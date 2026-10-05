"""Matched C0/C1 fixed-checkpoint students with bounded calibrated contrast VJP."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import signal
import sys
import threading
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import train_public_value_network as training
import native_value_dataset as native
from action_contrast_dataset import build_groups
from action_contrast_loss import BundleObjective, TrainingBundle, gradient_add, gradient_norm
from run_native_value_pilot import guarded, read_capture, test_command, PREDICT_TEST
from run_native_value_preflight import atomic_json, sha256
from run_postflop_gap_pilot import PilotMemoryGuard
from validate_public_value_parity import python_prediction
from training_coverage import extra_calibration, extension_families


def value_loss(current, context, query, projection, scales, targets, weights):
    error = current(context, query, projection, scales) - targets
    def huber(v):
        absolute = mx.abs(v); quadratic = mx.minimum(absolute, .05)
        return .5 * quadratic * quadratic + .05 * (absolute - quadratic)
    denominator = mx.maximum(mx.sum(weights), 1e-8)
    return (mx.sum(weights * huber(error)) + .25 * mx.sum(weights * huber(error * scales[:, None] / 20))) / denominator


def prepare(args):
    source = read_capture(args.corpus)
    reference = read_capture(args.split_reference)
    # The retained 508 prefix already refreshed TRAIN labels. Holdout/tuning
    # labels remain byte-identical to the original 284 split reference.
    split = native.family_split(source, 10601, .25, .25, reference=reference, refresh_training=True)
    if tuple(map(len, split)) != (474, 69, 72): raise ValueError("frozen split changed")
    forbidden = {native.board_family(source["targets"][i]["board"]) for i in np.concatenate(split[1:])}
    allowed = {native.board_family(source["targets"][i]["board"]) for i in split[0]}
    # The arrays and Dataset.source preserve training/export metadata. Do not
    # retain two extra parsed copies of the 615/284 targets during preprocessing.
    del source, reference
    dataset = training.load_dataset(args.corpus, 1, "payoff-exposure")
    contexts, queries, cache = training.feature_dataset_cached(dataset,
        training.FEATURE_SCHEMA_EXACT_RUNOUT, args.feature_workers, args.feature_cache)
    manifest = json.loads(args.bundles.read_text())
    if manifest.get("status") != "complete" or manifest.get("calibrationOnly"):
        raise ValueError("completed training bundles required")
    registry = manifest.get("trainingExtension")
    expected_roots = [2, 3, 4]
    if registry:
        added = extension_families(registry, args.corpus_sha256, args.split_reference_sha256, allowed, forbidden)
        allowed |= added
        expected_roots += [100, 101, 102]
        plan = json.loads(Path(registry["path"]).read_text())
        registered = {f["root"]: tuple(f["family"]) for f in plan["families"]}
        if any(tuple(f["family"]) != registered.get(f["root"]) for f in manifest["families"] if f["root"] >= 100):
            raise ValueError("training family/id differs from the registry")
    if sorted(f["root"] for f in manifest["families"]) != expected_roots:
        raise ValueError("predeclared training family set changed")
    bundles = []
    for family in sorted(manifest["families"], key=lambda f: f["root"]):
        if tuple(family["family"]) not in allowed or tuple(family["family"]) in forbidden:
            raise ValueError("training bundle leaks held-out family")
        for receipt in [family["prefix"], family["calibrationCorpus"], *family["labels"]]:
            if sha256(Path(receipt["path"])) != receipt["sha256"]: raise ValueError("bundle source changed")
        prefix = json.loads(Path(family["prefix"]["path"]).read_text())
        packets = [json.loads(Path(r["path"]).read_text()) for r in family["labels"]]
        groups, ordered = build_groups(prefix, packets, require_full_chance=bool(family.get("all49Turns")))
        if family.get("extraCapture"):
            if not registry or family["root"] < 100: raise ValueError("unregistered extra search calibration")
            receipt = family["extraCapture"]; path = Path(receipt["path"])
            if sha256(path) != receipt["sha256"]: raise ValueError("extra calibration capture changed")
            groups, ordered = extra_calibration(groups, ordered, read_capture(path), family["candidateSha256"])
        data = training.load_dataset(Path(family["calibrationCorpus"]["path"]), 1, "payoff-exposure")
        if data.source["targets"] != ordered: raise ValueError("bundle feature/affine leaf ordering differs")
        # Affine tensors no longer depend on the separately decoded packets.
        # Retain the complete validated calibration source, not both copies.
        del ordered, packets, prefix
        c, q, _ = training.feature_dataset_cached(data, training.FEATURE_SCHEMA_EXACT_RUNOUT, args.feature_workers, args.feature_cache)
        if any(float((g.weights * g.support).sum()) <= 0 for g in groups):
            raise ValueError("training bundle has no profile-consistent authentic contrast support")
        bundles.append(TrainingBundle(data, c, q, groups, tuple(family["family"])))
    if len(bundles) != len(expected_roots): raise ValueError("predeclared training families required")
    return dataset, contexts, queries, split, bundles, cache


def budgeted_cadence(family_seconds, available_seconds):
    """Resolve equal-arm work BEFORE fitting, never from response/holdout scores."""
    times = np.asarray(family_seconds, dtype=float)
    if times.shape not in ((3,), (6,)) or not np.isfinite(times).all() or (times <= 0).any():
        raise ValueError("three or six finite positive complete-family timings required")
    for cadence in (4, 8, 20):
        if (600 // cadence) % len(times): continue
        # Reject schedules that silently undertrain some registered families.
        projected = 1.5 * (float(np.mean(times)) * (600 // cadence) * 4 + 405 * 2 + 1200)
        if projected <= available_seconds:
            return cadence, projected
    raise ValueError("even reduced equal-arm bundle cadence exceeds remaining two-hour cap")


def preflight(args):
    started = time.monotonic()
    dataset, contexts, queries, split, bundles, cache = prepare(args)
    mx.random.seed(10601)
    model = training.SharedComboValueNetwork(True, "wide", "payoff-exposure", training.FEATURE_SCHEMA_EXACT_RUNOUT)
    train_rows = np.flatnonzero(np.isin(dataset.groups, split[0]))
    selected = training.primary_replay_batch_rows(np.random.default_rng(10601), train_rows,
        np.array([], dtype=np.int64), dataset.invested, 8, 0., np.ones(len(dataset.targets)))
    inputs = (mx.array(contexts[selected]), mx.array(queries[selected]),
              mx.array(dataset.projection_weights[selected]), mx.array(dataset.target_scales[selected]),
              mx.array(dataset.targets[selected]), mx.array(dataset.weights[selected]))
    ordinary_loss, ordinary = nn.value_and_grad(model, value_loss)(model, *inputs)
    objective = BundleObjective(bundles, 0.)
    norms, times, calibration, contrast = [], [], None, None
    for bundle in bundles:
        start = time.monotonic()
        cal, aux, report = objective.gradients(model, bundle, value_loss)
        times.append(time.monotonic() - start); norms.append(report)
        calibration = cal if calibration is None else gradient_add(calibration, cal)
        contrast = aux if contrast is None else gradient_add(contrast, aux)
    combined = gradient_add(ordinary, calibration, 1/len(bundles))
    contrast_norm = gradient_norm(contrast) / len(bundles)
    value_norm = gradient_norm(combined)
    if not np.isfinite([float(ordinary_loss), value_norm, contrast_norm]).all() or min(value_norm, contrast_norm) <= 0:
        raise ValueError("nonfinite/zero preflight component norm")
    coefficient = min(10., .25 * value_norm / contrast_norm)
    # Measured ordinary pair fitting was 405s; reserve 1,200s for exhaustive
    # cached-feature NumPy/Rust parity of all four students, plus 50% margin.
    conditioning = time.monotonic() - started
    cadence, projected = budgeted_cadence(times, 7200 - conditioning)
    settings = dict(schema="action-contrast-fit-settings-v1", status="complete", releaseAccepted=False,
        contrastWeight=coefficient, requestedContrastGradientFraction=.25,
        measuredContrastGradientFraction=coefficient*contrast_norm/value_norm,
        gradientWeightCap=10., valueGradientNorm=value_norm, contrastGradientNorm=contrast_norm,
        trainingSeedForConditioning=10601, frozenSplitTrainingRefresh=True,
        familyNorms=norms, familyGradientSeconds=times,
        projectedFitAndParitySeconds=projected, conditioningSeconds=conditioning,
        cadence=cadence, requestedCadence=4, reducedCadenceForResources=cadence != 4,
        chunkSize=4, fixedFinalStep=600, featureWorkers=args.feature_workers,
        pairedSeeds=[10601, 10602], featureCache=cache, armDifference="contrast_weight_only")
    atomic_json(args.output / "fit-settings.json", settings)
    return settings


def fit(args):
    settings = json.loads((args.output / "fit-settings.json").read_text())
    dataset, contexts, queries, split, bundles, cache = prepare(args)
    train, tuning, holdout = [np.flatnonzero(np.isin(dataset.groups, s)) for s in split]
    work = args.output / args.arm; work.mkdir()
    rows = []
    architecture = getattr(args, "architecture", "wide")
    transform = getattr(args, "initialization_transform", None)
    for seed in [10601, 10602]:
        initial_model = None
        initial_receipt = None
        if getattr(args, "initial_models", None) is not None:
            from retained_initialization import import_retained_weights, native_import_prediction
            initial_receipt = args.initial_models[seed]
            path = Path(initial_receipt["path"])
            if sha256(path) != initial_receipt["sha256"]:
                raise ValueError("retained initialization changed")
            initial_model = json.loads(path.read_text())
            probe = training.SharedComboValueNetwork(True, architecture, "payoff-exposure",
                training.FEATURE_SCHEMA_EXACT_RUNOUT)
            import_retained_weights(probe, initial_model, seed, transform)
            # Full private-vector comparison on two frozen TRAIN states. Every
            # serialized parameter is additionally validated by the importer.
            selected = train[:2]
            full_gpu = getattr(args, "matmul_precision", "default") == "full-float32"
            got = native_import_prediction(probe, contexts[selected], queries[selected],
                dataset.target_scales[selected], dataset.boards[selected], dataset.ranges[selected],
                device=mx.gpu if full_gpu else mx.cpu)
            expected = np.asarray([python_prediction(dataset, initial_model, int(i),
                (contexts[i], queries[i])) for i in selected])
            error = float(np.max(np.abs(got - expected)))
            if not np.isfinite(error) or error > .0001:
                raise ValueError("retained initialization independent NumPy/MLX parity failed")
            atomic_json(work / f"initialization-{seed}.json", dict(**initial_receipt,
                seed=seed, maximumParityErrorBb=error, states=selected.tolist(),
                optimizerState="fresh_not_imported",
                parityDevice="gpu_full_float32" if full_gpu else "cpu_full_float32",
                transform=transform,
                releaseAccepted=False))
            del probe; mx.clear_cache()
        objective = BundleObjective(bundles, settings["contrastWeight"] if args.arm == "C1" else 0.,
                                    settings["cadence"], settings["chunkSize"],
                                    serving_aligned=getattr(args,"serving_aligned",False))
        protection_settings = getattr(args, "protection_settings", None)
        if protection_settings is not None:
            from retained_value_protection import RetainedValueProtection
            if initial_model is None:
                raise ValueError("retained-value protection requires a frozen initialized reference")
            reference = training.SharedComboValueNetwork(True, "wide", "payoff-exposure",
                training.FEATURE_SCHEMA_EXACT_RUNOUT)
            import_retained_weights(reference, initial_model, seed)
            objective = RetainedValueProtection(objective, dataset, contexts, queries, train,
                reference, protection_settings["coefficient"], seed,
                protection_settings["cadence"], protection_settings["batchSize"])
            del reference
        def save(current, step):
            training.export_model(current, work / f"turn-value-range-seed{seed}-step{step}.json", seed,
                dataset.source_sha256, native.SCHEMA, "research_only", dataset.source["source_policy_sha256"], "payoff-exposure")
        current, _, _, metrics = training.train_one(dataset, contexts, queries, train, train,
            np.array([], dtype=np.int64), tuning, holdout, True, seed, 600, 8, .0003, .00003,
            True, 50, 6, architecture, "payoff-exposure", .05, .25, np.ones(len(dataset.targets)), 0.,
            training.FEATURE_SCHEMA_EXACT_RUNOUT, fixed_final_checkpoint=True,
            bundle_objective=objective, checkpoint_callback=save, initial_model=initial_model,
            initialization_transform=transform)
        model_path = work / f"turn-value-range-seed{seed}.json"
        training.export_model(current, model_path, seed, dataset.source_sha256, native.SCHEMA,
            "research_only", dataset.source["source_policy_sha256"], "payoff-exposure")
        counts = metrics["actionBundleObjective"]["familyCounts"]
        protection_report = metrics["actionBundleObjective"].get("retainedValueProtection")
        if protection_settings is not None and (protection_report is None
                or protection_report["updates"] != 150 or protection_report["draws"] != 1200):
            raise ValueError("retained-value protection exposure drift")
        if (metrics["selectedStep"] != 600
                or metrics["actionBundleObjective"]["bundleUpdates"] != 600 // settings["cadence"]
                or len(counts) != len(bundles) or set(counts.values()) != {600 // settings["cadence"] // len(bundles)}):
            raise ValueError("matched checkpoint/cadence drift")
        rows.append(dict(seed=seed, model=str(model_path), modelSha256=sha256(model_path), metrics=metrics,
            **({"initialization": initial_receipt} if initial_receipt is not None else {})))
        atomic_json(work / "fit-report.json", dict(arm=args.arm, variants=rows, settings=settings, status="running"))
        print(json.dumps(dict(event="contrast-seed-fit", arm=args.arm, seed=seed,
            selectedStep=600, surrogateHoldoutRmseBb=metrics["onPolicyReachMetrics"]["weightedRmseBb"])), flush=True)
        del current; gc.collect(); mx.clear_cache()
    atomic_json(work / "fit-report.json", dict(arm=args.arm, variants=rows, settings=settings,
        status="complete", featureCache=cache, split=[s.tolist() for s in split], releaseAccepted=False))


def verify(args, arm, stop):
    # Reuse HASH-VERIFIED deterministic feature arrays in the independent NumPy
    # predictor; neither framework forward passes nor only reached hands count
    # as export parity. All 615 states x 2 x 1326 queries remain checked.
    dataset = training.load_dataset(args.corpus, 1, "payoff-exposure")
    contexts, queries, _ = training.feature_dataset_cached(dataset, training.FEATURE_SCHEMA_EXACT_RUNOUT, 1, args.feature_cache)
    work = args.output / arm
    report = json.loads((work / "fit-report.json").read_text())
    record = dict(schema="native-value-student-pair-controller-v1", status="running", releaseAccepted=False,
                  corpusSha256=args.corpus_sha256, arm=arm, predictions=[],
                  bundleManifestSha256=args.bundles_sha256, fixedFinalStep=600,
                  fitSettingsSha256=sha256(args.output / "fit-settings.json"))
    for entry in report["variants"]:
        model_path = Path(entry["model"]); model = json.loads(model_path.read_text())
        output = work / f"predictions-{entry['seed']}.json"
        worker = guarded(test_command(args.binary, PREDICT_TEST), dict(
            POKER_NATIVE_PREDICT_DATASET=str(args.corpus), POKER_NATIVE_PREDICT_DATASET_SHA=args.corpus_sha256,
            POKER_NATIVE_PREDICT_MODEL=str(model_path), POKER_NATIVE_PREDICT_MODEL_SHA=entry["modelSha256"],
            POKER_NATIVE_PREDICT_OUTPUT=str(output)), work / f"predict-{entry['seed']}", 900, stop=stop)
        payload = json.loads(output.read_text()); values = np.asarray(payload["predictions"])
        if values.shape != (615, 2, 1326) or not np.isfinite(values).all(): raise ValueError("invalid native predictions")
        error = max(float(np.max(np.abs(python_prediction(dataset, model, i,
            (contexts[i], queries[i])) - values[i]))) for i in range(len(values)))
        if error > .0001: raise ValueError("independent NumPy/native parity failed")
        record["predictions"].append(dict(seed=entry["seed"], model=str(model_path),
            modelSha256=entry["modelSha256"], maximumParityErrorBb=error, states=615,
            predictionsSha256=sha256(output), worker=worker, fitMetrics=entry["metrics"]))
        atomic_json(work / "manifest.json", record)
        print(json.dumps(dict(event="contrast-student-verified", arm=arm, seed=entry["seed"], maximumParityErrorBb=error)), flush=True)
    record["status"] = "complete"; atomic_json(work / "manifest.json", record)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "corpus", "split-reference", "bundles"):
        p.add_argument("--" + name, type=Path, required=True)
        p.add_argument("--" + name + "-sha256", required=True)
    p.add_argument("--feature-cache", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--feature-workers", type=int, choices=(1, 2), default=1)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--quality-decision", type=Path)
    p.add_argument("--quality-decision-sha256")
    p.add_argument("--worker", choices=("preflight", "C0", "C1"))
    a = p.parse_args()
    pinned = {}
    for name in ("binary", "corpus", "split_reference", "bundles"):
        path = getattr(a, name).resolve(); setattr(a, name, path)
        if sha256(path) != getattr(a, name + "_sha256"): raise ValueError("pinned input changed")
        pinned[str(path)] = sha256(path)
    a.output, a.feature_cache = a.output.resolve(), a.feature_cache.resolve()
    bundle_manifest = json.loads(a.bundles.read_text())
    registry = bundle_manifest.get("trainingExtension")
    if registry:
        path = Path(registry["path"]).resolve()
        if sha256(path) != registry["sha256"]: raise ValueError("training registry changed")
        pinned[str(path)] = registry["sha256"]
        pinned.update(json.loads(path.read_text())["pinnedInputs"])
    if not a.preflight_only and a.worker != "preflight":
        if (a.quality_decision is None or a.quality_decision_sha256 is None
                or sha256(a.quality_decision) != a.quality_decision_sha256):
            raise ValueError("fitting requires a pinned, reviewed target-data decision")
        decision = json.loads(a.quality_decision.read_text())
        if (decision.get("schema") != "action-contrast-data-decision-v1"
                or decision.get("status") != "ready_for_fit_preflight"
                or decision.get("primaryTrainingManifestSha256") != a.bundles_sha256
                or decision.get("releaseAccepted") is not False):
            raise ValueError("inconclusive or mismatched target data cannot start C0/C1 fitting")
        pinned[str(a.quality_decision.resolve())] = a.quality_decision_sha256
    if a.worker:
        if a.worker == "preflight": preflight(a)
        else: a.arm = a.worker; fit(a)
        return
    if a.output.exists(): raise ValueError("never overwrite matched student stage")
    a.output.mkdir()
    for name in ("run_action_contrast_students.py", "action_contrast_dataset.py", "action_contrast_loss.py",
                 "train_public_value_network.py", "validate_public_value_parity.py", "native_value_dataset.py", "training_coverage.py"):
        path = Path(__file__).with_name(name); pinned[str(path)] = sha256(path)
    record = dict(schema="matched-action-contrast-students-v1", status="running", releaseAccepted=False,
                  pinnedInputs=pinned, arms=[], maximumSeconds=7200, maximumFitMemoryBytes=6*1024**3)
    atomic_json(a.output / "manifest.json", record)
    started = time.monotonic(); stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: stop.set())
    timer = threading.Timer(7200, stop.set); timer.daemon=True; timer.start()
    pressure = PilotMemoryGuard(stop, a.output / "system-memory.json").start()
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    try:
        record["preflight"] = guarded(command + ["--worker", "preflight"], {}, a.output / "preflight", 1200, 6*1024**3, stop)
        settings = json.loads((a.output / "fit-settings.json").read_text())
        record["fitSettings"] = settings; atomic_json(a.output / "manifest.json", record)
        print(json.dumps(dict(event="contrast-fit-preflight", coefficient=settings["contrastWeight"],
            projectedSeconds=settings["projectedFitAndParitySeconds"])), flush=True)
        for arm in (() if a.preflight_only else ("C0", "C1")):
            record[arm+"FitWorker"] = guarded(command + ["--worker", arm], {}, a.output / (arm+"-fit-worker"),
                min(3600, 7200-(time.monotonic()-started)), 6*1024**3, stop)
            verified = verify(a, arm, stop)
            record["arms"].append(dict(arm=arm, manifestPath=str(a.output/arm/"manifest.json"),
                                       manifestSha256=sha256(a.output/arm/"manifest.json")))
            atomic_json(a.output / "manifest.json", record)
        if stop.is_set() or any(sha256(Path(path)) != digest for path,digest in pinned.items()):
            raise ValueError("stage stopped or pinned source changed")
        record["status"] = "complete"
    except Exception as error:
        record["status"] = "failed"; record["failure"] = str(error); stop.set()
        raise
    finally:
        timer.cancel(); record["elapsedSeconds"] = time.monotonic()-started
        record["systemMemoryGuard"] = pressure.finish(); atomic_json(a.output/"manifest.json", record)
    print(json.dumps(dict(status=record["status"], seconds=record["elapsedSeconds"])), flush=True)


if __name__ == "__main__": main()
