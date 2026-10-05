"""Read-only frozen TRAIN decision probes; never policy/exploitability scores."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np

from action_contrast_dataset import build_groups, contrast_loss_and_q_gradient
from run_native_value_preflight import atomic_json, sha256
from run_student_value_pilot import select_students
import train_public_value_network as training
from validate_public_value_parity import python_prediction
from run_native_value_pilot import read_capture
from training_coverage import extra_calibration, extension_families


def decision_metrics(group, predicted):
    q = np.asarray(predicted, dtype=float)
    weights = group.weights * group.support
    if q.shape != group.target.shape or not np.isfinite(q).all() or weights.sum() <= 0:
        raise ValueError("invalid supported decision probe")
    weights = weights / weights.sum()
    best, chosen = np.argmax(group.target, axis=0), np.argmax(q, axis=0)
    loss = group.target[best, np.arange(len(best))] - group.target[chosen, np.arange(len(best))]
    errors = [(q[a] - q[b]) - (group.target[a] - group.target[b])
              for a in range(len(q)) for b in range(a + 1, len(q))]
    huber, _ = contrast_loss_and_q_gradient(q, group.target, weights)
    return dict(**group.report(), bestActionAgreement=float(weights @ (best == chosen)),
                nativeLossFromPredictedBestBb=float(weights @ loss),
                actionContrastRmseBb=float(np.sqrt(np.mean([weights @ (e * e) for e in errors]))),
                contrastHuberLoss=huber,
                interpretation="Local frozen TRAIN/profile decision values, not the student's own policy or exploitability.")


def probe(args):
    if args.output.exists():
        raise ValueError("never overwrite a frozen decision probe")
    pinned = {str(p.resolve()): digest for p, digest in
              ((args.students, args.students_sha256), (args.bundles, args.bundles_sha256))}
    for name in (Path(__file__).name, "action_contrast_dataset.py", "train_public_value_network.py",
                 "validate_public_value_parity.py", "run_student_value_pilot.py", "native_value_dataset.py", "training_coverage.py"):
        path = Path(__file__).with_name(name).resolve()
        pinned[str(path)] = sha256(path)
    if any(sha256(Path(p)) != digest for p, digest in pinned.items()):
        raise ValueError("pinned probe input changed")
    stage, bundles = json.loads(args.students.read_text()), json.loads(args.bundles.read_text())
    new_inputs = getattr(args, "new_training_probe", False)
    if (stage.get("status") != "complete" or stage.get("schema") != "matched-action-contrast-students-v1"
            or [a["arm"] for a in stage["arms"]] != ["C0", "C1"]
            or bundles.get("status") != "complete" or len(bundles["families"]) not in (3,6)):
        raise ValueError("complete matching C0/C1/full49 TRAIN set required")
    if new_inputs:
        registry = bundles.get("trainingExtension")
        if (bundles.get("schema") != "training-coverage-action-bundle-pilot-v1" or not registry
                or sorted(f["root"] for f in bundles["families"]) != [100,101,102]):
            raise ValueError("new-input probe requires exactly the registered TRAIN extension")
        path=Path(registry["path"])
        if sha256(path)!=registry["sha256"]: raise ValueError("new TRAIN registry changed")
        plan=json.loads(path.read_text())
        if not {plan["corpusSha256"],plan["splitReferenceSha256"]} <= set(stage["pinnedInputs"].values()):
            raise ValueError("new TRAIN probe uses a different primary/split source")
        extension_families(registry,plan["corpusSha256"],plan["splitReferenceSha256"],set(),set())
        registered={f["root"]:tuple(f["family"]) for f in plan["families"]}
        if any(tuple(f["family"])!=registered[f["root"]] for f in bundles["families"]):
            raise ValueError("new TRAIN probe differs from the registry")
        pinned[str(path.resolve())]=registry["sha256"]
        pinned.update(plan["pinnedInputs"])
    elif stage["pinnedInputs"].get(str(args.bundles.resolve())) != args.bundles_sha256:
        raise ValueError("complete matching C0/C1/full49 TRAIN set required")
    models = []
    for arm in stage["arms"]:
        pinned[str(Path(arm["manifestPath"]).resolve())] = arm["manifestSha256"]
        _, pair = select_students(Path(arm["manifestPath"]), arm["manifestSha256"])
        for seed, row in pair.items():
            path = Path(row["path"])
            pinned[str(path)] = row["sha256"]
            model = json.loads(path.read_text())
            if (model.get("predictionContract") != "native-turn-cfv-full-stack-v1"
                    or model.get("featureSchema") != training.FEATURE_SCHEMA_EXACT_RUNOUT):
                raise ValueError("probe prediction/feature contract changed")
            models.append((arm["arm"], seed, model))
    started, results = time.monotonic(), []
    for family in sorted(bundles["families"], key=lambda f: f["root"]):
        if not family.get("all49Turns"):
            raise ValueError("probe requires complete public chance")
        for receipt in [family["prefix"], family["calibrationCorpus"], *family["labels"]]:
            path = Path(receipt["path"])
            if sha256(path) != receipt["sha256"]:
                raise ValueError("frozen probe source changed")
            pinned[str(path)] = receipt["sha256"]
        prefix = json.loads(Path(family["prefix"]["path"]).read_text())
        packets = [json.loads(Path(r["path"]).read_text()) for r in family["labels"]]
        groups, ordered = build_groups(prefix, packets, require_full_chance=True)
        if family.get("extraCapture"):
            receipt=family["extraCapture"]; path=Path(receipt["path"])
            if sha256(path)!=receipt["sha256"]: raise ValueError("probe search calibration changed")
            pinned[str(path)]=receipt["sha256"]
            groups,ordered=extra_calibration(groups,ordered,read_capture(path),family["candidateSha256"])
        dataset = training.load_dataset(Path(family["calibrationCorpus"]["path"]), 1, "payoff-exposure")
        if dataset.source["targets"] != ordered:
            raise ValueError("probe feature/affine ordering differs")
        del ordered, packets, prefix
        key = training.feature_cache_key(dataset, training.FEATURE_SCHEMA_EXACT_RUNOUT)
        if not new_inputs and not (args.feature_cache / f"{key}.json").is_file():
            raise ValueError("probe requires an existing complete feature cache")
        contexts, queries, cache = training.feature_dataset_cached(
            dataset, training.FEATURE_SCHEMA_EXACT_RUNOUT, 2 if new_inputs else 1, args.feature_cache)
        pinned[str(Path(cache["metadata"]).resolve())] = sha256(Path(cache["metadata"]))
        for arm, seed, model in models:
            predicted = np.array([python_prediction(dataset, model, i, (contexts[i], queries[i]))
                                  for i in range(len(dataset.targets))])
            rows = [decision_metrics(g, g.backup(predicted[:, g.actor])) for g in groups]
            bands={}
            if family.get("extraCapture"):
                for i in range(len(dataset.source["targets"])-16,len(dataset.source["targets"])):
                    target=dataset.source["targets"][i]; name=target["state_distribution"]
                    weight=np.asarray(target["ranges"])*np.asarray(target["opponent_compatible_mass"])
                    error=predicted[i]-np.asarray(target["counterfactual_values_bb"])
                    bands.setdefault(name,[]).append(float(np.sum(weight*error*error)/weight.sum()))
            results.append(dict(root=family["root"], family=family["family"], arm=arm,
                                solverSeed=seed, featureCache=cache, decisions=rows,
                                searchBandRmseBb={k:float(np.sqrt(np.mean(v))) for k,v in bands.items()}))
            print(json.dumps(dict(event="frozen-train-decision-probe", root=family["root"],
                                  arm=arm, seed=seed, decisions=rows)), flush=True)
            del predicted
        del dataset, contexts, queries, groups
        gc.collect()
    if any(sha256(Path(p)) != digest for p, digest in pinned.items()):
        raise ValueError("pinned source changed during probe")
    args.output.mkdir()
    atomic_json(args.output / "manifest.json", dict(schema="frozen-action-contrast-probe-v1",
        status="complete", pinnedInputs=pinned, results=results, elapsedSeconds=time.monotonic() - started,
        releaseAccepted=False, newTrainingInputsOnly=new_inputs, codeSha256=sha256(Path(__file__))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("students", "bundles"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--new-training-probe", action="store_true", help="Diagnose the three pinned new TRAIN families, never held-out policy scores.")
    probe(parser.parse_args())
