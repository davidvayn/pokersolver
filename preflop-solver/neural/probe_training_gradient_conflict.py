"""Scores-blind gradient/retention diagnosis on frozen TRAIN checkpoints."""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import run_action_contrast_students as base
import train_public_value_network as training
from action_contrast_loss import gradient_add
from retained_initialization import import_retained_weights
from retained_value_protection import eligible_rows
from run_native_value_pilot import guarded
from run_native_value_preflight import atomic_json, sha256
from run_postflop_gap_pilot import PilotMemoryGuard
from training_gradient_conflict import gradient_relationship


def average_gradient(model, dataset, contexts, queries, rows, weights, targets):
    total, value = None, 0.
    for start in range(0, len(rows), 8):
        selected = rows[start:start+8]
        inputs = (mx.array(contexts[selected]), mx.array(queries[selected]),
            mx.array(dataset.projection_weights[selected]), mx.array(dataset.target_scales[selected]),
            mx.array(targets[selected]), mx.array(weights[selected]))
        loss, gradient = nn.value_and_grad(model, base.value_loss)(model, *inputs)
        mx.eval(loss, gradient)
        if not np.isfinite(float(loss)): raise ValueError("nonfinite TRAIN calibration loss")
        fraction = len(selected) / len(rows)
        value += float(loss) * fraction
        zero = gradient_add(gradient, gradient, -1.)
        total = gradient_add(zero if total is None else total, gradient, fraction)
    return value, total


def worker(args, manifest):
    if os.environ.get("MLX_ENABLE_TF32") != "0": raise ValueError("full-float32 worker required")
    dataset, contexts, queries, split, bundles, _ = base.prepare(args)
    train = np.flatnonzero(np.isin(dataset.groups, split[0]))
    old = eligible_rows(dataset.groups, train, dataset.projection_weights)
    authentic = dataset.projection_weights.reshape((-1, 2652)).copy()
    authentic *= 2652 / np.maximum(authentic.sum(axis=1, keepdims=True), 1e-12)
    settings = manifest["fitSettings"]
    objective = base.BundleObjective(bundles, settings["contrastWeight"], 4, 4, serving_aligned=True)
    results = []
    for seed in (10601, 10602):
        for step in (0, 200, 400, 600):
            receipt = args.checkpoints[str(seed)][str(step)]
            path = Path(receipt["path"])
            if sha256(path) != receipt["sha256"]: raise ValueError("frozen checkpoint changed")
            model = training.SharedComboValueNetwork(True, "wide", "payoff-exposure", training.FEATURE_SCHEMA_EXACT_RUNOUT)
            import_retained_weights(model, json.loads(path.read_text()), seed)
            old_auth_loss, old_auth = average_gradient(model, dataset, contexts, queries, old, authentic, dataset.targets)
            old_loss, old_calibration = average_gradient(model, dataset, contexts, queries, old, dataset.weights, dataset.targets)
            primary_loss, primary = average_gradient(model, dataset, contexts, queries, train, dataset.weights, dataset.targets)
            calibration, contrast = None, None
            for bundle in bundles:
                cal, aux, _ = objective.gradients(model, bundle, base.value_loss)
                zero = gradient_add(cal, cal, -1.)
                calibration = gradient_add(zero if calibration is None else calibration, cal, 1/len(bundles))
                contrast = gradient_add(zero if contrast is None else contrast, aux, settings["contrastWeight"]/len(bundles))
            learning = gradient_add(gradient_add(primary, calibration), contrast)
            parts = dict(primary=primary, bundleCalibration=calibration,
                         weightedContrast=contrast, expectedLearning=learning)
            row = dict(seed=seed, step=step, originalAuthenticLoss=old_auth_loss,
                originalCalibrationLoss=old_loss, primaryCalibrationLoss=primary_loss,
                originalAuthenticRelationships={name: gradient_relationship(old_auth, gradient) for name, gradient in parts.items()},
                originalCalibrationRelationships={name: gradient_relationship(old_calibration, gradient) for name, gradient in parts.items()})
            results.append(row)
            atomic_json(args.output/"analysis.json", dict(status="running", results=results, releaseAccepted=False))
            print(json.dumps(dict(event="training-gradient-conflict", seed=seed, step=step,
                originalAuthenticLoss=old_auth_loss,
                learningCosine=row["originalAuthenticRelationships"]["expectedLearning"]["cosine"])), flush=True)
            del model; gc.collect(); mx.clear_cache()
    by_seed = {seed: [r for r in results if r["seed"] == seed] for seed in (10601, 10602)}
    worsening = {str(seed): all(r["originalAuthenticLoss"] > rows[0]["originalAuthenticLoss"]*1.05
        for r in rows if r["step"] in (400, 600)) for seed, rows in by_seed.items()}
    conflicts = sum(r["step"] > 0 and r["originalAuthenticRelationships"]["expectedLearning"]["cosine"] is not None
        and r["originalAuthenticRelationships"]["expectedLearning"]["cosine"] < -.2 for r in results)
    atomic_json(args.output/"analysis.json", dict(status="complete", results=results, originalStates=len(old),
        trainingStates=len(train), conflictCheckpoints=conflicts, bothSeedOriginalLossWorsens=all(worsening.values()),
        gradientProjectionPilotSupported=conflicts >= 4 and all(worsening.values()), releaseAccepted=False,
        interpretation="Frozen TRAIN gradients and calibration only. Expected learning means an active bundle step averaged across families, without the separate soft retention gradient. No evaluation scores, new solves or fitting; no equilibrium or AdamW-step guarantee."))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control", type=Path, required=True); p.add_argument("--control-sha256", required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--worker", action="store_true")
    args = p.parse_args(); args.control = args.control.resolve(); args.output = args.output.resolve()
    if sha256(args.control) != args.control_sha256: raise ValueError("frozen control changed")
    manifest = json.loads(args.control.read_text())
    if (manifest.get("status") != "complete" or manifest.get("armDifference") != "serving_contrast_forward_vjp_only"
            or manifest.get("matmulPrecision") != "full-float32"):
        raise ValueError("completed serving-aligned wide control required")
    arguments = manifest["fitWorker"]["command"]
    pinned = {str(args.control): args.control_sha256}
    for name in ("corpus", "split-reference", "bundles", "feature-cache"):
        path = Path(arguments[arguments.index("--"+name)+1]).resolve()
        setattr(args, name.replace("-", "_"), path)
        if name != "feature-cache":
            digest = arguments[arguments.index("--"+name+"-sha256")+1]
            if sha256(path) != digest: raise ValueError("frozen diagnostic input changed")
            pinned[str(path)] = digest; setattr(args, name.replace("-", "_")+"_sha256", digest)
    args.feature_workers = 1; args.checkpoints = {}
    directory = Path(arguments[arguments.index("--retained-directory")+1]).resolve()
    pair = Path(manifest["studentManifest"]["path"])
    if sha256(pair) != manifest["studentManifest"]["sha256"]: raise ValueError("paired control changed")
    students = json.loads(pair.read_text()); pinned[str(pair)] = sha256(pair)
    if students.get("status") != "complete" or [r["seed"] for r in students["predictions"]] != [10601,10602]:
        raise ValueError("both completed seeds required")
    for seed in (10601,10602):
        args.checkpoints[str(seed)] = {}
        for step in (0,200,400,600):
            path = directory/f"turn-value-range-seed{seed}.json" if step == 0 else pair.parent/f"turn-value-range-seed{seed}{'-step'+str(step) if step != 600 else ''}.json"
            digest = sha256(path)
            pinned[str(path)] = digest; args.checkpoints[str(seed)][str(step)] = dict(path=str(path),sha256=digest)
    bundle = json.loads(args.bundles.read_text()); registry = bundle["trainingExtension"]
    pinned[registry["path"]] = registry["sha256"]
    pinned.update(json.loads(Path(registry["path"]).read_text())["pinnedInputs"])
    for family in bundle["families"]:
        receipts = [family["prefix"], family["calibrationCorpus"], *family["labels"]]
        if family.get("extraCapture"): receipts.append(family["extraCapture"])
        for receipt in receipts: pinned[receipt["path"]] = receipt["sha256"]
    for name in (Path(__file__).name,"training_gradient_conflict.py","run_action_contrast_students.py",
            "action_contrast_loss.py","action_contrast_dataset.py","train_public_value_network.py",
            "retained_initialization.py","retained_value_protection.py","serving_value_projection.py",
            "native_value_dataset.py","training_coverage.py","run_native_value_pilot.py","worker_resources.py"):
        path = Path(__file__).with_name(name); pinned[str(path)] = sha256(path)
    if any(sha256(Path(path)) != digest for path,digest in pinned.items()): raise ValueError("diagnostic dependency changed")
    if args.worker: worker(args, manifest); return
    if args.output.exists(): raise ValueError("never overwrite frozen gradient diagnosis")
    args.output.mkdir(); stop = threading.Event(); started = time.monotonic()
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig, lambda *_: stop.set())
    pressure = PilotMemoryGuard(stop,args.output/"system-memory.json").start()
    record = dict(status="running",schema="training-gradient-conflict-probe-v1",pinnedInputs=pinned,
        checkpoints=args.checkpoints,releaseAccepted=False)
    atomic_json(args.output/"manifest.json",record)
    try:
        record["worker"] = guarded([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],"--worker"],
            {"MLX_ENABLE_TF32":"0"},args.output/"worker",1200,6*1024**3,stop)
        if stop.is_set() or any(sha256(Path(path)) != digest for path,digest in pinned.items()):
            raise ValueError("diagnosis stopped or dependency changed")
        record["analysisSha256"] = sha256(args.output/"analysis.json"); record["status"] = "complete"
    except Exception as error:
        record["status"] = "failed"; record["failure"] = str(error); stop.set(); raise
    finally:
        record["elapsedSeconds"] = time.monotonic()-started
        record["systemMemoryGuard"] = pressure.finish(); atomic_json(args.output/"manifest.json",record)


if __name__ == "__main__": main()
