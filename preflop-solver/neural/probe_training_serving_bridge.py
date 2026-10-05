"""Frozen TRAIN action backups: optimization forward versus bounded serving."""
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
import numpy as np

import run_action_contrast_students as base
import train_public_value_network as training
from retained_initialization import import_retained_weights, native_import_prediction
from run_native_value_pilot import guarded
from run_native_value_preflight import atomic_json, sha256
from run_postflop_gap_pilot import PilotMemoryGuard


def compare_bridge(group, raw_values, served_values):
    raw, served = (group.backup(values[:, group.actor]) for values in (raw_values, served_values))
    weights = group.weights * group.support
    if not np.isfinite(raw).all() or not np.isfinite(served).all() or weights.sum() <= 0:
        raise ValueError("nonfinite or unsupported TRAIN bridge")
    weights = weights / weights.sum()
    r, s = np.argmax(raw, axis=0), np.argmax(served, axis=0)
    truth = np.max(group.target, axis=0)
    combos = np.arange(len(weights))
    differences = [(raw[a]-raw[b])-(served[a]-served[b])
        for a in range(len(raw)) for b in range(a+1,len(raw))]
    if not differences: differences = [np.zeros_like(weights)]
    return dict(history=group.history, actor=group.actor,
        rawNativeRankingLossBb=float(weights @ (truth - group.target[r, combos])),
        servedNativeRankingLossBb=float(weights @ (truth - group.target[s, combos])),
        trainingServingBestAgreement=float(weights @ (r == s)),
        trainingServingContrastRmseBb=float(np.sqrt(np.mean(np.asarray(differences)**2, axis=0) @ weights)),
        supportedReachFraction=float((group.weights * group.support).sum() / group.weights.sum()))


def worker(args):
    dataset, contexts, queries, split, bundles, _ = base.prepare(args)
    del dataset, contexts, queries, split; gc.collect()
    students = json.loads(args.students.read_text())
    results = []
    for entry in students["predictions"]:
        model = training.SharedComboValueNetwork(True, "wide", "payoff-exposure", training.FEATURE_SCHEMA_EXACT_RUNOUT)
        payload = json.loads(Path(entry["model"]).read_text())
        import_retained_weights(model, payload, entry["seed"])
        for bundle in bundles:
            d = bundle.dataset
            raw, served = [], []
            for start in range(0, len(d.targets), 8):
                end = min(start+8, len(d.targets))
                inputs = base.BundleObjective.inputs(bundle, start, end)
                raw.append(np.asarray(model(*inputs)).reshape((-1, 2, 1326))
                    * d.target_scales[start:end, None, None])
                served.append(native_import_prediction(model, bundle.contexts[start:end],
                    bundle.queries[start:end], d.target_scales[start:end], d.boards[start:end], d.ranges[start:end]))
            raw, served = np.concatenate(raw), np.concatenate(served)
            decisions = [compare_bridge(g, raw, served) for g in bundle.groups]
            weights = d.projection_weights
            value_rmse = float(np.sqrt(np.sum(weights*(raw-served)**2) / weights.sum()))
            row = dict(seed=entry["seed"], family=list(bundle.family),
                authenticTrainingServingValueRmseBb=value_rmse, decisions=decisions)
            results.append(row)
            atomic_json(args.output / "analysis.json", dict(status="running", results=results, releaseAccepted=False))
            print(json.dumps(dict(event="training-serving-bridge", seed=entry["seed"], family=list(bundle.family),
                maximumContrastRmseBb=max(r["trainingServingContrastRmseBb"] for r in decisions))), flush=True)
        del model; gc.collect(); mx.clear_cache()
    atomic_json(args.output / "analysis.json", dict(status="complete", results=results, releaseAccepted=False,
        interpretation="Frozen TRAIN backups only. A different serving wrapper is not necessarily a policy error or an exploitability estimate."))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "split-reference", "bundles", "students"):
        p.add_argument("--"+name, type=Path, required=True)
        p.add_argument("--"+name+"-sha256", required=True)
    p.add_argument("--feature-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args(); a.feature_workers = 1
    pinned = {}
    for name in ("corpus", "split_reference", "bundles", "students"):
        path = getattr(a, name).resolve(); setattr(a, name, path)
        digest = getattr(a, name+"_sha256")
        if sha256(path) != digest: raise ValueError("pinned bridge input changed")
        pinned[str(path)] = digest
    students = json.loads(a.students.read_text())
    if (students.get("schema") != "native-value-student-pair-controller-v1"
            or students.get("status") != "complete" or students.get("corpusSha256") != a.corpus_sha256
            or students.get("bundleManifestSha256") != a.bundles_sha256
            or sorted(r["seed"] for r in students["predictions"]) != [10601,10602]):
        raise ValueError("complete hash-matched paired students required")
    for row in students["predictions"]:
        path = Path(row["model"]).resolve()
        if row["maximumParityErrorBb"] > .0001 or sha256(path) != row["modelSha256"]:
            raise ValueError("student export parity/identity changed")
        pinned[str(path)] = row["modelSha256"]
    bundles = json.loads(a.bundles.read_text())
    if bundles.get("status") != "complete": raise ValueError("completed TRAIN bundles required")
    registry = bundles["trainingExtension"]
    path = Path(registry["path"]).resolve()
    if sha256(path) != registry["sha256"]: raise ValueError("TRAIN registry changed")
    pinned[str(path)] = registry["sha256"]
    pinned.update(json.loads(path.read_text())["pinnedInputs"])
    for family in bundles["families"]:
        receipts = [family["prefix"],family["calibrationCorpus"],*family["labels"]]
        if family.get("extraCapture"): receipts.append(family["extraCapture"])
        for receipt in receipts:
            path=Path(receipt["path"]).resolve()
            if sha256(path) != receipt["sha256"]: raise ValueError("TRAIN bundle changed")
            pinned[str(path)]=receipt["sha256"]
    for name in (Path(__file__).name, "run_action_contrast_students.py", "action_contrast_dataset.py",
            "action_contrast_loss.py", "train_public_value_network.py", "retained_initialization.py",
            "native_value_dataset.py", "training_coverage.py", "worker_resources.py"):
        path = Path(__file__).with_name(name); pinned[str(path)] = sha256(path)
    a.output, a.feature_cache = a.output.resolve(), a.feature_cache.resolve()
    if a.worker:
        worker(a); return
    if a.output.exists(): raise ValueError("never overwrite a TRAIN bridge diagnosis")
    a.output.mkdir(); started = time.monotonic(); stop = threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *_:stop.set())
    pressure = PilotMemoryGuard(stop, a.output/"system-memory.json").start()
    record = dict(schema="training-serving-action-bridge-v1", status="running", pinnedInputs=pinned, releaseAccepted=False)
    atomic_json(a.output/"manifest.json",record)
    try:
        record["worker"] = guarded([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],"--worker"],
            {"MLX_ENABLE_TF32":"0"},a.output/"worker",1200,6*1024**3,stop)
        if stop.is_set() or any(sha256(Path(path)) != digest for path,digest in pinned.items()):
            raise ValueError("bridge stopped or dependencies changed")
        record["analysisSha256"]=sha256(a.output/"analysis.json"); record["status"]="complete"
    except Exception as error:
        record["status"]="failed"; record["failure"]=str(error); stop.set(); raise
    finally:
        record["elapsedSeconds"]=time.monotonic()-started; record["systemMemoryGuard"]=pressure.finish()
        atomic_json(a.output/"manifest.json",record)


if __name__ == "__main__": main()
