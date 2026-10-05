"""Initialization-only pilot, matched to the frozen six-family scratch C1 pair."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

import run_action_contrast_students as base
from run_native_value_pilot import guarded
from run_native_value_preflight import atomic_json, sha256
from run_postflop_gap_pilot import PilotMemoryGuard

SETTINGS_SHA = "7d7a3f011f94102e6068d43bd2a2cac292d060dbb8dc9cc866ba7d7304d3a275"
CONTROL_SHA = "00951cd6061cd244e63b734dfa9ef93e080cd39df422d046e7f4b132ad29ae11"
RETAINED = {
    10601: "331c42c5d58219febaaa5caa692c6bcf2fe1a6d4098d31678a663d606a54eca0",
    10602: "7e3481eec45d0973e4696a09500b9e27f9ce82512673e1ab5d578c0cf11201c0",
}


def checked_settings(path):
    if sha256(path) != SETTINGS_SHA:
        raise ValueError("initialization-only pilot requires unchanged scratch fit settings")
    settings = json.loads(path.read_text())
    if (settings.get("status") != "complete" or settings.get("fixedFinalStep") != 600
            or settings.get("cadence") != 4 or settings.get("pairedSeeds") != [10601, 10602]):
        raise ValueError("frozen matched schedule differs")
    return settings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "corpus", "split-reference", "bundles", "quality-decision"):
        p.add_argument("--" + name, type=Path, required=True)
        p.add_argument("--" + name + "-sha256", required=True)
    p.add_argument("--settings", type=Path, required=True)
    p.add_argument("--scratch-control", type=Path, required=True)
    p.add_argument("--retained-directory", type=Path, required=True)
    p.add_argument("--feature-cache", type=Path, required=True)
    p.add_argument("--feature-workers", type=int, choices=(1, 2), default=2)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--matmul-precision", choices=("default", "full-float32"), default="default")
    p.add_argument("--warm-control", type=Path)
    p.add_argument("--warm-control-sha256")
    a = p.parse_args()
    pinned = {}
    for name in ("binary", "corpus", "split_reference", "bundles", "quality_decision"):
        path = getattr(a, name).resolve(); setattr(a, name, path)
        digest = getattr(a, name + "_sha256")
        if sha256(path) != digest: raise ValueError("pinned input changed: " + name)
        pinned[str(path)] = digest
    settings = checked_settings(a.settings)
    if sha256(a.scratch_control) != CONTROL_SHA:
        raise ValueError("frozen from-scratch C1 control changed")
    control = json.loads(a.scratch_control.read_text())
    if (control.get("status") != "complete" or control.get("fitSettingsSha256") != SETTINGS_SHA
            or control.get("bundleManifestSha256") != a.bundles_sha256
            or control.get("corpusSha256") != a.corpus_sha256):
        raise ValueError("scratch control does not match the experiment")
    decision = json.loads(a.quality_decision.read_text())
    if (decision.get("status") != "ready_for_fit_preflight"
            or decision.get("primaryTrainingManifestSha256") != a.bundles_sha256
            or decision.get("releaseAccepted") is not False):
        raise ValueError("reviewed target-data decision required")
    for path in (a.settings, a.scratch_control): pinned[str(path.resolve())] = sha256(path)
    warm_control = None
    if a.matmul_precision == "full-float32":
        if (a.warm_control is None or a.warm_control_sha256 is None
                or sha256(a.warm_control) != a.warm_control_sha256):
            raise ValueError("precision-only pilot requires a frozen matched warm-start control")
        warm_control = json.loads(a.warm_control.read_text())
        if (warm_control.get("status") != "complete"
                or warm_control.get("schema") != "retained-initialization-contrast-pilot-v1"
                or warm_control.get("armDifference") != "initial_weights_only_fresh_optimizer"
                or warm_control.get("scratchControlSha256") != CONTROL_SHA
                or warm_control.get("fitSettings") != settings
                or warm_control.get("pinnedInputs", {}).get(str(a.bundles)) != a.bundles_sha256
                or warm_control.get("pinnedInputs", {}).get(str(a.corpus)) != a.corpus_sha256):
            raise ValueError("warm-start control does not match the precision experiment")
        pinned[str(a.warm_control.resolve())] = a.warm_control_sha256
        pair_receipt = warm_control["studentManifest"]
        if sha256(Path(pair_receipt["path"])) != pair_receipt["sha256"]:
            raise ValueError("warm-start student pair changed")
        pinned[str(Path(pair_receipt["path"]).resolve())] = pair_receipt["sha256"]
        pair = json.loads(Path(pair_receipt["path"]).read_text())
        if pair.get("status") != "complete" or sorted(row["seed"] for row in pair["predictions"]) != [10601, 10602]:
            raise ValueError("complete paired warm-start control required")
        for row in pair["predictions"]:
            if row["maximumParityErrorBb"] > .0001:
                raise ValueError("warm control lacks export parity")
            pinned[str(Path(row["model"]).resolve())] = row["modelSha256"]
    elif a.warm_control is not None or a.warm_control_sha256 is not None:
        raise ValueError("warm control is only for a precision intervention")
    a.initial_models = {}
    for seed, digest in RETAINED.items():
        path = (a.retained_directory / f"turn-value-range-seed{seed}.json").resolve()
        if sha256(path) != digest: raise ValueError("retained seed model changed")
        a.initial_models[seed] = dict(path=str(path), sha256=digest)
        pinned[str(path)] = digest
    bundle = json.loads(a.bundles.read_text())
    registry = bundle["trainingExtension"]
    if sha256(Path(registry["path"])) != registry["sha256"]:
        raise ValueError("TRAIN family registry changed")
    pinned[str(Path(registry["path"]).resolve())] = registry["sha256"]
    pinned.update(json.loads(Path(registry["path"]).read_text())["pinnedInputs"])
    for family in bundle["families"]:
        receipts = [family["prefix"], family["calibrationCorpus"], *family["labels"]]
        if family.get("extraCapture"): receipts.append(family["extraCapture"])
        for receipt in receipts: pinned[str(Path(receipt["path"]).resolve())] = receipt["sha256"]
    for name in ("run_retained_contrast_students.py", "retained_initialization.py",
            "run_action_contrast_students.py", "train_public_value_network.py", "action_contrast_loss.py",
            "action_contrast_dataset.py", "validate_public_value_parity.py", "native_value_dataset.py",
            "training_coverage.py", "run_native_value_pilot.py", "worker_resources.py"):
        path = Path(__file__).with_name(name); pinned[str(path)] = sha256(path)
    a.output, a.feature_cache = a.output.resolve(), a.feature_cache.resolve()
    a.arm = "C1"
    if a.worker:
        if a.matmul_precision == "full-float32" and os.environ.get("MLX_ENABLE_TF32") != "0":
            raise ValueError("full-float32 must be selected before importing MLX in the worker")
        base.fit(a)
        return
    if a.output.exists(): raise ValueError("never overwrite a retained-initialization stage")
    if any(sha256(Path(path)) != digest for path, digest in pinned.items()):
        raise ValueError("experiment dependency changed")
    a.output.mkdir()
    # Preserve the original settings bytes, not recondition lambda on a different model.
    (a.output / "fit-settings.json").write_bytes(a.settings.read_bytes())
    record = dict(schema="retained-initialization-contrast-pilot-v1", status="running",
        releaseAccepted=False, armDifference=("matmul_precision_only" if warm_control is not None
            else "initial_weights_only_fresh_optimizer"), matmulPrecision=a.matmul_precision,
        warmControlSha256=a.warm_control_sha256,
        scratchControlSha256=CONTROL_SHA, pinnedInputs=pinned, maximumSeconds=7200,
        maximumFitMemoryBytes=6*1024**3, fitSettings=settings)
    atomic_json(a.output / "manifest.json", record)
    stop = threading.Event(); started = time.monotonic()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: stop.set())
    timer = threading.Timer(7200, stop.set); timer.daemon = True; timer.start()
    pressure = PilotMemoryGuard(stop, a.output / "system-memory.json").start()
    try:
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
        env = {"MLX_ENABLE_TF32": "0"} if a.matmul_precision == "full-float32" else {}
        if a.matmul_precision == "default" and os.environ.get("MLX_ENABLE_TF32", "1") != "1":
            raise ValueError("default comparison requires the original backend-default precision")
        record["fitWorker"] = guarded(command, env, a.output / "fit-worker", 3600, 6*1024**3, stop)
        atomic_json(a.output / "manifest.json", record)
        base.verify(a, "C1", stop)
        record["studentManifest"] = dict(path=str(a.output / "C1" / "manifest.json"),
            sha256=sha256(a.output / "C1" / "manifest.json"))
        if stop.is_set() or any(sha256(Path(path)) != digest for path, digest in pinned.items()):
            raise ValueError("stopped or experiment dependencies changed")
        record["status"] = "complete"
    except Exception as error:
        record["status"] = "failed"; record["failure"] = str(error); stop.set()
        raise
    finally:
        timer.cancel(); record["elapsedSeconds"] = time.monotonic() - started
        record["systemMemoryGuard"] = pressure.finish(); atomic_json(a.output / "manifest.json", record)
    print(json.dumps(dict(status=record["status"], seconds=record["elapsedSeconds"])), flush=True)


if __name__ == "__main__": main()
