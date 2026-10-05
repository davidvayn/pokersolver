"""Exact native8/native32 parallel construction preflight; no new evaluation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import threading
import time

from run_native_value_preflight import atomic_json, sha256
from run_postflop_gap_pilot import HERE, PREFIX, PilotMemoryGuard, controller_lock, run_job, same_public_state


def construction_environment(root, digest, output, iterations, environ=None):
    inherited = os.environ if environ is None else environ
    # Reject rather than silently override stale research environment.
    if any(key.startswith("POKER_NATIVE_FLOP_") for key in inherited):
        raise ValueError("native parallel preflight cannot inherit flop settings")
    if iterations not in (8, 32):
        raise ValueError("only the predeclared eight/32-update cost stages are allowed")
    return dict(POKER_NATIVE_FLOP_INPUT=str(root), POKER_NATIVE_FLOP_INPUT_SHA=digest,
        POKER_NATIVE_FLOP_SEED="100101", POKER_NATIVE_FLOP_ITERATIONS=str(iterations),
        POKER_NATIVE_FLOP_TURN_ITERATIONS="64", POKER_NATIVE_FLOP_CHANCE_BASELINE="none",
        POKER_NATIVE_FLOP_TURN_SAMPLES="1", POKER_NATIVE_FLOP_LEAF_WORKERS="4",
        POKER_NATIVE_FLOP_OUTPUT=str(output))


def projection_passes(seconds):
    if not 0 < seconds < float("inf"):
        raise ValueError("finite positive completed construction time required")
    return seconds * 4 * 1.25 < 1800


def check_policy(policy, original, iterations):
    if (policy.get("schema") != "hu-native-counterfactual-turn-flop-pilot-v1"
            or not same_public_state(policy["state"], original["public"])
            or policy["game"] != original["game"] or policy["seed"] != 100101
            or policy["iterations"] != iterations or policy["turn_iterations"] != 64
            or policy.get("response_turn_iterations") is not None
            or policy.get("learned_leaf_model_sha256") is not None
            or policy.get("leaf_schedule") is not None or not policy["strategies"]):
        raise ValueError("construction changed frozen root, budgets, model or schedule")


def run(args):
    binary, root, output = args.binary.resolve(), args.root.resolve(), args.output.resolve()
    pinned = {}
    for name in ("binary", "root", "reference8", "reference32", "build"):
        path = getattr(args, name).resolve()
        digest = getattr(args, name + "_sha256")
        if sha256(path) != digest:
            raise ValueError(name + " changed")
        pinned[str(path)] = digest
    build = json.loads(args.build.read_text())
    if build.get("binary") != str(binary) or build.get("binarySha256") != args.binary_sha256:
        raise ValueError("source snapshot belongs to a different binary")
    sources = build["rustSources"]
    if not sources or any(sha256(Path(p)) != h for p, h in sources.items()):
        raise ValueError("frozen Rust source snapshot changed")
    pinned.update(sources)
    refs = {}
    original = json.loads(root.read_text())
    for iterations in (8, 32):
        reference = getattr(args, "reference" + str(iterations)).resolve()
        refs[iterations] = pinned[str(reference)]
        check_policy(json.loads(reference.read_text()), original, iterations)
    for name in (Path(__file__).name, "run_postflop_gap_pilot.py", "run_native_value_preflight.py",
                 "run_native_value_pilot.py", "worker_resources.py"):
        path = HERE / name
        pinned[str(path)] = sha256(path)
    if output.exists():
        raise ValueError("never overwrite a completed or interrupted cost preflight")
    output.mkdir()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    timer = threading.Timer(3000, stop.set)
    timer.daemon = True
    started = time.monotonic()
    record = dict(schema="native-parallel-construction-cost-v1", status="running", cases=[],
        pinnedInputs=pinned, leafWorkers=4, maximumSeconds=3000, constructionMemoryBytes=4*1024**3,
        releaseAccepted=False, interpretation="Exact implementation cost/parity only; no new policy-quality result")
    with controller_lock(output):
        pressure = None
        try:
            atomic_json(output / "manifest.json", record)
            timer.start()
            pressure = PilotMemoryGuard(stop, output / "system-memory.json").start()
            for iterations, limit in ((8, 900), (32, 1800)):
                candidate = output / f"candidate-{iterations}.json"
                receipt = run_job(binary, PREFIX + "saved_20bb_native_flop_pilot",
                    construction_environment(root, pinned[str(root)], candidate, iterations),
                    output / f"solve-{iterations}", [candidate], stop, seconds=limit, memory=4*1024**3)
                check_policy(json.loads(candidate.read_text()), original, iterations)
                digest = sha256(candidate)
                if digest != refs[iterations]:
                    raise ValueError(f"parallel native{iterations} candidate is not byte-identical to serial control")
                worker = receipt["worker"]
                row = dict(iterations=iterations, candidateSha256=digest, parityPassed=True,
                    solveSeconds=worker["workerElapsedSeconds"], peakMemoryBytes=worker["sampledPeakMemoryBytes"])
                record["cases"].append(row)
                atomic_json(output / "manifest.json", record)
                print(json.dumps(dict(event="native-parallel-cost", **row)), flush=True)
                if iterations == 8 and not projection_passes(row["solveSeconds"]):
                    record.update(status="cost_rejected", projected32Seconds=row["solveSeconds"]*5)
                    break
            if stop.is_set() or any(sha256(Path(p)) != h for p, h in pinned.items()):
                raise ValueError("stopped or pinned input/source changed")
            if record["status"] != "cost_rejected":
                record["status"] = "complete"
        except BaseException as error:
            stop.set()
            record.update(status="failed", failure=str(error))
            raise
        finally:
            timer.cancel()
            record["elapsedSeconds"] = time.monotonic()-started
            if pressure is not None:
                record["systemMemoryGuard"] = pressure.finish()
            atomic_json(output / "manifest.json", record)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "root", "reference8", "reference32", "build"):
        p.add_argument("--" + name, type=Path, required=True)
        p.add_argument("--" + name + "-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
