"""Bounded native64/256 (optional 1024) comparison at one frozen turn belief.

The probe measures finite-reference drift, not flop-action EV or whole-game
exploitability. It never edits the source policy or promotes a label budget.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import threading
import time

from run_native_value_pilot import test_command
from run_native_value_preflight import atomic_json, sha256
from run_postflop_gap_pilot import (
    PilotMemoryGuard, SERIAL_SOLVE_MEMORY, controller_lock, run_job,
)

HERE = Path(__file__).resolve().parent
TEST = "blueprint::public_belief::counterfactual_turn::flop_pilot::continuation::saved_fixed_belief_native_label_probe"
STAGE_SECONDS = 4 * 3600


def label_drift(reference, challenger):
    if (reference.get("schema") != "hu-fixed-belief-native-label-probe-v1"
            or challenger.get("schema") != reference["schema"]
            or reference["candidateSha256"] != challenger["candidateSha256"]
            or reference["beliefSha256"] != challenger["beliefSha256"]
            or (reference["turn"], reference["leafIndex"], reference["publicHistory"])
            != (challenger["turn"], challenger["leafIndex"], challenger["publicHistory"])
            or reference["nativeIterations"] >= challenger["nativeIterations"]
            or reference["ownReach"] != challenger["ownReach"]
            or reference["opponentCompatibleMass"] != challenger["opponentCompatibleMass"]):
        raise ValueError("native budgets do not share an identical fixed belief")
    per_seat = []
    for seat in (0, 1):
        reach = reference["ownReach"][seat]
        mass = reference["opponentCompatibleMass"][seat]
        a = reference["counterfactualBb"][seat]
        b = challenger["counterfactualBb"][seat]
        if any(len(row) != 1326 for row in (reach, mass, a, b)):
            raise ValueError("invalid exact-combo label shape")
        total = squared = absolute = 0.0
        for own, opponent, low, high in zip(reach, mass, a, b):
            if not all(math.isfinite(x) for x in (own, opponent, low, high)) or own < 0 or opponent < 0:
                raise ValueError("nonfinite or negative label mass")
            weight = own * opponent
            if weight <= 0:
                continue
            delta = (high - low) / opponent
            total += weight
            squared += weight * delta * delta
            absolute += weight * abs(delta)
        if total <= 0:
            raise ValueError("fixed belief has zero joint reach")
        per_seat.append(dict(jointMass=total, weightedConditionalRmseBb=math.sqrt(squared/total),
                             weightedConditionalMaeBb=absolute/total))
    return dict(referenceIterations=reference["nativeIterations"],
                challengerIterations=challenger["nativeIterations"],
                beliefSha256=reference["beliefSha256"], perSeat=per_seat,
                referenceConditionalResponseGainBb=reference["conditionalResponseGainBb"],
                challengerConditionalResponseGainBb=challenger["conditionalResponseGainBb"],
                interpretation="Same fixed turn belief; reach-weighted conditional value drift against a finite native budget, not flop action-EV error or an equilibrium proof")


def run(args):
    binary, candidate, output = args.binary.resolve(), args.candidate.resolve(), args.output.resolve()
    if sha256(binary) != args.binary_sha256 or sha256(candidate) != args.candidate_sha256:
        raise ValueError("pinned binary or candidate changed")
    if output.exists() and not (output / "manifest.json").is_file():
        raise ValueError("output exists without resumable manifest")
    output.mkdir(parents=True, exist_ok=True)
    sources = {str(path): sha256(path) for path in (
        Path(__file__), HERE / "run_postflop_gap_pilot.py", HERE / "run_native_value_pilot.py",
        HERE / "worker_resources.py",
        HERE.parent / "src/blueprint/public_belief/counterfactual_turn/flop_pilot/continuation.rs")}
    budgets = [64, 256] if not args.include_1024 else [64, 256, 1024]
    identity = dict(schema="fixed-native-turn-label-budget-probe-v1", binarySha256=args.binary_sha256,
                    candidateSha256=args.candidate_sha256, turn=args.turn, leafIndex=args.leaf_index,
                    budgets=budgets, sourceHashes=sources, releaseAccepted=False)
    manifest = output / "manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("fixed label probe identity changed on resume")
    record = {**identity, "status": "running", "cases": [], "comparisons": []}
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    timer = threading.Timer(STAGE_SECONDS, stop.set)
    timer.daemon = True
    started = time.monotonic()
    with controller_lock(output):
        atomic_json(manifest, record)
        timer.start()
        memory_guard = None
        try:
            memory_guard = PilotMemoryGuard(stop, output / "system-memory.json").start()
            cases = []
            for budget in budgets:
                if stop.is_set():
                    raise ValueError("fixed belief probe stopped")
                if budget == 256 and cases and cases[0]["workerSeconds"] * 4 * 1.5 > 3600:
                    raise ValueError("native256 cost projection exceeds one-hour worker limit")
                if budget == 1024 and cases[-1]["workerSeconds"] * 4 * 1.5 > 10800:
                    raise ValueError("native1024 cost projection exceeds three-hour worker limit")
                work = output / f"native-{budget}"
                work.mkdir(exist_ok=True)
                result = work / "label.json"
                env = dict(POKER_FIXED_BELIEF_CANDIDATE=str(candidate),
                           POKER_FIXED_BELIEF_CANDIDATE_SHA=args.candidate_sha256,
                           POKER_FIXED_BELIEF_TURN=str(args.turn),
                           POKER_FIXED_BELIEF_LEAF_INDEX=str(args.leaf_index),
                           POKER_FIXED_BELIEF_ITERATIONS=str(budget),
                           POKER_FIXED_BELIEF_OUTPUT=str(result))
                receipt = run_job(binary, TEST, env, work / "solve", [result], stop,
                                  seconds={64: 900, 256: 3600, 1024: 10800}[budget],
                                  memory=SERIAL_SOLVE_MEMORY)
                value = json.loads(result.read_text())
                if (value["candidateSha256"] != args.candidate_sha256
                        or value["turn"] != args.turn or value["leafIndex"] != args.leaf_index
                        or value["nativeIterations"] != budget):
                    raise ValueError("fixed native label identity changed")
                row = dict(budget=budget, labelSha256=sha256(result),
                           workerSeconds=receipt["worker"]["workerElapsedSeconds"],
                           peakMemoryBytes=receipt["worker"]["sampledPeakMemoryBytes"],
                           beliefSha256=value["beliefSha256"],
                           conditionalResponseGainBb=value["conditionalResponseGainBb"])
                cases.append(row)
                record["cases"] = cases
                if len(cases) >= 2:
                    prior = json.loads((output / f"native-{budgets[len(cases)-2]}" / "label.json").read_text())
                    record["comparisons"].append(label_drift(prior, value))
                atomic_json(manifest, record)
                print(json.dumps(dict(event="fixed-label-budget", **row)), flush=True)
            if (stop.is_set() or memory_guard.reason
                    or sha256(binary) != args.binary_sha256
                    or sha256(candidate) != args.candidate_sha256
                    or any(sha256(Path(path)) != digest for path, digest in sources.items())):
                raise ValueError("fixed label probe stopped or input changed")
            record["status"] = "complete"
        except (OSError, ValueError, KeyError, AssertionError) as error:
            stop.set()
            record["status"], record["failure"] = "failed", str(error)
        finally:
            timer.cancel()
            record["elapsedSeconds"] = time.monotonic() - started
            if memory_guard:
                record["systemMemoryGuard"] = memory_guard.finish()
            atomic_json(manifest, record)
    print(json.dumps(dict(status=record["status"], failure=record.get("failure"),
                          cases=record["cases"], comparisons=record["comparisons"])), flush=True)
    if record["status"] != "complete":
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--turn", type=int, required=True)
    parser.add_argument("--leaf-index", type=int, required=True)
    parser.add_argument("--include-1024", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
