"""Pinned accurate64 full-hand integration/cost probe, never a strength gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import threading
import time

from run_native_inner_budget_pilot import controls
from run_native_update_quality import first_verdict, paired_verdict, pin_screen
from run_native_value_preflight import atomic_json, sha256
from run_postflop_gap_pilot import HERE, PilotMemoryGuard, controller_lock, run_job, select_cases

TEST = "blueprint::response::native_policy::pilot::compact_native_full_hand_candidate_serving_probe"
DEFAULT_POLICY_SHA = "f1faa095fc4b905786d446f32bf9927db645fb8008258187e49b33d812e1fe4d"


def expected_route_sha256(preflop_sha, flop_iterations=64, seed=100101):
    payload = dict(schema="research-pinned-full-hand-route-v1", preflopSha256=preflop_sha,
        learnedLeafModelSha256=None, seed=seed, flopIterations=flop_iterations, trainingTurnIterations=64,
        playedTurnIterations=64, rootSeedRule="seed-xor-first8-le-sha256-public-input-v1",
        playedContinuation="frozen-native-turn-river-average-v1",
        rootSupport="all-board-legal-own-realization-average-v1",
        beliefReplay="true-reaches-from-ones-normalize-once-v1")
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_pair(path, record, pinned, binary, baseline, old, spot):
    first_paths = [Path(p) for p, h in record["pinnedInputs"].items()
                   if h == record.get("firstScreenSha256")]
    if len(first_paths) != 1:
        raise ValueError("pair omits its first-screen identity")
    first_path = first_paths[0]
    pinned[str(first_path)] = record["firstScreenSha256"]
    first = json.loads(first_path.read_text())
    root = baseline / "inputs" / (spot + ".json")
    a, b = (old[(spot, seed, "native")] for seed in (100101, 100102))
    pin_screen(first_path, first, pinned, binary, root, a)
    pin_screen(path, record, pinned, binary, root, b)
    expected = paired_verdict(record["gainBb"], b["gainBb"], first["gainBb"], a["gainBb"], spot)
    if (first.get("result") != first_verdict(first["gainBb"], a["gainBb"])
            or record.get("result") != expected or not expected["promising"]):
        raise ValueError("both original paired quality criteria must pass")


def validate_probe(value, preflop_sha):
    d = value["diagnostics"]
    if (value.get("schema") != "native-full-hand-integration-probe-v1"
            or value["checkpointSha256"] != preflop_sha or d["preflopSha256"] != preflop_sha
            or value["cardsSeed"] != 881901 or d["learnedLeafModelSha256"] is not None
            or d["completeRootSupport"] is not True or d["rootRealizationTurnAverages"] is not False
            or d["route"] != "research-strict-preflop-native-flop-turn-river-v1"
            or d["safeResolving"] is not False or d["releaseQualified"] is not False
            or value["solveBudget"] != dict(flopIterations=64, trainingTurnIterations=64,
                responseTurnIterations=64, leafWorkers=4)
            or [d[k] for k in ("flopIterations", "trainingTurnIterations", "responseTurnIterations")] != [64]*3
            or d["executionLeafWorkers"] != 4 or d["cachedTurnSolves"] < 1):
        raise ValueError("probe route/model/budget/trajectory differs")
    if d["routeSha256"] != expected_route_sha256(preflop_sha):
        raise ValueError("native/Python full-hand route identity differs")
    queries, missing = value["preflopQueries"], value["preflopMissing"]
    error = value["maximumProbabilitySumError"]
    if (not isinstance(queries, int) or isinstance(queries, bool) or queries <= 0
            or not isinstance(missing, int) or isinstance(missing, bool) or not 0 <= missing <= queries
            or missing/queries > .0001 or value["preflopExhaustiveCoveragePass"] is not True
            or not math.isfinite(error) or not 0 <= error <= 1e-6):
        raise ValueError("frozen preflop coverage/probability check failed")
    rows = value["decisions"]
    if [str(r["street"]).lower() for r in rows] != [s for s in ("preflop", "flop", "turn", "river") for _ in (0, 1)]:
        raise ValueError("forced full-hand probe must cover all eight decisions")
    for row in rows:
        probabilities, seconds = row["probabilities"], row["seconds"]
        if (not probabilities or len(probabilities) != len(row["actions"])
                or row["forcedAction"] not in row["actions"]
                or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities)
                or abs(math.fsum(probabilities)-1) > 1e-6
                or not math.isfinite(seconds) or seconds < 0):
            raise ValueError("invalid served probabilities/action/cost")
    return dict(routeSha256=d["routeSha256"], preflopQueries=queries, preflopMissing=missing,
        maximumProbabilitySumError=error,
        streetSeconds=[dict(street=r["street"], actor=r["actor"], seconds=r["seconds"]) for r in rows],
        fullGameGateEvaluated=False, fullHandCoverageQualified=False, releaseAccepted=False)


def run(args):
    if any(k.startswith("POKER_NATIVE_") for k in os.environ):
        raise ValueError("cannot inherit a different research policy configuration")
    pinned = {}
    for name in ("binary", "build", "default_parity", "quality_build", "reference",
                 "limped_pair", "single_raised_pair", "preflop"):
        path = getattr(args, name).resolve(); digest = getattr(args, name + "_sha256")
        if sha256(path) != digest: raise ValueError(name + " changed")
        setattr(args, name, path); pinned[str(path)] = digest
    baseline = args.baseline_root.resolve()
    complete = baseline / "complete/manifest.json"
    if sha256(complete) != args.baseline_sha256: raise ValueError("baseline changed")
    pinned[str(complete)] = args.baseline_sha256
    protocol_path = baseline / "protocol.json"; pinned[str(protocol_path)] = sha256(protocol_path)
    protocol = json.loads(protocol_path.read_text()); protocol["sha256"] = pinned[str(protocol_path)]
    if protocol["pinnedInputs"].get(str(args.preflop)) != args.preflop_sha256:
        raise ValueError("preflop must be the unchanged benchmark checkpoint")
    old = controls(json.loads(args.reference.read_text()),
                   select_cases(protocol, json.loads(complete.read_text()), ["limped-paired", "single-raised-high-rainbow"]))
    quality = json.loads(args.quality_build.read_text()); quality_binary = Path(quality["binary"])
    pinned[str(quality_binary)] = quality["binarySha256"]
    for path, spot in ((args.limped_pair, "limped-paired"),
                       (args.single_raised_pair, "single-raised-high-rainbow")):
        verify_pair(path, json.loads(path.read_text()), pinned, quality_binary, baseline, old, spot)
    build = json.loads(args.build.read_text())
    if build["binary"] != str(args.binary) or build["binarySha256"] != args.binary_sha256 or not build["rustSources"]:
        raise ValueError("wrong or empty rebuilt source snapshot")
    pinned.update(build["rustSources"])
    default = json.loads(args.default_parity.read_text()); worker = default["worker"]
    if (default.get("status") != "complete" or default.get("parityPassed") is not True
            or default.get("binarySha256") != args.binary_sha256
            or default.get("candidateSha256") != DEFAULT_POLICY_SHA or worker.get("status") != "complete"
            or worker.get("exitCode") != 0 or worker.get("resourceStopReason")):
        raise ValueError("rebuilt default policy parity required")
    pinned[str(args.default_parity.parent / "candidate.json")] = DEFAULT_POLICY_SHA
    for name in (Path(__file__).name, "run_native_update_quality.py", "run_native_inner_budget_pilot.py",
                 "run_postflop_gap_pilot.py", "run_native_value_preflight.py", "run_native_value_pilot.py", "worker_resources.py"):
        path = HERE / name; pinned[str(path)] = sha256(path)
    if any(sha256(Path(p)) != h for p, h in pinned.items()): raise ValueError("input/source changed")
    output = args.output.resolve()
    if output.exists(): raise ValueError("never overwrite a full-hand cost probe")
    output.mkdir(); stop = threading.Event(); started = time.monotonic()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: stop.set())
    timer = threading.Timer(3900, stop.set); timer.daemon = True
    record = dict(schema="accurate-native-full-hand-cost-preflight-v1", status="running",
        pinnedInputs=pinned, maximumSeconds=3900, maximumWorkerSeconds=3600,
        maximumWorkerMemoryBytes=4*1024**3, releaseAccepted=False,
        interpretation="Forced check/call integration and cost only; not strength or full-hand coverage qualification")
    with controller_lock(output):
        pressure = None
        try:
            atomic_json(output / "manifest.json", record); timer.start()
            pressure = PilotMemoryGuard(stop, output / "system-memory.json").start()
            probe = output / "probe.json"
            receipt = run_job(args.binary, TEST, dict(POKER_NATIVE_CHECKPOINT=str(args.preflop),
                POKER_NATIVE_CHECKPOINT_SHA=args.preflop_sha256, POKER_NATIVE_COMPACT_CONTINUATION="1",
                POKER_NATIVE_OUTPUT=str(probe)), output / "worker", [probe], stop,
                seconds=3600, memory=4*1024**3)
            summary = validate_probe(json.loads(probe.read_text()), args.preflop_sha256)
            if stop.is_set() or any(sha256(Path(p)) != h for p, h in pinned.items()):
                raise ValueError("stopped or pinned source changed")
            record.update(status="complete", probeSha256=sha256(probe), worker=receipt["worker"], summary=summary)
            print(json.dumps(dict(event="native-full-hand-cost", **summary)), flush=True)
        except BaseException as error:
            stop.set(); record.update(status="failed", failure=str(error)); raise
        finally:
            timer.cancel(); record["elapsedSeconds"] = time.monotonic()-started
            if pressure is not None: record["systemMemoryGuard"] = pressure.finish()
            atomic_json(output / "manifest.json", record)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-root", type=Path, required=True); p.add_argument("--baseline-sha256", required=True)
    for name in ("binary", "build", "default_parity", "quality_build", "reference", "limped_pair", "single_raised_pair", "preflop"):
        option = name.replace("_", "-")
        p.add_argument("--"+option, type=Path, required=True); p.add_argument("--"+option+"-sha256", required=True)
    p.add_argument("--output", type=Path, required=True); run(p.parse_args())


if __name__ == "__main__": main()
