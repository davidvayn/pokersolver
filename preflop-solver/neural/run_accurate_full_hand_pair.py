"""One fixed-deal native32/native64 complete-hand LBR cost/response pilot."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import threading
import time

from run_native_full_hand_preflight import DEFAULT_POLICY_SHA, expected_route_sha256, validate_probe
from run_native_value_preflight import atomic_json, sha256
from run_postflop_gap_pilot import HERE, PilotMemoryGuard, controller_lock, run_job

TEST = "blueprint::response::native_policy::full_hand_probe::accurate_native_full_hand_lbr_probe"


def validate_cluster(row, preflop_sha, iterations, index=0, seed=100101):
    d, r = row["diagnostics"], row["report"]
    if (row["schema"] != "accurate-native-full-hand-lbr-cluster-v1" or row["releaseAccepted"] is not False
            or row["cohort"] != "screening" or row["index"] != index
            or d["preflopSha256"] != preflop_sha or d["preflopRounds"] != 32
            or d["learnedLeafModelSha256"] is not None or d["completeRootSupport"] is not True
            or d["rootRealizationTurnAverages"] is not False or d["safeResolving"] is not False
            or d["releaseQualified"] is not False or d["executionLeafWorkers"] != 4
            or d["flopIterations"] != iterations or d["trainingTurnIterations"] != 64
            or d["responseTurnIterations"] != 64
            or d["routeSha256"] != expected_route_sha256(preflop_sha, iterations, seed)
            or r["schema"] != "paired-frozen-lbr-hand-v1" or r["lbrSeed"] != 90001
            or r["earlyRunoutsPerCombo"] != 16 or len(r["attacks"]) != 2
            or len(r["seatAttackerUtilityBb"]) != 2):
        raise ValueError("complete-hand route/deal/attacker/budget identity differs")
    cards = [c for holding in row["deal"]["holes"] for c in holding] + row["deal"]["board"]
    if (len(row["deal"]["holes"]) != 2 or any(len(h) != 2 for h in row["deal"]["holes"])
            or len(row["deal"]["board"]) != 5 or len(cards) != 9 or len(set(cards)) != 9
            or any(not isinstance(c, int) or isinstance(c, bool) or not 0 <= c < 52 for c in cards)):
        raise ValueError("complete deal must contain nine unique exact cards")
    values, gain = r["seatAttackerUtilityBb"], r["pairedTotalResponseGainBb"]
    if (any(not math.isfinite(v) or abs(v) > 20.000001 for v in values)
            or not math.isfinite(gain) or abs(gain-math.fsum(values)) > 1e-10):
        raise ValueError("invalid total-scale paired payoff")
    postflop = 0
    for attack, utility in zip(r["attacks"], values):
        counts = attack["decisions"]
        if (len(counts) != 4 or any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in counts)
                or sum(counts) != len(attack["actions"]) or not math.isfinite(attack["utility"])
                or abs(attack["utility"]-utility) > 1e-10):
            raise ValueError("incomplete attacker trajectory/payoff")
        postflop += sum(counts[1:])
        for action in attack["actions"]:
            if action["action"] not in action["legalActions"]:
                raise ValueError("attacker selected an illegal action")
    return dict(totalResponseGainBbPerHand=gain, postflopAttackerDecisions=postflop,
        routeSha256=d["routeSha256"])


def compare(rows, preflop_sha, index=0, seed=100101):
    if len(rows) != 2: raise ValueError("both complete matched arms required")
    a, b = rows
    for key in ("cohort", "index", "chanceSeed", "deal"):
        if a[key] != b[key]: raise ValueError("cannot compare different full-deal clusters")
    if a["report"]["actionSeed"] != b["report"]["actionSeed"]:
        raise ValueError("attacker action randomness must be matched")
    first, second = (validate_cluster(row, preflop_sha, budget, index, seed) for row, budget in zip(rows, (32, 64)))
    return dict(dealClusters=1, native32=first, native64=second,
        pairedImprovementBbPerHand=first["totalResponseGainBbPerHand"]-second["totalResponseGainBbPerHand"],
        confidence="One fixed deal only; no strength decision or uncertainty bound",
        fullGameGateEvaluated=False, releaseAccepted=False,
        interpretation="Total response gain, not half-scale exploitability; frozen heuristic LBR is not an upper-bound certificate")


def expansion(index, has_first):
    if index not in range(4) or has_first != (index > 0):
        raise ValueError("only declared indices 0..3; expansion requires the completed first pair")
    return 100101 if index % 2 == 0 else 100102


def pin_first_pair(path, prior, pinned, binary, preflop):
    if (prior.get("schema") != "accurate-native-full-hand-pair-v1" or prior.get("status") != "complete"
            or prior.get("cohort") != "screening" or prior.get("index") != 0 or prior.get("policySeed") != 100101
            or prior["pinnedInputs"].get(str(binary)) != pinned[str(binary)]
            or prior["pinnedInputs"].get(str(preflop)) != pinned[str(preflop)]
            or len(prior.get("jobs", [])) != 2 or prior["systemMemoryGuard"].get("stopReason")
            or not math.isfinite(prior.get("elapsedSeconds", math.nan)) or not 0 < prior["elapsedSeconds"] <= 7200):
        raise ValueError("complete compatible first-pair feasibility receipt required")
    rows = []
    for job, iterations in zip(prior["jobs"], (32,64)):
        worker = job["worker"]
        if (job["iterations"] != iterations or worker.get("status") != "complete"
                or worker.get("exitCode") != 0 or worker.get("resourceStopReason")
                or not 0 < worker.get("sampledPeakMemoryBytes", math.inf) <= 2*1024**3):
            raise ValueError("first pair must pass both original worker/resource guards")
        report = path.parent / f"native{iterations}.json"
        if job["output"] != str(report): raise ValueError("first-pair report path differs")
        pinned[str(report)] = job["outputSha256"]
        row = json.loads(report.read_text())
        if job["summary"] != validate_cluster(row, pinned[str(preflop)], iterations):
            raise ValueError("first-pair output summary differs")
        rows.append(row)
    if prior["summary"] != compare(rows, pinned[str(preflop)]):
        raise ValueError("first-pair matched comparison differs")


def run(args):
    if any(k.startswith("POKER_NATIVE_") for k in os.environ):
        raise ValueError("cannot inherit another native policy configuration")
    pinned = {}
    seed = expansion(args.index, args.first_pair is not None)
    if (args.first_pair is None) != (args.first_pair_sha256 is None):
        raise ValueError("first pair and SHA must be supplied together")
    for name in ("binary", "build", "default_parity", "preflight", "preflop"):
        path = getattr(args, name).resolve(); digest = getattr(args, name+"_sha256")
        if sha256(path) != digest: raise ValueError(name+" changed")
        setattr(args, name, path); pinned[str(path)] = digest
    if args.first_pair is not None:
        path = args.first_pair.resolve(); pinned[str(path)] = args.first_pair_sha256
        if sha256(path) != args.first_pair_sha256: raise ValueError("first pair changed")
        pin_first_pair(path, json.loads(path.read_text()), pinned, args.binary, args.preflop)
    cost = json.loads(args.preflight.read_text()); probe = args.preflight.parent / "probe.json"
    pinned[str(probe)] = cost["probeSha256"]; worker = cost["worker"]
    if (cost["schema"] != "accurate-native-full-hand-cost-preflight-v1" or cost["status"] != "complete"
            or cost["pinnedInputs"].get(str(args.preflop)) != args.preflop_sha256
            or worker["status"] != "complete" or worker["exitCode"] != 0 or worker["resourceStopReason"]
            or not 0 < worker["workerElapsedSeconds"] <= 3600 or cost["systemMemoryGuard"]["stopReason"]):
        raise ValueError("completed compatible accurate full-hand cost probe required")
    if cost["summary"] != validate_probe(json.loads(probe.read_text()), args.preflop_sha256):
        raise ValueError("cost probe summary/route changed")
    build = json.loads(args.build.read_text()); default = json.loads(args.default_parity.read_text())
    if build["binary"] != str(args.binary) or build["binarySha256"] != args.binary_sha256 or not build["rustSources"]:
        raise ValueError("new binary/source snapshot required")
    pinned.update(build["rustSources"])
    w = default["worker"]
    if (default["status"] != "complete" or default["parityPassed"] is not True
            or default["binarySha256"] != args.binary_sha256 or default["candidateSha256"] != DEFAULT_POLICY_SHA
            or w["status"] != "complete" or w["exitCode"] != 0 or w["resourceStopReason"]):
        raise ValueError("rebuilt default policy parity failed")
    pinned[str(args.default_parity.parent / "candidate.json")] = DEFAULT_POLICY_SHA
    for name in (Path(__file__).name, "run_native_full_hand_preflight.py", "run_native_update_quality.py",
                 "run_native_inner_budget_pilot.py", "run_postflop_gap_pilot.py", "run_native_value_preflight.py",
                 "run_native_value_pilot.py", "worker_resources.py"):
        path = HERE / name; pinned[str(path)] = sha256(path)
    if any(sha256(Path(p)) != h for p, h in pinned.items()): raise ValueError("pinned input/source changed")
    output = args.output.resolve()
    if output.exists(): raise ValueError("never overwrite a complete-hand pair")
    output.mkdir(); stop = threading.Event(); started = time.monotonic()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: stop.set())
    timer = threading.Timer(12000, stop.set); timer.daemon = True
    record = dict(schema="accurate-native-full-hand-pair-v1", status="running", pinnedInputs=pinned,
        cohort="screening", index=args.index, policySeed=seed, maximumSeconds=12000,
        firstPairSha256=args.first_pair_sha256, controllerPid=os.getpid(),
        maximumWorkerMemoryBytes=4*1024**3, jobs=[], releaseAccepted=False)
    with controller_lock(output):
        pressure = None; rows = []
        try:
            atomic_json(output / "manifest.json", record); timer.start()
            pressure = PilotMemoryGuard(stop, output / "system-memory.json").start()
            for iterations, seconds in ((32, 4500), (64, 7200)):
                path = output / f"native{iterations}.json"
                receipt = run_job(args.binary, TEST, dict(POKER_NATIVE_CHECKPOINT=str(args.preflop),
                    POKER_NATIVE_CHECKPOINT_SHA=args.preflop_sha256, POKER_NATIVE_FLOP_ITERATIONS=str(iterations),
                    POKER_NATIVE_POLICY_SEED=str(seed), POKER_NATIVE_LBR_COHORT="screening", POKER_NATIVE_LBR_INDEX=str(args.index),
                    POKER_NATIVE_OUTPUT=str(path)), output / f"native{iterations}", [path], stop,
                    seconds=seconds, memory=4*1024**3)
                row = json.loads(path.read_text()); summary = validate_cluster(row, args.preflop_sha256, iterations, args.index, seed)
                rows.append(row)
                record["jobs"].append(dict(iterations=iterations, output=str(path), outputSha256=sha256(path),
                    worker=receipt["worker"], summary=summary))
                atomic_json(output / "manifest.json", record)
                print(json.dumps(dict(event="accurate-full-hand-arm", iterations=iterations, **summary)), flush=True)
            if stop.is_set() or any(sha256(Path(p)) != h for p, h in pinned.items()):
                raise ValueError("stopped or pinned source changed")
            record.update(status="complete", summary=compare(rows, args.preflop_sha256, args.index, seed))
            print(json.dumps(dict(event="accurate-full-hand-pair", **record["summary"])), flush=True)
        except BaseException as error:
            stop.set(); record.update(status="failed", failure=str(error)); raise
        finally:
            timer.cancel(); record["elapsedSeconds"] = time.monotonic()-started
            if pressure is not None: record["systemMemoryGuard"] = pressure.finish()
            atomic_json(output / "manifest.json", record)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=int, choices=range(4), default=0)
    p.add_argument("--first-pair", type=Path); p.add_argument("--first-pair-sha256")
    for name in ("binary", "build", "default_parity", "preflight", "preflop"):
        option = name.replace("_", "-")
        p.add_argument("--"+option, type=Path, required=True); p.add_argument("--"+option+"-sha256", required=True)
    p.add_argument("--output", type=Path, required=True); run(p.parse_args())


if __name__ == "__main__": main()
