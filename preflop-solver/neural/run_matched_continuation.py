"""Matched 2x2 continuation allocation screen; no policy training or release.

Two/four fitting boards x 128/64 or 256/128 continuation budgets. Every choice
is evaluated on the same four separate 256/128 development boards. Only the
existing fold/call probability mass at two disjoint roots is reassigned; all
other actions and the reference continuation ranges stay fixed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import signal
import threading
import time

import numpy as np
from run_native_value_pilot import guarded, test_command
from run_native_value_preflight import atomic_json, sha256

TEST = "blueprint::preflop_continuation::matched_continuation::matched_call_fold_capture"


def summarize(captures):
    by_key = {(c["strength"], c["boardIndex"]): c for c in captures}
    expected = {(s, b) for s in (1, 2) for b in range(4)} | {(2, b) for b in range(4, 8)}
    if len(by_key) != len(captures) or set(by_key) != expected:
        raise ValueError("missing/duplicate matched allocation cell")
    first = by_key[1, 0]
    weights = np.asarray(first["multiplicities"], float) / 1326
    if weights.shape != (169,) or abs(weights.sum() - 1) > 1e-12 or (weights <= 0).any():
        raise ValueError("invalid private-class weights")
    for c in captures:
        if (c.get("schema") != "matched-call-fold-capture-v1" or c.get("releaseAccepted") is not False
                or c.get("fullGameGateEvaluated") is not False or len(c["records"]) != 2
                or any(c[k] != first[k] for k in ("preflopSha256", "modelSha256", "kernelSha256", "classes", "multiplicities", "chanceSeed"))
                or len(set(c["board"] + [c["turn"]])) != 4):
            raise ValueError("mixed identity or invalid chance")
        if c["boardIndex"] < 4 and any(c[k] != by_key[1, c["boardIndex"]][k] for k in ("board", "turn")):
            raise ValueError("chance not matched across solve budgets")
        for i, row in enumerate(c["records"]):
            reference = first["records"][i]
            if any(row[k] != reference[k] for k in ("history", "seat", "ownPrefixReach", "foldProbability", "callProbability", "foldCfvBb")):
                raise ValueError("frozen reference changed")
            for k in ("callCfvBb", "foldCfvBb", "ownPrefixReach", "foldProbability", "callProbability"):
                v = np.asarray(row[k], float)
                if v.shape != (169,) or not np.isfinite(v).all(): raise ValueError("invalid vector")
                if k not in ("callCfvBb", "foldCfvBb") and ((v < 0).any() or (v > 1+1e-12).any()):
                    raise ValueError("invalid probability/reach")
            if np.max(np.asarray(row["callProbability"]) + row["foldProbability"]) > 1+1e-12:
                raise ValueError("eligible probability exceeds one")
    cells = []
    for strength in (1, 2):
        for count in (2, 4):
            choices = []
            for root in range(2):
                # Average public chance BEFORE selecting an observable action.
                gap = np.mean([np.asarray(by_key[strength, b]["records"][root]["callCfvBb"])
                               - by_key[strength, b]["records"][root]["foldCfvBb"] for b in range(count)], axis=0)
                choices.append(gap > 0)
            board_gains = []
            for b in range(4, 8):
                gain = 0.0
                for i, row in enumerate(by_key[2, b]["records"]):
                    call, fold = np.asarray(row["callCfvBb"]), np.asarray(row["foldCfvBb"])
                    cp, fp = np.asarray(row["callProbability"]), np.asarray(row["foldProbability"])
                    selected = np.where(choices[i], call, fold)
                    gain += float(weights @ (np.asarray(row["ownPrefixReach"]) *
                                  ((cp + fp) * selected - cp * call - fp * fold)))
                board_gains.append(gain)
            cells.append(dict(strength=strength, fittingBoards=count, evaluationGainsBb=board_gains,
                meanEvaluationGainBb=float(np.mean(board_gains)),
                boardClusterStandardErrorBb=float(np.std(board_gains, ddof=1)/2),
                choices=[c.astype(int).tolist() for c in choices]))
    return dict(cells=cells, fitBoardIndices=list(range(4)), evaluationBoardIndices=list(range(4, 8)),
        releaseAccepted=False, fullGameGateEvaluated=False,
        interpretation="Higher gain is better for the same restricted fold/call adjustment under a common finite-budget continuation reference. Not full-game exploitability, a bound, or a deployed policy. Four evaluation board clusters, not 169 classes, are independent.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--binary-sha256", required=True)
    p.add_argument("--reference", type=Path, action="append", required=True)
    p.add_argument("--reference-sha256", action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args()
    if len(args.reference) != 2 or len(args.reference_sha256) != 2:
        raise ValueError("two pinned seed references required")
    pinned = {str(args.binary.resolve()): args.binary_sha256}
    envs = []
    for path, digest in zip(args.reference, args.reference_sha256):
        path = path.resolve()
        if sha256(path) != digest: raise ValueError("reference changed")
        pinned[str(path)] = digest
        m = json.loads(path.read_text())
        if m["status"] != "complete": raise ValueError("reference incomplete")
        old = m["jobs"][0]["worker"]["environment"]
        env = {k: v for k, v in old.items() if k in ("POKER_NOISE_PREFLOP", "POKER_NOISE_PREFLOP_SHA",
            "POKER_COMPACT_MODEL", "POKER_COMPACT_MODEL_SHA", "POKER_COMPACT_CHECKDOWN", "POKER_COMPACT_CHECKDOWN_SHA")}
        for key in ("POKER_NOISE_PREFLOP", "POKER_COMPACT_MODEL", "POKER_COMPACT_CHECKDOWN"):
            pinned[env[key]] = env[key+"_SHA"]
        envs.append(env)
    for name in ("run_native_value_pilot.py", "run_native_value_preflight.py", "worker_resources.py"):
        path = Path(__file__).with_name(name).resolve(); pinned[str(path)] = sha256(path)
    if any(sha256(Path(k)) != v for k, v in pinned.items()): raise ValueError("pinned input changed")
    output = args.output.resolve(); output.mkdir(exist_ok=False)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: stop.set())
    timer = threading.Timer(7200, stop.set); timer.daemon = True; timer.start()
    started = time.monotonic()
    record = dict(schema="matched-continuation-controller-v1", status="running", pinnedInputs=pinned,
        runnerSha256=sha256(Path(__file__)), jobs=[], releaseAccepted=False, fullGameGateEvaluated=False,
        maximumStageSeconds=7200, maximumConcurrentWorkers=2, maximumWorkerMemoryBytes=2*1024**3,
        preflightOnly=args.preflight_only)
    atomic_json(output/"manifest.json", record)
    def job(seed, strength, board):
        if stop.is_set() or any(sha256(Path(k)) != v for k, v in pinned.items()): raise ValueError("stopped or changed inputs")
        name = f"seed{seed}-s{strength}-b{board}"
        target = output/(name+".json")
        worker = guarded(test_command(args.binary.resolve(), TEST), {**envs[seed],
            "POKER_MATCH_BOARD": str(board), "POKER_MATCH_STRENGTH": str(strength),
            "POKER_COMPACT_OUTPUT": str(target)}, output/(name+"-worker"), 900, 2*1024**3, stop)
        value = json.loads(target.read_text())
        if value["boardIndex"] != board or value["strength"] != strength: raise ValueError("wrong output cell")
        return value, dict(seed=seed, strength=strength, board=board, output=str(target), outputSha256=sha256(target), worker=worker)
    values = [[], []]
    try:
        # Strongest cell establishes the actual cost and memory envelope first.
        value, report = job(0, 2, 0)
        values[0].append(value); record["jobs"].append(report)
        projected = 24*report["worker"]["workerElapsedSeconds"]/2*1.5
        record.update(preflightPassed=True, conservativeProjectedSeconds=projected)
        atomic_json(output/"manifest.json", record)
        print(json.dumps(dict(event="matched-preflight", seconds=value["seconds"], projectedSeconds=projected)), flush=True)
        if projected > 7200: raise ValueError("measured cost exceeds bounded two-hour stage; reconsider allocation")
        if not args.preflight_only:
            jobs = [(seed, strength, board) for seed in range(2) for strength in (1, 2)
                    for board in range(4 if strength == 1 else 8) if (seed, strength, board) != (0, 2, 0)]
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(job, *j) for j in jobs]
                try:
                    for f in as_completed(futures):
                        value, report = f.result(); values[report["seed"]].append(value); record["jobs"].append(report)
                        atomic_json(output/"manifest.json", record)
                        print(json.dumps(dict(event="matched-cell", seed=report["seed"], strength=report["strength"],
                            board=report["board"], completed=len(record["jobs"]), seconds=value["seconds"])), flush=True)
                except Exception:
                    stop.set(); raise
            record["summaries"] = [summarize(v) for v in values]
        if stop.is_set() or any(sha256(Path(k)) != v for k, v in pinned.items()): raise ValueError("stopped or changed inputs")
        record["status"] = "complete"
    except (ValueError, OSError, KeyError) as error:
        stop.set(); record.update(status="failed", failure=str(error))
    finally:
        timer.cancel(); record["elapsedSeconds"] = time.monotonic()-started
        atomic_json(output/"manifest.json", record)
    print(json.dumps({k: record.get(k) for k in ("status", "elapsedSeconds", "failure")}), flush=True)
    if record["status"] != "complete": raise SystemExit(1)


if __name__ == "__main__": main()
