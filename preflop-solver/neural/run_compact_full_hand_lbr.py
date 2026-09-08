"""Guarded complete-hand LBR screening. No exploitability upper-bound claim."""
import argparse
import json
import math
from pathlib import Path
import signal
import threading
import time

from run_native_value_pilot import guarded, test_command
from run_native_value_preflight import atomic_json, sha256

TEST = "blueprint::response::native_policy::full_hand_probe::compact_native_full_hand_lbr_probe"


def summarize(rows):
    if not rows:
        raise ValueError("no completed full-hand clusters")
    identities, indices, gains = set(), set(), []
    for row in rows:
        d, r = row["diagnostics"], row["report"]
        key = row["cohort"], row["index"]
        if (row["schema"] != "compact-native-full-hand-lbr-cluster-v1"
                or row["releaseAccepted"] is not False or key in indices
                or d["completeRootSupport"] is not True
                or r["lbrSeed"] != 90001 or r["earlyRunoutsPerCombo"] != 16
                or len(r["attacks"]) != 2 or len(r["seatAttackerUtilityBb"]) != 2):
            raise ValueError("incomplete or duplicate full-hand cluster")
        values = r["seatAttackerUtilityBb"]
        gain = r["pairedTotalResponseGainBb"]
        if (any(not math.isfinite(v) or abs(v) > 20.000001 for v in values)
                or not math.isfinite(gain) or abs(gain - sum(values)) > 1e-10):
            raise ValueError("invalid paired payoff accounting")
        identities.add((d["routeSha256"], row["cohort"]))
        indices.add(key)
        gains.append(gain)
    if len(identities) != 1:
        raise ValueError("cannot pool different routes or evaluation cohorts")
    n = len(gains)
    mean = sum(gains) / n
    se = (math.sqrt(sum((v - mean)**2 for v in gains) / (n * (n - 1)))
          if n > 1 else None)
    return dict(dealClusters=n, totalResponseGainMeanBbPerHand=mean,
        dealClusterStandardErrorBb=se,
        diagnosticNormal99PercentInterval=(None if se is None else
            [mean - 2.5758293035489004*se, mean + 2.5758293035489004*se]),
        fullGameGateEvaluated=False, releaseAccepted=False,
        interpretation="Paired total, not half-scale exploitability. Normal interval is diagnostic, especially at tiny n. Negative/zero gains cannot certify equilibrium or attacker strength.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "preflop", "model"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--cohort", choices=["screening", "calibration", "holdout"], default="screening")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--hands", type=int, choices=[1, 2, 4, 8, 16, 32], default=1)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--preflight-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.start_index < args.start_index + args.hands <= 4096:
        raise ValueError("invalid hand cluster indices")
    pinned = {}
    for name in ("binary", "preflop", "model"):
        path = getattr(args, name).resolve()
        if sha256(path) != getattr(args, name + "_sha256"):
            raise ValueError("pinned input changed")
        setattr(args, name, path)
        pinned[str(path)] = sha256(path)
    if args.hands > 1 or args.cohort != "screening":
        if not args.preflight or sha256(args.preflight) != args.preflight_sha256:
            raise ValueError("larger screen requires pinned one-hand cost preflight")
        prior = json.loads(args.preflight.read_text())
        if (prior["status"] != "complete" or len(prior["jobs"]) != 1
                or any(prior["pinnedInputs"].get(p) != h for p, h in pinned.items())):
            raise ValueError("incompatible full-hand cost preflight")
        pinned[str(args.preflight.resolve())] = args.preflight_sha256
    for name in ("run_native_value_pilot.py", "run_native_value_preflight.py", "worker_resources.py"):
        path = Path(__file__).with_name(name).resolve()
        pinned[str(path)] = sha256(path)
    args.output = args.output.resolve()
    args.output.mkdir(exist_ok=False)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    record = dict(schema="compact-full-hand-lbr-controller-v1", status="running",
        pinnedInputs=pinned, runnerSha256=sha256(Path(__file__)), jobs=[],
        cohort=args.cohort, startIndex=args.start_index, requestedHands=args.hands,
        maximumWorkerSeconds=300, maximumWorkerMemoryBytes=2*1024**3,
        releaseAccepted=False)
    started = time.monotonic()
    atomic_json(args.output / "manifest.json", record)
    rows = []
    try:
        for index in range(args.start_index, args.start_index + args.hands):
            if stop.is_set() or any(sha256(Path(p)) != h for p, h in pinned.items()):
                raise ValueError("stopped or changed full-hand input")
            output = args.output / ("hand-%04d.json" % index)
            worker = guarded(test_command(args.binary, TEST), {
                "POKER_NATIVE_CHECKPOINT": str(args.preflop),
                "POKER_NATIVE_CHECKPOINT_SHA": args.preflop_sha256,
                "POKER_NATIVE_VALUE_MODEL": str(args.model),
                "POKER_NATIVE_VALUE_MODEL_SHA": args.model_sha256,
                "POKER_NATIVE_LBR_COHORT": args.cohort,
                "POKER_NATIVE_LBR_INDEX": str(index), "POKER_NATIVE_OUTPUT": str(output),
            }, args.output / ("worker-%04d" % index), 300, 2*1024**3, stop)
            row = json.loads(output.read_text())
            if (row["cohort"] != args.cohort or row["index"] != index
                    or row["diagnostics"]["preflopSha256"] != args.preflop_sha256
                    or row["diagnostics"]["learnedLeafModelSha256"] != args.model_sha256):
                raise ValueError("full-hand worker identity mismatch")
            rows.append(row)
            record["summary"] = summarize(rows)
            record["jobs"].append(dict(index=index, output=str(output),
                outputSha256=sha256(output), worker=worker))
            atomic_json(args.output / "manifest.json", record)
            print(json.dumps(dict(event="full-hand-cluster-complete", index=index,
                seconds=row["seconds"], totalGain=row["report"]["pairedTotalResponseGainBb"])), flush=True)
        if stop.is_set() or any(sha256(Path(p)) != h for p, h in pinned.items()):
            raise ValueError("stopped or changed full-hand input")
        record["status"] = "complete"
    except (OSError, ValueError, KeyError) as error:
        record.update(status="failed", failure=str(error))
    record["elapsedSeconds"] = time.monotonic() - started
    atomic_json(args.output / "manifest.json", record)
    print(json.dumps({k:record.get(k) for k in ("status", "elapsedSeconds", "summary", "failure")}), flush=True)
    if record["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
