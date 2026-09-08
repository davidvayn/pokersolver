"""Exact fixed-checkdown diagnostic for the shared preflop update code. No policy export."""
import argparse
import json
import math
from pathlib import Path
import signal
import threading
import time

from run_native_value_pilot import guarded, test_command
from run_native_value_preflight import atomic_json, sha256

TEST = "blueprint::preflop_continuation::fixed_control::exact_checkdown_preflop_convergence_control"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "kernel"):
        p.add_argument("--"+name, type=Path, required=True)
        p.add_argument("--"+name+"-sha256", required=True)
    p.add_argument("--rounds", type=int, choices=[8, 32, 128, 512], default=8)
    p.add_argument("--preflight", type=Path)
    p.add_argument("--preflight-sha256")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    pinned = {}
    for name in ("binary", "kernel"):
        path = getattr(args, name).resolve()
        if sha256(path) != getattr(args, name+"_sha256"): raise ValueError("changed exact-control input")
        setattr(args, name, path)
        pinned[str(path)] = sha256(path)
    if args.rounds > 8:
        if not args.preflight or sha256(args.preflight) != args.preflight_sha256:
            raise ValueError("longer control requires pinned eight-round preflight")
        prior = json.loads(args.preflight.read_text())
        if (prior["status"] != "complete" or prior["rounds"] != 8
                or any(prior["pinnedInputs"].get(path) != h for path,h in pinned.items())):
            raise ValueError("incompatible fixed-control preflight")
        if prior["elapsedSeconds"]/8*args.rounds*2 > 600:
            raise ValueError("projected control exceeds ten-minute guard")
        pinned[str(args.preflight.resolve())] = args.preflight_sha256
    for name in ("run_native_value_pilot.py", "run_native_value_preflight.py", "worker_resources.py"):
        path = Path(__file__).with_name(name).resolve()
        pinned[str(path)] = sha256(path)
    args.output = args.output.resolve()
    args.output.mkdir(exist_ok=False)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_:stop.set())
    started = time.monotonic()
    record = dict(schema="fixed-preflop-control-controller-v1", status="running",
        rounds=args.rounds, pinnedInputs=pinned, runnerSha256=sha256(Path(__file__)),
        maximumWorkerSeconds=600, maximumWorkerMemoryBytes=2*1024**3, releaseAccepted=False)
    atomic_json(args.output/"manifest.json",record)
    try:
        output = args.output/"control.json"
        record["worker"] = guarded(test_command(args.binary,TEST), {
            "POKER_COMPACT_CHECKDOWN":str(args.kernel), "POKER_COMPACT_CHECKDOWN_SHA":args.kernel_sha256,
            "POKER_FIXED_CONTROL_ROUNDS":str(args.rounds), "POKER_COMPACT_OUTPUT":str(output),
        },args.output/"worker",600,2*1024**3,stop)
        if stop.is_set() or any(sha256(Path(path)) != h for path,h in pinned.items()):
            raise ValueError("stopped or changed fixed-control input")
        v = json.loads(output.read_text())
        if (v["schema"] != "fixed-checkdown-preflop-control-v1" or v["releaseAccepted"] is not False
                or v["kernelSha256"] != args.kernel_sha256 or v["rounds"] != args.rounds
                or v["checkpoints"][-1]["rounds"] != args.rounds):
            raise ValueError("wrong fixed-game control identity")
        for r in v["checkpoints"]:
            if (not math.isfinite(r["exactCheckdownGameNashConvBb"])
                    or r["exactCheckdownGameNashConvBb"] < -1e-9
                    or r["fullHandExploitability"] is not None or r["releaseAccepted"] is not False):
                raise ValueError("invalid or mislabeled exact surrogate result")
        record.update(status="complete", outputSha256=sha256(output), checkpoints=v["checkpoints"],
            interpretation=__doc__)
    except (OSError, ValueError, KeyError) as error:
        record.update(status="failed",failure=str(error))
    record["elapsedSeconds"] = time.monotonic()-started
    atomic_json(args.output/"manifest.json",record)
    print(json.dumps({k:record.get(k) for k in ("status", "elapsedSeconds", "checkpoints", "failure")}), flush=True)
    if record["status"] != "complete": raise SystemExit(1)


if __name__ == "__main__": main()
