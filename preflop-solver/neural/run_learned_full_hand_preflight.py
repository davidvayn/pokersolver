"""Pinned 128/64/64 full-hand integration/cost preflight, not a strength test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import threading
import time

from run_native_value_pilot import guarded, test_command
from run_native_value_preflight import atomic_json, sha256

TEST = "blueprint::response::native_policy::pilot::learned_full_hand_candidate_serving_probe"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "preflop", "model"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compact-continuation", action="store_true")
    args = parser.parse_args()
    pinned = {}
    for name in ("binary", "preflop", "model"):
        path = getattr(args, name).resolve()
        digest = getattr(args, name + "_sha256")
        if sha256(path) != digest:
            raise ValueError("pinned input changed: " + name)
        setattr(args, name, path)
        pinned[str(path)] = digest
    for name in ("run_native_value_pilot.py", "run_native_value_preflight.py", "worker_resources.py"):
        path = Path(__file__).with_name(name).resolve()
        pinned[str(path)] = sha256(path)
    args.output = args.output.resolve()
    args.output.mkdir(exist_ok=False)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    started = time.monotonic()
    record = dict(schema="learned-full-hand-preflight-v1", status="running",
                  pinnedInputs=pinned, runnerSha256=sha256(Path(__file__)),
                  compactContinuation=args.compact_continuation,
                  releaseAccepted=False, interpretation=__doc__)
    atomic_json(args.output / "manifest.json", record)
    try:
        output = args.output / "probe.json"
        environment = {
            "POKER_NATIVE_CHECKPOINT": str(args.preflop),
            "POKER_NATIVE_CHECKPOINT_SHA": args.preflop_sha256,
            "POKER_NATIVE_VALUE_MODEL": str(args.model),
            "POKER_NATIVE_VALUE_MODEL_SHA": args.model_sha256,
            "POKER_NATIVE_OUTPUT": str(output),
        }
        if args.compact_continuation:
            environment["POKER_NATIVE_COMPACT_CONTINUATION"] = "1"
        record["worker"] = guarded(test_command(args.binary, TEST), environment,
            args.output / "worker", 300, 2 * 1024**3, stop)
        if stop.is_set() or any(sha256(Path(p)) != h for p, h in pinned.items()):
            raise ValueError("stopped stage or changed input")
        result = json.loads(output.read_text())
        if (result["diagnostics"]["learnedLeafModelSha256"] != args.model_sha256
                or result["diagnostics"]["preflopSha256"] != args.preflop_sha256
                or result["diagnostics"]["completeRootSupport"] != args.compact_continuation
                or len(result["decisions"]) != 8):
            raise ValueError("full-hand route identity/incomplete trajectory")
        record.update(status="complete", outputSha256=sha256(output),
                      routeSha256=result["diagnostics"]["routeSha256"],
                      preflopMissing=result["preflopMissing"],
                      maximumProbabilitySumError=result["maximumProbabilitySumError"],
                      streetSeconds=[dict(street=d["street"], seconds=d["seconds"])
                                     for d in result["decisions"]])
    except (OSError, ValueError, KeyError) as error:
        record.update(status="failed", failure=str(error))
    record["elapsedSeconds"] = time.monotonic() - started
    atomic_json(args.output / "manifest.json", record)
    print(json.dumps(record), flush=True)
    if record["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
