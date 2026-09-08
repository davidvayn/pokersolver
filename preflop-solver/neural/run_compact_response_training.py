"""Cost-preflight the existing full-continuation learned responder, not a strength test."""
import argparse
import json
from pathlib import Path
import signal
import threading
import time

from run_native_value_pilot import guarded, test_command
from run_native_value_preflight import atomic_json, sha256

TEST = "blueprint::response::native_policy::full_hand_probe::compact_native_learned_response_train"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "preflop", "model"):
        parser.add_argument("--"+name, type=Path, required=True)
        parser.add_argument("--"+name+"-sha256", required=True)
    parser.add_argument("--seat", type=int, choices=[0, 1], default=0)
    parser.add_argument("--deals", type=int, choices=[2, 8, 16, 32], default=2)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--preflight-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pinned = {}
    for name in ("binary", "preflop", "model"):
        path = getattr(args, name).resolve()
        if sha256(path) != getattr(args, name+"_sha256"):
            raise ValueError("response training input changed")
        setattr(args, name, path)
        pinned[str(path)] = sha256(path)
    if args.deals > 2:
        if not args.preflight or sha256(args.preflight) != args.preflight_sha256:
            raise ValueError("larger training requires pinned two-deal preflight")
        prior = json.loads(args.preflight.read_text())
        if (prior["status"] != "complete" or prior["deals"] != 2
                or any(prior["pinnedInputs"].get(p) != h for p, h in pinned.items())):
            raise ValueError("incompatible response training preflight")
        pinned[str(args.preflight.resolve())] = args.preflight_sha256
    for name in ("run_native_value_pilot.py", "run_native_value_preflight.py", "worker_resources.py"):
        path = Path(__file__).with_name(name).resolve()
        pinned[str(path)] = sha256(path)
    args.output = args.output.resolve()
    args.output.mkdir(exist_ok=False)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_:stop.set())
    record = dict(schema="compact-response-training-controller-v1", status="running",
        deals=args.deals, seat=args.seat, pinnedInputs=pinned,
        runnerSha256=sha256(Path(__file__)), maximumWorkerSeconds=150*args.deals,
        maximumWorkerMemoryBytes=2*1024**3, releaseAccepted=False)
    atomic_json(args.output/"manifest.json", record)
    started = time.monotonic()
    try:
        output = args.output/"response.json"
        record["worker"] = guarded(test_command(args.binary, TEST), {
            "POKER_NATIVE_CHECKPOINT":str(args.preflop), "POKER_NATIVE_CHECKPOINT_SHA":args.preflop_sha256,
            "POKER_NATIVE_VALUE_MODEL":str(args.model), "POKER_NATIVE_VALUE_MODEL_SHA":args.model_sha256,
            "POKER_NATIVE_RESPONSE_SEAT":str(args.seat), "POKER_NATIVE_RESPONSE_DEALS":str(args.deals),
            "POKER_NATIVE_OUTPUT":str(output),
        }, args.output/"worker", 150*args.deals, 2*1024**3, stop)
        if stop.is_set() or any(sha256(Path(p)) != h for p, h in pinned.items()):
            raise ValueError("stopped or changed response input")
        result = json.loads(output.read_text())
        if (result["schema"] != "compact-native-learned-response-v1"
                or result["responder"] != args.seat or result["trainingDeals"] != args.deals
                or result["diagnostics"]["preflopSha256"] != args.preflop_sha256
                or result["diagnostics"]["learnedLeafModelSha256"] != args.model_sha256
                or result["diagnostics"]["completeRootSupport"] is not True
                or result["releaseAccepted"] is not False):
            raise ValueError("response training identity mismatch")
        record.update(status="complete", output=str(output), outputSha256=sha256(output),
            trainingSeconds=result["seconds"], preflopRows=len(result["preflop"]),
            postflopRows=len(result["resolver"]["decisions"]),
            interpretation="Training/cost only; sparse critic coverage is not evidence of an unexploitable defender.")
    except (OSError, ValueError, KeyError) as error:
        record.update(status="failed", failure=str(error))
    record["elapsedSeconds"] = time.monotonic()-started
    atomic_json(args.output/"manifest.json", record)
    print(json.dumps({k:record.get(k) for k in ("status", "elapsedSeconds", "preflopRows", "postflopRows", "failure")}), flush=True)
    if record["status"] != "complete": raise SystemExit(1)


if __name__ == "__main__": main()
