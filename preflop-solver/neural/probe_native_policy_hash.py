"""Serial, hash-pinned packet parity/cost check for streamed native policy hashing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import threading
import time

from run_native_value_preflight import atomic_json, sha256
from run_postflop_gap_pilot import HERE, MEMORY, RESPONSE, PilotMemoryGuard, controller_lock, run_job


def verify_source_change(baseline):
    changed = [path for path, digest in baseline["sourceHashes"].items()
               if sha256(Path(path)) != digest]
    expected = {str((HERE / "../src/blueprint/public_belief" / name).resolve()) for name in
                ("counterfactual_turn.rs", "counterfactual_turn/flop_pilot/frozen_response.rs")}
    if set(changed) != expected:
        raise ValueError("streaming comparison permits only the hash implementation and canonical export test seam")
    return changed


def run(args):
    args.output = args.output.resolve()
    if args.output.exists():
        raise ValueError("refusing to overwrite a policy hash preflight")
    baseline_path, binary, candidate, reference = [path.resolve() for path in
        (args.baseline, args.binary, args.candidate, args.reference_packet)]
    if sha256(baseline_path) != args.baseline_sha256:
        raise ValueError("frozen build baseline changed")
    baseline = json.loads(baseline_path.read_text())
    changed = verify_source_change(baseline)
    old_binary = Path(baseline["binary"])
    pinned = {str(baseline_path): args.baseline_sha256, str(old_binary): baseline["binarySha256"],
              str(binary): args.binary_sha256, str(candidate): args.candidate_sha256,
              str(reference): args.reference_packet_sha256}
    for path in (Path(__file__), HERE / "run_postflop_gap_pilot.py", HERE / "run_native_value_pilot.py",
                 HERE / "worker_resources.py", HERE / "run_native_value_preflight.py"):
        pinned[str(path.resolve())] = sha256(path)
    pinned.update({path: sha256(Path(path)) for path in baseline["sourceHashes"]})
    if any(sha256(Path(path)) != digest for path, digest in pinned.items()):
        raise ValueError("hash preflight input changed")
    payload = json.loads(candidate.read_text()); packet = json.loads(reference.read_text())
    if (payload["turn_iterations"] != 64 or payload.get("response_turn_iterations") is not None
            or payload.get("learned_leaf_model_sha256") is not None
            or packet["turn_iterations"] != 64 or packet["candidate_sha256"] != args.candidate_sha256):
        raise ValueError("unmodified native64 policy/packet control required")
    args.output.mkdir()
    record = dict(schema="native-streamed-policy-hash-preflight-v1", status="running",
                  pinnedInputs=pinned, changedSourcePaths=changed, packets=[], releaseAccepted=False,
                  maximumSeconds=1800, maximumWorkerMemoryBytes=MEMORY)
    stop = threading.Event(); started = time.monotonic()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    timer = threading.Timer(1800, stop.set); timer.daemon = True
    with controller_lock(args.output):
        atomic_json(args.output / "manifest.json", record); timer.start()
        pressure = PilotMemoryGuard(stop, args.output / "system-memory.json").start()
        try:
            for arm, executable in (("buffered", old_binary), ("streamed", binary)):
                output = args.output / f"{arm}-packet.json"
                receipt = run_job(executable, RESPONSE + "saved_native_flop_turn_packet",
                    dict(POKER_NATIVE_FLOP_CANDIDATE=str(candidate), POKER_NATIVE_FLOP_CANDIDATE_SHA=args.candidate_sha256,
                         POKER_NATIVE_FLOP_TURN=str(packet["turn"]), POKER_NATIVE_FLOP_OUTPUT=str(output)),
                    args.output / arm, [output], stop, seconds=600, memory=MEMORY)
                if output.read_bytes() != reference.read_bytes():
                    raise ValueError(f"{arm} packet differs from the cached original native64 control")
                row = dict(arm=arm, output=str(output), sha256=sha256(output),
                           seconds=receipt["worker"]["workerElapsedSeconds"],
                           peakMemoryBytes=receipt["worker"]["sampledPeakMemoryBytes"])
                record["packets"].append(row)
                print(json.dumps(dict(event="native-hash-packet", **row)), flush=True)
                atomic_json(args.output / "manifest.json", record)
            if stop.is_set() or any(sha256(Path(path)) != digest for path, digest in pinned.items()):
                raise ValueError("preflight stopped or inputs changed")
            record.update(status="complete", exactPacketParity=True,
                          interpretation="One matched public turn, identical export bytes/values; not game-wide policy qualification")
        except Exception as error:
            record.update(status="failed", failure=str(error)); stop.set(); raise
        finally:
            timer.cancel(); record["elapsedSeconds"] = time.monotonic() - started
            record["systemMemoryGuard"] = pressure.finish()
            atomic_json(args.output / "manifest.json", record)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "binary", "candidate", "reference-packet"):
        p.add_argument("--" + name, type=Path, required=True)
        p.add_argument("--" + name + "-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
