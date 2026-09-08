"""Guarded 16-state native continuation cost pilot; never trains or activates a model."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time

from native_value_dataset import SCHEMA, board_family, validate_dataset
from worker_resources import WorkerResourceGuard

RESERVE_BYTES = 20 * 1024**3
PREFLIGHT_BYTES = 32 * 1024**2
TEST = "blueprint::public_belief::counterfactual_turn::flop_pilot::value_targets::saved_native_value_preflight"


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("x") as output:
        json.dump(payload, output, indent=2, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def preflight(binary: Path, expected_hash: str, root: Path, root_hash: str, output: Path) -> None:
    if output.exists():
        raise ValueError("refusing to overwrite an existing preflight")
    if not output.parent.is_dir():
        raise ValueError("output parent must already exist")
    if sha256(binary) != expected_hash or sha256(root) != root_hash:
        raise ValueError("binary or input digest mismatch")
    if root.stat().st_size > 1024**2:
        raise ValueError("public input exceeds 1MiB")
    if shutil.disk_usage(output.parent).free < RESERVE_BYTES + PREFLIGHT_BYTES:
        raise ValueError("native preflight needs 20GiB disk reserve plus 32MiB output headroom")


def analyze(path: Path, expected_input_hash: str, turn_iterations: int, seed: int) -> dict:
    # Bound decompression independently of the file's compressed length.
    if path.stat().st_size > PREFLIGHT_BYTES:
        raise ValueError("preflight compressed output exceeds its storage budget")
    with gzip.open(path, "rb") as source:
        payload = source.read(PREFLIGHT_BYTES + 1)
    if len(payload) > PREFLIGHT_BYTES:
        raise ValueError("preflight decoded output exceeds its storage budget")
    value = json.loads(payload)
    validate_dataset(value)
    if (value["source_public_input_sha256"] != expected_input_hash
            or value["turn_iterations"] != turn_iterations or value["seed"] != seed
            or value["flop_iterations"] != 2 or len(value["targets"]) != 16):
        raise ValueError("preflight configuration/output mismatch")
    return dict(
        datasetSchema=SCHEMA, compressedBytes=path.stat().st_size,
        decodedBytes=len(payload), outputSha256=sha256(path),
        decodedSha256=hashlib.sha256(payload).hexdigest(),
        sourcePolicySha256=value["source_policy_sha256"], capturedStates=16,
        observedQueries=value["observed_queries"],
        distinctFlopFamilies=len({board_family(t["board"]) for t in value["targets"]}),
        capturedIterations=sorted({t["iteration"] for t in value["targets"]}),
        zeroOwnReachCompletions=[sum(t["completed_zero_own_reach"][p] for t in value["targets"]) for p in (0, 1)],
        undefinedZeroJointResiduals=sum(t["conditional_response_gain_bb"] is None for t in value["targets"]),
        interpretation="Cost/target-contract preflight only. First-N intermediate leaves are not a representative training/holdout corpus.",
        releaseAccepted=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--turn-iterations", type=int, choices=(4, 64, 128), default=64)
    parser.add_argument("--seed", type=int, choices=(100101, 100102), default=100101)
    args = parser.parse_args()
    binary, root, stage = args.binary.resolve(), args.input.resolve(), args.output.resolve()
    preflight(binary, args.binary_sha256, root, args.input_sha256, stage)
    stage.mkdir()
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    env = {
        "POKER_NATIVE_VALUE_INPUT": str(root), "POKER_NATIVE_VALUE_INPUT_SHA": args.input_sha256,
        "POKER_NATIVE_VALUE_OUTPUT": str(stage / "targets.json.gz"),
        "POKER_NATIVE_VALUE_MAX_STATES": "16", "POKER_NATIVE_VALUE_ITERATIONS": "2",
        "POKER_NATIVE_VALUE_TURN_ITERATIONS": str(args.turn_iterations),
        "POKER_NATIVE_VALUE_SEED": str(args.seed), "POKER_NATIVE_VALUE_WORKERS": str(args.workers),
    }
    command = [str(binary), TEST, "--exact", "--ignored", "--nocapture", "--test-threads=1"]
    record = dict(schema="native-value-preflight-controller-v1", status="running",
                  command=command, environment=env, binarySha256=args.binary_sha256,
                  inputSha256=args.input_sha256, runnerSha256=sha256(Path(__file__)),
                  startedAtUnix=time.time(), minimumFreeDiskBytes=RESERVE_BYTES,
                  maximumWorkerMemoryBytes=2 * 1024**3, maximumSeconds=900,
                  releaseAccepted=False)
    atomic_json(stage / "manifest.json", record)
    try:
        with (stage / "worker.log").open("x") as log:
            process = subprocess.Popen(command, env={**os.environ, **env}, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            guard = WorkerResourceGuard(process, stage, max_memory_bytes=2 * 1024**3,
                                        max_seconds=900, minimum_free_disk_bytes=RESERVE_BYTES,
                                        stop_event=stop).start()
            try:
                record["exitCode"] = process.wait()
            finally:
                if process.poll() is None:
                    guard.request_stop("controller interrupted")
                    process.wait()
                record.update(guard.finish())
        if record["exitCode"] != 0 or record["resourceStopReason"]:
            raise ValueError("native worker failed or resource guard stopped the pilot")
        if sha256(binary) != args.binary_sha256 or sha256(root) != args.input_sha256:
            raise ValueError("pinned native inputs changed during capture")
        record["analysis"] = analyze(stage / "targets.json.gz", args.input_sha256, args.turn_iterations, args.seed)
        record["status"] = "complete"
    except (OSError, ValueError, KeyError) as error:
        record["status"] = "failed"
        record["failure"] = str(error)
    atomic_json(stage / "manifest.json", record)
    print(json.dumps(record, indent=2, allow_nan=False), flush=True)
    if record["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
