"""Bounded sequential native corpus/prediction jobs; no training or activation by default."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

import native_value_dataset as native
from run_native_value_preflight import atomic_json, sha256, RESERVE_BYTES
from worker_resources import WorkerResourceGuard

ROOT_TEST = "blueprint::response::native_policy::pilot::native_value_authentic_roots"
CAPTURE_TEST = "blueprint::public_belief::counterfactual_turn::flop_pilot::value_targets::saved_native_value_preflight"
PREDICT_TEST = "blueprint::public_belief::counterfactual_turn::flop_pilot::continuation::saved_native_prediction_probe"


def guarded(command, env, output: Path, seconds=900, memory=2 * 1024**3, stop=None):
    """One process at a time; native parallelism stays inside its memory guard."""
    if output.exists():
        raise ValueError("refusing to overwrite a worker stage")
    output.mkdir()
    record = dict(status="running", command=command, environment=env,
                  maximumSeconds=seconds, maximumMemoryBytes=memory,
                  minimumFreeDiskBytes=RESERVE_BYTES, releaseAccepted=False)
    atomic_json(output / "manifest.json", record)
    try:
        with (output / "worker.log").open("x") as log:
            process = subprocess.Popen(command, env={**os.environ, **env}, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            guard = WorkerResourceGuard(process, output, max_memory_bytes=memory,
                                        max_seconds=seconds, minimum_free_disk_bytes=RESERVE_BYTES,
                                        stop_event=stop).start()
            try:
                record["exitCode"] = process.wait()
            finally:
                if process.poll() is None:
                    guard.request_stop("controller interrupted")
                    process.wait()
                record.update(guard.finish())
        if record["exitCode"] != 0 or record["resourceStopReason"]:
            raise ValueError("worker failed or resource guard stopped the stage")
        record["status"] = "complete"
    except (OSError, ValueError) as error:
        record["status"], record["failure"] = "failed", str(error)
    atomic_json(output / "manifest.json", record)
    if record["status"] != "complete":
        raise ValueError(str(record))
    return record


def test_command(binary, test):
    return [str(binary), test, "--exact", "--ignored", "--nocapture", "--test-threads=1"]


def read_capture(path):
    if path.stat().st_size > 128 * 1024**2:
        raise ValueError("native capture exceeds compressed budget")
    with gzip.open(path, "rb") as stream:
        payload = stream.read(256 * 1024**2 + 1)
    if len(payload) > 256 * 1024**2:
        raise ValueError("native capture exceeds decoded budget")
    source = json.loads(payload)
    native.validate_dataset(source)
    return source


def merge_captures(paths, search_proposals=False, reference=None):
    sources, targets, manifests = [], [], []
    if reference is not None:
        native.validate_dataset(reference)
        if not reference.get("source_captures"):
            raise ValueError("extension requires a complete collection reference")
        sources.append(reference)
        targets.extend(reference["targets"])
        manifests.extend(reference["source_captures"])
    for path in paths:
        index = len(manifests)
        source = read_capture(path)
        if search_proposals:
            if (source.get("capture_selection") != "stratified_learned_search_and_final_average_beliefs_with_native_labels"
                    or source.get("native_label_queries") != len(source["targets"])
                    or not native._sha(source.get("proposal_model_sha256"))
                    or type(source.get("sampling_seed")) is not int
                    or source.get("proposal_policy_kind") != "frozen_learned_leaf_search_only"):
                raise ValueError("incomplete or unpinned learned-belief native labels")
            distributions = {t.get("state_distribution") for t in source["targets"]}
            if distributions != {"learned_flop_search_early_belief_native_label",
                                  "learned_flop_search_middle_belief_native_label",
                                  "learned_flop_search_late_belief_native_label",
                                  "learned_flop_final_average_belief_native_label"}:
                raise ValueError("search-belief corpus omits a declared distribution")
        elif source.get("capture_selection") != "all_intermediate_queries_bounded_feasibility" or source["observed_queries"] != len(source["targets"]):
            raise ValueError("a truncated cost preflight is not a feasibility corpus")
        if sources and (source["game"] != sources[0]["game"] or source["turn_iterations"] != sources[0]["turn_iterations"]):
            raise ValueError("cannot merge different games or reference budgets")
        sources.append(source)
        manifests.append(dict(capture_sha256=sha256(path), input_sha256=source["source_public_input_sha256"],
                              policy_sha256=source["source_policy_sha256"], states=len(source["targets"]),
                              flop_iterations=source["flop_iterations"], turn_iterations=source["turn_iterations"],
                              seed=source["seed"]))
        if search_proposals:
            manifests[-1].update(proposal_model_sha256=source["proposal_model_sha256"],
                                 sampling_seed=source["sampling_seed"], native_label_queries=len(source["targets"]))
        targets.extend({**row, "source_capture_index": index} for row in source["targets"])
    if not sources or not 256 <= len(targets) <= 512:
        raise ValueError("native feasibility corpus requires 256-512 complete states")
    result = dict(schema=native.SCHEMA, game=sources[0]["game"],
                  source_identity_semantics="ordered_capture_manifest_not_one_policy", source_captures=manifests,
                  source_public_input_sha256=native.identity_hash([s["input_sha256"] for s in manifests]),
                  source_policy_sha256=native.identity_hash([s["policy_sha256"] for s in manifests]),
                  flop_iterations=max(s["flop_iterations"] for s in sources), turn_iterations=sources[0]["turn_iterations"],
                  maximum_states=len(targets), observed_queries=sum(s["observed_queries"] for s in sources),
                  capture_selection=("native_labels_on_stratified_learned_search_and_final_average_beliefs"
                                     if search_proposals else "all_queries_across_eight_authentic_live_flop_roots"),
                  validation=dict(status="research_only", reasons=["finite-budget multi-family feasibility corpus"]), targets=targets)
    native.validate_dataset(result)
    native.family_split(result, 10601, .25, .25)
    return result


def generate_corpus(args, stop):
    stage = args.output
    if sha256(args.preflop) != args.preflop_sha256:
        raise ValueError("preflop input hash mismatch")
    preflight = json.loads(args.preflight.read_text())
    if preflight.get("status") != "complete" or preflight.get("analysis", {}).get("capturedStates") != 16:
        raise ValueError("a successful native cost preflight is required")
    if sha256(args.preflight.parent / "targets.json.gz") != preflight["analysis"]["outputSha256"]:
        raise ValueError("preflight dataset hash mismatch")
    roots_path = stage / "roots.json"
    guarded(test_command(args.binary, ROOT_TEST), {
        "POKER_NATIVE_CHECKPOINT": str(args.preflop), "POKER_NATIVE_CHECKPOINT_SHA": args.preflop_sha256,
        "POKER_NATIVE_ROOTS_OUTPUT": str(roots_path), "POKER_NATIVE_ROOTS_SEED": str(args.root_seed),
    }, stage / "root-worker", stop=stop)
    roots = json.loads(roots_path.read_text())["roots"]
    if len(roots) != 8 or len({native.board_family(r["public"]["board"]) for r in roots}) != 8:
        raise ValueError("eight disjoint authentic flop families required")
    captures, workers = [], []
    for index, root in enumerate(roots):
        if stop.is_set(): raise ValueError("operator stopped corpus generation")
        if sha256(args.binary) != args.binary_sha256: raise ValueError("pinned binary changed")
        leaf_count = root["turn_leaf_count"]
        if not 1 <= leaf_count <= 16: raise ValueError("unpreflighted native leaf count")
        rounds = math.ceil(32 / leaf_count)
        path = stage / ("root-%02d.json" % index)
        atomic_json(path, root)
        target = stage / ("capture-%02d.json.gz" % index)
        seed = (100101, 100102)[index % 2]
        env = {"POKER_NATIVE_VALUE_INPUT": str(path), "POKER_NATIVE_VALUE_INPUT_SHA": sha256(path),
               "POKER_NATIVE_VALUE_OUTPUT": str(target), "POKER_NATIVE_VALUE_MAX_STATES": "64",
               "POKER_NATIVE_VALUE_ITERATIONS": str(rounds), "POKER_NATIVE_VALUE_TURN_ITERATIONS": "64",
               "POKER_NATIVE_VALUE_SEED": str(seed), "POKER_NATIVE_VALUE_WORKERS": "4",
               "POKER_NATIVE_VALUE_CAPTURE_ALL": "1"}
        record = guarded(test_command(args.binary, CAPTURE_TEST), env, stage / ("worker-%02d" % index), stop=stop)
        source = read_capture(target)
        if (source["source_public_input_sha256"] != sha256(path) or source["flop_iterations"] != rounds
                or source["turn_iterations"] != 64 or source["seed"] != seed or len(source["targets"]) != rounds * leaf_count):
            raise ValueError("native capture configuration/count mismatch")
        captures.append(target)
        workers.append(record)
        print(json.dumps(dict(event="capture-complete", root=index, states=len(source["targets"]),
                              seconds=record["workerElapsedSeconds"], invested=root["public"]["invested_bb"])), flush=True)
    result = merge_captures(captures)
    encoded = json.dumps(result, separators=(",", ":"), allow_nan=False).encode()
    target = stage / "corpus.json.gz"
    with target.open("xb") as output:
        output.write(gzip.compress(encoded, mtime=0))
    if sha256(args.binary) != args.binary_sha256 or sha256(args.preflop) != args.preflop_sha256:
        raise ValueError("pinned source changed during corpus generation")
    return dict(corpusSha256=sha256(target), states=len(result["targets"]), families=8,
                rootsSha256=sha256(roots_path), preflightSha256=sha256(args.preflight),
                preflopSha256=args.preflop_sha256, decodedBytes=len(encoded), compressedBytes=target.stat().st_size,
                maximumWorkerMemoryBytes=max(w["sampledPeakMemoryBytes"] for w in workers))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--preflop", type=Path, required=True)
    parser.add_argument("--preflop-sha256", required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root-seed", type=int, default=881902)
    args = parser.parse_args()
    for name in ("binary", "preflop", "preflight", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists() or sha256(args.binary) != args.binary_sha256:
        raise ValueError("existing output or binary digest mismatch")
    args.output.mkdir()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: stop.set())
    record = dict(schema="native-value-feasibility-controller-v1", status="running",
                  binarySha256=args.binary_sha256, runnerSha256=sha256(Path(__file__)),
                  startedAtUnix=time.time(), releaseAccepted=False)
    atomic_json(args.output / "manifest.json", record)
    try:
        record["analysis"] = generate_corpus(args, stop)
        record["status"] = "complete"
    except (ValueError, KeyError, OSError) as error:
        record["status"], record["failure"] = "failed", str(error)
    record["elapsedSeconds"] = time.time() - record["startedAtUnix"]
    atomic_json(args.output / "manifest.json", record)
    print(json.dumps(record, indent=2), flush=True)
    if record["status"] != "complete": raise SystemExit(1)


if __name__ == "__main__":
    main()
