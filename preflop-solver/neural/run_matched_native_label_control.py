"""Re-label identical high-budget search beliefs at native64 for causal pairing.

The proposer, sampled queries, game and training families must reproduce exactly.
Only the native reference budget changes. This is an offline research control,
not a new web policy or a release qualification.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import signal
import threading
import time

import numpy as np

import native_value_dataset as native
from run_native_value_pilot import merge_captures, read_capture, test_command
from run_native_value_preflight import atomic_json, sha256
from run_postflop_gap_pilot import PilotMemoryGuard, controller_lock, run_job

TEST = "blueprint::public_belief::counterfactual_turn::flop_pilot::search_beliefs::saved_search_distribution_native_capture"
PUBLIC_KEYS = ("actor", "board", "completed_zero_own_reach", "invested_bb", "iteration",
               "opponent_compatible_mass", "public_state", "ranges", "raw_reach_totals",
               "state_distribution", "value_semantics")


def label_drift(high: dict, low: dict) -> dict:
    if (len(high["targets"]) != len(low["targets"])
            or high["turn_iterations"] != 256 or low["turn_iterations"] != 64
            or any(high.get(key) != low.get(key) for key in
                   ("game", "source_public_input_sha256", "source_policy_sha256",
                    "proposal_model_sha256", "seed", "sampling_seed", "flop_iterations",
                    "observed_queries", "native_label_queries"))):
        raise ValueError("matched labels have different proposer or query provenance")
    numerators = np.zeros(2, dtype=np.float64)
    denominators = np.zeros(2, dtype=np.float64)
    for h, l in zip(high["targets"], low["targets"], strict=True):
        if any(h.get(key) != l.get(key) for key in PUBLIC_KEYS):
            raise ValueError("native budgets labeled different public beliefs")
        weights = np.asarray(h["ranges"]) * np.asarray(h["opponent_compatible_mass"])
        delta = np.asarray(h["counterfactual_values_bb"]) - np.asarray(l["counterfactual_values_bb"])
        numerators += np.sum(weights * delta * delta, axis=1)
        denominators += np.sum(weights, axis=1)
    if not np.isfinite(numerators).all() or (denominators <= 0).any():
        raise ValueError("matched labels lack positive finite authentic reach")
    return dict(reachWeightedRmseBbBySeat=np.sqrt(numerators / denominators).tolist(),
                beliefs=len(high["targets"]))


def run(args):
    high_path, binary, output = args.high_manifest.resolve(), args.binary.resolve(), args.output.resolve()
    retained_path, split_path = args.retained_corpus.resolve(), args.split_reference.resolve()
    if sha256(high_path) != args.high_sha256 or sha256(binary) != args.binary_sha256:
        raise ValueError("pinned high-budget stage or binary changed")
    if (sha256(retained_path) != args.retained_sha256
            or sha256(split_path) != args.split_sha256):
        raise ValueError("retained corpus or split reference changed")
    high = json.loads(high_path.read_text())
    if (high.get("status") != "complete" or high.get("schema") != "native-search-distribution-controller-v1"
            or high.get("labelTurnIterations") != 256 or high.get("selectedRootIndices") != [2, 3, 4]
            or not high.get("trainingOnly") or len(high.get("workers", [])) != 3
            or high.get("pinnedInputs", {}).get(str(binary)) != args.binary_sha256
            or high.get("pinnedInputs", {}).get(str(retained_path)) != args.retained_sha256
            or high.get("pinnedInputs", {}).get(str(split_path)) != args.split_sha256):
        raise ValueError("complete training-only native256 stage required")
    sources = {str(path): sha256(path) for path in
               (Path(__file__), Path(__file__).with_name("run_native_value_pilot.py"),
                Path(__file__).with_name("run_postflop_gap_pilot.py"),
                Path(__file__).with_name("native_value_dataset.py"),
                Path(__file__).with_name("worker_resources.py"))}
    identity = dict(schema="matched-native-label-control-v1",
                    highManifestSha256=args.high_sha256, binarySha256=args.binary_sha256,
                    retainedCorpusSha256=args.retained_sha256, splitReferenceSha256=args.split_sha256,
                    sourceHashes=sources, selectedRootIndices=[2, 3, 4],
                    highBudget=256, controlBudget=64, maximumStageSeconds=5400,
                    maximumWorkerMemoryBytes=2 * 1024**3, releaseAccepted=False)
    if output.exists() and not (output / "manifest.json").is_file():
        raise ValueError("output exists without matched-label manifest")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("matched-label identity changed on resume")
    record = {**identity, "status": "running", "cases": []}
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    timer = threading.Timer(5400, stop.set)
    timer.daemon = True
    started = time.monotonic()
    with controller_lock(output):
        atomic_json(manifest, record)
        timer.start()
        guard = None
        try:
            guard = PilotMemoryGuard(stop, output / "system-memory.json").start()
            for row in high["workers"]:
                root = row["root"]
                if root not in (2, 3, 4) or stop.is_set():
                    raise ValueError("matched label root or resource state changed")
                high_capture = Path(row["path"])
                high_env = row["worker"]["environment"]
                high_proposer = Path(high_env["POKER_SEARCH_POLICY_OUTPUT"])
                if (sha256(high_capture) != row["sha256"]
                        or sha256(high_proposer) != read_capture(high_capture)["source_policy_sha256"]
                        or sha256(Path(high_env["POKER_SEARCH_ROOT"])) != high_env["POKER_SEARCH_ROOT_SHA"]
                        or sha256(Path(high_env["POKER_SEARCH_MODEL"])) != high_env["POKER_SEARCH_MODEL_SHA"]):
                    raise ValueError("high-budget capture or proposer changed")
                work = output / f"root-{root}"
                work.mkdir(exist_ok=True)
                low_capture = work / "capture.json.gz"
                low_proposer = work / "proposer.json"
                env = {**high_env,
                       "POKER_SEARCH_OUTPUT": str(low_capture),
                       "POKER_SEARCH_POLICY_OUTPUT": str(low_proposer),
                       "POKER_SEARCH_LABEL_TURN_ITERATIONS": "64"}
                receipt = run_job(binary, TEST, env, work / "label", [low_capture, low_proposer],
                                  stop, seconds=900, memory=2 * 1024**3)
                high_source, low_source = read_capture(high_capture), read_capture(low_capture)
                if sha256(low_proposer) != sha256(high_proposer):
                    raise ValueError("native64 relabeling changed the learned proposer")
                drift = label_drift(high_source, low_source)
                result = dict(root=root, states=row["states"], highCaptureSha256=row["sha256"],
                              lowCapture=str(low_capture), lowCaptureSha256=sha256(low_capture),
                              proposerSha256=sha256(low_proposer),
                              seconds=receipt["worker"]["workerElapsedSeconds"], **drift)
                record["cases"].append(result)
                atomic_json(manifest, record)
                print(json.dumps(dict(event="matched-native-labels", **result)), flush=True)
            retained = read_capture(retained_path)
            split_reference = read_capture(split_path)
            low_paths = [Path(row["lowCapture"]) for row in record["cases"]]
            corpus = merge_captures(low_paths, search_proposals=True,
                                    reference=retained, maximum_states=640)
            old_train, old_tuning, old_holdout = native.family_split(
                retained, 10601, .25, .25, reference=split_reference, refresh_training=True)
            new_train, new_tuning, new_holdout = native.family_split(
                corpus, 10601, .25, .25, reference=split_reference, refresh_training=True)
            if (len(new_train) != len(old_train) + sum(row["states"] for row in record["cases"])
                    or not np.array_equal(new_tuning, old_tuning)
                    or not np.array_equal(new_holdout, old_holdout)
                    or corpus["targets"][:len(retained["targets"])] != retained["targets"]):
                raise ValueError("native64 control changed the frozen evaluation split")
            encoded = json.dumps(corpus, separators=(",", ":"), allow_nan=False).encode()
            compressed = gzip.compress(encoded, mtime=0)
            corpus_path = output / "corpus.json.gz"
            if corpus_path.exists():
                if sha256(corpus_path) != hashlib.sha256(compressed).hexdigest():
                    raise ValueError("resumed native64 corpus output changed")
            else:
                temporary = output / "corpus.json.gz.tmp"
                if temporary.exists():
                    temporary.rename(output / f"corpus.interrupted-{time.time_ns()}.tmp")
                temporary.write_bytes(compressed)
                temporary.replace(corpus_path)
            record["corpus"] = dict(path=str(corpus_path), sha256=sha256(corpus_path),
                                    states=len(corpus["targets"]), decodedBytes=len(encoded),
                                    compressedBytes=len(compressed),
                                    frozenTuningStates=len(new_tuning), frozenHoldoutStates=len(new_holdout))
            if (stop.is_set() or guard.reason or sha256(high_path) != args.high_sha256
                    or sha256(binary) != args.binary_sha256
                    or sha256(retained_path) != args.retained_sha256
                    or sha256(split_path) != args.split_sha256
                    or any(sha256(Path(path)) != digest for path, digest in sources.items())):
                raise ValueError("matched-label stage stopped or source changed")
            if {row["root"] for row in record["cases"]} != {2, 3, 4}:
                raise ValueError("matched-label control lacks a training family")
            record["status"] = "complete"
        except (OSError, ValueError, KeyError, AssertionError) as error:
            stop.set()
            record["status"], record["failure"] = "failed", str(error)
        finally:
            timer.cancel()
            record["elapsedSeconds"] = time.monotonic() - started
            if guard:
                record["systemMemoryGuard"] = guard.finish()
            atomic_json(manifest, record)
    print(json.dumps(dict(status=record["status"], failure=record.get("failure"),
                          cases=record["cases"])), flush=True)
    if record["status"] != "complete":
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--high-manifest", type=Path, required=True)
    parser.add_argument("--high-sha256", required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--retained-corpus", type=Path, required=True)
    parser.add_argument("--retained-sha256", required=True)
    parser.add_argument("--split-reference", type=Path, required=True)
    parser.add_argument("--split-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
