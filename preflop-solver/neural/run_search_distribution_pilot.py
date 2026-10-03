"""Sequential 16-state preflight, then same-root/same-size native search-belief labels.

The completed learned model proposes inputs only. Every target is independently
computed by the explicitly pinned native continuation budget. No fitting or
activation here. A preflight-only run measures higher-quality label cost before
committing to a full corpus.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
import signal
import threading
import time

import native_value_dataset as native
from run_native_value_pilot import guarded, merge_captures, read_capture, test_command
from run_native_value_preflight import atomic_json, sha256
from run_postflop_gap_pilot import PilotMemoryGuard
from mixed_native_reference import append_training_labels

TEST = "blueprint::public_belief::counterfactual_turn::flop_pilot::search_beliefs::saved_search_distribution_native_capture"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "baseline", "students", "response", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
        if name != "output": parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--label-turn-iterations", type=int, choices=(64, 256, 1024), default=64)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-manifest", type=Path)
    parser.add_argument("--preflight-manifest-sha256")
    parser.add_argument("--selected-root-indices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--append-reference", type=Path)
    parser.add_argument("--append-reference-sha256")
    parser.add_argument("--split-reference", type=Path)
    parser.add_argument("--split-reference-sha256")
    parser.add_argument("--reuse-completed-from", type=Path)
    parser.add_argument("--reuse-completed-sha256")
    args = parser.parse_args()
    pinned = {}
    for name in ("binary", "baseline", "students", "response", "output"):
        path = getattr(args, name).resolve()
        setattr(args, name, path)
        if name != "output":
            expected = getattr(args, name + "_sha256")
            if sha256(path) != expected: raise ValueError("pinned input mismatch: " + name)
            pinned[str(path)] = expected
    baseline = json.loads(args.baseline.read_text())
    students = json.loads(args.students.read_text())
    response = json.loads(args.response.read_text())
    if any(m.get("status") != "complete" for m in (baseline, students, response)):
        raise ValueError("completed corpus, paired students, and policy comparison required")
    if (baseline["analysis"]["states"] != 284 or len(students["predictions"]) != 2
            or len(response["profiles"]) != 2):
        raise ValueError("unmatched feasibility stage")
    if response["pinnedInputs"].get(str(args.students)) != args.students_sha256:
        raise ValueError("policy comparison did not use this student pair")
    roots_path = args.baseline.parent / "roots.json"
    if sha256(roots_path) != baseline["analysis"]["rootsSha256"]:
        raise ValueError("authentic roots changed")
    pinned[str(roots_path)] = sha256(roots_path)
    roots = json.loads(roots_path.read_text())["roots"]
    if len(roots) != 8: raise ValueError("same eight flop roots required")
    selected = [int(value) for value in args.selected_root_indices.split(",")]
    if len(set(selected)) != len(selected) or selected not in (list(range(8)), [2, 3, 4]):
        raise ValueError("only all roots or the frozen training-only 2,3,4 pilot is allowed")
    training_only = selected == [2, 3, 4]
    if bool(args.append_reference) != bool(args.split_reference) or training_only != bool(args.append_reference):
        raise ValueError("training-only capture requires retained and split-reference corpora")
    if training_only and (args.preflight_only or args.label_turn_iterations != 256):
        raise ValueError("training-only pilot requires complete native-256 capture")
    reference = split_reference = None
    for name in ("append_reference", "split_reference"):
        path = getattr(args, name)
        digest = getattr(args, name + "_sha256")
        if path is None:
            if digest is not None: raise ValueError("reference SHA has no path")
            continue
        path = path.resolve()
        setattr(args, name, path)
        if not digest or sha256(path) != digest:
            raise ValueError("pinned " + name + " changed")
        pinned[str(path)] = digest
    if training_only:
        reference = read_capture(args.append_reference)
        split_reference = read_capture(args.split_reference)
        helper = Path(__file__).with_name("mixed_native_reference.py").resolve()
        pinned[str(helper)] = sha256(helper)
    if bool(args.preflight_manifest) != bool(args.preflight_manifest_sha256):
        raise ValueError("reused preflight needs a manifest and SHA")
    if args.preflight_only and args.preflight_manifest:
        raise ValueError("a preflight-only stage must measure its own worker")
    if args.preflight_manifest:
        args.preflight_manifest = args.preflight_manifest.resolve()
        if sha256(args.preflight_manifest) != args.preflight_manifest_sha256:
            raise ValueError("preflight manifest changed")
        previous_preflight = json.loads(args.preflight_manifest.read_text())
        if (previous_preflight.get("status") != "complete"
                or previous_preflight.get("preflightOnly") is not True
                or previous_preflight.get("labelTurnIterations") != args.label_turn_iterations
                or previous_preflight.get("analysis", {}).get("states") != 16
                or previous_preflight.get("analysis", {}).get("root") != 1
                or any(previous_preflight.get("pinnedInputs", {}).get(str(path)) != digest
                       for path, digest in ((args.binary, args.binary_sha256),
                                            (args.baseline, args.baseline_sha256),
                                            (args.students, args.students_sha256),
                                            (args.response, args.response_sha256)))):
            raise ValueError("preflight did not use this frozen model and native-label budget")
        old_capture = previous_preflight["workers"][0]
        if (old_capture["root"] != 1 or old_capture["states"] != 16
                or sha256(Path(old_capture["path"])) != old_capture["sha256"]
                or not read_capture(Path(old_capture["path"]))["policy_observation_parity_checked"]):
            raise ValueError("preflight capture or policy parity changed")
        pinned[str(args.preflight_manifest)] = args.preflight_manifest_sha256
    elif training_only:
        raise ValueError("training-only pilot requires measured native-256 preflight")
    model_paths = []
    for index, row in enumerate(students["predictions"]):
        path = Path(row["model"])
        if (sha256(path) != row["modelSha256"] or row["maximumParityErrorBb"] > 1e-4
                or response["profiles"][index]["modelSha256"] != row["modelSha256"]):
            raise ValueError("proposer weights or actual inference qualification changed")
        pinned[str(path)] = row["modelSha256"]
        model_paths.append(path)
    for name in ("run_native_value_pilot.py", "native_value_dataset.py", "run_native_value_preflight.py",
                 "run_postflop_gap_pilot.py", "worker_resources.py"):
        path = Path(__file__).with_name(name).resolve()
        pinned[str(path)] = sha256(path)
    reusable = {}
    if args.reuse_completed_from:
        if args.label_turn_iterations != 64:
            raise ValueError("higher-budget labels cannot reuse a native-64 source")
        previous_path = args.reuse_completed_from.resolve()
        if not args.reuse_completed_sha256 or sha256(previous_path) != args.reuse_completed_sha256:
            raise ValueError("reuse requires the exact previous manifest hash")
        previous = json.loads(previous_path.read_text())
        if previous.get("schema") != "native-search-distribution-controller-v1" or previous.get("status") not in ("failed", "complete"):
            raise ValueError("only a stopped or completed compatible stage can be reused")
        if any(previous["pinnedInputs"].get(p) != h for p, h in pinned.items()):
            raise ValueError("previous stage used different pinned inputs or label helpers")
        pinned[str(previous_path)] = args.reuse_completed_sha256
        for row in previous["workers"]:
            if row["worker"]["status"] != "complete" or row["worker"].get("resourceStopReason"):
                raise ValueError("cannot reuse an interrupted worker")
            key = (row["root"], row["states"])
            if key in reusable: raise ValueError("ambiguous repeated source capture")
            reusable[key] = row
    elif args.reuse_completed_sha256:
        raise ValueError("reuse hash has no manifest")
    if args.output.exists(): raise ValueError("refusing to overwrite a pilot")
    args.output.mkdir()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: stop.set())
    timer = threading.Timer(7200, stop.set)
    timer.daemon = True
    timer.start()
    started = time.monotonic()
    record = dict(schema="native-search-distribution-controller-v1", status="running", releaseAccepted=False,
                  pinnedInputs=pinned, runnerSha256=sha256(Path(__file__)), workers=[],
                  maximumStageSeconds=7200, distributionChangeOnly=True,
                  nativeWorkers=args.workers, reusedManifestSha256=args.reuse_completed_sha256,
                  labelTurnIterations=args.label_turn_iterations, preflightOnly=args.preflight_only,
                  selectedRootIndices=selected, trainingOnly=training_only,
                  reusedPreflightSha256=args.preflight_manifest_sha256,
                  interpretation=("Bounded native-label cost probe on a held-out source family; no fitting or activation."
                                  if args.preflight_only else
                                  "Training-only stronger native labels appended to a frozen research corpus; no fitting or activation."
                                  if training_only else
                                  "Same eight source flop families and 284-label budget, but stratified learned-search and final-average beliefs; native labels, no activation."))
    atomic_json(args.output / "manifest.json", record)

    def capture(index, count, name, check_policy=False):
        if stop.is_set() or any(sha256(Path(p)) != h for p,h in pinned.items()):
            raise ValueError("operator stop or pinned input changed")
        root = args.output / (name + "-root.json")
        atomic_json(root, roots[index])
        model = model_paths[index % 2]
        output = args.output / (name + ".json.gz")
        policy = args.output / (name + "-proposer.json")
        seed, sample_seed = 100101 + index % 2, 772801 + index
        previous_row = reusable.get((index, count))
        if previous_row and not check_policy:
            old_env = previous_row["worker"]["environment"]
            old_output = Path(previous_row["path"])
            old_root = Path(old_env["POKER_SEARCH_ROOT"])
            old_policy = Path(old_env["POKER_SEARCH_POLICY_OUTPUT"])
            source = read_capture(old_output)
            if (sha256(old_output) != previous_row["sha256"]
                    or sha256(old_root) != source["source_public_input_sha256"]
                    or json.loads(old_root.read_text()) != roots[index]
                    or sha256(old_policy) != source["source_policy_sha256"]
                    or source["proposal_model_sha256"] != pinned[str(model)]
                    or source["seed"] != seed or source["sampling_seed"] != sample_seed
                    or source["flop_iterations"] != 32 or source["turn_iterations"] != args.label_turn_iterations
                    or len(source["targets"]) != count or source["native_label_queries"] != count):
                raise ValueError("completed capture failed reuse integrity checks")
            for path in (old_output, old_root, old_policy): pinned[str(path)] = sha256(path)
            record["workers"].append({**previous_row, "reused": True,
                                       "originManifestSha256":args.reuse_completed_sha256})
            atomic_json(args.output / "manifest.json", record)
            print(json.dumps(dict(event="search-belief-labels-reused", root=index, states=count)), flush=True)
            return old_output
        env = {"POKER_SEARCH_ROOT": str(root), "POKER_SEARCH_ROOT_SHA": sha256(root),
               "POKER_SEARCH_MODEL": str(model), "POKER_SEARCH_MODEL_SHA": pinned[str(model)],
               "POKER_SEARCH_OUTPUT": str(output), "POKER_SEARCH_POLICY_OUTPUT": str(policy),
               "POKER_SEARCH_TRUNK_SEED": str(seed), "POKER_SEARCH_SAMPLE_SEED": str(sample_seed),
               "POKER_SEARCH_LABEL_COUNT": str(count), "POKER_SEARCH_WORKERS": str(args.workers),
               "POKER_SEARCH_LABEL_TURN_ITERATIONS": str(args.label_turn_iterations),
               "POKER_SEARCH_VERIFY_POLICY_PARITY": "1" if check_policy else "0"}
        worker = guarded(test_command(args.binary, TEST), env, args.output / (name + "-worker"),
                         3000 if args.label_turn_iterations > 64 and not check_policy else 900,
                         2 * 1024**3, stop)
        source = read_capture(output)
        if (len(source["targets"]) != count or source["native_label_queries"] != count
                or source["source_public_input_sha256"] != sha256(root)
                or source["source_policy_sha256"] != sha256(policy)
                or source["proposal_model_sha256"] != pinned[str(model)]
                or source["sampling_seed"] != sample_seed or source["seed"] != seed
                or source["flop_iterations"] != 32 or source["turn_iterations"] != args.label_turn_iterations
                or source.get("proposal_turn_iterations") != 64
                or (check_policy and not source["policy_observation_parity_checked"])):
            raise ValueError("search-belief/native label provenance or count drift")
        # Re-run the small preflight after a scheduling change. Exact dataset
        # equality checks both proposal and native-label worker-order parity.
        if check_policy and previous_row and sha256(output) != previous_row["sha256"]:
            raise ValueError("worker-count change altered the preflight dataset")
        row = dict(root=index, states=count, path=str(output), sha256=sha256(output), worker=worker)
        record["workers"].append(row)
        atomic_json(args.output / "manifest.json", record)
        print(json.dumps(dict(event="search-belief-native-labels", root=index, states=count,
                              seconds=worker["workerElapsedSeconds"], preflight=check_policy)), flush=True)
        return output

    memory_guard = None
    try:
        memory_guard = PilotMemoryGuard(stop, args.output / "system-memory.json").start()
        # The low-pot, nine-leaf root exercises the expensive continuation tree.
        if args.preflight_manifest:
            record["preflightSha256"] = previous_preflight["analysis"]["preflightSha256"]
            record["preflightSeconds"] = previous_preflight["analysis"]["workerSeconds"]
        else:
            preflight = capture(1, 16, "preflight", True)
            record["preflightSha256"] = sha256(preflight)
            record["preflightSeconds"] = record["workers"][-1]["worker"]["workerElapsedSeconds"]
        if args.preflight_only:
            record["status"] = "complete"
            record["analysis"] = dict(states=16, root=1, labelTurnIterations=args.label_turn_iterations,
                                      preflightSha256=record["preflightSha256"],
                                      workerSeconds=record["workers"][-1]["worker"]["workerElapsedSeconds"],
                                      peakMemoryBytes=record["workers"][-1]["worker"]["sampledPeakMemoryBytes"])
        else:
            expected_states = sum(math.ceil(32 / roots[index]["turn_leaf_count"]) *
                                  roots[index]["turn_leaf_count"] for index in selected)
            estimate = record["preflightSeconds"] * expected_states / 16
            if training_only and (estimate * 1.5 >= 7200 or
                                  len(reference["targets"]) + expected_states > 640):
                raise ValueError("measured stronger-label projection exceeds time or corpus budget")
            record["estimatedCaptureSeconds"] = estimate
            atomic_json(args.output / "manifest.json", record)
            captures = []
            for index in selected:
                root = roots[index]
                count = math.ceil(32 / root["turn_leaf_count"]) * root["turn_leaf_count"]
                captures.append(capture(index, count, "capture-%02d" % index))
            corpus = (append_training_labels(reference, split_reference, captures)
                      if training_only else merge_captures(captures, search_proposals=True))
            if len(corpus["targets"]) != (len(reference["targets"]) + expected_states
                                           if training_only else baseline["analysis"]["states"]):
                raise ValueError("changed native labeling budget")
            encoded = json.dumps(corpus, separators=(",", ":"), allow_nan=False).encode()
            output = args.output / "corpus.json.gz"
            with output.open("xb") as stream: stream.write(gzip.compress(encoded, mtime=0))
            if stop.is_set() or any(sha256(Path(p)) != h for p,h in pinned.items()):
                raise ValueError("stopped stage or pinned inputs changed")
            record["status"] = "complete"
            record["analysis"] = dict(states=len(corpus["targets"]),
                                      families=len({native.board_family(row["board"]) for row in corpus["targets"]}),
                                      corpusSha256=sha256(output),
                                      compressedBytes=output.stat().st_size, decodedBytes=len(encoded),
                                      maximumWorkerMemoryBytes=max(w["worker"]["sampledPeakMemoryBytes"] for w in record["workers"]))
    except (ValueError, KeyError, OSError) as error:
        stop.set()
        record["status"], record["failure"] = "failed", str(error)
    finally:
        timer.cancel()
        record["elapsedSeconds"] = time.monotonic() - started
        if memory_guard:
            record["systemMemoryGuard"] = memory_guard.finish()
        atomic_json(args.output / "manifest.json", record)
    print(json.dumps({k:record.get(k) for k in ("status", "elapsedSeconds", "analysis", "failure")}), flush=True)
    if record["status"] != "complete": raise SystemExit(1)


if __name__ == "__main__": main()
