"""Append 128 native labels at current and explicit-deviation training beliefs.

Eight 16-state captures: two preflop seeds x two diagnosed histories x actual /
forced-call ranges. Use only matched-comparison fitting boards. Existing 508
targets are immutable; independent response roots and held-out families excluded.
"""
import argparse
import copy
import gzip
import json
from pathlib import Path
import signal
import threading
import time

import numpy as np
import native_value_dataset as native
from run_native_value_pilot import guarded, merge_captures, read_capture, test_command
from run_native_value_preflight import atomic_json, sha256
from run_search_distribution_pilot import TEST


def forced_call_public(row, labels):
    """Legitimate public intervention: call with every hand reaching this node.

    Restore the acting player's PRE-action range; never infer unseen cards or
    divide by the original call probability (which can be zero).
    """
    public = copy.deepcopy(row["publicInput"])
    label_index = {label: i for i, label in enumerate(labels)}
    ranks = "23456789TJQKA"
    weights = []
    for a, b in native.COMBOS:
        high, low = max(a//4, b//4), min(a//4, b//4)
        label = ranks[high] + ranks[low] + ("" if high == low else "s" if a % 4 == b % 4 else "o")
        weights.append(0.0 if a in public["board"] or b in public["board"]
                       else row["ownPrefixReach"][label_index[label]])
    weights = np.asarray(weights)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("invalid forced-call pre-action reach")
    public["ranges"][row["seat"]] = (weights/weights.sum()).tolist()
    return public


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "comparison", "corpus", "split-reference", "students"):
        p.add_argument("--"+name, type=Path, required=True)
        p.add_argument("--"+name+"-sha256", required=True)
    p.add_argument("--exclude-root", type=Path, action="append", required=True)
    p.add_argument("--exclude-root-sha256", action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    pinned = {}
    for name in ("binary", "comparison", "corpus", "split_reference", "students"):
        path = getattr(args, name).resolve(); setattr(args, name, path)
        pinned[str(path)] = getattr(args, name+"_sha256")
    if len(args.exclude_root) != len(args.exclude_root_sha256): raise ValueError("missing excluded root hash")
    for path, digest in zip(args.exclude_root, args.exclude_root_sha256): pinned[str(path.resolve())] = digest
    for name in ("native_value_dataset.py", "run_native_value_pilot.py", "run_native_value_preflight.py", "worker_resources.py"):
        path = Path(__file__).with_name(name).resolve(); pinned[str(path)] = sha256(path)
    if any(sha256(Path(k)) != v for k, v in pinned.items()): raise ValueError("input changed")
    comparison, students = [json.loads(path.read_text()) for path in (args.comparison, args.students)]
    if comparison.get("status") != "complete" or len(comparison.get("summaries", [])) != 2:
        raise ValueError("finish the matched comparison before Path A")
    if students.get("status") != "complete" or {p["seed"] for p in students["predictions"]} != {10601,10602}:
        raise ValueError("verified paired proposers required")
    corpus, reference = read_capture(args.corpus), read_capture(args.split_reference)
    if len(corpus["targets"]) != 508: raise ValueError("expected immutable 508-state reference")
    split = native.family_split(corpus, 10601, .25, .25, reference=reference, refresh_training=True)
    forbidden = {native.board_family(corpus["targets"][i]["board"]) for i in np.concatenate((split[1], split[2]))}
    forbidden.update(native.board_family(json.loads(p.read_text())["public"]["board"]) for p in args.exclude_root)
    captured = {}
    for job in comparison["jobs"]:
        path = Path(job["output"])
        if sha256(path) != job["outputSha256"]: raise ValueError("comparison capture changed")
        pinned[str(path)] = job["outputSha256"]
        data = json.loads(path.read_text())
        captured[job["seed"], job["strength"], job["board"]] = data
        if job["board"] >= 4: forbidden.add(native.board_family(data["board"]))
    models = {}
    for entry in students["predictions"]:
        path = Path(entry["model"])
        if entry["maximumParityErrorBb"] > 1e-4 or sha256(path) != entry["modelSha256"]:
            raise ValueError("unverified proposer")
        models[entry["seed"]] = path; pinned[str(path)] = entry["modelSha256"]
    selected = []
    # Deterministic allocation, never select labels based on evaluation values.
    for seed in range(2):
        for root in range(2):
            eligible = [b for b in range(4) if native.board_family(captured[seed, 1, b]["board"]) not in forbidden]
            if not eligible: raise ValueError("no training-only fitting family available")
            board = eligible[(2*seed + root) % len(eligible)]
            data = captured[seed, 1, board]; row = data["records"][root]
            for forced in (False, True):
                public = forced_call_public(row, data["classes"]) if forced else row["publicInput"]
                selected.append((seed, root, board, forced, public))
    output = args.output.resolve(); output.mkdir(exist_ok=False)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: stop.set())
    timer = threading.Timer(3600, stop.set); timer.daemon = True; timer.start()
    started = time.monotonic()
    record = dict(schema="path-a-current-deviation-native-labels-v1", status="running", pinnedInputs=pinned,
        runnerSha256=sha256(Path(__file__)), workers=[], releaseAccepted=False, additionalStates=128,
        maximumStageSeconds=3600, maximumWorkerMemoryBytes=2*1024**3,
        interpretation="Current preflop-derived call ranges and explicit always-call deviations on training families only. Native turn/river targets, never model self-labels. Existing target prefix and held-outs unchanged.")
    atomic_json(output/"manifest.json", record)
    try:
        paths = []
        for index, (seed, root, board, forced, public) in enumerate(selected):
            if stop.is_set() or any(sha256(Path(k)) != v for k, v in pinned.items()): raise ValueError("stopped or changed input")
            source = output/f"root-{index}.json"
            atomic_json(source, dict(schema="native-flop-public-root-v1", game=corpus["game"], public=public))
            target, policy = output/f"capture-{index}.json.gz", output/f"policy-{index}.json"
            model = models[10601+seed]
            worker = guarded(test_command(args.binary, TEST), dict(
                POKER_SEARCH_ROOT=str(source), POKER_SEARCH_ROOT_SHA=sha256(source),
                POKER_SEARCH_MODEL=str(model), POKER_SEARCH_MODEL_SHA=pinned[str(model)],
                POKER_SEARCH_OUTPUT=str(target), POKER_SEARCH_POLICY_OUTPUT=str(policy),
                POKER_SEARCH_TRUNK_SEED=str(100101+seed), POKER_SEARCH_SAMPLE_SEED=str(71301+index),
                POKER_SEARCH_LABEL_COUNT="16", POKER_SEARCH_WORKERS="2", POKER_SEARCH_FLOP_ITERATIONS="128",
                POKER_SEARCH_VERIFY_POLICY_PARITY="1" if index == 0 else "0"), output/f"worker-{index}", 600, 2*1024**3, stop)
            value = read_capture(target)
            if (len(value["targets"]) != 16 or value["source_public_input_sha256"] != sha256(source)
                    or value["source_policy_sha256"] != sha256(policy) or value["proposal_model_sha256"] != pinned[str(model)]
                    or any(native.board_family(t["board"]) in forbidden for t in value["targets"])):
                raise ValueError("target count, provenance or held-out exclusion failed")
            paths.append(target)
            record["workers"].append(dict(seed=seed, root=root, board=board, forcedCall=forced,
                path=str(target), sha256=sha256(target), worker=worker))
            atomic_json(output/"manifest.json", record)
            print(json.dumps(dict(event="path-a-native-labels", captures=index+1, total=8, seconds=worker["workerElapsedSeconds"])), flush=True)
        merged = merge_captures(paths, search_proposals=True, reference=corpus, maximum_states=640)
        actual = native.family_split(merged, 10601, .25, .25, reference=reference, refresh_training=True)
        if (merged["targets"][:508] != corpus["targets"] or len(merged["targets"]) != 636
                or any(actual[i].tolist() != split[i].tolist() for i in (1, 2))):
            raise ValueError("old target/split parity failed")
        encoded = json.dumps(merged, separators=(",", ":"), allow_nan=False).encode()
        compressed = gzip.compress(encoded, mtime=0)
        if len(encoded) > 256*1024**2 or len(compressed) > 128*1024**2: raise ValueError("corpus memory/storage budget exceeded")
        target = output/"corpus.json.gz"
        with target.open("xb") as stream: stream.write(compressed)
        if stop.is_set() or any(sha256(Path(k)) != v for k, v in pinned.items()): raise ValueError("stopped or changed input")
        record.update(status="complete", analysis=dict(states=636, corpusSha256=sha256(target),
            compressedBytes=len(compressed), decodedBytes=len(encoded), exactOldTargetParity=True,
            split={k:v.tolist() for k,v in zip(("train", "tuning", "holdout"), actual)}))
    except (OSError, ValueError, KeyError) as error:
        stop.set(); record.update(status="failed", failure=str(error))
    finally:
        timer.cancel(); record["elapsedSeconds"] = time.monotonic()-started
        atomic_json(output/"manifest.json", record)
    print(json.dumps({k:record.get(k) for k in ("status", "elapsedSeconds", "failure")}), flush=True)
    if record["status"] != "complete": raise SystemExit(1)


if __name__ == "__main__": main()
