"""Complete one frozen TRAIN family's 49 turns, reusing verified native64 labels.

Only chance coverage changes. No policy update, fitting, or model promotion.
Each resulting calibration shard remains under existing native corpus guards.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import gzip
import json
import math
from pathlib import Path
import signal
import statistics
import threading
import time

from action_contrast_dataset import build_groups
from inspect_action_bundle_quality import compare
import native_value_dataset as native
from run_action_bundle_pilot import BASE
from run_native_value_pilot import guarded, test_command
from run_native_value_preflight import atomic_json, sha256
from run_postflop_gap_pilot import PilotMemoryGuard


def verified_json(receipt):
    path = Path(receipt["path"])
    if sha256(path) != receipt["sha256"]:
        raise ValueError("immutable action-bundle input changed")
    return json.loads(path.read_text())


def begin_finalization(stop, timer):
    """No native work after the label cap; bounded validation gets its own phase."""
    if stop.is_set(): raise ValueError("resource stop before finalization")
    timer.cancel()
    final_timer = threading.Timer(180, stop.set)
    final_timer.daemon = True; final_timer.start()
    return final_timer


def inventory(prefix, receipts):
    """No duplicate native64 turn, wrong candidate, or forbidden card can resume."""
    legal = set(range(52)) - set(prefix["root"]["board"])
    result = {}
    for receipt in receipts:
        packet = verified_json(receipt)
        if (packet.get("schema") != "hu-frozen-action-turn-labels-v1"
                or packet.get("candidate_sha256") != prefix["candidate_sha256"]
                or packet.get("releaseAccepted") is not False
                or type(packet.get("turn")) is not int or packet["turn"] not in legal):
            raise ValueError("invalid cached turn identity")
        if packet["turn_iterations"] != 64:
            if packet["turn_iterations"] not in (256, 1024):
                raise ValueError("unplanned native reference budget")
            continue
        if packet["turn"] in result:
            raise ValueError("duplicate cached native64 turn")
        result[packet["turn"]] = (receipt, packet)
    return result


def diagnose(prefix, complete, original, high):
    full, _ = build_groups(prefix, complete, require_full_chance=True)
    by_turn = {p["turn"]: p for p in complete}
    sample_groups = {}
    for name, turns in [("eightA", original["trainingTurns"]),
                        ("eightB", original["sensitivityTurns"]),
                        ("sixteen", original["trainingTurns"] + original["sensitivityTurns"])]:
        sample_groups[name], _ = build_groups(prefix, [by_turn[t] for t in turns])
    if len(high) != 1 or high[0]["turn_iterations"] != 256:
        raise ValueError("one planned native256 sentinel required")
    low_groups, _ = build_groups(prefix, [by_turn[high[0]["turn"]]])
    high_groups, _ = build_groups(prefix, high)
    upgraded, _ = build_groups(prefix, [high[0] if p["turn"] == high[0]["turn"] else p for p in complete],
                               require_full_chance=True)
    return dict(schema="full-chance-action-bundle-quality-v1", root=original["root"],
        all49Turns=True, nativeBudget=64, releaseAccepted=False, groups=[dict(
            history=g.history, support=g.report(),
            sampleLossAgainstAll49={name: compare(g, groups[i]) for name, groups in sample_groups.items()},
            sentinel64To256=compare(low_groups[i], high_groups[i]),
            oneSentinelUpgradeEffectOnAll49=compare(g, upgraded[i]),
            reachWeightedActionEvBb=((g.target * g.weights[None, :]).sum(1) / g.weights.sum()).tolist())
            for i, g in enumerate(full)],
        interpretation="Exact public-turn integration of finite native64 labels; not exact continuation values or full-game exploitability.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-sha256", required=True)
    p.add_argument("--root", type=int, choices=(2, 3, 4), default=2)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); a.source, a.output = a.source.resolve(), a.output.resolve()
    if sha256(a.source) != a.source_sha256 or a.output.exists():
        raise ValueError("changed source or existing stage")
    source = json.loads(a.source.read_text())
    if (source.get("status") != "complete" or source.get("calibrationOnly")
            or source.get("schema") != "action-bundle-pilot-v1"):
        raise ValueError("complete pinned training/sensitivity source required")
    originals = [f for f in source["families"] if f["root"] == a.root]
    if len(originals) != 1: raise ValueError("ambiguous training family")
    original = originals[0]; prefix = verified_json(original["prefix"])
    cached = inventory(prefix, original["labels"] + original["sentinels"])
    if (len(cached) != 16 or len(original["trainingTurns"]) != 8
            or len(original["sensitivityTurns"]) != 8
            or set(cached) != set(original["trainingTurns"] + original["sensitivityTurns"])):
        raise ValueError("complete disjoint 8+8 reuse required")
    # Validate ranges, complete branches, and card/accounting contracts before
    # spending any new compute. These are already TRAIN-only pinned snapshots.
    build_groups(prefix, [packet for _, packet in cached.values()])
    binaries = [path for path in source["pinnedInputs"] if "local-compact-binaries/" in path]
    if len(binaries) != 1: raise ValueError("ambiguous native binary")
    binary = Path(binaries[0])
    pinned = {str(a.source): a.source_sha256, str(binary): source["pinnedInputs"][str(binary)],
              original["candidatePath"]: original["candidateSha256"],
              original["prefix"]["path"]: original["prefix"]["sha256"],
              **{r["path"]: r["sha256"] for r in original["labels"] + original["sentinels"]}}
    for name in ("complete_action_bundle_chance.py", "action_contrast_dataset.py",
                 "inspect_action_bundle_quality.py", "run_native_value_pilot.py", "worker_resources.py"):
        path = Path(__file__).with_name(name); pinned[str(path)] = sha256(path)
    if any(sha256(Path(path)) != digest for path, digest in pinned.items()):
        raise ValueError("pinned source identity changed")
    turns = sorted(set(range(52)) - set(prefix["root"]["board"]))
    missing = [t for t in turns if t not in cached]
    measured = [r["worker"]["workerElapsedSeconds"] for r, _ in cached.values()]
    projected = 1.5 * statistics.median(measured) * math.ceil(len(missing) / 2)
    if projected > 2700: raise ValueError("projected chance completion exceeds 45-minute cap")
    a.output.mkdir(); started = time.monotonic(); stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: stop.set())
    timer = threading.Timer(2700, stop.set); timer.daemon = True; timer.start()
    memory = PilotMemoryGuard(stop, a.output / "system-memory.json").start()
    family = copy.deepcopy(original)
    family.update(trainingTurns=turns, sampledTrainingTurns=original["trainingTurns"],
                  sampledSensitivityTurns=original["sensitivityTurns"],
                  labels=[cached[t][0] for t in sorted(cached)], all49Turns=False)
    record = dict(schema="full-chance-action-bundle-pilot-v1", status="running", releaseAccepted=False,
        sourceManifestSha256=a.source_sha256, pinnedInputs=pinned, families=[family],
        reusedNative64Turns=16, newNative64Turns=33, maximumSeconds=2880,
        maximumLabelingSeconds=2700, maximumFinalizationSeconds=180,
        maximumWorkers=2, maximumWorkerMemoryBytes=2*1024**3, projectedLabelSeconds=projected)
    atomic_json(a.output / "manifest.json", record)

    def label(turn):
        if stop.is_set() or sha256(binary) != pinned[str(binary)]:
            raise ValueError("stage stopped or binary changed")
        output = a.output / f"turn-{turn}-64.json"
        worker = guarded(test_command(binary, BASE + "saved_action_bundle_turn_labels"), dict(
            POKER_NATIVE_FLOP_CANDIDATE=original["candidatePath"],
            POKER_NATIVE_FLOP_CANDIDATE_SHA=original["candidateSha256"],
            POKER_NATIVE_FLOP_OUTPUT=str(output), POKER_NATIVE_FLOP_TURN=str(turn),
            POKER_ACTION_LABEL_ITERATIONS="64"), a.output / f"turn-{turn}-worker", 900, 2*1024**3, stop)
        return dict(path=str(output), sha256=sha256(output), worker=worker)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(label, turn) for turn in missing]
            for future in as_completed(futures):
                try: receipt = future.result()
                except Exception: stop.set(); raise
                family["labels"].append(receipt)
                atomic_json(a.output / "manifest.json", record)
                print(json.dumps(dict(event="full-chance-turn", root=a.root, completeTurns=len(family["labels"]))), flush=True)
        if stop.is_set(): raise ValueError("resource stop")
        record["labelingElapsedSeconds"] = time.monotonic() - started
        timer = begin_finalization(stop, timer)
        complete = [verified_json(r) for r in family["labels"]]
        groups, ordered = build_groups(prefix, complete, require_full_chance=True)
        high = [verified_json(r) for r in original["sentinels"]
                if verified_json(r)["turn_iterations"] == 256]
        quality = diagnose(prefix, complete, original, high)
        atomic_json(a.output / "quality.json", quality)
        corpus = dict(schema=native.SCHEMA, game=prefix["game"],
            source_public_input_sha256=original["candidateSha256"], source_policy_sha256=original["candidateSha256"],
            validation=dict(status="research_only"), flop_iterations=1, turn_iterations=64,
            maximum_states=len(ordered), observed_queries=len(ordered), targets=ordered)
        native.validate_dataset(corpus)
        encoded = json.dumps(corpus, separators=(",", ":"), allow_nan=False).encode()
        if len(ordered) > 640 or len(encoded) > 256*1024**2:
            raise ValueError("full-chance shard exceeds existing corpus guards")
        path = a.output / "calibration.json.gz"
        with path.open("xb") as file: file.write(gzip.compress(encoded, mtime=0))
        if path.stat().st_size > 128*1024**2: raise ValueError("compressed calibration shard exceeds guard")
        family.update(all49Turns=True, groups=[g.report() for g in groups], calibrationCorpus=dict(
            path=str(path), sha256=sha256(path), states=len(ordered)))
        record["quality"] = dict(path=str(a.output / "quality.json"), sha256=sha256(a.output / "quality.json"))
        if stop.is_set() or any(sha256(Path(path)) != digest for path, digest in pinned.items()):
            raise ValueError("finalization stopped or immutable input changed during reference")
        record["status"] = "complete"
    except Exception as error:
        record["status"] = "failed"; record["failure"] = str(error); stop.set(); raise
    finally:
        timer.cancel(); record["elapsedSeconds"] = time.monotonic() - started
        record["systemMemoryGuard"] = memory.finish(); atomic_json(a.output / "manifest.json", record)
    print(json.dumps(dict(status=record["status"], root=a.root, seconds=record["elapsedSeconds"], quality=quality)), flush=True)


if __name__ == "__main__": main()
