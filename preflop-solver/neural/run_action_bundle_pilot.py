"""Bounded complete-branch label pilot, preceded by cached affine parity.

Immutable training-only prefix snapshots, paired common public turn samples,
finite native labels. No fitting, model activation, or paid compute.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import gzip
import json
from pathlib import Path
import signal
import threading
import time

import numpy as np

import native_value_dataset as native
from action_contrast_dataset import build_groups
from decision_pilot_screen import verified_rows
from run_native_value_pilot import guarded, test_command, read_capture
from run_native_value_preflight import atomic_json, sha256
from run_postflop_gap_pilot import PilotMemoryGuard
from run_flop_update_pilot import DEFAULT_BASELINE

BASE = "blueprint::public_belief::counterfactual_turn::flop_pilot::frozen_response::action_bundles::"


def cached_targets(prefix, packet):
    """Reconstruct inputs, not policies, for a read-only full-chance parity test."""
    targets = []
    states = {tuple(s["public_history"]): s for s in prefix["leaf_states"]}
    for leaf in packet["leaves"]:
        state = copy.deepcopy(states[tuple(leaf["history"])])
        state["board"] += [packet["turn"]]
        legal = native.legal_combos(state["board"])
        raw_ranges = np.asarray(state["ranges"])
        raw_ranges[:, ~legal] = 0
        state["ranges"] = raw_ranges.tolist()
        totals = raw_ranges.sum(1)
        ranges = np.divide(raw_ranges, totals[:, None], out=np.zeros_like(raw_ranges), where=totals[:, None] > 0)
        masses = native.compatible_masses(ranges)
        raw_mass = masses * totals[::-1, None]
        profile, best = np.asarray(leaf["profile_bb"]), np.asarray(leaf["best_response_bb"])
        completed = np.where(ranges == 0, best, profile)
        conditional = np.divide(completed, raw_mass, out=np.zeros_like(completed), where=raw_mass > 0)
        joint = float(np.sum(raw_ranges[0] * raw_mass[0]))
        gains = np.sum(raw_ranges * (best - profile), axis=1) / joint if joint > 0 else None
        targets.append(dict(iteration=1, board=state["board"], actor=state["actor"],
            invested_bb=state["invested_bb"], public_state=state, ranges=ranges.tolist(),
            raw_reach_totals=totals.tolist(), opponent_compatible_mass=masses.tolist(),
            counterfactual_values_bb=conditional.tolist(), raw_counterfactual_bb=completed.tolist(),
            raw_profile_counterfactual_bb=profile.tolist(), raw_best_response_counterfactual_bb=best.tolist(),
            completed_zero_own_reach=((ranges == 0) & legal).sum(axis=1).tolist(),
            conditional_response_gain_bb=None if gains is None else gains.tolist(),
            policy_sha256=leaf["policy_sha256"], policy_rows=1, turn_iterations=packet["turn_iterations"],
            value_semantics=native.SEMANTICS, state_distribution="cached_profile_parity_only"))
    return dict(schema="hu-frozen-action-turn-labels-v1", releaseAccepted=False,
        candidate_sha256=packet["candidate_sha256"], turn=packet["turn"],
        turn_iterations=packet["turn_iterations"], targets=targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "source-manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibrate-case", action="store_true")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--calibration-sha256")
    parser.add_argument("--split-reference", type=Path)
    parser.add_argument("--split-reference-sha256")
    args = parser.parse_args()
    pinned = {}
    for name in ("binary", "source_manifest", "calibration", "split_reference"):
        p, digest = getattr(args, name), getattr(args, name + "_sha256")
        if p is None:
            if digest is not None: raise ValueError("hash without input")
            continue
        p = p.resolve(); setattr(args, name, p)
        if sha256(p) != digest: raise ValueError("pinned input changed: " + name)
        pinned[str(p)] = digest
    here = Path(__file__).resolve().parent
    for name in ("action_contrast_dataset.py", "run_action_bundle_pilot.py", "native_value_dataset.py",
                 "run_native_value_pilot.py", "worker_resources.py", "audit_native_flop_response.mjs"):
        path = here / name; pinned[str(path)] = sha256(path)
    if args.output.exists(): raise ValueError("never overwrite a bundle stage")
    args.output = args.output.resolve(); args.output.mkdir()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: stop.set())
    timer = threading.Timer(1800 if args.calibrate_case else 5400, stop.set); timer.daemon = True; timer.start()
    memory_guard = PilotMemoryGuard(stop, args.output / "system-memory.json"); memory_guard.start()
    started = time.monotonic()
    record = dict(schema="action-bundle-pilot-v1", status="running", releaseAccepted=False,
                  pinnedInputs=pinned, calibrationOnly=args.calibrate_case, families=[],
                  maximumStageSeconds=1800 if args.calibrate_case else 5400,
                  maximumWorkers=2, maximumWorkerMemoryBytes=2 * 1024**3,
                  sampleSeed=20261004, auxiliaryTarget="profile_on_support_consistent_holdings",
                  offSupportTarget="existing_completed_values_calibration_only")
    atomic_json(args.output / "manifest.json", record)

    def check():
        if stop.is_set() or any(sha256(Path(p)) != h for p, h in pinned.items()):
            raise ValueError("resource/operator stop or pinned identity changed")

    def export(candidate, output, test, extra=None):
        check()
        env = dict(POKER_NATIVE_FLOP_CANDIDATE=str(candidate),
                   POKER_NATIVE_FLOP_CANDIDATE_SHA=pinned[str(candidate)], POKER_NATIVE_FLOP_OUTPUT=str(output))
        worker = guarded(test_command(args.binary, BASE + test), {**env, **(extra or {})},
                         output.parent / (output.stem + "-worker"), 900, 2 * 1024**3, stop)
        return dict(path=str(output), sha256=sha256(output), worker=worker)

    try:
        if args.calibrate_case:
            _, rows = verified_rows(args.source_manifest, args.source_manifest_sha256)
            row = next(r for r in rows if r["spot"] == "three-bet-high-rainbow" and r["seed"] == 100101)
            case = args.source_manifest.parent / row["spot"] / str(row["seed"])
            candidate = case / "candidate.json"; pinned[str(candidate)] = row["candidateSha256"]
            prefix_path = args.output / "prefix.json"
            record["prefixExport"] = export(candidate, prefix_path, "saved_action_bundle_prefix")
            prefix = json.loads(prefix_path.read_text())
            packets = [cached_targets(prefix, json.loads((case / "packets" / f"turn-{t}.json").read_text()))
                       for t in range(52) if t not in prefix["root"]["board"]]
            groups, _ = build_groups(prefix, packets, require_full_chance=True)
            equity = DEFAULT_BASELINE / "jobs" / row["spot"] / str(row["seed"]) / "equity.json"
            command = ["node", "--input-type=module", "-e",
                "import {audit} from './audit_native_flop_response.mjs'; console.log(JSON.stringify(audit(...process.argv.slice(1),{actionDiagnostics:true})));",
                str(candidate), str(case / "packets"), str(equity), str(case / "response.json")]
            record["jsAudit"] = guarded(command, {}, args.output / "js-audit", 180, stop=stop)
            audit = json.loads((args.output / "js-audit" / "worker.log").read_text())
            max_difference = 0.
            for group in groups:
                node = next(n for n in audit["actionDiagnostics"] if n["history"] == group.history)
                for hand in node["costlyHands"]:
                    max_difference = max(max_difference, float(np.max(np.abs(group.target[:, hand["combo"]] - hand["actionEvBb"]))))
                weighted = (group.target * group.weights[None, :]).sum(1) / group.weights.sum()
                max_difference = max(max_difference, float(np.max(np.abs(weighted - node["reachWeightedActionEvBb"]))))
            if max_difference > 1e-8: raise ValueError("affine backup differs from cached native/JS actions")
            record["maximumCachedActionDifferenceBb"] = max_difference
            record["groups"] = [g.report() for g in groups]
        else:
            if args.calibration is None or args.split_reference is None:
                raise ValueError("complete cached parity and pinned family split required before labeling")
            calibration = json.loads(args.calibration.read_text())
            if calibration.get("status") != "complete" or calibration.get("maximumCachedActionDifferenceBb", 1) > 1e-8:
                raise ValueError("cached affine calibration did not pass")
            source = json.loads(args.source_manifest.read_text())
            if source.get("status") != "complete" or source.get("schema") != "matched-native-label-control-v1":
                raise ValueError("complete matched snapshots required")
            reference = read_capture(args.split_reference)
            train, tuning, holdout = native.family_split(reference, 10601, .25, .25)
            allowed = {native.board_family(reference["targets"][i]["board"]) for i in train}
            forbidden = {native.board_family(reference["targets"][i]["board"]) for i in np.concatenate((tuning, holdout))}
            benchmark = json.loads((DEFAULT_BASELINE / "protocol.json").read_text())
            forbidden |= {native.board_family(s["board"]) for s in benchmark["spots"]}
            selected = []
            for index in [3, 2, 4]: # measured expensive family first
                row = next(c for c in source["cases"] if c["root"] == index)
                candidate = args.source_manifest.parent / f"root-{index}" / "proposer.json"
                if sha256(candidate) != row["proposerSha256"]: raise ValueError("proposer changed")
                pinned[str(candidate)] = row["proposerSha256"]
                work = args.output / f"root-{index}"; work.mkdir()
                prefix_path = work / "prefix.json"
                receipt = export(candidate, prefix_path, "saved_action_bundle_prefix")
                prefix = json.loads(prefix_path.read_text())
                family = native.board_family(prefix["root"]["board"])
                if family not in allowed or family in forbidden: raise ValueError("bundle leaks held-out/consumed family")
                if len(prefix["leaves"]) != (5 if index == 2 else 9): raise ValueError("complete leaf count changed")
                rng = np.random.default_rng(record["sampleSeed"] + index)
                cards = rng.permutation([c for c in range(52) if c not in prefix["root"]["board"]]).tolist()
                family_record = dict(root=index, prefix=receipt, candidateSha256=row["proposerSha256"],
                    candidatePath=str(candidate), family=list(family), trainingTurns=cards[:8],
                    sensitivityTurns=cards[8:16], labels=[], sentinels=[])
                record["families"].append(family_record)
                selected.append((work, prefix, candidate, family_record))
            atomic_json(args.output / "manifest.json", record)

            def label(job, turn, iterations=64):
                work, _, candidate, entry = job
                path = work / f"turn-{turn}-{iterations}.json"
                return export(candidate, path, "saved_action_bundle_turn_labels", dict(
                    POKER_NATIVE_FLOP_TURN=str(turn), POKER_ACTION_LABEL_ITERATIONS=str(iterations)))

            with ThreadPoolExecutor(max_workers=2) as pool:
                job = selected[0]; before = time.monotonic()
                job[3]["labels"] += list(pool.map(lambda t: label(job, t), job[3]["trainingTurns"][:2]))
                preflight_seconds = time.monotonic() - before
                # Full labels + one complete 256 sentinel turn per family,
                # plus a disjoint sensitivity turn per family; 1.5x safety margin.
                projected = preflight_seconds / 18 * (184 + 4 * 23 + 23) * 1.5
                record["preflight"] = dict(seconds=preflight_seconds, labels=18, projectedLabelSeconds=projected)
                atomic_json(args.output / "manifest.json", record)
                print(json.dumps(dict(event="bundle-preflight", **record["preflight"])), flush=True)
                if projected > 5400 - (time.monotonic() - started):
                    raise ValueError("complete bundle/sentinel work exceeds declared 90-minute cap")
                for job in selected:
                    done = {json.loads(Path(r["path"]).read_text())["turn"] for r in job[3]["labels"]}
                    turns = [t for t in job[3]["trainingTurns"] if t not in done]
                    job[3]["labels"] += list(pool.map(lambda t: label(job, t), turns))
                    job[3]["sentinels"] = list(pool.map(lambda item: label(job, *item),
                        [(job[3]["trainingTurns"][0], 256), (job[3]["sensitivityTurns"][0], 64)]))
                    packets = [json.loads(Path(r["path"]).read_text()) for r in job[3]["labels"]]
                    groups, ordered = build_groups(job[1], packets)
                    # Separate bounded calibration corpus; never append to 615.
                    corpus = dict(schema=native.SCHEMA, game=job[1]["game"],
                        source_public_input_sha256=job[3]["candidateSha256"],
                        source_policy_sha256=job[3]["candidateSha256"], validation=dict(status="research_only"),
                        flop_iterations=1, turn_iterations=64, maximum_states=len(ordered),
                        observed_queries=len(ordered), targets=ordered)
                    native.validate_dataset(corpus)
                    path = job[0] / "calibration.json.gz"
                    with path.open("xb") as file: file.write(gzip.compress(json.dumps(corpus, separators=(",", ":")).encode(), mtime=0))
                    job[3]["calibrationCorpus"] = dict(path=str(path), sha256=sha256(path), states=len(ordered))
                    job[3]["groups"] = [g.report() for g in groups]
                    atomic_json(args.output / "manifest.json", record)
                    print(json.dumps(dict(event="bundle-family-complete", root=job[3]["root"], states=len(ordered), groups=job[3]["groups"])), flush=True)
        check(); record["status"] = "complete"
    except Exception as error:
        stop.set(); record["status"] = "failed"; record["failure"] = str(error)
        raise
    finally:
        timer.cancel()
        record["elapsedSeconds"] = time.monotonic() - started
        record["systemMemoryGuard"] = memory_guard.finish()
        atomic_json(args.output / "manifest.json", record)
    print(json.dumps(record), flush=True)


if __name__ == "__main__": main()
