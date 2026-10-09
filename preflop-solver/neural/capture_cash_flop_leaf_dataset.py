#!/usr/bin/env python3
"""Bounded forced leaf coverage from a pinned, offline cash flop policy.

This is distribution-shift training data, not an authentic frequency sample,
not values of a frozen full-hand continuation, and never release activation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np

from cash_flop_leaves import leaf_input, selected_branches
from cash_turn_roots import validate_roots
from cash_value_dataset import build_dataset, validate_dataset
from native_value_dataset import board_family, family_split, identity_hash
from probe_cash_flop_leaves import reference
from run_cash_flop_pilot import sha, validate_solution


def capture(binary: Path, roots_path: Path, policy: Path, pair_report: Path, output: Path,
            split_reference: Path, excluded_roots: Path, seed: int = 971):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("invalid public proposal seed")
    roots = json.loads(roots_path.read_text()); pair = json.loads(pair_report.read_text())
    validate_roots(roots, json.loads(policy.read_text()), sha(policy), "nl25", street="flop")
    if (len(roots["roots"]) > 4 or pair.get("roots_sha256") != sha(roots_path)
            or pair.get("source_policy_sha256") != sha(policy)
            or pair.get("native_binary_sha256") != sha(binary) or pair.get("status") != "research_only"
            or pair.get("active") is not False or len(pair.get("value_network_sha256s", [])) != 2
            or len(pair.get("captures", [])) != 2 * len(roots["roots"])):
        raise ValueError("leaf corpus crossed its frozen bounded flop comparison")
    original = json.loads(split_reference.read_text()); validate_dataset(original)
    if original.get("source_datasets") is not None:
        raise ValueError("leaf coverage needs the original unmixed split reference")
    split = dict(game=original["game"], targets=[dict(board=l["input"]["state"]["board"]) for l in original["labels"]])
    _, tuning, holdout = family_split(split, 937, .2, .2)
    forbidden = {board_family(original["labels"][i]["input"]["state"]["board"]) for i in np.concatenate((tuning, holdout))}
    excluded = json.loads(excluded_roots.read_text())
    validate_roots(excluded, json.loads(policy.read_text()), sha(policy), "nl25", street="flop")
    forbidden.update(board_family(r["solve_input"]["state"]["board"]) for r in excluded["roots"])
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed); tasks = []; omitted = []
    for index, root in enumerate(roots["roots"]):
        if root["solve_input"]["game"] != original["game"]:
            raise ValueError("leaf corpus cannot change the reference's game")
        if board_family(root["solve_input"]["state"]["board"]) in forbidden:
            omitted.append(root["root_sha256"])
            continue
        saved = pair["captures"][index * 2]; path = Path(saved["path"])
        if (saved.get("root_sha256") != root["root_sha256"]
                or saved.get("value_network_sha256") != pair["value_network_sha256s"][0]
                or saved.get("native_binary_sha256") != sha(binary)
                or saved.get("solution_sha256") != sha(path)):
            raise ValueError("leaf coverage needs byte-pinned frozen flop solutions")
        solution = json.loads(path.read_text())
        validate_solution(solution, root["solve_input"], pair["value_network_sha256s"][0], roots["rules_sha256"])
        deck = [c for c in range(52) if c not in root["solve_input"]["state"]["board"]]
        turns = rng.choice(deck, size=2, replace=False)
        for branch in selected_branches(solution):
            history = root["solve_input"]["state"]["public_history"].copy(); branch_rows = []
            for label in branch:
                row = next(row for row in solution["strategies"] if row["public_history"] == history)
                branch_rows.append({key:row[key] for key in ("actor", "public_history", "action_labels", "probabilities")})
                history.append(f"Flop:p{row['actor']}:{label}")
            for turn in turns:
                config = leaf_input(root, solution, int(turn), branch)
                provenance = dict(root_sha256=root["root_sha256"], flop_solution_sha256=sha(path),
                                  branch_labels=branch, public_turn_proposal=int(turn), sampling_seed=seed)
                replay = dict(root=root, branch_policy_rows=branch_rows, **provenance)
                tasks.append((config, provenance, replay))
    if not tasks:
        raise ValueError("leaf coverage supplied no unheldout board families")
    with ThreadPoolExecutor(max_workers=2) as pool:
        generated = list(pool.map(lambda task:reference(binary, output, task[0], task[1]), tasks))
    dataset = build_dataset([path for path, _ in generated])
    dataset["flop_leaf_provenance"] = dict(schema="hu-cash-frozen-flop-leaf-lineage-v1",
        kind="forced-frozen-flop-average-leaves", rules_sha256=roots["rules_sha256"],
        source_policy_sha256=sha(policy), root_corpus_sha256=sha(roots_path),
        flop_pair_report_sha256=sha(pair_report), native_binary_sha256=sha(binary),
        source_flop_value_network_sha256=pair["value_network_sha256s"][0],
        split_reference_sha256=sha(split_reference), excluded_roots_sha256=sha(excluded_roots),
        split_seed=937, rows=[dict(task[2], label_input_sha256=identity_hash(label["input"]))
                             for task, (_, label) in zip(tasks, generated)])
    validate_dataset(dataset)
    path = output / "dataset.json"
    path.write_text(json.dumps(dataset, separators=(",", ":")) + "\n")
    report = dict(schema="hu-cash-frozen-flop-leaf-coverage-report-v1", status="research_only", active=False,
                  dataset_sha256=sha(path), contexts=len(tasks), omitted_forbidden_roots=omitted,
                  source_pair_report_sha256=sha(pair_report), native_binary_sha256=sha(binary),
                  maximum_reference_nash_conv_bb=max(label["metrics"]["cash"]["nash_conv_bb_per_hand"] for _, label in generated),
                  maximum_accounting_residual_bb=max(label["metrics"]["cash"]["conservation_residual_bb"] for _, label in generated),
                  full_game_exploitability="unmeasured",
                  limitations=["Forced two-branch/two-turn coverage per root, not authentic leaf frequencies.",
                               "Only seed one's frozen flop policy supplies beliefs; no full-game safety guarantee.",
                               "Exact finite-budget fresh subgames, not a pinned full-hand continuation.",
                               "External byte-pinned captures are needed to verify extracted branch rows against complete flop solutions."])
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("binary", "roots", "policy", "pair-report", "output", "split-reference", "excluded-roots"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    parser.add_argument("--seed", default=971, type=int)
    args = parser.parse_args()
    print(json.dumps(capture(args.binary, args.roots, args.policy, args.pair_report, args.output,
                             args.split_reference, args.excluded_roots, args.seed), indent=2))
