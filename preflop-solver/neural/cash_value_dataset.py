"""Strict own-payoff turn-start research labels, separate from Home corpora."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from cash_profiles import PAYOFF_CONTRACT, rules_digest, equal_cash_investment_units
from native_value_dataset import compatible_masses, legal_combos, identity_hash, family_split, board_family
from cash_flop_leaves import leaf_input
from cash_turn_roots import root_fingerprint

SCHEMA = "hu-cash-turn-start-cfv-dataset-v1"
LABEL_SCHEMA = "hu-cash-turn-river-continuation-values-v1"
NETWORK_SCHEMA = "hu-cash-public-belief-combo-value-network-v2"
POOLED_NETWORK_SCHEMA = "hu-cash-public-belief-combo-value-network-v3"
BLOCKER_POOLED_NETWORK_SCHEMA = "hu-cash-public-belief-combo-value-network-v4"
BASELINE_CONDITIONED_NETWORK_SCHEMA = "hu-cash-public-belief-combo-value-network-v5"
BASELINE_CONDITIONED_CONTRACT = "cash-turn-start-cfv-full-stack-baseline-conditioned-v1"
PREDICTION_CONTRACT = "cash-turn-start-cfv-full-stack-v1"
BLOCKER_POOLED_CONTRACT = "cash-turn-start-cfv-full-stack-blocker-pooled-v1"
MAX_FLAT_SOURCES = 5  # One bounded coverage extension; still reject nested merges.


def validate_label(label: dict) -> None:
    if label.get("schema") != LABEL_SCHEMA or label.get("validation", {}).get("status") != "research_only":
        raise ValueError("cash labels must be explicit finite-budget own-payoff research references")
    config = label["input"]
    game, state = config["game"], config["state"]
    rules = game["cash_rules"]
    digest = rules_digest(rules)
    if game["effective_stack_bb"] != 20 or game["small_blind_bb"] != .4 or game["big_blind_bb"] != 1:
        raise ValueError("cash value pilot currently supports only pinned 20bb NL25/control games")
    board = state["board"]
    if len(board) != 4 or len(set(board)) != 4 or any(type(c) is not int or not 0 <= c < 52 for c in board):
        raise ValueError("invalid turn board")
    if (state["street"] != "turn" or state["actor"] not in (0, 1)
            or state["street_invested_bb"] != [0, 0] or state["aggressions"] != 0
            or state["checks"] != 0 or state["last_full_raise_bb"] != 1
            or state["raise_reopened"] is not True or state.get("trajectory", [])
            or state["public_history"] != ["public_belief:turn_start"]):
        raise ValueError("cash feature contract only describes fresh turn roots")
    investments = state["invested_bb"]
    equal_cash_investment_units(investments,rules["unitsPerBb"])
    if any(v >= 20 for v in investments):
        raise ValueError("cash investments must be equal and cent aligned")
    ranges = np.asarray(state["ranges"], dtype=float)
    values = np.asarray(label["counterfactual_values_bb"], dtype=float)
    masses = np.asarray(label["opponent_compatible_mass"], dtype=float)
    if any(v.shape != (2, 1326) or not np.isfinite(v).all() for v in (ranges, values, masses)):
        raise ValueError("cash values must be finite paired exact-combo vectors")
    legal = legal_combos(board)
    if (ranges < 0).any() or (ranges[:, ~legal] != 0).any() or not np.allclose(ranges.sum(axis=1), 1, atol=1e-10):
        raise ValueError("cash beliefs are not normalized legal reaches")
    if not np.allclose(masses, compatible_masses(ranges), atol=1e-7, rtol=1e-6):
        raise ValueError("cash labels have incorrect blocker-compatible mass")
    if (values[:, ~legal] != 0).any() or (np.abs(values) > 20 + 1e-5).any():
        raise ValueError("cash values violate card removal or full-stack bounds")
    metrics, cash = label["metrics"], label["metrics"]["cash"]
    if (cash["rules_sha256"] != digest or label["joint_iterations"] != config["iterations"]
            or metrics["exact_river_cards"] != 48 or config["iterations"] < 2
            or config["averaging_delay"] >= config["iterations"]):
        raise ValueError("cash continuation identity/budget mismatch")
    weights = ranges * masses
    joint = weights.sum(axis=1)
    if min(joint) <= 0 or not np.allclose(joint[0], joint[1], atol=1e-7):
        raise ValueError("cash label lacks compatible joint belief")
    aggregate = np.sum(weights * values, axis=1) / joint
    house = cash["expected_house_rake_bb"]
    if (not np.isfinite(house) or not 0 <= house <= rules["rake"]["capUnits"] / rules["unitsPerBb"]
            or abs(aggregate.sum() + house) > 1e-5 or cash["conservation_residual_bb"] > 1e-8):
        raise ValueError("cash own CFVs disagree with independently walked house ledger")
    gains = np.asarray(cash["unilateral_gain_bb"], dtype=float)
    if (gains.shape != (2,) or not np.isfinite(gains).all() or (gains < 0).any()
            or not np.isclose(gains.sum(), cash["nash_conv_bb_per_hand"], atol=1e-8)
            or not np.isclose(gains.sum(), metrics["exact_abstract_exploitability_bb_per_hand"], atol=1e-8)
            or metrics["maximum_probability_sum_error"] > 1e-6):
        raise ValueError("cash response gains or probability diagnostics are inconsistent")


def build_dataset(paths: list[Path]) -> dict:
    labels, captures = [], []
    game = None
    for path in paths:
        payload = path.read_bytes()
        label = json.loads(payload)
        validate_label(label)
        current_game = label["input"]["game"]
        if game is not None and current_game != game:
            raise ValueError("cash value corpus cannot mix rules, depth or abstraction")
        game = current_game
        captures.append(hashlib.sha256(payload).hexdigest())
        labels.append(label)
    if not labels:
        raise ValueError("cash corpus is empty")
    result = {"schema": SCHEMA, "game": game, "rules_sha256": rules_digest(game["cash_rules"]),
              "payoff_contract": PAYOFF_CONTRACT, "validation": {"status": "research_only"},
              "capture_sha256": captures, "label_canonical_sha256": [identity_hash(l) for l in labels], "labels": labels}
    validate_dataset(result)
    return result


def validate_dataset(source: dict) -> None:
    if source.get("schema") != SCHEMA or source.get("payoff_contract") != PAYOFF_CONTRACT or source.get("validation", {}).get("status") != "research_only":
        raise ValueError("incompatible cash continuation dataset")
    if source.get("rules_sha256") != rules_digest(source["game"]["cash_rules"]):
        raise ValueError("cash dataset rules digest mismatch")
    if not source.get("labels") or len(source["labels"]) != len(source.get("capture_sha256", [])):
        raise ValueError("incomplete cash source captures")
    if source.get("label_canonical_sha256") != [identity_hash(l) for l in source["labels"]]:
        raise ValueError("cash labels changed after capture")
    for label, digest in zip(source["labels"], source["capture_sha256"]):
        validate_label(label)
        if label["input"]["game"] != source["game"] or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("mixed cash target provenance")
    provenance = source.get("reach_provenance")
    if provenance is not None:
        hashes = [provenance.get("policy_sha256"),provenance.get("root_corpus_sha256"),*provenance.get("root_sha256",[])]
        if (provenance.get("schema") != "hu-cash-public-reach-lineage-v1"
                or provenance.get("kind") != "authentic-frozen-average-turn-roots"
                or provenance.get("rules_sha256") != source["rules_sha256"]
                or len(provenance.get("root_sha256",[])) != len(source["labels"])
                or len(set(provenance["root_sha256"])) != len(source["labels"])
                or any(not isinstance(h,str) or re.fullmatch(r"[a-f0-9]{64}",h) is None for h in hashes)
                or provenance.get("label_input_sha256") != [identity_hash(label["input"]) for label in source["labels"]]):
            raise ValueError("authentic cash public-reach lineage differs from captured label inputs")
    leaves = source.get("flop_leaf_provenance")
    if leaves is not None:
        hashes = [leaves.get(key) for key in ("source_policy_sha256", "root_corpus_sha256", "flop_pair_report_sha256",
                  "native_binary_sha256", "source_flop_value_network_sha256", "split_reference_sha256", "excluded_roots_sha256")]
        if (provenance is not None or leaves.get("schema") != "hu-cash-frozen-flop-leaf-lineage-v1"
                or leaves.get("kind") != "forced-frozen-flop-average-leaves"
                or leaves.get("rules_sha256") != source["rules_sha256"] or leaves.get("split_seed") != 937
                or len(leaves.get("rows", [])) != len(source["labels"])
                or any(not isinstance(h, str) or re.fullmatch(r"[a-f0-9]{64}", h) is None for h in hashes)):
            raise ValueError("invalid frozen-flop leaf lineage")
        for row, label in zip(leaves["rows"], source["labels"]):
            root = row["root"]
            if (root["root_sha256"] != row["root_sha256"]
                    or root_fingerprint(leaves["source_policy_sha256"], root) != row["root_sha256"]
                    or root["solve_input"]["game"] != source["game"]
                    or row["label_input_sha256"] != identity_hash(label["input"])
                    or not isinstance(row.get("flop_solution_sha256"), str)
                    or re.fullmatch(r"[a-f0-9]{64}", row["flop_solution_sha256"]) is None):
                raise ValueError("frozen-flop leaf identities differ from captured inputs")
            replayed = leaf_input(root, dict(strategies=row["branch_policy_rows"]), row["public_turn_proposal"],
                                  row["branch_labels"], label["input"]["iterations"])
            if replayed != label["input"]:
                raise ValueError("frozen-flop leaf labels differ from public likelihood replay")
    inputs = [identity_hash(label["input"]) for label in source["labels"]]
    if len(set(inputs)) != len(inputs):
        raise ValueError("duplicate cash inputs are not independent data")
    merged = source.get("source_datasets")
    if merged is not None:
        version = merged.get("schema")
        if (provenance is not None or leaves is not None or version not in ("hu-cash-value-source-prefix-merge-v1","hu-cash-value-source-prefix-merge-v2")
                or not isinstance(merged.get("sources"),list)
                or not 2 <= len(merged["sources"]) <= MAX_FLAT_SOURCES
                or (version == "hu-cash-value-source-prefix-merge-v1" and len(merged["sources"]) != 2)):
            raise ValueError("invalid cash continuation merge lineage")
        labels, captures = [], []
        for entry in merged["sources"]:
            original = entry["dataset"]
            if original.get("source_datasets") is not None:
                raise ValueError("nested cash merges are not supported by this bounded pilot")
            validate_dataset(original)
            indices = entry["selected_rows"]
            if (original["game"] != source["game"] or entry["canonical_sha256"] != identity_hash(original)
                    or not isinstance(indices,list) or not indices
                    or any(type(i) is not int or not 0 <= i < len(original["labels"]) for i in indices)
                    or indices != sorted(set(indices))):
                raise ValueError("cash continuation parent identity or selected rows changed")
            labels.extend(original["labels"][i] for i in indices)
            captures.extend(original["capture_sha256"][i] for i in indices)
        if source["labels"] != labels or source["capture_sha256"] != captures:
            raise ValueError("cash merged labels differ from their captured parent rows")
        if version == "hu-cash-value-source-prefix-merge-v2":
            first = merged["sources"][0]
            pinned = first["dataset"]
            seed = merged.get("split_seed")
            if (type(seed) is not int or not 0 <= seed < 2**32
                    or merged.get("split_reference_canonical_sha256") != identity_hash(pinned)
                    or first["selected_rows"] != list(range(len(pinned["labels"])) )):
                raise ValueError("cash merge differs from its pinned split reference")
            split_source = {"game":pinned["game"],"targets":[{"board":l["input"]["state"]["board"]} for l in pinned["labels"]]}
            _,tuning,holdout = family_split(split_source,seed,.2,.2)
            forbidden = {board_family(pinned["labels"][i]["input"]["state"]["board"]) for i in np.concatenate((tuning,holdout))}
            if any(board_family(l["input"]["state"]["board"]) in forbidden for l in labels[len(pinned["labels"]):]):
                raise ValueError("cash merge added a pinned tuning/holdout family to training")


def extend_training_corpus(reference: dict, addition: dict, split_seed: int, split_reference: dict | None = None) -> dict:
    """Add contexts without moving or leaking a pinned tuning/holdout family.

    Parents are retained in this bounded research corpus so lineage can be
    rechecked independently, including authentic root hashes and source mixes.
    This deliberately does not pretend mixed corpora came from one root file.
    """
    validate_dataset(reference); validate_dataset(addition)
    if reference["game"] != addition["game"] or addition.get("source_datasets") is not None:
        raise ValueError("cash extension requires an unmixed addition in the same frozen game")
    merged = reference.get("source_datasets")
    if merged is not None and split_reference is None:
        raise ValueError("extending a merged corpus requires an explicit original split reference")
    pinned = reference if split_reference is None else split_reference
    validate_dataset(pinned)
    if (pinned.get("source_datasets") is not None or pinned["game"] != reference["game"]
            or reference["labels"][:len(pinned["labels"])] != pinned["labels"]
            or reference["capture_sha256"][:len(pinned["labels"])] != pinned["capture_sha256"]):
        raise ValueError("cash extension changed its pinned original target prefix")
    if merged is not None:
        if (len(merged["sources"]) >= MAX_FLAT_SOURCES or merged["sources"][0]["dataset"] != pinned):
            raise ValueError("cash extension requires the same original parent and at most five flat sources")
        if merged["schema"] == "hu-cash-value-source-prefix-merge-v2" and merged["split_seed"] != split_seed:
            raise ValueError("cash extension changed its pinned split seed")
    if type(split_seed) is not int or not 0 <= split_seed < 2**32:
        raise ValueError("invalid pinned cash extension split seed")
    split_source = {"game":pinned["game"],"targets":[{"board":l["input"]["state"]["board"]} for l in pinned["labels"]]}
    _,tuning,holdout = family_split(split_source,split_seed,.2,.2)
    forbidden = {board_family(pinned["labels"][i]["input"]["state"]["board"]) for i in np.concatenate((tuning,holdout))}
    known = {identity_hash(label["input"]) for label in reference["labels"]}
    rows = [i for i,label in enumerate(addition["labels"])
            if board_family(label["input"]["state"]["board"]) not in forbidden and identity_hash(label["input"]) not in known]
    if not rows: raise ValueError("cash extension supplies no new unheldout training contexts")
    source = copy.deepcopy(reference)
    source.pop("reach_provenance",None)
    source.pop("flop_leaf_provenance",None)
    source["labels"].extend(copy.deepcopy(addition["labels"][i]) for i in rows)
    source["capture_sha256"].extend(addition["capture_sha256"][i] for i in rows)
    source["label_canonical_sha256"] = [identity_hash(label) for label in source["labels"]]
    parents = (copy.deepcopy(merged["sources"]) if merged is not None else [
        {"canonical_sha256":identity_hash(reference),"selected_rows":list(range(len(reference["labels"]))),"dataset":copy.deepcopy(reference)}])
    parents.append({"canonical_sha256":identity_hash(addition),"selected_rows":rows,"dataset":copy.deepcopy(addition)})
    source["source_datasets"] = {"schema":"hu-cash-value-source-prefix-merge-v2" if split_reference is not None else "hu-cash-value-source-prefix-merge-v1","sources":parents}
    if split_reference is not None:
        source["source_datasets"].update(split_seed=split_seed,split_reference_canonical_sha256=identity_hash(pinned))
    validate_dataset(source)
    return source


def main():
    parser = argparse.ArgumentParser(description="Merge rule-pinned cash labels without relabeling games or targets")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--label", type=Path, nargs="+", help="individual label files or pilot directories")
    inputs.add_argument("--extend-reference",type=Path,help="Pinned corpus prefix whose tuning/holdout must remain unchanged")
    parser.add_argument("--addition",type=Path,help="Additional independently captured corpus, training-only after family filtering")
    parser.add_argument("--split-seed",type=int,default=937)
    parser.add_argument("--split-reference",type=Path,help="Original unmixed split corpus; required when extending an existing merge")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.extend_reference) != bool(args.addition):
        raise ValueError("cash corpus extension requires both reference and addition")
    if args.split_reference and not args.extend_reference:
        raise ValueError("a pinned split reference requires a corpus extension")
    paths = []
    for path in args.label or []:
        if path.is_dir():
            matches = sorted(p for p in path.iterdir() if re.fullmatch(r"label-[a-f0-9]{64}\.json",p.name))
            if not matches:
                raise ValueError(f"cash pilot directory has no immutable label captures: {path}")
            paths.extend(matches)
        else:
            paths.append(path)
    dataset = (extend_training_corpus(json.loads(args.extend_reference.read_bytes()),json.loads(args.addition.read_bytes()),args.split_seed,
                                     json.loads(args.split_reference.read_bytes()) if args.split_reference else None)
               if args.extend_reference else build_dataset(paths))
    # Validate every source before creating output. Training/tuning/holdout
    # splits group whole flop families across pots, never individual rows.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, separators=(",", ":")) + "\n")
    print(json.dumps({"schema": SCHEMA, "status": "research_only", "labels": len(dataset["labels"]),
                      "rules_sha256": dataset["rules_sha256"], "output": str(args.output)}))


if __name__ == "__main__": main()
