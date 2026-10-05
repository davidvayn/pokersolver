"""Scores-blind TRAIN selection and coherent extra calibration, not evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from action_contrast_dataset import AffineGroup
import native_value_dataset as native
from run_native_value_preflight import sha256


def select_families(roots, excluded):
    """First two high-card rainbow and first two-tone five-leaf three-bet roots."""
    seen = set(excluded)
    selected, counts = [], {3: 0, 2: 0}
    for index, root in enumerate(roots):
        state = root["public"]
        board = state["board"]
        family = native.board_family(board)
        suits = len({c % 4 for c in board})
        raises = sum(h.startswith("Preflop:") and ":raise_to_" in h for h in state["public_history"])
        if (family in seen or len({c // 4 for c in board}) != 3 or max(c // 4 for c in board) < 10
                or suits not in counts or counts[suits] >= (2 if suits == 3 else 1)
                or root["turn_leaf_count"] != 5 or raises < 2):
            continue
        counts[suits] += 1
        seen.add(family)
        selected.append(index)
    if counts != {3: 2, 2: 1}:
        raise ValueError("insufficient disjoint high-card three-bet TRAIN strata; no score-based fallback")
    return selected


def extra_calibration(groups, ordered, capture, candidate_sha):
    """Search queries train values only; never enter the frozen affine backup."""
    if (not ordered or capture.get("source_policy_sha256") != candidate_sha or capture.get("turn_iterations") != 64
            or capture.get("flop_iterations") != 128 or len(capture["targets"]) != 16
            or capture.get("capture_selection") != "stratified_learned_search_and_final_average_beliefs_with_native_labels"):
        raise ValueError("extra calibration has different proposer or search/native budget")
    family = native.board_family(ordered[0]["board"])
    if any(native.board_family(t["board"]) != family for t in capture["targets"]):
        raise ValueError("search calibration crosses board families")
    required = {"learned_flop_search_early_belief_native_label",
                "learned_flop_search_middle_belief_native_label",
                "learned_flop_search_late_belief_native_label",
                "learned_flop_final_average_belief_native_label"}
    if {t.get("state_distribution") for t in capture["targets"]} != required:
        raise ValueError("search calibration omits a declared band")
    count = len(capture["targets"])
    augmented = [AffineGroup(g.history, g.actor, g.actions, g.terminal,
        np.pad(g.coefficients, ((0, 0), (0, count), (0, 0))), g.target, g.weights, g.support)
        for g in groups]
    return augmented, ordered + capture["targets"]


def extension_families(receipt, corpus_sha, split_sha, existing, forbidden):
    """Allow only a pinned scores-blind registry; held-out/old families stay closed."""
    path = Path(receipt["path"])
    if sha256(path) != receipt["sha256"]:
        raise ValueError("TRAIN extension registry changed")
    plan = json.loads(path.read_text())
    if (plan.get("schema") != "new-training-coverage-protocol-v1" or plan.get("status") != "complete"
            or plan.get("corpusSha256") != corpus_sha or plan.get("splitReferenceSha256") != split_sha
            or plan.get("releaseAccepted") is not False or len(plan["families"]) != 3):
        raise ValueError("TRAIN registry does not match the frozen split")
    added = {tuple(f["family"]) for f in plan["families"]}
    excluded = {tuple(f) for f in plan["excludedFamilies"]}
    if len(added) != 3 or added & (existing | forbidden | excluded):
        raise ValueError("TRAIN extension leaks a consumed or held-out family")
    if not (existing | forbidden) <= excluded:
        raise ValueError("TRAIN registry omits existing split exclusions")
    for p, digest in plan["pinnedInputs"].items():
        if sha256(Path(p)) != digest:
            raise ValueError("TRAIN registry input changed")
    if sorted(f["root"] for f in plan["families"]) != [100, 101, 102]:
        raise ValueError("TRAIN registry repeats or omits a root id")
    for family in plan["families"]:
        if family.get("role") != "TRAIN" or family["root"] not in (100, 101, 102):
            raise ValueError("unexpected TRAIN extension role/id")
        root_path = Path(family["rootPath"])
        if (sha256(root_path) != family["rootSha256"]
                or plan["pinnedInputs"].get(str(root_path)) != family["rootSha256"]):
            raise ValueError("TRAIN registry root identity changed")
        root = json.loads(root_path.read_text())
        if native.board_family(root["public"]["board"]) != tuple(family["family"]):
            raise ValueError("TRAIN registry root family differs")
    return added
