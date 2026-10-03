"""Append stronger native labels to training only, preserving frozen evaluation.

The collection-level ``turn_iterations`` remains the retained corpus's 64 for
backward compatibility. Every target is validated against its own capture's
explicit budget; no 256/1024 label is ever presented as a native-64 result.
"""
from __future__ import annotations

import copy
from pathlib import Path

import native_value_dataset as native
from run_native_value_pilot import read_capture
from run_native_value_preflight import sha256


def append_training_labels(retained: dict, split_reference: dict,
                           captures: list[Path], maximum_states: int = 640) -> dict:
    native.validate_dataset(retained)
    native.validate_dataset(split_reference)
    if (not retained.get("source_captures") or not captures
            or retained["game"] != split_reference["game"]
            or retained["turn_iterations"] != split_reference["turn_iterations"]):
        raise ValueError("retained and split-reference native corpora do not match")
    train, tuning, holdout = native.family_split(
        retained, 10601, .25, .25, reference=split_reference, refresh_training=True)
    allowed = {native.board_family(retained["targets"][int(i)]["board"]) for i in train}
    forbidden = {native.board_family(retained["targets"][int(i)]["board"])
                 for i in (*tuning, *holdout)}
    if allowed & forbidden:
        raise ValueError("retained train and evaluation families overlap")
    manifests = copy.deepcopy(retained["source_captures"])
    targets = copy.deepcopy(retained["targets"])
    observed = retained["observed_queries"]
    added = 0
    seen = set()
    for path in captures:
        digest = sha256(path)
        if digest in seen:
            raise ValueError("duplicate stronger-label capture")
        seen.add(digest)
        source = read_capture(path)
        if (source["game"] != retained["game"]
                or source.get("capture_selection") !=
                "stratified_learned_search_and_final_average_beliefs_with_native_labels"
                or source.get("proposal_policy_kind") != "frozen_learned_leaf_search_only"
                or source.get("proposal_turn_iterations") != 64
                or source.get("turn_iterations") not in (256, 1024)
                or source.get("native_label_queries") != len(source["targets"])):
            raise ValueError("capture is not an independently labeled stronger reference")
        distributions = {row.get("state_distribution") for row in source["targets"]}
        if distributions != {"learned_flop_search_early_belief_native_label",
                             "learned_flop_search_middle_belief_native_label",
                             "learned_flop_search_late_belief_native_label",
                             "learned_flop_final_average_belief_native_label"}:
            raise ValueError("stronger-label capture omits a search distribution")
        families = {native.board_family(row["board"]) for row in source["targets"]}
        if not families or families - allowed or families & forbidden:
            raise ValueError("stronger labels touch a tuning, holdout, or unknown family")
        index = len(manifests)
        manifests.append(dict(capture_sha256=digest,
                              input_sha256=source["source_public_input_sha256"],
                              policy_sha256=source["source_policy_sha256"],
                              states=len(source["targets"]),
                              flop_iterations=source["flop_iterations"],
                              turn_iterations=source["turn_iterations"],
                              seed=source["seed"],
                              proposal_model_sha256=source["proposal_model_sha256"],
                              sampling_seed=source["sampling_seed"],
                              native_label_queries=source["native_label_queries"]))
        targets.extend({**row, "source_capture_index": index} for row in source["targets"])
        observed += source["observed_queries"]
        added += len(source["targets"])
    if not 256 <= len(targets) <= maximum_states or observed < len(targets):
        raise ValueError("stronger-label collection exceeds its state budget")
    result = dict(schema=native.SCHEMA, game=retained["game"],
                  source_identity_semantics="ordered_capture_manifest_not_one_policy",
                  source_captures=manifests,
                  source_public_input_sha256=native.identity_hash([m["input_sha256"] for m in manifests]),
                  source_policy_sha256=native.identity_hash([m["policy_sha256"] for m in manifests]),
                  flop_iterations=max(m["flop_iterations"] for m in manifests),
                  turn_iterations=retained["turn_iterations"],
                  maximum_states=len(targets), observed_queries=observed,
                  capture_selection="retained_reference_with_training_only_stronger_native_search_labels",
                  validation=dict(status="research_only", reasons=[
                      "mixed reference budgets are capture-specific; evaluation targets and families remain frozen"]),
                  targets=targets)
    native.validate_dataset(result)
    new_train, new_tuning, new_holdout = native.family_split(
        result, 10601, .25, .25, reference=split_reference, refresh_training=True)
    if (list(map(int, new_tuning)) != list(map(int, tuning))
            or list(map(int, new_holdout)) != list(map(int, holdout))
            or len(new_train) != len(train) + added):
        raise ValueError("stronger labels changed the frozen evaluation split")
    return result
