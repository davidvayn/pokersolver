#!/usr/bin/env python3
"""Reach-weighted stability comparison for paired tabular preflop policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("policy_a", type=Path)
    parser.add_argument("policy_b", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def compare(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    first_entries = {entry["key"]: entry for entry in first["strategies"]}
    second_entries = {entry["key"]: entry for entry in second["strategies"]}
    union = set(first_entries) | set(second_entries)
    common = sorted(set(first_entries) & set(second_entries))
    if not common:
        raise ValueError("paired tabular policies share no information sets")
    weights: list[float] = []
    maes: list[float] = []
    agreements: list[float] = []
    tie_aware: list[float] = []
    action_groups: list[tuple[list[str], np.ndarray, np.ndarray]] = []
    for key in common:
        a = first_entries[key]
        b = second_entries[key]
        if a["action_labels"] != b["action_labels"]:
            raise ValueError("paired information set has incompatible legal actions")
        first_probabilities = np.asarray(a["probabilities"], dtype=np.float64)
        second_probabilities = np.asarray(b["probabilities"], dtype=np.float64)
        weight = max(
            1.0,
            min(
                a.get("average_reach_weight", a["average_visits"]),
                b.get("average_reach_weight", b["average_visits"]),
            ),
        )
        first_primary = int(np.argmax(first_probabilities))
        second_primary = int(np.argmax(second_probabilities))
        weights.append(float(weight))
        maes.append(float(np.mean(np.abs(first_probabilities - second_probabilities))))
        agreements.append(float(first_primary == second_primary))
        tie_aware.append(
            float(
                first_primary == second_primary
                or first_probabilities[second_primary] >= np.max(first_probabilities) - 0.01
                or second_probabilities[first_primary] >= np.max(second_probabilities) - 0.01
            )
        )
        action_groups.append((a["action_labels"], first_probabilities, second_probabilities))
    normalized = np.asarray(weights) / np.sum(weights)
    aggregate_first: dict[str, float] = {}
    aggregate_second: dict[str, float] = {}
    for reach, (labels, first_probabilities, second_probabilities) in zip(
        normalized, action_groups
    ):
        for label, first_probability, second_probability in zip(
            labels, first_probabilities, second_probabilities
        ):
            aggregate_first[label] = aggregate_first.get(label, 0.0) + float(
                reach * first_probability
            )
            aggregate_second[label] = aggregate_second.get(label, 0.0) + float(
                reach * second_probability
            )
    action_labels = sorted(set(aggregate_first) | set(aggregate_second))
    aggregate_deltas = {
        label: abs(aggregate_first.get(label, 0.0) - aggregate_second.get(label, 0.0))
        for label in action_labels
    }
    return {
        "schema": "hu-tabular-preflop-cross-seed-v1",
        "seeds": [first["seed"], second["seed"]],
        "iterations": [first["iterations"], second["iterations"]],
        "informationSets": [len(first_entries), len(second_entries)],
        "commonInformationSets": len(common),
        "unionInformationSets": len(union),
        "lookupIntersectionCoverage": len(common) / len(union),
        "reachWeightedActionFrequencyMae": float(
            np.sum(normalized * np.asarray(maes))
        ),
        "reachWeightedPrimaryAgreement": float(
            np.sum(normalized * np.asarray(agreements))
        ),
        "reachWeightedTieAwarePrimaryAgreementAt0_01": float(
            np.sum(normalized * np.asarray(tie_aware))
        ),
        "maximumAggregateActionDelta": max(aggregate_deltas.values(), default=0.0),
        "aggregateActionDeltas": aggregate_deltas,
    }


def main() -> None:
    args = parse_args()
    first = json.loads(args.policy_a.read_text(encoding="utf-8"))
    second = json.loads(args.policy_b.read_text(encoding="utf-8"))
    result = compare(first, second)
    output = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
