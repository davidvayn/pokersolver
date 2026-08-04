#!/usr/bin/env python3
"""Fail-closed release gates for paired tabular preflop candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-a", type=Path, required=True)
    parser.add_argument("--policy-b", type=Path, required=True)
    parser.add_argument("--evaluation-a", type=Path, required=True)
    parser.add_argument("--evaluation-b", type=Path, required=True)
    parser.add_argument("--cross-seed", type=Path, required=True)
    parser.add_argument("--upper-bound", type=Path)
    parser.add_argument("--action-ev", type=Path)
    parser.add_argument("--projected-storage-bytes", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def probability_evidence(policy: dict[str, Any]) -> dict[str, Any]:
    strategies = policy.get("strategies")
    if not isinstance(strategies, list) or not strategies:
        return {"informationSets": 0, "valid": False}
    valid = True
    probabilities = 0
    for entry in strategies:
        values = entry.get("probabilities") if isinstance(entry, dict) else None
        if not isinstance(values, list) or not values:
            valid = False
            continue
        probabilities += len(values)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value < 0
            for value in values
        ):
            valid = False
        elif abs(sum(values) - 1.0) > 1e-9:
            valid = False
    return {
        "informationSets": len(strategies),
        "probabilities": probabilities,
        "valid": valid,
    }


def finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and value == value
        and abs(value) != float("inf")
    )


def evaluate_gates(
    policies: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    cross_seed: dict[str, Any],
    upper_bound: dict[str, Any] | None,
    action_ev: dict[str, Any] | None,
    projected_storage_bytes: int | None,
) -> dict[str, Any]:
    probability_checks = [probability_evidence(policy) for policy in policies]
    exploitabilities = [
        evaluation.get("exploitability_bb_per_hand") for evaluation in evaluations
    ]
    lookup_coverages = [evaluation.get("policy_lookup_coverage") for evaluation in evaluations]
    exploitability_evidence_valid = all(finite_number(value) for value in exploitabilities)
    lookup_evidence_valid = all(finite_number(value) for value in lookup_coverages)

    upper_values = None if upper_bound is None else upper_bound.get("upperBounds99BbPerHand")
    upper_evidence_valid = (
        isinstance(upper_values, list)
        and len(upper_values) == 2
        and all(finite_number(value) for value in upper_values)
    )
    action_coverage = (
        None if action_ev is None else action_ev.get("reachWeightedCoverageAt0_02Bb")
    )
    action_ev_evidence_valid = finite_number(action_coverage)
    storage_evidence_valid = (
        isinstance(projected_storage_bytes, int) and projected_storage_bytes >= 0
    )

    cross_mae = cross_seed.get("reachWeightedActionFrequencyMae")
    primary_agreement = cross_seed.get("reachWeightedPrimaryAgreement")
    aggregate_delta = cross_seed.get("maximumAggregateActionDelta")
    intersection_coverage = cross_seed.get("lookupIntersectionCoverage")
    cross_evidence_valid = all(
        finite_number(value)
        for value in [cross_mae, primary_agreement, aggregate_delta, intersection_coverage]
    )

    gates = {
        "exploitabilityEstimate": exploitability_evidence_valid
        and max(exploitabilities) <= 0.05,
        "exploitabilityUpperBound99": upper_evidence_valid and max(upper_values) <= 0.10,
        "crossSeedActionFrequencyMae": cross_evidence_valid and cross_mae <= 0.05,
        "crossSeedPrimaryAgreement": cross_evidence_valid and primary_agreement >= 0.85,
        "aggregateActionDelta": cross_evidence_valid and aggregate_delta <= 0.03,
        "lookupCoverage": lookup_evidence_valid
        and cross_evidence_valid
        and min([*lookup_coverages, intersection_coverage]) >= 0.9999,
        "probabilitySums": all(check["valid"] for check in probability_checks),
        "actionEvStandardErrorCoverage": action_ev_evidence_valid and action_coverage >= 0.95,
        "projectedStorage": storage_evidence_valid
        and projected_storage_bytes <= 20 * 1024**3,
    }
    return {
        "schema": "hu-v28-preflop-release-gates-v1",
        "status": "accepted" if all(gates.values()) else "rejected_not_activated",
        "evidence": {
            "exploitabilityBbPerHand": exploitabilities,
            "upperBounds99BbPerHand": upper_values,
            "lookupCoverage": lookup_coverages,
            "crossSeed": {
                "reachWeightedActionFrequencyMae": cross_mae,
                "primaryAgreement": primary_agreement,
                "maximumAggregateActionDelta": aggregate_delta,
                "lookupIntersectionCoverage": intersection_coverage,
            },
            "probabilities": probability_checks,
            "actionEvReachWeightedCoverageAt0_02Bb": action_coverage,
            "projectedStorageBytes": projected_storage_bytes,
        },
        "evidenceComplete": {
            "sampledExploitability": exploitability_evidence_valid,
            "exploitabilityUpperBound99": upper_evidence_valid,
            "crossSeed": cross_evidence_valid,
            "lookup": lookup_evidence_valid,
            "probabilities": all(check["valid"] for check in probability_checks),
            "actionEv": action_ev_evidence_valid,
            "storage": storage_evidence_valid,
        },
        "gates": gates,
        "allPassed": all(gates.values()),
        "interpretation": "fail-closed release decision; missing evidence is a failed gate, and sampled preflop exploitability is not a full-game equilibrium certificate",
    }


def main() -> None:
    args = parse_args()
    result = evaluate_gates(
        [read_json(args.policy_a), read_json(args.policy_b)],
        [read_json(args.evaluation_a), read_json(args.evaluation_b)],
        read_json(args.cross_seed),
        None if args.upper_bound is None else read_json(args.upper_bound),
        None if args.action_ev is None else read_json(args.action_ev),
        args.projected_storage_bytes,
    )
    output = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
