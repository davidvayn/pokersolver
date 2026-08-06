#!/usr/bin/env python3
"""Validate every frozen V49 range-consistent response artifact fail-closed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import freeze_range_response_release as response_freeze
import freeze_resolver_reach_release as release_freeze
import run_range_response_release as runner


SCHEMA = "hu-range-response-release-validation-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("freeze", type=Path)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summarize(
    plan: dict[str, Any], evidence: list[dict[str, Any]], freeze_path: Path
) -> dict[str, Any]:
    if not evidence:
        raise ValueError("range-response validation has no evidence")
    expected_directions = {
        (plan["models"][0]["seed"], plan["models"][1]["seed"]),
        (plan["models"][1]["seed"], plan["models"][0]["seed"]),
    }
    boards = {entry["board"] for entry in evidence}
    directions = {
        board: {
            (entry["strategySeed"], entry["evaluationSeed"])
            for entry in evidence
            if entry["board"] == board
        }
        for board in boards
    }
    gates = {
        "completeFreshRootCoverage": len(boards) == plan["rootCount"]
        and len(evidence) == plan["rootCount"] * 2,
        "bothCrossSeedDirectionsPerRoot": all(
            value == expected_directions for value in directions.values()
        ),
        "exactArtifactProvenance": all(
            entry["response"]["gates"]["artifactProvenance"]
            for entry in evidence
        ),
        "exactFrozenStrategyLink": all(
            entry["response"]["gates"]["strategyArtifactLink"]
            for entry in evidence
        ),
        "validProbabilitySums": all(
            entry["convergence"]["accepted"] for entry in evidence
        ),
        "validGainArithmetic": all(
            entry["response"]["gates"]["gainArithmetic"] for entry in evidence
        ),
        "maximumRangeConsistentResponseGain": all(
            entry["response"]["gates"]["maximumResponseGain"]
            for entry in evidence
        ),
        "finalCheckpointConvergence": all(
            entry["response"]["gates"]["finalCheckpointIncrease"]
            for entry in evidence
        ),
        "zeroSumResidual": all(
            entry["response"]["gates"]["zeroSumResidual"]
            for entry in evidence
        ),
    }
    accepted = all(gates.values())
    return {
        "schema": SCHEMA,
        "status": (
            "accepted-as-learned-response-rejection-gate-awaiting-authentic-recheck-preflop-and-full-game-upper-bound"
            if accepted
            else "rejected"
        ),
        "activationAllowed": False,
        "interpretation": (
            "finite range-consistent learned-response search is rejection evidence and "
            "does not establish an exploitability upper bound"
        ),
        "freeze": {
            "path": str(freeze_path),
            "sha256": release_freeze.sha256_file(freeze_path),
        },
        "controls": plan["controls"],
        "declaredGates": plan["gates"],
        "rootCount": plan["rootCount"],
        "directionalEvidenceCount": len(evidence),
        "maximumObservedResponseGainBbPerHand": max(
            entry["response"]["maximumResponseGainBbPerHand"]
            for entry in evidence
        ),
        "maximumObservedFinalCheckpointIncreaseBbPerHand": max(
            entry["response"]["finalCheckpointIncreaseBbPerHand"]
            for entry in evidence
        ),
        "maximumObservedZeroSumResidualBb": max(
            entry["response"]["maximumZeroSumResidualBb"] for entry in evidence
        ),
        "maximumObservedProbabilitySumError": max(
            entry["convergence"]["maximumProbabilitySumError"]
            for entry in evidence
        ),
        "gates": gates,
        "evidence": evidence,
        "remainingRequiredGates": [
            "fresh authentic value recheck on successor seeds 15501 and 15502",
            "preflop continuation-cache and tabular DCFR regeneration",
            "full-game lookup coverage and probability validation",
            "reach-weighted action-EV standard-error coverage",
            "cross-seed action-frequency stability",
            "independent one-sided 99% full-game exploitability upper bound at or below 0.10bb/hand",
            "hosted-policy storage projection at or below 20GB",
        ],
    }


def validate(
    freeze_path: Path, repository_root: Path
) -> dict[str, Any]:
    freeze = response_freeze.validate_freeze(freeze_path, repository_root)
    plan = runner.build_plan(freeze, freeze_path)
    evidence = [
        runner.inspect_job(
            job,
            plan["controls"],
            plan["gates"],
            repository_root,
        )
        for job in plan["jobs"]
    ]
    return summarize(plan, evidence, freeze_path)


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root or args.freeze.resolve().parent.parent
    result = validate(args.freeze, repository_root)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(args.output)
    print(encoded, end="")


if __name__ == "__main__":
    main()
