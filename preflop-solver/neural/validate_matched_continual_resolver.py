#!/usr/bin/env python3
"""Validate the frozen cross-seed continual-resolver evidence fail-closed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import freeze_resolver_reach_release as release_freeze
import run_matched_continual_resolver as matched
import run_resolver_reach_release as release_runner


SCHEMA = "hu-matched-continual-resolver-validation-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_freeze", type=Path)
    parser.add_argument("--value-validation", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate(
    release_path: Path,
    value_validation_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    release = release_runner.validate_release_freeze(release_path, repository_root)
    matched.validate_release_contract(release, repository_root)
    value_report = matched.validate_value_report(
        value_validation_path, release_path, repository_root
    )
    plan = matched.build_plan(
        release,
        value_report,
        release_path,
        value_validation_path,
        repository_root,
    )
    evidence = [
        matched.inspect_solution(
            job,
            plan["controls"],
            matched.resolved(repository_root, job["output"]),
        )
        for job in plan["jobs"]
    ]
    return summarize(plan, evidence, release_path, value_validation_path)


def summarize(
    plan: dict[str, Any],
    evidence: list[dict[str, Any]],
    release_path: Path,
    value_validation_path: Path,
) -> dict[str, Any]:
    if not evidence:
        raise ValueError("matched continual-resolver validation has no evidence")
    roots = {entry["board"] for entry in evidence}
    directions = {
        board: {
            (entry["strategySeed"], entry["evaluationSeed"])
            for entry in evidence
            if entry["board"] == board
        }
        for board in roots
    }
    expected_directions = {
        (plan["models"][0]["seed"], plan["models"][1]["seed"]),
        (plan["models"][1]["seed"], plan["models"][0]["seed"]),
    }
    gates = {
        "completeReservedRootCoverage": len(roots) == plan["rootCount"]
        and len(evidence) == plan["rootCount"] * 2,
        "bothCrossSeedDirectionsPerRoot": all(
            value == expected_directions for value in directions.values()
        ),
        "exactArtifactProvenance": all(
            entry["gates"]["artifactProvenance"] for entry in evidence
        ),
        "acceptedSolverValidation": all(
            entry["gates"]["acceptedSolverValidation"] for entry in evidence
        ),
        "validProbabilitySums": all(
            entry["gates"]["probabilitySums"] for entry in evidence
        ),
        "maximumLocalExploitability": all(
            entry["gates"]["maximumLocalExploitability"] for entry in evidence
        ),
        "positiveResolverImprovement": all(
            entry["gates"]["positiveResolverImprovement"] for entry in evidence
        ),
        "zeroSumResidual": all(
            entry["gates"]["zeroSumResidual"] for entry in evidence
        ),
    }
    accepted = all(gates.values())
    return {
        "schema": SCHEMA,
        "status": (
            "accepted-awaiting-preflop-and-full-game-gates"
            if accepted
            else "rejected"
        ),
        "activationAllowed": False,
        "releaseFreeze": {
            "path": str(release_path),
            "sha256": release_freeze.sha256_file(release_path),
        },
        "valueReleaseValidation": {
            "path": str(value_validation_path),
            "sha256": release_freeze.sha256_file(value_validation_path),
        },
        "controls": plan["controls"],
        "models": plan["models"],
        "rootCount": plan["rootCount"],
        "solutionCount": len(evidence),
        "maximumDepthLimitedExploitabilityBbPerHand": max(
            entry["depthLimitedExploitabilityBbPerHand"] for entry in evidence
        ),
        "minimumResolverRelativeExploitabilityImprovement": min(
            entry["resolverRelativeExploitabilityImprovement"] for entry in evidence
        ),
        "maximumZeroSumResidualBb": max(
            entry["zeroSumResidualBb"] for entry in evidence
        ),
        "maximumProbabilitySumError": max(
            entry["maximumProbabilitySumError"] for entry in evidence
        ),
        "gates": gates,
        "evidence": evidence,
        "remainingRequiredGates": [
            "preflop continuation-cache and tabular DCFR regeneration",
            "full-game lookup coverage and probability validation",
            "reach-weighted action-EV standard-error coverage",
            "cross-seed action-frequency stability",
            "learned-response red-team evaluation",
            "independent one-sided 99% full-game exploitability upper bound at or below 0.10bb/hand",
            "hosted-policy storage projection at or below 20GB",
        ],
    }


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root or args.release_freeze.resolve().parent.parent
    result = validate(
        args.release_freeze, args.value_validation, repository_root
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(args.output)
    print(encoded, end="")


if __name__ == "__main__":
    main()
