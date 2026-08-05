#!/usr/bin/env python3
"""Compose fail-closed gates for the exact-belief shared-combo v30 pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--turn-report", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--flop-range", type=Path)
    parser.add_argument("--flop-no-range", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def compose(
    targets: dict[str, Any],
    turn: dict[str, Any],
    parity: dict[str, Any],
    flop_range: dict[str, Any] | None = None,
    flop_no_range: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gates: dict[str, dict[str, Any]] = {}

    def gate(name: str, passed: bool, measured: Any, threshold: str) -> None:
        gates[name] = {
            "passed": bool(passed),
            "measured": measured,
            "threshold": threshold,
        }

    target_rows = targets["targets"]
    maximum_river = max(
        row["maximum_river_exploitability_bb_per_hand"] for row in target_rows
    )
    maximum_zero_sum = max(row["zero_sum_residual_bb"] for row in target_rows)
    maximum_range_tv = max(
        row.get("range_maximum_total_variation", float("inf")) for row in target_rows
    )
    minimum_particles = min(row.get("range_particles", 0) for row in target_rows)
    minimum_replicates = min(row.get("range_replicates", 0) for row in target_rows)
    exact_beliefs = all(
        row.get("belief_method", "").startswith("exact_per-player_reach_factors")
        for row in target_rows
    )
    distinct_boards = len({tuple(row["board"]) for row in target_rows})
    gate("targetCorpusAccepted", targets["validation"]["status"] == "accepted",
         targets["validation"], "accepted")
    policy_sha = targets.get("source_policy_sha256")
    gate("pinnedSourcePolicy", isinstance(policy_sha, str) and len(policy_sha) == 64,
         policy_sha, "64-character SHA-256")
    gate("authenticStateCount", len(target_rows) >= 64, len(target_rows), ">=64")
    gate("distinctTurnBoards", distinct_boards * 100 >= len(target_rows) * 95,
         distinct_boards, ">=95% of target states")
    gate("exactReachFactors", exact_beliefs,
         sorted({str(row.get("belief_method")) for row in target_rows}),
         "exact per-player reach factors")
    gate("beliefReplicateParticles", minimum_particles >= 4096, minimum_particles,
         ">=4096 per replicate")
    gate("beliefReplicateCount", minimum_replicates >= 2, minimum_replicates,
         ">=2 independent rotations")
    gate("beliefReplicateAgreement", maximum_range_tv <= 0.15, maximum_range_tv,
         "maximum exact-combo total variation <=0.15")
    gate("riverExploitability", maximum_river <= 0.05, maximum_river,
         "<=0.05bb/hand in every solved river abstraction")
    gate("targetZeroSum", maximum_zero_sum <= 1e-7, maximum_zero_sum, "<=1e-7bb")
    gate("turnNetworkAccepted", turn["validation"]["status"] == "accepted",
         turn["validation"], "accepted")
    gate("turnHoldoutRmse", turn["meanRangeRmseBb"] <= 0.25,
         turn["meanRangeRmseBb"], "<=0.25bb research gate")
    gate("turnRangeAblation", turn["rangeRelativeImprovement"] >= 0.02,
         turn["rangeRelativeImprovement"], ">=2% relative RMSE improvement")
    gate("turnCrossSeed", turn["crossSeedPredictionCorrelation"]["range"] >= 0.95,
         turn["crossSeedPredictionCorrelation"]["range"], ">=0.95")
    gate("suitEquivariance", turn.get("structurallySuitEquivariant") is True,
         turn.get("structurallySuitEquivariant"), "true by feature construction")
    gate("zeroSumProjection", turn.get("structurallyZeroSumProjected") is True,
         turn.get("structurallyZeroSumProjected"), "true")
    gate("pythonRustParity", parity["validation"]["status"] == "accepted"
         and parity["maximumAbsoluteErrorBb"] <= 1e-4,
         parity["maximumAbsoluteErrorBb"], "<=0.0001bb")

    if flop_range is not None and flop_no_range is not None:
        range_exploit = flop_range["metrics"]["depth_limited_exploitability_bb_per_hand"]
        no_range_exploit = flop_no_range["metrics"]["depth_limited_exploitability_bb_per_hand"]
        gate("flopDepthLimitedExploitability", range_exploit <= 0.05, range_exploit,
             "<=0.05bb/hand against the same learned leaf game")
        gate("flopRangeAblation", range_exploit < no_range_exploit,
             {"range": range_exploit, "noRange": no_range_exploit}, "range < no-range")
        gate("fullFlopActionAbstraction", flop_range["validation"]["status"] == "accepted",
             flop_range["validation"], "accepted including exact all-in branches")
    else:
        gate("flopDepthLimitedExploitability", False, None, "measured paired resolver result")
        gate("flopRangeAblation", False, None, "range < no-range")
        gate("fullFlopActionAbstraction", False, None,
             "accepted including exact all-in branches")

    # A depth-limited learned leaf game is not a full-game exploitability
    # certificate. Keep activation blocked until an independent upper bound is
    # measured against the routed full-hand policy.
    gate("fullGameExploitabilityUpperBound", False, None,
         "independent one-sided 99% upper bound <=0.10bb/hand")
    failed = [name for name, result in gates.items() if not result["passed"]]
    return {
        "schema": "hu-v30-exact-belief-shared-combo-release-gates-v1",
        "status": "accepted" if not failed else "rejected",
        "activationAllowed": not failed,
        "failedGates": failed,
        "gates": gates,
        "interpretation": (
            "Fail-closed research composition. Exact beliefs, parity, and a learned leaf "
            "ablation cannot replace full-action safe resolving or a full-game upper bound."
        ),
    }


def main() -> None:
    args = parse_args()
    report = compose(
        load(args.targets),
        load(args.turn_report),
        load(args.parity),
        load(args.flop_range) if args.flop_range else None,
        load(args.flop_no_range) if args.flop_no_range else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
