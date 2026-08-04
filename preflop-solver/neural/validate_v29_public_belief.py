#!/usr/bin/env python3
"""Fail-closed gate composition for the v29 public-belief research sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--turn-report", type=Path, required=True)
    parser.add_argument("--flop-range", type=Path, required=True)
    parser.add_argument("--flop-no-range", type=Path, required=True)
    parser.add_argument("--preflop-evaluation-a", type=Path, required=True)
    parser.add_argument("--preflop-evaluation-b", type=Path, required=True)
    parser.add_argument("--preflop-cross-seed", type=Path, required=True)
    parser.add_argument("--full-game-lbr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    targets = load(args.targets)
    turn = load(args.turn_report)
    flop_range = load(args.flop_range)
    flop_no_range = load(args.flop_no_range)
    evaluations = [load(args.preflop_evaluation_a), load(args.preflop_evaluation_b)]
    cross = load(args.preflop_cross_seed)
    lbr = load(args.full_game_lbr)
    gates: dict[str, dict[str, Any]] = {}

    def gate(name: str, passed: bool, measured: Any, threshold: str) -> None:
        gates[name] = {
            "passed": bool(passed),
            "measured": measured,
            "threshold": threshold,
        }

    maximum_river = max(
        target["maximum_river_exploitability_bb_per_hand"]
        for target in targets["targets"]
    )
    gate("targetCorpusAccepted", targets["validation"]["status"] == "accepted",
         targets["validation"], "accepted")
    gate("riverExploitability", maximum_river <= 0.05, maximum_river, "<=0.05bb/hand")
    gate("turnNetworkAccepted", turn["validation"]["status"] == "accepted",
         turn["validation"], "accepted")
    gate("turnHoldoutRmse", turn["meanRangeRmseBb"] <= 0.25,
         turn["meanRangeRmseBb"], "<=0.25bb research gate")
    gate("turnRangeAblation", turn["rangeRelativeImprovement"] >= 0.02,
         turn["rangeRelativeImprovement"], ">=2% relative RMSE improvement")
    gate("turnCrossSeed", turn["crossSeedPredictionCorrelation"]["range"] >= 0.95,
         turn["crossSeedPredictionCorrelation"]["range"], ">=0.95")
    gate("flopResolverAccepted", flop_range["validation"]["status"] == "accepted",
         flop_range["validation"], "accepted with full action abstraction")
    range_exploit = flop_range["metrics"]["depth_limited_exploitability_bb_per_hand"]
    no_range_exploit = flop_no_range["metrics"]["depth_limited_exploitability_bb_per_hand"]
    gate("flopDepthLimitedExploitability", range_exploit <= 0.05, range_exploit,
         "<=0.05bb/hand against the same leaf evaluator")
    gate("flopRangeAblation", range_exploit < no_range_exploit,
         {"range": range_exploit, "noRange": no_range_exploit}, "range < no-range")
    preflop_exploit = [value["exploitability_bb_per_hand"] for value in evaluations]
    gate("preflopIndependentEvaluation", max(preflop_exploit) <= 0.05,
         preflop_exploit, "both <=0.05bb/hand")
    gate("preflopCrossSeedMae", cross["reachWeightedActionFrequencyMae"] <= 0.05,
         cross["reachWeightedActionFrequencyMae"], "<=0.05")
    gate("preflopPrimaryAgreement", cross["reachWeightedPrimaryAgreement"] >= 0.85,
         cross["reachWeightedPrimaryAgreement"], ">=0.85")
    gate("preflopLookupCoverage", cross["lookupIntersectionCoverage"] >= 0.9999,
         cross["lookupIntersectionCoverage"], ">=0.9999")
    lbr_lower = lbr["approximate_exploitability_lower_confidence_bound_99_percent_bb_per_hand"]
    gate("fullGameLearnedResponseLeak", lbr_lower <= 0.05, lbr_lower,
         "99% approximate lower confidence bound <=0.05bb/hand")
    # A learned response is a lower bound. It can reject a candidate but can
    # never provide the upper-bound certificate required for activation.
    gate("fullGameExploitabilityUpperBound", False, None,
         "independent one-sided 99% upper bound <=0.10bb/hand")
    failed = [name for name, result in gates.items() if not result["passed"]]
    report = {
        "schema": "hu-v29-public-belief-release-gates-v1",
        "status": "accepted" if not failed else "rejected",
        "activationAllowed": not failed,
        "failedGates": failed,
        "gates": gates,
        "interpretation": (
            "Fail-closed research composition. River and depth-limited values are not a "
            "full-game equilibrium certificate; learned response is rejection evidence only."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
