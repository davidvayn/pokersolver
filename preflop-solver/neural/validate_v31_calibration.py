#!/usr/bin/env python3
"""Compose the fail-closed v31 value-calibration sequence decision.

This composer deliberately separates research sequencing gates from release
gates.  A failed calibration pilot prevents the larger corpus and exact
low-SPR branch from being attempted; it does not silently convert an unrun
conditional step into evidence for the candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any


POT_BANDS = ("small", "medium", "large")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--pot-report", type=Path, required=True)
    parser.add_argument("--payoff-report", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--v31-resolver", type=Path, action="append", default=[])
    parser.add_argument("--v30-resolver", type=Path, action="append", default=[])
    parser.add_argument("--full-game-lbr", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def range_variants(report: dict[str, Any]) -> list[dict[str, Any]]:
    variants = report.get("variants", {}).get("range", [])
    if len(variants) != 2:
        raise ValueError("each normalization requires exactly two range seeds")
    return variants


def mean_tuning_rmse(report: dict[str, Any]) -> float:
    return mean(
        float(row["metrics"]["bestTuningRmseBb"])
        for row in range_variants(report)
    )


def selected_seed(report: dict[str, Any]) -> dict[str, Any]:
    return min(
        range_variants(report),
        key=lambda row: (float(row["metrics"]["bestTuningRmseBb"]), int(row["seed"])),
    )


def mean_band_rmse(report: dict[str, Any]) -> dict[str, float]:
    return {
        band: mean(
            float(row["metrics"]["potBandMetrics"][band]["weightedRmseBb"])
            for row in range_variants(report)
        )
        for band in POT_BANDS
    }


def resolver_exploitability(report: dict[str, Any]) -> float:
    return float(report["metrics"]["depth_limited_exploitability_bb_per_hand"])


def compose(
    baseline: dict[str, Any],
    pot: dict[str, Any],
    payoff: dict[str, Any],
    parity: dict[str, Any],
    v31_resolvers: list[dict[str, Any]] | None = None,
    v30_resolvers: list[dict[str, Any]] | None = None,
    full_game_lbr: dict[str, Any] | None = None,
) -> dict[str, Any]:
    v31_resolvers = v31_resolvers or []
    v30_resolvers = v30_resolvers or []
    if len(v31_resolvers) != len(v30_resolvers):
        raise ValueError("v31 and v30 resolver reports must be paired")

    reports = {"pot": pot, "payoff-exposure": payoff}
    tuning = {name: mean_tuning_rmse(report) for name, report in reports.items()}
    selected_name = min(tuning, key=lambda name: (tuning[name], name))
    selected = reports[selected_name]
    seed = selected_seed(selected)
    bands = mean_band_rmse(selected)
    baseline_bands = baseline["turnValueEvaluation"]["rmseByInvestedPotBandBb"]
    baseline_by_band = {
        "small": float(baseline_bands["smallAtMost3_5Each"]),
        "medium": float(baseline_bands["medium4To7_5Each"]),
        "large": float(baseline_bands["largeAtLeast10_5Each"]),
    }
    band_improvement = {
        band: (baseline_by_band[band] - bands[band]) / baseline_by_band[band]
        for band in POT_BANDS
    }

    resolver_pairs = [
        {
            "index": index,
            "board": candidate.get("state", {}).get("board"),
            "iterations": candidate.get("iterations"),
            "matched": candidate.get("state", {}).get("board")
            == previous.get("state", {}).get("board")
            and candidate.get("iterations") == previous.get("iterations"),
            "v31ExploitabilityBbPerHand": resolver_exploitability(candidate),
            "v30ExploitabilityBbPerHand": resolver_exploitability(previous),
        }
        for index, (candidate, previous) in enumerate(
            zip(v31_resolvers, v30_resolvers, strict=True)
        )
    ]
    for pair in resolver_pairs:
        old = pair["v30ExploitabilityBbPerHand"]
        pair["relativeImprovement"] = (
            (old - pair["v31ExploitabilityBbPerHand"]) / old if old else 0.0
        )
    mean_v31_resolver = (
        mean(row["v31ExploitabilityBbPerHand"] for row in resolver_pairs)
        if resolver_pairs
        else None
    )
    mean_v30_resolver = (
        mean(row["v30ExploitabilityBbPerHand"] for row in resolver_pairs)
        if resolver_pairs
        else None
    )

    gates: dict[str, dict[str, Any]] = {}

    def gate(name: str, passed: bool, measured: Any, threshold: str) -> None:
        gates[name] = {
            "passed": bool(passed),
            "measured": measured,
            "threshold": threshold,
        }

    gate("normalizationSelectedByTuningOnly",
         selected.get("valueNormalization") == selected_name, {
        "selected": selected_name, "meanTuningRmseBb": tuning,
    }, "report with lowest paired mean tuning RMSE")
    gate("pairedCrossSeedCorrelation",
         float(selected["crossSeedPredictionCorrelation"]["range"]) >= 0.95,
         selected["crossSeedPredictionCorrelation"]["range"], ">=0.95")
    gate("pythonRustParity",
         parity.get("validation", {}).get("status") == "accepted"
         and float(parity["maximumAbsoluteErrorBb"]) <= 1e-4,
         parity["maximumAbsoluteErrorBb"], "accepted and <=0.0001bb")
    gate("smallPotNoRegression", band_improvement["small"] >= -0.05,
         band_improvement["small"], ">=-5% relative improvement")
    gate("mediumPotImprovement", band_improvement["medium"] >= 0.25,
         band_improvement["medium"], ">=25% relative improvement")
    gate("largePotImprovement", band_improvement["large"] >= 0.25,
         band_improvement["large"], ">=25% relative improvement")
    gate("turnHoldoutRmse", float(selected["meanRangeRmseBb"]) <= 0.25,
         selected["meanRangeRmseBb"], "<=0.25bb")
    gate("resolverPairsMatched", bool(resolver_pairs)
         and all(row["matched"] for row in resolver_pairs),
         [{"board": row["board"], "iterations": row["iterations"],
           "matched": row["matched"]} for row in resolver_pairs],
         "at least one pair with identical board and iteration count")
    gate("matchedResolverImprovement",
         bool(resolver_pairs) and all(row["matched"] for row in resolver_pairs)
         and mean_v31_resolver < mean_v30_resolver,
         {"v31": mean_v31_resolver, "v30": mean_v30_resolver}, "v31 < v30")
    gate("flopDepthLimitedExploitability",
         mean_v31_resolver is not None and mean_v31_resolver <= 0.05,
         mean_v31_resolver, "<=0.05bb/hand")

    calibration_prerequisites = all(
        gates[name]["passed"]
        for name in (
            "pairedCrossSeedCorrelation", "pythonRustParity",
            "smallPotNoRegression", "mediumPotImprovement", "largePotImprovement",
            "turnHoldoutRmse", "resolverPairsMatched", "matchedResolverImprovement",
        )
    )
    # These are conditional implementation decisions, not fabricated results.
    corpus_512 = "eligible" if calibration_prerequisites else "not_run_prerequisite_failed"
    exact_hybrid = "eligible_after_512_validation" if calibration_prerequisites else "not_run_prerequisite_failed"
    gate("balanced512CorpusValidated", False, corpus_512,
         "run only after all 128-state calibration and resolver improvement gates pass")
    gate("exactLowSprAllInHybridValidated", False, exact_hybrid,
         "run only after the 512-state model improves downstream resolving")

    if full_game_lbr is None:
        lbr_measurement: Any = None
    else:
        players = full_game_lbr.get("players", [])
        lbr_measurement = {
            "networkSha256": full_game_lbr.get("network_sha256"),
            "trainingDeals": full_game_lbr.get("training_deals"),
            "evaluationDeals": full_game_lbr.get("evaluation_deals"),
            "lowerBoundBbPerHand": full_game_lbr.get(
                "approximate_exploitability_lower_bound_bb_per_hand"
            ),
            "lowerConfidenceBound99BbPerHand": full_game_lbr.get(
                "approximate_exploitability_lower_confidence_bound_99_percent_bb_per_hand"
            ),
            "minimumResolverLookupCoverage": min(
                (float(player.get("resolver_lookup_coverage", 0.0)) for player in players),
                default=0.0,
            ),
            "confidentInformationSets": sum(
                int(player.get("confident_information_sets", 0)) for player in players
            ),
            "interpretation": full_game_lbr.get("interpretation"),
        }
    gate("independentLearnedResponseEvaluated", full_game_lbr is not None
         and full_game_lbr.get("network_sha256") == selected.get("sourcePolicySha256"),
         lbr_measurement, "fresh frozen-policy response evaluation present")
    # Learned response is a lower bound and cannot satisfy this release gate.
    gate("fullGameExploitabilityUpperBound", False, None,
         "independent one-sided 99% upper bound <=0.10bb/hand")

    failed = [name for name, result in gates.items() if not result["passed"]]
    return {
        "schema": "hu-v31-value-calibration-sequence-v1",
        "modelVersion": "20bb-v31-calibration-candidate",
        "status": "rejected" if failed else "accepted",
        "activationAllowed": not failed,
        "activeManifestModified": False,
        "sourcePolicySha256": selected.get("sourcePolicySha256"),
        "sourceDatasetSha256": selected.get("datasetSha256"),
        "normalizationSelection": {
            "rule": "lowest paired mean range tuning RMSE; holdout is evaluation only",
            "meanTuningRmseBb": tuning,
            "selected": selected_name,
            "selectedSeed": seed["seed"],
            "selectedSeedTuningRmseBb": seed["metrics"]["bestTuningRmseBb"],
            "selectedWeights": seed["weights"],
        },
        "turnValueEvaluation": {
            "meanRangeRmseBb": selected["meanRangeRmseBb"],
            "meanNoRangeRmseBb": selected["meanNoRangeRmseBb"],
            "rangeRelativeImprovement": selected["rangeRelativeImprovement"],
            "crossSeedPredictionCorrelation": selected[
                "crossSeedPredictionCorrelation"
            ]["range"],
            "meanRmseByPotBandBb": bands,
            "baselineRmseByPotBandBb": baseline_by_band,
            "relativeImprovementByPotBand": band_improvement,
            "maximumPythonRustParityErrorBb": parity["maximumAbsoluteErrorBb"],
        },
        "resolverEvaluation": {
            "pairs": resolver_pairs,
            "meanV31ExploitabilityBbPerHand": mean_v31_resolver,
            "meanV30ExploitabilityBbPerHand": mean_v30_resolver,
        },
        "conditionalSteps": {
            "balanced512Corpus": corpus_512,
            "exactLowSprAllInHybrid": exact_hybrid,
        },
        "failedGates": failed,
        "gates": gates,
        "interpretation": (
            "The v31 calibration sequence is fail-closed. Conditional scale and exact-branch "
            "work remains unrun when the preceding measured gates fail. Learned-response "
            "results are rejection evidence only, never an exploitability upper bound."
        ),
    }


def main() -> None:
    args = parse_args()
    report = compose(
        load(args.baseline),
        load(args.pot_report),
        load(args.payoff_report),
        load(args.parity),
        [load(path) for path in args.v31_resolver],
        [load(path) for path in args.v30_resolver],
        load(args.full_game_lbr) if args.full_game_lbr else None,
    )
    selected_report_path = (
        args.pot_report
        if report["normalizationSelection"]["selected"] == "pot"
        else args.payoff_report
    )
    selected_weights = (
        selected_report_path.parent
        / report["normalizationSelection"]["selectedWeights"]
    )
    report["normalizationSelection"]["selectedWeightsSha256"] = hashlib.sha256(
        selected_weights.read_bytes()
    ).hexdigest()
    report["normalizationSelection"]["selectedWeightsBytes"] = (
        selected_weights.stat().st_size
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
