#!/usr/bin/env python3
"""Compose the fail-closed v33 resolver-leaf conditioning decision."""

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
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-leaf", type=Path, required=True)
    parser.add_argument("--candidate-leaf", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--candidate-resolver", type=Path, action="append", default=[])
    parser.add_argument("--baseline-resolver", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def range_variants(report: dict[str, Any]) -> list[dict[str, Any]]:
    variants = report.get("variants", {}).get("range", [])
    if len(variants) != 2:
        raise ValueError("paired evaluation requires exactly two range seeds")
    return variants


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


def relative_improvement(previous: float, candidate: float) -> float:
    return (previous - candidate) / previous


def compose(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    baseline_leaf: dict[str, Any],
    candidate_leaf: dict[str, Any],
    parity: dict[str, Any],
    candidate_resolvers: list[dict[str, Any]] | None = None,
    baseline_resolvers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    candidate_resolvers = candidate_resolvers or []
    baseline_resolvers = baseline_resolvers or []
    if len(candidate_resolvers) != len(baseline_resolvers):
        raise ValueError("candidate and baseline resolver reports must be paired")

    baseline_bands = mean_band_rmse(baseline)
    candidate_bands = mean_band_rmse(candidate)
    authentic_improvement = relative_improvement(
        float(baseline["meanRangeRmseBb"]), float(candidate["meanRangeRmseBb"])
    )
    band_improvement = {
        band: relative_improvement(baseline_bands[band], candidate_bands[band])
        for band in POT_BANDS
    }
    baseline_leaf_rmse = float(
        baseline_leaf["resolverReachEvaluation"]["reachWeightedRmseBb"]
    )
    candidate_leaf_rmse = float(
        candidate_leaf["resolverReachEvaluation"]["reachWeightedRmseBb"]
    )
    leaf_improvement = relative_improvement(baseline_leaf_rmse, candidate_leaf_rmse)

    primary_states = int(candidate.get("primaryStates", 0))
    supplemental = [int(value) for value in candidate.get("supplementalTrainingStates", [])]
    train = {int(value) for value in candidate.get("trainStates", [])}
    tuning = {int(value) for value in candidate.get("tuningStates", [])}
    holdout = {int(value) for value in candidate.get("validationStates", [])}
    supplemental_train_only = bool(supplemental) and all(
        value >= primary_states and value in train and value not in tuning and value not in holdout
        for value in supplemental
    )
    evaluation_hash = candidate_leaf.get("sourceDatasetSha256")
    component_hashes = candidate.get("componentDatasetSha256", [])

    gates: dict[str, dict[str, Any]] = {}

    def gate(name: str, passed: bool, measured: Any, threshold: str) -> None:
        gates[name] = {
            "passed": bool(passed),
            "measured": measured,
            "threshold": threshold,
        }

    gate(
        "matchedAuthenticHoldout",
        candidate.get("validationStates") == baseline.get("validationStates"),
        candidate.get("validationStates"),
        "identical untouched authentic validation states",
    )
    gate(
        "supplementalDataTrainingOnly",
        supplemental_train_only,
        {"primaryStates": primary_states, "supplementalStates": supplemental},
        "non-empty resolver supplement appears only in training",
    )
    gate(
        "resolverEvaluationCorpusUntouched",
        bool(evaluation_hash) and evaluation_hash not in component_hashes,
        {"evaluation": evaluation_hash, "trainingComponents": component_hashes},
        "evaluation dataset hash is absent from every training component",
    )
    gate(
        "sourceTargetsAccepted",
        candidate.get("sourceValidation", {}).get("status") == "accepted",
        candidate.get("sourceValidation"),
        "combined training targets accepted",
    )
    gate(
        "sourcePolicyPinned",
        candidate.get("sourcePolicySha256") == baseline.get("sourcePolicySha256")
        == candidate_leaf.get("sourcePolicySha256"),
        candidate.get("sourcePolicySha256"),
        "same frozen source policy across training and evaluation",
    )
    gate(
        "pairedCrossSeedCorrelation",
        float(candidate["crossSeedPredictionCorrelation"]["range"]) >= 0.95,
        candidate["crossSeedPredictionCorrelation"]["range"],
        ">=0.95",
    )
    gate(
        "pythonRustParity",
        parity.get("validation", {}).get("status") == "accepted"
        and float(parity["maximumAbsoluteErrorBb"]) <= 1e-4,
        parity.get("maximumAbsoluteErrorBb"),
        "accepted and <=0.0001bb",
    )
    gate(
        "authenticHoldoutNoRegression",
        authentic_improvement >= -0.05,
        authentic_improvement,
        ">=-5% matched authentic RMSE improvement",
    )
    for band in POT_BANDS:
        threshold = -0.10 if band == "small" else -0.05
        gate(
            f"{band}PotNoRegression",
            band_improvement[band] >= threshold,
            band_improvement[band],
            f">={threshold:.0%} matched authentic improvement",
        )
    gate(
        "resolverLeafReachWeightedImprovement",
        leaf_improvement >= 0.10,
        leaf_improvement,
        ">=10% on untouched resolver-reach-weighted leaves",
    )

    resolver_pairs = []
    for candidate_report, baseline_report in zip(
        candidate_resolvers, baseline_resolvers, strict=True
    ):
        matched = (
            candidate_report.get("state", {}).get("board")
            == baseline_report.get("state", {}).get("board")
            and candidate_report.get("iterations") == baseline_report.get("iterations")
        )
        candidate_value = resolver_exploitability(candidate_report)
        baseline_value = resolver_exploitability(baseline_report)
        resolver_pairs.append(
            {
                "board": candidate_report.get("state", {}).get("board"),
                "iterations": candidate_report.get("iterations"),
                "matched": matched,
                "candidateExploitabilityBbPerHand": candidate_value,
                "baselineExploitabilityBbPerHand": baseline_value,
                "relativeImprovement": relative_improvement(
                    baseline_value, candidate_value
                ),
            }
        )
    candidate_resolver_mean = (
        mean(row["candidateExploitabilityBbPerHand"] for row in resolver_pairs)
        if resolver_pairs
        else None
    )
    baseline_resolver_mean = (
        mean(row["baselineExploitabilityBbPerHand"] for row in resolver_pairs)
        if resolver_pairs
        else None
    )
    improved_boards = sum(row["relativeImprovement"] > 0.0 for row in resolver_pairs)
    prediction_prerequisites = all(
        gates[name]["passed"]
        for name in (
            "matchedAuthenticHoldout",
            "supplementalDataTrainingOnly",
            "resolverEvaluationCorpusUntouched",
            "sourceTargetsAccepted",
            "sourcePolicyPinned",
            "pairedCrossSeedCorrelation",
            "pythonRustParity",
            "authenticHoldoutNoRegression",
            "smallPotNoRegression",
            "mediumPotNoRegression",
            "largePotNoRegression",
            "resolverLeafReachWeightedImprovement",
        )
    )
    resolver_improvement = (
        relative_improvement(baseline_resolver_mean, candidate_resolver_mean)
        if candidate_resolver_mean is not None and baseline_resolver_mean is not None
        else None
    )
    gate(
        "matchedResolverImprovement",
        prediction_prerequisites
        and bool(resolver_pairs)
        and all(row["matched"] for row in resolver_pairs)
        and resolver_improvement is not None
        and resolver_improvement >= 0.02
        and improved_boards >= 2,
        {
            "eligible": prediction_prerequisites,
            "candidate": candidate_resolver_mean,
            "baseline": baseline_resolver_mean,
            "relativeImprovement": resolver_improvement,
            "improvedBoards": improved_boards,
        },
        ">=2% mean improvement and improvement on at least two of three matched boards",
    )
    gate(
        "fullGameExploitabilityUpperBound",
        False,
        None,
        "independent one-sided 99% upper bound <=0.10bb/hand",
    )

    selected = selected_seed(candidate)
    research_preferred = gates["matchedResolverImprovement"]["passed"]
    return {
        "schema": "hu-v33-resolver-leaf-conditioning-sequence-v1",
        "modelVersion": "20bb-v33-resolver-leaf-candidate",
        "status": "rejected",
        "activationAllowed": False,
        "activeManifestModified": False,
        "researchSelection": "v33" if research_preferred else "v31",
        "sourcePolicySha256": candidate.get("sourcePolicySha256"),
        "sourceDatasetSha256": candidate.get("datasetSha256"),
        "selectedSeed": selected["seed"],
        "selectedWeights": selected["weights"],
        "selectedSeedTuningRmseBb": selected["metrics"]["bestTuningRmseBb"],
        "authenticEvaluation": {
            "baselineMeanRangeRmseBb": baseline["meanRangeRmseBb"],
            "candidateMeanRangeRmseBb": candidate["meanRangeRmseBb"],
            "relativeImprovement": authentic_improvement,
            "baselinePotBandRmseBb": baseline_bands,
            "candidatePotBandRmseBb": candidate_bands,
            "relativeImprovementByPotBand": band_improvement,
            "crossSeedPredictionCorrelation": candidate[
                "crossSeedPredictionCorrelation"
            ]["range"],
            "maximumPythonRustParityErrorBb": parity.get(
                "maximumAbsoluteErrorBb"
            ),
        },
        "resolverLeafEvaluation": {
            "datasetSha256": evaluation_hash,
            "baselineReachWeightedRmseBb": baseline_leaf_rmse,
            "candidateReachWeightedRmseBb": candidate_leaf_rmse,
            "relativeImprovement": leaf_improvement,
            "sampledLeafReachMass": candidate_leaf[
                "resolverReachEvaluation"
            ]["sampledLeafReachMass"],
        },
        "resolverEvaluation": {
            "eligible": prediction_prerequisites,
            "pairs": resolver_pairs,
            "candidateMeanExploitabilityBbPerHand": candidate_resolver_mean,
            "baselineMeanExploitabilityBbPerHand": baseline_resolver_mean,
            "relativeImprovement": resolver_improvement,
            "improvedBoards": improved_boards,
        },
        "failedGates": [name for name, result in gates.items() if not result["passed"]],
        "gates": gates,
        "interpretation": (
            "Resolver-leaf targets are sampled from a frozen average resolver policy on "
            "training-only root boards. A disjoint leaf corpus and the original matched "
            "resolver boards remain untouched. Local improvement never substitutes for "
            "the missing full-game exploitability upper bound."
        ),
    }


def main() -> None:
    args = parse_args()
    report = compose(
        load(args.baseline),
        load(args.candidate),
        load(args.baseline_leaf),
        load(args.candidate_leaf),
        load(args.parity),
        [load(path) for path in args.candidate_resolver],
        [load(path) for path in args.baseline_resolver],
    )
    selected_path = args.candidate.parent / report["selectedWeights"]
    report["selectedWeightsSha256"] = hashlib.sha256(selected_path.read_bytes()).hexdigest()
    report["selectedWeightsBytes"] = selected_path.stat().st_size
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
