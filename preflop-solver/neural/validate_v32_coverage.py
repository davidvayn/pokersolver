#!/usr/bin/env python3
"""Compose the fail-closed v32 off-policy coverage pilot decision."""

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
    parser.add_argument("--deep-pilot", type=Path, required=True)
    parser.add_argument("--deep-parity", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-parity", type=Path, required=True)
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


def compose(
    baseline: dict[str, Any],
    deep: dict[str, Any],
    deep_parity: dict[str, Any],
    candidate: dict[str, Any],
    candidate_parity: dict[str, Any],
    candidate_resolvers: list[dict[str, Any]] | None = None,
    baseline_resolvers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    candidate_resolvers = candidate_resolvers or []
    baseline_resolvers = baseline_resolvers or []
    if len(candidate_resolvers) != len(baseline_resolvers):
        raise ValueError("candidate and baseline resolver reports must be paired")

    baseline_bands = mean_band_rmse(baseline)
    candidate_bands = mean_band_rmse(candidate)
    deep_bands = mean_band_rmse(deep)
    improvement = {
        band: (baseline_bands[band] - candidate_bands[band]) / baseline_bands[band]
        for band in POT_BANDS
    }
    overall_improvement = (
        float(baseline["meanRangeRmseBb"]) - float(candidate["meanRangeRmseBb"])
    ) / float(baseline["meanRangeRmseBb"])

    primary_states = int(candidate.get("primaryStates", 0))
    supplemental = [int(value) for value in candidate.get("supplementalTrainingStates", [])]
    train = {int(value) for value in candidate.get("trainStates", [])}
    tuning = {int(value) for value in candidate.get("tuningStates", [])}
    holdout = {int(value) for value in candidate.get("validationStates", [])}
    supplemental_train_only = bool(supplemental) and all(
        value >= primary_states and value in train and value not in tuning and value not in holdout
        for value in supplemental
    )

    gates: dict[str, dict[str, Any]] = {}

    def gate(name: str, passed: bool, measured: Any, threshold: str) -> None:
        gates[name] = {
            "passed": bool(passed),
            "measured": measured,
            "threshold": threshold,
        }

    gate(
        "matchedPrimaryHoldout",
        candidate.get("validationStates") == baseline.get("validationStates"),
        {
            "baseline": baseline.get("validationStates"),
            "candidate": candidate.get("validationStates"),
        },
        "identical untouched primary validation-state indices",
    )
    gate(
        "supplementalDataTrainingOnly",
        supplemental_train_only,
        {"primaryStates": primary_states, "supplementalStates": supplemental},
        "non-empty supplemental states appear only in training",
    )
    gate(
        "sourceTargetsAccepted",
        candidate.get("sourceValidation", {}).get("status") == "accepted",
        candidate.get("sourceValidation"),
        "combined target validation accepted",
    )
    gate(
        "sourcePolicyPinned",
        candidate.get("sourcePolicySha256") == baseline.get("sourcePolicySha256"),
        candidate.get("sourcePolicySha256"),
        "same frozen source policy as baseline",
    )
    gate(
        "compactPotModel",
        candidate.get("architecture") == "compact"
        and candidate.get("valueNormalization") == "pot",
        {
            "architecture": candidate.get("architecture"),
            "normalization": candidate.get("valueNormalization"),
        },
        "compact architecture with pot normalization",
    )
    gate(
        "pairedCrossSeedCorrelation",
        float(candidate["crossSeedPredictionCorrelation"]["range"]) >= 0.95,
        candidate["crossSeedPredictionCorrelation"]["range"],
        ">=0.95",
    )
    gate(
        "pythonRustParity",
        candidate_parity.get("validation", {}).get("status") == "accepted"
        and float(candidate_parity["maximumAbsoluteErrorBb"]) <= 1e-4,
        candidate_parity.get("maximumAbsoluteErrorBb"),
        "accepted and <=0.0001bb",
    )
    gate(
        "overallHoldoutImprovement",
        overall_improvement >= 0.05,
        overall_improvement,
        ">=5% matched-holdout RMSE improvement",
    )
    gate(
        "smallPotNoRegression",
        improvement["small"] >= -0.05,
        improvement["small"],
        ">=-5% matched-holdout improvement",
    )
    for band in ("medium", "large"):
        gate(
            f"{band}PotImprovement",
            improvement[band] >= 0.05,
            improvement[band],
            ">=5% matched-holdout improvement",
        )

    deep_relative = (
        float(baseline["meanRangeRmseBb"]) - float(deep["meanRangeRmseBb"])
    ) / float(baseline["meanRangeRmseBb"])
    gate(
        "deepCapacityPilotPreferred",
        deep_relative >= 0.05,
        {
            "relativeImprovement": deep_relative,
            "parityErrorBb": deep_parity.get("maximumAbsoluteErrorBb"),
        },
        ">=5% matched-holdout improvement (research selection gate only)",
    )

    upstream_names = (
        "matchedPrimaryHoldout",
        "supplementalDataTrainingOnly",
        "sourceTargetsAccepted",
        "sourcePolicyPinned",
        "compactPotModel",
        "pairedCrossSeedCorrelation",
        "pythonRustParity",
        "overallHoldoutImprovement",
        "smallPotNoRegression",
        "mediumPotImprovement",
        "largePotImprovement",
    )
    resolver_eligible = all(gates[name]["passed"] for name in upstream_names)
    resolver_pairs = []
    for candidate_report, baseline_report in zip(
        candidate_resolvers, baseline_resolvers, strict=True
    ):
        matched = (
            candidate_report.get("state", {}).get("board")
            == baseline_report.get("state", {}).get("board")
            and candidate_report.get("iterations") == baseline_report.get("iterations")
        )
        resolver_pairs.append(
            {
                "board": candidate_report.get("state", {}).get("board"),
                "iterations": candidate_report.get("iterations"),
                "matched": matched,
                "candidateExploitabilityBbPerHand": resolver_exploitability(candidate_report),
                "baselineExploitabilityBbPerHand": resolver_exploitability(baseline_report),
            }
        )
    candidate_mean = (
        mean(row["candidateExploitabilityBbPerHand"] for row in resolver_pairs)
        if resolver_pairs
        else None
    )
    baseline_mean = (
        mean(row["baselineExploitabilityBbPerHand"] for row in resolver_pairs)
        if resolver_pairs
        else None
    )
    gate(
        "matchedResolverImprovement",
        resolver_eligible
        and bool(resolver_pairs)
        and all(row["matched"] for row in resolver_pairs)
        and candidate_mean < baseline_mean,
        {
            "eligible": resolver_eligible,
            "candidate": candidate_mean,
            "baseline": baseline_mean,
        },
        "upstream gates pass and matched candidate mean < baseline mean",
    )
    gate(
        "fullGameExploitabilityUpperBound",
        False,
        None,
        "independent one-sided 99% upper bound <=0.10bb/hand",
    )

    selected = selected_seed(candidate)
    failed_release = [name for name, value in gates.items() if not value["passed"]]
    return {
        "schema": "hu-v32-off-policy-coverage-sequence-v1",
        "modelVersion": "20bb-v32-off-policy-coverage-candidate",
        "status": "rejected",
        "activationAllowed": False,
        "activeManifestModified": False,
        "sourcePolicySha256": candidate.get("sourcePolicySha256"),
        "sourceDatasetSha256": candidate.get("datasetSha256"),
        "selectedSeed": selected["seed"],
        "selectedWeights": selected["weights"],
        "selectedSeedTuningRmseBb": selected["metrics"]["bestTuningRmseBb"],
        "turnValueEvaluation": {
            "baselineMeanRangeRmseBb": baseline["meanRangeRmseBb"],
            "candidateMeanRangeRmseBb": candidate["meanRangeRmseBb"],
            "relativeImprovement": overall_improvement,
            "baselinePotBandRmseBb": baseline_bands,
            "candidatePotBandRmseBb": candidate_bands,
            "relativeImprovementByPotBand": improvement,
            "crossSeedPredictionCorrelation": candidate[
                "crossSeedPredictionCorrelation"
            ]["range"],
            "maximumPythonRustParityErrorBb": candidate_parity.get(
                "maximumAbsoluteErrorBb"
            ),
        },
        "capacityPilot": {
            "architecture": deep.get("architecture"),
            "baselineMeanRangeRmseBb": baseline["meanRangeRmseBb"],
            "deepMeanRangeRmseBb": deep["meanRangeRmseBb"],
            "relativeImprovement": deep_relative,
            "potBandRmseBb": deep_bands,
            "maximumPythonRustParityErrorBb": deep_parity.get(
                "maximumAbsoluteErrorBb"
            ),
            "decision": "rejected_keep_compact",
        },
        "resolverEvaluation": {
            "eligible": resolver_eligible,
            "pairs": resolver_pairs,
            "candidateMeanExploitabilityBbPerHand": candidate_mean,
            "baselineMeanExploitabilityBbPerHand": baseline_mean,
        },
        "failedGates": failed_release,
        "gates": gates,
        "interpretation": (
            "The controlled-exploration pilot is evaluated on the identical untouched "
            "primary holdout. Supplemental states may train the network but cannot enter "
            "tuning or validation. Passing prediction gates only permits matched resolver "
            "evaluation; it never substitutes for a full-game exploitability upper bound."
        ),
    }


def main() -> None:
    args = parse_args()
    report = compose(
        load(args.baseline),
        load(args.deep_pilot),
        load(args.deep_parity),
        load(args.candidate),
        load(args.candidate_parity),
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
