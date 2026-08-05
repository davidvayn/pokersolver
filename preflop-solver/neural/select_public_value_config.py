#!/usr/bin/env python3
"""Freeze public-value hyperparameters using tuning evidence only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "hu-public-value-tuning-selection-v1"
REPORT_SCHEMA = "hu-turn-public-belief-value-network-pilot-v4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--minimum-cross-seed-correlation", type=float, default=0.95)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def comparable_identity(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "datasetSha256": report.get("datasetSha256"),
        "componentDatasetSha256": report.get("componentDatasetSha256"),
        "splitSeed": report.get("splitSeed"),
        "trainStates": report.get("trainStates"),
        "tuningStates": report.get("tuningStates"),
        "validationStates": report.get("validationStates"),
        "sourcePolicySha256": report.get("sourcePolicySha256"),
    }


def summarize_candidate(
    path: Path, minimum_cross_seed_correlation: float
) -> dict[str, Any]:
    report = json.loads(path.read_text())
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError(f"incompatible value report: {path}")
    if report.get("sourceValidation", {}).get("status") != "accepted":
        raise ValueError(f"value report source labels are rejected: {path}")
    variants = report.get("variants", {}).get("range", [])
    if len(variants) != 2 or len({entry.get("seed") for entry in variants}) != 2:
        raise ValueError(f"value report does not contain two independent seeds: {path}")
    tuning_rmse = []
    weights = []
    for entry in variants:
        metrics = entry.get("metrics", {})
        best = float(metrics.get("bestTuningRmseBb", float("nan")))
        final = float(
            metrics.get("finalTuningMetrics", {}).get(
                "weightedRmseBb", float("nan")
            )
        )
        if not np.isfinite(best) or not np.isfinite(final) or abs(best - final) > 1e-6:
            raise ValueError(f"value report did not restore its best tuning checkpoint: {path}")
        weight_path = path.parent / str(entry.get("weights", ""))
        if not weight_path.is_file():
            raise ValueError(f"value report weight file is missing: {weight_path}")
        tuning_rmse.append(best)
        weights.append(
            {
                "seed": int(entry["seed"]),
                "path": str(weight_path),
                "sha256": sha256_file(weight_path),
            }
        )
    correlation = float(
        report.get("tuningCrossSeedPredictionCorrelation", {}).get(
            "range", float("nan")
        )
    )
    if not np.isfinite(correlation) or correlation < minimum_cross_seed_correlation:
        raise ValueError(f"value report fails tuning cross-seed agreement: {path}")
    ensemble_tuning = float(
        report.get("twoSeedOutputEnsembleMetrics", {})
        .get("range", {})
        .get("tuning", {})
        .get("weightedRmseBb", float("nan"))
    )
    if not np.isfinite(ensemble_tuning):
        raise ValueError(f"value report lacks tuning ensemble diagnostics: {path}")
    return {
        "report": str(path),
        "reportSha256": sha256_file(path),
        "identity": comparable_identity(report),
        "configuration": {
            "architecture": report.get("architecture"),
            "featureSchema": report.get("featureSchema"),
            "valueNormalization": report.get("valueNormalization"),
            "steps": report.get("steps"),
            "batchSize": report.get("batchSize"),
            "learningRate": report.get("learningRate"),
            "learningRateFinal": report.get("learningRateFinal"),
            "learningRateSchedule": report.get("learningRateSchedule", "constant"),
            "earlyStoppingPatience": report.get("earlyStoppingPatience"),
            "huberDelta": report.get("loss", {}).get("huberDelta"),
            "rawBbAuxiliaryWeight": report.get("loss", {}).get(
                "rawBbAuxiliaryWeight"
            ),
            "minimumPrimaryBatchFraction": report.get(
                "minimumPrimaryBatchFraction"
            ),
            "supplementalSamplingWeight": report.get("supplementalSamplingWeight"),
        },
        "seedTuningRmseBb": tuning_rmse,
        "maximumSeedTuningRmseBb": max(tuning_rmse),
        "meanSeedTuningRmseBb": float(np.mean(tuning_rmse)),
        "tuningCrossSeedPredictionCorrelation": correlation,
        "twoSeedOutputEnsembleTuningRmseBb": ensemble_tuning,
        "weights": weights,
    }


def select_candidate(
    paths: list[Path], minimum_cross_seed_correlation: float = 0.95
) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one value report is required")
    candidates = [
        summarize_candidate(path, minimum_cross_seed_correlation) for path in paths
    ]
    identity = candidates[0]["identity"]
    if any(candidate["identity"] != identity for candidate in candidates[1:]):
        raise ValueError("value candidates do not use identical data and split membership")
    selected = min(
        candidates,
        key=lambda candidate: (
            candidate["maximumSeedTuningRmseBb"],
            candidate["meanSeedTuningRmseBb"],
            candidate["twoSeedOutputEnsembleTuningRmseBb"],
            candidate["reportSha256"],
        ),
    )
    return {
        "schema": SCHEMA,
        "status": "frozen-for-fresh-holdout",
        "selectionCriterion": (
            "minimum maximum per-seed tuning RMSE, then mean per-seed tuning "
            "RMSE, then diagnostic output-ensemble tuning RMSE"
        ),
        "holdoutMetricsConsulted": False,
        "minimumTuningCrossSeedPredictionCorrelation": (
            minimum_cross_seed_correlation
        ),
        "comparableIdentity": identity,
        "selectedReportSha256": selected["reportSha256"],
        "selectedConfiguration": selected["configuration"],
        "candidates": candidates,
    }


def main() -> None:
    args = parse_args()
    selection = select_candidate(args.reports, args.minimum_cross_seed_correlation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
