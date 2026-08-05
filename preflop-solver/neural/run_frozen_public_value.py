#!/usr/bin/env python3
"""Verify and execute one frozen public-value training configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA = "hu-public-value-frozen-training-config-v1"
TARGET_SCHEMA = "hu-turn-public-belief-cfv-dataset-v2"
REPORT_SCHEMA = "hu-turn-public-belief-value-network-pilot-v4"
SOLVER_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required JSON input is missing: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def resolve(path: str) -> Path:
    return SOLVER_ROOT / path


def verify_digest(path: Path, expected: str) -> None:
    if not path.is_file():
        raise ValueError(f"frozen input is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"frozen input SHA-256 mismatch for {path}: expected {expected}, got {actual}"
        )


def validate_target_dataset(
    dataset: dict[str, Any],
    expected_count: int,
    label: str,
    allowed_standalone_reasons: list[str] | None = None,
) -> None:
    if dataset.get("schema") != TARGET_SCHEMA:
        raise ValueError(f"{label} has an incompatible target schema")
    validation = dataset.get("validation", {})
    allowed = allowed_standalone_reasons or []
    if validation.get("status") == "accepted":
        if validation.get("reasons", []):
            raise ValueError(f"{label} has inconsistent accepted validation")
    elif validation.get("status") == "rejected":
        if validation.get("reasons") != allowed or not allowed:
            raise ValueError(f"{label} source labels are rejected")
    else:
        raise ValueError(f"{label} has an invalid validation status")
    if len(dataset.get("targets", [])) != expected_count:
        raise ValueError(f"{label} target count differs from the frozen config")


def verify_selection(config: dict[str, Any]) -> None:
    evidence = config["selectionEvidence"]
    selector_path = resolve(evidence["selectorArtifact"])
    report_path = resolve(evidence["selectedReport"])
    verify_digest(selector_path, evidence["selectorArtifactSha256"])
    verify_digest(report_path, evidence["selectedReportSha256"])

    selector = load_json(selector_path)
    if selector.get("status") != "frozen-for-fresh-holdout":
        raise ValueError("selector artifact is not frozen for a fresh holdout")
    if selector.get("holdoutMetricsConsulted") is not False:
        raise ValueError("selector artifact consulted holdout metrics")
    if selector.get("selectedReportSha256") != evidence["selectedReportSha256"]:
        raise ValueError("selector and frozen config disagree on the selected report")
    trainer = config["trainer"]
    frozen_training = {
        "architecture": trainer["architecture"],
        "variantSet": trainer["variantSet"],
        "featureSchema": trainer["featureSchema"],
        "suitAugmentationsPerState": trainer["suitAugmentationsPerState"],
        "valueNormalization": trainer["valueNormalization"],
        "steps": trainer["steps"],
        "batchSize": trainer["batchSize"],
        "evaluationInterval": trainer["evaluationInterval"],
        "learningRate": trainer["learningRate"],
        "learningRateFinal": trainer["learningRateFinal"],
        "learningRateSchedule": trainer["learningRateSchedule"],
        "adamBiasCorrection": trainer["adamBiasCorrection"],
        "earlyStoppingPatience": trainer["earlyStoppingPatience"],
        "huberDelta": trainer["huberDelta"],
        "rawBbAuxiliaryWeight": trainer["rawBbAuxiliaryWeight"],
        "minimumPrimaryBatchFraction": trainer["minimumPrimaryBatchFraction"],
        "supplementalSamplingWeight": trainer["supplementalSamplingWeight"],
    }
    if selector.get("selectedConfiguration") != frozen_training:
        raise ValueError("trainer controls differ from the tuning-selected configuration")

    report = load_json(report_path)
    variants = report.get("variants", {}).get("range", [])
    by_seed = {int(entry["seed"]): entry for entry in variants}
    seeds = [int(seed) for seed in evidence["selectedTrainingSeeds"]]
    expected_hashes = evidence["selectedWeightSha256"]
    if len(seeds) != 2 or len(set(seeds)) != 2 or len(expected_hashes) != 2:
        raise ValueError("frozen selection must contain two independent weights")
    for seed, expected_hash in zip(seeds, expected_hashes):
        entry = by_seed.get(seed)
        if entry is None:
            raise ValueError("selected report is missing a frozen training seed")
        verify_digest(report_path.parent / entry["weights"], expected_hash)


def verify_datasets(config: dict[str, Any]) -> None:
    primary_config = config["primaryDataset"]
    primary_path = resolve(primary_config["path"])
    primary = load_json(primary_path)
    expected_count = int(primary_config["expectedStateCount"])
    if primary_config.get("expectedSchema") != TARGET_SCHEMA:
        raise ValueError("frozen primary schema is incompatible")
    validate_target_dataset(primary, expected_count, "primary dataset")
    if primary.get("component_seeds") != primary_config["expectedComponentSeeds"]:
        raise ValueError("primary component seeds differ from the frozen config")
    component_counts = primary.get("component_target_counts")
    if (
        not isinstance(component_counts, list)
        or len(component_counts) != len(primary_config["expectedComponentSeeds"])
        or sum(component_counts) != expected_count
    ):
        raise ValueError("primary component target counts are invalid")
    component_hashes = primary.get("component_dataset_sha256")
    if (
        not isinstance(component_hashes, list)
        or len(component_hashes) != len(component_counts)
        or len(set(component_hashes)) != len(component_hashes)
        or any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in component_hashes
        )
    ):
        raise ValueError("primary component SHA-256 provenance is invalid")
    holdout_start = int(primary_config["holdoutStartIndex"])
    if holdout_start + int(primary_config["expectedHoldoutStateCount"]) != expected_count:
        raise ValueError("frozen holdout boundary does not cover the reserved tail")
    reserved = primary_config["reservedHoldoutComponentIndices"]
    if (
        not isinstance(reserved, list)
        or not reserved
        or reserved != list(range(reserved[0], len(component_counts)))
    ):
        raise ValueError("reserved holdout components must be a contiguous tail")
    if sum(component_counts[: reserved[0]]) != holdout_start:
        raise ValueError("component ordering does not match the frozen holdout boundary")

    for supplement in config["trainingOnlySupplements"]:
        path = resolve(supplement["path"])
        verify_digest(path, supplement["sha256"])
        validate_target_dataset(
            load_json(path),
            int(supplement["stateCount"]),
            f"supplement {path}",
            supplement.get("allowedStandaloneValidationReasons"),
        )


def trainer_command(config: dict[str, Any]) -> list[str]:
    primary = config["primaryDataset"]
    trainer = config["trainer"]
    command = [
        sys.executable,
        "neural/train_public_value_network.py",
        "--dataset",
        primary["path"],
    ]
    for supplement in config["trainingOnlySupplements"]:
        command.extend(("--supplemental-dataset", supplement["path"]))
    command.extend(
        (
            "--supplemental-sampling-weight",
            str(trainer["supplementalSamplingWeight"]),
            "--minimum-primary-batch-fraction",
            str(trainer["minimumPrimaryBatchFraction"]),
            "--output-dir",
            trainer["outputDirectory"],
            "--architecture",
            trainer["architecture"],
            "--feature-schema",
            trainer["featureSchema"],
            "--feature-workers",
            str(trainer["featureWorkers"]),
            "--feature-cache-dir",
            trainer["featureCacheDirectory"],
            "--value-normalization",
            trainer["valueNormalization"],
            "--variant-set",
            trainer["variantSet"],
            "--steps",
            str(trainer["steps"]),
            "--batch-size",
            str(trainer["batchSize"]),
            "--learning-rate",
            str(trainer["learningRate"]),
            "--seeds",
            ",".join(str(seed) for seed in trainer["trainingSeeds"]),
            "--split-seed",
            str(trainer["splitSeed"]),
            "--validation-fraction",
            str(trainer["validationFraction"]),
            "--tuning-fraction",
            str(trainer["tuningFraction"]),
            "--holdout-start-index",
            str(primary["holdoutStartIndex"]),
            "--evaluation-interval",
            str(trainer["evaluationInterval"]),
            "--early-stopping-patience",
            str(trainer["earlyStoppingPatience"]),
            "--maximum-rmse-bb",
            str(config["valueReleaseGates"]["maximumPerSeedHoldoutRmseBb"]),
            "--minimum-cross-seed-correlation",
            str(
                config["valueReleaseGates"][
                    "minimumHoldoutCrossSeedPredictionCorrelation"
                ]
            ),
            "--minimum-tuning-cross-seed-correlation",
            str(
                config["valueReleaseGates"][
                    "minimumTuningCrossSeedPredictionCorrelation"
                ]
            ),
            "--suit-augmentations",
            str(trainer["suitAugmentationsPerState"]),
            "--huber-delta",
            str(trainer["huberDelta"]),
            "--raw-bb-auxiliary-weight",
            str(trainer["rawBbAuxiliaryWeight"]),
        )
    )
    if trainer["learningRateFinal"] is not None:
        command.extend(("--learning-rate-final", str(trainer["learningRateFinal"])))
    if trainer["adamBiasCorrection"]:
        command.append("--adam-bias-correction")
    return command


def verify_output_report(config: dict[str, Any]) -> dict[str, Any]:
    primary = config["primaryDataset"]
    trainer = config["trainer"]
    gates = config["valueReleaseGates"]
    output = resolve(trainer["outputDirectory"])
    report_path = output / "turn-value-paired-report.json"
    report = load_json(report_path)
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError("frozen training report schema is incompatible")
    expected_components = [
        sha256_file(resolve(primary["path"])),
        *(entry["sha256"] for entry in config["trainingOnlySupplements"]),
    ]
    if report.get("componentDatasetSha256") != expected_components:
        raise ValueError("frozen training report used different corpus bytes")
    if report.get("sourceValidation", {}).get("status") != gates["sourceValidationStatus"]:
        raise ValueError("frozen training report source labels are rejected")
    if report.get("primaryStates") != primary["expectedStateCount"]:
        raise ValueError("frozen training report has the wrong primary state count")
    if report.get("holdoutStartIndex") != primary["holdoutStartIndex"]:
        raise ValueError("frozen training report has the wrong holdout boundary")
    expected_holdout = list(
        range(primary["holdoutStartIndex"], primary["expectedStateCount"])
    )
    if report.get("validationStates") != expected_holdout:
        raise ValueError("frozen training report did not evaluate the reserved holdout")
    if report.get("splitSeed") != trainer["splitSeed"]:
        raise ValueError("frozen training report used the wrong split seed")
    variants = report.get("variants", {}).get("range", [])
    if [entry.get("seed") for entry in variants] != trainer["trainingSeeds"]:
        raise ValueError("frozen training report used different release seeds")
    holdout_rmse = [
        float(entry.get("metrics", {}).get("weightedRmseBb", float("nan")))
        for entry in variants
    ]
    if (
        len(holdout_rmse) != 2
        or any(not value <= gates["maximumPerSeedHoldoutRmseBb"] for value in holdout_rmse)
    ):
        raise ValueError("frozen training report failed the per-seed holdout RMSE gate")
    holdout_correlation = float(
        report.get("crossSeedPredictionCorrelation", {}).get("range", float("nan"))
    )
    tuning_correlation = float(
        report.get("tuningCrossSeedPredictionCorrelation", {}).get(
            "range", float("nan")
        )
    )
    if not holdout_correlation >= gates["minimumHoldoutCrossSeedPredictionCorrelation"]:
        raise ValueError("frozen training report failed holdout cross-seed agreement")
    if not tuning_correlation >= gates["minimumTuningCrossSeedPredictionCorrelation"]:
        raise ValueError("frozen training report failed tuning cross-seed agreement")
    if report.get("validation", {}).get("status") != "accepted":
        raise ValueError("frozen training report is rejected")

    weights = []
    for entry in variants:
        path = output / entry["weights"]
        weights.append(
            {"seed": entry["seed"], "path": str(path), "sha256": sha256_file(path)}
        )
    return {
        "status": "accepted",
        "report": str(report_path),
        "reportSha256": sha256_file(report_path),
        "holdoutRmseBb": holdout_rmse,
        "holdoutCrossSeedPredictionCorrelation": holdout_correlation,
        "tuningCrossSeedPredictionCorrelation": tuning_correlation,
        "weights": weights,
    }


def verify_config(config_path: Path) -> tuple[dict[str, Any], list[str]]:
    config = load_json(config_path)
    if config.get("schema") != SCHEMA:
        raise ValueError("incompatible frozen public-value config")
    if config.get("status") != "frozen-for-fresh-holdout":
        raise ValueError("public-value config is not frozen for a fresh holdout")
    if config.get("activationAllowed") is not False:
        raise ValueError("research training config must keep activation disabled")
    seeds = config["trainer"]["trainingSeeds"]
    if len(seeds) != 2 or len(set(seeds)) != 2:
        raise ValueError("release training requires two independent seeds")
    if set(seeds) & set(config["selectionEvidence"]["selectedTrainingSeeds"]):
        raise ValueError("release seeds must be independent of tuning seeds")
    verify_selection(config)
    verify_datasets(config)
    output = resolve(config["trainer"]["outputDirectory"])
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"frozen output directory is not empty: {output}")
    return config, trainer_command(config)


def main() -> None:
    args = parse_args()
    try:
        config, command = verify_config(args.config)
        print(json.dumps({"cwd": str(SOLVER_ROOT), "command": command}, indent=2))
        if not args.dry_run:
            subprocess.run(command, cwd=SOLVER_ROOT, check=True)
            print(json.dumps(verify_output_report(config), indent=2, sort_keys=True))
    except (
        KeyError,
        TypeError,
        ValueError,
        OSError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
