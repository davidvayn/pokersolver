#!/usr/bin/env python3
"""Select a value configuration from disjoint resolver-reach cross-validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import select_public_value_config as value_selection


SPEC_SCHEMA = "hu-resolver-reach-crossfit-selection-spec-v1"
OUTPUT_SCHEMA = "hu-resolver-reach-crossfit-selection-v1"
DIAGNOSTIC_SCHEMA = "hu-public-value-texture-diagnostics-v1"
DATASET_SCHEMA = "hu-turn-public-belief-cfv-dataset-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument(
        "--repository-root",
        type=Path,
        help="preflop-solver directory; defaults to the spec's parent directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(repository_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else repository_root / path


def repository_relative_path(repository_root: Path, path: Path) -> str:
    root = repository_root.resolve()
    candidate = path.resolve()
    try:
        return str(candidate.relative_to(root))
    except ValueError as error:
        raise ValueError(
            f"resolver selection evidence escapes the repository root: {candidate}"
        ) from error


def accepted_dataset(path: Path) -> str:
    payload = json.loads(path.read_text())
    if payload.get("schema") != DATASET_SCHEMA:
        raise ValueError(f"resolver cross-fit dataset has the wrong schema: {path}")
    if payload.get("validation", {}).get("status") != "accepted":
        raise ValueError(f"resolver cross-fit dataset is not accepted: {path}")
    return sha256_file(path)


def checked_model_hashes(
    repository_root: Path, entries: list[dict[str, Any]]
) -> dict[int, str]:
    hashes: dict[int, str] = {}
    for entry in entries:
        path = resolve_path(repository_root, str(entry["path"]))
        actual = sha256_file(path)
        if actual != entry.get("sha256"):
            raise ValueError(f"baseline model hash mismatch: {path}")
        seed = int(entry["seed"])
        if seed in hashes:
            raise ValueError("baseline model seeds must be unique")
        hashes[seed] = actual
    if len(hashes) != 2:
        raise ValueError("baseline must pin exactly two independent models")
    return hashes


def diagnostic_metric(
    repository_root: Path,
    path: Path,
    dataset_sha256: str,
    expected_model_hashes: dict[int, str],
) -> dict[str, Any]:
    report = json.loads(path.read_text())
    if report.get("schema") != DIAGNOSTIC_SCHEMA:
        raise ValueError(f"resolver diagnostic has the wrong schema: {path}")
    if report.get("sourceDatasetSha256") != dataset_sha256:
        raise ValueError(f"resolver diagnostic uses the wrong dataset: {path}")
    seed = int(report.get("modelSeed", -1))
    if seed not in expected_model_hashes:
        raise ValueError(f"resolver diagnostic uses an unexpected model seed: {path}")
    if report.get("modelSha256") != expected_model_hashes[seed]:
        raise ValueError(f"resolver diagnostic model hash mismatch: {path}")
    evaluation = report.get("resolverReachEvaluation") or {}
    rmse = float(evaluation.get("reachWeightedRmseBb", float("nan")))
    reach_mass = float(evaluation.get("sampledLeafReachMass", float("nan")))
    if not np.isfinite(rmse) or rmse < 0.0 or not np.isfinite(reach_mass) or reach_mass <= 0.0:
        raise ValueError(f"resolver diagnostic lacks a valid reach metric: {path}")
    return {
        "path": repository_relative_path(repository_root, path),
        "sha256": sha256_file(path),
        "modelSeed": seed,
        "modelSha256": expected_model_hashes[seed],
        "datasetSha256": dataset_sha256,
        "resolverReachWeightedRmseBb": rmse,
        "sampledLeafReachMass": reach_mass,
    }


def fold_diagnostics(
    repository_root: Path,
    fold: dict[str, Any],
    expected_model_hashes: dict[int, str],
) -> dict[str, Any]:
    dataset_path = resolve_path(repository_root, str(fold["evaluationDataset"]))
    dataset_sha256 = accepted_dataset(dataset_path)
    diagnostics = [
        diagnostic_metric(
            repository_root,
            resolve_path(repository_root, str(path)),
            dataset_sha256,
            expected_model_hashes,
        )
        for path in fold.get("diagnostics", [])
    ]
    if len(diagnostics) != len(expected_model_hashes) or {
        entry["modelSeed"] for entry in diagnostics
    } != set(expected_model_hashes):
        raise ValueError("each resolver fold requires one diagnostic per model seed")
    return {
        "name": str(fold["name"]),
        "evaluationDataset": repository_relative_path(repository_root, dataset_path),
        "evaluationDatasetSha256": dataset_sha256,
        "diagnostics": diagnostics,
    }


def baseline_summary(
    repository_root: Path, baseline: dict[str, Any]
) -> dict[str, Any]:
    model_hashes = checked_model_hashes(repository_root, baseline.get("models", []))
    folds = [
        fold_diagnostics(repository_root, fold, model_hashes)
        for fold in baseline.get("folds", [])
    ]
    if len(folds) != 2 or len({fold["name"] for fold in folds}) != 2:
        raise ValueError("baseline requires two named resolver cross-validation folds")
    if len({fold["evaluationDatasetSha256"] for fold in folds}) != 2:
        raise ValueError("baseline resolver folds must use two different datasets")
    values = [
        metric["resolverReachWeightedRmseBb"]
        for fold in folds
        for metric in fold["diagnostics"]
    ]
    return {
        "models": [
            {
                **entry,
                "path": repository_relative_path(
                    repository_root,
                    resolve_path(repository_root, str(entry["path"])),
                ),
            }
            for entry in baseline["models"]
        ],
        "folds": folds,
        "maximumResolverReachWeightedRmseBb": max(values),
        "meanResolverReachWeightedRmseBb": float(np.mean(values)),
    }


def candidate_summary(
    repository_root: Path,
    candidate: dict[str, Any],
    minimum_cross_seed_correlation: float,
    maximum_authentic_tuning_rmse_bb: float,
    maximum_resolver_rmse_bb: float,
    expected_fold_dataset_hashes: dict[str, str],
) -> dict[str, Any]:
    folds = []
    configurations = []
    all_seeds: set[int] = set()
    for fold in candidate.get("folds", []):
        report_path = resolve_path(repository_root, str(fold["trainingReport"]))
        training = value_selection.summarize_candidate(
            report_path, minimum_cross_seed_correlation
        )
        training["report"] = repository_relative_path(repository_root, report_path)
        for weight in training["weights"]:
            weight["path"] = repository_relative_path(
                repository_root,
                resolve_path(repository_root, str(weight["path"])),
            )
        seeds = {entry["seed"] for entry in training["weights"]}
        if all_seeds & seeds:
            raise ValueError("resolver cross-validation folds reuse a training seed")
        all_seeds |= seeds
        model_hashes = {entry["seed"]: entry["sha256"] for entry in training["weights"]}
        evaluation = fold_diagnostics(repository_root, fold, model_hashes)
        held_out_hash = evaluation["evaluationDatasetSha256"]
        report = json.loads(report_path.read_text())
        if report.get("holdoutStartIndex") is not None:
            raise ValueError(
                "resolver cross-validation cannot reuse a predecessor release holdout"
            )
        validation_rmse = [
            float(entry.get("metrics", {}).get("weightedRmseBb", float("nan")))
            for entry in report.get("variants", {}).get("range", [])
        ]
        if len(validation_rmse) != 2 or any(
            not np.isfinite(value) for value in validation_rmse
        ):
            raise ValueError("resolver cross-validation lacks authentic validation metrics")
        components = set(report.get("componentDatasetSha256", []))
        if held_out_hash == report.get("datasetSha256") or held_out_hash in components:
            raise ValueError("resolver cross-validation evaluation leaked into training")
        configurations.append(training["configuration"])
        folds.append(
            {
                "name": evaluation["name"],
                "trainingReport": training,
                "seedAuthenticValidationRmseBb": validation_rmse,
                **evaluation,
            }
        )
    if len(folds) != 2 or len({fold["name"] for fold in folds}) != 2:
        raise ValueError("each candidate requires two named resolver cross-validation folds")
    if configurations[0] != configurations[1]:
        raise ValueError("resolver cross-validation folds use different configurations")
    held_out_hashes = {fold["evaluationDatasetSha256"] for fold in folds}
    if len(held_out_hashes) != 2:
        raise ValueError("resolver cross-validation folds must hold out different datasets")
    candidate_fold_hashes = {
        fold["name"]: fold["evaluationDatasetSha256"] for fold in folds
    }
    if candidate_fold_hashes != expected_fold_dataset_hashes:
        raise ValueError("candidate and baseline resolver folds are not identical")
    for fold in folds:
        components = set(fold["trainingReport"]["identity"]["componentDatasetSha256"])
        if not (held_out_hashes - {fold["evaluationDatasetSha256"]}) <= components:
            raise ValueError("resolver cross-validation folds do not swap training datasets")

    authentic = [
        value
        for fold in folds
        for value in fold["trainingReport"]["seedTuningRmseBb"]
    ]
    resolver = [
        metric["resolverReachWeightedRmseBb"]
        for fold in folds
        for metric in fold["diagnostics"]
    ]
    validation = [
        value for fold in folds for value in fold["seedAuthenticValidationRmseBb"]
    ]
    maximum_authentic = max(authentic)
    maximum_validation = max(validation)
    maximum_resolver = max(resolver)
    reasons = []
    if maximum_authentic > maximum_authentic_tuning_rmse_bb:
        reasons.append(
            f"maximum authentic tuning RMSE {maximum_authentic:.6f}bb exceeds "
            f"{maximum_authentic_tuning_rmse_bb:.6f}bb"
        )
    if maximum_validation > maximum_authentic_tuning_rmse_bb:
        reasons.append(
            f"maximum authentic validation RMSE {maximum_validation:.6f}bb exceeds "
            f"{maximum_authentic_tuning_rmse_bb:.6f}bb"
        )
    if maximum_resolver > maximum_resolver_rmse_bb:
        reasons.append(
            f"maximum cross-fit resolver-reach RMSE {maximum_resolver:.6f}bb exceeds "
            f"{maximum_resolver_rmse_bb:.6f}bb"
        )
    return {
        "name": str(candidate["name"]),
        "configuration": configurations[0],
        "configurationSha256": value_selection.canonical_sha256(configurations[0]),
        "folds": folds,
        "trainingSeeds": sorted(all_seeds),
        "maximumAuthenticTuningRmseBb": maximum_authentic,
        "meanAuthenticTuningRmseBb": float(np.mean(authentic)),
        "maximumAuthenticValidationRmseBb": maximum_validation,
        "meanAuthenticValidationRmseBb": float(np.mean(validation)),
        "maximumResolverReachWeightedRmseBb": maximum_resolver,
        "meanResolverReachWeightedRmseBb": float(np.mean(resolver)),
        "status": "accepted" if not reasons else "rejected",
        "reasons": reasons,
    }


def select(spec_path: Path, repository_root: Path) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    spec_path = resolve_path(repository_root, str(spec_path)).resolve()
    spec = json.loads(spec_path.read_text())
    if spec.get("schema") != SPEC_SCHEMA:
        raise ValueError("resolver-reach selection spec has the wrong schema")
    gates = spec.get("gates", {})
    minimum_correlation = float(gates["minimumCrossSeedPredictionCorrelation"])
    maximum_authentic = float(gates["maximumAuthenticTuningRmseBb"])
    minimum_improvement = float(
        gates["minimumMaximumResolverReachRmseImprovementFraction"]
    )
    if not -1.0 <= minimum_correlation <= 1.0:
        raise ValueError("cross-seed correlation gate is invalid")
    if maximum_authentic <= 0.0 or not 0.0 < minimum_improvement < 1.0:
        raise ValueError("resolver-reach selection gates are invalid")

    baseline = baseline_summary(repository_root, spec.get("baseline", {}))
    baseline_fold_hashes = {
        fold["name"]: fold["evaluationDatasetSha256"] for fold in baseline["folds"]
    }
    maximum_resolver = baseline["maximumResolverReachWeightedRmseBb"] * (
        1.0 - minimum_improvement
    )
    candidates = [
        candidate_summary(
            repository_root,
            candidate,
            minimum_correlation,
            maximum_authentic,
            maximum_resolver,
            baseline_fold_hashes,
        )
        for candidate in spec.get("candidates", [])
    ]
    if not candidates or len({entry["name"] for entry in candidates}) != len(candidates):
        raise ValueError("resolver-reach selection requires uniquely named candidates")
    accepted = [entry for entry in candidates if entry["status"] == "accepted"]
    selected = (
        min(
            accepted,
            key=lambda entry: (
                entry["maximumResolverReachWeightedRmseBb"],
                entry["meanResolverReachWeightedRmseBb"],
                entry["maximumAuthenticValidationRmseBb"],
                entry["meanAuthenticValidationRmseBb"],
                entry["maximumAuthenticTuningRmseBb"],
                entry["meanAuthenticTuningRmseBb"],
                entry["configurationSha256"],
            ),
        )
        if accepted
        else None
    )
    return {
        "schema": OUTPUT_SCHEMA,
        "status": "frozen-for-fresh-evaluation" if selected else "rejected",
        "activationAllowed": False,
        "releaseHoldoutMetricsConsulted": False,
        "selectionCriterion": (
            "pass authentic tuning and baseline-relative resolver-reach gates; "
            "then minimize maximum and mean cross-fit resolver-reach RMSE, "
            "authentic validation RMSE, and authentic tuning RMSE"
        ),
        "spec": repository_relative_path(repository_root, spec_path),
        "specSha256": sha256_file(spec_path),
        "gates": {
            **gates,
            "maximumCrossFitResolverReachWeightedRmseBb": maximum_resolver,
        },
        "baseline": baseline,
        "candidates": candidates,
        "selectedCandidate": selected,
    }


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root or args.spec.resolve().parent.parent
    report = select(args.spec, repository_root)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(args.output)
    print(encoded, end="")


if __name__ == "__main__":
    main()
