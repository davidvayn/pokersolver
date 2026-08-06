#!/usr/bin/env python3
"""Validate the immutable V49 cross-fit experiment plan and its inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import select_public_value_config as value_selection
import validate_resolver_reach_corpus as corpus_validation


SCHEMA = "hu-resolver-reach-crossfit-experiment-freeze-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--repository-root",
        type=Path,
        help="preflop-solver directory; defaults to the config's parent directory",
    )
    parser.add_argument("--require-completed-training-corpus", action="store_true")
    parser.add_argument("--output", type=Path)
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


def checked_path(repository_root: Path, entry: dict[str, Any]) -> Path:
    path = resolve_path(repository_root, str(entry["path"]))
    if not path.is_file() or sha256_file(path) != entry.get("sha256"):
        raise ValueError(f"experiment input is missing or has the wrong hash: {path}")
    return path


def validate_plan_structure(
    payload: dict[str, Any], corpus: dict[str, Any]
) -> dict[str, Any]:
    if payload.get("schema") != SCHEMA:
        raise ValueError("resolver-reach experiment plan has the wrong schema")
    if payload.get("activationAllowed") is not False:
        raise ValueError("cross-fit experiments cannot activate a model")
    if payload.get("releaseHoldoutMetricsConsulted") is not False:
        raise ValueError("cross-fit planning cannot consult a release holdout")

    expected_folds = {
        f"seed{int(shard['seed'])}": {
            "dataset": str(shard["output"]),
            "expectedStateCount": int(shard["expectedStateCount"]),
        }
        for shard in corpus.get("trainingShards", [])
    }
    folds = payload.get("resolverFolds", [])
    actual_folds = {str(fold["name"]): fold for fold in folds}
    if len(expected_folds) != 2 or set(actual_folds) != set(expected_folds):
        raise ValueError("experiment plan must use both frozen resolver training shards")
    for name, expected in expected_folds.items():
        fold = actual_folds[name]
        if (
            str(fold["dataset"]) != expected["dataset"]
            or int(fold["expectedStateCount"]) != expected["expectedStateCount"]
            or str(fold["oppositeFold"]) not in expected_folds
            or str(fold["oppositeFold"]) == name
        ):
            raise ValueError("resolver fold differs from the corpus freeze")
    if any(
        actual_folds[str(fold["oppositeFold"])]["oppositeFold"] != fold["name"]
        for fold in folds
    ):
        raise ValueError("resolver folds must be symmetric opposites")

    base_count = len(payload.get("baseSupplementalDatasets", []))
    if base_count == 0:
        raise ValueError("experiment plan requires base supplemental datasets")
    common = payload.get("commonTrainer", {})
    positive_fields = (
        "featureWorkers",
        "steps",
        "batchSize",
        "evaluationInterval",
        "learningRate",
        "earlyStoppingPatience",
        "huberDelta",
        "splitSeed",
    )
    if any(float(common.get(field, 0.0)) <= 0.0 for field in positive_fields):
        raise ValueError("common trainer parameters must be positive")
    if not 0.0 < float(common.get("validationFraction", 0.0)) < 1.0:
        raise ValueError("validation fraction is invalid")
    if not 0.0 < float(common.get("tuningFraction", 0.0)) < 1.0:
        raise ValueError("tuning fraction is invalid")
    if common.get("holdoutStartIndex") is not None:
        raise ValueError("opened predecessor data cannot remain a release holdout")

    candidates = payload.get("candidates", [])
    if len(candidates) < 2 or len({candidate["name"] for candidate in candidates}) != len(
        candidates
    ):
        raise ValueError("experiment plan requires uniquely named candidates")
    configuration_hashes: set[str] = set()
    training_seeds: set[int] = set()
    output_directories: set[str] = set()
    for candidate in candidates:
        primary_fraction = float(candidate["minimumPrimaryBatchFraction"])
        weights = [
            float(value) for value in candidate["supplementalDatasetSamplingWeights"]
        ]
        if not 0.0 <= primary_fraction <= 1.0:
            raise ValueError("minimum primary batch fraction is invalid")
        if len(weights) != base_count + 1 or any(
            not np.isfinite(value) or value <= 0.0 for value in weights
        ):
            raise ValueError("candidate supplemental weights are invalid")
        identity = {
            "commonTrainer": common,
            "minimumPrimaryBatchFraction": primary_fraction,
            "supplementalDatasetSamplingWeights": weights,
        }
        identity_hash = value_selection.canonical_sha256(identity)
        if identity_hash in configuration_hashes:
            raise ValueError("experiment candidates contain duplicate configurations")
        configuration_hashes.add(identity_hash)
        candidate_folds = candidate.get("folds", [])
        if {fold["trainingFold"] for fold in candidate_folds} != set(expected_folds):
            raise ValueError("candidate does not train once on each resolver fold")
        for fold in candidate_folds:
            training_fold = str(fold["trainingFold"])
            if str(fold["evaluationFold"]) != str(
                actual_folds[training_fold]["oppositeFold"]
            ):
                raise ValueError("candidate cross-fit training and evaluation are not opposite")
            seeds = [int(seed) for seed in fold.get("trainingSeeds", [])]
            if len(seeds) != 2 or len(set(seeds)) != 2 or training_seeds & set(seeds):
                raise ValueError("candidate folds require globally unique paired seeds")
            training_seeds |= set(seeds)
            output = str(fold["outputDirectory"])
            if output in output_directories:
                raise ValueError("candidate output directories must be unique")
            output_directories.add(output)

    gates = payload.get("selectionGates", {})
    correlation = float(gates.get("minimumCrossSeedPredictionCorrelation", -2.0))
    maximum_authentic = float(gates.get("maximumAuthenticTuningRmseBb", 0.0))
    minimum_improvement = float(
        gates.get("minimumMaximumResolverReachRmseImprovementFraction", 0.0)
    )
    if (
        not -1.0 <= correlation <= 1.0
        or maximum_authentic <= 0.0
        or not 0.0 < minimum_improvement < 1.0
    ):
        raise ValueError("experiment selection gates are invalid")

    release_seeds = [
        int(seed) for seed in payload.get("postSelection", {}).get("releaseTrainingSeeds", [])
    ]
    if len(release_seeds) != 2 or len(set(release_seeds)) != 2:
        raise ValueError("post-selection release training requires two seeds")
    corpus_seeds = {
        int(shard["seed"])
        for key in ("trainingShards", "reservedEvaluationShards")
        for shard in corpus.get(key, [])
    }
    source_seeds = {
        int(entry["trainingSeed"]) for entry in corpus.get("sourceValueNetworks", [])
    }
    if set(release_seeds) & (training_seeds | corpus_seeds | source_seeds):
        raise ValueError("release training seeds overlap prior experiment seeds")
    return {
        "candidateCount": len(candidates),
        "crossFitTrainingSeedCount": len(training_seeds),
        "configurationSha256": sorted(configuration_hashes),
        "releaseTrainingSeeds": release_seeds,
    }


def validate_experiment(
    config_path: Path,
    repository_root: Path,
    require_completed_training_corpus: bool = False,
) -> dict[str, Any]:
    payload = json.loads(config_path.read_text())
    corpus_entry = payload.get("corpusFreeze", {})
    corpus_path = checked_path(repository_root, corpus_entry)
    corpus = json.loads(corpus_path.read_text())
    structure = validate_plan_structure(payload, corpus)
    checked_path(repository_root, payload["primaryDataset"])
    for entry in payload.get("baseSupplementalDatasets", []):
        checked_path(repository_root, entry)

    corpus_report = corpus_validation.validate_config(corpus_path, repository_root)
    completed_training = len(corpus_report["completedShards"]["training"])
    completed_evaluation = len(corpus_report["completedShards"]["reservedEvaluation"])
    if completed_evaluation:
        raise ValueError("reserved resolver evaluation was generated before selection freeze")
    if require_completed_training_corpus and completed_training != len(
        corpus.get("trainingShards", [])
    ):
        raise ValueError("resolver training corpus is not complete")
    return {
        "schema": "hu-resolver-reach-crossfit-experiment-validation-v1",
        "status": "accepted",
        "config": str(config_path),
        "configSha256": sha256_file(config_path),
        "activationAllowed": False,
        "completedTrainingShards": completed_training,
        "completedReservedEvaluationShards": completed_evaluation,
        **structure,
    }


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root or args.config.resolve().parent.parent
    report = validate_experiment(
        args.config, repository_root, args.require_completed_training_corpus
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(encoded)
        temporary.replace(args.output)
    print(encoded, end="")


if __name__ == "__main__":
    main()
