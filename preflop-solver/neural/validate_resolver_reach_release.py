#!/usr/bin/env python3
"""Validate frozen V49 value-release evidence without activating a policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import freeze_resolver_reach_release as release_freeze
import run_resolver_reach_release as release_runner
import train_public_value_network as training
import validate_public_value_parity as parity
import validate_resolver_reach_corpus as corpus_validation


SCHEMA = "hu-resolver-reach-value-release-validation-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_freeze", type=Path)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def aggregate_error(reports: list[dict[str, Any]], resolver: bool) -> dict[str, float]:
    if not reports:
        raise ValueError("error aggregation requires at least one diagnostic")
    if resolver:
        entries = [report.get("resolverReachEvaluation") for report in reports]
        if any(entry is None for entry in entries):
            raise ValueError("resolver diagnostic lacks reach-weighted evidence")
        mass_key = "reachWeightMass"
        squared_key = "reachWeightedSquaredErrorBb2Sum"
        absolute_key = "reachWeightedAbsoluteErrorBbSum"
    else:
        entries = reports
        mass_key = "weightMass"
        squared_key = "weightedSquaredErrorBb2Sum"
        absolute_key = "weightedAbsoluteErrorBbSum"
    mass = sum(float(entry[mass_key]) for entry in entries)
    squared = sum(float(entry[squared_key]) for entry in entries)
    absolute = sum(float(entry[absolute_key]) for entry in entries)
    if not np.isfinite([mass, squared, absolute]).all() or mass <= 0.0:
        raise ValueError("diagnostic sufficient statistics are invalid")
    return {
        "weightMass": mass,
        "weightedSquaredErrorBb2Sum": squared,
        "weightedAbsoluteErrorBbSum": absolute,
        "weightedRmseBb": float(np.sqrt(squared / mass)),
        "weightedMaeBb": absolute / mass,
    }


def diagnostic_payload(
    job: dict[str, Any], repository_root: Path, expected_states: int
) -> dict[str, Any]:
    release_runner.validate_diagnostic_job(job, repository_root)
    payload = json.loads((repository_root / job["output"]).read_text())
    if int(payload.get("states", -1)) != expected_states:
        raise ValueError(f"diagnostic has the wrong state count: {job['output']}")
    return payload


def validate_fresh_holdout_dataset(
    path: Path,
    expected_seed: int,
    expected_states: int,
    source_policy_sha256: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if (
        payload.get("schema") != "hu-turn-public-belief-cfv-dataset-v2"
        or int(payload.get("seed", -1)) != expected_seed
        or len(payload.get("targets", [])) != expected_states
        or payload.get("source_policy_sha256") != source_policy_sha256
        or payload.get("validation", {}).get("status") != "accepted"
        or payload.get("state_distribution")
        != "frozen_v26_self_play_exact_reach_factor_public_beliefs"
    ):
        raise ValueError(f"fresh authentic holdout shard is invalid: {path}")
    fingerprints = [target.get("input_sha256") for target in payload["targets"]]
    if None in fingerprints or len(fingerprints) != len(set(fingerprints)):
        raise ValueError(f"fresh authentic holdout repeats a state: {path}")
    return payload


def dataset_fingerprints(path: Path) -> set[str]:
    payload = json.loads(path.read_text())
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError(f"release dataset has no fingerprinted targets: {path}")
    fingerprints = [target.get("input_sha256") for target in targets]
    if (
        any(not isinstance(value, str) or len(value) != 64 for value in fingerprints)
        or len(fingerprints) != len(set(fingerprints))
    ):
        raise ValueError(f"release dataset fingerprints are invalid or repeated: {path}")
    return set(fingerprints)


def cross_seed_prediction_correlation(
    dataset_paths: list[Path], model_paths: list[Path]
) -> float:
    if len(model_paths) != 2:
        raise ValueError("cross-seed correlation requires exactly two models")
    models = [json.loads(path.read_text()) for path in model_paths]
    if len({model.get("valueNormalization") for model in models}) != 1:
        raise ValueError("release models use different value normalization")
    predictions: list[list[np.ndarray]] = [[], []]
    masks: list[np.ndarray] = []
    for dataset_path in dataset_paths:
        dataset = training.load_dataset(
            dataset_path, 1, models[0].get("valueNormalization", "depth")
        )
        for index in range(len(dataset.boards)):
            masks.append(
                dataset.weights[index].reshape(2, training.COMBO_COUNT) > 0.0
            )
            for model_index, model in enumerate(models):
                predictions[model_index].append(
                    parity.python_prediction(dataset, model, index)
                )
    first = np.concatenate(
        [value[mask] for value, mask in zip(predictions[0], masks, strict=True)]
    )
    second = np.concatenate(
        [value[mask] for value, mask in zip(predictions[1], masks, strict=True)]
    )
    correlation = float(np.corrcoef(first, second)[0, 1])
    if not np.isfinite(correlation):
        raise ValueError("fresh cross-seed prediction correlation is invalid")
    return correlation


def validate(release_path: Path, repository_root: Path) -> dict[str, Any]:
    release = release_runner.validate_release_freeze(release_path, repository_root)
    protocol_path = release_freeze.resolved(repository_root, release["protocol"]["path"])
    _, _, corpus = release_freeze.validate_protocol(
        protocol_path, repository_root, require_unopened=False
    )
    plan = release_runner.build_plan(release, corpus)
    release_runner.validate_training_job(plan, repository_root)

    corpus_path = release_freeze.resolved(
        repository_root,
        json.loads(protocol_path.read_text())["resolverCorpus"]["path"],
    )
    corpus_report = corpus_validation.validate_config(corpus_path, repository_root)
    completed = corpus_report["completedShards"]
    if len(completed["training"]) != 2 or len(completed["reservedEvaluation"]) != 2:
        raise ValueError("resolver release validation requires every frozen shard")

    holdout = release["freshAuthenticHoldout"]
    opened_fingerprints: set[str] = set()
    opened_sources = [
        release["primaryDataset"],
        *release["supplementalDatasets"],
        *release["reservedResolverEvaluation"],
    ]
    for source in opened_sources:
        opened_fingerprints.update(
            dataset_fingerprints(repository_root / source.get("path", source.get("output")))
        )
    holdout_paths = []
    holdout_fingerprints: set[str] = set()
    for shard in holdout["shards"]:
        path = repository_root / shard["output"]
        payload = validate_fresh_holdout_dataset(
            path,
            int(shard["seed"]),
            int(holdout["generator"]["statesPerShard"]),
            holdout["sourcePolicy"]["sha256"],
        )
        fingerprints = {target["input_sha256"] for target in payload["targets"]}
        if opened_fingerprints & fingerprints:
            raise ValueError("fresh authentic holdout overlaps opened release evidence")
        if holdout_fingerprints & fingerprints:
            raise ValueError("fresh authentic holdout shards overlap")
        holdout_fingerprints.update(fingerprints)
        holdout_paths.append(path)

    models = plan["trainingJob"]["models"]
    model_paths = [repository_root / model["path"] for model in models]
    candidate_resolver: dict[int, list[dict[str, Any]]] = {
        int(model["seed"]): [] for model in models
    }
    candidate_authentic: dict[int, list[dict[str, Any]]] = {
        int(model["seed"]): [] for model in models
    }
    resolver_counts = {
        str(shard["output"]): int(shard["expectedStateCount"])
        for shard in release["reservedResolverEvaluation"]
    }
    authentic_count = int(holdout["generator"]["statesPerShard"])
    for job in plan["diagnosticJobs"]:
        command = job["command"]
        dataset = release_runner.crossfit.option_values(command, "--dataset")[0]
        model = release_runner.crossfit.option_values(command, "--model")[0]
        model_seed = next(
            int(entry["seed"]) for entry in models if entry["path"] == model
        )
        if dataset in resolver_counts:
            candidate_resolver[model_seed].append(
                diagnostic_payload(job, repository_root, resolver_counts[dataset])
            )
        else:
            candidate_authentic[model_seed].append(
                diagnostic_payload(job, repository_root, authentic_count)
            )
    if any(len(values) != 2 for values in candidate_resolver.values()) or any(
        len(values) != 2 for values in candidate_authentic.values()
    ):
        raise ValueError("release candidate diagnostics are incomplete")

    baseline_reports: dict[int, list[dict[str, Any]]] = {
        int(model["trainingSeed"]): [] for model in corpus["sourceValueNetworks"]
    }
    for job in plan["baselineResolverDiagnosticJobs"]:
        command = job["command"]
        dataset = release_runner.crossfit.option_values(command, "--dataset")[0]
        model = release_runner.crossfit.option_values(command, "--model")[0]
        seed = next(
            int(entry["trainingSeed"])
            for entry in corpus["sourceValueNetworks"]
            if entry["path"] == model
        )
        baseline_reports[seed].append(
            diagnostic_payload(job, repository_root, resolver_counts[dataset])
        )
    if any(len(values) != 2 for values in baseline_reports.values()):
        raise ValueError("release baseline diagnostics are incomplete")

    baseline_metrics = {
        seed: aggregate_error(reports, resolver=True)
        for seed, reports in baseline_reports.items()
    }
    strongest_baseline = min(
        metric["weightedRmseBb"] for metric in baseline_metrics.values()
    )
    resolver_metrics = {
        seed: aggregate_error(reports, resolver=True)
        for seed, reports in candidate_resolver.items()
    }
    authentic_metrics = {
        seed: aggregate_error(reports, resolver=False)
        for seed, reports in candidate_authentic.items()
    }
    resolver_improvements = {
        seed: (strongest_baseline - metric["weightedRmseBb"]) / strongest_baseline
        for seed, metric in resolver_metrics.items()
    }
    fresh_correlation = cross_seed_prediction_correlation(holdout_paths, model_paths)

    parity_evidence = []
    for job, model in zip(plan["parityJobs"], models, strict=True):
        report = json.loads((repository_root / job["output"]).read_text())
        model_path = repository_root / model["path"]
        dataset_path = repository_root / release["reservedResolverEvaluation"][0]["output"]
        valid = (
            report.get("validation", {}).get("status") == "accepted"
            and report.get("modelSha256") == release_freeze.sha256_file(model_path)
            and report.get("datasetSha256") == release_freeze.sha256_file(dataset_path)
            and float(report.get("maximumAbsoluteErrorBb", float("inf")))
            <= float(release["releaseGates"]["maximumRustPythonParityErrorBb"])
        )
        parity_evidence.append(
            {
                "seed": model["seed"],
                "maximumAbsoluteErrorBb": report.get("maximumAbsoluteErrorBb"),
                "valid": valid,
            }
        )

    gates = {
        "freshAuthenticPerSeedRmse": all(
            metric["weightedRmseBb"]
            <= float(
                release["releaseGates"]["maximumPerSeedAuthenticFreshHoldoutRmseBb"]
            )
            for metric in authentic_metrics.values()
        ),
        "freshAuthenticCrossSeedCorrelation": fresh_correlation
        >= float(release["releaseGates"]["minimumCrossSeedPredictionCorrelation"]),
        "reservedResolverPerSeedImprovement": all(
            improvement
            >= float(
                release["releaseGates"][
                    "minimumPerSeedResolverReachWeightedRmseImprovementFraction"
                ]
            )
            for improvement in resolver_improvements.values()
        ),
        "pythonRustParity": all(entry["valid"] for entry in parity_evidence),
    }
    value_passed = all(gates.values())
    return {
        "schema": SCHEMA,
        "status": (
            "accepted-awaiting-strategy-and-full-game-gates"
            if value_passed
            else "rejected"
        ),
        "activationAllowed": False,
        "releaseFreeze": {
            "path": str(release_path),
            "sha256": release_freeze.sha256_file(release_path),
        },
        "freshAuthentic": {
            "perSeed": authentic_metrics,
            "crossSeedPredictionCorrelation": fresh_correlation,
            "uniqueStateFingerprints": len(holdout_fingerprints),
            "openedStateFingerprintsExcluded": len(opened_fingerprints),
        },
        "reservedResolver": {
            "baselinePerSeed": baseline_metrics,
            "strongestBaselineRmseBb": strongest_baseline,
            "candidatePerSeed": resolver_metrics,
            "perSeedImprovementFraction": resolver_improvements,
        },
        "parity": parity_evidence,
        "gates": gates,
        "remainingRequiredGates": [
            "matched cross-fit continual-resolver local exploitability at or below 0.05bb/hand",
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
    result = validate(args.release_freeze, repository_root)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(args.output)
    print(encoded, end="")


if __name__ == "__main__":
    main()
