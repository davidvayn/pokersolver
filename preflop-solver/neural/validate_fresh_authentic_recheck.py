#!/usr/bin/env python3
"""Validate the successor V49 authentic-value recheck fail-closed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import freeze_resolver_reach_release as release_freeze
import run_fresh_authentic_recheck as runner
import run_resolver_reach_release as old_runner
import validate_resolver_reach_release as old_validator


SCHEMA = "hu-fresh-authentic-value-recheck-validation-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def opened_fingerprints(
    protocol: dict[str, Any], repository_root: Path
) -> set[str]:
    release_path = runner.resolved(
        repository_root, protocol["predecessor"]["releaseFreeze"]["path"]
    )
    release = json.loads(release_path.read_text())
    references = [
        release["primaryDataset"],
        *release["supplementalDatasets"],
        *release["reservedResolverEvaluation"],
        *release["freshAuthenticHoldout"]["shards"],
    ]
    result: set[str] = set()
    for reference in references:
        path = runner.resolved(
            repository_root, reference.get("path", reference.get("output", ""))
        )
        result |= old_validator.dataset_fingerprints(path)
    return result


def summarize(
    protocol: dict[str, Any],
    protocol_path: Path,
    metrics: dict[int, dict[str, float]],
    correlation: float,
    unique_fingerprints: int,
    opened_count: int,
    artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    gates = {
        "freshAuthenticPerSeedRmse": len(metrics) == 2
        and all(
            values["weightedRmseBb"]
            <= float(protocol["gates"]["maximumPerSeedRmseBb"])
            for values in metrics.values()
        ),
        "freshAuthenticCrossSeedCorrelation": correlation
        >= float(protocol["gates"]["minimumCrossSeedPredictionCorrelation"]),
        "uniqueAndDisjointStateFingerprints": unique_fingerprints
        == int(protocol["generator"]["statesPerShard"]) * len(protocol["shards"]),
        "completeArtifactProvenance": len(artifacts)
        == len(protocol["shards"]) + len(protocol["shards"]) * len(protocol["models"]),
    }
    accepted = all(gates.values())
    return {
        "schema": SCHEMA,
        "status": (
            "accepted-awaiting-strategy-preflop-and-full-game-gates"
            if accepted
            else "rejected"
        ),
        "activationAllowed": False,
        "protocol": {
            "path": str(protocol_path),
            "sha256": release_freeze.sha256_file(protocol_path),
        },
        "perSeed": metrics,
        "crossSeedPredictionCorrelation": correlation,
        "uniqueStateFingerprints": unique_fingerprints,
        "openedStateFingerprintsExcluded": opened_count,
        "gates": gates,
        "artifacts": artifacts,
        "remainingRequiredGates": [
            "fresh range-consistent postflop response gate",
            "preflop continuation-cache and tabular DCFR regeneration",
            "full-game lookup coverage and probability validation",
            "reach-weighted action-EV standard-error coverage",
            "cross-seed action-frequency stability",
            "independent one-sided 99% full-game exploitability upper bound at or below 0.10bb/hand",
            "hosted-policy storage projection at or below 20GB",
        ],
    }


def validate(protocol_path: Path, repository_root: Path) -> dict[str, Any]:
    protocol = runner.validate_protocol(
        protocol_path, repository_root, require_unopened=False
    )
    plan = runner.build_plan(protocol, protocol_path)
    opened = opened_fingerprints(protocol, repository_root)
    holdout_paths: list[Path] = []
    fresh: set[str] = set()
    artifacts: list[dict[str, Any]] = []
    for shard in protocol["shards"]:
        path = runner.resolved(repository_root, shard["output"])
        payload = old_validator.validate_fresh_holdout_dataset(
            path,
            int(shard["seed"]),
            int(protocol["generator"]["statesPerShard"]),
            protocol["sourcePolicy"]["sha256"],
        )
        fingerprints = {target["input_sha256"] for target in payload["targets"]}
        if opened & fingerprints or fresh & fingerprints:
            raise ValueError("successor authentic holdout states are not fresh")
        fresh |= fingerprints
        holdout_paths.append(path)
        artifacts.append(
            {
                "kind": "holdout",
                "seed": int(shard["seed"]),
                "path": str(path),
                "sha256": release_freeze.sha256_file(path),
            }
        )

    reports: dict[int, list[dict[str, Any]]] = {
        int(model["seed"]): [] for model in protocol["models"]
    }
    for job in plan["diagnosticJobs"]:
        output = runner.resolved(repository_root, job["output"])
        dataset = runner.resolved(
            repository_root,
            old_runner.crossfit.option_values(job["command"], "--dataset")[0],
        )
        model = runner.resolved(
            repository_root,
            old_runner.crossfit.option_values(job["command"], "--model")[0],
        )
        old_runner.validate_authentic_diagnostic(
            output,
            release_freeze.sha256_file(dataset),
            int(job["modelSeed"]),
            release_freeze.sha256_file(model),
        )
        payload = json.loads(output.read_text())
        if int(payload.get("states", -1)) != int(
            protocol["generator"]["statesPerShard"]
        ):
            raise ValueError("successor authentic diagnostic has the wrong state count")
        reports[int(job["modelSeed"])].append(payload)
        artifacts.append(
            {
                "kind": "diagnostic",
                "datasetSeed": int(job["datasetSeed"]),
                "modelSeed": int(job["modelSeed"]),
                "path": str(output),
                "sha256": release_freeze.sha256_file(output),
            }
        )
    if any(len(values) != len(protocol["shards"]) for values in reports.values()):
        raise ValueError("successor authentic diagnostics are incomplete")
    metrics = {
        seed: old_validator.aggregate_error(values, resolver=False)
        for seed, values in reports.items()
    }
    model_paths = [
        runner.resolved(repository_root, model["path"])
        for model in protocol["models"]
    ]
    correlation = old_validator.cross_seed_prediction_correlation(
        holdout_paths, model_paths
    )
    return summarize(
        protocol,
        protocol_path,
        metrics,
        correlation,
        len(fresh),
        len(opened),
        artifacts,
    )


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root or args.protocol.resolve().parent.parent
    result = validate(args.protocol, repository_root)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(args.output)
    print(encoded, end="")


if __name__ == "__main__":
    main()
