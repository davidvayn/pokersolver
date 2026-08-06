#!/usr/bin/env python3
"""Freeze the V49 release/evaluation configuration after cross-fit selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import select_resolver_reach_value_config as resolver_selection


PROTOCOL_SCHEMA = "hu-resolver-reach-release-evaluation-protocol-v1"
RELEASE_SCHEMA = "hu-resolver-reach-release-freeze-v1"
EXPERIMENT_SCHEMA = "hu-resolver-reach-crossfit-experiment-freeze-v1"
CORPUS_SCHEMA = "hu-resolver-reach-corpus-freeze-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-protocol-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolved(repository_root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repository_root / candidate


def load_pinned(
    repository_root: Path, reference: dict[str, Any], expected_schema: str
) -> tuple[Path, dict[str, Any]]:
    path = resolved(repository_root, reference["path"])
    if sha256_file(path) != reference["sha256"]:
        raise ValueError(f"pinned file hash mismatch: {path}")
    payload = json.loads(path.read_text())
    if payload.get("schema") != expected_schema:
        raise ValueError(f"pinned file has the wrong schema: {path}")
    return path, payload


def experiment_seeds(experiment: dict[str, Any], corpus: dict[str, Any]) -> set[int]:
    seeds = {
        int(seed)
        for candidate in experiment["candidates"]
        for fold in candidate["folds"]
        for seed in fold["trainingSeeds"]
    }
    seeds.update(int(seed) for seed in experiment["postSelection"]["releaseTrainingSeeds"])
    for key in ("trainingShards", "reservedEvaluationShards"):
        seeds.update(int(shard["seed"]) for shard in corpus[key])
    seeds.update(int(source["trainingSeed"]) for source in corpus["sourceValueNetworks"])
    return seeds


def validate_protocol(
    protocol_path: Path, repository_root: Path, require_unopened: bool = True
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("release evaluation protocol has the wrong schema")
    if protocol.get("status") != "frozen-before-crossfit-completion":
        raise ValueError("release evaluation protocol was not frozen before selection")
    if protocol.get("activationAllowed") is not False:
        raise ValueError("release evaluation protocol must fail closed")
    _, experiment = load_pinned(
        repository_root, protocol["crossfitExperiment"], EXPERIMENT_SCHEMA
    )
    _, corpus = load_pinned(repository_root, protocol["resolverCorpus"], CORPUS_SCHEMA)
    if experiment["corpusFreeze"] != protocol["resolverCorpus"]:
        raise ValueError("experiment and release protocol pin different corpora")

    release = protocol["releaseTraining"]
    post_selection = experiment["postSelection"]
    if (
        not release.get("mergeBothResolverTrainingFolds")
        or not release.get("preserveSelectedResolverReplayMass")
        or float(release.get("perFoldResolverWeightMultiplier", -1.0)) != 0.5
        or release.get("trainingSeeds") != post_selection["releaseTrainingSeeds"]
        or int(release.get("steps", -1)) != int(post_selection["releaseTrainingSteps"])
    ):
        raise ValueError("release training controls do not match the frozen experiment")
    if protocol.get("releaseGates") != corpus.get("releasePolicy"):
        raise ValueError("release gates do not match the frozen resolver corpus")

    holdout = protocol["freshAuthenticHoldout"]
    source_path = resolved(repository_root, holdout["sourcePolicy"]["path"])
    if sha256_file(source_path) != holdout["sourcePolicy"]["sha256"]:
        raise ValueError("fresh authentic holdout source policy hash mismatch")
    generator = holdout["generator"]
    expected_controls = {
        "command": "turn-pbs-self-play-targets",
        "effectiveStackBb": 20.0,
        "statesPerShard": 64,
        "rangeParticles": 4096,
        "riverIterations": 200,
        "riverAveragingDelay": 20,
        "threads": 10,
        "beliefReplicates": 2,
        "explorationProbability": 0.0,
        "minimumPotBb": 0.0,
    }
    if generator != expected_controls or not holdout.get(
        "generateOnlyAfterReleaseConfigurationFreeze"
    ):
        raise ValueError("fresh authentic holdout controls are invalid")
    shards = holdout.get("shards", [])
    holdout_seeds = [int(shard["seed"]) for shard in shards]
    if len(shards) != 2 or len(set(holdout_seeds)) != 2:
        raise ValueError("fresh authentic holdout requires two unique shards")
    if set(holdout_seeds) & experiment_seeds(experiment, corpus):
        raise ValueError("fresh authentic holdout seeds overlap training or resolver seeds")
    outputs = [resolved(repository_root, shard["output"]) for shard in shards]
    checkpoints = [
        resolved(repository_root, shard["checkpointDirectory"]) for shard in shards
    ]
    if len(set(outputs)) != 2 or len(set(checkpoints)) != 2:
        raise ValueError("fresh authentic holdout paths must be unique")
    if require_unopened and any(path.exists() for path in outputs):
        raise ValueError("fresh authentic holdout was generated before release freeze")
    return protocol, experiment, corpus


def release_supplements(
    experiment: dict[str, Any], candidate: dict[str, Any]
) -> tuple[list[str], list[float]]:
    base_paths = [entry["path"] for entry in experiment["baseSupplementalDatasets"]]
    resolver_paths = [entry["dataset"] for entry in experiment["resolverFolds"]]
    selected_weights = [
        float(value) for value in candidate["supplementalDatasetSamplingWeights"]
    ]
    if len(selected_weights) != len(base_paths) + 1 or len(resolver_paths) != 2:
        raise ValueError("selected candidate has incompatible resolver replay weights")
    resolver_weight = selected_weights[-1] / 2.0
    return base_paths + resolver_paths, selected_weights[:-1] + [resolver_weight] * 2


def build_release_freeze(
    protocol_path: Path,
    selection_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    protocol, experiment, corpus = validate_protocol(
        protocol_path, repository_root, require_unopened=True
    )
    selection = json.loads(selection_path.read_text())
    if (
        selection.get("schema") != resolver_selection.OUTPUT_SCHEMA
        or selection.get("status") != "frozen-for-fresh-evaluation"
        or selection.get("activationAllowed") is not False
        or selection.get("releaseHoldoutMetricsConsulted") is not False
    ):
        raise ValueError("cross-fit selection is not eligible for release freezing")
    spec_path = resolved(repository_root, selection["spec"])
    if sha256_file(spec_path) != selection["specSha256"]:
        raise ValueError("cross-fit selector spec hash mismatch")
    recomputed = resolver_selection.select(spec_path, repository_root)
    if recomputed != selection:
        raise ValueError("cross-fit selection does not reproduce from frozen evidence")
    selected = selection["selectedCandidate"]
    candidates = {
        candidate["name"]: candidate for candidate in experiment["candidates"]
    }
    if selected["name"] not in candidates:
        raise ValueError("selected candidate is absent from the frozen experiment")
    candidate = candidates[selected["name"]]
    supplements, weights = release_supplements(experiment, candidate)
    supplement_sources = [
        {
            "path": path,
            "sha256": sha256_file(resolved(repository_root, path)),
        }
        for path in supplements
    ]
    release = protocol["releaseTraining"]
    return {
        "schema": RELEASE_SCHEMA,
        "modelVersion": protocol["modelVersion"],
        "status": "frozen-for-fresh-evaluation",
        "activationAllowed": False,
        "selection": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
            "selectedCandidate": selected["name"],
            "selectedConfigurationSha256": selected["configurationSha256"],
            "releaseHoldoutMetricsConsulted": False,
        },
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "primaryDataset": experiment["primaryDataset"],
        "supplementalDatasets": supplement_sources,
        "supplementalDatasetSamplingWeights": weights,
        "minimumPrimaryBatchFraction": candidate["minimumPrimaryBatchFraction"],
        "trainer": {
            **experiment["commonTrainer"],
            "steps": release["steps"],
            "trainingSeeds": release["trainingSeeds"],
            "outputDirectory": (
                f"{release['outputDirectoryRoot']}/{selected['name']}"
            ),
        },
        "reservedResolverEvaluation": corpus["reservedEvaluationShards"],
        "freshAuthenticHoldout": protocol["freshAuthenticHoldout"],
        "releaseGates": protocol["releaseGates"],
        "failurePolicy": protocol["failurePolicy"],
    }


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root or args.protocol.resolve().parent.parent
    if args.validate_protocol_only:
        protocol, experiment, corpus = validate_protocol(
            args.protocol, repository_root, require_unopened=True
        )
        print(
            json.dumps(
                {
                    "schema": PROTOCOL_SCHEMA,
                    "status": "accepted",
                    "activationAllowed": False,
                    "modelVersion": protocol["modelVersion"],
                    "candidateCount": len(experiment["candidates"]),
                    "reservedResolverEvaluationShards": len(
                        corpus["reservedEvaluationShards"]
                    ),
                    "freshAuthenticHoldoutSeeds": [
                        shard["seed"]
                        for shard in protocol["freshAuthenticHoldout"]["shards"]
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.selection is None or args.output is None:
        raise ValueError("--selection and --output are required to freeze a release")
    result = build_release_freeze(args.protocol, args.selection, repository_root)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(args.output)
    print(encoded, end="")


if __name__ == "__main__":
    main()
