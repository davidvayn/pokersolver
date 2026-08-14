#!/usr/bin/env python3
"""Fine-tune independent range policies toward a shared policy consensus.

The release holdouts are used only to identify and exclude their public states
from the bounded training cap.  Consensus targets are the equal probability
average of two independently trained served policies on the remaining
high-reach states.  Each student retains its own parent parameters and uses an
independent optimizer seed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from distill_range_policy import (
    RangeConditionedPolicy,
    add_features,
    batch,
    export_model_from_source,
    inspect_dataset,
    load_dataset,
    load_exported_model,
    sha256,
    split_rows,
    train,
)


def state_identity(record: dict[str, Any]) -> bytes:
    state = record.get("state", {})
    return json.dumps(
        {
            "board": state.get("board"),
            "history": state.get("public_history"),
            "actor": state.get("actor"),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def heldout_identities(path: Path) -> set[bytes]:
    identities: set[bytes] = set()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        metadata = json.loads(next(stream))
        for line in stream:
            if line.strip():
                identities.add(state_identity(json.loads(line)))
    if metadata.get("records") != len(identities) or not identities:
        raise ValueError(f"release holdout contains duplicate or invalid states: {path}")
    return identities


def training_cap(
    source: Path,
    release_holdout: Path,
    output: Path,
    capacity: int,
) -> Path:
    source_sha256 = sha256(source)
    holdout_sha256 = sha256(release_holdout)
    if output.exists():
        with gzip.open(output, "rt", encoding="utf-8") as stream:
            cached = json.loads(next(stream))
        if (
            cached.get("records") == capacity
            and cached.get("subset_of_sha256") == source_sha256
            and cached.get("excluded_release_holdout_sha256") == holdout_sha256
            and cached.get("subset")
            == "deterministic_authentic_reach_priority_excluding_release_holdout_v1"
        ):
            return output
    excluded = heldout_identities(release_holdout)
    metadata: dict[str, Any] | None = None
    candidates: list[tuple[tuple[float, bytes], int, str]] = []
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        metadata = json.loads(next(stream))
        for index, line in enumerate(stream):
            if not line.strip():
                continue
            record = json.loads(line)
            identity = state_identity(record)
            if identity in excluded:
                continue
            street = str(record.get("state", {}).get("street", ""))
            weight = float(record.get("weight", float("nan")))
            if street not in ("flop", "turn", "river") or not np.isfinite(weight) or weight <= 0:
                raise ValueError(f"invalid training row {index} in {source}")
            tie = hashlib.sha256(b"hu-range-consensus-cap-v1\0" + identity).digest()
            candidates.append(((-weight, tie), index, street))
    if metadata is None or capacity < 12 or len(candidates) <= capacity:
        raise ValueError("consensus cap must remove records and cover every street")

    selected: set[int] = set()
    for street in ("flop", "turn", "river"):
        street_rows = sorted(
            (candidate for candidate in candidates if candidate[2] == street),
            key=lambda candidate: candidate[0],
        )
        if len(street_rows) < 4:
            raise ValueError(f"training partition omits {street}")
        selected.update(candidate[1] for candidate in street_rows[:4])
    for _, index, _ in sorted(candidates, key=lambda candidate: candidate[0]):
        if len(selected) >= capacity:
            break
        selected.add(index)
    if len(selected) != capacity:
        raise RuntimeError("consensus cap did not select the requested record count")

    retained: list[dict[str, Any]] = []
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        next(stream)
        for index, line in enumerate(stream):
            if index in selected:
                retained.append(json.loads(line))
    if len(retained) != capacity:
        raise RuntimeError("consensus cap could not replay every selected record")

    capped_metadata = dict(metadata)
    capped_metadata.update(
        {
            "records": len(retained),
            "subset_of_sha256": source_sha256,
            "subset": "deterministic_authentic_reach_priority_excluding_release_holdout_v1",
            "excluded_release_holdout_sha256": holdout_sha256,
            "inverse_inclusion_weight_correction": False,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            compressed.write(
                (json.dumps(capped_metadata, separators=(",", ":")) + "\n").encode()
            )
            for record in retained:
                compressed.write(
                    (json.dumps(record, separators=(",", ":")) + "\n").encode()
                )
    temporary.replace(output)
    return output


def validation_cap(source: Path, output: Path, capacity: int) -> Path:
    """Create a deterministic research subset without changing corpus ancestry."""
    source_sha256 = sha256(source)
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        metadata = json.loads(next(stream))
        ancestor_sha256 = metadata.get("subset_of_sha256", source_sha256)
        candidates: list[tuple[tuple[float, bytes], int, str]] = []
        for index, line in enumerate(stream):
            if not line.strip():
                continue
            record = json.loads(line)
            street = str(record.get("state", {}).get("street", ""))
            weight = float(record.get("weight", float("nan")))
            if street not in ("flop", "turn", "river") or not np.isfinite(weight) or weight <= 0:
                raise ValueError(f"invalid validation row {index} in {source}")
            tie = hashlib.sha256(
                b"hu-range-consensus-validation-cap-v1\0" + state_identity(record)
            ).digest()
            candidates.append(((-weight, tie), index, street))
    if (
        capacity < 12
        or len(candidates) <= capacity
        or not isinstance(ancestor_sha256, str)
        or len(ancestor_sha256) != 64
    ):
        raise ValueError("validation cap must remove records and preserve corpus ancestry")

    selected: set[int] = set()
    for street in ("flop", "turn", "river"):
        street_rows = sorted(
            (candidate for candidate in candidates if candidate[2] == street),
            key=lambda candidate: candidate[0],
        )
        if len(street_rows) < 4:
            raise ValueError(f"validation partition omits {street}")
        selected.update(candidate[1] for candidate in street_rows[:4])
    for _, index, _ in sorted(candidates, key=lambda candidate: candidate[0]):
        if len(selected) >= capacity:
            break
        selected.add(index)
    if len(selected) != capacity:
        raise RuntimeError("validation cap did not select the requested record count")

    capped_metadata = dict(metadata)
    capped_metadata.update(
        {
            "records": capacity,
            "subset_of_sha256": ancestor_sha256,
            "validation_subset_of_sha256": source_sha256,
            "subset": "deterministic_authentic_reach_priority_research_validation_v1",
            "inverse_inclusion_weight_correction": False,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    written = 0
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            compressed.write(
                (json.dumps(capped_metadata, separators=(",", ":")) + "\n").encode()
            )
            with gzip.open(source, "rt", encoding="utf-8") as stream:
                next(stream)
                for index, line in enumerate(stream):
                    if index in selected:
                        compressed.write(line.encode())
                        written += 1
    if written != capacity:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("validation cap could not replay every selected record")
    temporary.replace(output)
    return output


def load_policy(path: Path) -> tuple[RangeConditionedPolicy, dict[str, Any]]:
    payload = json.loads(path.read_text())
    model = RangeConditionedPolicy(
        str(payload["architecture"]),
        str(payload.get("policyComposition", "replace")),
    )
    load_exported_model(model, path)
    return model, payload


def predictions(
    model: RangeConditionedPolicy,
    dataset: Any,
    batch_size: int,
) -> np.ndarray:
    result = np.zeros_like(dataset.targets)
    for start in range(0, len(dataset.records), batch_size):
        rows = np.arange(start, min(start + batch_size, len(dataset.records)))
        arguments = batch(dataset, rows)
        logits = model(*arguments[:6], arguments[-1])
        probabilities = np.asarray(mx.softmax(logits, axis=2), dtype=np.float32)
        result[rows] = probabilities
    reachable = dataset.combo_weights > 0
    if (
        not np.all(np.isfinite(result))
        or np.any(result < 0)
        or np.any(np.abs(result.sum(axis=2)[reachable] - 1.0) > 1e-5)
    ):
        raise RuntimeError("consensus source inference produced invalid probabilities")
    return result


def consensus_sha256(dataset: Any, targets: np.ndarray, parents: list[Path]) -> str:
    digest = hashlib.sha256(b"hu-range-policy-consensus-targets-v1\0")
    digest.update(dataset.sha256.encode())
    for parent in parents:
        digest.update(sha256(parent).encode())
    digest.update(np.asarray(targets, dtype=np.float32).tobytes())
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-a", type=Path, required=True)
    parser.add_argument("--network-b", type=Path, required=True)
    parser.add_argument("--dataset-a", type=Path, required=True)
    parser.add_argument("--dataset-b", type=Path, required=True)
    parser.add_argument("--heldout-a", type=Path, required=True)
    parser.add_argument("--heldout-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-records-per-teacher", type=int, default=256)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--final-learning-rate", type=float)
    parser.add_argument("--seeds", default="20901,20902")
    parser.add_argument("--feature-cache-dir", type=Path)
    parser.add_argument("--feature-workers", type=int, default=1)
    parser.add_argument("--maximum-gradient-norm", type=float, default=1.0)
    parser.add_argument("--pilot-heldout-records", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    if (
        len(seeds) != 2
        or seeds[0] == seeds[1]
        or args.maximum_records_per_teacher < 12
        or min(args.steps, args.batch_size, args.feature_workers) <= 0
        or (args.pilot_heldout_records != 0 and args.pilot_heldout_records < 12)
        or not np.isfinite(args.learning_rate)
        or args.learning_rate <= 0
        or (
            args.final_learning_rate is not None
            and not 0 < args.final_learning_rate <= args.learning_rate
        )
    ):
        raise ValueError("invalid consensus training controls")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parents = [args.network_a, args.network_b]
    if sha256(parents[0]) == sha256(parents[1]):
        raise ValueError("consensus training requires independent parent policies")

    capped = [
        training_cap(
            args.dataset_a,
            args.heldout_a,
            args.output_dir / "teacher-a-training-cap.jsonl.gz",
            args.maximum_records_per_teacher,
        ),
        training_cap(
            args.dataset_b,
            args.heldout_b,
            args.output_dir / "teacher-b-training-cap.jsonl.gz",
            args.maximum_records_per_teacher,
        ),
    ]
    pilot_heldouts = (
        [
            validation_cap(
                args.heldout_a,
                args.output_dir / "pilot-heldout-a.jsonl.gz",
                args.pilot_heldout_records,
            ),
            validation_cap(
                args.heldout_b,
                args.output_dir / "pilot-heldout-b.jsonl.gz",
                args.pilot_heldout_records,
            ),
        ]
        if args.pilot_heldout_records
        else []
    )
    maximum_actions = max(inspect_dataset(path)[1] for path in capped)
    datasets = [load_dataset(path, maximum_actions) for path in capped]
    for dataset in datasets:
        add_features(dataset, args.feature_cache_dir, args.feature_workers)

    parent_models = [load_policy(path)[0] for path in parents]
    consensus_hashes: list[str] = []
    for dataset in datasets:
        targets = 0.5 * (
            predictions(parent_models[0], dataset, args.batch_size)
            + predictions(parent_models[1], dataset, args.batch_size)
        )
        dataset.targets = targets.astype(np.float32)
        consensus_hashes.append(consensus_sha256(dataset, dataset.targets, parents))
    del parent_models

    splits = [split_rows(dataset) for dataset in datasets]
    students: list[dict[str, Any]] = []
    for index, (parent, seed) in enumerate(zip(parents, seeds, strict=True)):
        payload = json.loads(parent.read_text())
        model, losses, selection = train(
            datasets[0],
            datasets[1],
            splits[0][0],
            splits[1][0],
            splits[0][1],
            splits[1][1],
            seed,
            args.steps,
            args.batch_size,
            args.learning_rate,
            args.final_learning_rate,
            0.5,
            0.0,
            str(payload["architecture"]),
            str(payload.get("policyComposition", "replace")),
            1.0,
            0.0 + 1e-8,
            40.0,
            True,
            args.maximum_gradient_norm,
            initial_network_path=parent,
        )
        output = args.output_dir / f"range-policy-seed-{seed}.json"
        export_model_from_source(
            model,
            payload,
            output,
            seed,
            sha256(parent),
            consensus_hashes,
            provenance_key="consensusDatasetSha256s",
        )
        students.append(
            {
                "seed": seed,
                "parent": str(parent),
                "parentSha256": sha256(parent),
                "network": str(output),
                "networkSha256": sha256(output),
                "firstLoss": losses[0],
                "finalLoss": losses[-1],
                "selectedCheckpoint": selection,
            }
        )

    report = {
        "schema": "hu-paired-range-policy-consensus-distillation-v1",
        "method": "independent_parent_equal_probability_consensus_on_training_partition",
        "parents": [sha256(path) for path in parents],
        "releaseHoldouts": [sha256(args.heldout_a), sha256(args.heldout_b)],
        "pilotHeldouts": [sha256(path) for path in pilot_heldouts],
        "trainingCaps": [sha256(path) for path in capped],
        "consensusTargetSha256s": consensus_hashes,
        "controls": {
            "maximumRecordsPerTeacher": args.maximum_records_per_teacher,
            "steps": args.steps,
            "batchSize": args.batch_size,
            "learningRate": args.learning_rate,
            "finalLearningRate": args.final_learning_rate,
            "maximumGradientNorm": args.maximum_gradient_norm,
        },
        "students": students,
        "validation": {
            "status": "research_only",
            "reasons": [
                "exact Rust cross-seed and policy-value gates have not yet been run"
            ],
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
