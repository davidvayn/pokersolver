#!/usr/bin/env python3
"""Distill paired range-conditioned solver policies into postflop students."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from distill_tabular_preflop import batch_features, dataset_arrays, metrics, probabilities
from train import (
    INPUT_FEATURE_COUNT,
    ActionScorer,
    make_compiled_policy_step,
    save_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-a", type=Path, required=True)
    parser.add_argument("--dataset-b", type=Path, required=True)
    parser.add_argument("--initial-weights-a", type=Path, required=True)
    parser.add_argument("--initial-weights-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-sizes", default="512,256")
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--cross-seed-replay-probability", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=16_101)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corpus_groups(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    groups: list[int] = []
    street_counts: dict[str, int] = {"flop": 0, "turn": 0, "river": 0}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        metadata = json.loads(next(stream))
        teacher = metadata.get("teacher", {})
        if teacher.get("schema") != "hu-range-conditioned-postflop-action-teacher-v1":
            raise ValueError("dataset is not a range-conditioned postflop action teacher")
        if teacher.get("validation", {}).get("status") != "accepted_for_training":
            raise ValueError("postflop action teacher was not accepted for training")
        for line in stream:
            record = json.loads(line)
            street = record["state"]["street"]
            if street not in street_counts:
                raise ValueError("postflop corpus contains a preflop or unknown decision")
            street_counts[street] += 1
            # Hold out entire public boards instead of adjacent private-card
            # rows from the same information set.
            board = bytes(record["state"]["board"][:3])
            groups.append(int.from_bytes(hashlib.sha256(board).digest()[:8], "little"))
    return np.asarray(groups, dtype=np.uint64), street_counts


def split_indices(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    heldout = np.flatnonzero(groups % 10 == 0)
    training = np.flatnonzero(groups % 10 != 0)
    if len(heldout) == 0 or len(training) == 0:
        # One-root smoke corpora cannot hold out a board. Keep their split
        # deterministic while release corpora use the board-disjoint branch.
        heldout = np.arange(0, len(groups), 10)
        training = np.setdiff1d(np.arange(len(groups)), heldout, assume_unique=True)
    if len(heldout) == 0 or len(training) == 0:
        raise ValueError("postflop corpus is too small for train/heldout separation")
    return training, heldout


def subset(data: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[indices] for key, value in data.items()}


def aggregate_action_delta(
    model: ActionScorer,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
) -> float:
    selected = subset(data, indices)
    predicted = probabilities(model, selected)
    targets = selected["targets"]
    masks = selected["masks"]
    weights = selected["weights"].reshape(-1)
    normalized = weights / np.sum(weights)
    action_count = targets.shape[1]
    deltas = []
    for action in range(action_count):
        active = masks[:, action]
        denominator = np.sum(normalized * active)
        if denominator <= 0:
            continue
        deltas.append(
            abs(
                float(
                    np.sum(normalized * active * (predicted[:, action] - targets[:, action]))
                    / denominator
                )
            )
        )
    return max(deltas, default=0.0)


def fit(
    data: dict[str, np.ndarray],
    training: np.ndarray,
    auxiliary_data: dict[str, np.ndarray],
    auxiliary_training: np.ndarray,
    cross_seed_replay_probability: float,
    initial: Path,
    hidden: tuple[int, int],
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[ActionScorer, list[float]]:
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = ActionScorer(INPUT_FEATURE_COUNT, hidden)
    model.load_weights(str(initial))
    mx.eval(model.parameters())
    optimizer = optim.Adam(learning_rate=learning_rate)
    step = make_compiled_policy_step(model, optimizer)
    losses: list[float] = []
    for _ in range(steps):
        batch_data = data
        batch_training = training
        if (
            cross_seed_replay_probability > 0
            and rng.random() < cross_seed_replay_probability
        ):
            batch_data = auxiliary_data
            batch_training = auxiliary_training
        indices = rng.choice(
            batch_training,
            size=min(batch_size, len(batch_training)),
            replace=True,
        )
        loss = step(
            batch_features(batch_data, indices),
            mx.array(batch_data["targets"][indices]),
            mx.array(batch_data["masks"][indices]),
            mx.array(batch_data["weights"][indices]),
        )
        mx.eval(loss, model.parameters(), optimizer.state)
        losses.append(float(loss.item()))
    return model, losses


def evaluated_metrics(
    model: ActionScorer,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, float]:
    result = metrics(model, data, indices)
    result["maximumAggregateActionDelta"] = aggregate_action_delta(model, data, indices)
    return result


def main() -> None:
    args = parse_args()
    hidden_values = tuple(int(value) for value in args.hidden_sizes.split(","))
    if len(hidden_values) != 2 or min(hidden_values) <= 0:
        raise ValueError("--hidden-sizes requires two positive widths")
    if min(args.steps, args.batch_size) <= 0 or args.learning_rate <= 0:
        raise ValueError("distillation optimization settings must be positive")
    if not 0 <= args.cross_seed_replay_probability <= 1:
        raise ValueError("--cross-seed-replay-probability must be between zero and one")
    hidden = (hidden_values[0], hidden_values[1])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaded: list[tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray, np.ndarray]] = []
    descriptions: list[dict[str, Any]] = []
    for path in (args.dataset_a, args.dataset_b):
        metadata, data = dataset_arrays(path)
        groups, street_counts = corpus_groups(path)
        if len(groups) != len(data["states"]):
            raise ValueError("postflop group scan disagrees with dataset record count")
        training, heldout = split_indices(groups)
        loaded.append((metadata, data, training, heldout))
        descriptions.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "records": len(groups),
                "trainingRecords": len(training),
                "heldoutRecords": len(heldout),
                "streetRecords": street_counts,
                "teacher": metadata["teacher"],
            }
        )
    if loaded[0][0]["depth_bb"] != loaded[1][0]["depth_bb"]:
        raise ValueError("paired action teachers use different depths")

    reports: list[dict[str, Any]] = []
    models: list[ActionScorer] = []
    for index, ((_, data, training, heldout), initial) in enumerate(
        zip(loaded, (args.initial_weights_a, args.initial_weights_b))
    ):
        other = 1 - index
        _, other_data, other_training, other_heldout = loaded[other]
        baseline = {
            "training": None,
            "heldout": None,
            "otherSeedHeldout": None,
        }
        baseline_model = ActionScorer(INPUT_FEATURE_COUNT, hidden)
        baseline_model.load_weights(str(initial))
        mx.eval(baseline_model.parameters())
        baseline["training"] = evaluated_metrics(baseline_model, data, training)
        baseline["heldout"] = evaluated_metrics(baseline_model, data, heldout)
        baseline["otherSeedHeldout"] = evaluated_metrics(
            baseline_model, other_data, other_heldout
        )
        del baseline_model
        model, losses = fit(
            data,
            training,
            other_data,
            other_training,
            args.cross_seed_replay_probability,
            initial,
            hidden,
            args.steps,
            args.batch_size,
            args.learning_rate,
            args.seed + index,
        )
        output = args.output_dir / f"seed-{index}.safetensors"
        save_model(model, output)
        reports.append(
            {
                "student": index,
                "initialWeightsSha256": sha256(initial),
                "studentWeightsSha256": sha256(output),
                "studentBytes": output.stat().st_size,
                "firstLoss": losses[0],
                "finalLoss": losses[-1],
                "baseline": baseline,
                "teacherFit": {
                    "training": evaluated_metrics(model, data, training),
                    "heldout": evaluated_metrics(model, data, heldout),
                },
            }
        )
        models.append(model)

    for index, model in enumerate(models):
        other = 1 - index
        _, other_data, _, other_heldout = loaded[other]
        reports[index]["otherSeedHeldoutTeacherFit"] = evaluated_metrics(
            model, other_data, other_heldout
        )
    report = {
        "schema": "hu-range-conditioned-postflop-policy-distillation-v1",
        "depthBb": loaded[0][0]["depth_bb"],
        "hiddenSizes": list(hidden),
        "steps": args.steps,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "crossSeedReplayProbability": args.cross_seed_replay_probability,
        "seed": args.seed,
        "datasets": descriptions,
        "students": reports,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    del loaded, models
    gc.collect()


if __name__ == "__main__":
    main()
