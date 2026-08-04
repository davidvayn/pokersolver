#!/usr/bin/env python3
"""Distill paired tabular preflop average policies into compact MLX students."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from train import (
    ACTION_FEATURE_COUNT,
    INPUT_FEATURE_COUNT,
    MAX_POLICY_ACTIONS,
    STATE_FEATURE_COUNT,
    STATE_FEATURE_SCHEMA,
    ActionScorer,
    expand_action,
    expand_state,
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
    parser.add_argument("--hidden-sizes", default="256,128")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=7301)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_arrays(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        metadata = json.loads(next(stream))
        if metadata.get("schema") != "hu-neural-traversal-jsonl-v7":
            raise ValueError("tabular teacher has an incompatible record schema")
        if metadata.get("state_feature_count") != STATE_FEATURE_COUNT:
            raise ValueError("tabular teacher state feature count is incompatible")
        if metadata.get("state_feature_schema") != STATE_FEATURE_SCHEMA:
            raise ValueError("tabular teacher state feature schema is incompatible")
        if metadata.get("action_feature_count") != ACTION_FEATURE_COUNT:
            raise ValueError("tabular teacher action feature count is incompatible")
        count = int(metadata["records"])
        if count <= 0:
            raise ValueError("tabular teacher has no records")
    depth = float(metadata["depth_bb"])
    states = np.zeros((count, STATE_FEATURE_COUNT), dtype=np.float32)
    actions = np.zeros(
        (count, MAX_POLICY_ACTIONS, ACTION_FEATURE_COUNT), dtype=np.float32
    )
    targets = np.zeros((count, MAX_POLICY_ACTIONS), dtype=np.float32)
    masks = np.zeros((count, MAX_POLICY_ACTIONS), dtype=np.float32)
    weights = np.zeros((count, 1), dtype=np.float32)
    observed = 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        next(stream)
        for index, line in enumerate(stream):
            if index >= count:
                raise ValueError("tabular teacher contains more records than declared")
            record = json.loads(line)
            legal = record["actions"]
            probabilities = np.asarray(record["targets"], dtype=np.float32)
            if (
                not legal
                or len(legal) != len(probabilities)
                or len(legal) > MAX_POLICY_ACTIONS
            ):
                raise ValueError("tabular teacher record has an invalid action group")
            if np.any(probabilities < 0) or not np.isclose(
                np.sum(probabilities), 1.0
            ):
                raise ValueError("tabular teacher probabilities are invalid")
            states[index] = expand_state(record["state"], depth)
            actions[index, : len(legal)] = np.stack(
                [expand_action(record["state"], action, depth) for action in legal]
            )
            targets[index, : len(legal)] = probabilities
            masks[index, : len(legal)] = 1.0
            weights[index, 0] = max(float(record["weight"]), 1e-6)
            observed += 1
    if observed != count:
        raise ValueError("tabular teacher record count is invalid")
    return metadata, {
        "states": states,
        "actions": actions,
        "targets": targets,
        "masks": masks,
        "weights": weights,
    }


def batch_features(data: dict[str, np.ndarray], indices: np.ndarray) -> mx.array:
    states = data["states"][indices]
    actions = data["actions"][indices]
    expanded = np.broadcast_to(
        states[:, None, :],
        (len(indices), MAX_POLICY_ACTIONS, STATE_FEATURE_COUNT),
    )
    return mx.array(np.concatenate((expanded, actions), axis=2))


def probabilities(model: ActionScorer, data: dict[str, np.ndarray]) -> np.ndarray:
    output: list[np.ndarray] = []
    for start in range(0, len(data["states"]), 2048):
        indices = np.arange(start, min(start + 2048, len(data["states"])))
        logits = np.asarray(
            model(batch_features(data, indices).reshape((-1, INPUT_FEATURE_COUNT)))
        ).reshape((len(indices), MAX_POLICY_ACTIONS))
        masks = data["masks"][indices]
        logits = np.where(masks > 0, logits, -1e9)
        logits -= np.max(logits, axis=1, keepdims=True)
        exponentials = np.exp(logits) * masks
        output.append(exponentials / np.sum(exponentials, axis=1, keepdims=True))
    return np.concatenate(output)


def metrics(model: ActionScorer, data: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, float]:
    selected = {key: value[indices] for key, value in data.items()}
    predicted = probabilities(model, selected)
    targets = selected["targets"]
    masks = selected["masks"]
    weights = selected["weights"].reshape(-1)
    per_decision_mae = np.sum(np.abs(predicted - targets) * masks, axis=1) / np.sum(
        masks, axis=1
    )
    primary = np.argmax(predicted, axis=1) == np.argmax(targets, axis=1)
    target_probability = np.sum(targets * predicted, axis=1)
    normalized = weights / np.sum(weights)
    return {
        "weightedActionFrequencyMae": float(np.sum(normalized * per_decision_mae)),
        "weightedPrimaryAgreement": float(np.sum(normalized * primary)),
        "weightedTeacherProbability": float(np.sum(normalized * target_probability)),
    }


def train_one(
    data: dict[str, np.ndarray],
    initial: Path,
    hidden: tuple[int, int],
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[ActionScorer, list[float], dict[str, float]]:
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = ActionScorer(INPUT_FEATURE_COUNT, hidden)
    model.load_weights(str(initial))
    mx.eval(model.parameters())
    initial_teacher_fit = metrics(model, data, np.arange(len(data["states"])))
    optimizer = optim.Adam(learning_rate=learning_rate)
    step = make_compiled_policy_step(model, optimizer)
    training = np.arange(len(data["states"]))
    losses: list[float] = []
    for _ in range(steps):
        indices = rng.choice(training, size=min(batch_size, len(training)), replace=True)
        loss = step(
            batch_features(data, indices),
            mx.array(data["targets"][indices]),
            mx.array(data["masks"][indices]),
            mx.array(data["weights"][indices]),
        )
        mx.eval(loss, model.parameters(), optimizer.state)
        losses.append(float(loss.item()))
    return model, losses, initial_teacher_fit


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise ValueError("distillation optimization settings must be positive")
    hidden_values = tuple(int(value) for value in args.hidden_sizes.split(","))
    if len(hidden_values) != 2 or min(hidden_values) <= 0:
        raise ValueError("--hidden-sizes requires two positive widths")
    hidden = (hidden_values[0], hidden_values[1])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    depth_bb: float | None = None
    for index, (dataset_path, initial) in enumerate(
        zip(
            (args.dataset_a, args.dataset_b),
            (args.initial_weights_a, args.initial_weights_b),
        )
    ):
        metadata, data = dataset_arrays(dataset_path)
        current_depth = float(metadata["depth_bb"])
        if depth_bb is None:
            depth_bb = current_depth
        elif depth_bb != current_depth:
            raise ValueError("paired tabular teachers use different depths")
        model, losses, initial_teacher_fit = train_one(
            data,
            initial.resolve(),
            hidden,
            args.steps,
            args.batch_size,
            args.learning_rate,
            args.seed + index,
        )
        output = args.output_dir / f"seed-{index}.safetensors"
        save_model(model, output)
        all_records = np.arange(len(data["states"]))
        reports.append(
            {
                "student": index,
                "dataset": str(dataset_path.resolve()),
                "datasetSha256": sha256(dataset_path),
                "teacher": metadata.get("teacher"),
                "records": len(data["states"]),
                "initialWeightsSha256": sha256(initial),
                "studentWeightsSha256": sha256(output),
                "studentBytes": output.stat().st_size,
                "firstLoss": losses[0],
                "finalLoss": losses[-1],
                "initialTeacherFit": initial_teacher_fit,
                "teacherFit": metrics(model, data, all_records),
            }
        )
        del data, model
        gc.collect()
    report = {
        "schema": "hu-tabular-preflop-distillation-v1",
        "depthBb": depth_bb,
        "hiddenSizes": list(hidden),
        "steps": args.steps,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "seed": args.seed,
        "students": reports,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
