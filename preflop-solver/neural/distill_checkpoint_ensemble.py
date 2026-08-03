#!/usr/bin/env python3
"""Distill a temporal checkpoint ensemble into paired single-network policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from train import (
    INPUT_FEATURE_COUNT,
    MAX_POLICY_ACTIONS,
    ActionScorer,
    make_compiled_policy_step,
    save_model,
    softmax,
)
from validate_seeds import (
    StreetRoutedModel,
    compare,
    evaluation_records,
    expand_state_action,
    load_run,
)


def parse_round_weights(source: str) -> list[tuple[int, float]]:
    values: list[tuple[int, float]] = []
    for item in source.split(","):
        round_text, separator, weight_text = item.partition(":")
        if not separator:
            raise ValueError("checkpoint weights must use round:weight entries")
        round_number = int(round_text)
        weight = float(weight_text)
        if round_number <= 0 or not np.isfinite(weight) or weight <= 0:
            raise ValueError("checkpoint rounds and weights must be positive")
        values.append((round_number, weight))
    total = sum(weight for _, weight in values)
    return [(round_number, weight / total) for round_number, weight in values]


class LogitEnsemble:
    def __init__(self, models: list[ActionScorer], weights: list[float]):
        self.models = models
        self.weights = weights

    def __call__(self, features: Any) -> Any:
        result = self.models[0](features) * self.weights[0]
        for model, weight in zip(self.models[1:], self.weights[1:]):
            result = result + model(features) * weight
        return result


def teacher_targets(
    models: list[ActionScorer], weights: list[float], features: np.ndarray
) -> np.ndarray:
    logits = sum(
        weight * np.asarray(model(mx.array(features))).reshape(-1)
        for model, weight in zip(models, weights)
    )
    return softmax(logits.astype(np.float64)).astype(np.float32)


def decision_dataset(
    records: list[dict[str, Any]],
    depth_bb: int,
    teacher_models: list[ActionScorer],
    weights: list[float],
    distill_street: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    dataset: list[tuple[np.ndarray, np.ndarray]] = []
    for record in records:
        is_preflop = record["state"]["street"] == "preflop"
        if is_preflop != (distill_street == "preflop"):
            continue
        features = np.stack(
            [
                expand_state_action(record["state"], action, depth_bb)
                for action in record["actions"]
            ]
        ).astype(np.float16)
        dataset.append((features, teacher_targets(teacher_models, weights, features)))
    if not dataset:
        raise RuntimeError(
            f"distillation corpus reached no {distill_street} decisions"
        )
    return dataset


def train_student(
    student: ActionScorer,
    dataset: list[tuple[np.ndarray, np.ndarray]],
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> list[float]:
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    optimizer.init(student.trainable_parameters())
    mx.eval(student.parameters(), optimizer.state)
    step = make_compiled_policy_step(student, optimizer)
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    for _ in range(steps):
        selected = rng.integers(0, len(dataset), size=batch_size)
        features = np.zeros(
            (batch_size, MAX_POLICY_ACTIONS, INPUT_FEATURE_COUNT), dtype=np.float32
        )
        targets = np.zeros((batch_size, MAX_POLICY_ACTIONS), dtype=np.float32)
        masks = np.zeros((batch_size, MAX_POLICY_ACTIONS), dtype=np.float32)
        for row, index in enumerate(selected):
            decision_features, decision_targets = dataset[int(index)]
            action_count = len(decision_targets)
            features[row, :action_count] = decision_features
            targets[row, :action_count] = decision_targets
            masks[row, :action_count] = 1.0
        loss = step(
            mx.array(features),
            mx.array(targets),
            mx.array(masks),
            mx.ones((batch_size, 1)),
        )
        mx.eval(loss, student.parameters(), optimizer.state)
        losses.append(float(loss.item()))
    return losses


def metrics(
    model_a: Any,
    model_b: Any,
    reach: list[list[dict[str, Any]]],
    forced: list[dict[str, Any]],
    depth_bb: int,
) -> dict[str, Any]:
    authentic = compare(
        model_a,
        model_b,
        reach,
        depth_bb,
        "held-out fixed authentic reach",
        True,
    )
    deviation = compare(
        model_a,
        model_b,
        [forced],
        depth_bb,
        "held-out forced reach",
        False,
    )
    return {
        "authentic": {
            "mae": authentic["action_frequency_mae"],
            "agreement": authentic["primary_action_agreement"],
            "maximum_aggregate_delta": authentic["maximum_aggregate_action_delta"],
        },
        "forced": {
            "mae": deviation["action_frequency_mae"],
            "agreement": deviation["primary_action_agreement"],
            "maximum_aggregate_delta": deviation["maximum_aggregate_action_delta"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("narrow_a", type=Path)
    parser.add_argument("narrow_b", type=Path)
    parser.add_argument("wide_a", type=Path)
    parser.add_argument("wide_b", type=Path)
    parser.add_argument(
        "--distill-street", choices=("preflop", "postflop"), default="postflop"
    )
    parser.add_argument("--narrow-round", type=int, default=250)
    parser.add_argument("--wide-round", type=int, default=100)
    parser.add_argument("--student-round", type=int)
    parser.add_argument("--narrow-weights-a", type=Path)
    parser.add_argument("--narrow-weights-b", type=Path)
    parser.add_argument("--wide-weights-a", type=Path)
    parser.add_argument("--wide-weights-b", type=Path)
    parser.add_argument("--checkpoint-weights", default="100:0.8,200:0.2")
    parser.add_argument("--training-trajectories", type=int, default=1000)
    parser.add_argument("--evaluation-trajectories", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--training-seed", type=int, default=3141592653)
    parser.add_argument("--evaluation-seed", type=int, default=2718281828)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(
        args.training_trajectories,
        args.evaluation_trajectories,
        args.steps,
        args.batch_size,
        args.narrow_round,
        args.wide_round,
    ) <= 0:
        raise ValueError("trajectory, step, and batch budgets must be positive")
    if args.student_round is not None and args.student_round <= 0:
        raise ValueError("student round must be positive")
    for first, second, label in (
        (args.narrow_weights_a, args.narrow_weights_b, "narrow"),
        (args.wide_weights_a, args.wide_weights_b, "wide"),
    ):
        if (first is None) != (second is None):
            raise ValueError(f"both {label} weight overrides are required")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("optimizer settings are invalid")
    round_weights = parse_round_weights(args.checkpoint_weights)
    root = Path(__file__).resolve().parents[2]
    narrow_dirs = [args.narrow_a.resolve(), args.narrow_b.resolve()]
    wide_dirs = [args.wide_a.resolve(), args.wide_b.resolve()]
    narrow_pairs = [load_run(path, args.narrow_round) for path in narrow_dirs]
    wide_pairs = [load_run(path, args.wide_round) for path in wide_dirs]
    narrow_models = [pair[1] for pair in narrow_pairs]
    wide_models = [pair[1] for pair in wide_pairs]
    if args.narrow_weights_a is not None:
        for model, path in zip(
            narrow_models, (args.narrow_weights_a, args.narrow_weights_b)
        ):
            model.load_weights(str(path.resolve()))
            mx.eval(model.parameters())
    if args.wide_weights_a is not None:
        for model, path in zip(wide_models, (args.wide_weights_a, args.wide_weights_b)):
            model.load_weights(str(path.resolve()))
            mx.eval(model.parameters())
    target_dirs = narrow_dirs if args.distill_street == "preflop" else wide_dirs
    protected_round = (
        args.narrow_round if args.distill_street == "preflop" else args.wide_round
    )
    student_round = args.student_round or protected_round
    teachers = [
        [load_run(path, round_number)[1] for round_number, _ in round_weights]
        for path in target_dirs
    ]
    weights = [weight for _, weight in round_weights]
    config = narrow_pairs[0][0]["config"]

    def records(traversals: int, seed: int, model_index: int | None = None):
        arguments = (
            root,
            20,
            traversals,
            seed,
            int(config["preflop_runout_samples"]),
            int(config["flop_runout_samples"]),
            not bool(config["exact_turn_rivers"]),
            bool(config["compact_serving_grid"]),
        )
        if model_index is None:
            return evaluation_records(*arguments)
        return evaluation_records(
            *arguments,
            policy_model=narrow_models[model_index],
            postflop_policy_model=wide_models[model_index],
        )

    training_forced = records(args.training_trajectories, args.training_seed)
    training_reach = [
        records(args.training_trajectories, args.training_seed, index) for index in range(2)
    ]
    students = [load_run(path, student_round)[1] for path in target_dirs]
    loss_summaries = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, student in enumerate(students):
        dataset = decision_dataset(
            training_reach[index] + training_forced,
            20,
            teachers[index],
            weights,
            args.distill_street,
        )
        losses = train_student(
            student,
            dataset,
            args.steps,
            args.batch_size,
            args.learning_rate,
            args.weight_decay,
            args.training_seed + index * 1009,
        )
        save_model(student, args.output_dir / f"seed-{index}.safetensors")
        loss_summaries.append(
            {
                "decisions": len(dataset),
                "initial_mean": float(np.mean(losses[: min(50, len(losses))])),
                "final_mean": float(np.mean(losses[-min(50, len(losses)) :])),
            }
        )

    heldout_forced = records(args.evaluation_trajectories, args.evaluation_seed)
    heldout_reach = [
        records(args.evaluation_trajectories, args.evaluation_seed, index)
        for index in range(2)
    ]
    baseline_routed = [
        StreetRoutedModel(narrow_models[index], wide_models[index]) for index in range(2)
    ]
    if args.distill_street == "preflop":
        student_routed = [
            StreetRoutedModel(students[index], wide_models[index]) for index in range(2)
        ]
        teacher_routed = [
            StreetRoutedModel(
                LogitEnsemble(teachers[index], weights), wide_models[index]
            )
            for index in range(2)
        ]
    else:
        student_routed = [
            StreetRoutedModel(narrow_models[index], students[index]) for index in range(2)
        ]
        teacher_routed = [
            StreetRoutedModel(
                narrow_models[index], LogitEnsemble(teachers[index], weights)
            )
            for index in range(2)
        ]
    report = {
        "schema": "hu-checkpoint-ensemble-distillation-v1",
        "distill_street": args.distill_street,
        "protected_rounds": {
            "preflop": args.narrow_round,
            "postflop": args.wide_round,
        },
        "protected_weight_overrides": {
            "preflop": [
                str(path.resolve())
                for path in (args.narrow_weights_a, args.narrow_weights_b)
                if path is not None
            ],
            "postflop": [
                str(path.resolve())
                for path in (args.wide_weights_a, args.wide_weights_b)
                if path is not None
            ],
        },
        "student_round": student_round,
        "checkpoint_weights": round_weights,
        "optimizer": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        "training": loss_summaries,
        "baseline": metrics(
            baseline_routed[0],
            baseline_routed[1],
            heldout_reach,
            heldout_forced,
            20,
        ),
        "teacher": metrics(
            teacher_routed[0],
            teacher_routed[1],
            heldout_reach,
            heldout_forced,
            20,
        ),
        "student": metrics(
            student_routed[0],
            student_routed[1],
            heldout_reach,
            heldout_forced,
            20,
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
