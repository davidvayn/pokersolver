#!/usr/bin/env python3
"""Train paired full-vector public-belief counterfactual-value networks."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

SCHEMA = "hu-turn-public-belief-value-network-pilot-v2"
COMBO_COUNT = 1326
DEPTH_BB = 20.0
RANGE_INPUT = COMBO_COUNT * 2
PUBLIC_INPUT = 52 + 2 + 2
OUTPUT_COUNT = COMBO_COUNT * 2


@dataclass
class Dataset:
    public: np.ndarray
    ranges: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    groups: np.ndarray
    source: dict[str, Any]
    source_sha256: str


class PublicValueNetwork(nn.Module):
    def __init__(self, use_ranges: bool):
        super().__init__()
        self.use_ranges = use_ranges
        self.public_tower = nn.Sequential(
            nn.Linear(PUBLIC_INPUT, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )
        self.range_tower = nn.Sequential(
            nn.Linear(RANGE_INPUT, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(192, 512), nn.ReLU(), nn.Linear(512, OUTPUT_COUNT)
        )

    def __call__(self, public: mx.array, ranges: mx.array) -> mx.array:
        if not self.use_ranges:
            ranges = mx.zeros_like(ranges)
        features = mx.concatenate(
            (self.public_tower(public), self.range_tower(ranges)), axis=1
        )
        return mx.tanh(self.head(features))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seeds", default="9601,9602")
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--maximum-rmse-bb", type=float, default=0.25)
    parser.add_argument("--minimum-range-relative-improvement", type=float, default=0.02)
    parser.add_argument("--minimum-cross-seed-correlation", type=float, default=0.95)
    parser.add_argument("--suit-augmentations", type=int, choices=(1, 24), default=24)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combo_cards(key: int) -> tuple[int, int]:
    high = 1
    while high * (high - 1) // 2 <= key:
        high += 1
    high -= 1
    return high, key - high * (high - 1) // 2


def suit_permutations(count: int) -> list[tuple[int, int, int, int]]:
    permutations = list(itertools.permutations(range(4)))
    return permutations if count == 24 else [permutations[0]]


def permute_card(card: int, permutation: tuple[int, int, int, int]) -> int:
    return (card >> 2) * 4 + permutation[card & 3]


def combo_permutation(permutation: tuple[int, int, int, int]) -> np.ndarray:
    mapping = np.empty(COMBO_COUNT, dtype=np.int32)
    for key in range(COMBO_COUNT):
        first, second = combo_cards(key)
        permuted_first = permute_card(first, permutation)
        permuted_second = permute_card(second, permutation)
        high, low = max(permuted_first, permuted_second), min(permuted_first, permuted_second)
        mapping[key] = high * (high - 1) // 2 + low
    if len(np.unique(mapping)) != COMBO_COUNT:
        raise AssertionError("suit permutation must be a combination bijection")
    return mapping


def load_dataset(path: Path, suit_augmentation_count: int = 1) -> Dataset:
    raw = json.loads(path.read_text())
    if raw.get("schema") != "hu-turn-public-belief-cfv-dataset-v1":
        raise ValueError("incompatible public-belief target dataset")
    public_rows: list[np.ndarray] = []
    range_rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    groups: list[int] = []
    permutations = suit_permutations(suit_augmentation_count)
    mappings = [combo_permutation(permutation) for permutation in permutations]
    for group, state in enumerate(raw["targets"]):
        original_board = [int(card) for card in state["board"]]
        ranges = np.asarray(state["ranges"], dtype=np.float32)
        values = np.asarray(state["counterfactual_values_bb"], dtype=np.float32)
        masses = np.asarray(state["opponent_compatible_mass"], dtype=np.float32)
        if ranges.shape != (2, COMBO_COUNT) or values.shape != ranges.shape:
            raise ValueError("public beliefs and values must use exact 1326-combo vectors")
        for permutation, mapping in zip(permutations, mappings):
            board = [permute_card(card, permutation) for card in original_board]
            board_features = np.zeros(52, dtype=np.float32)
            board_features[board] = 1.0
            actor = np.zeros(2, dtype=np.float32)
            actor[int(state["actor"])] = 1.0
            public_rows.append(
                np.concatenate(
                    (
                        board_features,
                        actor,
                        np.asarray(state["invested_bb"], dtype=np.float32) / DEPTH_BB,
                    )
                )
            )
            permuted_ranges = np.zeros_like(ranges)
            permuted_values = np.zeros_like(values)
            permuted_masses = np.zeros_like(masses)
            for player in range(2):
                permuted_ranges[player, mapping] = ranges[player]
                permuted_values[player, mapping] = values[player]
                permuted_masses[player, mapping] = masses[player]
                for combo in np.flatnonzero(permuted_ranges[player] > 0):
                    high, low = combo_cards(int(combo))
                    if high in board or low in board:
                        raise ValueError("target contains a board-blocked private combination")
            range_rows.append(permuted_ranges.reshape(-1) * COMBO_COUNT)
            targets.append(permuted_values.reshape(-1) / DEPTH_BB)
            weights.append((permuted_ranges * permuted_masses).reshape(-1))
            groups.append(group)
    weight_array = np.stack(weights)
    weight_array *= weight_array.size / max(float(weight_array.sum()), 1e-12)
    return Dataset(
        public=np.stack(public_rows),
        ranges=np.stack(range_rows),
        targets=np.stack(targets),
        weights=weight_array,
        groups=np.asarray(groups, dtype=np.int32),
        source=raw,
        source_sha256=sha256_file(path),
    )


def state_split(
    state_count: int, seed: int, validation_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    if state_count < 2:
        raise ValueError("paired evaluation requires at least two public-belief states")
    order = np.random.default_rng(seed ^ 0x51A7E).permutation(state_count)
    validation_count = min(
        state_count - 1, max(1, int(round(state_count * validation_fraction)))
    )
    return np.sort(order[validation_count:]), np.sort(order[:validation_count])


def weighted_metrics(
    truth: np.ndarray, prediction: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    normalized = weights / max(float(weights.sum()), 1e-12)
    error = prediction - truth
    return {
        "weightedRmseBb": float(
            np.sqrt(np.sum(normalized * error * error)) * DEPTH_BB
        ),
        "weightedMaeBb": float(np.sum(normalized * np.abs(error)) * DEPTH_BB),
        "correlation": float(
            np.corrcoef(truth[weights > 0].reshape(-1), prediction[weights > 0].reshape(-1))[0, 1]
        ),
    }


def train_one(
    dataset: Dataset,
    train_states: np.ndarray,
    validation_states: np.ndarray,
    use_ranges: bool,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[PublicValueNetwork, np.ndarray, dict[str, Any]]:
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = PublicValueNetwork(use_ranges)
    mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=1e-5)

    def loss_fn(
        current: PublicValueNetwork,
        public: mx.array,
        ranges: mx.array,
        targets: mx.array,
        weights: mx.array,
    ) -> mx.array:
        errors = current(public, ranges) - targets
        return mx.sum(weights * errors * errors) / mx.maximum(mx.sum(weights), 1e-8)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    for _ in range(steps):
        selected = rng.choice(
            train_states, size=min(batch_size, len(train_states)), replace=True
        )
        loss, gradients = loss_and_grad(
            model,
            mx.array(dataset.public[selected]),
            mx.array(dataset.ranges[selected]),
            mx.array(dataset.targets[selected]),
            mx.array(dataset.weights[selected]),
        )
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss)
    prediction = np.asarray(
        model(
            mx.array(dataset.public[validation_states]),
            mx.array(dataset.ranges[validation_states]),
        )
    )
    metrics = weighted_metrics(
        dataset.targets[validation_states],
        prediction,
        dataset.weights[validation_states],
    )
    return model, prediction, metrics


def layer_payload(layer: nn.Linear, activation: str) -> dict[str, Any]:
    weights = np.asarray(layer.weight)
    return {
        "inputSize": int(weights.shape[1]),
        "outputSize": int(weights.shape[0]),
        "activation": activation,
        "weights": weights.astype(np.float32).reshape(-1).tolist(),
        "biases": np.asarray(layer.bias).astype(np.float32).tolist(),
    }


def export_model(
    model: PublicValueNetwork,
    path: Path,
    seed: int,
    source_dataset_sha256: str,
    source_validation_status: str,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "hu-public-belief-value-network-v2",
                "seed": seed,
                "usesExactRanges": model.use_ranges,
                "targetScaleBb": DEPTH_BB,
                "rangeScale": COMBO_COUNT,
                "sourceDatasetSha256": source_dataset_sha256,
                "sourceValidationStatus": source_validation_status,
                "publicTower": [
                    layer_payload(model.public_tower.layers[0], "relu"),
                    layer_payload(model.public_tower.layers[2], "relu"),
                ],
                "rangeTower": [
                    layer_payload(model.range_tower.layers[0], "relu"),
                    layer_payload(model.range_tower.layers[2], "relu"),
                ],
                "head": [
                    layer_payload(model.head.layers[0], "relu"),
                    layer_payload(model.head.layers[2], "tanh"),
                ],
            },
            separators=(",", ":"),
        )
        + "\n"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset, args.suit_augmentations)
    seeds = [int(seed) for seed in args.seeds.split(",")]
    if len(seeds) < 2:
        raise ValueError("paired training requires at least two independent seeds")
    train_states, validation_states = state_split(
        len(dataset.source["targets"]), seeds[0], args.validation_fraction
    )
    train_rows = np.flatnonzero(np.isin(dataset.groups, train_states))
    validation_rows = np.flatnonzero(np.isin(dataset.groups, validation_states))
    variants: dict[str, list[dict[str, Any]]] = {"range": [], "noRange": []}
    predictions: dict[str, list[np.ndarray]] = {"range": [], "noRange": []}
    for variant, use_ranges in (("range", True), ("noRange", False)):
        for seed in seeds:
            model, prediction, metrics = train_one(
                dataset,
                train_rows,
                validation_rows,
                use_ranges,
                seed,
                args.steps,
                args.batch_size,
                args.learning_rate,
            )
            model_path = args.output_dir / f"turn-value-{variant}-seed{seed}.json"
            export_model(
                model,
                model_path,
                seed,
                dataset.source_sha256,
                dataset.source["validation"]["status"],
            )
            variants[variant].append(
                {"seed": seed, "metrics": metrics, "weights": model_path.name}
            )
            predictions[variant].append(prediction)
    cross_seed = {
        variant: float(
            np.corrcoef(values[0].reshape(-1), values[1].reshape(-1))[0, 1]
        )
        for variant, values in predictions.items()
    }
    range_rmse = float(
        np.mean([entry["metrics"]["weightedRmseBb"] for entry in variants["range"]])
    )
    no_range_rmse = float(
        np.mean([entry["metrics"]["weightedRmseBb"] for entry in variants["noRange"]])
    )
    relative_improvement = (no_range_rmse - range_rmse) / max(no_range_rmse, 1e-12)
    reasons: list[str] = []
    if dataset.source["validation"]["status"] != "accepted":
        reasons.append("source target corpus is not release-accepted")
    if range_rmse > args.maximum_rmse_bb:
        reasons.append(
            f"range-network mean holdout RMSE {range_rmse:.6f}bb exceeds {args.maximum_rmse_bb:.6f}bb"
        )
    if relative_improvement < args.minimum_range_relative_improvement:
        reasons.append(
            f"range input improves RMSE by {relative_improvement:.3%}, below {args.minimum_range_relative_improvement:.3%}"
        )
    if cross_seed["range"] < args.minimum_cross_seed_correlation:
        reasons.append(
            f"range-network cross-seed prediction correlation {cross_seed['range']:.6f} is below {args.minimum_cross_seed_correlation:.6f}"
        )
    report = {
        "schema": SCHEMA,
        "dataset": str(args.dataset),
        "datasetSha256": dataset.source_sha256,
        "states": int(len(dataset.source["targets"])),
        "augmentedStates": int(len(dataset.targets)),
        "suitAugmentationsPerState": args.suit_augmentations,
        "trainStates": train_states.tolist(),
        "validationStates": validation_states.tolist(),
        "steps": args.steps,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "variants": variants,
        "crossSeedPredictionCorrelation": cross_seed,
        "meanRangeRmseBb": range_rmse,
        "meanNoRangeRmseBb": no_range_rmse,
        "rangeRelativeImprovement": relative_improvement,
        "targetSamplingStandardErrorBb": 0.0,
        "validation": {"status": "accepted" if not reasons else "rejected", "reasons": reasons},
    }
    report_path = args.output_dir / "turn-value-paired-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
