#!/usr/bin/env python3
"""Train a leakage-safe range-conditioned flop continuation-value pilot.

The continuation caches contain Monte-Carlo values for exact deals at every
reachable flop leaf.  This pilot predicts those values from information that
is observable to one player: their hole cards, the flop, the public preflop
line, and Bayesian hand-class ranges induced by a frozen paired preflop policy.
It never exposes the opponent's cards or the cached turn/river to the model.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from train import TEXTURE_FEATURE_COUNT, texture_features


SCHEMA = "hu-range-conditioned-flop-value-oracle-pilot-v1"
RANKS = "23456789TJQKA"
DEPTH_BB = 20.0
CARD_FEATURES = 104
HAND_CLASS_FEATURES = 169
ACTOR_FEATURES = 2
SCALAR_FEATURES = 4
RANGE_FEATURES = 338
HUBER_DELTA = 0.05  # one big blind after targets are depth-normalized


def hand_classes() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    labels: list[str] = []
    high: list[int] = []
    low: list[int] = []
    kinds: list[int] = []  # 0 pair, 1 suited, 2 offsuit
    for high_rank in range(13):
        labels.append(RANKS[high_rank] * 2)
        high.append(high_rank)
        low.append(high_rank)
        kinds.append(0)
        for low_rank in range(high_rank):
            for suffix, kind in (("s", 1), ("o", 2)):
                labels.append(RANKS[high_rank] + RANKS[low_rank] + suffix)
                high.append(high_rank)
                low.append(low_rank)
                kinds.append(kind)
    if len(labels) != 169 or len(set(labels)) != 169:
        raise AssertionError("hold'em class enumeration must contain 169 classes")
    return (
        labels,
        np.asarray(high, dtype=np.int16),
        np.asarray(low, dtype=np.int16),
        np.asarray(kinds, dtype=np.int8),
    )


CLASS_LABELS, CLASS_HIGH, CLASS_LOW, CLASS_KIND = hand_classes()
CLASS_INDEX = {label: index for index, label in enumerate(CLASS_LABELS)}


@dataclass
class CacheArrays:
    holes: np.ndarray
    flops: np.ndarray
    targets: np.ndarray
    standard_errors: np.ndarray
    history_keys: list[str]
    histories: list[list[str]]
    meaningful_histories: np.ndarray
    history_scalars: np.ndarray
    source_sha256: str

    @property
    def deals(self) -> int:
        return int(self.holes.shape[0])

    @property
    def history_count(self) -> int:
        return len(self.histories)


@dataclass(frozen=True)
class SampleSelection:
    deals: np.ndarray
    histories: np.ndarray
    actors: np.ndarray


@dataclass
class GroupedTargets:
    bucket_ids: np.ndarray
    means: np.ndarray
    standard_errors: np.ndarray
    counts: np.ndarray
    bucket_count: int


class ValueOracle(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int, int],
        architecture: str = "flat",
        bounded_output: bool = False,
    ):
        super().__init__()
        self.architecture = architecture
        self.bounded_output = bounded_output
        if architecture == "flat":
            self.layers = nn.Sequential(
                nn.Linear(input_size, hidden_sizes[0]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[1], 1),
            )
        elif architecture == "range_tower":
            state_size = input_size - RANGE_FEATURES
            range_hidden = max(hidden_sizes[1], 64)
            self.state_encoder = nn.Sequential(
                nn.Linear(state_size, hidden_sizes[0]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                nn.ReLU(),
            )
            self.range_encoder = nn.Sequential(
                nn.Linear(RANGE_FEATURES, range_hidden),
                nn.ReLU(),
                nn.Linear(range_hidden, range_hidden),
                nn.ReLU(),
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_sizes[1] + range_hidden, hidden_sizes[1]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[1], 1),
            )
        else:
            raise ValueError(f"unsupported value-oracle architecture: {architecture}")

    def __call__(self, inputs: mx.array) -> mx.array:
        if self.architecture == "flat":
            values = self.layers(inputs)
        else:
            state = self.state_encoder(inputs[:, :-RANGE_FEATURES])
            ranges = self.range_encoder(inputs[:, -RANGE_FEATURES:])
            values = self.head(mx.concatenate((state, ranges), axis=1))
        return mx.tanh(values) if self.bounded_output else values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache-a", type=Path, required=True)
    parser.add_argument("--train-cache-b", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--holdout-cache", type=Path)
    parser.add_argument("--policy-a", type=Path, required=True)
    parser.add_argument("--policy-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4_000)
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--evaluation-samples", type=int, default=200_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-sizes", default="256,128")
    parser.add_argument("--seeds", default="9401,9402")
    parser.add_argument("--architecture", choices=("flat", "range_tower"), default="flat")
    parser.add_argument("--loss", choices=("huber", "mse"), default="huber")
    parser.add_argument("--bounded-output", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--target-mode",
        choices=("raw_rollout", "texture_group_mean"),
        default="raw_rollout",
    )
    parser.add_argument(
        "--target-bucketing",
        choices=("texture", "texture_hand_class"),
        default="texture_hand_class",
    )
    parser.add_argument("--load-weights-dir", type=Path)
    parser.add_argument("--minimum-range-relative-improvement", type=float, default=0.01)
    parser.add_argument(
        "--minimum-cross-seed-prediction-correlation", type=float, default=0.95
    )
    # Keep reports strict-JSON serializable even when this research gate is not
    # intentionally constraining a pilot.
    parser.add_argument("--maximum-grouped-rmse-bb", type=float, default=1e9)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_history_state(history: list[str]) -> tuple[bool, np.ndarray]:
    invested = [0.5, 1.0]
    action_count = 0
    for token in history[1:]:
        if token == "deal:Flop":
            continue
        _, player_token, label = token.split(":", 2)
        actor = int(player_token[1:])
        opponent = 1 - actor
        action_count += 1
        if label == "limp":
            invested[actor] = 1.0
        elif label in ("call", "call_all_in"):
            invested[actor] = min(DEPTH_BB, invested[opponent])
        elif label == "check":
            pass
        elif "_to_" in label:
            amount = label.rsplit("_to_", 1)[1].removesuffix("bb")
            invested[actor] = min(DEPTH_BB, float(amount))
        else:
            raise ValueError(f"unsupported public action in continuation cache: {label}")
    remaining = [DEPTH_BB - amount for amount in invested]
    meaningful = min(remaining) > 0.5
    scalars = np.asarray(
        [sum(invested) / (2 * DEPTH_BB), remaining[0] / DEPTH_BB,
         remaining[1] / DEPTH_BB, action_count / 6.0],
        dtype=np.float32,
    )
    return meaningful, scalars


def load_cache(path: Path) -> CacheArrays:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        raw = json.load(stream)
    if raw.get("schema") != "hu-preflop-continuation-cache-v1":
        raise ValueError(f"incompatible continuation cache: {path}")
    if float(raw.get("depth_bb", 0.0)) != DEPTH_BB:
        raise ValueError("range-value pilot currently accepts only 20bb caches")
    history_keys = list(raw["public_histories"])
    histories = [raw["public_histories"][key] for key in history_keys]
    meaningful_and_scalars = [parse_history_state(history) for history in histories]
    meaningful = np.asarray([value[0] for value in meaningful_and_scalars], dtype=bool)
    scalars = np.stack([value[1] for value in meaningful_and_scalars])
    deals = raw["deals"]
    holes = np.empty((len(deals), 2, 2), dtype=np.uint8)
    flops = np.empty((len(deals), 3), dtype=np.uint8)
    targets = np.empty((len(deals), len(histories)), dtype=np.float32)
    standard_errors = np.empty_like(targets)
    for deal_index, deal in enumerate(deals):
        holes[deal_index] = deal["holes"]
        flops[deal_index] = deal["board"][:3]
        for history_index, key in enumerate(history_keys):
            estimate = deal["continuations"][key]
            targets[deal_index, history_index] = estimate["mean_utility_p0_bb"]
            standard_errors[deal_index, history_index] = estimate[
                "action_standard_error_bb"
            ]
    del raw, deals
    gc.collect()
    if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(standard_errors)):
        raise ValueError("continuation cache contains non-finite targets")
    return CacheArrays(
        holes=holes,
        flops=flops,
        targets=targets,
        standard_errors=standard_errors,
        history_keys=history_keys,
        histories=histories,
        meaningful_histories=meaningful,
        history_scalars=scalars,
        source_sha256=sha256_file(path),
    )


def concatenate_caches(first: CacheArrays, second: CacheArrays) -> CacheArrays:
    if first.history_keys != second.history_keys or first.histories != second.histories:
        raise ValueError("training caches do not share identical public flop leaves")
    return CacheArrays(
        holes=np.concatenate((first.holes, second.holes)),
        flops=np.concatenate((first.flops, second.flops)),
        targets=np.concatenate((first.targets, second.targets)),
        standard_errors=np.concatenate((first.standard_errors, second.standard_errors)),
        history_keys=first.history_keys,
        histories=first.histories,
        meaningful_histories=first.meaningful_histories,
        history_scalars=first.history_scalars,
        source_sha256=hashlib.sha256(
            (first.source_sha256 + second.source_sha256).encode("ascii")
        ).hexdigest(),
    )


def averaged_policy(paths: Iterable[Path]) -> tuple[dict[str, dict[str, float]], list[str]]:
    artifacts = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(artifact.get("schema") != "hu-tabular-preflop-dcfr-v1" for artifact in artifacts):
        raise ValueError("range inference requires tabular DCFR policy artifacts")
    per_policy: list[dict[str, dict[str, float]]] = []
    for artifact in artifacts:
        lookup: dict[str, dict[str, float]] = {}
        for entry in artifact["strategies"]:
            lookup[entry["key"]] = dict(zip(entry["action_labels"], entry["probabilities"]))
        per_policy.append(lookup)
    common = set(per_policy[0])
    for lookup in per_policy[1:]:
        common &= set(lookup)
    if len(common) != len(per_policy[0]) or any(len(value) != len(common) for value in per_policy):
        raise ValueError("paired policies do not have identical information-set coverage")
    average: dict[str, dict[str, float]] = {}
    for key in sorted(common):
        labels = set(per_policy[0][key])
        if any(set(lookup[key]) != labels for lookup in per_policy[1:]):
            raise ValueError(f"paired policy action mismatch at {key}")
        average[key] = {
            label: sum(lookup[key][label] for lookup in per_policy) / len(per_policy)
            for label in labels
        }
    hashes = [sha256_file(path) for path in paths]
    return average, hashes


def range_likelihoods(
    histories: list[list[str]], policy: dict[str, dict[str, float]]
) -> np.ndarray:
    likelihoods = np.ones((len(histories), 2, 169), dtype=np.float64)
    for history_index, history in enumerate(histories):
        prefix = [history[0]]
        for token in history[1:]:
            if token == "deal:Flop":
                continue
            _, player_token, action = token.split(":", 2)
            actor = int(player_token[1:])
            for class_index, label in enumerate(CLASS_LABELS):
                key = f"p{actor}|{label}|{'/'.join(prefix)}"
                try:
                    likelihoods[history_index, actor, class_index] *= policy[key][action]
                except KeyError as error:
                    raise ValueError(
                        f"policy cannot replay public line at {key}: {action}"
                    ) from error
            prefix.append(token)
    return likelihoods


def exact_class_indices(holes: np.ndarray) -> np.ndarray:
    first_rank = holes[..., 0] // 4
    second_rank = holes[..., 1] // 4
    high = np.maximum(first_rank, second_rank)
    low = np.minimum(first_rank, second_rank)
    suited = holes[..., 0] % 4 == holes[..., 1] % 4
    labels = np.empty(high.shape, dtype=object)
    for index in np.ndindex(high.shape):
        if high[index] == low[index]:
            label = RANKS[int(high[index])] * 2
        else:
            label = RANKS[int(high[index])] + RANKS[int(low[index])] + (
                "s" if suited[index] else "o"
            )
        labels[index] = CLASS_INDEX[label]
    return labels.astype(np.int16)


def build_grouped_targets(
    cache: CacheArrays, include_hand_class: bool = True
) -> GroupedTargets:
    classes = exact_class_indices(cache.holes)
    textures = np.empty(
        (cache.deals, 2, TEXTURE_FEATURE_COUNT), dtype=np.float32
    )
    for deal_index in range(cache.deals):
        flop = cache.flops[deal_index].tolist()
        for actor in range(2):
            textures[deal_index, actor] = texture_features(
                cache.holes[deal_index, actor].tolist(), flop, "flop"
            )
    texture_signatures = np.rint(
        textures.reshape((-1, TEXTURE_FEATURE_COUNT)) * 100.0
    ).astype(np.int16)
    signatures = (
        np.concatenate(
            (classes.reshape((-1, 1)).astype(np.int16), texture_signatures),
            axis=1,
        )
        if include_hand_class
        else texture_signatures
    )
    _, inverse = np.unique(signatures, axis=0, return_inverse=True)
    bucket_ids = inverse.reshape((cache.deals, 2)).astype(np.int32)
    bucket_count = int(inverse.max()) + 1
    means = np.zeros((2, cache.history_count, bucket_count), dtype=np.float32)
    standard_errors = np.full_like(means, DEPTH_BB)
    counts = np.zeros((2, bucket_count), dtype=np.int32)
    for actor in range(2):
        actor_buckets = bucket_ids[:, actor]
        actor_counts = np.bincount(actor_buckets, minlength=bucket_count)
        counts[actor] = actor_counts
        observed = actor_counts > 0
        for history in range(cache.history_count):
            raw = cache.targets[:, history].astype(np.float64)
            values = raw if actor == 0 else -raw
            sums = np.bincount(
                actor_buckets, weights=values, minlength=bucket_count
            )
            squared = np.bincount(
                actor_buckets, weights=values**2, minlength=bucket_count
            )
            means[actor, history, observed] = (
                sums[observed] / actor_counts[observed]
            ).astype(np.float32)
            repeated = actor_counts > 1
            variance = np.zeros(bucket_count, dtype=np.float64)
            variance[repeated] = np.maximum(
                (
                    squared[repeated]
                    - sums[repeated] ** 2 / actor_counts[repeated]
                )
                / (actor_counts[repeated] - 1),
                0.0,
            )
            standard_errors[actor, history, repeated] = np.sqrt(
                variance[repeated] / actor_counts[repeated]
            ).astype(np.float32)
    return GroupedTargets(
        bucket_ids=bucket_ids,
        means=means,
        standard_errors=standard_errors,
        counts=counts,
        bucket_count=bucket_count,
    )


def compatible_class_counts(blocked_cards: np.ndarray) -> np.ndarray:
    blocked_cards = np.asarray(blocked_cards, dtype=np.uint8)
    available = np.ones((len(blocked_cards), 13, 4), dtype=np.float32)
    rows = np.arange(len(blocked_cards))[:, None]
    available[rows, blocked_cards // 4, blocked_cards % 4] = 0.0
    high_available = available[:, CLASS_HIGH, :]
    low_available = available[:, CLASS_LOW, :]
    high_count = np.sum(high_available, axis=2)
    low_count = np.sum(low_available, axis=2)
    suited_count = np.sum(high_available * low_available, axis=2)
    pair_count = high_count * np.maximum(high_count - 1.0, 0.0) / 2.0
    offsuit_count = high_count * low_count - suited_count
    return np.where(
        CLASS_KIND[None, :] == 0,
        pair_count,
        np.where(CLASS_KIND[None, :] == 1, suited_count, offsuit_count),
    ).astype(np.float32)


def normalized_ranges(counts: np.ndarray, likelihoods: np.ndarray) -> np.ndarray:
    values = counts * likelihoods.astype(np.float32)
    totals = np.sum(values, axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("public line induces an empty card-removed range")
    return values / totals


def canonical_card_features(hero_holes: np.ndarray, flops: np.ndarray) -> np.ndarray:
    output = np.zeros((len(hero_holes), CARD_FEATURES), dtype=np.float32)
    for row, (holes, flop) in enumerate(zip(hero_holes, flops)):
        suit_map: dict[int, int] = {}
        ordered_holes = sorted((int(holes[0]), int(holes[1])), reverse=True)
        for section, cards in ((0, ordered_holes), (52, [int(card) for card in flop])):
            for card in cards:
                suit = card % 4
                if suit not in suit_map:
                    suit_map[suit] = len(suit_map)
                canonical = (card // 4) * 4 + suit_map[suit]
                output[row, section + canonical] = 1.0
    return output


def history_reach_weights(cache: CacheArrays, likelihoods: np.ndarray) -> np.ndarray:
    classes = exact_class_indices(cache.holes)
    weights = np.zeros(cache.history_count, dtype=np.float64)
    for history in range(cache.history_count):
        p0 = likelihoods[history, 0, classes[:, 0]]
        p1 = likelihoods[history, 1, classes[:, 1]]
        weights[history] = np.mean(p0 * p1)
    weights[~cache.meaningful_histories] = 0.0
    if weights.sum() <= 0.0:
        raise ValueError("selected policy reaches no meaningful-stack flop leaves")
    return weights / weights.sum()


def sampling_distribution(cache: CacheArrays, likelihoods: np.ndarray) -> np.ndarray:
    reach = history_reach_weights(cache, likelihoods)
    uniform = cache.meaningful_histories.astype(np.float64)
    uniform /= uniform.sum()
    return 0.8 * reach + 0.2 * uniform


def sample_selection(
    cache: CacheArrays,
    distribution: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> SampleSelection:
    if distribution.shape != (cache.history_count,) or not np.isclose(
        distribution.sum(), 1.0
    ):
        raise ValueError("flop-leaf sampling distribution is invalid")
    return SampleSelection(
        deals=rng.integers(0, cache.deals, size=count, dtype=np.int32),
        histories=rng.choice(cache.history_count, size=count, p=distribution).astype(np.int16),
        actors=rng.integers(0, 2, size=count, dtype=np.int8),
    )


def feature_batch(
    cache: CacheArrays,
    likelihoods: np.ndarray,
    selection: SampleSelection,
    include_ranges: bool,
    grouped_targets: GroupedTargets | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    holes = cache.holes[selection.deals]
    flops = cache.flops[selection.deals]
    rows = np.arange(len(selection.deals))
    hero_holes = holes[rows, selection.actors]
    opponent = 1 - selection.actors
    class_indices = exact_class_indices(hero_holes)
    cards = canonical_card_features(hero_holes, flops)
    hand_classes = np.eye(HAND_CLASS_FEATURES, dtype=np.float32)[class_indices]
    textures = np.stack(
        [
            texture_features(hole.tolist(), flop.tolist(), "flop")
            for hole, flop in zip(hero_holes, flops)
        ]
    )
    actor_features = np.eye(2, dtype=np.float32)[selection.actors]
    history_features = np.eye(cache.history_count, dtype=np.float32)[selection.histories]
    scalars = cache.history_scalars[selection.histories].copy()
    swap = selection.actors == 1
    scalars[swap, 1], scalars[swap, 2] = scalars[swap, 2].copy(), scalars[swap, 1].copy()
    if include_ranges:
        public_counts = compatible_class_counts(flops)
        opponent_counts = compatible_class_counts(np.concatenate((flops, hero_holes), axis=1))
        hero_likelihood = likelihoods[selection.histories, selection.actors]
        opponent_likelihood = likelihoods[selection.histories, opponent]
        hero_range = normalized_ranges(public_counts, hero_likelihood)
        opponent_range = normalized_ranges(opponent_counts, opponent_likelihood)
        # Square roots retain zeros and ordering while putting the probability
        # channels on a useful numerical scale beside one-hot card features.
        ranges = np.concatenate((np.sqrt(hero_range), np.sqrt(opponent_range)), axis=1)
    else:
        ranges = np.zeros((len(selection.deals), RANGE_FEATURES), dtype=np.float32)
    features = np.concatenate(
        (
            cards,
            hand_classes,
            textures,
            actor_features,
            history_features,
            scalars,
            ranges,
        ),
        axis=1,
    ).astype(np.float32)
    if grouped_targets is None:
        target_p0 = cache.targets[selection.deals, selection.histories]
        targets = np.where(selection.actors == 0, target_p0, -target_p0).astype(
            np.float32
        )
        standard_errors = cache.standard_errors[
            selection.deals, selection.histories
        ]
    else:
        buckets = grouped_targets.bucket_ids[
            selection.deals, selection.actors
        ]
        targets = grouped_targets.means[
            selection.actors, selection.histories, buckets
        ]
        standard_errors = grouped_targets.standard_errors[
            selection.actors, selection.histories, buckets
        ]
    return features, targets, standard_errors, class_indices


def make_step(model: ValueOracle, optimizer: optim.Optimizer, loss_kind: str = "huber"):
    def loss_fn(model: ValueOracle, features: mx.array, targets: mx.array) -> mx.array:
        errors = model(features).reshape((-1,)) - targets
        if loss_kind == "mse":
            return mx.mean(mx.square(errors))
        absolute = mx.abs(errors)
        huber = mx.where(
            absolute <= HUBER_DELTA,
            0.5 * mx.square(errors),
            HUBER_DELTA * (absolute - 0.5 * HUBER_DELTA),
        )
        return mx.mean(huber)

    value_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(features: mx.array, targets: mx.array) -> mx.array:
        loss, gradients = value_and_grad(model, features, targets)
        optimizer.update(model, gradients)
        return loss

    return step


def oracle_input_size(history_count: int) -> int:
    return (
        CARD_FEATURES
        + HAND_CLASS_FEATURES
        + TEXTURE_FEATURE_COUNT
        + ACTOR_FEATURES
        + history_count
        + SCALAR_FEATURES
        + RANGE_FEATURES
    )


def train_model(
    cache: CacheArrays,
    likelihoods: np.ndarray,
    grouped_targets: GroupedTargets | None,
    include_ranges: bool,
    seed: int,
    hidden_sizes: tuple[int, int],
    steps: int,
    batch_size: int,
    learning_rate: float,
    architecture: str = "flat",
    loss_kind: str = "huber",
    bounded_output: bool = False,
    weight_decay: float = 1e-5,
) -> tuple[ValueOracle, list[float]]:
    input_size = oracle_input_size(cache.history_count)
    mx.random.seed(seed)
    model = ValueOracle(input_size, hidden_sizes, architecture, bounded_output)
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    optimizer.init(model.trainable_parameters())
    mx.eval(model.parameters(), optimizer.state)
    step = make_step(model, optimizer, loss_kind)
    rng = np.random.default_rng(seed ^ 0x5A17)
    distribution = sampling_distribution(cache, likelihoods)
    losses: list[float] = []
    for iteration in range(steps):
        selection = sample_selection(cache, distribution, batch_size, rng)
        features, targets, _, _ = feature_batch(
            cache,
            likelihoods,
            selection,
            include_ranges,
            grouped_targets,
        )
        loss = step(mx.array(features), mx.array(targets / DEPTH_BB))
        mx.eval(loss, model.parameters(), optimizer.state)
        if iteration == 0 or (iteration + 1) % max(steps // 20, 1) == 0:
            losses.append(float(loss))
    return model, losses


def predict(model: ValueOracle, features: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    parts: list[np.ndarray] = []
    for start in range(0, len(features), batch_size):
        values = np.asarray(model(mx.array(features[start : start + batch_size]))).reshape(-1)
        parts.append(values.astype(np.float64) * DEPTH_BB)
    return np.concatenate(parts)


def calibration(predicted: np.ndarray, targets: np.ndarray) -> list[dict[str, float | int]]:
    boundaries = np.quantile(predicted, np.linspace(0.0, 1.0, 11))
    bins: list[dict[str, float | int]] = []
    for index in range(10):
        upper_closed = index == 9
        mask = (predicted >= boundaries[index]) & (
            predicted <= boundaries[index + 1]
            if upper_closed
            else predicted < boundaries[index + 1]
        )
        if not np.any(mask):
            continue
        bins.append(
            {
                "count": int(mask.sum()),
                "predictedMeanBb": float(np.mean(predicted[mask])),
                "observedMeanBb": float(np.mean(targets[mask])),
            }
        )
    return bins


def pearson_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def regression_metrics(
    predicted: np.ndarray,
    targets: np.ndarray,
    standard_errors: np.ndarray,
    group_ids: np.ndarray,
) -> dict[str, Any]:
    error = predicted - targets
    inverse_variance = 1.0 / np.maximum(standard_errors**2 + 0.25, 0.25)
    inverse_variance /= np.sum(inverse_variance)
    group_count = int(group_ids.max()) + 1
    counts = np.bincount(group_ids, minlength=group_count)
    predicted_sums = np.bincount(group_ids, weights=predicted, minlength=group_count)
    target_sums = np.bincount(group_ids, weights=targets, minlength=group_count)
    accepted = counts >= 10
    group_predicted = predicted_sums[accepted] / counts[accepted]
    group_targets = target_sums[accepted] / counts[accepted]
    group_error = group_predicted - group_targets
    correlation = pearson_correlation(group_predicted, group_targets)
    raw_mse = float(np.mean(error**2))
    noise_variance = float(np.mean(standard_errors**2))
    return {
        "samples": len(targets),
        "rmseBb": math.sqrt(raw_mse),
        "maeBb": float(np.mean(np.abs(error))),
        "inverseVarianceWeightedRmseBb": math.sqrt(float(np.sum(inverse_variance * error**2))),
        "noiseAdjustedRmseBb": math.sqrt(max(raw_mse - noise_variance, 0.0)),
        "targetStandardErrorMeanBb": float(np.mean(standard_errors)),
        "fractionTargetSeAtMost0_02bb": float(np.mean(standard_errors <= 0.02)),
        "groupedSamples": int(accepted.sum()),
        "groupedRmseBb": math.sqrt(float(np.mean(group_error**2))),
        "groupedMaeBb": float(np.mean(np.abs(group_error))),
        "groupedPearsonCorrelation": correlation,
        "calibration": calibration(predicted, targets),
    }


def evaluate_models(
    cache: CacheArrays,
    likelihoods: np.ndarray,
    grouped_targets: GroupedTargets | None,
    models: dict[str, list[ValueOracle]],
    count: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, list[np.ndarray]]]:
    selection = sample_selection(
        cache,
        sampling_distribution(cache, likelihoods),
        count,
        np.random.default_rng(seed),
    )
    prediction_parts: dict[str, list[list[np.ndarray]]] = {
        kind: [[] for _ in kind_models] for kind, kind_models in models.items()
    }
    target_parts: list[np.ndarray] = []
    standard_error_parts: list[np.ndarray] = []
    class_parts: list[np.ndarray] = []
    for start in range(0, count, 8192):
        stop = min(start + 8192, count)
        batch_selection = SampleSelection(
            deals=selection.deals[start:stop],
            histories=selection.histories[start:stop],
            actors=selection.actors[start:stop],
        )
        range_features, batch_targets, batch_standard_errors, batch_classes = feature_batch(
            cache,
            likelihoods,
            batch_selection,
            True,
            grouped_targets,
        )
        no_range_features = range_features.copy()
        no_range_features[:, -RANGE_FEATURES:] = 0.0
        for kind, kind_models in models.items():
            features = range_features if kind == "range" else no_range_features
            for model_index, model in enumerate(kind_models):
                prediction_parts[kind][model_index].append(predict(model, features))
        target_parts.append(batch_targets)
        standard_error_parts.append(batch_standard_errors)
        class_parts.append(batch_classes)
    targets = np.concatenate(target_parts)
    standard_errors = np.concatenate(standard_error_parts)
    classes = np.concatenate(class_parts)
    group_ids = (
        (selection.actors.astype(np.int64) * cache.history_count + selection.histories)
        * 169
        + classes
    )
    results: dict[str, Any] = {}
    predictions = {
        kind: [np.concatenate(parts) for parts in model_parts]
        for kind, model_parts in prediction_parts.items()
    }
    for kind, kind_models in models.items():
        per_seed = [
            regression_metrics(values, targets, standard_errors, group_ids)
            for values in predictions[kind]
        ]
        ensemble = np.mean(predictions[kind], axis=0)
        results[kind] = {
            "seeds": per_seed,
            "ensemble": regression_metrics(ensemble, targets, standard_errors, group_ids),
            "crossSeedPredictionMaeBb": float(
                np.mean(np.abs(predictions[kind][0] - predictions[kind][1]))
            ),
            "crossSeedPredictionCorrelation": pearson_correlation(
                predictions[kind][0], predictions[kind][1]
            ),
        }
    range_rmse = results["range"]["ensemble"]["groupedRmseBb"]
    baseline_rmse = results["noRange"]["ensemble"]["groupedRmseBb"]
    results["pairedComparison"] = {
        "groupedRmseRelativeImprovement": (baseline_rmse - range_rmse) / baseline_rmse,
        "rangeWinsEverySeed": all(
            results["range"]["seeds"][index]["groupedRmseBb"]
            < results["noRange"]["seeds"][index]["groupedRmseBb"]
            for index in range(2)
        ),
    }
    return results, predictions


def passes_freeze_criteria(
    validation_metrics: dict[str, Any],
    minimum_range_relative_improvement: float,
    minimum_cross_seed_prediction_correlation: float,
    maximum_grouped_rmse_bb: float,
) -> bool:
    """Apply the predeclared V-only gate before a sealed H cache is opened."""
    paired = validation_metrics["pairedComparison"]
    requires_seed_wins = minimum_range_relative_improvement > 0.0
    return bool(
        (not requires_seed_wins or paired["rangeWinsEverySeed"])
        and paired["groupedRmseRelativeImprovement"]
        >= minimum_range_relative_improvement
        and validation_metrics["range"]["crossSeedPredictionCorrelation"]
        >= minimum_cross_seed_prediction_correlation
        and validation_metrics["range"]["ensemble"]["groupedRmseBb"]
        <= maximum_grouped_rmse_bb
    )


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.evaluation_samples <= 0:
        raise ValueError("steps, batch size, and evaluation samples must be positive")
    numeric_gates = (
        args.learning_rate,
        args.weight_decay,
        args.minimum_range_relative_improvement,
        args.minimum_cross_seed_prediction_correlation,
        args.maximum_grouped_rmse_bb,
    )
    if not all(math.isfinite(value) for value in numeric_gates):
        raise ValueError("optimizer and freeze-gate values must be finite")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("learning rate must be positive and weight decay non-negative")
    if not -1.0 <= args.minimum_cross_seed_prediction_correlation <= 1.0:
        raise ValueError("cross-seed prediction correlation gate must be in [-1, 1]")
    if args.maximum_grouped_rmse_bb < 0.0:
        raise ValueError("maximum grouped RMSE must be non-negative")
    hidden_sizes = tuple(int(value) for value in args.hidden_sizes.split(","))
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if len(hidden_sizes) != 2 or len(seeds) != 2:
        raise ValueError("pilot requires exactly two hidden sizes and two seeds")
    if any(size <= 0 for size in hidden_sizes):
        raise ValueError("hidden sizes must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("loading paired preflop policies", flush=True)
    policy, policy_hashes = averaged_policy((args.policy_a, args.policy_b))
    print("loading T1 and T2 continuation caches", flush=True)
    first = load_cache(args.train_cache_a)
    second = load_cache(args.train_cache_b)
    training = concatenate_caches(first, second)
    del first, second
    gc.collect()
    likelihoods = range_likelihoods(training.histories, policy)
    training_grouped = None
    if args.target_mode == "texture_group_mean":
        print("building T1+T2 conditional-mean targets", flush=True)
        training_grouped = build_grouped_targets(
            training, args.target_bucketing == "texture_hand_class"
        )
    print("loading V selection cache", flush=True)
    validation = load_cache(args.validation_cache)
    validation_grouped = None
    if args.target_mode == "texture_group_mean":
        print("building independent V conditional-mean targets", flush=True)
        validation_grouped = build_grouped_targets(
            validation, args.target_bucketing == "texture_hand_class"
        )
    models: dict[str, list[ValueOracle]] = {"range": [], "noRange": []}
    training_losses: dict[str, list[list[float]]] = {"range": [], "noRange": []}
    if args.load_weights_dir is not None:
        print(f"loading frozen weights from {args.load_weights_dir}", flush=True)
        for kind in models:
            for seed in seeds:
                model = ValueOracle(
                    oracle_input_size(training.history_count),
                    hidden_sizes,
                    args.architecture,
                    args.bounded_output,
                )
                model.load_weights(
                    str(args.load_weights_dir / f"{kind}-seed{seed}.safetensors")
                )
                mx.eval(model.parameters())
                models[kind].append(model)
    else:
        for kind, include_ranges in (("range", True), ("noRange", False)):
            for seed in seeds:
                print(f"training {kind} seed {seed}", flush=True)
                model, losses = train_model(
                    training,
                    likelihoods,
                    training_grouped,
                    include_ranges,
                    seed,
                    hidden_sizes,
                    args.steps,
                    args.batch_size,
                    args.learning_rate,
                    args.architecture,
                    args.loss,
                    args.bounded_output,
                    args.weight_decay,
                )
                models[kind].append(model)
                training_losses[kind].append(losses)
                print(
                    f"complete {kind} seed {seed}; final sampled loss={losses[-1]:.8f}",
                    flush=True,
                )
    print("evaluating paired models on V", flush=True)
    validation_metrics, _ = evaluate_models(
        validation,
        likelihoods,
        validation_grouped,
        models,
        args.evaluation_samples,
        0xC0FFEE,
    )
    architecture_frozen = passes_freeze_criteria(
        validation_metrics,
        args.minimum_range_relative_improvement,
        args.minimum_cross_seed_prediction_correlation,
        args.maximum_grouped_rmse_bb,
    )
    holdout_metrics = None
    holdout_hash = None
    holdout_target_buckets = None
    if architecture_frozen and args.holdout_cache is not None:
        print("V pilot passed; opening H once for confirmation", flush=True)
        holdout = load_cache(args.holdout_cache)
        holdout_hash = holdout.source_sha256
        holdout_grouped = (
            build_grouped_targets(
                holdout, args.target_bucketing == "texture_hand_class"
            )
            if args.target_mode == "texture_group_mean"
            else None
        )
        holdout_target_buckets = (
            None if holdout_grouped is None else holdout_grouped.bucket_count
        )
        holdout_metrics, _ = evaluate_models(
            holdout,
            likelihoods,
            holdout_grouped,
            models,
            args.evaluation_samples,
            0x51A1ED,
        )
    elif args.holdout_cache is not None:
        print("V pilot rejected; H remains unused by the value-oracle pilot", flush=True)
    weights: list[dict[str, Any]] = []
    for kind, kind_models in models.items():
        for seed, model in zip(seeds, kind_models):
            path = args.output_dir / f"{kind}-seed{seed}.safetensors"
            model.save_weights(str(path))
            weights.append(
                {"kind": kind, "seed": seed, "path": path.name, "sha256": sha256_file(path)}
            )
    result = {
        "schema": SCHEMA,
        "status": "promising_representation_pilot" if architecture_frozen else "pilot_rejected",
        "releaseEligible": False,
        "interpretation": (
            "representation/value-function pilot against frozen v26 continuation rollouts; "
            "not a postflop equilibrium solver, exploitability certificate, or GTO release gate"
        ),
        "leakageControls": {
            "observedInputs": [
                "hero_hole_cards",
                "flop",
                "public_preflop_line",
                "card_removed_ranges",
            ],
            "hiddenFromInputs": ["opponent_hole_cards", "turn", "river"],
            "exactCardRemoval": True,
        },
        "rangeEncoding": "square_root_of_normalized_169_class_probability",
        "training": {
            "deals": training.deals,
            "meaningfulFlopHistories": int(training.meaningful_histories.sum()),
            "steps": args.steps,
            "batchSize": args.batch_size,
            "learningRate": args.learning_rate,
            "hiddenSizes": hidden_sizes,
            "architecture": args.architecture,
            "loss": args.loss,
            "boundedOutput": args.bounded_output,
            "weightDecay": args.weight_decay,
            "targetMode": args.target_mode,
            "targetBucketing": args.target_bucketing,
            "trainingTargetBuckets": (
                None if training_grouped is None else training_grouped.bucket_count
            ),
            "validationTargetBuckets": (
                None if validation_grouped is None else validation_grouped.bucket_count
            ),
            "holdoutTargetBuckets": holdout_target_buckets,
            "loadedFrozenWeights": args.load_weights_dir is not None,
            "seeds": seeds,
            "lossSamples": training_losses,
        },
        "sources": {
            "trainingCacheCombinedSha256": training.source_sha256,
            "validationCacheSha256": validation.source_sha256,
            "holdoutCacheSha256": holdout_hash,
            "preflopPolicySha256s": policy_hashes,
        },
        "validation": validation_metrics,
        "architectureFrozenBeforeHoldout": architecture_frozen,
        "freezeCriteria": {
            "minimumRangeRelativeImprovement": args.minimum_range_relative_improvement,
            "minimumCrossSeedPredictionCorrelation": (
                args.minimum_cross_seed_prediction_correlation
            ),
            "maximumGroupedRmseBb": args.maximum_grouped_rmse_bb,
        },
        "holdout": holdout_metrics,
        "weights": weights,
        "limitations": [
            "targets inherit approximation and sampling error from the frozen v26 routed policies",
            "169-class marginal ranges do not encode all joint private-card correlations",
            "pilot predicts continuation values but does not resolve postflop actions",
        ],
    }
    atomic_json(args.output_dir / "report.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
