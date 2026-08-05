#!/usr/bin/env python3
"""Train paired, suit-equivariant public-belief counterfactual-value networks."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
import numpy as np

SCHEMA = "hu-turn-public-belief-value-network-pilot-v4"
NETWORK_SCHEMA = "hu-public-belief-combo-value-network-v4"
FEATURE_SCHEMA = "rank-suit-invariant-combo-query-v1"
FEATURE_SCHEMA_BOARD_RELATIVE = "rank-suit-invariant-combo-query-v2"
FEATURE_SCHEMA_EXACT_RUNOUT = "rank-suit-invariant-combo-query-v3"
FEATURE_IMPLEMENTATION_VERSION = "public-value-features-v3-exact-runout-1"
COMBO_COUNT = 1326
DEPTH_BB = 20.0
MINIMUM_VALUE_SCALE_BB = 1.0
POT_BAND_NAMES = ("small", "medium", "large")
CONTEXT_PUBLIC_COUNT = 21
CONTEXT_RANGE_COUNT = 338
CONTEXT_COUNT = CONTEXT_PUBLIC_COUNT + CONTEXT_RANGE_COUNT
QUERY_STRUCTURAL_COUNT = 76
QUERY_RANGE_COUNT = 19
QUERY_COUNT = QUERY_STRUCTURAL_COUNT + QUERY_RANGE_COUNT
CONTEXT_BOARD_RELATIVE_COUNT = 58
QUERY_BOARD_RELATIVE_COUNT = 29
HAND_CLASS_COUNT = 169
LEGACY_TARGET_SCHEMA = "hu-turn-public-belief-cfv-dataset-v1"
COMPLETE_TURN_TARGET_SCHEMA = "hu-turn-public-belief-cfv-dataset-v2"


def feature_sizes(feature_schema: str) -> tuple[int, int]:
    if feature_schema == FEATURE_SCHEMA:
        return CONTEXT_COUNT, QUERY_COUNT
    if feature_schema in (FEATURE_SCHEMA_BOARD_RELATIVE, FEATURE_SCHEMA_EXACT_RUNOUT):
        return (
            CONTEXT_COUNT + CONTEXT_BOARD_RELATIVE_COUNT,
            QUERY_COUNT + QUERY_BOARD_RELATIVE_COUNT,
        )
    raise ValueError(f"unknown value-network feature schema {feature_schema}")


def combo_cards(key: int) -> tuple[int, int]:
    high = 1
    while high * (high - 1) // 2 <= key:
        high += 1
    high -= 1
    return high, key - high * (high - 1) // 2


COMBO_CARDS = np.asarray(
    [combo_cards(key) for key in range(COMBO_COUNT)], dtype=np.int16
)


def hand_class_index(first: int, second: int) -> int:
    first_rank, second_rank = first >> 2, second >> 2
    if first_rank == second_rank:
        return first_rank
    high, low = max(first_rank, second_rank), min(first_rank, second_rank)
    unordered_index = high * (high - 1) // 2 + low
    return 13 + unordered_index * 2 + int((first & 3) == (second & 3))


HAND_CLASS_IDS = np.asarray(
    [hand_class_index(int(first), int(second)) for first, second in COMBO_CARDS],
    dtype=np.int16,
)


def combo_conflicts() -> np.ndarray:
    result = np.empty((COMBO_COUNT, 101), dtype=np.int16)
    for own, (first, second) in enumerate(COMBO_CARDS):
        conflicts = np.flatnonzero(
            (COMBO_CARDS[:, 0] == first)
            | (COMBO_CARDS[:, 1] == first)
            | (COMBO_CARDS[:, 0] == second)
            | (COMBO_CARDS[:, 1] == second)
        )
        if len(conflicts) != 101:
            raise AssertionError("every exact combo must conflict with 101 combos")
        result[own] = conflicts
    return result


COMBO_CONFLICTS = combo_conflicts()


def straight_high(mask: int) -> int:
    for high in range(12, 3, -1):
        needed = 0b11111 << (high - 4)
        if mask & needed == needed:
            return high
    wheel = (1 << 12) | (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)
    return 3 if mask & wheel == wheel else -1


def top_n(mask: int, count: int) -> int:
    result = 0
    found = 0
    for rank in range(12, -1, -1):
        if mask & (1 << rank):
            result = (result << 4) | rank
            found += 1
            if found == count:
                break
    return result


def evaluate_cards(cards: list[int]) -> int:
    rank_count = [0] * 13
    suit_rank_mask = [0] * 4
    rank_mask = 0
    for card in cards:
        rank, suit = card >> 2, card & 3
        rank_count[rank] += 1
        suit_rank_mask[suit] |= 1 << rank
        rank_mask |= 1 << rank
    for mask in suit_rank_mask:
        if mask.bit_count() >= 5:
            straight_flush = straight_high(mask)
            if straight_flush >= 0:
                return (8 << 24) | straight_flush
            return (5 << 24) | top_n(mask, 5)
    quads = [rank for rank in range(12, -1, -1) if rank_count[rank] == 4]
    trips = [rank for rank in range(12, -1, -1) if rank_count[rank] == 3]
    pairs = [rank for rank in range(12, -1, -1) if rank_count[rank] == 2]
    if quads:
        available = rank_mask & ~(1 << quads[0])
        return (7 << 24) | (quads[0] << 4) | top_n(available, 1)
    if trips and (len(trips) >= 2 or pairs):
        pair = trips[1] if len(trips) >= 2 else pairs[0]
        return (6 << 24) | (trips[0] << 4) | pair
    straight = straight_high(rank_mask)
    if straight >= 0:
        return (4 << 24) | straight
    if trips:
        return (3 << 24) | (trips[0] << 16) | top_n(rank_mask & ~(1 << trips[0]), 2)
    if len(pairs) >= 2:
        available = rank_mask & ~(1 << pairs[0]) & ~(1 << pairs[1])
        return (2 << 24) | (pairs[0] << 8) | (pairs[1] << 4) | top_n(available, 1)
    if pairs:
        return (1 << 24) | (pairs[0] << 16) | top_n(rank_mask & ~(1 << pairs[0]), 3)
    return top_n(rank_mask, 5)


def poker_query_features(
    board: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    legal = np.ones(COMBO_COUNT, dtype=bool)
    strengths = np.zeros(COMBO_COUNT, dtype=np.int64)
    category = np.zeros((COMBO_COUNT, 9), dtype=np.float32)
    river_categories = np.zeros((COMBO_COUNT, 9), dtype=np.float32)
    improvement = np.zeros(COMBO_COUNT, dtype=np.float32)
    board_set = set(int(card) for card in board)
    for combo, (first_raw, second_raw) in enumerate(COMBO_CARDS):
        first, second = int(first_raw), int(second_raw)
        if first in board_set or second in board_set:
            legal[combo] = False
            continue
        current = evaluate_cards([*(int(card) for card in board), first, second])
        strengths[combo] = current
        current_category = current >> 24
        category[combo, current_category] = 1.0
        rivers = 0
        improved = 0
        for river in range(52):
            if river in board_set or river == first or river == second:
                continue
            final_category = (
                evaluate_cards([*(int(card) for card in board), river, first, second])
                >> 24
            )
            river_categories[combo, final_category] += 1.0
            improved += int(final_category > current_category)
            rivers += 1
        river_categories[combo] /= rivers
        improvement[combo] = improved / rivers
    legal_strengths = strengths[legal]
    unique, counts = np.unique(legal_strengths, return_counts=True)
    lower = np.concatenate(([0], np.cumsum(counts)[:-1]))
    percentile_by_strength = {
        int(value): (float(before) + float(count) / 2.0) / len(legal_strengths)
        for value, before, count in zip(unique, lower, counts)
    }
    percentile = np.zeros(COMBO_COUNT, dtype=np.float32)
    for combo in np.flatnonzero(legal):
        percentile[combo] = percentile_by_strength[int(strengths[combo])]
    return (
        strengths,
        category,
        percentile,
        np.concatenate((river_categories, improvement[:, None]), axis=1),
    )


def immediate_range_equity(
    strengths: np.ndarray, opponent_range: np.ndarray, compatible_mass: np.ndarray
) -> np.ndarray:
    legal_strengths = strengths[strengths > 0]
    unique = np.unique(legal_strengths)
    strength_rank = {int(strength): rank for rank, strength in enumerate(unique)}
    group_mass = np.zeros(len(unique), dtype=np.float64)
    for combo, strength in enumerate(strengths):
        if strength > 0:
            group_mass[strength_rank[int(strength)]] += opponent_range[combo]
    lower_by_rank = np.concatenate(([0.0], np.cumsum(group_mass)[:-1]))
    result = np.zeros(COMBO_COUNT, dtype=np.float32)
    for combo, strength in enumerate(strengths):
        if strength <= 0 or compatible_mass[combo] <= 0:
            continue
        rank = strength_rank[int(strength)]
        lower = lower_by_rank[rank]
        equal = group_mass[rank]
        conflicts = COMBO_CONFLICTS[combo]
        conflict_strengths = strengths[conflicts]
        conflict_weights = opponent_range[conflicts]
        lower -= conflict_weights[conflict_strengths < strength].sum()
        equal -= conflict_weights[conflict_strengths == strength].sum()
        result[combo] = max(0.0, lower + 0.5 * equal) / compatible_mass[combo]
    return result


def exact_turn_range_equities(
    board: np.ndarray, ranges: np.ndarray, compatible_masses: np.ndarray
) -> np.ndarray:
    """Return exact river-runout showdown equity for every player/combo."""
    if np.asarray(board).shape != (4,):
        raise ValueError("exact turn equity requires a four-card public board")
    numerators = np.zeros((2, COMBO_COUNT), dtype=np.float64)
    board_set = set(int(card) for card in board)
    for river in range(52):
        if river in board_set:
            continue
        strengths = np.zeros(COMBO_COUNT, dtype=np.int64)
        for combo, (first_raw, second_raw) in enumerate(COMBO_CARDS):
            first, second = int(first_raw), int(second_raw)
            if (
                first in board_set
                or second in board_set
                or first == river
                or second == river
            ):
                continue
            strengths[combo] = evaluate_cards(
                [*(int(card) for card in board), river, first, second]
            )
        river_blocked = (COMBO_CARDS[:, 0] == river) | (
            COMBO_CARDS[:, 1] == river
        )
        for player in range(2):
            opponent_range = ranges[1 - player].copy()
            opponent_range[river_blocked] = 0.0
            river_masses = np.maximum(
                opponent_range.sum()
                - opponent_range[COMBO_CONFLICTS].sum(axis=1),
                0.0,
            )
            numerators[player] += (
                immediate_range_equity(strengths, opponent_range, river_masses)
                * river_masses
            )
    # Every compatible pair of private hands leaves exactly 44 possible rivers.
    denominators = compatible_masses.astype(np.float64) * 44.0
    return np.clip(
        np.divide(
            numerators,
            denominators,
            out=np.zeros_like(numerators),
            where=denominators > 1e-8,
        ),
        0.0,
        1.0,
    ).astype(np.float32)


@dataclass
class Dataset:
    boards: np.ndarray
    actors: np.ndarray
    invested: np.ndarray
    ranges: np.ndarray
    masses: np.ndarray
    targets: np.ndarray
    target_scales: np.ndarray
    weights: np.ndarray
    projection_weights: np.ndarray
    groups: np.ndarray
    source: dict[str, Any]
    source_sha256: str


class SharedComboValueNetwork(nn.Module):
    """One shared query head evaluates every player/private-card combination."""

    def __init__(
        self,
        use_ranges: bool,
        architecture: str = "compact",
        value_normalization: str = "pot",
        feature_schema: str = FEATURE_SCHEMA,
    ):
        super().__init__()
        self.use_ranges = use_ranges
        self.architecture = architecture
        self.value_normalization = value_normalization
        self.feature_schema = feature_schema
        context_count, query_count = feature_sizes(feature_schema)
        if architecture == "compact":
            context_hidden, embedding, query_hidden, head_hidden = 64, 32, 48, 32
            self.context_tower = nn.Sequential(
                nn.Linear(context_count, context_hidden),
                nn.ReLU(),
                nn.Linear(context_hidden, embedding),
                nn.ReLU(),
            )
            self.query_tower = nn.Sequential(
                nn.Linear(query_count, query_hidden),
                nn.ReLU(),
                nn.Linear(query_hidden, embedding),
                nn.ReLU(),
            )
            self.head = nn.Sequential(
                nn.Linear(embedding * 2, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, 1),
            )
        elif architecture == "wide":
            context_hidden, embedding, query_hidden, head_hidden = 128, 64, 96, 64
            self.context_tower = nn.Sequential(
                nn.Linear(context_count, context_hidden),
                nn.ReLU(),
                nn.Linear(context_hidden, embedding),
                nn.ReLU(),
            )
            self.query_tower = nn.Sequential(
                nn.Linear(query_count, query_hidden),
                nn.ReLU(),
                nn.Linear(query_hidden, embedding),
                nn.ReLU(),
            )
            self.head = nn.Sequential(
                nn.Linear(embedding * 2, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, 1),
            )
        elif architecture == "deep-gelu":
            embedding = 64
            self.context_tower = nn.Sequential(
                nn.Linear(context_count, 128),
                nn.GELU(approx="fast"),
                nn.Linear(128, 128),
                nn.GELU(approx="fast"),
                nn.Linear(128, embedding),
                nn.GELU(approx="fast"),
            )
            self.query_tower = nn.Sequential(
                nn.Linear(query_count, 128),
                nn.GELU(approx="fast"),
                nn.Linear(128, 128),
                nn.GELU(approx="fast"),
                nn.Linear(128, embedding),
                nn.GELU(approx="fast"),
            )
            self.head = nn.Sequential(
                nn.Linear(embedding * 2, 128),
                nn.GELU(approx="fast"),
                nn.Linear(128, 64),
                nn.GELU(approx="fast"),
                nn.Linear(64, 1),
            )
        elif architecture == "xwide-gelu":
            embedding = 128
            self.context_tower = nn.Sequential(
                nn.Linear(context_count, 256),
                nn.GELU(approx="fast"),
                nn.Linear(256, 256),
                nn.GELU(approx="fast"),
                nn.Linear(256, embedding),
                nn.GELU(approx="fast"),
            )
            self.query_tower = nn.Sequential(
                nn.Linear(query_count, 192),
                nn.GELU(approx="fast"),
                nn.Linear(192, 192),
                nn.GELU(approx="fast"),
                nn.Linear(192, embedding),
                nn.GELU(approx="fast"),
            )
            self.head = nn.Sequential(
                nn.Linear(embedding * 2, 256),
                nn.GELU(approx="fast"),
                nn.Linear(256, 128),
                nn.GELU(approx="fast"),
                nn.Linear(128, 64),
                nn.GELU(approx="fast"),
                nn.Linear(64, 1),
            )
        else:
            raise ValueError(f"unknown shared-combo architecture {architecture}")

    def __call__(
        self,
        context: mx.array,
        queries: mx.array,
        projection_weights: mx.array,
        value_scales: mx.array,
    ) -> mx.array:
        equity = queries[:, :, :, 94] if self.use_ranges else queries[:, :, :, 65]
        own_invested = context[:, :, 19, None]
        opponent_invested = context[:, :, 20, None]
        scale = value_scales[:, None, None]
        baseline = (equity * opponent_invested - (1.0 - equity) * own_invested) * (
            DEPTH_BB / scale
        )
        if not self.use_ranges:
            context = mx.concatenate(
                (
                    context[:, :, :CONTEXT_PUBLIC_COUNT],
                    mx.zeros_like(context[:, :, CONTEXT_PUBLIC_COUNT:]),
                ),
                axis=-1,
            )
            queries = mx.concatenate(
                (
                    queries[:, :, :, :QUERY_STRUCTURAL_COUNT],
                    mx.zeros_like(queries[:, :, :, QUERY_STRUCTURAL_COUNT:]),
                ),
                axis=-1,
            )
        context_embedding = self.context_tower(context)
        query_embedding = self.query_tower(queries)
        expanded_context = mx.broadcast_to(
            context_embedding[:, :, None, :], query_embedding.shape
        )
        combined = mx.concatenate((expanded_context, query_embedding), axis=-1)
        residual = self.head(combined).reshape((combined.shape[0], 2, COMBO_COUNT))
        raw = baseline + residual
        joint_mass = mx.maximum(mx.sum(projection_weights[:, 0, :], axis=1), 1e-8)
        aggregate = mx.sum(raw * projection_weights, axis=2) / joint_mass[:, None]
        residual = mx.sum(aggregate, axis=1)
        projected = raw - residual[:, None, None] / 2.0
        return projected.reshape((combined.shape[0], COMBO_COUNT * 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--supplemental-dataset",
        type=Path,
        action="append",
        default=[],
        help="add validated targets to training only; primary tuning/holdout stays pinned",
    )
    parser.add_argument(
        "--supplemental-sampling-weight",
        type=float,
        default=1.0,
        help="relative within-pot-band draw weight for supplemental training states",
    )
    parser.add_argument(
        "--minimum-primary-batch-fraction",
        type=float,
        default=0.0,
        help=(
            "minimum fraction of each pot-stratified batch drawn from the "
            "authentic primary corpus when supplements are present"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seeds", default="10601,10602")
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--tuning-fraction", type=float, default=0.15)
    parser.add_argument(
        "--holdout-start-index",
        type=int,
        help="restrict untouched validation states to this index and later",
    )
    parser.add_argument("--evaluation-interval", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--maximum-rmse-bb", type=float, default=0.25)
    parser.add_argument(
        "--minimum-range-relative-improvement", type=float, default=0.02
    )
    parser.add_argument("--minimum-cross-seed-correlation", type=float, default=0.95)
    parser.add_argument("--suit-augmentations", type=int, choices=(1, 24), default=1)
    parser.add_argument(
        "--architecture",
        choices=("compact", "wide", "deep-gelu", "xwide-gelu"),
        default="compact",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        help="pin train/tuning/holdout membership independently of model seeds",
    )
    parser.add_argument(
        "--variant-set",
        choices=("both", "range-only"),
        default="both",
        help="range-only is for architecture pilots after the range ablation is established",
    )
    parser.add_argument(
        "--feature-schema",
        choices=(
            FEATURE_SCHEMA,
            FEATURE_SCHEMA_BOARD_RELATIVE,
            FEATURE_SCHEMA_EXACT_RUNOUT,
        ),
        default=FEATURE_SCHEMA,
    )
    parser.add_argument("--feature-workers", type=int, default=1)
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        help=(
            "reuse content-addressed deterministic feature arrays; cache entries "
            "are accepted only after their metadata and SHA-256 digests match"
        ),
    )
    parser.add_argument(
        "--value-normalization",
        choices=("pot", "payoff-exposure"),
        default="pot",
    )
    parser.add_argument("--huber-delta", type=float, default=0.05)
    parser.add_argument("--raw-bb-auxiliary-weight", type=float, default=0.25)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_hashes(inputs: list[tuple[Path, str]]) -> None:
    """Fail closed if a target corpus changes while a long training job runs."""
    changed = [str(path) for path, expected in inputs if sha256_file(path) != expected]
    if changed:
        raise RuntimeError(
            "training source changed after it was loaded: " + ", ".join(changed)
        )


def suit_permutations(count: int) -> list[tuple[int, int, int, int]]:
    permutations = list(itertools.permutations(range(4)))
    return permutations if count == 24 else [permutations[0]]


def permute_card(card: int, permutation: tuple[int, int, int, int]) -> int:
    return (card >> 2) * 4 + permutation[card & 3]


def combo_permutation(permutation: tuple[int, int, int, int]) -> np.ndarray:
    mapping = np.empty(COMBO_COUNT, dtype=np.int32)
    for key, (first, second) in enumerate(COMBO_CARDS):
        permuted_first = permute_card(int(first), permutation)
        permuted_second = permute_card(int(second), permutation)
        high, low = max(permuted_first, permuted_second), min(
            permuted_first, permuted_second
        )
        mapping[key] = high * (high - 1) // 2 + low
    if len(np.unique(mapping)) != COMBO_COUNT:
        raise AssertionError("suit permutation must be a combination bijection")
    return mapping


def value_scale_bb(
    invested: np.ndarray | list[float], normalization: str, depth_bb: float = DEPTH_BB
) -> float:
    invested_array = np.asarray(invested, dtype=np.float64)
    if invested_array.shape != (2,) or np.any(invested_array < 0):
        raise ValueError("value normalization requires two non-negative investments")
    if normalization == "pot":
        scale = float(invested_array.sum())
    elif normalization == "payoff-exposure":
        remaining = np.maximum(depth_bb - invested_array, 0.0)
        scale = float(invested_array.max() + remaining.min())
    elif normalization == "depth":
        # Kept for loading historical v3 models in parity tests and diagnostics.
        scale = depth_bb
    else:
        raise ValueError(f"unknown value normalization {normalization}")
    return max(scale, MINIMUM_VALUE_SCALE_BB)


def pot_band(invested: np.ndarray | list[float]) -> int:
    maximum = float(np.max(np.asarray(invested, dtype=np.float64)))
    if maximum <= 3.5:
        return 0
    if maximum <= 7.5:
        return 1
    return 2


def public_board_texture(board: np.ndarray | list[int]) -> dict[str, str]:
    cards = np.asarray(board, dtype=np.int16)
    if cards.shape != (4,) or len(set(int(card) for card in cards)) != 4:
        raise ValueError("turn texture requires four unique board cards")
    if np.any(cards < 0) or np.any(cards >= 52):
        raise ValueError("turn texture contains an invalid card")

    rank_counts = np.bincount(cards >> 2, minlength=13)
    pairs = int(np.sum(rank_counts == 2))
    maximum_rank_count = int(rank_counts.max())
    if maximum_rank_count == 4:
        rank_texture = "quads"
    elif maximum_rank_count == 3:
        rank_texture = "trips"
    elif pairs == 2:
        rank_texture = "two-pair"
    elif pairs == 1:
        rank_texture = "paired"
    else:
        rank_texture = "unpaired"

    suit_counts = sorted(
        (int(value) for value in np.bincount(cards & 3, minlength=4) if value),
        reverse=True,
    )
    if suit_counts[0] == 4:
        suit_texture = "four-flush"
    elif suit_counts[0] == 3:
        suit_texture = "three-flush"
    elif suit_counts == [2, 2]:
        suit_texture = "two-tone"
    elif suit_counts == [2, 1, 1]:
        suit_texture = "single-suit-pair"
    else:
        suit_texture = "rainbow"

    ranks = {int(rank) for rank in cards >> 2}
    windows = [set(range(start, start + 5)) for start in range(9)]
    windows.append({12, 0, 1, 2, 3})
    maximum_window = max(len(ranks & window) for window in windows)
    if maximum_window >= 4:
        connectivity = "four-straight"
    elif maximum_window == 3:
        connectivity = "connected"
    else:
        connectivity = "disconnected"
    return {
        "rank": rank_texture,
        "suit": suit_texture,
        "connectivity": connectivity,
    }


def load_dataset(
    path: Path,
    suit_augmentation_count: int = 1,
    value_normalization: str = "depth",
) -> Dataset:
    # Hash and parse the same immutable byte snapshot. Hashing the path after
    # parsing would permit a rewrite between the two reads and could stamp a
    # model with the digest of labels it did not train on.
    payload = path.read_bytes()
    raw = json.loads(payload)
    source_sha256 = hashlib.sha256(payload).hexdigest()
    if raw.get("schema") not in {
        LEGACY_TARGET_SCHEMA,
        COMPLETE_TURN_TARGET_SCHEMA,
    }:
        raise ValueError("incompatible public-belief target dataset")
    boards: list[np.ndarray] = []
    actors: list[int] = []
    invested_rows: list[np.ndarray] = []
    range_rows: list[np.ndarray] = []
    mass_rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    target_scales: list[float] = []
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
            raise ValueError(
                "public beliefs and values must use exact 1326-combo vectors"
            )
        for permutation, mapping in zip(permutations, mappings):
            board = np.asarray(
                [permute_card(card, permutation) for card in original_board],
                dtype=np.int16,
            )
            permuted_ranges = np.zeros_like(ranges)
            permuted_values = np.zeros_like(values)
            permuted_masses = np.zeros_like(masses)
            for player in range(2):
                permuted_ranges[player, mapping] = ranges[player]
                permuted_values[player, mapping] = values[player]
                permuted_masses[player, mapping] = masses[player]
                for combo in np.flatnonzero(permuted_ranges[player] > 0):
                    first, second = combo_cards(int(combo))
                    if first in board or second in board:
                        raise ValueError(
                            "target contains a board-blocked private combination"
                        )
            boards.append(board)
            actors.append(int(state["actor"]))
            invested = np.asarray(state["invested_bb"], dtype=np.float32)
            scale = value_scale_bb(invested, value_normalization)
            invested_rows.append(invested)
            range_rows.append(permuted_ranges)
            mass_rows.append(permuted_masses)
            targets.append(permuted_values.reshape(-1) / scale)
            target_scales.append(scale)
            weights.append((permuted_ranges * permuted_masses).reshape(-1))
            groups.append(group)
    projection_weights = np.stack(weights).reshape((-1, 2, COMBO_COUNT))
    weight_array = projection_weights.reshape((-1, COMBO_COUNT * 2)).copy()
    # Every sampled public state receives equal loss mass. Exact combo reach
    # still determines the within-state weighting, but a high-reach pot band
    # cannot drown out another state solely because of blocker-compatible mass.
    row_totals = np.maximum(weight_array.sum(axis=1, keepdims=True), 1e-12)
    weight_array *= (COMBO_COUNT * 2) / row_totals
    return Dataset(
        boards=np.stack(boards),
        actors=np.asarray(actors, dtype=np.int8),
        invested=np.stack(invested_rows),
        ranges=np.stack(range_rows),
        masses=np.stack(mass_rows),
        targets=np.stack(targets),
        target_scales=np.asarray(target_scales, dtype=np.float32),
        weights=weight_array,
        projection_weights=projection_weights,
        groups=np.asarray(groups, dtype=np.int32),
        source=raw,
        source_sha256=source_sha256,
    )


def complete_turn_target_reasons(target: dict[str, Any], index: int) -> list[str]:
    reasons: list[str] = []
    exploitability = float(
        target.get("turn_river_exploitability_bb_per_hand", float("inf"))
    )
    if not np.isfinite(exploitability) or exploitability > 0.05:
        reasons.append(
            f"target {index} exceeds the complete turn-river exploitability gate"
        )
    if target.get("current_turn_river_exploitability_bb_per_hand") is None:
        reasons.append(f"target {index} lacks the current-policy diagnostic")
    probability_error = float(
        target.get("turn_river_maximum_probability_sum_error", float("inf"))
    )
    if not np.isfinite(probability_error) or probability_error > 1e-6:
        reasons.append(f"target {index} exceeds the probability-sum gate")
    for street in ("turn", "river"):
        gain = float(
            target.get(
                f"{street}_only_best_response_gain_bb_per_hand", float("inf")
            )
        )
        if (
            not np.isfinite(gain)
            or gain < 0.0
            or gain > exploitability + 1e-8
        ):
            reasons.append(
                f"target {index} has invalid {street}-only best-response attribution"
            )
    method = str(target.get("turn_river_solver_method", ""))
    if "complete_turn_river_betting" not in method:
        reasons.append(f"target {index} lacks complete turn-river solver provenance")
    if "paired_alternating" not in method:
        reasons.append(
            f"target {index} predates paired alternating turn-river updates"
        )
    if int(target.get("turn_information_sets", 0)) <= 0:
        reasons.append(f"target {index} lacks turn information-set provenance")
    if int(target.get("river_information_sets", 0)) <= 0:
        reasons.append(f"target {index} lacks river information-set provenance")
    if int(target.get("exact_river_cards", 0)) != 48:
        reasons.append(f"target {index} does not enumerate 48 public river cards")
    if abs(float(target.get("zero_sum_residual_bb", float("inf")))) > 1e-7:
        reasons.append(f"target {index} exceeds the zero-sum residual gate")
    return reasons


def complete_turn_release_reasons(source: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if source.get("schema") != COMPLETE_TURN_TARGET_SCHEMA:
        reasons.append("source target corpus omits complete turn betting")
        return reasons
    if source.get("validation", {}).get("status") != "accepted":
        reasons.append("source target corpus is not release-accepted")
    for index, target in enumerate(source.get("targets", [])):
        reasons.extend(complete_turn_target_reasons(target, index))
    return list(dict.fromkeys(reasons))


def combine_training_datasets(primary: Dataset, supplements: list[Dataset]) -> Dataset:
    if not supplements:
        return primary
    components = [primary, *supplements]
    source_policy = primary.source.get("source_policy_sha256")
    if not source_policy or any(
        component.source.get("source_policy_sha256") != source_policy
        for component in components
    ):
        raise ValueError("combined datasets must pin the same frozen source policy")
    if any(
        component.source.get("schema") != primary.source.get("schema")
        for component in components
    ):
        raise ValueError("combined datasets must use the same target schema")

    reasons: list[str] = complete_turn_release_reasons(primary.source)
    # The primary corpus must independently pass every corpus-size/release gate.
    # A small research supplement may fail only its standalone minimum-size gate;
    # every one of its targets is revalidated below and the combined corpus is
    # subjected to the distinct-board gate.
    if primary.source.get("validation", {}).get("status") != "accepted":
        reasons.append("primary target corpus is not accepted")
    all_targets = [
        target for component in components for target in component.source["targets"]
    ]
    for index, target in enumerate(all_targets):
        reasons.extend(complete_turn_target_reasons(target, index))
        belief_method = str(target.get("belief_method", ""))
        if belief_method == "exact_resolver_average_strategy_counterfactual_reach":
            reach = float(target.get("resolver_leaf_reach_probability", 0.0))
            if not np.isfinite(reach) or reach <= 0.0:
                reasons.append(f"target {index} lacks positive resolver leaf reach")
            if len(target.get("resolver_root_board", [])) != 3:
                reasons.append(f"target {index} lacks a resolver root board")
            if not target.get("resolver_public_history"):
                reasons.append(f"target {index} lacks resolver public history")
        else:
            if int(target.get("range_particles", 0)) < 4096:
                reasons.append(f"target {index} lacks 4096-particle belief validation")
            if int(target.get("range_replicates", 0)) < 2:
                reasons.append(f"target {index} lacks paired belief validation")
            particles = int(target.get("range_particles", 0))
            if float(target.get("range_effective_sample_size", 0.0)) < particles * 0.1:
                reasons.append(
                    f"target {index} has insufficient effective belief samples"
                )
            if not belief_method.startswith("exact_per-player_reach_factors"):
                reasons.append(f"target {index} lacks exact reach-factor beliefs")
            if float(target.get("range_maximum_total_variation", float("inf"))) > 0.15:
                reasons.append(f"target {index} exceeds the belief variation gate")
    distinct_boards = len({tuple(target["board"]) for target in all_targets})
    if distinct_boards * 100 < len(all_targets) * 95:
        reasons.append("combined target corpus has fewer than 95% distinct turn boards")

    source = dict(primary.source)
    source["targets"] = all_targets
    source["state_distribution"] = (
        "accepted_primary_with_validated_training_supplements"
    )
    source["component_dataset_sha256"] = [
        component.source_sha256 for component in components
    ]
    source["validation"] = {
        "status": "accepted" if not reasons else "rejected",
        "reasons": list(dict.fromkeys(reasons)),
    }
    digest = hashlib.sha256(
        "|".join(source["component_dataset_sha256"]).encode("ascii")
    ).hexdigest()

    groups = []
    offset = 0
    for component in components:
        groups.append(component.groups + offset)
        offset += len(component.source["targets"])
    return Dataset(
        boards=np.concatenate([component.boards for component in components]),
        actors=np.concatenate([component.actors for component in components]),
        invested=np.concatenate([component.invested for component in components]),
        ranges=np.concatenate([component.ranges for component in components]),
        masses=np.concatenate([component.masses for component in components]),
        targets=np.concatenate([component.targets for component in components]),
        target_scales=np.concatenate(
            [component.target_scales for component in components]
        ),
        weights=np.concatenate([component.weights for component in components]),
        projection_weights=np.concatenate(
            [component.projection_weights for component in components]
        ),
        groups=np.concatenate(groups),
        source=source,
        source_sha256=digest,
    )


def scaled_log(value: float | np.ndarray, scale: float) -> float | np.ndarray:
    return np.log1p(np.maximum(value, 0.0) * scale) / np.log1p(scale)


def range_statistics(
    ranges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    card = np.zeros((2, 52), dtype=np.float32)
    rank = np.zeros((2, 13), dtype=np.float32)
    suit = np.zeros((2, 4), dtype=np.float32)
    classes = np.zeros((2, HAND_CLASS_COUNT), dtype=np.float32)
    for player in range(2):
        classes[player] = np.bincount(
            HAND_CLASS_IDS, weights=ranges[player], minlength=HAND_CLASS_COUNT
        )
        for combo, (first_raw, second_raw) in enumerate(COMBO_CARDS):
            weight = ranges[player, combo]
            if weight <= 0:
                continue
            first, second = int(first_raw), int(second_raw)
            card[player, first] += weight
            card[player, second] += weight
            first_rank, second_rank = first >> 2, second >> 2
            rank[player, first_rank] += weight
            if second_rank != first_rank:
                rank[player, second_rank] += weight
            first_suit, second_suit = first & 3, second & 3
            suit[player, first_suit] += weight
            if second_suit != first_suit:
                suit[player, second_suit] += weight
    return card, rank, suit, classes


def canonical_combo_parts(
    first: int, second: int, board: np.ndarray
) -> tuple[int, int, np.ndarray, np.ndarray]:
    first_rank, second_rank = first >> 2, second >> 2
    if first_rank != second_rank:
        high, low = (first, second) if first_rank > second_rank else (second, first)
        high_mask = np.zeros(13, dtype=np.float32)
        low_mask = np.zeros(13, dtype=np.float32)
        for card in board:
            high_mask[int(card) >> 2] += float((int(card) & 3) == (high & 3))
            low_mask[int(card) >> 2] += float((int(card) & 3) == (low & 3))
        return high, low, high_mask, low_mask
    first_mask = np.zeros(13, dtype=np.float32)
    second_mask = np.zeros(13, dtype=np.float32)
    for card in board:
        first_mask[int(card) >> 2] += float((int(card) & 3) == (first & 3))
        second_mask[int(card) >> 2] += float((int(card) & 3) == (second & 3))
    return first, second, first_mask + second_mask, np.abs(first_mask - second_mask)


def build_features(
    board: np.ndarray,
    actor: int,
    invested: np.ndarray,
    ranges: np.ndarray,
    masses: np.ndarray,
    feature_schema: str = FEATURE_SCHEMA,
) -> tuple[np.ndarray, np.ndarray]:
    card_mass, rank_mass, suit_mass, class_mass = range_statistics(ranges)
    strengths, category, percentile, river_potential = poker_query_features(board)
    immediate_equities = np.stack(
        [
            immediate_range_equity(strengths, ranges[1], masses[0]),
            immediate_range_equity(strengths, ranges[0], masses[1]),
        ]
    )
    board_rank = np.bincount(board >> 2, minlength=13).astype(np.float32) / 4.0
    board_suit = (
        np.sort(np.bincount(board & 3, minlength=4).astype(np.float32))[::-1] / 4.0
    )
    context_count, query_count = feature_sizes(feature_schema)
    context = np.zeros((2, context_count), dtype=np.float32)
    queries = np.zeros((2, COMBO_COUNT, query_count), dtype=np.float32)
    strength_bins = np.minimum((percentile * 10).astype(np.int16), 9)
    board_relative = np.concatenate(
        (
            category,
            river_potential,
            np.eye(10, dtype=np.float32)[strength_bins],
        ),
        axis=1,
    )
    exact_runout_equities = (
        exact_turn_range_equities(board, ranges, masses)
        if feature_schema == FEATURE_SCHEMA_EXACT_RUNOUT
        else None
    )
    for player in range(2):
        opponent = 1 - player
        context[player, :17] = np.concatenate((board_rank, board_suit))
        context[player, 17:19] = [float(actor == player), float(actor == opponent)]
        context[player, 19:21] = [
            invested[player] / DEPTH_BB,
            invested[opponent] / DEPTH_BB,
        ]
        context[player, 21:190] = scaled_log(class_mass[player], HAND_CLASS_COUNT)
        context[player, 190:CONTEXT_COUNT] = scaled_log(
            class_mass[opponent], HAND_CLASS_COUNT
        )
        if feature_schema in (
            FEATURE_SCHEMA_BOARD_RELATIVE,
            FEATURE_SCHEMA_EXACT_RUNOUT,
        ):
            own_mass = max(float(ranges[player].sum()), 1e-8)
            opponent_mass = max(float(ranges[opponent].sum()), 1e-8)
            context[player, CONTEXT_COUNT:] = np.concatenate(
                (
                    (ranges[player, :, None] * category).sum(axis=0) / own_mass,
                    (ranges[opponent, :, None] * category).sum(axis=0)
                    / opponent_mass,
                    np.bincount(
                        strength_bins,
                        weights=ranges[player],
                        minlength=10,
                    )
                    / own_mass,
                    np.bincount(
                        strength_bins,
                        weights=ranges[opponent],
                        minlength=10,
                    )
                    / opponent_mass,
                    (ranges[player, :, None] * river_potential).sum(axis=0)
                    / own_mass,
                    (ranges[opponent, :, None] * river_potential).sum(axis=0)
                    / opponent_mass,
                )
            )
            weighted_opponent_features = ranges[opponent, :, None] * board_relative
            compatible_opponent_features = weighted_opponent_features.sum(
                axis=0, keepdims=True
            ) - weighted_opponent_features[COMBO_CONFLICTS].sum(axis=1)
            compatible_opponent_features = np.divide(
                compatible_opponent_features,
                masses[player, :, None],
                out=np.zeros_like(compatible_opponent_features),
                where=masses[player, :, None] > 1e-8,
            )
            compatible_opponent_features = np.clip(
                compatible_opponent_features, 0.0, 1.0
            )
        for combo, (first_raw, second_raw) in enumerate(COMBO_CARDS):
            first, second = int(first_raw), int(second_raw)
            high, low, high_suit_board, low_suit_board = canonical_combo_parts(
                first, second, board
            )
            high_rank, low_rank = high >> 2, low >> 2
            pair = high_rank == low_rank
            query = queries[player, combo]
            query[high_rank] = 1.0
            query[13 + low_rank] = 1.0
            query[26:28] = [float(pair), float((high & 3) == (low & 3))]
            query[28:41] = high_suit_board
            query[41:54] = low_suit_board
            query[54:56] = [high_suit_board.sum() / 4.0, low_suit_board.sum() / 4.0]
            query[56:65] = category[combo]
            query[65] = percentile[combo]
            query[66:76] = river_potential[combo]
            offset = QUERY_STRUCTURAL_COUNT
            query[offset : offset + 4] = [
                scaled_log(ranges[player, combo], COMBO_COUNT),
                scaled_log(ranges[opponent, combo], COMBO_COUNT),
                masses[player, combo],
                masses[opponent, combo],
            ]
            if pair:
                own_cards = [
                    card_mass[player, high] + card_mass[player, low],
                    abs(card_mass[player, high] - card_mass[player, low]),
                ]
                opponent_cards = [
                    card_mass[opponent, high] + card_mass[opponent, low],
                    abs(card_mass[opponent, high] - card_mass[opponent, low]),
                ]
                own_suits = [
                    suit_mass[player, high & 3] + suit_mass[player, low & 3],
                    abs(suit_mass[player, high & 3] - suit_mass[player, low & 3]),
                ]
                opponent_suits = [
                    suit_mass[opponent, high & 3] + suit_mass[opponent, low & 3],
                    abs(suit_mass[opponent, high & 3] - suit_mass[opponent, low & 3]),
                ]
            else:
                own_cards = [card_mass[player, high], card_mass[player, low]]
                opponent_cards = [card_mass[opponent, high], card_mass[opponent, low]]
                own_suits = [suit_mass[player, high & 3], suit_mass[player, low & 3]]
                opponent_suits = [
                    suit_mass[opponent, high & 3],
                    suit_mass[opponent, low & 3],
                ]
            query[offset + 4 : offset + 8] = scaled_log(
                np.asarray(own_cards + opponent_cards), 26.0
            )
            query[offset + 8 : offset + 12] = scaled_log(
                np.asarray(
                    [
                        rank_mass[player, high_rank],
                        rank_mass[player, low_rank],
                        rank_mass[opponent, high_rank],
                        rank_mass[opponent, low_rank],
                    ]
                ),
                6.5,
            )
            query[offset + 12 : offset + 16] = scaled_log(
                np.asarray(own_suits + opponent_suits), 2.0
            )
            query[offset + 16 : offset + 18] = [
                float(ranges[player].sum()),
                float(ranges[opponent].sum()),
            ]
            query[offset + 18] = immediate_equities[player, combo]
            if exact_runout_equities is not None:
                query[offset + 18] = exact_runout_equities[player, combo]
            if feature_schema in (
                FEATURE_SCHEMA_BOARD_RELATIVE,
                FEATURE_SCHEMA_EXACT_RUNOUT,
            ):
                query[QUERY_COUNT:] = compatible_opponent_features[combo]
    return context, queries


def build_feature_task(
    task: tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, str],
) -> tuple[np.ndarray, np.ndarray]:
    return build_features(*task)


def feature_dataset(
    dataset: Dataset, feature_schema: str = FEATURE_SCHEMA, workers: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    if workers <= 0:
        raise ValueError("feature workers must be positive")
    tasks = [
        (board, int(actor), invested, ranges, masses, feature_schema)
        for board, actor, invested, ranges, masses in zip(
            dataset.boards,
            dataset.actors,
            dataset.invested,
            dataset.ranges,
            dataset.masses,
        )
    ]
    if workers == 1:
        built = map(build_feature_task, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        built = executor.map(build_feature_task, tasks, chunksize=1)
    try:
        features = list(built)
    finally:
        if executor is not None:
            executor.shutdown()
    contexts = [context for context, _ in features]
    queries = [query for _, query in features]
    return np.stack(contexts), np.stack(queries)


def feature_cache_key(dataset: Dataset, feature_schema: str) -> str:
    """Identify feature tensors without trusting a mutable path or timestamp."""
    payload = {
        "implementation": FEATURE_IMPLEMENTATION_VERSION,
        "datasetSha256": dataset.source_sha256,
        "featureSchema": feature_schema,
        "rows": int(len(dataset.targets)),
        "groups": int(len(np.unique(dataset.groups))),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def atomic_save_npy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    temporary.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def feature_dataset_cached(
    dataset: Dataset,
    feature_schema: str = FEATURE_SCHEMA,
    workers: int = 1,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build features once and reuse only byte-verified content-addressed arrays."""
    if cache_dir is None:
        contexts, queries = feature_dataset(dataset, feature_schema, workers)
        return contexts, queries, {"enabled": False, "hit": False}

    cache_dir.mkdir(parents=True, exist_ok=True)
    key = feature_cache_key(dataset, feature_schema)
    context_path = cache_dir / f"{key}-contexts.npy"
    query_path = cache_dir / f"{key}-queries.npy"
    metadata_path = cache_dir / f"{key}.json"
    expected_header = {
        "schema": "hu-public-value-feature-cache-v1",
        "key": key,
        "implementation": FEATURE_IMPLEMENTATION_VERSION,
        "datasetSha256": dataset.source_sha256,
        "featureSchema": feature_schema,
        "rows": int(len(dataset.targets)),
        "groups": int(len(np.unique(dataset.groups))),
    }

    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        for field, expected in expected_header.items():
            if metadata.get(field) != expected:
                raise RuntimeError(f"feature cache metadata mismatch for {field}")
        for name, path in (("contexts", context_path), ("queries", query_path)):
            entry = metadata.get(name, {})
            if not path.is_file() or sha256_file(path) != entry.get("sha256"):
                raise RuntimeError(f"feature cache {name} digest mismatch")
        contexts = np.load(context_path, mmap_mode="r", allow_pickle=False)
        queries = np.load(query_path, mmap_mode="r", allow_pickle=False)
        for name, values in (("contexts", contexts), ("queries", queries)):
            entry = metadata[name]
            if list(values.shape) != entry.get("shape") or str(values.dtype) != entry.get(
                "dtype"
            ):
                raise RuntimeError(f"feature cache {name} shape or dtype mismatch")
        return contexts, queries, {
            "enabled": True,
            "hit": True,
            "key": key,
            "metadata": str(metadata_path),
        }

    contexts, queries = feature_dataset(dataset, feature_schema, workers)
    atomic_save_npy(context_path, contexts)
    atomic_save_npy(query_path, queries)
    metadata = {
        **expected_header,
        "contexts": {
            "path": context_path.name,
            "shape": list(contexts.shape),
            "dtype": str(contexts.dtype),
            "sha256": sha256_file(context_path),
        },
        "queries": {
            "path": query_path.name,
            "shape": list(queries.shape),
            "dtype": str(queries.dtype),
            "sha256": sha256_file(query_path),
        },
    }
    atomic_write_json(metadata_path, metadata)
    # Return read-only memory maps as on a cache hit, both to keep behavior
    # identical and to release the large construction buffers promptly.
    del contexts, queries
    return (
        np.load(context_path, mmap_mode="r", allow_pickle=False),
        np.load(query_path, mmap_mode="r", allow_pickle=False),
        {"enabled": True, "hit": False, "key": key, "metadata": str(metadata_path)},
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


def three_way_state_split(
    state_count: int,
    seed: int,
    validation_fraction: float,
    tuning_fraction: float,
    holdout_start_index: int | None = None,
    strata: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if state_count < 3:
        raise ValueError("early-stopped evaluation requires at least three states")
    rng = np.random.default_rng(seed ^ 0x51A7E)
    if holdout_start_index is None:
        validation_pool = np.arange(state_count)
    else:
        if holdout_start_index < 0 or holdout_start_index >= state_count:
            raise ValueError("holdout start index must select at least one state")
        validation_pool = np.arange(holdout_start_index, state_count)
    if strata is not None:
        strata = np.asarray(strata)
        if strata.shape != (state_count,):
            raise ValueError("split strata must have one label per state")

    def select(pool: np.ndarray, count: int) -> np.ndarray:
        if strata is None:
            return np.sort(rng.permutation(pool)[:count])
        labels, sizes = np.unique(strata[pool], return_counts=True)
        ideal = count * sizes.astype(np.float64) / len(pool)
        quotas = np.minimum(np.floor(ideal).astype(np.int64), sizes)
        if count >= len(labels):
            for offset in np.flatnonzero(quotas == 0):
                donors = np.flatnonzero(quotas > 1)
                if not len(donors):
                    break
                donor = donors[np.argmax(quotas[donors] - ideal[donors])]
                quotas[donor] -= 1
                quotas[offset] = 1
        while int(quotas.sum()) < count:
            available = np.flatnonzero(quotas < sizes)
            priority = ideal[available] - quotas[available]
            chosen = available[np.argmax(priority)]
            quotas[chosen] += 1
        selected = [
            rng.permutation(pool[strata[pool] == label])[:quota]
            for label, quota in zip(labels, quotas)
            if quota
        ]
        return np.sort(np.concatenate(selected))

    validation_count = min(
        len(validation_pool),
        state_count - 2,
        max(1, int(round(state_count * validation_fraction))),
    )
    validation = select(validation_pool, validation_count)
    remaining_pool = np.setdiff1d(
        np.arange(state_count), validation, assume_unique=True
    )
    remaining = len(remaining_pool)
    tuning_count = min(remaining - 1, max(1, int(round(state_count * tuning_fraction))))
    tuning = select(remaining_pool, tuning_count)
    train = np.setdiff1d(remaining_pool, tuning, assume_unique=True)
    return train, tuning, validation


def weighted_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
    target_scales: np.ndarray,
) -> dict[str, Any]:
    normalized = weights / max(float(weights.sum()), 1e-12)
    error = prediction - truth
    mask = weights > 0
    scale = target_scales.reshape((-1, 1))
    truth_bb = truth * scale
    prediction_bb = prediction * scale
    correlation = float(np.corrcoef(truth_bb[mask], prediction_bb[mask])[0, 1])
    absolute_bb = np.abs(prediction_bb - truth_bb)
    signed_bb = prediction_bb - truth_bb
    player_signed = signed_bb.reshape((-1, 2, COMBO_COUNT))
    player_weights = weights.reshape((-1, 2, COMBO_COUNT))
    player_bias = [
        float(
            np.sum(player_weights[:, player] * player_signed[:, player])
            / max(float(player_weights[:, player].sum()), 1e-12)
        )
        for player in range(2)
    ]
    return {
        "weightedRmseBb": float(
            np.sqrt(np.sum(normalized * absolute_bb * absolute_bb))
        ),
        "weightedMaeBb": float(np.sum(normalized * absolute_bb)),
        "weightedMeanErrorBb": float(np.sum(normalized * signed_bb)),
        "playerWeightedMeanErrorBb": player_bias,
        "maximumAbsolutePlayerWeightedMeanErrorBb": max(
            abs(value) for value in player_bias
        ),
        "reachWeightWithin025Bb": float(np.sum(normalized[absolute_bb <= 0.25])),
        "reachWeightWithin050Bb": float(np.sum(normalized[absolute_bb <= 0.50])),
        "correlation": correlation,
    }


def stratified_batch_rows(
    rng: np.random.Generator,
    rows: np.ndarray,
    invested: np.ndarray,
    batch_size: int,
    sampling_weights: np.ndarray | None = None,
) -> np.ndarray:
    if batch_size <= 0 or len(rows) == 0:
        raise ValueError("stratified sampling requires rows and a positive batch size")
    buckets = [
        rows[[pot_band(invested[row]) == band for row in rows]] for band in range(3)
    ]
    available = [band for band, bucket in enumerate(buckets) if len(bucket)]
    selected = []
    for offset in range(batch_size):
        bucket = buckets[available[offset % len(available)]]
        probabilities = None
        if sampling_weights is not None:
            local = sampling_weights[bucket]
            probabilities = local / local.sum()
        selected.append(int(rng.choice(bucket, p=probabilities)))
    rng.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)


def primary_replay_batch_rows(
    rng: np.random.Generator,
    primary_rows: np.ndarray,
    supplemental_rows: np.ndarray,
    invested: np.ndarray,
    batch_size: int,
    minimum_primary_fraction: float,
    sampling_weights: np.ndarray | None = None,
) -> np.ndarray:
    if not 0.0 <= minimum_primary_fraction <= 1.0:
        raise ValueError("primary replay fraction must be between zero and one")
    if len(supplemental_rows) == 0 or minimum_primary_fraction == 0.0:
        rows = np.concatenate((primary_rows, supplemental_rows))
        return stratified_batch_rows(rng, rows, invested, batch_size, sampling_weights)
    rows = np.concatenate((primary_rows, supplemental_rows))
    buckets = [
        rows[[pot_band(invested[row]) == band for row in rows]] for band in range(3)
    ]
    available = [band for band, bucket in enumerate(buckets) if len(bucket)]
    if batch_size <= 0 or not available:
        raise ValueError("primary replay requires rows and a positive batch size")
    primary_slots = int(np.ceil(batch_size * minimum_primary_fraction))
    selected: list[int] = []
    for offset in range(batch_size):
        band = available[offset % len(available)]
        preferred = primary_rows if offset < primary_slots else supplemental_rows
        bucket = preferred[[pot_band(invested[row]) == band for row in preferred]]
        if len(bucket) == 0:
            bucket = buckets[band]
        probabilities = None
        if sampling_weights is not None:
            local = sampling_weights[bucket]
            probabilities = local / local.sum()
        selected.append(int(rng.choice(bucket, p=probabilities)))
    rng.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)


def pot_band_metrics(
    dataset: Dataset,
    rows: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for band, name in enumerate(POT_BAND_NAMES):
        local = np.flatnonzero(
            np.asarray([pot_band(dataset.invested[row]) == band for row in rows])
        )
        if len(local) == 0:
            result[name] = {"states": 0, "weightedRmseBb": float("nan")}
            continue
        selected_rows = rows[local]
        metrics = weighted_metrics(
            dataset.targets[selected_rows],
            prediction[local],
            dataset.weights[selected_rows],
            dataset.target_scales[selected_rows],
        )
        result[name] = {
            "states": int(len(np.unique(dataset.groups[selected_rows]))),
            "weightedRmseBb": metrics["weightedRmseBb"],
            "weightedMaeBb": metrics["weightedMaeBb"],
            "weightedMeanErrorBb": metrics["weightedMeanErrorBb"],
            "playerWeightedMeanErrorBb": metrics["playerWeightedMeanErrorBb"],
            "maximumAbsolutePlayerWeightedMeanErrorBb": metrics[
                "maximumAbsolutePlayerWeightedMeanErrorBb"
            ],
        }
    return result


def weighted_prediction_correlation(
    first: np.ndarray,
    second: np.ndarray,
    weights: np.ndarray,
    target_scales: np.ndarray | None = None,
) -> float:
    if target_scales is not None:
        scale = target_scales.reshape((-1, 1))
        first = first * scale
        second = second * scale
    mask = weights > 0
    return float(np.corrcoef(first[mask], second[mask])[0, 1])


def every_seed_within_rmse(
    variants: list[dict[str, Any]], maximum_rmse_bb: float
) -> bool:
    values = [float(entry["metrics"]["weightedRmseBb"]) for entry in variants]
    return bool(values) and all(
        np.isfinite(value) and value <= maximum_rmse_bb for value in values
    )


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    target = quantile * float(cumulative[-1])
    return float(
        values[order[min(np.searchsorted(cumulative, target), len(order) - 1)]]
    )


def corpus_diagnostics(
    dataset: Dataset, contexts: np.ndarray, queries: np.ndarray
) -> dict[str, Any]:
    base_rows = np.asarray(
        [
            np.flatnonzero(dataset.groups == group)[0]
            for group in np.unique(dataset.groups)
        ]
    )
    ranges = dataset.ranges[base_rows]
    legal_rows = []
    for board in dataset.boards[base_rows]:
        blocked = set(map(int, board))
        legal_rows.append(
            [
                first not in blocked and second not in blocked
                for first, second in COMBO_CARDS
            ]
        )
    legal = np.asarray(legal_rows, dtype=bool)
    coverage = (ranges > 0).sum(axis=2) / legal.sum(axis=1)[:, None]
    entropy_effective = np.exp(
        -np.sum(
            np.where(ranges > 0, ranges * np.log(np.maximum(ranges, 1e-30)), 0.0),
            axis=2,
        )
    )
    baseline = np.zeros((len(base_rows), 2, COMBO_COUNT), dtype=np.float32)
    for row_index, row in enumerate(base_rows):
        for player in range(2):
            equity = queries[row, player, :, 94]
            baseline[row_index, player] = (
                equity * dataset.invested[row, 1 - player]
                - (1.0 - equity) * dataset.invested[row, player]
            )
    truth = (
        dataset.targets[base_rows].reshape((-1, 2, COMBO_COUNT))
        * dataset.target_scales[base_rows, None, None]
    )
    error = np.abs(truth - baseline)
    reach = dataset.projection_weights[base_rows]
    normalized = reach / max(float(reach.sum()), 1e-12)
    return {
        "distinctTurnBoards": int(
            len({tuple(map(int, board)) for board in dataset.boards[base_rows]})
        ),
        "minimumExactComboReachCoverage": float(coverage.min()),
        "meanExactComboReachCoverage": float(coverage.mean()),
        "minimumRangeEntropyEffectiveCombos": float(entropy_effective.min()),
        "meanRangeEntropyEffectiveCombos": float(entropy_effective.mean()),
        "baselineWeightedRmseBb": float(np.sqrt(np.sum(normalized * error * error))),
        "baselineAbsoluteErrorP95Bb": weighted_quantile(
            error.reshape(-1), reach.reshape(-1), 0.95
        ),
        "baselineAbsoluteErrorP99Bb": weighted_quantile(
            error.reshape(-1), reach.reshape(-1), 0.99
        ),
        "maximumValueScaleBb": float(dataset.target_scales[base_rows].max()),
        "minimumValueScaleBb": float(dataset.target_scales[base_rows].min()),
    }


def train_one(
    dataset: Dataset,
    contexts: np.ndarray,
    queries: np.ndarray,
    train_rows: np.ndarray,
    primary_train_rows: np.ndarray,
    supplemental_train_rows: np.ndarray,
    tuning_rows: np.ndarray,
    validation_rows: np.ndarray,
    use_ranges: bool,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    evaluation_interval: int,
    early_stopping_patience: int,
    architecture: str,
    value_normalization: str,
    huber_delta: float,
    raw_bb_auxiliary_weight: float,
    row_sampling_weights: np.ndarray,
    minimum_primary_batch_fraction: float,
    feature_schema: str,
) -> tuple[SharedComboValueNetwork, np.ndarray, np.ndarray, dict[str, Any]]:
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = SharedComboValueNetwork(
        use_ranges, architecture, value_normalization, feature_schema
    )
    mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=1e-5)

    def loss_fn(
        current: SharedComboValueNetwork,
        context: mx.array,
        query: mx.array,
        projection_weights: mx.array,
        value_scales: mx.array,
        targets: mx.array,
        weights: mx.array,
    ) -> mx.array:
        errors = current(context, query, projection_weights, value_scales) - targets

        def huber(values: mx.array) -> mx.array:
            absolute = mx.abs(values)
            quadratic = mx.minimum(absolute, huber_delta)
            linear = absolute - quadratic
            return 0.5 * quadratic * quadratic + huber_delta * linear

        normalized = mx.sum(weights * huber(errors)) / mx.maximum(mx.sum(weights), 1e-8)
        # Express the raw-BB error in depth units so this auxiliary remains
        # numerically comparable with the normalized objective. At 20bb a
        # Huber delta of 0.05 therefore changes regime at one raw big blind.
        raw_depth_units = errors * (value_scales[:, None] / DEPTH_BB)
        raw = mx.sum(weights * huber(raw_depth_units)) / mx.maximum(
            mx.sum(weights), 1e-8
        )
        return normalized + raw_bb_auxiliary_weight * raw

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    best_tuning_rmse = float("inf")
    best_step = 0
    best_parameters: Any = None
    stale_evaluations = 0
    completed_steps = 0
    tuning_history: list[dict[str, float | int]] = []
    for step in range(1, steps + 1):
        selected = primary_replay_batch_rows(
            rng,
            primary_train_rows,
            supplemental_train_rows,
            dataset.invested,
            min(batch_size, max(len(train_rows), 1)),
            minimum_primary_batch_fraction,
            row_sampling_weights,
        )
        loss, gradients = loss_and_grad(
            model,
            mx.array(contexts[selected]),
            mx.array(queries[selected]),
            mx.array(dataset.projection_weights[selected]),
            mx.array(dataset.target_scales[selected]),
            mx.array(dataset.targets[selected]),
            mx.array(dataset.weights[selected]),
        )
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss)
        completed_steps = step
        if step % evaluation_interval == 0 or step == steps:
            tuning_prediction = np.asarray(
                model(
                    mx.array(contexts[tuning_rows]),
                    mx.array(queries[tuning_rows]),
                    mx.array(dataset.projection_weights[tuning_rows]),
                    mx.array(dataset.target_scales[tuning_rows]),
                )
            )
            tuning_rmse = weighted_metrics(
                dataset.targets[tuning_rows],
                tuning_prediction,
                dataset.weights[tuning_rows],
                dataset.target_scales[tuning_rows],
            )["weightedRmseBb"]
            tuning_history.append({"step": step, "weightedRmseBb": tuning_rmse})
            if tuning_rmse < best_tuning_rmse - 1e-6:
                best_tuning_rmse = tuning_rmse
                best_step = step
                best_parameters = tree_map(
                    lambda value: mx.array(np.asarray(value).copy()), model.parameters()
                )
                mx.eval(best_parameters)
                stale_evaluations = 0
            else:
                stale_evaluations += 1
                if stale_evaluations >= early_stopping_patience:
                    break
    if best_parameters is None:
        raise RuntimeError("training did not produce an early-stopping checkpoint")
    model.update(best_parameters)
    mx.eval(model.parameters())
    prediction = np.asarray(
        model(
            mx.array(contexts[validation_rows]),
            mx.array(queries[validation_rows]),
            mx.array(dataset.projection_weights[validation_rows]),
            mx.array(dataset.target_scales[validation_rows]),
        )
    )
    metrics = weighted_metrics(
        dataset.targets[validation_rows],
        prediction,
        dataset.weights[validation_rows],
        dataset.target_scales[validation_rows],
    )
    metrics["bestTuningRmseBb"] = best_tuning_rmse
    metrics["bestStep"] = best_step
    metrics["completedSteps"] = completed_steps
    metrics["tuningHistory"] = tuning_history
    metrics["potBandMetrics"] = pot_band_metrics(dataset, validation_rows, prediction)
    final_tuning_prediction = np.asarray(
        model(
            mx.array(contexts[tuning_rows]),
            mx.array(queries[tuning_rows]),
            mx.array(dataset.projection_weights[tuning_rows]),
            mx.array(dataset.target_scales[tuning_rows]),
        )
    )
    metrics["finalTuningMetrics"] = weighted_metrics(
        dataset.targets[tuning_rows],
        final_tuning_prediction,
        dataset.weights[tuning_rows],
        dataset.target_scales[tuning_rows],
    )
    return model, prediction, final_tuning_prediction, metrics


def layer_payload(layer: nn.Linear, activation: str) -> dict[str, Any]:
    weights = np.asarray(layer.weight)
    return {
        "inputSize": int(weights.shape[1]),
        "outputSize": int(weights.shape[0]),
        "activation": activation,
        "weights": weights.astype(np.float32).reshape(-1).tolist(),
        "biases": np.asarray(layer.bias).astype(np.float32).tolist(),
    }


def tower_payload(
    tower: nn.Sequential, hidden_activation: str, final_activation: str
) -> list[dict[str, Any]]:
    linear_layers = [layer for layer in tower.layers if isinstance(layer, nn.Linear)]
    return [
        layer_payload(
            layer,
            final_activation if index == len(linear_layers) - 1 else hidden_activation,
        )
        for index, layer in enumerate(linear_layers)
    ]


def export_model(
    model: SharedComboValueNetwork,
    path: Path,
    seed: int,
    source_dataset_sha256: str,
    source_dataset_schema: str,
    source_validation_status: str,
    source_policy_sha256: str | None,
    value_normalization: str,
) -> None:
    hidden_activation = (
        "gelu-fast"
        if model.architecture in ("deep-gelu", "xwide-gelu")
        else "relu"
    )
    path.write_text(
        json.dumps(
            {
                "schema": NETWORK_SCHEMA,
                "architecture": model.architecture,
                "featureSchema": model.feature_schema,
                "seed": seed,
                "usesExactRanges": model.use_ranges,
                "targetScaleBb": DEPTH_BB,
                "valueNormalization": value_normalization,
                "rangeScale": COMBO_COUNT,
                "residualUnit": "normalized_state_value_scale",
                "baseline": (
                    "range_conditioned_exact_turn_runout_equity"
                    if model.use_ranges
                    and model.feature_schema == FEATURE_SCHEMA_EXACT_RUNOUT
                    else (
                        "range_conditioned_current_showdown_equity"
                        if model.use_ranges
                        else "structural_hand_strength_percentile"
                    )
                ),
                "sourceDatasetSha256": source_dataset_sha256,
                "sourceDatasetSchema": source_dataset_schema,
                "sourcePolicySha256": source_policy_sha256,
                "sourceValidationStatus": source_validation_status,
                "contextPublicCount": CONTEXT_PUBLIC_COUNT,
                "contextSize": feature_sizes(model.feature_schema)[0],
                "queryStructuralCount": QUERY_STRUCTURAL_COUNT,
                "querySize": feature_sizes(model.feature_schema)[1],
                "contextTower": tower_payload(
                    model.context_tower, hidden_activation, hidden_activation
                ),
                "queryTower": tower_payload(
                    model.query_tower, hidden_activation, hidden_activation
                ),
                "head": tower_payload(model.head, hidden_activation, "linear"),
            },
            separators=(",", ":"),
        )
        + "\n"
    )


def main() -> None:
    args = parse_args()
    if (
        args.evaluation_interval <= 0
        or args.early_stopping_patience <= 0
        or args.huber_delta <= 0
        or args.raw_bb_auxiliary_weight < 0
        or args.feature_workers <= 0
        or not 0.0 < args.supplemental_sampling_weight <= 1.0
        or not 0.0 <= args.minimum_primary_batch_fraction <= 1.0
    ):
        raise ValueError("early-stopping and robust-loss settings are invalid")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    primary_dataset = load_dataset(
        args.dataset, args.suit_augmentations, args.value_normalization
    )
    primary_state_count = len(primary_dataset.source["targets"])
    supplemental_datasets = [
        load_dataset(path, args.suit_augmentations, args.value_normalization)
        for path in args.supplemental_dataset
    ]
    source_inputs = list(
        zip(
            [args.dataset, *args.supplemental_dataset],
            [
                primary_dataset.source_sha256,
                *(dataset.source_sha256 for dataset in supplemental_datasets),
            ],
        )
    )
    dataset = combine_training_datasets(primary_dataset, supplemental_datasets)
    source_dataset_schema = str(dataset.source.get("schema", ""))
    source_release_reasons = complete_turn_release_reasons(dataset.source)
    source_release_status = (
        "accepted" if not source_release_reasons else "rejected"
    )
    contexts, queries, feature_cache = feature_dataset_cached(
        dataset,
        args.feature_schema,
        args.feature_workers,
        args.feature_cache_dir,
    )
    primary_state_bands = np.asarray(
        [
            pot_band(target["invested_bb"])
            for target in primary_dataset.source["targets"]
        ],
        dtype=np.int8,
    )
    seeds = [int(seed) for seed in args.seeds.split(",")]
    if len(seeds) != 2:
        raise ValueError("paired training requires exactly two independent seeds")
    split_seed = args.split_seed if args.split_seed is not None else seeds[0]
    train_states, tuning_states, validation_states = three_way_state_split(
        primary_state_count,
        split_seed,
        args.validation_fraction,
        args.tuning_fraction,
        args.holdout_start_index,
        primary_state_bands,
    )
    primary_train_states = train_states.copy()
    supplemental_states = np.arange(primary_state_count, len(dataset.source["targets"]))
    if len(supplemental_states):
        train_states = np.concatenate((train_states, supplemental_states))
    train_rows = np.flatnonzero(np.isin(dataset.groups, train_states))
    primary_train_rows = np.flatnonzero(np.isin(dataset.groups, primary_train_states))
    supplemental_train_rows = np.flatnonzero(
        np.isin(dataset.groups, supplemental_states)
    )
    tuning_rows = np.flatnonzero(np.isin(dataset.groups, tuning_states))
    validation_rows = np.flatnonzero(np.isin(dataset.groups, validation_states))
    row_sampling_weights = np.ones(len(dataset.boards), dtype=np.float64)
    if len(supplemental_states):
        row_sampling_weights[np.isin(dataset.groups, supplemental_states)] = (
            args.supplemental_sampling_weight
        )
    variant_specs = (
        (("range", True), ("noRange", False))
        if args.variant_set == "both"
        else (("range", True),)
    )
    variants: dict[str, list[dict[str, Any]]] = {
        variant: [] for variant, _ in variant_specs
    }
    predictions: dict[str, list[np.ndarray]] = {
        variant: [] for variant, _ in variant_specs
    }
    tuning_predictions: dict[str, list[np.ndarray]] = {
        variant: [] for variant, _ in variant_specs
    }
    for variant, use_ranges in variant_specs:
        for seed in seeds:
            model, prediction, tuning_prediction, metrics = train_one(
                dataset,
                contexts,
                queries,
                train_rows,
                primary_train_rows,
                supplemental_train_rows,
                tuning_rows,
                validation_rows,
                use_ranges,
                seed,
                args.steps,
                args.batch_size,
                args.learning_rate,
                args.evaluation_interval,
                args.early_stopping_patience,
                args.architecture,
                args.value_normalization,
                args.huber_delta,
                args.raw_bb_auxiliary_weight,
                row_sampling_weights,
                args.minimum_primary_batch_fraction,
                args.feature_schema,
            )
            model_path = args.output_dir / f"turn-value-{variant}-seed{seed}.json"
            verify_source_hashes(source_inputs)
            export_model(
                model,
                model_path,
                seed,
                dataset.source_sha256,
                source_dataset_schema,
                source_release_status,
                dataset.source.get("source_policy_sha256"),
                args.value_normalization,
            )
            variants[variant].append(
                {"seed": seed, "metrics": metrics, "weights": model_path.name}
            )
            predictions[variant].append(prediction)
            tuning_predictions[variant].append(tuning_prediction)
    validation_weights = dataset.weights[validation_rows]
    validation_scales = dataset.target_scales[validation_rows]
    cross_seed = {
        variant: weighted_prediction_correlation(
            values[0], values[1], validation_weights, validation_scales
        )
        for variant, values in predictions.items()
    }
    tuning_cross_seed = {
        variant: weighted_prediction_correlation(
            values[0],
            values[1],
            dataset.weights[tuning_rows],
            dataset.target_scales[tuning_rows],
        )
        for variant, values in tuning_predictions.items()
    }
    ensemble_metrics = {
        variant: {
            "tuning": weighted_metrics(
                dataset.targets[tuning_rows],
                np.mean(values, axis=0),
                dataset.weights[tuning_rows],
                dataset.target_scales[tuning_rows],
            ),
            "validation": weighted_metrics(
                dataset.targets[validation_rows],
                np.mean(predictions[variant], axis=0),
                dataset.weights[validation_rows],
                dataset.target_scales[validation_rows],
            ),
        }
        for variant, values in tuning_predictions.items()
    }
    range_rmse = float(
        np.mean([entry["metrics"]["weightedRmseBb"] for entry in variants["range"]])
    )
    no_range_rmse = (
        float(
            np.mean(
                [entry["metrics"]["weightedRmseBb"] for entry in variants["noRange"]]
            )
        )
        if "noRange" in variants
        else None
    )
    relative_improvement = (
        (no_range_rmse - range_rmse) / max(no_range_rmse, 1e-12)
        if no_range_rmse is not None
        else None
    )
    reasons: list[str] = list(source_release_reasons)
    if not every_seed_within_rmse(variants["range"], args.maximum_rmse_bb):
        maximum_seed_rmse = max(
            float(entry["metrics"]["weightedRmseBb"])
            for entry in variants["range"]
        )
        reasons.append(
            f"range-network maximum seed holdout RMSE {maximum_seed_rmse:.6f}bb exceeds {args.maximum_rmse_bb:.6f}bb"
        )
    if relative_improvement is not None and (
        not np.isfinite(relative_improvement)
        or relative_improvement < args.minimum_range_relative_improvement
    ):
        reasons.append(
            f"range input improves RMSE by {relative_improvement:.3%}, below {args.minimum_range_relative_improvement:.3%}"
        )
    if (
        not np.isfinite(cross_seed["range"])
        or cross_seed["range"] < args.minimum_cross_seed_correlation
    ):
        reasons.append(
            f"range-network cross-seed prediction correlation {cross_seed['range']:.6f} is below {args.minimum_cross_seed_correlation:.6f}"
        )
    report = {
        "schema": SCHEMA,
        "networkSchema": NETWORK_SCHEMA,
        "architecture": args.architecture,
        "variantSet": args.variant_set,
        "splitSeed": split_seed,
        "potStratifiedSplit": True,
        "splitPotBandStates": {
            name: {
                "train": int(
                    np.sum(
                        [
                            primary_state_bands[state] == band
                            for state in primary_train_states
                        ]
                    )
                ),
                "tuning": int(
                    np.sum(
                        [
                            primary_state_bands[state] == band
                            for state in tuning_states
                        ]
                    )
                ),
                "validation": int(
                    np.sum(
                        [
                            primary_state_bands[state] == band
                            for state in validation_states
                        ]
                    )
                ),
            }
            for band, name in enumerate(POT_BAND_NAMES)
        },
        "valueNormalization": args.value_normalization,
        "featureSchema": args.feature_schema,
        "featureWorkers": args.feature_workers,
        "featureCache": feature_cache,
        "dataset": str(args.dataset),
        "supplementalDatasets": [str(path) for path in args.supplemental_dataset],
        "datasetSha256": dataset.source_sha256,
        "sourceDatasetSchema": source_dataset_schema,
        "componentDatasetSha256": dataset.source.get(
            "component_dataset_sha256", [dataset.source_sha256]
        ),
        "sourcePolicySha256": dataset.source.get("source_policy_sha256"),
        "sourceValidation": dataset.source.get("validation"),
        "states": int(len(dataset.source["targets"])),
        "primaryStates": primary_state_count,
        "supplementalTrainingStates": supplemental_states.tolist(),
        "supplementalSamplingWeight": args.supplemental_sampling_weight,
        "minimumPrimaryBatchFraction": args.minimum_primary_batch_fraction,
        "primaryTrainingRows": int(len(primary_train_rows)),
        "supplementalTrainingRows": int(len(supplemental_train_rows)),
        "augmentedStates": int(len(dataset.targets)),
        "suitAugmentationsPerState": args.suit_augmentations,
        "structurallySuitEquivariant": True,
        "structurallyZeroSumProjected": True,
        "baseline": (
            "range_conditioned_exact_turn_runout_equity_with_structural_percentile_ablation"
            if args.feature_schema == FEATURE_SCHEMA_EXACT_RUNOUT
            else "range_conditioned_current_showdown_equity_with_structural_percentile_ablation"
        ),
        "residualUnit": "normalized_state_value_scale",
        "loss": {
            "kind": (
                "state-balanced reach-weighted Huber with depth-normalized "
                "raw-bb auxiliary"
            ),
            "huberDelta": args.huber_delta,
            "rawBbAuxiliaryWeight": args.raw_bb_auxiliary_weight,
            "potStratifiedBatches": True,
            "authenticPrimaryReplay": args.minimum_primary_batch_fraction > 0.0,
        },
        "trainStates": train_states.tolist(),
        "tuningStates": tuning_states.tolist(),
        "validationStates": validation_states.tolist(),
        "holdoutStartIndex": args.holdout_start_index,
        "steps": args.steps,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "evaluationInterval": args.evaluation_interval,
        "earlyStoppingPatience": args.early_stopping_patience,
        "variants": variants,
        "crossSeedPredictionCorrelation": cross_seed,
        "tuningCrossSeedPredictionCorrelation": tuning_cross_seed,
        "twoSeedOutputEnsembleMetrics": ensemble_metrics,
        "meanRangeRmseBb": range_rmse,
        "meanNoRangeRmseBb": no_range_rmse,
        "rangeRelativeImprovement": relative_improvement,
        "targetSamplingStandardErrorBb": 0.0,
        "corpusDiagnostics": corpus_diagnostics(dataset, contexts, queries),
        "validation": {
            "status": "accepted" if not reasons else "rejected",
            "reasons": reasons,
        },
    }
    verify_source_hashes(source_inputs)
    report_path = args.output_dir / "turn-value-paired-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
