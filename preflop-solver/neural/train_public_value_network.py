#!/usr/bin/env python3
"""Train paired, suit-equivariant public-belief counterfactual-value networks."""

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
from mlx.utils import tree_map
import numpy as np

SCHEMA = "hu-turn-public-belief-value-network-pilot-v3"
NETWORK_SCHEMA = "hu-public-belief-combo-value-network-v3"
FEATURE_SCHEMA = "rank-suit-invariant-combo-query-v1"
COMBO_COUNT = 1326
DEPTH_BB = 20.0
RESIDUAL_SCALE_BB = 5.0
CONTEXT_PUBLIC_COUNT = 21
CONTEXT_RANGE_COUNT = 338
CONTEXT_COUNT = CONTEXT_PUBLIC_COUNT + CONTEXT_RANGE_COUNT
QUERY_STRUCTURAL_COUNT = 76
QUERY_RANGE_COUNT = 19
QUERY_COUNT = QUERY_STRUCTURAL_COUNT + QUERY_RANGE_COUNT
HAND_CLASS_COUNT = 169


def combo_cards(key: int) -> tuple[int, int]:
    high = 1
    while high * (high - 1) // 2 <= key:
        high += 1
    high -= 1
    return high, key - high * (high - 1) // 2


COMBO_CARDS = np.asarray([combo_cards(key) for key in range(COMBO_COUNT)], dtype=np.int16)


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


def poker_query_features(board: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            final_category = evaluate_cards(
                [*(int(card) for card in board), river, first, second]
            ) >> 24
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
    return strengths, category, percentile, np.concatenate(
        (river_categories, improvement[:, None]), axis=1
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


@dataclass
class Dataset:
    boards: np.ndarray
    actors: np.ndarray
    invested: np.ndarray
    ranges: np.ndarray
    masses: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    projection_weights: np.ndarray
    groups: np.ndarray
    source: dict[str, Any]
    source_sha256: str


class SharedComboValueNetwork(nn.Module):
    """One shared query head evaluates every player/private-card combination."""

    def __init__(self, use_ranges: bool, architecture: str = "compact"):
        super().__init__()
        self.use_ranges = use_ranges
        self.architecture = architecture
        if architecture == "compact":
            context_hidden, embedding, query_hidden, head_hidden = 64, 32, 48, 32
        elif architecture == "wide":
            context_hidden, embedding, query_hidden, head_hidden = 128, 64, 96, 64
        else:
            raise ValueError(f"unknown shared-combo architecture {architecture}")
        self.context_tower = nn.Sequential(
            nn.Linear(CONTEXT_COUNT, context_hidden),
            nn.ReLU(),
            nn.Linear(context_hidden, embedding),
            nn.ReLU(),
        )
        self.query_tower = nn.Sequential(
            nn.Linear(QUERY_COUNT, query_hidden),
            nn.ReLU(),
            nn.Linear(query_hidden, embedding),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(embedding * 2, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
            nn.Tanh(),
        )

    def __call__(
        self, context: mx.array, queries: mx.array, projection_weights: mx.array
    ) -> mx.array:
        equity = queries[:, :, :, 94] if self.use_ranges else queries[:, :, :, 65]
        own_invested = context[:, :, 19, None]
        opponent_invested = context[:, :, 20, None]
        baseline = equity * opponent_invested - (1.0 - equity) * own_invested
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
        raw = baseline + residual * (RESIDUAL_SCALE_BB / DEPTH_BB)
        joint_mass = mx.maximum(mx.sum(projection_weights[:, 0, :], axis=1), 1e-8)
        aggregate = mx.sum(raw * projection_weights, axis=2) / joint_mass[:, None]
        residual = mx.sum(aggregate, axis=1)
        projected = raw - residual[:, None, None] / 2.0
        return projected.reshape((combined.shape[0], COMBO_COUNT * 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
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
    parser.add_argument("--minimum-range-relative-improvement", type=float, default=0.02)
    parser.add_argument("--minimum-cross-seed-correlation", type=float, default=0.95)
    parser.add_argument("--suit-augmentations", type=int, choices=(1, 24), default=1)
    parser.add_argument("--architecture", choices=("compact", "wide"), default="compact")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        high, low = max(permuted_first, permuted_second), min(permuted_first, permuted_second)
        mapping[key] = high * (high - 1) // 2 + low
    if len(np.unique(mapping)) != COMBO_COUNT:
        raise AssertionError("suit permutation must be a combination bijection")
    return mapping


def load_dataset(path: Path, suit_augmentation_count: int = 1) -> Dataset:
    raw = json.loads(path.read_text())
    if raw.get("schema") != "hu-turn-public-belief-cfv-dataset-v1":
        raise ValueError("incompatible public-belief target dataset")
    boards: list[np.ndarray] = []
    actors: list[int] = []
    invested_rows: list[np.ndarray] = []
    range_rows: list[np.ndarray] = []
    mass_rows: list[np.ndarray] = []
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
            board = np.asarray(
                [permute_card(card, permutation) for card in original_board], dtype=np.int16
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
                        raise ValueError("target contains a board-blocked private combination")
            boards.append(board)
            actors.append(int(state["actor"]))
            invested_rows.append(np.asarray(state["invested_bb"], dtype=np.float32))
            range_rows.append(permuted_ranges)
            mass_rows.append(permuted_masses)
            targets.append(permuted_values.reshape(-1) / DEPTH_BB)
            weights.append((permuted_ranges * permuted_masses).reshape(-1))
            groups.append(group)
    projection_weights = np.stack(weights).reshape((-1, 2, COMBO_COUNT))
    weight_array = projection_weights.reshape((-1, COMBO_COUNT * 2)).copy()
    weight_array *= weight_array.size / max(float(weight_array.sum()), 1e-12)
    return Dataset(
        boards=np.stack(boards),
        actors=np.asarray(actors, dtype=np.int8),
        invested=np.stack(invested_rows),
        ranges=np.stack(range_rows),
        masses=np.stack(mass_rows),
        targets=np.stack(targets),
        weights=weight_array,
        projection_weights=projection_weights,
        groups=np.asarray(groups, dtype=np.int32),
        source=raw,
        source_sha256=sha256_file(path),
    )


def scaled_log(value: float | np.ndarray, scale: float) -> float | np.ndarray:
    return np.log1p(np.maximum(value, 0.0) * scale) / np.log1p(scale)


def range_statistics(ranges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def canonical_combo_parts(first: int, second: int, board: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray]:
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
    board_suit = np.sort(np.bincount(board & 3, minlength=4).astype(np.float32))[::-1] / 4.0
    context = np.zeros((2, CONTEXT_COUNT), dtype=np.float32)
    queries = np.zeros((2, COMBO_COUNT, QUERY_COUNT), dtype=np.float32)
    for player in range(2):
        opponent = 1 - player
        context[player, :17] = np.concatenate((board_rank, board_suit))
        context[player, 17:19] = [float(actor == player), float(actor == opponent)]
        context[player, 19:21] = [invested[player] / DEPTH_BB, invested[opponent] / DEPTH_BB]
        context[player, 21:190] = scaled_log(class_mass[player], HAND_CLASS_COUNT)
        context[player, 190:] = scaled_log(class_mass[opponent], HAND_CLASS_COUNT)
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
                opponent_suits = [suit_mass[opponent, high & 3], suit_mass[opponent, low & 3]]
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
    return context, queries


def feature_dataset(dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
    contexts: list[np.ndarray] = []
    queries: list[np.ndarray] = []
    for board, actor, invested, ranges, masses in zip(
        dataset.boards,
        dataset.actors,
        dataset.invested,
        dataset.ranges,
        dataset.masses,
    ):
        context, query = build_features(board, int(actor), invested, ranges, masses)
        contexts.append(context)
        queries.append(query)
    return np.stack(contexts), np.stack(queries)


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
    validation_count = min(
        len(validation_pool),
        state_count - 2,
        max(1, int(round(state_count * validation_fraction))),
    )
    validation = np.sort(rng.permutation(validation_pool)[:validation_count])
    remaining_pool = np.setdiff1d(np.arange(state_count), validation, assume_unique=True)
    remaining = len(remaining_pool)
    tuning_count = min(
        remaining - 1, max(1, int(round(state_count * tuning_fraction)))
    )
    remaining_order = rng.permutation(remaining_pool)
    tuning = np.sort(remaining_order[:tuning_count])
    train = np.sort(remaining_order[tuning_count:])
    return train, tuning, validation


def weighted_metrics(
    truth: np.ndarray, prediction: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    normalized = weights / max(float(weights.sum()), 1e-12)
    error = prediction - truth
    mask = weights > 0
    correlation = float(np.corrcoef(truth[mask], prediction[mask])[0, 1])
    absolute_bb = np.abs(error) * DEPTH_BB
    return {
        "weightedRmseBb": float(np.sqrt(np.sum(normalized * error * error)) * DEPTH_BB),
        "weightedMaeBb": float(np.sum(normalized * absolute_bb)),
        "reachWeightWithin025Bb": float(np.sum(normalized[absolute_bb <= 0.25])),
        "reachWeightWithin050Bb": float(np.sum(normalized[absolute_bb <= 0.50])),
        "correlation": correlation,
    }


def weighted_prediction_correlation(
    first: np.ndarray, second: np.ndarray, weights: np.ndarray
) -> float:
    mask = weights > 0
    return float(np.corrcoef(first[mask], second[mask])[0, 1])


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    target = quantile * float(cumulative[-1])
    return float(values[order[min(np.searchsorted(cumulative, target), len(order) - 1)]])


def corpus_diagnostics(
    dataset: Dataset, contexts: np.ndarray, queries: np.ndarray
) -> dict[str, Any]:
    base_rows = np.asarray(
        [np.flatnonzero(dataset.groups == group)[0] for group in np.unique(dataset.groups)]
    )
    ranges = dataset.ranges[base_rows]
    legal_rows = []
    for board in dataset.boards[base_rows]:
        blocked = set(map(int, board))
        legal_rows.append(
            [first not in blocked and second not in blocked for first, second in COMBO_CARDS]
        )
    legal = np.asarray(legal_rows, dtype=bool)
    coverage = (ranges > 0).sum(axis=2) / legal.sum(axis=1)[:, None]
    entropy_effective = np.exp(
        -np.sum(np.where(ranges > 0, ranges * np.log(np.maximum(ranges, 1e-30)), 0.0), axis=2)
    )
    baseline = np.zeros((len(base_rows), 2, COMBO_COUNT), dtype=np.float32)
    for row_index, row in enumerate(base_rows):
        for player in range(2):
            equity = queries[row, player, :, 94]
            baseline[row_index, player] = (
                equity * dataset.invested[row, 1 - player]
                - (1.0 - equity) * dataset.invested[row, player]
            )
    truth = dataset.targets[base_rows].reshape((-1, 2, COMBO_COUNT)) * DEPTH_BB
    error = np.abs(truth - baseline)
    reach = dataset.projection_weights[base_rows]
    normalized = reach / max(float(reach.sum()), 1e-12)
    return {
        "distinctTurnBoards": int(len({tuple(map(int, board)) for board in dataset.boards[base_rows]})),
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
        "reachWeightBeyondResidualScale": float(
            reach[error > RESIDUAL_SCALE_BB].sum() / max(float(reach.sum()), 1e-12)
        ),
    }


def train_one(
    dataset: Dataset,
    contexts: np.ndarray,
    queries: np.ndarray,
    train_rows: np.ndarray,
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
) -> tuple[SharedComboValueNetwork, np.ndarray, dict[str, Any]]:
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = SharedComboValueNetwork(use_ranges, architecture)
    mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=1e-5)

    def loss_fn(
        current: SharedComboValueNetwork,
        context: mx.array,
        query: mx.array,
        projection_weights: mx.array,
        targets: mx.array,
        weights: mx.array,
    ) -> mx.array:
        errors = current(context, query, projection_weights) - targets
        return mx.sum(weights * errors * errors) / mx.maximum(mx.sum(weights), 1e-8)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    best_tuning_rmse = float("inf")
    best_step = 0
    best_parameters: Any = None
    stale_evaluations = 0
    completed_steps = 0
    tuning_history: list[dict[str, float | int]] = []
    for step in range(1, steps + 1):
        selected = rng.choice(
            train_rows, size=min(batch_size, len(train_rows)), replace=True
        )
        loss, gradients = loss_and_grad(
            model,
            mx.array(contexts[selected]),
            mx.array(queries[selected]),
            mx.array(dataset.projection_weights[selected]),
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
                )
            )
            tuning_rmse = weighted_metrics(
                dataset.targets[tuning_rows],
                tuning_prediction,
                dataset.weights[tuning_rows],
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
        )
    )
    metrics = weighted_metrics(
        dataset.targets[validation_rows], prediction, dataset.weights[validation_rows]
    )
    metrics["bestTuningRmseBb"] = best_tuning_rmse
    metrics["bestStep"] = best_step
    metrics["completedSteps"] = completed_steps
    metrics["tuningHistory"] = tuning_history
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
    model: SharedComboValueNetwork,
    path: Path,
    seed: int,
    source_dataset_sha256: str,
    source_validation_status: str,
    source_policy_sha256: str | None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": NETWORK_SCHEMA,
                "architecture": model.architecture,
                "featureSchema": FEATURE_SCHEMA,
                "seed": seed,
                "usesExactRanges": model.use_ranges,
                "targetScaleBb": DEPTH_BB,
                "rangeScale": COMBO_COUNT,
                "residualScaleBb": RESIDUAL_SCALE_BB,
                "baseline": (
                    "range_conditioned_current_showdown_equity"
                    if model.use_ranges
                    else "structural_hand_strength_percentile"
                ),
                "sourceDatasetSha256": source_dataset_sha256,
                "sourcePolicySha256": source_policy_sha256,
                "sourceValidationStatus": source_validation_status,
                "contextPublicCount": CONTEXT_PUBLIC_COUNT,
                "contextSize": CONTEXT_COUNT,
                "queryStructuralCount": QUERY_STRUCTURAL_COUNT,
                "querySize": QUERY_COUNT,
                "contextTower": [
                    layer_payload(model.context_tower.layers[0], "relu"),
                    layer_payload(model.context_tower.layers[2], "relu"),
                ],
                "queryTower": [
                    layer_payload(model.query_tower.layers[0], "relu"),
                    layer_payload(model.query_tower.layers[2], "relu"),
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
    if args.evaluation_interval <= 0 or args.early_stopping_patience <= 0:
        raise ValueError("early-stopping interval and patience must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset, args.suit_augmentations)
    contexts, queries = feature_dataset(dataset)
    seeds = [int(seed) for seed in args.seeds.split(",")]
    if len(seeds) != 2:
        raise ValueError("paired training requires exactly two independent seeds")
    train_states, tuning_states, validation_states = three_way_state_split(
        len(dataset.source["targets"]),
        seeds[0],
        args.validation_fraction,
        args.tuning_fraction,
        args.holdout_start_index,
    )
    train_rows = np.flatnonzero(np.isin(dataset.groups, train_states))
    tuning_rows = np.flatnonzero(np.isin(dataset.groups, tuning_states))
    validation_rows = np.flatnonzero(np.isin(dataset.groups, validation_states))
    variants: dict[str, list[dict[str, Any]]] = {"range": [], "noRange": []}
    predictions: dict[str, list[np.ndarray]] = {"range": [], "noRange": []}
    for variant, use_ranges in (("range", True), ("noRange", False)):
        for seed in seeds:
            model, prediction, metrics = train_one(
                dataset,
                contexts,
                queries,
                train_rows,
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
            )
            model_path = args.output_dir / f"turn-value-{variant}-seed{seed}.json"
            export_model(
                model,
                model_path,
                seed,
                dataset.source_sha256,
                dataset.source["validation"]["status"],
                dataset.source.get("source_policy_sha256"),
            )
            variants[variant].append(
                {"seed": seed, "metrics": metrics, "weights": model_path.name}
            )
            predictions[variant].append(prediction)
    validation_weights = dataset.weights[validation_rows]
    cross_seed = {
        variant: weighted_prediction_correlation(values[0], values[1], validation_weights)
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
    if not np.isfinite(range_rmse) or range_rmse > args.maximum_rmse_bb:
        reasons.append(
            f"range-network mean holdout RMSE {range_rmse:.6f}bb exceeds {args.maximum_rmse_bb:.6f}bb"
        )
    if (
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
        "featureSchema": FEATURE_SCHEMA,
        "dataset": str(args.dataset),
        "datasetSha256": dataset.source_sha256,
        "sourcePolicySha256": dataset.source.get("source_policy_sha256"),
        "states": int(len(dataset.source["targets"])),
        "augmentedStates": int(len(dataset.targets)),
        "suitAugmentationsPerState": args.suit_augmentations,
        "structurallySuitEquivariant": True,
        "structurallyZeroSumProjected": True,
        "baseline": "range_conditioned_current_showdown_equity_with_structural_percentile_ablation",
        "residualScaleBb": RESIDUAL_SCALE_BB,
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
        "meanRangeRmseBb": range_rmse,
        "meanNoRangeRmseBb": no_range_rmse,
        "rangeRelativeImprovement": relative_improvement,
        "targetSamplingStandardErrorBb": 0.0,
        "corpusDiagnostics": corpus_diagnostics(dataset, contexts, queries),
        "validation": {"status": "accepted" if not reasons else "rejected", "reasons": reasons},
    }
    report_path = args.output_dir / "turn-value-paired-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
