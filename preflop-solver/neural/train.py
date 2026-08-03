#!/usr/bin/env python3
"""Bounded-memory MLX Deep DCFR+ trainer for the exact Rust heads-up game.

Rust generates external-sampling advantages using a frozen pair of cumulative
advantage networks. This process keeps grouped samples in fixed-size float16
memory maps, performs compiled MLX updates, checkpoints optimizer state, and
exports framework-neutral browser weights at configured checkpoints.

Artifacts remain experimental until independent validation promotes them.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import random
import resource
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np


STATE_FEATURE_COUNT = 716
ACTION_FEATURE_COUNT = 9
INPUT_FEATURE_COUNT = STATE_FEATURE_COUNT + ACTION_FEATURE_COUNT
PROFILE_FEATURE_COUNT = 16
MAX_TRAJECTORY_ACTIONS = 32
MAX_POLICY_ACTIONS = 8
STREETS = ("preflop", "flop", "turn", "river")
ACTIONS = ("fold", "check", "call", "bet", "raise", "all_in")
RUN_SCHEMA = "hu-neural-mlx-run-v14"
LEGACY_RESUME_SCHEMA = "hu-neural-mlx-run-v9"
NETWORK_SCHEMA = "hu-neural-training-networks-v4"
STATE_FEATURE_SCHEMA = "hu-cash-trajectory-poker-aware-v4"
POKER_FEATURE_OFFSET = 604
TEXTURE_FEATURE_OFFSET = 652
TEXTURE_FEATURE_COUNT = 64
ACTION_VALUE_STANDARD_ERROR_FLOOR_BB = 0.005
UNEVALUATED_STANDARD_ERROR_PRIOR_BB = 0.10
ADVANTAGE_UPDATE = "bootstrapped_deep_dcfr_plus_vr_zero_init_texture"
STOP_REQUESTED = False
LEADING_20BB_PROFILE = {
    "traversals_per_round": 400,
    "reservoir_capacity": 100_000,
    "hidden_sizes": "256,128",
    "batch_size": 1_024,
    "steps_per_round": 100,
    "learning_rate": 1e-3,
    "advantage_alpha": 2.0,
    "variance_baseline_scale": 0.5,
    "artifact_every": 10,
    "preflop_runout_samples": 256,
    "flop_runout_samples": 128,
    "value_rollouts_per_action": 4,
}


@dataclass(frozen=True)
class RunConfig:
    schema: str
    depth_bb: int
    seed: int
    reservoir_capacity: int
    hidden_sizes: tuple[int, int]
    batch_size: int
    learning_rate: float
    learning_rate_final: float | None
    learning_rate_decay_start_round: int | None
    learning_rate_decay_end_round: int | None
    traversals_per_round: int
    steps_per_round: int
    advantage_alpha: float
    variance_baseline_scale: float
    replay_street_proposal: tuple[float, float, float, float] | None
    value_rollouts_per_action: int
    artifact_every: int
    preflop_runout_samples: int
    flop_runout_samples: int
    exact_turn_rivers: bool
    compact_serving_grid: bool


class ActionScorer(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: tuple[int, int], output_size: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size),
        )

    def __call__(self, inputs: mx.array) -> mx.array:
        return self.layers(inputs)


def stratified_replay_indices(
    street_ids: np.ndarray,
    size: int,
    batch_size: int,
    street_proposal: tuple[float, float, float, float] | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample streets deliberately and correct back to the empirical objective."""
    if size <= 0 or batch_size <= 0:
        raise ValueError("replay size and batch size must be positive")
    if street_proposal is None:
        return rng.integers(0, size, size=batch_size), np.ones(batch_size, dtype=np.float32)
    proposal = np.asarray(street_proposal, dtype=np.float64)
    if proposal.shape != (4,) or np.any(proposal <= 0) or not np.isclose(np.sum(proposal), 1.0):
        raise ValueError("street replay proposal must contain four positive probabilities")

    active_streets = np.asarray(street_ids[:size], dtype=np.uint8)
    groups = [np.flatnonzero(active_streets == street) for street in range(4)]
    available = [street for street, group in enumerate(groups) if len(group) > 0]
    if len(available) < 2 or batch_size < len(available):
        return rng.integers(0, size, size=batch_size), np.ones(batch_size, dtype=np.float32)

    active_proposal = proposal[available]
    active_proposal /= np.sum(active_proposal)
    remaining = batch_size - len(available)
    raw_extra = remaining * active_proposal
    counts = np.floor(raw_extra).astype(np.int64) + 1
    leftover = batch_size - int(np.sum(counts))
    order = np.argsort(-(raw_extra - np.floor(raw_extra)), kind="stable")
    for offset in range(leftover):
        counts[order[offset]] += 1

    selected: list[np.ndarray] = []
    correction_parts: list[np.ndarray] = []
    for street, count in zip(available, counts):
        selected.append(rng.choice(groups[street], size=int(count), replace=True))
        empirical_probability = len(groups[street]) / size
        realized_probability = int(count) / batch_size
        correction_parts.append(
            np.full(
                int(count),
                empirical_probability / realized_probability,
                dtype=np.float32,
            )
        )
    indices = np.concatenate(selected)
    corrections = np.concatenate(correction_parts)
    permutation = rng.permutation(batch_size)
    return indices[permutation], corrections[permutation]


class ReplayReservoir:
    """A deterministic, fixed-size reservoir backed by raw memory maps."""

    def __init__(
        self,
        directory: Path,
        name: str,
        capacity: int,
        target_size: int,
        size: int = 0,
        seen: int = 0,
    ) -> None:
        self.name = name
        self.capacity = capacity
        self.target_size = target_size
        self.size = size
        self.seen = seen
        directory.mkdir(parents=True, exist_ok=True)
        self.feature_path = directory / f"{name}.features.f16"
        self.target_path = directory / f"{name}.targets.f32"
        self.weight_path = directory / f"{name}.weights.f32"
        self.street_path = directory / f"{name}.street.u8"
        paths = (self.feature_path, self.target_path, self.weight_path, self.street_path)
        existing = all(path.exists() for path in paths)
        if any(path.exists() for path in paths) and not existing:
            raise RuntimeError(f"reservoir {name} is incomplete")
        mode = "r+" if existing else "w+"
        self.features = np.memmap(
            self.feature_path,
            dtype=np.float16,
            mode=mode,
            shape=(capacity, INPUT_FEATURE_COUNT),
        )
        self.targets = np.memmap(
            self.target_path,
            dtype=np.float32,
            mode=mode,
            shape=(capacity, target_size),
        )
        self.weights = np.memmap(
            self.weight_path,
            dtype=np.float32,
            mode=mode,
            shape=(capacity, 1),
        )
        self.streets = np.memmap(
            self.street_path,
            dtype=np.uint8,
            mode=mode,
            shape=(capacity,),
        )

    def add(
        self,
        features: np.ndarray,
        target: np.ndarray,
        weight: float,
        street: int,
        rng: random.Random,
    ) -> None:
        if street < 0 or street >= len(STREETS):
            raise ValueError("replay street is outside the pinned schema")
        self.seen += 1
        if self.size < self.capacity:
            index = self.size
            self.size += 1
        else:
            index = rng.randrange(self.seen)
            if index >= self.capacity:
                return
        self.features[index] = features
        self.targets[index] = target
        self.weights[index, 0] = weight
        self.streets[index] = street

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        street_proposal: tuple[float, float, float, float] | None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        if self.size == 0:
            raise RuntimeError(f"reservoir {self.name} is empty")
        # Sampling with replacement keeps the compiled MLX graph shape fixed,
        # including during the first partially filled replay round.
        indices, corrections = stratified_replay_indices(
            self.streets,
            self.size,
            batch_size,
            street_proposal,
            rng,
        )
        features = np.asarray(self.features[indices], dtype=np.float32)
        targets = np.asarray(self.targets[indices], dtype=np.float32)
        weights = np.asarray(self.weights[indices], dtype=np.float32) * corrections[:, None]
        return mx.array(features), mx.array(targets), mx.array(weights)

    def flush(self) -> None:
        self.features.flush()
        self.targets.flush()
        self.weights.flush()
        self.streets.flush()

    def summary(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "seen": self.seen,
            "target_size": self.target_size,
            "street_sizes": [
                int(np.count_nonzero(self.streets[: self.size] == street))
                for street in range(4)
            ],
        }


class DecisionReservoir:
    """A fixed-size reservoir that preserves complete legal-action decisions."""

    def __init__(
        self,
        directory: Path,
        name: str,
        capacity: int,
        normalize_targets: bool,
        size: int = 0,
        seen: int = 0,
    ) -> None:
        self.name = name
        self.capacity = capacity
        self.normalize_targets = normalize_targets
        self.size = size
        self.seen = seen
        directory.mkdir(parents=True, exist_ok=True)
        self.state_path = directory / f"{name}.states.f16"
        self.action_path = directory / f"{name}.actions.f16"
        self.target_path = directory / f"{name}.targets.f32"
        self.mask_path = directory / f"{name}.masks.u8"
        self.weight_path = directory / f"{name}.weights.f32"
        self.street_path = directory / f"{name}.street.u8"
        paths = (
            self.state_path,
            self.action_path,
            self.target_path,
            self.mask_path,
            self.weight_path,
            self.street_path,
        )
        existing = all(path.exists() for path in paths)
        if any(path.exists() for path in paths) and not existing:
            raise RuntimeError(f"decision reservoir {name} is incomplete")
        mode = "r+" if existing else "w+"
        self.states = np.memmap(
            self.state_path,
            dtype=np.float16,
            mode=mode,
            shape=(capacity, STATE_FEATURE_COUNT),
        )
        self.actions = np.memmap(
            self.action_path,
            dtype=np.float16,
            mode=mode,
            shape=(capacity, MAX_POLICY_ACTIONS, ACTION_FEATURE_COUNT),
        )
        self.targets = np.memmap(
            self.target_path,
            dtype=np.float32,
            mode=mode,
            shape=(capacity, MAX_POLICY_ACTIONS),
        )
        self.masks = np.memmap(
            self.mask_path,
            dtype=np.uint8,
            mode=mode,
            shape=(capacity, MAX_POLICY_ACTIONS),
        )
        self.weights = np.memmap(
            self.weight_path,
            dtype=np.float32,
            mode=mode,
            shape=(capacity, 1),
        )
        self.streets = np.memmap(
            self.street_path,
            dtype=np.uint8,
            mode=mode,
            shape=(capacity,),
        )

    def add(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
        weight: float,
        street: int,
        rng: random.Random,
    ) -> None:
        if street < 0 or street >= len(STREETS):
            raise ValueError("decision street is outside the pinned schema")
        if actions.ndim != 2 or actions.shape[1] != ACTION_FEATURE_COUNT:
            raise ValueError("policy decision action features have the wrong shape")
        if len(actions) == 0 or len(actions) > MAX_POLICY_ACTIONS or len(actions) != len(targets):
            raise ValueError("policy decision exceeds the grouped action schema")
        target_sum = float(np.sum(targets))
        if not np.all(np.isfinite(targets)) or (
            self.normalize_targets and target_sum <= 0
        ):
            raise ValueError("policy decision targets are invalid")
        self.seen += 1
        if self.size < self.capacity:
            index = self.size
            self.size += 1
        else:
            index = rng.randrange(self.seen)
            if index >= self.capacity:
                return
        action_count = len(actions)
        self.states[index] = state
        self.actions[index] = 0
        self.actions[index, :action_count] = actions
        self.targets[index] = 0
        self.targets[index, :action_count] = (
            targets / target_sum if self.normalize_targets else targets
        )
        self.masks[index] = 0
        self.masks[index, :action_count] = 1
        self.weights[index, 0] = weight
        self.streets[index] = street

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        street_proposal: tuple[float, float, float, float] | None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        if self.size == 0:
            raise RuntimeError(f"decision reservoir {self.name} is empty")
        indices, corrections = stratified_replay_indices(
            self.streets,
            self.size,
            batch_size,
            street_proposal,
            rng,
        )
        states = np.asarray(self.states[indices], dtype=np.float32)
        actions = np.asarray(self.actions[indices], dtype=np.float32)
        expanded_states = np.broadcast_to(
            states[:, None, :],
            (batch_size, MAX_POLICY_ACTIONS, STATE_FEATURE_COUNT),
        )
        features = np.concatenate((expanded_states, actions), axis=2)
        targets = np.asarray(self.targets[indices], dtype=np.float32)
        masks = np.asarray(self.masks[indices], dtype=np.float32)
        weights = np.asarray(self.weights[indices], dtype=np.float32) * corrections[:, None]
        return mx.array(features), mx.array(targets), mx.array(masks), mx.array(weights)

    def clear(self) -> None:
        """Forget prior decisions without reallocating the bounded memory maps."""
        self.size = 0
        self.seen = 0

    def flush(self) -> None:
        self.states.flush()
        self.actions.flush()
        self.targets.flush()
        self.masks.flush()
        self.weights.flush()
        self.streets.flush()

    def summary(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "seen": self.seen,
            "street_sizes": [
                int(np.count_nonzero(self.streets[: self.size] == street))
                for street in range(4)
            ],
            "max_actions": MAX_POLICY_ACTIONS,
            "storage": "grouped_decisions",
            "target_mode": "probability" if self.normalize_targets else "raw",
        }


def one_hot(size: int, selected: int, target: np.ndarray, offset: int) -> None:
    if selected < 0 or selected >= size:
        raise ValueError("one-hot index outside schema")
    target[offset + selected] = 1.0


def canonical_suit_map(private_cards: Iterable[int], board: Iterable[int]) -> list[int]:
    private_masks = [0] * 4
    board_masks = [0] * 4
    for card in private_cards:
        private_masks[int(card) % 4] |= 1 << (int(card) // 4)
    for card in board:
        board_masks[int(card) % 4] |= 1 << (int(card) // 4)
    ordered = sorted(
        range(4),
        key=lambda suit: (-private_masks[suit], -board_masks[suit], suit),
    )
    mapping = [0] * 4
    for canonical, original in enumerate(ordered):
        mapping[original] = canonical
    return mapping


def canonical_card(card: int, suit_map: list[int]) -> int:
    return (int(card) // 4) * 4 + suit_map[int(card) % 4]


def rank_mask_has_straight(mask: int) -> bool:
    return any(((mask >> low) & 0b11111) == 0b11111 for low in range(9)) or (
        mask & ((1 << 12) | 0b1111)
    ) == ((1 << 12) | 0b1111)


def straight_window_density(mask: int) -> int:
    regular = max((((mask >> low) & 0b11111).bit_count() for low in range(9)), default=0)
    wheel = (mask & ((1 << 12) | 0b1111)).bit_count()
    return max(regular, wheel)


def made_hand_category(cards: list[int]) -> int:
    if len(cards) < 5:
        raise ValueError("made-hand category requires at least five cards")
    rank_counts = [0] * 13
    suit_masks = [0] * 4
    rank_mask = 0
    for card in cards:
        rank = int(card) // 4
        suit = int(card) % 4
        rank_counts[rank] += 1
        suit_masks[suit] |= 1 << rank
        rank_mask |= 1 << rank
    for suit_mask in suit_masks:
        if suit_mask.bit_count() >= 5 and rank_mask_has_straight(suit_mask):
            return 8
    if max(rank_counts) == 4:
        return 7
    trips = sum(count == 3 for count in rank_counts)
    pairs = sum(count == 2 for count in rank_counts)
    if trips >= 2 or (trips >= 1 and pairs >= 1):
        return 6
    if any(suit_mask.bit_count() >= 5 for suit_mask in suit_masks):
        return 5
    if rank_mask_has_straight(rank_mask):
        return 4
    if trips:
        return 3
    if pairs >= 2:
        return 2
    if pairs == 1:
        return 1
    return 0


def texture_features(private_cards: list[int], board: list[int], street: str) -> np.ndarray:
    """Mirror the Rust/TypeScript suit-invariant 64-feature poker summary."""
    output = np.zeros(TEXTURE_FEATURE_COUNT, dtype=np.float32)
    hole_ranks = [int(card) // 4 for card in private_cards]
    if not board:
        output[30] = float(hole_ranks[0] == hole_ranks[1])
        return output

    output[0] = 1.0
    output[1 + made_hand_category([*private_cards, *board])] = 1.0
    board_rank_counts = [0] * 13
    board_suit_counts = [0] * 4
    board_rank_mask = 0
    for card in board:
        rank = int(card) // 4
        board_rank_counts[rank] += 1
        board_suit_counts[int(card) % 4] += 1
        board_rank_mask |= 1 << rank
    board_max_rank_count = max(board_rank_counts)
    board_max_suit_count = max(board_suit_counts)
    board_density = straight_window_density(board_rank_mask)
    output[10 + min(board_max_rank_count - 1, 3)] = 1.0
    output[14 + min(board_max_suit_count - 1, 4)] = 1.0
    output[19 + min(board_density - 1, 4)] = 1.0
    board_high = max(int(card) // 4 for card in board)
    board_low = min(int(card) // 4 for card in board)
    output[24 + (2 if board_high >= 10 else 1 if board_high >= 7 else 0)] = 1.0
    overcards = sum(rank > board_high for rank in hole_ranks)
    output[27 + min(overcards, 2)] = 1.0
    pocket_pair = hole_ranks[0] == hole_ranks[1]
    output[30] = float(pocket_pair)
    output[31] = float(pocket_pair and hole_ranks[0] > board_high)
    matches = [board_rank_counts[rank] > 0 for rank in hole_ranks]
    output[32] = float(any(rank == board_high for rank in hole_ranks))
    output[33] = float(
        any(rank != board_high and rank != board_low and board_rank_counts[rank] > 0 for rank in hole_ranks)
    )
    output[34] = float(board_low != board_high and any(rank == board_low for rank in hole_ranks))
    output[35] = float(matches[0] and matches[1])
    output[36] = float(matches[0] ^ matches[1])
    board_pairs = sum(count == 2 for count in board_rank_counts)
    output[37] = float(board_pairs >= 1)
    output[38] = float(board_pairs >= 2)
    output[39] = float(3 in board_rank_counts)
    output[40] = float(4 in board_rank_counts)

    full_rank_counts = board_rank_counts.copy()
    full_suit_counts = board_suit_counts.copy()
    full_rank_mask = board_rank_mask
    for card in private_cards:
        rank = int(card) // 4
        full_rank_counts[rank] += 1
        full_suit_counts[int(card) % 4] += 1
        full_rank_mask |= 1 << rank
    full_max_rank = max(full_rank_counts)
    full_max_suit = max(full_suit_counts)
    output[41 + min(full_max_rank - 1, 3)] = 1.0
    output[45 + min(full_max_suit - 1, 4)] = 1.0
    made_straight = rank_mask_has_straight(full_rank_mask)
    output[50] = float(made_straight)
    output[51] = float(street != "river" and full_max_suit == 4)
    output[52] = float(street == "flop" and full_max_suit == 3)
    straight_outs = 0
    if street != "river" and not made_straight:
        straight_outs = sum(
            not (full_rank_mask & (1 << rank))
            and rank_mask_has_straight(full_rank_mask | (1 << rank))
            for rank in range(13)
        )
    output[53 + min(straight_outs, 2)] = 1.0
    output[56] = float(board_max_suit_count == 1)
    output[57] = float(board_max_suit_count == 2)
    output[58] = float(board_max_suit_count >= 3)
    output[59] = float(board_density >= 3)
    output[60] = float(board_density >= 4)
    output[61] = sum(int(card) // 4 >= 10 for card in board) / 5.0
    output[62] = sum(count > 0 for count in board_rank_counts) / 5.0
    output[63] = (
        float(board_max_rank_count >= 2)
        + float(board_max_suit_count >= 2)
        + float(board_density >= 3)
    ) / 3.0
    return output


def expand_state(state: dict[str, Any], depth_bb: float) -> np.ndarray:
    features = np.zeros(STATE_FEATURE_COUNT, dtype=np.float32)
    suit_map = canonical_suit_map(state["private_cards"], state["board"])
    for card in state["private_cards"]:
        features[canonical_card(int(card), suit_map)] = 1.0
    for card in state["board"]:
        features[52 + canonical_card(int(card), suit_map)] = 1.0
    street = STREETS.index(state["street"])
    actor = int(state["actor"])
    opponent = 1 - actor
    button = int(state["button"])
    one_hot(4, street, features, 104)
    one_hot(2, actor, features, 108)
    one_hot(2, button, features, 110)
    trajectory = state["trajectory"]
    if len(trajectory) > MAX_TRAJECTORY_ACTIONS:
        raise ValueError("trajectory exceeds pinned browser schema")
    features[112:124] = np.asarray(
        [
            state["pot_bb"] / depth_bb,
            state["stacks_bb"][actor] / depth_bb,
            state["stacks_bb"][opponent] / depth_bb,
            state["street_bets_bb"][actor] / depth_bb,
            state["street_bets_bb"][opponent] / depth_bb,
            state["total_committed_bb"][actor] / depth_bb,
            state["total_committed_bb"][opponent] / depth_bb,
            state["to_call_bb"] / depth_bb,
            state["last_full_raise_bb"] / depth_bb,
            1.0 if state["raise_reopened"] else 0.0,
            len(state["board"]) / 5.0,
            len(trajectory) / MAX_TRAJECTORY_ACTIONS,
        ],
        dtype=np.float32,
    )
    for index, history in enumerate(trajectory):
        offset = 124 + index * 15
        one_hot(2, int(history["actor"]), features, offset)
        one_hot(4, STREETS.index(history["street"]), features, offset + 2)
        one_hot(6, ACTIONS.index(history["kind"]), features, offset + 6)
        features[offset + 12] = history["amount_bb"] / depth_bb
        features[offset + 13] = (history.get("amount_to_bb") or 0.0) / depth_bb
        features[offset + 14] = history["pot_after_bb"] / depth_bb

    for card in state["private_cards"]:
        rank = int(card) // 4
        suit = suit_map[int(card) % 4]
        features[POKER_FEATURE_OFFSET + rank] += 0.5
        features[POKER_FEATURE_OFFSET + 26 + rank] += 0.25
        features[POKER_FEATURE_OFFSET + 43 + suit] += 1.0 / 7.0
    for card in state["board"]:
        rank = int(card) // 4
        suit = suit_map[int(card) % 4]
        features[POKER_FEATURE_OFFSET + 13 + rank] += 0.25
        features[POKER_FEATURE_OFFSET + 26 + rank] += 0.25
        features[POKER_FEATURE_OFFSET + 39 + suit] += 0.2
        features[POKER_FEATURE_OFFSET + 43 + suit] += 1.0 / 7.0
    features[POKER_FEATURE_OFFSET + 47] = float(
        int(state["private_cards"][0]) % 4 == int(state["private_cards"][1]) % 4
    )
    features[TEXTURE_FEATURE_OFFSET : TEXTURE_FEATURE_OFFSET + TEXTURE_FEATURE_COUNT] = (
        texture_features(state["private_cards"], state["board"], state["street"])
    )

    if not np.all(np.isfinite(features)):
        raise ValueError("expanded state vector contains non-finite values")
    return features


def expand_action(state: dict[str, Any], action: dict[str, Any], depth_bb: float) -> np.ndarray:
    features = np.zeros(ACTION_FEATURE_COUNT, dtype=np.float32)
    offset = 0
    kind = action["kind"]
    one_hot(6, ACTIONS.index(kind), features, offset)
    actor = int(state["actor"])
    current = float(state["street_bets_bb"][actor])
    highest = max(float(value) for value in state["street_bets_bb"])
    if kind == "call":
        target = highest
    else:
        target = float(action.get("amount_to_bb") or current)
    paid = max(0.0, target - current)
    total_pot = float(state["pot_bb"]) + sum(float(value) for value in state["street_bets_bb"])
    features[offset + 6] = target / depth_bb
    features[offset + 7] = paid / depth_bb
    features[offset + 8] = paid / max(total_pot, 1.0)
    if not np.all(np.isfinite(features)):
        raise ValueError("expanded action vector contains non-finite values")
    return features


def expand_state_action(state: dict[str, Any], action: dict[str, Any], depth_bb: float) -> np.ndarray:
    return np.concatenate((expand_state(state, depth_bb), expand_action(state, action, depth_bb)))


def load_jsonl_gzip(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        metadata = json.loads(next(stream))
        records = [json.loads(line) for line in stream if line.strip()]
    if metadata.get("schema") not in (
        "hu-neural-traversal-jsonl-v6",
        "hu-neural-traversal-jsonl-v7",
    ):
        raise ValueError("Rust traversal shard has an incompatible schema")
    if metadata.get("state_feature_count") != STATE_FEATURE_COUNT:
        raise ValueError("Rust and MLX state feature schemas differ")
    if metadata.get("action_feature_count") != ACTION_FEATURE_COUNT:
        raise ValueError("Rust and MLX action feature schemas differ")
    if metadata.get("state_feature_schema") != STATE_FEATURE_SCHEMA:
        raise ValueError("Rust and MLX suit-canonical feature schemas differ")
    if metadata.get("sampling_mode") not in ("external_sampling", "trajectory"):
        raise ValueError("Rust traversal shard has an unknown sampling mode")
    if metadata.get("records") != len(records):
        raise ValueError("Rust traversal shard record count is invalid")
    return metadata, records


def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def bootstrap_dcfr_plus_targets(
    prior_cumulative: np.ndarray,
    instantaneous: np.ndarray,
    round_number: int,
    alpha: float,
) -> np.ndarray:
    """Apply the Deep DCFR+ cumulative-advantage target recurrence."""
    if round_number <= 0 or alpha <= 0:
        raise ValueError("round number and Deep DCFR+ alpha must be positive")
    previous = float(round_number - 1)
    discount = 0.0 if previous == 0 else previous**alpha / (previous**alpha + 1.0)
    return np.maximum(prior_cumulative, 0.0) * discount + instantaneous


def ingest_records(
    records: list[dict[str, Any]],
    depth_bb: int,
    reservoirs: dict[str, ReplayReservoir | DecisionReservoir],
    models: dict[str, ActionScorer],
    round_number: int,
    advantage_alpha: float,
    reservoir_rng: random.Random,
) -> dict[str, list[tuple[np.ndarray, np.ndarray, float]]]:
    heldout: dict[str, list[tuple[np.ndarray, np.ndarray, float]]] = {
        "advantage_p0": [],
        "advantage_p1": [],
        "average_strategy": [],
        "value": [],
    }
    for record_index, record in enumerate(records):
        actions = record["actions"]
        targets = record["targets"]
        values = record.get("action_values_bb")
        standard_errors = record.get("action_value_standard_errors_bb")
        if len(actions) == 0 or len(actions) != len(targets):
            raise ValueError("training record has inconsistent action targets")
        kind = record["kind"]
        if kind not in ("advantage_p0", "advantage_p1", "average_strategy"):
            raise ValueError(f"unknown training sample kind: {kind}")
        street = STREETS.index(record["state"]["street"])
        state_features = expand_state(record["state"], depth_bb)
        action_features = np.stack(
            [expand_action(record["state"], action, depth_bb) for action in actions]
        )
        group_features = np.concatenate(
            (
                np.broadcast_to(state_features, (len(actions), STATE_FEATURE_COUNT)),
                action_features,
            ),
            axis=1,
        )
        feature_hashes = [
            hashlib.sha256(
                np.rint(np.asarray(features, dtype=np.float64) * 1_000_000.0)
                .astype("<i4")
                .tobytes()
            ).hexdigest()
            for features in group_features
        ]
        if feature_hashes != record.get("feature_sha256"):
            raise ValueError("Rust and MLX feature encoders disagree")
        group_targets = np.asarray(targets, dtype=np.float32)
        if kind in ("advantage_p0", "advantage_p1"):
            prior_cumulative = np.asarray(
                models[kind](mx.array(group_features)),
                dtype=np.float32,
            ).reshape(-1)
            group_targets = bootstrap_dcfr_plus_targets(
                prior_cumulative,
                group_targets / depth_bb,
                round_number,
                advantage_alpha,
            )
        holdout_group = record_index % 10 == 0
        if holdout_group:
            heldout[kind].append((group_features, group_targets, float(record["weight"])))
        else:
            grouped_reservoir = reservoirs[kind]
            if not isinstance(grouped_reservoir, DecisionReservoir):
                raise TypeError("strategy and advantage samples require grouped reservoirs")
            grouped_reservoir.add(
                state_features,
                action_features,
                group_targets,
                float(record["weight"]),
                street,
                reservoir_rng,
            )
        if values is not None:
            if len(values) != len(actions):
                raise ValueError("action-value target count is invalid")
            if standard_errors is not None and (
                len(standard_errors) != len(actions)
                or not np.all(np.isfinite(standard_errors))
                or np.any(np.asarray(standard_errors) < 0)
            ):
                raise ValueError("action-value standard-error targets are invalid")
            # Fit values in stack-normalized utility units so Huber regression
            # does not spend nearly every update in its saturated linear tail.
            # Exporters scale the value mean back to big blinds at boundaries.
            value_targets = np.asarray(values, dtype=np.float32) / depth_bb
            uncertainty_bb = (
                np.asarray(standard_errors, dtype=np.float32)
                if standard_errors is not None
                else np.full(
                    len(actions),
                    UNEVALUATED_STANDARD_ERROR_PRIOR_BB,
                    dtype=np.float32,
                )
            )
            raw_uncertainty_targets = np.asarray(
                [
                    inverse_softplus(
                        max(
                            float(standard_error)
                            - ACTION_VALUE_STANDARD_ERROR_FLOOR_BB,
                            1e-4,
                        )
                    )
                    for standard_error in uncertainty_bb
                ],
                dtype=np.float32,
            )
            combined_value_targets = np.stack(
                (value_targets, raw_uncertainty_targets), axis=1
            )
            if holdout_group:
                heldout["value"].append((group_features, combined_value_targets, 1.0))
            else:
                for action_index, features in enumerate(group_features):
                    value_reservoir = reservoirs["value"]
                    if not isinstance(value_reservoir, ReplayReservoir):
                        raise TypeError("value samples require an action reservoir")
                    value_reservoir.add(
                        features,
                        combined_value_targets[action_index],
                        1.0,
                        street,
                        reservoir_rng,
                    )
    return heldout


def make_compiled_step(model: ActionScorer, optimizer: optim.Optimizer):
    def loss_fn(
        active_model: ActionScorer,
        features: mx.array,
        targets: mx.array,
        weights: mx.array,
    ) -> mx.array:
        predictions = active_model(features)
        error = predictions - targets
        absolute = mx.abs(error)
        huber = mx.where(absolute < 1.0, 0.5 * mx.square(error), absolute - 0.5)
        normalized_weights = weights / mx.maximum(mx.mean(weights), mx.array(1e-8))
        return mx.mean(huber * normalized_weights)

    value_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(features: mx.array, targets: mx.array, weights: mx.array) -> mx.array:
        loss, gradients = value_and_grad(model, features, targets, weights)
        optimizer.update(model, gradients)
        return loss

    return step


def make_compiled_policy_step(model: ActionScorer, optimizer: optim.Optimizer):
    def loss_fn(
        active_model: ActionScorer,
        features: mx.array,
        targets: mx.array,
        masks: mx.array,
        weights: mx.array,
    ) -> mx.array:
        batch_size = features.shape[0]
        logits = active_model(features.reshape((-1, INPUT_FEATURE_COUNT))).reshape(
            (batch_size, MAX_POLICY_ACTIONS)
        )
        masked_logits = mx.where(masks > 0, logits, mx.array(-1e9))
        log_probabilities = masked_logits - mx.logsumexp(
            masked_logits,
            axis=1,
            keepdims=True,
        )
        per_decision = -mx.sum(targets * log_probabilities * masks, axis=1, keepdims=True)
        normalized_weights = weights / mx.maximum(mx.mean(weights), mx.array(1e-8))
        return mx.mean(per_decision * normalized_weights)

    value_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(
        features: mx.array,
        targets: mx.array,
        masks: mx.array,
        weights: mx.array,
    ) -> mx.array:
        loss, gradients = value_and_grad(model, features, targets, masks, weights)
        optimizer.update(model, gradients)
        return loss

    return step


def make_compiled_group_regression_step(model: ActionScorer, optimizer: optim.Optimizer):
    def loss_fn(
        active_model: ActionScorer,
        features: mx.array,
        targets: mx.array,
        masks: mx.array,
        weights: mx.array,
    ) -> mx.array:
        batch_size = features.shape[0]
        predictions = active_model(features.reshape((-1, INPUT_FEATURE_COUNT))).reshape(
            (batch_size, MAX_POLICY_ACTIONS)
        )
        error = predictions - targets
        action_counts = mx.maximum(mx.sum(masks, axis=1, keepdims=True), mx.array(1.0))
        per_decision = mx.sum(mx.square(error) * masks, axis=1, keepdims=True) / action_counts
        normalized_weights = weights / mx.maximum(mx.mean(weights), mx.array(1e-8))
        return mx.mean(per_decision * normalized_weights)

    value_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(
        features: mx.array,
        targets: mx.array,
        masks: mx.array,
        weights: mx.array,
    ) -> mx.array:
        loss, gradients = value_and_grad(model, features, targets, masks, weights)
        optimizer.update(model, gradients)
        return loss

    return step


def train_reservoir(
    model: ActionScorer,
    optimizer: optim.Optimizer,
    step: Any,
    reservoir: ReplayReservoir,
    steps: int,
    batch_size: int,
    rng: np.random.Generator,
    street_proposal: tuple[float, float, float, float] | None,
) -> float | None:
    if reservoir.size == 0 or steps == 0:
        return None
    losses: list[float] = []
    for _ in range(steps):
        features, targets, weights = reservoir.sample(
            batch_size,
            rng,
            street_proposal,
        )
        loss = step(features, targets, weights)
        mx.eval(loss, model.parameters(), optimizer.state)
        losses.append(float(loss.item()))
    return sum(losses) / len(losses)


def train_grouped_reservoir(
    model: ActionScorer,
    optimizer: optim.Optimizer,
    step: Any,
    reservoir: DecisionReservoir,
    steps: int,
    batch_size: int,
    rng: np.random.Generator,
    street_proposal: tuple[float, float, float, float] | None,
) -> float | None:
    if reservoir.size == 0 or steps == 0:
        return None
    losses: list[float] = []
    for _ in range(steps):
        features, targets, masks, weights = reservoir.sample(
            batch_size,
            rng,
            street_proposal,
        )
        loss = step(features, targets, masks, weights)
        mx.eval(loss, model.parameters(), optimizer.state)
        losses.append(float(loss.item()))
    return sum(losses) / len(losses)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


def evaluate_models(
    models: dict[str, ActionScorer],
    heldout: dict[str, list[tuple[np.ndarray, np.ndarray, float]]],
    depth_bb: int,
) -> dict[str, float | int | None]:
    metrics: dict[str, float | int | None] = {}
    for kind in ("advantage_p0", "advantage_p1"):
        squared: list[float] = []
        for features, targets, _ in heldout[kind][:4096]:
            predicted = np.asarray(models[kind](mx.array(features))).reshape(-1)
            squared.extend(np.square(predicted - targets).tolist())
        metrics[f"{kind}_rmse_normalized"] = math.sqrt(sum(squared) / len(squared)) if squared else None
        metrics[f"{kind}_actions"] = len(squared)

    absolute_policy_errors: list[float] = []
    policy_kls: list[float] = []
    for features, targets, _ in heldout["average_strategy"][:4096]:
        logits = np.asarray(models["average_strategy"](mx.array(features))).reshape(-1)
        predicted = softmax(logits)
        expected = np.asarray(targets, dtype=np.float64)
        expected /= expected.sum()
        absolute_policy_errors.extend(np.abs(predicted - expected).tolist())
        policy_kls.append(float(np.sum(expected * np.log((expected + 1e-9) / (predicted + 1e-9)))))
    metrics["current_strategy_snapshot_mae"] = (
        sum(absolute_policy_errors) / len(absolute_policy_errors) if absolute_policy_errors else None
    )
    metrics["current_strategy_snapshot_kl"] = (
        sum(policy_kls) / len(policy_kls) if policy_kls else None
    )
    metrics["current_strategy_snapshot_actions"] = len(absolute_policy_errors)

    value_squared: list[float] = []
    uncertainty_absolute: list[float] = []
    for features, targets, _ in heldout["value"][:4096]:
        predicted = np.asarray(models["value"](mx.array(features)))
        value_squared.extend(np.square(predicted[:, 0] - targets[:, 0]).tolist())
        predicted_standard_error = ACTION_VALUE_STANDARD_ERROR_FLOOR_BB + np.logaddexp(
            0.0, predicted[:, 1]
        )
        target_standard_error = ACTION_VALUE_STANDARD_ERROR_FLOOR_BB + np.logaddexp(
            0.0, targets[:, 1]
        )
        uncertainty_absolute.extend(
            np.abs(predicted_standard_error - target_standard_error).tolist()
        )
    metrics["action_value_rmse_bb"] = (
        math.sqrt(sum(value_squared) / len(value_squared)) * depth_bb
        if value_squared
        else None
    )
    metrics["action_value_standard_error_mae_bb"] = (
        sum(uncertainty_absolute) / len(uncertainty_absolute)
        if uncertainty_absolute
        else None
    )
    metrics["value_actions"] = len(value_squared)
    return metrics


def save_optimizer(optimizer: optim.Optimizer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    mx.savez(temporary, **dict(tree_flatten(optimizer.state)))
    os.replace(temporary, path)


def load_optimizer(optimizer: optim.Optimizer, path: Path) -> None:
    arrays = mx.load(path)
    optimizer.state = tree_unflatten(list(arrays.items()))


def save_model(model: ActionScorer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.safetensors")
    model.save_weights(str(temporary))
    os.replace(temporary, path)


def linear_layers(model: ActionScorer) -> list[nn.Linear]:
    return [model.layers.layers[index] for index in (0, 2, 4)]


def scorer_json(
    model: ActionScorer,
    output_index: int | None = None,
    output_scale: float = 1.0,
) -> dict[str, Any]:
    layers: list[dict[str, Any]] = []
    linears = linear_layers(model)
    for index, layer in enumerate(linears):
        weights = np.asarray(layer.weight, dtype=np.float32)
        biases = np.asarray(layer.bias, dtype=np.float32)
        if index == len(linears) - 1 and output_index is not None:
            if output_index < 0 or output_index >= weights.shape[0]:
                raise ValueError("scorer output index is outside the final layer")
            weights = weights[output_index : output_index + 1]
            biases = biases[output_index : output_index + 1]
        if index == len(linears) - 1 and output_scale != 1.0:
            weights = weights * output_scale
            biases = biases * output_scale
        layers.append(
            {
                "input_size": int(weights.shape[1]),
                "output_size": int(weights.shape[0]),
                "activation": "linear" if index == len(linears) - 1 else "relu",
                "weights": weights.reshape(-1).tolist(),
                "biases": biases.tolist(),
            }
        )
    return {"layers": layers}


def export_traversal_networks(
    models: dict[str, ActionScorer],
    path: Path,
    variance_baseline_scale: float,
    depth_bb: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    source = {
        "schema": NETWORK_SCHEMA,
        "input_size": INPUT_FEATURE_COUNT,
        "strategy_transform": "regret_matching",
        "networks": [scorer_json(models["advantage_p0"]), scorer_json(models["advantage_p1"])],
        "sampling_baseline": scorer_json(
            models["value"],
            output_index=0,
            output_scale=depth_bb,
        ),
        "sampling_baseline_scale": variance_baseline_scale,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(source, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)


def append_runtime_network(
    parameters: list[float],
    model: ActionScorer,
    output_scales: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    descriptors: list[dict[str, Any]] = []
    linears = linear_layers(model)
    for index, layer in enumerate(linears):
        weights = np.asarray(layer.weight, dtype=np.float32)
        biases = np.asarray(layer.bias, dtype=np.float32)
        if index == len(linears) - 1 and output_scales is not None:
            if len(output_scales) != weights.shape[0]:
                raise ValueError("runtime output scales do not match the final layer")
            scales = np.asarray(output_scales, dtype=np.float32)
            weights = weights * scales[:, None]
            biases = biases * scales
        weight_offset = len(parameters)
        parameters.extend(weights.reshape(-1).tolist())
        bias_offset = len(parameters)
        parameters.extend(biases.tolist())
        descriptors.append(
            {
                "inputSize": int(weights.shape[1]),
                "outputSize": int(weights.shape[0]),
                "activation": "linear" if index == len(linears) - 1 else "relu",
                "weightOffset": weight_offset,
                "biasOffset": bias_offset,
            }
        )
    return {"layers": descriptors}


def response_from_baseline(
    baseline: ActionScorer,
    hidden_sizes: tuple[int, int],
) -> ActionScorer:
    response = ActionScorer(INPUT_FEATURE_COUNT + PROFILE_FEATURE_COUNT, hidden_sizes)
    baseline_layers = linear_layers(baseline)
    response_layers = linear_layers(response)
    first = np.asarray(baseline_layers[0].weight, dtype=np.float32)
    padded = np.zeros((first.shape[0], first.shape[1] + PROFILE_FEATURE_COUNT), dtype=np.float32)
    padded[:, : first.shape[1]] = first
    response_layers[0].weight = mx.array(padded)
    response_layers[0].bias = mx.array(np.asarray(baseline_layers[0].bias, dtype=np.float32))
    for source, destination in zip(baseline_layers[1:], response_layers[1:]):
        destination.weight = mx.array(np.asarray(source.weight, dtype=np.float32))
        destination.bias = mx.array(np.asarray(source.bias, dtype=np.float32))
    mx.eval(response.parameters())
    return response


def camel_action_abstraction(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "openSizesBb": source["open_sizes_bb"],
        "limpRaiseSizesBb": source["limp_raise_sizes_bb"],
        "threeBetSizesBb": source["three_bet_sizes_bb"],
        "fourBetSizesBb": source["four_bet_sizes_bb"],
        "deeperRaisePotFractions": source["deeper_raise_pot_fractions"],
        "preflopRaiseCap": source["preflop_raise_cap"],
        "flopBetPotFractions": source["flop_bet_pot_fractions"],
        "turnRiverBetPotFractions": source["turn_river_bet_pot_fractions"],
        "postflopRaisePotFractions": source["postflop_raise_pot_fractions"],
        "postflopRaiseCap": source["postflop_raise_cap"],
        "includeAllIn": source["include_all_in"],
    }


def export_teacher_snapshot(
    models: dict[str, ActionScorer],
    artifact_dir: Path,
    config: RunConfig,
    round_number: int,
) -> Path:
    """Retain a bounded, offline SD-CFR teacher candidate at artifact rounds."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    teacher_models: dict[str, dict[str, Any]] = {}
    for player in ("advantage_p0", "advantage_p1"):
        teacher_path = artifact_dir / f"{player}.safetensors"
        save_model(models[player], teacher_path)
        payload = teacher_path.read_bytes()
        teacher_models[player] = {
            "file": teacher_path.name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    completed_traversals = round_number * config.traversals_per_round
    manifest_path = artifact_dir / "teacher-snapshot.json"
    atomic_json(
        manifest_path,
        {
            "schema": "hu-sparse-sd-cfr-teacher-v1",
            "purpose": "offline_direct_average_policy_comparison_only",
            "round": round_number,
            "completedTraversals": completed_traversals,
            "strategyWeight": float(completed_traversals**2),
            "strategyTransform": "regret_matching",
            "networkSchema": NETWORK_SCHEMA,
            "models": teacher_models,
        },
    )
    return manifest_path


def export_browser_artifact(
    root: Path,
    run_dir: Path,
    models: dict[str, ActionScorer],
    config: RunConfig,
    round_number: int,
    action_abstraction: dict[str, Any],
) -> tuple[Path, Path]:
    config_fingerprint = hashlib.sha256(
        json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    model_version = (
        f"deep-dcfr-plus-v12-{config.depth_bb}bb-cfg{config_fingerprint}"
        f"-seed{config.seed}"
        f"-r{round_number:06d}-experimental"
    )
    response = response_from_baseline(models["average_strategy"], config.hidden_sizes)
    parameters: list[float] = []
    baseline_descriptor = append_runtime_network(parameters, models["average_strategy"])
    response_descriptor = append_runtime_network(parameters, response)
    value_descriptor = append_runtime_network(
        parameters,
        models["value"],
        (float(config.depth_bb), 1.0),
    )
    metadata = {
        "schemaVersion": 1,
        "kind": "deep-cfr-baseline-response",
        "modelVersion": model_version,
        "depthBb": config.depth_bb,
        "stateFeatureSchema": STATE_FEATURE_SCHEMA,
        "stateFeatureCount": STATE_FEATURE_COUNT,
        "actionFeatureSchema": "hu-cash-legal-action-v1",
        "actionFeatureCount": ACTION_FEATURE_COUNT,
        "opponentProfileSchema": "local-opponent-profile-v1",
        "opponentProfileFeatureCount": PROFILE_FEATURE_COUNT,
        "parameterCount": len(parameters),
        "trainingAlgorithm": ADVANTAGE_UPDATE,
        "advantageAlpha": config.advantage_alpha,
        "samplingVarianceReduction": {
            "kind": "action_dependent_control_variate",
            "scale": config.variance_baseline_scale,
            "activationRound": 2,
        },
        "valueTrainingUnits": "effective_stack_fraction_exported_as_bb",
        "optimizerSchedule": {
            "kind": "linear_after_round"
            if config.learning_rate_final is not None
            else "constant",
            "initialLearningRate": config.learning_rate,
            "finalLearningRate": config.learning_rate_final,
            "decayStartRound": config.learning_rate_decay_start_round,
            "decayEndRound": config.learning_rate_decay_end_round,
        },
        "textureFeatureInitialization": "zero_first_layer_columns",
        "replaySampling": {
            "kind": "four_street_stratified_importance_corrected"
            if config.replay_street_proposal is not None
            else "authentic_uniform",
            "streetProposal": list(config.replay_street_proposal)
            if config.replay_street_proposal is not None
            else None,
            "importanceCorrection": "empirical_street_probability_over_proposal_probability"
            if config.replay_street_proposal is not None
            else None,
        },
        "valueTargetSampling": {
            "kind": "independent_external_sampling_average",
            "samplesPerAction": config.value_rollouts_per_action,
            "primaryTraversalSampleReused": True,
            "extraRandomness": "canonical_state_action_seeded",
        },
        "valueUncertaintyTraining": {
            "kind": "sample_standard_error"
            if config.value_rollouts_per_action >= 2
            else "unevaluated_prior",
            "standardErrorFloorBb": ACTION_VALUE_STANDARD_ERROR_FLOOR_BB,
            "unevaluatedPriorBb": UNEVALUATED_STANDARD_ERROR_PRIOR_BB,
        },
        "actionAbstraction": camel_action_abstraction(action_abstraction),
        "adaptation": {
            "minimumObservations": 50,
            "fullConfidenceObservations": 250,
            "maximumResponseWeight": 0.5,
        },
        "valueCalibration": {
            "standardErrorFloorBb": ACTION_VALUE_STANDARD_ERROR_FLOOR_BB,
            "highConfidenceMaximumBb": 0.02,
        },
        "networks": {
            "baselinePolicy": baseline_descriptor,
            "exploitResponse": response_descriptor,
            "baselineActionValue": value_descriptor,
        },
    }
    artifact_dir = run_dir / "artifacts" / f"round-{round_number:06d}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    export_teacher_snapshot(models, artifact_dir, config, round_number)
    source_path = artifact_dir / "model-source.json"
    binary_path = artifact_dir / "model.bin"
    source_path.write_text(
        json.dumps({"metadata": metadata, "parameters": parameters}, separators=(",", ":")),
        encoding="utf-8",
    )
    relative_url = f"/models/practice/{model_version}/{config.depth_bb}bb.bin"
    result = subprocess.run(
        [
            "node",
            str(root / "scripts/policy/export-neural-artifact.mjs"),
            "--input",
            str(source_path),
            "--output",
            str(binary_path),
            "--url",
            relative_url,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    (artifact_dir / "descriptor.json").write_text(result.stdout, encoding="utf-8")
    return source_path, binary_path


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def migrate_legacy_resume_state(
    state: dict[str, Any], config: RunConfig, config_hash: str
) -> tuple[dict[str, Any], bool]:
    """Upgrade the exact v9 leading run without weakening config pinning."""
    if state.get("schema") == RUN_SCHEMA:
        return state, False
    if state.get("schema") != LEGACY_RESUME_SCHEMA:
        raise RuntimeError("resume configuration differs from the existing neural run")
    legacy_config = copy.deepcopy(state.get("config", {}))
    if legacy_config.get("schema") != LEGACY_RESUME_SCHEMA:
        raise RuntimeError("legacy neural run has an inconsistent config schema")
    legacy_config["schema"] = RUN_SCHEMA
    legacy_config.setdefault("learning_rate_final", None)
    legacy_config.setdefault("learning_rate_decay_start_round", None)
    legacy_config.setdefault("learning_rate_decay_end_round", None)
    legacy_config.setdefault("replay_street_proposal", None)
    legacy_config.setdefault("value_rollouts_per_action", 1)
    expected_config = json.loads(json.dumps(asdict(config)))
    if legacy_config != expected_config:
        raise RuntimeError("resume configuration differs from the existing neural run")
    migrated = copy.deepcopy(state)
    migrated["schema"] = RUN_SCHEMA
    migrated["config"] = expected_config
    migrated["config_hash"] = config_hash
    migrated["migrations"] = migrated.get("migrations", []) + [
        {
            "from": LEGACY_RESUME_SCHEMA,
            "to": RUN_SCHEMA,
            "kind": "additive_default_fields_only",
        }
    ]
    return migrated, True


def backfill_street_file(
    source_path: Path,
    street_path: Path,
    capacity: int,
    size: int,
    feature_width: int,
) -> bool:
    """Recover v9 replay street ids from the pinned state one-hot features."""
    if street_path.exists():
        return False
    expected_bytes = capacity * feature_width * np.dtype(np.float16).itemsize
    if not source_path.exists() or source_path.stat().st_size != expected_bytes:
        raise RuntimeError(f"cannot reconstruct replay streets from {source_path}")
    if size < 0 or size > capacity:
        raise RuntimeError("legacy replay reservoir size is invalid")
    source = np.memmap(
        source_path,
        dtype=np.float16,
        mode="r",
        shape=(capacity, feature_width),
    )
    temporary = street_path.with_suffix(".tmp")
    streets = np.memmap(temporary, dtype=np.uint8, mode="w+", shape=(capacity,))
    streets[:] = 0
    for start in range(0, size, 8192):
        stop = min(start + 8192, size)
        encoded = np.asarray(source[start:stop, 104:108], dtype=np.float32)
        if not np.all(np.sum(encoded, axis=1) == 1.0) or not np.all(
            (encoded == 0.0) | (encoded == 1.0)
        ):
            raise RuntimeError("legacy replay contains invalid street one-hot features")
        streets[start:stop] = np.argmax(encoded, axis=1).astype(np.uint8)
    streets.flush()
    del streets
    del source
    os.replace(temporary, street_path)
    return True


def backfill_legacy_replay_streets(run_dir: Path, state: dict[str, Any]) -> None:
    if int(state.get("completed_rounds", 0)) == 0:
        return
    migrated_from_v9 = any(
        migration.get("from") == LEGACY_RESUME_SCHEMA
        for migration in state.get("migrations", [])
    )
    replay_dir = run_dir / "replay"
    summaries = state.get("reservoirs", {})
    missing = []
    for name in ("value", "advantage_p0", "advantage_p1", "average_strategy"):
        street_path = replay_dir / f"{name}.street.u8"
        if not street_path.exists():
            missing.append(name)
    if missing and not migrated_from_v9:
        raise RuntimeError("replay reservoir is missing pinned street metadata")
    capacity = int(state["config"]["reservoir_capacity"])
    for name in missing:
        is_value = name == "value"
        source_path = replay_dir / (
            f"{name}.features.f16" if is_value else f"{name}.states.f16"
        )
        backfill_street_file(
            source_path,
            replay_dir / f"{name}.street.u8",
            capacity,
            int(summaries[name]["size"]),
            INPUT_FEATURE_COUNT if is_value else STATE_FEATURE_COUNT,
        )


def nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(nested_tuple(item) for item in value)
    return value


def peak_resident_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage) if sys.platform == "darwin" else int(usage * 1024)


def scheduled_learning_rate(config: RunConfig, round_number: int) -> float:
    final = config.learning_rate_final
    start = config.learning_rate_decay_start_round
    end = config.learning_rate_decay_end_round
    if final is None:
        return config.learning_rate
    if start is None or end is None:
        raise ValueError("a final learning rate requires decay start and end rounds")
    if round_number <= start:
        return config.learning_rate
    if round_number >= end:
        return final
    progress = (round_number - start) / (end - start)
    return config.learning_rate + (final - config.learning_rate) * progress


def set_optimizer_learning_rate(
    optimizers: dict[str, optim.Optimizer], learning_rate: float
) -> None:
    for optimizer in optimizers.values():
        optimizer.learning_rate = learning_rate
    mx.eval(*(optimizer.state for optimizer in optimizers.values()))


def initialize_models(config: RunConfig) -> tuple[dict[str, ActionScorer], dict[str, optim.Optimizer]]:
    names_outputs = {
        "advantage_p0": 1,
        "advantage_p1": 1,
        "average_strategy": 1,
        "value": 2,
    }
    models: dict[str, ActionScorer] = {}
    optimizers: dict[str, optim.Optimizer] = {}
    for offset, (name, output_size) in enumerate(names_outputs.items()):
        mx.random.seed(config.seed + offset * 1009)
        model = ActionScorer(INPUT_FEATURE_COUNT, config.hidden_sizes, output_size)
        first_layer = linear_layers(model)[0]
        first_weights = np.asarray(first_layer.weight, dtype=np.float32)
        first_weights[:, TEXTURE_FEATURE_OFFSET : TEXTURE_FEATURE_OFFSET + TEXTURE_FEATURE_COUNT] = 0
        first_layer.weight = mx.array(first_weights)
        optimizer = optim.AdamW(learning_rate=config.learning_rate, weight_decay=1e-5)
        optimizer.init(model.trainable_parameters())
        mx.eval(model.parameters(), optimizer.state)
        models[name] = model
        optimizers[name] = optimizer
    return models, optimizers


def load_checkpoint(
    run_dir: Path,
    models: dict[str, ActionScorer],
    optimizers: dict[str, optim.Optimizer],
    state: dict[str, Any],
) -> None:
    checkpoint_name = state.get("checkpoint")
    if not checkpoint_name:
        raise RuntimeError("completed run state does not name an atomic checkpoint")
    checkpoint = run_dir / checkpoint_name
    for name, model in models.items():
        model.load_weights(str(checkpoint / f"{name}.safetensors"))
        load_optimizer(optimizers[name], checkpoint / f"{name}.optimizer.npz")
    mx.eval(*(model.parameters() for model in models.values()))


def save_checkpoint(
    run_dir: Path,
    models: dict[str, ActionScorer],
    optimizers: dict[str, optim.Optimizer],
    round_number: int,
    variance_baseline_scale: float,
    depth_bb: int,
) -> Path:
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoints / f"round-{round_number:06d}"
    staging = checkpoints / f".round-{round_number:06d}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    if checkpoint.exists():
        shutil.rmtree(checkpoint)
    staging.mkdir()
    for name, model in models.items():
        save_model(model, staging / f"{name}.safetensors")
        save_optimizer(optimizers[name], staging / f"{name}.optimizer.npz")
    export_traversal_networks(
        models,
        staging / "traversal-networks.json",
        variance_baseline_scale,
        depth_bb,
    )
    os.replace(staging, checkpoint)
    return checkpoint


def request_stop(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(
        json.dumps(
            {
                "event": "stop_requested",
                "signal": signum,
                "behavior": "finish_and_checkpoint_current_round",
            }
        ),
        flush=True,
    )


def build_rust(root: Path) -> Path:
    subprocess.run(
        ["cargo", "build", "--release", "--manifest-path", "preflop-solver/Cargo.toml"],
        cwd=root,
        check=True,
    )
    return root / "preflop-solver/target/release/preflop-solver"


def generate_shard(
    binary: Path,
    run_dir: Path,
    config: RunConfig,
    round_number: int,
    global_iteration: int,
    traversal_networks: Path | None,
) -> Path:
    shard_dir = run_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard = shard_dir / f"round-{round_number:06d}.jsonl.gz"
    command = [
        str(binary),
        "neural-samples",
        "--effective-stack-bb",
        str(config.depth_bb),
        "--traversals",
        str(config.traversals_per_round),
        "--start-iteration",
        str(global_iteration),
        "--seed",
        str(config.seed ^ (round_number * 0x9E3779B1)),
        "--max-records",
        str(max(config.reservoir_capacity, 10_000)),
        "--preflop-runout-samples",
        str(config.preflop_runout_samples),
        "--flop-runout-samples",
        str(config.flop_runout_samples),
        "--value-rollouts-per-action",
        str(config.value_rollouts_per_action),
        "--output",
        str(shard),
    ]
    if not config.exact_turn_rivers:
        command.append("--sample-turn-rivers")
    if config.compact_serving_grid:
        command.append("--compact-serving-grid")
    if traversal_networks is not None:
        command.extend(("--networks", str(traversal_networks)))
    subprocess.run(command, check=True, cwd=run_dir, start_new_session=True)
    return shard


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--depth-bb", type=int, choices=(20, 50, 100), default=20)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument(
        "--target-round",
        type=int,
        help="stop at this total round; safer than --rounds when resuming",
    )
    parser.add_argument("--max-minutes", type=float)
    parser.add_argument(
        "--traversals-per-round",
        type=int,
        default=LEADING_20BB_PROFILE["traversals_per_round"],
    )
    parser.add_argument(
        "--reservoir-capacity",
        type=int,
        default=LEADING_20BB_PROFILE["reservoir_capacity"],
    )
    parser.add_argument("--hidden-sizes", default=LEADING_20BB_PROFILE["hidden_sizes"])
    parser.add_argument(
        "--batch-size", type=int, default=LEADING_20BB_PROFILE["batch_size"]
    )
    parser.add_argument(
        "--steps-per-round",
        type=int,
        default=LEADING_20BB_PROFILE["steps_per_round"],
    )
    parser.add_argument(
        "--advantage-alpha",
        type=float,
        default=LEADING_20BB_PROFILE["advantage_alpha"],
        help="positive cumulative-advantage discount exponent for Deep DCFR+",
    )
    parser.add_argument(
        "--variance-baseline-scale",
        type=float,
        default=LEADING_20BB_PROFILE["variance_baseline_scale"],
        help="action-value control-variate scale in [0,1] for sampled opponent nodes",
    )
    parser.add_argument(
        "--replay-street-proposal",
        default="authentic",
        help="'authentic' or importance-corrected preflop,flop,turn,river probabilities",
    )
    parser.add_argument(
        "--artifact-every",
        type=int,
        default=LEADING_20BB_PROFILE["artifact_every"],
        help="export a browser artifact every N rounds; the final round is always exported",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=LEADING_20BB_PROFILE["learning_rate"]
    )
    parser.add_argument("--learning-rate-final", type=float)
    parser.add_argument("--learning-rate-decay-start-round", type=int)
    parser.add_argument("--learning-rate-decay-end-round", type=int)
    parser.add_argument(
        "--preflop-runout-samples",
        type=int,
        default=LEADING_20BB_PROFILE["preflop_runout_samples"],
    )
    parser.add_argument(
        "--flop-runout-samples",
        type=int,
        default=LEADING_20BB_PROFILE["flop_runout_samples"],
    )
    parser.add_argument(
        "--value-rollouts-per-action",
        type=int,
        default=LEADING_20BB_PROFILE["value_rollouts_per_action"],
        help="independent external samples averaged into each action-value target",
    )
    parser.add_argument("--sample-turn-rivers", action="store_true")
    parser.add_argument("--compact-serving-grid", action="store_true")
    parser.add_argument("--keep-shards", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = False
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    args = parse_args()
    street_proposal = (
        None
        if args.replay_street_proposal == "authentic"
        else tuple(float(value) for value in args.replay_street_proposal.split(","))
    )
    if street_proposal is not None and len(street_proposal) != len(STREETS):
        raise ValueError("--replay-street-proposal requires four probabilities")
    if (
        args.rounds <= 0
        or (args.target_round is not None and args.target_round <= 0)
        or args.traversals_per_round <= 0
        or args.steps_per_round < 0
        or args.advantage_alpha <= 0
        or args.value_rollouts_per_action <= 0
        or args.learning_rate <= 0
        or not 0 <= args.variance_baseline_scale <= 1
        or (
            street_proposal is not None
            and (
                any(value <= 0 for value in street_proposal)
                or not np.isclose(sum(street_proposal), 1.0)
            )
        )
        or args.artifact_every <= 0
    ):
        raise ValueError("training counts or replay/control parameters are invalid")
    decay_values = (
        args.learning_rate_final,
        args.learning_rate_decay_start_round,
        args.learning_rate_decay_end_round,
    )
    if any(value is not None for value in decay_values) and (
        any(value is None for value in decay_values)
        or args.learning_rate_final <= 0
        or args.learning_rate_decay_start_round < 1
        or args.learning_rate_decay_end_round <= args.learning_rate_decay_start_round
    ):
        raise ValueError("learning-rate decay requires a positive final rate and start < end")
    if args.reservoir_capacity < 100 or args.batch_size <= 0:
        raise ValueError("reservoir capacity and batch size are too small")
    hidden = tuple(int(value) for value in args.hidden_sizes.split(","))
    if len(hidden) != 2 or min(hidden) <= 0:
        raise ValueError("--hidden-sizes must contain two positive integers")
    config = RunConfig(
        schema=RUN_SCHEMA,
        depth_bb=args.depth_bb,
        seed=args.seed,
        reservoir_capacity=args.reservoir_capacity,
        hidden_sizes=(hidden[0], hidden[1]),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        learning_rate_final=args.learning_rate_final,
        learning_rate_decay_start_round=args.learning_rate_decay_start_round,
        learning_rate_decay_end_round=args.learning_rate_decay_end_round,
        traversals_per_round=args.traversals_per_round,
        steps_per_round=args.steps_per_round,
        advantage_alpha=args.advantage_alpha,
        variance_baseline_scale=args.variance_baseline_scale,
        replay_street_proposal=None
        if street_proposal is None
        else (
            street_proposal[0],
            street_proposal[1],
            street_proposal[2],
            street_proposal[3],
        ),
        value_rollouts_per_action=args.value_rollouts_per_action,
        artifact_every=args.artifact_every,
        preflop_runout_samples=args.preflop_runout_samples,
        flop_runout_samples=args.flop_runout_samples,
        exact_turn_rivers=not args.sample_turn_rivers,
        compact_serving_grid=args.compact_serving_grid,
    )
    root = Path(__file__).resolve().parents[2]
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.json"
    config_hash = hashlib.sha256(
        json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    state: dict[str, Any]
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state, migrated = migrate_legacy_resume_state(state, config, config_hash)
        if state.get("config_hash") != config_hash:
            raise RuntimeError("resume configuration differs from the existing neural run")
        if migrated:
            atomic_json(state_path, state)
    else:
        state = {
            "schema": RUN_SCHEMA,
            "config": asdict(config),
            "config_hash": config_hash,
            "completed_rounds": 0,
            "completed_traversals": 0,
            "reservoirs": {},
            "metrics": [],
            "artifacts": [],
            "checkpoint": None,
            "status": "training",
        }
        atomic_json(state_path, state)

    backfill_legacy_replay_streets(run_dir, state)

    models, optimizers = initialize_models(config)
    if int(state["completed_rounds"]) > 0:
        load_checkpoint(run_dir, models, optimizers, state)
    compiled_steps = {
        name: (
            make_compiled_policy_step(models[name], optimizers[name])
            if name == "average_strategy"
            else make_compiled_group_regression_step(models[name], optimizers[name])
            if name in ("advantage_p0", "advantage_p1")
            else make_compiled_step(models[name], optimizers[name])
        )
        for name in models
    }

    replay_dir = run_dir / "replay"
    reservoirs: dict[str, ReplayReservoir | DecisionReservoir] = {
        "value": ReplayReservoir(
            replay_dir,
            "value",
            config.reservoir_capacity,
            2,
            size=int(state.get("reservoirs", {}).get("value", {}).get("size", 0)),
            seen=int(state.get("reservoirs", {}).get("value", {}).get("seen", 0)),
        )
    }
    for name in ("advantage_p0", "advantage_p1", "average_strategy"):
        reservoirs[name] = DecisionReservoir(
            replay_dir,
            name,
            config.reservoir_capacity,
            normalize_targets=name == "average_strategy",
            size=int(state.get("reservoirs", {}).get(name, {}).get("size", 0)),
            seen=int(state.get("reservoirs", {}).get(name, {}).get("seen", 0)),
        )
    reservoir_rng = random.Random(config.seed ^ 0xD1B54A32D192ED03)
    if state.get("reservoir_rng_state") is not None:
        reservoir_rng.setstate(nested_tuple(state["reservoir_rng_state"]))
    np_rng = np.random.default_rng(config.seed ^ 0xA24BAED4963EE407)
    if state.get("numpy_rng_state") is not None:
        np_rng.bit_generator.state = state["numpy_rng_state"]

    binary = build_rust(root)
    traversal_networks = (
        run_dir / state["checkpoint"] / "traversal-networks.json"
        if state.get("checkpoint")
        else None
    )
    started = time.monotonic()
    deadline = started + args.max_minutes * 60 if args.max_minutes is not None else None
    first_round = int(state["completed_rounds"]) + 1
    last_round = (
        args.target_round
        if args.target_round is not None
        else int(state["completed_rounds"]) + args.rounds
    )
    if last_round < first_round:
        raise ValueError("target round must exceed the completed round")
    action_abstraction: dict[str, Any] | None = None
    exported_rounds = {int(artifact["round"]) for artifact in state.get("artifacts", [])}

    for round_number in range(first_round, last_round + 1):
        if STOP_REQUESTED or (deadline is not None and time.monotonic() >= deadline):
            break
        round_started = time.monotonic()
        round_learning_rate = scheduled_learning_rate(config, round_number)
        set_optimizer_learning_rate(optimizers, round_learning_rate)
        shard = generate_shard(
            binary,
            run_dir,
            config,
            round_number,
            int(state["completed_traversals"]),
            traversal_networks,
        )
        metadata, records = load_jsonl_gzip(shard)
        if int(metadata["depth_bb"]) != config.depth_bb:
            raise RuntimeError("sample shard depth changed during a pinned run")
        if metadata["sampling_mode"] != "external_sampling":
            raise RuntimeError("training requires external-sampling traversal shards")
        action_abstraction = metadata["action_abstraction"]
        for name in ("advantage_p0", "advantage_p1"):
            reservoir = reservoirs[name]
            if not isinstance(reservoir, DecisionReservoir):
                raise TypeError("advantage samples require grouped reservoirs")
            reservoir.clear()
        heldout = ingest_records(
            records,
            config.depth_bb,
            reservoirs,
            models,
            round_number,
            config.advantage_alpha,
            reservoir_rng,
        )
        losses: dict[str, float | None] = {}
        for name, reservoir in reservoirs.items():
            if isinstance(reservoir, ReplayReservoir):
                losses[name] = train_reservoir(
                    models[name],
                    optimizers[name],
                    compiled_steps[name],
                    reservoir,
                    config.steps_per_round,
                    config.batch_size,
                    np_rng,
                    config.replay_street_proposal,
                )
            else:
                losses[name] = train_grouped_reservoir(
                    models[name],
                    optimizers[name],
                    compiled_steps[name],
                    reservoir,
                    config.steps_per_round,
                    config.batch_size,
                    np_rng,
                    config.replay_street_proposal,
                )
        validation = evaluate_models(models, heldout, config.depth_bb)
        previous_checkpoint = state.get("checkpoint")
        checkpoint = save_checkpoint(
            run_dir,
            models,
            optimizers,
            round_number,
            config.variance_baseline_scale,
            config.depth_bb,
        )
        traversal_networks = checkpoint / "traversal-networks.json"
        exported_artifact: tuple[Path, Path] | None = None
        if round_number % config.artifact_every == 0:
            exported_artifact = export_browser_artifact(
                root,
                run_dir,
                models,
                config,
                round_number,
                action_abstraction,
            )
            exported_rounds.add(round_number)
        for reservoir in reservoirs.values():
            reservoir.flush()
        round_metrics = {
            "round": round_number,
            "traversals": config.traversals_per_round,
            "advantage_update": ADVANTAGE_UPDATE,
            "advantage_alpha": config.advantage_alpha,
            "variance_baseline_scale": config.variance_baseline_scale,
            "replay_street_proposal": config.replay_street_proposal,
            "value_rollouts_per_action": config.value_rollouts_per_action,
            "learning_rate": round_learning_rate,
            "records": len(records),
            "records_truncated": bool(metadata["truncated"]),
            "losses": losses,
            "heldout": validation,
            "elapsed_seconds": time.monotonic() - round_started,
            "peak_resident_bytes": peak_resident_bytes(),
            "mlx_active_memory_bytes": int(mx.get_active_memory()),
            "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
        }
        state["completed_rounds"] = round_number
        state["completed_traversals"] = int(state["completed_traversals"]) + config.traversals_per_round
        state["reservoirs"] = {name: reservoir.summary() for name, reservoir in reservoirs.items()}
        state["reservoir_rng_state"] = reservoir_rng.getstate()
        state["numpy_rng_state"] = np_rng.bit_generator.state
        state["checkpoint"] = str(checkpoint.relative_to(run_dir))
        state["metrics"] = (state.get("metrics", []) + [round_metrics])[-100:]
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as metrics_stream:
            metrics_stream.write(json.dumps(round_metrics, sort_keys=True) + "\n")
        if exported_artifact is not None:
            source_path, binary_path = exported_artifact
            state["artifacts"].append(
                {
                    "round": round_number,
                    "source": str(source_path),
                    "binary": str(binary_path),
                    "validation_status": "training_not_activated",
                }
            )
        atomic_json(state_path, state)
        if previous_checkpoint and previous_checkpoint != state["checkpoint"]:
            previous_path = run_dir / previous_checkpoint
            if previous_path.exists():
                shutil.rmtree(previous_path)
        if not args.keep_shards:
            shard.unlink()
        print(json.dumps(round_metrics, sort_keys=True), flush=True)
        if STOP_REQUESTED:
            break

    completed_round = int(state["completed_rounds"])
    if (
        completed_round > 0
        and completed_round not in exported_rounds
        and action_abstraction is not None
    ):
        source_path, binary_path = export_browser_artifact(
            root,
            run_dir,
            models,
            config,
            completed_round,
            action_abstraction,
        )
        state["artifacts"].append(
            {
                "round": completed_round,
                "source": str(source_path),
                "binary": str(binary_path),
                "validation_status": "training_not_activated",
            }
        )
    state["status"] = "paused_resumable"
    state["elapsed_seconds_last_invocation"] = time.monotonic() - started
    atomic_json(state_path, state)


if __name__ == "__main__":
    main()
