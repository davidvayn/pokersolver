"""Strict research-only native CFV data contract and leakage-safe splitting.

Conditional training targets, raw opponent-scaled CFVs, and on-policy reach
weights are different quantities. Never make a range positive to train a value
for a zero-own-reach deviation.
"""
from __future__ import annotations

import itertools
import hashlib
import json
import re

import numpy as np

SCHEMA = "hu-native-turn-cfv-dataset-v1"
SEMANTICS = "profile-positive-own-reach-cbr-zero-own-reach-v1"
COMBOS = np.asarray([(high, low) for high in range(1, 52) for low in range(high)])
COMBO_COUNT = len(COMBOS)


def identity_hash(values) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def _sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def legal_combos(board: list[int] | np.ndarray) -> np.ndarray:
    return ~np.isin(COMBOS, board).any(axis=1)


def compatible_masses(ranges: np.ndarray) -> np.ndarray:
    result = []
    for player in (0, 1):
        opponent = ranges[1 - player]
        cards = np.bincount(COMBOS.reshape(-1), weights=np.repeat(opponent, 2), minlength=52)
        result.append(np.maximum(0.0, opponent.sum() - cards[COMBOS[:, 0]] - cards[COMBOS[:, 1]] + opponent))
    return np.stack(result)


def validate_dataset(source: dict) -> None:
    """Reject semantic corruption before any feature allocation or fitting."""
    if source.get("schema") != SCHEMA or source.get("game", {}).get("effective_stack_bb") != 20.0:
        raise ValueError("native value pilot requires its explicit schema and 20bb game")
    if source.get("validation", {}).get("status") != "research_only":
        raise ValueError("native value corpus is research-only, never release-accepted")
    for key in ("source_public_input_sha256", "source_policy_sha256"):
        if not _sha(source.get(key)):
            raise ValueError(f"native corpus lacks pinned {key}")
    targets = source.get("targets")
    if not isinstance(targets, list) or not targets or len(targets) != source.get("maximum_states"):
        raise ValueError("native capture is incomplete")
    if source.get("observed_queries", -1) < len(targets):
        raise ValueError("native capture has fewer queries than targets")
    captures = source.get("source_captures")
    if captures is not None:
        if (not captures or source.get("source_identity_semantics") != "ordered_capture_manifest_not_one_policy"
                or source["source_public_input_sha256"] != identity_hash([s["input_sha256"] for s in captures])
                or source["source_policy_sha256"] != identity_hash([s["policy_sha256"] for s in captures])):
            raise ValueError("invalid native collection provenance")
        for capture in captures:
            if not all(_sha(capture.get(k)) for k in ("input_sha256", "policy_sha256", "capture_sha256")):
                raise ValueError("unhashed native source capture")
        if len(targets) != sum(c["states"] for c in captures):
            raise ValueError("incomplete native source collection")
    for index, target in enumerate(targets):
        try:
            header = source
            if captures is not None:
                ci = target.get("source_capture_index")
                if type(ci) is not int or not 0 <= ci < len(captures):
                    raise ValueError("missing source capture index")
                capture = captures[ci]
                if sum(t.get("source_capture_index") == ci for t in targets) != capture["states"]:
                    raise ValueError("source capture membership/count mismatch")
                header = {**source, "flop_iterations": capture["flop_iterations"], "turn_iterations": capture["turn_iterations"]}
            _validate_target(target, header)
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"native target {index}: {error}") from error


def _validate_target(target: dict, source: dict) -> None:
    board = target["board"]
    if (len(board) != 4 or len(set(board)) != 4
            or any(type(c) is not int or not 0 <= c < 52 for c in board)):
        raise ValueError("invalid turn board")
    if target["value_semantics"] != SEMANTICS or not _sha(target["policy_sha256"]):
        raise ValueError("missing counterfactual semantics or frozen policy identity")
    if not 1 <= target["iteration"] <= source["flop_iterations"]:
        raise ValueError("iteration outside frozen native capture")
    if target["turn_iterations"] != source["turn_iterations"] or target["policy_rows"] <= 0:
        raise ValueError("reference solver budget or policy provenance mismatch")
    public = target["public_state"]
    if public["board"] != board or public["actor"] != target["actor"] or public["invested_bb"] != target["invested_bb"]:
        raise ValueError("public state mismatch")
    if target["actor"] not in (0, 1) or len(target["invested_bb"]) != 2:
        raise ValueError("invalid actor or investments")
    if any(not np.isfinite(v) or not 0 <= v <= 20 for v in target["invested_bb"]):
        raise ValueError("invalid investments")
    if not public.get("public_history") or not public.get("trajectory"):
        raise ValueError("missing exact public history")
    legal = legal_combos(board)

    def vector(value, nonnegative=False):
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (2, COMBO_COUNT) or not np.isfinite(result).all():
            raise ValueError("expected finite 2 x 1326 vectors")
        if nonnegative and (result < 0).any():
            raise ValueError("negative reach or mass")
        return result

    raw_ranges = vector(public["ranges"], True)
    ranges = vector(target["ranges"], True)
    masses = vector(target["opponent_compatible_mass"], True)
    conditional = vector(target["counterfactual_values_bb"])
    raw = vector(target["raw_counterfactual_bb"])
    profile = vector(target["raw_profile_counterfactual_bb"])
    best = vector(target["raw_best_response_counterfactual_bb"])
    totals = np.asarray(target["raw_reach_totals"], dtype=np.float64)
    if totals.shape != (2,) or not np.isfinite(totals).all() or (totals < 0).any():
        raise ValueError("invalid raw reach totals")
    if (raw_ranges[:, ~legal] != 0).any() or (ranges[:, ~legal] != 0).any():
        raise ValueError("board-blocked private reach")
    if not np.allclose(raw_ranges.sum(axis=1), totals, rtol=1e-10, atol=1e-14):
        raise ValueError("raw reach total mismatch")
    normalized = np.divide(raw_ranges, totals[:, None], out=np.zeros_like(raw_ranges), where=totals[:, None] > 0)
    if not np.allclose(ranges, normalized, rtol=1e-10, atol=1e-14):
        raise ValueError("native ranges were changed or a zero range was invented")
    if not np.allclose(masses, compatible_masses(ranges), rtol=1e-9, atol=1e-12):
        raise ValueError("blocker-compatible opponent mass mismatch")
    raw_mass = masses * totals[::-1, None]
    if not np.allclose(raw, conditional * raw_mass, rtol=1e-9, atol=1e-10):
        raise ValueError("conditional EV / raw opponent-scaled CFV mismatch")
    if any((values[:, ~legal] != 0).any() for values in (raw, profile, best, conditional)):
        raise ValueError("board-blocked counterfactual value")
    for values in (raw, profile, best):
        if (np.abs(values) > 20 * raw_mass + 1e-8).any():
            raise ValueError("counterfactual value exceeds full-stack payoff bound")
    if not np.allclose(raw, np.where(ranges == 0, best, profile), rtol=1e-9, atol=1e-10):
        raise ValueError("zero-own-reach completion/profile selection mismatch")
    if abs(float(np.sum(raw_ranges * profile))) > 1e-8:
        raise ValueError("frozen profile violates zero-sum accounting")
    completed = ((ranges == 0) & legal[None, :]).sum(axis=1)
    if not np.array_equal(completed, target["completed_zero_own_reach"]):
        raise ValueError("zero-own-reach completion count mismatch")
    joint = float(np.sum(ranges[0] * masses[0]))
    gains = target["conditional_response_gain_bb"]
    if gains is None:
        if joint > 1e-12:
            raise ValueError("reachable reference lacks response residual")
    else:
        gains = np.asarray(gains, dtype=float)
        if gains.shape != (2,) or not np.isfinite(gains).all() or (gains < -1e-8).any() or joint <= 0:
            raise ValueError("invalid conditional response residual")
        raw_joint = joint * totals.prod()
        expected = np.sum(raw_ranges * (best - profile), axis=1) / raw_joint
        if not np.allclose(gains, expected, rtol=1e-7, atol=1e-8):
            raise ValueError("reference response residual mismatch")


def training_weights(board, ranges, masses, counterfactual_fraction: float = 0.1) -> np.ndarray:
    if not 0 < counterfactual_fraction <= 1:
        raise ValueError("native counterfactual training fraction must be in (0,1]")
    legal = legal_combos(board)
    uniform = legal.astype(np.float64) / legal.sum()
    # Do not reuse this for projection or pooling: this is a loss distribution,
    # not a modification of either player's actual public belief.
    return ((1 - counterfactual_fraction) * ranges + counterfactual_fraction * uniform) * masses


def board_family(board) -> tuple[int, ...]:
    """Hold out whole flop families, including all suit variants and turns."""
    return min(tuple(sorted((int(c) // 4) * 4 + p[int(c) % 4] for c in board[:3]))
               for p in itertools.permutations(range(4)))


def family_split(source: dict, seed: int, validation_fraction: float, tuning_fraction: float,
                 reference: dict | None = None, refresh_training: bool = False):
    if not (0 < validation_fraction < 1 and 0 < tuning_fraction < 1
            and validation_fraction + tuning_fraction < 1):
        raise ValueError("native train/tuning/holdout fractions are invalid")
    if refresh_training and reference is None:
        raise ValueError("training refresh requires a pinned split reference")
    if reference is not None:
        if source["game"] != reference["game"]:
            raise ValueError("reference split game changed")
        count = len(reference["targets"])
        if len(source["targets"]) < count:
            raise ValueError("reference split target prefix was truncated")
        if not refresh_training and source["targets"][:count] != reference["targets"]:
            raise ValueError("reference split target prefix changed or was truncated")
        train, tuning, holdout = family_split(reference, seed, validation_fraction, tuning_fraction)
        if refresh_training:
            if any(source["targets"][i] != reference["targets"][i] for i in np.concatenate((tuning, holdout))):
                raise ValueError("training refresh changed a held-out tuning/validation target")
            if any(board_family(source["targets"][i]["board"]) != board_family(reference["targets"][i]["board"])
                   for i in train):
                raise ValueError("training refresh changed a reference flop family")
        forbidden = {board_family(reference["targets"][i]["board"]) for i in np.concatenate((tuning, holdout))}
        if any(board_family(row["board"]) in forbidden for row in source["targets"][count:]):
            raise ValueError("training extension overlaps a held-out tuning/validation flop family")
        return np.concatenate((train, np.arange(count, len(source["targets"]), dtype=np.int64))), tuning, holdout
    families = [board_family(row["board"]) for row in source["targets"]]
    unique = sorted(set(families))
    if len(unique) < 3:
        raise ValueError("native training needs at least three disjoint flop families; a cost preflight is not a training corpus")
    order = np.random.default_rng(seed ^ 0x51A7E).permutation(len(unique))
    holdout_count = min(len(unique) - 2, max(1, round(len(unique) * validation_fraction)))
    tuning_count = min(len(unique) - holdout_count - 1, max(1, round(len(unique) * tuning_fraction)))
    holdout = {unique[i] for i in order[:holdout_count]}
    tuning = {unique[i] for i in order[holdout_count:holdout_count + tuning_count]}
    train = set(unique) - holdout - tuning
    return tuple(np.asarray([i for i, family in enumerate(families) if family in split], dtype=np.int64)
                 for split in (train, tuning, holdout))


def project_native_predictions(values, board, ranges, depth=20.0):
    """Serving-equivalent bounded zero-sum projection; retain zero-own queries."""
    result = np.asarray(values, dtype=np.float64).copy()
    ranges = np.asarray(ranges, dtype=np.float64)
    legal = legal_combos(board)
    masses = compatible_masses(ranges)
    weights = ranges * masses
    # Match blueprint::EPSILON, including its near-zero correction branch.
    joint = max(float(weights[0].sum()), 1e-9)
    result = np.clip(result, -depth, depth)
    result[:, ~legal] = 0.0
    aggregate = np.sum(weights * result, axis=1) / joint
    midpoint = (aggregate[0] - aggregate[1]) / 2
    for p in (0, 1):
        target = midpoint if p == 0 else -midpoint
        if abs(aggregate[p] - target) <= 1e-12:
            continue
        low, high = -depth - result[p, legal].max(), depth - result[p, legal].min()
        for _ in range(80):
            shift = (low + high) / 2
            shifted = np.sum(weights[p] * np.clip(result[p] + shift, -depth, depth)) / joint
            if shifted < target: low = shift
            else: high = shift
        result[p, legal] = np.clip(result[p, legal] + (low + high) / 2, -depth, depth)
    return result
