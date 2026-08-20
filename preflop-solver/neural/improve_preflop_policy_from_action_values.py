#!/usr/bin/env python3
"""Build a bounded preflop policy-improvement candidate from measured action EVs.

The action-value artifact evaluates every legal action while holding the rest of
the frozen policy fixed.  This tool performs one conservative policy-iteration
step: it forms a regret-matching target from statistically conservative action
advantages and mixes that target with the parent policy.  It never changes the
game tree, action labels, or information-set coverage.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT_HISTORY = ("blinds:0.500/1.000",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--action-values", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mix", type=float, required=True)
    parser.add_argument(
        "--actors",
        default="0,1",
        help="comma-separated actors to update (player 0 is BTN/SB)",
    )
    parser.add_argument(
        "--root-only",
        action="store_true",
        help="update only the initial BTN/SB decision",
    )
    parser.add_argument(
        "--confidence-z",
        type=float,
        default=2.5758293035489004,
        help="one-sided normal multiplier subtracted from each action EV",
    )
    parser.add_argument(
        "--minimum-advantage-bb",
        type=float,
        default=0.01,
        help="ignore conservative advantages no larger than this amount",
    )
    parser.add_argument("--model-version", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(values: list[float]) -> list[float]:
    total = math.fsum(values)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("policy target has invalid total probability")
    output = [max(value / total, 0.0) for value in values]
    # Put the final floating-point residual on the largest component.  A pure
    # target can end in a zero-probability action, where assigning a tiny
    # negative residual to the last element would violate policy validation.
    largest = max(range(len(output)), key=output.__getitem__)
    output[largest] += 1.0 - math.fsum(output)
    return output


def conservative_regret_target(
    probabilities: list[float],
    action_values: list[float],
    standard_errors: list[float],
    confidence_z: float,
    minimum_advantage_bb: float,
) -> tuple[list[float], float]:
    if not (
        len(probabilities) == len(action_values) == len(standard_errors)
        and probabilities
    ):
        raise ValueError("action-value row dimensions do not match")
    if any(
        not math.isfinite(value)
        for value in probabilities + action_values + standard_errors
    ):
        raise ValueError("action-value row contains a non-finite number")
    parent = normalize(probabilities)
    policy_value = math.fsum(
        probability * value for probability, value in zip(parent, action_values)
    )
    conservative_values = [
        value - confidence_z * max(standard_error, 0.0)
        for value, standard_error in zip(action_values, standard_errors)
    ]
    advantages = [
        max(value - policy_value - minimum_advantage_bb, 0.0)
        for value in conservative_values
    ]
    if math.fsum(advantages) <= 0:
        return parent, 0.0
    target = normalize(advantages)
    target_value = math.fsum(
        probability * value for probability, value in zip(target, action_values)
    )
    return target, target_value - policy_value


def improve_policy(
    policy: dict[str, Any],
    action_values: dict[str, Any],
    *,
    mix: float,
    actors: set[int],
    root_only: bool,
    confidence_z: float,
    minimum_advantage_bb: float,
    model_version: str,
    policy_sha256: str,
    action_values_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not 0 < mix <= 1:
        raise ValueError("mix must be in (0, 1]")
    if not actors or not actors <= {0, 1}:
        raise ValueError("actors must be a non-empty subset of 0,1")
    if confidence_z < 0 or minimum_advantage_bb < 0:
        raise ValueError("confidence controls must be non-negative")
    if policy.get("schema") != "hu-tabular-preflop-dcfr-v1":
        raise ValueError("policy has an incompatible schema")
    if action_values.get("schema") != "hu-preflop-canonical-range-action-values-v1":
        raise ValueError("action values have an incompatible schema")
    if action_values.get("source_policy_sha256") != policy.get("source_policy_sha256"):
        raise ValueError("action values target a different frozen neural policy")

    value_rows = {
        row["key"]: row
        for player_rows in action_values.get("players", [])
        for row in player_rows
    }
    strategies = policy.get("strategies")
    if not isinstance(strategies, list) or len(value_rows) != len(strategies):
        raise ValueError("policy and action values do not have identical coverage")

    updated = 0
    predicted_gain = 0.0
    maximum_probability_delta = 0.0
    missing: list[str] = []
    for strategy in strategies:
        row = value_rows.get(strategy.get("key"))
        if row is None:
            missing.append(str(strategy.get("key")))
            continue
        if strategy.get("action_labels") != row.get("action_labels"):
            raise ValueError(f"legal actions changed at {strategy.get('key')}")
        actor = int(strategy["actor"])
        history = tuple(strategy["public_history"])
        if actor not in actors or (root_only and history != ROOT_HISTORY):
            continue
        target, full_step_gain = conservative_regret_target(
            [float(value) for value in strategy["probabilities"]],
            [float(value) for value in row["action_values_bb"]],
            [float(value) for value in row["action_value_standard_errors_bb"]],
            confidence_z,
            minimum_advantage_bb,
        )
        parent = normalize([float(value) for value in strategy["probabilities"]])
        candidate = normalize(
            [
                (1.0 - mix) * old + mix * desired
                for old, desired in zip(parent, target)
            ]
        )
        strategy["probabilities"] = candidate
        updated += 1
        reach = max(float(row.get("reach_probability", 0.0)), 0.0)
        predicted_gain += reach * mix * full_step_gain
        maximum_probability_delta = max(
            maximum_probability_delta,
            max(abs(old - new) for old, new in zip(parent, candidate)),
        )
    if missing:
        raise ValueError(f"action values missed {len(missing)} policy rows")
    if updated == 0:
        raise ValueError("selection updated no policy rows")

    policy["model_version"] = model_version
    policy["source_policy_sha256"] = None
    policy["training_evaluation"] = {
        **policy.get("training_evaluation", {}),
        "interpretation": (
            "stale parent evaluation; candidate requires independent evaluation before use"
        ),
    }
    provenance = {
        "schema": "hu-preflop-action-value-policy-improvement-v1",
        "modelVersion": model_version,
        "parentPolicySha256": policy_sha256,
        "actionValuesSha256": action_values_sha256,
        "mix": mix,
        "actors": sorted(actors),
        "rootOnly": root_only,
        "confidenceZ": confidence_z,
        "minimumAdvantageBb": minimum_advantage_bb,
        "updatedInformationSets": updated,
        "predictedFrozenContinuationGainBbPerHand": predicted_gain,
        "maximumProbabilityDelta": maximum_probability_delta,
        "activationEligible": False,
        "interpretation": (
            "one bounded policy-iteration proposal against frozen continuations; "
            "not an exploitability certificate"
        ),
    }
    policy["policy_improvement"] = provenance
    return policy, provenance


def main() -> None:
    args = parse_args()
    actors = {int(value) for value in args.actors.split(",") if value.strip()}
    policy = read_json(args.policy)
    values = read_json(args.action_values)
    candidate, report = improve_policy(
        policy,
        values,
        mix=args.mix,
        actors=actors,
        root_only=args.root_only,
        confidence_z=args.confidence_z,
        minimum_advantage_bb=args.minimum_advantage_bb,
        model_version=args.model_version,
        policy_sha256=sha256(args.policy),
        action_values_sha256=sha256(args.action_values),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(candidate, separators=(",", ":")), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({**report, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
