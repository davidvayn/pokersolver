#!/usr/bin/env python3
"""Reach-aware cross-seed checks for frozen neural average policies.

This validator deliberately fails closed. Cross-seed agreement, finite
probabilities, and artifact hashes are useful reproducibility gates, but they
do not supply a full-game exploitability upper bound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from train import (
    ACTIONS,
    NETWORK_SCHEMA,
    ActionScorer,
    INPUT_FEATURE_COUNT,
    expand_state_action,
    linear_layers,
    load_jsonl_gzip,
    scorer_json,
    softmax,
)


def load_artifact_model(source_path: Path) -> ActionScorer:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    descriptors = source["metadata"]["networks"]["baselinePolicy"]["layers"]
    if len(descriptors) != 3 or int(descriptors[0]["inputSize"]) != INPUT_FEATURE_COUNT:
        raise RuntimeError("browser artifact has an incompatible baseline network")
    hidden = (int(descriptors[0]["outputSize"]), int(descriptors[1]["outputSize"]))
    model = ActionScorer(INPUT_FEATURE_COUNT, hidden)
    parameters = np.asarray(source["parameters"], dtype=np.float32)
    for layer, descriptor in zip(linear_layers(model), descriptors):
        input_size = int(descriptor["inputSize"])
        output_size = int(descriptor["outputSize"])
        weight_offset = int(descriptor["weightOffset"])
        bias_offset = int(descriptor["biasOffset"])
        weight_count = input_size * output_size
        layer.weight = mx.array(
            parameters[weight_offset : weight_offset + weight_count].reshape(
                (output_size, input_size)
            )
        )
        layer.bias = mx.array(parameters[bias_offset : bias_offset + output_size])
    mx.eval(model.parameters())
    return model


def load_run(
    run_dir: Path,
    round_number: int | None = None,
) -> tuple[dict[str, Any], ActionScorer]:
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    if round_number is not None:
        artifact = latest_artifact(state, round_number)
        return state, load_artifact_model(Path(artifact["source"]))
    config = state["config"]
    hidden = tuple(int(value) for value in config["hidden_sizes"])
    model = ActionScorer(INPUT_FEATURE_COUNT, (hidden[0], hidden[1]))
    checkpoint = state.get("checkpoint")
    if not checkpoint:
        raise RuntimeError("run state does not name an atomic checkpoint")
    model.load_weights(str(run_dir / checkpoint / "average_strategy.safetensors"))
    mx.eval(model.parameters())
    return state, model


def immutable_game_config(state: dict[str, Any]) -> dict[str, Any]:
    config = dict(state["config"])
    config.pop("seed", None)
    return config


def latest_artifact(
    state: dict[str, Any],
    round_number: int | None = None,
) -> dict[str, Any]:
    if not state.get("artifacts"):
        raise RuntimeError("run has no exported browser artifact")
    if round_number is not None:
        matches = [
            artifact
            for artifact in state["artifacts"]
            if int(artifact["round"]) == round_number
        ]
        if len(matches) != 1:
            raise RuntimeError(f"run has no unique round-{round_number} browser artifact")
        return matches[0]
    return max(state["artifacts"], key=lambda artifact: int(artifact["round"]))


def verify_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    binary = Path(artifact["binary"])
    descriptor = json.loads((binary.parent / "descriptor.json").read_text(encoding="utf-8"))
    payload = binary.read_bytes()
    sha256 = hashlib.sha256(payload).hexdigest()
    return {
        "path": str(binary),
        "bytes": len(payload),
        "sha256": sha256,
        "descriptor_matches": descriptor.get("artifactBytes") == len(payload)
        and descriptor.get("artifactSha256") == sha256,
        "magic_valid": payload[:4] == b"PLNP",
    }


def evaluation_records(
    root: Path,
    depth_bb: int,
    traversals: int,
    seed: int,
    preflop_samples: int,
    flop_samples: int,
    sample_turn_rivers: bool,
    compact_grid: bool,
    policy_model: ActionScorer | None = None,
) -> list[dict[str, Any]]:
    subprocess.run(
        ["cargo", "build", "--release", "--manifest-path", "preflop-solver/Cargo.toml"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    binary = root / "preflop-solver/target/release/preflop-solver"
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "evaluation.jsonl.gz"
        network_path = Path(directory) / "policy-networks.json"
        command = [
            str(binary),
            "neural-samples",
            "--effective-stack-bb",
            str(depth_bb),
            "--traversals",
            str(traversals),
            "--seed",
            str(seed),
            "--max-records",
            "100000",
            "--sample-trajectories",
            "--preflop-runout-samples",
            str(preflop_samples),
            "--flop-runout-samples",
            str(flop_samples),
            "--output",
            str(output),
        ]
        if sample_turn_rivers:
            command.append("--sample-turn-rivers")
        if compact_grid:
            command.append("--compact-serving-grid")
        if policy_model is not None:
            network = scorer_json(policy_model)
            network_path.write_text(
                json.dumps(
                    {
                        "schema": NETWORK_SCHEMA,
                        "input_size": INPUT_FEATURE_COUNT,
                        "strategy_transform": "softmax",
                        "networks": [network, network],
                    },
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            command.extend(("--networks", str(network_path)))
        subprocess.run(command, cwd=root, check=True, capture_output=True)
        metadata, records = load_jsonl_gzip(output)
    if metadata["truncated"]:
        raise RuntimeError("independent evaluation trajectory set was truncated")
    if metadata["sampling_mode"] != "trajectory":
        raise RuntimeError("independent evaluation did not use pure trajectory sampling")
    return records


def policy(model: ActionScorer, features: np.ndarray) -> np.ndarray:
    logits = np.asarray(model(mx.array(features))).reshape(-1)
    probabilities = softmax(logits.astype(np.float64))
    if not np.all(np.isfinite(probabilities)) or abs(float(probabilities.sum()) - 1.0) > 1e-6:
        raise RuntimeError("neural average policy produced invalid probabilities")
    return probabilities


def compare(
    first_model: ActionScorer,
    second_model: ActionScorer,
    datasets: list[list[dict[str, Any]]],
    depth_bb: int,
    distribution: str,
    reach_weighted: bool,
) -> dict[str, Any]:
    weighted_records: list[tuple[dict[str, Any], float]] = []
    for records in datasets:
        if not records:
            raise RuntimeError("evaluation distribution reached no decisions")
        dataset_scale = 1.0 / len(datasets)
        decision_weight = dataset_scale / len(records)
        weighted_records.extend((record, decision_weight) for record in records)

    absolute = 0.0
    primary_agreements = 0.0
    tie_aware_agreements = 0.0
    first_top_probability = 0.0
    second_top_probability = 0.0
    first_entropy = 0.0
    second_entropy = 0.0
    street_metrics: dict[str, dict[str, float]] = {}
    probability_count = 0
    first_kind_mass = np.zeros(len(ACTIONS), dtype=np.float64)
    second_kind_mass = np.zeros(len(ACTIONS), dtype=np.float64)
    total_weight = sum(weight for _, weight in weighted_records)
    squared_weight = 0.0
    for record, raw_weight in weighted_records:
        decision_weight = raw_weight / total_weight
        squared_weight += decision_weight * decision_weight
        features = np.stack(
            [expand_state_action(record["state"], action, depth_bb) for action in record["actions"]]
        )
        first = policy(first_model, features)
        second = policy(second_model, features)
        absolute += decision_weight * float(np.mean(np.abs(first - second)))
        first_primary = int(np.argmax(first))
        second_primary = int(np.argmax(second))
        primary_agreements += decision_weight * int(first_primary == second_primary)
        first_near_best = set(np.flatnonzero(first >= float(np.max(first)) - 0.01).tolist())
        second_near_best = set(np.flatnonzero(second >= float(np.max(second)) - 0.01).tolist())
        tie_aware_agreements += decision_weight * int(bool(first_near_best & second_near_best))
        first_top_probability += decision_weight * float(np.max(first))
        second_top_probability += decision_weight * float(np.max(second))
        first_entropy += decision_weight * float(-np.sum(first * np.log(first + 1e-12)))
        second_entropy += decision_weight * float(-np.sum(second * np.log(second + 1e-12)))
        street = str(record["state"]["street"])
        street_metric = street_metrics.setdefault(
            street,
            {"weight": 0.0, "mae": 0.0, "agreement": 0.0, "decisions": 0.0},
        )
        street_metric["weight"] += decision_weight
        street_metric["mae"] += decision_weight * float(np.mean(np.abs(first - second)))
        street_metric["agreement"] += decision_weight * int(first_primary == second_primary)
        street_metric["decisions"] += 1
        probability_count += len(first)
        for action, first_probability, second_probability in zip(record["actions"], first, second):
            kind_index = ACTIONS.index(action["kind"])
            first_kind_mass[kind_index] += decision_weight * first_probability
            second_kind_mass[kind_index] += decision_weight * second_probability
    decisions = len(weighted_records)
    if decisions == 0:
        raise RuntimeError("independent evaluation reached no decisions")
    deltas = np.abs(first_kind_mass - second_kind_mass)
    street_breakdown = {
        street: {
            "reach_mass": values["weight"],
            "action_frequency_mae": values["mae"] / values["weight"],
            "primary_action_agreement": values["agreement"] / values["weight"],
            "decisions": int(values["decisions"]),
        }
        for street, values in street_metrics.items()
        if values["weight"] > 0
    }
    return {
        "evaluation_distribution": distribution,
        "reach_weighted": reach_weighted,
        "sampling_method": "empirical_pure_trajectories",
        "decisions": decisions,
        "effective_decisions": 1.0 / squared_weight,
        "action_probabilities": probability_count,
        "action_frequency_mae": absolute,
        "primary_action_agreement": primary_agreements,
        "primary_action_agreement_tie_aware_0_01": tie_aware_agreements,
        "mean_top_probability": [first_top_probability, second_top_probability],
        "mean_policy_entropy": [first_entropy, second_entropy],
        "street_breakdown": street_breakdown,
        "maximum_aggregate_action_delta": float(np.max(deltas)),
        "aggregate_action_deltas": {
            kind: float(deltas[index]) for index, kind in enumerate(ACTIONS)
        },
        "probability_sums_valid": True,
        "lookup_coverage": 1.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--traversals", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0x8A5CD789)
    parser.add_argument("--round", type=int, dest="round_number")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.traversals <= 0:
        raise ValueError("evaluation traversals must be positive")
    if args.round_number is not None and args.round_number <= 0:
        raise ValueError("artifact round must be positive")
    root = Path(__file__).resolve().parents[2]
    first_state, first_model = load_run(args.run_a.resolve(), args.round_number)
    second_state, second_model = load_run(args.run_b.resolve(), args.round_number)
    if immutable_game_config(first_state) != immutable_game_config(second_state):
        raise RuntimeError("cross-seed runs do not share an identical training configuration")
    config = first_state["config"]
    if first_state["config"]["seed"] == second_state["config"]["seed"]:
        raise RuntimeError("cross-seed validation requires independent training seeds")
    evaluation_arguments = (
        root,
        int(config["depth_bb"]),
        args.traversals,
        args.seed,
        int(config["preflop_runout_samples"]),
        int(config["flop_runout_samples"]),
        not bool(config["exact_turn_rivers"]),
        bool(config["compact_serving_grid"]),
    )
    first_reach_records = evaluation_records(*evaluation_arguments, policy_model=first_model)
    second_reach_records = evaluation_records(*evaluation_arguments, policy_model=second_model)
    forced_records = evaluation_records(*evaluation_arguments)
    cross_seed = compare(
        first_model,
        second_model,
        [first_reach_records, second_reach_records],
        int(config["depth_bb"]),
        "equal mixture of both frozen policies over authentic exact-card deals",
        True,
    )
    forced_deviation = compare(
        first_model,
        second_model,
        [forced_records],
        int(config["depth_bb"]),
        "fixed-seed uniform-policy forced-deviation exact-card trajectories",
        False,
    )
    gates = {
        "action_frequency_mae_at_most_0_05": cross_seed["action_frequency_mae"] <= 0.05,
        "primary_action_agreement_at_least_0_85": cross_seed["primary_action_agreement"] >= 0.85,
        "aggregate_action_delta_at_most_0_03": cross_seed["maximum_aggregate_action_delta"] <= 0.03,
        "probabilities_valid": cross_seed["probability_sums_valid"],
        "reach_weighting_valid": cross_seed["reach_weighted"],
        "coverage_at_least_0_9999": cross_seed["lookup_coverage"] >= 0.9999,
        "independent_seed_count_at_least_2": True,
        "exploitability_upper_99_at_most_0_10": False,
        "action_ev_standard_error_coverage_at_least_0_95": False,
    }
    research_pilot_gates = {
        "action_frequency_mae_at_most_0_06": cross_seed["action_frequency_mae"] <= 0.06,
        "primary_action_agreement_at_least_0_80": cross_seed["primary_action_agreement"] >= 0.80,
        "aggregate_action_delta_at_most_0_04": cross_seed["maximum_aggregate_action_delta"] <= 0.04,
        "coverage_at_least_0_9999": cross_seed["lookup_coverage"] >= 0.9999,
        "probabilities_valid": cross_seed["probability_sums_valid"],
    }
    report = {
        "schema": "hu-neural-cross-seed-validation-v4",
        "depth_bb": config["depth_bb"],
        "seeds": [first_state["config"]["seed"], second_state["config"]["seed"]],
        "completed_traversals": [
            args.round_number * int(config["traversals_per_round"])
            if args.round_number is not None
            else first_state["completed_traversals"],
            args.round_number * int(config["traversals_per_round"])
            if args.round_number is not None
            else second_state["completed_traversals"],
        ],
        "cross_seed": cross_seed,
        "forced_deviation": forced_deviation,
        "artifacts": [
            verify_artifact(latest_artifact(first_state, args.round_number)),
            verify_artifact(latest_artifact(second_state, args.round_number)),
        ],
        "gates": gates,
        "research_pilot": {
            "purpose": "continuation evidence only; never sufficient for model activation",
            "gates": research_pilot_gates,
            "status": "promising"
            if all(research_pilot_gates.values())
            else "not_yet_promising",
        },
        "status": "rejected_not_activated",
        "reasons": [
            "Cross-seed stability is a reproducibility check, not equilibrium proof.",
            "A statistically valid full-game exploitability upper bound has not been computed.",
            "Independent action-EV standard-error coverage has not reached the release gate.",
        ],
    }
    report["integrity_valid"] = all(
        artifact["descriptor_matches"] and artifact["magic_valid"] for artifact in report["artifacts"]
    )
    output = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
