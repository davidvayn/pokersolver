#!/usr/bin/env python3
"""Apply a trust-region policy update from causal certificate attribution.

The Rust corpus contains policy-player action values under one frozen,
information-set-consistent best response from the causal sample game. Values
are already negated into the policy player's utility direction. This script
uses a per-node KL-capped exponentiated update, distills it from frozen
postflop weights, and requires improvement on an independently seeded corpus
before declaring the student eligible for routed evaluation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from distill_tabular_preflop import batch_features, probabilities
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
    scorer_json,
)


SCHEMA = "hu-neural-causal-policy-attribution-jsonl-v1"
REPORT_SCHEMA = "hu-neural-causal-policy-trust-region-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--initial-weights", type=Path, required=True)
    parser.add_argument("--source-networks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-sizes", default="512,256")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--mirror-step-per-bb", type=float, default=0.1)
    parser.add_argument("--maximum-target-node-kl", type=float, default=0.001)
    parser.add_argument("--maximum-realized-node-kl", type=float, default=0.005)
    parser.add_argument("--maximum-realized-weighted-kl", type=float, default=0.0015)
    parser.add_argument("--source-parity-tolerance", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=16_201)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_sha256(features: np.ndarray) -> str:
    canonical = np.rint(np.asarray(features, dtype=np.float64) * 1_000_000.0).astype(
        "<i4"
    )
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def load_dataset(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        metadata = json.loads(next(stream))
        if metadata.get("schema") != SCHEMA:
            raise ValueError("dataset is not a causal policy attribution corpus")
        if metadata.get("state_feature_schema") != STATE_FEATURE_SCHEMA:
            raise ValueError("causal attribution state feature schema is incompatible")
        if metadata.get("state_feature_count") != STATE_FEATURE_COUNT:
            raise ValueError("causal attribution state feature count is incompatible")
        if metadata.get("action_feature_count") != ACTION_FEATURE_COUNT:
            raise ValueError("causal attribution action feature count is incompatible")
        if not metadata.get("preflop_policy_frozen") or not metadata.get(
            "postflop_only"
        ):
            raise ValueError("causal attribution must freeze preflop and contain postflop only")
        count = int(metadata.get("records", 0))
        if count <= 0:
            raise ValueError("causal attribution corpus has no records")
        records = [json.loads(line) for line in stream if line.strip()]
    if len(records) != count:
        raise ValueError("causal attribution record count is invalid")

    depth = float(metadata["depth_bb"])
    states = np.zeros((count, STATE_FEATURE_COUNT), dtype=np.float32)
    actions = np.zeros(
        (count, MAX_POLICY_ACTIONS, ACTION_FEATURE_COUNT), dtype=np.float32
    )
    current = np.zeros((count, MAX_POLICY_ACTIONS), dtype=np.float32)
    values = np.zeros((count, MAX_POLICY_ACTIONS), dtype=np.float32)
    masks = np.zeros((count, MAX_POLICY_ACTIONS), dtype=np.float32)
    weights = np.zeros((count, 1), dtype=np.float32)
    iterations = np.zeros(count, dtype=np.uint64)
    maximum_probability_error = 0.0
    for index, record in enumerate(records):
        state = record["state"]
        if state.get("street") == "preflop":
            raise ValueError("causal attribution unexpectedly contains preflop")
        legal = record["actions"]
        probabilities = np.asarray(record["targets"], dtype=np.float32)
        action_values = np.asarray(record.get("action_values_bb"), dtype=np.float32)
        hashes = record.get("feature_sha256", [])
        if (
            not legal
            or len(legal) > MAX_POLICY_ACTIONS
            or probabilities.shape != (len(legal),)
            or action_values.shape != (len(legal),)
            or len(hashes) != len(legal)
        ):
            raise ValueError("causal attribution action row is incompatible")
        if (
            np.any(probabilities <= 0)
            or not np.all(np.isfinite(probabilities))
            or not np.all(np.isfinite(action_values))
        ):
            raise ValueError("causal attribution probabilities or values are invalid")
        probability_error = abs(float(np.sum(probabilities, dtype=np.float64)) - 1.0)
        maximum_probability_error = max(maximum_probability_error, probability_error)
        if probability_error > 1e-6:
            raise ValueError("causal attribution probabilities do not sum to one")
        state_features = expand_state(state, depth)
        action_features = np.stack(
            [expand_action(state, action, depth) for action in legal]
        )
        for action_index, expected_hash in enumerate(hashes):
            measured = feature_sha256(
                np.concatenate((state_features, action_features[action_index]))
            )
            if measured != expected_hash:
                raise ValueError("Rust/Python causal attribution feature hash differs")
        weight = float(record["weight"])
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("causal attribution weight must be finite and positive")
        states[index] = state_features
        actions[index, : len(legal)] = action_features
        current[index, : len(legal)] = probabilities
        values[index, : len(legal)] = action_values
        masks[index, : len(legal)] = 1.0
        weights[index, 0] = weight
        iterations[index] = int(record["iteration"])
    metadata = dict(metadata)
    metadata["maximum_loader_probability_sum_error"] = maximum_probability_error
    return metadata, {
        "states": states,
        "actions": actions,
        "current": current,
        "targets": current.copy(),
        "action_values_bb": values,
        "masks": masks,
        "weights": weights,
        "iterations": iterations,
    }


def categorical_kl(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    active = first > 0
    safe_first = np.where(active, first, 1.0)
    safe_second = np.where(active, np.maximum(second, 1e-30), 1.0)
    return np.sum(
        np.where(active, first * np.log(safe_first / safe_second), 0.0),
        axis=1,
    )


def mirror_descent_targets(
    current: np.ndarray,
    action_values_bb: np.ndarray,
    masks: np.ndarray,
    step_per_bb: float,
    maximum_node_kl: float,
) -> np.ndarray:
    if step_per_bb <= 0 or maximum_node_kl <= 0:
        raise ValueError("mirror step and node KL bound must be positive")
    output = np.zeros_like(current, dtype=np.float64)
    for row in range(len(current)):
        legal = masks[row] > 0
        old = np.asarray(current[row, legal], dtype=np.float64)
        values = np.asarray(action_values_bb[row, legal], dtype=np.float64)
        old /= np.sum(old)

        def update(step: float) -> np.ndarray:
            scores = np.log(old) + step * (values - np.max(values))
            scores -= np.max(scores)
            result = np.exp(scores)
            return result / np.sum(result)

        target = update(step_per_bb)
        if categorical_kl(target[None, :], old[None, :])[0] > maximum_node_kl:
            low = 0.0
            high = step_per_bb
            for _ in range(60):
                middle = (low + high) / 2.0
                candidate = update(middle)
                if (
                    categorical_kl(candidate[None, :], old[None, :])[0]
                    <= maximum_node_kl
                ):
                    low = middle
                    target = candidate
                else:
                    high = middle
        output[row, legal] = target
    return output.astype(np.float32)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    normalized = weights.reshape(-1).astype(np.float64)
    normalized /= np.sum(normalized)
    return float(np.sum(normalized * values))


def policy_metrics(
    candidate: np.ndarray,
    data: dict[str, np.ndarray],
) -> dict[str, float]:
    old = data["current"].astype(np.float64)
    masks = data["masks"].astype(np.float64)
    values = data["action_values_bb"].astype(np.float64)
    candidate = candidate.astype(np.float64) * masks
    candidate /= np.sum(candidate, axis=1, keepdims=True)
    baseline_value = np.sum(old * values, axis=1)
    candidate_value = np.sum(candidate * values, axis=1)
    reverse_kl = categorical_kl(candidate, old)
    forward_kl = categorical_kl(old, candidate)
    l1 = np.sum(np.abs(candidate - old), axis=1)
    weights = data["weights"]
    return {
        "weightedBaselinePolicyValueBb": weighted_mean(baseline_value, weights),
        "weightedCandidatePolicyValueBb": weighted_mean(candidate_value, weights),
        "weightedPolicyValueGainBb": weighted_mean(
            candidate_value - baseline_value, weights
        ),
        "weightedReverseKlFromFrozen": weighted_mean(reverse_kl, weights),
        "maximumReverseKlFromFrozen": float(np.max(reverse_kl)),
        "weightedForwardKlFromFrozen": weighted_mean(forward_kl, weights),
        "maximumForwardKlFromFrozen": float(np.max(forward_kl)),
        "weightedL1ActionDelta": weighted_mean(l1, weights),
        "maximumL1ActionDelta": float(np.max(l1)),
        "weightedPrimaryActionAgreement": weighted_mean(
            np.argmax(candidate, axis=1) == np.argmax(old, axis=1), weights
        ),
        "maximumProbabilitySumError": float(
            np.max(np.abs(np.sum(candidate, axis=1) - 1.0))
        ),
    }


def model_probabilities(model: ActionScorer, data: dict[str, np.ndarray]) -> np.ndarray:
    compatible = dict(data)
    compatible["targets"] = data["current"]
    return probabilities(model, compatible)


def validate_source_artifact_identity(
    model: ActionScorer,
    source_networks: Path,
    expected_sha256: str,
) -> None:
    if sha256(source_networks) != expected_sha256:
        raise ValueError("source network hash differs from the attribution corpus")
    bundle = json.loads(source_networks.read_text(encoding="utf-8"))
    postflop = bundle.get("postflop_networks")
    if (
        not isinstance(postflop, list)
        or len(postflop) != 2
        or postflop[0] != postflop[1]
        or scorer_json(model) != postflop[0]
    ):
        raise ValueError(
            "initial weights are not parameter-identical to the attributed policy"
        )


def main() -> None:
    args = parse_args()
    hidden = tuple(int(value) for value in args.hidden_sizes.split(","))
    if len(hidden) != 2 or min(hidden) <= 0:
        raise ValueError("--hidden-sizes requires two positive widths")
    if min(args.steps, args.batch_size) <= 0 or args.learning_rate <= 0:
        raise ValueError("trust-region optimization settings must be positive")
    for value in (
        args.mirror_step_per_bb,
        args.maximum_target_node_kl,
        args.maximum_realized_node_kl,
        args.maximum_realized_weighted_kl,
        args.source_parity_tolerance,
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("trust-region bounds must be finite and positive")

    training_metadata, training = load_dataset(args.dataset)
    validation_metadata, validation = load_dataset(args.validation_dataset)
    if training_metadata["depth_bb"] != validation_metadata["depth_bb"]:
        raise ValueError("training and validation attribution depths differ")
    if training_metadata["source_network_sha256"] != validation_metadata[
        "source_network_sha256"
    ]:
        raise ValueError("training and validation attributions use different policies")
    if training_metadata["seed"] == validation_metadata["seed"]:
        raise ValueError("causal trust-region validation must use an independent seed")

    model = ActionScorer(INPUT_FEATURE_COUNT, (hidden[0], hidden[1]))
    model.load_weights(str(args.initial_weights.resolve()))
    mx.eval(model.parameters())
    validate_source_artifact_identity(
        model,
        args.source_networks.resolve(),
        str(training_metadata["source_network_sha256"]),
    )
    baseline_training = model_probabilities(model, training)
    baseline_validation = model_probabilities(model, validation)
    maximum_source_error = max(
        float(np.max(np.abs(baseline_training - training["current"]))),
        float(np.max(np.abs(baseline_validation - validation["current"]))),
    )
    if maximum_source_error > args.source_parity_tolerance:
        raise ValueError(
            "Rust/MLX source inference difference exceeds its declared tolerance"
        )

    training_targets = mirror_descent_targets(
        training["current"],
        training["action_values_bb"],
        training["masks"],
        args.mirror_step_per_bb,
        args.maximum_target_node_kl,
    )
    validation_targets = mirror_descent_targets(
        validation["current"],
        validation["action_values_bb"],
        validation["masks"],
        args.mirror_step_per_bb,
        args.maximum_target_node_kl,
    )
    target_training_metrics = policy_metrics(training_targets, training)
    target_validation_metrics = policy_metrics(validation_targets, validation)
    if (
        target_training_metrics["maximumReverseKlFromFrozen"]
        > args.maximum_target_node_kl + 1e-7
        or target_validation_metrics["maximumReverseKlFromFrozen"]
        > args.maximum_target_node_kl + 1e-7
        or target_training_metrics["weightedPolicyValueGainBb"] < -1e-9
        or target_validation_metrics["weightedPolicyValueGainBb"] < -1e-9
    ):
        raise RuntimeError("mirror-descent targets failed their analytic trust-region gates")
    training["targets"] = training_targets

    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    step = make_compiled_policy_step(model, optimizer)
    indices = np.arange(len(training["states"]))
    losses: list[float] = []
    for _ in range(args.steps):
        batch = rng.choice(
            indices, size=min(args.batch_size, len(indices)), replace=True
        )
        loss = step(
            batch_features(training, batch),
            mx.array(training_targets[batch]),
            mx.array(training["masks"][batch]),
            mx.array(training["weights"][batch]),
        )
        mx.eval(loss, model.parameters(), optimizer.state)
        losses.append(float(loss.item()))

    student_training = model_probabilities(model, training)
    student_validation = model_probabilities(model, validation)
    student_training_metrics = policy_metrics(student_training, training)
    student_validation_metrics = policy_metrics(student_validation, validation)
    selection_checks = {
        "sourcePolicyArtifactIdentity": True,
        "sourceInferenceDifferenceBounded": maximum_source_error
        <= args.source_parity_tolerance,
        "trainingPolicyValueImproved": student_training_metrics[
            "weightedPolicyValueGainBb"
        ]
        > 0,
        "independentPolicyValueImproved": student_validation_metrics[
            "weightedPolicyValueGainBb"
        ]
        > 0,
        "trainingMaximumKlBounded": student_training_metrics[
            "maximumReverseKlFromFrozen"
        ]
        <= args.maximum_realized_node_kl,
        "independentMaximumKlBounded": student_validation_metrics[
            "maximumReverseKlFromFrozen"
        ]
        <= args.maximum_realized_node_kl,
        "trainingWeightedKlBounded": student_training_metrics[
            "weightedReverseKlFromFrozen"
        ]
        <= args.maximum_realized_weighted_kl,
        "independentWeightedKlBounded": student_validation_metrics[
            "weightedReverseKlFromFrozen"
        ]
        <= args.maximum_realized_weighted_kl,
        "probabilitySumsValid": max(
            student_training_metrics["maximumProbabilitySumError"],
            student_validation_metrics["maximumProbabilitySumError"],
        )
        <= 1e-6,
    }
    selection_checks_passed = all(selection_checks.values())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "student.safetensors"
    save_model(model, output)
    report = {
        "schema": REPORT_SCHEMA,
        "status": (
            "accepted_for_routed_evaluation"
            if selection_checks_passed
            else "rejected_before_routing"
        ),
        "selectionChecksPassed": selection_checks_passed,
        "selectionChecks": selection_checks,
        "depthBb": training_metadata["depth_bb"],
        "hiddenSizes": list(hidden),
        "steps": args.steps,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "mirrorStepPerBb": args.mirror_step_per_bb,
        "maximumTargetNodeKl": args.maximum_target_node_kl,
        "maximumRealizedNodeKl": args.maximum_realized_node_kl,
        "maximumRealizedWeightedKl": args.maximum_realized_weighted_kl,
        "seed": args.seed,
        "initialWeightsSha256": sha256(args.initial_weights),
        "sourceNetworksSha256": sha256(args.source_networks),
        "studentWeightsSha256": sha256(output),
        "studentBytes": output.stat().st_size,
        "maximumRustMlxSourcePolicyProbabilityDifference": maximum_source_error,
        "firstLoss": losses[0],
        "finalLoss": losses[-1],
        "trainingDataset": {
            "path": str(args.dataset.resolve()),
            "sha256": sha256(args.dataset),
            "metadata": training_metadata,
        },
        "validationDataset": {
            "path": str(args.validation_dataset.resolve()),
            "sha256": sha256(args.validation_dataset),
            "metadata": validation_metadata,
        },
        "analyticTarget": {
            "training": target_training_metrics,
            "independentValidation": target_validation_metrics,
        },
        "student": {
            "training": student_training_metrics,
            "independentValidation": student_validation_metrics,
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
