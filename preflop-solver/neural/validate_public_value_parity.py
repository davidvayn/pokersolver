#!/usr/bin/env python3
"""Verify exported shared-combo value inference is identical in Python and Rust."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

import train_public_value_network as training


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--solver", type=Path, default=Path("target/release/preflop-solver")
    )
    parser.add_argument("--state-index", type=int, default=0)
    parser.add_argument(
        "--state-indices",
        help="comma-separated target-state indices; overrides --state-index",
    )
    parser.add_argument("--maximum-absolute-error-bb", type=float, default=1e-4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def selected_state_indices(raw: str | None, fallback: int) -> list[int]:
    if raw is None:
        return [fallback]
    indices = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not indices:
        raise ValueError("--state-indices must contain at least one index")
    return list(dict.fromkeys(indices))


def activate(values: np.ndarray, activation: str) -> np.ndarray:
    if activation == "relu":
        return np.maximum(values, 0.0)
    if activation == "tanh":
        return np.tanh(values)
    if activation == "linear":
        return values
    if activation == "gelu-fast":
        return values / (1.0 + np.exp(-1.702 * values))
    raise ValueError(f"unknown activation {activation}")


def dense_forward(values: np.ndarray, layers: list[dict[str, Any]]) -> np.ndarray:
    for layer in layers:
        weights = np.asarray(layer["weights"], dtype=np.float32).reshape(
            layer["outputSize"], layer["inputSize"]
        )
        biases = np.asarray(layer["biases"], dtype=np.float32)
        values = activate(values @ weights.T + biases, layer["activation"])
    return values


def python_prediction(
    dataset: training.Dataset, model: dict[str, Any], state_index: int
) -> np.ndarray:
    context, queries = training.build_features(
        dataset.boards[state_index],
        int(dataset.actors[state_index]),
        dataset.invested[state_index],
        dataset.ranges[state_index],
        dataset.masses[state_index],
        model.get("featureSchema", training.FEATURE_SCHEMA),
    )
    equity = (
        queries[:, :, 94].copy()
        if model["usesExactRanges"]
        else queries[:, :, 65].copy()
    )
    baseline_depth = (
        equity * context[:, 20, None]
        - (1.0 - equity) * context[:, 19, None]
    )
    if not model["usesExactRanges"]:
        context[:, training.CONTEXT_PUBLIC_COUNT :] = 0.0
        queries[:, :, training.QUERY_STRUCTURAL_COUNT :] = 0.0
    context_embedding = dense_forward(context, model["contextTower"])
    query_embedding = dense_forward(queries, model["queryTower"])
    expanded = np.broadcast_to(context_embedding[:, None, :], query_embedding.shape)
    if model["schema"] == training.POOLED_NETWORK_SCHEMA:
        projection_weights = dataset.projection_weights[state_index]
        reach = projection_weights / np.maximum(
            projection_weights.sum(axis=1, keepdims=True), 1e-8
        )
        pooled = (query_embedding * reach[:, :, None]).sum(axis=1)
        own_pool = np.broadcast_to(pooled[:, None, :], query_embedding.shape)
        opponent_pool = np.broadcast_to(pooled[::-1, None, :], query_embedding.shape)
        head_input = np.concatenate(
            (expanded, own_pool, opponent_pool, query_embedding), axis=-1
        )
    else:
        head_input = np.concatenate((expanded, query_embedding), axis=-1)
    residual = dense_forward(head_input, model["head"]).reshape(
        2, training.COMBO_COUNT
    )
    if model["schema"] in {
        training.NETWORK_SCHEMA,
        training.POOLED_NETWORK_SCHEMA,
    }:
        scale_bb = training.value_scale_bb(
            dataset.invested[state_index], model["valueNormalization"]
        )
        raw = baseline_depth * (training.DEPTH_BB / scale_bb) + residual
    else:
        scale_bb = float(model["targetScaleBb"])
        raw = baseline_depth + residual * (
            float(model["residualScaleBb"]) / scale_bb
        )
    projection_weights = dataset.projection_weights[state_index]
    joint_mass = max(float(projection_weights[0].sum()), 1e-8)
    aggregate = (raw * projection_weights).sum(axis=1) / joint_mass
    projected = raw - float(aggregate.sum()) / 2.0
    projected[dataset.ranges[state_index] <= 0] = 0.0
    return projected * scale_bb


def rust_prediction(
    solver: Path, dataset: Path, model: Path, state_index: int
) -> np.ndarray:
    completed = subprocess.run(
        [
            str(solver),
            "turn-pbs-value-predict",
            "--dataset",
            str(dataset),
            "--value-network",
            str(model),
            "--state-index",
            str(state_index),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    return np.asarray(payload["counterfactualValuesBb"], dtype=np.float64)


def main() -> None:
    args = parse_args()
    model = json.loads(args.model.read_text())
    if model.get("schema") not in {
        training.NETWORK_SCHEMA,
        training.POOLED_NETWORK_SCHEMA,
        "hu-public-belief-combo-value-network-v3",
    }:
        raise ValueError("parity validation requires a shared-combo v3, v4, or v5 model")
    normalization = model.get("valueNormalization", "depth")
    dataset = training.load_dataset(args.dataset, 1, normalization)
    state_indices = selected_state_indices(args.state_indices, args.state_index)
    per_state = []
    for state_index in state_indices:
        python_values = python_prediction(dataset, model, state_index)
        rust_values = rust_prediction(args.solver, args.dataset, args.model, state_index)
        per_state.append(
            {
                "stateIndex": state_index,
                "maximumAbsoluteErrorBb": float(
                    np.max(np.abs(python_values - rust_values))
                ),
            }
        )
    maximum_error = max(row["maximumAbsoluteErrorBb"] for row in per_state)
    report = {
        "schema": "hu-public-belief-value-parity-v1",
        "model": str(args.model),
        "modelSha256": sha256_file(args.model),
        "dataset": str(args.dataset),
        "datasetSha256": sha256_file(args.dataset),
        "stateIndices": state_indices,
        "perState": per_state,
        "maximumAbsoluteErrorBb": maximum_error,
        "thresholdBb": args.maximum_absolute_error_bb,
        "validation": {
            "status": "accepted"
            if maximum_error <= args.maximum_absolute_error_bb
            else "rejected",
            "reasons": []
            if maximum_error <= args.maximum_absolute_error_bb
            else ["Python and Rust value-network inference differ"],
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    if report["validation"]["status"] != "accepted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
