#!/usr/bin/env python3
"""Verify exported shared-combo value inference is identical in Python and Rust."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

import train_public_value_network as training


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
    )
    equity = (
        queries[:, :, 94].copy()
        if model["usesExactRanges"]
        else queries[:, :, 65].copy()
    )
    baseline = (
        equity * context[:, 20, None]
        - (1.0 - equity) * context[:, 19, None]
    )
    if not model["usesExactRanges"]:
        context[:, training.CONTEXT_PUBLIC_COUNT :] = 0.0
        queries[:, :, training.QUERY_STRUCTURAL_COUNT :] = 0.0
    context_embedding = dense_forward(context, model["contextTower"])
    query_embedding = dense_forward(queries, model["queryTower"])
    expanded = np.broadcast_to(context_embedding[:, None, :], query_embedding.shape)
    residual = dense_forward(
        np.concatenate((expanded, query_embedding), axis=-1), model["head"]
    ).reshape(2, training.COMBO_COUNT)
    raw = baseline + residual * (
        float(model["residualScaleBb"]) / float(model["targetScaleBb"])
    )
    projection_weights = dataset.projection_weights[state_index]
    joint_mass = max(float(projection_weights[0].sum()), 1e-8)
    aggregate = (raw * projection_weights).sum(axis=1) / joint_mass
    projected = raw - float(aggregate.sum()) / 2.0
    projected[dataset.ranges[state_index] <= 0] = 0.0
    return projected * float(model["targetScaleBb"])


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
    dataset = training.load_dataset(args.dataset, 1)
    model = json.loads(args.model.read_text())
    if model.get("schema") != training.NETWORK_SCHEMA:
        raise ValueError("parity validation requires a shared-combo v3 model")
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
        "dataset": str(args.dataset),
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
