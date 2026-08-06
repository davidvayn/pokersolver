#!/usr/bin/env python3
"""Report state- and texture-conditioned errors for an exported value model."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import train_public_value_network as training
import validate_public_value_parity as parity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--state-indices", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def selected_indices(raw: str) -> list[int]:
    indices = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("state indices must be a non-empty unique list")
    return indices


def weighted_error_bb(
    truth_bb: np.ndarray, prediction_bb: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    normalized = weights / max(float(weights.sum()), 1e-12)
    error = prediction_bb - truth_bb
    return (
        float(np.sqrt(np.sum(normalized * error * error))),
        float(np.sum(normalized * np.abs(error))),
    )


def player_weighted_signed_error_bb(
    truth_bb: np.ndarray, prediction_bb: np.ndarray, weights: np.ndarray
) -> list[float]:
    errors = (prediction_bb - truth_bb).reshape((-1, 2, training.COMBO_COUNT))
    player_weights = weights.reshape((-1, 2, training.COMBO_COUNT))
    return [
        float(
            np.sum(player_weights[:, player] * errors[:, player])
            / max(float(player_weights[:, player].sum()), 1e-12)
        )
        for player in range(2)
    ]


def resolver_reach_weighted_error_bb(
    truth_bb: np.ndarray,
    prediction_bb: np.ndarray,
    weights: np.ndarray,
    resolver_reaches: np.ndarray,
) -> tuple[float, float]:
    if weights.shape[0] != resolver_reaches.shape[0]:
        raise ValueError("resolver reach count must match the state count")
    reshape = (len(resolver_reaches),) + (1,) * (weights.ndim - 1)
    return weighted_error_bb(
        truth_bb,
        prediction_bb,
        weights * resolver_reaches.reshape(reshape),
    )


def compose(
    dataset: training.Dataset,
    model: dict[str, Any],
    indices: list[int],
    model_sha256: str | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for index in indices:
        if index < 0 or index >= len(dataset.boards):
            raise ValueError(f"state index {index} is outside the dataset")
        prediction = parity.python_prediction(dataset, model, index)
        truth = dataset.targets[index].reshape(2, training.COMBO_COUNT)
        truth_bb = truth * dataset.target_scales[index]
        state_weights = dataset.weights[index].reshape(2, training.COMBO_COUNT)
        rmse, mae = weighted_error_bb(truth_bb, prediction, state_weights)
        player_bias = player_weighted_signed_error_bb(
            truth_bb, prediction, state_weights
        )
        texture = training.public_board_texture(dataset.boards[index])
        source_target = dataset.source["targets"][index]
        resolver_reach = source_target.get("resolver_leaf_reach_probability")
        rows.append(
            {
                "stateIndex": index,
                "board": [int(card) for card in dataset.boards[index]],
                "potBand": training.POT_BAND_NAMES[
                    training.pot_band(dataset.invested[index])
                ],
                "texture": texture,
                "weightedRmseBb": rmse,
                "weightedMaeBb": mae,
                "playerWeightedMeanErrorBb": player_bias,
                "maximumAbsolutePlayerWeightedMeanErrorBb": max(
                    abs(value) for value in player_bias
                ),
                "resolverLeafReachProbability": resolver_reach,
                "resolverRootBoard": source_target.get("resolver_root_board"),
                "resolverPublicHistory": source_target.get("resolver_public_history"),
            }
        )
        predictions.append(prediction)
        truths.append(truth_bb)
        weights.append(state_weights)

    grouped: dict[str, dict[str, list[int]]] = {
        "potBand": defaultdict(list),
        "rankTexture": defaultdict(list),
        "suitTexture": defaultdict(list),
        "connectivity": defaultdict(list),
    }
    for offset, row in enumerate(rows):
        grouped["potBand"][row["potBand"]].append(offset)
        grouped["rankTexture"][row["texture"]["rank"]].append(offset)
        grouped["suitTexture"][row["texture"]["suit"]].append(offset)
        grouped["connectivity"][row["texture"]["connectivity"]].append(offset)

    prediction_array = np.stack(predictions)
    truth_array = np.stack(truths)
    weight_array = np.stack(weights)
    summaries: dict[str, dict[str, dict[str, float | int]]] = {}
    for facet, values in grouped.items():
        summaries[facet] = {}
        for label, offsets in sorted(values.items()):
            rmse, mae = weighted_error_bb(
                truth_array[offsets], prediction_array[offsets], weight_array[offsets]
            )
            player_bias = player_weighted_signed_error_bb(
                truth_array[offsets],
                prediction_array[offsets],
                weight_array[offsets],
            )
            summaries[facet][label] = {
                "states": len(offsets),
                "weightedRmseBb": rmse,
                "weightedMaeBb": mae,
                "playerWeightedMeanErrorBb": player_bias,
                "maximumAbsolutePlayerWeightedMeanErrorBb": max(
                    abs(value) for value in player_bias
                ),
            }
    overall_rmse, overall_mae = weighted_error_bb(
        truth_array, prediction_array, weight_array
    )
    overall_player_bias = player_weighted_signed_error_bb(
        truth_array, prediction_array, weight_array
    )
    resolver_reaches = np.asarray(
        [
            (
                float(row["resolverLeafReachProbability"])
                if row["resolverLeafReachProbability"] is not None
                else np.nan
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    if np.all(np.isfinite(resolver_reaches)) and np.all(resolver_reaches > 0.0):
        resolver_rmse, resolver_mae = resolver_reach_weighted_error_bb(
            truth_array,
            prediction_array,
            weight_array,
            resolver_reaches,
        )
        reach_shape = (len(resolver_reaches),) + (1,) * (weight_array.ndim - 1)
        resolver_bias = player_weighted_signed_error_bb(
            truth_array,
            prediction_array,
            weight_array * resolver_reaches.reshape(reach_shape),
        )
        resolver_evaluation: dict[str, Any] | None = {
            "sampledLeafReachMass": float(resolver_reaches.sum()),
            "reachWeightedRmseBb": resolver_rmse,
            "reachWeightedMaeBb": resolver_mae,
            "playerWeightedMeanErrorBb": resolver_bias,
            "maximumAbsolutePlayerWeightedMeanErrorBb": max(
                abs(value) for value in resolver_bias
            ),
        }
    else:
        resolver_evaluation = None
    return {
        "schema": "hu-public-value-texture-diagnostics-v1",
        "sourceDatasetSha256": dataset.source_sha256,
        "sourcePolicySha256": dataset.source.get("source_policy_sha256"),
        "resolverSourceValueNetworkSha256": dataset.source.get(
            "resolver_source_value_network_sha256"
        ),
        "modelSchema": model["schema"],
        "modelSeed": model["seed"],
        "modelSha256": model_sha256,
        "states": len(rows),
        "weightedRmseBb": overall_rmse,
        "weightedMaeBb": overall_mae,
        "playerWeightedMeanErrorBb": overall_player_bias,
        "maximumAbsolutePlayerWeightedMeanErrorBb": max(
            abs(value) for value in overall_player_bias
        ),
        "resolverReachEvaluation": resolver_evaluation,
        "facets": summaries,
        "perState": rows,
    }


def main() -> None:
    args = parse_args()
    model_bytes = args.model.read_bytes()
    model = json.loads(model_bytes)
    normalization = model.get("valueNormalization", "depth")
    dataset = training.load_dataset(args.dataset, 1, normalization)
    report = compose(
        dataset,
        model,
        selected_indices(args.state_indices),
        hashlib.sha256(model_bytes).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
