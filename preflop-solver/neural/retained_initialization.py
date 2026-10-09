"""Strict, research-only weight transfer. Never restores optimizer or regrets."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import native_value_dataset as native

RANGE_AUGMENTATION = "wide-to-wide-pooled-zero-columns-v1"


def native_import_prediction(model, contexts, queries, scales, boards, ranges, device=mx.cpu):
    """Imported towers with the *serving* clip/mask/projection, not train loss.

    The training forward zero-sums an unbounded regression vector. Serving
    clips/masks BEFORE its bounded zero-sum correction. These are not generally
    interchangeable, so import parity must compare the same wrapper.
    """
    # MLX 0.32 may silently use reduced-precision GPU float32 matmul. Import
    # verification uses CPU full float32; this does NOT alter training's device
    # or precision, so the initialization-only experiment remains matched.
    with mx.stream(device):
        context, query, scale = mx.array(contexts), mx.array(queries), mx.array(scales)
        embedding = model.query_tower(query)
        expanded = mx.broadcast_to(model.context_tower(context)[:, :, None, :], embedding.shape)
        if model.pools_exact_ranges:
            weights = mx.array(np.asarray([r * native.compatible_masses(r) for r in ranges], dtype=np.float32))
            reach = weights / mx.maximum(mx.sum(weights, axis=2, keepdims=True), 1e-9)
            pooled = mx.sum(embedding * reach[:, :, :, None], axis=2)
            own = mx.broadcast_to(pooled[:, :, None, :], embedding.shape)
            opponent = mx.broadcast_to(mx.stack((pooled[:, 1], pooled[:, 0]), axis=1)[:, :, None, :], embedding.shape)
            combined = mx.concatenate((expanded, own, opponent, embedding), axis=-1)
        else:
            combined = mx.concatenate((expanded, embedding), axis=-1)
        residual = model.head(combined).reshape((-1, 2, 1326))
        equity = query[:, :, :, 94]
        baseline = (equity * context[:, :, 20, None]
            - (1-equity) * context[:, :, 19, None]) * 20.
        raw = np.asarray(baseline + residual * scale[:, None, None])
    return np.asarray([native.project_native_predictions(values, board, weights)
        for values, board, weights in zip(raw, boards, ranges)])


def import_retained_weights(model, payload: dict, expected_seed: int, transform: str | None = None) -> dict:
    # Deliberately narrow: unsupported encoders/architectures must not receive
    # an approximately compatible subset of another network's parameters.
    if transform not in (None, RANGE_AUGMENTATION):
        raise ValueError("unsupported retained initialization transform")
    augment = transform == RANGE_AUGMENTATION
    if (model.architecture != ("wide-pooled" if augment else "wide") or not model.use_ranges
            or model.value_normalization != "payoff-exposure"
            or model.feature_schema != "rank-suit-invariant-combo-query-v3"):
        raise ValueError("retained initialization supports only the pinned wide/v3 contract")
    expected = dict(schema="hu-public-belief-combo-value-network-v4", architecture="wide",
        featureSchema="rank-suit-invariant-combo-query-v3", seed=expected_seed,
        usesExactRanges=True, targetScaleBb=20., valueNormalization="payoff-exposure",
        rangeScale=1326, residualUnit="normalized_state_value_scale",
        rangeAggregation="handcrafted-public-range-summaries",
        baseline="range_conditioned_exact_turn_runout_equity",
        sourceDatasetSchema="hu-native-turn-cfv-dataset-v1",
        predictionContract="native-turn-cfv-full-stack-v1", contextPublicCount=21,
        contextSize=417, queryStructuralCount=76, querySize=124, potExpertHeads=[])
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("retained model schema/seed/value contract differs")
    pending = []
    for name, tower in (("contextTower", model.context_tower),
                        ("queryTower", model.query_tower), ("head", model.head)):
        layers = [layer for layer in tower.layers if isinstance(layer, nn.Linear)]
        source = payload.get(name)
        if not isinstance(source, list) or len(source) != len(layers):
            raise ValueError("retained tower layer count differs")
        for index, (layer, item) in enumerate(zip(layers, source)):
            activation = "linear" if name == "head" and index == len(layers)-1 else "relu"
            output_size, destination_size = layer.weight.shape
            expand = augment and name == "head" and index == 0
            input_size = 128 if expand else destination_size
            if expand and (destination_size, output_size) != (256, 64):
                raise ValueError("range augmentation requires exact wide-pooled head")
            if (not isinstance(item, dict) or item.get("activation") != activation
                    or item.get("inputSize") != input_size or item.get("outputSize") != output_size):
                raise ValueError("retained layer shape/activation differs")
            try:
                weights = np.asarray(item["weights"], dtype=np.float32)
                biases = np.asarray(item["biases"], dtype=np.float32)
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raise ValueError("invalid retained parameter payload") from error
            if (weights.shape != (input_size * output_size,) or biases.shape != (output_size,)
                    or not np.isfinite(weights).all() or not np.isfinite(biases).all()):
                raise ValueError("retained parameter dimensions/nonfinite values")
            weights = weights.reshape(output_size, input_size)
            if expand:
                enlarged = np.zeros((64, 256), dtype=np.float32)
                enlarged[:, :64] = weights[:, :64]
                enlarged[:, 192:] = weights[:, 64:]
                weights = enlarged
            pending.append((layer, weights, biases))
    # Complete validation before mutation; no hybrid old/new model on rejection.
    for layer, weights, biases in pending:
        layer.weight, layer.bias = mx.array(weights), mx.array(biases)
    mx.eval(model.parameters())
    return dict(seed=expected_seed, optimizerState="fresh_not_imported", transform=transform,
        sourceValidationStatus=payload.get("sourceValidationStatus"), releaseAccepted=False)
