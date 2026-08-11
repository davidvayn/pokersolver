#!/usr/bin/env python3
"""Trust-region updates for a served range policy on full-game response rows."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
import numpy as np

from distill_causal_policy import mirror_descent_targets
from distill_range_policy import (
    ACTION_FEATURE_COUNT,
    ACTION_FEATURE_SCHEMA,
    COMBO_CONFLICTS,
    COMBO_COUNT,
    CONTEXT_SIZE,
    LoadedDataset,
    RANGE_POLICY_FEATURE_SCHEMA,
    RangeConditionedPolicy,
    add_features,
    export_model_from_source,
    load_exported_model,
    sha256,
)

CAUSAL_SCHEMA = "hu-range-conditioned-causal-policy-attribution-jsonl-v1"
SELF_PLAY_SCHEMA = "hu-range-conditioned-self-play-regret-jsonl-v1"
DIRECTIONAL_RECORD_TYPES = {
    CAUSAL_SCHEMA: "range_conditioned_causal_policy_attribution",
    SELF_PLAY_SCHEMA: "range_conditioned_self_play_regret",
}
REPORT_SCHEMA = "hu-paired-range-conditioned-directional-trust-region-v2"
MAXIMUM_SOURCE_PARITY_ABSOLUTE_ERROR = 0.0025
MAXIMUM_SOURCE_PARITY_WEIGHTED_KL = 1e-6
# MLX/Metal and the scalar Rust evaluator accumulate compact-network matrix
# products in a different order. Keep the local parity gate tight enough to
# catch behavioral mismatches while allowing the measured backend-only drift;
# exact candidate acceptance is still decided by the Rust evaluator below.
MAXIMUM_SOURCE_PARITY_NODE_KL = 2e-5
REALIZED_TRUST_REGION_SELECTION_FRACTION = 0.95
ONE_SIDED_99_PERCENT_Z = 2.3263478740408408


@dataclass
class CausalRangeDataset:
    path: Path
    sha256: str
    metadata: dict[str, Any]
    records: list[dict[str, Any]]
    feature_dataset: LoadedDataset
    actions: np.ndarray
    action_masks: np.ndarray
    focal_combos: np.ndarray
    current: np.ndarray
    action_values: np.ndarray
    action_value_standard_errors: np.ndarray | None
    weights: np.ndarray

    @property
    def contexts(self) -> np.ndarray:
        assert self.feature_dataset.contexts is not None
        return self.feature_dataset.contexts

    @property
    def queries(self) -> np.ndarray:
        assert self.feature_dataset.queries is not None
        return self.feature_dataset.queries

    @property
    def projection_weights(self) -> np.ndarray:
        return self.feature_dataset.projection_weights

    @property
    def actors(self) -> np.ndarray:
        return self.feature_dataset.actors


def _record_identity(record: dict[str, Any]) -> bytes:
    return json.dumps(
        {
            "state": record.get("state"),
            "focal_combo": record.get("focal_combo"),
            "action_labels": record.get("action_labels"),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _cap_records(records: list[dict[str, Any]], capacity: int) -> list[dict[str, Any]]:
    if capacity <= 0 or len(records) <= capacity:
        return records
    if capacity < 6:
        raise ValueError("causal range-policy cap must cover every street")

    def rank(record: dict[str, Any]) -> tuple[float, bytes]:
        weight = float(record.get("weight", float("nan")))
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("causal range-policy rows need positive weights")
        return (-weight, hashlib.sha256(_record_identity(record)).digest())

    selected: set[int] = set()
    for street in ("flop", "turn", "river"):
        rows = [
            index
            for index, record in enumerate(records)
            if record.get("state", {}).get("street") == street
        ]
        if not rows:
            raise ValueError(f"causal range-policy corpus omits {street}")
        selected.update(sorted(rows, key=lambda row: rank(records[row]))[:2])
    for index in sorted(range(len(records)), key=lambda row: rank(records[row])):
        if len(selected) >= capacity:
            break
        selected.add(index)
    return [
        records[index] for index in sorted(selected, key=lambda row: rank(records[row]))
    ]


def load_dataset(
    path: Path,
    maximum_records: int,
    feature_cache_dir: Path | None,
    feature_workers: int,
) -> CausalRangeDataset:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        metadata = json.loads(next(stream))
        records = [json.loads(line) for line in stream if line.strip()]
    if (
        metadata.get("record_type") != "metadata"
        or metadata.get("schema") not in DIRECTIONAL_RECORD_TYPES
        or metadata.get("state_feature_schema") != RANGE_POLICY_FEATURE_SCHEMA
        or metadata.get("state_feature_count") != CONTEXT_SIZE
        or metadata.get("action_feature_schema") != ACTION_FEATURE_SCHEMA
        or metadata.get("action_feature_count") != ACTION_FEATURE_COUNT
        or metadata.get("uses_exact_ranges") is not True
        or metadata.get("focal_combo_attribution") is not True
        or metadata.get("postflop_only") is not True
        or metadata.get("preflop_policy_frozen") is not True
        or metadata.get("records") != len(records)
        or not records
        or not isinstance(metadata.get("source_range_policy_sha256"), str)
        or len(metadata["source_range_policy_sha256"]) != 64
    ):
        raise ValueError(f"incompatible causal range-policy dataset: {path}")
    records = _cap_records(records, maximum_records)
    expected_record_type = DIRECTIONAL_RECORD_TYPES[metadata["schema"]]
    count = len(records)
    maximum_actions = max(len(record.get("action_labels", [])) for record in records)
    if maximum_actions <= 0:
        raise ValueError("causal range-policy corpus has no actions")

    boards: list[np.ndarray] = []
    actors = np.empty(count, dtype=np.int32)
    invested = np.empty((count, 2), dtype=np.float32)
    ranges = np.empty((count, 2, COMBO_COUNT), dtype=np.float32)
    actions = np.zeros((count, maximum_actions, ACTION_FEATURE_COUNT), dtype=np.float32)
    action_masks = np.zeros((count, maximum_actions), dtype=np.float32)
    focal_combos = np.empty(count, dtype=np.int32)
    current = np.zeros((count, maximum_actions), dtype=np.float32)
    action_values = np.zeros_like(current)
    action_value_standard_errors = (
        np.zeros_like(current) if metadata["schema"] == SELF_PLAY_SCHEMA else None
    )
    weights = np.empty(count, dtype=np.float64)
    summaries: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        state = record.get("state", {})
        action_count = len(record.get("action_labels", []))
        record_ranges = np.asarray(record.get("ranges"), dtype=np.float32)
        record_actions = np.asarray(record.get("action_features"), dtype=np.float32)
        record_current = np.asarray(record.get("probabilities"), dtype=np.float32)
        record_values = np.asarray(record.get("action_values_bb"), dtype=np.float32)
        record_standard_errors = (
            np.asarray(record.get("action_value_standard_errors_bb"), dtype=np.float32)
            if action_value_standard_errors is not None
            else None
        )
        board = np.asarray(state.get("board"), dtype=np.int16)
        actor = int(state.get("actor", -1))
        focal = int(record.get("focal_combo", -1))
        weight = float(record.get("weight", float("nan")))
        if (
            record.get("record_type") != expected_record_type
            or not 0 < action_count <= maximum_actions
            or record_ranges.shape != (2, COMBO_COUNT)
            or record_actions.shape != (action_count, ACTION_FEATURE_COUNT)
            or record_current.shape != (action_count,)
            or record_values.shape != (action_count,)
            or (
                record_standard_errors is not None
                and (
                    record_standard_errors.shape != (action_count,)
                    or not np.all(np.isfinite(record_standard_errors))
                    or not np.all(record_standard_errors >= 0)
                )
            )
            or board.shape not in ((3,), (4,), (5,))
            or actor not in (0, 1)
            or not 0 <= focal < COMBO_COUNT
            or not np.isfinite(weight)
            or weight <= 0
            or not np.all(np.isfinite(record_ranges))
            or not np.all(record_ranges >= 0)
            or not np.all(np.isfinite(record_actions))
            or not np.all(np.isfinite(record_current))
            or not np.all(record_current > 0)
            or not np.all(np.isfinite(record_values))
            or np.any(np.abs(record_ranges.sum(axis=1) - 1.0) > 1e-5)
            or abs(float(record_current.sum()) - 1.0) > 1e-5
            or record_ranges[actor, focal] <= 0
        ):
            raise ValueError(f"invalid causal range-policy record {index} in {path}")
        boards.append(board)
        actors[index] = actor
        invested[index] = np.asarray(state.get("invested_bb"), dtype=np.float32)
        ranges[index] = record_ranges
        actions[index, :action_count] = record_actions
        action_masks[index, :action_count] = 1.0
        focal_combos[index] = focal
        current[index, :action_count] = record_current
        action_values[index, :action_count] = record_values
        if action_value_standard_errors is not None:
            assert record_standard_errors is not None
            action_value_standard_errors[index, :action_count] = record_standard_errors
        weights[index] = weight
        summaries.append({"state": state})

    masses = np.maximum(
        ranges.sum(axis=2)[:, ::-1, None]
        - ranges[:, ::-1, :][:, :, COMBO_CONFLICTS].sum(axis=3),
        0.0,
    ).astype(np.float32)
    projection = ranges * masses
    selection_digest = hashlib.sha256()
    selection_digest.update(sha256(path).encode())
    for record in records:
        selection_digest.update(_record_identity(record))
    feature_identity = selection_digest.hexdigest()
    empty = np.empty((0,), dtype=np.float32)
    feature_dataset = LoadedDataset(
        path=path,
        sha256=feature_identity,
        metadata={"depth_bb": float(metadata["depth_bb"])},
        records=summaries,
        boards=boards,
        actors=actors,
        invested=invested,
        ranges=ranges,
        masses=masses,
        projection_weights=projection,
        actions=actions,
        action_masks=action_masks,
        source_probabilities=empty,
        targets=empty,
        action_values=empty,
        combo_weights=empty,
    )
    add_features(feature_dataset, feature_cache_dir, feature_workers)
    return CausalRangeDataset(
        path=path,
        sha256=sha256(path),
        metadata=metadata,
        records=records,
        feature_dataset=feature_dataset,
        actions=actions,
        action_masks=action_masks,
        focal_combos=focal_combos,
        current=current,
        action_values=action_values,
        action_value_standard_errors=action_value_standard_errors,
        weights=weights,
    )


def _model_arguments(
    dataset: CausalRangeDataset, rows: np.ndarray
) -> tuple[mx.array, ...]:
    return (
        mx.array(dataset.contexts[rows]),
        mx.array(dataset.queries[rows]),
        mx.array(dataset.projection_weights[rows]),
        mx.array(dataset.actors[rows]),
        mx.array(dataset.actions[rows]),
        mx.array(dataset.action_masks[rows]),
    )


def predict(
    model: RangeConditionedPolicy,
    dataset: CausalRangeDataset,
    batch_size: int,
    retain_full: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    focal = np.zeros_like(dataset.current)
    full: list[np.ndarray] = []
    for start in range(0, len(dataset.records), batch_size):
        rows = np.arange(start, min(start + batch_size, len(dataset.records)))
        logits = model(*_model_arguments(dataset, rows))
        probabilities = mx.softmax(logits, axis=2)
        mx.eval(probabilities)
        measured = np.asarray(probabilities)
        focal[rows] = measured[np.arange(len(rows)), dataset.focal_combos[rows]]
        if retain_full:
            full.append(measured)
    return focal, np.concatenate(full) if retain_full else None


def metrics(
    candidate: np.ndarray,
    dataset: CausalRangeDataset,
    full_candidate: np.ndarray | None = None,
    full_source: np.ndarray | None = None,
    frozen: np.ndarray | None = None,
) -> dict[str, float]:
    if frozen is None:
        frozen = dataset.current
    if frozen.shape != candidate.shape:
        raise ValueError("causal range-policy baseline shape differs")
    legal = dataset.action_masks > 0
    safe_candidate = np.where(legal, np.maximum(candidate, 1e-30), 1.0)
    safe_source = np.where(legal, np.maximum(frozen, 1e-30), 1.0)
    reverse = np.sum(
        np.where(legal, safe_candidate * np.log(safe_candidate / safe_source), 0.0),
        axis=1,
    )
    gains = np.sum((candidate - frozen) * dataset.action_values, axis=1)
    normalized = dataset.weights / dataset.weights.sum()
    result = {
        "weightedPolicyValueGainBb": float(np.sum(normalized * gains)),
        "weightedReverseKlFromFrozen": float(np.sum(normalized * reverse)),
        "maximumReverseKlFromFrozen": float(np.max(reverse)),
        "weightedL1ActionDelta": float(
            np.sum(normalized * np.sum(np.abs(candidate - frozen), axis=1))
        ),
        "primaryActionAgreement": float(
            np.sum(
                normalized * (np.argmax(candidate, axis=1) == np.argmax(frozen, axis=1))
            )
        ),
    }
    if dataset.action_value_standard_errors is not None:
        node_variances = np.sum(
            np.square(candidate - frozen)
            * np.square(dataset.action_value_standard_errors),
            axis=1,
        )
        gain_standard_error = float(
            np.sqrt(np.sum(np.square(normalized) * node_variances))
        )
        result["weightedPolicyValueGainActionRolloutStandardErrorBb"] = (
            gain_standard_error
        )
        result["weightedPolicyValueGainActionRolloutLowerBound99Bb"] = (
            result["weightedPolicyValueGainBb"]
            - ONE_SIDED_99_PERCENT_Z * gain_standard_error
        )
    if full_candidate is not None and full_source is not None:
        source = np.maximum(full_source, 1e-30)
        candidate_full = np.maximum(full_candidate, 1e-30)
        local = np.sum(candidate_full * np.log(candidate_full / source), axis=2)
        actor_projection = dataset.projection_weights[
            np.arange(len(dataset.records)), dataset.actors
        ].astype(np.float64)
        actor_projection /= np.maximum(
            actor_projection.sum(axis=1, keepdims=True), 1e-30
        )
        node_kl = np.sum(actor_projection * local, axis=1)
        result["weightedAllComboAnchorKl"] = float(np.sum(normalized * node_kl))
        result["maximumAllComboAnchorKl"] = float(np.max(node_kl))
    return result


def source_parity_metrics(
    measured: np.ndarray, dataset: CausalRangeDataset
) -> dict[str, float | bool]:
    result: dict[str, float | bool] = metrics(measured, dataset)
    legal = dataset.action_masks > 0
    result["maximumAbsoluteError"] = float(
        np.max(np.abs(measured - dataset.current)[legal])
    )
    result["maximumProbabilitySumError"] = float(
        np.max(np.abs(np.sum(measured, axis=1) - 1.0))
    )
    result["accepted"] = bool(
        result["maximumAbsoluteError"] <= MAXIMUM_SOURCE_PARITY_ABSOLUTE_ERROR
        and result["weightedReverseKlFromFrozen"] <= MAXIMUM_SOURCE_PARITY_WEIGHTED_KL
        and result["maximumReverseKlFromFrozen"] <= MAXIMUM_SOURCE_PARITY_NODE_KL
        and result["primaryActionAgreement"] >= 1.0 - 1e-12
        and result["maximumProbabilitySumError"] <= 1e-6
    )
    return result


def rust_evaluate(
    evaluator: Path,
    candidate: Path,
    frozen: Path,
    attribution: Path,
    dataset: Path,
    minimum_policy_value_gain_bb: float,
    maximum_node_kl: float,
    maximum_weighted_kl: float,
) -> dict[str, Any]:
    command = [
        str(evaluator),
        "range-policy-causal-evaluate",
        "--network",
        str(candidate),
        "--frozen-network",
        str(frozen),
        "--attribution-network",
        str(attribution),
        "--dataset",
        str(dataset),
        "--minimum-policy-value-gain-bb",
        str(minimum_policy_value_gain_bb),
        "--maximum-node-kl",
        str(maximum_node_kl),
        "--maximum-weighted-kl",
        str(maximum_weighted_kl),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            "Rust directional range-policy evaluation failed"
            + (f": {detail}" if detail else " without diagnostics")
        )
    report = json.loads(completed.stdout)
    if (
        report.get("schema") != "hu-range-conditioned-causal-policy-rust-evaluation-v1"
        or report.get("networkSha256") != sha256(candidate)
        or report.get("frozenNetworkSha256") != sha256(frozen)
        or report.get("attributionNetworkSha256") != sha256(attribution)
        or report.get("datasetSha256") != sha256(dataset)
    ):
        raise RuntimeError("Rust causal range-policy evaluation is not pinned")
    return report


def reuse_exact_dataset_parity(
    report_path: Path,
    datasets: list[CausalRangeDataset],
    sources: list[Path],
    attributions: list[Path],
) -> list[dict[str, Any]]:
    payload = json.loads(report_path.read_text())
    reports = payload.get("exactRustDatasetParity")
    if not isinstance(reports, list) or len(reports) != len(datasets):
        raise ValueError("cached exact Rust dataset parity is incomplete")
    for report, dataset, source, attribution in zip(
        reports, datasets, sources, attributions, strict=True
    ):
        if (
            report.get("schema")
            != "hu-range-conditioned-causal-policy-rust-evaluation-v1"
            or report.get("networkSha256") != sha256(source)
            or report.get("frozenNetworkSha256") != sha256(source)
            or report.get("attributionNetworkSha256") != sha256(attribution)
            or report.get("datasetSha256") != dataset.sha256
            or report.get("records") != len(dataset.records)
            or report.get("maximumStoredSourceProbabilityDifference", float("inf"))
            > 1e-6
            or report.get("maximumProbabilitySumError", float("inf")) > 1e-6
        ):
            raise ValueError("cached exact Rust dataset parity is not pinned")
    return reports


def train_candidate(
    source_path: Path,
    training: CausalRangeDataset,
    validation: CausalRangeDataset,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    mirror_step_per_bb: float,
    maximum_target_node_kl: float,
    maximum_realized_node_kl: float,
    maximum_realized_weighted_kl: float,
    anchor_scale: float,
    maximum_gradient_norm: float,
    evaluation_interval: int,
    full_corpus_gradient: bool = False,
    attribution_path: Path | None = None,
    paired_corpus_gradient: bool = False,
) -> tuple[RangeConditionedPolicy, dict[str, Any], dict[str, Any]]:
    if paired_corpus_gradient and not full_corpus_gradient:
        raise ValueError("paired-corpus gradient requires full-corpus accumulation")
    source_payload = json.loads(source_path.read_text())
    model = RangeConditionedPolicy(
        str(source_payload["architecture"]),
        str(source_payload.get("policyComposition", "replace")),
    )
    load_exported_model(model, source_path)
    source_training, full_source_training = predict(model, training, batch_size, True)
    source_validation, full_source_validation = predict(
        model, validation, batch_size, True
    )
    assert full_source_training is not None and full_source_validation is not None
    attribution_path = attribution_path or source_path
    source_sha256 = sha256(source_path)
    attribution_sha256 = sha256(attribution_path)
    same_attribution_policy = attribution_sha256 == source_sha256
    if same_attribution_policy:
        attribution_training = source_training
    else:
        attribution_payload = json.loads(attribution_path.read_text())
        attribution_model = RangeConditionedPolicy(
            str(attribution_payload["architecture"]),
            str(attribution_payload.get("policyComposition", "replace")),
        )
        load_exported_model(attribution_model, attribution_path)
        attribution_training, _ = predict(
            attribution_model, training, batch_size, False
        )
        del attribution_model
    attribution_parity = source_parity_metrics(attribution_training, training)
    if same_attribution_policy and not attribution_parity["accepted"]:
        raise ValueError(
            "Rust/MLX attribution policy behavior parity failed: "
            f"{json.dumps(attribution_parity, sort_keys=True)}"
        )
    targets = mirror_descent_targets(
        source_training,
        training.action_values,
        training.action_masks,
        mirror_step_per_bb,
        maximum_target_node_kl,
    )
    validation_targets = mirror_descent_targets(
        source_validation,
        validation.action_values,
        validation.action_masks,
        mirror_step_per_bb,
        maximum_target_node_kl,
    )
    target_metrics = metrics(targets, training, frozen=source_training)
    validation_target_metrics = metrics(
        validation_targets, validation, frozen=source_validation
    )
    if any(
        measured["maximumReverseKlFromFrozen"] > maximum_target_node_kl + 1e-6
        or measured["weightedPolicyValueGainBb"] < -1e-9
        for measured in (target_metrics, validation_target_metrics)
    ):
        raise RuntimeError(
            "causal range-policy mirror targets are invalid: "
            f"{json.dumps({'training': target_metrics, 'validation': validation_target_metrics}, sort_keys=True)}"
        )

    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    optimizer = optim.Adam(learning_rate=learning_rate)

    def loss_fn(
        current_model: RangeConditionedPolicy,
        contexts: mx.array,
        queries: mx.array,
        projection: mx.array,
        actors: mx.array,
        actions: mx.array,
        masks: mx.array,
        focal_combos: mx.array,
        target: mx.array,
        frozen_full: mx.array,
        row_weights: mx.array,
    ) -> mx.array:
        logits = current_model(contexts, queries, projection, actors, actions, masks)
        log_probabilities = logits - mx.logsumexp(logits, axis=2, keepdims=True)
        selected = mx.take_along_axis(
            log_probabilities, focal_combos[:, None, None], axis=1
        ).squeeze(1)
        denominator = mx.maximum(mx.sum(row_weights), 1e-8)
        focal_loss = (
            -mx.sum(row_weights * mx.sum(target * selected, axis=1)) / denominator
        )
        selector = mx.stack((actors == 0, actors == 1), axis=1)[:, :, None]
        actor_reach = mx.sum(projection * selector, axis=1)
        actor_reach /= mx.maximum(mx.sum(actor_reach, axis=1, keepdims=True), 1e-8)
        anchor_cross_entropy = -mx.sum(frozen_full * log_probabilities, axis=2)
        anchor_loss = (
            mx.sum(row_weights * mx.sum(actor_reach * anchor_cross_entropy, axis=1))
            / denominator
        )
        return focal_loss + anchor_scale * anchor_loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    sampling = training.weights / training.weights.sum()
    best: dict[str, Any] | None = None
    best_parameters: Any = None
    losses: list[float] = []
    checkpoints: list[dict[str, Any]] = []
    for step_index in range(steps):
        if full_corpus_gradient:
            gradients = None
            step_loss = 0.0
            corpora = [
                (training, targets, full_source_training),
            ]
            if paired_corpus_gradient:
                corpora.append((validation, validation_targets, full_source_validation))
            corpus_mass = 1.0 / len(corpora)
            for corpus, corpus_targets, corpus_source in corpora:
                corpus_weight = float(corpus.weights.sum())
                for start in range(0, len(corpus.records), batch_size):
                    rows = np.arange(
                        start, min(start + batch_size, len(corpus.records))
                    )
                    row_weights = corpus.weights[rows].astype(np.float32)
                    chunk_fraction = (
                        corpus_mass * float(row_weights.sum()) / corpus_weight
                    )
                    loss, chunk_gradients = loss_and_grad(
                        model,
                        *_model_arguments(corpus, rows),
                        mx.array(corpus.focal_combos[rows]),
                        mx.array(corpus_targets[rows]),
                        mx.array(corpus_source[rows]),
                        mx.array(row_weights),
                    )
                    chunk_gradients = tree_map(
                        lambda value: value * chunk_fraction, chunk_gradients
                    )
                    gradients = (
                        chunk_gradients
                        if gradients is None
                        else tree_map(
                            lambda accumulated, value: accumulated + value,
                            gradients,
                            chunk_gradients,
                        )
                    )
                    mx.eval(gradients, loss)
                    step_loss += float(loss.item()) * chunk_fraction
            assert gradients is not None
        else:
            rows = rng.choice(
                len(training.records),
                size=min(batch_size, len(training.records)),
                replace=True,
                p=sampling,
            )
            loss, gradients = loss_and_grad(
                model,
                *_model_arguments(training, rows),
                mx.array(training.focal_combos[rows]),
                mx.array(targets[rows]),
                mx.array(full_source_training[rows]),
                mx.ones((len(rows),)),
            )
            mx.eval(loss, gradients)
            step_loss = float(loss.item())
        if maximum_gradient_norm > 0:
            gradients, _ = optim.clip_grad_norm(gradients, maximum_gradient_norm)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state)
        losses.append(step_loss)
        step = step_index + 1
        if step > 4 and step % evaluation_interval != 0 and step != steps:
            continue
        training_focal, training_full = predict(model, training, batch_size, True)
        validation_focal, validation_full = predict(model, validation, batch_size, True)
        assert training_full is not None and validation_full is not None
        training_metrics = metrics(
            training_focal,
            training,
            training_full,
            full_source_training,
            frozen=source_training,
        )
        validation_metrics = metrics(
            validation_focal,
            validation,
            validation_full,
            full_source_validation,
            frozen=source_validation,
        )
        feasible = all(
            measured["maximumReverseKlFromFrozen"]
            <= maximum_realized_node_kl * REALIZED_TRUST_REGION_SELECTION_FRACTION
            and measured["weightedReverseKlFromFrozen"]
            <= maximum_realized_weighted_kl * REALIZED_TRUST_REGION_SELECTION_FRACTION
            for measured in (training_metrics, validation_metrics)
        )

        def conservative_gain(measured: dict[str, float]) -> float:
            return measured.get(
                "weightedPolicyValueGainActionRolloutLowerBound99Bb",
                measured["weightedPolicyValueGainBb"],
            )

        rank = (
            int(feasible),
            min(
                conservative_gain(training_metrics),
                conservative_gain(validation_metrics),
            ),
            -max(
                training_metrics["weightedReverseKlFromFrozen"],
                validation_metrics["weightedReverseKlFromFrozen"],
            ),
            -step,
        )
        checkpoint = {
            "step": step,
            "insideRealizedTrustRegion": feasible,
            "rank": list(rank),
            "training": training_metrics,
            "validation": validation_metrics,
        }
        checkpoints.append(checkpoint)
        if best is None or tuple(checkpoint["rank"]) > tuple(best["rank"]):
            best = checkpoint
            best_parameters = tree_map(
                lambda value: np.asarray(value), model.parameters()
            )
    if best is None or best_parameters is None:
        raise RuntimeError("causal range-policy training produced no checkpoint")
    model.update(tree_map(mx.array, best_parameters))
    mx.eval(model.parameters())
    diagnostics = {
        "sourceParity": (attribution_parity if same_attribution_policy else None),
        "attributionParity": attribution_parity,
        "attributionParityRequiredForTraining": same_attribution_policy,
        "attributionRangePolicySha256": attribution_sha256,
        "targetAnchorRangePolicySha256": source_sha256,
        "target": target_metrics,
        "validationTarget": validation_target_metrics,
        "pairedCorpusGradient": paired_corpus_gradient,
        "firstLoss": losses[0],
        "finalLoss": losses[-1],
        "checkpoints": checkpoints,
        "selectedCheckpoint": best,
    }
    return model, source_payload, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-a", type=Path, required=True)
    parser.add_argument("--dataset-b", type=Path, required=True)
    parser.add_argument("--source-a", type=Path, required=True)
    parser.add_argument("--source-b", type=Path, required=True)
    parser.add_argument(
        "--attribution-a",
        type=Path,
        help="policy that generated dataset A; defaults to source A",
    )
    parser.add_argument(
        "--attribution-b",
        type=Path,
        help="policy that generated dataset B; defaults to source B",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path)
    parser.add_argument("--feature-workers", type=int, default=4)
    parser.add_argument("--maximum-records", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--mirror-step-per-bb", type=float, default=0.1)
    parser.add_argument("--maximum-target-node-kl", type=float, default=0.002)
    parser.add_argument("--maximum-realized-node-kl", type=float, default=0.005)
    parser.add_argument("--maximum-realized-weighted-kl", type=float, default=0.0015)
    parser.add_argument("--minimum-policy-value-gain-bb", type=float, default=1e-6)
    parser.add_argument("--anchor-scale", type=float, default=0.25)
    parser.add_argument("--maximum-gradient-norm", type=float, default=1.0)
    parser.add_argument("--evaluation-interval", type=int, default=25)
    parser.add_argument("--full-corpus-gradient", action="store_true")
    parser.add_argument("--paired-corpus-gradient", action="store_true")
    parser.add_argument("--seeds", default="18301,18302")
    parser.add_argument(
        "--rust-evaluator",
        type=Path,
        default=Path("target/release/preflop-solver"),
    )
    parser.add_argument(
        "--reuse-exact-dataset-parity",
        type=Path,
        help="reuse fully pinned exactRustDatasetParity from an earlier report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    if len(seeds) != 2 or seeds[0] == seeds[1]:
        raise ValueError("causal range-policy training needs two distinct seeds")
    for value in (
        args.steps,
        args.batch_size,
        args.feature_workers,
        args.maximum_records,
        args.evaluation_interval,
    ):
        if value <= 0:
            raise ValueError("causal range-policy counts must be positive")
    for value in (
        args.learning_rate,
        args.mirror_step_per_bb,
        args.maximum_target_node_kl,
        args.maximum_realized_node_kl,
        args.maximum_realized_weighted_kl,
        args.minimum_policy_value_gain_bb,
        args.anchor_scale,
        args.maximum_gradient_norm,
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("causal range-policy bounds must be positive")
    if args.paired_corpus_gradient and not args.full_corpus_gradient:
        raise ValueError("paired-corpus gradient requires --full-corpus-gradient")

    datasets = [
        load_dataset(
            path,
            args.maximum_records,
            args.feature_cache_dir,
            args.feature_workers,
        )
        for path in (args.dataset_a, args.dataset_b)
    ]
    sources = [args.source_a, args.source_b]
    attributions = [
        args.attribution_a or args.source_a,
        args.attribution_b or args.source_b,
    ]
    if datasets[0].metadata["seed"] == datasets[1].metadata["seed"]:
        raise ValueError("causal range-policy datasets need independent seeds")
    if datasets[0].metadata["depth_bb"] != datasets[1].metadata["depth_bb"]:
        raise ValueError("causal range-policy dataset depths differ")
    for dataset, attribution in zip(datasets, attributions, strict=True):
        if dataset.metadata["source_range_policy_sha256"] != sha256(attribution):
            raise ValueError("directional dataset does not pin its attribution policy")

    if args.reuse_exact_dataset_parity is not None:
        exact_dataset_parity_reports = reuse_exact_dataset_parity(
            args.reuse_exact_dataset_parity,
            datasets,
            sources,
            attributions,
        )
        exact_dataset_parity_source = str(args.reuse_exact_dataset_parity)
    else:

        def exact_dataset_parity(dataset_index: int) -> dict[str, Any]:
            report = rust_evaluate(
                args.rust_evaluator,
                sources[dataset_index],
                sources[dataset_index],
                attributions[dataset_index],
                datasets[dataset_index].path,
                args.minimum_policy_value_gain_bb,
                args.maximum_realized_node_kl,
                args.maximum_realized_weighted_kl,
            )
            if (
                report["maximumStoredSourceProbabilityDifference"] > 1e-6
                or report["maximumProbabilitySumError"] > 1e-6
            ):
                raise RuntimeError("exact Rust directional-dataset parity failed")
            return report

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            exact_dataset_parity_reports = list(
                executor.map(exact_dataset_parity, range(2))
            )
        exact_dataset_parity_source = "computed"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    students = []
    attribution_hashes = [dataset.sha256 for dataset in datasets]
    dataset_schemas = {dataset.metadata["schema"] for dataset in datasets}
    if dataset_schemas == {SELF_PLAY_SCHEMA}:
        provenance_key = "selfPlayRegretDatasetSha256s"
    elif dataset_schemas == {CAUSAL_SCHEMA}:
        provenance_key = "causalAttributionSha256s"
    else:
        provenance_key = "directionalDatasetSha256s"
    for index, seed in enumerate(seeds):
        model, source_payload, diagnostics = train_candidate(
            sources[index],
            datasets[index],
            datasets[1 - index],
            seed,
            args.steps,
            args.batch_size,
            args.learning_rate,
            args.mirror_step_per_bb,
            args.maximum_target_node_kl,
            args.maximum_realized_node_kl,
            args.maximum_realized_weighted_kl,
            args.anchor_scale,
            args.maximum_gradient_norm,
            args.evaluation_interval,
            args.full_corpus_gradient,
            attributions[index],
            args.paired_corpus_gradient,
        )
        output = args.output_dir / f"range-policy-seed-{seed}.json"
        export_model_from_source(
            model,
            source_payload,
            output,
            seed,
            sha256(sources[index]),
            attribution_hashes,
            provenance_key,
        )
        selected = diagnostics["selectedCheckpoint"]
        measurements = [selected["training"], selected["validation"]]
        python_accepted = all(
            value["weightedPolicyValueGainBb"] >= args.minimum_policy_value_gain_bb
            and value.get(
                "weightedPolicyValueGainActionRolloutLowerBound99Bb",
                value["weightedPolicyValueGainBb"],
            )
            >= args.minimum_policy_value_gain_bb
            and value["maximumReverseKlFromFrozen"] <= args.maximum_realized_node_kl
            and value["weightedReverseKlFromFrozen"]
            <= args.maximum_realized_weighted_kl
            for value in measurements
        )
        students.append(
            {
                "seed": seed,
                "source": str(sources[index]),
                "sourceSha256": sha256(sources[index]),
                "attribution": str(attributions[index]),
                "attributionSha256": sha256(attributions[index]),
                "network": str(output),
                "networkSha256": sha256(output),
                "diagnostics": diagnostics,
                "pythonPairedDirectionalTrustGate": python_accepted,
            }
        )
        del model

    evaluation_tasks = [
        (student_index, dataset_index)
        for student_index in range(2)
        for dataset_index in range(2)
    ]

    def exact_evaluation(task: tuple[int, int]) -> tuple[int, int, dict[str, Any]]:
        student_index, dataset_index = task
        evaluation = rust_evaluate(
            args.rust_evaluator,
            Path(students[student_index]["network"]),
            sources[student_index],
            attributions[dataset_index],
            datasets[dataset_index].path,
            args.minimum_policy_value_gain_bb,
            args.maximum_realized_node_kl,
            args.maximum_realized_weighted_kl,
        )
        return student_index, dataset_index, evaluation

    rust_evaluations: list[list[dict[str, Any] | None]] = [
        [None, None],
        [None, None],
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for student_index, dataset_index, evaluation in executor.map(
            exact_evaluation, evaluation_tasks
        ):
            rust_evaluations[student_index][dataset_index] = evaluation
    for student, evaluations in zip(students, rust_evaluations, strict=True):
        if any(evaluation is None for evaluation in evaluations):
            raise RuntimeError("paired exact Rust evaluation is incomplete")
        complete = [evaluation for evaluation in evaluations if evaluation is not None]
        student["rustEvaluations"] = complete
        student["passesPairedDirectionalTrustGate"] = bool(
            student["pythonPairedDirectionalTrustGate"]
            and all(
                evaluation["validation"]["status"]
                == "accepted_for_directional_evaluation"
                for evaluation in complete
            )
        )

    report = {
        "schema": REPORT_SCHEMA,
        "depthBb": datasets[0].metadata["depth_bb"],
        "datasets": [
            {
                "path": str(dataset.path),
                "sha256": dataset.sha256,
                "schema": dataset.metadata["schema"],
                "seed": dataset.metadata["seed"],
                "sourceRangePolicySha256": dataset.metadata[
                    "source_range_policy_sha256"
                ],
                "attributionRangePolicySha256": sha256(attributions[dataset_index]),
                "records": len(dataset.records),
                "featureCache": dataset.feature_dataset.feature_cache,
            }
            for dataset_index, dataset in enumerate(datasets)
        ],
        "steps": args.steps,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "mirrorStepPerBb": args.mirror_step_per_bb,
        "maximumTargetNodeKl": args.maximum_target_node_kl,
        "maximumRealizedNodeKl": args.maximum_realized_node_kl,
        "maximumRealizedWeightedKl": args.maximum_realized_weighted_kl,
        "realizedTrustRegionSelectionFraction": REALIZED_TRUST_REGION_SELECTION_FRACTION,
        "minimumPolicyValueGainBb": args.minimum_policy_value_gain_bb,
        "anchorScale": args.anchor_scale,
        "fullCorpusGradient": args.full_corpus_gradient,
        "pairedCorpusGradient": args.paired_corpus_gradient,
        "pairedCorpusWeighting": (
            "equal_corpus_mass" if args.paired_corpus_gradient else None
        ),
        "exactRustDatasetParity": exact_dataset_parity_reports,
        "exactRustDatasetParitySource": exact_dataset_parity_source,
        "updateOperator": (
            "mirror_prox_corrector_from_frozen_parent"
            if any(
                sha256(source) != sha256(attribution)
                for source, attribution in zip(sources, attributions, strict=True)
            )
            else "current_profile_mirror_descent"
        ),
        "students": students,
        "acceptedForDirectionalEvaluation": all(
            student["passesPairedDirectionalTrustGate"] for student in students
        ),
        "activationEligible": False,
        "activationReason": "full-game exploitability and release gates remain mandatory",
    }
    report_path = args.output_dir / "report.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(report_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
