#!/usr/bin/env python3
"""Distill paired exact-range postflop solver policies into served networks."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
import numpy as np

from train_public_value_network import (
    COMBO_CONFLICTS,
    COMBO_COUNT,
    CONTEXT_COUNT,
    CONTEXT_BOARD_RELATIVE_COUNT,
    QUERY_COUNT,
    QUERY_BOARD_RELATIVE_COUNT,
    RANGE_POLICY_FEATURE_SCHEMA as DATASET_RANGE_POLICY_FEATURE_SCHEMA,
    build_features,
    tower_payload,
)

DATASET_SCHEMA = "hu-range-conditioned-postflop-policy-dataset-v1"
NETWORK_SCHEMA = "hu-public-belief-combo-policy-network-v1"
RANGE_POLICY_FEATURE_SCHEMA = "rank-suit-invariant-combo-policy-query-v2"
ACTION_FEATURE_SCHEMA = "hu-cash-legal-action-v1"
ACTION_FEATURE_COUNT = 9
BASE_CONTEXT_SIZE = CONTEXT_COUNT + CONTEXT_BOARD_RELATIVE_COUNT
QUERY_SIZE = QUERY_COUNT + QUERY_BOARD_RELATIVE_COUNT
MAX_TRAJECTORY_ACTIONS = 32
TRAJECTORY_FEATURE_COUNT = 15
PUBLIC_STATE_FEATURE_COUNT = 20 + MAX_TRAJECTORY_ACTIONS * TRAJECTORY_FEATURE_COUNT
CONTEXT_SIZE = BASE_CONTEXT_SIZE + PUBLIC_STATE_FEATURE_COUNT
STREETS = ("preflop", "flop", "turn", "river")
TRAJECTORY_KINDS = ("fold", "check", "call", "bet", "raise", "all_in")
FEATURE_CACHE_IMPLEMENTATION = "range-policy-features-v2-public-trajectory-1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class LoadedDataset:
    path: Path
    sha256: str
    metadata: dict[str, Any]
    records: list[dict[str, Any]]
    boards: list[np.ndarray]
    actors: np.ndarray
    invested: np.ndarray
    ranges: np.ndarray
    masses: np.ndarray
    projection_weights: np.ndarray
    actions: np.ndarray
    action_masks: np.ndarray
    source_probabilities: np.ndarray
    targets: np.ndarray
    action_values: np.ndarray
    combo_weights: np.ndarray
    target_corpus_sha256: str = ""
    feature_cache: dict[str, Any] | None = None
    contexts: np.ndarray | None = None
    queries: np.ndarray | None = None


def read_records(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        metadata = json.loads(next(stream))
        records = [json.loads(line) for line in stream if line.strip()]
    if (
        metadata.get("record_type") != "metadata"
        or metadata.get("schema") != DATASET_SCHEMA
        or metadata.get("feature_schema") != DATASET_RANGE_POLICY_FEATURE_SCHEMA
        or metadata.get("context_size") != BASE_CONTEXT_SIZE
        or metadata.get("query_size") != QUERY_SIZE
        or metadata.get("action_feature_schema") != ACTION_FEATURE_SCHEMA
        or metadata.get("action_feature_count") != ACTION_FEATURE_COUNT
        or metadata.get("teacher", {}).get("validation", {}).get("status")
        != "accepted_for_training"
        or metadata.get("records") != len(records)
        or not records
    ):
        raise ValueError(f"incompatible or unvalidated range-policy dataset: {path}")
    return metadata, records


def inspect_dataset(path: Path) -> tuple[dict[str, Any], int]:
    """Validate a dataset stream without retaining its large target tensors."""
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        metadata = json.loads(next(stream))
        if (
            metadata.get("record_type") != "metadata"
            or metadata.get("schema") != DATASET_SCHEMA
            or metadata.get("feature_schema") != DATASET_RANGE_POLICY_FEATURE_SCHEMA
            or metadata.get("context_size") != BASE_CONTEXT_SIZE
            or metadata.get("query_size") != QUERY_SIZE
            or metadata.get("action_feature_schema") != ACTION_FEATURE_SCHEMA
            or metadata.get("action_feature_count") != ACTION_FEATURE_COUNT
            or metadata.get("teacher", {}).get("validation", {}).get("status")
            != "accepted_for_training"
            or not isinstance(metadata.get("records"), int)
            or metadata["records"] <= 0
        ):
            raise ValueError(f"incompatible or unvalidated range-policy dataset: {path}")
        maximum_actions = 0
        records = 0
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            maximum_actions = max(
                maximum_actions, len(record.get("action_labels", []))
            )
            records += 1
    if records != metadata["records"] or maximum_actions <= 0:
        raise ValueError(f"range-policy dataset record count mismatch: {path}")
    return metadata, maximum_actions


def cap_dataset(source: Path, output: Path, capacity: int) -> Path:
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        header = json.loads(next(stream))
    if isinstance(header.get("records"), int) and header["records"] <= capacity:
        return source
    metadata, records = read_records(source)
    if capacity < 12:
        raise ValueError("range-policy corpus cap must leave room for every street")
    grouped = {
        street: [
            record
            for record in records
            if record.get("state", {}).get("street") == street
        ]
        for street in ("flop", "turn", "river")
    }
    if any(not candidates for candidates in grouped.values()):
        raise ValueError("range-policy source corpus must contain every postflop street")
    selected = reach_priority_cap(records, capacity)
    source_reach = sum(float(record["weight"]) for record in records)
    retained_reach = sum(float(record["weight"]) for record in selected)
    capped_metadata = dict(metadata)
    capped_metadata["records"] = len(selected)
    capped_metadata["subset_of_sha256"] = sha256(source)
    capped_metadata["subset"] = "deterministic_authentic_reach_priority_v1"
    capped_metadata["inverse_inclusion_weight_correction"] = False
    capped_metadata["retained_authentic_reach_fraction"] = (
        retained_reach / source_reach
    )
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            compressed.write(
                (json.dumps(capped_metadata, separators=(",", ":")) + "\n").encode()
            )
            for record in selected:
                compressed.write(
                    (json.dumps(record, separators=(",", ":")) + "\n").encode()
                )
    temporary.replace(output)
    return output


def reach_priority_cap(
    records: list[dict[str, Any]], capacity: int
) -> list[dict[str, Any]]:
    """Retain the action states carrying the most authentic public reach.

    A uniform reservoir is not representative when a complete betting tree's
    node reaches span many orders of magnitude.  Inverse-count correction is
    also invalid for a deterministic corpus because it can multiply the mass
    of a retained high-reach node.  Reserve enough rows for train/heldout
    coverage on every street, then fill the cap by authentic reach without
    changing any retained weight.
    """
    if capacity < 6 or capacity >= len(records):
        raise ValueError("reach-priority cap must remove records and cover streets")

    def rank(index: int) -> tuple[float, bytes]:
        weight = float(records[index].get("weight", float("nan")))
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("range-policy cap needs positive authentic reach")
        tie_break = hashlib.sha256(
            b"hu-range-policy-reach-priority-v1\0"
            + record_selection_identity(records[index])
        ).digest()
        return -weight, tie_break

    streets = ("flop", "turn", "river")
    by_street = {
        street: [
            index
            for index, record in enumerate(records)
            if record.get("state", {}).get("street") == street
        ]
        for street in streets
    }
    if any(len(indices) < 2 for indices in by_street.values()):
        raise ValueError("range-policy cap needs two or more rows per street")
    selected_indices: set[int] = set()
    minimum_per_street = min(4, capacity // len(streets))
    for street in streets:
        retained = sorted(by_street[street], key=rank)[
            : min(minimum_per_street, len(by_street[street]))
        ]
        selected_indices.update(retained)
    for index in sorted(range(len(records)), key=rank):
        if len(selected_indices) >= capacity:
            break
        selected_indices.add(index)
    if len(selected_indices) != capacity:
        raise RuntimeError("reach-priority cap did not select the requested capacity")
    return [records[index] for index in sorted(selected_indices, key=rank)]


def record_selection_identity(record: dict[str, Any]) -> bytes:
    """Identify a solver target without depending on its attached source policy."""
    return json.dumps(
        canonical_training_numbers(
            {
                "state": record.get("state"),
                "action_labels": record.get("action_labels"),
            }
        ),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def target_corpus_sha256(dataset: LoadedDataset) -> str:
    if len(dataset.target_corpus_sha256) != 64:
        raise ValueError("range-policy target corpus hash is unavailable")
    return dataset.target_corpus_sha256


def target_record_identity(record: dict[str, Any]) -> bytes:
    """Canonicalize a solver target across Rust's typed JSON round trip.

    Every numeric tensor is loaded as float32 below.  Canonicalize the hash to
    that same representation so harmless decimal reformatting by serde does
    not make a cross-augmented copy look like a different training target.
    """
    target = dict(record)
    target.pop("source_policy_probabilities", None)
    return json.dumps(
        canonical_training_numbers(target), sort_keys=True, separators=(",", ":")
    ).encode()


def canonical_training_numbers(value: Any) -> Any:
    if isinstance(value, float):
        return float(np.float32(value))
    if isinstance(value, list):
        return [canonical_training_numbers(item) for item in value]
    if isinstance(value, dict):
        return {
            key: canonical_training_numbers(item) for key, item in value.items()
        }
    return value


def load_dataset(path: Path, maximum_actions: int) -> LoadedDataset:
    metadata, records = read_records(path)
    count = len(records)
    target_digest = hashlib.sha256(b"hu-range-policy-target-corpus-v1\0")
    record_summaries: list[dict[str, Any]] = []
    boards: list[np.ndarray] = []
    actors = np.empty(count, dtype=np.int32)
    invested = np.empty((count, 2), dtype=np.float32)
    ranges = np.empty((count, 2, COMBO_COUNT), dtype=np.float32)
    actions = np.zeros((count, maximum_actions, ACTION_FEATURE_COUNT), dtype=np.float32)
    action_masks = np.zeros((count, maximum_actions), dtype=np.float32)
    source_probabilities = np.zeros(
        (count, COMBO_COUNT, maximum_actions), dtype=np.float32
    )
    targets = np.zeros((count, COMBO_COUNT, maximum_actions), dtype=np.float32)
    action_values = np.zeros_like(targets)
    node_weights = np.empty(count, dtype=np.float64)
    for index, record in enumerate(records):
        target_digest.update(target_record_identity(record))
        target_digest.update(b"\n")
        state = record.get("state", {})
        action_count = len(record.get("action_labels", []))
        record_ranges = np.asarray(record.get("ranges"), dtype=np.float32)
        record_actions = np.asarray(record.get("action_features"), dtype=np.float32)
        record_targets = np.asarray(record.get("probabilities"), dtype=np.float32)
        record_source = np.asarray(
            record.get("source_policy_probabilities", []), dtype=np.float32
        )
        record_values = np.asarray(record.get("action_values_bb"), dtype=np.float32)
        if (
            record.get("record_type") != "range_conditioned_average_strategy"
            or not 0 < action_count <= maximum_actions
            or record_ranges.shape != (2, COMBO_COUNT)
            or record_actions.shape != (action_count, ACTION_FEATURE_COUNT)
            or record_targets.shape != (COMBO_COUNT * action_count,)
            or record_source.shape not in (
                (0,),
                (COMBO_COUNT * action_count,),
            )
            or record_values.shape != (COMBO_COUNT * action_count,)
            or not np.all(np.isfinite(record_ranges))
            or not np.all(record_ranges >= 0)
            or not np.all(np.isfinite(record_actions))
            or not np.all(np.isfinite(record_targets))
            or not np.all(record_targets >= 0)
            or not np.all(np.isfinite(record_values))
        ):
            raise ValueError(f"invalid range-policy record {index} in {path}")
        actor = int(state.get("actor", -1))
        board = np.asarray(state.get("board"), dtype=np.int16)
        if actor not in (0, 1) or board.shape not in ((3,), (4,), (5,)):
            raise ValueError(f"invalid public state in record {index} of {path}")
        totals = record_ranges.sum(axis=1)
        if np.any(np.abs(totals - 1.0) > 1e-5):
            raise ValueError(f"unnormalized public ranges in record {index} of {path}")
        boards.append(board)
        actors[index] = actor
        invested[index] = np.asarray(state.get("invested_bb"), dtype=np.float32)
        ranges[index] = record_ranges
        actions[index, :action_count] = record_actions
        action_masks[index, :action_count] = 1.0
        targets[index, :, :action_count] = record_targets.reshape(
            COMBO_COUNT, action_count
        )
        if record_source.size:
            source_probabilities[index, :, :action_count] = record_source.reshape(
                COMBO_COUNT, action_count
            )
        action_values[index, :, :action_count] = record_values.reshape(
            COMBO_COUNT, action_count
        )
        node_weights[index] = float(record.get("weight", 0.0))
        if not np.isfinite(node_weights[index]) or node_weights[index] <= 0:
            raise ValueError(f"invalid reach weight in record {index} of {path}")
        record_summaries.append(
            {
                "record_type": record["record_type"],
                "weight": record["weight"],
                "state": state,
                "action_labels": record["action_labels"],
            }
        )
    del records
    masses = np.maximum(
        ranges.sum(axis=2)[:, ::-1, None]
        - ranges[:, ::-1, :][:, :, COMBO_CONFLICTS].sum(axis=3),
        0.0,
    ).astype(np.float32)
    projection = ranges * masses
    combo_weights = np.empty((count, COMBO_COUNT), dtype=np.float32)
    for row, actor in enumerate(actors):
        joint = max(float(projection[row, actor].sum()), 1e-12)
        combo_weights[row] = (
            projection[row, actor] / joint * node_weights[row]
        ).astype(np.float32)
        reachable = combo_weights[row] > 0
        sums = targets[row].sum(axis=1)
        if np.any(np.abs(sums[reachable] - 1.0) > 1e-5):
            raise ValueError(f"teacher probabilities do not normalize in record {row}")
    return LoadedDataset(
        path=path,
        sha256=sha256(path),
        metadata=metadata,
        records=record_summaries,
        boards=boards,
        actors=actors,
        invested=invested,
        ranges=ranges,
        masses=masses,
        projection_weights=projection,
        actions=actions,
        action_masks=action_masks,
        source_probabilities=source_probabilities,
        targets=targets,
        action_values=action_values,
        combo_weights=combo_weights,
        target_corpus_sha256=target_digest.hexdigest(),
    )


def build_range_feature_task(
    task: tuple[
        dict[str, Any],
        np.ndarray,
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
    ]
) -> tuple[np.ndarray, np.ndarray]:
    state, board, actor, invested, ranges, masses, depth_bb = task
    context, queries = build_features(
        board,
        actor,
        invested,
        ranges,
        masses,
        DATASET_RANGE_POLICY_FEATURE_SCHEMA,
    )
    public_state = range_policy_state_features(state, depth_bb)
    context = np.concatenate(
        (context, np.broadcast_to(public_state, (2, len(public_state)))), axis=1
    )
    return context, queries


def range_feature_cache_key(dataset: LoadedDataset) -> str:
    payload = {
        "implementation": FEATURE_CACHE_IMPLEMENTATION,
        "datasetSha256": dataset.sha256,
        "featureSchema": RANGE_POLICY_FEATURE_SCHEMA,
        "contextSize": CONTEXT_SIZE,
        "querySize": QUERY_SIZE,
        "records": len(dataset.records),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_feature_cache_array(
    path: Path,
    expected_shape: tuple[int, ...],
    expected_sha256: str,
) -> np.memmap:
    if not path.is_file() or sha256(path) != expected_sha256:
        raise RuntimeError(f"range-policy feature cache integrity failure: {path}")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.shape != expected_shape or values.dtype != np.float32:
        raise RuntimeError(f"range-policy feature cache shape mismatch: {path}")
    return values


def add_features(
    dataset: LoadedDataset,
    cache_dir: Path | None = None,
    workers: int = 1,
) -> None:
    if workers <= 0:
        raise ValueError("range-policy feature workers must be positive")
    depth_bb = float(dataset.metadata["depth_bb"])
    count = len(dataset.records)
    context_shape = (count, 2, CONTEXT_SIZE)
    query_shape = (count, 2, COMBO_COUNT, QUERY_SIZE)
    if cache_dir is None:
        built = [
            build_range_feature_task(
                (
                    record["state"],
                    board,
                    int(actor),
                    invested,
                    ranges,
                    masses,
                    depth_bb,
                )
            )
            for record, board, actor, invested, ranges, masses in zip(
                dataset.records,
                dataset.boards,
                dataset.actors,
                dataset.invested,
                dataset.ranges,
                dataset.masses,
                strict=True,
            )
        ]
        dataset.contexts = np.stack([context for context, _ in built])
        dataset.queries = np.stack([queries for _, queries in built])
        dataset.feature_cache = {"enabled": False, "hit": False}
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    key = range_feature_cache_key(dataset)
    context_path = cache_dir / f"{key}-contexts.npy"
    query_path = cache_dir / f"{key}-queries.npy"
    metadata_path = cache_dir / f"{key}.json"
    expected = {
        "schema": "hu-range-policy-feature-cache-v1",
        "key": key,
        "implementation": FEATURE_CACHE_IMPLEMENTATION,
        "datasetSha256": dataset.sha256,
        "featureSchema": RANGE_POLICY_FEATURE_SCHEMA,
        "records": count,
        "contextShape": list(context_shape),
        "queryShape": list(query_shape),
        "dtype": "float32",
    }
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        for field, value in expected.items():
            if metadata.get(field) != value:
                raise RuntimeError(
                    f"range-policy feature cache metadata mismatch for {field}"
                )
        dataset.contexts = validate_feature_cache_array(
            context_path, context_shape, metadata["contextsSha256"]
        )
        dataset.queries = validate_feature_cache_array(
            query_path, query_shape, metadata["queriesSha256"]
        )
        dataset.feature_cache = {
            "enabled": True,
            "hit": True,
            "key": key,
            "metadata": str(metadata_path),
            "bytes": context_path.stat().st_size + query_path.stat().st_size,
        }
        return

    projected_bytes = (
        int(np.prod(context_shape)) + int(np.prod(query_shape))
    ) * np.dtype(np.float32).itemsize
    if projected_bytes > 20 * 1024**3:
        raise RuntimeError("range-policy feature cache exceeds the 20GB safety ceiling")
    free_bytes = shutil.disk_usage(cache_dir).free
    if free_bytes < projected_bytes + 1024**3:
        raise RuntimeError("insufficient disk headroom for range-policy feature cache")

    context_temporary = Path(f"{context_path}.tmp")
    query_temporary = Path(f"{query_path}.tmp")
    context_temporary.unlink(missing_ok=True)
    query_temporary.unlink(missing_ok=True)
    contexts = np.lib.format.open_memmap(
        context_temporary,
        mode="w+",
        dtype=np.float32,
        shape=context_shape,
    )
    queries = np.lib.format.open_memmap(
        query_temporary,
        mode="w+",
        dtype=np.float32,
        shape=query_shape,
    )

    def task(row: int) -> tuple[Any, ...]:
        return (
            dataset.records[row]["state"],
            dataset.boards[row],
            int(dataset.actors[row]),
            dataset.invested[row],
            dataset.ranges[row],
            dataset.masses[row],
            depth_bb,
        )

    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    chunk_size = max(1, workers * 2)
    try:
        for start in range(0, count, chunk_size):
            end = min(count, start + chunk_size)
            tasks = [task(row) for row in range(start, end)]
            results = (
                executor.map(build_range_feature_task, tasks, chunksize=1)
                if executor is not None
                else map(build_range_feature_task, tasks)
            )
            for row, (context, query) in zip(range(start, end), results, strict=True):
                contexts[row] = context
                queries[row] = query
        contexts.flush()
        queries.flush()
    except BaseException:
        del contexts, queries
        context_temporary.unlink(missing_ok=True)
        query_temporary.unlink(missing_ok=True)
        raise
    finally:
        if executor is not None:
            executor.shutdown()
    del contexts, queries
    context_temporary.replace(context_path)
    query_temporary.replace(query_path)
    metadata = {
        **expected,
        "contextsSha256": sha256(context_path),
        "queriesSha256": sha256(query_path),
    }
    metadata_temporary = Path(f"{metadata_path}.tmp")
    metadata_temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    metadata_temporary.replace(metadata_path)
    dataset.contexts = np.load(context_path, mmap_mode="r", allow_pickle=False)
    dataset.queries = np.load(query_path, mmap_mode="r", allow_pickle=False)
    dataset.feature_cache = {
        "enabled": True,
        "hit": False,
        "key": key,
        "metadata": str(metadata_path),
        "bytes": context_path.stat().st_size + query_path.stat().st_size,
    }


def range_policy_state_features(
    state: dict[str, Any], depth_bb: float
) -> np.ndarray:
    """Encode the complete public action state used by the policy target.

    Exact ranges alone do not identify whether a passive action closes the
    street, which player has position, or which perfect-recall betting line
    reached the belief state.  Keep those public distinctions explicit.
    """
    if not np.isfinite(depth_bb) or depth_bb <= 0:
        raise ValueError("range-policy depth must be finite and positive")
    street = str(state.get("street", ""))
    actor = int(state.get("actor", -1))
    trajectory = state.get("trajectory", [])
    invested = np.asarray(state.get("invested_bb"), dtype=np.float32)
    street_invested = np.asarray(
        state.get("street_invested_bb"), dtype=np.float32
    )
    if (
        street not in STREETS
        or actor not in (0, 1)
        or not isinstance(trajectory, list)
        or len(trajectory) > MAX_TRAJECTORY_ACTIONS
        or invested.shape != (2,)
        or street_invested.shape != (2,)
        or not np.all(np.isfinite(invested))
        or not np.all(np.isfinite(street_invested))
    ):
        raise ValueError("invalid range-policy public action state")
    opponent = 1 - actor
    last_full_raise = float(state.get("last_full_raise_bb", 0.0))
    aggressions = int(state.get("aggressions", -1))
    checks = int(state.get("checks", -1))
    board = state.get("board", [])
    if (
        not np.isfinite(last_full_raise)
        or last_full_raise <= 0
        or aggressions < 0
        or checks not in (0, 1)
        or not isinstance(board, list)
        or len(board) not in (3, 4, 5)
    ):
        raise ValueError("invalid range-policy betting state")
    features = np.zeros(PUBLIC_STATE_FEATURE_COUNT, dtype=np.float32)
    features[STREETS.index(street)] = 1.0
    features[4 + actor] = 1.0
    settled_pot = float(invested.sum() - street_invested.sum())
    to_call = max(float(street_invested[opponent] - street_invested[actor]), 0.0)
    features[6:18] = np.asarray(
        [
            settled_pot / depth_bb,
            (depth_bb - float(invested[actor])) / depth_bb,
            (depth_bb - float(invested[opponent])) / depth_bb,
            float(street_invested[actor]) / depth_bb,
            float(street_invested[opponent]) / depth_bb,
            float(invested[actor]) / depth_bb,
            float(invested[opponent]) / depth_bb,
            to_call / depth_bb,
            last_full_raise / depth_bb,
            1.0 if state.get("raise_reopened") else 0.0,
            len(board) / 5.0,
            len(trajectory) / MAX_TRAJECTORY_ACTIONS,
        ],
        dtype=np.float32,
    )
    features[18] = aggressions / 2.0
    features[19] = float(checks)
    for index, action in enumerate(trajectory):
        action_actor = int(action.get("actor", -1))
        action_street = str(action.get("street", ""))
        kind = str(action.get("kind", ""))
        amount = float(action.get("amount_bb", float("nan")))
        amount_to = float(action.get("amount_to_bb") or 0.0)
        pot_after = float(action.get("pot_after_bb", float("nan")))
        if (
            action_actor not in (0, 1)
            or action_street not in STREETS
            or kind not in TRAJECTORY_KINDS
            or not all(np.isfinite(value) for value in (amount, amount_to, pot_after))
        ):
            raise ValueError("invalid range-policy trajectory action")
        offset = 20 + index * TRAJECTORY_FEATURE_COUNT
        features[offset + action_actor] = 1.0
        features[offset + 2 + STREETS.index(action_street)] = 1.0
        features[offset + 6 + TRAJECTORY_KINDS.index(kind)] = 1.0
        features[offset + 12 : offset + 15] = [
            amount / depth_bb,
            amount_to / depth_bb,
            pot_after / depth_bb,
        ]
    if not np.all(np.isfinite(features)):
        raise ValueError("range-policy public action features are non-finite")
    return features


class RangeConditionedPolicy(nn.Module):
    def __init__(
        self, architecture: str = "compact", composition: str = "replace"
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.composition = composition
        if architecture == "compact":
            self.embedding_size = 64
            self.action_embedding_size = 32
            self.context_tower = nn.Sequential(
                nn.Linear(CONTEXT_SIZE, 128),
                nn.GELU(approx="fast"),
                nn.Linear(128, self.embedding_size),
                nn.GELU(approx="fast"),
            )
            self.query_tower = nn.Sequential(
                nn.Linear(QUERY_SIZE, 128),
                nn.GELU(approx="fast"),
                nn.Linear(128, self.embedding_size),
                nn.GELU(approx="fast"),
            )
            self.action_tower = nn.Sequential(
                nn.Linear(ACTION_FEATURE_COUNT, 32),
                nn.GELU(approx="fast"),
                nn.Linear(32, self.action_embedding_size),
                nn.GELU(approx="fast"),
            )
            self.head = nn.Sequential(
                nn.Linear(self.embedding_size * 4 + self.action_embedding_size, 128),
                nn.GELU(approx="fast"),
                nn.Linear(128, 64),
                nn.GELU(approx="fast"),
                nn.Linear(64, 1),
            )
        elif architecture == "wide":
            self.embedding_size = 128
            self.action_embedding_size = 64
            self.context_tower = nn.Sequential(
                nn.Linear(CONTEXT_SIZE, 256),
                nn.GELU(approx="fast"),
                nn.Linear(256, self.embedding_size),
                nn.GELU(approx="fast"),
            )
            self.query_tower = nn.Sequential(
                nn.Linear(QUERY_SIZE, 192),
                nn.GELU(approx="fast"),
                nn.Linear(192, self.embedding_size),
                nn.GELU(approx="fast"),
            )
            self.action_tower = nn.Sequential(
                nn.Linear(ACTION_FEATURE_COUNT, 64),
                nn.GELU(approx="fast"),
                nn.Linear(64, self.action_embedding_size),
                nn.GELU(approx="fast"),
            )
            self.head = nn.Sequential(
                nn.Linear(self.embedding_size * 4 + self.action_embedding_size, 256),
                nn.GELU(approx="fast"),
                nn.Linear(256, 128),
                nn.GELU(approx="fast"),
                nn.Linear(128, 64),
                nn.GELU(approx="fast"),
                nn.Linear(64, 1),
            )
        else:
            raise ValueError(f"unknown range-policy architecture {architecture}")
        if composition not in ("replace", "source_bundle_logit_residual"):
            raise ValueError(f"unknown range-policy composition {composition}")
        if composition == "source_bundle_logit_residual":
            final = self.head.layers[-1]
            final.weight = mx.zeros_like(final.weight)
            if final.bias is not None:
                final.bias = mx.zeros_like(final.bias)

    def __call__(
        self,
        contexts: mx.array,
        queries: mx.array,
        projection_weights: mx.array,
        actors: mx.array,
        actions: mx.array,
        action_masks: mx.array,
        source_probabilities: mx.array | None = None,
    ) -> mx.array:
        context_embedding = self.context_tower(contexts)
        query_embedding = self.query_tower(queries)
        action_embedding = self.action_tower(actions)
        normalized = projection_weights / mx.maximum(
            mx.sum(projection_weights, axis=2, keepdims=True), 1e-8
        )
        pooled = mx.sum(query_embedding * normalized[:, :, :, None], axis=2)
        selector = mx.stack((actors == 0, actors == 1), axis=1)[:, :, None]
        own_context = mx.sum(context_embedding * selector, axis=1)
        own_pool = mx.sum(pooled * selector, axis=1)
        opponent_pool = mx.sum(pooled * selector[:, ::-1, :], axis=1)
        hand_query = mx.sum(query_embedding * selector[:, :, None, :], axis=1)
        batch, hands, actions_count = (
            hand_query.shape[0],
            hand_query.shape[1],
            action_embedding.shape[1],
        )
        combined = mx.concatenate(
            (
                mx.broadcast_to(
                    own_context[:, None, None, :],
                    (batch, hands, actions_count, self.embedding_size),
                ),
                mx.broadcast_to(
                    own_pool[:, None, None, :],
                    (batch, hands, actions_count, self.embedding_size),
                ),
                mx.broadcast_to(
                    opponent_pool[:, None, None, :],
                    (batch, hands, actions_count, self.embedding_size),
                ),
                mx.broadcast_to(
                    hand_query[:, :, None, :],
                    (batch, hands, actions_count, self.embedding_size),
                ),
                mx.broadcast_to(
                    action_embedding[:, None, :, :],
                    (batch, hands, actions_count, self.action_embedding_size),
                ),
            ),
            axis=3,
        )
        logits = self.head(combined).reshape((batch, hands, actions_count))
        if self.composition == "source_bundle_logit_residual":
            if source_probabilities is None:
                raise ValueError("residual policy requires source probabilities")
            logits = logits + mx.log(mx.maximum(source_probabilities, 1e-12))
        return mx.where(action_masks[:, None, :] > 0, logits, -1e9)


def split_rows(dataset: LoadedDataset) -> tuple[np.ndarray, np.ndarray]:
    hashes = []
    streets = []
    for record in dataset.records:
        identity = json.dumps(
            {
                "board": record["state"]["board"],
                "history": record["state"]["public_history"],
                "actor": record["state"]["actor"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        hashes.append(int.from_bytes(hashlib.sha256(identity).digest()[:8], "little"))
        streets.append(record["state"]["street"])
    if len(hashes) < 2:
        raise ValueError("range-policy distillation needs at least two public nodes")
    hashes_array = np.asarray(hashes, dtype=np.uint64)
    streets_array = np.asarray(streets)
    training: list[int] = []
    heldout: list[int] = []
    for street in ("flop", "turn", "river"):
        rows = np.flatnonzero(streets_array == street)
        if len(rows) < 2:
            raise ValueError(f"range-policy split needs two or more {street} nodes")
        order = rows[np.argsort(hashes_array[rows])]
        heldout_count = max(1, len(order) // 5)
        heldout.extend(int(row) for row in order[:heldout_count])
        training.extend(int(row) for row in order[heldout_count:])
    return np.sort(np.asarray(training)), np.sort(np.asarray(heldout))


def reach_sampling_probabilities(
    combo_weights: np.ndarray, rows: np.ndarray
) -> np.ndarray:
    """Sample public nodes in proportion to their authentic reach mass.

    ``combo_weights`` already contains public-node reach multiplied by the
    acting player's conditional hand reach.  Sampling rows uniformly and then
    dividing by each minibatch's sampled mass is a biased self-normalized
    estimator when node reaches are highly skewed.  Sampling by node mass and
    conditioning the per-combo weights below gives an unbiased estimator of
    the reach-weighted policy objective.
    """
    masses = np.asarray(combo_weights[rows].sum(axis=1), dtype=np.float64)
    total = float(masses.sum())
    if (
        masses.shape != (len(rows),)
        or not np.all(np.isfinite(masses))
        or np.any(masses <= 0)
        or not np.isfinite(total)
        or total <= 0
    ):
        raise ValueError("range-policy training rows need positive reach mass")
    return masses / total


def batch(
    dataset: LoadedDataset,
    rows: np.ndarray,
    condition_on_node_reach: bool = False,
) -> tuple[mx.array, ...]:
    assert dataset.contexts is not None and dataset.queries is not None
    combo_weights = dataset.combo_weights[rows]
    if condition_on_node_reach:
        node_masses = combo_weights.sum(axis=1, keepdims=True)
        if (
            not np.all(np.isfinite(node_masses))
            or np.any(node_masses <= 0)
        ):
            raise ValueError("range-policy batch needs positive node reach mass")
        combo_weights = combo_weights / node_masses
    return (
        mx.array(dataset.contexts[rows]),
        mx.array(dataset.queries[rows]),
        mx.array(dataset.projection_weights[rows]),
        mx.array(dataset.actors[rows]),
        mx.array(dataset.actions[rows]),
        mx.array(dataset.action_masks[rows]),
        mx.array(dataset.targets[rows]),
        mx.array(dataset.action_values[rows]),
        mx.array(combo_weights),
        mx.array(dataset.source_probabilities[rows]),
    )


def concatenate_training_batches(
    first: tuple[mx.array, ...], second: tuple[mx.array, ...]
) -> tuple[mx.array, ...]:
    if len(first) != len(second):
        raise ValueError("range-policy training batches are incompatible")
    return tuple(
        mx.concatenate((left, right), axis=0)
        for left, right in zip(first, second, strict=True)
    )


def train(
    primary: LoadedDataset,
    auxiliary: LoadedDataset,
    primary_rows: np.ndarray,
    auxiliary_rows: np.ndarray,
    primary_heldout: np.ndarray,
    auxiliary_heldout: np.ndarray,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    final_learning_rate: float | None,
    auxiliary_probability: float,
    ev_regret_scale: float,
    architecture: str,
    composition: str,
    maximum_weighted_kl: float,
    minimum_primary_agreement: float,
    maximum_teacher_ev_loss_bb: float,
    balanced_teacher_batches: bool,
) -> tuple[RangeConditionedPolicy, list[float], dict[str, Any]]:
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = RangeConditionedPolicy(architecture, composition)
    mx.eval(model.parameters())
    schedule = (
        learning_rate
        if final_learning_rate is None
        else optim.cosine_decay(learning_rate, steps, final_learning_rate)
    )
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=1e-5)
    primary_sampling_probabilities = reach_sampling_probabilities(
        primary.combo_weights, primary_rows
    )
    auxiliary_sampling_probabilities = reach_sampling_probabilities(
        auxiliary.combo_weights, auxiliary_rows
    )

    def loss_fn(
        current: RangeConditionedPolicy,
        contexts: mx.array,
        queries: mx.array,
        projection: mx.array,
        actors: mx.array,
        actions: mx.array,
        action_masks: mx.array,
        targets: mx.array,
        values: mx.array,
        weights: mx.array,
        source_probabilities: mx.array,
    ) -> mx.array:
        logits = current(
            contexts,
            queries,
            projection,
            actors,
            actions,
            action_masks,
            source_probabilities,
        )
        log_probabilities = logits - mx.logsumexp(logits, axis=2, keepdims=True)
        cross_entropy = -mx.sum(targets * log_probabilities, axis=2)
        denominator = mx.maximum(mx.sum(weights), 1e-8)
        loss = mx.sum(weights * cross_entropy) / denominator
        if ev_regret_scale > 0:
            probabilities = mx.softmax(logits, axis=2)
            masked_values = mx.where(action_masks[:, None, :] > 0, values, -1e9)
            best = mx.max(masked_values, axis=2, keepdims=True)
            regret = mx.maximum(best - values, 0.0)
            expected_regret = mx.sum(probabilities * regret, axis=2)
            loss = loss + ev_regret_scale * mx.sum(weights * expected_regret) / denominator
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    losses: list[float] = []
    best_rank: tuple[float, float, float, float] | None = None
    best_parameters: Any = None
    best_selection: dict[str, Any] = {}
    evaluation_interval = min(100, steps)
    for step_index in range(steps):
        if balanced_teacher_batches:
            auxiliary_count = max(
                1, min(batch_size - 1, round(batch_size * auxiliary_probability))
            )
            primary_count = batch_size - auxiliary_count
            primary_selected = rng.choice(
                primary_rows,
                size=primary_count,
                replace=True,
                p=primary_sampling_probabilities,
            )
            auxiliary_selected = rng.choice(
                auxiliary_rows,
                size=auxiliary_count,
                replace=True,
                p=auxiliary_sampling_probabilities,
            )
            arguments = concatenate_training_batches(
                batch(primary, primary_selected, condition_on_node_reach=True),
                batch(auxiliary, auxiliary_selected, condition_on_node_reach=True),
            )
        else:
            use_auxiliary = rng.random() < auxiliary_probability
            selected_dataset = auxiliary if use_auxiliary else primary
            available = auxiliary_rows if use_auxiliary else primary_rows
            probabilities = (
                auxiliary_sampling_probabilities
                if use_auxiliary
                else primary_sampling_probabilities
            )
            selected = rng.choice(
                available,
                size=min(batch_size, len(available)),
                replace=True,
                p=probabilities,
            )
            arguments = batch(
                selected_dataset, selected, condition_on_node_reach=True
            )
        loss, gradients = loss_and_grad(model, *arguments)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss)
        losses.append(float(loss.item()))
        step = step_index + 1
        if step % evaluation_interval == 0 or step == steps:
            own = python_diagnostic(model, primary, primary_heldout)
            other = python_diagnostic(model, auxiliary, auxiliary_heldout)
            worst_ev = max(
                own["teacherEvMinusCandidateEvBb"],
                other["teacherEvMinusCandidateEvBb"],
            )
            worst_kl = max(
                own["weightedTeacherKl"], other["weightedTeacherKl"]
            )
            worst_agreement = min(
                own["reachWeightedPrimaryActionAgreement"],
                other["reachWeightedPrimaryActionAgreement"],
            )
            rank = (
                max(
                    worst_ev / maximum_teacher_ev_loss_bb,
                    worst_kl / maximum_weighted_kl,
                    minimum_primary_agreement / max(worst_agreement, 1e-8),
                ),
                worst_ev,
                worst_kl,
                -worst_agreement,
            )
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_parameters = tree_map(
                    lambda value: np.asarray(value), model.parameters()
                )
                best_selection = {
                    "step": step,
                    "ownHeldout": own,
                    "otherSeedHeldout": other,
                    "rank": list(rank),
                }
    if best_parameters is None:
        raise RuntimeError("range-policy training did not produce a selectable checkpoint")
    model.update(tree_map(mx.array, best_parameters))
    mx.eval(model.parameters())
    return model, losses, best_selection


def python_diagnostic(
    model: RangeConditionedPolicy,
    dataset: LoadedDataset,
    rows: np.ndarray,
) -> dict[str, float]:
    weighted_kl = 0.0
    agreement = 0.0
    ev_loss = 0.0
    total_weight = 0.0
    for row in rows:
        arguments = batch(dataset, np.asarray([row]))
        logits = model(*arguments[:6], arguments[9])
        probabilities = np.asarray(mx.softmax(logits, axis=2))[0]
        targets = dataset.targets[row]
        values = dataset.action_values[row]
        weights = dataset.combo_weights[row]
        reachable = weights > 0
        target_rows = targets[reachable]
        policy_rows = np.maximum(probabilities[reachable], 1e-12)
        local_kl = np.sum(
            np.where(
                target_rows > 0,
                target_rows
                * (np.log(np.maximum(target_rows, 1e-12)) - np.log(policy_rows)),
                0.0,
            ),
            axis=1,
        )
        local_ev_loss = np.sum(
            (target_rows - policy_rows) * values[reachable], axis=1
        )
        local_agreement = (
            np.argmax(target_rows, axis=1) == np.argmax(policy_rows, axis=1)
        )
        selected_weights = weights[reachable].astype(np.float64)
        weighted_kl += float(np.sum(selected_weights * local_kl))
        ev_loss += float(np.sum(selected_weights * local_ev_loss))
        agreement += float(np.sum(selected_weights * local_agreement))
        total_weight += float(selected_weights.sum())
    return {
        "weightedTeacherKl": weighted_kl / total_weight,
        "reachWeightedPrimaryActionAgreement": agreement / total_weight,
        "teacherEvMinusCandidateEvBb": ev_loss / total_weight,
    }


def source_policy_diagnostic(
    dataset: LoadedDataset, rows: np.ndarray
) -> dict[str, float]:
    weighted_kl = 0.0
    agreement = 0.0
    ev_loss = 0.0
    total_weight = 0.0
    for row in rows:
        targets = dataset.targets[row]
        source = np.maximum(dataset.source_probabilities[row], 1e-12)
        values = dataset.action_values[row]
        weights = dataset.combo_weights[row]
        reachable = weights > 0
        target_rows = targets[reachable]
        source_rows = source[reachable]
        local_kl = np.sum(
            np.where(
                target_rows > 0,
                target_rows
                * (
                    np.log(np.maximum(target_rows, 1e-12))
                    - np.log(source_rows)
                ),
                0.0,
            ),
            axis=1,
        )
        local_ev_loss = np.sum(
            (target_rows - source_rows) * values[reachable], axis=1
        )
        local_agreement = (
            np.argmax(target_rows, axis=1) == np.argmax(source_rows, axis=1)
        )
        selected_weights = weights[reachable].astype(np.float64)
        weighted_kl += float(np.sum(selected_weights * local_kl))
        ev_loss += float(np.sum(selected_weights * local_ev_loss))
        agreement += float(np.sum(selected_weights * local_agreement))
        total_weight += float(selected_weights.sum())
    return {
        "weightedTeacherKl": weighted_kl / total_weight,
        "reachWeightedPrimaryActionAgreement": agreement / total_weight,
        "teacherEvMinusCandidateEvBb": ev_loss / total_weight,
    }


def export_model(
    model: RangeConditionedPolicy,
    path: Path,
    seed: int,
    primary: LoadedDataset,
    auxiliary: LoadedDataset,
) -> None:
    payload = {
        "schema": NETWORK_SCHEMA,
        "architecture": model.architecture,
        "seed": seed,
        "depthBb": float(primary.metadata["depth_bb"]),
        "usesExactRanges": True,
        "featureSchema": RANGE_POLICY_FEATURE_SCHEMA,
        "contextSize": CONTEXT_SIZE,
        "querySize": QUERY_SIZE,
        "actionFeatureSchema": ACTION_FEATURE_SCHEMA,
        "actionFeatureSize": ACTION_FEATURE_COUNT,
        "rangeAggregation": "joint-reach-weighted-own-and-opponent-query-pooling",
        "sourceDatasetSha256": primary.sha256,
        "auxiliaryDatasetSha256": auxiliary.sha256,
        "sourceDatasetSchema": DATASET_SCHEMA,
        "sourceValidationStatus": "accepted_for_training",
        "policyComposition": model.composition,
        "sourcePolicySha256": primary.metadata.get("source_policy_baseline", {}).get(
            "sha256"
        ),
        "contextTower": tower_payload(model.context_tower, "gelu-fast", "gelu-fast"),
        "queryTower": tower_payload(model.query_tower, "gelu-fast", "gelu-fast"),
        "actionTower": tower_payload(model.action_tower, "gelu-fast", "gelu-fast"),
        "head": tower_payload(model.head, "gelu-fast", "linear"),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    temporary.replace(path)


def write_subset(
    dataset: LoadedDataset, rows: np.ndarray, path: Path
) -> None:
    selected = {int(row) for row in rows}
    if len(selected) != len(rows) or min(selected, default=-1) < 0 or max(
        selected, default=-1
    ) >= len(dataset.records):
        raise ValueError("invalid deterministic heldout row selection")
    metadata = dict(dataset.metadata)
    metadata["records"] = int(len(rows))
    metadata["subset_of_sha256"] = dataset.sha256
    metadata["subset"] = "deterministic_heldout"
    temporary = path.with_suffix(path.suffix + ".tmp")
    written = 0
    with gzip.open(dataset.path, "rb") as source, temporary.open("wb") as raw:
        next(source)
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            compressed.write(
                (json.dumps(metadata, separators=(",", ":")) + "\n").encode()
            )
            row = 0
            for line in source:
                if not line.strip():
                    continue
                if row in selected:
                    compressed.write(line)
                    written += 1
                row += 1
    if written != len(selected):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("heldout stream did not contain every selected row")
    temporary.replace(path)


def rust_evaluate(
    evaluator: Path,
    network: Path,
    dataset: Path,
    independent: bool,
    source_network: Path | None,
) -> dict[str, Any]:
    command = [
        str(evaluator),
        "range-policy-evaluate",
        "--network",
        str(network),
        "--dataset",
        str(dataset),
    ]
    if independent:
        command.append("--allow-independent-dataset")
    if source_network is not None:
        command.extend(("--source-network", str(source_network)))
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-a", type=Path, required=True)
    parser.add_argument("--dataset-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rust-evaluator", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path)
    parser.add_argument("--feature-workers", type=int, default=1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--final-learning-rate", type=float)
    parser.add_argument("--auxiliary-probability", type=float, default=0.25)
    parser.add_argument("--balanced-teacher-batches", action="store_true")
    parser.add_argument("--ev-regret-scale", type=float, default=0.02)
    parser.add_argument(
        "--architecture", choices=("compact", "wide"), default="compact"
    )
    parser.add_argument(
        "--composition",
        choices=("replace", "source_bundle_logit_residual"),
        default="replace",
    )
    parser.add_argument("--source-network", type=Path)
    parser.add_argument("--source-network-a", type=Path)
    parser.add_argument("--source-network-b", type=Path)
    parser.add_argument(
        "--dataset-a-with-source-b",
        type=Path,
        help="teacher A targets augmented with source network B",
    )
    parser.add_argument(
        "--dataset-b-with-source-a",
        type=Path,
        help="teacher B targets augmented with source network A",
    )
    parser.add_argument("--seeds", default="17601,17602")
    parser.add_argument("--maximum-records-per-teacher", type=int, default=256)
    parser.add_argument("--maximum-weighted-kl", type=float, default=0.10)
    parser.add_argument("--minimum-primary-agreement", type=float, default=0.70)
    parser.add_argument("--maximum-teacher-ev-loss-bb", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    shared_source = args.source_network is not None
    paired_sources = args.source_network_a is not None and args.source_network_b is not None
    if (
        len(seeds) != 2
        or min(args.steps, args.batch_size) <= 0
        or args.learning_rate <= 0
        or (
            args.final_learning_rate is not None
            and not 0 < args.final_learning_rate <= args.learning_rate
        )
        or not 0 <= args.auxiliary_probability <= 1
        or (
            args.balanced_teacher_batches
            and (args.batch_size < 2 or not 0 < args.auxiliary_probability < 1)
        )
        or args.ev_regret_scale < 0
        or args.maximum_records_per_teacher < 12
        or args.feature_workers <= 0
        or args.maximum_weighted_kl <= 0
        or not 0 < args.minimum_primary_agreement <= 1
        or args.maximum_teacher_ev_loss_bb <= 0
        or (
            args.composition == "source_bundle_logit_residual"
            and shared_source == paired_sources
        )
        or (
            paired_sources
            and (
                args.dataset_a_with_source_b is None
                or args.dataset_b_with_source_a is None
            )
        )
        or (
            not paired_sources
            and (
                args.dataset_a_with_source_b is not None
                or args.dataset_b_with_source_a is not None
                or args.source_network_a is not None
                or args.source_network_b is not None
            )
        )
    ):
        raise ValueError("invalid paired range-policy optimization controls")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    primary_paths = [
        cap_dataset(
            args.dataset_a,
            args.output_dir / "teacher-a-primary-capped.jsonl.gz",
            args.maximum_records_per_teacher,
        ),
        cap_dataset(
            args.dataset_b,
            args.output_dir / "teacher-b-primary-capped.jsonl.gz",
            args.maximum_records_per_teacher,
        ),
    ]
    cross_paths = (
        [
            cap_dataset(
                args.dataset_a_with_source_b,
                args.output_dir / "teacher-a-source-b-capped.jsonl.gz",
                args.maximum_records_per_teacher,
            ),
            cap_dataset(
                args.dataset_b_with_source_a,
                args.output_dir / "teacher-b-source-a-capped.jsonl.gz",
                args.maximum_records_per_teacher,
            ),
        ]
        if paired_sources
        else primary_paths
    )
    paths_to_inspect = primary_paths + (cross_paths if paired_sources else [])
    inspections = [inspect_dataset(path) for path in paths_to_inspect]
    metadata_a, metadata_b = inspections[0][0], inspections[1][0]
    maximum_actions = max(maximum for _, maximum in inspections)
    primary_datasets = [
        load_dataset(primary_paths[0], maximum_actions),
        load_dataset(primary_paths[1], maximum_actions),
    ]
    cross_datasets = (
        [
            load_dataset(cross_paths[0], maximum_actions),
            load_dataset(cross_paths[1], maximum_actions),
        ]
        if paired_sources
        else primary_datasets
    )
    source_networks = (
        [args.source_network_a, args.source_network_b]
        if paired_sources
        else [args.source_network, args.source_network]
    )
    if (
        primary_datasets[0].sha256 == primary_datasets[1].sha256
        or primary_datasets[0].metadata["seed"]
        == primary_datasets[1].metadata["seed"]
        or primary_datasets[0].metadata["depth_bb"]
        != primary_datasets[1].metadata["depth_bb"]
        or metadata_a["teacher"]["valueNetworkSha256"]
        == metadata_b["teacher"]["valueNetworkSha256"]
    ):
        raise ValueError("paired policy teachers must have independent identities at one depth")
    if paired_sources and (
        source_networks[0] is None
        or source_networks[1] is None
        or sha256(source_networks[0]) == sha256(source_networks[1])
        or target_corpus_sha256(primary_datasets[0])
        != target_corpus_sha256(cross_datasets[0])
        or target_corpus_sha256(primary_datasets[1])
        != target_corpus_sha256(cross_datasets[1])
    ):
        raise ValueError(
            "paired residual sources must be independent and cross-augmented from identical targets"
        )
    expected_sources = (
        [source_networks[0], source_networks[1], source_networks[1], source_networks[0]]
        if paired_sources
        else [source_networks[0], source_networks[1]]
    )
    datasets_to_prepare = (
        primary_datasets + cross_datasets if paired_sources else primary_datasets
    )
    for dataset, source_network in zip(
        datasets_to_prepare, expected_sources, strict=True
    ):
        baseline = dataset.metadata.get("source_policy_baseline", {})
        if args.composition == "source_bundle_logit_residual" and (
            baseline.get("composition") != "source_bundle_logit_residual"
            or len(str(baseline.get("sha256", ""))) != 64
            or source_network is None
            or baseline.get("sha256") != sha256(source_network)
            or np.any(
                (dataset.combo_weights > 0)
                & (dataset.source_probabilities.sum(axis=2) <= 0)
            )
        ):
            raise ValueError(
                "residual policy datasets require pinned source probabilities"
            )
        add_features(dataset, args.feature_cache_dir, args.feature_workers)
    primary_splits = [split_rows(dataset) for dataset in primary_datasets]
    cross_splits = (
        [split_rows(dataset) for dataset in cross_datasets]
        if paired_sources
        else primary_splits
    )
    students = []
    for index, seed in enumerate(seeds):
        other = 1 - index
        primary = primary_datasets[index]
        auxiliary = cross_datasets[other]
        primary_split = primary_splits[index]
        auxiliary_split = cross_splits[other]
        source_network = source_networks[index]
        own_heldout_path = args.output_dir / f"student-{index}-own-heldout.jsonl.gz"
        cross_heldout_path = args.output_dir / f"student-{index}-cross-heldout.jsonl.gz"
        write_subset(primary, primary_split[1], own_heldout_path)
        write_subset(auxiliary, auxiliary_split[1], cross_heldout_path)
        model, losses, selection = train(
            primary,
            auxiliary,
            primary_split[0],
            auxiliary_split[0],
            primary_split[1],
            auxiliary_split[1],
            seed,
            args.steps,
            args.batch_size,
            args.learning_rate,
            args.final_learning_rate,
            args.auxiliary_probability,
            args.ev_regret_scale,
            args.architecture,
            args.composition,
            args.maximum_weighted_kl,
            args.minimum_primary_agreement,
            args.maximum_teacher_ev_loss_bb,
            args.balanced_teacher_batches,
        )
        network = args.output_dir / f"range-policy-seed-{seed}.json"
        export_model(model, network, seed, primary, auxiliary)
        python_diagnostics = {
            "sourceFull": python_diagnostic(
                model, primary, np.arange(len(primary.records))
            ),
            "ownHeldout": python_diagnostic(model, primary, primary_split[1]),
            "otherSeedHeldout": python_diagnostic(
                model, auxiliary, auxiliary_split[1]
            ),
        }
        evaluations = {
            "sourceFull": rust_evaluate(
                args.rust_evaluator,
                network,
                primary.path,
                False,
                source_network,
            ),
            "ownHeldout": rust_evaluate(
                args.rust_evaluator,
                network,
                own_heldout_path,
                True,
                source_network,
            ),
            "otherSeedHeldout": rust_evaluate(
                args.rust_evaluator,
                network,
                cross_heldout_path,
                True,
                source_network,
            ),
        }
        passes = all(
            evaluation["validation"]["status"] == "accepted_for_comparison"
            and evaluation["minimumScoredComboCoverage"] >= 0.999999
            and evaluation["maximumProbabilitySumError"] <= 1e-6
            and evaluation["weightedTeacherKl"] <= args.maximum_weighted_kl
            and evaluation["reachWeightedPrimaryActionAgreement"]
            >= args.minimum_primary_agreement
            and evaluation["teacherEvMinusCandidateEvBb"]
            <= args.maximum_teacher_ev_loss_bb
            for evaluation in evaluations.values()
        )
        students.append(
            {
                "seed": seed,
                "network": str(network),
                "networkSha256": sha256(network),
                "firstLoss": losses[0],
                "finalLoss": losses[-1],
                "selectedCheckpoint": selection,
                "evaluations": evaluations,
                "mlxDiagnostics": python_diagnostics,
                "passesPilotTrustGate": passes,
            }
        )
        del model
    accepted = all(student["passesPilotTrustGate"] for student in students)
    report = {
        "schema": "hu-paired-range-conditioned-policy-distillation-v1",
        "depthBb": primary_datasets[0].metadata["depth_bb"],
        "steps": args.steps,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "finalLearningRate": args.final_learning_rate,
        "auxiliaryProbability": args.auxiliary_probability,
        "teacherBatching": (
            "exact-proportional-mixed-v1"
            if args.balanced_teacher_batches
            else "stochastic-whole-batch-v1"
        ),
        "evRegretScale": args.ev_regret_scale,
        "architecture": args.architecture,
        "composition": args.composition,
        "trainingSampling": "public-node-reach-proportional-combo-conditional-v1",
        "featureWorkers": args.feature_workers,
        "featureCaches": [dataset.feature_cache for dataset in primary_datasets],
        "sourceNetwork": str(args.source_network) if args.source_network else None,
        "sourceNetworks": [
            str(source_network) if source_network else None
            for source_network in source_networks
        ],
        "datasets": [
            {
                "path": str(dataset.path.resolve()),
                "sha256": dataset.sha256,
                "seed": dataset.metadata["seed"],
                "records": len(dataset.records),
                "trainingRecords": len(primary_splits[index][0]),
                "heldoutRecords": len(primary_splits[index][1]),
            }
            for index, dataset in enumerate(primary_datasets)
        ],
        "crossAugmentedDatasets": [
            {
                "path": str(dataset.path.resolve()),
                "sha256": dataset.sha256,
                "seed": dataset.metadata["seed"],
                "records": len(dataset.records),
                "targetCorpusSha256": target_corpus_sha256(dataset),
            }
            for dataset in cross_datasets
        ],
        "sourcePolicyDiagnostics": (
            [
                {
                    "sourceFull": source_policy_diagnostic(
                        primary_datasets[index],
                        np.arange(len(primary_datasets[index].records)),
                    ),
                    "ownHeldout": source_policy_diagnostic(
                        primary_datasets[index], primary_splits[index][1]
                    ),
                    "crossHeldout": source_policy_diagnostic(
                        cross_datasets[1 - index], cross_splits[1 - index][1]
                    ),
                }
                for index in range(2)
            ]
            if args.composition == "source_bundle_logit_residual"
            else []
        ),
        "gates": {
            "maximumWeightedKl": args.maximum_weighted_kl,
            "minimumPrimaryAgreement": args.minimum_primary_agreement,
            "maximumTeacherEvLossBb": args.maximum_teacher_ev_loss_bb,
        },
        "students": students,
        "validation": {
            "status": "accepted_for_full_game_pilot" if accepted else "rejected",
            "reasons": [
                (
                    "both independent students passed exact Rust source, heldout, and cross-seed teacher-fit gates; this is not a release activation"
                    if accepted
                    else "at least one student failed an exact Rust source, heldout, or cross-seed teacher-fit gate"
                )
            ],
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
