"""Provenance-pinned replay inside one native, family-split research corpus.

Sampling weights are NOT poker reaches. This module never edits target values,
projection weights, ranges, or the split. Disabled replay consumes no RNG.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np

import native_value_dataset as native

MAXIMUM_REFERENCE_BYTES = 256 * 1024**2
BANDS = ("small", "medium", "large")


def load_reference(path: Path, expected_sha256: str) -> tuple[dict, str]:
    if path.stat().st_size > MAXIMUM_REFERENCE_BYTES:
        raise ValueError("native replay reference exceeds input budget")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError("native replay reference hash mismatch")
    if payload.startswith(b"\x1f\x8b"):
        with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
            decoded = stream.read(MAXIMUM_REFERENCE_BYTES + 1)
    else:
        decoded = payload
    if len(decoded) > MAXIMUM_REFERENCE_BYTES:
        raise ValueError("decoded native replay reference exceeds input budget")
    source = json.loads(decoded)
    native.validate_dataset(source)
    return source, digest


def reference_boundary(source: dict, reference: dict) -> int:
    """Require an exact target AND capture prefix, not merely a chosen count."""
    if (source.get("schema") != native.SCHEMA
            or reference.get("schema") != native.SCHEMA
            or source.get("game") != reference.get("game")):
        raise ValueError("native replay reference schema/game changed")
    old, new = reference.get("targets", []), source.get("targets", [])
    if not old or len(old) >= len(new) or new[:len(old)] != old:
        raise ValueError("native replay reference is not an unchanged proper prefix")
    captures = reference.get("source_captures", [])
    if (not captures or source.get("source_captures", [])[:len(captures)] != captures
            or sum(row["states"] for row in captures) != len(old)):
        raise ValueError("native replay capture provenance changed")
    return len(old)


def partition_rows(groups: np.ndarray, train_states: np.ndarray,
                   tuning_states: np.ndarray, holdout_states: np.ndarray,
                   boundary: int) -> tuple[np.ndarray, np.ndarray]:
    if type(boundary) is not int or boundary <= 0 or not len(groups):
        raise ValueError("invalid native replay boundary/groups")
    splits = [set(map(int, values)) for values in
              (train_states, tuning_states, holdout_states)]
    if any(len(values) != len(selected) for values, selected in
           zip((train_states, tuning_states, holdout_states), splits, strict=True)):
        raise ValueError("duplicate native split state")
    if (any(splits[a] & splits[b] for a, b in ((0, 1), (0, 2), (1, 2)))
            or set(map(int, groups)) != set.union(*splits)):
        raise ValueError("native replay split overlaps or does not cover the corpus")
    rows = np.flatnonzero(np.isin(groups, train_states))
    retained = rows[groups[rows] < boundary]
    appended = rows[groups[rows] >= boundary]
    if not len(retained) or not len(appended):
        raise ValueError("native replay requires retained and appended training rows")
    return retained, appended


class NativeReplaySampler:
    """Preserve pooled pot quotas, rotate cohort slots, audit actual exposure."""

    def __init__(self, retained: np.ndarray, appended: np.ndarray,
                 invested: np.ndarray, batch_size: int, retained_fraction: float,
                 sampling_weights: np.ndarray | None = None):
        if (batch_size <= 0 or not np.isfinite(retained_fraction)
                or not 0 < retained_fraction <= 1):
            raise ValueError("invalid native replay batch/fraction")
        cohorts = [np.asarray(retained, dtype=np.int64),
                   np.asarray(appended, dtype=np.int64)]
        all_rows = np.concatenate(cohorts)
        if (not len(cohorts[0]) or not len(cohorts[1])
                or len(np.unique(all_rows)) != len(all_rows)
                or np.any(all_rows < 0) or np.any(all_rows >= len(invested))):
            raise ValueError("native replay rows must be nonempty, disjoint and valid")
        maximum = np.max(invested, axis=1)
        if not np.isfinite(maximum[all_rows]).all():
            raise ValueError("non-finite native replay pot")
        self.bands = np.where(maximum <= 3.5, 0, np.where(maximum <= 7.5, 1, 2))
        self.buckets = [[rows[self.bands[rows] == band] for band in range(3)]
                        for rows in cohorts]
        self.pooled = [all_rows[self.bands[all_rows] == band] for band in range(3)]
        available = [band for band in range(3) if len(self.pooled[band])]
        self.schedule = [available[offset % len(available)]
                         for offset in range(batch_size)]
        self.batch_size = batch_size
        self.retained_slots = int(np.ceil(batch_size * retained_fraction))
        self.appended_slots = batch_size - self.retained_slots
        self.retained_set = set(map(int, retained))
        self.weights = sampling_weights
        if sampling_weights is not None and (
                len(sampling_weights) != len(invested)
                or not np.isfinite(sampling_weights[all_rows]).all()
                or np.any(sampling_weights[all_rows] <= 0)):
            raise ValueError("invalid native replay sampling weights")
        self.draws = np.zeros((2, 3), dtype=np.int64)
        self.fallbacks = np.zeros((2, 3), dtype=np.int64)
        self.batches = 0

    def sample(self, rng: np.random.Generator, step: int) -> np.ndarray:
        if type(step) is not int or step < 0:
            raise ValueError("native replay step must be a nonnegative integer")
        appended_positions = {(step * self.appended_slots + offset) % self.batch_size
                              for offset in range(self.appended_slots)}
        selected = []
        for offset, band in enumerate(self.schedule):
            preferred = int(offset in appended_positions)
            bucket = self.buckets[preferred][band]
            if not len(bucket):
                bucket = self.pooled[band]
                self.fallbacks[preferred, band] += 1
            probabilities = None
            if self.weights is not None:
                local = self.weights[bucket]
                probabilities = local / local.sum()
            row = int(rng.choice(bucket, p=probabilities))
            actual = int(row not in self.retained_set)
            self.draws[actual, band] += 1
            selected.append(row)
        rng.shuffle(selected)
        self.batches += 1
        return np.asarray(selected, dtype=np.int64)

    def report(self) -> dict:
        def counts(values):
            return {name: {band: int(values[index, b]) for b, band in enumerate(BANDS)}
                    for index, name in enumerate(("retained", "appended"))}
        return dict(kind="native_provenance_rotating_slots_v1", batches=self.batches,
                    retainedSlots=self.retained_slots, appendedSlots=self.appended_slots,
                    potSlotBands=[BANDS[band] for band in self.schedule],
                    sampledRowsByOriginPot=counts(self.draws),
                    emptyBandFallbacksByOriginPot=counts(self.fallbacks))
