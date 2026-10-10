"""Bounded process-local preparation reuse for serial cash pilot fits.

Only immutable corpus bytes and the feature schema determine these arrays;
network seeds, fit objectives and split assignments do not. This is not a
continuation-value cache, persistence layer or cross-process cache. The byte
budget bounds cache-owned storage, not arrays still held by an active fit.
"""
from __future__ import annotations

import re
import time
from collections.abc import Callable
from threading import Lock

import numpy as np


class CashFeatureCache:
    def __init__(self, maximum_bytes: int = 512 * 1024**2):
        if type(maximum_bytes) is not int or maximum_bytes <= 0:
            raise ValueError("cash feature cache budget must be a positive integer")
        self._maximum_bytes = maximum_bytes
        self._entry = None
        self._lock = Lock()

    def get(self, dataset_sha256: str, feature_schema: str,
            prepare: Callable[[], tuple[np.ndarray, ...]]):
        """Return exact read-only arrays and JSON-safe preparation metadata.

        The producer must return eight newly owned FP32 NumPy arrays. A miss
        releases the previous entry before preparation to avoid retaining two
        corpora. Oversize results are returned but not cached; failures are
        never cached. Read-only flags protect against accidental fit mutation.
        """
        if type(dataset_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", dataset_sha256) is None:
            raise ValueError("cash feature cache requires a lowercase dataset SHA-256")
        if type(feature_schema) is not str or not feature_schema:
            raise ValueError("cash feature cache requires a nonempty feature schema")
        key = (dataset_sha256, feature_schema)
        with self._lock:
            hit = self._entry is not None and self._entry[0] == key
            if hit:
                _, arrays, size = self._entry
                elapsed = 0.
            else:
                self._entry = None
                started = time.perf_counter()
                arrays = prepare()
                if (type(arrays) is not tuple or len(arrays) != 8
                        or any(not isinstance(a, np.ndarray) or a.dtype != np.float32
                               or not a.flags.owndata for a in arrays)):
                    raise ValueError("cash preparation must return eight owned FP32 arrays")
                size = sum(a.nbytes for a in arrays)
                for array in arrays:
                    array.setflags(write=False)
                elapsed = time.perf_counter() - started
                if size <= self._maximum_bytes:
                    self._entry = (key, arrays, size)
            return arrays, dict(cache_hit=hit, retained=self._entry is not None,
                array_bytes=size, maximum_retained_bytes=self._maximum_bytes,
                preparation_seconds=elapsed, dataset_sha256=dataset_sha256,
                feature_schema=feature_schema)
