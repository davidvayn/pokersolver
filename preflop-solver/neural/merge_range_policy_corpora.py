#!/usr/bin/env python3
"""Merge disjoint, identically configured range-policy teacher windows."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any


DATASET_SCHEMA = "hu-range-conditioned-postflop-policy-dataset-v1"
VARIABLE_TEACHER_FIELDS = {
    "rootOffset",
    "roots",
    "turnLeaves",
    "flopConvergence",
    "flopRangeResponse",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def read_metadata(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt") as source:
        line = source.readline()
    if not line:
        raise ValueError(f"empty range-policy corpus: {path}")
    metadata = json.loads(line)
    teacher = metadata.get("teacher", {})
    if (
        metadata.get("record_type") != "metadata"
        or metadata.get("schema") != DATASET_SCHEMA
        or int(metadata.get("records", 0)) <= 0
        or teacher.get("validation", {}).get("status") != "accepted_for_training"
        or int(teacher.get("roots", 0)) <= 0
        or int(teacher.get("rootOffset", -1)) < 0
    ):
        raise ValueError(f"unvalidated range-policy corpus: {path}")
    return metadata


def fixed_identity(metadata: dict[str, Any]) -> bytes:
    stable = copy.deepcopy(metadata)
    stable.pop("records", None)
    teacher = stable["teacher"]
    for field in VARIABLE_TEACHER_FIELDS:
        teacher.pop(field, None)
    return canonical(stable)


def record_identity(record: dict[str, Any]) -> bytes:
    return hashlib.sha256(
        canonical(
            {
                "state": record.get("state"),
                "action_labels": record.get("action_labels"),
            }
        )
    ).digest()


def validate_records(path: Path, expected: int, identities: set[bytes]) -> int:
    measured = 0
    with gzip.open(path, "rt") as source:
        next(source)
        for line in source:
            record = json.loads(line)
            if record.get("record_type") != "range_conditioned_average_strategy":
                raise ValueError(f"invalid range-policy record in {path}")
            identity = record_identity(record)
            if identity in identities:
                raise ValueError(
                    "range-policy corpus windows contain a duplicate public node"
                )
            identities.add(identity)
            measured += 1
    if measured != expected:
        raise ValueError(f"range-policy record count mismatch: {path}")
    return measured


def merged_metadata(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    if len(metadata) < 2:
        raise ValueError("range-policy merge requires at least two corpora")
    identity = fixed_identity(metadata[0])
    if any(fixed_identity(value) != identity for value in metadata[1:]):
        raise ValueError("range-policy corpora do not share one frozen configuration")
    if any(
        int(value["teacher"]["turnLeaves"]) % int(value["teacher"]["roots"])
        for value in metadata
    ):
        raise ValueError("range-policy corpus has fractional turn-leaf coverage")
    leaves_per_root = {
        int(value["teacher"]["turnLeaves"]) / int(value["teacher"]["roots"])
        for value in metadata
    }
    if len(leaves_per_root) != 1 or next(iter(leaves_per_root)) <= 0:
        raise ValueError("range-policy corpora use different turn-leaf coverage")
    windows = sorted(
        (
            int(value["teacher"]["rootOffset"]),
            int(value["teacher"]["roots"]),
            value,
        )
        for value in metadata
    )
    cursor = windows[0][0]
    for offset, roots, _ in windows:
        if offset != cursor:
            raise ValueError(
                "range-policy corpus root windows must be contiguous and disjoint"
            )
        cursor += roots
    result = copy.deepcopy(windows[0][2])
    result["records"] = sum(int(value["records"]) for value in metadata)
    teacher = result["teacher"]
    teacher["rootOffset"] = windows[0][0]
    teacher["roots"] = sum(roots for _, roots, _ in windows)
    teacher["turnLeaves"] = sum(
        int(value["teacher"]["turnLeaves"]) for value in metadata
    )
    for field in ("flopConvergence", "flopRangeResponse"):
        rows = [
            row for _, _, value in windows for row in value["teacher"].get(field, [])
        ]
        teacher[field] = sorted(
            rows, key=lambda row: (int(row["root"]), canonical(row))
        )
        roots = [int(row["root"]) for row in teacher[field]]
        if roots != list(range(teacher["rootOffset"], cursor)):
            raise ValueError(
                f"range-policy teacher {field} does not cover each root once"
            )
    return result


def merge(paths: list[Path], output: Path) -> dict[str, Any]:
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved) or output.resolve() in resolved:
        raise ValueError("range-policy merge paths must be distinct")
    metadata = [read_metadata(path) for path in paths]
    combined = merged_metadata(metadata)
    identities: set[bytes] = set()
    measured = sum(
        validate_records(path, int(value["records"]), identities)
        for path, value in zip(paths, metadata, strict=True)
    )
    if measured != combined["records"]:
        raise ValueError("merged range-policy record count is inconsistent")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(
                fileobj=raw, mode="wb", filename="", mtime=0
            ) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8") as destination:
                    destination.write(
                        json.dumps(combined, separators=(",", ":")) + "\n"
                    )
                    for path in paths:
                        with gzip.open(path, "rt") as source:
                            next(source)
                            for line in source:
                                destination.write(line)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "schema": "hu-range-conditioned-policy-corpus-merge-v1",
        "output": str(output),
        "records": measured,
        "rootOffset": combined["teacher"]["rootOffset"],
        "roots": combined["teacher"]["roots"],
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def main() -> None:
    args = parse_args()
    print(json.dumps(merge(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
