#!/usr/bin/env python3
"""Validate a frozen resolver-reach corpus plan and any completed shards."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA = "hu-resolver-reach-corpus-freeze-v1"
DATASET_SCHEMA = "hu-turn-public-belief-cfv-dataset-v2"
RANKS = "23456789TJQKA"
SUITS = "cdhs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--repository-root",
        type=Path,
        help="preflop-solver directory; defaults to the config's parent directory",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_board(board: str) -> tuple[int, int, int]:
    cards = board.split(",")
    if len(cards) != 3:
        raise ValueError(f"flop must contain three cards: {board}")
    parsed: list[int] = []
    for card in cards:
        if len(card) != 2 or card[0] not in RANKS or card[1] not in SUITS:
            raise ValueError(f"invalid card in flop {board}")
        parsed.append(RANKS.index(card[0]) * 4 + SUITS.index(card[1]))
    if len(set(parsed)) != 3:
        raise ValueError(f"flop contains duplicate cards: {board}")
    return tuple(parsed)


def suit_isomorphism_key(cards: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if len(cards) != len(set(cards)):
        raise ValueError("board contains duplicate cards")
    canonical = []
    for permutation in itertools.permutations(range(4)):
        transformed = sorted((card // 4) * 4 + permutation[card % 4] for card in cards)
        canonical.append(tuple(transformed))
    return min(canonical)


def flop_texture_key(cards: tuple[int, int, int]) -> str:
    rank_counts = Counter(card // 4 for card in cards)
    suit_counts = Counter(card % 4 for card in cards)
    maximum_rank_count = max(rank_counts.values())
    if maximum_rank_count == 3:
        return "trips"
    suit = (
        "Monotone"
        if max(suit_counts.values()) == 3
        else "TwoTone"
        if max(suit_counts.values()) == 2
        else "Rainbow"
    )
    if maximum_rank_count == 2:
        return f"paired{suit}"
    ranks = set(rank_counts)
    windows = [set(range(start, start + 5)) for start in range(9)]
    windows.append({12, 0, 1, 2, 3})
    connectivity = (
        "Connected" if max(len(ranks & window) for window in windows) == 3 else "Disconnected"
    )
    return f"unpaired{suit}{connectivity}"


def resolve_path(repository_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else repository_root / path


def checked_hash(repository_root: Path, entry: dict[str, Any]) -> Path:
    path = resolve_path(repository_root, str(entry["path"]))
    if not path.is_file():
        raise ValueError(f"pinned artifact is missing: {path}")
    actual = sha256_file(path)
    if actual != entry.get("sha256"):
        raise ValueError(f"pinned artifact hash mismatch: {path}")
    return path


def legacy_roots(path: Path) -> set[tuple[int, ...]]:
    payload = json.loads(path.read_text())
    roots = set()
    for target in payload.get("targets", []):
        root = target.get("resolver_root_board")
        if not isinstance(root, list) or len(root) != 3:
            raise ValueError(f"legacy resolver target lacks a three-card root: {path}")
        roots.add(suit_isomorphism_key(tuple(int(card) for card in root)))
    if not roots:
        raise ValueError(f"legacy resolver dataset has no roots: {path}")
    return roots


def validate_completed_shard(
    repository_root: Path,
    shard: dict[str, Any],
    source: dict[str, Any],
    planned_keys: set[tuple[int, ...]],
) -> dict[str, Any] | None:
    path = resolve_path(repository_root, str(shard["output"]))
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if payload.get("schema") != DATASET_SCHEMA:
        raise ValueError(f"completed shard has the wrong schema: {path}")
    if payload.get("validation", {}).get("status") != "accepted":
        raise ValueError(f"completed shard is not accepted: {path}")
    if int(payload.get("seed", -1)) != int(shard["seed"]):
        raise ValueError(f"completed shard has the wrong seed: {path}")
    if payload.get("resolver_source_value_network_sha256") != source["sha256"]:
        raise ValueError(f"completed shard has the wrong source value network: {path}")
    targets = payload.get("targets", [])
    if len(targets) != int(shard["expectedStateCount"]):
        raise ValueError(f"completed shard has the wrong state count: {path}")
    actual_keys = {
        suit_isomorphism_key(tuple(int(card) for card in target["resolver_root_board"]))
        for target in targets
    }
    if actual_keys != planned_keys:
        raise ValueError(f"completed shard roots differ from the frozen plan: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "stateCount": len(targets),
        "validationStatus": "accepted",
    }


def validate_config(config_path: Path, repository_root: Path) -> dict[str, Any]:
    payload = json.loads(config_path.read_text())
    if payload.get("schema") != SCHEMA:
        raise ValueError("resolver-reach corpus config has the wrong schema")
    if payload.get("activationAllowed") is not False:
        raise ValueError("corpus planning cannot activate a model")

    sources: dict[int, dict[str, Any]] = {}
    for source in payload.get("sourceValueNetworks", []):
        checked_hash(repository_root, source)
        seed = int(source["trainingSeed"])
        if seed in sources:
            raise ValueError("source value-network seeds must be unique")
        sources[seed] = source
    if len(sources) < 2:
        raise ValueError("resolver-reach generation requires two source value networks")

    phases: dict[str, list[dict[str, Any]]] = {
        "training": payload.get("trainingShards", []),
        "reservedEvaluation": payload.get("reservedEvaluationShards", []),
    }
    phase_keys: dict[str, set[tuple[int, ...]]] = {}
    phase_textures: dict[str, Counter[str]] = {}
    completed: dict[str, list[dict[str, Any]]] = {key: [] for key in phases}
    all_seeds: set[int] = set()
    for phase, shards in phases.items():
        keys: set[tuple[int, ...]] = set()
        textures: Counter[str] = Counter()
        for shard in shards:
            seed = int(shard["seed"])
            if seed in all_seeds:
                raise ValueError("resolver-reach shard seeds must be unique")
            all_seeds.add(seed)
            source_seed = int(shard["sourceTrainingSeed"])
            if source_seed not in sources:
                raise ValueError("resolver-reach shard references an unknown source seed")
            boards = [parse_board(board) for board in shard.get("boards", [])]
            if len(boards) < 3 or int(shard["expectedStateCount"]) != 3 * len(boards):
                raise ValueError("each resolver-reach shard must freeze three states per board")
            shard_keys = {suit_isomorphism_key(board) for board in boards}
            if len(shard_keys) != len(boards) or keys & shard_keys:
                raise ValueError(f"{phase} contains suit-isomorphic duplicate roots")
            keys |= shard_keys
            textures.update(flop_texture_key(board) for board in boards)
            evidence = validate_completed_shard(
                repository_root, shard, sources[source_seed], shard_keys
            )
            if evidence is not None:
                completed[phase].append(evidence)
        phase_keys[phase] = keys
        phase_textures[phase] = textures

    if phase_keys["training"] & phase_keys["reservedEvaluation"]:
        raise ValueError("training and reserved evaluation roots are suit-isomorphic")
    separation = payload.get("separationPolicy", {})
    if len(phase_keys["training"]) != int(separation["trainingRootCount"]):
        raise ValueError("training root count differs from the separation policy")
    if len(phase_keys["reservedEvaluation"]) != int(
        separation["reservedEvaluationRootCount"]
    ):
        raise ValueError("evaluation root count differs from the separation policy")

    expected_coverage = payload.get("coverage", {})
    for phase, textures in phase_textures.items():
        if dict(textures) != expected_coverage.get(phase):
            raise ValueError(f"{phase} texture coverage differs from the frozen declaration")

    legacy_keys: set[tuple[int, ...]] = set()
    for entry in payload.get("legacyDiagnosticDatasets", []):
        legacy_keys |= legacy_roots(checked_hash(repository_root, entry))
    if legacy_keys & (phase_keys["training"] | phase_keys["reservedEvaluation"]):
        raise ValueError("new roots overlap a legacy resolver corpus under suit isomorphism")

    return {
        "schema": "hu-resolver-reach-corpus-freeze-validation-v1",
        "status": "accepted",
        "config": str(config_path),
        "configSha256": sha256_file(config_path),
        "trainingRootCount": len(phase_keys["training"]),
        "reservedEvaluationRootCount": len(phase_keys["reservedEvaluation"]),
        "textureCoverage": {phase: dict(value) for phase, value in phase_textures.items()},
        "completedShards": completed,
        "activationAllowed": False,
    }


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root or args.config.resolve().parent.parent
    report = validate_config(args.config, repository_root)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(encoded)
        temporary.replace(args.output)
    print(encoded, end="")


if __name__ == "__main__":
    main()
