#!/usr/bin/env python3
"""Freeze fresh, suit-disjoint roots for the V49 range-response gate."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import freeze_resolver_reach_release as release_freeze
import validate_resolver_reach_corpus as corpus_validator


PROTOCOL_SCHEMA = "hu-range-response-release-protocol-v1"
FREEZE_SCHEMA = "hu-range-response-release-freeze-v1"
VALUE_SCHEMA = "hu-resolver-reach-value-release-validation-v1"
RANKS = "23456789TJQKA"
SUITS = "cdhs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def resolved(repository_root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repository_root / candidate


def checked_reference(repository_root: Path, reference: dict[str, Any]) -> Path:
    path = resolved(repository_root, reference.get("path", ""))
    if not path.is_file() or release_freeze.sha256_file(path) != reference.get("sha256"):
        raise ValueError(f"range-response pinned artifact is missing or changed: {path}")
    return path


def card_label(card: int) -> str:
    return RANKS[card // 4] + SUITS[card % 4]


def board_label(board: tuple[int, int, int]) -> str:
    return ",".join(card_label(card) for card in board)


def stable_digest(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def excluded_root_keys(
    repository_root: Path, corpus_path: Path
) -> set[tuple[int, ...]]:
    corpus = json.loads(corpus_path.read_text())
    keys: set[tuple[int, ...]] = set()
    for phase in ("trainingShards", "reservedEvaluationShards"):
        for shard in corpus.get(phase, []):
            for board in shard.get("boards", []):
                keys.add(
                    corpus_validator.suit_isomorphism_key(
                        corpus_validator.parse_board(board)
                    )
                )
    for reference in corpus.get("legacyDiagnosticDatasets", []):
        keys |= corpus_validator.legacy_roots(
            checked_reference(repository_root, reference)
        )
    if not keys:
        raise ValueError("range-response root exclusion set is empty")
    return keys


def select_roots(
    seed: int,
    texture_counts: dict[str, Any],
    excluded: set[tuple[int, ...]],
) -> list[dict[str, Any]]:
    candidates: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    seen: set[tuple[int, ...]] = set()
    for board in itertools.combinations(range(52), 3):
        key = corpus_validator.suit_isomorphism_key(board)
        if key in seen or key in excluded:
            continue
        seen.add(key)
        texture = corpus_validator.flop_texture_key(board)
        candidates[texture].append(tuple(int(card) for card in key))
    selected: list[dict[str, Any]] = []
    for texture, raw_count in texture_counts.items():
        count = int(raw_count)
        if count < 1 or len(candidates.get(texture, [])) < count:
            raise ValueError(f"range-response texture has insufficient roots: {texture}")
        ranked = sorted(
            candidates[texture],
            key=lambda board: hashlib.sha256(
                f"{seed}:{texture}:{','.join(map(str, board))}".encode()
            ).hexdigest(),
        )
        for board in ranked[:count]:
            selected.append(
                {
                    "board": board_label(board),
                    "boardIndices": list(board),
                    "texture": texture,
                    "suitIsomorphismKey": list(
                        corpus_validator.suit_isomorphism_key(board)
                    ),
                }
            )
    keys = {tuple(root["suitIsomorphismKey"]) for root in selected}
    if len(keys) != len(selected) or keys & excluded:
        raise ValueError("fresh range-response roots are not suit-disjoint")
    return selected


def validate_protocol(
    protocol_path: Path,
    repository_root: Path,
    require_unopened: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], set[tuple[int, ...]]]:
    protocol = json.loads(protocol_path.read_text())
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != "frozen-before-fresh-range-response-evaluation"
        or protocol.get("activationAllowed") is not False
    ):
        raise ValueError("range-response protocol is not a fail-closed v1 freeze")

    predecessor = protocol.get("predecessor", {})
    release_path = checked_reference(repository_root, predecessor["releaseFreeze"])
    value_path = checked_reference(
        repository_root, predecessor["acceptedValueValidation"]
    )
    release = json.loads(release_path.read_text())
    value = json.loads(value_path.read_text())
    if (
        value.get("schema") != VALUE_SCHEMA
        or value.get("status") != "accepted-awaiting-strategy-and-full-game-gates"
        or value.get("activationAllowed") is not False
        or not value.get("gates")
        or not all(gate is True for gate in value["gates"].values())
        or value.get("releaseFreeze", {}).get("sha256")
        != predecessor["releaseFreeze"]["sha256"]
        or release.get("activationAllowed") is not False
    ):
        raise ValueError("range-response protocol is not bound to accepted V49 value gates")
    rejected = predecessor.get("rejectedMatchedEvaluator", {})
    if rejected.get("burnEveryPreviouslyReservedRoot") is not True:
        raise ValueError("range-response successor must burn the rejected v1 roots")

    models: list[dict[str, Any]] = []
    for reference in protocol.get("models", []):
        path = checked_reference(repository_root, reference)
        payload = json.loads(path.read_text())
        seed = int(reference["seed"])
        if (
            int(payload.get("seed", -1)) != seed
            or payload.get("usesExactRanges") is not True
            or payload.get("sourceValidationStatus") != "accepted"
            or not isinstance(payload.get("sourceDatasetSha256"), str)
            or len(payload["sourceDatasetSha256"]) != 64
            or not isinstance(payload.get("sourcePolicySha256"), str)
            or len(payload["sourcePolicySha256"]) != 64
        ):
            raise ValueError(f"range-response model is not eligible: {path}")
        models.append(
            {
                **reference,
                "sourceDatasetSha256": payload["sourceDatasetSha256"],
                "sourcePolicySha256": payload["sourcePolicySha256"],
            }
        )
    if len(models) != 2 or len({model["seed"] for model in models}) != 2:
        raise ValueError("range-response protocol requires two independent V49 models")

    implementation = protocol.get("implementation", {})
    if len(str(implementation.get("commit", ""))) != 40:
        raise ValueError("range-response implementation commit is not pinned")
    for reference in implementation.get("files", []):
        checked_reference(repository_root, reference)
    if len(implementation.get("files", [])) < 3:
        raise ValueError("range-response implementation sources are not fully pinned")

    controls = protocol.get("controls", {})
    strategy_checkpoints = [int(value) for value in controls.get("strategyCheckpoints", [])]
    response_checkpoints = [int(value) for value in controls.get("responseCheckpoints", [])]
    if (
        float(controls.get("effectiveStackBb", 0.0)) != 20.0
        or int(controls.get("strategyIterations", -1)) != 100
        or strategy_checkpoints != sorted(set(strategy_checkpoints))
        or not strategy_checkpoints
        or strategy_checkpoints[-1] != 100
        or response_checkpoints != sorted(set(response_checkpoints))
        or len(response_checkpoints) < 3
        or int(controls.get("strategyAveragingDelay", -1)) >= strategy_checkpoints[0]
        or int(controls.get("responseAveragingDelay", -1)) >= response_checkpoints[0]
        or int(controls.get("threads", 0)) < 1
        or controls.get("crossEvaluateBothDirections") is not True
    ):
        raise ValueError("range-response solver controls are invalid")
    gates = protocol.get("gates", {})
    if (
        float(gates.get("maximumRangeConsistentResponseGainBbPerHand", -1.0))
        != 0.05
        or float(gates.get("maximumFinalCheckpointIncreaseBbPerHand", -1.0))
        != 0.005
        or float(gates.get("maximumZeroSumResidualBb", -1.0)) > 1e-6
        or float(gates.get("maximumProbabilitySumError", -1.0)) > 1e-5
        or gates.get("requireEveryRootAndDirection") is not True
        or gates.get("interpretAsExploitabilityUpperBound") is not False
        or gates.get("activationRequiresIndependentFullGameUpperBound") is not True
    ):
        raise ValueError("range-response release gates were weakened")

    selection = protocol.get("rootSelection", {})
    corpus_path = checked_reference(repository_root, selection["excludedCorpus"])
    corpus_validator.validate_config(corpus_path, repository_root)
    excluded = excluded_root_keys(repository_root, corpus_path)
    roots = select_roots(
        int(selection["seed"]), selection.get("textureCounts", {}), excluded
    )
    if len(roots) < 12 or selection.get("requireSuitIsomorphicDisjointness") is not True:
        raise ValueError("range-response fresh-root coverage is insufficient")

    recheck = protocol.get("freshAuthenticRecheck", {})
    seeds = [int(seed) for seed in recheck.get("seeds", [])]
    if (
        recheck.get("requiredBeforeRoutedPolicyPromotion") is not True
        or recheck.get("useForModelSelection") is not False
        or len(seeds) != 2
        or len(set(seeds)) != 2
        or set(seeds) & {model["seed"] for model in models}
    ):
        raise ValueError("range-response successor lacks a fresh authentic recheck")

    if require_unopened:
        output_directory = resolved(repository_root, controls["outputDirectory"])
        for root in roots:
            label = root["board"].replace(",", "")
            if any(output_directory.glob(f"flop-{label}-*.json")):
                raise ValueError("fresh range-response output existed before its freeze")
    return protocol, models, excluded


def build_freeze(
    protocol_path: Path,
    repository_root: Path,
    require_unopened: bool = True,
) -> dict[str, Any]:
    protocol, models, excluded = validate_protocol(
        protocol_path, repository_root, require_unopened=require_unopened
    )
    selection = protocol["rootSelection"]
    roots = select_roots(
        int(selection["seed"]), selection["textureCounts"], excluded
    )
    excluded_encoded = [list(key) for key in sorted(excluded)]
    return {
        "schema": FREEZE_SCHEMA,
        "modelVersion": protocol["modelVersion"],
        "status": "frozen-before-fresh-range-response-evaluation",
        "activationAllowed": False,
        "protocol": {
            "path": str(protocol_path),
            "sha256": release_freeze.sha256_file(protocol_path),
        },
        "predecessor": protocol["predecessor"],
        "models": models,
        "implementation": protocol["implementation"],
        "rootSelection": {
            **selection,
            "excludedSuitIsomorphismKeyCount": len(excluded),
            "excludedSuitIsomorphismKeysSha256": stable_digest(excluded_encoded),
            "selectedRootCount": len(roots),
            "selectedRootsSha256": stable_digest(roots),
            "roots": roots,
        },
        "controls": protocol["controls"],
        "gates": protocol["gates"],
        "freshAuthenticRecheck": protocol["freshAuthenticRecheck"],
        "failurePolicy": protocol["failurePolicy"],
    }


def validate_freeze(
    freeze_path: Path, repository_root: Path
) -> dict[str, Any]:
    payload = json.loads(freeze_path.read_text())
    if payload.get("schema") != FREEZE_SCHEMA:
        raise ValueError("range-response freeze has the wrong schema")
    protocol_path = resolved(repository_root, payload.get("protocol", {}).get("path", ""))
    if (
        not protocol_path.is_file()
        or release_freeze.sha256_file(protocol_path)
        != payload.get("protocol", {}).get("sha256")
    ):
        raise ValueError("range-response freeze protocol is missing or changed")
    expected = build_freeze(protocol_path, repository_root, require_unopened=False)
    if payload != expected:
        raise ValueError("range-response freeze does not reproduce from its protocol")
    return payload


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root or args.protocol.resolve().parent.parent
    result = build_freeze(args.protocol, repository_root, require_unopened=True)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(args.output)
    print(encoded, end="")


if __name__ == "__main__":
    main()
