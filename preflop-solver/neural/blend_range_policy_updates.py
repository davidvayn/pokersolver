#!/usr/bin/env python3
"""Compose pinned range-policy updates by blending or rebasing parameter deltas."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


NETWORK_SCHEMA = "hu-public-belief-combo-policy-network-v1"
TOWER_KEYS = ("contextTower", "queryTower", "actionTower", "head")
PARAMETER_KEYS = (
    "weights",
    "biases",
    "normalizationWeights",
    "normalizationBiases",
)
PROVENANCE_KEYS = (
    "causalAttributionSha256s",
    "selfPlayRegretDatasetSha256s",
    "directionalDatasetSha256s",
    "hybridUpdateComponentSha256s",
    "hybridUpdateWeights",
    "hybridUpdateMethod",
    "hybridUpdateSourceSha256",
    "rebasedUpdateSourceSha256",
    "rebasedUpdateDonorSourceSha256",
    "rebasedUpdateDonorCandidateSha256",
    "rebasedUpdateWeight",
    "rebasedUpdateMethod",
)
POLICY_STATIC_KEYS = (
    "schema",
    "architecture",
    "depthBb",
    "usesExactRanges",
    "featureSchema",
    "contextSize",
    "querySize",
    "actionFeatureSchema",
    "actionFeatureSize",
    "rangeAggregation",
    "policyComposition",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_policy(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != NETWORK_SCHEMA:
        raise ValueError(f"{path} is not a range-conditioned policy network")
    return payload


def _validate_static_layer(
    source: dict[str, Any],
    first: dict[str, Any],
    second: dict[str, Any],
) -> None:
    keys = set(source) | set(first) | set(second)
    for key in keys - set(PARAMETER_KEYS):
        if source.get(key) != first.get(key) or source.get(key) != second.get(key):
            raise ValueError(f"range-policy layer metadata differs at {key}")


def _blend_parameter(
    source: Any,
    first: Any,
    second: Any,
    first_weight: float,
    second_weight: float,
    location: str,
) -> list[float]:
    if not all(isinstance(values, list) for values in (source, first, second)):
        raise ValueError(f"range-policy parameter is missing at {location}")
    if len(source) != len(first) or len(source) != len(second):
        raise ValueError(f"range-policy parameter shape differs at {location}")
    result = []
    for index, (base, left, right) in enumerate(
        zip(source, first, second, strict=True)
    ):
        if not all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in (base, left, right)
        ):
            raise ValueError(
                f"range-policy parameter is invalid at {location}[{index}]"
            )
        blended = float(base) + first_weight * (float(left) - float(base))
        blended += second_weight * (float(right) - float(base))
        if not math.isfinite(blended):
            raise ValueError(f"range-policy blend is non-finite at {location}[{index}]")
        result.append(blended)
    return result


def _rebase_parameter(
    target: Any,
    donor_source: Any,
    donor_candidate: Any,
    weight: float,
    location: str,
) -> list[float]:
    if not all(
        isinstance(values, list) for values in (target, donor_source, donor_candidate)
    ):
        raise ValueError(f"range-policy parameter is missing at {location}")
    if len(target) != len(donor_source) or len(target) != len(donor_candidate):
        raise ValueError(f"range-policy parameter shape differs at {location}")
    result = []
    for index, (base, before, after) in enumerate(
        zip(target, donor_source, donor_candidate, strict=True)
    ):
        if not all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in (base, before, after)
        ):
            raise ValueError(
                f"range-policy parameter is invalid at {location}[{index}]"
            )
        rebased = float(base) + weight * (float(after) - float(before))
        if not math.isfinite(rebased):
            raise ValueError(
                f"range-policy rebase is non-finite at {location}[{index}]"
            )
        result.append(rebased)
    return result


def _validate_policy_metadata(*policies: dict[str, Any]) -> None:
    for key in POLICY_STATIC_KEYS:
        values = [policy.get(key) for policy in policies]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"range-policy metadata differs at {key}")


def _validate_hashes(*hashes: str) -> None:
    if any(len(value) != 64 for value in hashes):
        raise ValueError("range-policy provenance hashes are invalid")


def blend_policy(
    source: dict[str, Any],
    first: dict[str, Any],
    second: dict[str, Any],
    source_sha256: str,
    first_sha256: str,
    second_sha256: str,
    first_weight: float,
    second_weight: float,
    seed: int,
) -> dict[str, Any]:
    if (
        not math.isfinite(first_weight)
        or not math.isfinite(second_weight)
        or first_weight < 0.0
        or second_weight < 0.0
        or abs(first_weight + second_weight - 1.0) > 1e-12
    ):
        raise ValueError(
            "hybrid update weights must be finite, nonnegative, and sum to one"
        )
    _validate_hashes(source_sha256, first_sha256, second_sha256)
    for component in (first, second):
        if component.get("parentRangePolicySha256") != source_sha256:
            raise ValueError("hybrid update component does not pin the frozen parent")
    _validate_policy_metadata(source, first, second)

    result = dict(source)
    for key in PROVENANCE_KEYS:
        result.pop(key, None)
    for tower_key in TOWER_KEYS:
        towers = (source.get(tower_key), first.get(tower_key), second.get(tower_key))
        if not all(isinstance(tower, list) for tower in towers):
            raise ValueError(f"range-policy tower is missing at {tower_key}")
        if len(towers[0]) != len(towers[1]) or len(towers[0]) != len(towers[2]):
            raise ValueError(f"range-policy tower depth differs at {tower_key}")
        blended_tower = []
        for layer_index, (base, left, right) in enumerate(zip(*towers, strict=True)):
            if not all(isinstance(layer, dict) for layer in (base, left, right)):
                raise ValueError(
                    f"range-policy layer is invalid at {tower_key}[{layer_index}]"
                )
            _validate_static_layer(base, left, right)
            blended_layer = dict(base)
            for parameter_key in PARAMETER_KEYS:
                parameter_presence = tuple(
                    parameter_key in layer for layer in (base, left, right)
                )
                if not any(parameter_presence):
                    continue
                if not all(parameter_presence):
                    raise ValueError(
                        "range-policy parameter presence differs at "
                        f"{tower_key}[{layer_index}].{parameter_key}"
                    )
                blended_layer[parameter_key] = _blend_parameter(
                    base.get(parameter_key),
                    left.get(parameter_key),
                    right.get(parameter_key),
                    first_weight,
                    second_weight,
                    f"{tower_key}[{layer_index}].{parameter_key}",
                )
            blended_tower.append(blended_layer)
        result[tower_key] = blended_tower
    result.update(
        {
            "seed": seed,
            "parentRangePolicySha256": source_sha256,
            "hybridUpdateSourceSha256": source_sha256,
            "hybridUpdateComponentSha256s": [first_sha256, second_sha256],
            "hybridUpdateWeights": [first_weight, second_weight],
            "hybridUpdateMethod": "source_plus_weighted_component_deltas",
        }
    )
    return result


def rebase_policy_update(
    target_source: dict[str, Any],
    donor_source: dict[str, Any],
    donor_candidate: dict[str, Any],
    target_source_sha256: str,
    donor_source_sha256: str,
    donor_candidate_sha256: str,
    weight: float,
    seed: int,
) -> dict[str, Any]:
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError("rebased update weight must be finite and positive")
    _validate_hashes(
        target_source_sha256,
        donor_source_sha256,
        donor_candidate_sha256,
    )
    if donor_candidate.get("parentRangePolicySha256") != donor_source_sha256:
        raise ValueError("rebased update candidate does not pin its donor parent")
    _validate_policy_metadata(target_source, donor_source, donor_candidate)

    result = dict(target_source)
    for key in PROVENANCE_KEYS:
        result.pop(key, None)
    for tower_key in TOWER_KEYS:
        towers = (
            target_source.get(tower_key),
            donor_source.get(tower_key),
            donor_candidate.get(tower_key),
        )
        if not all(isinstance(tower, list) for tower in towers):
            raise ValueError(f"range-policy tower is missing at {tower_key}")
        if len(towers[0]) != len(towers[1]) or len(towers[0]) != len(towers[2]):
            raise ValueError(f"range-policy tower depth differs at {tower_key}")
        rebased_tower = []
        for layer_index, (base, before, after) in enumerate(zip(*towers, strict=True)):
            if not all(isinstance(layer, dict) for layer in (base, before, after)):
                raise ValueError(
                    f"range-policy layer is invalid at {tower_key}[{layer_index}]"
                )
            _validate_static_layer(base, before, after)
            rebased_layer = dict(base)
            for parameter_key in PARAMETER_KEYS:
                parameter_presence = tuple(
                    parameter_key in layer for layer in (base, before, after)
                )
                if not any(parameter_presence):
                    continue
                if not all(parameter_presence):
                    raise ValueError(
                        "range-policy parameter presence differs at "
                        f"{tower_key}[{layer_index}].{parameter_key}"
                    )
                rebased_layer[parameter_key] = _rebase_parameter(
                    base.get(parameter_key),
                    before.get(parameter_key),
                    after.get(parameter_key),
                    weight,
                    f"{tower_key}[{layer_index}].{parameter_key}",
                )
            rebased_tower.append(rebased_layer)
        result[tower_key] = rebased_tower
    result.update(
        {
            "seed": seed,
            "parentRangePolicySha256": target_source_sha256,
            "rebasedUpdateSourceSha256": target_source_sha256,
            "rebasedUpdateDonorSourceSha256": donor_source_sha256,
            "rebasedUpdateDonorCandidateSha256": donor_candidate_sha256,
            "rebasedUpdateWeight": weight,
            "rebasedUpdateMethod": "target_source_plus_weighted_donor_delta",
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--component-a", type=Path)
    parser.add_argument("--component-b", type=Path)
    parser.add_argument("--weight-a", type=float, default=0.5)
    parser.add_argument("--weight-b", type=float, default=0.5)
    parser.add_argument("--donor-source", type=Path)
    parser.add_argument("--donor-candidate", type=Path)
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_sha256 = sha256(args.source)
    blend_mode = args.component_a is not None or args.component_b is not None
    rebase_mode = args.donor_source is not None or args.donor_candidate is not None
    if blend_mode == rebase_mode:
        raise ValueError("select exactly one complete blend or rebase update")
    if blend_mode:
        if args.component_a is None or args.component_b is None:
            raise ValueError("blend update requires both components")
        first_sha256 = sha256(args.component_a)
        second_sha256 = sha256(args.component_b)
        payload = blend_policy(
            load_policy(args.source),
            load_policy(args.component_a),
            load_policy(args.component_b),
            source_sha256,
            first_sha256,
            second_sha256,
            args.weight_a,
            args.weight_b,
            args.seed,
        )
        report = {
            "schema": "hu-range-policy-hybrid-update-report-v1",
            "sourceSha256": source_sha256,
            "componentSha256s": [first_sha256, second_sha256],
            "weights": [args.weight_a, args.weight_b],
        }
    else:
        if args.donor_source is None or args.donor_candidate is None:
            raise ValueError("rebased update requires a donor source and candidate")
        donor_source_sha256 = sha256(args.donor_source)
        donor_candidate_sha256 = sha256(args.donor_candidate)
        payload = rebase_policy_update(
            load_policy(args.source),
            load_policy(args.donor_source),
            load_policy(args.donor_candidate),
            source_sha256,
            donor_source_sha256,
            donor_candidate_sha256,
            args.weight,
            args.seed,
        )
        report = {
            "schema": "hu-range-policy-rebased-update-report-v1",
            "sourceSha256": source_sha256,
            "donorSourceSha256": donor_source_sha256,
            "donorCandidateSha256": donor_candidate_sha256,
            "weight": args.weight,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    temporary.replace(args.output)
    report.update(
        {
            "seed": args.seed,
            "output": str(args.output),
            "outputSha256": sha256(args.output),
        }
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
