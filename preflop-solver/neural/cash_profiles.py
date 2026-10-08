"""Pinned cash economics for training, not permission to publish a policy."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path

REGISTRY = Path(__file__).resolve().parents[2] / "data/practice/cash-game-rules.json"
PAYOFF_CONTRACT = "own-net-bb-after-refunds-and-house-rake-v1"
CASH_DATASET_SCHEMA = "hu-neural-own-payoff-traversal-jsonl-v3"
CASH_NETWORK_SCHEMA = "hu-neural-own-payoff-training-networks-v3"
CASH_STATE_FEATURE_SCHEMA = "hu-cash-card-arrival-recall-v5"
CASH_STATE_FEATURE_COUNT = 820


def equal_cash_investment_units(investments, units_per_bb: int) -> int:
    """Compare paid money as integers, not equality of serialized bb floats."""
    if (len(investments) != 2 or type(units_per_bb) is not int or units_per_bb <= 0
            or any(type(v) not in (int,float) or not math.isfinite(v) or v <= 0
                   or abs(v*units_per_bb-round(v*units_per_bb)) > 1e-8 for v in investments)):
        raise ValueError("cash investments must be equal and cent aligned")
    amounts = [round(v*units_per_bb) for v in investments]
    if amounts[0] != amounts[1]:
        raise ValueError("cash investments must be equal and cent aligned")
    return amounts[0]


def profile_rules(name: str | None) -> dict | None:
    if name is None or name == "home": return None
    if name not in ("nl25", "nl25-rake-off-control"): raise ValueError("unknown cash study profile")
    rules = copy.deepcopy(json.loads(REGISTRY.read_text())["nl25"])
    if name == "nl25-rake-off-control":
        rules["id"] = "nl25-rake-off-control-v1"
        rules["rake"]["rateBasisPoints"] = rules["rake"]["capUnits"] = 0
    return rules


def rules_digest(rules: dict) -> str:
    if rules not in [profile_rules("nl25"), profile_rules("nl25-rake-off-control")]:
        raise ValueError("cash rules differ from the frozen study profiles")
    rake = rules["rake"]
    fields = [rules["schema"], rules["id"], rules["currency"], rules["unitsPerBb"],
              *rules["blindsUnits"], rules["playersDealt"], rules["anteUnits"],
              rake["rateBasisPoints"], rake["capUnits"], int(rake["noFlopNoDrop"]),
              rake["rounding"], rules["splitPotRule"], rules["betRounding"]]
    return hashlib.sha256("|".join(map(str, fields)).encode()).hexdigest()


def validate_cash_metadata(metadata: dict, expected_profile: str | None = None) -> None:
    schema = metadata.get("schema")
    cash = metadata.get("cash_rules")
    if schema != CASH_DATASET_SCHEMA:
        if cash is not None or metadata.get("rules_sha256") is not None or expected_profile not in (None, "home"):
            raise ValueError("legacy traversal data cannot be relabelled as cash")
        return
    if cash is None or metadata.get("rules_sha256") != rules_digest(cash):
        raise ValueError("cash traversal rules identity is invalid")
    if metadata.get("payoff_contract") != PAYOFF_CONTRACT:
        raise ValueError("cash traversal targets must be own net payoffs")
    if metadata.get("cash_terminal_action_integration") is not True:
        raise ValueError("cash traversal lacks the pinned terminal expectation estimator")
    if expected_profile is not None and cash != profile_rules(expected_profile):
        raise ValueError("cash traversal profile differs from the pinned run")


def validate_training_profile(name: str | None, depth: int, baseline_scale: float) -> None:
    if profile_rules(name) is not None:
        if depth not in (20, 40, 50, 100, 200, 1000, 2000): raise ValueError("cash depth is not a frozen study target")
        if baseline_scale != 0: raise ValueError("cash pilots require baseline scale zero; actor-only baselines cannot be negated")
    elif depth not in (20,50,100):
        raise ValueError("legacy Home training supports only 20, 50 or 100bb")
