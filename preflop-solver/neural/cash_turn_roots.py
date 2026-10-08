"""Validate native public-range captures without relabeling arbitrary ranges."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from cash_profiles import CASH_NETWORK_SCHEMA, CASH_STATE_FEATURE_COUNT, profile_rules, rules_digest
from native_value_dataset import compatible_masses, legal_combos

HASH_SCHEMA = "hu-cash-public-range-root-v1"


def validate_public_money(actions: list[dict], investments: list[float]) -> None:
    def cents(value):
        if type(value) not in (int,float) or not math.isfinite(value) or value < 0 or abs(value*25-round(value*25)) > 1e-8:
            raise ValueError("public line uses invalid cent money")
        return round(value*25)
    committed, bets = [10,25], [10,25]
    actor, street, checks, aggressions = 0, "preflop", 0, 0
    for action in actions:
        if action["actor"] != actor or action["street"] != street:
            raise ValueError("public action order differs from the reached turn")
        kind = action["kind"]; paid = cents(action["amount_bb"])
        facing = max(bets)-bets[actor]
        target = action["amount_to_bb"]
        if kind in ("bet","raise"):
            if target is None or cents(target) != bets[actor]+paid or cents(target) <= max(bets) or paid == 0:
                raise ValueError("public aggressive payment differs from its wager target")
            aggressions += 1; checks = 0
            closes = False
        elif kind == "call":
            if target is not None or facing <= 0 or paid != facing:
                raise ValueError("public call does not pay the outstanding wager")
            closes = street != "preflop" or aggressions > 0
        elif kind == "check":
            if target is not None or facing != 0 or paid != 0:
                raise ValueError("public check faces an unpaid wager")
            checks += 1; closes = (street == "preflop" and actor == 1) or checks == 2
        else:
            raise ValueError("folds/all-ins cannot precede a live fresh-turn root")
        committed[actor] += paid; bets[actor] += paid
        if committed[actor] >= 500 or cents(action["pot_after_bb"]) != sum(committed):
            raise ValueError("public line settled an all-in or has an inconsistent gross pot")
        actor = 1-actor
        if closes:
            if committed[0] != committed[1]: raise ValueError("closed street has unequal commitments")
            street = "flop" if street == "preflop" else "turn"
            actor, bets, checks, aggressions = 1, [0,0], 0, 0
    if street != "turn" or [cents(v) for v in investments] != committed:
        raise ValueError("public prior line does not reach the captured turn commitments")


def root_fingerprint(policy_sha256: str, root: dict) -> str:
    source = copy.deepcopy(root["solve_input"])
    source["state"]["ranges"] = [hashlib.sha256(np.asarray(row, dtype="<f8").tobytes()).hexdigest()
                                  for row in source["state"]["ranges"]]
    payload = dict(schema=HASH_SCHEMA, policy_sha256=policy_sha256,
                   source_public_actions=root["source_public_actions"], solve_input=source)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def validate_roots(source: dict, model: dict, weight_hash: str, profile: str) -> None:
    rules = profile_rules(profile)
    if (model.get("schema") != CASH_NETWORK_SCHEMA or model.get("strategy_transform") != "softmax"
            or model.get("input_size") != CASH_STATE_FEATURE_COUNT+9 or model.get("cash_depth_bb") != 20
            or model.get("cash_rules") != rules or source.get("schema") != "hu-cash-authentic-turn-roots-v1"
            or source.get("root_hash_schema") != HASH_SCHEMA or source.get("validation_status") != "research_only"
            or source.get("rules_sha256") != rules_digest(rules) or source.get("policy_sha256") != weight_hash
            or type(source.get("sampled_deals")) is not int or not 1 <= source["sampled_deals"] <= 10_000
            or type(source.get("sampling_seed")) is not int or source["sampling_seed"] < 0
            or not isinstance(source.get("roots"), list) or not 1 <= len(source["roots"]) <= 64):
        raise ValueError("authentic roots crossed frozen source weights/rules or bounded sampling provenance")
    seen, prior_index = set(), -1
    for root in source["roots"]:
        config = root["solve_input"]; game, state = config["game"], config["state"]
        if (game.get("cash_rules") != rules or game.get("effective_stack_bb") != 20
                or game.get("small_blind_bb") != .4 or game.get("big_blind_bb") != 1
                or game.get("action_abstraction") != model["cash_action_abstraction"]
                or state.get("street") != "turn" or state.get("actor") != 1
                or state.get("street_invested_bb") != [0,0] or state.get("checks") != 0
                or state.get("aggressions") != 0 or state.get("last_full_raise_bb") != 1
                or state.get("raise_reopened") is not True or state.get("trajectory", [])
                or state.get("public_history") != ["public_belief:turn_start"]
                or config.get("averaging_delay") != 0 or config.get("river_refinement_iterations") != 0
                or config.get("regret_matching_plus") is not False or type(config.get("iterations")) is not int
                or not 2 <= config["iterations"] <= 128):
            raise ValueError("authentic root is not the pinned fresh-turn abstraction")
        board = state["board"]; investments = state["invested_bb"]
        if (len(board) != 4 or len(set(board)) != 4 or any(type(c) is not int or not 0 <= c < 52 for c in board)
                or len(investments) != 2
                or any(type(v) not in (int,float) or not math.isfinite(v) or not 0 < v < 20
                       or abs(v*25-round(v*25)) > 1e-8 for v in investments)
                or round(investments[0]*25) != round(investments[1]*25)):
            raise ValueError("authentic root has invalid cards or cent commitments")
        ranges = np.asarray(state["ranges"], dtype=float)
        legal = legal_combos(board)
        if (ranges.shape != (2,1326) or not np.isfinite(ranges).all() or (ranges < 0).any()
                or (ranges[:,~legal] != 0).any() or not np.allclose(ranges.sum(axis=1),1,atol=1e-10,rtol=0)):
            raise ValueError("authentic ranges violate public card removal or normalization")
        joint = np.sum(ranges * compatible_masses(ranges), axis=1)
        if (min(joint) <= 0 or not np.allclose(joint[0],joint[1],atol=1e-10,rtol=0)
                or type(root.get("compatible_joint_mass")) not in (int,float)
                or not math.isfinite(root["compatible_joint_mass"])
                or abs(joint[0]-root["compatible_joint_mass"]) > 1e-9):
            raise ValueError("authentic root has an invalid compatible joint belief")
        actions = root["source_public_actions"]
        if not 1 <= len(actions) <= 32 or any(set(a) != {"actor","street","kind","amount_bb","amount_to_bb","pot_after_bb"}
                or type(a["actor"]) is not int or a["actor"] not in (0,1) or a["street"] not in ("preflop","flop")
                or a["kind"] not in ("check","call","bet","raise","all_in") for a in actions):
            raise ValueError("authentic source must contain prior public actions only")
        validate_public_money(actions, investments)
        index = root["source_deal_index"]
        if (type(index) is not int or not prior_index < index < source["sampled_deals"]
                or root.get("root_sha256") != root_fingerprint(weight_hash, root) or root["root_sha256"] in seen):
            raise ValueError("authentic public line/input changed after capture or duplicates a root")
        prior_index = index; seen.add(root["root_sha256"])


def load_roots(path: Path, weights: Path, profile: str) -> tuple[dict, str]:
    raw, weight_bytes = path.read_bytes(), weights.read_bytes()
    source = json.loads(raw)
    validate_roots(source, json.loads(weight_bytes), hashlib.sha256(weight_bytes).hexdigest(), profile)
    return source, hashlib.sha256(raw).hexdigest()
