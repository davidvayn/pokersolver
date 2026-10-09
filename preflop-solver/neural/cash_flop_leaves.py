"""Replay revealed flop actions into fresh-turn own-payoff research inputs.

No private deal or later card is accepted. A branch is deliberately selected,
not sampled as an authentic full-hand distribution; both behavioral reaches
are conditioned on the frozen policy's exact-combo action likelihoods.
"""
from __future__ import annotations

import copy
import math
import re

import numpy as np

from cash_turn_roots import validate_public_money
from native_value_dataset import COMBOS, compatible_masses


def leaf_input(root: dict, solution: dict, turn: int, labels: list[str], iterations: int = 64) -> dict:
    if type(iterations) is not int or not 2 <= iterations <= 128:
        raise ValueError("turn reference probe permits 2..128 updates")
    state = copy.deepcopy(root["solve_input"]["state"])
    if (state["street"] != "flop" or state["actor"] != 1 or state["street_invested_bb"] != [0, 0]
            or state["checks"] != 0 or state["aggressions"] != 0 or len(state["board"]) != 3):
        raise ValueError("leaf replay requires a fresh flop root")
    if type(turn) is not int or not 0 <= turn < 52 or turn in state["board"]:
        raise ValueError("turn probe must be a distinct public card")
    if not isinstance(labels, list) or not 2 <= len(labels) <= 4:
        raise ValueError("leaf replay requires a bounded completed public branch")

    def cents(value):
        if not math.isfinite(value) or abs(value * 25 - round(value * 25)) > 1e-8:
            raise ValueError("leaf replay cannot invent subcent payments")
        return round(value * 25)

    reaches = np.asarray(state["ranges"], dtype=float).copy()
    history = state["public_history"].copy()
    actions = copy.deepcopy(root["source_public_actions"])
    committed = [cents(v) for v in state["invested_bb"]]
    bets, checks, actor, closed = [0, 0], 0, 1, False
    for label in labels:
        if closed:
            raise ValueError("public branch continued after closing the flop")
        rows = [row for row in solution["strategies"] if row["public_history"] == history]
        if len(rows) != 1 or rows[0]["actor"] != actor or label not in rows[0]["action_labels"]:
            raise ValueError("frozen flop policy lacks the requested exact branch")
        row = rows[0]
        mix = np.asarray(row["probabilities"], dtype=float).reshape(1326, len(row["action_labels"]))
        likelihood = mix[:, row["action_labels"].index(label)]
        if not np.isfinite(likelihood).all() or (likelihood < 0).any() or (likelihood > 1).any():
            raise ValueError("invalid frozen branch likelihoods")
        reaches[actor] *= likelihood
        facing = max(bets) - bets[actor]
        target = None
        if label == "check" and facing == 0:
            kind, paid = "check", 0
            checks += 1
            closed = checks == 2
        elif label == "call" and facing > 0:
            kind, paid = "call", facing
            closed = True
        else:
            match = re.fullmatch(r"(bet|raise)_to_([0-9]+\.[0-9]{3})bb", label)
            if match is None or (match[1] == "bet") != (max(bets) == 0):
                raise ValueError("folds, all-ins, and invalid actions do not reach a live turn")
            kind, target = match[1], float(match[2])
            to = cents(target)
            if to <= max(bets):
                raise ValueError("public aggression did not increase the wager")
            paid, checks = to - bets[actor], 0
        bets[actor] += paid
        committed[actor] += paid
        history.append(f"Flop:p{actor}:{label}")
        actions.append(dict(actor=actor, street="flop", kind=kind, amount_bb=paid / 25,
                            amount_to_bb=target, pot_after_bb=sum(committed) / 25))
        actor = 1 - actor
    if not closed or committed[0] != committed[1] or max(committed) >= 500:
        raise ValueError("public branch did not reach a live equal-investment turn")
    investments = [v / 25 for v in committed]
    validate_public_money(actions, investments)
    # Mask only the newly revealed card. Future runout / sampled private cards
    # cannot condition either range, even when the turn was a forced proposal.
    reaches[:, (COMBOS == turn).any(axis=1)] = 0.
    totals = reaches.sum(axis=1)
    if (totals <= 0).any():
        raise ValueError("frozen flop leaf has no public reach; no fallback")
    reaches /= totals[:, None]
    if np.sum(reaches * compatible_masses(reaches), axis=1).min() <= 0:
        raise ValueError("frozen flop leaf has no compatible private pair")
    state.update(street="turn", board=[*state["board"], turn], actor=1, invested_bb=investments,
                 public_history=["public_belief:turn_start"], ranges=reaches.tolist())
    return dict(game=copy.deepcopy(root["solve_input"]["game"]), state=state, iterations=iterations,
                averaging_delay=0, river_refinement_iterations=0, regret_matching_plus=False)


def selected_branches(solution: dict) -> list[list[str]]:
    """Small forced coverage grid, not a frequency-weighted policy sample."""
    bets = [label for label in solution["root"]["action_labels"] if label.startswith("bet_to_")]
    branches = [["check", "check"]]
    if bets:
        branches.append([bets[0], "call"])
    return branches
