"""Frozen, chance-integrated affine decisions; never a response evaluator.

Profile contrasts and off-support BR-completed calibration targets are distinct.
Only holdings with profile-consistent support at every contributing leaf enter
the contrast auxiliary. No range is floored to make this condition true.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

import native_value_dataset as native

N = native.COMBO_COUNT


@dataclass
class AffineGroup:
    history: list[str]
    actor: int
    actions: list[str]
    terminal: np.ndarray                 # [action, holding], conditional bb
    coefficients: np.ndarray             # [action, leaf, holding]
    target: np.ndarray                   # [action, holding], profile bb
    weights: np.ndarray                  # authentic parent joint reach
    support: np.ndarray                  # coherent profile target mask

    def backup(self, leaf_ev: np.ndarray) -> np.ndarray:
        if leaf_ev.shape != self.coefficients.shape[1:]:
            raise ValueError("leaf values do not match exact affine ordering")
        if not np.isfinite(leaf_ev).all():
            raise ValueError("nonfinite leaf values")
        return self.terminal + np.einsum("alh,lh->ah", self.coefficients, leaf_ev)

    def report(self) -> dict:
        total = float(self.weights.sum())
        return dict(history=self.history, actor=self.actor, actions=self.actions,
                    authenticReach=total,
                    profileConsistentReachFraction=(min(1., float(self.weights[self.support].sum()) / total)
                                                    if total else None),
                    supportedHoldings=int(self.support.sum()),
                    totalReachedHoldings=int((self.weights > 0).sum()))


def contrast_loss_and_q_gradient(q, target, weights, delta=.05, depth=20.):
    """Exact unique-pair weighted Huber and its derivative in conditional bb."""
    q, target, weights = map(lambda v: np.asarray(v, dtype=np.float64), (q, target, weights))
    if (q.shape != target.shape or q.ndim != 2 or q.shape[0] < 2
            or weights.shape != q.shape[1:] or (weights < 0).any()
            or not all(np.isfinite(v).all() for v in (q, target, weights))
            or not math.isfinite(delta) or delta <= 0 or not math.isfinite(depth) or depth <= 0):
        raise ValueError("invalid contrast tensors/units")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("contrast has no supported training reach")
    pairs = q.shape[0] * (q.shape[0] - 1) // 2
    gradient = np.zeros_like(q)
    loss = 0.
    for a in range(q.shape[0]):
        for b in range(a + 1, q.shape[0]):
            error = ((q[a] - q[b]) - (target[a] - target[b])) / depth
            absolute = np.abs(error)
            quadratic = np.minimum(absolute, delta)
            loss += float(np.sum(weights * (.5 * quadratic**2 + delta * (absolute - quadratic)))) / total / pairs
            derivative = weights / total / pairs * np.clip(error, -delta, delta) / depth
            gradient[a] += derivative
            gradient[b] -= derivative
    return loss, gradient


def leaf_gradient(group: AffineGroup, q_gradient: np.ndarray) -> np.ndarray:
    return np.einsum("ah,alh->lh", q_gradient, group.coefficients)


def build_groups(prefix: dict, packets: list[dict], *, require_full_chance=False):
    """Use Rust-exported legal topology/terminal CFVs, not a duplicate game tree.

    All supplied turns contribute before action pairs are compared. Uniform
    public sampling has inclusion factor 49/K, with raw CFV factor 1/45.
    Opponent blocker mass is applied once at the leaf and removed once at the
    parent. Later own policies multiply coefficients, never own reach.
    """
    if prefix.get("schema") != "hu-frozen-action-prefix-v1" or prefix.get("releaseAccepted") is not False:
        raise ValueError("not a research frozen prefix")
    root = prefix["root"]
    board = root["board"]
    if len(board) != 3 or len(set(board)) != 3 or prefix["game"]["effective_stack_bb"] != 20:
        raise ValueError("contrast pilot requires exact 20bb flop")
    board_legal = native.legal_combos(board)
    legal = np.asarray(prefix["legal"], dtype=bool)
    if legal.shape != (2, N) or (legal[:, ~board_legal]).any():
        raise ValueError("invalid frozen legal support")
    key = lambda history: tuple(history)
    nodes = {key(r["history"]): r for r in prefix["nodes"]}
    terminals = {key(r["history"]): np.asarray(r["raw_cfvs"], dtype=float) for r in prefix["terminals"]}
    leaves = {key(h) for h in prefix["leaves"]}
    leaf_states = {key(s["public_history"]): s for s in prefix["leaf_states"]}
    if (len(nodes) != len(prefix["nodes"]) or len(terminals) != len(prefix["terminals"])
            or len(leaves) != len(prefix["leaves"]) or not 1 <= len(leaves) <= 16
            or set(leaf_states) != leaves or len(leaf_states) != len(prefix["leaf_states"])
            or set(nodes) & (set(terminals) | leaves) or set(terminals) & leaves):
        raise ValueError("invalid/duplicate frozen topology")
    root_key = key(root["public_history"])
    if root_key not in nodes:
        raise ValueError("missing root")
    seen = set()

    def validate(cursor, reaches):
        if cursor in seen:
            raise ValueError("prefix is not a tree")
        seen.add(cursor)
        if cursor in terminals:
            cfvs = terminals[cursor]
            if cfvs.shape != (2, N) or not np.isfinite(cfvs).all():
                raise ValueError("invalid terminal CFVs")
            return
        if cursor in leaves:
            if not np.allclose(leaf_states[cursor]["ranges"], reaches, rtol=1e-10, atol=1e-14):
                raise ValueError("leaf input differs from reconstructed frozen reaches")
            return
        if cursor not in nodes:
            raise ValueError("missing prefix branch")
        node = nodes[cursor]
        actor, actions = node["actor"], node["actions"]
        actual = np.asarray(node["reaches"], dtype=float)
        sigma = np.asarray(node["probabilities"], dtype=float).reshape(N, len(actions))
        if (actor not in (0, 1) or not actions or len(set(actions)) != len(actions)
                or len(actions) != len(node["children"]) or actual.shape != (2, N)
                or not np.allclose(actual, reaches, rtol=1e-10, atol=1e-14)
                or not np.isfinite(sigma).all() or (sigma < 0).any()
                or not np.allclose(sigma[legal[actor]].sum(1), 1, atol=1e-6)
                or (sigma[~legal[actor]] != 0).any()):
            raise ValueError("frozen policy/reach mismatch")
        for a, child in enumerate(node["children"]):
            child_key = key(child["history"])
            kind = "node" if child_key in nodes else "leaf" if child_key in leaves else "terminal" if child_key in terminals else None
            if child["kind"] != kind:
                raise ValueError("child kind mismatch")
            updated = reaches.copy()
            updated[actor] *= sigma[:, a]
            validate(child_key, updated)

    validate(root_key, np.asarray(root["ranges"], dtype=float))
    if seen != set(nodes) | set(terminals) | leaves:
        raise ValueError("unused prefix branch")
    turns = [p["turn"] for p in packets]
    if (not packets or len(packets) > 49 or len(set(turns)) != len(turns)
            or any(type(t) is not int or t < 0 or t >= 52 or t in board for t in turns)
            or require_full_chance and len(packets) != 49):
        raise ValueError("incomplete/invalid chance block")
    ordered = []
    for packet in sorted(packets, key=lambda p: p["turn"]):
        if (packet.get("schema") != "hu-frozen-action-turn-labels-v1"
                or packet["candidate_sha256"] != prefix["candidate_sha256"]
                or packet.get("releaseAccepted") is not False
                or {key(t["public_state"]["public_history"]) for t in packet["targets"]} != leaves
                or len(packet["targets"]) != len(leaves)):
            raise ValueError("incomplete or mismatched native leaf packet")
        source = dict(schema=native.SCHEMA, game=prefix["game"],
                      source_public_input_sha256=prefix["candidate_sha256"],
                      source_policy_sha256=prefix["candidate_sha256"],
                      validation=dict(status="research_only"), flop_iterations=1,
                      turn_iterations=packet["turn_iterations"], observed_queries=len(leaves),
                      maximum_states=len(leaves), targets=packet["targets"])
        native.validate_dataset(source)
        for target in sorted(packet["targets"], key=lambda t: key(t["public_state"]["public_history"])):
            if target["board"] != board + [packet["turn"]]:
                raise ValueError("leaf turn/board mismatch")
            expected = np.asarray(leaf_states[key(target["public_state"]["public_history"])]["ranges"], dtype=float).copy()
            expected[:, ~native.legal_combos(target["board"])] = 0
            if not np.allclose(target["public_state"]["ranges"], expected, rtol=1e-10, atol=1e-14):
                raise ValueError("native label does not use exact frozen-prefix ranges")
            ordered.append(target)
    indexes = {}
    for i, t in enumerate(ordered):
        indexes.setdefault(key(t["public_state"]["public_history"]), []).append(i)
    raw_ranges = np.asarray([t["public_state"]["ranges"] for t in ordered])
    masses = np.asarray([native.compatible_masses(r) for r in raw_ranges])
    # Opponent compatibility alone does not know that HERO contains the turn.
    # Such a query is impossible, not a zero-own-reach strategic deviation.
    for i, t in enumerate(ordered):
        masses[i, :, ~native.legal_combos(t["board"])] = 0
    profile_raw = np.asarray([t["raw_profile_counterfactual_bb"] for t in ordered])
    profile_ev = np.divide(profile_raw, masses, out=np.zeros_like(profile_raw), where=masses > 0)
    group_keys = [root_key]
    root_node = nodes[root_key]
    if "check" in root_node["actions"]:
        child = root_node["children"][root_node["actions"].index("check")]
        if child["kind"] == "node": group_keys.append(key(child["history"]))
    groups = []
    chance = 49. / len(packets) / 45.
    for group_key in group_keys:
        node = nodes[group_key]
        actor = node["actor"]
        parent_mass = native.compatible_masses(np.asarray(node["reaches"]))[actor]
        count = len(ordered)

        def affine(cursor):
            const = np.zeros(N)
            coeff = np.zeros((count, N))
            if cursor in terminals:
                return terminals[cursor][actor].copy(), coeff
            if cursor in leaves:
                for i in indexes[cursor]: coeff[i] = chance * masses[i, actor]
                return const, coeff
            future = nodes[cursor]
            sigma = np.asarray(future["probabilities"]).reshape(N, len(future["actions"]))
            for a, child in enumerate(future["children"]):
                sub_const, sub_coeff = affine(key(child["history"]))
                factor = sigma[:, a] if future["actor"] == actor else np.ones(N)
                const += sub_const * factor
                coeff += sub_coeff * factor
            return const, coeff

        consts, coeffs = zip(*(affine(key(child["history"])) for child in node["children"]))
        consts, coeffs = np.asarray(consts), np.asarray(coeffs)
        np.divide(consts, parent_mass[None, :], out=consts, where=parent_mass[None, :] > 0)
        np.divide(coeffs, parent_mass[None, None, :], out=coeffs, where=parent_mass[None, None, :] > 0)
        consts[:, parent_mass <= 0] = 0
        coeffs[:, :, parent_mass <= 0] = 0
        consts[:, ~legal[actor]] = 0
        coeffs[:, :, ~legal[actor]] = 0
        support = legal[actor] & (parent_mass > 0)
        # Do not equate training's zero-own-reach BR completion with profile.
        support &= ~np.any((np.abs(coeffs) > 0) & (raw_ranges[:, actor] == 0)[None, :, :], axis=(0, 1))
        weights = np.asarray(node["reaches"])[actor] * parent_mass
        target = consts + np.einsum("alh,lh->ah", coeffs, profile_ev[:, actor])
        groups.append(AffineGroup(list(group_key), actor, node["actions"], consts, coeffs, target, weights, support))
    return groups, ordered
