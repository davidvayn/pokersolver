import copy
import unittest
from unittest.mock import patch

import numpy as np

from action_contrast_dataset import AffineGroup, build_groups, contrast_loss_and_q_gradient, leaf_gradient, N
import native_value_dataset as native


class ActionContrastTests(unittest.TestCase):
    def test_chance_integrated_before_ranking_and_exact_terms(self):
        # A clairvoyant chooser wins either draw; a legal earlier chooser ties.
        coeff = np.array([[[.5], [0]], [[0], [.5]]])
        group = AffineGroup([], 0, ["check", "bet"], np.array([[.25], [-.25]]),
                            coeff, np.zeros((2, 1)), np.ones(1), np.ones(1, dtype=bool))
        np.testing.assert_allclose(group.backup(np.array([[1.], [2.]])), [[.75], [.75]])
        self.assertEqual(contrast_loss_and_q_gradient(group.backup(np.array([[1.], [2.]])),
                         np.zeros((2, 1)), np.ones(1))[0], 0)

    def test_all_pairs_gradient_matches_finite_differences_and_chunks(self):
        rng = np.random.default_rng(710)
        coeff = rng.uniform(size=(3, 4, 2))
        group = AffineGroup([], 1, ["a", "b", "c"], rng.normal(size=(3, 2)), coeff,
                            rng.normal(size=(3, 2)), np.array([.25, .75]), np.ones(2, dtype=bool))
        values = rng.normal(size=(4, 2))
        q = group.backup(values)
        loss, dq = contrast_loss_and_q_gradient(q, group.target, group.weights)
        dv = leaf_gradient(group, dq)
        self.assertGreater(loss, 0)
        for l in range(4):
            for h in range(2):
                plus, minus = values.copy(), values.copy()
                plus[l, h] += 1e-5
                minus[l, h] -= 1e-5
                f = lambda v: contrast_loss_and_q_gradient(group.backup(v), group.target, group.weights)[0]
                self.assertAlmostEqual(dv[l, h], (f(plus) - f(minus)) / 2e-5, places=9)
        # Two-pass decomposition gives the same affine Q and leaf derivatives.
        chunk_q = group.terminal.copy()
        for l in range(4): chunk_q += coeff[:, l] * values[l]
        np.testing.assert_allclose(chunk_q, q)
        np.testing.assert_allclose(dv, np.stack([np.sum(dq * coeff[:, l], axis=0) for l in range(4)]))

    def test_common_bias_cancels_and_bad_units_reject(self):
        target = np.array([[2., 1.], [1., 3.]])
        self.assertEqual(contrast_loss_and_q_gradient(target + 42, target, [1., 2.])[0], 0)
        for weights in ([0., 0.], [-1., 2.], [float("nan"), 2.]):
            with self.assertRaises(ValueError): contrast_loss_and_q_gradient(target, target, weights)

    def fixture(self):
        board = [0, 5, 10]
        legal = native.legal_combos(board)
        reaches = np.tile(legal / legal.sum(), (2, 1))
        sigma = np.zeros((N, 2)); sigma[legal] = [.2, .8]
        children = [dict(history=["root", "check"], kind="leaf"),
                    dict(history=["root", "bet"], kind="leaf")]
        prefix = dict(schema="hu-frozen-action-prefix-v1", releaseAccepted=False,
                      candidate_sha256="a" * 64, game=dict(effective_stack_bb=20.),
                      root=dict(board=board, public_history=["root"], ranges=reaches.tolist()),
                      legal=np.tile(legal, (2, 1)).tolist(),
                      nodes=[dict(history=["root"], actor=0, actions=["check", "bet"],
                                  reaches=reaches.tolist(), probabilities=sigma.reshape(-1).tolist(), children=children)],
                      leaves=[c["history"] for c in children], terminals=[], leaf_states=[])
        for a, child in enumerate(children):
            leaf_ranges = reaches.copy(); leaf_ranges[0] *= sigma[:, a]
            prefix["leaf_states"].append(dict(public_history=child["history"], ranges=leaf_ranges.tolist()))
        packets = []
        for turn in [15, 19]:
            targets = []
            for a, child in enumerate(children):
                raw = reaches.copy(); raw[0] *= sigma[:, a]
                raw[:, ~native.legal_combos(board + [turn])] = 0
                mass = native.compatible_masses(raw)
                # These simplified tensors are for affine accounting; full
                # target validation is separately exercised on real packets.
                value = np.zeros((2, N)); value[0] = (a + 1) * mass[0]
                targets.append(dict(board=board + [turn], public_state=dict(
                    public_history=child["history"], ranges=raw.tolist()),
                    raw_profile_counterfactual_bb=value.tolist()))
            packets.append(dict(schema="hu-frozen-action-turn-labels-v1", releaseAccepted=False,
                                candidate_sha256="a" * 64, turn=turn, turn_iterations=64, targets=targets))
        return prefix, packets

    @patch("action_contrast_dataset.native.validate_dataset")
    def test_masses_once_no_own_reach_and_blockers_with_shared_turns(self, _):
        prefix, packets = self.fixture()
        groups, ordered = build_groups(prefix, packets)
        self.assertEqual(len(groups), 1)
        group = groups[0]
        legal = native.legal_combos([0, 5, 10, 15, 19])
        values = np.ones((4, N))
        q = group.backup(values)
        # Two public cards use 49/2/45, NOT 1/49 or a own-policy .2/.8 factor.
        raw_parent = np.asarray(prefix["root"]["ranges"])
        parent_mass = native.compatible_masses(raw_parent)[0]
        numerator = sum(native.compatible_masses(np.asarray(t["public_state"]["ranges"]))[0]
                        for t in ordered if t["public_state"]["public_history"][-1] == "check")
        np.testing.assert_allclose(q[0, legal], (49/2/45 * numerator/parent_mass)[legal])
        np.testing.assert_allclose(q[0], q[1])
        self.assertTrue(group.support[legal].all())
        self.assertEqual(group.report()["profileConsistentReachFraction"], 1.)
        self.assertEqual(q[:, ~native.legal_combos([0, 5, 10])].sum(), 0)

    @patch("action_contrast_dataset.native.validate_dataset")
    def test_zero_own_completion_is_not_a_profile_contrast(self, _):
        prefix, packets = self.fixture()
        holding = int(np.flatnonzero(native.legal_combos([0, 5, 10, 15, 19]))[0])
        root_weight = prefix["root"]["ranges"][0][holding]
        prefix["nodes"][0]["probabilities"][holding * 2:holding * 2 + 2] = [1., 0.]
        prefix["leaf_states"][0]["ranges"][0][holding] = root_weight
        prefix["leaf_states"][1]["ranges"][0][holding] = 0
        for packet in packets:
            packet["targets"][0]["public_state"]["ranges"][0][holding] = root_weight
            packet["targets"][1]["public_state"]["ranges"][0][holding] = 0
        group = build_groups(prefix, packets)[0][0]
        self.assertFalse(group.support[holding])
        # The true ranges are unchanged. Off-support coverage belongs to the
        # completed-value calibration loss, not a contradictory profile target.
        self.assertEqual(packets[0]["targets"][1]["public_state"]["ranges"][0][holding], 0)
        self.assertGreater(group.weights[holding], 0)

    @patch("action_contrast_dataset.native.validate_dataset")
    def test_missing_branches_turns_and_changed_identity_fail(self, _):
        prefix, packets = self.fixture()
        for corrupt in (lambda p: p[0]["targets"].pop(),
                        lambda p: p.append(copy.deepcopy(p[0])),
                        lambda p: p[0].update(candidate_sha256="b" * 64)):
            changed = copy.deepcopy(packets); corrupt(changed)
            with self.assertRaises(ValueError): build_groups(prefix, changed)
        with self.assertRaises(ValueError): build_groups(prefix, packets, require_full_chance=True)


if __name__ == "__main__": unittest.main()
