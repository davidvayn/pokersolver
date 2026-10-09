import unittest
import threading
from unittest.mock import patch

import numpy as np

from action_contrast_dataset import AffineGroup
import complete_action_bundle_chance as module


class FullChanceBundleTests(unittest.TestCase):
    def test_resource_stop_blocks_finalization_and_phase_gets_separate_bounded_timer(self):
        stop = threading.Event()
        with patch.object(module.threading, "Timer") as timer:
            old = unittest.mock.Mock()
            module.begin_finalization(stop, old)
            old.cancel.assert_called_once()
            timer.assert_called_once_with(180, stop.set)
            timer.return_value.start.assert_called_once()
        stop.set()
        with patch.object(module.threading, "Timer") as timer:
            with self.assertRaisesRegex(ValueError, "before finalization"):
                module.begin_finalization(stop, old)
            timer.assert_not_called()

    def setUp(self):
        self.prefix = dict(candidate_sha256="a" * 64, root=dict(board=[0, 5, 10]))
        self.turns = sorted(set(range(52)) - {0, 5, 10})

    def packet(self, turn, budget=64):
        return dict(schema="hu-frozen-action-turn-labels-v1", candidate_sha256="a" * 64,
                    releaseAccepted=False, turn=turn, turn_iterations=budget)

    def test_inventory_reuses_only_unique_verified_native64_cards(self):
        packets = [self.packet(t) for t in self.turns[:16]] + [self.packet(self.turns[0], 256)]
        with patch.object(module, "verified_json", side_effect=packets) as verified:
            inventory = module.inventory(self.prefix, list(range(17)))
        self.assertEqual(set(inventory), set(self.turns[:16]))
        self.assertEqual(verified.call_count, 17)

    def test_bad_cached_identity_duplicate_card_and_unbounded_budget_rejected(self):
        invalid = [self.packet(0), self.packet(52), self.packet(True), self.packet(self.turns[0], 32),
                   dict(self.packet(self.turns[0]), candidate_sha256="b" * 64),
                   dict(self.packet(self.turns[0]), releaseAccepted=True)]
        for packet in invalid:
            with patch.object(module, "verified_json", return_value=packet):
                with self.assertRaises(ValueError): module.inventory(self.prefix, [0])
        with patch.object(module, "verified_json", return_value=self.packet(self.turns[0])):
            with self.assertRaisesRegex(ValueError, "duplicate"):
                module.inventory(self.prefix, [0, 1])

    def test_diagnosis_uses_all49_as_reference_not_a_sample_selected_winner(self):
        complete = [self.packet(t) for t in self.turns]
        original = dict(root=2, trainingTurns=self.turns[:8], sensitivityTurns=self.turns[8:16])
        def build(prefix, packets, **kwargs):
            full = len(packets) in (16, 49)
            target = [[2, 1], [1, 3]] if full else [[1, 1], [2, 3]]
            group = AffineGroup(["root"], 1, ["check", "bet"], np.zeros((2, 2)),
                np.zeros((2, len(packets), 2)), np.asarray(target, dtype=float),
                np.array([.25, .75]), np.ones(2, dtype=bool))
            return [group], []
        with patch.object(module, "build_groups", side_effect=build) as builder:
            report = module.diagnose(self.prefix, complete, original, [self.packet(self.turns[0], 256)])
        group = report["groups"][0]
        self.assertAlmostEqual(group["sampleLossAgainstAll49"]["eightA"]["firstTargetLossFromSecondBestBb"], .25)
        self.assertEqual(group["sampleLossAgainstAll49"]["sixteen"]["firstTargetLossFromSecondBestBb"], 0)
        self.assertEqual(group["oneSentinelUpgradeEffectOnAll49"]["firstTargetLossFromSecondBestBb"], 0)
        self.assertFalse(report["releaseAccepted"])
        full_calls = [call for call in builder.call_args_list if call.kwargs.get("require_full_chance")]
        self.assertEqual(len(full_calls), 2)
        self.assertEqual([len(c.args[1]) for c in full_calls], [49, 49])
        self.assertEqual(sum(p["turn_iterations"] == 256 for p in full_calls[-1].args[1]), 1)
        self.assertEqual(complete, [self.packet(t) for t in self.turns])


if __name__ == "__main__": unittest.main()
