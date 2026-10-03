"""The native64 control must use exactly the native256 sampled beliefs."""
import copy
import math
import unittest

from run_matched_native_label_control import label_drift


class MatchedNativeLabelTests(unittest.TestCase):
    def setUp(self):
        self.high = dict(turn_iterations=256, game={"depth": 20},
                         source_public_input_sha256="a", source_policy_sha256="b",
                         proposal_model_sha256="c", seed=100101,
                         sampling_seed=77, flop_iterations=32,
                         observed_queries=10, native_label_queries=1,
                         targets=[dict(actor=0, board=[2, 12, 15, 19],
                                       completed_zero_own_reach=[0, 0],
                                       invested_bb=[1, 1], iteration=9,
                                       opponent_compatible_mass=[[1, 1], [1, 1]],
                                       public_state={"history": ["check"]},
                                       ranges=[[.5, .5], [.5, .5]],
                                       raw_reach_totals=[1, 1],
                                       state_distribution="learned_flop_search_early_belief_native_label",
                                       value_semantics="v1",
                                       counterfactual_values_bb=[[2, 4], [1, 3]])])
        self.low = copy.deepcopy(self.high)
        self.low["turn_iterations"] = 64
        self.low["targets"][0]["counterfactual_values_bb"] = [[1, 2], [1, 1]]

    def test_weights_conditional_value_drift_by_authentic_reach(self):
        result = label_drift(self.high, self.low)
        self.assertEqual(result["beliefs"], 1)
        self.assertAlmostEqual(result["reachWeightedRmseBbBySeat"][0], math.sqrt(2.5))
        self.assertAlmostEqual(result["reachWeightedRmseBbBySeat"][1], math.sqrt(2))

    def test_rejects_different_sampled_belief_or_proposer(self):
        wrong = copy.deepcopy(self.low)
        wrong["targets"][0]["public_state"] = {"history": ["bet"]}
        with self.assertRaisesRegex(ValueError, "different public beliefs"):
            label_drift(self.high, wrong)
        wrong = copy.deepcopy(self.low)
        wrong["source_policy_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "proposer"):
            label_drift(self.high, wrong)


if __name__ == "__main__":
    unittest.main()
