import copy
import unittest

from run_fixed_native_label_probe import label_drift


class FixedNativeLabelProbeTest(unittest.TestCase):
    def fixture(self):
        base = dict(schema="hu-fixed-belief-native-label-probe-v1",
                    candidateSha256="a" * 64, beliefSha256="b" * 64,
                    turn=7, leafIndex=2, publicHistory=["flop-check"],
                    nativeIterations=64,
                    ownReach=[[1.] + [0.] * 1325, [1.] + [0.] * 1325],
                    opponentCompatibleMass=[[.5] + [0.] * 1325, [.5] + [0.] * 1325],
                    counterfactualBb=[[1.] + [0.] * 1325, [1.] + [0.] * 1325],
                    conditionalResponseGainBb=[.2, .3])
        higher = copy.deepcopy(base)
        higher["nativeIterations"] = 256
        higher["counterfactualBb"][0][0] = 1.5
        return base, higher

    def test_drift_uses_conditional_bb_and_joint_reach(self):
        low, high = self.fixture()
        result = label_drift(low, high)
        self.assertAlmostEqual(result["perSeat"][0]["weightedConditionalRmseBb"], 1.)
        self.assertAlmostEqual(result["perSeat"][0]["jointMass"], .5)
        self.assertAlmostEqual(result["perSeat"][1]["weightedConditionalRmseBb"], 0.)

    def test_mismatched_belief_or_reversed_budgets_rejected(self):
        low, high = self.fixture()
        high["beliefSha256"] = "c" * 64
        with self.assertRaises(ValueError):
            label_drift(low, high)
        high["beliefSha256"] = low["beliefSha256"]
        with self.assertRaises(ValueError):
            label_drift(high, low)


if __name__ == "__main__":
    unittest.main()
