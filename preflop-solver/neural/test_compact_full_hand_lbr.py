import copy
import unittest
from run_compact_full_hand_lbr import summarize


def row(index, values):
    return dict(schema="compact-native-full-hand-lbr-cluster-v1", releaseAccepted=False,
        cohort="screening", index=index, diagnostics=dict(completeRootSupport=True, routeSha256="pin"),
        report=dict(lbrSeed=90001, earlyRunoutsPerCombo=16, attacks=[{}, {}],
            seatAttackerUtilityBb=values, pairedTotalResponseGainBb=sum(values)))


class FullHandClusters(unittest.TestCase):
    def test_counts_complete_deals_not_seats_and_preserves_total_scale(self):
        first, second = row(0, [1, 2]), row(1, [-1, 0])
        result = summarize([first, second])
        self.assertEqual(result["dealClusters"], 2)
        self.assertEqual(result["totalResponseGainMeanBbPerHand"], 1)
        self.assertEqual(result["dealClusterStandardErrorBb"], 2)
        self.assertIsNone(summarize([first])["dealClusterStandardErrorBb"])
        self.assertFalse(result["fullGameGateEvaluated"])
        for path, value in [("index", 0), ("cohort", "holdout")]:
            bad = copy.deepcopy(second)
            bad[path] = value
            with self.assertRaises(ValueError): summarize([first, bad])
        bad = copy.deepcopy(second)
        bad["report"]["pairedTotalResponseGainBb"] = float("nan")
        with self.assertRaises(ValueError): summarize([bad])
        bad = copy.deepcopy(second)
        bad["diagnostics"]["routeSha256"] = "different"
        with self.assertRaises(ValueError): summarize([first, bad])


if __name__ == "__main__": unittest.main()
