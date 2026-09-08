import unittest

from analyze_preflop_targets import inspect_tick


class TargetGapTest(unittest.TestCase):
    def test_counterfactual_gap_can_hide_from_current_profile_accounting(self):
        d = dict(trainingRootActionValuesBb=[[1.0], [3.0]],
            profileRootActionValuesBb=[[1.0], [-2.0]], rootProbabilities=[[1.0], [0.0]],
            multiplicities=[1326], classes=["fixture"], actions=["stay", "deviate"],
            rootUpdatedThisRound=True, zeroOwnRootHoldings=[1, 0],
            maxNativeGapOnPositiveRootReachBb=[0, 0], maxNativeGapOnZeroRootReachBb=[5, 0])
        r = inspect_tick(d)
        self.assertEqual(r["maximumCurrentProfileValueChangeBb"], 0)
        self.assertEqual(r["maximumRootAdvantageGapBb"], 5)
        self.assertEqual(r["bestActionChangedComboFraction"], 1)
        self.assertEqual(r["trainingOnlyPositiveRegretComboFraction"], 1)
        self.assertEqual(r["examples"][0]["currentActionProbability"], 0)
