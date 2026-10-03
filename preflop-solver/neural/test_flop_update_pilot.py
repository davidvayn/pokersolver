"""Fast checks for the paired flop-update pilot's scoring contract."""
import unittest

from run_flop_update_pilot import paired_summary


class FlopUpdatePilotTests(unittest.TestCase):
    def setUp(self):
        self.cases = [
            ({"id": "limped-paired", "startingPotBb": 2}, {"seed": 100101}, {"halfSummedGainBb": .5}),
            ({"id": "limped-paired", "startingPotBb": 2}, {"seed": 100102}, {"halfSummedGainBb": .4}),
            ({"id": "single-raised-high-rainbow", "startingPotBb": 5}, {"seed": 100101}, {"halfSummedGainBb": .9}),
            ({"id": "single-raised-high-rainbow", "startingPotBb": 5}, {"seed": 100102}, {"halfSummedGainBb": .8}),
        ]
        self.rows = [
            {"spot": "limped-paired", "seed": 100101, "gainBb": .3},
            {"spot": "limped-paired", "seed": 100102, "gainBb": .3},
            {"spot": "single-raised-high-rainbow", "seed": 100101, "gainBb": .8},
            {"spot": "single-raised-high-rainbow", "seed": 100102, "gainBb": .9},
        ]

    def test_scores_equal_cases_in_bb_and_starting_pot_units(self):
        result = paired_summary(self.cases, self.rows)
        self.assertAlmostEqual(result["equalCaseMeanImprovementBb"], .075)
        self.assertAlmostEqual(result["equalCaseMeanImprovementPercentPot"], 3.75)
        self.assertTrue(result["bothSeedsImproveByRoot"]["limped-paired"])
        self.assertFalse(result["bothSeedsImproveByRoot"]["single-raised-high-rainbow"])

    def test_refuses_partial_duplicate_or_nonfinite_response(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            paired_summary(self.cases, self.rows[:3])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            paired_summary(self.cases, self.rows[:3] + self.rows[:1])
        bad = self.rows[:-1] + [{**self.rows[-1], "gainBb": float("nan")}]
        with self.assertRaisesRegex(ValueError, "invalid"):
            paired_summary(self.cases, bad)


if __name__ == "__main__":
    unittest.main()
