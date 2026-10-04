import copy
import math
import unittest

from decision_pilot_screen import screen, SPOTS, SEEDS


class DecisionPilotScreenTests(unittest.TestCase):
    def rows(self):
        return [dict(spot=spot, seed=seed, oldGainBb=.3, gainBb=.25)
                for spot in SPOTS for seed in SEEDS]

    def test_rejects_control_regression_even_if_mean_improves(self):
        rows = self.rows()
        rows[0]["gainBb"] = .34
        rows[2]["gainBb"] = .05
        result = screen(rows)
        self.assertGreater(result["equalCaseMeanImprovementBb"], 0)
        self.assertEqual(result["status"], "rejected")
        self.assertFalse(result["releaseAccepted"])

    def test_requires_positive_paired_means_and_useful_effect(self):
        self.assertEqual(screen(self.rows())["status"], "promising")
        rows = self.rows()
        for row in rows:
            row["gainBb"] = .29
        self.assertEqual(screen(rows)["status"], "inconclusive")

    def test_matched_control_cannot_be_worse_than_treatment(self):
        rows = self.rows()
        for row in rows:
            row["matchedGainBb"] = .2
        self.assertEqual(screen(rows)["status"], "inconclusive")

    def test_rejects_duplicates_partial_or_invalid_gains(self):
        for rows in (self.rows()[:3], [self.rows()[0]] * 4):
            with self.assertRaises(ValueError):
                screen(rows)
        rows = copy.deepcopy(self.rows())
        rows[0]["gainBb"] = math.nan
        with self.assertRaises(ValueError):
            screen(rows)


if __name__ == "__main__":
    unittest.main()
