import json
import unittest

import numpy as np

import native_value_dataset as native
from diagnose_native_value_errors import diagnose, error_statistics, paired_statistics


class NativeErrorDiagnosticsTest(unittest.TestCase):
    def test_completion_diagnostic_divides_by_raw_opponent_mass(self):
        board = [0, 5, 10, 15]
        legal = native.legal_combos(board)
        ranges = np.zeros((2, 1326))
        ranges[1] = legal / legal.sum()
        masses = native.compatible_masses(ranges)
        profile = np.zeros_like(ranges)
        profile[0, legal] = 1
        truth = profile * 2
        target = dict(board=board, invested_bb=[1, 1], counterfactual_values_bb=truth,
                      ranges=ranges, opponent_compatible_mass=masses,
                      raw_reach_totals=[0, 3], raw_profile_counterfactual_bb=profile * masses * np.asarray([[3], [0]]),
                      state_distribution="test", public_state={"public_history": "test"})
        result = diagnose({"targets": [target]}, profile[None, :], {"train": np.asarray([0])})["train"]
        self.assertAlmostEqual(result["zeroOwnAgainstFrozenProfile"]["weightedRmseBb"], 0)
        self.assertAlmostEqual(result["zeroOwnCounterfactual"]["weightedRmseBb"], 1)
        self.assertAlmostEqual(result["zeroOwnCompletionGap"]["weightedBiasBb"], 1)

    def test_combined_bias_does_not_hide_opposite_seat_errors(self):
        truth = np.zeros((1, 2, 1))
        report = paired_statistics(truth, np.asarray([[[2], [-2]]]), np.ones_like(truth))
        self.assertEqual(report["weightedBiasBb"], 0)
        self.assertEqual([s["weightedBiasBb"] for s in report["perSeat"]], [2, -2])

    def test_report_serializes_numpy_split_indices(self):
        values = np.zeros((2, 1326))
        target = dict(board=[0, 5, 10, 15], invested_bb=[1, 1],
                      counterfactual_values_bb=values, ranges=values,
                      opponent_compatible_mass=values, state_distribution="test",
                      raw_reach_totals=[0, 0], raw_profile_counterfactual_bb=values,
                      public_state={"public_history": "test"})
        report = diagnose({"targets": [target]}, values[None, :], {"train": np.asarray([0])})
        json.dumps(report, allow_nan=False)

    def test_uses_reach_not_row_count_and_preserves_scale(self):
        result = error_statistics([0, 0, 0], [1, -3, 100], [3, 1, 0])
        self.assertAlmostEqual(result["weightedRmseBb"], np.sqrt(3))
        self.assertEqual(result["weightedMaeBb"], 1.5)
        self.assertEqual(result["weightedBiasBb"], 0)
        self.assertEqual(result["weightAbove050Bb"], 1)
        scaled = error_statistics([0, 0, 0], [1, -3, 100], [6, 2, 0])
        self.assertEqual(scaled["weightedRmseBb"], result["weightedRmseBb"])

    def test_absent_reach_is_unknown_not_passing_zero(self):
        result = error_statistics([0], [10], [0])
        self.assertEqual(result["weightMass"], 0)
        self.assertIsNone(result["weightedRmseBb"])
        self.assertIsNone(result["weightAbove050Bb"])

    def test_rejects_broadcasting_nonfinite_and_negative_weights(self):
        for truth, predicted, weights in (([0, 0], [1], [1, 1]), ([0], [np.nan], [1]), ([0], [1], [-1])):
            with self.assertRaises(ValueError):
                error_statistics(truth, predicted, weights)


if __name__ == "__main__":
    unittest.main()
