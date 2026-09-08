import unittest

import numpy as np

from diagnose_prefilled_baseline import endpoint_lottery, fitting_shrinkage


class EndpointLotteryTest(unittest.TestCase):
    def test_arbitrary_stale_baseline_preserves_each_board_mean(self):
        values = np.arange(24, dtype=float).reshape(2, 3, 2, 2)-7
        baseline = np.full((3, 2, 2), -19.0)
        result = endpoint_lottery(values, baseline)
        np.testing.assert_allclose(result.mean(axis=1), values.sum(axis=1))

    def test_exact_baseline_eliminates_endpoint_variance(self):
        values = np.arange(12, dtype=float).reshape(1, 3, 2, 2)
        self.assertGreater(float(endpoint_lottery(values, np.zeros_like(values[0])).var(axis=1).sum()), 0)
        np.testing.assert_array_equal(endpoint_lottery(values, values[0]).var(axis=1), 0)

    def test_nonfinite_or_wrong_shape_rejected(self):
        values = np.zeros((1, 3, 2, 2))
        with self.assertRaises(ValueError):
            endpoint_lottery(values, np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            endpoint_lottery(values, np.full((3, 2, 2), np.nan))

    def test_shrinkage_uses_only_cross_fitting_agreement(self):
        x = np.arange(12, dtype=float).reshape(3, 2, 2)
        weights = np.ones((2, 2))/4
        self.assertEqual(fitting_shrinkage(np.stack([x, x]), weights), 1)
        self.assertEqual(fitting_shrinkage(np.stack([x, -x]), weights), 0)
        self.assertEqual(fitting_shrinkage(np.zeros((2, 3, 2, 2)), weights), 0)
        with self.assertRaises(ValueError):
            fitting_shrinkage(np.zeros((4, 3, 2, 2)), weights)
