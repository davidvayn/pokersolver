from types import SimpleNamespace
import unittest

import numpy as np

from probe_action_contrast_students import decision_metrics


class FrozenDecisionProbeTests(unittest.TestCase):
    def group(self):
        return SimpleNamespace(target=np.array([[2., 1., 99.], [1., 3., -99.]]),
            weights=np.array([1., 3., 100.]), support=np.array([True, True, False]),
            report=lambda: {"history": [], "profileConsistentReachFraction": 4 / 104})

    def test_supported_weights_and_common_value_offset(self):
        group = self.group()
        exact = decision_metrics(group, group.target + 10)
        self.assertEqual(exact["nativeLossFromPredictedBestBb"], 0.)
        self.assertEqual(exact["actionContrastRmseBb"], 0.)
        self.assertEqual(exact["bestActionAgreement"], 1.)
        wrong = decision_metrics(group, group.target[::-1])
        self.assertEqual(wrong["nativeLossFromPredictedBestBb"], 1.75)
        self.assertEqual(wrong["bestActionAgreement"], 0.)
        self.assertAlmostEqual(wrong["actionContrastRmseBb"], np.sqrt(13))

    def test_invalid_output_and_empty_support_fail_closed(self):
        group = self.group()
        with self.assertRaises(ValueError):
            decision_metrics(group, np.ones((2, 2)))
        with self.assertRaises(ValueError):
            decision_metrics(group, np.full((2, 3), np.nan))
        group.support[:] = False
        with self.assertRaises(ValueError):
            decision_metrics(group, group.target)


if __name__ == "__main__":
    unittest.main()
