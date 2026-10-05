import unittest

import numpy as np

from action_contrast_dataset import AffineGroup
from probe_training_serving_bridge import compare_bridge


class TrainingServingBridgeTests(unittest.TestCase):
    def test_reports_real_action_ranking_change_not_only_value_offset(self):
        coefficients = np.zeros((2, 1, 1326))
        coefficients[1, 0] = 1.
        target = np.zeros((2, 1326)); target[1] = 1.
        group = AffineGroup([], 0, ["check", "bet"], np.zeros_like(target), coefficients,
            target, np.ones(1326), np.ones(1326, dtype=bool))
        raw = np.full((1, 2, 1326), 1.)
        served = np.full((1, 2, 1326), -1.)
        result = compare_bridge(group, raw, served)
        self.assertEqual(result["rawNativeRankingLossBb"], 0.)
        self.assertEqual(result["servedNativeRankingLossBb"], 1.)
        self.assertEqual(result["trainingServingBestAgreement"], 0.)
        self.assertEqual(result["trainingServingContrastRmseBb"], 2.)
        same = compare_bridge(group, raw, raw)
        self.assertEqual(same["trainingServingContrastRmseBb"], 0.)


if __name__ == "__main__": unittest.main()
