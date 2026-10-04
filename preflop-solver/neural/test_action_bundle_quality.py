import unittest
import numpy as np

from action_contrast_dataset import AffineGroup
from inspect_action_bundle_quality import compare, qualification


class BundleQualityTests(unittest.TestCase):
    @staticmethod
    def group(target):
        return AffineGroup(["root"], 0, ["check", "bet"], np.zeros((2, 2)),
            np.zeros((2, 1, 2)), np.asarray(target, dtype=float), np.array([.25, .75]), np.ones(2, dtype=bool))

    def test_bias_drift_is_not_contrast_drift(self):
        first = self.group([[2, 1], [1, 3]])
        second = self.group(first.target + np.array([100, -100]))
        self.assertEqual(compare(first, second)["actionContrastRmseBb"], 0)
        self.assertEqual(compare(first, second)["firstTargetLossFromSecondBestBb"], 0)

    def test_teacher_rank_drift_reports_actual_loss_not_just_rmse(self):
        first = self.group([[2, 1], [1, 3]])
        second = self.group([[1, 1], [2, 3]])
        result = compare(first, second)
        self.assertAlmostEqual(result["firstTargetLossFromSecondBestBb"], .25)
        self.assertAlmostEqual(result["bestActionAgreement"], .75)
        self.assertAlmostEqual(result["actionContrastRmseBb"], 1.)

    def test_prefix_and_support_cannot_be_faked(self):
        first = self.group([[2, 1], [1, 3]])
        second = self.group(first.target)
        second.history = ["different"]
        with self.assertRaises(ValueError): compare(first, second)
        second.history = first.history; second.support[:] = False
        with self.assertRaises(ValueError): compare(first, second)

    def test_data_qualification_rejects_material_chance_loss_without_a_release_claim(self):
        group = dict(sensitivityTurnCount=8,
                     primaryVsDisjointSensitivity=dict(firstTargetLossFromSecondBestBb=.001),
                     sameTurn64To256=dict(firstTargetLossFromSecondBestBb=.001))
        report = dict(bundleManifestSha256="a" * 64, primaryTrainingManifestSha256="b" * 64,
                      families=[dict(groups=[dict(group) for _ in range(6)])])
        self.assertEqual(qualification(report)["status"], "ready_for_fit_preflight")
        report["families"][0]["groups"][0] = dict(group,
            primaryVsDisjointSensitivity=dict(firstTargetLossFromSecondBestBb=.314))
        result = qualification(report)
        self.assertEqual(result["status"], "inconclusive")
        self.assertFalse(result["modelPromotionAllowed"])
        self.assertFalse(result["releaseAccepted"])


if __name__ == "__main__": unittest.main()
