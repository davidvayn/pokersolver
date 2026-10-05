import copy
import unittest

from join_full_chance_bundles import data_decision


class FullChanceDecisionTests(unittest.TestCase):
    def fixture(self):
        group = dict(sentinel64To256=dict(firstTargetLossFromSecondBestBb=.001),
                     oneSentinelUpgradeEffectOnAll49=dict(firstTargetLossFromSecondBestBb=.0001),
                     support=dict(profileConsistentReachFraction=1.))
        return [dict(schema="full-chance-action-bundle-quality-v1", root=root, all49Turns=True,
                     nativeBudget=64, releaseAccepted=False, groups=[copy.deepcopy(group), copy.deepcopy(group)])
                for root in (2, 3, 4)]

    def test_exact_chance_allows_only_training_not_promotion(self):
        result = data_decision(self.fixture(), "a" * 64)
        self.assertEqual(result["status"], "ready_for_fit_preflight")
        self.assertEqual(result["primaryTrainingManifestSha256"], "a" * 64)
        self.assertFalse(result["modelPromotionAllowed"])
        self.assertFalse(result["releaseAccepted"])
        self.assertIn("native64", result["chanceIntegration"])

    def test_teacher_uncertainty_can_still_stop_full_chance(self):
        qualities = self.fixture()
        qualities[2]["groups"][1]["sentinel64To256"]["firstTargetLossFromSecondBestBb"] = .02
        self.assertEqual(data_decision(qualities, "a" * 64)["status"], "inconclusive")

    def test_new_training_ids_require_explicit_declaration(self):
        qualities=self.fixture()
        for q, root in zip(qualities,[100,101,102]): q["root"]=root
        with self.assertRaises(ValueError): data_decision(qualities,"a"*64)
        self.assertEqual(data_decision(qualities,"a"*64,expected_roots=[100,101,102])["status"],"ready_for_fit_preflight")
        with self.assertRaises(ValueError): data_decision(qualities,"a"*64,expected_roots=[100,101,103])

    def test_missing_duplicate_unbounded_or_nonfinite_targets_rejected(self):
        with self.assertRaises(ValueError): data_decision(self.fixture()[:2], "a" * 64)
        for mutate in (lambda q: q.update(root=2), lambda q: q.update(all49Turns=False),
                       lambda q: q.update(nativeBudget=32), lambda q: q.update(releaseAccepted=True),
                       lambda q: q["groups"][0]["support"].update(profileConsistentReachFraction=0),
                       lambda q: q["groups"][0]["sentinel64To256"].update(firstTargetLossFromSecondBestBb=float("nan"))):
            qualities = self.fixture(); mutate(qualities[-1])
            with self.assertRaises(ValueError): data_decision(qualities, "a" * 64)


if __name__ == "__main__": unittest.main()
