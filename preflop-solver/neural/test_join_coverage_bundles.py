import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from join_coverage_bundles import join
from join_full_chance_bundles import data_decision


class CoverageJoinTests(unittest.TestCase):
    def test_incomplete_unregistered_set_cannot_create_output(self):
        args=SimpleNamespace(baseline=Path("old"),baseline_sha256="a"*64,
                             coverage=Path("new"),coverage_sha256="b"*64,output=Path("output"))
        old=dict(schema="full-chance-action-bundle-set-v1",status="complete",releaseAccepted=False,
                 families=[dict(root=r) for r in (2,3,4)])
        new=dict(schema="training-coverage-action-bundle-pilot-v1",status="complete",releaseAccepted=False,
                 families=[dict(root=r) for r in (100,101,102)])
        for mutate in (lambda x:x.update(status="failed"),lambda x:x.update(releaseAccepted=True),
                       lambda x:x["families"][0].update(root=103),lambda x:x.update(schema="arbitrary")):
            bad=copy.deepcopy(new); mutate(bad)
            with patch("join_coverage_bundles.verified_json",side_effect=[old,bad]),patch.object(Path,"mkdir") as mkdir:
                with self.assertRaises(ValueError): join(args)
                mkdir.assert_not_called()

    def test_six_family_quality_cannot_omit_one_weaker_teacher(self):
        group=dict(sentinel64To256=dict(firstTargetLossFromSecondBestBb=.001),
                   oneSentinelUpgradeEffectOnAll49=dict(firstTargetLossFromSecondBestBb=.0001),
                   support=dict(profileConsistentReachFraction=1.))
        roots=[2,3,4,100,101,102]
        qualities=[dict(schema="full-chance-action-bundle-quality-v1",root=r,all49Turns=True,
                        nativeBudget=64,releaseAccepted=False,groups=[copy.deepcopy(group),copy.deepcopy(group)]) for r in roots]
        self.assertEqual(data_decision(qualities,"a"*64,expected_roots=roots)["status"],"ready_for_fit_preflight")
        qualities[-1]["groups"][1]["sentinel64To256"]["firstTargetLossFromSecondBestBb"]=.04
        self.assertEqual(data_decision(qualities,"a"*64,expected_roots=roots)["status"],"inconclusive")
        with self.assertRaises(ValueError): data_decision(qualities[:-1],"a"*64,expected_roots=roots)


if __name__=="__main__": unittest.main()
