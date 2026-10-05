"""Student policy pilot rejects incomplete pairs and scores starting-pot units."""
import math
from pathlib import Path
import tempfile
import unittest
import json
from types import SimpleNamespace

from run_native_value_preflight import sha256
from run_student_value_pilot import select_students, summarize, run
from decision_pilot_screen import completed_control_regression


class StudentValuePilotTests(unittest.TestCase):
    def test_early_stop_cannot_be_applied_to_an_undeclared_broad_pilot(self):
        args=SimpleNamespace(packet_workers=4,maximum_seconds=900,stop_on_known_regression=True,spots="limped-paired")
        with self.assertRaisesRegex(ValueError,"control-first"): run(args)

    def test_one_bad_control_rejects_but_one_good_control_cannot_accept(self):
        row=dict(spot="three-bet-high-rainbow",seed=100101,gainBb=.3507304900105628,oldGainBb=.2783353664552195)
        result=completed_control_regression(row)
        self.assertEqual(result["status"],"rejected")
        self.assertAlmostEqual(result["regressionBb"],.07239512355534328)
        self.assertFalse(result["releaseAccepted"])
        row["gainBb"]=.25
        self.assertIsNone(completed_control_regression(row))
        for changed in (dict(seed=1),dict(spot="unknown"),dict(gainBb=float("nan")),dict(oldGainBb=-1.)):
            with self.assertRaises(ValueError): completed_control_regression({**row,**changed})

    def test_requires_two_parity_checked_independent_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = []
            for seed in (10601, 10602):
                path = root / f"model-{seed}.json"
                path.write_text("{}")
                models.append(dict(seed=seed, model=str(path), modelSha256=sha256(path),
                                   maximumParityErrorBb=1e-5))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(dict(status="complete", predictions=models)))
            _, selected = select_students(manifest, sha256(manifest))
            self.assertEqual(set(selected), {100101, 100102})
            self.assertEqual(selected[100101]["seed"], 100101)
            self.assertEqual(selected[100102]["seed"], 100102)
            models[1]["maximumParityErrorBb"] = 1e-3
            manifest.write_text(json.dumps(dict(status="complete", predictions=models)))
            with self.assertRaisesRegex(ValueError, "parity"):
                select_students(manifest, sha256(manifest))

    def test_scoring_is_paired_and_pot_normalized(self):
        cases = [({"id": "limped-paired", "startingPotBb": 2}, {"seed": 100101}, {"halfSummedGainBb": .5}),
                 ({"id": "limped-paired", "startingPotBb": 2}, {"seed": 100102}, {"halfSummedGainBb": .4})]
        rows = [{"spot": "limped-paired", "seed": 100101, "gainBb": .3},
                {"spot": "limped-paired", "seed": 100102, "gainBb": .35}]
        result = summarize(cases, rows)
        self.assertAlmostEqual(result["equalCaseMeanImprovementBb"], .125)
        self.assertAlmostEqual(result["equalCaseMeanImprovementPercentPot"], 6.25)
        self.assertTrue(result["bothSeedsImproveByRoot"]["limped-paired"])
        rows[1]["gainBb"] = math.nan
        with self.assertRaisesRegex(ValueError, "invalid"):
            summarize(cases, rows)


if __name__ == "__main__":
    unittest.main()
