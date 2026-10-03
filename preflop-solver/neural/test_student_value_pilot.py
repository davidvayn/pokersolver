"""Student policy pilot rejects incomplete pairs and scores starting-pot units."""
import math
from pathlib import Path
import tempfile
import unittest
import json

from run_native_value_preflight import sha256
from run_student_value_pilot import select_students, summarize


class StudentValuePilotTests(unittest.TestCase):
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
