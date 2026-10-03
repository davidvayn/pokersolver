import copy
import json
from pathlib import Path
import tempfile
import unittest

from run_native_value_preflight import sha256
from summarize_postflop_gap_pilot import summarize, verify_saved_outputs


class PairedResponseSummaryTest(unittest.TestCase):
    def fixture(self):
        protocol = {"schema": "postflop-benchmark-protocol-v1",
                    "spots": [{"id": "limped-paired", "potType": "limped", "startingPotBb": 2},
                              {"id": "single-raised-high-rainbow", "potType": "single-raised",
                               "startingPotBb": 5}]}
        pilot = {"schema": "postflop-gap-matched-leaf-pilot-v1", "phase": "compare",
                 "status": "complete", "iterations": 32,
                 "spots": [spot["id"] for spot in protocol["spots"]], "cases": []}
        for spot in pilot["spots"]:
            for seed in (100101, 100102):
                for arm, gain in (("native", .2), ("learned", .3)):
                    pilot["cases"].append({"spot": spot, "seed": seed,
                                           "arm": arm, "gainBb": gain})
        return pilot, protocol

    def test_requires_all_matched_responses_and_reports_both_units(self):
        pilot, protocol = self.fixture()
        result = summarize(pilot, protocol)
        self.assertAlmostEqual(result["meanNativeGainBb"], .2)
        self.assertAlmostEqual(result["meanLearnedGainBb"], .3)
        self.assertAlmostEqual(result["meanImprovementBb"], .1)
        self.assertAlmostEqual(result["meanImprovementPercentagePointsPot"], 3.5)
        self.assertTrue(result["allSeedRootPairsImprove"])
        self.assertFalse(result["releaseAccepted"])
        pilot["cases"].pop()
        with self.assertRaises(ValueError):
            summarize(pilot, protocol)

    def test_no_cherry_pick_of_duplicate_or_failed_case(self):
        pilot, protocol = self.fixture()
        bad = copy.deepcopy(pilot)
        bad["cases"].append(copy.deepcopy(bad["cases"][0]))
        with self.assertRaises(ValueError):
            summarize(bad, protocol)
        pilot["status"] = "failed"
        with self.assertRaises(ValueError):
            summarize(pilot, protocol)

    def test_saved_response_must_match_pinned_hash_and_manifest_gain(self):
        pilot, _ = self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for row in pilot["cases"]:
                work = root / row["spot"] / str(row["seed"]) / row["arm"]
                work.mkdir(parents=True)
                candidate, response = work / "candidate.json", work / "response.json"
                candidate.write_text("{}")
                response.write_text(json.dumps({"half_summed_gain_bb": row["gainBb"]}))
                row["candidateSha256"] = sha256(candidate)
                row["responseSha256"] = sha256(response)
            verify_saved_outputs(pilot, root)
            pilot["cases"][0]["gainBb"] += .01
            with self.assertRaises(ValueError):
                verify_saved_outputs(pilot, root)


if __name__ == "__main__":
    unittest.main()
