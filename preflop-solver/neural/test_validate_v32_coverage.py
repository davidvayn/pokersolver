import json
import unittest
from pathlib import Path

import validate_v32_coverage as module


def training_report(rmse, bands, *, supplemental=False, architecture="compact"):
    report = {
        "architecture": architecture,
        "valueNormalization": "pot",
        "sourcePolicySha256": "a" * 64,
        "sourceValidation": {"status": "accepted", "reasons": []},
        "meanRangeRmseBb": rmse,
        "crossSeedPredictionCorrelation": {"range": 0.99},
        "validationStates": [6, 7],
        "trainStates": [0, 1, 2, 3],
        "tuningStates": [4, 5],
        "primaryStates": 8,
        "supplementalTrainingStates": [],
        "variants": {
            "range": [
                {
                    "seed": seed,
                    "weights": f"seed-{seed}.json",
                    "metrics": {
                        "bestTuningRmseBb": tuning,
                        "potBandMetrics": {
                            band: {"weightedRmseBb": value}
                            for band, value in bands.items()
                        },
                    },
                }
                for seed, tuning in ((1, 0.2), (2, 0.3))
            ]
        },
    }
    if supplemental:
        report["trainStates"] += [8, 9]
        report["supplementalTrainingStates"] = [8, 9]
    return report


def parity(error=1e-6):
    return {"maximumAbsoluteErrorBb": error, "validation": {"status": "accepted"}}


def resolver(value):
    return {
        "state": {"board": [0, 5, 10]},
        "iterations": 100,
        "metrics": {"depth_limited_exploitability_bb_per_hand": value},
    }


class V32CoverageTests(unittest.TestCase):
    def setUp(self):
        self.baseline = training_report(
            0.60, {"small": 0.30, "medium": 0.90, "large": 1.80}
        )
        self.deep = training_report(
            0.62,
            {"small": 0.31, "medium": 0.93, "large": 1.86},
            architecture="deep-gelu",
        )
        self.candidate = training_report(
            0.54,
            {"small": 0.31, "medium": 0.81, "large": 1.62},
            supplemental=True,
        )

    def test_passing_prediction_gates_only_make_resolver_eligible(self):
        report = module.compose(
            self.baseline, self.deep, parity(), self.candidate, parity()
        )
        self.assertTrue(report["resolverEvaluation"]["eligible"])
        self.assertFalse(report["gates"]["matchedResolverImprovement"]["passed"])
        self.assertFalse(report["activationAllowed"])
        self.assertFalse(report["gates"]["fullGameExploitabilityUpperBound"]["passed"])

    def test_mismatched_holdout_fails_closed(self):
        self.candidate["validationStates"] = [5, 7]
        report = module.compose(
            self.baseline, self.deep, parity(), self.candidate, parity()
        )
        self.assertFalse(report["gates"]["matchedPrimaryHoldout"]["passed"])
        self.assertFalse(report["resolverEvaluation"]["eligible"])

    def test_supplement_cannot_leak_into_validation(self):
        self.candidate["validationStates"] = [6, 8]
        self.baseline["validationStates"] = [6, 8]
        report = module.compose(
            self.baseline, self.deep, parity(), self.candidate, parity()
        )
        self.assertFalse(report["gates"]["supplementalDataTrainingOnly"]["passed"])
        self.assertFalse(report["resolverEvaluation"]["eligible"])

    def test_deep_capacity_gate_is_reported_but_does_not_block_compact_candidate(self):
        report = module.compose(
            self.baseline, self.deep, parity(), self.candidate, parity()
        )
        self.assertFalse(report["gates"]["deepCapacityPilotPreferred"]["passed"])
        self.assertTrue(report["resolverEvaluation"]["eligible"])
        self.assertEqual(report["capacityPilot"]["decision"], "rejected_keep_compact")

    def test_matched_resolver_must_improve_but_never_activates_without_upper_bound(self):
        report = module.compose(
            self.baseline,
            self.deep,
            parity(),
            self.candidate,
            parity(),
            [resolver(0.4)],
            [resolver(0.5)],
        )
        self.assertTrue(report["gates"]["matchedResolverImprovement"]["passed"])
        self.assertFalse(report["activationAllowed"])
        self.assertIn("fullGameExploitabilityUpperBound", report["failedGates"])

    def test_checked_candidate_records_resolver_rejection(self):
        candidate = json.loads(
            Path(__file__).with_name(
                "20bb-v32-off-policy-coverage-candidate.json"
            ).read_text()
        )
        self.assertEqual(candidate["status"], "rejected")
        self.assertFalse(candidate["activationAllowed"])
        self.assertFalse(candidate["activeManifestModified"])
        self.assertTrue(candidate["resolverEvaluation"]["eligible"])
        self.assertFalse(candidate["gates"]["matchedResolverImprovement"]["passed"])
        self.assertIn("fullGameExploitabilityUpperBound", candidate["failedGates"])


if __name__ == "__main__":
    unittest.main()
