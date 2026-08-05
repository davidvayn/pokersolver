import json
import unittest
from pathlib import Path

import validate_v31_calibration as module


def training_report(normalization, tuning, overall, bands, correlation=0.99):
    return {
        "valueNormalization": normalization,
        "sourcePolicySha256": "a" * 64,
        "meanRangeRmseBb": overall,
        "meanNoRangeRmseBb": overall * 2,
        "rangeRelativeImprovement": 0.5,
        "crossSeedPredictionCorrelation": {"range": correlation},
        "variants": {
            "range": [
                {
                    "seed": seed,
                    "weights": f"seed-{seed}.json",
                    "metrics": {
                        "bestTuningRmseBb": value,
                        "potBandMetrics": {
                            band: {"weightedRmseBb": rmse}
                            for band, rmse in bands.items()
                        },
                    },
                }
                for seed, value in zip((1, 2), tuning, strict=True)
            ]
        },
    }


def resolver(value):
    return {
        "state": {"board": [0, 1, 2]},
        "iterations": 100,
        "metrics": {"depth_limited_exploitability_bb_per_hand": value},
    }


class V31CalibrationGateTests(unittest.TestCase):
    def setUp(self):
        self.baseline = {
            "turnValueEvaluation": {
                "rmseByInvestedPotBandBb": {
                    "smallAtMost3_5Each": 0.30,
                    "medium4To7_5Each": 1.00,
                    "largeAtLeast10_5Each": 2.00,
                }
            }
        }
        self.parity = {
            "validation": {"status": "accepted"},
            "maximumAbsoluteErrorBb": 1e-6,
        }
        self.payoff = training_report(
            "payoff-exposure", [0.40, 0.42], 0.30,
            {"small": 0.25, "medium": 0.70, "large": 1.40},
        )

    def test_failed_medium_gate_blocks_conditional_scale_and_exact_branch(self):
        pot = training_report(
            "pot", [0.30, 0.31], 0.24,
            {"small": 0.25, "medium": 0.80, "large": 1.40},
        )
        report = module.compose(
            self.baseline, pot, self.payoff, self.parity,
            [resolver(0.04)], [resolver(0.06)], None,
        )
        self.assertFalse(report["gates"]["mediumPotImprovement"]["passed"])
        self.assertEqual(
            report["conditionalSteps"]["balanced512Corpus"],
            "not_run_prerequisite_failed",
        )
        self.assertEqual(
            report["conditionalSteps"]["exactLowSprAllInHybrid"],
            "not_run_prerequisite_failed",
        )
        self.assertFalse(report["activationAllowed"])

    def test_tuning_selects_normalization_and_seed_without_holdout_leakage(self):
        pot = training_report(
            "pot", [0.20, 0.10], 9.0,
            {"small": 0.20, "medium": 0.70, "large": 1.40},
        )
        report = module.compose(
            self.baseline, pot, self.payoff, self.parity,
            [resolver(0.04)], [resolver(0.06)], None,
        )
        self.assertEqual(report["normalizationSelection"]["selected"], "pot")
        self.assertEqual(report["normalizationSelection"]["selectedSeed"], 2)
        self.assertEqual(
            report["normalizationSelection"]["rule"],
            "lowest paired mean range tuning RMSE; holdout is evaluation only",
        )

    def test_learned_response_never_satisfies_upper_bound_gate(self):
        pot = training_report(
            "pot", [0.20, 0.21], 0.20,
            {"small": 0.20, "medium": 0.70, "large": 1.40},
        )
        lbr = {
            "network_sha256": "a" * 64,
            "approximate_exploitability_lower_bound_bb_per_hand": 0.0,
            "approximate_exploitability_lower_confidence_bound_99_percent_bb_per_hand": 0.0,
            "interpretation": "lower bound only",
        }
        report = module.compose(
            self.baseline, pot, self.payoff, self.parity,
            [resolver(0.04)], [resolver(0.06)], lbr,
        )
        self.assertTrue(report["gates"]["independentLearnedResponseEvaluated"]["passed"])
        self.assertFalse(report["gates"]["fullGameExploitabilityUpperBound"]["passed"])
        self.assertFalse(report["activationAllowed"])

    def test_checked_candidate_records_conditional_rejection(self):
        candidate = json.loads(
            Path(__file__).with_name("20bb-v31-calibration-candidate.json").read_text()
        )
        self.assertEqual(candidate["status"], "rejected")
        self.assertFalse(candidate["activationAllowed"])
        self.assertFalse(candidate["activeManifestModified"])
        self.assertEqual(
            candidate["conditionalSteps"]["balanced512Corpus"],
            "not_run_prerequisite_failed",
        )
        self.assertEqual(
            candidate["conditionalSteps"]["exactLowSprAllInHybrid"],
            "not_run_prerequisite_failed",
        )
        self.assertIn("fullGameExploitabilityUpperBound", candidate["failedGates"])
        correction = candidate["turnValueEvaluation"]["matchedHoldoutCorrection"]
        self.assertEqual(len(correction["stateIndices"]), 32)
        self.assertGreater(correction["relativeImprovement"], 0.10)


if __name__ == "__main__":
    unittest.main()
