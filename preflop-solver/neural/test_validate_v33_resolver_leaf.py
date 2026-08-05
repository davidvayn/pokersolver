import json
import unittest
from pathlib import Path

import validate_v33_resolver_leaf as module


def training_report(rmse, bands, *, supplemental=False):
    report = {
        "sourcePolicySha256": "a" * 64,
        "sourceValidation": {"status": "accepted", "reasons": []},
        "meanRangeRmseBb": rmse,
        "crossSeedPredictionCorrelation": {"range": 0.99},
        "validationStates": [6, 7],
        "trainStates": [0, 1, 2, 3],
        "tuningStates": [4, 5],
        "primaryStates": 8,
        "supplementalTrainingStates": [],
        "componentDatasetSha256": ["1" * 64],
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
        report["componentDatasetSha256"].append("2" * 64)
    return report


def leaf_report(rmse):
    return {
        "sourceDatasetSha256": "3" * 64,
        "sourcePolicySha256": "a" * 64,
        "resolverReachEvaluation": {
            "reachWeightedRmseBb": rmse,
            "reachWeightedMaeBb": rmse / 2,
            "sampledLeafReachMass": 0.1,
        },
    }


def parity():
    return {"maximumAbsoluteErrorBb": 1e-6, "validation": {"status": "accepted"}}


def resolver(value, board=None):
    return {
        "state": {"board": board or [0, 5, 10]},
        "iterations": 100,
        "metrics": {"depth_limited_exploitability_bb_per_hand": value},
    }


class V33ResolverLeafTests(unittest.TestCase):
    def setUp(self):
        self.baseline = training_report(
            0.60, {"small": 0.30, "medium": 0.90, "large": 1.80}
        )
        self.candidate = training_report(
            0.59,
            {"small": 0.31, "medium": 0.88, "large": 1.75},
            supplemental=True,
        )

    def test_prediction_gates_require_disjoint_leaf_evaluation(self):
        self.candidate["componentDatasetSha256"].append("3" * 64)
        report = module.compose(
            self.baseline,
            self.candidate,
            leaf_report(1.0),
            leaf_report(0.8),
            parity(),
        )
        self.assertFalse(report["gates"]["resolverEvaluationCorpusUntouched"]["passed"])
        self.assertFalse(report["resolverEvaluation"]["eligible"])

    def test_candidate_must_improve_two_boards_and_mean_by_two_percent(self):
        boards = ([0, 5, 10], [1, 6, 11], [2, 7, 12])
        report = module.compose(
            self.baseline,
            self.candidate,
            leaf_report(1.0),
            leaf_report(0.8),
            parity(),
            [resolver(0.45, boards[0]), resolver(0.48, boards[1]), resolver(0.53, boards[2])],
            [resolver(0.50, boards[0]), resolver(0.50, boards[1]), resolver(0.50, boards[2])],
        )
        self.assertTrue(report["gates"]["matchedResolverImprovement"]["passed"])
        self.assertEqual(report["researchSelection"], "v33")
        self.assertFalse(report["activationAllowed"])

    def test_leaf_improvement_cannot_replace_full_game_upper_bound(self):
        report = module.compose(
            self.baseline,
            self.candidate,
            leaf_report(1.0),
            leaf_report(0.8),
            parity(),
        )
        self.assertTrue(report["gates"]["resolverLeafReachWeightedImprovement"]["passed"])
        self.assertFalse(report["gates"]["fullGameExploitabilityUpperBound"]["passed"])
        self.assertFalse(report["activationAllowed"])

    def test_checked_candidate_remains_fail_closed_and_evaluation_is_disjoint(self):
        candidate = json.loads(
            Path(__file__).with_name("20bb-v33-resolver-leaf-candidate.json").read_text()
        )
        self.assertEqual(candidate["status"], "rejected")
        self.assertFalse(candidate["activationAllowed"])
        self.assertFalse(candidate["activeManifestModified"])
        self.assertEqual(candidate["researchSelection"], "v31")
        self.assertTrue(
            candidate["gates"]["resolverEvaluationCorpusUntouched"]["passed"]
        )
        self.assertFalse(
            candidate["gates"]["fullGameExploitabilityUpperBound"]["passed"]
        )
        self.assertIn("smallPotNoRegression", candidate["failedGates"])


if __name__ == "__main__":
    unittest.main()
