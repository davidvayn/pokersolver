import json
import unittest
from pathlib import Path

import validate_v33_resolver_leaf as module


def training_report(rmse, bands, *, supplemental=False, seed_rmses=None):
    seed_rmses = seed_rmses or (rmse, rmse)
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
                        "weightedRmseBb": seed_rmse,
                        "potBandMetrics": {
                            band: {"weightedRmseBb": value}
                            for band, value in bands.items()
                        },
                    },
                }
                for (seed, tuning), seed_rmse in zip(
                    ((1, 0.2), (2, 0.3)), seed_rmses, strict=True
                )
            ]
        },
    }
    if supplemental:
        report["trainStates"] += [8, 9]
        report["supplementalTrainingStates"] = [8, 9]
        report["componentDatasetSha256"].append("2" * 64)
    return report


def leaf_report(rmse, seed=None):
    report = {
        "sourceDatasetSha256": "3" * 64,
        "sourcePolicySha256": "a" * 64,
        "resolverReachEvaluation": {
            "reachWeightedRmseBb": rmse,
            "reachWeightedMaeBb": rmse / 2,
            "sampledLeafReachMass": 0.1,
        },
    }
    if seed is not None:
        report["modelSeed"] = seed
    return report


def leaf_reports(rmse):
    return [leaf_report(rmse, seed) for seed in (1, 2)]


def parity():
    return [
        {
            "model": f"seed-{seed}.json",
            "maximumAbsoluteErrorBb": 1e-6,
            "validation": {"status": "accepted"},
        }
        for seed in (1, 2)
    ]


def resolver(value, board=None, seed=1, evaluation_seed=None):
    evaluation_seed = evaluation_seed or 10_000 + seed
    return {
        "value_network_seed": seed,
        "evaluation_value_network_seed": evaluation_seed,
        "evaluation_value_network_source_dataset_sha256": "e" * 64,
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
            leaf_reports(0.8),
            parity(),
        )
        self.assertFalse(report["gates"]["resolverEvaluationCorpusUntouched"]["passed"])
        self.assertFalse(report["resolverEvaluation"]["eligible"])

    def test_candidate_must_improve_two_boards_and_mean_by_two_percent(self):
        boards = ([0, 5, 10], [1, 6, 11], [2, 7, 12])
        candidate_resolvers = [
            resolver(value, board, seed)
            for seed, values in ((1, (0.45, 0.48, 0.53)), (2, (0.46, 0.49, 0.52)))
            for value, board in zip(values, boards, strict=True)
        ]
        baseline_resolvers = [
            resolver(0.50, board, seed) for seed in (1, 2) for board in boards
        ]
        report = module.compose(
            self.baseline,
            self.candidate,
            leaf_report(1.0),
            leaf_reports(0.8),
            parity(),
            candidate_resolvers,
            baseline_resolvers,
        )
        self.assertTrue(report["gates"]["matchedResolverCoverage"]["passed"])
        self.assertTrue(report["gates"]["matchedResolverImprovement"]["passed"])
        self.assertTrue(report["gates"]["modelSelectionEligible"]["passed"])
        self.assertEqual(report["researchSelection"], "v33")
        self.assertFalse(report["activationAllowed"])

    def test_downstream_resolver_selects_seed_instead_of_tuning_rmse(self):
        boards = ([0, 5, 10], [1, 6, 11], [2, 7, 12])
        candidate_resolvers = [
            resolver(value, board, seed)
            for seed, values in ((1, (0.49, 0.49, 0.51)), (2, (0.40, 0.45, 0.48)))
            for value, board in zip(values, boards, strict=True)
        ]
        baseline_resolvers = [
            resolver(0.50, board, seed) for seed in (1, 2) for board in boards
        ]
        report = module.compose(
            self.baseline,
            self.candidate,
            leaf_report(1.0),
            leaf_reports(0.8),
            parity(),
            candidate_resolvers,
            baseline_resolvers,
        )
        self.assertEqual(module.selected_seed(self.candidate)["seed"], 1)
        self.assertEqual(report["selectedSeed"], 2)
        self.assertEqual(
            report["resolverEvaluation"]["selectionBasis"],
            "lowest matched mean downstream resolver exploitability",
        )
        self.assertEqual(report["researchSelection"], "v33")

    def test_cross_seed_resolver_disagreement_fails_closed(self):
        boards = ([0, 5, 10], [1, 6, 11], [2, 7, 12])
        candidate_resolvers = [
            resolver(value, board, seed)
            for seed, values in ((1, (0.40, 0.45, 0.48)), (2, (0.55, 0.45, 0.55)))
            for value, board in zip(values, boards, strict=True)
        ]
        baseline_resolvers = [
            resolver(0.50, board, seed) for seed in (1, 2) for board in boards
        ]
        report = module.compose(
            self.baseline,
            self.candidate,
            leaf_report(1.0),
            leaf_reports(0.8),
            parity(),
            candidate_resolvers,
            baseline_resolvers,
        )
        self.assertTrue(report["gates"]["matchedResolverImprovement"]["passed"])
        self.assertFalse(report["gates"]["matchedResolverCrossSeedAgreement"]["passed"])
        self.assertFalse(report["gates"]["modelSelectionEligible"]["passed"])
        self.assertEqual(report["researchSelection"], "v31")

    def test_missing_second_seed_resolver_is_fail_closed(self):
        boards = ([0, 5, 10], [1, 6, 11], [2, 7, 12])
        report = module.compose(
            self.baseline,
            self.candidate,
            leaf_report(1.0),
            leaf_reports(0.8),
            parity(),
            [
                resolver(value, board, 1)
                for value, board in zip((0.4, 0.45, 0.48), boards, strict=True)
            ],
            [resolver(0.5, board, 1) for board in boards],
        )
        self.assertFalse(report["gates"]["matchedResolverCoverage"]["passed"])
        self.assertFalse(report["gates"]["modelSelectionEligible"]["passed"])
        self.assertEqual(report["researchSelection"], "v31")

    def test_self_evaluated_resolver_is_not_matched_evidence(self):
        boards = ([0, 5, 10], [1, 6, 11], [2, 7, 12])
        candidates = [
            resolver(value, board, 1, evaluation_seed=1)
            for value, board in zip((0.4, 0.45, 0.48), boards, strict=True)
        ]
        baselines = [
            resolver(0.5, board, 1, evaluation_seed=1) for board in boards
        ]
        report = module.compose(
            self.baseline,
            self.candidate,
            leaf_report(1.0),
            leaf_reports(0.8),
            parity(),
            candidates,
            baselines,
        )
        self.assertFalse(report["gates"]["matchedResolverCoverage"]["passed"])
        self.assertFalse(report["gates"]["modelSelectionEligible"]["passed"])

    def test_leaf_improvement_cannot_replace_full_game_upper_bound(self):
        report = module.compose(
            self.baseline,
            self.candidate,
            leaf_report(1.0),
            leaf_reports(0.8),
            parity(),
        )
        self.assertTrue(
            report["gates"]["resolverLeafReachWeightedImprovement"]["passed"]
        )
        self.assertFalse(report["gates"]["fullGameExploitabilityUpperBound"]["passed"])
        self.assertFalse(report["activationAllowed"])

    def test_absolute_authentic_gate_overrides_directional_resolver_gain(self):
        boards = ([0, 5, 10], [1, 6, 11], [2, 7, 12])
        candidate_resolvers = [
            resolver(value, board, seed)
            for seed, values in ((1, (0.45, 0.48, 0.53)), (2, (0.46, 0.49, 0.52)))
            for value, board in zip(values, boards, strict=True)
        ]
        baseline_resolvers = [
            resolver(0.50, board, seed) for seed in (1, 2) for board in boards
        ]
        report = module.compose(
            self.baseline,
            self.candidate,
            leaf_report(1.0),
            leaf_reports(0.8),
            parity(),
            candidate_resolvers,
            baseline_resolvers,
            model_version="20bb-v36-primary-replay-candidate",
            research_candidate="v36",
            research_baseline="v31",
            maximum_authentic_rmse_bb=0.25,
        )
        self.assertEqual(report["modelVersion"], "20bb-v36-primary-replay-candidate")
        self.assertFalse(report["gates"]["absoluteAuthenticHoldoutRmse"]["passed"])
        self.assertFalse(report["gates"]["modelSelectionEligible"]["passed"])
        self.assertEqual(report["researchSelection"], "v31")

    def test_absolute_authentic_gate_requires_every_seed(self):
        candidate = training_report(
            0.24,
            {"small": 0.20, "medium": 0.24, "large": 0.30},
            supplemental=True,
            seed_rmses=(0.20, 0.28),
        )
        report = module.compose(
            self.baseline,
            candidate,
            leaf_report(1.0),
            leaf_reports(0.8),
            parity(),
            maximum_authentic_rmse_bb=0.25,
        )
        gate = report["gates"]["absoluteAuthenticHoldoutRmse"]
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["measured"]["maximumSeedRmseBb"], 0.28)

    def test_parity_must_cover_both_candidate_seeds(self):
        report = module.compose(
            self.baseline,
            self.candidate,
            leaf_report(1.0),
            leaf_reports(0.8),
            parity()[:1],
        )
        self.assertFalse(report["gates"]["pythonRustParity"]["passed"])
        self.assertFalse(report["gates"]["modelSelectionEligible"]["passed"])

    def test_leaf_evaluation_must_cover_both_candidate_seeds(self):
        report = module.compose(
            self.baseline,
            self.candidate,
            leaf_report(1.0),
            leaf_reports(0.8)[:1],
            parity(),
        )
        self.assertFalse(
            report["gates"]["resolverLeafReachWeightedImprovement"]["passed"]
        )
        self.assertFalse(report["gates"]["modelSelectionEligible"]["passed"])

    def test_checked_candidate_remains_fail_closed_and_evaluation_is_disjoint(self):
        candidate = json.loads(
            Path(__file__)
            .with_name("20bb-v33-resolver-leaf-candidate.json")
            .read_text()
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
