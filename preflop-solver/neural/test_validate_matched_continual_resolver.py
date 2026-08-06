import json
import tempfile
import unittest
from pathlib import Path

import run_matched_continual_resolver as runner
import validate_matched_continual_resolver as validator


class ValidateMatchedContinualResolverTests(unittest.TestCase):
    def job(self):
        return {
            "board": "2d,8h,Ks",
            "boardIndices": runner.board_indices("2d,8h,Ks"),
            "strategyModel": {
                "seed": 15301,
                "sha256": "a" * 64,
                "sourceDatasetSha256": "b" * 64,
                "sourcePolicySha256": "c" * 64,
            },
            "evaluationModel": {
                "seed": 15302,
                "sha256": "d" * 64,
                "sourceDatasetSha256": "e" * 64,
                "sourcePolicySha256": "f" * 64,
            },
        }

    def controls(self):
        return {
            "effectiveStackBb": 20.0,
            "rootPotBb": 4.0,
            "rootActor": 1,
            "iterations": 100,
            "averagingDelay": 10,
            "threads": 10,
            "maximumExploitabilityBbPerHand": 0.05,
        }

    def payload(self, exploitability: float = 0.04):
        job = self.job()
        actions = 2
        probabilities = []
        board = set(job["boardIndices"])
        legal_count = sum(
            runner.legal_combo(combo, board)
            for combo in range(runner.COMBO_COUNT)
        )
        root_range = [
            1.0 / legal_count if runner.legal_combo(combo, board) else 0.0
            for combo in range(runner.COMBO_COUNT)
        ]
        for combo in range(runner.COMBO_COUNT):
            probabilities.extend(
                [0.25, 0.75]
                if runner.legal_combo(combo, board)
                else [0.0, 0.0]
            )
        return {
            "schema": runner.SOLUTION_SCHEMA,
            "method": runner.SOLUTION_METHOD,
            "approximate": True,
            "effective_stack_bb": 20.0,
            "value_network_seed": 15301,
            "value_network_sha256": "a" * 64,
            "uses_exact_ranges": True,
            "value_network_source_dataset_sha256": "b" * 64,
            "value_network_source_policy_sha256": "c" * 64,
            "evaluation_value_network_seed": 15302,
            "evaluation_value_network_sha256": "d" * 64,
            "evaluation_value_network_source_dataset_sha256": "e" * 64,
            "evaluation_value_network_source_policy_sha256": "f" * 64,
            "state": {
                "street": "flop",
                "board": job["boardIndices"],
                "actor": 1,
                "invested_bb": [2.0, 2.0],
                "street_invested_bb": [0.0, 0.0],
                "public_history": ["public_belief:flop_start"],
                "aggressions": 0,
                "checks": 0,
                "raise_reopened": True,
                "ranges": [root_range, root_range],
            },
            "iterations": 100,
            "averaging_delay": 10,
            "threads": 10,
            "strategies": [
                {
                    "action_labels": ["check", "bet"],
                    "probabilities": probabilities,
                }
            ],
            "counterfactual_values_bb": [[0.0] * runner.COMBO_COUNT for _ in range(2)],
            "opponent_compatible_mass": [[1.0] * runner.COMBO_COUNT for _ in range(2)],
            "metrics": {
                "depth_limited_exploitability_bb_per_hand": exploitability,
                "resolver_relative_exploitability_improvement": 0.9,
                "zero_sum_residual_after_projection_bb": 1e-12,
            },
            "validation": {"status": "accepted", "reasons": []},
        }

    def test_solution_requires_exact_artifacts_and_valid_probability_rows(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            path = Path(raw_directory) / "solution.json"
            payload = self.payload()
            path.write_text(json.dumps(payload))
            evidence = runner.inspect_solution(self.job(), self.controls(), path)
            self.assertTrue(evidence["accepted"])
            self.assertLessEqual(evidence["maximumProbabilitySumError"], 1e-7)

            payload["value_network_sha256"] = "9" * 64
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "provenance"):
                runner.inspect_solution(self.job(), self.controls(), path)

    def test_exploitability_failure_is_reported_without_activating(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            path = Path(raw_directory) / "solution.json"
            path.write_text(json.dumps(self.payload(exploitability=0.050001)))
            evidence = runner.inspect_solution(self.job(), self.controls(), path)
            self.assertFalse(evidence["accepted"])
            self.assertFalse(evidence["gates"]["maximumLocalExploitability"])

    def test_aggregate_requires_both_cross_seed_directions_per_root(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            release_path = root / "release.json"
            release_path.write_text("release")
            value_path = root / "value.json"
            value_path.write_text("value")
            first = {
                "board": "2d,8h,Ks",
                "strategySeed": 15301,
                "evaluationSeed": 15302,
                "depthLimitedExploitabilityBbPerHand": 0.04,
                "resolverRelativeExploitabilityImprovement": 0.9,
                "zeroSumResidualBb": 0.0,
                "maximumProbabilitySumError": 0.0,
                "gates": {
                    "artifactProvenance": True,
                    "acceptedSolverValidation": True,
                    "probabilitySums": True,
                    "maximumLocalExploitability": True,
                    "positiveResolverImprovement": True,
                    "zeroSumResidual": True,
                },
            }
            second = {
                **first,
                "strategySeed": 15302,
                "evaluationSeed": 15301,
            }
            plan = {
                "controls": self.controls(),
                "models": [{"seed": 15301}, {"seed": 15302}],
                "rootCount": 1,
            }
            accepted = validator.summarize(
                plan, [first, second], release_path, value_path
            )
            self.assertEqual(
                accepted["status"],
                "accepted-awaiting-preflop-and-full-game-gates",
            )
            self.assertFalse(accepted["activationAllowed"])
            rejected = validator.summarize(
                plan, [first, first], release_path, value_path
            )
            self.assertEqual(rejected["status"], "rejected")
            self.assertFalse(rejected["gates"]["bothCrossSeedDirectionsPerRoot"])


if __name__ == "__main__":
    unittest.main()
