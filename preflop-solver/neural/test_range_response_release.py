import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import freeze_range_response_release as freezer
import run_matched_continual_resolver as matched_v1
import run_range_response_release as runner
import validate_range_response_release as validator
import validate_resolver_reach_corpus as corpus_validator


class RangeResponseReleaseTests(unittest.TestCase):
    def controls(self):
        return {
            "effectiveStackBb": 20.0,
            "rootPotBb": 4.0,
            "rootActor": 1,
            "strategyIterations": 100,
            "strategyCheckpoints": [25, 50, 100],
            "strategyAveragingDelay": 10,
            "responseCheckpoints": [100, 200, 400],
            "responseAveragingDelay": 10,
            "threads": 10,
            "crossEvaluateBothDirections": True,
            "outputDirectory": "runs/range-response",
        }

    def gates(self):
        return {
            "maximumRangeConsistentResponseGainBbPerHand": 0.05,
            "maximumFinalCheckpointIncreaseBbPerHand": 0.005,
            "maximumZeroSumResidualBb": 1e-6,
            "maximumProbabilitySumError": 1e-5,
            "requireEveryRootAndDirection": True,
            "interpretAsExploitabilityUpperBound": False,
            "activationRequiresIndependentFullGameUpperBound": True,
        }

    def model(self, seed: int, marker: str):
        return {
            "seed": seed,
            "path": f"model-{seed}.json",
            "sha256": marker * 64,
            "sourceDatasetSha256": ("c" if marker != "c" else "d") * 64,
            "sourcePolicySha256": "e" * 64,
        }

    def root(self):
        board = "2d,8h,Ks"
        return {
            "board": board,
            "boardIndices": matched_v1.board_indices(board),
            "texture": "unpairedRainbowDisconnected",
            "suitIsomorphismKey": list(
                corpus_validator.suit_isomorphism_key(
                    tuple(matched_v1.board_indices(board))
                )
            ),
        }

    def job(self):
        return {
            "name": "fixture",
            "root": self.root(),
            "strategyModel": self.model(15301, "a"),
            "evaluationModel": self.model(15302, "b"),
            "convergenceOutput": "convergence.json",
            "responseOutput": "response.json",
        }

    def state(self):
        board = self.root()["boardIndices"]
        blocked = set(board)
        legal_count = sum(
            matched_v1.legal_combo(combo, blocked)
            for combo in range(matched_v1.COMBO_COUNT)
        )
        root_range = [
            1.0 / legal_count if matched_v1.legal_combo(combo, blocked) else 0.0
            for combo in range(matched_v1.COMBO_COUNT)
        ]
        return {
            "street": "flop",
            "board": board,
            "actor": 1,
            "invested_bb": [2.0, 2.0],
            "street_invested_bb": [0.0, 0.0],
            "public_history": ["public_belief:flop_start"],
            "aggressions": 0,
            "checks": 0,
            "raise_reopened": True,
            "ranges": [root_range, root_range],
        }

    def strategies(self):
        board = set(self.root()["boardIndices"])
        probabilities = []
        for combo in range(matched_v1.COMBO_COUNT):
            probabilities.extend(
                [0.25, 0.75]
                if matched_v1.legal_combo(combo, board)
                else [0.0, 0.0]
            )
        return [
            {
                "public_history": ["public_belief:flop_start"],
                "actor": 1,
                "action_labels": ["check", "bet:0.5pot"],
                "probabilities": probabilities,
            }
        ]

    def convergence_payload(self):
        job = self.job()
        strategy = job["strategyModel"]
        evaluation = job["evaluationModel"]
        strategies = self.strategies()
        state = self.state()
        metrics = {"depth_limited_exploitability_bb_per_hand": 0.3}
        final = {
            "schema": matched_v1.SOLUTION_SCHEMA,
            "method": matched_v1.SOLUTION_METHOD,
            "approximate": True,
            "effective_stack_bb": 20.0,
            "value_network_seed": strategy["seed"],
            "value_network_sha256": strategy["sha256"],
            "uses_exact_ranges": True,
            "value_network_source_dataset_sha256": strategy[
                "sourceDatasetSha256"
            ],
            "value_network_source_policy_sha256": strategy["sourcePolicySha256"],
            "evaluation_value_network_seed": evaluation["seed"],
            "evaluation_value_network_sha256": evaluation["sha256"],
            "evaluation_value_network_source_dataset_sha256": evaluation[
                "sourceDatasetSha256"
            ],
            "evaluation_value_network_source_policy_sha256": evaluation[
                "sourcePolicySha256"
            ],
            "state": state,
            "iterations": 100,
            "averaging_delay": 10,
            "threads": 10,
            "strategies": strategies,
        }
        checkpoint_solutions = []
        for iterations in (25, 50, 100):
            solution = copy.deepcopy(final)
            solution["iterations"] = iterations
            checkpoint_solutions.append(solution)
        return {
            "schema": runner.CONVERGENCE_SCHEMA,
            "method": runner.CONVERGENCE_METHOD,
            "approximate": True,
            "value_network_seed": strategy["seed"],
            "value_network_sha256": strategy["sha256"],
            "value_network_source_dataset_sha256": strategy[
                "sourceDatasetSha256"
            ],
            "value_network_source_policy_sha256": strategy["sourcePolicySha256"],
            "evaluation_value_network_seed": evaluation["seed"],
            "evaluation_value_network_sha256": evaluation["sha256"],
            "evaluation_value_network_source_dataset_sha256": evaluation[
                "sourceDatasetSha256"
            ],
            "evaluation_value_network_source_policy_sha256": evaluation[
                "sourcePolicySha256"
            ],
            "state": state,
            "averaging_delay": 10,
            "threads": 10,
            "checkpoints": [
                {"iterations": value, "metrics": metrics, "validation": {}}
                for value in (25, 50, 100)
            ],
            "checkpoint_solutions": checkpoint_solutions,
            "final_strategy_sha256": runner.strategy_sha256(strategies),
            "final_solution": final,
        }

    def response_payload(self, final_gain: float = 0.028):
        job = self.job()
        strategy = job["strategyModel"]
        evaluation = job["evaluationModel"]
        gains = [0.02, 0.025, final_gain]
        return {
            "schema": runner.RESPONSE_SCHEMA,
            "method": runner.RESPONSE_METHOD,
            "approximate": True,
            "interpretation": "finite response; not an exploitability upper bound",
            "frozen_strategy_sha256": runner.strategy_sha256(self.strategies()),
            "frozen_strategy_iterations": 100,
            "strategy_value_network_seed": strategy["seed"],
            "strategy_value_network_sha256": strategy["sha256"],
            "strategy_value_network_source_dataset_sha256": strategy[
                "sourceDatasetSha256"
            ],
            "strategy_value_network_source_policy_sha256": strategy[
                "sourcePolicySha256"
            ],
            "evaluation_value_network_seed": evaluation["seed"],
            "evaluation_value_network_sha256": evaluation["sha256"],
            "evaluation_value_network_source_dataset_sha256": evaluation[
                "sourceDatasetSha256"
            ],
            "evaluation_value_network_source_policy_sha256": evaluation[
                "sourcePolicySha256"
            ],
            "state": self.state(),
            "baseline_profile_value_p0_bb": 0.1,
            "baseline_profile_value_p1_bb": -0.1,
            "response_averaging_delay": 10,
            "threads": 10,
            "checkpoints": [
                {
                    "iterations": iteration,
                    "response_value_p0_bb": 0.1 + gain,
                    "response_value_p1_bb": -0.1 + gain,
                    "response_gain_p0_bb": gain,
                    "response_gain_p1_bb": gain,
                    "range_consistent_response_gain_bb_per_hand": gain,
                    "maximum_zero_sum_residual_bb": 1e-12,
                }
                for iteration, gain in zip((100, 200, 400), gains)
            ],
            "validation": {"status": "diagnostic_only"},
        }

    def test_fresh_root_selection_is_deterministic_balanced_and_disjoint(self):
        counts = {
            "unpairedRainbowDisconnected": 1,
            "unpairedRainbowConnected": 1,
            "unpairedTwoToneDisconnected": 1,
            "unpairedTwoToneConnected": 1,
            "unpairedMonotoneDisconnected": 1,
            "unpairedMonotoneConnected": 1,
            "pairedRainbow": 1,
            "pairedTwoTone": 1,
            "trips": 1,
        }
        excluded = {
            corpus_validator.suit_isomorphism_key(
                corpus_validator.parse_board("2d,8h,Ks")
            )
        }
        first = freezer.select_roots(15601, counts, excluded)
        second = freezer.select_roots(15601, counts, excluded)
        self.assertEqual(first, second)
        self.assertEqual(Counter(root["texture"] for root in first), Counter(counts))
        self.assertFalse(
            {tuple(root["suitIsomorphismKey"]) for root in first} & excluded
        )

    def test_freeze_paths_are_portable_with_an_absolute_repository_root(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory).resolve()
            protocol = root / "neural" / "protocol.json"
            protocol.parent.mkdir()
            protocol.write_text("{}")
            self.assertEqual(
                freezer.portable_path(root, protocol), "neural/protocol.json"
            )

    def test_burned_freeze_roots_are_explicitly_excluded(self):
        first = self.root()
        second_board = "3c,4c,Ac"
        second = {
            "board": second_board,
            "boardIndices": matched_v1.board_indices(second_board),
            "texture": "unpairedMonotoneConnected",
            "suitIsomorphismKey": list(
                corpus_validator.suit_isomorphism_key(
                    tuple(matched_v1.board_indices(second_board))
                )
            ),
        }
        payload = {
            "schema": freezer.RANGE_RESPONSE_FREEZE_SCHEMA,
            "activationAllowed": False,
            "rootSelection": {"roots": [first, second]},
        }
        expected = {
            tuple(first["suitIsomorphismKey"]),
            tuple(second["suitIsomorphismKey"]),
        }
        self.assertEqual(freezer.frozen_root_keys(payload), expected)

        payload["activationAllowed"] = True
        with self.assertRaisesRegex(ValueError, "fail-closed"):
            freezer.frozen_root_keys(payload)

    def test_fresh_authentic_recheck_must_be_fully_accepted(self):
        payload = {
            "schema": freezer.FRESH_AUTHENTIC_RECHECK_SCHEMA,
            "status": "accepted-awaiting-strategy-preflop-and-full-game-gates",
            "activationAllowed": False,
            "gates": {
                "maximumPerSeedRmseBb": 0.25,
                "minimumCrossSeedPredictionCorrelation": 0.95,
                "freshAuthenticPerSeedRmse": True,
                "freshAuthenticCrossSeedCorrelation": True,
                "uniqueAndDisjointStateFingerprints": True,
                "completeArtifactProvenance": True,
            },
        }
        freezer.validate_accepted_fresh_authentic_recheck(payload)

        payload["gates"]["completeArtifactProvenance"] = False
        with self.assertRaisesRegex(ValueError, "fresh authentic recheck"):
            freezer.validate_accepted_fresh_authentic_recheck(payload)

    def test_strategy_hash_matches_the_rust_binary_fixture(self):
        fixture = [
            {
                "public_history": ["root", "check"],
                "actor": 1,
                "action_labels": ["fold", "call"],
                "probabilities": [0.25, 0.75],
            }
        ]
        self.assertEqual(
            runner.strategy_sha256(fixture),
            "caa0399fc945c99975cf5d3466dcd84f395f1fa19d9149622efd07e567e75983",
        )

    def test_strengthened_algorithms_are_pinned_in_commands_and_artifacts(self):
        controls = self.controls()
        controls["strategyRegretMatchingPlus"] = True
        controls["responseRegretMatchingPlus"] = True
        job = self.job()
        convergence_command = runner.convergence_command(
            controls,
            job["root"],
            job["strategyModel"],
            job["evaluationModel"],
            job["convergenceOutput"],
        )
        response_command = runner.response_command(
            controls,
            job["convergenceOutput"],
            job["evaluationModel"],
            job["responseOutput"],
        )
        self.assertIn("--regret-matching-plus", convergence_command)
        self.assertIn("--regret-matching-plus", response_command)
        self.assertEqual(
            response_command[response_command.index("--strategy-iterations") + 1],
            "100",
        )

        convergence_payload = self.convergence_payload()
        convergence_payload["method"] = runner.convergence_method(controls)
        convergence_payload["regret_matching_plus"] = True
        for solution in [
            *convergence_payload["checkpoint_solutions"],
            convergence_payload["final_solution"],
        ]:
            solution["method"] = runner.solution_method(controls)
            solution["regret_matching_plus"] = True
        response_payload = self.response_payload()
        response_payload["method"] = runner.response_method(controls)
        response_payload["response_regret_matching_plus"] = True
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            convergence_path = root / "convergence.json"
            response_path = root / "response.json"
            convergence_path.write_text(json.dumps(convergence_payload))
            response_path.write_text(json.dumps(response_payload))
            convergence = runner.inspect_convergence(
                job, controls, self.gates(), convergence_path
            )
            response = runner.inspect_response(
                job, controls, self.gates(), convergence, response_path
            )
        self.assertTrue(convergence["accepted"])
        self.assertTrue(response["accepted"])

    def test_artifacts_require_exact_policy_link_and_response_bounds(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            convergence_path = root / "convergence.json"
            response_path = root / "response.json"
            convergence_payload = self.convergence_payload()
            convergence_path.write_text(json.dumps(convergence_payload))
            response_path.write_text(json.dumps(self.response_payload()))
            convergence = runner.inspect_convergence(
                self.job(), self.controls(), self.gates(), convergence_path
            )
            response = runner.inspect_response(
                self.job(), self.controls(), self.gates(), convergence, response_path
            )
            self.assertTrue(convergence["accepted"])
            self.assertTrue(response["accepted"])
            self.assertAlmostEqual(response["finalCheckpointIncreaseBbPerHand"], 0.003)

            convergence_payload["final_solution"]["strategies"][0][
                "probabilities"
            ][0] = 0.5
            convergence_payload["checkpoint_solutions"][-1]["strategies"][0][
                "probabilities"
            ][0] = 0.5
            convergence_path.write_text(json.dumps(convergence_payload))
            with self.assertRaisesRegex(ValueError, "strategy hash"):
                runner.inspect_convergence(
                    self.job(), self.controls(), self.gates(), convergence_path
                )

            response_path.write_text(json.dumps(self.response_payload(final_gain=0.051)))
            response = runner.inspect_response(
                self.job(), self.controls(), self.gates(), convergence, response_path
            )
            self.assertFalse(response["accepted"])
            self.assertFalse(response["gates"]["maximumResponseGain"])

    def test_aggregate_requires_every_fresh_root_in_both_directions(self):
        plan = {
            "controls": self.controls(),
            "gates": self.gates(),
            "models": [self.model(15301, "a"), self.model(15302, "b")],
            "rootCount": 1,
        }
        response = {
            "maximumResponseGainBbPerHand": 0.04,
            "finalCheckpointIncreaseBbPerHand": 0.003,
            "maximumZeroSumResidualBb": 1e-12,
            "gates": {
                "artifactProvenance": True,
                "strategyArtifactLink": True,
                "gainArithmetic": True,
                "maximumResponseGain": True,
                "finalCheckpointIncrease": True,
                "zeroSumResidual": True,
            },
        }
        convergence = {"accepted": True, "maximumProbabilitySumError": 1e-8}
        first = {
            "board": "2d,8h,Ks",
            "strategySeed": 15301,
            "evaluationSeed": 15302,
            "response": response,
            "convergence": convergence,
        }
        second = {**first, "strategySeed": 15302, "evaluationSeed": 15301}
        with tempfile.TemporaryDirectory() as raw_directory:
            freeze_path = Path(raw_directory) / "freeze.json"
            freeze_path.write_text("freeze")
            accepted = validator.summarize(plan, [first, second], freeze_path)
            self.assertTrue(all(accepted["gates"].values()))
            self.assertFalse(accepted["activationAllowed"])
            self.assertIn("not establish", accepted["interpretation"])
            rejected = validator.summarize(plan, [first, first], freeze_path)
            self.assertEqual(rejected["status"], "rejected")
            self.assertFalse(rejected["gates"]["bothCrossSeedDirectionsPerRoot"])


if __name__ == "__main__":
    unittest.main()
