import unittest

import validate_v30_public_belief as module


class V30GateTests(unittest.TestCase):
    def test_upper_bound_and_all_in_gates_fail_closed(self) -> None:
        target = {
            "validation": {"status": "accepted", "reasons": []},
            "source_policy_sha256": "a" * 64,
            "targets": [
                {
                    "maximum_river_exploitability_bb_per_hand": 0.01,
                    "zero_sum_residual_bb": 0.0,
                    "board": [index, 100, 101, 102],
                    "range_maximum_total_variation": 0.1,
                    "range_particles": 4096,
                    "range_replicates": 2,
                    "belief_method": "exact_per-player_reach_factors_test",
                }
                for index in range(64)
            ],
        }
        turn = {
            "validation": {"status": "accepted", "reasons": []},
            "meanRangeRmseBb": 0.2,
            "rangeRelativeImprovement": 0.1,
            "crossSeedPredictionCorrelation": {"range": 0.99},
            "structurallySuitEquivariant": True,
            "structurallyZeroSumProjected": True,
        }
        parity = {
            "validation": {"status": "accepted", "reasons": []},
            "maximumAbsoluteErrorBb": 1e-6,
        }
        report = module.compose(target, turn, parity)
        self.assertEqual(report["status"], "rejected")
        self.assertIn("fullFlopActionAbstraction", report["failedGates"])
        self.assertIn("fullGameExploitabilityUpperBound", report["failedGates"])


if __name__ == "__main__":
    unittest.main()
