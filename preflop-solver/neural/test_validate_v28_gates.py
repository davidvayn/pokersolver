import unittest

from validate_v28_gates import evaluate_gates


def policy(seed: int):
    return {
        "seed": seed,
        "strategies": [
            {"key": "root", "probabilities": [0.25, 0.75]},
            {"key": "response", "probabilities": [1.0, 0.0]},
        ],
    }


def evaluation(exploitability: float):
    return {
        "exploitability_bb_per_hand": exploitability,
        "policy_lookup_coverage": 1.0,
    }


def cross_seed():
    return {
        "reachWeightedActionFrequencyMae": 0.04,
        "reachWeightedPrimaryAgreement": 0.9,
        "maximumAggregateActionDelta": 0.02,
        "lookupIntersectionCoverage": 1.0,
    }


class ValidateV28GateTests(unittest.TestCase):
    def test_accepts_only_complete_passing_evidence(self):
        result = evaluate_gates(
            [policy(1), policy(2)],
            [evaluation(0.04), evaluation(0.05)],
            cross_seed(),
            {"upperBounds99BbPerHand": [0.08, 0.09]},
            {"reachWeightedCoverageAt0_02Bb": 0.96},
            200_000,
        )
        self.assertTrue(result["allPassed"])
        self.assertEqual(result["status"], "accepted")

    def test_missing_statistical_evidence_fails_closed(self):
        result = evaluate_gates(
            [policy(1), policy(2)],
            [evaluation(0.04), evaluation(0.04)],
            cross_seed(),
            None,
            None,
            None,
        )
        self.assertFalse(result["allPassed"])
        self.assertFalse(result["gates"]["exploitabilityUpperBound99"])
        self.assertFalse(result["gates"]["actionEvStandardErrorCoverage"])
        self.assertFalse(result["gates"]["projectedStorage"])
        self.assertEqual(result["status"], "rejected_not_activated")

    def test_invalid_probabilities_and_one_bad_seed_are_rejected(self):
        invalid = policy(2)
        invalid["strategies"][0]["probabilities"] = [0.6, 0.6]
        result = evaluate_gates(
            [policy(1), invalid],
            [evaluation(0.04), evaluation(0.051)],
            cross_seed(),
            {"upperBounds99BbPerHand": [0.08, 0.09]},
            {"reachWeightedCoverageAt0_02Bb": 0.96},
            200_000,
        )
        self.assertFalse(result["gates"]["probabilitySums"])
        self.assertFalse(result["gates"]["exploitabilityEstimate"])
        self.assertFalse(result["allPassed"])


if __name__ == "__main__":
    unittest.main()
