import unittest

from compare_tabular_preflop import compare


class CompareTabularPreflopTests(unittest.TestCase):
    def test_comparison_is_reach_weighted_and_tie_aware(self):
        def artifact(seed, probabilities):
            return {
                "seed": seed,
                "iterations": 10,
                "strategies": [
                    {
                        "key": "root",
                        "action_labels": ["fold", "call"],
                        "probabilities": probabilities,
                        "average_visits": 100,
                    }
                ],
            }

        result = compare(artifact(1, [0.505, 0.495]), artifact(2, [0.495, 0.505]))
        self.assertEqual(result["reachWeightedPrimaryAgreement"], 0.0)
        self.assertEqual(result["reachWeightedTieAwarePrimaryAgreementAt0_01"], 1.0)
        self.assertAlmostEqual(result["reachWeightedActionFrequencyMae"], 0.01)
        self.assertAlmostEqual(result["maximumAggregateActionDelta"], 0.01)

    def test_average_reach_weight_controls_aggregation(self):
        def artifact(seed, first):
            return {
                "seed": seed,
                "iterations": 10,
                "strategies": [
                    {
                        "key": "common",
                        "action_labels": ["fold", "call"],
                        "probabilities": first,
                        "average_visits": 1,
                        "average_reach_weight": 99.0,
                    },
                    {
                        "key": "rare",
                        "action_labels": ["fold", "call"],
                        "probabilities": [0.5, 0.5],
                        "average_visits": 10000,
                        "average_reach_weight": 1.0,
                    },
                ],
            }

        result = compare(artifact(1, [1.0, 0.0]), artifact(2, [0.0, 1.0]))
        self.assertAlmostEqual(result["reachWeightedActionFrequencyMae"], 0.99)


if __name__ == "__main__":
    unittest.main()
