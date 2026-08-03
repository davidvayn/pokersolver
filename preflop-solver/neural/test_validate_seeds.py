import unittest

import mlx.core as mx
import numpy as np

from train import INPUT_FEATURE_COUNT, STATE_FEATURE_COUNT
from validate_seeds import (
    StreetRoutedModel,
    action_ev_standard_error_summary,
    compare,
)


def state(private_cards):
    return {
        "private_cards": private_cards,
        "board": [],
        "street": "preflop",
        "actor": 0,
        "button": 0,
        "pot_bb": 0.0,
        "stacks_bb": [19.5, 19.0],
        "street_bets_bb": [0.5, 1.0],
        "total_committed_bb": [0.5, 1.0],
        "to_call_bb": 0.5,
        "last_full_raise_bb": 1.0,
        "raise_reopened": True,
        "trajectory": [],
    }


def record(private_cards):
    return {
        "state": state(private_cards),
        "actions": [
            {"kind": "fold", "amount_to_bb": None},
            {"kind": "call", "amount_to_bb": None},
        ],
    }


class UniformModel:
    def __call__(self, features):
        return mx.zeros((features.shape[0], 1))


class RankMarkerModel:
    """Prefers call only when canonical card zero is visible."""

    def __call__(self, features):
        values = np.asarray(features)
        marker = values[:, 0]
        fold = values[:, STATE_FEATURE_COUNT]
        call = values[:, STATE_FEATURE_COUNT + 2]
        logits = marker * (-4.0 * fold + 4.0 * call)
        return mx.array(logits.reshape((-1, 1)))


class ReachAwareValidationTests(unittest.TestCase):
    def test_action_ev_gate_requires_every_action_at_a_served_decision(self):
        precise = {
            **record([0, 4]),
            "action_value_standard_errors_bb": [0.01, 0.02],
        }
        noisy = {
            **record([48, 44]),
            "action_value_standard_errors_bb": [0.01, 0.03],
        }
        result = action_ev_standard_error_summary([[precise, noisy]])
        self.assertTrue(result["available"])
        self.assertEqual(result["decisions"], 2)
        self.assertAlmostEqual(result["decision_coverage"], 0.5)
        self.assertAlmostEqual(result["action_coverage"], 0.75)
        self.assertAlmostEqual(result["maximum_standard_error_bb"], 0.03)

    def test_street_routed_model_selects_the_matching_component(self):
        features = np.zeros((2, INPUT_FEATURE_COUNT), dtype=np.float32)
        features[0, 104] = 1.0

        class ConstantModel:
            def __init__(self, value):
                self.value = value

            def __call__(self, values):
                return mx.full((values.shape[0], 1), self.value)

        result = np.asarray(
            StreetRoutedModel(ConstantModel(3.0), ConstantModel(7.0))(mx.array(features))
        ).reshape(-1)
        np.testing.assert_array_equal(result, np.asarray([3.0, 7.0]))

    def test_empirical_trajectory_comparison_keeps_repeated_visits(self):
        divergent = record([0, 4])
        agreeing = record([48, 44])
        result = compare(
            UniformModel(),
            RankMarkerModel(),
            [[divergent, divergent, agreeing]],
            20,
            "test trajectory distribution",
            True,
        )
        self.assertEqual(result["decisions"], 3)
        self.assertAlmostEqual(result["primary_action_agreement"], 1.0 / 3.0)
        self.assertEqual(result["sampling_method"], "empirical_pure_trajectories")
        self.assertAlmostEqual(result["street_breakdown"]["preflop"]["reach_mass"], 1.0)

    def test_policy_corpora_receive_equal_mixture_mass(self):
        divergent = record([0, 4])
        agreeing = record([48, 44])
        result = compare(
            UniformModel(),
            RankMarkerModel(),
            [[divergent, divergent], [agreeing]],
            20,
            "equal policy mixture",
            True,
        )
        self.assertEqual(result["decisions"], 3)
        self.assertAlmostEqual(result["primary_action_agreement"], 0.5)
        self.assertAlmostEqual(result["effective_decisions"], 8.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
