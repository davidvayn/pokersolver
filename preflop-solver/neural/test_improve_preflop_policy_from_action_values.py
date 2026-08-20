import copy
import unittest

from improve_preflop_policy_from_action_values import (
    conservative_regret_target,
    improve_policy,
    normalize,
)


class ImprovePreflopPolicyTests(unittest.TestCase):
    def test_normalize_keeps_zero_tail_non_negative(self) -> None:
        normalized = normalize([0.7904304627504474, 0.2095695372495529, 0.0])
        self.assertTrue(all(value >= 0 for value in normalized))
        self.assertEqual(sum(normalized), 1.0)

    def test_conservative_regret_target_uses_measured_advantages(self) -> None:
        target, gain = conservative_regret_target(
            [0.5, 0.5], [0.0, 1.0], [0.0, 0.0], 0.0, 0.0
        )
        self.assertEqual(target, [0.0, 1.0])
        self.assertAlmostEqual(gain, 0.5)

    def test_root_actor_selection_preserves_every_other_row(self) -> None:
        root = {
            "key": "p0|22|blinds:0.500/1.000",
            "actor": 0,
            "public_history": ["blinds:0.500/1.000"],
            "action_labels": ["fold", "limp"],
            "probabilities": [0.75, 0.25],
        }
        other = {
            "key": "p1|22|later",
            "actor": 1,
            "public_history": ["blinds:0.500/1.000", "Preflop:p0:limp"],
            "action_labels": ["check", "raise"],
            "probabilities": [0.5, 0.5],
        }
        policy = {
            "schema": "hu-tabular-preflop-dcfr-v1",
            "source_policy_sha256": "source",
            "strategies": [copy.deepcopy(root), copy.deepcopy(other)],
            "training_evaluation": {},
        }
        values = {
            "schema": "hu-preflop-canonical-range-action-values-v1",
            "source_policy_sha256": "source",
            "players": [
                [
                    {
                        **root,
                        "player": 0,
                        "reach_probability": 1.0,
                        "policy_probabilities": root["probabilities"],
                        "action_values_bb": [-0.5, 0.5],
                        "action_value_standard_errors_bb": [0.0, 0.0],
                    }
                ],
                [
                    {
                        **other,
                        "player": 1,
                        "reach_probability": 0.5,
                        "policy_probabilities": other["probabilities"],
                        "action_values_bb": [0.0, 1.0],
                        "action_value_standard_errors_bb": [0.0, 0.0],
                    }
                ],
            ],
        }
        candidate, report = improve_policy(
            policy,
            values,
            mix=0.5,
            actors={0},
            root_only=True,
            confidence_z=0.0,
            minimum_advantage_bb=0.0,
            model_version="candidate",
            policy_sha256="parent-hash",
            action_values_sha256="value-hash",
        )
        self.assertEqual(candidate["strategies"][0]["probabilities"], [0.375, 0.625])
        self.assertEqual(candidate["strategies"][1], other)
        self.assertEqual(report["updatedInformationSets"], 1)
        self.assertAlmostEqual(report["predictedFrozenContinuationGainBbPerHand"], 0.375)
        self.assertFalse(report["activationEligible"])

    def test_source_policy_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "different frozen neural policy"):
            improve_policy(
                {
                    "schema": "hu-tabular-preflop-dcfr-v1",
                    "source_policy_sha256": "a",
                    "strategies": [],
                },
                {
                    "schema": "hu-preflop-canonical-range-action-values-v1",
                    "source_policy_sha256": "b",
                    "players": [],
                },
                mix=0.5,
                actors={0},
                root_only=True,
                confidence_z=0.0,
                minimum_advantage_bb=0.0,
                model_version="candidate",
                policy_sha256="parent-hash",
                action_values_sha256="value-hash",
            )


if __name__ == "__main__":
    unittest.main()
