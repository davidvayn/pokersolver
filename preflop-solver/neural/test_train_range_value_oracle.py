import unittest

import mlx.core as mx
import numpy as np

from train_range_value_oracle import (
    CLASS_INDEX,
    CacheArrays,
    SampleSelection,
    ValueOracle,
    build_grouped_targets,
    compatible_class_counts,
    exact_class_indices,
    feature_batch,
    passes_freeze_criteria,
    pearson_correlation,
    parse_history_state,
    range_likelihoods,
)


class RangeValueOracleTests(unittest.TestCase):
    def test_freeze_criteria_support_strict_improvement_and_noninferiority(self):
        metrics = {
            "range": {
                "crossSeedPredictionCorrelation": 0.97,
                "ensemble": {"groupedRmseBb": 0.8},
            },
            "pairedComparison": {
                "groupedRmseRelativeImprovement": -0.035,
                "rangeWinsEverySeed": False,
            },
        }
        self.assertFalse(passes_freeze_criteria(metrics, 0.01, 0.95, 1.0))
        self.assertTrue(passes_freeze_criteria(metrics, -0.05, 0.95, 1.0))
        self.assertFalse(passes_freeze_criteria(metrics, -0.05, 0.98, 1.0))
        self.assertFalse(passes_freeze_criteria(metrics, -0.05, 0.95, 0.7))

    def test_range_tower_has_bounded_value_head(self):
        model = ValueOracle(400, (32, 16), "range_tower", True)
        values = np.asarray(model(mx.ones((3, 400))))
        self.assertEqual(values.shape, (3, 1))
        self.assertTrue(np.all(np.abs(values) <= 1.0))

    def test_constant_prediction_correlation_is_finite(self):
        self.assertEqual(
            pearson_correlation(np.zeros(4), np.asarray([1.0, 2.0, 3.0, 4.0])),
            0.0,
        )

    def test_exact_class_counts_apply_card_removal(self):
        unblocked = compatible_class_counts(np.empty((1, 0), dtype=np.uint8))[0]
        self.assertEqual(unblocked[CLASS_INDEX["AA"]], 6)
        self.assertEqual(unblocked[CLASS_INDEX["AKs"]], 4)
        self.assertEqual(unblocked[CLASS_INDEX["AKo"]], 12)

        ace_of_first_suit = np.asarray([[12 * 4]], dtype=np.uint8)
        blocked = compatible_class_counts(ace_of_first_suit)[0]
        self.assertEqual(blocked[CLASS_INDEX["AA"]], 3)
        self.assertEqual(blocked[CLASS_INDEX["AKs"]], 3)
        self.assertEqual(blocked[CLASS_INDEX["AKo"]], 9)

    def test_exact_hole_cards_map_to_holdem_classes(self):
        holes = np.asarray(
            [
                [[12 * 4, 11 * 4], [12 * 4, 11 * 4 + 1]],
                [[7 * 4, 7 * 4 + 3], [0, 1 * 4]],
            ],
            dtype=np.uint8,
        )
        actual = exact_class_indices(holes)
        self.assertEqual(actual[0, 0], CLASS_INDEX["AKs"])
        self.assertEqual(actual[0, 1], CLASS_INDEX["AKo"])
        self.assertEqual(actual[1, 0], CLASS_INDEX["99"])
        self.assertEqual(actual[1, 1], CLASS_INDEX["32s"])

    def test_history_state_excludes_preflop_all_ins(self):
        meaningful, scalars = parse_history_state(
            [
                "blinds:0.500/1.000",
                "Preflop:p0:raise_all_in_to_20.000bb",
                "Preflop:p1:call_all_in",
                "deal:Flop",
            ]
        )
        self.assertFalse(meaningful)
        self.assertEqual(scalars[1], 0.0)
        self.assertEqual(scalars[2], 0.0)

        meaningful, _ = parse_history_state(
            [
                "blinds:0.500/1.000",
                "Preflop:p0:raise_to_3.000bb",
                "Preflop:p1:call",
                "deal:Flop",
            ]
        )
        self.assertTrue(meaningful)

    def test_range_likelihood_replays_public_actions_by_hand_class(self):
        history = [
            "blinds:0.500/1.000",
            "Preflop:p0:limp",
            "Preflop:p1:check",
            "deal:Flop",
        ]
        policy = {}
        from train_range_value_oracle import CLASS_LABELS

        for class_index, label in enumerate(CLASS_LABELS):
            limp = 0.25 if class_index == CLASS_INDEX["AA"] else 0.5
            policy[f"p0|{label}|blinds:0.500/1.000"] = {"limp": limp}
            policy[
                f"p1|{label}|blinds:0.500/1.000/Preflop:p0:limp"
            ] = {"check": 0.75}
        likelihoods = range_likelihoods([history], policy)
        self.assertEqual(likelihoods[0, 0, CLASS_INDEX["AA"]], 0.25)
        self.assertEqual(likelihoods[0, 0, CLASS_INDEX["AKs"]], 0.5)
        self.assertEqual(likelihoods[0, 1, CLASS_INDEX["AA"]], 0.75)

    def test_features_never_depend_on_actual_opponent_cards(self):
        history = [
            "blinds:0.500/1.000",
            "Preflop:p0:limp",
            "Preflop:p1:check",
            "deal:Flop",
        ]
        meaningful, scalars = parse_history_state(history)
        cache = CacheArrays(
            holes=np.asarray(
                [
                    [[48, 45], [44, 41]],
                    [[48, 45], [36, 33]],
                ],
                dtype=np.uint8,
            ),
            flops=np.asarray([[0, 5, 10], [0, 5, 10]], dtype=np.uint8),
            targets=np.asarray([[1.0], [-1.0]], dtype=np.float32),
            standard_errors=np.asarray([[0.1], [0.1]], dtype=np.float32),
            history_keys=["1"],
            histories=[history],
            meaningful_histories=np.asarray([meaningful]),
            history_scalars=np.stack([scalars]),
            source_sha256="0" * 64,
        )
        selection = SampleSelection(
            deals=np.asarray([0, 1], dtype=np.int32),
            histories=np.asarray([0, 0], dtype=np.int16),
            actors=np.asarray([0, 0], dtype=np.int8),
        )
        likelihoods = np.ones((1, 2, 169), dtype=np.float64)
        features, targets, _, _ = feature_batch(cache, likelihoods, selection, True)
        np.testing.assert_array_equal(features[0], features[1])
        self.assertNotEqual(targets[0], targets[1])

        ablated, _, _, _ = feature_batch(cache, likelihoods, selection, False)
        np.testing.assert_array_equal(ablated[:, -338:], 0.0)

        grouped = build_grouped_targets(cache)
        _, grouped_values, grouped_se, _ = feature_batch(
            cache, likelihoods, selection, True, grouped
        )
        np.testing.assert_allclose(grouped_values, 0.0)
        np.testing.assert_allclose(grouped_se, 1.0)


if __name__ == "__main__":
    unittest.main()
