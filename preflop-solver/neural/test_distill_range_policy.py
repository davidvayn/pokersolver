import unittest

import mlx.core as mx
import numpy as np

import distill_range_policy as module
import train_public_value_network as value_features


class RangePolicyDistillationTests(unittest.TestCase):
    def test_reach_priority_cap_preserves_weights_and_street_coverage(self) -> None:
        records = []
        for street_index, street in enumerate(("flop", "turn", "river")):
            for index in range(6):
                records.append(
                    {
                        "weight": float(100 * street_index + index + 1),
                        "state": {
                            "street": street,
                            "board": [street_index, index],
                        },
                        "action_labels": ["check"],
                    }
                )
        selected = module.reach_priority_cap(records, 12)
        self.assertEqual(len(selected), 12)
        self.assertEqual(
            {record["state"]["street"] for record in selected},
            {"flop", "turn", "river"},
        )
        selected_weights = [record["weight"] for record in selected]
        self.assertEqual(
            selected_weights,
            [206.0, 205.0, 204.0, 203.0, 106.0, 105.0, 104.0, 103.0, 6.0, 5.0, 4.0, 3.0],
        )

    def test_reach_sampling_matches_global_combo_weight_objective(self) -> None:
        combo_weights = np.asarray(
            [
                [0.1, 0.3, 0.0],
                [0.6, 0.9, 0.1],
                [0.0, 0.2, 0.2],
            ],
            dtype=np.float32,
        )
        rows = np.asarray([0, 2], dtype=np.int64)
        probabilities = module.reach_sampling_probabilities(combo_weights, rows)
        np.testing.assert_allclose(probabilities, [0.5, 0.5])

        all_probabilities = module.reach_sampling_probabilities(
            combo_weights, np.arange(3)
        )
        np.testing.assert_allclose(all_probabilities, [1 / 6, 2 / 3, 1 / 6])

    def test_training_batch_conditions_combo_weights_on_sampled_node(self) -> None:
        combo_weights = np.zeros((2, module.COMBO_COUNT), dtype=np.float32)
        combo_weights[0, :2] = [0.1, 0.3]
        combo_weights[1, :2] = [0.6, 0.2]
        dataset = module.LoadedDataset(
            path=module.Path("unused"),
            sha256="0" * 64,
            metadata={},
            records=[],
            boards=[],
            actors=np.zeros(2, dtype=np.int32),
            invested=np.zeros((2, 2), dtype=np.float32),
            ranges=np.zeros((2, 2, module.COMBO_COUNT), dtype=np.float32),
            masses=np.zeros((2, 2, module.COMBO_COUNT), dtype=np.float32),
            projection_weights=np.zeros(
                (2, 2, module.COMBO_COUNT), dtype=np.float32
            ),
            actions=np.zeros((2, 1, module.ACTION_FEATURE_COUNT), dtype=np.float32),
            action_masks=np.ones((2, 1), dtype=np.float32),
            source_probabilities=np.zeros(
                (2, module.COMBO_COUNT, 1), dtype=np.float32
            ),
            targets=np.zeros((2, module.COMBO_COUNT, 1), dtype=np.float32),
            action_values=np.zeros(
                (2, module.COMBO_COUNT, 1), dtype=np.float32
            ),
            combo_weights=combo_weights,
            contexts=np.zeros((2, 2, module.CONTEXT_SIZE), dtype=np.float32),
            queries=np.zeros(
                (2, 2, module.COMBO_COUNT, module.QUERY_SIZE), dtype=np.float32
            ),
        )
        conditioned = np.asarray(
            module.batch(
                dataset,
                np.asarray([0, 1]),
                condition_on_node_reach=True,
            )[-2]
        )
        np.testing.assert_allclose(conditioned.sum(axis=1), 1.0)
        np.testing.assert_allclose(conditioned[0, :2], [0.25, 0.75])
        np.testing.assert_allclose(conditioned[1, :2], [0.75, 0.25])

    def test_record_selection_is_source_policy_invariant(self) -> None:
        first = {
            "state": {
                "street": "flop",
                "public_history": ["root"],
                "invested_bb": [1.0000000000000002, 2.0],
            },
            "action_labels": ["check", "bet"],
            "source_policy_probabilities": [0.9, 0.1],
        }
        second = dict(first)
        second["state"] = {
            "street": "flop",
            "public_history": ["root"],
            "invested_bb": [float(module.np.float32(1.0000000000000002)), 2.0],
        }
        second["source_policy_probabilities"] = [0.2, 0.8]
        self.assertEqual(
            module.record_selection_identity(first),
            module.record_selection_identity(second),
        )

    def test_target_identity_is_source_policy_and_f32_tensor_invariant(self) -> None:
        first = {
            "record_type": "range_conditioned_average_strategy",
            "weight": 3.3185869866666664,
            "state": {
                "street": "flop",
                "public_history": ["root"],
                "invested_bb": [1.0000000000000002, 2.0],
            },
            "action_labels": ["check", "bet"],
            "probabilities": [0.7, 0.3],
            "action_values_bb": [-1.7674874000000002e-17, 0.25],
            "source_policy_probabilities": [0.9, 0.1],
        }
        second = module.canonical_training_numbers(first)
        second["source_policy_probabilities"] = [0.2, 0.8]
        self.assertEqual(
            module.target_record_identity(first),
            module.target_record_identity(second),
        )

    def test_network_scores_every_combo_and_masks_padded_actions(self) -> None:
        for architecture in ("compact", "wide"):
            with self.subTest(architecture=architecture):
                model = module.RangeConditionedPolicy(architecture)
                logits = model(
                    mx.zeros((1, 2, module.CONTEXT_SIZE)),
                    mx.zeros((1, 2, module.COMBO_COUNT, module.QUERY_SIZE)),
                    mx.ones((1, 2, module.COMBO_COUNT)),
                    mx.array([1]),
                    mx.zeros((1, 4, module.ACTION_FEATURE_COUNT)),
                    mx.array([[1.0, 1.0, 1.0, 0.0]]),
                )
                mx.eval(logits)
                self.assertEqual(logits.shape, (1, module.COMBO_COUNT, 4))
                values = np.asarray(logits)
                self.assertTrue(np.all(np.isfinite(values)))
                self.assertTrue(np.all(values[:, :, 3] == -1e9))
                probabilities = np.asarray(mx.softmax(logits, axis=2))
                np.testing.assert_allclose(probabilities[:, :, :3].sum(axis=2), 1.0)
                np.testing.assert_allclose(probabilities[:, :, 3], 0.0)

    def test_policy_features_support_flop_turn_and_river(self) -> None:
        for board in (
            np.asarray([0, 5, 10]),
            np.asarray([0, 5, 10, 15]),
            np.asarray([0, 5, 10, 15, 20]),
        ):
            ranges = np.ones((2, module.COMBO_COUNT), dtype=np.float32)
            for player in range(2):
                blocked = np.isin(value_features.COMBO_CARDS[:, 0], board) | np.isin(
                    value_features.COMBO_CARDS[:, 1], board
                )
                ranges[player, blocked] = 0.0
                ranges[player] /= ranges[player].sum()
            masses = np.maximum(
                ranges.sum(axis=1)[:, None]
                - ranges[::-1][:, value_features.COMBO_CONFLICTS].sum(axis=2),
                0.0,
            )
            context, queries = value_features.build_features(
                board,
                1,
                np.asarray([2.0, 2.0]),
                ranges,
                masses,
                value_features.RANGE_POLICY_FEATURE_SCHEMA,
            )
            self.assertEqual(context.shape, (2, module.BASE_CONTEXT_SIZE))
            self.assertEqual(
                queries.shape, (2, module.COMBO_COUNT, module.QUERY_SIZE)
            )
            self.assertTrue(np.all(np.isfinite(context)))
            self.assertTrue(np.all(np.isfinite(queries)))
            legal = ranges[0] > 0
            np.testing.assert_allclose(
                queries[0, legal, 66:75].sum(axis=1), 1.0, atol=1e-6
            )

    def test_policy_state_features_preserve_markov_state_and_trajectory(self) -> None:
        state = {
            "street": "flop",
            "board": [0, 5, 10],
            "actor": 0,
            "invested_bb": [4.0, 4.0],
            "street_invested_bb": [0.0, 0.0],
            "last_full_raise_bb": 1.0,
            "aggressions": 0,
            "checks": 1,
            "raise_reopened": True,
            "trajectory": [
                {
                    "actor": 1,
                    "street": "flop",
                    "kind": "check",
                    "amount_bb": 0.0,
                    "amount_to_bb": None,
                    "pot_after_bb": 8.0,
                }
            ],
        }
        features = module.range_policy_state_features(state, 20.0)
        self.assertEqual(features.shape, (module.PUBLIC_STATE_FEATURE_COUNT,))
        self.assertEqual(features[1], 1.0)
        self.assertEqual(features[4], 1.0)
        self.assertAlmostEqual(float(features[6]), 0.4)
        self.assertEqual(features[19], 1.0)
        self.assertEqual(features[21], 1.0)
        self.assertEqual(features[23], 1.0)
        self.assertEqual(features[27], 1.0)
        self.assertAlmostEqual(float(features[34]), 0.4)

        first_check = features.copy()
        state["checks"] = 0
        state["trajectory"] = []
        root = module.range_policy_state_features(state, 20.0)
        self.assertFalse(np.array_equal(first_check, root))

    def test_zero_initialized_residual_preserves_source_probabilities(self) -> None:
        model = module.RangeConditionedPolicy(
            "compact", "source_bundle_logit_residual"
        )
        source = np.zeros((1, module.COMBO_COUNT, 4), dtype=np.float32)
        source[:, :, :3] = np.asarray([0.7, 0.2, 0.1], dtype=np.float32)
        logits = model(
            mx.zeros((1, 2, module.CONTEXT_SIZE)),
            mx.zeros((1, 2, module.COMBO_COUNT, module.QUERY_SIZE)),
            mx.ones((1, 2, module.COMBO_COUNT)),
            mx.array([1]),
            mx.zeros((1, 4, module.ACTION_FEATURE_COUNT)),
            mx.array([[1.0, 1.0, 1.0, 0.0]]),
            mx.array(source),
        )
        probabilities = np.asarray(mx.softmax(logits, axis=2))
        np.testing.assert_allclose(probabilities[:, :, :3], source[:, :, :3])
        np.testing.assert_allclose(probabilities[:, :, 3], 0.0)


if __name__ == "__main__":
    unittest.main()
