import unittest

import mlx.core as mx
import numpy as np

import distill_range_policy as module
import train_public_value_network as value_features


class RangePolicyDistillationTests(unittest.TestCase):
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
            self.assertEqual(context.shape, (2, module.CONTEXT_SIZE))
            self.assertEqual(
                queries.shape, (2, module.COMBO_COUNT, module.QUERY_SIZE)
            )
            self.assertTrue(np.all(np.isfinite(context)))
            self.assertTrue(np.all(np.isfinite(queries)))
            legal = ranges[0] > 0
            np.testing.assert_allclose(
                queries[0, legal, 66:75].sum(axis=1), 1.0, atol=1e-6
            )

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
