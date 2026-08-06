import unittest

import numpy as np

import diagnose_public_value_model as module


class PublicValueDiagnosticsTests(unittest.TestCase):
    def test_weighted_error_uses_combo_reach(self) -> None:
        truth = np.asarray([[0.0, 0.0]])
        prediction = np.asarray([[1.0, 10.0]])
        weights = np.asarray([[1.0, 0.0]])
        rmse, mae = module.weighted_error_bb(truth, prediction, weights)
        self.assertEqual(rmse, 1.0)
        self.assertEqual(mae, 1.0)

    def test_player_signed_error_does_not_cancel_zero_sum_bias(self) -> None:
        truth = np.zeros((1, 2, module.training.COMBO_COUNT))
        prediction = np.zeros_like(truth)
        prediction[:, 0, :] = 0.4
        prediction[:, 1, :] = -0.2
        bias = module.player_weighted_signed_error_bb(
            truth, prediction, np.ones_like(truth)
        )
        self.assertAlmostEqual(bias[0], 0.4)
        self.assertAlmostEqual(bias[1], -0.2)

    def test_selected_indices_reject_duplicates(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            module.selected_indices("1,2,1")

    def test_resolver_reach_weights_states_without_changing_combo_weights(self) -> None:
        truth = np.asarray([[[0.0]], [[0.0]]])
        prediction = np.asarray([[[1.0]], [[3.0]]])
        weights = np.ones_like(truth)
        rmse, mae = module.resolver_reach_weighted_error_bb(
            truth,
            prediction,
            weights,
            np.asarray([0.9, 0.1]),
        )
        self.assertAlmostEqual(rmse, (1.8**0.5))
        self.assertAlmostEqual(mae, 1.2)

    def test_sufficient_statistics_combine_shards_exactly(self) -> None:
        truth = np.zeros((4, 1))
        prediction = np.asarray([[1.0], [2.0], [3.0], [4.0]])
        weights = np.asarray([[1.0], [2.0], [3.0], [4.0]])
        whole = module.weighted_error_sufficient_statistics(
            truth, prediction, weights
        )
        shards = [
            module.weighted_error_sufficient_statistics(
                truth[offset : offset + 2],
                prediction[offset : offset + 2],
                weights[offset : offset + 2],
            )
            for offset in (0, 2)
        ]
        for key in whole:
            self.assertAlmostEqual(whole[key], sum(shard[key] for shard in shards))
        combined_rmse = (
            sum(shard["weightedSquaredErrorBb2Sum"] for shard in shards)
            / sum(shard["weightMass"] for shard in shards)
        ) ** 0.5
        direct_rmse, _ = module.weighted_error_bb(truth, prediction, weights)
        self.assertAlmostEqual(combined_rmse, direct_rmse)


if __name__ == "__main__":
    unittest.main()
