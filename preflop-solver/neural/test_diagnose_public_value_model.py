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


if __name__ == "__main__":
    unittest.main()
