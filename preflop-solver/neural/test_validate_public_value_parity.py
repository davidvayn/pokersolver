import unittest

import numpy as np

import validate_public_value_parity as module


class PublicValueParityTests(unittest.TestCase):
    def test_state_index_list_is_deduplicated_and_preserves_order(self) -> None:
        self.assertEqual(module.selected_state_indices(None, 7), [7])
        self.assertEqual(module.selected_state_indices("4, 2,4", 7), [4, 2])
        with self.assertRaisesRegex(ValueError, "at least one"):
            module.selected_state_indices(",", 7)

    def test_dense_forward_applies_exported_row_major_weights(self) -> None:
        layer = {
            "inputSize": 2,
            "outputSize": 2,
            "activation": "linear",
            "weights": [1.0, 2.0, 3.0, 4.0],
            "biases": [0.5, -0.5],
        }
        result = module.dense_forward(np.asarray([[2.0, 1.0]]), [layer])
        np.testing.assert_allclose(result, [[4.5, 9.5]])


if __name__ == "__main__":
    unittest.main()
