import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

import validate_public_value_parity as module
from test_train_public_value_network import PublicValueNetworkTests


class PublicValueParityTests(unittest.TestCase):
    def test_cached_features_preserve_independent_prediction_and_readonly_inputs(self):
        training = module.training
        dataset = PublicValueNetworkTests.synthetic_dataset([0, 5, 10, 15], "a" * 64)
        features = training.build_features(dataset.boards[0], int(dataset.actors[0]),
            dataset.invested[0], dataset.ranges[0], dataset.masses[0], training.FEATURE_SCHEMA)
        features[0].flags.writeable = False; features[1].flags.writeable = False
        originals = tuple(f.copy() for f in features)
        with tempfile.TemporaryDirectory() as directory:
            for use_ranges in [True, False]:
                current = training.SharedComboValueNetwork(use_ranges, "compact", "payoff-exposure", training.FEATURE_SCHEMA)
                path = Path(directory) / "model.json"
                training.export_model(current, path, 1, "a" * 64, training.native_values.SCHEMA,
                    "research_only", "b" * 64, "payoff-exposure")
                model = json.loads(path.read_text())
                np.testing.assert_array_equal(module.python_prediction(dataset, model, 0),
                    module.python_prediction(dataset, model, 0, features))
        for original, cached in zip(originals, features): np.testing.assert_array_equal(original, cached)

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
