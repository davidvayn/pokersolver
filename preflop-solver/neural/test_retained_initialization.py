import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

import train_public_value_network as training
import native_value_dataset as native
from retained_initialization import import_retained_weights, native_import_prediction
from validate_public_value_parity import python_prediction


class RetainedInitializationTests(unittest.TestCase):
    def setUp(self):
        self.source = training.SharedComboValueNetwork(True, "wide", "payoff-exposure",
            training.FEATURE_SCHEMA_EXACT_RUNOUT)
        mx.eval(self.source.parameters())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            training.export_model(self.source, path, 10601, "a" * 64,
                "hu-native-turn-cfv-dataset-v1", "rejected", "b" * 64, "payoff-exposure")
            self.payload = json.loads(path.read_text())
        self.destination = training.SharedComboValueNetwork(True, "wide", "payoff-exposure",
            training.FEATURE_SCHEMA_EXACT_RUNOUT)
        mx.eval(self.destination.parameters())

    def test_exact_weight_and_forward_roundtrip(self):
        receipt = import_retained_weights(self.destination, self.payload, 10601)
        self.assertEqual(receipt["optimizerState"], "fresh_not_imported")
        rng = np.random.default_rng(11)
        context = mx.array(rng.random((2, 2, 417), dtype=np.float32))
        query = mx.array(rng.random((2, 2, 1326, 124), dtype=np.float32))
        projection = mx.ones((2, 2, 1326))
        scales = mx.ones(2)
        np.testing.assert_array_equal(np.asarray(self.source(context, query, projection, scales)),
            np.asarray(self.destination(context, query, projection, scales)))
        for name, tower in (("contextTower", self.destination.context_tower),
                            ("queryTower", self.destination.query_tower), ("head", self.destination.head)):
            self.assertEqual(training.tower_payload(tower, "relu", "linear" if name == "head" else "relu"),
                self.payload[name])

    def test_invalid_contract_seed_activation_shape_or_nonfinite_fails_before_mutation(self):
        before = training.tower_payload(self.destination.context_tower, "relu", "relu")
        for field, value in (("schema", "unknown"), ("seed", 10602), ("usesExactRanges", False),
                             ("targetScaleBb", 40), ("predictionContract", "other"),
                             ("featureSchema", "v1"), ("valueNormalization", "pot")):
            payload = copy.deepcopy(self.payload); payload[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                import_retained_weights(self.destination, payload, 10601)
        for edit in ("activation", "shape", "nan", "missing"):
            payload = copy.deepcopy(self.payload)
            if edit == "activation": payload["head"][-1]["activation"] = "relu"
            elif edit == "shape": payload["head"][-1]["weights"].pop()
            elif edit == "nan": payload["head"][-1]["biases"][0] = float("nan")
            else: payload.pop("queryTower")
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                import_retained_weights(self.destination, payload, 10601)
            self.assertEqual(before, training.tower_payload(self.destination.context_tower, "relu", "relu"))

    def test_serving_wrapper_matches_independent_numpy_even_when_clipping(self):
        import_retained_weights(self.destination, self.payload, 10601)
        # Force clipping to lock down the actual failed preflight seam. Train
        # zero-sum-then-clip is not interchangeable with serving clip-then-sum.
        self.payload["head"][-1]["biases"][0] = 2.
        import_retained_weights(self.destination, self.payload, 10601)
        rng = np.random.default_rng(17)
        context = rng.random((1, 2, 417), dtype=np.float32)
        context[:, :, 19:21] = .375
        query = rng.random((1, 2, 1326, 124), dtype=np.float32)
        board = np.array([[0, 5, 10, 15]])
        ranges = np.tile(native.legal_combos(board[0]), (1, 2, 1)).astype(np.float32)
        data = SimpleNamespace(boards=board, ranges=ranges, invested=np.array([[7.5, 7.5]]),
            projection_weights=ranges * native.compatible_masses(ranges[0])[None])
        expected = python_prediction(data, self.payload, 0, (context[0], query[0]))
        got = native_import_prediction(self.destination, context, query, np.array([20.]), board, ranges)
        np.testing.assert_allclose(got[0], expected, atol=1e-5, rtol=0)


if __name__ == "__main__": unittest.main()
