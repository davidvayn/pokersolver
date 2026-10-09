import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import train_public_value_network as training
import native_value_dataset as native
from retained_initialization import RANGE_AUGMENTATION, import_retained_weights, native_import_prediction
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

    def test_function_preserving_range_augmentation_has_trainable_new_columns(self):
        pooled = training.SharedComboValueNetwork(True, "wide-pooled", "payoff-exposure",
            training.FEATURE_SCHEMA_EXACT_RUNOUT)
        with self.assertRaises(ValueError):
            import_retained_weights(pooled, self.payload, 10601)
        with self.assertRaises(ValueError):
            import_retained_weights(self.destination, self.payload, 10601, RANGE_AUGMENTATION)
        receipt = import_retained_weights(pooled, self.payload, 10601, RANGE_AUGMENTATION)
        self.assertEqual(receipt["transform"], RANGE_AUGMENTATION)
        source = np.asarray(self.source.head.layers[0].weight)
        target = np.asarray(pooled.head.layers[0].weight)
        np.testing.assert_array_equal(target[:, :64], source[:, :64])
        np.testing.assert_array_equal(target[:, 192:], source[:, 64:])
        np.testing.assert_array_equal(target[:, 64:192], 0.)
        rng = np.random.default_rng(27)
        context = mx.array(rng.random((1, 2, 417), dtype=np.float32))
        query = mx.array(rng.random((1, 2, 1326, 124), dtype=np.float32))
        reach = mx.array(rng.random((1, 2, 1326), dtype=np.float32))
        scales = mx.array([20.])
        with mx.stream(mx.cpu):
            np.testing.assert_allclose(np.asarray(pooled.raw_values(context, query, reach, scales)),
                np.asarray(self.source.raw_values(context, query, reach, scales)), atol=1e-6, rtol=0)
            np.testing.assert_allclose(np.asarray(pooled(context, query, reach, scales)),
                np.asarray(self.source(context, query, reach, scales)), atol=1e-6, rtol=0)
            _, gradient = nn.value_and_grad(pooled, lambda model: mx.sum(
                model.raw_values(context, query, reach, scales)))(pooled)
            new_columns = np.asarray(gradient["head"]["layers"][0]["weight"])[:, 64:192]
        self.assertGreater(float(np.linalg.norm(new_columns)), 0)
        self.assertTrue(np.isfinite(new_columns).all())
        before = training.tower_payload(pooled.context_tower, "relu", "relu")
        bad = copy.deepcopy(self.payload); bad["head"][-1]["biases"][0] = float("nan")
        with self.assertRaises(ValueError):
            import_retained_weights(pooled, bad, 10601, RANGE_AUGMENTATION)
        self.assertEqual(before, training.tower_payload(pooled.context_tower, "relu", "relu"))

    def test_pooled_import_serving_wrapper_matches_numpy_with_active_pool_columns(self):
        pooled = training.SharedComboValueNetwork(True, "wide-pooled", "payoff-exposure",
            training.FEATURE_SCHEMA_EXACT_RUNOUT)
        import_retained_weights(pooled, self.payload, 10601, RANGE_AUGMENTATION)
        weight = np.asarray(pooled.head.layers[0].weight).copy()
        weight[:, 64:192] = .001
        pooled.head.layers[0].weight = mx.array(weight)
        rng = np.random.default_rng(29)
        context = rng.random((1, 2, 417), dtype=np.float32); context[:, :, 19:21] = .375
        query = rng.random((1, 2, 1326, 124), dtype=np.float32)
        board = np.array([[0, 5, 10, 15]])
        ranges = rng.random((1, 2, 1326), dtype=np.float32) * native.legal_combos(board[0])
        data = SimpleNamespace(boards=board, ranges=ranges, invested=np.array([[7.5, 7.5]]),
            projection_weights=(ranges * native.compatible_masses(ranges[0])[None]).astype(np.float32))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pooled.json"
            training.export_model(pooled, path, 10601, "a"*64, native.SCHEMA,
                "research_only", "b"*64, "payoff-exposure")
            expected = python_prediction(data, json.loads(path.read_text()), 0, (context[0], query[0]))
        got = native_import_prediction(pooled, context, query, np.array([20.]), board, ranges)
        np.testing.assert_allclose(got[0], expected, atol=1e-5, rtol=0)


if __name__ == "__main__": unittest.main()
