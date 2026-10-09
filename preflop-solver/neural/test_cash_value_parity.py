import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from cash_profiles import profile_rules, rules_digest
from native_value_dataset import COMBOS
from train_cash_value_network import OwnComboValueNetwork, export_cash_model
from validate_cash_value_parity import cash_architecture, cash_feature_schema, mlx_predictions, python_predictions
from train_public_value_network import FEATURE_SCHEMA_BOARD_RELATIVE, FEATURE_SCHEMA_EXACT_RUNOUT


class CashValueParityTests(unittest.TestCase):
    def test_feature_schema_is_required_and_cannot_silently_fall_back(self):
        for invalid in (None, "unknown", "rank-suit-invariant-combo-query-v1"):
            with self.assertRaisesRegex(ValueError,"feature schema"):
                cash_feature_schema({"featureSchema":invalid})
        for supported in (FEATURE_SCHEMA_BOARD_RELATIVE, FEATURE_SCHEMA_EXACT_RUNOUT):
            self.assertEqual(cash_feature_schema({"featureSchema":supported}),supported)

    def test_architecture_and_cash_schema_cannot_be_relabelled(self):
        for payload in (
            {"schema": "hu-cash-public-belief-combo-value-network-v2", "architecture": "wide-pooled"},
            {"schema": "hu-cash-public-belief-combo-value-network-v3", "architecture": "wide"},
            {"schema": "hu-public-belief-combo-value-network-v5", "architecture": "wide-pooled"},
            {"schema": "hu-cash-public-belief-combo-value-network-v3", "architecture": "unknown"},
            {"schema": "hu-cash-public-belief-combo-value-network-v4", "architecture": "wide-blocker-pooled",
             "predictionContract": "cash-turn-start-cfv-full-stack-v1"},
            {"schema": "hu-cash-public-belief-combo-value-network-v5", "architecture": "wide-baseline-conditioned",
             "predictionContract": "cash-turn-start-cfv-full-stack-v1"},
        ):
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, "architecture"):
                cash_architecture(payload)

    def test_numpy_and_mlx_preserve_pooled_ranges_and_zero_own_reach_queries(self):
        rules = profile_rules("nl25")
        source = dict(game=dict(cash_rules=rules), rules_sha256=rules_digest(rules))
        with tempfile.TemporaryDirectory() as directory:
            for architecture in ("compact", "wide", "wide-pooled", "wide-blocker-pooled", "wide-baseline-conditioned"):
                with self.subTest(architecture=architecture), mx.stream(mx.cpu):
                    model = OwnComboValueNetwork(architecture)
                    for tower in (model.context_tower, model.query_tower, model.head):
                        for layer in tower.layers:
                            if hasattr(layer, "weight"):
                                layer.weight = mx.zeros_like(layer.weight)
                                layer.bias = mx.zeros_like(layer.bias)
                    weight = np.array(model.query_tower.layers[0].weight)
                    weight[0, 94] = 1.
                    model.query_tower.layers[0].weight = mx.array(weight)
                    weight = np.array(model.query_tower.layers[2].weight)
                    weight[0, 0] = 1.
                    model.query_tower.layers[2].weight = mx.array(weight)
                    embedding = weight.shape[0]
                    weight = np.array(model.head.layers[0].weight)
                    if architecture in ("wide-pooled","wide-blocker-pooled","wide-baseline-conditioned"):
                        weight[0, embedding] = .2
                        weight[0, embedding * 2] = .1
                        weight[0, embedding * 3] = .05
                        if architecture == "wide-baseline-conditioned":
                            weight[0, -1] = .1
                    else:
                        weight[0, embedding] = .05
                    model.head.layers[0].weight = mx.array(weight)
                    weight = np.array(model.head.layers[2].weight)
                    weight[0, 0] = 1.
                    model.head.layers[2].weight = mx.array(weight)
                    path = Path(directory) / f"{architecture}.json"
                    export_cash_model(model, path, 7101, source, "a" * 64)
                    payload = json.loads(path.read_text())
                    context = np.zeros((1, 2, payload["contextTower"][0]["inputSize"]), np.float32)
                    queries = np.zeros((1, 2, 1326, payload["queryTower"][0]["inputSize"]), np.float32)
                    queries[0, 0, 1:4, 94] = [.2, .8, .9]
                    queries[0, 1, 1:4, 94] = [.4, .6, .7]
                    reaches = np.zeros((1, 2, 1326), np.float32)
                    reaches[0, 0, 1:3] = [1., 3.]
                    reaches[0, 1, 1:3] = [3., 1.]
                    raw_ranges = reaches / reaches.sum(axis=2,keepdims=True)
                    source["labels"] = [{"input":{"state":{"ranges":raw_ranges[0].tolist()}}}]
                    scales = np.array([4.], np.float32)
                    baselines = np.zeros((1, 2, 1326), np.float32)
                    baselines[:, 0] = 2.
                    baselines[:, 1] = -3.
                    legal = np.ones((1, 1326), np.float32)
                    legal[:, 0] = 0.
                    arrays = (context, queries, reaches, scales, baselines, legal, None, None)
                    with patch("validate_cash_value_parity.feature_arrays", return_value=arrays):
                        dense = python_predictions(source, payload)
                        cpu = mlx_predictions(source, payload, mx.cpu)
                    expected = baselines + .05 * scales[:, None, None] * queries[:, :, :, 94]
                    if architecture in ("wide-pooled","wide-blocker-pooled","wide-baseline-conditioned"):
                        own = np.array([.65, .45], np.float32)
                        expected += scales[:, None, None] * (.2 * own)[None, :, None]
                        if architecture == "wide-blocker-pooled":
                            for player in (0,1):
                                for query,(a,b) in enumerate(COMBOS):
                                    weights = raw_ranges[0,1-player,1:3] * ~((COMBOS[1:3] == a).any(axis=1) | (COMBOS[1:3] == b).any(axis=1))
                                    if weights.sum() > 0:
                                        expected[0,player,query] += .4 * (weights @ queries[0,1-player,1:3,94]) / weights.sum()
                        else:
                            expected += scales[:,None,None] * (.1 * own[::-1])[None,:,None]
                    if architecture == "wide-baseline-conditioned":
                        expected += scales[:,None,None] * .1 * baselines / 20.
                    expected *= legal[:, None, :]
                    np.testing.assert_allclose(dense, expected, atol=2e-6, rtol=0)
                    np.testing.assert_allclose(cpu, expected, atol=2e-6, rtol=0)
                    self.assertNotEqual(float(dense[0, 0, 3]), 0.)  # Own reach is zero.
                    self.assertEqual(float(dense[0, 0, 0]), 0.)  # Board-blocked hand.


if __name__ == "__main__":
    unittest.main()
