"""Real optimizer and unequal-mass regressions for bounded cash fitting."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from cash_feature_cache import CashFeatureCache
from cash_profiles import profile_rules, rules_digest
import train_cash_value_network as training


class CashGradientAccumulationTests(unittest.TestCase):
    def test_invalid_batch_budget_fails_before_reading_data(self):
        for invalid in (0, -1, True, 1.5, "32", 10001):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "gradient batch size"):
                training.run(Path("must-not-read"), Path("must-not-create"), 7101, 1,
                             gradient_batch_size=invalid)

    def test_global_normalization_preserves_unequal_mass_objective_and_gradient(self):
        class Scalar(nn.Module):
            def __init__(self):
                super().__init__()
                self.offset = mx.array([.3])

        with mx.stream(mx.cpu):
            model = Scalar()
            targets = mx.array([[1., 2.], [-1., 4.], [5., -3.]])
            weights = mx.array([[.1, .2], [20., 1.], [.4, .7]])
            total = float(mx.sum(weights).item())
            rows = mx.arange(3)
            for regression in ("mse", "huber"):
                def full(current, selected):
                    return training.cash_regression_losses(
                        current.offset-targets[selected], weights[selected], regression, .5)[1]
                def additive(current, selected):
                    return training.cash_regression_losses(
                        current.offset-targets[selected], weights[selected], regression, .5,
                        normalization_mass=total)[1]
                full_value, full_gradient = nn.value_and_grad(model, full)(model, rows)
                value, gradient = training.cash_accumulated_value_and_grad(
                    model, nn.value_and_grad(model, additive), rows, 2)
                self.assertAlmostEqual(float(value.item()), float(full_value.item()), places=5)
                np.testing.assert_allclose(np.array(gradient["offset"]),
                                           np.array(full_gradient["offset"]), rtol=2e-6, atol=2e-6)
                # Averaging independently normalized batch losses is wrong
                # when their reach/curriculum weight masses differ.
                wrong = (full(model, rows[:2])+full(model, rows[2:]))/2
                self.assertGreater(abs(float(wrong.item())-float(full_value.item())), .1)

    def test_actual_fit_accumulates_before_adam_and_bounds_all_forwards(self):
        dimensions = training.OwnComboValueNetwork()
        contexts = np.zeros((7, 2, dimensions.context_tower.layers[0].weight.shape[1]), np.float32)
        queries = np.zeros((7, 2, 1326, dimensions.query_tower.layers[0].weight.shape[1]), np.float32)
        contexts[:, :, 0] = np.arange(7)[:, None]/10
        queries[:, :, :, 94] = .5
        masses = np.ones((7, 2, 1326), np.float32)/1326
        scales = np.array([1., 2., 3., 2., 1., 2., 1.], np.float32)
        targets = np.array([.1, -.3, .7, .2, -.1, .3, .1], np.float32)[:, None]*np.ones((7, 2652), np.float32)
        loss_weights = np.ones((7, 2652), np.float32)/1326
        loss_weights *= np.array([1., 10., .3, 1., 1., 3., 1.], np.float32)[:, None]
        arrays = (contexts, queries, masses, scales, np.zeros((7, 2, 1326), np.float32),
                  np.ones((7, 1326), np.float32), targets, loss_weights)
        rules = profile_rules("nl25")
        ranges = np.full((2, 1326), 1/1326).tolist()
        source = dict(game=dict(cash_rules=rules), rules_sha256=rules_digest(rules), labels=[
            dict(input=dict(state=dict(ranges=ranges)),
                 counterfactual_values_bb=np.full((2, 1326), float(targets[i, 0]*scales[i])).tolist(),
                 metrics=dict(cash=dict(expected_house_rake_bb=.1))) for i in range(7)])
        split = (np.arange(3), np.array([3,4,5]), np.array([6]))
        with tempfile.TemporaryDirectory() as directory, mx.stream(mx.cpu):
            root = Path(directory)
            dataset = root/"dataset.json"
            dataset.write_text(json.dumps(source))
            with (patch.object(training, "_FEATURE_CACHE", CashFeatureCache()),
                  patch.object(training, "split_cash_families", return_value=split),
                  patch.object(training, "feature_arrays", return_value=arrays),
                  patch.object(training, "context_loss_multipliers", return_value=np.array([1., .25, .1, 1., 1., 1., 1.], np.float32))):
                knobs = dict(accounting_loss_weight=.3, profile_value_loss_weight=.2,
                             training_output_mode="raw")
                full = training.run(dataset, root/"full", 7101, 3, **knobs)
                observed = []
                original = training.OwnComboValueNetwork.raw_values
                def prediction(current, context, *inputs):
                    observed.append(context.shape[0])
                    return original(current, context, *inputs)
                with patch.object(training.OwnComboValueNetwork, "raw_values", new=prediction):
                    batched = training.run(dataset, root/"batched", 7101, 3,
                                           gradient_batch_size=2, **knobs)
                self.assertTrue(observed)
                self.assertLessEqual(max(observed), 2)
                self.assertEqual(full["selected_step"], batched["selected_step"])
                for metric in ("training_initial_mse_bb", "training_initial_objective_bb_squared", "heldout_rmse_bb"):
                    self.assertAlmostEqual(full[metric], batched[metric], places=5)
                full_weights = json.loads((root/"full/value-network.json").read_bytes())
                batched_weights = json.loads((root/"batched/value-network.json").read_bytes())
                for tower in ("contextTower", "queryTower", "head"):
                    self.assertEqual(len(full_weights[tower]), len(batched_weights[tower]))
                    for left,right in zip(full_weights[tower], batched_weights[tower]):
                        self.assertEqual(left.keys(), right.keys())
                        for key,value in left.items():
                            if isinstance(value, str):
                                self.assertEqual(value, right[key])
                            else:
                                np.testing.assert_allclose(value, right[key], rtol=2e-5, atol=2e-5)
                self.assertNotIn("gradientBatchSize", full_weights)
                self.assertEqual(batched_weights["gradientBatchSize"], 2)
                self.assertEqual(batched["gradient_batching"]["microbatches_per_update"], 2)
                self.assertEqual(batched["gradient_batching"]["contexts_per_optimizer_update"], 3)
                self.assertFalse(batched["active"])

    def test_invalid_global_mass_and_empty_accumulation_fail_closed(self):
        for mass in (0, -1, True, float("inf"), float("nan")):
            with self.assertRaisesRegex(ValueError, "normalization mass"):
                training.cash_regression_losses(mx.array([1.]), mx.array([1.]),
                                                normalization_mass=mass)
        with self.assertRaisesRegex(ValueError, "nonempty rows"):
            training.cash_accumulated_value_and_grad(None, None, mx.array([], dtype=mx.int32), 2)

    def test_precision_request_is_reported_without_relabeling_serving_or_legacy_exports(self):
        rules = profile_rules("nl25")
        source = dict(game=dict(cash_rules=rules), rules_sha256=rules_digest(rules))
        with tempfile.TemporaryDirectory() as directory, mx.stream(mx.cpu):
            for flag in (None, "1", "0"):
                environment = dict(os.environ)
                environment.pop("MLX_ENABLE_TF32", None)
                if flag is not None:
                    environment["MLX_ENABLE_TF32"] = flag
                with patch.dict(os.environ, environment, clear=True):
                    precision = training.cash_training_precision()
                    self.assertEqual(precision["mlx_enable_tf32"], flag)
                    self.assertTrue(precision["mlx_version"])
                    path = Path(directory)/f"network-{flag}.json"
                    training.export_cash_model(training.OwnComboValueNetwork(),path,7101,source,"a"*64)
                    network = json.loads(path.read_bytes())
                    self.assertEqual(network["projection"], "independent-full-stack-clip-and-board-mask-no-zero-sum")
                    if flag == "0":
                        self.assertEqual(network["trainingMatmulPrecisionRequest"], "full-float32")
                        self.assertEqual(network["trainingMlxVersion"], precision["mlx_version"])
                    else:
                        self.assertNotIn("trainingMatmulPrecisionRequest", network)

    def test_invalid_precision_environment_fails_before_reading_data(self):
        for flag in ("false", "", "2", "0 "):
            with patch.dict(os.environ, MLX_ENABLE_TF32=flag):
                with self.assertRaisesRegex(ValueError, "MLX_ENABLE_TF32"):
                    training.run(Path("must-not-read"), Path("must-not-create"), 7101, 1)


if __name__ == "__main__":
    unittest.main()
