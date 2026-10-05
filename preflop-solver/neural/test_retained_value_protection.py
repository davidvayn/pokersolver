import unittest
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from retained_value_protection import eligible_rows, retention_loss, conditioned_weight, RetainedValueProtection


class RetainedValueProtectionTests(unittest.TestCase):
    def test_only_original_training_positive_joint_states_are_eligible(self):
        groups = np.array([0, 1, 2, 508, 509])
        projection = np.ones((5, 2, 3)); projection[1] = 0.
        np.testing.assert_array_equal(eligible_rows(groups, np.array([0, 1, 3, 4]), projection), [0])
        with self.assertRaisesRegex(ValueError, "eligible"):
            eligible_rows(groups, np.array([1, 3, 4]), projection)
        with self.assertRaises(ValueError): eligible_rows(groups, np.array([0, 0]), projection)

    def test_frozen_output_loss_is_zero_at_reference_and_gradient_restores_drift(self):
        reference = mx.array([[.1, -.1, 0.]])
        weights = mx.array([[1., 2., 0.]])
        scales = mx.array([20.])
        loss = lambda prediction: retention_loss(prediction, reference, weights, scales)
        self.assertEqual(float(loss(reference)), 0.)
        np.testing.assert_array_equal(np.asarray(mx.grad(loss)(reference)), np.zeros((1, 3)))
        drifted = reference + mx.array([[.01, -.02, 10.]])
        gradient = np.asarray(mx.grad(loss)(drifted))
        self.assertGreater(gradient[0, 0], 0)
        self.assertLess(gradient[0, 1], 0)
        self.assertEqual(gradient[0, 2], 0.)

    def test_coefficient_conditioning_uses_finite_training_norms_and_cap(self):
        self.assertEqual(conditioned_weight(1., .25), 2.)
        self.assertEqual(conditioned_weight(1., .001), 32.)
        for args in ((1., 0.), (float("nan"), 1.), (-1., 1.)):
            with self.assertRaises(ValueError): conditioned_weight(*args)

    def test_real_gradient_adapter_preserves_cadence_and_frozen_reference(self):
        class ConstantNetwork(nn.Module):
            def __init__(self, value): super().__init__(); self.weight = mx.array(value)
            def __call__(self, context, query, projection, scales):
                return mx.broadcast_to(self.weight, (len(context), 2652))
        class BaseObjective:
            def accumulate(self, model, gradients, loss_fn, step): return gradients
            def report(self): return {"bundleUpdates": 1}
        projection = np.ones((5, 2, 1326), dtype=np.float32); projection[1] = 0.
        dataset = SimpleNamespace(groups=np.array([0, 1, 2, 508, 509]), projection_weights=projection,
            targets=np.zeros((5, 2652), dtype=np.float32), target_scales=np.full(5, 20., dtype=np.float32),
            invested=np.full((5, 2), 7.5))
        reference, current = ConstantNetwork(.1), ConstantNetwork(.2)
        objective = RetainedValueProtection(BaseObjective(), dataset,
            np.zeros((5, 2, 1), dtype=np.float32), np.zeros((5, 2, 1326, 1), dtype=np.float32),
            np.array([0, 1, 3, 4]), reference, 2., 10601)
        zero = {"weight": mx.array(0.)}
        self.assertIs(objective.accumulate(current, zero, None, 1), zero)
        reference.weight = mx.array(.9)
        result = objective.accumulate(current, zero, None, 4)
        self.assertGreater(float(result["weight"]), 0.)
        np.testing.assert_allclose(objective.reference[0], .1)
        report = objective.report()["retainedValueProtection"]
        self.assertEqual((report["updates"], report["draws"], report["trainingRows"]), (1, 8, [0]))
        self.assertEqual(objective.report()["bundleUpdates"], 1)


if __name__ == "__main__": unittest.main()
