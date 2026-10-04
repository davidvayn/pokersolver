from types import SimpleNamespace
import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np

from action_contrast_dataset import AffineGroup
from action_contrast_loss import BundleObjective, TrainingBundle


class TinyValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def __call__(self, contexts, queries, projection_weights, scales):
        return self.linear(queries).reshape(queries.shape[0], -1)


class ContrastGradientTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(518)
        count = 3
        data = SimpleNamespace(targets=rng.normal(size=(count, 2652)).astype(np.float32),
                               target_scales=np.array([2., 5., 8.], dtype=np.float32),
                               weights=np.ones((count, 2652), dtype=np.float32),
                               projection_weights=np.ones((count, 2, 1326), dtype=np.float32))
        coeff = rng.uniform(size=(3, count, 1326))
        group = AffineGroup([], 0, ["check", "bet", "jam"], np.zeros((3, 1326)),
                            coeff, rng.normal(size=(3, 1326)), np.ones(1326), np.ones(1326, dtype=bool))
        bundle = TrainingBundle(data, np.zeros((count, 1), dtype=np.float32),
            rng.normal(size=(count, 2, 1326, 1)).astype(np.float32), [group], (0, 5, 10))
        return bundle

    @staticmethod
    def value_loss(model, contexts, queries, projection, scales, targets, weights):
        return mx.sum(weights * (model(contexts, queries, projection, scales) - targets)**2) / mx.sum(weights)

    def test_two_pass_matches_full_graph_with_unequal_scales(self):
        mx.random.seed(67)
        model = TinyValue()
        bundle = self.fixture(); group = bundle.groups[0]
        objective = BundleObjective([bundle], 1., chunk_size=1)
        calibration, contrast, _ = objective.gradients(model, bundle, self.value_loss)
        args = objective.inputs(bundle, 0, 3)
        def full_cal(current):
            return self.value_loss(current, *args, mx.array(bundle.dataset.targets), mx.array(bundle.dataset.weights))
        def full_contrast(current):
            values = current(*args).reshape(3, 2, 1326)[:, 0] * mx.array(bundle.dataset.target_scales[:, None])
            q = mx.sum(mx.array(group.coefficients.astype(np.float32)) * values[None, :, :], axis=1)
            loss = mx.array(0.)
            for a, b in [(0, 1), (0, 2), (1, 2)]:
                error = ((q[a]-q[b]) - mx.array((group.target[a]-group.target[b]).astype(np.float32))) / 20
                absolute = mx.abs(error); quadratic = mx.minimum(absolute, .05)
                loss += mx.mean(.5 * quadratic**2 + .05 * (absolute-quadratic)) / 3
            return loss
        _, expected_cal = nn.value_and_grad(model, full_cal)(model)
        _, expected_aux = nn.value_and_grad(model, full_contrast)(model)
        for actual, expected in [(calibration, expected_cal), (contrast, expected_aux)]:
            for (pa, ga), (pe, ge) in zip(tree_flatten(actual), tree_flatten(expected)):
                self.assertEqual(pa, pe)
                np.testing.assert_allclose(np.asarray(ga), np.asarray(ge), rtol=2e-5, atol=2e-7)

    def test_c0_same_schedule_and_gradients_except_auxiliary(self):
        mx.random.seed(89); model = TinyValue(); bundle = self.fixture()
        off, on = BundleObjective([bundle], 0.), BundleObjective([bundle], .5)
        empty = {"linear": {"weight": mx.zeros_like(model.linear.weight), "bias": mx.zeros_like(model.linear.bias)}}
        self.assertIs(off.accumulate(model, empty, self.value_loss, 1), empty)
        first = off.accumulate(model, empty, self.value_loss, 4)
        second = on.accumulate(model, empty, self.value_loss, 4)
        cal, aux, _ = on.gradients(model, bundle, self.value_loss)
        for (_, ga), (_, gb), (_, gc), (_, gd) in zip(tree_flatten(first), tree_flatten(second), tree_flatten(cal), tree_flatten(aux)):
            np.testing.assert_allclose(np.asarray(ga), np.asarray(gc))
            np.testing.assert_allclose(np.asarray(gb), np.asarray(gc + .5*gd))
        self.assertEqual(off.scheduled_families, on.scheduled_families)
        self.assertEqual(off.updates, 1)


if __name__ == "__main__": unittest.main()
