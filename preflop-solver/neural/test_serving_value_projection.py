import unittest

import numpy as np

import native_value_dataset as native
from serving_value_projection import BoundedValueProjection


class ServingValueProjectionTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(34)
        boards = np.array([[0, 5, 10, 15], [20, 25, 30, 35], [1, 6, 11, 16]])
        legal = np.array([native.legal_combos(b) for b in boards])
        ranges = rng.random((3, 2, 1326)) * legal[:, None]
        ranges /= ranges.sum(axis=2, keepdims=True)
        ranges[1] *= 1e-7  # authentic joint below the native denominator floor
        ranges[2, 0] = 0.  # zero-joint deviations are still served
        weights = np.asarray([r * native.compatible_masses(r) for r in ranges])
        raw = rng.normal(0., 18., (3, 2, 1326))
        return raw, weights, legal, boards, ranges

    def test_matches_actual_native_projection_including_clipping_tiny_and_zero_joint(self):
        raw, weights, legal, boards, ranges = self.fixture()
        projection = BoundedValueProjection(raw, weights, legal)
        expected = np.array([native.project_native_predictions(v, b, r)
            for v, b, r in zip(raw, boards, ranges)])
        np.testing.assert_allclose(projection.values, expected, atol=1e-10, rtol=0)
        self.assertLessEqual(abs(projection.values).max(), 20.)
        self.assertTrue(np.all(projection.values[~np.broadcast_to(legal[:,None], raw.shape)] == 0))

    def test_vjp_matches_directional_finite_difference_through_both_clips_and_coupled_target(self):
        raw, weights, legal, _, _ = self.fixture()
        rng = np.random.default_rng(18)
        upstream = rng.normal(size=raw.shape)
        direction = rng.normal(size=raw.shape)
        projection = BoundedValueProjection(raw, weights, legal)
        gradient = projection.vjp(upstream)
        # Stay away from either clipping boundary for the differentiable probe.
        direction[(abs(abs(raw)-20.) < .05) | (abs(abs(projection.shifted)-20.) < .05)] = 0
        epsilon = 1e-5
        plus = BoundedValueProjection(raw+epsilon*direction, weights, legal).values
        minus = BoundedValueProjection(raw-epsilon*direction, weights, legal).values
        measured = np.sum(upstream*(plus-minus))/(2*epsilon)
        expected = np.sum(gradient*direction)
        self.assertAlmostEqual(measured, expected, delta=1e-4)


if __name__ == "__main__": unittest.main()
