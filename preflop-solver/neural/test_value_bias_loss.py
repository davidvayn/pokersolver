"""A zero-sum output can still systematically overvalue one player's calls."""
import unittest

import mlx.core as mx
import numpy as np

from train_public_value_network import player_value_bias_loss


class PlayerValueBiasTest(unittest.TestCase):
    def test_opposite_player_errors_do_not_cancel(self):
        errors = mx.array([[1., 1., -1., -1.]])
        weights = mx.ones((1, 2, 2))
        self.assertAlmostEqual(float(player_value_bias_loss(errors, weights)), .5)
        grad = mx.grad(lambda e: player_value_bias_loss(e, weights))(errors)
        np.testing.assert_allclose(np.asarray(grad), [[.25, .25, -.25, -.25]])

    def test_reach_weights_and_empty_players(self):
        errors = mx.array([[1., 500., -1., -1.]])
        weights = mx.array([[[1., 0.], [0., 0.]]])
        self.assertAlmostEqual(float(player_value_bias_loss(errors, weights)), .5)
        self.assertEqual(float(player_value_bias_loss(errors, mx.zeros_like(weights))), 0.)

    def test_unbiased_errors_and_range_scale_invariance(self):
        errors = mx.array([[1., -1., 2., -2.]])
        weights = mx.ones((1, 2, 2))
        self.assertEqual(float(player_value_bias_loss(errors, weights)), 0.)
        other = mx.array([[1., 2., 3., 4.]])
        self.assertAlmostEqual(float(player_value_bias_loss(other, weights)),
                               float(player_value_bias_loss(other, weights * 7)))


if __name__ == '__main__':
    unittest.main()
