import unittest

import mlx.core as mx
import numpy as np

from distill_checkpoint_ensemble import ProbabilityEnsemble, parse_round_weights


class DistillCheckpointEnsembleTests(unittest.TestCase):
    def test_round_weights_are_normalized(self):
        self.assertEqual(parse_round_weights("100:8,200:2"), [(100, 0.8), (200, 0.2)])

    def test_round_weights_reject_non_positive_values(self):
        with self.assertRaises(ValueError):
            parse_round_weights("100:0")

    def test_probability_ensemble_averages_policies_not_logits(self):
        class Fixed:
            def __init__(self, logits):
                self.logits = mx.array(logits)[:, None]

            def __call__(self, _features):
                return self.logits

        ensemble = ProbabilityEnsemble(
            [Fixed([2.0, 0.0]), Fixed([0.0, 0.0])], [0.5, 0.5]
        )
        actual = np.exp(np.asarray(ensemble(mx.zeros((2, 1)))).reshape(-1))
        expected = 0.5 * np.asarray([0.880797, 0.119203]) + 0.5 * np.asarray(
            [0.5, 0.5]
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
