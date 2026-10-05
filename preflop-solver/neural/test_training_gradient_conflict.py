import unittest

import mlx.core as mx

from training_gradient_conflict import gradient_dot, gradient_relationship


class TrainingGradientConflictTests(unittest.TestCase):
    def test_conflict_and_alignment_directions_are_not_reversed(self):
        reference = dict(weight=mx.array([1., 2.]))
        opposed = dict(weight=mx.array([-1., -2.]))
        result = gradient_relationship(reference, opposed)
        self.assertAlmostEqual(result["cosine"], -1.)
        self.assertGreater(result["referenceLossDirectionalDerivativeUnderNegativeGradient"], 0.)
        self.assertAlmostEqual(gradient_relationship(reference, reference)["cosine"], 1.)
        self.assertIsNone(gradient_relationship(reference, dict(weight=mx.zeros(2)))["cosine"])

    def test_invalid_paths_shapes_and_nonfinite_fail_closed(self):
        reference = dict(weight=mx.array([1., 2.]))
        for wrong in ({}, dict(bias=mx.ones(2)), dict(weight=mx.ones(3)),
                      dict(weight=mx.array([1., float("nan")]))):
            with self.assertRaises(ValueError): gradient_dot(reference, wrong)


if __name__ == "__main__": unittest.main()
