import itertools
import unittest

import numpy as np

from endpoint_variance import conditional_variances, importance_variance, fit_proposal


class EndpointVarianceTests(unittest.TestCase):
    def test_importance_sampling_keeps_the_full_gradient_expectation(self):
        values = np.array([[1., -2.], [-3., 4.], [8., 2.], [2., -1.], [-1., 3.]])
        q = np.array([.1, .2, .25, .3, .15])
        estimates = values/q[:, None]
        mean = (q[:, None]*estimates).sum(axis=0)
        np.testing.assert_allclose(mean, values.sum(axis=0), atol=1e-12)
        expected = (q[:, None]*(estimates-mean)**2).sum(axis=0)
        np.testing.assert_allclose(importance_variance(values, q), expected, atol=1e-12)
        proposal = fit_proposal(np.array([[0., 1., 9., 0., 4.]]))
        self.assertTrue((proposal >= .5/5).all())
        self.assertAlmostEqual(proposal.sum(), 1.)
        self.assertGreater(proposal[2], proposal[1])
        np.testing.assert_allclose(fit_proposal(np.zeros((2, 5))), np.full(5, .2))

    def test_exact_enumeration_matches_unbiased_sampling_variances(self):
        values = np.array([[1., -2.], [-3., 4.], [8., 2.], [2., -1.], [-1., 3.]])
        groups = [0, 0, 1, 1, 1]
        result = conditional_variances(values, groups, 2)
        samples = {
            "uniformOne": np.array([5*row for row in values]),
            "uniformBatch": np.array([2.5*values[list(pair)].sum(axis=0)
                                      for pair in itertools.combinations(range(5), 2)]),
            "rootStratified": np.array([2*values[a]+3*values[b]
                                        for a in range(2) for b in range(2, 5)]),
        }
        for kind, estimates in samples.items():
            np.testing.assert_allclose(estimates.mean(axis=0), values.sum(axis=0), atol=1e-12)
            np.testing.assert_allclose(result[kind], estimates.var(axis=0), atol=1e-12)

    def test_constant_within_strata_has_zero_stratified_variance(self):
        result = conditional_variances(np.array([[1.], [1.], [3.], [3.]]), [0, 0, 1, 1], 2)
        np.testing.assert_array_equal(result["rootStratified"], [0.])
        self.assertGreater(result["uniformBatch"][0], 0)
        with self.assertRaises(ValueError):
            conditional_variances(np.array([[float("nan")], [1.]]), [0, 1], 2)
        with self.assertRaises(ValueError):
            conditional_variances(np.ones((3, 1)), [0, 1], 2)


if __name__ == "__main__":
    unittest.main()
