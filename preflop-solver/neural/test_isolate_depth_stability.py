import unittest
from isolate_depth_stability import comparison, decompose, draw_saved_mixture


class StabilityIsolationTests(unittest.TestCase):
    def test_saved_namespaces_are_preserved(self):
        rows = [[{'iteration': 10000001}], [{'iteration': 11000002}]]
        sampled = draw_saved_mixture(rows, 1000, 1)
        self.assertEqual({r['iteration'] for r in sampled}, {10000001, 11000002})
        self.assertEqual(sampled, draw_saved_mixture(rows, 1000, 1))

    def test_stable_underlying_average_does_not_hide_compression_disagreement(self):
        result = decompose([[[.9, .1]], [[.1, .9]]], [[[.6, .4]], [[.6, .4]]])
        self.assertAlmostEqual(result['exported_cross_seed']['frequency_mae'], .8)
        self.assertEqual(result['direct_average_cross_seed']['frequency_mae'], 0)
        self.assertEqual(result['direct_average_cross_seed']['primary_agreement'], 1)

    def test_decision_not_action_count_weighting(self):
        result = comparison([[1., 0.], [.25] * 4], [[0., 1.], [.25] * 4])
        self.assertEqual(result['frequency_mae'], .5)
        with self.assertRaises(ValueError):
            comparison([[.5, .5]], [[.5]])


if __name__ == '__main__':
    unittest.main()
