import copy
import unittest

from run_preflop_average_pilot import stability


class StabilityTests(unittest.TestCase):
    def fixture(self):
        return {'rows': [dict(key=str(i), actor=0, history=[], hand=hand,
            actions=['fold', 'call'], comboWeight=663, probabilities=probabilities)
            for i, hand, probabilities in [(0, 'AA', [0.8, 0.2]), (1, 'KK', [0.4, 0.6])]]}

    def test_probability_deltas_and_incomplete_coverage_are_explicit(self):
        a = self.fixture()
        b = copy.deepcopy(a)
        b['rows'][0]['probabilities'] = [0.6, 0.4]
        result = stability(a, b)['initialRoot']
        self.assertTrue(result['completeCoverage'])
        self.assertAlmostEqual(result['comboWeightedPerActionMae'], 0.1)
        self.assertAlmostEqual(result['maximumAggregateDelta'], 0.1)
        self.assertEqual(result['comboWeightedPrimaryAgreement'], 1)
        b['rows'][1]['probabilities'] = None
        result = stability(a, b)['initialRoot']
        self.assertFalse(result['completeCoverage'])
        self.assertEqual(result['comparedCombos'], 663)
        self.assertAlmostEqual(result['comboWeightedPerActionMae'], 0.2)

    def test_rejects_action_mismatch_and_invalid_probability_sums(self):
        a = self.fixture()
        b = copy.deepcopy(a)
        b['rows'][0]['actions'].reverse()
        with self.assertRaises(AssertionError):
            stability(a, b)
        b = copy.deepcopy(a)
        b['rows'][0]['probabilities'] = [0.6, 0.6]
        with self.assertRaises(AssertionError):
            stability(a, b)

    def test_initial_root_includes_posted_blinds(self):
        a = self.fixture()
        for row in a['rows']:
            row['history'] = ['post_sb_0.500bb', 'post_bb_1.000bb']
        result = stability(a, a)
        self.assertEqual(result['initialRoot']['history'], tuple(a['rows'][0]['history']))


if __name__ == '__main__':
    unittest.main()
