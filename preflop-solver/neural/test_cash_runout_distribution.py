import unittest

import numpy as np

from cash_checkdown import exact_cash_checkdown_features, exact_cash_runout_distribution
from cash_profiles import profile_rules
from native_value_dataset import COMBOS, legal_combos
from train_public_value_network import evaluate_cards


def key(a, b):
    high, low = max(a, b), min(a, b)
    return high * (high - 1) // 2 + low


class CashRunoutDistributionTests(unittest.TestCase):
    def setUp(self):
        self.board = [8, 13, 22, 31]
        self.ranges = np.zeros((2, 1326))
        self.ranges[0, key(50, 51)] = 1.
        self.ranges[1, key(46, 47)] = .7
        self.ranges[1, key(42, 37)] = .3
        self.rules = profile_rules('nl25')

    def brute(self, player, hand):
        cards = list(map(int, COMBOS[hand]))
        opponents = [(list(map(int, COMBOS[k])), self.ranges[1-player, k])
                     for k in np.flatnonzero(self.ranges[1-player])
                     if not set(cards).intersection(COMBOS[k])]
        mass = sum(weight for _, weight in opponents)
        mean = second = 0.
        histogram = np.zeros(8)
        for river in sorted(set(range(52)) - set(self.board + cards)):
            compatible = [(other, weight) for other, weight in opponents if river not in other]
            river_mass = sum(weight for _, weight in compatible)
            if not river_mass:
                continue
            hero_strength = evaluate_cards(self.board + cards + [river])
            numerator = 0.
            for other, weight in compatible:
                strength = evaluate_cards(self.board + other + [river])
                numerator += weight * (1. if hero_strength > strength else .5 if hero_strength == strength else 0.)
            equity = numerator / river_mass
            probability = river_mass / (44 * mass)
            mean += probability * equity
            second += probability * equity ** 2
            histogram[min(7, int(equity * 8))] += probability
        return mean, max(0., second - mean ** 2), histogram

    def test_exact_conditional_river_distribution_with_removal_and_zero_own_reach(self):
        mean, variance, histogram = exact_cash_runout_distribution(
            self.board, self.ranges, [7.6, 7.6], self.rules)
        for player in (0, 1):
            for hand in (key(50, 51), key(34, 35), key(46, 47)):
                expected = self.brute(player, hand)
                self.assertAlmostEqual(mean[player, hand], expected[0], places=11)
                self.assertAlmostEqual(variance[player, hand], expected[1], places=11)
                np.testing.assert_allclose(histogram[player, hand], expected[2], atol=1e-11, rtol=0)
        self.assertEqual(self.ranges[0, key(34, 35)], 0.)
        self.assertAlmostEqual(histogram[0, key(34, 35)].sum(), 1., places=11)
        np.testing.assert_array_equal(mean, exact_cash_checkdown_features(
            self.board, self.ranges, [7.6, 7.6], self.rules)[1])
        legal = legal_combos(self.board)
        self.assertEqual(histogram.shape, (2, 1326, 8))
        self.assertTrue(np.all(histogram[:, ~legal] == 0))
        self.assertTrue(np.all(variance[:, ~legal] == 0))
        self.assertTrue(np.all((variance >= 0) & (variance <= .25 + 1e-12)))
        self.assertGreater(np.max(variance), .001)

    def test_copies_rake_independence_and_range_scale_invariance(self):
        values = exact_cash_runout_distribution(self.board, self.ranges, [7.6, 7.6], self.rules)
        saved = tuple(value.copy() for value in values)
        for value in values:
            value[:] = 99.
        again = exact_cash_runout_distribution(self.board, self.ranges, [7.6, 7.6], self.rules)
        for left, right in zip(saved, again):
            np.testing.assert_array_equal(left, right)
        unraked = {**self.rules, 'rake': {**self.rules['rake'], 'rateBasisPoints': 0}}
        for left, right in zip(saved, exact_cash_runout_distribution(
                self.board, self.ranges, [7.6, 7.6], unraked)):
            np.testing.assert_array_equal(left, right)
        scaled = self.ranges * np.array([[3.], [7.]])
        for left, right in zip(saved, exact_cash_runout_distribution(
                self.board, scaled, [7.6, 7.6], self.rules)):
            np.testing.assert_allclose(left, right, atol=1e-11, rtol=0)

    def test_invalid_public_cards_or_beliefs_are_rejected(self):
        for board in ([8, 13, 22], [8, 13, 22, 22], [8, 13, 22, 52], [8, 13, 22, 31.5]):
            with self.assertRaises(ValueError):
                exact_cash_runout_distribution(board, self.ranges, [7.6, 7.6], self.rules)
        for ranges in (self.ranges[0], self.ranges.astype(complex),
                       np.full((2, 1326), np.nan), -self.ranges,
                       np.ones((2, 1326)), np.zeros((2, 1326))):
            with self.assertRaises(ValueError):
                exact_cash_runout_distribution(self.board, ranges, [7.6, 7.6], self.rules)


if __name__ == '__main__':
    unittest.main()
