import unittest

import numpy as np

from cash_checkdown import exact_cash_checkdown, exact_cash_checkdown_features
from cash_profiles import profile_rules
from native_value_dataset import COMBOS, legal_combos
from train_public_value_network import evaluate_cards


def key(a, b):
    return max(a, b) * (max(a, b) - 1) // 2 + min(a, b)


class CashCheckdownFeatureTests(unittest.TestCase):
    def test_shared_runouts_preserve_own_payoffs_and_independent_showdown_equity(self):
        board = [8, 13, 22, 31]
        ranges = np.zeros((2, 1326))
        ranges[0, key(50, 51)] = 1.
        ranges[1, key(46, 47)] = .7
        ranges[1, key(42, 37)] = .3
        rules = profile_rules("nl25")
        own, equity = exact_cash_checkdown_features(board, ranges, [7.6, 7.6], rules)
        np.testing.assert_array_equal(own, exact_cash_checkdown(board, ranges, [7.6, 7.6], rules))
        for player in (0, 1):
            for hand in (key(50, 51), key(34, 35), key(46, 47)):
                cards = list(map(int, COMBOS[hand]))
                expected_equity = expected_own = mass = 0.
                for other in np.flatnonzero(ranges[1-player]):
                    opponent = list(map(int, COMBOS[other]))
                    if set(cards) & set(opponent):
                        continue
                    weight = ranges[1-player, other]
                    remaining = sorted(set(range(52)) - set(board + cards + opponent))
                    self.assertEqual(len(remaining), 44)
                    for river in remaining:
                        a = evaluate_cards(board + cards + [river])
                        b = evaluate_cards(board + opponent + [river])
                        expected_equity += weight * (1. if a > b else .5 if a == b else 0.) / 44
                        # NL25 pot 380 cents, rake 17 cents, net 363 cents.
                        # The odd-cent tie is deliberately not equity * pot.
                        payoff = 173/25 if a > b else (-9+player)/25 if a == b else -190/25
                        expected_own += weight * payoff / 44
                    mass += weight
                self.assertAlmostEqual(equity[player, hand], expected_equity / mass if mass else 0., places=10)
                self.assertAlmostEqual(own[player, hand], expected_own / mass if mass else 0., places=10)
        legal = legal_combos(board)
        self.assertTrue(np.all(own[:, ~legal] == 0))
        self.assertTrue(np.all(equity[:, ~legal] == 0))
        self.assertTrue(np.all((equity >= 0) & (equity <= 1)))
        self.assertEqual(ranges[0, key(34, 35)], 0.)
        self.assertGreater(equity[0, key(34, 35)], 0.)

    def test_cached_arrays_cannot_be_mutated_and_equity_does_not_include_rake(self):
        board = [8, 13, 22, 31]
        legal = legal_combos(board)
        ranges = np.tile(legal / legal.sum(), (2, 1))
        rules = profile_rules("nl25")
        own, equity = exact_cash_checkdown_features(board, ranges, [2., 2.], rules)
        saved_own, saved_equity = own.copy(), equity.copy()
        own[:] = 100.; equity[:] = 100.
        again = exact_cash_checkdown_features(board, ranges, [2., 2.], rules)
        np.testing.assert_array_equal(again[0], saved_own)
        np.testing.assert_array_equal(again[1], saved_equity)
        rules["rake"]["rateBasisPoints"] = 0
        unraked_own, unraked_equity = exact_cash_checkdown_features(board, ranges, [2., 2.], rules)
        np.testing.assert_array_equal(unraked_equity, saved_equity)
        self.assertGreater(np.max(np.abs(unraked_own - saved_own)), .01)


if __name__ == "__main__":
    unittest.main()
