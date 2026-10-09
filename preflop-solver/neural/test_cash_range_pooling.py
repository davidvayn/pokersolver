import unittest
import mlx.core as mx
import numpy as np
from native_value_dataset import COMBOS, legal_combos
from cash_range_pooling import card_removed_opponent_pool, numpy_card_removed_opponent_pool


def key(first, second):
    return int(np.flatnonzero(((COMBOS == first).any(axis=1)) & ((COMBOS == second).any(axis=1)))[0])


class CashRangePoolingTests(unittest.TestCase):
    def test_every_query_matches_explicit_card_removal_including_censored_joint_hands(self):
        rng = np.random.default_rng(123)
        embeddings = rng.normal(size=(1,2,1326,3)).astype(np.float32)
        ranges = np.zeros((1,2,1326),np.float32)
        ranges[0,0,key(50,51)] = 1.
        for cards,weight in [((50,40),.25),((51,44),.25),((32,33),.5)]:
            ranges[0,1,key(*cards)] = weight
        expected = np.zeros_like(embeddings)
        for player in (0,1):
            for query,(a,b) in enumerate(COMBOS):
                compatible = ~((COMBOS == a).any(axis=1) | (COMBOS == b).any(axis=1))
                weights = ranges[0,1-player] * compatible
                if weights.sum() > 0:
                    expected[0,player,query] = weights @ embeddings[0,1-player] / weights.sum()
        with mx.stream(mx.cpu):
            actual = np.array(card_removed_opponent_pool(mx.array(embeddings),mx.array(ranges)))
        dense = numpy_card_removed_opponent_pool(embeddings,ranges)
        np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=0)
        np.testing.assert_allclose(dense,expected,atol=2e-6,rtol=0)
        zero_own = key(32,33)
        self.assertEqual(ranges[0,0,zero_own],0.)
        np.testing.assert_allclose(actual[0,0,zero_own],
            (embeddings[0,1,key(50,40)]+embeddings[0,1,key(51,44)])/2,atol=2e-6,rtol=0)
        self.assertTrue(np.isfinite(actual).all())

    def test_gradients_exclude_blocked_opponent_hands(self):
        ranges = np.zeros((1,2,1326),np.float32)
        ranges[0,0,key(50,51)] = 1.
        ranges[0,1,key(50,40)] = .5
        ranges[0,1,key(32,33)] = .5
        with mx.stream(mx.cpu):
            gradient = mx.grad(lambda e: card_removed_opponent_pool(e,mx.array(ranges))[0,0,key(50,51),0])(
                mx.ones((1,2,1326,1)))
            actual = np.array(gradient)
        self.assertEqual(float(actual[0,1,key(50,40),0]),0.)
        self.assertAlmostEqual(float(actual[0,1,key(32,33),0]),1.,places=6)
        self.assertEqual(float(np.abs(actual[0,0]).sum()),0.)


if __name__ == "__main__":
    unittest.main()
