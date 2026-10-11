"""Compare factorized kernel aggregation with explicit compatible pairs."""
import unittest
import mlx.core as mx
import numpy as np

from cash_range_attention import positive_kernel_opponent_pool,numpy_positive_kernel_opponent_pool
from cash_range_pooling import numpy_card_removed_opponent_pool
from native_value_dataset import COMBOS


def key(a,b):
    return max(a,b)*(max(a,b)-1)//2+min(a,b)


class CashRangeAttentionTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(101)
        values=rng.normal(size=(1,2,1326,3)).astype(np.float32)
        queries=rng.uniform(.25,1.5,size=(1,2,1326,4)).astype(np.float32)
        keys=rng.uniform(.25,1.5,size=(1,2,1326,4)).astype(np.float32)
        ranges=np.zeros((1,2,1326),np.float32)
        ranges[0,0,key(50,51)]=1
        for cards,weight in (((50,40),.25),((51,44),.25),((32,33),.5)):
            ranges[0,1,key(*cards)]=weight
        return values,ranges,queries,keys

    def explicit(self,values,ranges,queries,keys):
        expected=np.zeros_like(values)
        for player in (0,1):
            other=1-player
            for index,(a,b) in enumerate(COMBOS):
                legal=~((COMBOS==a).any(axis=1)|(COMBOS==b).any(axis=1))
                weights=ranges[0,other].astype(np.float64)*legal*(keys[0,other].astype(np.float64)@queries[0,player,index].astype(np.float64))
                if weights.sum()>1e-9:
                    expected[0,player,index]=weights@values[0,other].astype(np.float64)/weights.sum()
        return expected

    def test_sparse_all_query_forward_matches_explicit_kernel_and_card_removal(self):
        arrays=self.fixture(); expected=self.explicit(*arrays)
        with mx.stream(mx.cpu):
            actual=np.array(positive_kernel_opponent_pool(*(mx.array(a) for a in arrays)))
        dense=numpy_positive_kernel_opponent_pool(*arrays)
        np.testing.assert_allclose(actual,expected,atol=4e-6,rtol=0)
        np.testing.assert_allclose(dense,expected,atol=4e-6,rtol=0)
        self.assertEqual(arrays[1][0,0,key(32,33)],0.)
        self.assertGreater(float(np.abs(actual[0,0,key(32,33)]).sum()),0.)

    def test_constant_kernel_is_the_card_removed_mean_not_uniform_play(self):
        values,ranges,queries,keys=self.fixture()
        queries=np.ones_like(queries); keys=np.ones_like(keys)
        actual=numpy_positive_kernel_opponent_pool(values,ranges,queries,keys)
        expected=numpy_card_removed_opponent_pool(values,ranges)
        np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=0)

    def test_query_features_change_compatible_opponent_weighting(self):
        values,ranges,queries,keys=self.fixture()
        i,j=key(0,1),key(2,3)
        values[:]=0; values[0,1,key(50,40),0]=1
        keys[:]=1; keys[0,1,key(50,40),0]=10
        queries[:]=1; queries[0,0,i,0]=10
        actual=numpy_positive_kernel_opponent_pool(values,ranges,queries,keys)
        self.assertGreater(float(actual[0,0,i,0]),float(actual[0,0,j,0]))

    def test_floating_mass_cancellation_never_invents_all_blocked_support(self):
        values,ranges,queries,keys=self.fixture(); rng=np.random.default_rng(173)
        ranges[:]=0
        indices=[key(50,card) for card in range(50)]
        weights=rng.uniform(size=len(indices)).astype(np.float32); weights/=weights.sum()
        ranges[0,1,indices]=weights
        with mx.stream(mx.cpu):
            actual=np.array(positive_kernel_opponent_pool(*(mx.array(a) for a in (values,ranges,queries,keys))))
        dense=numpy_positive_kernel_opponent_pool(values,ranges,queries,keys)
        np.testing.assert_array_equal(actual[0,0,key(50,51)],0.)
        np.testing.assert_array_equal(dense[0,0,key(50,51)],0.)
        np.testing.assert_array_equal(actual[0,1],0.)

    def test_value_and_key_gradients_exclude_blocked_hands_and_reach_active_hands(self):
        values,ranges,queries,keys=self.fixture()
        with mx.stream(mx.cpu):
            value_gradient=mx.grad(lambda v:positive_kernel_opponent_pool(v,mx.array(ranges),mx.array(queries),mx.array(keys))[0,0,key(50,51),0])(mx.array(values))
            key_gradient=mx.grad(lambda k:positive_kernel_opponent_pool(mx.array(values),mx.array(ranges),mx.array(queries),k)[0,0,key(0,1),0])(mx.array(keys))
            vg=np.array(value_gradient); kg=np.array(key_gradient)
        self.assertEqual(float(vg[0,1,key(50,40),0]),0.)
        self.assertEqual(float(vg[0,1,key(51,44),0]),0.)
        self.assertAlmostEqual(float(vg[0,1,key(32,33),0]),1.,places=5)
        self.assertEqual(float(np.abs(vg[0,0]).sum()),0.)
        self.assertGreater(float(np.abs(kg[0,1,key(50,40)]).sum()),0.)
        self.assertEqual(float(np.abs(kg[0,0]).sum()),0.)

    def test_query_gradient_matches_explicit_compatible_kernel(self):
        values,ranges,queries,keys=self.fixture(); index=key(0,1)
        with mx.stream(mx.cpu):
            constants=tuple(mx.array(a) for a in (values,ranges,keys))
            actual=mx.grad(lambda q:positive_kernel_opponent_pool(constants[0],constants[1],q,constants[2])[0,0,index,0])(mx.array(queries))
            def explicit(q):
                weights=constants[1][0,1]*mx.sum(constants[2][0,1]*q[0,0,index],axis=-1)
                return mx.sum(weights*constants[0][0,1,:,0])/mx.sum(weights)
            expected=mx.grad(explicit)(mx.array(queries))
            np.testing.assert_allclose(np.array(actual),np.array(expected),atol=3e-6,rtol=0)

    def test_invalid_shapes_fail_before_pairwise_expansion(self):
        values,ranges,queries,keys=self.fixture()
        for arrays in ((values,ranges[:,:,:2],queries,keys),(values,ranges,queries,keys[:,:,:,:2]),
                       (values,ranges,queries[:,:,:,:0],keys[:,:,:,:0])):
            with self.assertRaisesRegex(ValueError,'1326-combo'):
                numpy_positive_kernel_opponent_pool(*arrays)


if __name__=='__main__': unittest.main()
