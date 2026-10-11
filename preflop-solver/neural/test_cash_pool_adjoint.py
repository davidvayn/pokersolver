"""Repeated-card backward reductions must be reproducible on the GPU."""
import unittest

import mlx.core as mx
import numpy as np

import cash_range_pooling as pooling


class CashCardAdjointTests(unittest.TestCase):
    def test_card_views_preserve_exact_forward_values(self):
        generator=np.random.default_rng(9174)
        with mx.stream(mx.cpu):
            cards=mx.array(generator.normal(size=(3,2,52,5)).astype(np.float32))
            first,second=pooling._card_query_views(cards)
            np.testing.assert_array_equal(np.array(first),np.array(cards[:,:,pooling.FIRST,:]))
            np.testing.assert_array_equal(np.array(second),np.array(cards[:,:,pooling.SECOND,:]))

    def test_adjoint_matches_independent_signed_card_sums(self):
        generator=np.random.default_rng(9175)
        first=generator.normal(size=(2,2,1326,3)).astype(np.float32)
        second=generator.normal(size=first.shape).astype(np.float32)
        expected=np.zeros((2,2,52,3),np.float64)
        for index,(a,b) in enumerate(pooling.COMBOS):
            expected[:,:,a,:]+=first[:,:,index,:]
            expected[:,:,b,:]+=second[:,:,index,:]
        with mx.stream(mx.cpu):
            cards=mx.zeros(expected.shape)
            first_mx,second_mx=mx.array(first),mx.array(second)
            def objective(value):
                a,b=pooling._card_query_views(value)
                return mx.sum(a*first_mx)+mx.sum(b*second_mx)
            actual=np.array(mx.grad(objective)(cards))
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=1e-5)

    def test_cpu_pool_value_and_range_gradients_match_indexed_backward(self):
        from unittest.mock import patch
        generator=np.random.default_rng(9176)
        with mx.stream(mx.cpu):
            values=mx.array(generator.normal(size=(1,2,1326,4)).astype(np.float32))
            reaches=generator.uniform(.001,1.,size=(1,2,1326)).astype(np.float32)
            reaches/=reaches.sum(axis=2,keepdims=True)
            ranges=mx.array(reaches)
            gradient=mx.grad(lambda v,r:mx.mean(pooling.card_removed_opponent_pool(v,r)**2),argnums=(0,1))
            actual=gradient(values,ranges); mx.eval(actual)
            def indexed(cards):
                return cards[:,:,pooling.FIRST,:],cards[:,:,pooling.SECOND,:]
            with patch.object(pooling,'_card_query_views',indexed):
                expected=gradient(values,ranges); mx.eval(expected)
            for value,reference in zip(actual,expected):
                np.testing.assert_allclose(np.array(value),np.array(reference),rtol=2e-5,atol=1e-8)

    @unittest.skipUnless(mx.metal.is_available(),'Metal GPU required for scatter-repeat regression')
    def test_repeated_real_card_pool_gradients_are_identical_on_gpu(self):
        generator=np.random.default_rng(9173)
        with mx.stream(mx.gpu):
            values=mx.array(generator.normal(size=(1,2,1326,4)).astype(np.float32))
            reaches=generator.uniform(.001,1.,size=(1,2,1326)).astype(np.float32)
            reaches/=reaches.sum(axis=2,keepdims=True)
            ranges=mx.array(reaches)
            gradient=mx.grad(lambda v:mx.mean(pooling.card_removed_opponent_pool(v,ranges)**2))
            baseline=None
            for _ in range(20):
                actual=gradient(values); mx.eval(actual); array=np.array(actual)
                if baseline is None: baseline=array.copy()
                np.testing.assert_array_equal(array,baseline)

    @unittest.skipUnless(mx.metal.is_available(),'Metal GPU required for kernel-repeat regression')
    def test_kernel_value_query_and_key_gradients_are_identical_on_gpu(self):
        from cash_range_attention import positive_kernel_opponent_pool
        generator=np.random.default_rng(9177)
        with mx.stream(mx.gpu):
            values=mx.array(generator.normal(size=(1,2,1326,3)).astype(np.float32))
            reaches=generator.uniform(.001,1.,size=(1,2,1326)).astype(np.float32)
            reaches/=reaches.sum(axis=2,keepdims=True)
            ranges=mx.array(reaches)
            queries=mx.array(generator.uniform(.1,2.,size=(1,2,1326,8)).astype(np.float32))
            keys=mx.array(generator.uniform(.1,2.,size=queries.shape).astype(np.float32))
            gradient=mx.grad(lambda v,q,k:mx.mean(positive_kernel_opponent_pool(v,ranges,q,k)**2),argnums=(0,1,2))
            baseline=None
            for _ in range(10):
                actual=gradient(values,queries,keys); mx.eval(actual)
                arrays=[np.array(value) for value in actual]
                if baseline is None: baseline=[value.copy() for value in arrays]
                for value,reference in zip(arrays,baseline):
                    np.testing.assert_array_equal(value,reference)


if __name__=='__main__': unittest.main()
