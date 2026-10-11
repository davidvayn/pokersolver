import unittest

import mlx.core as mx
import numpy as np

from cash_gradient_projection import combine_cash_domain_gradients
from train_cash_value_network import cash_prefix_value_and_grad, run
import mlx.nn as nn
from pathlib import Path


class CashGradientProjectionTests(unittest.TestCase):
    def gradient(self, values):
        return {'tower': [{'weight': mx.array(values,dtype=mx.float32)}]}

    def test_conflicting_domains_preserve_both_first_order_descent_directions(self):
        old=self.gradient([2.,0.]); added=self.gradient([-1.,1.])
        before=[np.array(g['tower'][0]['weight']) for g in (old,added)]
        combined,metrics=combine_cash_domain_gradients(old,added)
        actual=np.array(combined['tower'][0]['weight'])
        np.testing.assert_allclose(actual,[1.,2.],atol=1e-7)
        self.assertTrue(metrics['projected'])
        self.assertLess(metrics['cosine'],0)
        for original in before: self.assertGreater(float(original@actual),0)
        for g,expected in zip((old,added),before):
            np.testing.assert_array_equal(np.array(g['tower'][0]['weight']),expected)

    def test_nonconflicting_and_zero_domains_return_exact_original_sum(self):
        for left,right in [([2.,1.],[1.,3.]),([0.,0.],[1.,3.]),([0.,0.],[0.,0.])]:
            old,added=self.gradient(left),self.gradient(right)
            combined,metrics=combine_cash_domain_gradients(old,added)
            self.assertFalse(metrics['projected'])
            np.testing.assert_array_equal(np.array(combined['tower'][0]['weight']),
                np.array(mx.array(left,dtype=mx.float32)+mx.array(right,dtype=mx.float32)))

    def test_projection_is_global_not_independent_per_layer(self):
        old={'a':mx.array([2.,0.]),'b':mx.array([1.])}
        added={'a':mx.array([-1.,1.]),'b':mx.array([3.])}
        combined,metrics=combine_cash_domain_gradients(old,added)
        self.assertGreater(metrics['dot'],0)
        self.assertFalse(metrics['projected'])
        np.testing.assert_array_equal(np.array(combined['a']),[1.,1.])
        np.testing.assert_array_equal(np.array(combined['b']),[4.])

    def test_float64_reductions_avoid_float32_norm_overflow_and_underflow(self):
        for scale in [1e-25,1.,1e25]:
            old=self.gradient(np.asarray([2.,0.])*scale)
            added=self.gradient(np.asarray([-1.,1.])*scale)
            combined,metrics=combine_cash_domain_gradients(old,added)
            self.assertTrue(metrics['projected'])
            self.assertAlmostEqual(metrics['cosine'],-1/np.sqrt(2),places=6)
            np.testing.assert_allclose(np.array(combined['tower'][0]['weight'])/scale,[1.,2.],rtol=1e-6)

    def test_mismatched_shapes_parameters_empty_or_nonfinite_domains_rejected(self):
        ordinary=self.gradient([1.,2.])
        for bad in [self.gradient([1.]),{'different':mx.array([1.,2.])},
                    self.gradient([float('nan'),0.]),self.gradient([float('inf'),0.]),{}]:
            with self.assertRaises(ValueError): combine_cash_domain_gradients(ordinary,bad)
        with self.assertRaises(ValueError): combine_cash_domain_gradients({}, {})

    def test_real_split_accumulation_keeps_one_frozen_model_and_global_loss_denominator(self):
        model=nn.Linear(1,1,bias=False); model.weight=mx.array([[0.]])
        x=mx.array([[1.],[2.],[3.],[4.]])
        y=mx.array([[1.],[1.],[-1.],[-1.]])
        def loss(current,rows): return mx.sum((current(x[rows])-y[rows])**2)/4.
        gradient_fn=nn.value_and_grad(model,loss)
        expected,gradient=gradient_fn(model,mx.array([0,1,2,3]))
        actual,combined,metrics=cash_prefix_value_and_grad(model,gradient_fn,
            mx.array([0,1]),mx.array([2,3]),1,'prefix-sum')
        self.assertAlmostEqual(float(actual.item()),float(expected.item()),places=7)
        np.testing.assert_array_equal(np.array(combined['weight']),np.array(gradient['weight']))
        np.testing.assert_array_equal(np.array(model.weight),[[0.]])
        self.assertFalse(metrics['projected'])
        _,projected,metrics=cash_prefix_value_and_grad(model,gradient_fn,
            mx.array([0,1]),mx.array([2,3]),1,'prefix-pcgrad')
        self.assertTrue(metrics['projected'])
        # Exactly antiparallel single-parameter gradients have no common
        # strict descent direction. Projection must return zero, not invent one.
        np.testing.assert_allclose(np.array(projected['weight']),[[0.]],atol=1e-7)

    def test_training_option_is_bounded_and_rejects_missing_warm_prefix_before_io(self):
        for mode in ['prefix-sum','prefix-pcgrad','unknown',True]:
            with self.assertRaises(ValueError):
                run(Path('missing'),Path('must-not-create'),7101,1,
                    gradient_domain_mode=mode,gradient_batch_size=8,training_output_mode='raw')


if __name__=='__main__': unittest.main()
