import copy
import unittest
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np

from cash_context_conditioning import (ContextConditionedHead, export_context_head,
    restore_context_head, context_dense_forward)


def inputs():
    x=np.zeros((1,2,1326,256),np.float32); x[:,:,:,:64]=1.; x[:,:,:,100]=2.
    return x


class ContextConditioningTests(unittest.TestCase):
    def test_identical_capacity_neutral_initialization_and_non_dead_output_gradients(self):
        mx.random.seed(7101); first=ContextConditionedHead(); mx.eval(first.parameters())
        mx.random.seed(7101); second=ContextConditionedHead(); mx.eval(second.parameters())
        self.assertEqual(sum(v.size for _,v in tree_flatten(first.parameters())),12417)
        self.assertEqual(export_context_head(first),export_context_head(second))
        x=mx.array(inputs())
        for mode in ('additive','film'):
            np.testing.assert_array_equal(np.asarray(first(x,mode)),np.zeros((1,2,1326),np.float32))
            def loss_fn(current): return mx.sum(current(x,mode))
            loss,grads=nn.value_and_grad(first,loss_fn)(first); mx.eval(loss,grads)
            self.assertGreater(np.count_nonzero(np.asarray(grads['output_tower']['layers'][0]['weight'])),0)
            self.assertTrue(all(np.count_nonzero(np.asarray(v))==0 for _,v in tree_flatten(grads['context_tower'])))

    def test_known_additive_and_multiplicative_arithmetic_and_cpu_decode(self):
        mx.random.seed(7102); model=ContextConditionedHead(); mx.eval(model.parameters())
        head=export_context_head(model)
        for tower in head.values():
            for layer in tower:
                layer['weights']=[0.]*len(layer['weights']); layer['biases']=[0.]*len(layer['biases'])
        head['input_tower'][0]['weights'][100]=1.
        head['context_tower'][0]['weights'][0]=.5
        head['context_tower'][0]['biases'][32]=.2
        head['output_tower'][0]['weights'][0]=1.
        restore_context_head(model,head)
        for mode,expected in (('additive',2.7),('film',3.2)):
            with mx.stream(mx.cpu): actual=np.asarray(model(mx.array(inputs()),mode))
            np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=0)
            np.testing.assert_allclose(actual,context_dense_forward(inputs(),head,mode),atol=1e-6,rtol=0)

    def test_decoder_rejects_invalid_context_broadcast_tensors_axes_and_modes(self):
        model=ContextConditionedHead(); mx.eval(model.parameters()); head=export_context_head(model)
        changed=inputs(); changed[:,:,1,0]=2.
        for x,mode in ((changed,'film'),(inputs().astype(np.float64),'film'),(inputs(),'unknown')):
            with self.assertRaises(ValueError): context_dense_forward(x,head,mode)
        bad=copy.deepcopy(head); bad['context_tower'][0]['weights'][0]=float('nan')
        with self.assertRaises(ValueError): restore_context_head(model,bad)
        bad=copy.deepcopy(head); bad['context_tower'][0]['inputSize']=63
        with self.assertRaises(ValueError): restore_context_head(model,bad)
        with self.assertRaises(ValueError): restore_context_head(model,{**head,'unknown':[]})


if __name__=='__main__': unittest.main()
