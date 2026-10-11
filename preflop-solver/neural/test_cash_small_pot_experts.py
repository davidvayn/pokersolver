import copy
import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np

from cash_profiles import profile_rules, rules_digest
from cash_small_pot_experts import (SmallPotExpertHead, small_pot_expert_weights,
    export_expert_heads, restore_expert_heads, expert_dense_forward)


def source(amounts):
    rules=profile_rules('nl25'); game={'effective_stack_bb':20.,'cash_rules':rules}
    return {'game':game,'rules_sha256':rules_digest(rules),
        'labels':[{'input':{'game':copy.deepcopy(game),'state':{
            'board':[1,27,22,25],'invested_bb':[cents/25.,cents/25.]}}} for cents in amounts]}


class SmallPotExpertTests(unittest.TestCase):
    def test_weights_use_exact_public_money_and_smooth_partition_without_mutation(self):
        item=source([10,25,26,38,49,50,75]); original=copy.deepcopy(item)
        weights=small_pot_expert_weights(item,'routed')
        z=np.clip((np.array([10,25,26,38,49,50,75])-25)/25.,0,1)
        bet=z*z*(3-2*z)
        np.testing.assert_allclose(weights,np.stack((1-bet,bet),axis=1),atol=1e-7,rtol=0)
        np.testing.assert_array_equal(weights.sum(axis=1),np.ones(7,np.float32))
        np.testing.assert_array_equal(small_pot_expert_weights(item,'shared'),np.full((7,2),.5,np.float32))
        self.assertEqual(item,original)
        bad=copy.deepcopy(item); bad['labels'][0]['input']['state']['invested_bb']=[1.,1.04]
        with self.assertRaises(ValueError): small_pot_expert_weights(bad,'routed')
        with self.assertRaises(ValueError): small_pot_expert_weights(item,'category')

    def test_control_and_routed_models_have_identical_capacity_initialization_and_zero_output(self):
        mx.random.seed(7101); first=SmallPotExpertHead(); mx.eval(first.parameters())
        mx.random.seed(7101); second=SmallPotExpertHead(); mx.eval(second.parameters())
        self.assertEqual(sum(v.size for _,v in tree_flatten(first.parameters())),8258)
        self.assertEqual(export_expert_heads(first),export_expert_heads(second))
        x=mx.ones((1,2,1326,256),mx.float32)
        for weights in (mx.array([[.5,.5]]),mx.array([[1.,0.]]),mx.array([[0.,1.]])):
            np.testing.assert_array_equal(np.asarray(first(x,weights)),np.zeros((1,2,1326),np.float32))

    def test_routing_localizes_gradients_and_shared_control_updates_both_experts(self):
        mx.random.seed(7102); model=SmallPotExpertHead(); mx.eval(model.parameters())
        x=mx.ones((1,2,1326,256),mx.float32)
        for weights,inactive in (([1.,0.],1),([0.,1.],0),([.5,.5],None)):
            def loss_fn(current): return mx.sum(current(x,mx.array([weights],mx.float32)))
            loss,grads=nn.value_and_grad(model,loss_fn)(model); mx.eval(loss,grads)
            for index,expert in enumerate(grads['heads']):
                values=[np.asarray(v) for _,v in tree_flatten(expert)]
                if index==inactive: self.assertTrue(all(np.count_nonzero(v)==0 for v in values))
                else: self.assertTrue(any(np.count_nonzero(v)>0 for v in values))

    def test_export_restore_and_cpu_numpy_decoder_reproduce_endpoints_and_transition(self):
        mx.random.seed(7101); model=SmallPotExpertHead(); mx.eval(model.parameters())
        heads=export_expert_heads(model)
        heads[0][-1]['biases']=[.3]; heads[0][-1]['weights']=[.02]*16
        heads[1][-1]['biases']=[-.2]; heads[1][-1]['weights']=[-.03]*16
        restore_expert_heads(model,heads)
        x=np.linspace(-.5,.5,2*1326*256,dtype=np.float32).reshape((1,2,1326,256))
        for weights in (np.array([[1.,0.]],np.float32),np.array([[0.,1.]],np.float32),
                        np.array([[.5,.5]],np.float32),small_pot_expert_weights(source([38]),'routed')):
            with mx.stream(mx.cpu): actual=np.asarray(model(mx.array(x),mx.array(weights)))
            np.testing.assert_allclose(actual,expert_dense_forward(x,heads,weights),atol=1e-6,rtol=0)
        bad=copy.deepcopy(heads); bad[0][0]['weights'][0]=float('nan')
        with self.assertRaises(ValueError): restore_expert_heads(model,bad)
        for weights in (np.array([[.3,.3]],np.float32),np.array([[1.,0.]],np.float64),
                        np.array([[np.nan,0.]],np.float32)):
            with self.assertRaises(ValueError): expert_dense_forward(x,heads,weights)
        with self.assertRaises(ValueError): expert_dense_forward(x.astype(np.float64),heads,np.array([[.5,.5]],np.float32))


if __name__=='__main__': unittest.main()
