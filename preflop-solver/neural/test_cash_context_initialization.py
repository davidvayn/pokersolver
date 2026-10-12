import unittest
import numpy as np
import mlx.core as mx
from cash_context_activation import LeakyContextHead
from cash_context_conditioning import export_context_head,restore_context_head
from cash_context_initialization import normalize_cold_context_input


def fixture():
    model=LeakyContextHead()
    matrix=np.zeros((32,256),np.float32); matrix[:,64]=-1.
    model.input_tower.layers[0].weight=mx.array(matrix)
    model.input_tower.layers[0].bias=mx.zeros((32,),dtype=mx.float32)
    mx.eval(model.parameters()); payload=export_context_head(model)
    x=np.zeros((2,2,1326,256),np.float32); x[0,:,:,64]=1.; x[1,:,:,64]=2.
    return model,payload,x,np.ones((2,2,1326),np.float32)


class ColdContextInitializationTests(unittest.TestCase):
    def test_training_moments_center_scale_and_preserve_neutral_parent(self):
        model,payload,x,w=fixture()
        result,report=normalize_cold_context_input(payload,x,w)
        self.assertTrue(np.allclose(report['preactivation_mean_after'],0.))
        self.assertTrue(np.allclose(report['preactivation_std_after'],1.))
        self.assertTrue(np.allclose(report['positive_reach_fraction_after'],.5))
        self.assertEqual(result['output_tower'],payload['output_tower'])
        self.assertEqual(result['context_tower'],payload['context_tower'])
        restore_context_head(model,result)
        self.assertTrue(np.array_equal(np.asarray(model(mx.array(x))),np.zeros((2,2,1326))))
        self.assertFalse(report['target_values_used'])

    def test_row_reordering_and_weight_scale_do_not_change_initialization(self):
        _,payload,x,w=fixture()
        first,_=normalize_cold_context_input(payload,x,w)
        reordered,_=normalize_cold_context_input(payload,x[::-1].copy(),w[::-1].copy()*10.)
        self.assertEqual(first,reordered)

    def test_constant_features_fail_without_manufacturing_variance(self):
        _,payload,x,w=fixture(); x[1]=x[0]
        with self.assertRaises(ValueError): normalize_cold_context_input(payload,x,w)

    def test_non_cold_or_invalid_weights_fail_closed(self):
        _,payload,x,w=fixture()
        payload['output_tower'][0]['biases'][0]=.1
        with self.assertRaises(ValueError): normalize_cold_context_input(payload,x,w)
        _,payload,x,w=fixture()
        with self.assertRaises(ValueError): normalize_cold_context_input(payload,x,w*0.)
        with self.assertRaises(ValueError): normalize_cold_context_input(payload,x,-w)


if __name__=='__main__': unittest.main()

