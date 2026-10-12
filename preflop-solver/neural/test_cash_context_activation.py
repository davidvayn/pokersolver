import unittest
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from cash_context_activation import LeakyContextHead as Model,leaky_context_dense_forward
from cash_context_conditioning import ContextConditionedHead,export_context_head


def negative_fixture():
    model=Model()
    weight=np.zeros((32,256),np.float32); weight[:,64]=-1.
    model.input_tower.layers[0].weight=mx.array(weight)
    model.input_tower.layers[0].bias=mx.zeros((32,),dtype=mx.float32)
    x=np.zeros((2,2,1326,256),np.float32); x[0,:,:,64]=1.; x[1,:,:,64]=2.
    target=mx.array(np.broadcast_to(np.asarray([1.,-1.],np.float32)[:,None,None],(2,2,1326)).copy())
    return model,mx.array(x),target


class ContextActivationTests(unittest.TestCase):
    def test_negative_hidden_channels_can_receive_gradient_with_zero_readout(self):
        model,x,target=negative_fixture()
        loss=lambda current:mx.mean((current(x,'film')-target)**2)
        value,gradient=nn.value_and_grad(model,loss)(model); mx.eval(value,gradient)
        output=np.asarray(gradient['output_tower']['layers'][0]['weight'])
        self.assertGreater(float(np.linalg.norm(output)),0.,
            'Zero-initialized readout leaves all negative ReLU channels untrainable')

    def test_neutral_cold_forward_and_original_parameter_count(self):
        mx.random.seed(30); model=Model(); mx.eval(model.parameters())
        x=mx.ones((1,2,1326,256),dtype=mx.float32)
        self.assertTrue(np.array_equal(np.asarray(model(x)),np.zeros((1,2,1326),np.float32)))
        from mlx.utils import tree_flatten
        self.assertEqual(sum(value.size for _,value in tree_flatten(model.parameters())),12417)

    def test_body_gradients_recover_after_first_nonzero_readout_update(self):
        model,x,target=negative_fixture()
        loss=lambda current:mx.mean((current(x)-target)**2)
        gradient_fn=nn.value_and_grad(model,loss)
        value,gradient=gradient_fn(model); mx.eval(value,gradient)
        import mlx.optimizers as optim
        optimizer=optim.SGD(learning_rate=.01)
        optimizer.update(model,gradient); mx.eval(model.parameters(),optimizer.state)
        _,second=gradient_fn(model); mx.eval(second)
        body=np.asarray(second['input_tower']['layers'][0]['weight'])
        self.assertTrue(np.all(np.abs(body[:,64])>0))

    def test_independent_decoder_and_strict_axes(self):
        model,x,target=negative_fixture()
        model.output_tower.layers[0].weight=mx.full((1,32),.1)
        model.output_tower.layers[0].bias=mx.array([.2])
        mx.eval(model.parameters()); payload=export_context_head(model)
        reference=leaky_context_dense_forward(np.asarray(x),payload)
        self.assertLess(float(np.max(np.abs(reference-np.asarray(model(x))))),1e-6)
        with self.assertRaises(ValueError): model(x,'additive')
        with self.assertRaises(ValueError): leaky_context_dense_forward(np.asarray(x).astype(np.float64),payload)


if __name__=='__main__': unittest.main()

