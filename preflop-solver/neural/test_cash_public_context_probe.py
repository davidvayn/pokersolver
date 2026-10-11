import unittest
import mlx.core as mx
import numpy as np
from cash_context_conditioning import (ContextConditionedHead, export_context_head,
    restore_context_head, context_dense_forward)
from cash_public_context_probe import (ExternalContextHead, make_public_sketch,
    sketch_public_context, match_training_sketch_gain, external_context_dense_forward)


class PublicContextProbeTests(unittest.TestCase):
    def test_deterministic_signed_projection_and_training_only_rms_match(self):
        raw=np.random.default_rng(12).normal(size=(3,2,417)).astype(np.float32)
        projection=make_public_sketch(20261031)
        np.testing.assert_array_equal(projection,make_public_sketch(20261031))
        self.assertFalse(np.array_equal(projection,make_public_sketch(20261032)))
        self.assertEqual(set(np.unique(projection)),{-.125,.125})
        encoded=np.ones((3,2,64),np.float32)*2
        gain=match_training_sketch_gain(raw,encoded,projection)
        projected=sketch_public_context(raw,projection,gain)
        self.assertEqual(projected.shape,(3,2,64)); self.assertEqual(projected.dtype,np.float32)
        self.assertAlmostEqual(float(np.sqrt(np.mean(projected.astype(np.float64)**2))),2.,places=6)
        np.testing.assert_array_equal(projected[::-1],sketch_public_context(raw[::-1],projection,gain))

    def test_cold_neutrality_same_parameters_and_non_dead_output_gradient(self):
        mx.random.seed(7101); old=ContextConditionedHead()
        mx.random.seed(7101); new=ExternalContextHead()
        for tower in ('input_tower','context_tower','output_tower'):
            for field in ('weight','bias'):
                np.testing.assert_array_equal(np.asarray(getattr(getattr(old,tower).layers[0],field)),
                    np.asarray(getattr(getattr(new,tower).layers[0],field)))
        self.assertEqual(sum(layer.weight.size+layer.bias.size for layer in
            (new.input_tower.layers[0],new.context_tower.layers[0],new.output_tower.layers[0])),12417)
        inputs=mx.ones((1,2,1326,256)); context=mx.ones((1,2,64))
        np.testing.assert_array_equal(np.asarray(new(inputs)),np.asarray(old(inputs,'film')))
        np.testing.assert_array_equal(np.asarray(new(inputs,context)),np.zeros((1,2,1326)))
        gradients=mx.grad(lambda model:mx.mean((model(inputs,context)-1)**2))(new)
        self.assertGreater(float(mx.sum(mx.abs(gradients['output_tower']['layers'][0]['weight'])).item()),0.)

    def test_encoded_control_and_external_cpu_arithmetic(self):
        model=ExternalContextHead()
        model.input_tower.layers[0].weight=mx.zeros((32,256))
        model.input_tower.layers[0].bias=mx.ones((32))*2
        model.context_tower.layers[0].weight=mx.zeros((64,64))
        model.context_tower.layers[0].bias=mx.concatenate((mx.ones(32)*.5,mx.ones(32)*.2))
        model.output_tower.layers[0].weight=mx.ones((1,32))/32
        model.output_tower.layers[0].bias=mx.zeros(1)
        payload=export_context_head(model)
        inputs=np.ones((1,2,1326,256),np.float32); context=inputs[:,:,0,:64].copy()
        actual=external_context_dense_forward(inputs,context,payload)
        np.testing.assert_allclose(actual,3.2,atol=1e-6,rtol=0)
        np.testing.assert_array_equal(actual,context_dense_forward(inputs,payload,'film'))
        with mx.stream(mx.cpu): decoded=np.asarray(model(mx.array(inputs),mx.array(context)))
        np.testing.assert_allclose(actual,decoded,atol=1e-6,rtol=0)
        # A nonconstant external context affects only the conditioner, not the
        # frozen combined input tower or a query's private coordinate axis.
        model.context_tower.layers[0].weight=mx.ones((64,64))*.01
        payload=export_context_head(model); context[0,1]=2
        actual=external_context_dense_forward(inputs,context,payload)
        with mx.stream(mx.cpu): decoded=np.asarray(model(mx.array(inputs),mx.array(context)))
        np.testing.assert_allclose(actual,decoded,atol=2e-6,rtol=0)
        np.testing.assert_array_equal(inputs,np.ones_like(inputs))

    def test_bad_shapes_private_axes_money_independent_projection_and_overflow_reject(self):
        raw=np.ones((1,2,417),np.float32); projection=make_public_sketch(1)
        for bad in (raw.astype(np.float64),raw[:,0],raw[:,:,:,None],raw*np.inf):
            with self.assertRaises(ValueError): sketch_public_context(bad,projection)
        for bad in (projection.astype(np.float64),projection[:416],projection*2):
            with self.assertRaises(ValueError): sketch_public_context(raw,bad)
        for gain in (0.,-1.,float('inf'),True,1e40):
            with self.assertRaises(ValueError): sketch_public_context(raw,projection,gain)
        with self.assertRaises(ValueError): make_public_sketch(True)
        with self.assertRaises(ValueError): make_public_sketch(-1)
        with self.assertRaises(ValueError): match_training_sketch_gain(raw,np.zeros((1,2,64),np.float32),projection)
        payload=export_context_head(ExternalContextHead())
        inputs=np.ones((1,2,1326,256),np.float32); context=np.ones((1,2,64),np.float32)
        bad=inputs.copy(); bad[0,0,1,0]=2
        with self.assertRaises(ValueError): external_context_dense_forward(bad,context,payload)
        with self.assertRaises(ValueError): external_context_dense_forward(inputs,context[:,:,None,:],payload)
        context[0,0,0]=np.nan
        with self.assertRaises(ValueError): external_context_dense_forward(inputs,context,payload)


if __name__=='__main__': unittest.main()
