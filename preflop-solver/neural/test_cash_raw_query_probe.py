import unittest

import mlx.core as mx
import numpy as np

from cash_raw_query_probe import raw_query_probe_features, RawQueryProbeHead
from train_public_value_network import tower_payload
from validate_cash_value_parity import dense_forward


class RawQueryProbeTests(unittest.TestCase):
    def test_true_and_zero_controls_preserve_all_axes_and_frozen_prefix(self):
        combined=np.arange(3*2*1326*256,dtype=np.float32).reshape(3,2,1326,256)/10000
        query=np.arange(3*2*1326*124,dtype=np.float32).reshape(3,2,1326,124)/10000
        combined.flags.writeable=False; query.flags.writeable=False
        true=raw_query_probe_features(combined,query,'raw-query')
        control=raw_query_probe_features(combined,query,'zero-query')
        self.assertEqual(true.shape,(3,2,1326,380)); self.assertEqual(true.dtype,np.float32)
        np.testing.assert_array_equal(true[...,:256],combined)
        np.testing.assert_array_equal(control[...,:256],combined)
        np.testing.assert_array_equal(true[...,256:],query)
        np.testing.assert_array_equal(control[...,256:],np.zeros_like(query))
        self.assertFalse(np.shares_memory(true,combined)); self.assertFalse(np.shares_memory(true,query))
        true[0,0,0,0]=-99.; true[0,0,0,256]=-99.
        self.assertEqual(combined[0,0,0,0],0.); self.assertEqual(query[0,0,0,0],0.)

    def test_rejects_mismatched_widths_axes_dtypes_and_nonfinite_values(self):
        c=np.zeros((1,2,1326,256),np.float32); q=np.zeros((1,2,1326,124),np.float32)
        cases=[(c,q,'guessed-mode'),(c.astype(np.float64),q,'raw-query'),
            (c,q.astype(np.float64),'zero-query'),(c[:,0],q,'raw-query'),
            (c,q[...,:123],'raw-query'),(c,np.zeros((2,2,1326,124),np.float32),'raw-query'),
            (c[:0],q[:0],'raw-query'),
            (np.broadcast_to(c,(513,2,1326,256)),np.broadcast_to(q,(513,2,1326,124)),'raw-query')]
        for inputs in cases:
            with self.assertRaises(ValueError): raw_query_probe_features(*inputs)
        for mode in ('raw-query','zero-query'):
            for invalid in (np.nan,np.inf,-np.inf):
                bad=q.copy(); bad[0,1,17,56]=invalid
                with self.assertRaises(ValueError): raw_query_probe_features(c,bad,mode)
                bad=c.copy(); bad[0,1,17,56]=invalid
                with self.assertRaises(ValueError): raw_query_probe_features(bad,q,mode)

    def test_same_size_heads_initialize_identically_and_start_at_zero(self):
        x=np.ones((1,2,1326,380),np.float32)
        heads=[]
        for _ in range(2):
            mx.random.seed(7101); model=RawQueryProbeHead(); mx.eval(model.parameters())
            heads.append(tower_payload(model.head,'relu','linear'))
            np.testing.assert_array_equal(np.asarray(model(mx.array(x))),np.zeros((1,2,1326)))
        self.assertEqual(heads[0],heads[1])
        self.assertEqual(heads[0][0]['inputSize'],380)
        self.assertEqual(heads[0][0]['outputSize'],32)

    def test_cpu_head_export_reproduces_prediction_at_the_real_input_width(self):
        mx.random.seed(7101); model=RawQueryProbeHead()
        model.head.layers[-1].weight=mx.full_like(model.head.layers[-1].weight,.01)
        model.head.layers[-1].bias=mx.full_like(model.head.layers[-1].bias,.4)
        mx.eval(model.parameters())
        x=np.linspace(-.5,.5,2*1326*380,dtype=np.float32).reshape(1,2,1326,380)
        with mx.stream(mx.cpu): actual=np.asarray(model(mx.array(x)))
        expected=dense_forward(x,tower_payload(model.head,'relu','linear')).reshape(1,2,1326)
        np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=0)


if __name__=='__main__': unittest.main()
