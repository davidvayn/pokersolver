"""Strict opt-in cash kernel network; same tensors/payoffs, learned interaction."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
import cash_value_dataset as datasets
from cash_value_initialization import initialize_cash_from_frozen
import test_cash_value_initialization as fixtures
import train_cash_value_network as training
import validate_cash_value_parity as parity

ARCHITECTURE='wide-kernel-pooled'


class CashKernelNetworkTests(unittest.TestCase):
    def arrays(self,model):
        context=np.zeros((1,2,model.context_tower.layers[0].weight.shape[1]),np.float32)
        context[:,:,19:21]=.1
        queries=np.zeros((1,2,1326,model.query_tower.layers[0].weight.shape[1]),np.float32)
        queries[0,0,:,94]=np.linspace(0,1,1326)
        queries[0,1,:,94]=np.linspace(1,0,1326)
        ranges=np.full((1,2,1326),1/1326,np.float32)
        legal=np.ones((1,1326),np.float32); legal[:,0]=0
        baseline=np.full((1,2,1326),-.86,np.float32); baseline[:,:,0]=0
        return context,queries,ranges,np.array([4.],np.float32),baseline,legal,None,None

    def test_zero_residual_and_shared_parameters_preserve_cold_baseline(self):
        with mx.stream(mx.cpu):
            mx.random.seed(7101); control=training.OwnComboValueNetwork('wide-blocker-pooled')
            mx.random.seed(7101); candidate=training.OwnComboValueNetwork(ARCHITECTURE)
            for name in ('context_tower','query_tower','head'):
                for (_,left),(_,right) in zip(tree_flatten(getattr(control,name).parameters()),tree_flatten(getattr(candidate,name).parameters())):
                    np.testing.assert_array_equal(np.array(left),np.array(right))
            self.assertEqual(candidate.kernel_query.layers[0].weight.shape,(8,64))
            self.assertEqual(candidate.kernel_key.layers[0].weight.shape,(8,64))
            arrays=self.arrays(candidate)
            inputs=[mx.array(v) for v in arrays[:6]]+[mx.array(arrays[2])]
            actual=np.array(candidate(*inputs)).reshape((1,2,1326))*4
            np.testing.assert_allclose(actual,arrays[4],atol=1e-7)
            with self.assertRaisesRegex(ValueError,'raw exact ranges'):
                candidate.raw_values(*inputs[:5])

    def test_exports_and_numpy_mlx_parity_use_explicit_cash_kernel_contract(self):
        with tempfile.TemporaryDirectory() as directory,mx.stream(mx.cpu):
            _,network,_,source,_=fixtures.CashValueInitializationTests().fixture(Path(directory))
            model=training.OwnComboValueNetwork(ARCHITECTURE)
            model.head.layers[-1].weight=mx.full(model.head.layers[-1].weight.shape,.1)
            model.head.layers[-1].bias=mx.array([-.1])
            training.export_cash_model(model,network,7101,source,'a'*64)
            payload=json.loads(network.read_bytes())
            self.assertEqual(payload['schema'],datasets.KERNEL_POOLED_NETWORK_SCHEMA)
            self.assertEqual(payload['predictionContract'],datasets.KERNEL_POOLED_CONTRACT)
            self.assertEqual(payload['kernelFeatureMap'],'elu-plus-one-v1')
            self.assertEqual(payload['kernelRank'],8)
            self.assertEqual(payload['kernelQuery'][0]['outputSize'],8)
            self.assertEqual(payload['kernelKey'][0]['activation'],'linear')
            arrays=self.arrays(model)
            source['labels']=[dict(input=dict(state=dict(ranges=arrays[2][0].tolist())))]
            with patch.object(parity,'feature_arrays',return_value=arrays):
                dense=parity.python_predictions(source,payload)
                cpu=parity.mlx_predictions(source,payload,mx.cpu)
            np.testing.assert_allclose(dense,cpu,atol=5e-6,rtol=0)
            self.assertEqual(payload['projection'],'independent-full-stack-clip-and-board-mask-no-zero-sum')
            for change in (lambda p:p.update(predictionContract=datasets.PREDICTION_CONTRACT),
                           lambda p:p.update(kernelRank=4),lambda p:p.update(kernelFeatureMap='softmax')):
                bad=copy.deepcopy(payload); change(bad)
                with self.assertRaises(ValueError): parity.cash_architecture(bad)
            for change in (lambda p:p.update(kernelQuery=[]),lambda p:p['kernelKey'][0].update(residual=True),
                           lambda p:p['kernelQuery'][0].update(normalization='layer'),
                           lambda p:p.update(schema=datasets.BLOCKER_POOLED_NETWORK_SCHEMA,architecture='wide-blocker-pooled',
                                predictionContract=datasets.BLOCKER_POOLED_CONTRACT)):
                bad=copy.deepcopy(payload); change(bad)
                with self.assertRaises(ValueError): parity.cash_architecture(bad)

    def test_warm_load_stages_every_kernel_tensor_and_rejects_a_silent_downgrade(self):
        with tempfile.TemporaryDirectory() as directory,mx.stream(mx.cpu):
            _,network,dataset,source,splits=fixtures.CashValueInitializationTests().fixture(Path(directory))
            model=training.OwnComboValueNetwork(ARCHITECTURE)
            training.export_cash_model(model,network,7101,source,hashlib.sha256(dataset.read_bytes()).hexdigest())
            payload=json.loads(network.read_bytes()); report_path=Path(directory)/'report.json'
            report=json.loads(report_path.read_bytes()); report['network_sha256']=hashlib.sha256(network.read_bytes()).hexdigest()
            report_path.write_text(json.dumps(report))
            target=training.OwnComboValueNetwork(ARCHITECTURE)
            initialize_cash_from_frozen(target,network,dataset,source,splits,7101,937)
            for (name,left),(other,right) in zip(tree_flatten(model.parameters()),tree_flatten(target.parameters())):
                self.assertEqual(name,other); np.testing.assert_array_equal(np.array(left),np.array(right))
            for change in (lambda p:p.update(kernelQuery=[]),lambda p:p.update(kernelRank=16),
                           lambda p:p['kernelKey'][0].update(biases=[float('nan')]*8),
                           lambda p:p['kernelKey'][0].update(residual=True),
                           lambda p:p['kernelQuery'][0].update(normalization='layer')):
                bad=copy.deepcopy(payload); change(bad); network.write_text(json.dumps(bad))
                report['network_sha256']=hashlib.sha256(network.read_bytes()).hexdigest(); report_path.write_text(json.dumps(report))
                target=training.OwnComboValueNetwork(ARCHITECTURE)
                before=[np.array(value).copy() for _,value in tree_flatten(target.parameters())]
                with self.assertRaises(ValueError): initialize_cash_from_frozen(target,network,dataset,source,splits,7101,937)
                for left,(_,right) in zip(before,tree_flatten(target.parameters())): np.testing.assert_array_equal(left,np.array(right))


if __name__=='__main__': unittest.main()
