"""Public-state expert routing without changing cash payoffs or cold RNG."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np

import cash_value_dataset as datasets
from cash_feature_cache import CashFeatureCache
from cash_value_initialization import initialize_cash_from_frozen
import test_cash_value_initialization as initialization_fixture
import train_cash_value_network as training
import validate_cash_value_parity as parity

ARCHITECTURE = 'wide-pooled-pot-experts'


class CashPotExpertTests(unittest.TestCase):
    def inputs(self, model, investments):
        count = len(investments)
        context = np.zeros((count,2,model.context_tower.layers[0].weight.shape[1]),np.float32)
        context[:,:,19:21] = np.asarray(investments,np.float32)[:,None,None]/20
        query = mx.zeros((count,2,1326,model.query_tower.layers[0].weight.shape[1]))
        weights = mx.full((count,2,1326),1/1326)
        scales = mx.array(np.asarray(investments,np.float32)*2)
        baseline = mx.full((count,2,1326),-.86)
        return mx.array(context),query,weights,scales,baseline

    def test_cold_shared_encoder_and_all_heads_match_control_without_new_rng_draws(self):
        with mx.stream(mx.cpu):
            mx.random.seed(7101); control = training.OwnComboValueNetwork('wide-pooled')
            control_next = np.array(mx.random.uniform(shape=(8,)))
            mx.random.seed(7101); candidate = training.OwnComboValueNetwork(ARCHITECTURE)
            candidate_next = np.array(mx.random.uniform(shape=(8,)))
            np.testing.assert_array_equal(control_next,candidate_next)
            self.assertTrue(candidate.uses_pot_experts)
            self.assertTrue(candidate.pools_exact_ranges)
            self.assertIsNone(candidate.head)
            self.assertEqual(len(candidate.pot_expert_heads),3)
            for name in ('context_tower','query_tower'):
                for (_,left),(_,right) in zip(tree_flatten(getattr(control,name).parameters()),tree_flatten(getattr(candidate,name).parameters())):
                    np.testing.assert_array_equal(np.array(left),np.array(right))
            for head in candidate.pot_expert_heads:
                for (_,left),(_,right) in zip(tree_flatten(control.head.parameters()),tree_flatten(head.parameters())):
                    np.testing.assert_array_equal(np.array(left),np.array(right))
            inputs = self.inputs(candidate,[3.48,3.52,7.48,7.52])
            np.testing.assert_allclose(np.array(candidate.raw_values(*inputs))*np.array(inputs[3])[:,None,None],-.86,atol=1e-7)
            candidate.pot_expert_heads[0].layers[-1].bias = mx.array([1.])
            self.assertEqual(float(candidate.pot_expert_heads[1].layers[-1].bias.item()),0.)

    def test_cent_threshold_neighbors_route_one_head_keep_mask_and_independent_clip(self):
        with mx.stream(mx.cpu):
            model = training.OwnComboValueNetwork(ARCHITECTURE)
            for index,head in enumerate(model.pot_expert_heads):
                head.layers[-1].bias = mx.array([index*.1])
            inputs = self.inputs(model,[3.48,3.52,7.48,7.52])
            expected = -.86 + np.array([0,.1,.1,.2])*np.array(inputs[3])
            raw_bb = np.array(model.raw_values(*inputs))*np.array(inputs[3])[:,None,None]
            np.testing.assert_allclose(raw_bb,expected[:,None,None]+np.zeros((4,2,1326)),atol=1e-6)
            for head in model.pot_expert_heads:
                head.layers[-1].bias = mx.array([10.])
            legal = np.ones((4,1326),np.float32); legal[:,0] = 0
            served = np.array(model(*inputs,mx.array(legal))).reshape((4,2,1326))*np.array(inputs[3])[:,None,None]
            np.testing.assert_allclose(served[:,:,1:],20.,atol=1e-6)
            np.testing.assert_array_equal(served[:,:,0],0.)
            # Both players may have the same signed value: no Home zero-sum shift.
            self.assertGreater(float(served.sum()),0.)

    def test_raw_bb_gradient_only_updates_the_active_head(self):
        with mx.stream(mx.cpu):
            model = training.OwnComboValueNetwork(ARCHITECTURE)
            inputs = self.inputs(model,[3.52])
            def loss(current):
                values = current.raw_values(*inputs)*inputs[3][:,None,None]
                return mx.mean(mx.square(values-1.))
            _,gradients = nn.value_and_grad(model,loss)(model)
            mx.eval(gradients)
            for index,head in enumerate(gradients['pot_expert_heads']):
                magnitude = sum(float(mx.sum(mx.abs(value)).item()) for _,value in tree_flatten(head))
                self.assertGreater(magnitude,0.) if index==1 else self.assertEqual(magnitude,0.)

    def test_export_and_strict_all_or_nothing_loading_use_cash_expert_contract(self):
        with tempfile.TemporaryDirectory() as directory, mx.stream(mx.cpu):
            _,network,dataset,source,splits = initialization_fixture.CashValueInitializationTests().fixture(Path(directory))
            mx.random.seed(7101); original = training.OwnComboValueNetwork(ARCHITECTURE)
            for index,head in enumerate(original.pot_expert_heads):
                head.layers[-1].bias = mx.array([index*.1])
            digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
            training.export_cash_model(original,network,7101,source,digest)
            payload = json.loads(network.read_bytes())
            self.assertEqual(payload['schema'],datasets.POT_EXPERT_NETWORK_SCHEMA)
            self.assertEqual(payload['predictionContract'],datasets.POT_EXPERT_CONTRACT)
            self.assertEqual(payload['head'],[])
            self.assertEqual(len(payload['potExpertHeads']),3)
            self.assertEqual(payload['projection'],'independent-full-stack-clip-and-board-mask-no-zero-sum')
            report_path = Path(directory)/'report.json'
            report = json.loads(report_path.read_bytes())
            report['network_sha256'] = hashlib.sha256(network.read_bytes()).hexdigest()
            report_path.write_text(json.dumps(report))
            target = training.OwnComboValueNetwork(ARCHITECTURE)
            initialize_cash_from_frozen(target,network,dataset,source,splits,7101,937)
            for (name,left),(actual_name,right) in zip(tree_flatten(original.parameters()),tree_flatten(target.parameters())):
                self.assertEqual(name,actual_name)
                np.testing.assert_array_equal(np.array(left),np.array(right))
            changes = (lambda p:p.update(potExpertHeads=p['potExpertHeads'][:2]),
                       lambda p:p.update(head=p['potExpertHeads'][0]),
                       lambda p:p.update(predictionContract=datasets.PREDICTION_CONTRACT),
                       lambda p:p['potExpertHeads'][2][-1].update(biases=[float('nan')]),
                       lambda p:p['potExpertHeads'][2][-1].update(activation='relu'))
            for change in changes:
                damaged = copy.deepcopy(payload); change(damaged)
                network.write_text(json.dumps(damaged))
                report['network_sha256'] = hashlib.sha256(network.read_bytes()).hexdigest()
                report_path.write_text(json.dumps(report))
                target = training.OwnComboValueNetwork(ARCHITECTURE)
                before = [np.array(value).copy() for _,value in tree_flatten(target.parameters())]
                with self.assertRaises(ValueError):
                    initialize_cash_from_frozen(target,network,dataset,source,splits,7101,937)
                for expected,(_,actual) in zip(before,tree_flatten(target.parameters())):
                    np.testing.assert_array_equal(expected,np.array(actual))

    def test_bounded_full_pass_fit_exports_inactive_three_head_model(self):
        # Synthetic unittest targets exercise the actual optimizer, not a poker reference.
        with tempfile.TemporaryDirectory() as directory, mx.stream(mx.cpu):
            _,_,dataset,source,_ = initialization_fixture.CashValueInitializationTests().fixture(Path(directory))
            model = training.OwnComboValueNetwork(ARCHITECTURE)
            inputs = self.inputs(model,[2.,4.,8.,12.])
            legal = np.ones((4,1326),np.float32)
            arrays = tuple(np.array(value) for value in inputs)+(legal,np.full((4,2652),-.1,np.float32),np.ones((4,2652),np.float32))
            output = Path(directory)/'fit'
            with (patch.object(training,'_FEATURE_CACHE',CashFeatureCache()),
                  patch.object(training,'feature_arrays',return_value=arrays)):
                report = training.run(dataset,output,7101,2,split_seed=937,
                    architecture=ARCHITECTURE,gradient_batch_size=1,training_output_mode='raw')
            payload = json.loads((output/'value-network.json').read_bytes())
            self.assertFalse(report['active'])
            self.assertEqual(report['gradient_batching']['batch_size'],1)
            self.assertEqual(payload['schema'],datasets.POT_EXPERT_NETWORK_SCHEMA)
            self.assertEqual(len(payload['potExpertHeads']),3)
            self.assertEqual(payload['head'],[])

    def test_numpy_and_mlx_frozen_parity_support_all_heads_and_actual_scale(self):
        with tempfile.TemporaryDirectory() as directory, mx.stream(mx.cpu):
            _,network,_,source,_ = initialization_fixture.CashValueInitializationTests().fixture(Path(directory))
            for normalization in ('pot','payoff-exposure'):
                model = training.OwnComboValueNetwork(ARCHITECTURE,value_normalization=normalization)
                for index,head in enumerate(model.pot_expert_heads):
                    head.layers[-1].bias = mx.array([index*.1])
                training.export_cash_model(model,network,7101,source,'a'*64)
                payload = json.loads(network.read_bytes())
                inputs = self.inputs(model,[3.48,3.52,7.48,7.52])
                if normalization != 'pot':
                    inputs = (*inputs[:3],mx.full((4,),20.),inputs[4])
                arrays = tuple(np.array(value) for value in inputs)+(np.ones((4,1326),np.float32),None,None)
                with patch.object(parity,'feature_arrays',return_value=arrays) as prepared:
                    dense = parity.python_predictions(source,payload)
                    actual = parity.mlx_predictions(source,payload,mx.cpu)
                    np.testing.assert_allclose(dense,actual,atol=2e-6,rtol=0)
                    if normalization != 'pot':
                        self.assertEqual(prepared.call_args.kwargs['value_normalization'],normalization)
                damaged = copy.deepcopy(payload); damaged['potExpertHeads'] = damaged['potExpertHeads'][:2]
                with self.assertRaises(ValueError):
                    parity.cash_architecture(damaged)


if __name__ == '__main__':
    unittest.main()
