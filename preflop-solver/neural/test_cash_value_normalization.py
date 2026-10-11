"""Unit conversion, cache isolation and provenance for opt-in cash scaling."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np

from cash_feature_cache import CashFeatureCache
from cash_value_dataset import build_dataset
from cash_value_initialization import initialize_cash_from_frozen
from test_cash_value_dataset import label_fixture
import test_cash_value_initialization as initialization_fixture
import train_cash_value_network as training


class CashValueNormalizationTests(unittest.TestCase):
    def test_unknown_normalization_fails_before_source_io(self):
        for invalid in ('depth','',None,True,[],1):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError,'cash value normalization'):
                training.run(Path('must-not-read'),Path('must-not-create'),7101,1,value_normalization=invalid)

    def test_feature_scaling_preserves_real_bb_targets_and_every_other_feature(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'unit-label.json'
            path.write_text(json.dumps(label_fixture()))
            source = build_dataset([path])
            dimensions = training.OwnComboValueNetwork()
            context = np.zeros((2,dimensions.context_tower.layers[0].weight.shape[1]))
            query = np.zeros((2,1326,dimensions.query_tower.layers[0].weight.shape[1]))
            baseline = np.asarray(source['labels'][0]['counterfactual_values_bb'])
            with (patch.object(training,'build_features',return_value=(context,query)),
                  patch.object(training,'exact_cash_checkdown_features',return_value=(baseline,np.full((2,1326),.5)))):
                original = training.feature_arrays(source)
                constant = training.feature_arrays(source,value_normalization='payoff-exposure')
            self.assertEqual(original[3][0],38)
            self.assertEqual(constant[3][0],20)
            for index in (0,1,2,4,5,7):
                np.testing.assert_array_equal(original[index],constant[index])
            target = np.asarray(source['labels'][0]['counterfactual_values_bb']).reshape(-1)
            np.testing.assert_allclose(original[6][0]*original[3][0],target,atol=1e-7)
            np.testing.assert_allclose(constant[6][0]*constant[3][0],target,atol=1e-7)
            self.assertEqual(source['labels'][0]['input']['state']['invested_bb'],[19,19])

    def test_cache_keys_separate_scales_without_changing_the_legacy_key(self):
        schema = training.FEATURE_SCHEMA_BOARD_RELATIVE
        self.assertEqual(training.cash_feature_cache_key(schema,'pot'),schema)
        alternate = training.cash_feature_cache_key(schema,'payoff-exposure')
        self.assertNotEqual(alternate,schema)
        cache = CashFeatureCache()
        arrays = lambda:tuple(np.ones((2,),dtype=np.float32) for _ in range(8))
        _,first = cache.get('a'*64,schema,arrays)
        _,second = cache.get('a'*64,alternate,arrays)
        self.assertFalse(first['cache_hit'])
        self.assertFalse(second['cache_hit'])
        _,third = cache.get('a'*64,alternate,arrays)
        self.assertTrue(third['cache_hit'])

    def test_zero_residual_preserves_exact_baseline_for_both_normalizations(self):
        with mx.stream(mx.cpu):
            predictions = []
            parameters = []
            for normalization,scale in (('pot',38.),('payoff-exposure',20.)):
                mx.random.seed(7101)
                model = training.OwnComboValueNetwork('wide-pooled',value_normalization=normalization)
                context = mx.zeros((1,2,model.context_tower.layers[0].weight.shape[1]))
                query = mx.zeros((1,2,1326,model.query_tower.layers[0].weight.shape[1]))
                legal = np.ones((1,1326),dtype=np.float32); legal[:,0] = 0
                baseline = np.full((1,2,1326),-.86,dtype=np.float32); baseline[:,:,0] = 0
                features = [context,query,mx.full((1,2,1326),1/1326),mx.array([scale]),mx.array(baseline),mx.array(legal)]
                predictions.append(training.reference_predictions(model,features,np.array([scale],np.float32),legal))
                parameters.append([np.array(value) for _,value in tree_flatten(model.parameters())])
                np.testing.assert_allclose(predictions[-1],baseline,atol=1e-7)
            for left,right in zip(parameters[0],parameters[1]):
                np.testing.assert_array_equal(left,right)
            np.testing.assert_allclose(predictions[0],predictions[1],atol=1e-7)

    def test_exports_use_the_actual_model_scale_and_do_not_shift_payoffs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'label.json'; path.write_text(json.dumps(label_fixture()))
            source = build_dataset([path])
            for normalization in ('pot','payoff-exposure'):
                model = training.OwnComboValueNetwork('wide-pooled',value_normalization=normalization)
                destination = Path(directory)/f'{normalization}.json'
                training.export_cash_model(model,destination,7101,source,'a'*64)
                payload = json.loads(destination.read_bytes())
                self.assertEqual(payload['valueNormalization'],normalization)
                self.assertEqual(payload['targetScaleBb'],20)
                self.assertEqual(payload['projection'],'independent-full-stack-clip-and-board-mask-no-zero-sum')
                self.assertEqual(payload['cashRules'],source['game']['cash_rules'])

    def test_matching_scale_loads_and_a_cross_scale_warm_start_fails_before_loading(self):
        with tempfile.TemporaryDirectory() as directory, mx.stream(mx.cpu):
            original,network,dataset,source,splits = initialization_fixture.CashValueInitializationTests().fixture(Path(directory))
            # Unit fixture only: export the same known tensors under a distinct
            # scale. Production pilots must cold-fit or supply matching lineage.
            alternate = training.OwnComboValueNetwork('wide-pooled',value_normalization='payoff-exposure')
            alternate.update(original.parameters())
            digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
            training.export_cash_model(alternate,network,7101,source,digest)
            report_path = Path(directory)/'report.json'
            report = json.loads(report_path.read_bytes())
            report.update(network_sha256=hashlib.sha256(network.read_bytes()).hexdigest(),value_normalization='payoff-exposure')
            report_path.write_text(json.dumps(report))
            loaded = training.OwnComboValueNetwork('wide-pooled',value_normalization='payoff-exposure')
            initialize_cash_from_frozen(loaded,network,dataset,source,splits,7101,937)
            for (_,expected),(_,actual) in zip(tree_flatten(alternate.parameters()),tree_flatten(loaded.parameters())):
                np.testing.assert_array_equal(np.array(expected),np.array(actual))
            mismatched = training.OwnComboValueNetwork('wide-pooled')
            before = [np.array(value).copy() for _,value in tree_flatten(mismatched.parameters())]
            with self.assertRaisesRegex(ValueError,'normalization|contract'):
                initialize_cash_from_frozen(mismatched,network,dataset,source,splits,7101,937)
            for expected,(_,actual) in zip(before,tree_flatten(mismatched.parameters())):
                np.testing.assert_array_equal(np.array(expected),np.array(actual))

    def test_actual_fit_uses_isolated_scale_features_and_records_the_opt_in(self):
        with tempfile.TemporaryDirectory() as directory, mx.stream(mx.cpu):
            _,_,dataset,_,_ = initialization_fixture.CashValueInitializationTests().fixture(Path(directory))
            dimensions = training.OwnComboValueNetwork()
            context = np.zeros((2,dimensions.context_tower.layers[0].weight.shape[1]))
            query = np.zeros((2,1326,dimensions.query_tower.layers[0].weight.shape[1]))
            def baseline(board,ranges,investments,rules):
                from native_value_dataset import legal_combos
                legal = legal_combos(board)
                return np.tile(legal*-.86,(2,1)),np.tile(legal*.5,(2,1))
            with (patch.object(training,'_FEATURE_CACHE',CashFeatureCache()),
                  patch.object(training,'build_features',return_value=(context,query)),
                  patch.object(training,'exact_cash_checkdown_features',side_effect=baseline)):
                reports = []
                for i,normalization in enumerate(('pot','payoff-exposure','payoff-exposure')):
                    report = training.run(dataset,Path(directory)/f'fit-{i}',7101,1,split_seed=937,
                        training_output_mode='raw',value_normalization=normalization)
                    reports.append(report)
                    payload = json.loads((Path(directory)/f'fit-{i}/value-network.json').read_bytes())
                    self.assertEqual(payload['valueNormalization'],normalization)
                    self.assertFalse(report['active'])
                    self.assertEqual(report['feature_preparation']['feature_schema'],training.FEATURE_SCHEMA_BOARD_RELATIVE)
                    if normalization == 'pot':
                        self.assertNotIn('value_normalization',report)
                        self.assertNotIn('cache_feature_key',report['feature_preparation'])
                    else:
                        self.assertEqual(report['value_normalization'],normalization)
                        self.assertEqual(report['feature_preparation']['value_normalization'],normalization)
                self.assertFalse(reports[0]['feature_preparation']['cache_hit'])
                self.assertFalse(reports[1]['feature_preparation']['cache_hit'])
                self.assertTrue(reports[2]['feature_preparation']['cache_hit'])


if __name__ == '__main__':
    unittest.main()
