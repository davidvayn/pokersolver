import unittest
from dataclasses import replace
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile

import numpy as np

from depth_repair import calibration_metrics, independent_response_gain, select_street_roots, fit_frozen_value_head
from run_depth_pilots import PILOT_GRID
from train import (INPUT_FEATURE_COUNT, RunConfig, run_config_payload, retain_sampling_policy,
                   expand_state_action, ingest_records)


class FrozenQualificationTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('POKER_CERTIFICATE_TEST_BINARY'),'explicit native binary required')
    def test_frozen_evaluation_is_deterministic_and_rejects_wrong_game_or_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); binary=os.environ['POKER_CERTIFICATE_TEST_BINARY']
            grid={**PILOT_GRID,'preflop_raise_cap':1}
            network={'layers':[{'input_size':INPUT_FEATURE_COUNT,'output_size':1,'activation':'linear',
                'weights':[0.]*INPUT_FEATURE_COUNT,'biases':[0.]}]}
            source={'schema':'hu-neural-training-networks-v4','input_size':INPUT_FEATURE_COUNT,
                'strategy_transform':'softmax','networks':[network,network],
                'qualification_game':{'depth_bb':50,'action_abstraction':grid}}
            model=root/'model.json'; model.write_text(json.dumps(source))
            samples=root/'sample.gz'
            subprocess.run([binary,'neural-samples','--networks',str(model),'--sample-trajectories',
                '--traversals','4','--effective-stack-bb','50','--seed','17321','--sample-turn-rivers',
                '--action-abstraction-json',json.dumps(grid),'--output',str(samples)],
                capture_output=True,check=True,timeout=30)
            with gzip.open(samples,'rt') as stream: rows=[json.loads(line) for line in stream][1:]
            input_path=root/'input.json'; input_path.write_text(json.dumps(rows[:3]))
            argv=[binary,'neural-frozen-values','--networks',str(model),'--input',str(input_path),
                '--effective-stack-bb','50','--hidden-blocks','8','--rollouts-per-block','2',
                '--action-abstraction-json',json.dumps(grid),'--seed','17322']
            first=subprocess.run(argv,capture_output=True,text=True,check=True,timeout=30)
            second=subprocess.run(argv,capture_output=True,text=True,check=True,timeout=30)
            self.assertEqual(first.stdout,second.stdout)
            report=json.loads(first.stdout)
            self.assertEqual(len(report['records']),3)
            self.assertEqual(len(report['records'][0]['independent_hidden_block_values_bb']),8)
            rows[0]['state']['raise_reopened']=not rows[0]['state']['raise_reopened']
            input_path.write_text(json.dumps(rows[:3]))
            self.assertNotEqual(subprocess.run(argv,capture_output=True,timeout=30).returncode,0)
            input_path.write_text(json.dumps(rows[1:3]))
            source['qualification_game']['depth_bb']=100; model.write_text(json.dumps(source))
            self.assertNotEqual(subprocess.run(argv,capture_output=True,timeout=30).returncode,0)

    def test_frozen_refit_keeps_families_disjoint_and_never_calls_interval_width_sampling_se(self):
        state = {'private_cards':[48,49], 'board':[], 'street':'preflop', 'actor':0,'button':0,
            'pot_bb':0.,'stacks_bb':[49.5,49.],'street_bets_bb':[.5,1.],
            'total_committed_bb':[.5,1.],'to_call_bb':.5,'last_full_raise_bb':1.,'raise_reopened':True,'trajectory':[]}
        records=[{'iteration':i,'state':state,'actions':[{'kind':'fold','amount_to_bb':None}]} for i in range(8)]
        reference={'records':[{'action_values_bb':[-.5],'standard_errors_bb':[.01]} for _ in records],
            'source_network_sha256s':['a'*64],'input_sha256':'b'*64}
        with tempfile.TemporaryDirectory() as directory:
            report=fit_frozen_value_head(records,reference,50,Path(directory),steps=2)
            train,calibration,test=map(set,report['family_splits'])
            self.assertFalse(train & calibration or train & test or calibration & test)
            self.assertEqual(report['status'],'research_not_activated')
            self.assertTrue(np.isfinite(report['empirical_prediction_interval_95_half_width_bb']))
            self.assertNotIn('standard_error_bb',report)

    def test_sampling_history_includes_uniform_start_and_correct_player_iteration_weights(self):
        config=RunConfig(schema='test',depth_bb=50,seed=1,reservoir_capacity=100,
            hidden_sizes=(8,4),batch_size=4,learning_rate=.001,learning_rate_final=None,
            learning_rate_decay_start_round=None,learning_rate_decay_end_round=None,
            traversals_per_round=4,steps_per_round=1,advantage_alpha=2.,variance_baseline_scale=.5,
            replay_street_proposal=None,value_rollouts_per_action=4,artifact_every=1,
            preflop_runout_samples=4,flop_runout_samples=4,exact_turn_rivers=False,compact_serving_grid=False)
        self.assertNotIn('regret_training',run_config_payload(config))
        self.assertNotIn('average_regret_rollouts',run_config_payload(config))
        self.assertNotIn('retain_policy_snapshots',run_config_payload(config))
        self.assertNotEqual(run_config_payload(config),run_config_payload(replace(config,average_regret_rollouts=True)))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            retain_sampling_policy(root,config,1,0,None)
            entries=json.loads((root/'policy-history/manifest.json').read_text())
            self.assertEqual(entries[0]['weights_by_player'],[20.,10.])
            initial=json.loads(Path(entries[0]['path']).read_text())
            self.assertTrue(all(v==0 for v in initial['networks'][0]['layers'][0]['weights']))
            retain_sampling_policy(root,replace(config,regret_training='retained_instantaneous'),3,8,None)
            entries=json.loads((root/'policy-history/manifest.json').read_text())
            self.assertEqual(entries[1]['weights_by_player'],[6.,6.])

    def test_raw_regrets_do_not_bootstrap_a_network_prior_and_use_linear_iteration_weights(self):
        state = {'private_cards':[48,49], 'board':[], 'street':'preflop', 'actor':0,'button':0,
            'pot_bb':0.,'stacks_bb':[49.5,49.],'street_bets_bb':[.5,1.],
            'total_committed_bb':[.5,1.],'to_call_bb':.5,'last_full_raise_bb':1.,'raise_reopened':True,'trajectory':[]}
        actions=[{'kind':'fold','amount_to_bb':None},{'kind':'call','amount_to_bb':None}]
        features=np.stack([expand_state_action(state,a,50) for a in actions])
        record={'kind':'advantage_p0','state':state,'actions':actions,'targets':[-2.,2.],
            'weight':1.,'feature_sha256':[hashlib.sha256(np.rint(f.astype(np.float64)*1e6).astype('<i4').tobytes()).hexdigest() for f in features]}
        # An empty model mapping makes accidental bootstrapping fail immediately.
        heldout=ingest_records([record],50,{}, {},7,2.,random.Random(1),regret_training='retained_instantaneous')
        np.testing.assert_allclose(heldout['advantage_p0'][0][1],[-.04,.04])
        self.assertEqual(heldout['advantage_p0'][0][2],7.)

    @unittest.skipUnless(os.environ.get('POKER_CERTIFICATE_TEST_BINARY'),'explicit native binary required')
    def test_actual_native_regret_targets_use_the_extra_rollouts(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'samples.gz'
            subprocess.run([os.environ['POKER_CERTIFICATE_TEST_BINARY'],'neural-samples','--traversals','4',
                '--effective-stack-bb','50','--seed','17301','--max-records','50000',
                '--value-rollouts-per-action','4','--average-regret-rollouts','--integrate-closed-opponent-actions','--sample-turn-rivers',
                '--action-abstraction-json',json.dumps({**PILOT_GRID,'preflop_raise_cap':1}),
                '--output',str(output)],check=True,capture_output=True,timeout=30)
            with gzip.open(output,'rt') as stream: rows=[json.loads(line) for line in stream]
            self.assertEqual(rows[0]['regret_target_estimator'],'averaged_action_values_v1')
            self.assertTrue(rows[0]['integrates_closed_opponent_actions'])
            advantages=[r for r in rows[1:] if r['kind'].startswith('advantage')]
            self.assertTrue(advantages)
            for record in advantages:
                values=np.asarray(record['action_values_bb'])
                np.testing.assert_allclose(record['targets'],values-values.mean(),atol=2e-5)
            baseline=Path(directory)/'baseline.gz'
            subprocess.run([os.environ['POKER_CERTIFICATE_TEST_BINARY'],'neural-samples','--traversals','4',
                '--effective-stack-bb','50','--seed','17301','--max-records','50000',
                '--value-rollouts-per-action','4','--average-regret-rollouts','--sample-turn-rivers',
                '--action-abstraction-json',json.dumps({**PILOT_GRID,'preflop_raise_cap':1}),
                '--output',str(baseline)],check=True,capture_output=True,timeout=30)
            with gzip.open(baseline,'rt') as stream: original=[json.loads(line) for line in stream][1:]
            # Integration changes estimates, not primary chance/visitation or
            # average-policy records when the frozen behavior policy is equal.
            keys=('kind','iteration','state','actions','feature_sha256','weight','reach_probability')
            self.assertEqual([{k:r[k] for k in keys} for r in rows[1:]],
                             [{k:r[k] for k in keys} for r in original])

    def test_confidence_includes_prediction_error_not_just_rollout_noise(self):
        result = calibration_metrics(np.array([[4., 0.]]), np.array([[0., 0.]]),
                                     np.array([[.001, .001]]), np.array([[.001, .001]]))
        self.assertEqual(result['predicted_precision_coverage'], 1.)
        self.assertEqual(result['empirical_error_at_most_0_02_fraction'], .5)
        self.assertEqual(result['nominal_95_interval_coverage'], .5)
        self.assertFalse(result['qualified'])

    def test_response_selection_and_measurement_use_different_hidden_blocks(self):
        blocks = np.array([[10., 0.], [10., 0.], [-2., 0.], [-2., 0.]])
        result = independent_response_gain(blocks, np.array([.5, .5]))
        self.assertEqual(result['selected_action'], 0)
        self.assertAlmostEqual(result['heldout_gain_bb'], -1.)
        self.assertAlmostEqual(result['selection_gain_bb'], 5.)

    def test_street_root_selection_is_bounded_and_deterministic(self):
        records = [{'iteration': i, 'state': {'street': street}} for i in range(16)
                   for street in ('preflop', 'flop', 'turn', 'river')]
        selected = select_street_roots(records, 2)
        self.assertEqual(len(selected), 8)
        self.assertEqual(selected, select_street_roots(records, 2))


if __name__ == '__main__':
    unittest.main()
