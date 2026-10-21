import unittest
import json
import os
import subprocess
import tempfile
from unittest.mock import patch
from pathlib import Path

import mlx.core as mx
import numpy as np

from train import INPUT_FEATURE_COUNT, STATE_FEATURE_COUNT, NETWORK_SCHEMA
from validate_seeds import (
    MAXIMUM_ESTIMATED_EXPLOITABILITY_BB_PER_HAND,
    MAXIMUM_ONE_SIDED_99_EXPLOITABILITY_UPPER_BOUND_BB_PER_HAND,
    StreetRoutedModel,
    action_ev_standard_error_summary,
    apply_weight_overrides,
    compare,
    select_exploitability_certificate,
    abstraction_arguments,
    verify_evaluation_game,
    evaluation_records,
    routing_compatible_config,
    served_action_value_audit,
)
from run_depth_pilots import PILOT_GRID


def state(private_cards):
    return {
        "private_cards": private_cards,
        "board": [],
        "street": "preflop",
        "actor": 0,
        "button": 0,
        "pot_bb": 0.0,
        "stacks_bb": [19.5, 19.0],
        "street_bets_bb": [0.5, 1.0],
        "total_committed_bb": [0.5, 1.0],
        "to_call_bb": 0.5,
        "last_full_raise_bb": 1.0,
        "raise_reopened": True,
        "trajectory": [],
    }


def record(private_cards):
    return {
        "state": state(private_cards),
        "actions": [
            {"kind": "fold", "amount_to_bb": None},
            {"kind": "call", "amount_to_bb": None},
        ],
    }


class UniformModel:
    def __call__(self, features):
        return mx.zeros((features.shape[0], 1))


class RankMarkerModel:
    """Prefers call only when canonical card zero is visible."""

    def __call__(self, features):
        values = np.asarray(features)
        marker = values[:, 0]
        fold = values[:, STATE_FEATURE_COUNT]
        call = values[:, STATE_FEATURE_COUNT + 2]
        logits = marker * (-4.0 * fold + 4.0 * call)
        return mx.array(logits.reshape((-1, 1)))


class ReachAwareValidationTests(unittest.TestCase):
    def test_value_audit_also_uses_native_aligned_cpu_inference(self):
        from contextlib import contextmanager
        active = [False]
        actual_stream = mx.stream
        @contextmanager
        def tracked_stream(device):
            self.assertEqual(device,mx.cpu)
            with actual_stream(device):
                active[0] = True
                try: yield
                finally: active[0] = False
        test = self
        class Value:
            def __call__(self,features):
                test.assertTrue(active[0],'value head was evaluated outside the CPU stream')
                return mx.zeros((features.shape[0],2))
        with patch('validate_seeds.mx.stream',tracked_stream):
            served_action_value_audit(UniformModel(),Value(),[[record([48,49])]],20)

    def test_artifact_json_source_cannot_differ_from_the_frozen_binary(self):
        from test_frozen_policy_values import FrozenPolicyValueTests
        from validate_seeds import verify_artifact
        with tempfile.TemporaryDirectory() as directory:
            source,binary = FrozenPolicyValueTests().fixture(directory)
            artifact = {'source':str(source),'binary':str(binary)}
            self.assertTrue(verify_artifact(artifact)['descriptor_matches'])
            data = json.loads(source.read_text()); data['parameters'][0] += 1
            source.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'source.*binary'):
                verify_artifact(artifact)

    def test_policy_comparisons_use_native_aligned_cpu_inference(self):
        from validate_seeds import policy
        with patch('validate_seeds.mx.stream',wraps=mx.stream) as stream:
            result = policy(UniformModel(),np.zeros((2,INPUT_FEATURE_COUNT),np.float32))
        stream.assert_called_with(mx.cpu)
        np.testing.assert_array_equal(result,[.5,.5])

    @unittest.skipUnless(os.environ.get('POKER_CERTIFICATE_TEST_BINARY'),'explicit native certificate binary required')
    def test_native_certificate_entry_point_preserves_custom_grid_identity(self):
        # Structural transport test, not an evaluation of trained poker strength.
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'uniform.json'
            network={'layers':[{'input_size':INPUT_FEATURE_COUNT,'output_size':1,'activation':'linear',
                'weights':[0.]*INPUT_FEATURE_COUNT,'biases':[0.]}]}
            source.write_text(json.dumps({'schema':NETWORK_SCHEMA,'input_size':INPUT_FEATURE_COUNT,
                'strategy_transform':'softmax','networks':[network,network]}))
            grid={**PILOT_GRID,'preflop_raise_cap':1}
            completed=subprocess.run([os.environ['POKER_CERTIFICATE_TEST_BINARY'],'neural-certificate',
                '--effective-stack-bb','50','--deals','2','--seed','73499','--threads','2',
                '--networks',str(source),'--action-abstraction-json',json.dumps(grid)],
                capture_output=True,text=True,check=True,timeout=30)
            certificate=json.loads(completed.stdout)
            verify_evaluation_game(certificate,50,grid)
            self.assertEqual(certificate['deals'],2)
            self.assertEqual(certificate['schema'],'hu-neural-clairvoyant-upper-bound-v1')

    def test_served_ev_head_precision_is_not_confused_with_reference_rollout_precision(self):
        class Value:
            def __init__(self,raw): self.raw=raw
            def __call__(self,features): return mx.tile(mx.array([[0.,self.raw]]),(features.shape[0],1))
        records=[[record([48,49]),record([44,45])]]
        good=served_action_value_audit(UniformModel(),Value(-10.),records,20)
        noisy=served_action_value_audit(UniformModel(),Value(0.),records,20)
        self.assertEqual(good['policy_action_weighted_predicted_precision_coverage'],1.)
        self.assertEqual(noisy['policy_action_weighted_predicted_precision_coverage'],0.)
        self.assertGreater(noisy['maximum_predicted_standard_error_bb'],.02)
        with self.assertRaises(RuntimeError):served_action_value_audit(UniformModel(),Value(float('nan')),records,20)

    def test_explicit_trained_grid_is_not_replaced_by_a_default_or_compact_tree(self):
        args=abstraction_arguments(False,PILOT_GRID)
        self.assertEqual(args[0],'--action-abstraction-json')
        self.assertEqual(json.loads(args[1]),PILOT_GRID)
        with self.assertRaises(ValueError): abstraction_arguments(True,PILOT_GRID)
        verify_evaluation_game({'depth_bb':50,'action_abstraction':PILOT_GRID},50,PILOT_GRID)
        for metadata in [{'depth_bb':20,'action_abstraction':PILOT_GRID}, {'depth_bb':50},
                         {'depth_bb':50,'action_abstraction':{**PILOT_GRID,'open_sizes_bb':[2,2.5,3]}}]:
            with self.assertRaises(RuntimeError): verify_evaluation_game(metadata,50,PILOT_GRID)

    def test_actual_trajectory_command_and_metadata_share_the_pinned_grid(self):
        metadata={'depth_bb':100,'action_abstraction':PILOT_GRID,'sampling_mode':'trajectory','truncated':False}
        with patch('validate_seeds.subprocess.run') as run, patch('validate_seeds.load_jsonl_gzip',return_value=(metadata,[record([48,49])])):
            records=evaluation_records(Path('/root'),100,64,123,256,128,True,False,
                action_abstraction=PILOT_GRID,binary_path=Path('/frozen/sampler'))
            argv=run.call_args.args[0]
            self.assertEqual(argv[0],'/frozen/sampler')
            self.assertEqual(json.loads(argv[argv.index('--action-abstraction-json')+1]),PILOT_GRID)
            self.assertEqual(len(records),1)
        config={'depth_bb':50,'preflop_runout_samples':256,'flop_runout_samples':128,
            'exact_turn_rivers':False,'compact_serving_grid':False,'action_abstraction_json':json.dumps(PILOT_GRID)}
        other={**config,'action_abstraction_json':json.dumps({**PILOT_GRID,'open_sizes_bb':[2,2.5,3]})}
        self.assertNotEqual(routing_compatible_config({'config':config}),routing_compatible_config({'config':other}))

    def test_total_exploitability_gate_is_half_a_blind_with_consistent_selection(self):
        self.assertEqual(MAXIMUM_ESTIMATED_EXPLOITABILITY_BB_PER_HAND, 0.50)
        self.assertEqual(
            MAXIMUM_ONE_SIDED_99_EXPLOITABILITY_UPPER_BOUND_BB_PER_HAND,
            0.50,
        )
        certificates = [
            {
                "sample_mean_exploitability_bb": 0.10,
                "exploitability_upper_bound_bb": 0.49,
                "network_sha256": "a" * 64,
            },
            {
                "sample_mean_exploitability_bb": 0.01,
                "exploitability_upper_bound_bb": 0.50,
                "network_sha256": "b" * 64,
            },
        ]
        selected = select_exploitability_certificate(certificates)
        self.assertIs(selected, certificates[0])
        self.assertEqual(selected["sample_mean_exploitability_bb"], 0.10)

    def test_weight_overrides_must_remain_a_paired_seed_comparison(self):
        with self.assertRaisesRegex(ValueError, "paired frozen weight"):
            apply_weight_overrides([], (Path("seed-a.safetensors"), None))

    def test_action_ev_gate_requires_every_action_at_a_served_decision(self):
        precise = {
            **record([0, 4]),
            "action_value_standard_errors_bb": [0.01, 0.02],
        }
        noisy = {
            **record([48, 44]),
            "action_value_standard_errors_bb": [0.01, 0.03],
        }
        result = action_ev_standard_error_summary([[precise, noisy]])
        self.assertTrue(result["available"])
        self.assertEqual(result["decisions"], 2)
        self.assertAlmostEqual(result["decision_coverage"], 0.5)
        self.assertAlmostEqual(result["action_coverage"], 0.75)
        self.assertAlmostEqual(result["maximum_standard_error_bb"], 0.03)

    def test_street_routed_model_selects_the_matching_component(self):
        features = np.zeros((2, INPUT_FEATURE_COUNT), dtype=np.float32)
        features[0, 104] = 1.0

        class ConstantModel:
            def __init__(self, value):
                self.value = value

            def __call__(self, values):
                return mx.full((values.shape[0], 1), self.value)

        result = np.asarray(
            StreetRoutedModel(ConstantModel(3.0), ConstantModel(7.0))(
                mx.array(features)
            )
        ).reshape(-1)
        np.testing.assert_array_equal(result, np.asarray([3.0, 7.0]))

    def test_empirical_trajectory_comparison_keeps_repeated_visits(self):
        divergent = record([0, 4])
        agreeing = record([48, 44])
        result = compare(
            UniformModel(),
            RankMarkerModel(),
            [[divergent, divergent, agreeing]],
            20,
            "test trajectory distribution",
            True,
        )
        self.assertEqual(result["decisions"], 3)
        self.assertAlmostEqual(result["primary_action_agreement"], 1.0 / 3.0)
        self.assertEqual(result["sampling_method"], "empirical_pure_trajectories")
        self.assertAlmostEqual(result["street_breakdown"]["preflop"]["reach_mass"], 1.0)

    def test_policy_corpora_receive_equal_mixture_mass(self):
        divergent = record([0, 4])
        agreeing = record([48, 44])
        result = compare(
            UniformModel(),
            RankMarkerModel(),
            [[divergent, divergent], [agreeing]],
            20,
            "equal policy mixture",
            True,
        )
        self.assertEqual(result["decisions"], 3)
        self.assertAlmostEqual(result["primary_action_agreement"], 0.5)
        self.assertAlmostEqual(result["effective_decisions"], 8.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
