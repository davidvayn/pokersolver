import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from run_depth_pilots import PILOT_GRID
from train import RunConfig, export_browser_artifact, initialize_models, atomic_json, scorer_json
from frozen_policy_values import (digest, fit_policy_values, load_policy_value_sidecar,
                                  native_equal, split_families, verify_frozen_targets)
from validate_seeds import load_artifact_model


class FrozenPolicyValueTests(unittest.TestCase):
    def fixture(self, directory):
        config = RunConfig(schema='test', depth_bb=50, seed=1, reservoir_capacity=16,
            hidden_sizes=(8, 4), batch_size=4, learning_rate=.001, learning_rate_final=None,
            learning_rate_decay_start_round=None, learning_rate_decay_end_round=None,
            traversals_per_round=2, steps_per_round=1, advantage_alpha=2,
            variance_baseline_scale=.5, replay_street_proposal=None, value_rollouts_per_action=4,
            artifact_every=1, preflop_runout_samples=4, flop_runout_samples=4,
            exact_turn_rivers=False, compact_serving_grid=False)
        models, _ = initialize_models(config)
        root = Path(__file__).resolve().parents[2]
        return export_browser_artifact(root, Path(directory), models, config, 1, PILOT_GRID)

    def test_training_export_cannot_claim_values_match_the_average_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _ = self.fixture(directory)
            metadata = json.loads(source.read_text())['metadata']
            self.assertTrue(metadata['modelVersion'].startswith('deep-dcfr-plus-v13-'))
            self.assertEqual(metadata['valueTargetPolicy']['kind'], 'historical_training_policies')
            self.assertFalse(metadata['valueTargetPolicy']['servingPolicyMatched'])
            self.assertFalse(metadata['valueTargetPolicy']['releaseEligible'])
            self.assertEqual(metadata['valueUncertaintyTraining']['scope'], 'continuation_actions_conditional_on_deal')

    def test_four_family_splits_do_not_leak_duplicates(self):
        records = [{'iteration': i} for i in range(24)] * 2
        splits = list(map(set, split_families(records, 1)))
        self.assertEqual(set.union(*splits), set(range(24)))
        self.assertTrue(all(not a & b for i, a in enumerate(splits) for b in splits[i + 1:]))
        self.assertTrue(native_equal({'amount': 2.333}, {'amount': float(np.float32(2.333))}))
        self.assertFalse(native_equal({'amount': 2.333}, {'amount': 2.334}))

    def test_frozen_fit_binds_policy_rejects_corruption_and_keeps_error_contract_separate(self):
        from test_train import initial_state
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, binary = self.fixture(directory)
            state = initial_state()
            state.update(stacks_bb=[49.5,49.])
            records = [{'iteration': i, 'state': state,
                'actions': [{'kind': 'fold', 'amount_to_bb': None}]} for i in range(24)]
            reference_path = root / 'reference.json'
            inputs, networks = reference_path.with_suffix('.input.json'), reference_path.with_suffix('.networks.json')
            atomic_json(inputs, records)
            policy = scorer_json(load_artifact_model(source))
            atomic_json(networks, {'strategy_transform': 'softmax', 'networks': [policy, policy]})
            reference = {'schema': 'hu-frozen-information-set-values-v1', 'depth_bb': 50,
                'policy_only': False, 'snapshot_average': False, 'hidden_blocks': 8,
                'action_abstraction': PILOT_GRID, 'input_sha256': digest(inputs),
                'source_network_sha256s': [digest(networks)],
                'uncertainty_contract': 'Independent outer opponent-posterior/future-board blocks; test fixture.',
                'records': [{**r, 'action_values_bb': [-.5], 'standard_errors_bb': [.01]} for r in records]}
            atomic_json(reference_path, reference)
            report = fit_policy_values(source, reference_path, root / 'values', trial_steps=(2,))
            self.assertFalse(report['release_eligible'])
            self.assertEqual(report['selected_steps'], 2)
            self.assertTrue(report['value_target_policy_matched'])
            sidecar = root / 'values/frozen-policy-values.json'
            loaded = load_policy_value_sidecar(sidecar, binary)
            self.assertFalse(loaded['confidence']['is_sampling_standard_error'])
            self.assertNotIn('standard_error_bb', loaded)
            changed = root / 'wrong-model.bin'; changed.write_bytes(b'not the pinned artifact')
            with self.assertRaises(ValueError):
                load_policy_value_sidecar(sidecar, changed)
            reference['source_network_sha256s'] = ['0' * 64]
            atomic_json(reference_path, reference)
            with self.assertRaises(ValueError):
                verify_frozen_targets(source, reference_path)
            reference['source_network_sha256s'] = [digest(networks)]
            reference['records'][0] = {**reference['records'][0], 'actions': [{'kind': 'call', 'amount_to_bb': None}]}
            atomic_json(reference_path, reference)
            with self.assertRaises(ValueError):
                verify_frozen_targets(source, reference_path)
            # A JSON source with the same file name cannot masquerade as the binary.
            value = json.loads(source.read_text()); value['parameters'][0] += 1
            atomic_json(source, value)
            with self.assertRaises(ValueError):
                verify_frozen_targets(source, reference_path)


if __name__ == '__main__':
    unittest.main()
