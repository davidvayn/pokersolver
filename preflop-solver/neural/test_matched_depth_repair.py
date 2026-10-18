import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from depth_repair import training_command
from run_matched_depth_repair import add_family_namespace, paired_response_summary, promotion_decision


class MatchedDepthRepairTests(unittest.TestCase):
    def test_training_command_pins_depth_budget_and_only_changes_estimator(self):
        common = (Path('/sampler'), Path('/run'), 'control', 51701, 64, 256, 160)
        with patch.object(Path, 'read_bytes', return_value=b'fixed sampler'):
            control = training_command(*common, depth=100, max_minutes=35)
            averaged = training_command(*common[:2], 'averaged', *common[3:], depth=100, max_minutes=35)
        self.assertEqual(averaged[:-1], control)
        self.assertEqual(averaged[-1], '--average-regret-rollouts')
        self.assertEqual(control[control.index('--depth-bb') + 1], '100')
        self.assertEqual(control[control.index('--max-minutes') + 1], '35')
        self.assertEqual(control[control.index('--traversals-per-round') + 1], '256')

    def test_legacy_training_budget_defaults_are_preserved(self):
        with patch.object(Path, 'read_bytes', return_value=b'fixed sampler'):
            argv = training_command(Path('/sampler'), Path('/run'), 'control', 1, 12, 64, 80)
            self.assertEqual(argv[argv.index('--depth-bb') + 1], '50')
            self.assertEqual(argv[argv.index('--max-minutes') + 1], '10')
            self.assertNotIn('--average-regret-rollouts', argv)
            with self.assertRaises(ValueError):
                training_command(Path('/sampler'), Path('/run'), 'control', 1, 129, 64, 80)

    def test_independent_corpora_never_share_trajectory_family_ids(self):
        rows = [{'iteration': 0}, {'iteration': 0}, {'iteration': 1}]
        first, second = add_family_namespace(rows, 0), add_family_namespace(rows, 1)
        self.assertEqual(first[0]['iteration'], first[1]['iteration'])
        self.assertFalse({r['iteration'] for r in first} & {r['iteration'] for r in second})
        self.assertEqual(rows[0]['iteration'], 0)

    def test_paired_response_retains_root_pairing_and_negative_estimates(self):
        result = paired_response_summary(np.ones((2, 16)), np.zeros((2, 16)))
        self.assertEqual(result['mean_delta_bb'], -1.)
        self.assertEqual(result['one_sided_95_bootstrap_upper_delta_bb'], -1.)
        self.assertEqual(result['seed_mean_deltas_bb'], [-1., -1.])
        with self.assertRaises(ValueError):
            paired_response_summary(np.ones((2, 16)), np.ones((2, 15)))
        with self.assertRaises(ValueError):
            paired_response_summary(np.full((2, 16), np.nan), np.ones((2, 16)))

    def test_stability_alone_cannot_promote_or_start_100bb(self):
        batches = [{'control': {'action_frequency_mae': .14, 'primary_action_agreement': .50},
                    'averaged': {'action_frequency_mae': .12, 'primary_action_agreement': .60}}] * 2
        noisy = {'seed_mean_deltas_bb': [-.1, -.2], 'one_sided_95_bootstrap_upper_delta_bb': .3}
        self.assertFalse(promotion_decision(batches, noisy)['improvements_hold'])
        credible = {**noisy, 'one_sided_95_bootstrap_upper_delta_bb': -.01}
        self.assertTrue(promotion_decision(batches, credible)['improvements_hold'])
        regression = {'control': batches[0]['control'], 'averaged': {'action_frequency_mae': .15, 'primary_action_agreement': .60}}
        self.assertFalse(promotion_decision([batches[0], regression], credible)['improvements_hold'])
        self.assertFalse(promotion_decision(batches, {**credible, 'seed_mean_deltas_bb': [-1., .1]})['improvements_hold'])


if __name__ == '__main__':
    unittest.main()
