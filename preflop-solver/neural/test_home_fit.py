import tempfile
import unittest
from pathlib import Path

import numpy as np

from probe_home_fit import batch, read_replay, statistics
from probe_regret_fit_budget import split_indices
from train import atomic_json


class HomeFitTests(unittest.TestCase):
    def test_arrival_ablation_preserves_actions_and_legacy_features(self):
        arrays = {'states': np.ones((2, 820)), 'actions': np.ones((2, 8, 9))*2,
                  'targets': np.zeros((2, 8)), 'masks': np.ones((2, 8)), 'weights': np.ones((2, 1))}
        x, _, _, w = batch(arrays, np.array([0, 1]), hide_arrivals=True, corrections=np.array([.5, 2]))
        x = np.asarray(x)
        np.testing.assert_array_equal(x[:, :, :716], 1)
        np.testing.assert_array_equal(x[:, :, 716:820], 0)
        np.testing.assert_array_equal(x[:, :, 820:], 2)
        np.testing.assert_array_equal(np.asarray(w)[:, 0], [.5, 2])

    def test_split_precedes_ablation_and_keeps_repeated_states_together(self):
        x = np.random.default_rng(1).normal(size=(100, 820)).astype(np.float16)
        x[-1] = x[0]
        x[-1, 716:] += 1  # Even distinct v5 histories collapse under the ablation.
        train, test = split_indices(x[:, :716])
        self.assertEqual(0 in train, 99 in train)
        self.assertFalse(set(train) & set(test))

    def test_policy_metrics_use_softmax_not_regret_matching_and_mask_padding(self):
        p = np.array([[.2, .8, 0.]])
        mask = np.array([[1., 1., 0.]])
        actual = statistics(np.array([[np.log(.2), np.log(.8), 1e10]]), p, mask, np.ones((1, 1)), True)
        self.assertAlmostEqual(actual['kl_sum'], 0)
        self.assertAlmostEqual(actual['mix_mae_sum'], 0)
        self.assertAlmostEqual(actual['primary_sum'], 1)

    def test_readonly_loader_rejects_wrong_v5_byte_width(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run/'replay').mkdir()
            atomic_json(run/'state.json', {'config': {'card_arrival_memory': True, 'reservoir_capacity': 2},
                'reservoirs': {'advantage_p0': {'size': 2, 'max_actions': 8}}})
            (run/'replay/advantage_p0.states.f16').write_bytes(bytes(2*716*2))
            with self.assertRaisesRegex(ValueError, 'byte length'):
                read_replay(run, 'advantage_p0')
