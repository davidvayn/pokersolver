import unittest
import os
from pathlib import Path
import tempfile

import mlx.core as mx

from depth_repair import ROOT, frozen_reference
from qualify_depth_repair_values import (chunk_seed, chunked_frozen_reference,
                                         reference_precision_summary, sample_authentic_decisions)
from run_depth_pilots import PILOT_GRID
from train import ActionScorer, INPUT_FEATURE_COUNT
from validate_seeds import evaluation_records


class AuthenticFrozenValueTests(unittest.TestCase):
    def test_chunk_seeds_preserve_the_native_global_root_index(self):
        mask = (1 << 64) - 1
        self.assertEqual(chunk_seed(17, 0), 17)
        self.assertEqual((chunk_seed(17, 32) + 3 * 0x9e3779b97f4a7c15) & mask, chunk_seed(17, 35))

    @unittest.skipUnless(os.environ.get('POKER_CERTIFICATE_TEST_BINARY'), 'explicit native binary required')
    def test_chunked_native_blocks_are_identical_to_one_call(self):
        binary = Path(os.environ['POKER_CERTIFICATE_TEST_BINARY'])
        mx.random.seed(2717141)
        model = ActionScorer(INPUT_FEATURE_COUNT, (8, 4))
        mx.eval(model.parameters())
        rows = evaluation_records(ROOT, 50, 2, 2717143, 64, 32, True, False,
            policy_model=model, action_abstraction=PILOT_GRID, binary_path=binary)[:2]
        self.assertEqual(len(rows), 2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            single = frozen_reference(binary, model, rows, 50, root/'single.json', blocks=8)
            chunks = chunked_frozen_reference(binary, model, rows, 50, root/'chunks.json', blocks=8, chunk_size=1)
            self.assertEqual(single['records'], chunks['records'])
            self.assertEqual(single['input_sha256'], chunks['input_sha256'])
            self.assertEqual(single['source_network_sha256s'], chunks['source_network_sha256s'])
            exact = frozen_reference(binary,model,rows,50,root/'exact.json',blocks=8,
                exact_river_values=True,exact_closed_values=True)
            self.assertTrue(exact['exact_river_values'] and exact['exact_closed_values'])
            first=exact['records'][0]
            fold=next(i for i,a in enumerate(first['actions']) if a['kind']=='fold')
            self.assertEqual(first['action_estimators'][fold],'exact_closed_posterior_runouts_v1')
            self.assertEqual(first['standard_errors_bb'][fold],0.)
            exact_chunks=chunked_frozen_reference(binary,model,rows,50,root/'exact-chunks.json',blocks=8,chunk_size=1,
                exact_river_values=True,exact_closed_values=True)
            self.assertEqual(exact['records'],exact_chunks['records'])
            prefix = frozen_reference(binary,model,rows,50,root/'prefix.json',blocks=8,action_expectation_depth=1)
            prefix_chunks = chunked_frozen_reference(binary,model,rows,50,root/'prefix-chunks.json',blocks=8,
                chunk_size=1,action_expectation_depth=1)
            self.assertEqual(prefix['records'],prefix_chunks['records'])
            self.assertEqual(prefix['action_expectation_depth'],1)

    def test_policy_mixture_is_equal_even_when_trajectory_lengths_differ(self):
        first = [{'iteration': 0, 'state': {'street': 'preflop'}}]
        second = [{'iteration': i, 'state': {'street': 'river'}} for i in range(100)]
        sampled = sample_authentic_decisions([first, second], 10000, 2717123)
        share = sum(row['state']['street'] == 'preflop' for row in sampled) / len(sampled)
        self.assertAlmostEqual(share, .5, delta=.02)
        self.assertEqual(sampled, sample_authentic_decisions([first, second], 10000, 2717123))
        self.assertTrue(all(row['iteration'] >= 1_000_000 for row in sampled if row['state']['street'] == 'river'))
        self.assertEqual(first[0]['iteration'], 0)

    def test_repeated_draws_are_not_discarded_or_street_balanced(self):
        rows = [{'iteration': 0, 'state': {'street': 'preflop'}}]
        sampled = sample_authentic_decisions([rows, rows], 32, 1)
        self.assertEqual(len(sampled), 32)
        self.assertEqual(len({row['iteration'] for row in sampled}), 2)
        with self.assertRaises(ValueError):
            sample_authentic_decisions([rows, []], 32, 1)

    def test_precision_requires_every_action_in_a_decision(self):
        result = reference_precision_summary([
            {'standard_errors_bb': [.01, .03]}, {'standard_errors_bb': [.01, .01]}], 2048)
        self.assertEqual(result['decision_precision_coverage'], .5)
        self.assertEqual(result['action_precision_coverage'], .75)
        self.assertGreater(result['optimistic_outer_blocks_for_95_percent_decisions'], 2048)
        with self.assertRaises(ValueError):
            reference_precision_summary([{'standard_errors_bb': [float('nan')]}], 2048)


if __name__ == '__main__':
    unittest.main()
