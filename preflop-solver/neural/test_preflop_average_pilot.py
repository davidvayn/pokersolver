import copy
import json
import unittest
from unittest.mock import patch

from cloud_blueprint_run import CANONICAL_HAND_CLASSES
from run_preflop_average_pilot import analyze, stability


class StabilityTests(unittest.TestCase):
    def fixture(self):
        return {'rows': [dict(key=str(i), actor=0, history=[], hand=hand,
            actions=['fold', 'call'], comboWeight=663, probabilities=probabilities)
            for i, hand, probabilities in [(0, 'AA', [0.8, 0.2]), (1, 'KK', [0.4, 0.6])]]}

    def test_probability_deltas_and_incomplete_coverage_are_explicit(self):
        a = self.fixture()
        b = copy.deepcopy(a)
        b['rows'][0]['probabilities'] = [0.6, 0.4]
        result = stability(a, b)['initialRoot']
        self.assertTrue(result['completeCoverage'])
        self.assertAlmostEqual(result['comboWeightedPerActionMae'], 0.1)
        self.assertAlmostEqual(result['maximumAggregateDelta'], 0.1)
        self.assertEqual(result['comboWeightedPrimaryAgreement'], 1)
        b['rows'][1]['probabilities'] = None
        result = stability(a, b)['initialRoot']
        self.assertFalse(result['completeCoverage'])
        self.assertEqual(result['comparedCombos'], 663)
        self.assertAlmostEqual(result['comboWeightedPerActionMae'], 0.2)

    def test_rejects_action_mismatch_and_invalid_probability_sums(self):
        a = self.fixture()
        b = copy.deepcopy(a)
        b['rows'][0]['actions'].reverse()
        with self.assertRaises(AssertionError):
            stability(a, b)
        b = copy.deepcopy(a)
        b['rows'][0]['probabilities'] = [0.6, 0.6]
        with self.assertRaises(AssertionError):
            stability(a, b)

    def test_initial_root_includes_posted_blinds(self):
        a = self.fixture()
        for row in a['rows']:
            row['history'] = ['post_sb_0.500bb', 'post_bb_1.000bb']
        result = stability(a, a)
        self.assertEqual(result['initialRoot']['history'], tuple(a['rows'][0]['history']))

    def test_established_gate_does_not_use_mean_over_actions(self):
        jobs, outputs = [], []
        for seed in (26001, 26002):
            for mode in ('sampled', 'exact'):
                jobs.append(dict(status='complete', output='mock-result',
                    outputSha256='digest', seed=seed, mode=mode))
                probabilities = [0.6, 0.3, 0.1] if seed == 26001 else [0.5, 0.4, 0.1]
                outputs.append(json.dumps(dict(
                    config=dict(seed=seed, iterations=64, exact_preflop_averaging=mode=='exact'),
                    regretStateSha256='same', postflopStateSha256='same', rngState=1,
                    sampledDeals=64, terminalEvaluations=64, absentAverageCombos=0,
                    untrainedCombos=0, trainingSeconds=1,
                    rows=[dict(key=hand, hand=hand, actor=0, history=['blinds:0.500/1.000'],
                        comboWeight=6 if len(hand)==2 else 4 if hand.endswith('s') else 12,
                        actions=['fold', 'call', 'jam'], probabilities=probabilities,
                        averageVisits=64, regretUpdates=64, trained=True)
                        for hand in sorted(CANONICAL_HAND_CLASSES)])))
        with patch('run_preflop_average_pilot.file_sha256', return_value='digest'), \
                patch('run_preflop_average_pilot.Path.read_text', side_effect=outputs):
            result = analyze(dict(jobs=jobs, rounds=64))
        for mode in ('sampled', 'exact'):
            established = result['establishedRootStability'][mode]
            self.assertAlmostEqual(established['maximumComboWeightedPerActionMae'], 0.1)
            self.assertFalse(established['passed'])
            self.assertAlmostEqual(result['stability'][mode]['initialRoot']['comboWeightedPerActionMae'], 0.2/3)


if __name__ == '__main__':
    unittest.main()
