import copy
import unittest

import numpy as np

from cash_flop_leaves import leaf_input
from cash_query_pairs import paired_trace_inputs
from native_value_dataset import legal_combos
import test_cash_flop_leaves as leaf_tests


class CashQueryPairTests(unittest.TestCase):
    def fixture(self):
        root, solution = leaf_tests.CashFlopLeafTests().both_seat_bets()
        root['solve_input']['iterations'] = solution['iterations'] = 32
        config = leaf_input(root, solution, 31, ['check', 'bet_to_2.480bb', 'call'], 2)
        # A current policy can have zero own support; keep its actual belief,
        # not the average solution's strictly positive likelihoods.
        legal = legal_combos(config['state']['board'])
        current = np.asarray(config['state']['ranges'])
        current[0, np.flatnonzero(legal)[::2]] = 0
        current /= current.sum(axis=1)[:, None]
        config['state']['ranges'] = current.tolist()
        query = dict(round=16, query_sha256='c'*64,
                     flop_public_history=['public_belief:flop_start', 'Flop:p1:check',
                                          'Flop:p0:bet_to_2.480bb', 'Flop:p1:call', 'deal:Turn'],
                     reach_probability_from_flop_root=.02, input=config)
        options = dict(solve=copy.deepcopy(root['solve_input']), sample_rounds=[1,16], leaves_per_round=1)
        trace = dict(schema='cash-current-policy-turn-trace-v1', status='research_only', active=False,
                     trace_input_sha256='d'*64, trace_input=options, solution=solution,
                     observed_turn_query_events_per_round={'1':49, '16':49}, queries=[query],
                     selection_method='smallest-sha256-unique-queries-per-selected-training-round')
        return root, trace

    def pair(self, root, trace, iterations=128):
        return paired_trace_inputs(trace, root, 'a'*64, 'b'*64, reference_iterations=iterations)

    def test_changes_only_ranges_and_explicit_reference_budget_without_mutation(self):
        root, trace = self.fixture(); before = copy.deepcopy((root, trace))
        pairs = self.pair(root, trace)
        self.assertEqual((root, trace), before)
        self.assertEqual(len(pairs), 1)
        pair = pairs[0]
        self.assertEqual(pair['round'], 16)
        self.assertEqual(pair['branch_labels'], ['check','bet_to_2.480bb','call'])
        self.assertEqual(pair['captured_prediction_input'], trace['queries'][0]['input'])
        current, average = pair['current'], pair['average']
        self.assertEqual(current['iterations'], 128)
        self.assertEqual(average['iterations'], 128)
        self.assertEqual(current['state']['invested_bb'], [3.48,3.48])
        current_ranges = current['state'].pop('ranges')
        average_ranges = average['state'].pop('ranges')
        self.assertEqual(current, average)
        self.assertNotEqual(current_ranges, average_ranges)
        self.assertTrue((np.asarray(current_ranges)[0] == 0).any())

    def test_adjacent_float_cent_representations_preserved_but_subcent_rejected(self):
        root, trace = self.fixture()
        money = trace['queries'][0]['input']['state']['invested_bb']
        money[:] = [np.nextafter(3.48, 4.).item()]*2
        pair = self.pair(root, trace)[0]
        self.assertEqual(pair['average']['state']['invested_bb'], money)
        for bad in [3.481, 3.52, 20., float('nan')]:
            root, trace = self.fixture()
            trace['queries'][0]['input']['state']['invested_bb'] = [bad,bad]
            with self.assertRaises(ValueError): self.pair(root, trace)

    def test_order_independent_and_repeated_queries_or_unobserved_rounds_rejected(self):
        root, trace = self.fixture()
        query = copy.deepcopy(trace['queries'][0]); query.update(round=1,query_sha256='e'*64)
        trace['queries'].append(query)
        expected = self.pair(root, trace)
        trace['queries'].reverse()
        self.assertEqual(self.pair(root, trace), expected)
        trace['queries'].append(copy.deepcopy(query))
        with self.assertRaises(ValueError): self.pair(root, trace)
        root, trace = self.fixture(); trace['observed_turn_query_events_per_round'].pop('16')
        with self.assertRaises(ValueError): self.pair(root, trace)

    def test_wrong_identity_private_information_and_unbounded_requests_fail_closed(self):
        mutations = [
            lambda t:t.update(active=True), lambda t:t.update(status='validated'),
            lambda t:t['solution'].update(value_network_sha256='f'*64),
            lambda t:t['trace_input']['solve'].update(iterations=31),
            lambda t:t['trace_input'].update(sample_rounds=[16,1]),
            lambda t:t['trace_input'].update(leaves_per_round=True),
            lambda t:t['queries'][0].update(round=True),
            lambda t:t['queries'][0].update(reach_probability_from_flop_root=0),
            lambda t:t['queries'][0].update(reach_probability_from_flop_root=float('nan')),
            lambda t:t['queries'][0]['input'].update(opponent_hole_cards=[0,1]),
            lambda t:t['queries'][0]['input'].update(iterations=64),
            lambda t:t['queries'][0]['input']['state'].update(actor=0),
            lambda t:t['queries'][0]['input']['state'].update(board=[0,1,2,3,4]),
            lambda t:t['queries'][0]['flop_public_history'].__setitem__(2,'Flop:p1:bet_to_2.480bb'),
        ]
        for mutate in mutations:
            root, trace = self.fixture(); mutate(trace)
            with self.assertRaises(ValueError): self.pair(root, trace)
        for iterations in [True, 1, 129, 2.5]:
            root, trace = self.fixture()
            with self.assertRaises(ValueError): self.pair(root, trace, iterations)

    def test_nonfinite_illegal_unnormalized_and_incompatible_beliefs_rejected(self):
        for mutate in [lambda r:r[0].__setitem__(0,float('nan')),
                       lambda r:r[0].__setitem__(0,-1),
                       lambda r:r[0].__setitem__(0,True),
                       lambda r:r[0].__setitem__(0,.1),
                       lambda r:r.__setitem__(0,[0.]*1326)]:
            root, trace = self.fixture(); mutate(trace['queries'][0]['input']['state']['ranges'])
            with self.assertRaises(ValueError): self.pair(root, trace)
        root, trace = self.fixture()
        ranges = np.zeros((2,1326)); legal_index = np.flatnonzero(legal_combos(trace['queries'][0]['input']['state']['board']))[0]
        ranges[:, legal_index] = 1
        trace['queries'][0]['input']['state']['ranges'] = ranges.tolist()
        with self.assertRaises(ValueError): self.pair(root, trace)

    def test_empty_observer_is_empty_not_fabricated_and_bad_average_support_rejected(self):
        root, trace = self.fixture(); trace['queries'] = []; trace['observed_turn_query_events_per_round'] = {}
        self.assertEqual(self.pair(root, trace), [])
        root, trace = self.fixture()
        # A trace cannot use a nonexistent final-average action as a control.
        trace['queries'][0]['flop_public_history'][2] = 'Flop:p0:bet_to_9.000bb'
        with self.assertRaises(ValueError): self.pair(root, trace)

    def test_weighted_observer_is_explicit_and_never_claims_inclusion_probabilities(self):
        root, trace = self.fixture()
        trace['trace_input']['reach_sampling_seed'] = 917
        with self.assertRaises(ValueError): self.pair(root, trace)
        trace['selection_method'] = 'seeded-exponential-race-unique-queries-weighted-by-joint-root-reach'
        pair = self.pair(root, trace)[0]
        self.assertEqual(pair['reach_probability_from_flop_root'], .02)
        self.assertNotIn('inclusion_probability', pair)
        for seed in [-1, 2**64, True, '917', None]:
            bad = copy.deepcopy(trace); bad['trace_input']['reach_sampling_seed'] = seed
            with self.assertRaises(ValueError): self.pair(root, bad)

    def test_no_control_when_average_policy_gives_the_branch_zero_probability(self):
        root, trace = self.fixture()
        row = trace['solution']['strategies'][1]
        mix = np.asarray(row['probabilities']).reshape(1326,2)
        legal = mix.sum(axis=1) > 0
        mix[legal] = [1.,0.]
        row['probabilities'] = mix.reshape(-1).tolist()
        with self.assertRaisesRegex(ValueError,'no public reach'): self.pair(root, trace)


if __name__ == '__main__': unittest.main()
