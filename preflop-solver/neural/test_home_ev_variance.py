import copy
import unittest

from run_home_ev_variance import comparison


class HomeEVVarianceTests(unittest.TestCase):
    def fixture(self):
        return {'records':[{'state':{},'actions':[{},{}],'probabilities':[.5,.5],
            'action_values_bb':[0.,1.],'standard_errors_bb':[.1,0.],
            'independent_hidden_block_values_bb':[[-2.,1.],[2.,1.],[-2.,1.],[2.,1.]],
            'action_estimators':['independent_hidden_blocks_v1','exact_closed_posterior_runouts_v1']}]}

    def test_cost_adjustment_prevents_promoting_expensive_variance_reduction(self):
        before = self.fixture(); after = copy.deepcopy(before)
        after['records'][0]['independent_hidden_block_values_bb']=[[-1.,1.],[1.,1.],[-1.,1.],[1.,1.]]
        result = comparison(before,after,1.,5.)
        self.assertAlmostEqual(result['variance_ratio'],.25)
        self.assertFalse(result['cost_adjusted_improvement'])
        self.assertTrue(comparison(before,after,1.,2.)['cost_adjusted_improvement'])

    def test_exact_constants_cannot_silently_change(self):
        before = self.fixture(); after = copy.deepcopy(before)
        after['records'][0]['independent_hidden_block_values_bb'][0][1] += 1
        with self.assertRaisesRegex(ValueError,'exact action'):
            comparison(before,after,1.,1.)
