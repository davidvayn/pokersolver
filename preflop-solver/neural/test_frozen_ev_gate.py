import copy
import unittest

from frozen_ev_gate import frozen_ev_gate


def receipt():
    return {'schema':'hu-frozen-ev-qualification-v1','depth_bb':50,'action_abstraction':{},
        'uncertainty_scope':'opponent_posterior_and_future_board_outer_blocks',
        'evaluation_distribution':'heldout_authentic_reach_mixture',
        'candidates':[{'artifact_sha256':sha,'frozen_policy_inference_verified':True,
            'independent_hidden_blocks':32,'heldout_decisions':128,
            'reference_decision_precision_coverage':.96,'served_predicted_precision_coverage':.97,
            'nominal_95_interval_coverage':.96} for sha in ('a'*64,'b'*64)]}


class FrozenEvGateTests(unittest.TestCase):
    def test_missing_qualification_is_unmeasured_not_a_numeric_failure(self):
        result=frozen_ev_gate(None,50,{},['a'*64,'b'*64])
        self.assertFalse(result['available']); self.assertFalse(result['qualified'])

    def test_exact_artifact_bound_full_hidden_receipt_can_pass(self):
        self.assertTrue(frozen_ev_gate(receipt(),50,{},['a'*64,'b'*64])['qualified'])

    def test_conditional_noise_and_pilot_distribution_cannot_qualify_release(self):
        for key,value in (('uncertainty_scope','fixed_deal_action_rollouts'),
                          ('evaluation_distribution','street_stratified_pilot')):
            record=receipt(); record[key]=value
            self.assertFalse(frozen_ev_gate(record,50,{},['a'*64,'b'*64])['qualified'])

    def test_small_predicted_noise_does_not_override_bad_calibration(self):
        record=receipt(); record['candidates'][0]['nominal_95_interval_coverage']=.6
        self.assertFalse(frozen_ev_gate(record,50,{},['a'*64,'b'*64])['qualified'])

    def test_wrong_weights_game_or_nonfinite_metric_are_rejected(self):
        for patch in ({'depth_bb':100},{'action_abstraction':{'other':True}},
                      {'candidates':[receipt()['candidates'][0]]}):
            record={**receipt(),**patch}
            with self.assertRaises(ValueError): frozen_ev_gate(record,50,{},['a'*64,'b'*64])
        record=copy.deepcopy(receipt());record['candidates'][0]['reference_decision_precision_coverage']=float('nan')
        with self.assertRaises(ValueError):frozen_ev_gate(record,50,{},['a'*64,'b'*64])
        record=receipt(); record['candidates']=[]
        with self.assertRaises(ValueError):frozen_ev_gate(record,50,{},[])


if __name__=='__main__': unittest.main()
