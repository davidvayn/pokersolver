import copy
import unittest

import numpy as np

from cash_paired_regime_probe import paired_small_pot_coefficients, split_paired_probe_rows
from cash_profiles import profile_rules, rules_digest
from native_value_dataset import board_family


def source(boards, cents):
    rules=profile_rules('nl25')
    game={'effective_stack_bb':20.,'cash_rules':rules}
    return {'game':game,'rules_sha256':rules_digest(rules),
        'labels':[{'input':{'game':copy.deepcopy(game),'state':{
            'board':board,'invested_bb':[amount/25.,amount/25.]}}}
            for board,amount in zip(boards,cents)]}


class PairedRegimeProbeTests(unittest.TestCase):
    def test_exact_money_fade_and_only_newly_paired_turns(self):
        paired=[1,27,22,25]
        item=source([paired]*6+[[1,27,22,40],[1,2,22,40]], [25,75,76,87,99,100,50,50])
        actual=paired_small_pot_coefficients(item)
        expected=[1.,1.,(.96**2)*(3-2*.96),(.52**2)*(3-2*.52),(.04**2)*(3-2*.04),0.,0.,0.]
        np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=0)
        self.assertEqual(actual.dtype,np.float32)

    def test_reject_incompatible_rules_board_and_money(self):
        good=source([[1,27,22,25]],[50])
        changes=[('game',{'effective_stack_bb':40.,'cash_rules':good['game']['cash_rules']}),
                 ('rules_sha256','0'*64)]
        for field,value in changes:
            bad=copy.deepcopy(good); bad[field]=value
            with self.assertRaises(ValueError): paired_small_pot_coefficients(bad)
        for board,amounts in (([1,27,22,22],[2.,2.]),([1,27,22,52],[2.,2.]),
                              ([1,27,22,25],[2.,2.04]),([1,27,22,25],[2.001,2.001]),
                              ([1,27,22,25],[20.,20.])):
            bad=copy.deepcopy(good); bad['labels'][0]['input']['state'].update(board=board,invested_bb=amounts)
            with self.assertRaises(ValueError): paired_small_pot_coefficients(bad)
        bad=copy.deepcopy(good); bad['labels'][0]['input']['game']['cash_rules']['rake']['capUnits']=0
        with self.assertRaises(ValueError): paired_small_pot_coefficients(bad)

    def test_split_holds_out_whole_flop_families_and_does_not_mutate_input(self):
        boards=[[4*i,4*(i+1)+1,4*(i+3)+2,4*(i+1)+3] for i in range(6)]
        item=source(boards+[boards[0],boards[1],[1,27,22,40]],[50]*9)
        original=copy.deepcopy(item); allowed=list(range(9))
        split=split_paired_probe_rows(item,allowed,20261011)
        self.assertEqual(split,split_paired_probe_rows(item,allowed[::-1],20261011))
        self.assertEqual(item,original)
        self.assertEqual(set(split['train']+split['development']),set(range(8)))
        self.assertEqual(split['excluded'],[8])
        train={board_family(item['labels'][i]['input']['state']['board']) for i in split['train']}
        dev={board_family(item['labels'][i]['input']['state']['board']) for i in split['development']}
        self.assertFalse(train.intersection(dev)); self.assertEqual(len(dev),1)
        self.assertEqual(len(split['development_families']),1)

    def test_split_rejects_empty_duplicate_out_of_bounds_or_insufficient_families(self):
        item=source([[1,27,22,25]]*2,[50,50])
        for allowed in ([],[0,0],[0,2],[-1],[True],[0.5],[0,1]):
            with self.assertRaises(ValueError): split_paired_probe_rows(item,allowed,20261011)
        with self.assertRaises(ValueError): split_paired_probe_rows(item,[0],True)


if __name__=='__main__': unittest.main()
