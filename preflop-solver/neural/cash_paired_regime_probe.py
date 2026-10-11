"""Non-serving paired-turn correction experiment, not a native model schema.

This public-state-only gate isolates a diagnosed regime; it never selects
hands or changes the actual belief ranges. The development split removes
whole flop families from adapter fitting, using only caller-authorized original
training rows. A frozen parent may already have seen these development labels;
they are not untouched full-model evaluation or release evidence.
"""
import hashlib

import numpy as np

from cash_profiles import equal_cash_investment_units, profile_rules, rules_digest
from native_value_dataset import board_family


def paired_small_pot_coefficients(source):
    game=source.get('game',{})
    rules=game.get('cash_rules')
    if (game.get('effective_stack_bb')!=20. or rules!=profile_rules('nl25')
            or source.get('rules_sha256')!=rules_digest(rules)):
        raise ValueError('paired regime probe requires pinned NL25 20bb rules')
    result=[]
    for label in source['labels']:
        request=label['input']; state=request['state']; board=state['board']
        if (request['game'].get('effective_stack_bb')!=20.
                or request['game'].get('cash_rules')!=rules
                or len(board)!=4 or len(set(board))!=4
                or any(type(card) is not int or not 0<=card<52 for card in board)):
            raise ValueError('paired regime probe requires matching exact public turn states')
        invested=equal_cash_investment_units(state['invested_bb'],25)
        if invested>=500:
            raise ValueError('all-in cash continuations require exact settlement')
        paired=board[-1]//4 in [card//4 for card in board[:3]]
        x=min(1.,max(0.,(100-invested)/25.)) if paired else 0.
        result.append(x*x*(3.-2.*x))
    return np.asarray(result,np.float32)


def split_paired_probe_rows(source,allowed_rows,seed):
    rows=np.asarray(allowed_rows)
    if (rows.ndim!=1 or rows.dtype.kind not in 'iu' or len(rows)==0
            or (rows<0).any() or (rows>=len(source['labels'])).any()
            or len(np.unique(rows))!=len(rows) or type(seed) is not int or not 0<=seed<2**64):
        raise ValueError('paired regime development split requires unique authorized training rows and a seed')
    coefficients=paired_small_pot_coefficients(source)
    families={int(i):board_family(source['labels'][int(i)]['input']['state']['board'])
              for i in rows if coefficients[i]>0}
    unique=set(families.values())
    if len(unique)<5:
        raise ValueError('paired regime development split requires at least five flop families')
    ordered=sorted(unique,key=lambda family:(hashlib.sha256(
        f'{seed}|{family}'.encode()).hexdigest(),family))
    dev=set(ordered[:max(1,len(ordered)//5)])
    return dict(train=sorted(i for i,family in families.items() if family not in dev),
        development=sorted(i for i,family in families.items() if family in dev),
        excluded=sorted(int(i) for i in rows if int(i) not in families),
        development_families=[list(family) for family in ordered if family in dev],
        seed=seed,status='non_serving_research',active=False)
