"""Exact 44-river own-payoff baseline with blockers and odd-cent ties.

This is a forced-checkdown reference, NOT an equilibrium continuation oracle.
Each query gets its own payoff against the compatible opponent range.
"""
from functools import lru_cache
import json
import numpy as np
from native_value_dataset import COMBOS, compatible_masses, legal_combos
from train_public_value_network import COMBO_CONFLICTS, evaluate_cards
from cash_profiles import equal_cash_investment_units


def exact_cash_checkdown(board, ranges, investments, rules):
    return exact_cash_checkdown_features(board, ranges, investments, rules)[0]


def exact_cash_checkdown_features(board, ranges, investments, rules):
    """Own net payoff and pure win-plus-half-tie equity from the same runouts.

    Equity is an input feature, not a cash payoff conversion: odd-cent ties
    still use each seat's actual award in the independently retained baseline.
    Return copies so callers cannot corrupt a later cached result.
    """
    board = tuple(map(int,board)); ranges = np.asarray(ranges,dtype=np.float64)
    if len(board) != 4: raise ValueError("checkdown expects equal-investment turn roots")
    amount = equal_cash_investment_units(investments,rules["unitsPerBb"])
    return tuple(values.copy() for values in _cached(board,ranges.tobytes(),amount,json.dumps(rules,sort_keys=True)))


@lru_cache(maxsize=16)
def _cached(board, range_bytes, amount, rules_json):
    rules = json.loads(rules_json); ranges = np.frombuffer(range_bytes,dtype=np.float64).reshape(2,1326)
    gross = 2 * amount; whole, remainder = divmod(gross * rules["rake"]["rateBasisPoints"],10000)
    rake = min(rules["rake"]["capUnits"],whole+int(remainder > 5000 or remainder == 5000 and whole % 2 == 1))
    unit = rules["unitsPerBb"]; net = gross-rake
    win, lose = (amount-rake)/unit, -amount/unit
    ties = [(net//2-amount)/unit, ((net+1)//2-amount)/unit]
    numerators = np.zeros((2,1326),dtype=np.float64)
    equity_numerators = np.zeros_like(numerators)
    legal = legal_combos(board)
    for river in range(52):
        if river in board: continue
        river_legal = legal & ~(COMBOS == river).any(axis=1)
        strengths = np.zeros(1326,dtype=np.int64)
        for key in np.flatnonzero(river_legal):
            strengths[key] = evaluate_cards([*board,river,*map(int,COMBOS[key])])
        _, group = np.unique(strengths,return_inverse=True)
        masked = ranges * river_legal[None,:]; masses = compatible_masses(masked)
        conflict_strengths = strengths[COMBO_CONFLICTS]
        for player in (0,1):
            opponent = masked[1-player]
            mass_by_group = np.bincount(group,weights=opponent)
            before = np.concatenate(([0.],np.cumsum(mass_by_group)[:-1]))
            conflict_weights = opponent[COMBO_CONFLICTS]
            better_mass = before[group] - np.sum(conflict_weights * (conflict_strengths < strengths[:,None]),axis=1)
            tie_mass = mass_by_group[group] - np.sum(conflict_weights * (conflict_strengths == strengths[:,None]),axis=1)
            losing_mass = masses[player] - better_mass - tie_mass
            numerators[player] += np.where(river_legal, better_mass*win+losing_mass*lose+tie_mass*ties[player],0)
            equity_numerators[player] += np.where(river_legal, better_mass + .5*tie_mass, 0)
    denominator = compatible_masses(ranges) * 44
    result = np.divide(numerators,denominator,out=np.zeros_like(numerators),where=denominator>0)
    result[:,~legal] = 0
    equity = np.clip(np.divide(equity_numerators,denominator,out=np.zeros_like(numerators),where=denominator>0),0.,1.)
    equity[:,~legal] = 0
    return result, equity
