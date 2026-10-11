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


def exact_cash_runout_distribution(board, ranges, investments, rules):
    """Research-only mean, variance and eight-bin future river-equity histogram.

    Condition each river equity on the query's compatible opponent range. Its
    chance weight is compatible mass after that river / (44 * original mass),
    not uniform over the46 rivers remaining after board plus queried hand.
    Bins are [0,1/8), ... [7/8,1]. Zero compatible support returns all zeros;
    zero own reach does not remove otherwise legal counterfactual queries.
    No betting policy, equilibrium value or future/private card is supplied.
    Legacy callers do not pay for these additional opt-in accumulations.
    """
    cards = np.asarray(board)
    beliefs = np.asarray(ranges)
    if (cards.shape != (4,) or cards.dtype.kind not in 'iu'
            or (cards < 0).any() or (cards >= 52).any() or len(np.unique(cards)) != 4):
        raise ValueError('runout distribution requires four unique exact public cards')
    if (beliefs.shape != (2,1326) or beliefs.dtype.kind not in 'iuf'
            or not np.isfinite(beliefs).all() or (beliefs < 0).any()
            or not (beliefs.sum(axis=1) > 0).all()
            or (beliefs[:, ~legal_combos(cards)] != 0).any()):
        raise ValueError('runout distribution requires finite nonnegative legal paired beliefs')
    beliefs = beliefs.astype(np.float64, copy=False)
    amount = equal_cash_investment_units(investments, rules['unitsPerBb'])
    values = _cached(tuple(map(int,cards)), beliefs.tobytes(), amount,
                     json.dumps(rules,sort_keys=True), include_distribution=True)
    return tuple(value.copy() for value in values[1:])


@lru_cache(maxsize=16)
def _cached(board, range_bytes, amount, rules_json, include_distribution=False):
    rules = json.loads(rules_json); ranges = np.frombuffer(range_bytes,dtype=np.float64).reshape(2,1326)
    gross = 2 * amount; whole, remainder = divmod(gross * rules["rake"]["rateBasisPoints"],10000)
    rake = min(rules["rake"]["capUnits"],whole+int(remainder > 5000 or remainder == 5000 and whole % 2 == 1))
    unit = rules["unitsPerBb"]; net = gross-rake
    win, lose = (amount-rake)/unit, -amount/unit
    ties = [(net//2-amount)/unit, ((net+1)//2-amount)/unit]
    numerators = np.zeros((2,1326),dtype=np.float64)
    equity_numerators = np.zeros_like(numerators)
    if include_distribution:
        second_numerators = np.zeros_like(numerators)
        histogram_numerators = np.zeros((2,1326,8),dtype=np.float64)
        keys = np.arange(1326)
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
            if include_distribution:
                conditional = np.clip(np.divide(better_mass+.5*tie_mass, masses[player],
                    out=np.zeros(1326), where=masses[player]>0),0.,1.)
                river_mass = np.where(river_legal, masses[player], 0.)
                second_numerators[player] += river_mass * conditional ** 2
                bins = np.minimum(7,(conditional * 8).astype(np.int64))
                histogram_numerators[player,keys,bins] += river_mass
    denominator = compatible_masses(ranges) * 44
    result = np.divide(numerators,denominator,out=np.zeros_like(numerators),where=denominator>0)
    result[:,~legal] = 0
    equity = np.clip(np.divide(equity_numerators,denominator,out=np.zeros_like(numerators),where=denominator>0),0.,1.)
    equity[:,~legal] = 0
    if include_distribution:
        second = np.divide(second_numerators,denominator,
            out=np.zeros_like(numerators),where=denominator>0)
        variance = np.clip(second-equity**2,0.,.25)
        variance[:,~legal] = 0
        histogram = np.divide(histogram_numerators,denominator[:,:,None],
            out=np.zeros_like(histogram_numerators),where=denominator[:,:,None]>0)
        histogram[:,~legal,:] = 0
        return result, equity, variance, histogram
    return result, equity
