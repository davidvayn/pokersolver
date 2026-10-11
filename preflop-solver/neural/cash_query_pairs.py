"""Match observed current-search beliefs to final-average controls.

The caller must pin the native binary, trace bytes, proposer and authentic
root lineage. A native query key is retained, not reconstructed with Python
float serialization. This module validates the observable contract and
replays public actions; it generates neither beliefs nor teacher values.
"""
from __future__ import annotations

import copy
import math
import re

import numpy as np

from cash_flop_leaves import leaf_input
from cash_profiles import equal_cash_investment_units
from native_value_dataset import compatible_masses, legal_combos
from run_cash_flop_pilot import validate_solution


def paired_trace_inputs(trace: dict, root: dict, network_sha: str, rules_sha: str,
                        *, reference_iterations: int = 128) -> list[dict]:
    """Return bounded matched requests; preserve actual zeros and cent floats.

    Only ranges differ between arms. The new reference budget is explicit;
    the captured 2-update prediction request is preserved separately. Missing
    live average support rejects the pair, with no invented control or range.
    """
    if type(reference_iterations) is not int or not 2 <= reference_iterations <= 128:
        raise ValueError('cash query pairs permit 2..128 reference updates')
    if (trace.get('schema') != 'cash-current-policy-turn-trace-v1'
            or trace.get('status') != 'research_only' or trace.get('active') is not False
            or not isinstance(trace.get('trace_input_sha256'), str)
            or re.fullmatch(r'[a-f0-9]{64}', trace['trace_input_sha256']) is None):
        raise ValueError('cash trace must retain its research identity')
    options = trace.get('trace_input', {})
    rounds, limit = options.get('sample_rounds'), options.get('leaves_per_round')
    seed = options.get('reach_sampling_seed')
    config = root['solve_input']
    if (set(options) - {'solve','sample_rounds','leaves_per_round','reach_sampling_seed'}
            or options.get('solve') != config
            or not isinstance(rounds, list) or not 1 <= len(rounds) <= 4
            or any(type(r) is not int or not 1 <= r <= config['iterations'] for r in rounds)
            or rounds != sorted(set(rounds)) or type(limit) is not int or not 1 <= limit <= 4
            or ('reach_sampling_seed' in options and
                (type(seed) is not int or not 0 <= seed < 2**64))):
        raise ValueError('cash trace changed its solve or bounded selection')
    selection = ('seeded-exponential-race-unique-queries-weighted-by-joint-root-reach'
                 if seed is not None else 'smallest-sha256-unique-queries-per-selected-training-round')
    if trace.get('selection_method') != selection:
        raise ValueError('cash trace selection provenance differs from options')
    validate_solution(trace['solution'], config, network_sha, rules_sha)
    queries, observed = trace.get('queries'), trace.get('observed_turn_query_events_per_round')
    if (not isinstance(queries, list) or len(queries) > len(rounds)*limit
            or not isinstance(observed, dict)
            or any(str(r) not in {str(n) for n in rounds} or type(n) is not int or n <= 0
                   for r,n in observed.items())):
        raise ValueError('cash trace exceeded its observation budget')
    seen, counts, result = set(), {}, []
    for query in queries:
        if not isinstance(query, dict) or set(query) != {
                'round','query_sha256','flop_public_history','reach_probability_from_flop_root','input'}:
            raise ValueError('cash query contains unexpected or private inputs')
        round_, digest, reach = query['round'], query['query_sha256'], query['reach_probability_from_flop_root']
        if (type(round_) is not int or round_ not in rounds or not isinstance(digest,str)
                or re.fullmatch(r'[a-f0-9]{64}',digest) is None or digest in seen
                or type(reach) not in (int,float) or not math.isfinite(reach) or not 0 < reach <= 1):
            raise ValueError('cash query lacks unique selected positive-reach identity')
        seen.add(digest); counts[round_] = counts.get(round_,0)+1
        if counts[round_] > limit or observed.get(str(round_),0) < counts[round_]:
            raise ValueError('cash query was not in its bounded observed round')
        history, captured = query['flop_public_history'], query['input']
        prefix = config['state']['public_history']
        if (not isinstance(history,list) or history[:len(prefix)] != prefix
                or len(history) < len(prefix)+3 or history[-1] != 'deal:Turn'
                or not isinstance(captured,dict) or not isinstance(captured.get('state'),dict)):
            raise ValueError('cash query is not a completed revealed flop branch')
        actions = history[len(prefix):-1]
        labels = []
        for index, action in enumerate(actions):
            actor = 1 if index%2 == 0 else 0
            if not isinstance(action,str) or not action.startswith(f'Flop:p{actor}:'):
                raise ValueError('cash query changed public actor order')
            labels.append(action.split(':',2)[2])
        board = captured['state'].get('board')
        if (not isinstance(board,list) or len(board) != 4
                or board[:3] != config['state']['board']):
            raise ValueError('cash query changed the flop or added a future card')
        average = leaf_input(root, trace['solution'], board[-1], labels, 2)
        if set(captured) != set(average) or set(captured['state']) != set(average['state']):
            raise ValueError('cash query changed its fresh-turn contract')
        actual_money = captured['state']['invested_bb']
        units = config['game']['cash_rules']['unitsPerBb']
        if (equal_cash_investment_units(actual_money,units)
                != equal_cash_investment_units(average['state']['invested_bb'],units)):
            raise ValueError('cash query changed the public branch payments')
        # Native cent arithmetic may serialize an adjacent f64. Retain the
        # exact captured representation in both arms rather than relabel it.
        average['state']['invested_bb'] = copy.deepcopy(actual_money)
        comparison = copy.deepcopy(captured)
        comparison['state']['ranges'] = average['state']['ranges']
        if comparison != average:
            raise ValueError('cash query differs from replay beyond its current ranges')
        raw = captured['state']['ranges']
        if (not isinstance(raw,list) or len(raw) != 2
                or any(not isinstance(row,list) or len(row) != 1326
                       or any(type(v) not in (int,float) for v in row) for row in raw)):
            raise ValueError('cash query ranges are not paired numeric combo vectors')
        ranges = np.asarray(raw,dtype=float); legal = legal_combos(board)
        if (not np.isfinite(ranges).all() or (ranges < 0).any()
                or (ranges[:,~legal] != 0).any()
                or not np.allclose(ranges.sum(axis=1),1.,atol=1e-10,rtol=0)
                or np.sum(ranges*compatible_masses(ranges),axis=1).min() <= 0):
            raise ValueError('cash query has invalid current public beliefs; no fallback')
        current = copy.deepcopy(captured)
        current['iterations'] = average['iterations'] = reference_iterations
        result.append(dict(round=round_,query_sha256=digest,branch_labels=labels,
                           reach_probability_from_flop_root=reach,
                           captured_prediction_input=copy.deepcopy(captured),
                           current=current,average=average))
    return sorted(result,key=lambda pair:(pair['round'],pair['query_sha256']))
