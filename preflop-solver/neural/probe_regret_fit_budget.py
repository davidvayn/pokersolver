#!/usr/bin/env python3
"""Read-only fixed-replay fit probe. Not a playing-strength or release test.

Split by encoded information set (not row); saved replay has no hand-family IDs.
Every budget starts from identical fresh weights. Original run data is mmap r.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from train import ActionScorer, atomic_json, linear_layers, make_compiled_group_regression_step


def split_indices(states):
    # All actions of repeated states share a split, irrespective of targets.
    buckets = np.array([int.from_bytes(hashlib.sha256(row.tobytes()).digest()[:4], 'little') % 10
                        for row in states])
    return np.flatnonzero(buckets < 8), np.flatnonzero(buckets >= 8)


def probe(run: Path, player: int, budgets=(240, 960)):
    state = json.loads((run / 'state.json').read_text())
    name = f'advantage_p{player}'
    size = state['reservoirs'][name]['size']
    cap = state['config']['reservoir_capacity']
    def read(suffix, dtype, shape):
        return np.memmap(run / 'replay' / f'{name}.{suffix}', mode='r', dtype=dtype, shape=(cap, *shape))[:size]
    states = read('states.f16', np.float16, (716,))
    actions = read('actions.f16', np.float16, (8, 9))
    targets = read('targets.f32', np.float32, (8,))
    masks = read('masks.u8', np.uint8, (8,))
    weights = read('weights.f32', np.float32, (1,))
    train, test = split_indices(states)
    if min(len(train), len(test)) < 64:
        raise ValueError('insufficient fixed replay for an information-set holdout')
    def arrays(ids):
        x = np.concatenate((np.broadcast_to(np.asarray(states[ids], np.float32)[:,None,:], (len(ids),8,716)),
                            np.asarray(actions[ids], np.float32)), axis=2)
        return tuple(mx.array(a) for a in (x, np.asarray(targets[ids]), np.asarray(masks[ids],np.float32), np.asarray(weights[ids])))
    def metrics(model, ids):
        squared = mass = mae = agree = 0.
        for chunk in np.array_split(ids, max(1, (len(ids)+127)//128)):
            x, y, m, w = arrays(chunk)
            predicted = np.asarray(model(x).squeeze(-1)); truth = np.asarray(y)
            mask, weight = np.asarray(m), np.asarray(w)
            per_decision = np.sum((predicted-truth)**2 * mask,axis=1) / mask.sum(1)
            squared += float(np.sum(per_decision * weight[:,0]))
            mass += float(np.sum(weight))
            def probabilities(v):
                v = np.maximum(v,0) * mask
                sums = v.sum(1,keepdims=True)
                return np.where(sums > 1e-10, v/np.maximum(sums,1e-10), mask/mask.sum(1,keepdims=True))
            p, q = probabilities(predicted), probabilities(truth)
            mae += float(np.sum(np.sum(abs(p-q),axis=1)/mask.sum(1)))
            agree += float(np.sum(np.argmax(p,axis=1) == np.argmax(q,axis=1)))
        return {'weighted_mse_stack_fraction': squared/mass, 'target_mix_mae': mae/len(ids), 'target_primary_agreement': agree/len(ids)}
    trials = []
    for budget in budgets:
        mx.random.seed(2919001 + player)
        model = ActionScorer(725, tuple(state['config']['hidden_sizes']))
        first = linear_layers(model)[0]
        initial = np.asarray(first.weight).copy(); initial[:,652:716] = 0
        first.weight = mx.array(initial)
        optimizer = optim.AdamW(learning_rate=state['config']['learning_rate'], weight_decay=1e-5)
        optimizer.init(model.trainable_parameters())
        step = make_compiled_group_regression_step(model,optimizer)
        rng = np.random.default_rng(2919003)
        for _ in range(budget):
            ids = rng.choice(train, size=512, replace=True)
            loss = step(*arrays(ids)); mx.eval(loss,model.parameters(),optimizer.state)
            if not np.isfinite(float(loss)): raise RuntimeError('nonfinite fit')
        result = {'steps':budget, 'train':metrics(model,train), 'heldout':metrics(model,test)}
        print(json.dumps({'run':str(run),'player':player,**result}),flush=True)
        trials.append(result)
    return {'run':str(run), 'player':player,'source_config_hash':state['config_hash'], 'records':size,
            'train_states':len(train),'heldout_states':len(test),'trials':trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('refuse to overwrite fit probe')
    report = {'schema':'hu-fixed-regret-fitting-budget-v1', 'release_eligible':False,
        'split':'encoded-information-set holdout; not hand-family heldout, not independent validation',
        'interpretation':'Fits historical noisy instantaneous targets; improvement is not evidence of equilibrium or playing strength.',
        'results':[probe(run,player) for run in args.run for player in (0,1)]}
    atomic_json(args.output,report)


if __name__ == '__main__': main()
