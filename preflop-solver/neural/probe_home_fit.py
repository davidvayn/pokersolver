#!/usr/bin/env python3
"""Read-only, fixed-target Home fitting/arrival ablation; never a release test.

All budgets are checkpoints of one identical minibatch stream. Splits group
encoded information sets, not independent hand families (absent from replay).
The arrival ablation retains the same width, initialization, targets and split.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from probe_regret_fit_budget import split_indices
from train import (ActionScorer, MAX_POLICY_ACTIONS, atomic_json, input_feature_count,
                   linear_layers, make_compiled_group_regression_step, make_compiled_policy_step,
                   state_feature_count, stratified_replay_indices)


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def read_replay(run, name):
    state = json.loads((run / 'state.json').read_text())
    config = state['config']
    if config.get('cash_profile') is not None:
        raise ValueError('Home probe cannot reinterpret cash replay')
    info = state['reservoirs'][name]
    width = state_feature_count(card_arrival_memory=config.get('card_arrival_memory', False))
    cap, size = config['reservoir_capacity'], info['size']
    if not 0 < size <= cap or info['max_actions'] != MAX_POLICY_ACTIONS:
        raise ValueError('invalid grouped replay metadata')
    arrays, hashes = {}, {}
    for key, suffix, dtype, shape in (
        ('states', 'states.f16', np.float16, (width,)),
        ('actions', 'actions.f16', np.float16, (8, 9)),
        ('targets', 'targets.f32', np.float32, (8,)),
        ('masks', 'masks.u8', np.uint8, (8,)),
        ('weights', 'weights.f32', np.float32, (1,)),
        ('streets', 'street.u8', np.uint8, ()),
    ):
        path = run / 'replay' / f'{name}.{suffix}'
        if path.stat().st_size != cap * int(np.prod(shape)) * np.dtype(dtype).itemsize:
            raise ValueError(f'replay byte length disagrees with declared width: {path}')
        arrays[key] = np.memmap(path, mode='r', dtype=dtype, shape=(cap, *shape))[:size]
        hashes[str(path)] = digest(path)
    mask = arrays['masks']
    if (not np.isin(mask, (0, 1)).all() or np.any(mask.sum(1) == 0)
            or np.any(arrays['streets'] > 3) or np.any(arrays['weights'] <= 0)
            or any(not np.isfinite(a).all() for a in arrays.values())):
        raise ValueError('invalid fixed replay values')
    if name == 'average_strategy' and (np.any(arrays['targets'] < 0)
            or not np.allclose((arrays['targets'] * mask).sum(1), 1., atol=2e-6)):
        raise ValueError('average targets must be probability vectors')
    return state, arrays, hashes


def batch(arrays, ids, *, hide_arrivals=False, corrections=None):
    states = np.asarray(arrays['states'][ids], np.float32).copy()
    if hide_arrivals:
        if states.shape[1] != 820:
            raise ValueError('arrival ablation needs the v5 replay schema')
        states[:, 716:820] = 0
    features = np.concatenate((np.broadcast_to(states[:, None, :], (len(ids), 8, states.shape[1])),
                               np.asarray(arrays['actions'][ids], np.float32)), axis=2)
    weights = np.asarray(arrays['weights'][ids], np.float32)
    if corrections is not None:
        weights = weights * corrections[:, None]
    return tuple(mx.array(a) for a in (features, np.asarray(arrays['targets'][ids]),
                                      np.asarray(arrays['masks'][ids], np.float32), weights))


def statistics(scores, truth, mask, weights, policy_head):
    counts = mask.sum(1)
    if policy_head:
        logits = np.where(mask > 0, scores, -1e30)
        p = np.exp(logits - logits.max(1, keepdims=True)) * mask
        p /= p.sum(1, keepdims=True)
        q = truth
        loss = -(q * np.log(np.maximum(p, 1e-30)) * mask).sum(1)
        # KL removes irreducible target entropy; training still uses cross entropy.
        kl = (q * (np.log(np.maximum(q, 1e-30)) - np.log(np.maximum(p, 1e-30))) * mask).sum(1)
    else:
        def regret_mix(values):
            positive = np.maximum(values, 0) * mask
            total = positive.sum(1, keepdims=True)
            return np.where(total > 1e-10, positive / np.maximum(total, 1e-10), mask / counts[:, None])
        p, q = regret_mix(scores), regret_mix(truth)
        loss = (((scores - truth)**2) * mask).sum(1) / counts
        kl = np.zeros_like(loss)
    weight = weights.reshape(-1)
    return {'mass': float(weight.sum()), 'loss_sum': float(loss @ weight),
            'kl_sum': float(kl @ weight),
            'mix_mae_sum': float((np.abs(p-q).sum(1)/counts) @ weight),
            'primary_sum': float((p.argmax(1) == q.argmax(1)) @ weight), 'decisions': len(scores)}


def metrics(model, arrays, ids, policy_head, hide_arrivals):
    groups = {'all': {}}
    for chunk in np.array_split(ids, max(1, (len(ids)+127)//128)):
        x, y, m, w = batch(arrays, chunk, hide_arrivals=hide_arrivals)
        with mx.stream(mx.cpu):
            prediction = np.asarray(model(x.reshape(-1, x.shape[-1]))).reshape(len(chunk), 8)
        truth, mask, weight = np.asarray(y), np.asarray(m), np.asarray(w)
        street = arrays['streets'][chunk]
        for key, selection in [('all', np.ones(len(chunk), bool)), *[(str(s), street == s) for s in range(4)]]:
            if not selection.any():
                continue
            sums = statistics(prediction[selection], truth[selection], mask[selection], weight[selection], policy_head)
            target = groups.setdefault(key, {})
            for field, value in sums.items():
                target[field] = target.get(field, 0) + value
    def finish(sums):
        mass = sums['mass']
        return {'decisions': sums['decisions'],
                'weighted_cross_entropy' if policy_head else 'weighted_mse_stack_fraction': sums['loss_sum']/mass,
                **({'weighted_kl': sums['kl_sum']/mass} if policy_head else {}),
                'target_mix_mae': sums['mix_mae_sum']/mass,
                'target_primary_agreement': sums['primary_sum']/mass}
    return {key: finish(value) for key, value in groups.items()}


def probe(run, name, budgets, variants):
    state, arrays, hashes = read_replay(run, name)
    config = state['config']
    # Group the union of both representations: histories collapsed by the
    # ablation must not land on opposite sides of its holdout.
    train_ids, test_ids = split_indices(arrays['states'][:, :716])
    if min(len(train_ids), len(test_ids)) < 64:
        raise ValueError('not enough independent encoded states for a fitting probe')
    # Fixed metric subsets bound cost; never used to alter targets or the split.
    metric_rng = np.random.default_rng(2919031)
    measured_train = metric_rng.choice(train_ids, min(2048, len(train_ids)), replace=False)
    measured_test = metric_rng.choice(test_ids, min(2048, len(test_ids)), replace=False)
    policy_head = name == 'average_strategy'
    results = []
    for hide in variants:
        mx.random.seed(2919001 + ('advantage_p0', 'advantage_p1', 'average_strategy').index(name))
        width = input_feature_count(card_arrival_memory=config.get('card_arrival_memory', False))
        model = ActionScorer(width, tuple(config['hidden_sizes']))
        first = linear_layers(model)[0]
        weights = np.asarray(first.weight).copy()
        weights[:, 652:width-9] = 0  # same neutral texture/arrival initialization in both arms
        first.weight = mx.array(weights)
        optimizer = optim.AdamW(learning_rate=config['learning_rate'], weight_decay=1e-5)
        optimizer.init(model.trainable_parameters())
        step = (make_compiled_policy_step if policy_head else make_compiled_group_regression_step)(model, optimizer)
        rng = np.random.default_rng(2919003)
        trials = []
        for update in range(1, max(budgets)+1):
            local, corrections = stratified_replay_indices(arrays['streets'][train_ids], len(train_ids),
                config['batch_size'], config['replay_street_proposal'], rng)
            loss = step(*batch(arrays, train_ids[local], hide_arrivals=hide, corrections=corrections))
            mx.eval(loss, model.parameters(), optimizer.state)
            if not np.isfinite(float(loss)):
                raise RuntimeError('nonfinite fixed-target fit')
            if update in budgets:
                row = {'steps': update, 'train': metrics(model, arrays, measured_train, policy_head, hide),
                       'heldout': metrics(model, arrays, measured_test, policy_head, hide)}
                trials.append(row)
                print(json.dumps({'run': str(run), 'head': name, 'hide_arrivals': hide, **row}), flush=True)
        results.append({'hide_arrivals': hide, 'trials': trials})
    if any(digest(Path(path)) != value for path, value in hashes.items()):
        raise RuntimeError('source replay changed during the readonly probe')
    return {'run': str(run), 'head': name, 'source_config_hash': state['config_hash'],
            'source_state_sha256': digest(run/'state.json'), 'replay_sha256s': hashes,
            'records': len(arrays['states']), 'train_states': len(train_ids), 'heldout_states': len(test_ids), 'results': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budgets', default='240,480,960')
    parser.add_argument('--head', choices=('advantage_p0', 'advantage_p1', 'average_strategy'), action='append')
    parser.add_argument('--ablate-arrivals', action='store_true')
    args = parser.parse_args()
    budgets = tuple(int(x) for x in args.budgets.split(','))
    if not budgets or min(budgets) < 1 or max(budgets) > 2000 or args.output.exists():
        raise ValueError('invalid bounded budgets or existing output')
    report = {'schema': 'hu-home-fixed-target-fitting-v1', 'release_eligible': False,
              'mlx_enable_tf32':os.environ.get('MLX_ENABLE_TF32','unset'),
              'split': 'legacy-projected information-set groups shared by both ablation arms; not independent hand families',
              'interpretation': 'Fresh fixed-target fitting diagnostics, not CFR training or playing strength.', 'results': []}
    for run in args.run:
        for head in args.head or ('advantage_p0', 'advantage_p1', 'average_strategy'):
            report['results'].append(probe(run, head, budgets, (False, True) if args.ablate_arrivals else (False,)))
            atomic_json(args.output, report)


if __name__ == '__main__':
    main()
