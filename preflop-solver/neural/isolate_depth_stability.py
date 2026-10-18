#!/usr/bin/env python3
"""Decompose seed disagreement before choosing another learning experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from depth_repair import frozen_reference
from train import atomic_json, expand_state_action
from validate_seeds import latest_artifact, load_artifact_model, load_run, policy, verify_artifact


def comparison(first: list[list[float]], second: list[list[float]]) -> dict:
    if not first or len(first) != len(second):
        raise ValueError('comparison requires aligned nonempty decision rows')
    errors, agreements = [], []
    for a, b in zip(first, second, strict=True):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if (a.ndim != 1 or a.shape != b.shape or not a.size
                or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b))
                or np.any(a < 0) or np.any(b < 0)
                or not np.isclose(a.sum(), 1) or not np.isclose(b.sum(), 1)):
            raise ValueError('invalid or unaligned policy probabilities')
        errors.append(float(np.abs(a - b).mean()))
        agreements.append(int(a.argmax() == b.argmax()))
    return {'decisions': len(first), 'frequency_mae': float(np.mean(errors)),
            'primary_agreement': float(np.mean(agreements))}


def decompose(students: list, references: list) -> dict:
    if len(students) != 2 or len(references) != 2:
        raise ValueError('decomposition requires two independent seeds')
    return {'exported_cross_seed': comparison(*students),
            'direct_average_cross_seed': comparison(*references),
            'compression_error_by_seed': [comparison(s, r) for s, r in zip(students, references)],
            'interpretation': 'Same decisions, exact own-reach historical averages. Components are not additive; agreement is not strength.'}


def draw_saved_mixture(corpora: list[list[dict]], count: int, seed: int) -> list[dict]:
    if not corpora or any(not rows for rows in corpora) or count < 1:
        raise ValueError('requires nonempty saved corpora')
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(count):
        rows = corpora[int(rng.integers(len(corpora)))]
        result.append(rows[int(rng.integers(len(rows)))])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--decisions-per-batch', type=int, default=128)
    args = parser.parse_args()
    if not 32 <= args.decisions_per_batch <= 256:
        raise ValueError('diagnostic budget is 32..256 decisions per batch')
    experiment = json.loads((args.experiment / 'summary.json').read_text())
    if experiment.get('status') != 'completed_not_activated':
        raise ValueError('requires completed immutable experiment')
    args.output.mkdir(parents=True, exist_ok=False)
    depth, seeds = experiment['depth_bb'], experiment['seeds']
    binary = args.experiment / 'sampler'
    if hashlib.sha256(binary.read_bytes()).hexdigest() != experiment['sampler_sha256']:
        raise ValueError('sampler changed')
    report = {'schema': 'hu-depth-stability-decomposition-v1', 'depth_bb': depth,
              'status': 'running_not_activated', 'batches': [], 'active_models_changed': False,
              'source_experiment_sha256': hashlib.sha256((args.experiment / 'summary.json').read_bytes()).hexdigest()}
    for batch in range(2):
        corpora = json.loads((args.experiment / f'reach-batch-{batch}.json').read_text())
        # Saved corpora already namespace separate whole-hand families.
        records = draw_saved_mixture(corpora, args.decisions_per_batch, 2818001 + batch)
        students, references, provenance = [], [], []
        for seed in seeds:
            run = args.experiment / f'averaged-{seed}'
            state, _ = load_run(run)
            artifact = latest_artifact(state)
            integrity = verify_artifact(artifact)
            if integrity['sha256'] != experiment['artifacts'][f'averaged-{seed}']['sha256']:
                raise ValueError('evaluated artifact changed')
            model = load_artifact_model(Path(artifact['source']))
            reference = frozen_reference(binary, model, records, depth,
                args.output / f'batch-{batch}-seed-{seed}.json',
                snapshots=run / 'policy-history/manifest.json', policy_only=True)
            with mx.stream(mx.cpu):
                students.append([policy(model, np.stack([expand_state_action(r['state'], a, depth)
                    for a in r['actions']])).tolist() for r in records])
            references.append([r['probabilities'] for r in reference['records']])
            provenance.append({'seed': seed, 'artifact_sha256': integrity['sha256'],
                               'snapshot_sha256s': reference['source_network_sha256s']})
        result = decompose(students, references)
        result.update(provenance=provenance, street_breakdown={})
        for street in ('preflop', 'flop', 'turn', 'river'):
            ids = [i for i, r in enumerate(records) if r['state']['street'] == street]
            if ids:
                result['street_breakdown'][street] = decompose(
                    [[rows[i] for i in ids] for rows in students],
                    [[rows[i] for i in ids] for rows in references])
        report['batches'].append(result)
        atomic_json(args.output / 'summary.json', report)
        print(json.dumps({'stage': 'decomposition', 'depth': depth, 'batch': batch,
            **{k: v for k, v in result.items() if k not in ('provenance', 'street_breakdown')}}), flush=True)
    report['status'] = 'completed_not_activated'
    atomic_json(args.output / 'summary.json', report)


if __name__ == '__main__':
    main()
