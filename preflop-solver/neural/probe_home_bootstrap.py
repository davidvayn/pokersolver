#!/usr/bin/env python3
"""Compare frozen CPU/GPU bootstrap priors with the real Rust traversal scorer."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import mlx.core as mx
import numpy as np

from probe_home_fit import batch, digest, read_replay
from train import ActionScorer, atomic_json, input_feature_count, scorer_json


def prepare(run):
    fixture, predictions, hashes = [], [], {}
    for name in ('advantage_p0', 'advantage_p1'):
        state, arrays, replay_hashes = read_replay(run, name)
        config = state['config']
        model = ActionScorer(input_feature_count(card_arrival_memory=config.get('card_arrival_memory', False)),
                             tuple(config['hidden_sizes']))
        weights = run/state['checkpoint']/f'{name}.safetensors'
        model.load_weights(str(weights))
        ids = np.random.default_rng(2919041).choice(len(arrays['states']), min(128, len(arrays['states'])), replace=False)
        x, _, mask, _ = batch(arrays, ids)
        scores = {}
        for label, device in (('cpu', mx.cpu), ('gpu', mx.gpu)):
            with mx.stream(device):
                scores[label] = np.asarray(model(x.reshape(-1, x.shape[-1]))).reshape(len(ids),8).copy()
        counts = np.asarray(mask).sum(1).astype(int)
        rows = [{'state': arrays['states'][i].astype(float).tolist(),
                 'actions': arrays['actions'][i, :n].astype(float).tolist()} for i, n in zip(ids, counts)]
        fixture.append({'network': scorer_json(model), 'rows': rows})
        predictions.append({key: [row[:n] for row, n in zip(value, counts)] for key, value in scores.items()})
        hashes.update(replay_hashes)
        hashes[str(weights)] = digest(weights)
    return state, fixture, predictions, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cargo', default='cargo')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'hu-home-bootstrap-inference-probe-v1', 'release_eligible': False,
              'mlx_enable_tf32':os.environ.get('MLX_ENABLE_TF32','unset'), 'results': []}
    root = Path(__file__).resolve().parents[2]
    for index, run in enumerate(args.run):
        state, fixture, predictions, hashes = prepare(run)
        inputs, outputs = args.output/f'{index}.input.json', args.output/f'{index}.native.json'
        atomic_json(inputs, fixture)
        environment = dict(os.environ, POKER_BOOTSTRAP_FIXTURE=str(inputs), POKER_BOOTSTRAP_OUTPUT=str(outputs))
        subprocess.run([args.cargo, 'test', '--release', '--lib', '--manifest-path', str(root/'preflop-solver/Cargo.toml'),
                        'real_weight_bootstrap_inference_fixture', '--', '--ignored'], cwd=root,
                       env=environment, check=True)
        native = json.loads(outputs.read_text())
        heads = []
        for name, actual, python in zip(('advantage_p0','advantage_p1'), native, predictions, strict=True):
            native_values = np.concatenate(actual)
            comparisons = {}
            for device, rows in python.items():
                delta = (np.concatenate(rows)-native_values)*state['config']['depth_bb']
                comparisons[device] = {'maximum_absolute_difference_bb':float(np.max(np.abs(delta))),
                    'mean_absolute_difference_bb':float(np.mean(np.abs(delta))),
                    'p95_absolute_difference_bb':float(np.quantile(np.abs(delta),.95))}
            heads.append({'head':name, 'actions':len(native_values), 'differences_from_native':comparisons})
        if any(digest(Path(path)) != value for path,value in hashes.items()):
            raise RuntimeError('source changed during parity probe')
        report['results'].append({'run':str(run),'source_config_hash':state['config_hash'],
            'native_scorer_source_sha256':digest(root/'preflop-solver/src/blueprint/neural.rs'),
            'source_sha256s':hashes, 'native_fixture_sha256':digest(inputs),
            'native_output_sha256':digest(outputs), 'heads':heads})
        atomic_json(args.output/'summary.json', report)
        print(json.dumps(report['results'][-1]), flush=True)


if __name__ == '__main__':
    main()
