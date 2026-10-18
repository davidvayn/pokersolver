#!/usr/bin/env python3
"""Compare offline estimators on identical frozen policies and hidden-block seeds."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from train import atomic_json
from qualify_depth_repair_values import reference_precision_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    experiment = json.loads((args.experiment / 'summary.json').read_text())
    if experiment.get('status') != 'completed_not_activated':
        raise ValueError('requires a completed frozen comparison')
    args.output.mkdir(parents=True, exist_ok=False)
    binary = args.output / 'sampler'
    shutil.copy2(args.binary, binary)
    # First saved batch: four independent whole-hand roots per street.
    records = json.loads((args.experiment / 'response-roots.json').read_text())[:16]
    inputs = args.output / 'decisions.json'
    atomic_json(inputs, records)
    report = {'schema': 'hu-offline-frozen-value-precision-probe-v1',
        'status': 'running_not_activated', 'depth_bb': experiment['depth_bb'],
        'decisions': len(records), 'hidden_blocks': 512, 'inner_rollouts': 2,
        'sampler_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
        'input_sha256': hashlib.sha256(inputs.read_bytes()).hexdigest(),
        'source_experiment_sha256': hashlib.sha256((args.experiment / 'summary.json').read_bytes()).hexdigest(),
        'active_models_changed': False, 'candidates': []}
    for index, seed in enumerate(experiment['seeds']):
        networks = args.experiment / f'averaged-{seed}/frozen-values.networks.json'
        results = {}
        # Reverse order in seed two; still not an isolated wall-time benchmark.
        for integrate in ([False, True] if index == 0 else [True, False]):
            label = 'integrated' if integrate else 'sampled'
            path = args.output / f'{seed}-{label}.json'
            argv = [str(binary), 'neural-frozen-values', '--networks', str(networks),
                '--input', str(inputs), '--output', str(path), '--effective-stack-bb', str(experiment['depth_bb']),
                '--hidden-blocks', '512', '--rollouts-per-block', '2', '--seed', '2818051',
                '--action-abstraction-json', json.dumps(json.loads(networks.read_text())['qualification_game']['action_abstraction'])]
            if integrate:
                argv.append('--integrate-terminal-responses')
            started = time.monotonic()
            subprocess.run(argv, check=True, capture_output=True, text=True, timeout=600)
            seconds = time.monotonic() - started
            reference = json.loads(path.read_text())
            if (reference['input_sha256'] != report['input_sha256']
                    or reference['source_network_sha256s'] != [hashlib.sha256(networks.read_bytes()).hexdigest()]
                    or reference['integrates_terminal_responses'] != integrate):
                raise RuntimeError('offline estimator lost its input/policy/estimator identity')
            errors = np.concatenate([r['standard_errors_bb'] for r in reference['records']])
            results[label] = {**reference_precision_summary(reference['records'], 512),
                'seconds': seconds, 'mean_action_sampling_variance_bb2': float(np.mean(errors ** 2)),
                'report_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        ratio = results['integrated']['mean_action_sampling_variance_bb2'] / results['sampled']['mean_action_sampling_variance_bb2']
        candidate = {'seed': seed, 'estimators': results, 'variance_ratio_integrated_over_sampled': ratio,
            'variance_times_wall_time_ratio': ratio * results['integrated']['seconds'] / results['sampled']['seconds']}
        report['candidates'].append(candidate)
        atomic_json(args.output / 'summary.json', report)
        print(json.dumps(candidate), flush=True)
    report.update(status='completed_not_activated',
        interpretation='Exact terminal expectations remove terminal-action noise, not hidden-card variance or prediction bias. Stratified pilot, not release qualification. Wall times may include host contention.')
    atomic_json(args.output / 'summary.json', report)


if __name__ == '__main__':
    main()
