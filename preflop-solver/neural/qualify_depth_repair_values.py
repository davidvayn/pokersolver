#!/usr/bin/env python3
"""Bounded authentic-reach EV audit of a frozen matched pair; never activates it."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import mlx.core as mx
import numpy as np

from depth_repair import (ROOT, audit_exported_values, fit_frozen_value_head,
                          frozen_reference)
from frozen_ev_gate import frozen_ev_gate
from run_depth_pilots import PILOT_GRID
from train import atomic_json, expand_state_action
from validate_seeds import (evaluation_records, latest_artifact, load_artifact_model,
                            load_run, verify_artifact)


def sample_authentic_decisions(datasets: list[list[dict]], count: int, seed: int) -> list[dict]:
    """IID draws from the same equal-policy decision mixture as the validator.

    Keep duplicate draws and whole-hand namespaces. Removing duplicates or
    forcing street quotas would alter the intended sampling distribution.
    """
    if count < 1 or not datasets or any(not rows for rows in datasets):
        raise ValueError('authentic EV sampling requires nonempty policy corpora')
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(count):
        source = int(rng.integers(len(datasets)))
        row = datasets[source][int(rng.integers(len(datasets[source])))]
        if not 0 <= row['iteration'] < 1_000_000:
            raise ValueError('trajectory family exceeds its namespace')
        result.append({**row, 'iteration': source * 1_000_000 + row['iteration']})
    return result


def reference_precision_summary(records: list[dict], blocks: int) -> dict:
    errors = [np.asarray(row['standard_errors_bb'], dtype=float) for row in records]
    if blocks < 4 or not errors or any(row.ndim != 1 or not len(row)
            or not np.all(np.isfinite(row)) or np.any(row < 0) for row in errors):
        raise ValueError('reference errors must be finite, nonnegative and aligned with actions')
    maxima = np.array([row.max() for row in errors])
    projected = np.maximum(blocks, blocks * (maxima / .02)**2)
    return {'decision_precision_coverage': float(np.mean(maxima <= .02)),
            'action_precision_coverage': float(np.mean(np.concatenate(errors) <= .02)),
            'median_decision_max_se_bb': float(np.median(maxima)),
            'p95_decision_max_se_bb': float(np.quantile(maxima, .95)),
            'optimistic_outer_blocks_for_95_percent_decisions': int(math.ceil(np.quantile(projected, .95))),
            'projection_assumption': 'IID square-root sampling-error scaling only; not a guarantee and does not reduce neural bias.'}


def chunk_seed(seed: int, start: int) -> int:
    return (seed + start * 0x9e3779b97f4a7c15) & ((1 << 64) - 1)


def chunked_frozen_reference(binary: Path, model, records: list[dict], depth: int,
                             output: Path, *, blocks: int, seed: int = 2471701,
                             chunk_size: int = 32, exact_river_values: bool = False,
                             exact_closed_values: bool = False) -> dict:
    # Preserve qualification.rs's GLOBAL root-index seed schedule exactly.
    # Small subprocess batches avoid a single timeout discarding all labels.
    if not records or not 1 <= chunk_size <= 32 or blocks < 4 or len(records) * blocks > 1_000_000:
        raise ValueError('invalid bounded frozen reference chunk budget')
    directory = output.parent / f'{output.stem}-chunks'
    directory.mkdir(parents=True, exist_ok=False)
    chunks, receipts = [], []
    for start in range(0, len(records), chunk_size):
        path = directory / f'roots-{start:04d}.json'
        options = {}
        if exact_river_values: options['exact_river_values'] = True
        if exact_closed_values: options['exact_closed_values'] = True
        part = frozen_reference(binary, model, records[start:start + chunk_size], depth,
            path, seed=chunk_seed(seed, start), blocks=blocks, **options)
        if chunks and part['source_network_sha256s'] != chunks[0]['source_network_sha256s']:
            raise ValueError('chunked reference changed its frozen policy')
        chunks.append(part)
        receipts.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'start': start, 'seed': chunk_seed(seed, start)})
        print(json.dumps({'stage': 'reference_chunk_finished', 'depth_bb': depth,
                          'start': start, 'decisions': len(part['records'])}), flush=True)
    input_path = output.with_suffix('.input.json')
    atomic_json(input_path, records)
    shutil.copy2((directory / 'roots-0000.json').with_suffix('.networks.json'), output.with_suffix('.networks.json'))
    merged = {**chunks[0], 'records': [row for chunk in chunks for row in chunk['records']],
              'seed': seed, 'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(),
              'chunk_receipts': receipts,
              'generation': 'Native chunks with the unchanged global root-index RNG schedule.'}
    atomic_json(output, merged)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', choices=('control', 'averaged'), default='averaged')
    parser.add_argument('--decisions', type=int, default=128)
    parser.add_argument('--hidden-blocks', type=int, default=2048)
    parser.add_argument('--refit-steps', type=int, default=400)
    args = parser.parse_args()
    if (not 32 <= args.decisions <= 256 or not 512 <= args.hidden_blocks <= 4096
            or args.decisions * args.hidden_blocks > 1_000_000 or not 1 <= args.refit_steps <= 1000):
        raise ValueError('EV audit exceeds its bounded label/refit budget')
    experiment_path = args.experiment / 'summary.json'
    experiment = json.loads(experiment_path.read_text())
    if experiment.get('schema') != 'hu-longer-matched-depth-repair-v1' or experiment.get('status') != 'completed_not_activated':
        raise ValueError('EV audit requires a completed frozen matched comparison')
    depth, seeds = experiment['depth_bb'], experiment['seeds']
    args.output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(args.output).free < 20 * 1024**3:
        raise RuntimeError('frozen EV audit requires 20GiB free disk headroom')
    binary = args.output / 'sampler'
    shutil.copy2(args.experiment / 'sampler', binary)
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    if digest != experiment['sampler_sha256']:
        raise ValueError('frozen matched sampler identity changed')
    paths = [Path(__file__), *[Path(__file__).with_name(name) for name in
             ('depth_repair.py', 'train.py', 'validate_seeds.py', 'frozen_ev_gate.py', 'run_depth_pilots.py')]]
    sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    source_directory = args.output / 'sources'
    source_directory.mkdir()
    for path in paths:
        shutil.copy2(path, source_directory / path.name)
    for name, sha in experiment['source_sha256s'].items():
        if name in sources and sources[name] != sha:
            raise ValueError('frozen EV audit learning/evaluation source differs from the experiment')
    report = {'schema': 'hu-authentic-depth-repair-ev-audit-v1', 'status': 'running_not_activated',
        'depth_bb': depth, 'profile': args.profile, 'seeds': seeds,
        'source_experiment_sha256': hashlib.sha256(experiment_path.read_bytes()).hexdigest(),
        'source_sha256s': sources, 'sampler_sha256': digest,
        'decisions': args.decisions, 'hidden_blocks': args.hidden_blocks, 'inner_rollouts': 2,
        'active_models_changed': False, 'candidates': [], 'release_eligible': False}

    def save(stage: str, **data) -> None:
        atomic_json(args.output / 'summary.json', report)
        print(json.dumps({'stage': stage, 'depth_bb': depth, **data}), flush=True)

    save('predeclared')
    try:
        states, models, artifacts = [], [], []
        for seed in seeds:
            state, _ = load_run(args.experiment / f'{args.profile}-{seed}')
            artifact = latest_artifact(state)
            check = verify_artifact(artifact)
            if (check['sha256'] != experiment['artifacts'][f'{args.profile}-{seed}']['sha256']
                    or not check['descriptor_matches'] or not check['magic_valid']):
                raise ValueError('EV audit export identity changed')
            states.append(state); artifacts.append(check)
            models.append(load_artifact_model(Path(artifact['source'])))
        reach = [evaluation_records(ROOT, depth, 512, 2717123, 64, 32, True, False,
                    policy_model=model, action_abstraction=PILOT_GRID, binary_path=binary) for model in models]
        records = sample_authentic_decisions(reach, args.decisions, 2717127)
        report['street_counts'] = dict(Counter(row['state']['street'] for row in records))
        report['trajectory_families'] = len({row['iteration'] for row in records})
        atomic_json(args.output / 'decisions.json', records)
        save('authentic_corpus_sampled', street_counts=report['street_counts'])
        candidates = []
        for seed, model, state, artifact in zip(seeds, models, states, artifacts, strict=True):
            if shutil.disk_usage(args.output).free < 20 * 1024**3:
                raise RuntimeError('frozen EV audit reached its disk headroom guard')
            prefix = args.output / f'seed-{seed}'
            reference_path = prefix.with_suffix('.json')
            reference = chunked_frozen_reference(binary, model, records, depth, reference_path,
                seed=2717129, blocks=args.hidden_blocks)
            value = load_artifact_model(Path(latest_artifact(state)['source']), 'baselineActionValue')
            audit = audit_exported_values(model, value, records, reference, depth)
            precision = reference_precision_summary(reference['records'], args.hidden_blocks)
            policy_weighted_precision = []
            with mx.stream(mx.cpu):
                for record, row in zip(records, reference['records'], strict=True):
                    features = np.stack([expand_state_action(record['state'], action, depth) for action in record['actions']])
                    errors = .005 + np.logaddexp(0., np.asarray(value(mx.array(features)))[:, 1].astype(float))
                    policy_weighted_precision.append(float(np.sum(np.asarray(row['probabilities']) * (errors <= .02))))
            refit = fit_frozen_value_head(records, reference, depth, prefix, seed=2717131, steps=args.refit_steps)
            candidate = {'seed': seed, 'artifact_sha256': artifact['sha256'],
                'reference_network_sha256': reference['source_network_sha256s'][0],
                'reference_report_sha256': hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                'frozen_policy_inference_verified': True, 'independent_hidden_blocks': args.hidden_blocks,
                'heldout_decisions': len(records),
                'reference_decision_precision_coverage': precision['decision_precision_coverage'],
                'served_predicted_precision_coverage': float(np.mean(policy_weighted_precision)),
                'nominal_95_interval_coverage': audit['calibration']['nominal_95_interval_coverage']}
            candidates.append(candidate)
            report['candidates'].append({'qualification': candidate, 'reference_precision': precision,
                'calibration': audit['calibration'], 'separate_mean_refit': refit})
            save('candidate_audited', seed=seed, precision=candidate['reference_decision_precision_coverage'],
                 calibration=candidate['nominal_95_interval_coverage'])
        receipt = {'schema': 'hu-frozen-ev-qualification-v1', 'depth_bb': depth,
            'action_abstraction': PILOT_GRID,
            'uncertainty_scope': 'opponent_posterior_and_future_board_outer_blocks',
            'evaluation_distribution': 'heldout_authentic_reach_mixture', 'candidates': candidates,
            'source_experiment_summary_sha256': report['source_experiment_sha256'],
            'evaluation_seed': 2717123, 'decision_selection_seed': 2717127,
            'status': 'measured_not_activated',
            'limitations': '128-draw default audit, clustered trajectories possible; calibration is marginal, not a conditional guarantee. A negative receipt does not qualify any model.'}
        receipt['assessment'] = frozen_ev_gate(receipt, depth, PILOT_GRID, [a['sha256'] for a in artifacts])
        receipt_path = args.output / 'frozen-ev-qualification.json'
        atomic_json(receipt_path, receipt)
        report['qualification'] = receipt['assessment']
        destination = args.output / 'release-validation.json'
        argv = [sys.executable, str(Path(__file__).with_name('validate_seeds.py')),
            *[str(args.experiment / f'{args.profile}-{seed}') for seed in seeds],
            '--round', str(experiment['rounds']), '--traversals', '512', '--seed', '2717137',
            '--evaluation-binary', str(binary), '--evaluation-binary-sha256', digest,
            '--frozen-ev-qualification', str(receipt_path), '--output', str(destination)]
        subprocess.run(argv, cwd=ROOT, check=True, capture_output=True, text=True, timeout=600)
        validation = json.loads(destination.read_text())
        report['release_validation'] = {'path': str(destination),
            'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
            'failed_measured_gates': validation['failed_measured_gates'],
            'unmeasured_gates': validation['unmeasured_gates'],
            'release_eligible': validation['release_eligible']}
        report['status'] = 'completed_not_activated'
        save('completed', qualification=report['qualification'])
    except BaseException as error:
        report.update(status='failed_not_activated', error=f'{type(error).__name__}: {error}')
        save('failed', error=report['error'])
        raise


if __name__ == '__main__':
    main()
