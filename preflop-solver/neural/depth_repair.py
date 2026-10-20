#!/usr/bin/env python3
"""Offline, hash-pinned depth-repair probes. No serving activation or gate waiver."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from run_depth_pilots import PILOT_GRID, process_family, stop_family
from run_depth_full import finite_metrics
from train import (ActionScorer, INPUT_FEATURE_COUNT, NETWORK_SCHEMA, ARRIVAL_NETWORK_SCHEMA, atomic_json,
                   expand_state_action, make_compiled_step, save_model, scorer_json)
from validate_seeds import (compare, evaluation_records, latest_artifact, load_artifact_model,
                            load_run, policy, verify_evaluation_game, model_has_arrival_memory, model_input_count, record_features)

ROOT = Path(__file__).resolve().parents[2]
PROFILES = {
    'control': [],
    'averaged': ['--average-regret-rollouts'],
    'recall': ['--average-regret-rollouts', '--card-arrival-memory'],
    'retained': ['--average-regret-rollouts', '--regret-training', 'retained_instantaneous'],
    'no-baseline': ['--average-regret-rollouts', '--variance-baseline-scale', '0'],
    'integrated': ['--average-regret-rollouts', '--integrate-closed-opponent-actions'],
}


def select_street_roots(records: list[dict], per_street: int) -> list[dict]:
    if per_street < 1:
        raise ValueError('root quota must be positive')
    selected = []
    selected_families = set()
    for street in ('preflop', 'flop', 'turn', 'river'):
        seen = set()
        for record in records:
            family = record['iteration']
            if record['state']['street'] == street and family not in seen and family not in selected_families:
                selected.append(record)
                seen.add(family)
                selected_families.add(family)
                if len(seen) == per_street:
                    break
    if not selected:
        raise ValueError('root sample is empty')
    return selected


def calibration_metrics(predicted: np.ndarray, predicted_se: np.ndarray,
                        reference: np.ndarray, reference_se: np.ndarray) -> dict:
    arrays = [np.asarray(a, dtype=np.float64).reshape(-1) for a in
              (predicted, predicted_se, reference, reference_se)]
    if not arrays[0].size or any(a.shape != arrays[0].shape or not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError('calibration arrays must be aligned, finite and nonempty')
    mean, se, target, target_se = arrays
    if np.any(se < 0) or np.any(target_se < 0):
        raise ValueError('negative standard errors are invalid')
    residual = np.abs(mean - target)
    combined = np.sqrt(se**2 + target_se**2)
    result = {
        'actions': len(mean),
        'rmse_against_reference_bb': float(np.sqrt(np.mean(residual**2))),
        'predicted_precision_coverage': float(np.mean(se <= .02)),
        'reference_precision_coverage': float(np.mean(target_se <= .02)),
        'empirical_error_at_most_0_02_fraction': float(np.mean(residual <= .02)),
        'nominal_95_interval_coverage': float(np.mean(residual <= 1.96 * combined)),
    }
    result['qualified'] = all(result[key] >= .95 for key in (
        'predicted_precision_coverage', 'reference_precision_coverage',
        'empirical_error_at_most_0_02_fraction', 'nominal_95_interval_coverage'))
    result['interpretation'] = 'Finite-sample diagnostics, not a conditional error guarantee or exploitability certificate.'
    return result


def independent_response_gain(blocks: np.ndarray, probabilities: np.ndarray) -> dict:
    blocks = np.asarray(blocks, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if blocks.ndim != 2 or len(blocks) < 4 or blocks.shape[1:] != probabilities.shape:
        raise ValueError('response probe needs >=4 independent hidden blocks and aligned probabilities')
    if not np.all(np.isfinite(blocks)) or not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1):
        raise ValueError('response probe values/probabilities are invalid')
    split = len(blocks) // 2
    selection, heldout = blocks[:split], blocks[split:]
    action = int(np.argmax(selection.mean(axis=0)))
    gains = heldout[:, action] - heldout @ probabilities
    return {
        'selected_action': action,
        'selection_gain_bb': float(selection[:, action].mean() - (selection @ probabilities).mean()),
        'heldout_gain_bb': float(gains.mean()),
        'heldout_standard_error_bb': float(gains.std(ddof=1) / np.sqrt(len(gains))),
        'interpretation': 'One information-set-consistent local deviation, selected on disjoint blocks; not full-game exploitability.'
    }


def frozen_reference(binary: Path, model, records: list[dict], depth: int, output: Path,
                     seed: int = 2471701, blocks: int = 32, inner: int = 2,
                     snapshots: Path | None = None, policy_only: bool = False,
                     exact_river_values: bool = False, exact_closed_values: bool = False) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    network_path = output.with_suffix('.networks.json')
    source = {'schema': ARRIVAL_NETWORK_SCHEMA if model_has_arrival_memory(model) else NETWORK_SCHEMA, 'input_size': model_input_count(model),
              'strategy_transform': 'softmax', 'networks': [scorer_json(model)] * 2,
              'qualification_game': {'depth_bb': depth, 'action_abstraction': PILOT_GRID}}
    atomic_json(network_path, source)
    input_path = output.with_suffix('.input.json')
    atomic_json(input_path, records)
    argv = [str(binary), 'neural-frozen-values', '--networks', str(network_path),
            '--input', str(input_path), '--output', str(output), '--effective-stack-bb', str(depth),
            '--seed', str(seed), '--hidden-blocks', str(blocks), '--rollouts-per-block', str(inner),
            '--action-abstraction-json', json.dumps(PILOT_GRID, sort_keys=True, separators=(',', ':'))]
    if snapshots is not None:
        argv += ['--snapshots', str(snapshots)]
    if policy_only:
        argv += ['--policy-only']
    if exact_river_values:
        argv += ['--exact-river-values']
    if exact_closed_values:
        argv += ['--exact-closed-values']
    try:
        subprocess.run(argv, cwd=ROOT, check=True, capture_output=True, text=True, timeout=600)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f'native qualification failed: {error.stderr.strip()}') from error
    report = json.loads(output.read_text())
    if (bool(report.get('exact_river_values')) != exact_river_values
            or bool(report.get('exact_closed_values')) != exact_closed_values):
        raise RuntimeError('native frozen reference changed the requested exact estimator')
    verify_evaluation_game(report, depth, PILOT_GRID)
    if report['input_sha256'] != hashlib.sha256(input_path.read_bytes()).hexdigest():
        raise RuntimeError('native frozen reference changed its input identity')
    if snapshots is None and report['source_network_sha256s'] != [hashlib.sha256(network_path.read_bytes()).hexdigest()]:
        raise RuntimeError('native frozen reference used different policy weights')
    return report


def audit_exported_values(model, value_model, records: list[dict], reference: dict, depth: int) -> dict:
    predicted, predicted_se, means, errors = [], [], [], []
    responses = []
    with mx.stream(mx.cpu):
        for record, row in zip(records, reference['records'], strict=True):
            features = record_features(value_model, record, depth)
            values = np.asarray(value_model(mx.array(features)))
            if values.shape != (len(record['actions']), 2):
                raise RuntimeError('exported value head has a wrong shape')
            predicted.extend(values[:, 0])
            predicted_se.extend(.005 + np.logaddexp(0., values[:, 1].astype(np.float64)))
            means.extend(row['action_values_bb'])
            errors.extend(row['standard_errors_bb'])
            # Verify both qualification and cross-seed audits query the exact
            # exported policy (not the current regret-network strategy).
            if not np.allclose(policy(model, features), row['probabilities'], atol=2e-5, rtol=2e-5):
                raise RuntimeError('native/exported frozen policy probabilities differ')
            responses.append(independent_response_gain(np.array(row['independent_hidden_block_values_bb']), np.array(row['probabilities'])))
    return {'calibration': calibration_metrics(np.array(predicted), np.array(predicted_se), np.array(means), np.array(errors)),
            'local_deviation_probes': responses,
            'mean_heldout_local_gain_bb': float(np.mean([r['heldout_gain_bb'] for r in responses])),
            'mean_heldout_local_gain_standard_error_bb': float(np.sqrt(sum(r['heldout_standard_error_bb']**2 for r in responses)) / len(responses)),
            'sampling': 'street-stratified bounded pilot; not authentic reach-weighted release qualification'}


def fit_frozen_value_head(records: list[dict], reference: dict, depth: int,
                          output: Path, seed: int = 2471717, steps: int = 200) -> dict:
    """Separate immutable-policy fit with trajectory-family train/cal/test splits.

    Prediction interval widths are empirical held-out residual envelopes. They
    are NEVER exported as sampling SE or used to activate a serving artifact.
    """
    groups = {}
    for record, row in zip(records, reference['records'], strict=True):
        groups.setdefault(record['iteration'], []).append((record, row))
    families = sorted(groups)
    if len(families) < 6:
        raise ValueError('frozen value fit requires >=6 disjoint trajectory families')
    rng = np.random.default_rng(seed)
    rng.shuffle(families)
    cuts = (max(2, len(families) // 2), max(4, len(families) * 3 // 4))
    splits = [families[:cuts[0]], families[cuts[0]:cuts[1]], families[cuts[1]:]]
    datasets = []
    for split in splits:
        features, targets, errors = [], [], []
        for family in split:
            for record, row in groups[family]:
                features.extend(expand_state_action(record['state'], a, depth) for a in record['actions'])
                targets.extend(row['action_values_bb'])
                errors.extend(row['standard_errors_bb'])
        datasets.append((np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32), np.asarray(errors)))
    mx.random.seed(seed)
    model = ActionScorer(INPUT_FEATURE_COUNT, (128, 64))
    optimizer = optim.AdamW(learning_rate=.001)
    optimizer.init(model.trainable_parameters())
    step = make_compiled_step(model, optimizer)
    train_x, train_y, _ = datasets[0]
    for _ in range(steps):
        indices = rng.integers(0, len(train_x), size=128)
        loss = step(mx.array(train_x[indices]), mx.array(train_y[indices, None] / depth), mx.ones((128, 1)))
        mx.eval(loss, model.parameters(), optimizer.state)
        if not np.isfinite(float(loss)):
            raise RuntimeError('frozen EV refit became nonfinite')
    with mx.stream(mx.cpu):
        predictions = [np.asarray(model(mx.array(data[0]))).reshape(-1) * depth for data in datasets]
    calibration_residuals = np.abs(predictions[1] - datasets[1][1]) + 1.96 * datasets[1][2]
    width = float(np.quantile(calibration_residuals, .95, method='higher'))
    residuals = np.abs(predictions[2] - datasets[2][1])
    output.mkdir(parents=True, exist_ok=True)
    weights = output / 'frozen-value.safetensors'
    save_model(model, weights)
    report = {'schema': 'hu-frozen-ev-refit-v1', 'status': 'research_not_activated',
        'source_policy_sha256s': reference['source_network_sha256s'],
        'reference_input_sha256': reference['input_sha256'],
        'family_splits': splits, 'seed': seed, 'steps': steps,
        'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
        'empirical_prediction_interval_95_half_width_bb': width,
        'heldout_interval_coverage': float(np.mean(residuals <= width)),
        'heldout_rmse_bb': float(np.sqrt(np.mean(residuals**2))),
        'heldout_reference_precision_coverage': float(np.mean(datasets[2][2] <= .02)),
        'interpretation': 'Pilot residual envelope includes reference noise; not sampling SE, a conditional guarantee, or qualification.'}
    atomic_json(output / 'report.json', report)
    return report


def training_command(binary: Path, output: Path, profile: str, seed: int, rounds: int,
                     traversals: int, steps: int, *, depth: int = 50, max_minutes: float = 10) -> list[str]:
    if depth not in (50, 100) or not 2 <= rounds <= 128 or traversals < 2 or traversals % 2 or steps < 1 or not 0 < max_minutes <= 60:
        raise ValueError('invalid bounded matched training budget')
    return [sys.executable, str(Path(__file__).with_name('train.py')), '--run-dir', str(output),
        '--depth-bb', str(depth), '--seed', str(seed), '--target-round', str(rounds), '--max-minutes', str(max_minutes),
        '--traversals-per-round', str(traversals), '--steps-per-round', str(steps),
        '--reservoir-capacity', '40000', '--batch-size', '512', '--hidden-sizes', '256,128',
        '--artifact-every', '1', '--sample-turn-rivers', '--preflop-runout-samples', '64',
        '--flop-runout-samples', '32', '--value-rollouts-per-action', '4',
        '--sampler-binary', str(binary), '--sampler-sha256', hashlib.sha256(binary.read_bytes()).hexdigest(),
        '--action-abstraction-json', json.dumps(PILOT_GRID, sort_keys=True, separators=(',', ':')),
        '--replay-street-proposal', '.25,.25,.25,.25', '--retain-policy-snapshots'] + PROFILES[profile]


def run_training(binary: Path, output: Path, profile: str, seed: int, rounds: int, traversals: int,
                 steps: int, *, depth: int = 50, max_minutes: float = 10, resume: bool = False) -> dict:
    argv = training_command(binary, output, profile, seed, rounds, traversals, steps,
                            depth=depth, max_minutes=max_minutes)
    if resume and not (output / 'state.json').is_file():
        raise ValueError('explicit recovery requires an existing atomic run state')
    output.mkdir(parents=True, exist_ok=resume)
    previous = json.loads((output / 'state.json').read_text()) if resume else None
    starting_round = previous['completed_rounds'] if previous else 0
    if previous:
        expected = {'depth_bb': depth, 'seed': seed, 'traversals_per_round': traversals,
                    'steps_per_round': steps, 'sampler_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
                    'hidden_sizes': [256, 128], 'reservoir_capacity': 40000, 'batch_size': 512,
                    'retain_policy_snapshots': True,
                    'action_abstraction_json': json.dumps(PILOT_GRID, sort_keys=True, separators=(',', ':'))}
        if (starting_round > rounds or any(previous['config'].get(key) != value for key, value in expected.items())
                or previous['config'].get('average_regret_rollouts', False) != (profile != 'control')
                or previous['config'].get('integrate_closed_opponent_actions', False) != (profile == 'integrated')
                or previous['config'].get('card_arrival_memory', False) != (profile == 'recall')
                or previous['config'].get('regret_training', 'bootstrapped_dcfr') != ('retained_instantaneous' if profile == 'retained' else 'bootstrapped_dcfr')
                or previous['config'].get('variance_baseline_scale') != (0. if profile == 'no-baseline' else .5)):
            raise ValueError('recovery run does not match the requested frozen configuration')
    started = time.monotonic()
    if starting_round < rounds:
        with (output / 'train.log').open('a' if resume else 'w') as log:
            process = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while process.poll() is None:
                    if time.monotonic() - started > max_minutes * 60 + 120 or sum(process_family(process.pid).values()) > 3.5 * 1024**3 or shutil.disk_usage(output).free < 20 * 1024**3:
                        raise RuntimeError('repair pilot exceeded its time, memory or disk guard')
                    time.sleep(2)
                if process.returncode:
                    detail = (output / 'train.log').read_text(errors='replace')[-2000:]
                    raise RuntimeError(f'repair training failed: {output}/train.log\n{detail}')
            finally:
                if process.poll() is None:
                    stop_family(process)
    state, _ = load_run(output)
    if state['completed_rounds'] != rounds or state['completed_traversals'] != rounds * traversals:
        raise RuntimeError('pilot did not finish its matched iteration budget')
    if any(metric.get('records_truncated') or not finite_metrics(metric) for metric in state['metrics']):
        raise RuntimeError('repair pilot contains truncated or nonfinite training data')
    return {'profile': profile, 'seed': seed, 'depth_bb': depth, 'seconds': time.monotonic() - started,
            'output': str(output), 'rounds': rounds, 'traversals': state['completed_traversals'],
            'starting_round': starting_round,
            'total_update_seconds': sum(metric['elapsed_seconds'] for metric in state['metrics'])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=12)
    parser.add_argument('--traversals', type=int, default=64)
    parser.add_argument('--steps', type=int, default=80)
    parser.add_argument('--existing-full-run', type=Path)
    args = parser.parse_args()
    if not 2 <= args.rounds <= 128 or args.traversals < 2 or args.traversals % 2 or args.steps < 1:
        raise ValueError('paired pilots require 2..128 rounds, an even traversal count and positive optimizer steps')
    args.output_root.mkdir(parents=True, exist_ok=False)
    binary = args.output_root / 'sampler'
    shutil.copy2(args.binary, binary)
    source_files = [Path(__file__).with_name(name) for name in
                    ('depth_repair.py', 'train.py', 'validate_seeds.py', 'run_depth_pilots.py', 'run_depth_full.py')]
    sources = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    report = {'schema': 'hu-depth-repair-pilot-v1', 'status': 'running_not_activated',
        'sampler_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(), 'source_sha256s': sources,
        'profiles': PROFILES, 'seeds': [51701, 51702], 'rounds': args.rounds,
        'traversals_per_round': args.traversals, 'optimizer_steps_per_round': args.steps,
        'existing_frozen_audits': [], 'training': [], 'comparisons': {}}
    atomic_json(args.output_root / 'summary.json', report)
    # First qualify existing frozen exports, without retraining their policy.
    if args.existing_full_run:
        summary = json.loads((args.existing_full_run / 'summary.json').read_text())
        for run in sorted(summary['results'], key=lambda r: (r['depthBb'], r['seed'])):
            started = time.monotonic()
            source = Path(run['artifacts'][-1]['source'])
            model = load_artifact_model(source)
            depth = run['depthBb']
            records = evaluation_records(ROOT, depth, 64, 2471703, 64, 32, True, False,
                policy_model=model, action_abstraction=PILOT_GRID, binary_path=binary)
            roots = select_street_roots(records, 2)
            prefix = args.output_root / f"frozen-{depth}-{run['seed']}"
            reference = frozen_reference(binary, model, roots, depth, prefix.with_suffix('.json'))
            audit = audit_exported_values(model, load_artifact_model(source, 'baselineActionValue'), roots, reference, depth)
            audit.update(depth_bb=depth, seed=run['seed'])
            audit['refit'] = fit_frozen_value_head(roots, reference, depth, prefix)
            audit['seconds'] = time.monotonic() - started
            report['existing_frozen_audits'].append(audit)
            atomic_json(args.output_root / 'summary.json', report)
            print(json.dumps({'stage':'frozen_audit','depth_bb':depth,'seed':run['seed'],'seconds':audit['seconds']}),flush=True)
    tasks = [(profile, seed) for profile in PROFILES for seed in report['seeds']]
    def train_task(task):
        if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha for path, sha in sources.items()):
            raise RuntimeError('repair source changed after experiment pinning')
        profile, seed = task
        return run_training(binary, args.output_root / f'{profile}-{seed}', profile, seed, args.rounds, args.traversals, args.steps)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(train_task, tasks):
            report['training'].append(result)
            atomic_json(args.output_root / 'summary.json', report)
            print(json.dumps({'stage':'training',**result}),flush=True)
    # Common authentic exact-card evaluation distribution for fair A/B deltas.
    models = {}; states = {}
    for profile in PROFILES:
        states[profile] = [load_run(args.output_root / f'{profile}-{seed}')[0] for seed in report['seeds']]
        models[profile] = [load_artifact_model(Path(latest_artifact(state)['source'])) for state in states[profile]]
    reach = [evaluation_records(ROOT, 50, 256, 2471704, 64, 32, True, False,
        policy_model=model, action_abstraction=PILOT_GRID, binary_path=binary) for model in models['control']]
    forced = evaluation_records(ROOT, 50, 128, 2471705, 64, 32, True, False,
        action_abstraction=PILOT_GRID, binary_path=binary)
    roots = select_street_roots(reach[0], 2)
    for profile in PROFILES:
        comparison = {'authentic': compare(*models[profile], reach, 50, 'fixed common control-policy reach', True),
                      'forced': compare(*models[profile], [forced], 50, 'fixed forced deviations', False),
                      'seeds': []}
        for index, model in enumerate(models[profile]):
            seed = report['seeds'][index]
            run = args.output_root / f'{profile}-{seed}'
            reference = frozen_reference(binary, model, roots, 50, run / 'frozen-values.json')
            value_model = load_artifact_model(Path(latest_artifact(states[profile][index])['source']), 'baselineActionValue')
            audit = audit_exported_values(model, value_model, roots, reference, 50)
            snapshot = frozen_reference(binary, model, select_street_roots(reach[0], 8), 50,
                run / 'snapshot-reference.json', snapshots=run / 'policy-history/manifest.json', policy_only=True)
            student_rows = [record for record in select_street_roots(reach[0], 8)]
            snapshot_mae = []
            with mx.stream(mx.cpu):
                for record, row in zip(student_rows, snapshot['records'], strict=True):
                    features = np.stack([expand_state_action(record['state'], a, 50) for a in record['actions']])
                    snapshot_mae.append(float(np.abs(policy(model, features) - np.asarray(row['probabilities'])).mean()))
            audit['student_vs_own_reach_snapshot_mae'] = float(np.mean(snapshot_mae))
            comparison['seeds'].append(audit)
        report['comparisons'][profile] = comparison
        atomic_json(args.output_root / 'summary.json', report)
        print(json.dumps({'stage':'comparison','profile':profile,'mae':comparison['authentic']['action_frequency_mae']}),flush=True)
    control = report['comparisons']['control']
    promising = []
    for profile, result in report['comparisons'].items():
        if profile == 'control': continue
        better_stability = result['authentic']['action_frequency_mae'] < control['authentic']['action_frequency_mae'] and result['authentic']['primary_action_agreement'] > control['authentic']['primary_action_agreement']
        # A tiny noisy local-response pilot cannot authorize an extended run.
        result['stability_directionally_better'] = better_stability
        if better_stability: promising.append(profile)
    report.update(status='completed_not_activated', promising_stability_profiles=promising,
        long_run_authorized=False, exploitability_measured=False,
        next_decision='Review independent response precision before extending; no automatic long run or model promotion.')
    atomic_json(args.output_root / 'summary.json', report)
    print(json.dumps({'status': report['status'], 'promising_stability_profiles': promising, 'output': str(args.output_root)}))


if __name__ == '__main__':
    main()
