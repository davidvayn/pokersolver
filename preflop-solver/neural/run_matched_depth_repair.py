#!/usr/bin/env python3
"""Longer paired estimator comparison, with a conditional 100bb follow-up.

This screen never activates exports or substitutes local gains for exploitability.
All four runs have the same compute budget; only averaged regret targets differ.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from depth_repair import (ROOT, audit_exported_values, frozen_reference, run_training,
                          select_street_roots)
from run_depth_pilots import PILOT_GRID
from train import atomic_json
from validate_seeds import (compare, evaluation_records, latest_artifact,
                            load_artifact_model, load_run, verify_artifact)

PROFILES = ('control', 'averaged')
SEEDS = (51701, 51702)
EVALUATION_SEEDS = (2717101, 2717102)


def paired_response_summary(control: np.ndarray, averaged: np.ndarray) -> dict:
    """Bootstrap whole roots, retaining correlation between the two seed pairs.

    This is an exploratory, conditional comparison across sampled decision roots,
    not a finite-sample equilibrium bound or a claim across all training seeds.
    """
    control, averaged = np.asarray(control, dtype=float), np.asarray(averaged, dtype=float)
    if (control.shape != averaged.shape or control.ndim != 2 or control.shape[0] != 2
            or control.shape[1] < 8 or not np.all(np.isfinite(control))
            or not np.all(np.isfinite(averaged))):
        raise ValueError('response comparison requires two aligned seeds and >=8 finite independent roots')
    delta = averaged - control
    root_delta = delta.mean(axis=0)
    rng = np.random.default_rng(2717111)
    resampled = root_delta[rng.integers(0, len(root_delta), size=(10000, len(root_delta)))].mean(axis=1)
    return {'control_mean_gain_bb': float(control.mean()),
            'averaged_mean_gain_bb': float(averaged.mean()),
            'mean_delta_bb': float(delta.mean()),
            'seed_mean_deltas_bb': delta.mean(axis=1).tolist(),
            'one_sided_95_bootstrap_upper_delta_bb': float(np.quantile(resampled, .95)),
            'independent_roots': control.shape[1], 'bootstrap_replicates': len(resampled),
            'interpretation': 'Per sampled decision, conditional on two training seeds; negative favors averaging. Not bb/hand exploitability.'}


def promotion_decision(batches: list[dict], response: dict) -> dict:
    stable = len(batches) == 2 and all(
        batch['averaged']['action_frequency_mae'] < batch['control']['action_frequency_mae']
        and batch['averaged']['primary_action_agreement'] > batch['control']['primary_action_agreement']
        for batch in batches)
    credible = (len(response['seed_mean_deltas_bb']) == 2
                and all(delta < 0 for delta in response['seed_mean_deltas_bb'])
                and response['one_sided_95_bootstrap_upper_delta_bb'] < 0)
    return {'stability_better_on_both_evaluation_batches': stable,
            'local_response_reduction_credible_in_this_screen': credible,
            'improvements_hold': stable and credible,
            'release_eligible': False,
            'interpretation': 'Screening criterion only, not release validation or proof of equilibrium strength.'}


def add_family_namespace(records: list[dict], namespace: int) -> list[dict]:
    # Each evaluation CLI starts iteration numbering at zero. Preserve whole-hand
    # grouping while making families from separate sampled corpora disjoint.
    if any(not 0 <= row['iteration'] < 1_000_000 for row in records):
        raise ValueError('evaluation family exceeds its namespace')
    return [{**row, 'iteration': namespace * 1_000_000 + row['iteration']} for row in records]


def recovery_identity(previous: dict, *, depth: int, rounds: int, traversals: int,
                      steps: int, digest: str, sources: dict) -> dict:
    expected = {'schema': 'hu-longer-matched-depth-repair-v1', 'status': 'failed_not_activated',
                'depth_bb': depth, 'rounds': rounds, 'traversals_per_round': traversals,
                'optimizer_steps_per_round': steps, 'sampler_sha256': digest}
    if any(previous.get(key) != value for key, value in expected.items()):
        raise ValueError('recovery requires a stopped attempt with the identical iteration/game budget')
    old_sources = previous['source_sha256s']
    drivers = {'depth_repair.py', 'run_matched_depth_repair.py'}
    if set(old_sources) != set(sources) or any(old_sources[name] != sha for name, sha in sources.items() if name not in drivers):
        raise ValueError('recovery cannot change the frozen learning/evaluation implementation')
    count = previous.get('recovery_attempt_count', 0) + 1
    if count > 2:
        raise ValueError('recovery exceeded its two-attempt bound')
    return {'training_configuration_unchanged': True, 'recovery_attempt_count': count,
            'prior_source_sha256s': old_sources,
            'changed_driver_only': [name for name in sorted(drivers) if old_sources.get(name) != sources.get(name)],
            'reason': 'Clean checkpoint recovery after external CPU/GPU contention; not a new training configuration.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--depth', type=int, choices=(50, 100), default=50)
    parser.add_argument('--prerequisite-50bb', type=Path)
    parser.add_argument('--rounds', type=int, default=64)
    parser.add_argument('--traversals', type=int, default=256)
    parser.add_argument('--steps', type=int, default=160)
    parser.add_argument('--max-minutes-per-run', type=float, default=35)
    parser.add_argument('--evaluation-traversals', type=int, default=256)
    parser.add_argument('--response-blocks', type=int, default=512)
    parser.add_argument('--resume-existing', action='store_true',
                        help='recover a stopped attempt, preserving checkpoints, logs and prior provenance')
    args = parser.parse_args()
    if (not 2 <= args.rounds <= 128 or args.traversals < 2 or args.traversals % 2
            or args.steps < 1 or not 0 < args.max_minutes_per_run <= 60
            or not 64 <= args.evaluation_traversals <= 2048 or not 256 <= args.response_blocks <= 4096):
        raise ValueError('invalid bounded comparison budget')
    prerequisite = None
    if args.depth == 100:
        if args.prerequisite_50bb is None:
            raise ValueError('100bb requires the completed positive 50bb comparison')
        prerequisite = json.loads(args.prerequisite_50bb.read_text())
        if (prerequisite.get('schema') != 'hu-longer-matched-depth-repair-v1'
                or prerequisite.get('depth_bb') != 50
                or prerequisite.get('status') != 'completed_not_activated'
                or not prerequisite['decision']['improvements_hold']):
            raise ValueError('50bb did not establish the conditional follow-up criterion')
        if any(prerequisite.get(key) != value for key, value in
               (('rounds', args.rounds), ('traversals_per_round', args.traversals),
                ('optimizer_steps_per_round', args.steps))):
            raise ValueError('100bb replication must preserve the accepted 50bb iteration budget')
    previous = json.loads((args.output / 'summary.json').read_text()) if args.resume_existing else None
    args.output.mkdir(parents=True, exist_ok=args.resume_existing)
    binary = args.output / 'sampler'
    if not args.resume_existing:
        shutil.copy2(args.binary, binary)
    elif hashlib.sha256(args.binary.read_bytes()).digest() != hashlib.sha256(binary.read_bytes()).digest():
        raise ValueError('recovery cannot replace its frozen sampler')
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    names = ('run_matched_depth_repair.py', 'depth_repair.py', 'train.py', 'validate_seeds.py',
             'frozen_ev_gate.py', 'run_depth_pilots.py', 'run_depth_full.py', 'cash_profiles.py')
    sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}
    if prerequisite is not None and (prerequisite['sampler_sha256'] != digest or prerequisite['source_sha256s'] != sources):
        raise ValueError('conditional 100bb comparison requires the same frozen implementation')
    report = {'schema': 'hu-longer-matched-depth-repair-v1', 'status': 'running_not_activated',
        'depth_bb': args.depth, 'sampler_sha256': digest, 'source_sha256s': sources,
        'profiles': PROFILES, 'seeds': SEEDS, 'rounds': args.rounds,
        'traversals_per_round': args.traversals, 'optimizer_steps_per_round': args.steps,
        'evaluation_seeds': EVALUATION_SEEDS, 'response_blocks': args.response_blocks,
        'evaluation_traversals_per_policy_per_batch': args.evaluation_traversals,
        'max_minutes_per_invocation': args.max_minutes_per_run,
        'criteria': 'Both batches: lower MAE and higher primary agreement; both seeds: lower held-out local gain; paired-root bootstrap one-sided 95% upper delta <0.',
        'release_eligible': False, 'full_game_exploitability_measured': False,
        'training': [], 'batches': [], 'frozen_audits': {}, 'validation': {}}
    if args.prerequisite_50bb:
        report['prerequisite_sha256'] = hashlib.sha256(args.prerequisite_50bb.read_bytes()).hexdigest()
    summary_path = args.output / 'summary.json'
    if previous is not None:
        identity = recovery_identity(previous, depth=args.depth, rounds=args.rounds,
            traversals=args.traversals, steps=args.steps, digest=digest, sources=sources)
        if (previous['evaluation_seeds'] != list(EVALUATION_SEEDS) or previous['response_blocks'] != args.response_blocks
                or previous.get('evaluation_traversals_per_policy_per_batch', 256) != args.evaluation_traversals):
            raise ValueError('recovery cannot change the predeclared evaluation budget')
        old_sha = hashlib.sha256(summary_path.read_bytes()).hexdigest()
        snapshot = args.output / f'attempt-{old_sha[:12]}-summary.json'
        shutil.copy2(summary_path, snapshot)
        report.update(recovery={**identity, 'prior_summary_sha256': old_sha, 'prior_summary_path': str(snapshot)},
                      recovery_attempt_count=identity['recovery_attempt_count'])

    def save(stage: str, **values) -> None:
        atomic_json(summary_path, report)
        print(json.dumps({'stage': stage, 'depth_bb': args.depth, **values}), flush=True)

    save('predeclared')
    models, states, artifacts = {}, {}, {}
    tasks = [(profile, seed) for profile in PROFILES for seed in SEEDS]

    def task(spec):
        if any(hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != sha for name, sha in sources.items()):
            raise RuntimeError('matched comparison sources changed while queued')
        profile, seed = spec
        return run_training(binary, args.output / f'{profile}-{seed}', profile, seed,
                            args.rounds, args.traversals, args.steps, depth=args.depth,
                            max_minutes=args.max_minutes_per_run, resume=args.resume_existing)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for result in pool.map(task, tasks):
                report['training'].append(result)
                save('training_finished', **result)
        for profile, seed in tasks:
            key = f'{profile}-{seed}'
            state, _ = load_run(args.output / key)
            states[key] = state
            artifact = latest_artifact(state)
            check = verify_artifact(artifact)
            if not check['descriptor_matches'] or not check['magic_valid']:
                raise RuntimeError('matched artifact integrity failed')
            artifacts[key] = check
            models[key] = load_artifact_model(Path(artifact['source']))
        configs = []
        for state in states.values():
            config = dict(state['config'])
            config.pop('seed'); config.pop('average_regret_rollouts', None)
            configs.append(config)
        if any(config != configs[0] for config in configs):
            raise RuntimeError('matched runs changed more than seed and estimator')
        report['artifacts'] = artifacts
        response_roots = []
        for batch_index, evaluation_seed in enumerate(EVALUATION_SEEDS):
            reach = []
            for index, model in enumerate(models.values()):
                records = evaluation_records(ROOT, args.depth, args.evaluation_traversals, evaluation_seed,
                    64, 32, True, False, policy_model=model, action_abstraction=PILOT_GRID, binary_path=binary)
                reach.append(add_family_namespace(records, batch_index * 10 + index))
            atomic_json(args.output / f'reach-batch-{batch_index}.json', reach)
            batch = {}
            for profile in PROFILES:
                batch[profile] = compare(models[f'{profile}-{SEEDS[0]}'], models[f'{profile}-{SEEDS[1]}'],
                    reach, args.depth, 'fixed equal mixture of all four frozen policies', True)
            roots = select_street_roots([row for corpus in reach for row in corpus], 4)
            if len(roots) != 16:
                raise RuntimeError('held-out root corpus did not cover four independent families per street')
            response_roots.extend(roots)
            report['batches'].append(batch)
            save('stability_batch_finished', batch=batch_index,
                 mae={profile: batch[profile]['action_frequency_mae'] for profile in PROFILES})
        atomic_json(args.output / 'response-roots.json', response_roots)
        response_arrays = {profile: [] for profile in PROFILES}
        for profile, seed in tasks:
            key = f'{profile}-{seed}'
            reference = frozen_reference(binary, models[key], response_roots, args.depth,
                args.output / key / 'frozen-values.json', seed=2717109, blocks=args.response_blocks)
            value_model = load_artifact_model(Path(latest_artifact(states[key])['source']), 'baselineActionValue')
            audit = audit_exported_values(models[key], value_model, response_roots, reference, args.depth)
            report['frozen_audits'][key] = audit
            response_arrays[profile].append([root['heldout_gain_bb'] for root in audit['local_deviation_probes']])
            save('response_finished', profile=profile, seed=seed,
                 gain=audit['mean_heldout_local_gain_bb'])
        report['paired_response'] = paired_response_summary(response_arrays['control'], response_arrays['averaged'])
        report['decision'] = promotion_decision(report['batches'], report['paired_response'])
        save('screen_decision', **report['decision'])
        # Still run the ordinary independent validation for BOTH pairs. A failed
        # stability/precision check blocks expensive strength qualification; do
        # not spend an unbounded certificate budget or invent an EV receipt.
        for profile in PROFILES:
            destination = args.output / f'{profile}-release-validation.json'
            argv = [sys.executable, str(Path(__file__).with_name('validate_seeds.py')),
                str(args.output / f'{profile}-{SEEDS[0]}'), str(args.output / f'{profile}-{SEEDS[1]}'),
                '--round', str(args.rounds), '--traversals', str(args.evaluation_traversals * 2),
                '--seed', '2717117', '--evaluation-binary', str(binary),
                '--evaluation-binary-sha256', digest, '--output', str(destination)]
            subprocess.run(argv, cwd=ROOT, check=True, capture_output=True, text=True, timeout=600)
            validation = json.loads(destination.read_text())
            report['validation'][profile] = {'path': str(destination),
                'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                'release_eligible': validation['release_eligible'],
                'gates': validation['gates'], 'unmeasured_gates': validation['unmeasured_gates']}
            save('normal_validation_finished', profile=profile,
                 release_eligible=validation['release_eligible'])
        report['status'] = 'completed_not_activated'
        save('completed', decision=report['decision'])
    except BaseException as error:
        report.update(status='failed_not_activated', error=f'{type(error).__name__}: {error}')
        save('failed', error=report['error'])
        raise


if __name__ == '__main__':
    main()
