#!/usr/bin/env python3
"""Short matched upstream-learning test after exact-average decomposition."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from depth_repair import (ROOT, frozen_reference, independent_response_gain,
                          run_training, select_street_roots)
from run_depth_pilots import PILOT_GRID
from run_matched_depth_repair import add_family_namespace, paired_response_summary, promotion_decision
from train import atomic_json
from validate_seeds import compare, evaluation_records, latest_artifact, load_artifact_model, load_run, verify_artifact
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', choices=('retained', 'recall'), default='retained')
    parser.add_argument('--depth', type=int, choices=(50,100), default=50)
    parser.add_argument('--prerequisite-50bb', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    binary = args.output / 'sampler'
    shutil.copy2(args.binary, binary)
    names = ('train.py', 'depth_repair.py', 'run_regret_learning_pilot.py', 'validate_seeds.py')
    sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    if args.depth == 100:
        if args.prerequisite_50bb is None:
            raise ValueError('100bb requires a positive 50bb screen')
        prior = json.loads(args.prerequisite_50bb.read_text())
        if (prior.get('status') != 'completed_not_activated' or prior.get('depth_bb') != 50
                or prior.get('profiles') != ['averaged', args.candidate]
                or not prior.get('decision',{}).get('improvements_hold')
                or prior.get('source_sha256s') != sources or prior.get('sampler_sha256') != digest):
            raise ValueError('100bb requires the same implementation and a positive matched 50bb comparison')
    report = {'schema': 'hu-upstream-regret-learning-pilot-v1', 'status': 'running_not_activated',
        'depth_bb': args.depth, 'seeds': [51701, 51702], 'profiles': ['averaged', args.candidate],
        'rounds': 24, 'traversals_per_round': 128, 'steps_per_round': 240,
        'source_sha256s': sources, 'sampler_sha256': digest,
        'criteria': 'Both batches MAE lower and primary agreement higher; both seeds lower local response gain; paired-root 95% upper delta negative.',
        'contrast': ('Only card-arrival inputs differ; zero-extended shared initialization, identical game and training budgets.'
            if args.candidate == 'recall' else 'Bootstrapped DCFR+ versus retained-instantaneous linear CFR with fresh per-round regret refits. Not a one-parameter discounting ablation.'),
        'active_models_changed': False, 'release_eligible': False, 'long_run_authorized': False,
        'full_game_exploitability_measured': False, 'training': [], 'batches': [], 'responses': {}, 'validation': {}}
    def save(stage, **values):
        atomic_json(args.output / 'summary.json', report)
        print(json.dumps({'stage': stage, **values}), flush=True)
    def task(spec):
        if any(hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != sha for name, sha in sources.items()):
            raise RuntimeError('frozen learning sources changed')
        profile, seed = spec
        return run_training(binary, args.output / f'{profile}-{seed}', profile, seed, 24, 128, 240,
                            depth=args.depth, max_minutes=20)
    save('predeclared')
    try:
        tasks = [(p, s) for p in report['profiles'] for s in report['seeds']]
        with ThreadPoolExecutor(max_workers=2) as pool:
            for result in pool.map(task, tasks):
                report['training'].append(result)
                save('training_finished', profile=result['profile'], seed=result['seed'])
        models = {}
        report['artifacts'] = {}
        for profile, seed in tasks:
            state, _ = load_run(args.output / f'{profile}-{seed}')
            artifact = latest_artifact(state)
            integrity = verify_artifact(artifact)
            if not integrity['descriptor_matches'] or not integrity['magic_valid']:
                raise RuntimeError('artifact integrity failed')
            key = f'{profile}-{seed}'
            report['artifacts'][key] = integrity
            models[key] = load_artifact_model(Path(artifact['source']))
        if args.candidate == 'recall':
            configs = []
            for profile, seed in tasks:
                config = dict(json.loads((args.output / f'{profile}-{seed}' / 'state.json').read_text())['config'])
                config.pop('seed'); config.pop('card_arrival_memory',None)
                configs.append(config)
            if any(config != configs[0] for config in configs):
                raise RuntimeError('arrival pilot changed more than card memory and seed')
        roots = []
        for batch, seed in enumerate((2818021, 2818023)):
            corpora = [add_family_namespace(evaluation_records(ROOT, args.depth, 256, seed, 64, 32, True, False,
                policy_model=m, action_abstraction=PILOT_GRID, binary_path=binary), batch * 10 + i)
                for i, m in enumerate(models.values())]
            atomic_json(args.output / f'reach-batch-{batch}.json', corpora)
            result = {p: compare(models[f'{p}-51701'], models[f'{p}-51702'], corpora, args.depth,
                'fixed equal-policy mixture; diagnostic only', True) for p in report['profiles']}
            report['batches'].append(result)
            selected = select_street_roots([r for rows in corpora for r in rows], 4)
            if len(selected) != 16:
                raise RuntimeError('response roots do not cover four families per street')
            roots.extend(selected)
            save('stability_finished', batch=batch, mae={p: result[p]['action_frequency_mae'] for p in result})
        atomic_json(args.output / 'response-roots.json', roots)
        gains = {p: [] for p in report['profiles']}
        for profile, seed in tasks:
            key = f'{profile}-{seed}'
            reference = frozen_reference(binary, models[key], roots, args.depth,
                args.output / key / 'response.json', seed=2818027, blocks=512)
            rows = [independent_response_gain(np.asarray(r['independent_hidden_block_values_bb']),
                np.asarray(r['probabilities'])) for r in reference['records']]
            report['responses'][key] = rows
            gains[profile].append([r['heldout_gain_bb'] for r in rows])
            save('response_finished', profile=profile, seed=seed,
                 gain=float(np.mean(gains[profile][-1])))
        response = paired_response_summary(gains['averaged'], gains[args.candidate])
        response['interpretation'] = f'{args.candidate} minus bootstrapped-v4 local response gain; negative favors candidate. Not full-game exploitability.'
        report['paired_response'] = response
        report['decision'] = promotion_decision([
            {'control': b['averaged'], 'averaged': b[args.candidate]} for b in report['batches']], response)
        for profile in report['profiles']:
            destination = args.output / f'{profile}-release-validation.json'
            subprocess.run([sys.executable,str(Path(__file__).with_name('validate_seeds.py')),
                str(args.output/f'{profile}-51701'), str(args.output/f'{profile}-51702'),
                '--round','24','--traversals','512','--seed','2919021',
                '--evaluation-binary',str(binary),'--evaluation-binary-sha256',digest,
                '--output',str(destination)],cwd=ROOT,check=True,capture_output=True,timeout=600)
            validation = json.loads(destination.read_text())
            report['validation'][profile] = {'path':str(destination),
                'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
                'release_eligible':validation['release_eligible'], 'gates':validation['gates'],
                'unmeasured_gates':validation['unmeasured_gates']}
        report['status'] = 'completed_not_activated'
        save('completed', decision=report['decision'])
    except BaseException as error:
        report.update(status='failed_not_activated', error=f'{type(error).__name__}: {error}')
        save('failed', error=report['error'])
        raise


if __name__ == '__main__':
    main()
