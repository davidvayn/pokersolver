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
from train import RunConfig, atomic_json, run_config_payload
from validate_seeds import compare, evaluation_records, latest_artifact, load_artifact_model, load_run, verify_artifact
import numpy as np


def reusable_control(summary_path, control, seed, depth, binary_digest):
    origin = json.loads(summary_path.read_text())
    if (origin.get('status') != 'completed_not_activated' or origin.get('depth_bb') != depth
            or origin.get('sampler_sha256') != binary_digest or control != 'recall-cpu'):
        raise ValueError('reused control requires a completed, same-game/binary CPU-prior pilot')
    run = summary_path.parent/f'{control}-{seed}'
    state = json.loads((run/'state.json').read_text())
    config = state['config']
    if (state['completed_rounds'] != 24 or state['completed_traversals'] != 24*128
            or config['depth_bb'] != depth or config['seed'] != seed or config['steps_per_round'] != 240
            or not config.get('card_arrival_memory') or not config.get('bootstrap_cpu_prior')
            or config.get('matmul_precision') is not None):
        raise ValueError('reused control changed its training recipe')
    canonical = json.dumps(run_config_payload(RunConfig(**config)),sort_keys=True,separators=(',',':'))
    if hashlib.sha256(canonical.encode()).hexdigest() != state['config_hash']:
        raise ValueError('reused control configuration identity changed')
    actual = verify_artifact(latest_artifact(state))
    if actual != origin['artifacts'][f'{control}-{seed}'] or not actual['descriptor_matches'] or not actual['magic_valid']:
        raise ValueError('reused control weights differ from their frozen origin')
    return run, {'profile':control,'seed':seed,'depth_bb':depth,'output':str(run),
        'rounds':24,'traversals':24*128,'reused_frozen_control':True,
        'origin_summary_sha256':hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        'origin_training_source_sha256s':origin['source_sha256s'],'artifact':actual}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', choices=('retained', 'recall', 'recall-data', 'recall-fit', 'recall-cpu', 'recall-fp32'), default='retained')
    parser.add_argument('--depth', type=int, choices=(50,100), default=50)
    parser.add_argument('--prerequisite-50bb', type=Path)
    parser.add_argument('--reuse-control-summary',type=Path,
                        help='full-FP32 treatment only: reuse hash-verified CPU-prior controls, never resume or modify them')
    args = parser.parse_args()
    control = 'recall-cpu' if args.candidate == 'recall-fp32' else 'recall' if args.candidate in ('recall-data','recall-fit','recall-cpu') else 'averaged'
    if args.reuse_control_summary is not None and args.candidate != 'recall-fp32':
        raise ValueError('frozen control reuse is only supported for the full-FP32 treatment')
    budgets = {p: {'traversals': 512 if p == 'recall-data' else 128, 'steps': 240,
                   'average_policy_steps':480 if p == 'recall-fit' else 240} for p in (control,args.candidate)}
    args.output.mkdir(parents=True, exist_ok=False)
    binary = args.output / 'sampler'
    shutil.copy2(args.binary, binary)
    names = ('train.py', 'depth_repair.py', 'run_regret_learning_pilot.py', 'validate_seeds.py', 'mlx_precision.py')
    sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    if args.depth == 100:
        if args.prerequisite_50bb is None:
            raise ValueError('100bb requires a positive 50bb screen')
        prior = json.loads(args.prerequisite_50bb.read_text())
        if (prior.get('status') != 'completed_not_activated' or prior.get('depth_bb') != 50
                or prior.get('profiles') != [control, args.candidate]
                or not prior.get('decision',{}).get('improvements_hold')
                or prior.get('source_sha256s') != sources or prior.get('sampler_sha256') != digest):
            raise ValueError('100bb requires the same implementation and a positive matched 50bb comparison')
    report = {'schema': 'hu-upstream-regret-learning-pilot-v1', 'status': 'running_not_activated',
        'depth_bb': args.depth, 'seeds': [51701, 51702], 'profiles': [control, args.candidate],
        'rounds': 24, 'traversals_per_round': 128, 'steps_per_round': 240,
        'training_budgets':budgets,
        'source_sha256s': sources, 'sampler_sha256': digest,
        'criteria': 'Both batches MAE lower and primary agreement higher; both seeds lower local response gain; paired-root 95% upper delta negative.',
        'contrast': ('Only card-arrival inputs differ; zero-extended shared initialization, identical game and training budgets.'
            if args.candidate == 'recall' else 'Only fresh traversals per round change, 128 to 512; both arms keep v5 memory and all fitting budgets.'
            if args.candidate == 'recall-data' else 'Only average-policy fitting changes, 240 to 480; both regret/value budgets and v5 traversal data are unchanged.'
            if args.candidate == 'recall-fit' else 'Only bootstrap prior inference changes to CPU FP32; traversal, fitting budgets and v5 encoding are unchanged.'
            if args.candidate == 'recall-cpu' else 'Only GPU fitting precision changes to full FP32; both arms use CPU bootstrap priors and identical v5 game/training budgets.'
            if args.candidate == 'recall-fp32' else 'Bootstrapped DCFR+ versus retained-instantaneous linear CFR with fresh per-round regret refits. Not a one-parameter discounting ablation.'),
        'active_models_changed': False, 'release_eligible': False, 'long_run_authorized': False,
        'full_game_exploitability_measured': False, 'training': [], 'batches': [], 'responses': {}, 'validation': {}}
    reused = {}
    if args.reuse_control_summary is not None:
        for seed in report['seeds']:
            reused[seed] = reusable_control(args.reuse_control_summary,control,seed,args.depth,digest)
    def run_path(profile,seed):
        return reused[seed][0] if profile == control and seed in reused else args.output/f'{profile}-{seed}'
    def save(stage, **values):
        atomic_json(args.output / 'summary.json', report)
        print(json.dumps({'stage': stage, **values}), flush=True)
    def task(spec):
        if any(hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != sha for name, sha in sources.items()):
            raise RuntimeError('frozen learning sources changed')
        profile, seed = spec
        if profile == control and seed in reused:
            return reused[seed][1]
        return run_training(binary, args.output / f'{profile}-{seed}', profile, seed, 24, budgets[profile]['traversals'], 240,
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
            state, _ = load_run(run_path(profile,seed))
            artifact = latest_artifact(state)
            integrity = verify_artifact(artifact)
            if not integrity['descriptor_matches'] or not integrity['magic_valid']:
                raise RuntimeError('artifact integrity failed')
            key = f'{profile}-{seed}'
            report['artifacts'][key] = integrity
            models[key] = load_artifact_model(Path(artifact['source']))
        if args.candidate in ('recall','recall-data','recall-fit','recall-cpu','recall-fp32'):
            configs = []
            for profile, seed in tasks:
                config = dict(json.loads((run_path(profile,seed) / 'state.json').read_text())['config'])
                if args.candidate != 'recall' and not config.get('card_arrival_memory'):
                    raise RuntimeError('learning repair lost card-arrival memory')
                config.pop('seed'); config.pop('card_arrival_memory',None)
                if args.candidate == 'recall-data':
                    if config['traversals_per_round'] != budgets[profile]['traversals']:
                        raise RuntimeError('fresh-data pilot changed its declared traversal budget')
                    config.pop('traversals_per_round')
                if args.candidate == 'recall-fit':
                    if config.get('average_policy_steps_per_round',240) != budgets[profile]['average_policy_steps']:
                        raise RuntimeError('average-policy pilot changed its declared fit budget')
                    config.pop('average_policy_steps_per_round',None)
                if args.candidate == 'recall-cpu':
                    if config.get('bootstrap_cpu_prior',False) != (profile == 'recall-cpu'):
                        raise RuntimeError('bootstrap pilot changed its declared inference path')
                    config.pop('bootstrap_cpu_prior',None)
                if args.candidate == 'recall-fp32':
                    if config.get('matmul_precision') != ('full-fp32-v1' if profile == 'recall-fp32' else None):
                        raise RuntimeError('precision pilot changed its declared math mode')
                    config.pop('matmul_precision',None)
                configs.append(config)
            if any(config != configs[0] for config in configs):
                raise RuntimeError('matched learning pilot changed more than its declared factor and seed')
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
        response = paired_response_summary(gains[control], gains[args.candidate])
        response['interpretation'] = f'{args.candidate} minus {control} local response gain; negative favors candidate. Not full-game exploitability.'
        report['paired_response'] = response
        report['decision'] = promotion_decision([
            {'control': b[control], 'averaged': b[args.candidate]} for b in report['batches']], response)
        for profile in report['profiles']:
            destination = args.output / f'{profile}-release-validation.json'
            subprocess.run([sys.executable,str(Path(__file__).with_name('validate_seeds.py')),
                str(run_path(profile,51701)), str(run_path(profile,51702)),
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
