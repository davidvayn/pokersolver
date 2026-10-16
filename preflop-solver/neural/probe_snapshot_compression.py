#!/usr/bin/env python3
"""Isolate average-head fitting from regret learning; no strategy-strength claim."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from depth_repair import ROOT, frozen_reference
from distill_checkpoint_ensemble import train_student
from run_depth_pilots import PILOT_GRID
from train import atomic_json, expand_state_action, save_model
from validate_seeds import compare, evaluation_records, latest_artifact, load_artifact_model, load_run, policy


def family_splits(records, seed):
    families = sorted({record['iteration'] for record in records})
    if len(families) < 8:
        raise ValueError('compression probe requires >=8 independent trajectory families')
    np.random.default_rng(seed).shuffle(families)
    first, second = len(families) // 2, len(families) * 3 // 4
    return [set(families[:first]), set(families[first:second]), set(families[second:])]


def mean_error(model, dataset):
    with mx.stream(mx.cpu):
        return float(np.mean([np.abs(policy(model, features) - targets).mean() for features, targets in dataset]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    binary = args.experiment_root / 'sampler'
    seeds = [51701, 51702]
    controls = [load_run(args.experiment_root / f'control-{seed}')[1] for seed in seeds]
    reach = [evaluation_records(ROOT, 50, 256, 2471704, 64, 32, True, False,
        policy_model=model, action_abstraction=PILOT_GRID, binary_path=binary) for model in controls]
    records = reach[0] + reach[1]
    split = family_splits(records, 2471711)
    report = {'schema':'hu-snapshot-compression-probe-v1','status':'research_not_activated',
        'selection':'100/400 steps chosen on tuning families; test families never select weights',
        'family_splits':[sorted(families) for families in split], 'profiles':{}}
    for profile in ('control', 'averaged'):
        results, students = [], []
        for seed in seeds:
            run = args.experiment_root / f'{profile}-{seed}'
            state, _ = load_run(run)
            model = load_artifact_model(Path(latest_artifact(state)['source']))
            reference = frozen_reference(binary, model, records, 50,
                args.output / f'{profile}-{seed}-reference.json', snapshots=run / 'policy-history/manifest.json', policy_only=True)
            datasets = [[],[],[]]
            for record, row in zip(records, reference['records'], strict=True):
                features = np.stack([expand_state_action(record['state'], a, 50) for a in record['actions']])
                target = np.asarray(row['probabilities'],dtype=np.float32)
                for index, families in enumerate(split):
                    if record['iteration'] in families:
                        datasets[index].append((features, target))
            baseline = {'tuning_mae':mean_error(model,datasets[1]),'test_mae':mean_error(model,datasets[2])}
            trials = [(0,model,baseline['tuning_mae'])]
            for steps in (100,400):
                student = copy.deepcopy(model)
                train_student(student,datasets[0],steps,128,.0001,1e-5,seed)
                trials.append((steps,student,mean_error(student,datasets[1])))
            selected_steps, selected, tuning = min(trials,key=lambda trial:trial[2])
            test = mean_error(selected,datasets[2])
            weights = args.output / f'{profile}-{seed}-student.safetensors'
            save_model(selected,weights)
            results.append({'seed':seed,'baseline':baseline,'selected_steps':selected_steps,
                'selected_tuning_mae':tuning,'selected_test_mae':test,
                'trial_tuning_maes':[[steps,error] for steps,_,error in trials],
                'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
                'source_snapshot_sha256s':reference['source_network_sha256s']})
            students.append(selected)
            print(json.dumps({'profile':profile,**{k:v for k,v in results[-1].items() if 'sha256' not in k}}),flush=True)
        report['profiles'][profile]={'seeds':results,
            'student_cross_seed':compare(*students,reach,50,'fixed control-policy reach; compression probe only',True)}
        atomic_json(args.output / 'summary.json',report)
    report['interpretation']='A reduction isolates average-network fitting error; neither distillation nor cross-seed agreement proves lower exploitability.'
    atomic_json(args.output / 'summary.json',report)


if __name__=='__main__':main()
