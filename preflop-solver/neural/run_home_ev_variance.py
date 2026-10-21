#!/usr/bin/env python3
"""Matched frozen-v5 open-action variance pilot. Never trains or activates policy.

Compare the same posterior/hidden blocks with and without a conditional action
expectation. Exact branches are identical constants in both arms. Judge variance
AND wall time; a smaller SE at greater cost is not automatically an improvement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from depth_repair import frozen_reference
from frozen_policy_values import verify_frozen_targets
from train import atomic_json
from validate_seeds import latest_artifact, load_artifact_model


def comparison(control, candidate, control_seconds, candidate_seconds):
    before, after, precise_before, precise_after, differences = [], [], [], [], []
    for a,b in zip(control['records'],candidate['records'],strict=True):
        if a['state'] != b['state'] or a['actions'] != b['actions'] or a['probabilities'] != b['probabilities']:
            raise ValueError('variance pilot changed the frozen decision/policy')
        precise_before.append(max(a['standard_errors_bb']) <= .02)
        precise_after.append(max(b['standard_errors_bb']) <= .02)
        if not a['independent_hidden_block_values_bb']:
            if a['action_values_bb'] != b['action_values_bb'] or a['standard_errors_bb'] != b['standard_errors_bb']:
                raise ValueError('exact reference changed')
            continue
        x,y = np.array(a['independent_hidden_block_values_bb']),np.array(b['independent_hidden_block_values_bb'])
        if x.shape != y.shape:
            raise ValueError('hidden block pairing changed')
        for i,method in enumerate(a['action_estimators']):
            if method != 'independent_hidden_blocks_v1':
                if not np.array_equal(x[:,i],y[:,i]):
                    raise ValueError('exact action changed')
                continue
            before.append(float(x[:,i].var(ddof=1)))
            after.append(float(y[:,i].var(ddof=1)))
            delta = y[:,i]-x[:,i]
            se = float(delta.std(ddof=1)/np.sqrt(len(delta)))
            differences.append({'mean_delta_bb':float(delta.mean()),'paired_delta_se_bb':se})
    if not before or np.mean(before) <= 0:
        raise ValueError('no noisy open actions to compare')
    variance_ratio = float(np.mean(after)/np.mean(before))
    time_ratio = candidate_seconds/control_seconds
    return {'sampled_actions':len(before),'mean_sampled_action_variance_before_bb2':float(np.mean(before)),
            'mean_sampled_action_variance_after_bb2':float(np.mean(after)), 'variance_ratio':variance_ratio,
            'wall_time_ratio':time_ratio, 'variance_times_wall_time_ratio':variance_ratio*time_ratio,
            'all_action_reference_precision_before':float(np.mean(precise_before)),
            'all_action_reference_precision_after':float(np.mean(precise_after)),
            'paired_mean_differences':differences,
            'precision_gate_passed_in_stratified_probe':bool(np.mean(precise_after)>=.95),
            'cost_adjusted_improvement':bool(variance_ratio<.95 and variance_ratio*time_ratio<1),
            'interpretation':'Small stratified estimator diagnostic, not authentic-reach confidence or neural accuracy.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--pilot',type=Path,required=True,help='completed v5 run pair directory')
    parser.add_argument('--roots',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--blocks',type=int,default=128)
    args = parser.parse_args()
    if not 32 <= args.blocks <= 512:
        raise ValueError('bounded pilot requires 32..512 hidden blocks')
    args.output.mkdir(parents=True,exist_ok=False)
    binary = args.output/'sampler'
    shutil.copy2(args.binary,binary)
    roots = json.loads(args.roots.read_text())[:16]
    if len(roots)!=16 or len({r['iteration'] for r in roots})!=16:
        raise ValueError('requires sixteen disjoint frozen hand-family roots')
    report = {'schema':'hu-home-open-action-variance-pilot-v1','release_eligible':False,
              'status':'running_not_activated','binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
              'criteria':'Both seeds reduce sampled-action variance >=5% AND variance-times-wall-time; exact actions unchanged.',
              'results':[],'active_models_changed':False}
    for seed in (51701,51702):
        run = args.pilot/f'recall-{seed}'
        state = json.loads((run/'state.json').read_text())
        source = Path(latest_artifact(state)['source'])
        model = load_artifact_model(source)
        refs, times, paths = [], [], []
        for depth in (0,1):
            path = args.output/f'{seed}-prefix{depth}.json'
            start = time.monotonic()
            ref = frozen_reference(binary,model,roots,50,path,seed=2919053,blocks=args.blocks,inner=1,
                exact_river_values=True,exact_closed_values=True,action_expectation_depth=depth)
            times.append(time.monotonic()-start)
            verify_frozen_targets(source,path)
            refs.append(ref)
            paths.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            print(json.dumps({'stage':'frozen_labels_finished','seed':seed,'prefix':depth,'seconds':times[-1]}),flush=True)
        result = {'seed':seed,'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                  'references':paths,'seconds':times,'comparison':comparison(*refs,*times)}
        report['results'].append(result)
        atomic_json(args.output/'summary.json',report)
        print(json.dumps(result['comparison']),flush=True)
    report['status']='completed_not_activated'
    report['improvements_hold']=all(row['comparison']['cost_adjusted_improvement'] for row in report['results'])
    atomic_json(args.output/'summary.json',report)


if __name__=='__main__':
    main()
