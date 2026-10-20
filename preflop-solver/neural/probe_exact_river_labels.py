#!/usr/bin/env python3
"""Paired frozen-policy exact-river label check. Other streets stay sampled."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from train import atomic_json


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_companions(output: Path, inputs: Path, networks: Path):
    """Keep labels consumable by the hash-bound frozen-value fitter, not just plots."""
    for source,target in ((inputs,output.with_suffix('.input.json')),
                          (networks,output.with_suffix('.networks.json'))):
        if target.exists():
            if digest(source)!=digest(target): raise ValueError('refuse to overwrite a different frozen companion')
        else:
            shutil.copyfile(source,target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--closed',action='store_true',help='Also integrate closed flop/turn all-in branches')
    args = parser.parse_args()
    experiment = json.loads((args.experiment/'summary.json').read_text())
    if experiment['status'] != 'completed_not_activated': raise ValueError('requires frozen completed experiment')
    args.output.mkdir(parents=True,exist_ok=False)
    binary = args.output/'sampler'; shutil.copy2(args.binary,binary)
    records = json.loads((args.experiment/'response-roots.json').read_text())[:16]
    inputs = args.output/'decisions.json'; atomic_json(inputs,records)
    report = {'schema':'hu-exact-closed-label-probe-v1' if args.closed else 'hu-exact-river-label-probe-v1','status':'running_not_activated',
        'sampler_sha256':digest(binary),'input_sha256':digest(inputs),'depth_bb':experiment['depth_bb'],
        'source_experiment_sha256':digest(args.experiment/'summary.json'),
        'release_eligible':False,'active_models_changed':False,'decisions':len(records),'results':[]}
    atomic_json(args.output/'summary.json',report)
    for seed in experiment['seeds']:
        network = args.experiment/f'averaged-{seed}/frozen-values.networks.json'
        grid = json.loads(network.read_text())['qualification_game']['action_abstraction']
        results = {}
        for exact in (False,True):
            output = args.output/f'{seed}-{"exact" if exact else "sampled"}.json'
            command = [str(binary),'neural-frozen-values','--networks',str(network),'--input',str(inputs),
                '--output',str(output),'--effective-stack-bb',str(experiment['depth_bb']),
                '--hidden-blocks','512','--rollouts-per-block','2','--seed','2919023',
                '--action-abstraction-json',json.dumps(grid)]
            if exact: command.append('--exact-river-values')
            if exact and args.closed: command.append('--exact-closed-values')
            started = time.monotonic()
            subprocess.run(command,check=True,capture_output=True,timeout=600)
            result = json.loads(output.read_text())
            if result['input_sha256'] != digest(inputs) or result['source_network_sha256s'] != [digest(network)]:
                raise ValueError('exact label input/policy identity changed')
            if exact and bool(result.get('exact_closed_values')) != args.closed:
                raise ValueError('native estimator did not apply the requested closed-branch setting')
            prepare_companions(output,inputs,network)
            results[exact] = (result,time.monotonic()-started,digest(output))
        nonriver_equal = True
        zscores = []; river_nodes = []; exact_actions=0; sampled_columns_identical=True; exact_runouts=0
        for old,new in zip(results[False][0]['records'],results[True][0]['records'],strict=True):
            exact_runouts += new.get('exact_closed_runouts',0)
            if old['state']['street'] != 'river':
                nonriver_equal &= old == new
                if not args.closed: continue
            elif (new['estimator'] != 'exact_river_posterior_and_action_tree_v1'
                    or any(new['standard_errors_bb']) or new['independent_hidden_block_values_bb']):
                raise ValueError('exact expectation was misrepresented as sampled blocks')
            if old['state']['street']=='river': river_nodes.append(new['exact_tree_nodes'])
            estimators=new.get('action_estimators',['exact_river_posterior_and_action_tree_v1']*len(new['actions']))
            for index,(a,b,se,estimator) in enumerate(zip(old['action_values_bb'],new['action_values_bb'],old['standard_errors_bb'],estimators,strict=True)):
                if estimator=='independent_hidden_blocks_v1':
                    sampled_columns_identical &= a==b and old['standard_errors_bb'][index]==new['standard_errors_bb'][index]
                    sampled_columns_identical &= [row[index] for row in old['independent_hidden_block_values_bb']]==[row[index] for row in new['independent_hidden_block_values_bb']]
                    continue
                exact_actions+=1
                if new['standard_errors_bb'][index] != 0.: raise ValueError('exact action has sampling error')
                if se > 1e-10: zscores.append((a-b)/se)
                elif abs(a-b) > 1e-8: raise ValueError('deterministic sampled action disagrees with exact reference')
        if not sampled_columns_identical or (not args.closed and not nonriver_equal):
            raise ValueError('exact estimator changed sampled action columns')
        row = {'seed':seed,'nonriver_outputs_identical':nonriver_equal,'river_decisions':len(river_nodes),
            'sampled_action_columns_identical':sampled_columns_identical,'exact_actions':exact_actions,
            'exact_closed_runouts':exact_runouts,
            'exact_nodes_per_river_decision':river_nodes,
            'sampled_minus_exact_z_scores':zscores,
            'fraction_within_sampled_95_interval':float(np.mean(np.abs(zscores)<=1.96)),
            'estimators':{('exact_river' if exact else 'sampled'):{'seconds':values[1],'sha256':values[2],
                'all_action_reference_precision_coverage':float(np.mean([max(r['standard_errors_bb'])<=.02 for r in values[0]['records']])),
                'action_reference_precision_coverage':float(np.mean([se<=.02 for r in values[0]['records'] for se in r['standard_errors_bb']])),
                'mean_action_sampling_variance_bb2':float(np.mean([se**2 for r in values[0]['records'] for se in r['standard_errors_bb']])),
                'median_max_action_se_bb':float(np.median([max(r['standard_errors_bb']) for r in values[0]['records']]))}
                for exact,values in results.items()}}
        report['results'].append(row); atomic_json(args.output/'summary.json',report)
        print(json.dumps(row),flush=True)
    report.update(status='completed_not_activated',
        interpretation='Exact actions enumerate the frozen-policy posterior and applicable finite action/chance trees, giving zero Monte Carlo SE. This is not zero neural error, an exploitability bound, or authentic-distribution EV qualification. Open preflop/flop/turn subtrees remain sampled.')
    atomic_json(args.output/'summary.json',report)


if __name__ == '__main__': main()
