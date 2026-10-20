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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    experiment = json.loads((args.experiment/'summary.json').read_text())
    if experiment['status'] != 'completed_not_activated': raise ValueError('requires frozen completed experiment')
    args.output.mkdir(parents=True,exist_ok=False)
    binary = args.output/'sampler'; shutil.copy2(args.binary,binary)
    records = json.loads((args.experiment/'response-roots.json').read_text())[:16]
    inputs = args.output/'decisions.json'; atomic_json(inputs,records)
    report = {'schema':'hu-exact-river-label-probe-v1','status':'running_not_activated',
        'sampler_sha256':digest(binary),'input_sha256':digest(inputs),'depth_bb':experiment['depth_bb'],
        'source_experiment_sha256':digest(args.experiment/'summary.json'),
        'release_eligible':False,'active_models_changed':False,'results':[]}
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
            started = time.monotonic()
            subprocess.run(command,check=True,capture_output=True,timeout=600)
            result = json.loads(output.read_text())
            if result['input_sha256'] != digest(inputs) or result['source_network_sha256s'] != [digest(network)]:
                raise ValueError('exact label input/policy identity changed')
            results[exact] = (result,time.monotonic()-started,digest(output))
        nonriver_equal = True
        zscores = []; river_nodes = []
        for old,new in zip(results[False][0]['records'],results[True][0]['records'],strict=True):
            if old['state']['street'] != 'river':
                nonriver_equal &= old == new
                continue
            if (new['estimator'] != 'exact_river_posterior_and_action_tree_v1'
                    or any(new['standard_errors_bb']) or new['independent_hidden_block_values_bb']):
                raise ValueError('exact expectation was misrepresented as sampled blocks')
            river_nodes.append(new['exact_tree_nodes'])
            for a,b,se in zip(old['action_values_bb'],new['action_values_bb'],old['standard_errors_bb']):
                if se > 1e-10: zscores.append((a-b)/se)
                elif abs(a-b) > 1e-8: raise ValueError('deterministic sampled action disagrees with exact reference')
        if not nonriver_equal: raise ValueError('exact-river flag changed sampled non-river labels')
        row = {'seed':seed,'nonriver_outputs_identical':nonriver_equal,'river_decisions':len(river_nodes),
            'exact_nodes_per_river_decision':river_nodes,
            'sampled_minus_exact_z_scores':zscores,
            'fraction_within_sampled_95_interval':float(np.mean(np.abs(zscores)<=1.96)),
            'estimators':{('exact_river' if exact else 'sampled'):{'seconds':values[1],'sha256':values[2],
                'all_action_reference_precision_coverage':float(np.mean([max(r['standard_errors_bb'])<=.02 for r in values[0]['records']])),
                'median_max_action_se_bb':float(np.median([max(r['standard_errors_bb']) for r in values[0]['records']]))}
                for exact,values in results.items()}}
        report['results'].append(row); atomic_json(args.output/'summary.json',report)
        print(json.dumps(row),flush=True)
    report.update(status='completed_not_activated',
        interpretation='Exact river labels have zero Monte Carlo SE conditional on the frozen modeled game and posterior. This is not zero neural error, an exploitability bound, or an authentic-distribution EV qualification. Other streets remain sampled.')
    atomic_json(args.output/'summary.json',report)


if __name__ == '__main__': main()
