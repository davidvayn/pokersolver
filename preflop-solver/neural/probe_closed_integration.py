#!/usr/bin/env python3
"""Matched follow-up: change only terminal-response integration, on one binary."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

from depth_repair import ROOT, audit_exported_values, frozen_reference, run_training, select_street_roots
from run_depth_pilots import PILOT_GRID
from train import atomic_json
from validate_seeds import compare, evaluation_records, latest_artifact, load_artifact_model, load_run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--response-hidden-blocks',type=int,default=256)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    binary=args.output/'sampler';shutil.copy2(args.binary,binary)
    profiles=('averaged','integrated');seeds=(51701,51702)
    sources={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
             for name in ('train.py','depth_repair.py','validate_seeds.py','probe_closed_integration.py',
                          'frozen_ev_gate.py','run_depth_pilots.py','run_depth_full.py')}
    report={'schema':'hu-closed-integration-pilot-v1','status':'running_not_activated',
        'sampler_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'source_sha256s':sources,'training':[],'comparisons':{}}
    atomic_json(args.output/'summary.json',report)
    tasks=[(profile,seed) for profile in profiles for seed in seeds]
    def task(spec):
        if any(hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=sha for name,sha in sources.items()):
            raise RuntimeError('closed-response pilot inputs changed while queued')
        profile,seed=spec
        return run_training(binary,args.output/f'{profile}-{seed}',profile,seed,12,64,80)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(task,tasks):
            report['training'].append(result);atomic_json(args.output/'summary.json',report)
            print(json.dumps(result),flush=True)
    models={profile:[load_artifact_model(Path(latest_artifact(load_run(args.output/f'{profile}-{seed}')[0])['source'])) for seed in seeds] for profile in profiles}
    reach=[evaluation_records(ROOT,50,256,2471704,64,32,True,False,policy_model=model,
        action_abstraction=PILOT_GRID,binary_path=binary) for model in models['averaged']]
    roots=select_street_roots(reach[0],2)
    for profile in profiles:
        result={'authentic':compare(*models[profile],reach,50,'fixed averaged-control reach',True),'seeds':[]}
        for seed,model in zip(seeds,models[profile]):
            run=args.output/f'{profile}-{seed}'; state,_=load_run(run)
            reference=frozen_reference(binary,model,roots,50,run/'frozen-values.json',blocks=args.response_hidden_blocks)
            value=load_artifact_model(Path(latest_artifact(state)['source']),'baselineActionValue')
            result['seeds'].append(audit_exported_values(model,value,roots,reference,50))
        report['comparisons'][profile]=result;atomic_json(args.output/'summary.json',report)
    report.update(status='completed_not_activated',exploitability_measured=False,
        interpretation='Terminal response estimator ablation only; no automatic long run or release.')
    atomic_json(args.output/'summary.json',report)
    print(json.dumps({'status':report['status'],'mae':{p:r['authentic']['action_frequency_mae'] for p,r in report['comparisons'].items()}}))


if __name__=='__main__':main()
