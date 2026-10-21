#!/usr/bin/env python3
"""Re-evaluate a completed pilot on its unchanged data with native-aligned math.

This is a measurement correction, never a new training improvement. Preserve
the original report and its source hashes; bind the correction to that report.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from depth_repair import ROOT, frozen_reference
from run_matched_depth_repair import promotion_decision
from train import atomic_json
from validate_seeds import compare, latest_artifact, load_artifact_model, policy, record_features, verify_artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    source = args.pilot/'summary.json'
    original = json.loads(source.read_text())
    if original['status'] != 'completed_not_activated':
        raise ValueError('requires a completed frozen pilot')
    binary = args.pilot/'sampler'
    if hashlib.sha256(binary.read_bytes()).hexdigest() != original['sampler_sha256']:
        raise ValueError('frozen sampler identity changed')
    models = {}
    report = {'schema':'hu-home-native-aligned-precision-audit-v1','release_eligible':False,
        'source_experiment_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_experiment':str(source),'inference':'MLX CPU FP32, checked against actual native frozen policy',
        'interpretation':'Measurement correction on unchanged weights/data; not an improvement due to training.',
        'native_probability_checks':[],'batches':[],'validation':{}}
    roots = json.loads((args.pilot/'response-roots.json').read_text())
    for profile in original['profiles']:
        for seed in original['seeds']:
            training = next(row for row in original['training'] if row['profile']==profile and row['seed']==seed)
            state = json.loads((Path(training['output'])/'state.json').read_text())
            artifact = latest_artifact(state)
            checked = verify_artifact(artifact)
            key = f'{profile}-{seed}'
            if checked != original['artifacts'][key]:
                raise ValueError('frozen weights changed before measurement correction')
            model = load_artifact_model(Path(artifact['source']))
            models[key] = model
            native = frozen_reference(binary,model,roots,original['depth_bb'],args.output/f'{key}.json',policy_only=True)
            delta = max(float(np.max(np.abs(policy(model,record_features(model,row,original['depth_bb']))-
                np.array(result['probabilities'])))) for row,result in zip(roots,native['records'],strict=True))
            if delta > 2e-5:
                raise ValueError('CPU probability audit does not match the actual native policy')
            report['native_probability_checks'].append({'profile':profile,'seed':seed,'maximum_probability_delta':delta,
                'artifact_sha256':checked['sha256']})
    for index, before in enumerate(original['batches']):
        corpora = json.loads((args.pilot/f'reach-batch-{index}.json').read_text())
        after = {p:compare(models[f'{p}-51701'],models[f'{p}-51702'],corpora,original['depth_bb'],
            'unchanged fixed equal-policy mixture; CPU inference correction',True) for p in original['profiles']}
        report['batches'].append({'before_recorded':before,'after_native_aligned_cpu':after})
    control,candidate = original['profiles']
    report['decision']=promotion_decision([{'control':b['after_native_aligned_cpu'][control],
        'averaged':b['after_native_aligned_cpu'][candidate]} for b in report['batches']],original['paired_response'])
    for profile in original['profiles']:
        output = args.output/f'{profile}-validation.json'
        subprocess.run([sys.executable,str(Path(__file__).with_name('validate_seeds.py')),
            *[next(row['output'] for row in original['training'] if row['profile']==profile and row['seed']==seed)
              for seed in (51701,51702)],
            '--round',str(original['rounds']),'--traversals','512','--seed','2919021',
            '--evaluation-binary',str(binary),'--evaluation-binary-sha256',original['sampler_sha256'],
            '--output',str(output)],cwd=ROOT,check=True,capture_output=True,timeout=600)
        result = json.loads(output.read_text())
        report['validation'][profile]={'path':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
            'release_eligible':result['release_eligible'],'gates':result['gates'],
            'unmeasured_gates':result['unmeasured_gates']}
    atomic_json(args.output/'summary.json',report)
    print(json.dumps({'decision':report['decision'],'native_checks':report['native_probability_checks']}))


if __name__=='__main__':
    main()
