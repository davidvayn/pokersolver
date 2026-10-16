"""Artifact-bound qualification of served EVs; old conditional SE is diagnostic only."""
from __future__ import annotations

import math


def write_research_receipts(experiment_root, full_run_root):
    """Bind the new stratified probes to old exports; never relabel them release evidence."""
    import hashlib
    import json
    from pathlib import Path
    import mlx.core as mx
    import numpy as np
    from train import atomic_json, expand_state_action, scorer_json
    from validate_seeds import load_artifact_model, verify_artifact

    experiment_root, full_run_root = Path(experiment_root), Path(full_run_root)
    summary_path = experiment_root / 'summary.json'
    summary = json.loads(summary_path.read_text())
    runs = json.loads((full_run_root / 'summary.json').read_text())['results']
    receipts = []
    for depth in (50, 100):
        candidates = []; grid = None
        for audit in summary['existing_frozen_audits']:
            if audit['depth_bb'] != depth:
                continue
            run = next(run for run in runs if run['seed'] == audit['seed'])
            artifact = run['artifacts'][-1]
            verified = verify_artifact(artifact)
            if not verified['descriptor_matches'] or not verified['magic_valid']:
                raise ValueError('frozen EV receipt artifact integrity failed')
            prefix = experiment_root / f"frozen-{depth}-{audit['seed']}"
            reference_path = prefix.with_suffix('.json')
            reference = json.loads(reference_path.read_text())
            inputs = prefix.with_suffix('.input.json').read_bytes()
            network_bytes = prefix.with_suffix('.networks.json').read_bytes()
            if reference['input_sha256'] != hashlib.sha256(inputs).hexdigest() or reference['source_network_sha256s'] != [hashlib.sha256(network_bytes).hexdigest()]:
                raise ValueError('frozen EV reference provenance changed')
            model = load_artifact_model(Path(artifact['source']))
            if json.loads(network_bytes)['networks'] != [scorer_json(model)] * 2:
                raise ValueError('frozen reference policy differs from the exported artifact')
            value = load_artifact_model(Path(artifact['source']), 'baselineActionValue')
            records = json.loads(inputs)
            precision = []
            with mx.stream(mx.cpu):
                for record, row in zip(records, reference['records'], strict=True):
                    features = np.stack([expand_state_action(record['state'], action, depth) for action in record['actions']])
                    errors = .005 + np.logaddexp(0., np.asarray(value(mx.array(features)))[:, 1].astype(np.float64))
                    precision.append(float(np.sum(np.asarray(row['probabilities']) * (errors <= .02))))
            if grid is not None and grid != reference['action_abstraction']:
                raise ValueError('frozen receipt candidates used different grids')
            grid = reference['action_abstraction']
            candidates.append({'seed': audit['seed'], 'artifact_sha256': verified['sha256'],
                'reference_network_sha256': reference['source_network_sha256s'][0],
                'reference_report_sha256': hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                'frozen_policy_inference_verified': True,
                'independent_hidden_blocks': reference['hidden_blocks'],
                'heldout_decisions': len(reference['records']),
                'reference_decision_precision_coverage': float(np.mean([all(se <= .02 for se in row['standard_errors_bb']) for row in reference['records']])),
                'served_predicted_precision_coverage': float(np.mean(precision)),
                'nominal_95_interval_coverage': audit['calibration']['nominal_95_interval_coverage']})
        if len(candidates) != 2:
            raise ValueError('frozen research receipt requires both independent seeds')
        receipt = {'schema': 'hu-frozen-ev-qualification-v1', 'depth_bb': depth, 'action_abstraction': grid,
            'uncertainty_scope': 'opponent_posterior_and_future_board_outer_blocks',
            'evaluation_distribution': 'street_stratified_pilot', 'candidates': candidates,
            'status': 'research_not_activated',
            'source_experiment_summary_sha256': hashlib.sha256(summary_path.read_bytes()).hexdigest()}
        receipt['assessment'] = frozen_ev_gate(receipt, depth, grid, [candidate['artifact_sha256'] for candidate in candidates])
        output = experiment_root / f'{depth}bb-frozen-ev-qualification.json'
        atomic_json(output, receipt)
        receipts.append(str(output))
    return receipts


def frozen_ev_gate(receipt: dict | None, depth: int, grid: dict | None,
                   artifact_sha256s: list[str]) -> dict:
    if receipt is None:
        return {'available': False, 'qualified': False,
                'failures': ['Full information-set EV qualification was not measured.']}
    if receipt.get('schema') != 'hu-frozen-ev-qualification-v1' or receipt.get('depth_bb') != depth or receipt.get('action_abstraction') != grid:
        raise ValueError('frozen EV qualification has a different schema, depth or grid')
    if not artifact_sha256s or len(set(artifact_sha256s)) != len(artifact_sha256s):
        raise ValueError('frozen EV qualification requires distinct evaluated artifacts')
    candidates = receipt.get('candidates', [])
    hashes = [candidate.get('artifact_sha256') for candidate in candidates]
    if any(not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha) for sha in hashes) or len(set(hashes)) != len(hashes) or sorted(hashes) != sorted(artifact_sha256s):
        raise ValueError('frozen EV qualification is not bound to every evaluated artifact')
    failures = []
    if receipt.get('uncertainty_scope') != 'opponent_posterior_and_future_board_outer_blocks':
        failures.append('Conditional continuation noise does not qualify information-set EVs.')
    if receipt.get('evaluation_distribution') != 'heldout_authentic_reach_mixture':
        failures.append('Stratified pilot roots do not qualify authentic reach-weighted serving precision.')
    for candidate in candidates:
        if not candidate.get('frozen_policy_inference_verified') or candidate.get('independent_hidden_blocks', 0) < 4 or candidate.get('heldout_decisions', 0) < 1:
            failures.append('Frozen-policy inference or independent hidden-scenario provenance is missing.')
        for key in ('reference_decision_precision_coverage', 'served_predicted_precision_coverage',
                    'nominal_95_interval_coverage'):
            value = candidate.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'frozen EV qualification has invalid {key}')
            if value < .95:
                failures.append(f'{key} did not reach 95% for {candidate["artifact_sha256"][:12]}.')
    return {'available': True, 'qualified': not failures, 'failures': failures,
            'interpretation': 'Measured precision/calibration of artifact-bound frozen values; not an exploitability certificate.'}


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', required=True)
    parser.add_argument('--full-run-root', required=True)
    args = parser.parse_args()
    print(json.dumps(write_research_receipts(args.experiment_root, args.full_run_root)))
