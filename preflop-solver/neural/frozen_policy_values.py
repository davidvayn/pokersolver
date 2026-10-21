#!/usr/bin/env python3
"""Fit serving-value candidates to one immutable exported policy, never training replay.

The one-output value sidecar is deliberately NOT the training control variate or
the browser's two-output (mean, sampling-SE) head. Its residual envelope includes
prediction error and must not be mislabeled as Monte Carlo standard error.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from train import (ActionScorer, INPUT_FEATURE_COUNT, atomic_json, camel_action_abstraction,
                   expand_state_action, linear_layers, make_compiled_step, scorer_json)
from validate_seeds import load_artifact_model, verify_artifact, model_input_count, model_has_arrival_memory


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_equal(first, second) -> bool:
    """Native compact cash amounts round-trip through FP32, not loose tolerances."""
    if isinstance(first, dict) and isinstance(second, dict):
        return first.keys() == second.keys() and all(native_equal(first[k], second[k]) for k in first)
    if isinstance(first, list) and isinstance(second, list):
        return len(first) == len(second) and all(native_equal(a, b) for a, b in zip(first, second))
    if isinstance(first, float) and isinstance(second, (float, int)):
        return np.float32(first) == np.float32(second)
    return first == second


def verify_frozen_targets(source: Path, reference_path: Path) -> tuple[dict, list[dict], dict]:
    artifact = {'source': str(source), 'binary': str(source.with_name('model.bin'))}
    checked = verify_artifact(artifact)
    if not checked['descriptor_matches'] or not checked['magic_valid']:
        raise ValueError('policy artifact integrity failed')
    contents = json.loads(source.read_text())
    binary = source.with_name('model.bin').read_bytes()
    metadata_size = struct.unpack_from('<I', binary, 8)[0]
    if (json.loads(binary[16:16 + metadata_size]) != contents['metadata']
            or binary[16 + metadata_size:] != np.asarray(contents['parameters'], dtype='<f4').tobytes()):
        raise ValueError('policy source and immutable binary disagree')
    reference = json.loads(reference_path.read_text())
    prefix_depth = reference.get('action_expectation_depth',0)
    if type(prefix_depth) is not int or prefix_depth not in (0,1,2):
        raise ValueError('invalid frozen action expectation depth')
    inputs = reference_path.with_suffix('.input.json')
    networks = reference_path.with_suffix('.networks.json')
    records = json.loads(inputs.read_text())
    depth = contents['metadata']['depthBb']
    if (reference.get('schema') != 'hu-frozen-information-set-values-v1'
            or reference.get('policy_only') or reference.get('snapshot_average')
            or reference.get('hidden_blocks', 0) < 4
            or not reference.get('uncertainty_contract', '').startswith('Independent outer opponent-posterior/future-board blocks')
            or reference.get('depth_bb') != depth
            or camel_action_abstraction(reference['action_abstraction']) != contents['metadata']['actionAbstraction']
            or reference.get('input_sha256') != digest(inputs)
            or reference.get('source_network_sha256s') != [digest(networks)]):
        raise ValueError('values are not bound to this frozen policy/game/input')
    policy = scorer_json(load_artifact_model(source))
    bundle = json.loads(networks.read_text())
    if (bundle.get('strategy_transform') != 'softmax' or bundle.get('networks') != [policy, policy]
            or set(bundle) - {'schema', 'input_size', 'strategy_transform', 'networks', 'qualification_game'}):
        raise ValueError('reference continuation policy differs from the served policy')
    if not records or len(records) != len(reference['records']):
        raise ValueError('frozen value records are not aligned')
    for record, row in zip(records, reference['records'], strict=True):
        if not native_equal(record['state'], row['state']) or not native_equal(record['actions'], row['actions']):
            raise ValueError('frozen value state/action alignment changed')
        n = len(record['actions'])
        values, errors = np.asarray(row['action_values_bb']), np.asarray(row['standard_errors_bb'])
        if (values.shape != (n,) or errors.shape != (n,) or not np.all(np.isfinite(values))
                or not np.all(np.isfinite(errors)) or np.any(errors < 0)):
            raise ValueError('invalid frozen value targets')
        if row.get('estimator') == 'exact_river_posterior_and_action_tree_v1':
            if (not reference.get('exact_river_values') or record['state']['street'] != 'river'
                    or np.any(errors) or row.get('independent_hidden_block_values_bb')
                    or not 0 < row.get('exact_tree_nodes',0) <= 2_000_000):
                raise ValueError('invalid exact river uncertainty contract')
        estimators = row.get('action_estimators')
        if estimators is not None:
            if len(estimators) != n: raise ValueError('exact action estimator count differs')
            for index,estimator in enumerate(estimators):
                if estimator not in ('independent_hidden_blocks_v1','exact_river_posterior_and_action_tree_v1','exact_closed_posterior_runouts_v1'):
                    raise ValueError('unknown frozen action estimator')
                if estimator == 'exact_river_posterior_and_action_tree_v1' and (
                        row.get('estimator') != estimator or not reference.get('exact_river_values')
                        or record['state']['street'] != 'river' or errors[index] != 0.):
                    raise ValueError('invalid per-action river uncertainty contract')
                if estimator == 'exact_closed_posterior_runouts_v1' and (
                        not reference.get('exact_closed_values') or errors[index] != 0.
                        or not 0 <= row.get('exact_closed_runouts',-1) <= 2_000_000):
                    raise ValueError('invalid closed-action uncertainty contract')
    return checked, records, reference


def split_families(records: list[dict], seed: int) -> list[list[int]]:
    families = sorted({r['iteration'] for r in records})
    if len(families) < 12:
        raise ValueError('requires >=12 disjoint families for train/tune/calibration/test')
    np.random.default_rng(seed).shuffle(families)
    return [list(map(int, part)) for part in np.split(np.asarray(families),
        [len(families) // 2, len(families) * 2 // 3, len(families) * 5 // 6])]


def mean_from_training_head(source: Path, depth: int):
    original = load_artifact_model(source, 'baselineActionValue')
    hidden = tuple(int(layer.weight.shape[0]) for layer in linear_layers(original)[:2])
    mean = ActionScorer(model_input_count(original), hidden)
    for i, (destination, old) in enumerate(zip(linear_layers(mean), linear_layers(original))):
        destination.weight = mx.array(np.asarray(old.weight) if i < 2 else np.asarray(old.weight)[:1] / depth)
        destination.bias = mx.array(np.asarray(old.bias) if i < 2 else np.asarray(old.bias)[:1] / depth)
    mx.eval(mean.parameters())
    return mean


def fit_policy_values(source: Path, reference_path: Path, output: Path, *, seed=2818031,
                      trial_steps=(400, 1600), learning_rate=.0001) -> dict:
    if (not trial_steps or any(not 1 <= steps <= 2000 for steps in trial_steps)
            or not 0 < learning_rate <= .001):
        raise ValueError('invalid short frozen-policy fitting budget')
    checked, records, reference = verify_frozen_targets(source, reference_path)
    depth = int(reference['depth_bb'])
    arrival = model_has_arrival_memory(load_artifact_model(source))
    splits = split_families(records, seed)
    datasets = []
    for families in splits:
        features, targets, errors, groups = [], [], [], []
        for i, (record, row) in enumerate(zip(records, reference['records'], strict=True)):
            if record['iteration'] in families:
                features.extend(expand_state_action(record['state'], a, depth, card_arrival_memory=arrival) for a in record['actions'])
                targets.extend(row['action_values_bb']); errors.extend(row['standard_errors_bb'])
                groups.extend([i] * len(record['actions']))
        datasets.append((np.asarray(features, dtype=np.float32), np.asarray(targets), np.asarray(errors), np.asarray(groups)))
    original = mean_from_training_head(source, depth)
    def predict(model, dataset):
        with mx.stream(mx.cpu):
            return np.asarray(model(mx.array(dataset[0]))).reshape(-1).astype(float) * depth
    def rmse(model, dataset):
        return float(np.sqrt(np.mean((predict(model, dataset) - dataset[1]) ** 2)))
    # The historical head is a CONTROL, never a frozen-policy-trained candidate.
    # Selecting step zero and relabeling it would recreate the original mismatch.
    baseline_tuning = rmse(original, datasets[1])
    trials = []
    for steps in trial_steps:
        model = copy.deepcopy(original)
        optimizer = optim.AdamW(learning_rate=learning_rate)
        optimizer.init(model.trainable_parameters())
        step = make_compiled_step(model, optimizer)
        rng = np.random.default_rng(seed)
        x, y, _, _ = datasets[0]
        for _ in range(steps):
            ids = rng.integers(len(x), size=128)
            loss = step(mx.array(x[ids]), mx.array(y[ids, None] / depth, dtype=mx.float32), mx.ones((128, 1)))
            mx.eval(loss, model.parameters(), optimizer.state)
            if not np.isfinite(float(loss)):
                raise RuntimeError('nonfinite frozen-policy fit')
        trials.append((steps, model, rmse(model, datasets[1])))
    steps, selected, _ = min(trials, key=lambda trial: trial[2])
    calibration = np.abs(predict(selected, datasets[2]) - datasets[2][1]) + 1.96 * datasets[2][2]
    radius = float(np.quantile(calibration, .95, method='higher'))
    before, after = predict(original, datasets[3]), predict(selected, datasets[3])
    target, se, groups = datasets[3][1:]
    def centered_rmse(values):
        residuals = []
        for group in np.unique(groups):
            error = values[groups == group] - target[groups == group]
            residuals.extend(error - error.mean())
        return float(np.sqrt(np.mean(np.square(residuals))))
    output.mkdir(parents=True, exist_ok=False)
    bundle = {'schema': 'hu-frozen-policy-value-sidecar-v1', 'status': 'research_not_activated',
        'value_target_policy_matched': True,
        'policy_artifact_sha256': checked['sha256'], 'policy_source_sha256': digest(source),
        'policy_network_sha256': reference['source_network_sha256s'][0],
        'reference_sha256': digest(reference_path), 'depth_bb': depth,
        'state_feature_schema': json.loads(source.read_text())['metadata']['stateFeatureSchema'],
        'action_abstraction': reference['action_abstraction'],
        'mean_network_bb': scorer_json(selected, output_index=0, output_scale=depth),
        'confidence': {'kind': 'empirical_prediction_residual_envelope',
            'nominal_level': .95, 'half_width_bb': radius, 'is_sampling_standard_error': False,
            'precision_qualified': False},
        'release_eligible': False}
    sidecar_path = output / 'frozen-policy-values.json'
    atomic_json(sidecar_path, bundle)
    report = {k: v for k, v in bundle.items() if k != 'mean_network_bb'}
    report.update(family_splits=splits, seed=seed, trial_steps=list(trial_steps), selected_steps=steps,
        baseline_tuning_rmse_bb=baseline_tuning,
        fixed_policy_fit_better_on_tuning=trials[[n for n, _, _ in trials].index(steps)][2] < baseline_tuning,
        learning_rate=learning_rate,
        source_sha256s={name: digest(Path(__file__).with_name(name)) for name in
            ('frozen_policy_values.py', 'train.py', 'validate_seeds.py')},
        trial_tuning_rmse_bb=[[n, error] for n, _, error in trials],
        test_actions=len(target), test_decisions=len(np.unique(groups)),
        matched_before_rmse_bb=float(np.sqrt(np.mean((before - target) ** 2))),
        matched_after_rmse_bb=float(np.sqrt(np.mean((after - target) ** 2))),
        matched_before_centered_action_rmse_bb=centered_rmse(before),
        matched_after_centered_action_rmse_bb=centered_rmse(after),
        heldout_empirical_interval_coverage=float(np.mean(np.abs(after - target) <= radius)),
        heldout_reference_precision_coverage=float(np.mean(se <= .02)),
        sidecar_sha256=digest(sidecar_path),
        interpretation='Matched held-out families; frozen-policy means, not an exploitability or confidence qualification. Residual bands include noisy references.')
    atomic_json(output / 'report.json', report)
    return report


def load_policy_value_sidecar(path: Path, artifact_binary: Path) -> dict:
    bundle = json.loads(path.read_text())
    if (bundle.get('schema') != 'hu-frozen-policy-value-sidecar-v1'
            or bundle.get('policy_artifact_sha256') != digest(artifact_binary)
            or bundle.get('value_target_policy_matched') is not True
            or bundle.get('confidence', {}).get('is_sampling_standard_error') is not False):
        raise ValueError('value sidecar belongs to a different policy or confidence contract')
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(fit_policy_values(args.source, args.reference, args.output)), flush=True)


if __name__ == '__main__':
    main()
