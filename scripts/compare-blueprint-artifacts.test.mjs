import assert from 'node:assert/strict';
import test from 'node:test';

import {
  ArtifactComparisonError,
  compareBlueprintArtifacts,
} from './compare-blueprint-artifacts.mjs';

const ranks = '23456789TJQKA';

function handLabels() {
  const labels = [];
  for (let high = 0; high < ranks.length; high += 1) {
    labels.push(`${ranks[high]}${ranks[high]}`);
    for (let low = 0; low < high; low += 1) {
      labels.push(`${ranks[high]}${ranks[low]}s`);
      labels.push(`${ranks[high]}${ranks[low]}o`);
    }
  }
  return labels;
}

function fixtureProbabilities(label) {
  const high = ranks.indexOf(label[0]);
  const low = ranks.indexOf(label[1]);
  const pairBonus = label.length === 2 ? 13 : 0;
  const suitedBonus = label.endsWith('s') ? 2 : 0;
  const rawStrength = high + low + pairBonus + suitedBonus;
  const premium = ['AA', 'AKs', 'KK', 'QQ'].includes(label);
  const continueFrequency = premium
    ? 0.995
    : 0.02 + (0.97 * rawStrength) / 39;
  const shoveFrequency = 0.005;
  return [
    1 - continueFrequency,
    continueFrequency - shoveFrequency,
    shoveFrequency,
  ];
}

function actionDistribution(label, probabilities) {
  const [fold, limp, shove] =
    typeof probabilities === 'function'
      ? probabilities(label)
      : probabilities ?? fixtureProbabilities(label);
  return [
    { action: 'fold', probability: fold },
    { action: 'limp', probability: limp },
    ...(shove === undefined
      ? []
      : [
          {
            action: 'raise_all_in_to_100.000bb',
            probability: shove,
          },
        ]),
  ];
}

function artifact(seed, probabilities = undefined, config = {}) {
  return {
    schema_version: 1,
    artifact_id: `fixture-${seed}`,
    model: 'fixture-blueprint-v1',
    approximate: true,
    config: {
      effective_stack_bb: 100,
      iterations: 100_000,
      seed,
      held_out_deals: 20_000,
      recall_mode: 'current_street',
      ...config,
    },
    metrics: {
      requested_iterations: 100_000,
      training_iterations: 100_000,
      stopped_early: false,
      sampled_deals: 200_000,
      terminal_evaluations: 1_000_000,
      information_sets: 10_000,
      preflop_information_sets: 1_000,
      postflop_information_sets: 9_000,
      trained_information_sets: 9_000,
      exported_information_sets: 1_000,
      held_out: {
        deals: 20_000,
        button_mean_net_bb: 0.2,
        button_net_standard_error_bb: 0.1,
        fold_terminal_fraction: 0.5,
        showdown_terminal_fraction: 0.5,
        unknown_information_set_fraction: 0.01,
        untrained_information_set_fraction: 0.01,
      },
      root_local_deviation: {
        kind: 'button-root-one-step-local-best-response-v1',
        samples_per_class: 512,
        classes: handLabels().map((label) => ({
          hand_class: label,
          exact_combo_count:
            label.length === 2 ? 6 : label.endsWith('s') ? 4 : 12,
          root_policy_trained: true,
          action_values: actionDistribution(label, probabilities).map(
            (action) => ({
              action: action.action,
              samples: 512,
            })
          ),
        })),
        aggregate_chosen_average_ev_bb: 0.2,
        aggregate_best_action_ev_bb: 0.24,
        aggregate_local_deviation_gain_bb: 0.04,
        aggregate_local_deviation_gain_standard_error_bb: 0.02,
        aggregate_local_deviation_gain_99pct_lower_bound_bb: 0.01,
        trained_root_combo_fraction: 1,
        continuation_coverage: {
          decisions: 100_000,
          unknown_information_set_fraction: 0.01,
          untrained_information_set_fraction: 0.01,
        },
      },
    },
    strategies: handLabels().map((label) => ({
      actor: 'button_small_blind',
      street: 'preflop',
      hand_bucket_trajectory: [`preflop:${label}`],
      public_history: ['blinds:0.500/1.000'],
      average_visits: 500,
      regret_updates: 600,
      trained_average: true,
      actions: actionDistribution(label, probabilities),
    })),
  };
}

test('identical root strategies pass and expose weighted metrics', () => {
  const report = compareBlueprintArtifacts([
    { file: 'seed-1.json', artifact: artifact(1) },
    { file: 'seed-2.json', artifact: artifact(2) },
  ]);

  assert.equal(
    report.passed,
    true,
    JSON.stringify(report.acceptance_gates.filter((gate) => !gate.passed))
  );
  assert.equal(
    report.artifacts[0].coverage_and_held_out_metrics.root.weighted_combos,
    1326
  );
  assert.deepEqual(
    Object.keys(
      report.artifacts[0].coverage_and_held_out_metrics.root
        .combo_weighted_aggregate_frequencies
    ),
    ['fold', 'limp', 'raise_all_in_to_100.000bb']
  );
  assert.ok(
    Math.abs(
      report.artifacts[0].poker_domain_sanity
        .aggregate_open_shove_frequency - 0.005
    ) < 1e-12
  );
  assert.ok(
    report.artifacts[0].poker_domain_sanity
      .hand_strength_continue_rank_order.spearman_correlation >= 0.65
  );
  assert.equal(
    report.comparisons[0].per_hand_total_variation.median,
    0
  );
  assert.equal(
    report.comparisons[0].primary_action_agreement.hand_class_fraction,
    1
  );
});

test('unstable strategies fail stability gates', () => {
  const report = compareBlueprintArtifacts([
    { file: 'seed-1.json', artifact: artifact(1, [0.1, 0.9]) },
    { file: 'seed-2.json', artifact: artifact(2, [0.9, 0.1]) },
  ]);

  assert.equal(report.passed, false);
  assert.ok(
    Math.abs(
      report.comparisons[0].combo_weighted_per_action_mae.fold - 0.8
    ) < 1e-12
  );
  assert.ok(
    Math.abs(
      report.comparisons[0].per_hand_total_variation.median - 0.8
    ) < 1e-12
  );
  assert.equal(
    report.comparisons[0].primary_action_agreement.hand_class_fraction,
    0
  );
});

test('training parameters other than seed, iterations, and evaluation controls must match', () => {
  assert.throws(
    () =>
      compareBlueprintArtifacts([
        { file: 'seed-1.json', artifact: artifact(1) },
        {
          file: 'seed-2.json',
          artifact: artifact(2, undefined, {
            recall_mode: 'full_recall',
          }),
        },
      ]),
    (error) =>
      error instanceof ArtifactComparisonError &&
      error.code === 'INCOMPATIBLE_ARTIFACTS'
  );

  assert.doesNotThrow(() =>
    compareBlueprintArtifacts([
      { file: 'seed-1.json', artifact: artifact(1) },
      {
        file: 'seed-2.json',
        artifact: artifact(2, undefined, {
          iterations: 200_000,
          held_out_deals: 50_000,
          max_information_sets: 20_000,
        }),
      },
    ])
  );
});

test('domain gates reject stable but strategically pathological roots', () => {
  const pathological = [0.1, 0.7, 0.2];
  const report = compareBlueprintArtifacts([
    { file: 'seed-1.json', artifact: artifact(1, pathological) },
    { file: 'seed-2.json', artifact: artifact(2, pathological) },
  ]);
  const gates = Object.fromEntries(
    report.acceptance_gates.map((entry) => [entry.name, entry])
  );

  assert.equal(report.comparisons[0].per_hand_total_variation.max, 0);
  assert.equal(
    gates.maximum_aggregate_open_shove_frequency.passed,
    false
  );
  assert.equal(gates.maximum_premium_hand_fold_frequency.passed, false);
  assert.equal(gates.maximum_trash_hand_open_shove_frequency.passed, false);
  assert.equal(gates.minimum_hand_strength_continue_spearman.passed, false);
  assert.equal(report.passed, false);
});

test('legacy artifacts remain comparable but cannot pass without a local-deviation audit', () => {
  const first = artifact(1);
  const second = artifact(2);
  delete first.metrics.root_local_deviation;
  delete second.metrics.root_local_deviation;

  const report = compareBlueprintArtifacts([
    { file: 'seed-1.json', artifact: first },
    { file: 'seed-2.json', artifact: second },
  ]);
  const gates = Object.fromEntries(
    report.acceptance_gates.map((entry) => [entry.name, entry])
  );

  assert.equal(report.artifacts[0].root_local_deviation_audit.available, false);
  assert.equal(gates.root_local_deviation_audit_available.passed, false);
  assert.equal(
    gates.minimum_root_local_deviation_samples_per_class.observed,
    null
  );
  assert.equal(report.passed, false);
});

test('local-deviation gates fail closed on insufficient or exploitable audits', () => {
  const first = artifact(1);
  const second = artifact(2);
  const audit = second.metrics.root_local_deviation;
  audit.samples_per_class = 128;
  audit.classes[0].action_values[0].samples = 64;
  audit.classes[0].root_policy_trained = false;
  audit.trained_root_combo_fraction = 0.99;
  audit.continuation_coverage.unknown_information_set_fraction = 0.051;
  audit.continuation_coverage.untrained_information_set_fraction = 0.026;
  audit.aggregate_local_deviation_gain_bb = 0.101;
  audit.aggregate_local_deviation_gain_99pct_lower_bound_bb = 0.051;

  const report = compareBlueprintArtifacts([
    { file: 'seed-1.json', artifact: first },
    { file: 'seed-2.json', artifact: second },
  ]);
  const gates = Object.fromEntries(
    report.acceptance_gates.map((entry) => [entry.name, entry])
  );
  const expectedFailures = [
    'minimum_root_local_deviation_samples_per_class',
    'full_trained_root_in_local_deviation_audit',
    'maximum_root_local_deviation_unknown_information_set_fraction',
    'maximum_root_local_deviation_untrained_information_set_fraction',
    'maximum_aggregate_root_local_deviation_gain_bb',
    'maximum_aggregate_root_local_deviation_99pct_lower_bound_bb',
  ];

  for (const name of expectedFailures) {
    assert.equal(gates[name].passed, false, name);
  }
  assert.equal(report.passed, false);
});
