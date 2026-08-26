#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { pathToFileURL } from 'node:url';
import { gunzipSync } from 'node:zlib';

const ANALYZER = 'hu-blueprint-root-stability-v1';
const PROBABILITY_TOLERANCE = 1e-6;
const EXPECTED_HAND_CLASSES = 169;
const EXPECTED_COMBOS = 1326;
const SUPPORTED_EFFECTIVE_STACKS_BB = Object.freeze([20, 50, 100]);

const IGNORED_CONFIG_KEYS = new Set([
  'seed',
  'iterations',
  'held_out_deals',
  'evaluation',
  'evaluation_controls',
  'evaluation_deals',
  'evaluation_iterations',
  'evaluation_seed',
  'max_information_sets',
]);

export const DEFAULT_THRESHOLDS = Object.freeze({
  minimum_root_average_visits: 100,
  minimum_held_out_deals: 10_000,
  maximum_held_out_unknown_information_set_fraction: 0.05,
  maximum_held_out_untrained_information_set_fraction: 0.025,
  maximum_aggregate_action_frequency_delta: 0.03,
  maximum_per_action_mae: 0.05,
  maximum_hand_total_variation_median: 0.2,
  maximum_hand_total_variation_p95: 0.35,
  maximum_hand_total_variation_max: 0.65,
  minimum_primary_action_agreement: 0.85,
  maximum_held_out_button_ev_z_score: 2.576,
  maximum_aggregate_open_shove_frequency: 0.01,
  maximum_premium_hand_fold_frequency: 0.02,
  maximum_trash_hand_open_shove_frequency: 0.02,
  minimum_hand_strength_continue_spearman: 0.65,
  minimum_strength_quartile_continue_gap: 0.3,
  minimum_root_local_deviation_samples_per_class: 256,
  maximum_root_local_deviation_unknown_information_set_fraction: 0.05,
  maximum_root_local_deviation_untrained_information_set_fraction: 0.025,
  maximum_aggregate_root_local_deviation_gain_bb: 0.1,
  maximum_aggregate_root_local_deviation_99pct_lower_bound_bb: 0.05,
});

export function parseBlueprintArtifactBytes(file, bytes) {
  const payload = file.endsWith('.gz') ? gunzipSync(bytes) : bytes;
  return JSON.parse(payload.toString('utf8'));
}

export class ArtifactComparisonError extends Error {
  constructor(code, message, details = undefined) {
    super(message);
    this.name = 'ArtifactComparisonError';
    this.code = code;
    this.details = details;
  }
}

function finiteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function stableObject(value) {
  if (Array.isArray(value)) return value.map(stableObject);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(
    Object.keys(value)
      .sort()
      .map((key) => [key, stableObject(value[key])])
  );
}

function stableStringify(value) {
  return JSON.stringify(stableObject(value));
}

function normalizedConfig(config) {
  if (!config || typeof config !== 'object' || Array.isArray(config)) {
    throw new ArtifactComparisonError(
      'INVALID_ARTIFACT',
      'artifact config must be an object'
    );
  }
  return Object.fromEntries(
    Object.entries(config)
      .filter(([key]) => !IGNORED_CONFIG_KEYS.has(key))
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, value]) => [key, stableObject(value)])
  );
}

function valueDifferences(left, right, prefix = '') {
  if (stableStringify(left) === stableStringify(right)) return [];
  if (
    !left ||
    !right ||
    typeof left !== 'object' ||
    typeof right !== 'object' ||
    Array.isArray(left) ||
    Array.isArray(right)
  ) {
    return [{ path: prefix || '<root>', left, right }];
  }
  const keys = [
    ...new Set([...Object.keys(left), ...Object.keys(right)]),
  ].sort();
  return keys.flatMap((key) =>
    valueDifferences(left[key], right[key], prefix ? `${prefix}.${key}` : key)
  );
}

function handClassLabels() {
  const ranks = '23456789TJQKA';
  const labels = [];
  for (let high = 0; high < ranks.length; high += 1) {
    labels.push(`${ranks[high]}${ranks[high]}`);
    for (let low = 0; low < high; low += 1) {
      labels.push(`${ranks[high]}${ranks[low]}s`);
      labels.push(`${ranks[high]}${ranks[low]}o`);
    }
  }
  return labels.sort();
}

const CANONICAL_HAND_CLASSES = new Set(handClassLabels());

function comboWeight(label) {
  if (label.length === 2) return 6;
  if (label.endsWith('s')) return 4;
  if (label.endsWith('o')) return 12;
  throw new ArtifactComparisonError(
    'INVALID_ROOT_STRATEGY',
    `invalid hand class ${label}`
  );
}

const CHEN_HIGH_CARD_POINTS = Object.freeze({
  A: 10,
  K: 8,
  Q: 7,
  J: 6,
  T: 5,
  9: 4.5,
  8: 4,
  7: 3.5,
  6: 3,
  5: 2.5,
  4: 2,
  3: 1.5,
  2: 1,
});
const RANK_INDEX = Object.fromEntries(
  [...'23456789TJQKA'].map((rank, index) => [rank, index])
);
const PREMIUM_HANDS = Object.freeze(['AA', 'AKs', 'KK', 'QQ']);
const TRASH_HANDS = Object.freeze(['32o', '72o']);

function chenStructuralScore(label) {
  const highRank = label[0];
  const lowRank = label[1];
  let score = CHEN_HIGH_CARD_POINTS[highRank];
  if (label.length === 2) {
    return Math.max(5, score * 2);
  }
  if (label.endsWith('s')) score += 2;
  const gap = RANK_INDEX[highRank] - RANK_INDEX[lowRank] - 1;
  score -= [0, 1, 2, 4, 5][Math.min(gap, 4)];
  if (gap <= 1 && RANK_INDEX[highRank] < RANK_INDEX.Q) score += 1;
  return Math.ceil(score * 2) / 2;
}

function tiedRanks(values) {
  const sorted = values
    .map((value, index) => ({ value, index }))
    .sort(
      (left, right) => left.value - right.value || left.index - right.index
    );
  const ranks = Array(values.length);
  for (let start = 0; start < sorted.length;) {
    let end = start;
    while (
      end + 1 < sorted.length &&
      sorted[end + 1].value === sorted[start].value
    ) {
      end += 1;
    }
    const averageRank = (start + end) / 2 + 1;
    for (let index = start; index <= end; index += 1) {
      ranks[sorted[index].index] = averageRank;
    }
    start = end + 1;
  }
  return ranks;
}

function pearsonCorrelation(left, right) {
  if (left.length !== right.length || left.length < 2) return null;
  const leftMean = left.reduce((sum, value) => sum + value, 0) / left.length;
  const rightMean = right.reduce((sum, value) => sum + value, 0) / right.length;
  let covariance = 0;
  let leftVariance = 0;
  let rightVariance = 0;
  for (let index = 0; index < left.length; index += 1) {
    const leftDelta = left[index] - leftMean;
    const rightDelta = right[index] - rightMean;
    covariance += leftDelta * rightDelta;
    leftVariance += leftDelta * leftDelta;
    rightVariance += rightDelta * rightDelta;
  }
  const denominator = Math.sqrt(leftVariance * rightVariance);
  return denominator > 0 ? covariance / denominator : null;
}

function spearmanCorrelation(left, right) {
  return pearsonCorrelation(tiedRanks(left), tiedRanks(right));
}

function domainSanitySummary(root, effectiveStackBb) {
  const shoveActions = root.actions.filter((action) =>
    /(?:all_in|shove)/i.test(action)
  );
  const hasFoldAction = root.actions.includes('fold');
  const handRows = [...root.hands.entries()]
    .map(([hand, strategy]) => {
      const fold = hasFoldAction ? strategy.probabilities.fold : null;
      const shove = shoveActions.reduce(
        (sum, action) => sum + strategy.probabilities[action],
        0
      );
      return {
        hand,
        combo_weight: strategy.combo_weight,
        structural_strength_score: chenStructuralScore(hand),
        continue_frequency: finiteNumber(fold) ? 1 - fold : null,
        fold_frequency: fold,
        open_shove_frequency: shove,
      };
    })
    .sort(
      (left, right) =>
        right.structural_strength_score - left.structural_strength_score ||
        left.hand.localeCompare(right.hand)
    );
  const aggregateOpenShove =
    handRows.reduce(
      (sum, hand) => sum + hand.combo_weight * hand.open_shove_frequency,
      0
    ) / EXPECTED_COMBOS;
  const premiumFoldFrequencies = Object.fromEntries(
    PREMIUM_HANDS.map((label) => [
      label,
      handRows.find((hand) => hand.hand === label)?.fold_frequency ?? null,
    ])
  );
  const trashShoveFrequencies = Object.fromEntries(
    TRASH_HANDS.map((label) => [
      label,
      handRows.find((hand) => hand.hand === label)?.open_shove_frequency ??
        null,
    ])
  );
  const strengthScores = handRows.map((hand) => hand.structural_strength_score);
  const continueFrequencies = handRows.map((hand) => hand.continue_frequency);
  const quartileSize = Math.ceil(EXPECTED_HAND_CLASSES * 0.25);
  const meanContinue = (hands) =>
    hands.every((hand) => finiteNumber(hand.continue_frequency))
      ? hands.reduce((sum, hand) => sum + hand.continue_frequency, 0) /
        hands.length
      : null;
  const topQuartileContinue = meanContinue(handRows.slice(0, quartileSize));
  const bottomQuartileContinue = meanContinue(handRows.slice(-quartileSize));

  return {
    effective_stack_bb: finiteNumber(effectiveStackBb)
      ? effectiveStackBb
      : null,
    applicable_to: finiteNumber(effectiveStackBb)
      ? `unopened heads-up button/small-blind at ${effectiveStackBb}bb`
      : 'unopened heads-up button/small-blind at unknown depth',
    action_detection: {
      fold_action_present: hasFoldAction,
      open_shove_actions: shoveActions,
    },
    aggregate_open_shove_frequency: aggregateOpenShove,
    premium_hand_fold_frequencies: premiumFoldFrequencies,
    maximum_premium_hand_fold_frequency: maximumFinite(
      Object.values(premiumFoldFrequencies)
    ),
    trash_hand_open_shove_frequencies: trashShoveFrequencies,
    maximum_trash_hand_open_shove_frequency: maximumFinite(
      Object.values(trashShoveFrequencies)
    ),
    hand_strength_continue_rank_order: {
      score: 'chen_style_structural_score_v1',
      response: 'one_minus_fold_frequency',
      hand_classes: handRows.length,
      spearman_correlation:
        continueFrequencies.every(finiteNumber) && shoveActions.length > 0
          ? spearmanCorrelation(strengthScores, continueFrequencies)
          : null,
      quartile_hand_classes: quartileSize,
      top_quartile_mean_continue_frequency: topQuartileContinue,
      bottom_quartile_mean_continue_frequency: bottomQuartileContinue,
      top_minus_bottom_continue_frequency:
        finiteNumber(topQuartileContinue) &&
        finiteNumber(bottomQuartileContinue)
          ? topQuartileContinue - bottomQuartileContinue
          : null,
    },
  };
}

function quantile(sortedValues, probability) {
  if (sortedValues.length === 0) return null;
  if (sortedValues.length === 1) return sortedValues[0];
  const position = (sortedValues.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sortedValues[lower];
  const fraction = position - lower;
  return sortedValues[lower] * (1 - fraction) + sortedValues[upper] * fraction;
}

function distributionSummary(values) {
  const sorted = [...values].sort((left, right) => left - right);
  const total = sorted.reduce((sum, value) => sum + value, 0);
  return {
    min: sorted[0] ?? null,
    median: quantile(sorted, 0.5),
    p95: quantile(sorted, 0.95),
    max: sorted.at(-1) ?? null,
    mean: sorted.length > 0 ? total / sorted.length : null,
  };
}

function optionalDistributionSummary(values) {
  return values.every(finiteNumber)
    ? distributionSummary(values)
    : {
        min: null,
        median: null,
        p95: null,
        max: null,
        mean: null,
      };
}

function primaryAction(probabilities, actions) {
  return [...actions].sort().reduce((best, action) => {
    if (best === null || probabilities[action] > probabilities[best]) {
      return action;
    }
    return best;
  }, null);
}

function extractRoot(file, artifact) {
  if (!artifact || typeof artifact !== 'object' || Array.isArray(artifact)) {
    throw new ArtifactComparisonError(
      'INVALID_ARTIFACT',
      `${file}: artifact root must be an object`
    );
  }
  if (!Array.isArray(artifact.strategies)) {
    throw new ArtifactComparisonError(
      'INVALID_ARTIFACT',
      `${file}: strategies must be an array`
    );
  }

  const entries = artifact.strategies.filter(
    (strategy) =>
      strategy?.street === 'preflop' &&
      strategy.actor === 'button_small_blind' &&
      Array.isArray(strategy.public_history) &&
      strategy.public_history.length === 1 &&
      typeof strategy.public_history[0] === 'string' &&
      strategy.public_history[0].startsWith('blinds:')
  );
  const hands = new Map();
  let actions = null;
  let maximumProbabilitySumError = 0;

  for (const entry of entries) {
    const trajectory = entry.hand_bucket_trajectory;
    const bucket = Array.isArray(trajectory) ? trajectory.at(-1) : undefined;
    const label =
      typeof bucket === 'string' && bucket.startsWith('preflop:')
        ? bucket.slice('preflop:'.length)
        : null;
    if (!label || !CANONICAL_HAND_CLASSES.has(label)) {
      throw new ArtifactComparisonError(
        'INVALID_ROOT_STRATEGY',
        `${file}: invalid root hand bucket ${String(bucket)}`
      );
    }
    if (hands.has(label)) {
      throw new ArtifactComparisonError(
        'INVALID_ROOT_STRATEGY',
        `${file}: duplicate root strategy for ${label}`
      );
    }
    if (!Array.isArray(entry.actions) || entry.actions.length === 0) {
      throw new ArtifactComparisonError(
        'INVALID_ROOT_STRATEGY',
        `${file}: ${label} has no actions`
      );
    }

    const probabilities = {};
    for (const action of entry.actions) {
      if (
        typeof action?.action !== 'string' ||
        Object.hasOwn(probabilities, action.action) ||
        !finiteNumber(action.probability) ||
        action.probability < 0 ||
        action.probability > 1
      ) {
        throw new ArtifactComparisonError(
          'INVALID_ROOT_STRATEGY',
          `${file}: invalid action distribution for ${label}`
        );
      }
      probabilities[action.action] = action.probability;
    }

    const entryActions = Object.keys(probabilities).sort();
    if (actions === null) {
      actions = entryActions;
    } else if (stableStringify(actions) !== stableStringify(entryActions)) {
      throw new ArtifactComparisonError(
        'INCOMPATIBLE_ACTIONS',
        `${file}: root action set differs between hand classes`,
        { label, expected: actions, actual: entryActions }
      );
    }

    const probabilitySum = Object.values(probabilities).reduce(
      (sum, value) => sum + value,
      0
    );
    const probabilitySumError = Math.abs(probabilitySum - 1);
    maximumProbabilitySumError = Math.max(
      maximumProbabilitySumError,
      probabilitySumError
    );
    if (probabilitySumError > PROBABILITY_TOLERANCE) {
      throw new ArtifactComparisonError(
        'INVALID_ROOT_STRATEGY',
        `${file}: ${label} probabilities sum to ${probabilitySum}`
      );
    }

    hands.set(label, {
      probabilities,
      combo_weight: comboWeight(label),
      primary_action: primaryAction(probabilities, entryActions),
      average_visits: entry.average_visits,
      regret_updates: entry.regret_updates,
      trained_average:
        typeof entry.trained_average === 'boolean'
          ? entry.trained_average
          : null,
    });
  }

  const missing = [...CANONICAL_HAND_CLASSES]
    .filter((label) => !hands.has(label))
    .sort();
  const extra = [...hands.keys()]
    .filter((label) => !CANONICAL_HAND_CLASSES.has(label))
    .sort();
  if (
    hands.size !== EXPECTED_HAND_CLASSES ||
    missing.length > 0 ||
    extra.length > 0
  ) {
    throw new ArtifactComparisonError(
      'INCOMPLETE_ROOT_STRATEGY',
      `${file}: expected ${EXPECTED_HAND_CLASSES} root hand classes, found ${hands.size}`,
      { missing, extra }
    );
  }

  const comboCount = [...hands.values()].reduce(
    (sum, hand) => sum + hand.combo_weight,
    0
  );
  if (comboCount !== EXPECTED_COMBOS) {
    throw new ArtifactComparisonError(
      'INCOMPLETE_ROOT_STRATEGY',
      `${file}: expected ${EXPECTED_COMBOS} weighted combos, found ${comboCount}`
    );
  }

  const aggregateFrequencies = Object.fromEntries(
    actions.map((action) => [
      action,
      [...hands.values()].reduce(
        (sum, hand) => sum + hand.combo_weight * hand.probabilities[action],
        0
      ) / comboCount,
    ])
  );

  return {
    actions,
    hands,
    summary: {
      hand_classes: hands.size,
      weighted_combos: comboCount,
      trained_average_hand_classes: [...hands.values()].every(
        (hand) => typeof hand.trained_average === 'boolean'
      )
        ? [...hands.values()].filter((hand) => hand.trained_average).length
        : null,
      maximum_probability_sum_error: maximumProbabilitySumError,
      average_visits: optionalDistributionSummary(
        [...hands.values()].map((hand) => hand.average_visits)
      ),
      regret_updates: optionalDistributionSummary(
        [...hands.values()].map((hand) => hand.regret_updates)
      ),
      combo_weighted_aggregate_frequencies: aggregateFrequencies,
    },
  };
}

function ensureCompatibility(inputs) {
  const first = inputs[0];
  const baseline = {
    schema_version: first.artifact.schema_version,
    model: first.artifact.model,
    approximate: first.artifact.approximate,
    config: normalizedConfig(first.artifact.config),
  };

  for (const input of inputs.slice(1)) {
    const candidate = {
      schema_version: input.artifact.schema_version,
      model: input.artifact.model,
      approximate: input.artifact.approximate,
      config: normalizedConfig(input.artifact.config),
    };
    const differences = valueDifferences(baseline, candidate);
    if (differences.length > 0) {
      throw new ArtifactComparisonError(
        'INCOMPATIBLE_ARTIFACTS',
        `${first.file} and ${input.file} use incompatible models or configs`,
        { differences }
      );
    }
  }

  return {
    schema_version: baseline.schema_version,
    model: baseline.model,
    approximate: baseline.approximate,
    normalized_config_sha256: createHash('sha256')
      .update(stableStringify(baseline.config))
      .digest('hex'),
    ignored_config_paths: [...IGNORED_CONFIG_KEYS]
      .sort()
      .map((key) => `config.${key}`),
  };
}

function coverageSummary(artifact, root) {
  const metrics = artifact.metrics ?? {};
  const heldOut = metrics.held_out ?? {};
  const informationSets = metrics.information_sets;
  const trainedInformationSets = metrics.trained_information_sets;
  const exportedInformationSets = metrics.exported_information_sets;
  return {
    root: root.summary,
    training: {
      requested_iterations: metrics.requested_iterations,
      training_iterations: metrics.training_iterations,
      stopped_early: metrics.stopped_early,
      stop_reason: metrics.stop_reason,
      sampled_deals: metrics.sampled_deals,
      terminal_evaluations: metrics.terminal_evaluations,
    },
    information_sets: {
      total: informationSets,
      preflop: metrics.preflop_information_sets,
      postflop: metrics.postflop_information_sets,
      trained: trainedInformationSets,
      trained_fraction:
        finiteNumber(trainedInformationSets) &&
        finiteNumber(informationSets) &&
        informationSets > 0
          ? trainedInformationSets / informationSets
          : null,
      exported: exportedInformationSets,
      exported_fraction:
        finiteNumber(exportedInformationSets) &&
        finiteNumber(informationSets) &&
        informationSets > 0
          ? exportedInformationSets / informationSets
          : null,
    },
    held_out: {
      deals: heldOut.deals,
      button_mean_net_bb: heldOut.button_mean_net_bb,
      button_net_standard_error_bb: heldOut.button_net_standard_error_bb,
      fold_terminal_fraction: heldOut.fold_terminal_fraction,
      showdown_terminal_fraction: heldOut.showdown_terminal_fraction,
      unknown_information_set_fraction:
        heldOut.unknown_information_set_fraction,
      untrained_information_set_fraction:
        heldOut.untrained_information_set_fraction,
    },
  };
}

function rootLocalDeviationSummary(artifact) {
  const audit = artifact.metrics?.root_local_deviation;
  if (!audit || typeof audit !== 'object' || Array.isArray(audit)) {
    return {
      available: false,
      kind: null,
      configured_samples_per_class: null,
      minimum_action_samples_per_class: null,
      hand_classes: null,
      canonical_hand_class_coverage: false,
      trained_root_hand_classes: null,
      trained_root_combo_fraction: null,
      fully_trained_root: false,
      continuation_coverage: {
        decisions: null,
        unknown_information_set_fraction: null,
        untrained_information_set_fraction: null,
      },
      aggregate_chosen_average_ev_bb: null,
      aggregate_best_action_ev_bb: null,
      aggregate_local_deviation_gain_bb: null,
      aggregate_local_deviation_gain_standard_error_bb: null,
      aggregate_local_deviation_gain_99pct_lower_bound_bb: null,
    };
  }

  const classes = Array.isArray(audit.classes) ? audit.classes : [];
  const labels = classes.map((entry) => entry?.hand_class);
  const uniqueLabels = new Set(labels);
  const canonicalCoverage =
    classes.length === EXPECTED_HAND_CLASSES &&
    uniqueLabels.size === EXPECTED_HAND_CLASSES &&
    [...CANONICAL_HAND_CLASSES].every((label) => uniqueLabels.has(label));
  const allActionSamples = classes.flatMap((entry) =>
    Array.isArray(entry?.action_values)
      ? entry.action_values.map((action) => action?.samples)
      : [null]
  );
  const completeActionSamples =
    classes.length === EXPECTED_HAND_CLASSES &&
    allActionSamples.length > 0 &&
    allActionSamples.every(finiteNumber);
  const minimumActionSamples = completeActionSamples
    ? Math.min(...allActionSamples)
    : null;
  const trainedRootHandClasses = classes.filter(
    (entry) => entry?.root_policy_trained === true
  ).length;
  const trainedRootComboFraction = audit.trained_root_combo_fraction;
  const fullyTrainedRoot =
    canonicalCoverage &&
    trainedRootHandClasses === EXPECTED_HAND_CLASSES &&
    finiteNumber(trainedRootComboFraction) &&
    trainedRootComboFraction >= 1 - PROBABILITY_TOLERANCE;
  const continuationCoverage = audit.continuation_coverage ?? {};

  return {
    available: true,
    kind: typeof audit.kind === 'string' ? audit.kind : null,
    configured_samples_per_class: finiteNumber(audit.samples_per_class)
      ? audit.samples_per_class
      : null,
    minimum_action_samples_per_class: minimumActionSamples,
    hand_classes: classes.length,
    canonical_hand_class_coverage: canonicalCoverage,
    trained_root_hand_classes: trainedRootHandClasses,
    trained_root_combo_fraction: finiteNumber(trainedRootComboFraction)
      ? trainedRootComboFraction
      : null,
    fully_trained_root: fullyTrainedRoot,
    continuation_coverage: {
      decisions: finiteNumber(continuationCoverage.decisions)
        ? continuationCoverage.decisions
        : null,
      unknown_information_set_fraction: finiteNumber(
        continuationCoverage.unknown_information_set_fraction
      )
        ? continuationCoverage.unknown_information_set_fraction
        : null,
      untrained_information_set_fraction: finiteNumber(
        continuationCoverage.untrained_information_set_fraction
      )
        ? continuationCoverage.untrained_information_set_fraction
        : null,
    },
    aggregate_chosen_average_ev_bb: finiteNumber(
      audit.aggregate_chosen_average_ev_bb
    )
      ? audit.aggregate_chosen_average_ev_bb
      : null,
    aggregate_best_action_ev_bb: finiteNumber(audit.aggregate_best_action_ev_bb)
      ? audit.aggregate_best_action_ev_bb
      : null,
    aggregate_local_deviation_gain_bb: finiteNumber(
      audit.aggregate_local_deviation_gain_bb
    )
      ? audit.aggregate_local_deviation_gain_bb
      : null,
    aggregate_local_deviation_gain_standard_error_bb: finiteNumber(
      audit.aggregate_local_deviation_gain_standard_error_bb
    )
      ? audit.aggregate_local_deviation_gain_standard_error_bb
      : null,
    aggregate_local_deviation_gain_99pct_lower_bound_bb: finiteNumber(
      audit.aggregate_local_deviation_gain_99pct_lower_bound_bb
    )
      ? audit.aggregate_local_deviation_gain_99pct_lower_bound_bb
      : null,
  };
}

function pairComparison(left, right) {
  if (
    stableStringify(left.root.actions) !== stableStringify(right.root.actions)
  ) {
    throw new ArtifactComparisonError(
      'INCOMPATIBLE_ACTIONS',
      `${left.file} and ${right.file} have different root action sets`,
      { left: left.root.actions, right: right.root.actions }
    );
  }

  const actions = left.root.actions;
  const actionAbsoluteErrors = Object.fromEntries(
    actions.map((action) => [action, 0])
  );
  let primaryClassMatches = 0;
  let primaryComboMatches = 0;
  const totalVariations = [];

  for (const label of [...CANONICAL_HAND_CLASSES].sort()) {
    const leftHand = left.root.hands.get(label);
    const rightHand = right.root.hands.get(label);
    let absoluteDifferenceSum = 0;
    for (const action of actions) {
      const difference = Math.abs(
        leftHand.probabilities[action] - rightHand.probabilities[action]
      );
      actionAbsoluteErrors[action] += leftHand.combo_weight * difference;
      absoluteDifferenceSum += difference;
    }
    const matches = leftHand.primary_action === rightHand.primary_action;
    if (matches) {
      primaryClassMatches += 1;
      primaryComboMatches += leftHand.combo_weight;
    }
    totalVariations.push({
      hand: label,
      combo_weight: leftHand.combo_weight,
      total_variation: absoluteDifferenceSum / 2,
      left_primary_action: leftHand.primary_action,
      right_primary_action: rightHand.primary_action,
    });
  }

  const aggregateDeltas = Object.fromEntries(
    actions.map((action) => {
      const leftFrequency =
        left.root.summary.combo_weighted_aggregate_frequencies[action];
      const rightFrequency =
        right.root.summary.combo_weighted_aggregate_frequencies[action];
      return [
        action,
        {
          left: leftFrequency,
          right: rightFrequency,
          right_minus_left: rightFrequency - leftFrequency,
          absolute_delta: Math.abs(rightFrequency - leftFrequency),
        },
      ];
    })
  );
  const perActionMae = Object.fromEntries(
    actions.map((action) => [
      action,
      actionAbsoluteErrors[action] / EXPECTED_COMBOS,
    ])
  );
  const sortedTotalVariations = totalVariations
    .map((hand) => hand.total_variation)
    .sort((leftValue, rightValue) => leftValue - rightValue);
  const worstHands = [...totalVariations]
    .sort(
      (leftHand, rightHand) =>
        rightHand.total_variation - leftHand.total_variation ||
        leftHand.hand.localeCompare(rightHand.hand)
    )
    .slice(0, 10);

  const leftHeldOut = left.artifact.metrics?.held_out ?? {};
  const rightHeldOut = right.artifact.metrics?.held_out ?? {};
  const meanDelta =
    finiteNumber(leftHeldOut.button_mean_net_bb) &&
    finiteNumber(rightHeldOut.button_mean_net_bb)
      ? rightHeldOut.button_mean_net_bb - leftHeldOut.button_mean_net_bb
      : null;
  const combinedStandardError =
    finiteNumber(leftHeldOut.button_net_standard_error_bb) &&
    finiteNumber(rightHeldOut.button_net_standard_error_bb)
      ? Math.hypot(
          leftHeldOut.button_net_standard_error_bb,
          rightHeldOut.button_net_standard_error_bb
        )
      : null;
  const zScore =
    finiteNumber(meanDelta) &&
    finiteNumber(combinedStandardError) &&
    combinedStandardError > 0
      ? Math.abs(meanDelta) / combinedStandardError
      : null;

  return {
    left_artifact_id: left.artifact.artifact_id,
    right_artifact_id: right.artifact.artifact_id,
    combo_weighted_aggregate_frequencies: aggregateDeltas,
    combo_weighted_per_action_mae: perActionMae,
    per_hand_total_variation: {
      median: quantile(sortedTotalVariations, 0.5),
      p95: quantile(sortedTotalVariations, 0.95),
      max: sortedTotalVariations.at(-1),
      mean:
        sortedTotalVariations.reduce((sum, value) => sum + value, 0) /
        sortedTotalVariations.length,
      worst_hands: worstHands,
    },
    primary_action_agreement: {
      hand_class_fraction: primaryClassMatches / EXPECTED_HAND_CLASSES,
      combo_weighted_fraction: primaryComboMatches / EXPECTED_COMBOS,
    },
    held_out_button_ev_consistency: {
      right_minus_left_bb: meanDelta,
      combined_standard_error_bb: combinedStandardError,
      absolute_z_score: zScore,
    },
  };
}

function maximumFinite(values) {
  return values.every(finiteNumber) && values.length > 0
    ? Math.max(...values)
    : null;
}

function minimumFinite(values) {
  return values.every(finiteNumber) && values.length > 0
    ? Math.min(...values)
    : null;
}

function gate(name, operator, threshold, observed, passed, scope) {
  return { name, scope, operator, threshold, observed, passed };
}

function buildGates(artifacts, comparisons, thresholds) {
  const completedTraining = artifacts.every(
    ({ artifact }) =>
      artifact.metrics?.stopped_early === false &&
      finiteNumber(artifact.metrics?.requested_iterations) &&
      artifact.metrics?.training_iterations ===
        artifact.metrics?.requested_iterations
  );
  const rootCoverage = artifacts.every(
    ({ root }) =>
      root.summary.hand_classes === EXPECTED_HAND_CLASSES &&
      root.summary.weighted_combos === EXPECTED_COMBOS
  );
  const allRootAveragesTrained = artifacts.every(
    ({ root }) =>
      root.summary.trained_average_hand_classes === EXPECTED_HAND_CLASSES
  );
  const rootMinimumVisits = minimumFinite(
    artifacts.map(({ root }) => root.summary.average_visits.min)
  );
  const heldOutMinimumDeals = minimumFinite(
    artifacts.map(({ artifact }) => artifact.metrics?.held_out?.deals)
  );
  const heldOutMaximumUnknown = maximumFinite(
    artifacts.map(
      ({ artifact }) =>
        artifact.metrics?.held_out?.unknown_information_set_fraction
    )
  );
  const heldOutMaximumUntrained = maximumFinite(
    artifacts.map(
      ({ artifact }) =>
        artifact.metrics?.held_out?.untrained_information_set_fraction
    )
  );
  const maximumAggregateDelta = maximumFinite(
    comparisons.flatMap((comparison) =>
      Object.values(comparison.combo_weighted_aggregate_frequencies).map(
        (action) => action.absolute_delta
      )
    )
  );
  const maximumActionMae = maximumFinite(
    comparisons.flatMap((comparison) =>
      Object.values(comparison.combo_weighted_per_action_mae)
    )
  );
  const maximumTvMedian = maximumFinite(
    comparisons.map((comparison) => comparison.per_hand_total_variation.median)
  );
  const maximumTvP95 = maximumFinite(
    comparisons.map((comparison) => comparison.per_hand_total_variation.p95)
  );
  const maximumTvMax = maximumFinite(
    comparisons.map((comparison) => comparison.per_hand_total_variation.max)
  );
  const minimumPrimaryAgreement = minimumFinite(
    comparisons.flatMap((comparison) => [
      comparison.primary_action_agreement.hand_class_fraction,
      comparison.primary_action_agreement.combo_weighted_fraction,
    ])
  );
  const maximumHeldOutEvZScore = maximumFinite(
    comparisons.map(
      (comparison) => comparison.held_out_button_ev_consistency.absolute_z_score
    )
  );
  const uniqueSeeds = new Set(
    artifacts.map(({ artifact }) => artifact.config?.seed)
  ).size;
  const observedEffectiveStacks = [
    ...new Set(
      artifacts.map(({ artifact }) => artifact.config?.effective_stack_bb)
    ),
  ];
  const domainSanityDepthSupported = observedEffectiveStacks.every(
    (depth) =>
      finiteNumber(depth) && SUPPORTED_EFFECTIVE_STACKS_BB.includes(depth)
  );
  const aggregateOpenShoveGateApplicable = observedEffectiveStacks.every(
    (depth) => depth === 100
  );
  const requiredDomainActionsPresent = artifacts.every(
    ({ domainSanity }) =>
      domainSanity.action_detection.fold_action_present &&
      domainSanity.action_detection.open_shove_actions.length > 0
  );
  const maximumAggregateOpenShove = maximumFinite(
    artifacts.map(
      ({ domainSanity }) => domainSanity.aggregate_open_shove_frequency
    )
  );
  const maximumPremiumFold = maximumFinite(
    artifacts.map(
      ({ domainSanity }) => domainSanity.maximum_premium_hand_fold_frequency
    )
  );
  const maximumTrashShove = maximumFinite(
    artifacts.map(
      ({ domainSanity }) => domainSanity.maximum_trash_hand_open_shove_frequency
    )
  );
  const minimumStrengthContinueSpearman = minimumFinite(
    artifacts.map(
      ({ domainSanity }) =>
        domainSanity.hand_strength_continue_rank_order.spearman_correlation
    )
  );
  const minimumStrengthQuartileContinueGap = minimumFinite(
    artifacts.map(
      ({ domainSanity }) =>
        domainSanity.hand_strength_continue_rank_order
          .top_minus_bottom_continue_frequency
    )
  );
  const rootLocalDeviationAuditsAvailable = artifacts.every(
    ({ rootLocalDeviation }) => rootLocalDeviation.available
  );
  const minimumRootLocalDeviationSamples = minimumFinite(
    artifacts.flatMap(({ rootLocalDeviation }) => [
      rootLocalDeviation.configured_samples_per_class,
      rootLocalDeviation.minimum_action_samples_per_class,
    ])
  );
  const allRootLocalDeviationRootsTrained = artifacts.every(
    ({ rootLocalDeviation }) => rootLocalDeviation.fully_trained_root
  );
  const maximumRootLocalDeviationUnknown = maximumFinite(
    artifacts.map(
      ({ rootLocalDeviation }) =>
        rootLocalDeviation.continuation_coverage
          .unknown_information_set_fraction
    )
  );
  const maximumRootLocalDeviationUntrained = maximumFinite(
    artifacts.map(
      ({ rootLocalDeviation }) =>
        rootLocalDeviation.continuation_coverage
          .untrained_information_set_fraction
    )
  );
  const maximumAggregateRootLocalDeviationGain = maximumFinite(
    artifacts.map(
      ({ rootLocalDeviation }) =>
        rootLocalDeviation.aggregate_local_deviation_gain_bb
    )
  );
  const maximumAggregateRootLocalDeviationLowerBound = maximumFinite(
    artifacts.map(
      ({ rootLocalDeviation }) =>
        rootLocalDeviation.aggregate_local_deviation_gain_99pct_lower_bound_bb
    )
  );

  return [
    gate(
      'distinct_training_seeds',
      '==',
      artifacts.length,
      uniqueSeeds,
      uniqueSeeds === artifacts.length,
      'all_artifacts'
    ),
    gate(
      'completed_requested_training',
      '==',
      true,
      completedTraining,
      completedTraining,
      'all_artifacts'
    ),
    gate(
      'complete_root_coverage',
      '==',
      true,
      rootCoverage,
      rootCoverage,
      'all_artifacts'
    ),
    gate(
      'all_root_hand_classes_have_trained_averages',
      '==',
      true,
      allRootAveragesTrained,
      allRootAveragesTrained,
      'all_artifacts'
    ),
    gate(
      'minimum_root_average_visits',
      '>=',
      thresholds.minimum_root_average_visits,
      rootMinimumVisits,
      finiteNumber(rootMinimumVisits) &&
        rootMinimumVisits >= thresholds.minimum_root_average_visits,
      'all_artifacts'
    ),
    gate(
      'minimum_held_out_deals',
      '>=',
      thresholds.minimum_held_out_deals,
      heldOutMinimumDeals,
      finiteNumber(heldOutMinimumDeals) &&
        heldOutMinimumDeals >= thresholds.minimum_held_out_deals,
      'all_artifacts'
    ),
    gate(
      'maximum_held_out_unknown_information_set_fraction',
      '<=',
      thresholds.maximum_held_out_unknown_information_set_fraction,
      heldOutMaximumUnknown,
      finiteNumber(heldOutMaximumUnknown) &&
        heldOutMaximumUnknown <=
          thresholds.maximum_held_out_unknown_information_set_fraction,
      'all_artifacts'
    ),
    gate(
      'maximum_held_out_untrained_information_set_fraction',
      '<=',
      thresholds.maximum_held_out_untrained_information_set_fraction,
      heldOutMaximumUntrained,
      finiteNumber(heldOutMaximumUntrained) &&
        heldOutMaximumUntrained <=
          thresholds.maximum_held_out_untrained_information_set_fraction,
      'all_artifacts'
    ),
    gate(
      'root_local_deviation_audit_available',
      '==',
      true,
      rootLocalDeviationAuditsAvailable,
      rootLocalDeviationAuditsAvailable,
      'all_artifacts'
    ),
    gate(
      'minimum_root_local_deviation_samples_per_class',
      '>=',
      thresholds.minimum_root_local_deviation_samples_per_class,
      minimumRootLocalDeviationSamples,
      finiteNumber(minimumRootLocalDeviationSamples) &&
        minimumRootLocalDeviationSamples >=
          thresholds.minimum_root_local_deviation_samples_per_class,
      'all_artifacts'
    ),
    gate(
      'full_trained_root_in_local_deviation_audit',
      '==',
      true,
      allRootLocalDeviationRootsTrained,
      allRootLocalDeviationRootsTrained,
      'all_artifacts'
    ),
    gate(
      'maximum_root_local_deviation_unknown_information_set_fraction',
      '<=',
      thresholds.maximum_root_local_deviation_unknown_information_set_fraction,
      maximumRootLocalDeviationUnknown,
      finiteNumber(maximumRootLocalDeviationUnknown) &&
        maximumRootLocalDeviationUnknown <=
          thresholds.maximum_root_local_deviation_unknown_information_set_fraction,
      'all_artifacts'
    ),
    gate(
      'maximum_root_local_deviation_untrained_information_set_fraction',
      '<=',
      thresholds.maximum_root_local_deviation_untrained_information_set_fraction,
      maximumRootLocalDeviationUntrained,
      finiteNumber(maximumRootLocalDeviationUntrained) &&
        maximumRootLocalDeviationUntrained <=
          thresholds.maximum_root_local_deviation_untrained_information_set_fraction,
      'all_artifacts'
    ),
    gate(
      'maximum_aggregate_root_local_deviation_gain_bb',
      '<=',
      thresholds.maximum_aggregate_root_local_deviation_gain_bb,
      maximumAggregateRootLocalDeviationGain,
      finiteNumber(maximumAggregateRootLocalDeviationGain) &&
        maximumAggregateRootLocalDeviationGain <=
          thresholds.maximum_aggregate_root_local_deviation_gain_bb,
      'all_artifacts'
    ),
    gate(
      'maximum_aggregate_root_local_deviation_99pct_lower_bound_bb',
      '<=',
      thresholds.maximum_aggregate_root_local_deviation_99pct_lower_bound_bb,
      maximumAggregateRootLocalDeviationLowerBound,
      finiteNumber(maximumAggregateRootLocalDeviationLowerBound) &&
        maximumAggregateRootLocalDeviationLowerBound <=
          thresholds.maximum_aggregate_root_local_deviation_99pct_lower_bound_bb,
      'all_artifacts'
    ),
    gate(
      'maximum_aggregate_action_frequency_delta',
      '<=',
      thresholds.maximum_aggregate_action_frequency_delta,
      maximumAggregateDelta,
      finiteNumber(maximumAggregateDelta) &&
        maximumAggregateDelta <=
          thresholds.maximum_aggregate_action_frequency_delta,
      'all_pairs'
    ),
    gate(
      'maximum_combo_weighted_per_action_mae',
      '<=',
      thresholds.maximum_per_action_mae,
      maximumActionMae,
      finiteNumber(maximumActionMae) &&
        maximumActionMae <= thresholds.maximum_per_action_mae,
      'all_pairs'
    ),
    gate(
      'maximum_hand_total_variation_median',
      '<=',
      thresholds.maximum_hand_total_variation_median,
      maximumTvMedian,
      finiteNumber(maximumTvMedian) &&
        maximumTvMedian <= thresholds.maximum_hand_total_variation_median,
      'all_pairs'
    ),
    gate(
      'maximum_hand_total_variation_p95',
      '<=',
      thresholds.maximum_hand_total_variation_p95,
      maximumTvP95,
      finiteNumber(maximumTvP95) &&
        maximumTvP95 <= thresholds.maximum_hand_total_variation_p95,
      'all_pairs'
    ),
    gate(
      'maximum_hand_total_variation_max',
      '<=',
      thresholds.maximum_hand_total_variation_max,
      maximumTvMax,
      finiteNumber(maximumTvMax) &&
        maximumTvMax <= thresholds.maximum_hand_total_variation_max,
      'all_pairs'
    ),
    gate(
      'minimum_primary_action_agreement',
      '>=',
      thresholds.minimum_primary_action_agreement,
      minimumPrimaryAgreement,
      finiteNumber(minimumPrimaryAgreement) &&
        minimumPrimaryAgreement >= thresholds.minimum_primary_action_agreement,
      'all_pairs'
    ),
    gate(
      'maximum_held_out_button_ev_z_score',
      '<=',
      thresholds.maximum_held_out_button_ev_z_score,
      maximumHeldOutEvZScore,
      finiteNumber(maximumHeldOutEvZScore) &&
        maximumHeldOutEvZScore <= thresholds.maximum_held_out_button_ev_z_score,
      'all_pairs'
    ),
    gate(
      'poker_domain_sanity_depth_supported',
      'in',
      SUPPORTED_EFFECTIVE_STACKS_BB,
      observedEffectiveStacks,
      domainSanityDepthSupported,
      'all_artifacts'
    ),
    gate(
      'poker_domain_sanity_actions_present',
      '==',
      true,
      requiredDomainActionsPresent,
      requiredDomainActionsPresent,
      'all_artifacts'
    ),
    {
      ...gate(
        'maximum_aggregate_open_shove_frequency',
        '<=',
        thresholds.maximum_aggregate_open_shove_frequency,
        maximumAggregateOpenShove,
        !aggregateOpenShoveGateApplicable ||
          (finiteNumber(maximumAggregateOpenShove) &&
            maximumAggregateOpenShove <=
              thresholds.maximum_aggregate_open_shove_frequency),
        'all_artifacts'
      ),
      applicable: aggregateOpenShoveGateApplicable,
      applicability: '100bb only',
    },
    gate(
      'maximum_premium_hand_fold_frequency',
      '<=',
      thresholds.maximum_premium_hand_fold_frequency,
      maximumPremiumFold,
      finiteNumber(maximumPremiumFold) &&
        maximumPremiumFold <= thresholds.maximum_premium_hand_fold_frequency,
      'all_artifacts'
    ),
    gate(
      'maximum_trash_hand_open_shove_frequency',
      '<=',
      thresholds.maximum_trash_hand_open_shove_frequency,
      maximumTrashShove,
      finiteNumber(maximumTrashShove) &&
        maximumTrashShove <= thresholds.maximum_trash_hand_open_shove_frequency,
      'all_artifacts'
    ),
    gate(
      'minimum_hand_strength_continue_spearman',
      '>=',
      thresholds.minimum_hand_strength_continue_spearman,
      minimumStrengthContinueSpearman,
      finiteNumber(minimumStrengthContinueSpearman) &&
        minimumStrengthContinueSpearman >=
          thresholds.minimum_hand_strength_continue_spearman,
      'all_artifacts'
    ),
    gate(
      'minimum_strength_quartile_continue_gap',
      '>=',
      thresholds.minimum_strength_quartile_continue_gap,
      minimumStrengthQuartileContinueGap,
      finiteNumber(minimumStrengthQuartileContinueGap) &&
        minimumStrengthQuartileContinueGap >=
          thresholds.minimum_strength_quartile_continue_gap,
      'all_artifacts'
    ),
  ];
}

export function compareBlueprintArtifacts(
  inputs,
  thresholds = DEFAULT_THRESHOLDS
) {
  if (!Array.isArray(inputs) || inputs.length < 2) {
    throw new ArtifactComparisonError(
      'INVALID_ARGUMENTS',
      'at least two blueprint artifacts are required'
    );
  }
  const sortedInputs = [...inputs].sort((left, right) =>
    left.file.localeCompare(right.file)
  );
  const compatibility = ensureCompatibility(sortedInputs);
  const artifacts = sortedInputs.map((input) => {
    const root = extractRoot(input.file, input.artifact);
    return {
      ...input,
      root,
      domainSanity: domainSanitySummary(
        root,
        input.artifact.config?.effective_stack_bb
      ),
      rootLocalDeviation: rootLocalDeviationSummary(input.artifact),
    };
  });
  const baselineActions = artifacts[0].root.actions;
  for (const artifact of artifacts.slice(1)) {
    if (
      stableStringify(baselineActions) !==
      stableStringify(artifact.root.actions)
    ) {
      throw new ArtifactComparisonError(
        'INCOMPATIBLE_ACTIONS',
        `${artifacts[0].file} and ${artifact.file} have different root action sets`,
        { left: baselineActions, right: artifact.root.actions }
      );
    }
  }

  const comparisons = [];
  for (let left = 0; left < artifacts.length; left += 1) {
    for (let right = left + 1; right < artifacts.length; right += 1) {
      comparisons.push(pairComparison(artifacts[left], artifacts[right]));
    }
  }
  const gates = buildGates(artifacts, comparisons, thresholds);

  return {
    schema_version: 1,
    analyzer: ANALYZER,
    interpretation: [
      'Passing certifies cross-seed root-strategy stability under these gates, not Nash equilibrium or exploitability.',
      'Frequencies and action MAE are weighted by the 1,326 physical starting-hand combinations; hand total variation treats each of the 169 classes equally.',
      'Held-out button EV consistency uses the absolute difference divided by the combined standard error.',
      'The one-step root local-deviation audit tests profitable unilateral root-action changes against fixed continuation policies; it is a necessary local check, not full-game exploitability.',
      'Poker-domain gates catch coarse heads-up root pathologies at supported 20/50/100bb depths; the aggregate open-shove cap is 100bb-only and non-applicable gates are identified explicitly.',
      'The monotonicity check correlates a tie-aware Chen-style structural score with one minus fold frequency because legitimate strong-hand limps make raise frequency an unsuitable strength target.',
    ],
    compatibility,
    thresholds: { ...thresholds },
    threshold_rationale: {
      maximum_aggregate_action_frequency_delta:
        'The aggregate frequency of any root action may differ by at most three percentage points across independent seeds.',
      maximum_per_action_mae:
        'The combo-weighted hand-class MAE of any root action may differ by at most five percentage points across independent seeds.',
      minimum_primary_action_agreement:
        'At least 85 percent of the 169 hand classes must share the same highest-probability root action across independent seeds.',
      maximum_aggregate_open_shove_frequency:
        'At 100bb an unopened all-in should be exceptional; one percent still allows residual sampling noise and isolated mixes while rejecting a material shove range. This gate is explicitly non-applicable at 20bb and 50bb because no depth-specific threshold has been validated.',
      maximum_premium_hand_fold_frequency:
        'AA, KK, QQ, and AKs should almost never fold unopened; two percent allows sampling noise while catching strategically inverted roots.',
      maximum_trash_hand_open_shove_frequency:
        '72o and 32o may mix small opens or limps, but an all-in above two percent is treated as a coarse pathology at supported depths.',
      minimum_hand_strength_continue_spearman:
        'A 0.65 rank correlation requires a clear strength signal without demanding strict hand-by-hand monotonicity from a mixed strategy.',
      minimum_strength_quartile_continue_gap:
        'The strongest structural quartile must continue at least 30 percentage points more often than the weakest quartile.',
      minimum_root_local_deviation_samples_per_class:
        'Every root action in every hand class must have at least 256 samples; the gate uses the minimum actual action count as well as the configured count.',
      maximum_root_local_deviation_unknown_information_set_fraction:
        'At most five percent of continuation decisions may use an unknown information set so the local response is mostly evaluated against learned continuation play.',
      maximum_root_local_deviation_untrained_information_set_fraction:
        'At most 2.5 percent of continuation decisions may use an untrained information set.',
      maximum_aggregate_root_local_deviation_gain_bb:
        'A weighted one-step root action change may improve no more than 0.10bb on average.',
      maximum_aggregate_root_local_deviation_99pct_lower_bound_bb:
        'The one-sided 99 percent lower confidence bound may exceed zero slightly for sampling noise, but not 0.05bb.',
    },
    artifacts: artifacts.map(
      ({ file, artifact, root, domainSanity, rootLocalDeviation }) => ({
        file,
        artifact_id: artifact.artifact_id,
        seed: artifact.config?.seed,
        iterations: artifact.config?.iterations,
        coverage_and_held_out_metrics: coverageSummary(artifact, root),
        root_local_deviation_audit: rootLocalDeviation,
        poker_domain_sanity: domainSanity,
      })
    ),
    comparisons,
    acceptance_gates: gates,
    passed: gates.every((entry) => entry.passed),
  };
}

function parseArguments(argv) {
  const files = [];
  let requirePass = false;
  for (const argument of argv) {
    if (argument === '--require-pass') {
      requirePass = true;
    } else if (argument === '--help' || argument === '-h') {
      return { help: true, files, requirePass };
    } else if (argument.startsWith('-')) {
      throw new ArtifactComparisonError(
        'INVALID_ARGUMENTS',
        `unknown option ${argument}`
      );
    } else {
      files.push(argument);
    }
  }
  return { help: false, files, requirePass };
}

async function runCli(argv) {
  const options = parseArguments(argv);
  if (options.help) {
    process.stdout.write(
      'Usage: node scripts/compare-blueprint-artifacts.mjs [--require-pass] <artifact-a.json[.gz]> <artifact-b.json[.gz]> [artifact-c.json[.gz] ...]\n'
    );
    return 0;
  }
  if (options.files.length < 2) {
    throw new ArtifactComparisonError(
      'INVALID_ARGUMENTS',
      'at least two blueprint artifact paths are required'
    );
  }

  const inputs = await Promise.all(
    [...options.files].sort().map(async (file) => ({
      file: path
        .relative(process.cwd(), path.resolve(file))
        .split(path.sep)
        .join('/'),
      artifact: parseBlueprintArtifactBytes(file, await readFile(file)),
    }))
  );
  const report = compareBlueprintArtifacts(inputs);
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  return options.requirePass && !report.passed ? 1 : 0;
}

const isCli =
  process.argv[1] &&
  import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href;

if (isCli) {
  try {
    process.exitCode = await runCli(process.argv.slice(2));
  } catch (error) {
    const known =
      error instanceof ArtifactComparisonError
        ? error
        : new ArtifactComparisonError(
            'UNEXPECTED_ERROR',
            error instanceof Error ? error.message : String(error)
          );
    process.stdout.write(
      `${JSON.stringify(
        {
          schema_version: 1,
          analyzer: ANALYZER,
          error: {
            code: known.code,
            message: known.message,
            ...(known.details === undefined ? {} : { details: known.details }),
          },
        },
        null,
        2
      )}\n`
    );
    process.exitCode = 2;
  }
}
