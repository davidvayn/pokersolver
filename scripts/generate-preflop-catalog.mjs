import { createHash } from 'node:crypto';
import { readdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

const root = process.cwd();
const sourceDirectory = path.join(root, 'preflop-solver', 'artifacts');
const outputPath = path.join(root, 'data', 'preflop', 'solved-scenarios.json');
const frequencyTolerance = 1e-6;

function fail(file, message) {
  throw new Error(`${file}: ${message}`);
}

function frequency(value) {
  return Number.isFinite(value) && value >= 0 && value <= 1;
}

function finite(value) {
  return Number.isFinite(value);
}

function validateArtifact(file, artifact) {
  if (!artifact || typeof artifact !== 'object') fail(file, 'invalid root');
  if (artifact.schema_version !== 1) fail(file, 'unsupported schema');
  if (artifact.model !== 'heads-up-push-fold-monte-carlo-v1') {
    fail(file, 'unsupported model');
  }
  if (artifact.validation?.status !== 'approximate') {
    fail(file, `validation status ${artifact.validation?.status}`);
  }
  const checks = artifact.validation.checks;
  const requiredChecks = [
    'finite_metrics',
    'best_response_ordering',
    'strategy_probability_sums',
    'aces_shove_and_call_sanity',
    'exploitability_advisory',
  ];
  if (
    !Array.isArray(checks) ||
    requiredChecks.some(
      (name) => !checks.some((check) => check.name === name && check.passed)
    ) ||
    checks.some(
      (check) =>
        check.name !== 'exploitability_high_precision' && !check.passed
    )
  ) {
    fail(file, 'a required validation check failed');
  }
  if (
    !Number.isFinite(artifact.metrics?.exploitability_bb) ||
    artifact.metrics.exploitability_bb < 0 ||
    artifact.metrics.exploitability_bb > 0.01
  ) {
    fail(file, 'exploitability is outside the advisory gate');
  }
  if (artifact.metrics.compatible_deals !== 1326 * 1225) {
    fail(file, 'compatible deal count is incomplete');
  }

  const combos = artifact.strategies?.exact_combos;
  if (!Array.isArray(combos) || combos.length !== 1326) {
    fail(file, 'exact combo catalog is incomplete');
  }
  const keys = new Set();
  const classTotals = new Map();
  const ranks = '23456789TJQKA';
  for (const combo of combos) {
    const [first, second] = combo.cards ?? [];
    const high = Math.max(first, second);
    const low = Math.min(first, second);
    const expectedKey = (high * (high - 1)) / 2 + low;
    const pairs = [
      combo.small_blind &&
        [combo.small_blind.fold, combo.small_blind.shove],
      combo.big_blind_vs_shove &&
        [combo.big_blind_vs_shove.fold, combo.big_blind_vs_shove.call],
    ];
    if (
      !Number.isInteger(first) ||
      !Number.isInteger(second) ||
      first < 0 ||
      second < 0 ||
      first > 51 ||
      second > 51 ||
      first === second ||
      combo.combo_key !== expectedKey ||
      keys.has(combo.combo_key) ||
      pairs.some(
        (pair) =>
          !pair ||
          !frequency(pair[0]) ||
          !frequency(pair[1]) ||
          Math.abs(pair[0] + pair[1] - 1) > frequencyTolerance
      )
    ) {
      fail(file, `invalid exact combo ${combo.combo_key}`);
    }
    const firstRank = Math.floor(first / 4);
    const secondRank = Math.floor(second / 4);
    const highRank = Math.max(firstRank, secondRank);
    const lowRank = Math.min(firstRank, secondRank);
    const label =
      firstRank === secondRank
        ? ranks[firstRank] + ranks[firstRank]
        : `${ranks[highRank]}${ranks[lowRank]}${
            first % 4 === second % 4 ? 's' : 'o'
          }`;
    if (combo.label !== label) fail(file, `mismatched label ${combo.label}`);
    const total = classTotals.get(label) ?? {
      count: 0,
      shove: 0,
      call: 0,
    };
    total.count += 1;
    total.shove += combo.small_blind.shove;
    total.call += combo.big_blind_vs_shove.call;
    classTotals.set(label, total);
    keys.add(combo.combo_key);
  }

  const classes = artifact.strategies?.hand_classes;
  if (!Array.isArray(classes) || classes.length !== 169) {
    fail(file, 'hand-class catalog is incomplete');
  }
  const labels = new Set();
  for (const hand of classes) {
    const exact = classTotals.get(hand.label);
    const expectedCount = hand.label.length === 2 ? 6 : hand.label.endsWith('s') ? 4 : 12;
    if (
      typeof hand.label !== 'string' ||
      labels.has(hand.label) ||
      !exact ||
      exact.count !== expectedCount ||
      hand.combo_count !== expectedCount ||
      !frequency(hand.small_blind?.fold) ||
      !frequency(hand.small_blind?.shove) ||
      !frequency(hand.big_blind_vs_shove?.fold) ||
      !frequency(hand.big_blind_vs_shove?.call)
    ) {
      fail(file, `invalid hand class ${hand.label}`);
    }
    if (
      Math.abs(hand.small_blind.fold + hand.small_blind.shove - 1) >
        frequencyTolerance ||
      Math.abs(
        hand.big_blind_vs_shove.fold + hand.big_blind_vs_shove.call - 1
      ) > frequencyTolerance ||
      Math.abs(hand.small_blind.shove - exact.shove / exact.count) >
        frequencyTolerance ||
      Math.abs(
        hand.big_blind_vs_shove.call - exact.call / exact.count
      ) > frequencyTolerance
    ) {
      fail(file, `hand class does not match exact combos: ${hand.label}`);
    }
    labels.add(hand.label);
  }
  if (classTotals.size !== 169 || labels.size !== classTotals.size) {
    fail(file, 'canonical hand-class set is incomplete');
  }
}

function validateActionValues(file, values, artifact, sourceSha256) {
  if (!values || typeof values !== 'object') fail(file, 'invalid root');
  if (
    values.schema_version !== 1 ||
    values.model !== 'heads-up-push-fold-action-values-v1' ||
    values.source_artifact_id !== artifact.artifact_id ||
    values.source_config_hash !== artifact.config_hash ||
    values.source_artifact_sha256 !== sourceSha256 ||
    values.evaluation_seed !== artifact.metrics.evaluation_seed ||
    values.equity_samples !== artifact.config.equity_samples ||
    values.called_payoff_standard_error_upper_bound_bb !==
      artifact.metrics.called_payoff_standard_error_upper_bound_bb
  ) {
    fail(file, 'action-value provenance does not match the policy artifact');
  }
  const hands = values.hand_classes;
  if (!Array.isArray(hands) || hands.length !== 169) {
    fail(file, 'action-value hand-class catalog is incomplete');
  }
  const policyClasses = new Map(
    artifact.strategies.hand_classes.map((hand) => [hand.label, hand])
  );
  const labels = new Set();
  for (const hand of hands) {
    const policy = policyClasses.get(hand?.label);
    const sb = hand?.small_blind;
    const bb = hand?.big_blind_vs_shove;
    if (
      !policy ||
      labels.has(hand.label) ||
      hand.combo_count !== policy.combo_count ||
      !finite(sb?.fold_ev_bb) ||
      !finite(sb?.shove_ev_bb) ||
      sb.fold_ev_bb !== -artifact.config.small_blind_bb ||
      sb.fold_standard_error_bb !== 0 ||
      sb.shove_standard_error_upper_bound_bb !==
        values.called_payoff_standard_error_upper_bound_bb ||
      !finite(bb?.fold_ev_bb) ||
      !finite(bb?.call_ev_bb) ||
      bb.fold_ev_bb !== -artifact.config.big_blind_bb ||
      bb.fold_standard_error_bb !== 0 ||
      bb.call_standard_error_upper_bound_bb !==
        values.called_payoff_standard_error_upper_bound_bb
    ) {
      fail(file, `invalid action values for ${hand?.label}`);
    }
    labels.add(hand.label);
  }
  if (labels.size !== policyClasses.size) {
    fail(file, 'action-value hand-class set does not match the policy');
  }
}

const files = (await readdir(sourceDirectory))
  .filter((file) => /^hu-push-fold-\d+bb\.json$/.test(file))
  .sort(
    (first, second) =>
      Number(first.match(/\d+/)?.[0]) - Number(second.match(/\d+/)?.[0])
  );

const summaries = [];
for (const file of files) {
  const source = await readFile(path.join(sourceDirectory, file), 'utf8');
  const artifact = JSON.parse(source);
  validateArtifact(file, artifact);
  const sourceSha256 = createHash('sha256').update(source).digest('hex');
  const valuesFile = file.replace(/\.json$/, '-action-values.json');
  const valuesSource = await readFile(
    path.join(sourceDirectory, valuesFile),
    'utf8'
  );
  const actionValues = JSON.parse(valuesSource);
  validateActionValues(valuesFile, actionValues, artifact, sourceSha256);
  summaries.push({
    artifact_id: artifact.artifact_id,
    config_hash: artifact.config_hash,
    solver_version: artifact.solver_version,
    model: artifact.model,
    generated_at_unix_seconds: artifact.generated_at_unix_seconds,
    effective_stack_bb: artifact.config.effective_stack_bb,
    iterations: artifact.config.iterations,
    equity_samples: artifact.config.equity_samples,
    seed: artifact.config.seed,
    exploitability_bb: artifact.metrics.exploitability_bb,
    quality: artifact.validation.quality,
    source_sha256: sourceSha256,
    action_values_source_sha256: createHash('sha256')
      .update(valuesSource)
      .digest('hex'),
    action_value_standard_error_upper_bound_bb:
      actionValues.called_payoff_standard_error_upper_bound_bb,
    hands: artifact.strategies.hand_classes.map((hand) => [
      hand.label,
      hand.small_blind.shove,
      hand.big_blind_vs_shove.call,
    ]),
    action_values: actionValues.hand_classes.map((hand) => [
      hand.label,
      hand.small_blind.fold_ev_bb,
      hand.small_blind.shove_ev_bb,
      hand.big_blind_vs_shove.fold_ev_bb,
      hand.big_blind_vs_shove.call_ev_bb,
    ]),
  });
}

await writeFile(outputPath, `${JSON.stringify(summaries)}\n`);
console.log(
  `Wrote ${path.relative(root, outputPath)} with ${summaries.length} scenarios.`
);
