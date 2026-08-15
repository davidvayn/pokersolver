import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  MAX_FULL_HAND_TOTAL_EXPLOITABILITY_BB,
  MAX_HOSTED_BYTES,
  parseArgs,
  required,
} from './lib.mjs';

const PROBABILITY_TOLERANCE = 1e-6;
const QUANTIZED_TOLERANCE = 1 / 65_535 + 1e-9;

function finite(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function nodeMap(seed) {
  if (!Array.isArray(seed.nodes) || seed.nodes.length === 0) throw new Error('Seed contains no served policy nodes');
  return new Map(seed.nodes.map((node) => [node.stateHash, node]));
}

function primary(node) {
  return [...node.actions].sort(
    (left, right) => right.probability - left.probability || left.id.localeCompare(right.id)
  )[0]?.id;
}

function probabilityAudit(seed) {
  let rawMaximumError = 0;
  let quantizedMaximumError = 0;
  let validInputs = true;
  for (const node of seed.nodes) {
    if (!Array.isArray(node.actions) || node.actions.length === 0) {
      validInputs = false;
      continue;
    }
    if (
      node.actions.some(
        (action) =>
          !finite(action.probability) ||
          action.probability < 0 ||
          action.probability > 1
      )
    ) {
      validInputs = false;
      continue;
    }
    const raw = node.actions.reduce((sum, action) => sum + action.probability, 0);
    rawMaximumError = Math.max(rawMaximumError, Math.abs(raw - 1));
    const quantized = node.actions.map((action) => Math.round(action.probability * 65_535));
    const difference = 65_535 - quantized.reduce((sum, value) => sum + value, 0);
    const largest = node.actions.reduce(
      (best, action, index) => action.probability > node.actions[best].probability ? index : best,
      0
    );
    quantized[largest] += difference;
    quantizedMaximumError = Math.max(
      quantizedMaximumError,
      Math.abs(quantized.reduce((sum, value) => sum + value, 0) / 65_535 - 1)
    );
  }
  return { rawMaximumError, quantizedMaximumError, validInputs };
}

function servedNodeIntegrity(seed) {
  return seed.nodes.every((node) => {
    const ids = new Set(node.actions?.map((action) => action.id));
    return (
      /^[a-f0-9]{64}$/.test(node.stateHash ?? '') &&
      node.depthBb === seed.depthBb &&
      finite(node.reachProbability) &&
      node.reachProbability >= 0 &&
      Array.isArray(node.actions) &&
      node.actions.length > 0 &&
      ids.size === node.actions.length &&
      node.actions.every(
        (action) =>
          typeof action.id === 'string' &&
          action.id.length > 0 &&
          finite(action.evBb) &&
          finite(action.standardErrorBb) &&
          action.standardErrorBb >= 0 &&
          ['high', 'low'].includes(action.confidence)
      )
    );
  });
}

function actionEvCoverage(seed) {
  let eligibleWeight = 0;
  let passingWeight = 0;
  for (const node of seed.nodes) {
    const reach = finite(node.reachProbability) ? node.reachProbability : 0;
    for (const action of node.actions) {
      const weight = reach * action.probability;
      eligibleWeight += weight;
      if (finite(action.standardErrorBb) && action.standardErrorBb <= 0.02) passingWeight += weight;
    }
  }
  return eligibleWeight > 0 ? passingWeight / eligibleWeight : 0;
}

function crossSeed(first, second) {
  const left = nodeMap(first);
  const right = nodeMap(second);
  const shared = [...left.keys()].filter((key) => right.has(key));
  let weight = 0;
  let primaryWeight = 0;
  const absolute = new Map();
  const aggregateLeft = new Map();
  const aggregateRight = new Map();
  for (const key of shared) {
    const a = left.get(key);
    const b = right.get(key);
    const reach = Math.sqrt(Math.max(0, a.reachProbability ?? 0) * Math.max(0, b.reachProbability ?? 0));
    if (reach === 0) continue;
    const actionsA = new Map(a.actions.map((action) => [action.id, action.probability]));
    const actionsB = new Map(b.actions.map((action) => [action.id, action.probability]));
    const actions = new Set([...actionsA.keys(), ...actionsB.keys()]);
    for (const action of actions) {
      const leftProbability = actionsA.get(action) ?? 0;
      const rightProbability = actionsB.get(action) ?? 0;
      absolute.set(action, (absolute.get(action) ?? 0) + reach * Math.abs(leftProbability - rightProbability));
      aggregateLeft.set(action, (aggregateLeft.get(action) ?? 0) + reach * leftProbability);
      aggregateRight.set(action, (aggregateRight.get(action) ?? 0) + reach * rightProbability);
    }
    weight += reach;
    if (primary(a) === primary(b)) primaryWeight += reach;
  }
  const actionFrequencyMae = Object.fromEntries(
    [...absolute].map(([action, value]) => [action, weight ? value / weight : Number.POSITIVE_INFINITY])
  );
  const aggregateActionDeltas = Object.fromEntries(
    [...new Set([...aggregateLeft.keys(), ...aggregateRight.keys()])].map((action) => [
      action,
      weight
        ? Math.abs((aggregateLeft.get(action) ?? 0) / weight - (aggregateRight.get(action) ?? 0) / weight)
        : Number.POSITIVE_INFINITY,
    ])
  );
  return {
    sharedNodes: shared.length,
    reachWeight: weight,
    actionFrequencyMae,
    maximumActionFrequencyMae: Math.max(...Object.values(actionFrequencyMae)),
    aggregateActionDeltas,
    maximumAggregateActionDelta: Math.max(...Object.values(aggregateActionDeltas)),
    primaryActionAgreement: weight ? primaryWeight / weight : 0,
  };
}

function seedChecks(seed) {
  const probabilities = probabilityAudit(seed);
  const computedActionEvCoverage = actionEvCoverage(seed);
  const checks = {
    trainingHours: finite(seed.trainingHours) && seed.trainingHours >= 8 && seed.trainingHours <= 12,
    exploitabilityEstimate:
      finite(seed.evaluation?.exploitabilityEstimateBb) &&
      seed.evaluation.exploitabilityEstimateBb <=
        MAX_FULL_HAND_TOTAL_EXPLOITABILITY_BB,
    exploitabilityUpper99:
      finite(seed.evaluation?.exploitabilityUpper99Bb) &&
      seed.evaluation.exploitabilityUpper99Bb <=
        MAX_FULL_HAND_TOTAL_EXPLOITABILITY_BB,
    policyLookupCoverage: finite(seed.evaluation?.policyLookupCoverage) && seed.evaluation.policyLookupCoverage >= 0.9999,
    servedNodeIntegrity: servedNodeIntegrity(seed),
    rawProbabilitySums: probabilities.validInputs && probabilities.rawMaximumError <= PROBABILITY_TOLERANCE,
    quantizedProbabilitySums: probabilities.quantizedMaximumError <= QUANTIZED_TOLERANCE,
    actionEvStandardErrorCoverage: computedActionEvCoverage >= 0.95,
    storage: finite(seed.projectedStorageBytes) && seed.projectedStorageBytes <= MAX_HOSTED_BYTES,
  };
  return { checks, probabilities, computedActionEvCoverage, passed: Object.values(checks).every(Boolean) };
}

export function validateSeeds(first, second) {
  if (first.depthBb !== second.depthBb || ![20, 50, 100].includes(first.depthBb)) {
    throw new Error('Seeds must share a supported 20/50/100bb depth');
  }
  if (first.seed === second.seed) throw new Error('Two distinct training seeds are required');
  if (first.model !== second.model || first.stateSchema !== second.stateSchema) {
    throw new Error('Seeds use incompatible model/state schemas');
  }
  const firstAudit = seedChecks(first);
  const secondAudit = seedChecks(second);
  const stability = crossSeed(first, second);
  const stabilityChecks = {
    crossSeedActionFrequencyMae: stability.maximumActionFrequencyMae <= 0.05,
    primaryActionAgreement: stability.primaryActionAgreement >= 0.85,
    aggregateActionDeltas: stability.maximumAggregateActionDelta <= 0.03,
  };
  const passed =
    firstAudit.passed &&
    secondAudit.passed &&
    Object.values(stabilityChecks).every(Boolean);
  const selected = [first, second].sort(
    (left, right) =>
      left.evaluation.exploitabilityUpper99Bb - right.evaluation.exploitabilityUpper99Bb
  )[0];
  return {
    schemaVersion: 1,
    depthBb: first.depthBb,
    passed,
    selectedSeed: passed ? selected.seed : null,
    seedAudits: [
      { seed: first.seed, ...firstAudit },
      { seed: second.seed, ...secondAudit },
    ],
    stability,
    stabilityChecks,
    interpretation:
      'Cross-seed stability is a reproducibility gate, not equilibrium proof. The exploitability estimate and bound come from the independent evaluation pass.',
  };
}

async function run() {
  const args = parseArgs(process.argv.slice(2));
  const firstPath = path.resolve(required(args, '--seed-a'));
  const secondPath = path.resolve(required(args, '--seed-b'));
  const first = JSON.parse(await readFile(firstPath, 'utf8'));
  const second = JSON.parse(await readFile(secondPath, 'utf8'));
  const report = validateSeeds(first, second);
  const output = args.get('--output');
  if (typeof output === 'string') {
    await writeFile(path.resolve(output), `${JSON.stringify(report, null, 2)}\n`, { flag: 'wx' });
  } else {
    process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  }
  if (!report.passed) process.exitCode = 2;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await run();
}
