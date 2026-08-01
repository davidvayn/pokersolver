import { createHash } from 'node:crypto';

export const MAX_HOSTED_BYTES = 20 * 1024 ** 3;
export const NORMAL_ITEM_BYTES = 24 * 1024;
export const MAX_ITEM_BYTES = 400 * 1024;
export const POLICY_REGION = 'us-west-2';

const MAGIC = Buffer.from('PLP1');
const SAMPLE_MAGIC = Buffer.from('PLS1');
const ACTION_KINDS = ['fold', 'check', 'call', 'bet', 'raise', 'all-in'];
const CONFIDENCE = ['high', 'low', 'unavailable'];

export function parseArgs(values) {
  const result = new Map();
  for (let index = 0; index < values.length; index++) {
    const value = values[index];
    if (!value.startsWith('--')) continue;
    const next = values[index + 1];
    if (!next || next.startsWith('--')) result.set(value, true);
    else {
      result.set(value, next);
      index++;
    }
  }
  return result;
}

export function required(args, name) {
  const value = args.get(name);
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`Missing required argument ${name}`);
  }
  return value;
}

export function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

function f32(value) {
  const buffer = Buffer.allocUnsafe(4);
  buffer.writeFloatLE(value ?? Number.NaN);
  return buffer;
}

function u16(value) {
  const buffer = Buffer.allocUnsafe(2);
  buffer.writeUInt16LE(value);
  return buffer;
}

function u32(value) {
  const buffer = Buffer.allocUnsafe(4);
  buffer.writeUInt32LE(value);
  return buffer;
}

function string(value) {
  const encoded = Buffer.from(value, 'utf8');
  if (encoded.length > 255) throw new Error('Policy strings must fit in 255 bytes');
  return Buffer.concat([Buffer.from([encoded.length]), encoded]);
}

export function validateNode(node) {
  if (!node || !/^[a-f0-9]{64}$/.test(node.stateHash ?? '')) {
    throw new Error('Every policy node requires a SHA-256 stateHash');
  }
  if (!Array.isArray(node.actions) || node.actions.length === 0) {
    throw new Error(`Policy node ${node.stateHash} has no actions`);
  }
  const sum = node.actions.reduce((total, action) => {
    if (!ACTION_KINDS.includes(action.kind)) throw new Error(`Invalid action kind ${action.kind}`);
    if (!CONFIDENCE.includes(action.confidence)) throw new Error(`Invalid confidence ${action.confidence}`);
    if (!Number.isFinite(action.probability) || action.probability < 0) {
      throw new Error(`Invalid probability at ${node.stateHash}`);
    }
    if (action.evBb !== null && (!Number.isFinite(action.evBb) || !Number.isFinite(action.standardErrorBb))) {
      throw new Error(`Invalid action EV at ${node.stateHash}`);
    }
    return total + action.probability;
  }, 0);
  if (Math.abs(sum - 1) > 1e-6) throw new Error(`Probabilities at ${node.stateHash} sum to ${sum}`);
}

export function encodeNodes(nodes) {
  const parts = [MAGIC, Buffer.from([1]), u32(nodes.length)];
  for (const node of nodes) {
    validateNode(node);
    const best = node.bestActionId == null
      ? 255
      : node.actions.findIndex((action) => action.id === node.bestActionId);
    if (best < 0) throw new Error(`Missing best action at ${node.stateHash}`);
    parts.push(
      Buffer.from(node.stateHash, 'hex'),
      Buffer.from([node.actions.length, best]),
      f32(node.bestActionEvBb),
      f32(node.reachProbability)
    );
    const probabilities = node.actions.map((action) =>
      Math.max(0, Math.min(65_535, Math.round(action.probability * 65_535)))
    );
    const difference = 65_535 - probabilities.reduce((sum, value) => sum + value, 0);
    if (difference !== 0) {
      const largest = node.actions.reduce(
        (best, action, index) => action.probability > node.actions[best].probability ? index : best,
        0
      );
      probabilities[largest] += difference;
    }
    for (const [actionIndex, action] of node.actions.entries()) {
      const amount = action.amountToBb == null ? null : action.amountToBb;
      const probability = probabilities[actionIndex];
      parts.push(
        Buffer.from([ACTION_KINDS.indexOf(action.kind)]),
        string(action.id),
        string(action.label),
        f32(amount),
        u16(probability),
        f32(action.evBb),
        f32(action.standardErrorBb),
        Buffer.from([CONFIDENCE.indexOf(action.confidence)])
      );
    }
  }
  return Buffer.concat(parts);
}

export function encodeSamples(samples) {
  const sorted = [...samples].sort((first, second) =>
    String(first.stateHash).localeCompare(String(second.stateHash))
  );
  const parts = [SAMPLE_MAGIC, Buffer.from([1]), u32(sorted.length)];
  for (const sample of sorted) {
    if (
      !sample ||
      !/^[a-f0-9]{64}$/.test(sample.stateHash ?? '') ||
      ![20, 50, 100].includes(sample.depthBb) ||
      !['flop', 'turn', 'river'].includes(sample.street) ||
      !sample.state ||
      !Array.isArray(sample.replayActions)
    ) {
      throw new Error('Invalid postflop sample');
    }
    const payload = Buffer.from(JSON.stringify(sample), 'utf8');
    parts.push(Buffer.from(sample.stateHash, 'hex'), u32(payload.length), payload);
  }
  return Buffer.concat(parts);
}

export function assertAcceptedManifest(manifest) {
  if (
    !manifest ||
    manifest.schemaVersion !== 1 ||
    manifest.label !== 'Approximate GTO' ||
    manifest.active !== true ||
    manifest.validation?.status !== 'accepted' ||
    !['full-hand', 'push-fold'].includes(manifest.subtype) ||
    !Array.isArray(manifest.depthsBb) ||
    manifest.depthsBb.length === 0
  ) {
    throw new Error('Manifest is not an active accepted Approximate GTO model');
  }
  if (manifest.subtype !== 'full-hand') return;
  if (!manifest.depthsBb.every((depth) => [20, 50, 100].includes(depth))) {
    throw new Error('Full-hand manifest contains an unsupported depth');
  }
  const validation = manifest.validation;
  const gates = [
    ['exploitabilityEstimateBb', validation.exploitabilityEstimateBb <= 0.05],
    ['exploitabilityUpper99Bb', validation.exploitabilityUpper99Bb <= 0.1],
    ['crossSeedFrequencyMae', validation.crossSeedFrequencyMae <= 0.05],
    ['primaryActionAgreement', validation.primaryActionAgreement >= 0.85],
    ['maximumAggregateActionDelta', validation.maximumAggregateActionDelta <= 0.03],
    ['policyCoverage', validation.policyCoverage >= 0.9999],
    ['actionEvStandardErrorCoverage', validation.actionEvStandardErrorCoverage >= 0.95],
    ['projectedStorageBytes', validation.projectedStorageBytes <= MAX_HOSTED_BYTES],
    ['rawProbabilitySumsValid', validation.rawProbabilitySumsValid === true],
    ['quantizedProbabilitySumsValid', validation.quantizedProbabilitySumsValid === true],
    ['independentSeedCount', validation.independentSeedCount === 2],
    [
      'trainingHoursPerSeed',
      Array.isArray(validation.trainingHoursPerSeed) &&
        validation.trainingHoursPerSeed.length === 2 &&
        validation.trainingHoursPerSeed.every((hours) =>
          Number.isFinite(hours) && hours >= 8 && hours <= 12
        ),
    ],
  ];
  const failed = gates.filter(([, passed]) => !passed).map(([name]) => name);
  if (failed.length > 0) throw new Error(`Full-hand manifest failed gates: ${failed.join(', ')}`);
}

export function splitBuffer(buffer, size = NORMAL_ITEM_BYTES) {
  if (size <= 0 || size >= MAX_ITEM_BYTES) throw new Error('Invalid DynamoDB item split size');
  const parts = [];
  for (let offset = 0; offset < buffer.length; offset += size) {
    parts.push(buffer.subarray(offset, Math.min(buffer.length, offset + size)));
  }
  return parts;
}

export function dynamoWriteDelayMs(payloadBytes, capacity = 25) {
  if (!Number.isFinite(payloadBytes) || payloadBytes < 0 || capacity <= 0) {
    throw new Error('Invalid DynamoDB throttle input');
  }
  const writeUnits = Math.ceil(payloadBytes / 1024) + 1;
  return Math.ceil((writeUnits / capacity) * 1000);
}

export function sleep(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
