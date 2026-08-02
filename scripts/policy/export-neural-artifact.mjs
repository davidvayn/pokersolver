import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs, required } from './lib.mjs';

const MAGIC = Buffer.from('PLNP');
const HEADER_BYTES = 16;
const MAX_PARAMETER_COUNT = 32_000_000;

function validateNetwork(network, expectedInput, expectedOutput, parameterCount, name) {
  if (!network || !Array.isArray(network.layers) || network.layers.length === 0) {
    throw new Error(`${name} has no layers`);
  }
  let inputSize = expectedInput;
  for (const [index, layer] of network.layers.entries()) {
    if (
      !Number.isInteger(layer.inputSize) ||
      layer.inputSize !== inputSize ||
      !Number.isInteger(layer.outputSize) ||
      layer.outputSize < 1 ||
      !['linear', 'relu', 'tanh'].includes(layer.activation)
    ) {
      throw new Error(`${name} layer ${index} has an invalid shape or activation`);
    }
    const weightEnd = layer.weightOffset + layer.inputSize * layer.outputSize;
    const biasEnd = layer.biasOffset + layer.outputSize;
    if (
      !Number.isInteger(layer.weightOffset) ||
      layer.weightOffset < 0 ||
      weightEnd > parameterCount ||
      !Number.isInteger(layer.biasOffset) ||
      layer.biasOffset < 0 ||
      biasEnd > parameterCount
    ) {
      throw new Error(`${name} layer ${index} points outside the parameter buffer`);
    }
    inputSize = layer.outputSize;
  }
  if (inputSize !== expectedOutput) throw new Error(`${name} has the wrong output size`);
}

export function encodeNeuralArtifact(source) {
  if (!source || typeof source !== 'object') throw new Error('Model source must be an object');
  const { metadata, parameters } = source;
  if (!metadata || typeof metadata !== 'object') throw new Error('Model metadata is required');
  if (metadata.schemaVersion !== 1 || metadata.kind !== 'deep-cfr-baseline-response') {
    throw new Error('Unsupported neural artifact schema');
  }
  if (
    metadata.stateFeatureSchema !== 'hu-cash-trajectory-poker-aware-v4' ||
    metadata.actionFeatureSchema !== 'hu-cash-legal-action-v1' ||
    metadata.opponentProfileSchema !== 'local-opponent-profile-v1'
  ) {
    throw new Error('Neural feature schema is incompatible with the browser runtime');
  }
  if (!Array.isArray(parameters) || parameters.length > MAX_PARAMETER_COUNT) {
    throw new Error('Neural parameter array is missing or too large');
  }
  if (parameters.some((value) => typeof value !== 'number' || !Number.isFinite(value))) {
    throw new Error('Neural parameters must all be finite numbers');
  }
  if (metadata.parameterCount !== parameters.length) {
    throw new Error('Metadata parameter count does not match the model');
  }
  if (
    metadata.stateFeatureCount !== 716 ||
    metadata.actionFeatureCount !== 9 ||
    metadata.opponentProfileFeatureCount !== 16
  ) {
    throw new Error('Neural feature counts are incompatible with the browser runtime');
  }
  const stateAction = metadata.stateFeatureCount + metadata.actionFeatureCount;
  validateNetwork(
    metadata.networks?.baselinePolicy,
    stateAction,
    1,
    parameters.length,
    'baselinePolicy'
  );
  validateNetwork(
    metadata.networks?.exploitResponse,
    stateAction + metadata.opponentProfileFeatureCount,
    1,
    parameters.length,
    'exploitResponse'
  );
  validateNetwork(
    metadata.networks?.baselineActionValue,
    stateAction,
    2,
    parameters.length,
    'baselineActionValue'
  );
  const metadataBytes = Buffer.from(JSON.stringify(metadata));
  const output = Buffer.allocUnsafe(HEADER_BYTES + metadataBytes.length + parameters.length * 4);
  MAGIC.copy(output, 0);
  output.writeUInt16LE(1, 4);
  output.writeUInt16LE(0, 6);
  output.writeUInt32LE(metadataBytes.length, 8);
  output.writeUInt32LE(parameters.length, 12);
  metadataBytes.copy(output, HEADER_BYTES);
  let offset = HEADER_BYTES + metadataBytes.length;
  for (const value of parameters) {
    output.writeFloatLE(value, offset);
    offset += 4;
  }
  return output;
}

export function artifactDescriptor(bytes, artifactUrl) {
  if (
    typeof artifactUrl !== 'string' ||
    (!artifactUrl.startsWith('/models/practice/') && !artifactUrl.startsWith('https://'))
  ) {
    throw new Error('Artifact URL must be immutable static model content');
  }
  return {
    artifactUrl,
    artifactSha256: createHash('sha256').update(bytes).digest('hex'),
    artifactBytes: bytes.length,
  };
}

async function run() {
  const args = parseArgs(process.argv.slice(2));
  const input = path.resolve(required(args, '--input'));
  const output = path.resolve(required(args, '--output'));
  const artifactUrl = required(args, '--url');
  const source = JSON.parse(await readFile(input, 'utf8'));
  const bytes = encodeNeuralArtifact(source);
  await mkdir(path.dirname(output), { recursive: true });
  await writeFile(output, bytes, { flag: 'wx' });
  process.stdout.write(`${JSON.stringify(artifactDescriptor(bytes, artifactUrl), null, 2)}\n`);
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  run().catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  });
}
