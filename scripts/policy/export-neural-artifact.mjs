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
  if (![1, 2].includes(metadata.schemaVersion) || metadata.kind !== 'deep-cfr-baseline-response') {
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
  const postflopNetworks = [
    metadata.networks?.postflopBaselinePolicy,
    metadata.networks?.postflopExploitResponse,
    metadata.networks?.postflopBaselineActionValue,
  ];
  const postflopNetworkCount = postflopNetworks.filter(Boolean).length;
  if (metadata.schemaVersion === 1 && (metadata.routing || postflopNetworkCount > 0)) {
    throw new Error('Artifact schema 1 cannot declare street routing');
  }
  if (
    metadata.schemaVersion === 2 &&
    (metadata.routing?.kind !== 'street-v1' ||
      !metadata.routing.preflopModelVersion ||
      !metadata.routing.postflopModelVersion ||
      postflopNetworkCount !== 3)
  ) {
    throw new Error('Street-routed artifact metadata is incomplete');
  }
  if (postflopNetworkCount > 0 && postflopNetworkCount !== 3) {
    throw new Error('Street-routed postflop network group is incomplete');
  }
  if (postflopNetworkCount === 3) {
    validateNetwork(
      metadata.networks.postflopBaselinePolicy,
      stateAction,
      1,
      parameters.length,
      'postflopBaselinePolicy'
    );
    validateNetwork(
      metadata.networks.postflopExploitResponse,
      stateAction + metadata.opponentProfileFeatureCount,
      1,
      parameters.length,
      'postflopExploitResponse'
    );
    validateNetwork(
      metadata.networks.postflopBaselineActionValue,
      stateAction,
      2,
      parameters.length,
      'postflopBaselineActionValue'
    );
  }
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

function sameJson(first, second) {
  return JSON.stringify(first) === JSON.stringify(second);
}

function shiftedNetwork(network, parameterOffset) {
  return {
    layers: network.layers.map((layer) => ({
      ...layer,
      weightOffset: layer.weightOffset + parameterOffset,
      biasOffset: layer.biasOffset + parameterOffset,
    })),
  };
}

export function composeStreetRoutedArtifact(preflop, postflop, modelVersion) {
  // Validate the independent components before copying any descriptor offsets.
  encodeNeuralArtifact(preflop);
  encodeNeuralArtifact(postflop);
  if (preflop.metadata.schemaVersion !== 1 || postflop.metadata.schemaVersion !== 1) {
    throw new Error('Street routing currently composes two single-network schema-1 artifacts');
  }
  if (typeof modelVersion !== 'string' || modelVersion.length === 0) {
    throw new Error('A composite model version is required');
  }
  const compatibleFields = [
    'depthBb',
    'stateFeatureSchema',
    'stateFeatureCount',
    'actionFeatureSchema',
    'actionFeatureCount',
    'opponentProfileSchema',
    'opponentProfileFeatureCount',
    'actionAbstraction',
    'adaptation',
    'valueCalibration',
  ];
  for (const field of compatibleFields) {
    if (!sameJson(preflop.metadata[field], postflop.metadata[field])) {
      throw new Error(`Street-routed components disagree on ${field}`);
    }
  }
  const parameterOffset = preflop.parameters.length;
  const parameters = [...preflop.parameters, ...postflop.parameters];
  return {
    metadata: {
      ...preflop.metadata,
      schemaVersion: 2,
      modelVersion,
      parameterCount: parameters.length,
      routing: {
        kind: 'street-v1',
        preflopModelVersion: preflop.metadata.modelVersion,
        postflopModelVersion: postflop.metadata.modelVersion,
      },
      networks: {
        baselinePolicy: preflop.metadata.networks.baselinePolicy,
        exploitResponse: preflop.metadata.networks.exploitResponse,
        baselineActionValue: preflop.metadata.networks.baselineActionValue,
        postflopBaselinePolicy: shiftedNetwork(
          postflop.metadata.networks.baselinePolicy,
          parameterOffset
        ),
        postflopExploitResponse: shiftedNetwork(
          postflop.metadata.networks.exploitResponse,
          parameterOffset
        ),
        postflopBaselineActionValue: shiftedNetwork(
          postflop.metadata.networks.baselineActionValue,
          parameterOffset
        ),
      },
    },
    parameters,
  };
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
  const output = path.resolve(required(args, '--output'));
  const artifactUrl = required(args, '--url');
  const input = args.get('--input');
  const preflopInput = args.get('--preflop-input');
  const postflopInput = args.get('--postflop-input');
  if (Boolean(input) === Boolean(preflopInput || postflopInput)) {
    throw new Error('Provide either --input or both street component inputs');
  }
  let source;
  if (input) {
    source = JSON.parse(await readFile(path.resolve(input), 'utf8'));
  } else {
    if (!preflopInput || !postflopInput) {
      throw new Error('Both --preflop-input and --postflop-input are required');
    }
    const [preflop, postflop] = await Promise.all(
      [preflopInput, postflopInput].map(async (component) =>
        JSON.parse(await readFile(path.resolve(component), 'utf8'))
      )
    );
    source = composeStreetRoutedArtifact(
      preflop,
      postflop,
      required(args, '--model-version')
    );
  }
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
