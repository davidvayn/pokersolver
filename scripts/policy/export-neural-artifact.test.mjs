import assert from 'node:assert/strict';
import test from 'node:test';
import {
  artifactDescriptor,
  encodeNeuralArtifact,
} from './export-neural-artifact.mjs';

function validSource() {
  const parameters = Array(2920).fill(0);
  return {
    metadata: {
      schemaVersion: 1,
      kind: 'deep-cfr-baseline-response',
      stateFeatureSchema: 'hu-cash-trajectory-poker-aware-v4',
      stateFeatureCount: 716,
      actionFeatureSchema: 'hu-cash-legal-action-v1',
      actionFeatureCount: 9,
      opponentProfileSchema: 'local-opponent-profile-v1',
      opponentProfileFeatureCount: 16,
      parameterCount: parameters.length,
      networks: {
        baselinePolicy: {
          layers: [{ inputSize: 725, outputSize: 1, activation: 'linear', weightOffset: 0, biasOffset: 725 }],
        },
        exploitResponse: {
          layers: [{ inputSize: 741, outputSize: 1, activation: 'linear', weightOffset: 726, biasOffset: 1467 }],
        },
        baselineActionValue: {
          layers: [{ inputSize: 725, outputSize: 2, activation: 'linear', weightOffset: 1468, biasOffset: 2918 }],
        },
      },
    },
    parameters,
  };
}

test('exports the browser neural binary envelope deterministically', () => {
  const source = validSource();
  source.parameters[0] = 1.25;
  source.parameters[1] = -2.5;
  const first = encodeNeuralArtifact(source);
  const second = encodeNeuralArtifact(source);
  assert.deepEqual(first, second);
  assert.equal(first.subarray(0, 4).toString(), 'PLNP');
  assert.equal(first.readUInt16LE(4), 1);
  assert.equal(first.readUInt32LE(12), source.parameters.length);
  const descriptor = artifactDescriptor(
    first,
    '/models/practice/test/20bb.bin'
  );
  assert.match(descriptor.artifactSha256, /^[a-f0-9]{64}$/);
  assert.equal(descriptor.artifactBytes, first.length);
});

test('refuses non-finite weights and mutable application URLs', () => {
  assert.throws(
    () =>
      encodeNeuralArtifact({
        ...validSource(),
        parameters: [Number.NaN, ...validSource().parameters.slice(1)],
      }),
    /finite/
  );
  assert.throws(() => artifactDescriptor(Buffer.alloc(1), '/api/model'), /immutable/);
});
