import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { gzipSync } from 'node:zlib';
import test from 'node:test';

const activator = path.resolve(
  import.meta.dirname,
  'activate-experimental-resolver.mjs'
);
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

async function fixture(coverage = 0.96) {
  const root = await mkdtemp(path.join(tmpdir(), 'experimental-activator-'));
  const modelDir = path.join(root, 'models');
  await mkdir(modelDir);
  const artifacts = {
    networks: Buffer.from('{"network":true}'),
    rangePolicy: Buffer.from('{"range":true}'),
    flopValueNetwork: Buffer.from('{"value":true}'),
  };
  const networkSha256 = sha256(artifacts.networks);
  artifacts.preflopActionValues = Buffer.from(
    JSON.stringify({
      schema: 'hu-preflop-canonical-range-action-values-v1',
      evaluated_information_sets: [8450, 8450],
      policy_lookup_coverage: 1,
      action_ev_standard_error_coverage: coverage,
      source_policy_sha256: networkSha256,
      policy_artifact_sha256: 'e'.repeat(64),
    })
  );
  const artifactFiles = {
    networks: 'networks.json.gz',
    rangePolicy: 'range-policy.json.gz',
    preflopActionValues: 'action-values.json.gz',
    flopValueNetwork: 'value-network.json.gz',
  };
  let projectedStorageBytes = 0;
  for (const [kind, file] of Object.entries(artifactFiles)) {
    const compressed = gzipSync(artifacts[kind]);
    projectedStorageBytes += compressed.length;
    await writeFile(path.join(modelDir, file), compressed);
  }
  const manifestPath = path.join(root, 'manifest.json');
  const manifest = {
    schemaVersion: 1,
    version: 'resolver-test',
    model: 'test',
    label: 'Experimental self-play',
    subtype: 'full-hand',
    active: false,
    depthsBb: [20],
    validation: {
      status: 'rejected',
      exploitabilityGateDeferred: true,
      crossSeedFrequencyMae: 0.049,
      primaryActionAgreement: 0.88,
      maximumAggregateActionDelta: 0.02,
      policyCoverage: 1,
      actionEvStandardErrorCoverage: coverage,
      projectedStorageBytes,
      rawProbabilitySumsValid: true,
      quantizedProbabilitySumsValid: true,
      independentSeedCount: 2,
      trainingHoursPerSeed: [8, 8],
    },
    runtime: {
      kind: 'rust-continual-resolver-v1',
      artifactFiles,
      networkSha256,
      rangePolicySha256: sha256(artifacts.rangePolicy),
      valueNetworkSha256: sha256(artifacts.flopValueNetwork),
      preflopActionValuesSha256: sha256(artifacts.preflopActionValues),
      resolver: {
        flopIterations: 2,
        flopResolvedActor: 1,
        turnIterations: 2,
        turnResolvedActor: 1,
        riverIterations: 2,
        riverResolvedActor: 1,
        deterministic: true,
      },
    },
  };
  await writeFile(manifestPath, JSON.stringify([manifest]));
  return { root, modelDir, manifestPath };
}

function activate(files) {
  return spawnSync(
    process.execPath,
    [
      activator,
      '--model-version',
      'resolver-test',
      '--manifest',
      files.manifestPath,
      '--model-dir',
      files.modelDir,
    ],
    { encoding: 'utf8' }
  );
}

test('activates only an explicitly experimental resolver with every normal gate', async () => {
  const files = await fixture();
  try {
    const result = activate(files);
    assert.equal(result.status, 0, result.stderr);
    const [manifest] = JSON.parse(await readFile(files.manifestPath, 'utf8'));
    assert.equal(manifest.active, true);
    assert.equal(manifest.validation.status, 'accepted');
    const output = JSON.parse(result.stdout);
    assert.equal(output.label, 'Experimental self-play');
    assert.equal(output.exploitabilityGateDeferred, true);
    assert.equal(output.artifactVerification.verified, true);
  } finally {
    await rm(files.root, { recursive: true, force: true });
  }
});

test('leaves the registry unchanged when action-EV precision fails', async () => {
  const files = await fixture(0.949);
  try {
    const result = activate(files);
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /actionEvPrecision/);
    const [manifest] = JSON.parse(await readFile(files.manifestPath, 'utf8'));
    assert.equal(manifest.active, false);
    assert.equal(manifest.validation.status, 'rejected');
  } finally {
    await rm(files.root, { recursive: true, force: true });
  }
});

test('leaves the registry unchanged when per-street actor routing is missing', async () => {
  const files = await fixture();
  try {
    const [manifest] = JSON.parse(await readFile(files.manifestPath, 'utf8'));
    delete manifest.runtime.resolver.riverResolvedActor;
    await writeFile(files.manifestPath, JSON.stringify([manifest]));
    const result = activate(files);
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /resolverConfiguration/);
    const [unchanged] = JSON.parse(await readFile(files.manifestPath, 'utf8'));
    assert.equal(unchanged.active, false);
    assert.equal(unchanged.validation.status, 'rejected');
  } finally {
    await rm(files.root, { recursive: true, force: true });
  }
});
