import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import {
  access,
  mkdir,
  mkdtemp,
  readFile,
  rm,
  writeFile,
} from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { gzipSync } from 'node:zlib';
import test from 'node:test';

const promoter = path.resolve(import.meta.dirname, 'promote-action-values.mjs');
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

async function fixture(coverage = 0.96) {
  const root = await mkdtemp(path.join(tmpdir(), 'action-value-promoter-'));
  const modelDir = path.join(root, 'models');
  await mkdir(modelDir);
  const policy = Buffer.from('{"policy":true}');
  const policyPath = path.join(root, 'policy.json');
  await writeFile(policyPath, policy);
  const actionValues = Buffer.from(
    JSON.stringify({
      schema: 'hu-preflop-canonical-range-action-values-v1',
      corpus_deals: 22_100,
      policy_artifact_sha256: sha256(policy),
      source_policy_sha256: 'a'.repeat(64),
      evaluated_information_sets: [8450, 8450],
      policy_lookup_coverage: 1,
      action_ev_standard_error_coverage: coverage,
    })
  );
  const compressed = gzipSync(actionValues);
  const artifactPath = path.join(root, 'candidate.json.gz');
  await writeFile(artifactPath, compressed);
  const artifactFiles = {
    networks: 'networks.json.gz',
    rangePolicy: 'range-policy.json.gz',
    preflopActionValues: 'old-action-values.json.gz',
    flopValueNetwork: 'value-network.json.gz',
  };
  for (const file of Object.values(artifactFiles)) {
    await writeFile(path.join(modelDir, file), gzipSync(Buffer.from(file)));
  }
  const manifestPath = path.join(root, 'manifest.json');
  await writeFile(
    manifestPath,
    JSON.stringify([
      {
        version: 'resolver-test',
        active: false,
        generatedAt: '2026-01-01T00:00:00.000Z',
        validation: {
          status: 'rejected',
          actionEvStandardErrorCoverage: 0.3,
          projectedStorageBytes: 0,
          notes: ['The preflop action-value corpus is too noisy.'],
        },
        runtime: {
          kind: 'rust-continual-resolver-v1',
          networkSha256: 'a'.repeat(64),
          preflopActionValuesSha256: 'b'.repeat(64),
          artifactFiles,
        },
      },
    ])
  );
  return {
    root,
    modelDir,
    manifestPath,
    policyPath,
    artifactPath,
    compressed,
    decodedSha256: sha256(actionValues),
  };
}

function promote(files, targetFile) {
  return spawnSync(
    process.execPath,
    [
      promoter,
      '--artifact',
      files.artifactPath,
      '--policy',
      files.policyPath,
      '--model-version',
      'resolver-test',
      '--target-file',
      targetFile,
      '--manifest',
      files.manifestPath,
      '--model-dir',
      files.modelDir,
    ],
    { encoding: 'utf8' }
  );
}

test('installs a passing artifact and updates the inactive manifest atomically', async () => {
  const files = await fixture();
  const targetFile = 'canonical-action-values.json.gz';
  try {
    const result = promote(files, targetFile);
    assert.equal(result.status, 0, result.stderr);
    const [manifest] = JSON.parse(await readFile(files.manifestPath, 'utf8'));
    assert.equal(manifest.active, false);
    assert.equal(manifest.validation.status, 'rejected');
    assert.equal(manifest.validation.actionEvStandardErrorCoverage, 0.96);
    assert.equal(manifest.runtime.artifactFiles.preflopActionValues, targetFile);
    assert.equal(
      manifest.runtime.preflopActionValuesSha256,
      files.decodedSha256
    );
    assert.deepEqual(
      await readFile(path.join(files.modelDir, targetFile)),
      files.compressed
    );
    assert.match(manifest.validation.notes.at(-1), /96\.000%/);
    assert.match(
      manifest.validation.notes.at(-1),
      /conservative full-hand sampling-error lower bound/
    );
  } finally {
    await rm(files.root, { recursive: true, force: true });
  }
});

test('does not install an artifact below the precision gate', async () => {
  const files = await fixture(0.949);
  const targetFile = 'rejected-action-values.json.gz';
  try {
    const result = promote(files, targetFile);
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /has not passed its serving gates/);
    await assert.rejects(access(path.join(files.modelDir, targetFile)));
  } finally {
    await rm(files.root, { recursive: true, force: true });
  }
});
