import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { gzipSync } from 'node:zlib';
import test from 'node:test';

const verifier = path.resolve(
  import.meta.dirname,
  'verify-resolver-artifacts.mjs'
);
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

async function fixture({ active = false, unsafe = false } = {}) {
  const root = await mkdtemp(path.join(tmpdir(), 'resolver-artifact-verifier-'));
  const modelDir = path.join(root, 'models');
  const actionValues = {
    schema: 'hu-preflop-local-leak-attribution-v1',
    evaluated_information_sets: [8450, 8450],
    policy_lookup_coverage: 1,
    action_ev_standard_error_coverage: 0.3,
  };
  const artifacts = {
    networks: Buffer.from('{}'),
    rangePolicy: Buffer.from('{}'),
    preflopActionValues: Buffer.from(JSON.stringify(actionValues)),
    flopValueNetwork: Buffer.from('{}'),
  };
  const files = {
    networks: 'networks.json.gz',
    rangePolicy: 'range-policy.json.gz',
    preflopActionValues: unsafe
      ? '../preflop-action-values.json.gz'
      : 'preflop-action-values.json.gz',
    flopValueNetwork: 'flop-value-network.json.gz',
  };
  await mkdir(modelDir, { recursive: true });
  for (const [kind, bytes] of Object.entries(artifacts)) {
    const file = path.basename(files[kind]);
    await writeFile(path.join(modelDir, file), gzipSync(bytes));
  }
  const manifest = [
    {
      version: 'resolver-test',
      active,
      validation: {
        status: active ? 'accepted' : 'rejected',
        actionEvStandardErrorCoverage: 0.3,
      },
      runtime: {
        kind: 'rust-continual-resolver-v1',
        artifactFiles: files,
        networkSha256: sha256(artifacts.networks),
        rangePolicySha256: sha256(artifacts.rangePolicy),
        preflopActionValuesSha256: sha256(artifacts.preflopActionValues),
        valueNetworkSha256: sha256(artifacts.flopValueNetwork),
      },
    },
  ];
  const manifestPath = path.join(root, 'manifest.json');
  await writeFile(manifestPath, JSON.stringify(manifest));
  return { root, modelDir, manifestPath };
}

function verify({ modelDir, manifestPath }) {
  return spawnSync(
    process.execPath,
    [verifier, '--manifest', manifestPath, '--model-dir', modelDir],
    { encoding: 'utf8' }
  );
}

test('verifies decoded resolver identities for an inactive candidate', async () => {
  const files = await fixture();
  try {
    const result = verify(files);
    assert.equal(result.status, 0, result.stderr);
    assert.equal(JSON.parse(result.stdout).verified, true);
  } finally {
    await rm(files.root, { recursive: true, force: true });
  }
});

test('rejects low-precision or unsafe action values at the serving boundary', async () => {
  const active = await fixture({ active: true });
  const unsafe = await fixture({ unsafe: true });
  try {
    const activeResult = verify(active);
    assert.notEqual(activeResult.status, 0);
    assert.match(activeResult.stderr, /cannot serve its preflop action values/);

    const unsafeResult = verify(unsafe);
    assert.notEqual(unsafeResult.status, 0);
    assert.match(unsafeResult.stderr, /invalid preflopActionValues artifact metadata/);
  } finally {
    await Promise.all([
      rm(active.root, { recursive: true, force: true }),
      rm(unsafe.root, { recursive: true, force: true }),
    ]);
  }
});
