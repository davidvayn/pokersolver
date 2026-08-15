import assert from 'node:assert/strict';
import { chmod, mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import test from 'node:test';

const verifier = path.resolve(import.meta.dirname, 'verify-resolver-build.mjs');

async function fixture({ omit = null, includeStale = false } = {}) {
  const root = await mkdtemp(path.join(tmpdir(), 'resolver-build-verifier-'));
  const modelDir = path.join(root, 'models');
  const binary = path.join(root, 'preflop-solver');
  const tracePath = path.join(root, '.next', 'route.js.nft.json');
  const artifactFiles = {
    networks: 'networks.json.gz',
    rangePolicy: 'range-policy.json.gz',
    preflopActionValues: 'action-values.json.gz',
    flopValueNetwork: 'value-network.json.gz',
  };
  await mkdir(modelDir, { recursive: true });
  await mkdir(path.dirname(tracePath), { recursive: true });
  await writeFile(binary, '#!/bin/sh\n');
  await chmod(binary, 0o755);
  for (const file of Object.values(artifactFiles)) {
    await writeFile(path.join(modelDir, file), file);
  }
  const manifestPath = path.join(root, 'manifest.json');
  await writeFile(
    manifestPath,
    JSON.stringify([
      {
        version: 'resolver-test',
        runtime: { kind: 'rust-continual-resolver-v1', artifactFiles },
      },
    ])
  );
  const traced = [binary, ...Object.values(artifactFiles).map((file) => path.join(modelDir, file))]
    .filter((file) => path.basename(file) !== omit);
  if (includeStale) {
    const stale = path.join(modelDir, 'stale.json.gz');
    await writeFile(stale, 'stale');
    traced.push(stale);
  }
  await writeFile(
    tracePath,
    JSON.stringify({
      version: 1,
      files: traced.map((file) => path.relative(path.dirname(tracePath), file)),
    })
  );
  return { root, modelDir, binary, tracePath, manifestPath };
}

function verify(files) {
  return spawnSync(
    process.execPath,
    [
      verifier,
      '--manifest',
      files.manifestPath,
      '--model-dir',
      files.modelDir,
      '--binary',
      files.binary,
      '--trace',
      files.tracePath,
    ],
    { encoding: 'utf8' }
  );
}

test('verifies the production trace contains the exact pinned resolver bundle', async () => {
  const files = await fixture();
  try {
    const result = verify(files);
    assert.equal(result.status, 0, result.stderr);
    const output = JSON.parse(result.stdout);
    assert.equal(output.verified, true);
    assert.equal(output.artifacts.length, 4);
    assert.ok(output.pinnedBundleBytes > 0);
  } finally {
    await rm(files.root, { recursive: true, force: true });
  }
});

test('rejects a trace that omits a pinned artifact', async () => {
  const files = await fixture({ omit: 'action-values.json.gz' });
  try {
    const result = verify(files);
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /omits pinned files.*action-values\.json\.gz/s);
  } finally {
    await rm(files.root, { recursive: true, force: true });
  }
});

test('rejects unpinned model payloads in the resolver bundle', async () => {
  const files = await fixture({ includeStale: true });
  try {
    const result = verify(files);
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /unpinned model files: stale\.json\.gz/);
  } finally {
    await rm(files.root, { recursive: true, force: true });
  }
});
