import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import test from 'node:test';

function manifest() {
  return {
    schemaVersion: 1,
    version: 'full-export-test-v1',
    model: 'dcfr-test',
    label: 'Approximate GTO',
    subtype: 'full-hand',
    active: true,
    depthsBb: [20],
    generatedAt: '2026-01-01T00:00:00.000Z',
    stateSchema: 'hu-cash-v1',
    shardSchema: 'plp1',
    abstraction: {},
    validation: {
      status: 'accepted',
      exploitabilityEstimateBb: 0.04,
      exploitabilityUpper99Bb: 0.08,
      crossSeedFrequencyMae: 0.03,
      primaryActionAgreement: 0.9,
      maximumAggregateActionDelta: 0.02,
      policyCoverage: 0.99995,
      actionEvStandardErrorCoverage: 0.97,
      projectedStorageBytes: 4096,
      rawProbabilitySumsValid: true,
      quantizedProbabilitySumsValid: true,
      independentSeedCount: 2,
      trainingHoursPerSeed: [10, 10],
      notes: [],
    },
  };
}

test('exports deterministic policy and postflop sample shard indexes under the size gate', async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'poker-policy-export-'));
  try {
    const input = path.join(directory, 'source.json');
    const output = path.join(directory, 'hosted');
    const stateHash = '12'.repeat(32);
    const sampleHash = '34'.repeat(32);
    await writeFile(input, JSON.stringify({
      manifest: manifest(),
      nodes: [{
        stateHash,
        depthBb: 20,
        bestActionId: 'check',
        bestActionEvBb: 0.1,
        reachProbability: 1,
        actions: [{
          id: 'check',
          kind: 'check',
          label: 'Check',
          probability: 1,
          evBb: 0.1,
          standardErrorBb: 0.01,
          confidence: 'high',
        }],
      }],
      samples: [{
        stateHash: sampleHash,
        depthBb: 20,
        street: 'flop',
        state: { id: 'sample' },
        replayActions: [{ kind: 'call' }, { kind: 'check' }],
      }],
    }));
    const result = spawnSync(
      process.execPath,
      ['scripts/policy/export-shards.mjs', '--input', input, '--output', output],
      { cwd: process.cwd(), encoding: 'utf8' }
    );
    assert.equal(result.status, 0, result.stderr);
    const indexPath = path.join(output, manifest().version, 'export-index.json');
    const index = JSON.parse(await readFile(indexPath, 'utf8'));
    assert.equal(index.shards.length, 2);
    assert.deepEqual(index.shards.map((shard) => shard.kind).sort(), ['policy', 'sample']);
    assert.ok(index.totalBytes > 0);
    assert.ok(index.projectedHostedBytes <= 20 * 1024 ** 3);
    const policy = await readFile(path.join(output, manifest().version, '20', '1212.bin'));
    const sample = await readFile(path.join(output, 'samples', manifest().version, '20', 'flop', '3434.bin'));
    assert.equal(policy.subarray(0, 4).toString(), 'PLP1');
    assert.equal(sample.subarray(0, 4).toString(), 'PLS1');

    const audit = spawnSync(
      process.execPath,
      ['scripts/policy/audit-size.mjs', '--indexes', indexPath],
      { cwd: process.cwd(), encoding: 'utf8' }
    );
    assert.equal(audit.status, 0, audit.stderr);
    assert.equal(JSON.parse(audit.stdout).passed, true);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
