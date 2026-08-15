import assert from 'node:assert/strict';
import test from 'node:test';
import {
  assertAcceptedManifest,
  dynamoWriteDelayMs,
  splitBuffer,
} from './lib.mjs';
import { validateSeeds } from './validate-full-hand-seeds.mjs';

function seed(id, delta = 0) {
  return {
    schemaVersion: 1,
    model: 'dcfr-test',
    stateSchema: 'trajectory-v1',
    depthBb: 20,
    seed: id,
    trainingHours: 10,
    projectedStorageBytes: 1_000_000,
    evaluation: {
      exploitabilityEstimateBb: 0.49,
      exploitabilityUpper99Bb: id === 1 ? 0.5 : 0.499,
      policyLookupCoverage: 0.99995,
    },
    nodes: [
      {
        stateHash: 'a'.repeat(64),
        depthBb: 20,
        reachProbability: 1,
        actions: [
          { id: 'fold', kind: 'fold', label: 'Fold', probability: 0.3 + delta, evBb: 0, standardErrorBb: 0.01, confidence: 'high' },
          { id: 'call', kind: 'call', label: 'Call', probability: 0.7 - delta, evBb: 0.1, standardErrorBb: 0.01, confidence: 'high' },
        ],
      },
    ],
  };
}

test('accepts two stable independent seeds and chooses the lower upper bound', () => {
  const report = validateSeeds(seed(1), seed(2, 0.01));
  assert.equal(report.passed, true);
  assert.equal(report.selectedSeed, 2);
  assert.ok(report.stability.maximumActionFrequencyMae <= 0.05);
});

test('fails closed on exploitability, coverage, EV error, and instability', () => {
  const bad = seed(2, 0.4);
  bad.evaluation.exploitabilityEstimateBb = 0.500001;
  bad.evaluation.exploitabilityUpper99Bb = 0.500001;
  bad.evaluation.policyLookupCoverage = 0.99;
  bad.nodes[0].actions[0].standardErrorBb = 0.03;
  bad.nodes[0].actions[1].evBb = null;
  const report = validateSeeds(seed(1), bad);
  assert.equal(report.passed, false);
  assert.equal(report.selectedSeed, null);
  assert.equal(report.seedAudits[1].checks.exploitabilityEstimate, false);
  assert.equal(report.seedAudits[1].checks.exploitabilityUpper99, false);
  assert.equal(report.seedAudits[1].checks.policyLookupCoverage, false);
  assert.equal(report.seedAudits[1].checks.servedNodeIntegrity, false);
  assert.equal(report.stabilityChecks.primaryActionAgreement, false);
});

test('export gate requires the complete full-hand validation summary', () => {
  const manifest = {
    schemaVersion: 1,
    version: 'full-test-v1',
    label: 'Approximate GTO',
    subtype: 'full-hand',
    active: true,
    depthsBb: [20],
    validation: {
      status: 'accepted',
      exploitabilityEstimateBb: 0.5,
      exploitabilityUpper99Bb: 0.5,
      crossSeedFrequencyMae: 0.05,
      primaryActionAgreement: 0.85,
      maximumAggregateActionDelta: 0.03,
      policyCoverage: 0.9999,
      actionEvStandardErrorCoverage: 0.95,
      projectedStorageBytes: 20 * 1024 ** 3,
      rawProbabilitySumsValid: true,
      quantizedProbabilitySumsValid: true,
      independentSeedCount: 2,
      trainingHoursPerSeed: [8, 12],
    },
  };
  assert.doesNotThrow(() => assertAcceptedManifest(manifest));
  manifest.validation.maximumAggregateActionDelta = 0.031;
  assert.throws(
    () => assertAcceptedManifest(manifest),
    /maximumAggregateActionDelta/
  );
});

test('Dynamo importer chunks below normal item size and respects 25 WCU', () => {
  const parts = splitBuffer(Buffer.alloc(50 * 1024));
  assert.deepEqual(parts.map((part) => part.byteLength), [24 * 1024, 24 * 1024, 2 * 1024]);
  assert.equal(dynamoWriteDelayMs(24 * 1024, 25), 1000);
  assert.equal(dynamoWriteDelayMs(2 * 1024, 25), 120);
});
