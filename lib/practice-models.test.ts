import { describe, expect, it } from 'vitest';
import { isValidatedFullHandManifest } from '@/lib/practice-models';
import type { PolicyManifest } from '@/lib/practice-types';

function acceptedManifest(): PolicyManifest {
  return {
    schemaVersion: 1,
    version: 'deep-cfr-20-v1',
    model: 'deep-cfr-baseline-response',
    label: 'Approximate GTO',
    subtype: 'full-hand',
    active: true,
    depthsBb: [20],
    generatedAt: '2026-08-01T00:00:00.000Z',
    stateSchema: 'hu-cash-trajectory-poker-aware-v4',
    shardSchema: 'neural-binary-v1',
    runtime: {
      kind: 'neural-deep-cfr-v1',
      artifactUrl: '/models/practice/deep-cfr-20-v1/20bb.bin',
      artifactSha256: 'a'.repeat(64),
      stateFeatureSchema: 'hu-cash-trajectory-poker-aware-v4',
      actionFeatureSchema: 'hu-cash-legal-action-v1',
      opponentProfileSchema: 'local-opponent-profile-v1',
      actionAbstraction: {
        openSizesBb: [2, 2.5, 3],
        limpRaiseSizesBb: [3, 4, 5],
        threeBetSizesBb: [7.5, 9, 11],
        fourBetSizesBb: [18, 22, 26],
        deeperRaisePotFractions: [0.75, 1, 1.25],
        preflopRaiseCap: 4,
        flopBetPotFractions: [1 / 3, 0.75, 1.25],
        turnRiverBetPotFractions: [0.5, 1],
        postflopRaisePotFractions: [1],
        postflopRaiseCap: 1,
        includeAllIn: true,
      },
      adaptation: {
        minimumObservations: 50,
        fullConfidenceObservations: 250,
        maximumResponseWeight: 0.5,
      },
    },
    abstraction: {
      blindsBb: [0.5, 1],
      anteBb: 0,
      rake: 'none',
      actionSizing: 'pinned grid',
      cardAbstraction: 'exact cards into learned features',
      recall: 'trajectory',
    },
    validation: {
      status: 'accepted',
      exploitabilityEstimateBb: 0.05,
      exploitabilityUpper99Bb: 0.1,
      crossSeedFrequencyMae: 0.05,
      primaryActionAgreement: 0.85,
      maximumAggregateActionDelta: 0.03,
      policyCoverage: 0.9999,
      actionEvStandardErrorCoverage: 0.95,
      projectedStorageBytes: 1_000_000,
      rawProbabilitySumsValid: true,
      quantizedProbabilitySumsValid: true,
      independentSeedCount: 2,
      trainingHoursPerSeed: [8, 12],
      notes: [],
    },
  };
}

describe('database-free full-hand activation registry', () => {
  it('accepts a neural manifest only when every promotion gate is present', () => {
    expect(isValidatedFullHandManifest(acceptedManifest())).toBe(true);
    expect(
      isValidatedFullHandManifest({
        ...acceptedManifest(),
        validation: {
          ...acceptedManifest().validation,
          exploitabilityUpper99Bb: 0.101,
        },
      })
    ).toBe(false);
  });

  it('rejects unverified artifact locations and experimental labels', () => {
    const manifest = acceptedManifest();
    expect(
      isValidatedFullHandManifest({
        ...manifest,
        runtime: {
          ...manifest.runtime,
          artifactUrl: '/api/practice/model/latest',
        },
      })
    ).toBe(false);
    expect(
      isValidatedFullHandManifest({
        ...manifest,
        label: 'Experimental self-play',
      })
    ).toBe(false);
  });
});
