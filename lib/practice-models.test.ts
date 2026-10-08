import { describe, expect, it } from 'vitest';
import fullHandManifests from '@/data/practice/full-hand-manifests.json';
import {
  ACTIVE_FULL_HAND_MANIFESTS,
  isExperimentalFullHandManifest,
  isValidatedFullHandManifest,
  modelForFullHandIdentity,
} from '@/lib/practice-models';
import { HOME_RULES_SHA256, NL25_RULES_SHA256 } from '@/lib/practice-game-identity';
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
      exploitabilityEstimateBb: 0.5,
      exploitabilityUpper99Bb: 0.5,
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
  it('selects an exact pinned version, depth, and rules identity without cross-profile fallback',()=>{
    for (const manifest of ACTIVE_FULL_HAND_MANIFESTS) {
      expect(modelForFullHandIdentity(manifest.version,manifest.depthsBb[0],HOME_RULES_SHA256)).toBe(manifest);
      expect(modelForFullHandIdentity(manifest.version,manifest.depthsBb[0],NL25_RULES_SHA256)).toBeNull();
      expect(modelForFullHandIdentity(manifest.version,2000)).toBeNull();
    }
    expect(modelForFullHandIdentity('missing',20)).toBeNull();
  });
  it('serves exactly the checked-in manifests that pass a complete serving predicate', () => {
    const expected = (fullHandManifests as unknown[]).filter(
      (manifest) =>
        isValidatedFullHandManifest(manifest) ||
        isExperimentalFullHandManifest(manifest)
    );
    expect(ACTIVE_FULL_HAND_MANIFESTS).toEqual(expected);
    expect(
      ACTIVE_FULL_HAND_MANIFESTS.every(
        (manifest) =>
          manifest.active && manifest.validation.status === 'accepted'
      )
    ).toBe(true);
  });

  it('accepts a neural manifest only when every promotion gate is present', () => {
    expect(isValidatedFullHandManifest(acceptedManifest())).toBe(true);
    expect(
      isValidatedFullHandManifest({
        ...acceptedManifest(),
        validation: {
          ...acceptedManifest().validation,
          exploitabilityEstimateBb: 0.500001,
        },
      })
    ).toBe(false);
    expect(
      isValidatedFullHandManifest({
        ...acceptedManifest(),
        validation: {
          ...acceptedManifest().validation,
          exploitabilityUpper99Bb: 0.500001,
        },
      })
    ).toBe(false);
  });

  it('allows additional trained stack depths without weakening activation gates', () => {
    for (const depth of [20, 40, 50, 75, 100, 150, 200, 1000, 2000]) {
      const manifest = { ...acceptedManifest(), depthsBb: [depth] };
      expect(isValidatedFullHandManifest(manifest)).toBe(true);
      expect(isValidatedFullHandManifest({
        ...manifest,
        validation: { ...manifest.validation, policyCoverage: 0.9 },
      })).toBe(false);
    }
    for (const depth of [0, -20, 1, 20.5, Infinity, NaN, '40']) {
      expect(isValidatedFullHandManifest({
        ...acceptedManifest(), depthsBb: [depth],
      })).toBe(false);
    }
  });

  it('rejects impossible or non-finite qualification measurements without relaxing thresholds', () => {
    const source = acceptedManifest();
    const invalid: Array<Partial<PolicyManifest['validation']>> = [
      { exploitabilityEstimateBb: -0.01 }, { exploitabilityUpper99Bb: -0.01 },
      { exploitabilityEstimateBb: NaN }, { exploitabilityUpper99Bb: Infinity },
      { exploitabilityEstimateBb: 0.4, exploitabilityUpper99Bb: 0.3 },
      { crossSeedFrequencyMae: -0.01 }, { maximumAggregateActionDelta: -0.01 },
      { primaryActionAgreement: Infinity }, { primaryActionAgreement: 1.01 },
      { policyCoverage: 1.01 }, { actionEvStandardErrorCoverage: 1.01 },
      { projectedStorageBytes: -1 }, { projectedStorageBytes: 0.5 },
    ];
    for (const measurement of invalid) {
      expect(isValidatedFullHandManifest({ ...source, validation: { ...source.validation, ...measurement } })).toBe(false);
      if (!('exploitabilityEstimateBb' in measurement) && !('exploitabilityUpper99Bb' in measurement)) {
        expect(isExperimentalFullHandManifest({ ...source, label: 'Experimental self-play',
          validation: { ...source.validation, exploitabilityGateDeferred: true, ...measurement } })).toBe(false);
      }
    }
  });

  it('does not serve raked or otherwise incompatible games as Home game', () => {
    const manifest = acceptedManifest();
    for (const abstraction of [
      { ...manifest.abstraction, rake: '5% capped at 3bb' },
      { ...manifest.abstraction, anteBb: 1 },
      { ...manifest.abstraction, blindsBb: [1, 2] },
      { ...manifest.abstraction, blindsBb: [0.5, 1, 2] },
      undefined,
    ]) {
      expect(isValidatedFullHandManifest({ ...manifest, abstraction })).toBe(false);
      expect(isExperimentalFullHandManifest({
        ...manifest,
        label: 'Experimental self-play',
        abstraction,
        validation: { ...manifest.validation, exploitabilityGateDeferred: true },
      })).toBe(false);
    }
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

  it('serves an experimental resolver only when exploitability is explicitly deferred and every normal gate passes', () => {
    const source = acceptedManifest();
    const experimental: PolicyManifest = {
      ...source,
      label: 'Experimental self-play',
      runtime: {
        kind: 'rust-continual-resolver-v1',
        endpoint: '/api/practice/resolve',
        artifactFiles: {
          networks: 'networks.json.gz',
          rangePolicy: 'range-policy.json.gz',
          preflopActionValues: 'preflop-action-values.json.gz',
          flopValueNetwork: 'flop-value-network.json.gz',
        },
        networkSha256: 'a'.repeat(64),
        rangePolicySha256: 'b'.repeat(64),
        valueNetworkSha256: 'c'.repeat(64),
        preflopActionValuesSha256: 'd'.repeat(64),
        stateFeatureSchema: 'hu-cash-trajectory-poker-aware-v4',
        rangeFeatureSchema: 'rank-suit-invariant-combo-policy-query-v2',
        actionFeatureSchema: 'hu-cash-legal-action-v1',
        actionAbstraction:
          source.runtime?.kind === 'neural-deep-cfr-v1'
            ? source.runtime.actionAbstraction
            : (() => {
                throw new Error('test source must use a neural runtime');
              })(),
        dcfr: {
          positiveRegretExponent: 1.5,
          negativeRegretExponent: 0,
          strategyExponent: 0,
        },
        resolver: {
          flopIterations: 2,
          flopResolvedActor: 1,
          flopDeploySolvedPolicy: true,
          turnIterations: 2,
          turnResolvedActor: 1,
          riverIterations: 2,
          riverResolvedActor: 1,
          riverSafeResolving: false,
          riverSafeMaxmargin: false,
          riverSafeIterations: null,
          riverSafeResolvedActor: null,
          deterministic: true,
        },
      },
      validation: {
        ...source.validation,
        exploitabilityGateDeferred: true,
      },
    };
    expect(isExperimentalFullHandManifest(experimental)).toBe(true);
    const resolverRuntime = experimental.runtime;
    if (resolverRuntime?.kind !== 'rust-continual-resolver-v1') {
      throw new Error('test manifest must use a continual resolver');
    }
    expect(
      isExperimentalFullHandManifest({
        ...experimental,
        runtime: {
          ...resolverRuntime,
          artifactFiles: {
            ...resolverRuntime.artifactFiles,
            preflopActionValues: '../unverified.json.gz',
          },
        },
      })
    ).toBe(false);
    expect(
      isExperimentalFullHandManifest({
        ...experimental,
        runtime: {
          ...resolverRuntime,
          resolver: {
            ...resolverRuntime.resolver,
            riverResolvedActor: null,
            riverSafeResolving: true,
            riverSafeMaxmargin: true,
            riverSafeIterations: 16,
            riverSafeResolvedActor: 0,
          },
        },
      })
    ).toBe(true);
    expect(
      isExperimentalFullHandManifest({
        ...experimental,
        runtime: {
          ...resolverRuntime,
          resolver: {
            ...resolverRuntime.resolver,
            riverSafeMaxmargin: true,
          },
        },
      })
    ).toBe(false);
    const { riverSafeResolving: _missingSafeMode, ...incompleteSafety } =
      resolverRuntime.resolver;
    expect(
      isExperimentalFullHandManifest({
        ...experimental,
        runtime: {
          ...resolverRuntime,
          resolver: incompleteSafety,
        },
      })
    ).toBe(false);
    const { dcfr: _missingDcfr, ...incompleteDcfrRuntime } = resolverRuntime;
    expect(
      isExperimentalFullHandManifest({
        ...experimental,
        runtime: incompleteDcfrRuntime,
      })
    ).toBe(false);
    const { flopDeploySolvedPolicy: _missingDeployment, ...incompleteSolver } =
      resolverRuntime.resolver;
    expect(
      isExperimentalFullHandManifest({
        ...experimental,
        runtime: {
          ...resolverRuntime,
          resolver: incompleteSolver,
        },
      })
    ).toBe(false);
    expect(
      isExperimentalFullHandManifest({
        ...experimental,
        runtime: {
          ...resolverRuntime,
          resolver: {
            ...resolverRuntime.resolver,
            riverResolvedActor: 2,
          },
        },
      })
    ).toBe(false);
    const { turnResolvedActor: _missing, ...incompleteRouting } =
      resolverRuntime.resolver;
    expect(
      isExperimentalFullHandManifest({
        ...experimental,
        runtime: {
          ...resolverRuntime,
          resolver: incompleteRouting,
        },
      })
    ).toBe(false);
    expect(
      isExperimentalFullHandManifest({
        ...experimental,
        validation: {
          ...experimental.validation,
          crossSeedFrequencyMae: 0.0501,
        },
      })
    ).toBe(false);
    expect(
      isExperimentalFullHandManifest({
        ...experimental,
        validation: {
          ...experimental.validation,
          exploitabilityGateDeferred: false,
        },
      })
    ).toBe(false);
  });
});

describe('practice model storage and warming', () => {
  it('stores, retrieves, and warms manifests in local storage and memory', async () => {
    const {
      getStoredPracticeManifests,
      storePracticeManifests,
      warmPracticeModels,
      activePracticeManifests,
      PRACTICE_MANIFESTS_STORAGE_KEY,
    } = await import('@/lib/practice-models');

    const embedded = activePracticeManifests();
    expect(embedded.length).toBeGreaterThanOrEqual(2);
    expect(embedded.some((m) => m.subtype === 'full-hand')).toBe(true);

    const warmed = warmPracticeModels();
    expect(warmed).toEqual(embedded);

    const retrieved = getStoredPracticeManifests();
    expect(retrieved).toEqual(embedded);

    // Verify stored in localStorage when window is available
    if (typeof window !== 'undefined') {
      const storedJson = window.localStorage.getItem(PRACTICE_MANIFESTS_STORAGE_KEY);
      expect(storedJson).not.toBeNull();
      const parsed = JSON.parse(storedJson!);
      expect(parsed.length).toBe(embedded.length);
    }
  });
});
