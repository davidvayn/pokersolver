import solvedScenarios from '@/data/preflop/solved-scenarios.json';
import fullHandManifests from '@/data/practice/full-hand-manifests.json';
import type { CompactPushFoldScenario } from '@/data/preflop/artifacts/types';
import type { ActionAbstraction, PolicyManifest } from '@/lib/practice-types';

const scenarios = solvedScenarios as CompactPushFoldScenario[];
const newestGeneratedAt = Math.max(
  ...scenarios.map((scenario) => scenario.generated_at_unix_seconds)
);

export const PUSH_FOLD_MANIFEST: PolicyManifest = {
  schemaVersion: 1,
  version: 'hu-push-fold-v1',
  model: 'heads-up-push-fold-monte-carlo-v1',
  label: 'Approximate GTO',
  subtype: 'push-fold',
  active: true,
  depthsBb: scenarios.map((scenario) => scenario.effective_stack_bb),
  generatedAt: new Date(newestGeneratedAt * 1000).toISOString(),
  stateSchema: 'hu-push-fold-hand-class-v1',
  shardSchema: 'embedded-compact-json-v1',
  runtime: { kind: 'binary-policy-shards-v1' },
  abstraction: {
    blindsBb: [0.5, 1],
    anteBb: 0,
    rake: 'none',
    actionSizing: 'fold/all-in; fold/call response',
    cardAbstraction: '169 preflop hand classes; exact-card removal during deals',
    recall: 'single preflop decision',
  },
  validation: {
    status: 'accepted',
    exploitabilityEstimateBb: Math.max(
      ...scenarios.map((scenario) => scenario.exploitability_bb)
    ),
    notes: [
      'All eight bundled depths pass the v1 finite-metric, probability-sum, sanity, and advisory exploitability checks.',
      'Per-action EV and EV-loss feedback is evaluated against the served 169-class policy with deterministic Monte Carlo showdown equity.',
      'Called-action uncertainty uses the conservative sampled-payoff standard-error upper bound and is labeled low confidence.',
    ],
  },
};

export function isValidatedFullHandManifest(
  value: unknown
): value is PolicyManifest {
  if (!value || typeof value !== 'object') return false;
  const manifest = value as Partial<PolicyManifest>;
  const validation = manifest.validation;
  if (
    manifest.schemaVersion !== 1 ||
    manifest.subtype !== 'full-hand' ||
    manifest.label !== 'Approximate GTO' ||
    manifest.active !== true ||
    typeof manifest.version !== 'string' ||
    !Array.isArray(manifest.depthsBb) ||
    manifest.depthsBb.length === 0 ||
    !manifest.depthsBb.every((depth) => [20, 50, 100].includes(depth)) ||
    validation?.status !== 'accepted'
  ) {
    return false;
  }
  if (!validFullHandRuntime(manifest.runtime)) return false;
  return (
    typeof validation.exploitabilityEstimateBb === 'number' &&
    validation.exploitabilityEstimateBb <= 0.05 &&
    typeof validation.exploitabilityUpper99Bb === 'number' &&
    validation.exploitabilityUpper99Bb <= 0.1 &&
    normalFullHandGatesPass(validation)
  );
}

function validActionAbstraction(action: ActionAbstraction | undefined): boolean {
  const grids = action
    ? [
        action.openSizesBb,
        action.limpRaiseSizesBb,
        action.threeBetSizesBb,
        action.fourBetSizesBb,
        action.deeperRaisePotFractions,
        action.flopBetPotFractions,
        action.turnRiverBetPotFractions,
        action.postflopRaisePotFractions,
      ]
    : [];
  return Boolean(
    action &&
      grids.length === 8 &&
      grids.every(
        (grid) =>
          Array.isArray(grid) &&
          grid.length > 0 &&
          grid.every((number) => Number.isFinite(number) && number > 0) &&
          grid.every((number, index) => index === 0 || grid[index - 1] < number)
      ) &&
      Number.isInteger(action.preflopRaiseCap) &&
      Number.isInteger(action.postflopRaiseCap) &&
      typeof action.includeAllIn === 'boolean'
  );
}

function validFullHandRuntime(runtime: PolicyManifest['runtime']): boolean {
  if (runtime?.kind === 'neural-deep-cfr-v1') {
    const action = runtime.actionAbstraction;
    const adaptation = runtime.adaptation;
    if (
      typeof runtime.artifactUrl !== 'string' ||
      !runtime.artifactUrl.startsWith('/models/practice/') ||
      !/^[a-f0-9]{64}$/.test(runtime.artifactSha256) ||
      runtime.stateFeatureSchema !== 'hu-cash-trajectory-poker-aware-v4' ||
      runtime.actionFeatureSchema !== 'hu-cash-legal-action-v1' ||
      runtime.opponentProfileSchema !== 'local-opponent-profile-v1' ||
      !adaptation ||
      !Number.isInteger(adaptation.minimumObservations) ||
      !Number.isInteger(adaptation.fullConfidenceObservations) ||
      adaptation.fullConfidenceObservations <=
        adaptation.minimumObservations ||
      !Number.isFinite(adaptation.maximumResponseWeight) ||
      adaptation.maximumResponseWeight < 0 ||
      adaptation.maximumResponseWeight > 1 ||
      !validActionAbstraction(action)
    ) {
      return false;
    }
    return true;
  }
  if (runtime?.kind === 'rust-continual-resolver-v1') {
    return (
      runtime.endpoint === '/api/practice/resolve' &&
      /^[a-f0-9]{64}$/.test(runtime.networkSha256) &&
      /^[a-f0-9]{64}$/.test(runtime.rangePolicySha256) &&
      /^[a-f0-9]{64}$/.test(runtime.valueNetworkSha256) &&
      /^[a-f0-9]{64}$/.test(runtime.preflopActionValuesSha256) &&
      runtime.stateFeatureSchema === 'hu-cash-trajectory-poker-aware-v4' &&
      runtime.rangeFeatureSchema ===
        'rank-suit-invariant-combo-policy-query-v2' &&
      runtime.actionFeatureSchema === 'hu-cash-legal-action-v1' &&
      validActionAbstraction(runtime.actionAbstraction) &&
      Number.isInteger(runtime.resolver.flopIterations) &&
      runtime.resolver.flopIterations >= 2 &&
      Number.isInteger(runtime.resolver.turnIterations) &&
      runtime.resolver.turnIterations >= 2 &&
      Number.isInteger(runtime.resolver.riverIterations) &&
      runtime.resolver.riverIterations >= 2 &&
      runtime.resolver.deterministic === true
    );
  }
  return runtime?.kind === 'binary-policy-shards-v1';
}

function normalFullHandGatesPass(
  validation: PolicyManifest['validation']
): boolean {
  return (
    typeof validation.crossSeedFrequencyMae === 'number' &&
    validation.crossSeedFrequencyMae <= 0.05 &&
    typeof validation.primaryActionAgreement === 'number' &&
    validation.primaryActionAgreement >= 0.85 &&
    typeof validation.maximumAggregateActionDelta === 'number' &&
    validation.maximumAggregateActionDelta <= 0.03 &&
    typeof validation.policyCoverage === 'number' &&
    validation.policyCoverage >= 0.9999 &&
    typeof validation.actionEvStandardErrorCoverage === 'number' &&
    validation.actionEvStandardErrorCoverage >= 0.95 &&
    typeof validation.projectedStorageBytes === 'number' &&
    validation.projectedStorageBytes <= 20 * 1024 ** 3 &&
    validation.rawProbabilitySumsValid === true &&
    validation.quantizedProbabilitySumsValid === true &&
    validation.independentSeedCount === 2 &&
    Array.isArray(validation.trainingHoursPerSeed) &&
    validation.trainingHoursPerSeed.length === 2 &&
    validation.trainingHoursPerSeed.every(
      (hours) => Number.isFinite(hours) && hours >= 8 && hours <= 12
    )
  );
}

export function isExperimentalFullHandManifest(
  value: unknown
): value is PolicyManifest {
  if (!value || typeof value !== 'object') return false;
  const manifest = value as Partial<PolicyManifest>;
  return Boolean(
    manifest.schemaVersion === 1 &&
      manifest.subtype === 'full-hand' &&
      manifest.label === 'Experimental self-play' &&
      manifest.active === true &&
      typeof manifest.version === 'string' &&
      Array.isArray(manifest.depthsBb) &&
      manifest.depthsBb.length > 0 &&
      manifest.depthsBb.every((depth) => [20, 50, 100].includes(depth)) &&
      manifest.validation?.status === 'accepted' &&
      manifest.validation.exploitabilityGateDeferred === true &&
      validFullHandRuntime(manifest.runtime) &&
      normalFullHandGatesPass(manifest.validation)
  );
}

export function isServableFullHandManifest(
  value: unknown
): value is PolicyManifest {
  return (
    isValidatedFullHandManifest(value) || isExperimentalFullHandManifest(value)
  );
}

// This checked-in registry is the database-free activation boundary.
// Approximate-GTO entries pass every gate; explicitly experimental entries
// may defer exploitability but still have to pass every normal serving gate.
export const ACTIVE_FULL_HAND_MANIFESTS: PolicyManifest[] = (
  fullHandManifests as unknown[]
).filter(isServableFullHandManifest);

export function activePracticeManifests(): PolicyManifest[] {
  return [...ACTIVE_FULL_HAND_MANIFESTS, PUSH_FOLD_MANIFEST].filter(
    (manifest) => manifest.active && manifest.validation.status === 'accepted'
  );
}

export function activeFullHandDepths(): number[] {
  return [
    ...new Set(
      ACTIVE_FULL_HAND_MANIFESTS.flatMap((manifest) => manifest.depthsBb)
    ),
  ].sort((first, second) => first - second);
}

export function modelForFullDepth(depthBb: number): PolicyManifest | null {
  return (
    ACTIVE_FULL_HAND_MANIFESTS.find((manifest) =>
      manifest.depthsBb.includes(depthBb)
    ) ?? null
  );
}
