import solvedScenarios from '@/data/preflop/solved-scenarios.json';
import fullHandManifests from '@/data/practice/full-hand-manifests.json';
import type { CompactPushFoldScenario } from '@/data/preflop/artifacts/types';
import type { ActionAbstraction, PolicyManifest } from '@/lib/practice-types';
import { isFullHandDepth } from '@/lib/practice-types';
import { HOME_RULES_SHA256 } from '@/lib/practice-game-identity';

const scenarios = solvedScenarios as CompactPushFoldScenario[];
// Frozen in 20bb-v50-full-hand-candidate-freeze.json. Both the point estimate
// and its one-sided 99% upper bound must fit under this total full-game limit.
export const MAX_FULL_HAND_TOTAL_EXPLOITABILITY_BB = 0.5;
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
    manifest.cashGame !== undefined ||
    manifest.subtype !== 'full-hand' ||
    manifest.label !== 'Approximate GTO' ||
    manifest.active !== true ||
    typeof manifest.version !== 'string' ||
    !Array.isArray(manifest.depthsBb) ||
    manifest.depthsBb.length === 0 ||
    !manifest.depthsBb.every(isFullHandDepth) ||
    !hasHomeGameRules(manifest.abstraction) ||
    validation?.status !== 'accepted'
  ) {
    return false;
  }
  if (!validFullHandRuntime(manifest.runtime)) return false;
  return (
    finiteInRange(validation.exploitabilityEstimateBb, 0, MAX_FULL_HAND_TOTAL_EXPLOITABILITY_BB) &&
    finiteInRange(validation.exploitabilityUpper99Bb, validation.exploitabilityEstimateBb, MAX_FULL_HAND_TOTAL_EXPLOITABILITY_BB) &&
    normalFullHandGatesPass(validation)
  );
}

// All installed policies and this legacy serving route are still rake-free.
// Relabeling a manifest as NL20 must never make those policies serve a raked
// game. Raked profiles need their own settlement/training/evaluation support.
export function hasHomeGameRules(
  abstraction: PolicyManifest['abstraction'] | undefined
): boolean {
  return Boolean(
    abstraction &&
      abstraction.rake === 'none' &&
      abstraction.anteBb === 0 &&
      Array.isArray(abstraction.blindsBb) &&
      abstraction.blindsBb.length === 2 &&
      abstraction.blindsBb[0] === 0.5 &&
      abstraction.blindsBb[1] === 1
  );
}

export function isLegacyHomeManifest(manifest: Partial<PolicyManifest>): boolean {
  return manifest.cashGame === undefined && hasHomeGameRules(manifest.abstraction);
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
    const artifactFiles = runtime.artifactFiles;
    const requiredArtifactFiles = artifactFiles
      ? [
          artifactFiles.networks,
          artifactFiles.rangePolicy,
          artifactFiles.preflopActionValues,
          artifactFiles.flopValueNetwork,
        ]
      : [];
    const resolver = runtime.resolver;
    const dcfr = runtime.dcfr;
    const validResolvedActor = (actor: unknown) =>
      actor === null || actor === 0 || actor === 1;
    const validExponent = (value: unknown) =>
      typeof value === 'number' && Number.isFinite(value) && value >= 0;
    return (
      runtime.endpoint === '/api/practice/resolve' &&
      Object.keys(artifactFiles ?? {}).length === 4 &&
      requiredArtifactFiles.length === 4 &&
      requiredArtifactFiles.every(
        (file) =>
          typeof file === 'string' &&
          /^[a-z0-9][a-z0-9.-]*\.json\.gz$/.test(file)
      ) &&
      /^[a-f0-9]{64}$/.test(runtime.networkSha256) &&
      /^[a-f0-9]{64}$/.test(runtime.rangePolicySha256) &&
      /^[a-f0-9]{64}$/.test(runtime.valueNetworkSha256) &&
      /^[a-f0-9]{64}$/.test(runtime.preflopActionValuesSha256) &&
      runtime.stateFeatureSchema === 'hu-cash-trajectory-poker-aware-v4' &&
      runtime.rangeFeatureSchema ===
        'rank-suit-invariant-combo-policy-query-v2' &&
      runtime.actionFeatureSchema === 'hu-cash-legal-action-v1' &&
      validActionAbstraction(runtime.actionAbstraction) &&
      validExponent(dcfr?.positiveRegretExponent) &&
      validExponent(dcfr?.negativeRegretExponent) &&
      validExponent(dcfr?.strategyExponent) &&
      Number.isInteger(resolver?.flopIterations) &&
      resolver.flopIterations >= 2 &&
      validResolvedActor(resolver.flopResolvedActor) &&
      typeof resolver.flopDeploySolvedPolicy === 'boolean' &&
      Number.isInteger(resolver.turnIterations) &&
      resolver.turnIterations >= 2 &&
      validResolvedActor(resolver.turnResolvedActor) &&
      Number.isInteger(resolver.riverIterations) &&
      resolver.riverIterations >= 2 &&
      validResolvedActor(resolver.riverResolvedActor) &&
      typeof resolver.riverSafeResolving === 'boolean' &&
      typeof resolver.riverSafeMaxmargin === 'boolean' &&
      (resolver.riverSafeIterations === null ||
        (Number.isInteger(resolver.riverSafeIterations) &&
          resolver.riverSafeIterations >= resolver.riverIterations)) &&
      validResolvedActor(resolver.riverSafeResolvedActor) &&
      (resolver.riverSafeResolving ||
        (!resolver.riverSafeMaxmargin &&
          resolver.riverSafeIterations === null &&
          resolver.riverSafeResolvedActor === null)) &&
      (!resolver.riverSafeMaxmargin || resolver.riverSafeResolving) &&
      (resolver.riverSafeResolvedActor === null ||
        resolver.riverResolvedActor === null ||
        resolver.riverSafeResolvedActor === resolver.riverResolvedActor) &&
      resolver.deterministic === true
    );
  }
  return runtime?.kind === 'binary-policy-shards-v1';
}

function normalFullHandGatesPass(
  validation: PolicyManifest['validation']
): boolean {
  return (
    finiteInRange(validation.crossSeedFrequencyMae, 0, 0.05) &&
    finiteInRange(validation.primaryActionAgreement, 0.85, 1) &&
    finiteInRange(validation.maximumAggregateActionDelta, 0, 0.03) &&
    finiteInRange(validation.policyCoverage, 0.9999, 1) &&
    finiteInRange(validation.actionEvStandardErrorCoverage, 0.95, 1) &&
    typeof validation.projectedStorageBytes === 'number' &&
    Number.isSafeInteger(validation.projectedStorageBytes) &&
    validation.projectedStorageBytes >= 0 &&
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

function finiteInRange(value: unknown, minimum: number, maximum: number): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= minimum && value <= maximum;
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
      manifest.depthsBb.every(isFullHandDepth) &&
      isLegacyHomeManifest(manifest) &&
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

/** Exact hand identity; never select another version just because depth matches. */
export function modelForFullHandIdentity(
  version: string,
  depthBb: number,
  rulesSha256: string = HOME_RULES_SHA256
): PolicyManifest | null {
  const matches = ACTIVE_FULL_HAND_MANIFESTS.filter((manifest) =>
    manifest.version === version && manifest.depthsBb.includes(depthBb)
      && (manifest.cashGame?.rulesSha256 ?? HOME_RULES_SHA256) === rulesSha256
  );
  return matches.length === 1 ? matches[0] : null;
}

export const PRACTICE_MANIFESTS_STORAGE_KEY = 'poker_lab_practice_manifests_v1';

let memoryManifestsCache: PolicyManifest[] | null = null;

export function storePracticeManifests(manifests: PolicyManifest[]): void {
  const valid = manifests.filter(
    (m) =>
      m &&
      typeof m === 'object' &&
      m.active === true &&
      m.validation?.status === 'accepted' &&
      (m.subtype === 'push-fold' || isServableFullHandManifest(m))
  );
  if (valid.length === 0) return;
  memoryManifestsCache = valid;
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(
      PRACTICE_MANIFESTS_STORAGE_KEY,
      JSON.stringify(valid)
    );
  } catch {
    // Ignore storage quota or disabled storage in privacy modes.
  }
}

export function getStoredPracticeManifests(): PolicyManifest[] {
  if (memoryManifestsCache && memoryManifestsCache.length > 0) {
    return memoryManifestsCache;
  }
  const embedded = activePracticeManifests();
  if (typeof window === 'undefined') {
    memoryManifestsCache = embedded;
    return embedded;
  }
  try {
    const raw = window.localStorage.getItem(PRACTICE_MANIFESTS_STORAGE_KEY);
    if (raw) {
      const parsed: unknown = JSON.parse(raw);
      if (Array.isArray(parsed)) {
        const validated = parsed.filter(
          (m): m is PolicyManifest =>
            Boolean(
              m &&
                typeof m === 'object' &&
                (m as PolicyManifest).active === true &&
                (m as PolicyManifest).validation?.status === 'accepted' &&
                ((m as PolicyManifest).subtype === 'push-fold' ||
                  isServableFullHandManifest(m))
            )
        );
        const hasFullHand = validated.some((m) => m.subtype === 'full-hand');
        if (hasFullHand) {
          memoryManifestsCache = validated;
          return validated;
        }
      }
    }
  } catch {
    // Ignore read or parse errors and fall back to embedded manifests
  }
  storePracticeManifests(embedded);
  return embedded;
}

export function warmPracticeModels(): PolicyManifest[] {
  const manifests = getStoredPracticeManifests();
  storePracticeManifests(manifests);
  return manifests;
}
