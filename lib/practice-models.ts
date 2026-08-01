import solvedScenarios from '@/data/preflop/solved-scenarios.json';
import type { CompactPushFoldScenario } from '@/data/preflop/artifacts/types';
import type { PolicyManifest } from '@/lib/practice-types';

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
      'Showdown equities are deterministic Monte Carlo estimates; this corpus does not contain per-action EV estimates.',
    ],
  },
};

// Full-hand manifests are deliberately empty until an independently validated
// two-seed artifact passes every activation gate. Do not add placeholder depths.
export const ACTIVE_FULL_HAND_MANIFESTS: PolicyManifest[] = [];

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
