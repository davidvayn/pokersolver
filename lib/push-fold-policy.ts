import solvedScenarios from '@/data/preflop/solved-scenarios.json';
import type { CompactPushFoldScenario } from '@/data/preflop/artifacts/types';
import {
  applyAction,
  canonicalPolicyHash,
  createHand,
  handBucket,
  type CreateHandOptions,
} from '@/lib/practice-engine';
import type {
  HandState,
  LegalAction,
  PolicyAction,
  PolicyNode,
  PushFoldDepth,
  Seat,
} from '@/lib/practice-types';

interface ScenarioIndex {
  scenario: CompactPushFoldScenario;
  hands: Map<string, { shove: number; call: number }>;
}

const indexes = new Map<number, ScenarioIndex>(
  (solvedScenarios as CompactPushFoldScenario[]).map((scenario) => [
    scenario.effective_stack_bb,
    {
      scenario,
      hands: new Map(
        scenario.hands.map(([label, shove, call]) => [label, { shove, call }])
      ),
    },
  ])
);

export interface PushFoldSpot {
  state: HandState;
  node: PolicyNode;
  scenario: CompactPushFoldScenario;
  replayActions: HandState['actionHistory'];
}

export function pushFoldDepths(): PushFoldDepth[] {
  return [...indexes.keys()].sort((first, second) => first - second) as PushFoldDepth[];
}

function indexFor(depthBb: number): ScenarioIndex {
  const index = indexes.get(depthBb);
  if (!index) throw new Error(`No accepted push/fold model at ${depthBb}bb`);
  return index;
}

function frequenciesFor(
  index: ScenarioIndex,
  state: HandState,
  seat: Seat
): { shove: number; call: number } {
  const label = handBucket(state.holeCards[seat]);
  const frequency = index.hands.get(label);
  if (!frequency) throw new Error(`Policy has no hand class ${label}`);
  return frequency;
}

function unavailableValue(
  action: LegalAction,
  probability: number
): PolicyAction {
  return {
    ...action,
    probability,
    evBb: null,
    standardErrorBb: null,
    confidence: 'unavailable',
  };
}

async function nodeFor(
  index: ScenarioIndex,
  state: HandState,
  hero: Seat
): Promise<PolicyNode> {
  const frequencies = frequenciesFor(index, state, hero);
  const actions: PolicyAction[] =
    hero === 'button-small-blind'
      ? [
          unavailableValue({ id: 'fold', kind: 'fold', label: 'Fold' }, 1 - frequencies.shove),
          unavailableValue(
            {
              id: 'all-in',
              kind: 'all-in',
              label: `All-in ${state.depthBb}bb`,
              amountToBb: state.depthBb,
            },
            frequencies.shove
          ),
        ]
      : [
          unavailableValue({ id: 'fold', kind: 'fold', label: 'Fold' }, 1 - frequencies.call),
          unavailableValue({ id: 'call', kind: 'call', label: `Call ${Math.min(state.stacksBb[hero], state.depthBb - 1).toFixed(1)}bb` }, frequencies.call),
        ];
  return {
    stateHash: await canonicalPolicyHash(state, hero),
    actions,
    bestActionId: null,
    bestActionEvBb: null,
  };
}

export async function createPushFoldSpot(options: {
  depthBb: PushFoldDepth;
  hero: Seat;
  handNumber: number;
  random?: () => number;
}): Promise<PushFoldSpot> {
  const random = options.random ?? Math.random;
  const index = indexFor(options.depthBb);
  const base: Omit<CreateHandOptions, 'id'> = {
    modelVersion: index.scenario.artifact_id,
    depthBb: options.depthBb,
    button: 'button-small-blind',
    hero: options.hero,
    random,
  };

  for (let attempt = 0; attempt < 10_000; attempt++) {
    let state = createHand({
      ...base,
      id: `pf-${options.handNumber}-${attempt}-${index.scenario.config_hash}`,
    });
    const replayActions: HandState['actionHistory'] = [];
    if (options.hero === 'big-blind') {
      const villain = frequenciesFor(index, state, 'button-small-blind');
      if (random() >= villain.shove) continue;
      state = applyAction(state, {
        id: 'all-in',
        kind: 'all-in',
        label: `All-in ${state.depthBb}bb`,
        amountToBb: state.depthBb,
      });
      replayActions.push(...state.actionHistory);
    }
    return {
      node: await nodeFor(index, state, options.hero),
      state,
      scenario: index.scenario,
      replayActions,
    };
  }
  throw new Error('Could not sample a reachable big-blind push/fold decision');
}

export function finishPushFoldHand(
  state: HandState,
  chosen: LegalAction,
  random: () => number = Math.random
): HandState {
  const index = indexFor(state.depthBb);
  let next = applyAction(state, chosen);
  if (next.terminal || state.hero === 'big-blind') return next;
  const bigBlind = frequenciesFor(index, next, 'big-blind');
  if (random() < bigBlind.call) {
    next = applyAction(next, {
      id: 'call',
      kind: 'call',
      label: `Call ${next.stacksBb['big-blind'].toFixed(1)}bb`,
    });
  } else {
    next = applyAction(next, { id: 'fold', kind: 'fold', label: 'Fold' });
  }
  return next;
}

export function pushFoldModelSummary(depthBb: number): CompactPushFoldScenario {
  return indexFor(depthBb).scenario;
}
