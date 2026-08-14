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

interface HandClassPolicy {
  shove: number;
  call: number;
  smallBlindFoldEvBb: number;
  smallBlindShoveEvBb: number;
  bigBlindFoldEvBb: number;
  bigBlindCallEvBb: number;
}

interface ScenarioIndex {
  scenario: CompactPushFoldScenario;
  hands: Map<string, HandClassPolicy>;
}

const indexes = new Map<number, ScenarioIndex>(
  (solvedScenarios as CompactPushFoldScenario[]).map((scenario) => {
    const values = new Map(
      scenario.action_values.map(
        ([
          label,
          smallBlindFoldEvBb,
          smallBlindShoveEvBb,
          bigBlindFoldEvBb,
          bigBlindCallEvBb,
        ]) => [
          label,
          {
            smallBlindFoldEvBb,
            smallBlindShoveEvBb,
            bigBlindFoldEvBb,
            bigBlindCallEvBb,
          },
        ]
      )
    );
    return [
      scenario.effective_stack_bb,
      {
        scenario,
        hands: new Map(
          scenario.hands.map(([label, shove, call]) => {
            const actionValues = values.get(label);
            if (!actionValues) {
              throw new Error(`Push/fold model has no action values for ${label}`);
            }
            return [label, { shove, call, ...actionValues }];
          })
        ),
      },
    ];
  })
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
): HandClassPolicy {
  const label = handBucket(state.holeCards[seat]);
  const frequency = index.hands.get(label);
  if (!frequency) throw new Error(`Policy has no hand class ${label}`);
  return frequency;
}

function estimatedValue(
  action: LegalAction,
  probability: number,
  evBb: number,
  standardErrorBb: number
): PolicyAction {
  return {
    ...action,
    probability,
    evBb,
    standardErrorBb,
    confidence: 'low',
  };
}

async function nodeFor(
  index: ScenarioIndex,
  state: HandState,
  hero: Seat
): Promise<PolicyNode> {
  const frequencies = frequenciesFor(index, state, hero);
  const sampledValueError = index.scenario.action_value_standard_error_upper_bound_bb;
  const actions: PolicyAction[] =
    hero === 'button-small-blind'
      ? [
          estimatedValue(
            { id: 'fold', kind: 'fold', label: 'Fold' },
            1 - frequencies.shove,
            frequencies.smallBlindFoldEvBb,
            0
          ),
          estimatedValue(
            {
              id: 'all-in',
              kind: 'all-in',
              label: `All-in ${state.depthBb}bb`,
              amountToBb: state.depthBb,
            },
            frequencies.shove,
            frequencies.smallBlindShoveEvBb,
            sampledValueError
          ),
        ]
      : [
          estimatedValue(
            { id: 'fold', kind: 'fold', label: 'Fold' },
            1 - frequencies.call,
            frequencies.bigBlindFoldEvBb,
            0
          ),
          estimatedValue(
            {
              id: 'call',
              kind: 'call',
              label: `Call ${Math.min(state.stacksBb[hero], state.depthBb - 1).toFixed(1)}bb`,
            },
            frequencies.call,
            frequencies.bigBlindCallEvBb,
            sampledValueError
          ),
        ];
  const best = actions.reduce(
    (current, action) =>
      current === null || (action.evBb ?? -Infinity) > (current.evBb ?? -Infinity)
        ? action
        : current,
    null as PolicyAction | null
  );
  return {
    stateHash: await canonicalPolicyHash(state, hero),
    actions,
    bestActionId: best?.id ?? null,
    bestActionEvBb: best?.evBb ?? null,
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
