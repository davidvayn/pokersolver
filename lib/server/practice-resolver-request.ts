import { totalPotBb } from '@/lib/practice-engine';
import type { HandState, Seat } from '@/lib/practice-types';

const seatIndex = (seat: Seat): number =>
  seat === 'button-small-blind' ? 0 : 1;

/**
 * Keep every continuation of the user's first preflop choice on the same
 * resolver worker. Its Rust process retains the resolved public subtree, so
 * postflop descendants can reuse that work instead of starting over.
 */
export function resolverAffinityKey(state: HandState): string {
  const firstHeroAction = state.actionHistory.findIndex(
    (action) => action.actor === state.hero
  );
  const branch = state.actionHistory
    .slice(0, firstHeroAction < 0 ? state.actionHistory.length : firstHeroAction + 1)
    .map((action) =>
      [action.street, action.actor, action.kind, action.amountToBb ?? ''].join(':')
    )
    .join('/');
  return `${state.id}|${branch || 'root'}`;
}

/**
 * Convert a browser hand snapshot into the exact Rust replay request. Only
 * the acting player's private cards cross this boundary; the policy engine
 * reconstructs the opponent range from the public trajectory.
 */
export function resolverQueryPayload(
  state: HandState,
  stateHash: string,
  modelVersion: string,
  depthBb: number
): Record<string, unknown> {
  if (!state.toAct || state.terminal) {
    throw new Error('A resolver query requires a live decision');
  }
  const actor = state.toAct;
  return {
    stateHash,
    modelVersion,
    depthBb,
    privateCards: state.holeCards[actor],
    board: state.board,
    street: state.street,
    actor: seatIndex(actor),
    totalPotBb: totalPotBb(state),
    stacksBb: [
      state.stacksBb['button-small-blind'],
      state.stacksBb['big-blind'],
    ],
    streetBetsBb: [
      state.streetBetsBb['button-small-blind'],
      state.streetBetsBb['big-blind'],
    ],
    totalCommittedBb: [
      state.totalCommittedBb['button-small-blind'],
      state.totalCommittedBb['big-blind'],
    ],
    lastFullRaiseBb: state.lastFullRaiseBb,
    raiseReopened: state.raiseReopened,
    actions: state.actionHistory.map((action) => ({
      actor: seatIndex(action.actor),
      street: action.street,
      kind: action.kind.replace('-', '_'),
      amountToBb: action.amountToBb,
    })),
  };
}
