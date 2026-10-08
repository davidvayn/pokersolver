import { cashGameRulesDigest } from '@/lib/cash-game-rules';
import type { CashGameRules } from '@/lib/cash-game-rules';
import { assertCashConservation } from '@/lib/practice-cash';
import { canonicalPolicyHash, totalPotBb } from '@/lib/practice-engine';
import type { HandState, Seat } from '@/lib/practice-types';

/** An inactive native-research contract, deliberately not a servable manifest. */
export interface CashPolicyIdentity {
  modelVersion: string;
  depthBb: number;
  rules: CashGameRules;
  rulesSha256: string;
  networkSha256: string;
}

export interface CashPolicyQuery {
  schema: 'hu-cash-average-policy-query-v1';
  rulesSha256: string;
  networkSha256: string;
  query: {
    requestId: string;
    stateHash: string;
    modelVersion: string;
    depthBb: number;
    privateCards: [number, number];
    board: number[];
    street: HandState['street'];
    actor: number;
    totalPotBb: number;
    stacksBb: [number, number];
    streetBetsBb: [number, number];
    totalCommittedBb: [number, number];
    lastFullRaiseBb: number;
    raiseReopened: boolean;
    actions: Array<{ actor: number; street: HandState['street']; kind: string; amountToBb?: number }>;
  };
}

/** Serialize only the acting hand and public line, never deck/opponent cards. */
export async function cashPolicyQueryPayload(
  state: HandState,
  identity: CashPolicyIdentity,
  requestId: string
): Promise<CashPolicyQuery> {
  if (!state.cash || state.terminal || !state.toAct || state.cash.houseRakeUnits !== 0) {
    throw new Error('Cash queries require a live, unsettled rules-pinned hand');
  }
  assertCashConservation(state);
  if (!requestId.trim() || requestId.length > 128 || !identity.modelVersion.trim()
    || identity.modelVersion.length > 128 || identity.modelVersion.includes('|')
    || state.modelVersion !== identity.modelVersion || state.depthBb !== identity.depthBb
    || !/^[a-f0-9]{64}$/.test(identity.networkSha256)
    || await cashGameRulesDigest(state.cash.rules) !== identity.rulesSha256
    || await cashGameRulesDigest(identity.rules) !== identity.rulesSha256) {
    throw new Error('Cash hand and pinned rules, depth or artifact identity differ');
  }
  if (state.actionHistory.length > 32) throw new Error('Cash action history exceeds the frozen feature contract');
  const seat = (actor: Seat) => actor === 'button-small-blind' ? 0 : 1;
  return {
    schema: 'hu-cash-average-policy-query-v1', rulesSha256: identity.rulesSha256,
    networkSha256: identity.networkSha256,
    query: {
      requestId, stateHash: await canonicalPolicyHash(state), modelVersion: identity.modelVersion,
      depthBb: identity.depthBb, privateCards: [...state.holeCards[state.toAct]], board: [...state.board],
      street: state.street, actor: seat(state.toAct), totalPotBb: totalPotBb(state),
      stacksBb: [state.stacksBb['button-small-blind'], state.stacksBb['big-blind']],
      streetBetsBb: [state.streetBetsBb['button-small-blind'], state.streetBetsBb['big-blind']],
      totalCommittedBb: [state.totalCommittedBb['button-small-blind'], state.totalCommittedBb['big-blind']],
      lastFullRaiseBb: state.lastFullRaiseBb, raiseReopened: state.raiseReopened,
      actions: state.actionHistory.map((action) => ({ actor: seat(action.actor), street: action.street,
        kind: action.kind.replace('-', '_'), amountToBb: action.amountToBb })),
    },
  };
}
