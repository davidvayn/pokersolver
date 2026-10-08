import { describe, expect, it } from 'vitest';
import { NL25_STUDY_RULES } from '@/lib/cash-game-rules';
import { NL25_RULES_SHA256 } from '@/lib/practice-game-identity';
import { applyAction, canonicalPolicyState, createHand, seededRandom } from '@/lib/practice-engine';
import { cashPolicyQueryPayload } from '@/lib/cash-policy-query';
import type { CashPolicyIdentity } from '@/lib/cash-policy-query';
import { neuralLegalActions } from '@/lib/neural-policy';
import type { ActionAbstraction } from '@/lib/practice-types';
import queryFixtures from '@/data/practice/cash-query-fixtures.json';

const identity: CashPolicyIdentity = { modelVersion: 'cash-pilot-test', depthBb: 20,
  rules: NL25_STUDY_RULES, rulesSha256: NL25_RULES_SHA256, networkSha256: 'a'.repeat(64) };
function hand() {
  return createHand({ id: 'cash-query', modelVersion: identity.modelVersion, depthBb: 20,
    button: 'button-small-blind', hero: 'button-small-blind', cashRules: identity.rules, random: seededRandom(7) });
}

describe('inactive cash average-policy query adapter', () => {
  it('remembers the flop and later card arrivals without distinguishing flop permutations', () => {
    const state = { ...hand(),street:'river' as const,board:[8,13,22,31,40] };
    expect(canonicalPolicyState(state)).not.toBe(canonicalPolicyState({ ...state,board:[8,13,31,22,40] }));
    expect(canonicalPolicyState(state)).not.toBe(canonicalPolicyState({ ...state,board:[8,13,22,40,31] }));
    expect(canonicalPolicyState(state)).toBe(canonicalPolicyState({ ...state,board:[22,8,13,31,40] }));
  });
  it('offers the native cent-aligned sizing grid and clamps duplicate undersized bets', () => {
    const grid: ActionAbstraction = { openSizesBb: [2, 2.5, 3], limpRaiseSizesBb: [3,4,5],
      threeBetSizesBb: [7.5,9,11], fourBetSizesBb: [18,22,26], deeperRaisePotFractions: [.75,1,1.25],
      preflopRaiseCap: 4, flopBetPotFractions: [1/3,.4,.75], turnRiverBetPotFractions: [.5,1],
      postflopRaisePotFractions: [1], postflopRaiseCap: 1, includeAllIn: true };
    const state = hand();
    expect(neuralLegalActions(state,grid).map((a) => a.amountToBb ?? a.kind)).toEqual(['fold','call',2,2.48,3,20]);
    const flop = applyAction(applyAction(state,{ id:'call',kind:'call',label:'Call' }),{ id:'check',kind:'check',label:'Check' });
    expect(neuralLegalActions(flop,grid).map((a) => a.amountToBb ?? a.kind)).toEqual(['check',1,1.52,19]);
    expect(neuralLegalActions(state,grid).find((a) => a.amountToBb === 2.48)?.label).toBe('Raise to 2.48bb');
    const opened = applyAction(state,{ id:'raise',kind:'raise',label:'Raise',amountToBb:2.48 });
    expect(neuralLegalActions(opened,grid).find((a) => a.kind === 'call')?.label).toBe('Call 1.48bb');
  });
  it('pins rules and weights without sending hidden cards, deck or an unbounded ledger', async () => {
    const state = hand();
    const payload = await cashPolicyQueryPayload(state, identity, 'request-1');
    expect(payload).toMatchObject({ schema: 'hu-cash-average-policy-query-v1', rulesSha256: NL25_RULES_SHA256,
      networkSha256: 'a'.repeat(64), query: { actor: 0, totalPotBb: 1.4, streetBetsBb: [.4, 1] } });
    expect(payload.query.privateCards).toEqual(state.holeCards['button-small-blind']);
    expect(payload).not.toHaveProperty('cash');
    expect(payload.query).not.toHaveProperty('holeCards');
    expect(payload.query).not.toHaveProperty('deck');
    expect((await cashPolicyQueryPayload({ ...state, deck: [...state.deck].reverse(),
      holeCards: { ...state.holeCards, 'big-blind': [0, 1] } }, identity, 'request-2')).query.stateHash).toBe(payload.query.stateHash);
  });

  it('rejects wrong profiles, artifacts, depths, malformed ledgers and stopped hands', async () => {
    const state = hand();
    for (const broken of [{ ...identity, depthBb: 40 }, { ...identity, rulesSha256: '0'.repeat(64) },
      { ...identity, networkSha256: '../unsafe' }, { ...identity, modelVersion: 'Home' }]) {
      await expect(cashPolicyQueryPayload(state, broken, 'request')).rejects.toThrow();
    }
    await expect(cashPolicyQueryPayload({ ...state, cash: undefined }, identity, 'request')).rejects.toThrow('live');
    await expect(cashPolicyQueryPayload({ ...state, terminal: true }, identity, 'request')).rejects.toThrow('live');
    await expect(cashPolicyQueryPayload({ ...state, potBb: 1 }, identity, 'request')).rejects.toThrow('disagree');
  });

  it('uses the same cash v3 root and limp hash strings as native exact replay', async () => {
    const state = { ...hand(), holeCards: { 'button-small-blind': [48,49] as [number,number], 'big-blind': [4,5] as [number,number] } };
    expect(canonicalPolicyState(state)).toContain('|20.000|button-small-blind|preflop|48,49||0.000|0.400|1.000|19.600|19.000|');
    const called = applyAction(state, { id: 'call', kind: 'call', label: 'Call' });
    expect(canonicalPolicyState(called)).toContain('preflop:button-small-blind:call:0.6');
    expect((await cashPolicyQueryPayload(called, identity, 'limp')).query.actions).toEqual([
      { actor: 0, street: 'preflop', kind: 'call', amountToBb: undefined },
    ]);
    const flop = { ...applyAction(called,{ id:'check',kind:'check',label:'Check' }), board: [8,13,22] };
    const check = { id:'check',kind:'check' as const,label:'Check' };
    const turn = { ...applyAction(applyAction(flop,check),check),board:[8,13,22,31] };
    const river = { ...applyAction(applyAction(applyAction(applyAction(flop,check),check),check),check),board:[8,13,22,31,40] };
    for (const [index,snapshot] of [state,called,flop,river,turn].entries()) {
      const query = (await cashPolicyQueryPayload(snapshot,identity,queryFixtures.cases[index].query.requestId)).query;
      expect(query.stateHash).toBe(queryFixtures.cases[index].stateHash);
      // JSON omits undefined action sizes; native nullable sizes mean the same.
      expect(JSON.parse(JSON.stringify(query.actions))).toEqual(queryFixtures.cases[index].query.actions.map((a) =>
        ({ actor:a.actor,street:a.street,kind:a.kind })));
    }
  });
});
