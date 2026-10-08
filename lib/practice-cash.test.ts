import { describe, expect, it } from 'vitest';
import { HOME_GAME_RULES, NL25_STUDY_RULES } from '@/lib/cash-game-rules';
import { applyAction, assertChipConservation, canonicalPolicyState, createHand, engineLegalActions,
  seededRandom, stopForReview, totalPotBb } from '@/lib/practice-engine';
import type { HandState, LegalAction } from '@/lib/practice-types';
import { settleCashState } from '@/lib/practice-cash';

function hand(seed = 1, depthBb = 20): HandState {
  return createHand({ id: `raked-${seed}`, modelVersion: 'inactive-cash-test', depthBb,
    button: 'button-small-blind', hero: 'button-small-blind', random: seededRandom(seed), cashRules: NL25_STUDY_RULES });
}
const call: LegalAction = { id: 'call', kind: 'call', label: 'Call' };
const check: LegalAction = { id: 'check', kind: 'check', label: 'Check' };
const fold: LegalAction = { id: 'fold', kind: 'fold', label: 'Fold' };
function flop(state: HandState): HandState { return applyAction(applyAction(state, call), check); }

describe('rules-pinned cash hand engine', () => {
  it('posts 10/25 cents and pins deeply frozen rules without changing Home hashes', () => {
    const state = hand();
    expect(state.cash?.streetBetsUnits).toEqual({ 'button-small-blind': 10, 'big-blind': 25 });
    expect(state.stacksBb['button-small-blind']).toBe(19.6);
    expect(totalPotBb(state)).toBe(1.4);
    expect(Object.isFrozen(state.cash?.rules.rake)).toBe(true);
    assertChipConservation(state);
    const home = createHand({ modelVersion: 'inactive-cash-test', depthBb: 20,
      button: 'button-small-blind', hero: 'button-small-blind', random: seededRandom(1) });
    expect(canonicalPolicyState(home).startsWith('hu-cash-v1|')).toBe(true);
    expect(canonicalPolicyState(state).startsWith('hu-cash-v3|')).toBe(true);
    expect(canonicalPolicyState(state)).not.toEqual(canonicalPolicyState(home));
    const control = createHand({ modelVersion: 'inactive-cash-test', depthBb: 20,
      button: 'button-small-blind', hero: 'button-small-blind', random: seededRandom(1),
      cashRules: { ...NL25_STUDY_RULES, id: 'nl25-rake-off-control-v1', rake: { ...NL25_STUDY_RULES.rake, rateBasisPoints: 0, capUnits: 0 } } });
    expect(canonicalPolicyState(control)).not.toEqual(canonicalPolicyState(state));
  });

  it('refunds an uncalled preflop shove without charging rake', () => {
    const shoved = applyAction(hand(), { id: 'jam', kind: 'all-in', label: 'All-in', amountToBb: 20 });
    const finished = applyAction(shoved, fold);
    expect(finished.result?.cashSettlement).toMatchObject({ grossPotUnits: 50,
      uncalledRefundUnits: [475, 0], rakeUnits: 0, netPayoffUnits: [25, -25] });
    expect(finished.result?.netBb).toEqual({ 'button-small-blind': 1, 'big-blind': -1 });
    assertChipConservation(finished);
    expect(shoved.cash?.houseRakeUnits).toBe(0);
  });

  it('charges called preflop all-ins after the runout, including at deep caps', () => {
    for (const depth of [20, 40, 50, 100, 200, 1000, 2000]) {
      const shoved = applyAction(hand(7, depth), { id: 'jam', kind: 'all-in', label: 'All-in', amountToBb: depth });
      const finished = applyAction(shoved, call);
      expect(finished.board).toHaveLength(5);
      expect(finished.result?.reason).toBe('showdown');
      expect(finished.cash?.houseRakeUnits).toBe(depth === 20 ? 45 : 50);
      const values = finished.result!.netBb;
      expect(values['button-small-blind'] + values['big-blind']).toBeCloseTo(-finished.cash!.houseRakeUnits / 25, 10);
      assertChipConservation(finished);
    }
  });

  it('deducts rake from only matched postflop wagers, once, preserving the input', () => {
    const initial = flop(hand());
    const bet = applyAction(initial, { id: 'bet', kind: 'bet', label: 'Bet', amountToBb: 8 });
    const finished = applyAction(bet, fold);
    expect(initial.cash?.committedUnits).toEqual({ 'button-small-blind': 25, 'big-blind': 25 });
    expect(finished.result?.cashSettlement).toMatchObject({ grossPotUnits: 50,
      uncalledRefundUnits: [0, 200], rakeUnits: 2, netPotUnits: 48 });
    expect(finished.result?.netBb).toEqual({ 'button-small-blind': -1, 'big-blind': 0.92 });
    assertChipConservation(finished);
    expect(() => applyAction(finished, check)).toThrow('complete');
    expect(() => settleCashState(finished, 'big-blind', 'fold')).toThrow('already been settled');
  });

  it('splits an odd net cent according to the pinned study rule', () => {
    let state = applyAction(hand(), { id: 'open', kind: 'raise', label: 'Raise', amountToBb: 5 });
    state = applyAction(state, call);
    state = { ...state, board: [32, 36, 40, 44, 48], street: 'river',
      holeCards: { 'button-small-blind': [0, 5], 'big-blind': [8, 13] } };
    const finished = applyAction(applyAction(state, check), check);
    expect(finished.result?.winner).toBe('split');
    expect(finished.result?.cashSettlement).toMatchObject({ rakeUnits: 11, awardsUnits: [119, 120], netPayoffUnits: [-6, -5] });
    assertChipConservation(finished);
  });

  it('rejects sub-cent bets, fractional-cent stacks and inconsistent ledgers', () => {
    expect(() => hand(1, 20.01)).toThrow('money units');
    expect(() => applyAction(hand(), { id: 'bad', kind: 'raise', label: 'Raise', amountToBb: 2.5 })).toThrow('money units');
    const state = hand();
    expect(() => assertChipConservation({ ...state, stacksBb: { ...state.stacksBb, 'big-blind': 18 } })).toThrow('disagree');
  });

  it('does not charge a hypothetical rake when a practice review stops early', () => {
    const state = flop(hand());
    const stopped = stopForReview(state, 'review-complete');
    expect(stopped.result?.cashSettlement).toBeUndefined();
    expect(stopped.cash?.houseRakeUnits).toBe(0);
    assertChipConservation(stopped);
    expect(() => settleCashState(stopped, 'big-blind', 'fold')).toThrow('stopped for review');
  });

  it('conserves exact integer chips through deterministic authentic hands', () => {
    for (let seed = 1; seed <= 100; seed++) {
      let state = hand(seed);
      const random = seededRandom(seed * 31);
      while (!state.terminal) {
        const choices = engineLegalActions(state);
        state = applyAction(state, choices[Math.floor(random() * choices.length)]);
        assertChipConservation(state);
      }
    }
    const home = createHand({ modelVersion: 'cash-home-test', depthBb: 20, button: 'button-small-blind', hero: 'big-blind', cashRules: HOME_GAME_RULES });
    assertChipConservation(applyAction(home, fold));
  });
});
