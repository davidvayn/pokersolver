import { describe, expect, it } from 'vitest';
import { createHand, seededRandom } from '@/lib/practice-engine';
import { publicRangeNode, type PublicRangeStrategy } from '@/lib/solver/public-range';

const state = createHand({ modelVersion: 'test', depthBb: 20, button: 'button-small-blind', hero: 'button-small-blind', random: seededRandom(4) });
const legal = [{ id: 'fold', kind: 'fold' as const, label: 'Fold' }, { id: 'call', kind: 'call' as const, label: 'Call' }];
const fixture = (): PublicRangeStrategy => ({ schema: 'hu-public-range-strategy-v1', actor: 0, street: 'preflop', board: [],
  actions: [{ kind: 'call', amountToBb: null }, { kind: 'fold', amountToBb: null }],
  combos: [
    { cards: [48, 44], weight: .25, probabilities: [.8, .2], actionValuesBb: [2, -1], standardErrorsBb: [.01, .01] },
    { cards: [49, 45], weight: .75, probabilities: [.4, .6], actionValuesBb: [1, -1], standardErrorsBb: [.01, .01] },
  ] });

describe('shared public range strategy', () => {
  it('aligns actual action identities, weights class mixes and preserves individual action EVs', () => {
    const node = publicRangeNode(fixture(), state, legal);
    expect(node.rows[0].class).toBe('AKs');
    expect(node.rows[0].actions[0].freq).toBeCloseTo(.5);
    expect(node.rows[0].actions[1].freq).toBeCloseTo(.5);
    expect(node.rows[0].combos_data![0].actions.map((a) => a.ev)).toEqual([-1, 2]);
    expect(node.rows[0].combos_data![0].ev).toBeCloseTo(1.4);
    expect(node.rows[0].ev).toBeCloseTo(.2);
    expect(node.rows[0].combos_data![0].equity).toBeNull();
  });
  it('never replaces unavailable action EVs with zero or guessed equity', () => {
    const range = fixture(); range.combos.forEach((c) => { c.actionValuesBb = null; c.standardErrorsBb = null; });
    const row = publicRangeNode(range, state, legal).rows[0];
    expect(row.ev).toBeNull(); expect(row.actions.every((a) => a.ev === null)).toBe(true);
    expect(row.combos_data!.every((c) => c.ev === null && c.equity === null)).toBe(true);
  });
  it('rejects corrupted probabilities, duplicate/blocked combinations and mismatched actor/values', () => {
    for (const change of [
      (r: PublicRangeStrategy) => { r.combos[0].probabilities = [.25, .25]; },
      (r: PublicRangeStrategy) => { r.combos[1].cards = [...r.combos[0].cards]; },
      (r: PublicRangeStrategy) => { r.actor = 1; },
      (r: PublicRangeStrategy) => { r.combos[0].actionValuesBb = [1]; },
      (r: PublicRangeStrategy) => { r.combos[0].weight = 2; },
    ]) { const range = fixture(); change(range); expect(() => publicRangeNode(range, state, legal)).toThrow(); }
  });
});
