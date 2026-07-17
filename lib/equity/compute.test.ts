import { describe, it, expect } from 'vitest';
import { parseCard, parseBoard, parseRange } from '../cards';
import { computeEquity, handVsHand, seedRng } from './compute';

const h = (s: string): [number, number] => [
  parseCard(s.slice(0, 2)),
  parseCard(s.slice(2, 4)),
];

describe('equity — known spots', () => {
  it('AA vs KK preflop is ~82/18', () => {
    seedRng(12345);
    const r = computeEquity([{ hand: h('AsAd') }, { hand: h('KsKd') }], {
      maxSamples: 300000,
    });
    expect(r.equities[0]).toBeGreaterThan(0.79);
    expect(r.equities[0]).toBeLessThan(0.85);
  });

  it('exact enumeration on the river gives a clean split', () => {
    // Both players play the board: royal-ish straight on board -> tie.
    const r = computeEquity([{ hand: h('2c3d') }, { hand: h('4c5d') }], {
      board: parseBoard('AsKsQsJsTs'),
    });
    expect(r.exact).toBe(true);
    expect(r.equities[0]).toBeCloseTo(0.5, 5);
    expect(r.equities[1]).toBeCloseTo(0.5, 5);
  });

  it('made hand on the turn beats a draw appropriately', () => {
    // AsAd vs KsQs on Js Ts 2h (flopped straight+flush draws for KQ)
    seedRng(999);
    const r = computeEquity([{ hand: h('AsAd') }, { hand: h('KcQc') }], {
      board: parseBoard('Js Ts 2h'.replace(/\s/g, '')),
    });
    // Both equities should be valid probabilities summing to ~1
    expect(r.equities[0] + r.equities[1]).toBeCloseTo(1, 5);
    expect(r.equities[0]).toBeGreaterThan(0);
    expect(r.equities[1]).toBeGreaterThan(0);
  });

  it('hand vs range works', () => {
    seedRng(7);
    const r = computeEquity(
      [{ hand: h('AsAd') }, { range: parseRange('QQ,JJ,AKs') }],
      { maxSamples: 200000 }
    );
    // AA should be a strong favorite vs QQ/JJ/AKs
    expect(r.equities[0]).toBeGreaterThan(0.7);
  });
});
