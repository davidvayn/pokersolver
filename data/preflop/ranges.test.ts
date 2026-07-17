import { describe, it, expect } from 'vitest';
import { CHARTS, chartsFor } from './ranges';
import { parseRange } from '@/lib/cards';

describe('preflop charts', () => {
  it('has a BB vs SB chart (and BB defenses vs every opener)', () => {
    expect(chartsFor('BB', 'SB').length).toBe(1);
    for (const vs of ['UTG', 'MP', 'CO', 'BTN', 'SB'] as const) {
      expect(chartsFor('BB', vs).length).toBeGreaterThan(0);
    }
  });

  it('keeps action ranges within each chart disjoint', () => {
    for (const chart of CHARTS) {
      const seen = new Map<number, string>();
      for (const action of chart.actions) {
        const range = parseRange(action.range);
        for (const key of range.keys()) {
          const prev = seen.get(key);
          expect(
            prev,
            `${chart.id}: combo shared between "${prev}" and "${action.name}"`
          ).toBeUndefined();
          seen.set(key, action.name);
        }
      }
    }
  });

  it('parses every chart range without error', () => {
    for (const chart of CHARTS) {
      for (const action of chart.actions) {
        expect(() => parseRange(action.range)).not.toThrow();
        expect(parseRange(action.range).size).toBeGreaterThan(0);
      }
    }
  });
});
