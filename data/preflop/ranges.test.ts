import { describe, it, expect } from 'vitest';
import { CHARTS, chartsFor } from './ranges';
import { parseRange } from '@/lib/cards';

describe('preflop charts', () => {
  it('has a BB vs SB chart (and BB defenses vs every opener)', () => {
    expect(chartsFor('BB', 'SB', 6).length).toBe(1);
    for (const vs of ['UTG', 'MP', 'CO', 'BTN', 'SB'] as const) {
      expect(chartsFor('BB', vs).length).toBeGreaterThan(0);
    }
  });

  it('has dedicated heads-up opening and defense charts', () => {
    expect(chartsFor('BTN', 'BB', 2).map((chart) => chart.id)).toContain(
      'hu-btn-rfi'
    );
    expect(chartsFor('BB', 'BTN', 2).map((chart) => chart.id)).toContain(
      'hu-bb-vs-btn'
    );
  });

  it('has dedicated 9-max opening charts for every opening seat', () => {
    for (const hero of [
      'UTG',
      'UTG1',
      'MP',
      'LJ',
      'HJ',
      'CO',
      'BTN',
      'SB',
    ] as const) {
      const charts = chartsFor(hero, undefined, 9).filter(
        (chart) => chart.category === 'RFI'
      );
      expect(charts, `${hero} 9-max RFI`).toHaveLength(1);
      expect(charts[0].formats).toEqual([9]);
    }
  });

  it('has dedicated 9-max BB defenses against every opener', () => {
    for (const villain of [
      'UTG',
      'UTG1',
      'MP',
      'LJ',
      'HJ',
      'CO',
      'BTN',
      'SB',
    ] as const) {
      const charts = chartsFor('BB', villain, 9).filter(
        (chart) => chart.category === 'vs-RFI'
      );
      expect(charts, `9-max BB vs ${villain}`).toHaveLength(1);
      expect(charts[0].formats).toEqual([9]);
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

  it('uses globally unique chart ids', () => {
    expect(new Set(CHARTS.map((chart) => chart.id)).size).toBe(CHARTS.length);
  });
});
