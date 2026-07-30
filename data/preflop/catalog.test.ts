import { describe, expect, it } from 'vitest';
import { comboLabelToCombos, comboKey, parseRange } from '@/lib/cards';
import {
  CURATED_SCENARIOS,
  SOLVED_SCENARIOS,
  scenariosForSeats,
} from '@/data/preflop/catalog';

function actionFrequency(rangeText: string): number {
  const range = parseRange(rangeText);
  let combos = 0;
  for (const weight of range.values()) combos += weight;
  return combos / 1326;
}

function handFrequency(rangeText: string, label: string): number {
  const range = parseRange(rangeText);
  const combos = comboLabelToCombos(label);
  return (
    combos.reduce(
      (sum, [first, second]) =>
        sum + (range.get(comboKey(first, second)) ?? 0),
      0
    ) / combos.length
  );
}

describe('preflop scenario catalog', () => {
  it('loads eight accepted heads-up depths without replacing curated charts', () => {
    expect(SOLVED_SCENARIOS.map((scenario) => scenario.effectiveStackBb)).toEqual(
      [2, 3, 5, 8, 10, 12, 15, 20]
    );
    expect(scenariosForSeats(2)).toHaveLength(9);
    expect(scenariosForSeats(6)).toHaveLength(1);
    expect(scenariosForSeats(9)).toHaveLength(1);

    for (const scenario of CURATED_SCENARIOS) {
      expect(scenario.openingSize).toEqual({ kind: 'raise-to', bb: 2.5 });
      expect(scenario.provenance.assumptions.join(' ')).toContain(
        'Standardized to a 2.5bb open'
      );
    }

    for (const scenario of SOLVED_SCENARIOS) {
      expect(scenario.provenance.status).toBe('approximate');
      expect(scenario.provenance.exploitabilityBb).toBeLessThanOrEqual(0.01);
      expect(scenario.openingSize).toEqual({ kind: 'all-in' });
      expect(scenario.charts.map((chart) => chart.actions[0].name)).toEqual([
        'All-in',
        'Call',
      ]);
    }
  });

  it('tightens aggregate shove and call ranges as stacks deepen', () => {
    const shove = SOLVED_SCENARIOS.map((scenario) =>
      actionFrequency(scenario.charts[0].actions[0].range)
    );
    const call = SOLVED_SCENARIOS.map((scenario) =>
      actionFrequency(scenario.charts[1].actions[0].range)
    );

    for (let index = 1; index < shove.length; index++) {
      expect(shove[index]).toBeLessThan(shove[index - 1]);
      expect(call[index]).toBeLessThan(call[index - 1]);
    }
  });

  it('keeps pocket aces as a near-pure shove and call', () => {
    for (const scenario of SOLVED_SCENARIOS) {
      expect(
        handFrequency(scenario.charts[0].actions[0].range, 'AA')
      ).toBeGreaterThan(0.99);
      expect(
        handFrequency(scenario.charts[1].actions[0].range, 'AA')
      ).toBeGreaterThan(0.99);
    }
  });
});
