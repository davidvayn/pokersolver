import { describe, expect, it } from 'vitest';
import { compactStrategy } from './strategy-display';
import type { NodeStrategy } from './client';

const actions = [
  { action: 'Fold', freq: 0.1, ev: -1 },
  { action: 'Call 1bb', freq: 0.2, ev: 2 },
  { action: 'Raise to 3bb', freq: 0.3, ev: 3 },
  { action: 'Raise to 5bb', freq: 0.1, ev: 5 },
  { action: 'All-in 20bb', freq: 0.3, ev: 4 },
];
const node: NodeStrategy = {
  title: 'IP',
  actions: actions.map((a) => a.action),
  rows: [
    {
      class: 'AKs',
      combos: 4,
      ev: 2.8,
      actions,
      combos_data: [
        { card0: 48, card1: 44, weight: 0.7, equity: null, ev: 2.8, actions },
      ],
    },
  ],
};

describe('compact strategy display', () => {
  it('keeps only the largest raise in every class and exact combo without changing probabilities or the source policy', () => {
    const snapshot = JSON.stringify(node);
    const compact = compactStrategy(node);
    expect(compact.actions).toEqual(['Fold', 'Call 1', 'Raise 5', 'All-in 20']);
    const row = compact.rows[0];
    for (const grouped of [row.actions, row.combos_data![0].actions]) {
      expect(grouped).toHaveLength(4);
      expect(grouped.reduce((sum, a) => sum + a.freq, 0)).toBeCloseTo(0.7);
      expect(grouped[2].freq).toBeCloseTo(0.1);
      expect(grouped[2].ev).toBe(5);
    }
    expect(row.ev).toBe(2.8);
    expect(row.combos_data![0]).toMatchObject({
      weight: 0.7,
      equity: null,
      ev: 2.8,
    });
    expect(JSON.stringify(node)).toBe(snapshot);
  });

  it('keeps missing EVs unavailable', () => {
    const compact = compactStrategy({
      ...node,
      rows: [
        {
          ...node.rows[0],
          actions: actions.map((a) => ({
            ...a,
            ev: a.action === 'Raise to 5bb' ? null : a.ev,
          })),
        },
      ],
    });
    expect(compact.rows[0].actions[2].ev).toBeNull();
  });
});
