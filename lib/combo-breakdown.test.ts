import { describe, expect, it } from 'vitest';
import { parseBoard, parseCard } from '@/lib/cards';
import {
  buildHandClassBreakdown,
  describeHandClass,
} from '@/lib/combo-breakdown';
import type { ClassRow } from '@/lib/solver/client';

describe('describeHandClass', () => {
  it('correctly labels suited hands', () => {
    const res = describeHandClass('AKs');
    expect(res.category).toBe('suited');
    expect(res.categoryLabel).toBe('Suited');
    expect(res.friendlyName).toBe('Ace-King Suited');
  });

  it('correctly labels offsuit hands', () => {
    const res = describeHandClass('AKo');
    expect(res.category).toBe('offsuit');
    expect(res.categoryLabel).toBe('Offsuit');
    expect(res.friendlyName).toBe('Ace-King Offsuit');
  });

  it('correctly labels pocket pairs', () => {
    const res = describeHandClass('AA');
    expect(res.category).toBe('pair');
    expect(res.categoryLabel).toBe('Pocket Pair');
    expect(res.friendlyName).toBe('Pocket Aces');
  });
});

describe('buildHandClassBreakdown', () => {
  it('builds 4 suited combos for AKs and detects board blockers', () => {
    // Board has Ah, so AhKh should be blocked!
    const board = parseBoard('Ah 7s 2c');
    const row: ClassRow = {
      class: 'AKs',
      combos: 3,
      actions: [
        { action: 'Check', freq: 0.25, ev: 2.5 },
        { action: 'Bet 75%', freq: 0.75, ev: 2.5 },
      ],
    };

    const breakdown = buildHandClassBreakdown('AKs', row, board);
    expect(breakdown.label).toBe('AKs');
    expect(breakdown.category).toBe('suited');
    expect(breakdown.totalCombos).toBe(4);
    expect(breakdown.blockedCombos).toBe(1); // AhKh
    expect(breakdown.activeCombos).toBe(3); // AsKs, AdKd, AcKc

    const blockedCombo = breakdown.combos.find((c) => c.isBlocked);
    expect(blockedCombo).toBeDefined();
    expect(blockedCombo?.blockedCards).toContain(parseCard('Ah'));
    expect(blockedCombo?.weight).toBe(0);
  });

  it('builds 12 combos for offsuit hands and merges solver combo data when present', () => {
    const board = parseBoard('Qh 7s 2c');
    const c0 = parseCard('As');
    const c1 = parseCard('Kh');
    const row: ClassRow = {
      class: 'AKo',
      combos: 11,
      actions: [{ action: 'Check', freq: 0.5, ev: 1.0 }],
      combos_data: [
        {
          card0: c0,
          card1: c1,
          weight: 1.0,
          actions: [
            { action: 'Check', freq: 0.1, ev: 2.1 },
            { action: 'Bet 75%', freq: 0.9, ev: 2.1 },
          ],
          ev: 2.1,
          equity: 0.65,
        },
      ],
    };

    const breakdown = buildHandClassBreakdown('AKo', row, board);
    expect(breakdown.totalCombos).toBe(12);
    const specificCombo = breakdown.combos.find(
      (c) =>
        (c.card0 === c0 && c.card1 === c1) || (c.card0 === c1 && c.card1 === c0)
    );
    expect(specificCombo).toBeDefined();
    expect(specificCombo?.hasData).toBe(true);
    expect(specificCombo?.ev).toBe(2.1);
    expect(specificCombo?.equity).toBe(0.65);
    expect(specificCombo?.actions[1].freq).toBe(0.9);
  });

  it('builds 6 combos for pocket pairs', () => {
    const board = parseBoard('As Kd 2c');
    const row: ClassRow = {
      class: 'AA',
      combos: 3,
      actions: [{ action: 'Bet', freq: 1.0, ev: 5.0 }],
    };

    const breakdown = buildHandClassBreakdown('AA', row, board);
    expect(breakdown.totalCombos).toBe(6);
    // As is on board, so AsAh, AsAd, AsAc are 3 blocked combos
    expect(breakdown.blockedCombos).toBe(3);
    expect(breakdown.activeCombos).toBe(3);
  });
});
