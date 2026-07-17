import { describe, it, expect } from 'vitest';
import {
  parseCard,
  cardToStr,
  makeCard,
  comboLabelToCombos,
  parseRange,
  serializeRange,
  rangeComboCount,
  handClassLabel,
  parseBoard,
} from './cards';

describe('card parsing', () => {
  it('round-trips card strings', () => {
    for (const s of ['As', 'Kd', 'Th', '2c', '9s']) {
      expect(cardToStr(parseCard(s))).toBe(s);
    }
  });

  it('encodes rank and suit', () => {
    expect(parseCard('2c')).toBe(makeCard(0, 0));
    expect(parseCard('As')).toBe(makeCard(12, 3));
  });

  it('parses boards', () => {
    expect(parseBoard('AsKd7h').map(cardToStr)).toEqual(['As', 'Kd', '7h']);
    expect(parseBoard('As Kd 7h').length).toBe(3);
  });
});

describe('hand class expansion', () => {
  it('pairs have 6 combos', () => {
    expect(comboLabelToCombos('AA')).toHaveLength(6);
  });
  it('suited hands have 4 combos', () => {
    expect(comboLabelToCombos('AKs')).toHaveLength(4);
  });
  it('offsuit hands have 12 combos', () => {
    expect(comboLabelToCombos('AKo')).toHaveLength(12);
  });
  it('no-suffix non-pair has 16 combos', () => {
    expect(comboLabelToCombos('AK')).toHaveLength(16);
  });
});

describe('grid labels', () => {
  it('diagonal is pairs', () => {
    expect(handClassLabel(0, 0)).toBe('AA');
    expect(handClassLabel(12, 12)).toBe('22');
  });
  it('upper triangle is suited', () => {
    expect(handClassLabel(0, 1)).toBe('AKs');
  });
  it('lower triangle is offsuit', () => {
    expect(handClassLabel(1, 0)).toBe('AKo');
  });
});

describe('range parsing', () => {
  it('parses pair-plus', () => {
    const r = parseRange('TT+');
    // TT,JJ,QQ,KK,AA = 5 * 6 = 30 combos
    expect(rangeComboCount(r)).toBe(30);
  });
  it('parses pair range', () => {
    const r = parseRange('66-99');
    // 66,77,88,99 = 4 * 6 = 24
    expect(rangeComboCount(r)).toBe(24);
  });
  it('parses suited-plus', () => {
    const r = parseRange('KTs+'); // KTs,KJs,KQs = 3 * 4 = 12
    expect(rangeComboCount(r)).toBe(12);
  });
  it('parses A5s-A2s', () => {
    const r = parseRange('A5s-A2s'); // A2s..A5s = 4 * 4 = 16
    expect(rangeComboCount(r)).toBe(16);
  });
  it('parses explicit combos', () => {
    const r = parseRange('AsKh');
    expect(rangeComboCount(r)).toBe(1);
  });
  it('parses weighted tokens', () => {
    const r = parseRange('AA:0.5');
    expect(rangeComboCount(r)).toBeCloseTo(3, 5);
  });
  it('round-trips a simple range', () => {
    const r = parseRange('AA,KK,AKs');
    const s = serializeRange(r);
    expect(s.split(',').sort()).toEqual(['AA', 'AKs', 'KK'].sort());
  });
});
