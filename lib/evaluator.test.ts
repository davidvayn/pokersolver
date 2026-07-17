import { describe, it, expect } from 'vitest';
import { parseBoard } from './cards';
import { evaluate, handCategory, HAND_CATEGORY_NAMES } from './evaluator';

function cat(cards: string): string {
  return HAND_CATEGORY_NAMES[handCategory(evaluate(parseBoard(cards)))];
}

describe('hand categories', () => {
  it('detects a royal/straight flush', () => {
    expect(cat('AsKsQsJsTs')).toBe('Straight Flush');
  });
  it('detects four of a kind', () => {
    expect(cat('AsAdAhAc2d')).toBe('Four of a Kind');
  });
  it('detects a full house', () => {
    expect(cat('AsAdAh2c2d')).toBe('Full House');
  });
  it('detects a flush', () => {
    expect(cat('As9s7s5s2s')).toBe('Flush');
  });
  it('detects a straight', () => {
    expect(cat('As2d3h4c5s')).toBe('Straight'); // wheel
    expect(cat('Ts9d8h7c6s')).toBe('Straight');
  });
  it('detects trips, two pair, pair, high card', () => {
    expect(cat('AsAdAh2c3d')).toBe('Three of a Kind');
    expect(cat('AsAd2h2c3d')).toBe('Two Pair');
    expect(cat('AsAd2h3c4d')).toBe('Pair');
    expect(cat('As2d3h4c6s')).toBe('High Card');
  });
});

describe('hand comparison', () => {
  it('ranks categories correctly', () => {
    const straightFlush = evaluate(parseBoard('AsKsQsJsTs'));
    const quads = evaluate(parseBoard('AsAdAhAc2d'));
    const boat = evaluate(parseBoard('AsAdAh2c2d'));
    const flush = evaluate(parseBoard('As9s7s5s2s'));
    expect(straightFlush).toBeGreaterThan(quads);
    expect(quads).toBeGreaterThan(boat);
    expect(boat).toBeGreaterThan(flush);
  });

  it('picks best 5 of 7 cards', () => {
    // 7 cards containing a flush should beat 7 cards with only a pair
    const flush7 = evaluate(parseBoard('As9s7s5s2s Kd Qh'.replace(/\s/g, '')));
    const pair7 = evaluate(parseBoard('AsAd2h3c4d 7s 9h'.replace(/\s/g, '')));
    expect(flush7).toBeGreaterThan(pair7);
  });

  it('kickers break ties', () => {
    const aceKing = evaluate(parseBoard('AsAdKh5c3d'));
    const aceQueen = evaluate(parseBoard('AcAhQd5s3h'));
    expect(aceKing).toBeGreaterThan(aceQueen);
  });

  it('higher straight beats lower straight', () => {
    const nine = evaluate(parseBoard('9s8d7h6c5s'));
    const wheel = evaluate(parseBoard('As2d3h4c5s'));
    expect(nine).toBeGreaterThan(wheel);
  });
});
