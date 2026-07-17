// Compact, dependency-free 5-to-7 card poker hand evaluator.
//
// Returns a single integer score where a HIGHER score is a better hand. Scores
// are comparable across all hands, so equity can be computed by counting wins,
// ties and losses. The scheme packs the hand category and kickers into one int:
//
//   score = (category << 20) | tiebreakers
//
// category: 8 = straight flush ... 0 = high card.
// This is a classic rank-histogram evaluator: correct and fast enough to drive
// Monte Carlo in a Web Worker (millions of evals/sec), with no lookup-table
// asset to ship. It can later be swapped for a WASM evaluator behind the same
// signature without touching callers.

import { Card, cardRank, cardSuit } from './cards';

export const HAND_CATEGORY_NAMES = [
  'High Card',
  'Pair',
  'Two Pair',
  'Three of a Kind',
  'Straight',
  'Flush',
  'Full House',
  'Four of a Kind',
  'Straight Flush',
] as const;

const STRAIGHT_FLUSH = 8;
const FOUR_KIND = 7;
const FULL_HOUSE = 6;
const FLUSH = 5;
const STRAIGHT = 4;
const THREE_KIND = 3;
const TWO_PAIR = 2;
const ONE_PAIR = 1;
const HIGH_CARD = 0;

/**
 * Evaluate the best 5-card hand from 5, 6 or 7 cards.
 * @param cards array of card ints (0..51)
 * @returns integer score (higher is better)
 */
export function evaluate(cards: Card[]): number {
  // Rank counts and per-suit rank bitmasks.
  const rankCount = new Int8Array(13);
  const suitRankMask = [0, 0, 0, 0]; // bitmask of ranks present per suit
  let rankMask = 0;

  for (let i = 0; i < cards.length; i++) {
    const c = cards[i];
    const r = cardRank(c);
    const s = cardSuit(c);
    rankCount[r]++;
    suitRankMask[s] |= 1 << r;
    rankMask |= 1 << r;
  }

  // Flush / straight flush detection
  for (let s = 0; s < 4; s++) {
    const mask = suitRankMask[s];
    if (popcount(mask) >= 5) {
      const sf = straightHighFromMask(mask);
      if (sf >= 0) return score(STRAIGHT_FLUSH, sf);
      // Plain flush: top 5 ranks in this suit
      return score(FLUSH, topN(mask, 5));
    }
  }

  // Group ranks by count (quads, trips, pairs)
  let quad = -1;
  const trips: number[] = [];
  const pairs: number[] = [];
  for (let r = 12; r >= 0; r--) {
    const n = rankCount[r];
    if (n === 4) quad = r;
    else if (n === 3) trips.push(r);
    else if (n === 2) pairs.push(r);
  }

  if (quad >= 0) {
    const kicker = highestExcept(rankMask, quad);
    return score(FOUR_KIND, (quad << 4) | kicker);
  }

  if (trips.length >= 1 && (trips.length >= 2 || pairs.length >= 1)) {
    const three = trips[0];
    const pair = trips.length >= 2 ? trips[1] : pairs[0];
    return score(FULL_HOUSE, (three << 4) | pair);
  }

  // Straight (non-flush)
  const st = straightHighFromMask(rankMask);
  if (st >= 0) return score(STRAIGHT, st);

  if (trips.length >= 1) {
    const three = trips[0];
    return score(THREE_KIND, (three << 16) | topNExcept(rankMask, 2, three));
  }

  if (pairs.length >= 2) {
    const hi = pairs[0];
    const lo = pairs[1];
    const kicker = highestExcept(rankMask, hi, lo);
    return score(TWO_PAIR, (hi << 8) | (lo << 4) | kicker);
  }

  if (pairs.length === 1) {
    const p = pairs[0];
    return score(ONE_PAIR, (p << 16) | topNExcept(rankMask, 3, p));
  }

  return score(HIGH_CARD, topN(rankMask, 5));
}

function score(category: number, tiebreak: number): number {
  return (category << 24) | tiebreak;
}

export function handCategory(score: number): number {
  return score >>> 24;
}

// --- bit helpers ---------------------------------------------------------

function popcount(x: number): number {
  let c = 0;
  while (x) {
    x &= x - 1;
    c++;
  }
  return c;
}

/**
 * Given a rank bitmask, return the high card rank of the best straight, or -1.
 * Handles the wheel (A-2-3-4-5) where the ace plays low.
 */
function straightHighFromMask(mask: number): number {
  // Scan from ace-high (T-J-Q-K-A) down to 6-high; return the first hit.
  for (let hi = 12; hi >= 4; hi--) {
    const need = 0b11111 << (hi - 4);
    if ((mask & need) === need) return hi;
  }
  // Wheel: A-2-3-4-5 (bits 12,0,1,2,3), ace plays low -> 5-high (rank index 3).
  const wheel = (1 << 12) | (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3);
  if ((mask & wheel) === wheel) return 3;
  return -1;
}

/** Top N ranks from a bitmask packed 4 bits each (highest first). */
function topN(mask: number, n: number): number {
  let out = 0;
  let count = 0;
  for (let r = 12; r >= 0 && count < n; r--) {
    if (mask & (1 << r)) {
      out = (out << 4) | r;
      count++;
    }
  }
  return out;
}

/** Top N ranks excluding a specific rank. */
function topNExcept(mask: number, n: number, except: number): number {
  return topN(mask & ~(1 << except), n);
}

function highestExcept(mask: number, ...except: number[]): number {
  let m = mask;
  for (const e of except) m &= ~(1 << e);
  for (let r = 12; r >= 0; r--) if (m & (1 << r)) return r;
  return 0;
}
