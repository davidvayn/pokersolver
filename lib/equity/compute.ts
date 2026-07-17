// Equity computation: exact enumeration when the runout space is small,
// Monte Carlo sampling otherwise. Pure functions so they can run in a Web
// Worker or in a test. "Accurate calculations" live here.

import { Card, Range, comboKey } from '../cards';
import { evaluate } from '../evaluator';

export interface PlayerInput {
  /** Either a fixed 2-card hand, or a weighted range (comboKey -> weight). */
  hand?: [Card, Card];
  range?: Range;
}

export interface EquityResult {
  equities: number[]; // per player, 0..1, sums to ~1
  wins: number[]; // fractional wins (no split)
  ties: number[]; // fractional ties (split contribution)
  samples: number; // number of complete boards evaluated
  exact: boolean; // true if fully enumerated
}

/** Expand a range map to an array of [c0, c1] combos with weights. */
function rangeToCombos(range: Range): { combo: [Card, Card]; weight: number }[] {
  const out: { combo: [Card, Card]; weight: number }[] = [];
  for (const [key, weight] of range) {
    if (weight <= 0) continue;
    // Recover the two cards from the triangular key.
    // key = hi*(hi-1)/2 + lo, with hi > lo.
    const hi = Math.floor((1 + Math.sqrt(1 + 8 * key)) / 2);
    const lo = key - (hi * (hi - 1)) / 2;
    out.push({ combo: [hi, lo], weight });
  }
  return out;
}

function playerCombos(
  p: PlayerInput
): { combo: [Card, Card]; weight: number }[] {
  if (p.hand) return [{ combo: p.hand, weight: 1 }];
  if (p.range) return rangeToCombos(p.range);
  throw new Error('Player must have a hand or a range');
}

const FULL_DECK: Card[] = Array.from({ length: 52 }, (_, i) => i);

export interface EquityOptions {
  board?: Card[];
  /** Target Monte Carlo samples; enumeration used when smaller than this. */
  maxSamples?: number;
  /** Called periodically with progress 0..1 and interim result. */
  onProgress?: (p: number, interim: EquityResult) => void;
  /** AbortSignal-like flag to stop early. */
  shouldStop?: () => boolean;
}

export function computeEquity(
  players: PlayerInput[],
  opts: EquityOptions = {}
): EquityResult {
  const board = opts.board ?? [];
  const maxSamples = opts.maxSamples ?? 100000;
  const n = players.length;

  const wins = new Float64Array(n);
  const ties = new Float64Array(n);
  let totalWeight = 0;
  let samples = 0;

  const combos = players.map(playerCombos);
  const need = 5 - board.length; // community cards still to come

  // Decide enumeration vs Monte Carlo by the size of the joint space.
  // Rough joint size = product of combo counts * C(remaining deck, need).
  const comboSpace = combos.reduce((a, c) => a * c.length, 1);
  const runoutSpace = need <= 1 ? 52 : need <= 2 ? 52 * 52 : Infinity;
  const enumerate =
    comboSpace * runoutSpace <= maxSamples && need <= 2 && comboSpace < 5000;

  const used = new Uint8Array(52);

  // Iterate over all combinations of player holdings (respecting card removal).
  function forEachHolding(
    idx: number,
    chosen: [Card, Card][],
    weightAcc: number
  ) {
    if (opts.shouldStop?.()) return;
    if (idx === n) {
      accumulate(chosen, weightAcc);
      return;
    }
    for (const { combo, weight } of combos[idx]) {
      const [a, b] = combo;
      if (used[a] || used[b]) continue;
      // Also skip if collides with board
      if (board.includes(a) || board.includes(b)) continue;
      used[a] = 1;
      used[b] = 1;
      forEachHolding(idx + 1, [...chosen, combo], weightAcc * weight);
      used[a] = 0;
      used[b] = 0;
    }
  }

  // Exact path: for one fixed set of holdings, enumerate every runout.
  function accumulate(holdings: [Card, Card][], weight: number) {
    if (weight <= 0) return;
    const dead = new Uint8Array(52);
    for (const c of board) dead[c] = 1;
    for (const [a, b] of holdings) {
      dead[a] = 1;
      dead[b] = 1;
    }
    const deck: Card[] = [];
    for (const c of FULL_DECK) if (!dead[c]) deck.push(c);

    let runouts = 0;
    enumerateRunouts(deck, board, need, (fullBoard) => {
      settle(holdings, fullBoard, weight);
      samples++;
      runouts++;
    });
    // Each runout contributed `weight`, so the denominator must count them all.
    totalWeight += weight * runouts;
  }

  // Monte Carlo path: sample one holding per player (weighted, with rejection
  // on card collisions) plus a runout, per iteration. This avoids enumerating
  // the full cartesian product of holdings, which explodes for wide/multi-way
  // ranges.
  function monteCarlo() {
    const cum = combos.map((list) => {
      const c: number[] = [];
      let acc = 0;
      for (const { weight } of list) {
        acc += Math.max(0, weight);
        c.push(acc);
      }
      return c;
    });
    const totals = cum.map((c) => c[c.length - 1] || 0);
    if (totals.some((t) => t <= 0)) return;

    let attempts = 0;
    const maxAttempts = maxSamples * 20 + 1000;
    while (samples < maxSamples && attempts < maxAttempts) {
      if (opts.shouldStop?.()) break;
      attempts++;

      const dead = new Uint8Array(52);
      for (const c of board) dead[c] = 1;
      const holdings: [Card, Card][] = [];
      let ok = true;
      for (let p = 0; p < n; p++) {
        const list = combos[p];
        const target = rand() * totals[p];
        let lo = 0;
        let hi = list.length - 1;
        while (lo < hi) {
          const mid = (lo + hi) >> 1;
          if (cum[p][mid] < target) lo = mid + 1;
          else hi = mid;
        }
        const [a, b] = list[lo].combo;
        if (dead[a] || dead[b]) {
          ok = false;
          break;
        }
        dead[a] = 1;
        dead[b] = 1;
        holdings.push([a, b]);
      }
      if (!ok) continue;

      const deck: Card[] = [];
      for (const c of FULL_DECK) if (!dead[c]) deck.push(c);
      const fullBoard = need > 0 ? board.concat(sampleN(deck, need)) : board;
      settle(holdings, fullBoard, 1);
      samples++;
      totalWeight += 1;
    }
  }

  function settle(holdings: [Card, Card][], fullBoard: Card[], weight: number) {
    let best = -1;
    let bestPlayers: number[] = [];
    for (let i = 0; i < n; i++) {
      const s = evaluate([holdings[i][0], holdings[i][1], ...fullBoard]);
      if (s > best) {
        best = s;
        bestPlayers = [i];
      } else if (s === best) {
        bestPlayers.push(i);
      }
    }
    if (bestPlayers.length === 1) {
      wins[bestPlayers[0]] += weight;
    } else {
      const share = weight / bestPlayers.length;
      for (const i of bestPlayers) ties[i] += share;
    }
  }

  if (enumerate) forEachHolding(0, [], 1);
  else monteCarlo();

  const equities = new Array(n).fill(0);
  const denom = totalWeight || 1;
  for (let i = 0; i < n; i++) {
    equities[i] = (wins[i] + ties[i]) / denom;
  }

  return {
    equities,
    wins: Array.from(wins, (w) => w / denom),
    ties: Array.from(ties, (t) => t / denom),
    samples,
    exact: enumerate,
  };
}

/** Enumerate every unordered combination of `need` cards from `deck`. */
function enumerateRunouts(
  deck: Card[],
  board: Card[],
  need: number,
  cb: (fullBoard: Card[]) => void
) {
  if (need === 0) {
    cb(board);
    return;
  }
  const idx = new Array(need).fill(0).map((_, i) => i);
  const m = deck.length;
  if (need > m) return;
  // Standard k-combination iteration
  const combo = () => board.concat(idx.map((i) => deck[i]));
  while (true) {
    cb(combo());
    // advance
    let i = need - 1;
    while (i >= 0 && idx[i] === m - need + i) i--;
    if (i < 0) break;
    idx[i]++;
    for (let j = i + 1; j < need; j++) idx[j] = idx[j - 1] + 1;
  }
}

// Simple xorshift RNG for reproducible-ish sampling without Math.random bias.
let rngState = 0x9e3779b9 >>> 0;
function rand(): number {
  rngState ^= rngState << 13;
  rngState ^= rngState >>> 17;
  rngState ^= rngState << 5;
  rngState >>>= 0;
  // Divide by 2^32 so the result is in [0, 1) and can never hit exactly 1.0
  // (which would cause out-of-bounds sampling downstream).
  return rngState / 0x100000000;
}

export function seedRng(seed: number) {
  rngState = (seed >>> 0) || 1;
}

/** Sample N distinct cards from a deck (partial Fisher-Yates, non-destructive). */
function sampleN(deck: Card[], n: number): Card[] {
  const local = deck.slice();
  const out: Card[] = [];
  for (let i = 0; i < n; i++) {
    const j = i + Math.floor(rand() * (local.length - i));
    [local[i], local[j]] = [local[j], local[i]];
    out.push(local[i]);
  }
  return out;
}

/** Convenience: exact equity for a heads-up hand-vs-hand with optional board. */
export function handVsHand(
  a: [Card, Card],
  b: [Card, Card],
  board: Card[] = []
): EquityResult {
  return computeEquity([{ hand: a }, { hand: b }], {
    board,
    maxSamples: 2000000,
  });
}

export { comboKey };
