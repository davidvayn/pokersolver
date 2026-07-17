// Canonical card model + range notation parsing/serialization.
//
// A card is encoded as an integer 0..51 where:
//   card = rank * 4 + suit
//   rank: 0 = 2, 1 = 3, ... 8 = T, 9 = J, 10 = Q, 11 = K, 12 = A
//   suit: 0 = clubs, 1 = diamonds, 2 = hearts, 3 = spades
//
// This compact integer form is what the evaluator and Monte Carlo engine use.

export const RANKS = '23456789TJQKA';
export const SUITS = 'cdhs';

export type Card = number; // 0..51

export function makeCard(rank: number, suit: number): Card {
  return rank * 4 + suit;
}

export function cardRank(card: Card): number {
  return card >> 2; // Math.floor(card / 4)
}

export function cardSuit(card: Card): number {
  return card & 3;
}

/** "As", "Td", "2c" -> card int. Throws on malformed input. */
export function parseCard(str: string): Card {
  if (str.length !== 2) throw new Error(`Invalid card: "${str}"`);
  const rank = RANKS.indexOf(str[0].toUpperCase());
  const suit = SUITS.indexOf(str[1].toLowerCase());
  if (rank < 0 || suit < 0) throw new Error(`Invalid card: "${str}"`);
  return makeCard(rank, suit);
}

/** card int -> "As" */
export function cardToStr(card: Card): string {
  return RANKS[cardRank(card)] + SUITS[cardSuit(card)];
}

/** Parse a board like "AsKd7h" or "As Kd 7h" into an array of cards. */
export function parseBoard(str: string): Card[] {
  const clean = str.replace(/\s+/g, '');
  const cards: Card[] = [];
  for (let i = 0; i < clean.length; i += 2) {
    cards.push(parseCard(clean.slice(i, i + 2)));
  }
  return cards;
}

// ---------------------------------------------------------------------------
// Range notation
// ---------------------------------------------------------------------------
//
// A "combo" is a specific two-card holding (e.g. AsKh). A "hand class" is a
// grid cell like "AKs", "AKo", or "TT". The 13x13 matrix has 169 hand classes.
//
// A Range maps a combo (encoded as a stable key) to a weight in [0, 1].

export type Combo = [Card, Card]; // always stored with c0 > c1

export function comboKey(a: Card, b: Card): number {
  // Order-independent key: 0..1325
  const hi = Math.max(a, b);
  const lo = Math.min(a, b);
  return (hi * (hi - 1)) / 2 + lo;
}

export type Range = Map<number, number>; // comboKey -> weight

/** All 6 (pair) / 4 (suited... ) combos for a hand-class label like "AKs". */
export function comboLabelToCombos(label: string): Combo[] {
  const r1 = RANKS.indexOf(label[0].toUpperCase());
  const r2 = RANKS.indexOf(label[1].toUpperCase());
  if (r1 < 0 || r2 < 0) throw new Error(`Invalid hand class: "${label}"`);
  const suffix = label[2]?.toLowerCase();
  const combos: Combo[] = [];

  if (r1 === r2) {
    // Pair: all 6 suit combinations
    for (let s1 = 0; s1 < 4; s1++) {
      for (let s2 = s1 + 1; s2 < 4; s2++) {
        combos.push(orderCombo(makeCard(r1, s1), makeCard(r1, s2)));
      }
    }
    return combos;
  }

  const hiR = Math.max(r1, r2);
  const loR = Math.min(r1, r2);
  if (suffix === 's') {
    for (let s = 0; s < 4; s++) {
      combos.push(orderCombo(makeCard(hiR, s), makeCard(loR, s)));
    }
  } else if (suffix === 'o') {
    for (let s1 = 0; s1 < 4; s1++) {
      for (let s2 = 0; s2 < 4; s2++) {
        if (s1 === s2) continue;
        combos.push(orderCombo(makeCard(hiR, s1), makeCard(loR, s2)));
      }
    }
  } else {
    // No suffix on a non-pair: both suited and offsuit
    return [
      ...comboLabelToCombos(RANKS[hiR] + RANKS[loR] + 's'),
      ...comboLabelToCombos(RANKS[hiR] + RANKS[loR] + 'o'),
    ];
  }
  return combos;
}

function orderCombo(a: Card, b: Card): Combo {
  return a > b ? [a, b] : [b, a];
}

/** The 169 grid labels, row-major from AA. Row i, col j (0=A..12=2). */
export function handClassLabel(row: number, col: number): string {
  // row/col index 0 -> Ace, 12 -> 2 (matrix is drawn A..2)
  const rHi = 12 - row;
  const rLo = 12 - col;
  if (row === col) return RANKS[rHi] + RANKS[rHi]; // pair on the diagonal
  if (row < col) return RANKS[12 - row] + RANKS[12 - col] + 's'; // upper triangle: suited
  return RANKS[12 - col] + RANKS[12 - row] + 'o'; // lower triangle: offsuit
}

/**
 * Parse a PioSolver-style range string into a Range map.
 * Supports: comma/space separated tokens, each of:
 *   - "AA", "AKs", "AKo", "AK" (both)
 *   - pairs range "66-99" or "TT+"
 *   - suited/offsuit range "A5s-A2s", "KTs+", "QJo+"
 *   - explicit combos "AsKh"
 *   - optional weight suffix ":0.5" applied to the token
 */
export function parseRange(input: string): Range {
  const range: Range = new Map();
  const tokens = input
    .split(/[,\s]+/)
    .map((t) => t.trim())
    .filter(Boolean);

  for (const raw of tokens) {
    let weight = 1;
    let token = raw;
    const colon = token.indexOf(':');
    if (colon >= 0) {
      weight = parseFloat(token.slice(colon + 1));
      token = token.slice(0, colon);
      if (!isFinite(weight)) weight = 1;
    }

    for (const label of expandToken(token)) {
      addLabel(range, label, weight);
    }
  }
  return range;
}

function addLabel(range: Range, label: string, weight: number) {
  // Explicit 4-char combo like "AsKh"
  if (/^[2-9TJQKA][cdhs][2-9TJQKA][cdhs]$/i.test(label)) {
    const a = parseCard(label.slice(0, 2));
    const b = parseCard(label.slice(2, 4));
    range.set(comboKey(a, b), weight);
    return;
  }
  for (const [a, b] of comboLabelToCombos(label)) {
    range.set(comboKey(a, b), weight);
  }
}

/** Expand a single token into an array of concrete hand-class labels. */
function expandToken(token: string): string[] {
  // Explicit 4-char combo like "AsKh" — passed through; addLabel() handles it.
  if (/^[2-9TJQKA][cdhs][2-9TJQKA][cdhs]$/i.test(token)) {
    return [token];
  }

  // "TT+" -> TT, JJ, QQ, KK, AA
  const pairPlus = /^([2-9TJQKA])\1\+$/i.exec(token);
  if (pairPlus) {
    const start = RANKS.indexOf(pairPlus[1].toUpperCase());
    const out: string[] = [];
    for (let r = start; r <= 12; r++) out.push(RANKS[r] + RANKS[r]);
    return out;
  }

  // "66-99" pair range
  const pairRange = /^([2-9TJQKA])\1-([2-9TJQKA])\2$/i.exec(token);
  if (pairRange) {
    let lo = RANKS.indexOf(pairRange[1].toUpperCase());
    let hi = RANKS.indexOf(pairRange[2].toUpperCase());
    if (lo > hi) [lo, hi] = [hi, lo];
    const out: string[] = [];
    for (let r = lo; r <= hi; r++) out.push(RANKS[r] + RANKS[r]);
    return out;
  }

  // "KTs+" / "QJo+" / "A5s+" -> fix high card, walk low card up to high-1
  const suitedPlus = /^([2-9TJQKA])([2-9TJQKA])([so])\+$/i.exec(token);
  if (suitedPlus) {
    const hi = RANKS.indexOf(suitedPlus[1].toUpperCase());
    const loStart = RANKS.indexOf(suitedPlus[2].toUpperCase());
    const suf = suitedPlus[3].toLowerCase();
    const out: string[] = [];
    for (let r = loStart; r < hi; r++) out.push(RANKS[hi] + RANKS[r] + suf);
    return out;
  }

  // "A5s-A2s" style range (shared high card)
  const suitedRange =
    /^([2-9TJQKA])([2-9TJQKA])([so])-([2-9TJQKA])([2-9TJQKA])\3$/i.exec(token);
  if (suitedRange) {
    const hi = RANKS.indexOf(suitedRange[1].toUpperCase());
    const a = RANKS.indexOf(suitedRange[2].toUpperCase());
    const b = RANKS.indexOf(suitedRange[5].toUpperCase());
    const suf = suitedRange[3].toLowerCase();
    let lo = Math.min(a, b);
    let hiLo = Math.max(a, b);
    const out: string[] = [];
    for (let r = lo; r <= hiLo; r++) {
      if (r === hi) continue;
      out.push(RANKS[hi] + RANKS[r] + suf);
    }
    return out;
  }

  // Plain class: "AA", "AKs", "AKo", "AK"
  return [token];
}

/** Serialize a range back to a compact, human-readable string of combos/labels. */
export function serializeRange(range: Range): string {
  // Group by hand-class label; emit label if fully present at weight 1.
  const labels: string[] = [];
  for (let row = 0; row < 13; row++) {
    for (let col = 0; col < 13; col++) {
      const label = handClassLabel(row, col);
      const combos = comboLabelToCombos(label);
      const weights = combos.map((c) => range.get(comboKey(c[0], c[1])) ?? 0);
      if (weights.every((w) => w === 0)) continue;
      const allSame = weights.every((w) => w === weights[0]);
      if (allSame && weights[0] === 1) labels.push(label);
      else if (allSame) labels.push(`${label}:${round2(weights[0])}`);
      else labels.push(label); // mixed weights within a class: emit label (approx)
    }
  }
  return labels.join(',');
}

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}

/** Total weighted combo count of a range (max 1326). */
export function rangeComboCount(range: Range): number {
  let total = 0;
  for (const w of range.values()) total += w;
  return total;
}

/** Convert per-hand-class weights (0..1) into a per-combo Range. */
export function weightsToRange(weights: Record<string, number>): Range {
  const range: Range = new Map();
  for (const [label, w] of Object.entries(weights)) {
    if (!w) continue;
    for (const [a, b] of comboLabelToCombos(label)) {
      range.set(comboKey(a, b), w);
    }
  }
  return range;
}

/** Convert a Range back to per-hand-class weights (averaged over each class). */
export function rangeToWeights(range: Range): Record<string, number> {
  const weights: Record<string, number> = {};
  for (let row = 0; row < 13; row++) {
    for (let col = 0; col < 13; col++) {
      const label = handClassLabel(row, col);
      const combos = comboLabelToCombos(label);
      let sum = 0;
      for (const [a, b] of combos) sum += range.get(comboKey(a, b)) ?? 0;
      const avg = sum / combos.length;
      if (avg > 0) weights[label] = avg;
    }
  }
  return weights;
}
