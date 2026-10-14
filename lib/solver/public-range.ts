import { cardRank, cardSuit, RANKS } from '@/lib/cards';
import type { HandState, LegalAction } from '@/lib/practice-types';
import type { ClassRow, ComboStrategy, NodeStrategy } from '@/lib/solver/client';

interface PublicRangeCombo {
  cards: [number, number]; weight: number; probabilities: number[];
  actionValuesBb: number[] | null; standardErrorsBb: number[] | null;
}
export interface PublicRangeStrategy {
  schema: 'hu-public-range-strategy-v1'; actor: number; street: string; board: number[];
  actions: Array<{ kind: string; amountToBb: number | null }>;
  combos: PublicRangeCombo[];
}

/** Validate the complete public matrix before presenting any probability/EV. */
export function publicRangeNode(value: unknown, state: HandState, legal: LegalAction[]): NodeStrategy {
  const range = value as PublicRangeStrategy | null;
  if (!range || range.schema !== 'hu-public-range-strategy-v1' ||
      range.actor !== (state.toAct === 'button-small-blind' ? 0 : 1) || range.street !== state.street ||
      JSON.stringify(range.board) !== JSON.stringify(state.board) || !Array.isArray(range.actions) ||
      !Array.isArray(range.combos) || !range.combos.length || range.combos.length > 1326 || range.actions.length !== legal.length) {
    throw new Error('Public range strategy does not match the requested hand');
  }
  const columns = legal.map((action) => {
    const indices = range.actions.flatMap((candidate, index) => {
      const kind = candidate.kind === 'all_in' ? 'all-in' : candidate.kind;
      const amountMatches = action.amountToBb === undefined ? candidate.amountToBb == null
        : typeof candidate.amountToBb === 'number' && Math.abs(candidate.amountToBb - action.amountToBb) <= 0.001;
      return kind === action.kind && amountMatches ? [index] : [];
    });
    if (indices.length !== 1) throw new Error('Public range legal action set differs from Practice');
    return indices[0];
  });
  const seen = new Set<string>();
  let totalWeight = 0;
  let maximumWeight = 0;
  for (const combo of range.combos) {
    if (!Array.isArray(combo.cards) || combo.cards.length !== 2 || combo.cards[0] === combo.cards[1] ||
        combo.cards.some((card) => !Number.isInteger(card) || card < 0 || card >= 52 || state.board.includes(card)) ||
        !Number.isFinite(combo.weight) || combo.weight <= 0 || !Array.isArray(combo.probabilities) ||
        combo.probabilities.length !== legal.length || combo.probabilities.some((p) => !Number.isFinite(p) || p < 0) ||
        Math.abs(combo.probabilities.reduce((sum, p) => sum + p, 0) - 1) > 1e-6) {
      throw new Error('Public range contains invalid cards, weights or probabilities');
    }
    const key = [...combo.cards].sort((a, b) => a - b).join('-');
    if (seen.has(key)) throw new Error('Public range contains duplicate combinations');
    seen.add(key);
    for (const vector of [combo.actionValuesBb, combo.standardErrorsBb]) {
      if (vector !== null && (!Array.isArray(vector) || vector.length !== legal.length || vector.some((v) => !Number.isFinite(v)))) {
        throw new Error('Public range action-value shape is invalid');
      }
    }
    if (combo.standardErrorsBb?.some((v) => v < 0)) throw new Error('Public range uncertainty is invalid');
    totalWeight += combo.weight; maximumWeight = Math.max(maximumWeight, combo.weight);
  }
  if (!Number.isFinite(totalWeight) || Math.abs(totalWeight - 1) > 1e-6) throw new Error('Public range weights are not normalized');
  const groups = new Map<string, PublicRangeCombo[]>();
  for (const combo of range.combos) {
    const ranks = combo.cards.map(cardRank).sort((a, b) => b - a);
    const label = `${RANKS[ranks[0]]}${RANKS[ranks[1]]}${ranks[0] === ranks[1] ? '' : cardSuit(combo.cards[0]) === cardSuit(combo.cards[1]) ? 's' : 'o'}`;
    groups.set(label, [...(groups.get(label) ?? []), combo]);
  }
  const rows: ClassRow[] = [...groups].map(([label, combos]) => {
    const mass = combos.reduce((sum, c) => sum + c.weight, 0);
    const combos_data: ComboStrategy[] = combos.map((combo) => {
      const actions = legal.map((action, i) => ({ action: action.label, freq: combo.probabilities[columns[i]], ev: combo.actionValuesBb?.[columns[i]] ?? null }));
      return { card0: combo.cards[0], card1: combo.cards[1], weight: combo.weight / maximumWeight,
        actions, ev: combo.actionValuesBb ? actions.reduce((sum, a) => sum + a.freq * (a.ev as number), 0) : null, equity: null };
    });
    const actions = legal.map((action, i) => ({ action: action.label,
      freq: combos.reduce((sum, c) => sum + c.weight * c.probabilities[columns[i]], 0) / mass,
      ev: combos.every((c) => c.actionValuesBb) ? combos.reduce((sum, c) => sum + c.weight * c.actionValuesBb![columns[i]], 0) / mass : null }));
    const ev = combos_data.every((c) => c.ev !== null)
      ? combos_data.reduce((sum, c, i) => sum + combos[i].weight * (c.ev as number), 0) / mass : null;
    return { class: label, combos: combos_data.length, actions, combos_data, ev };
  });
  return { title: `${state.toAct === 'button-small-blind' ? 'IP · BTN/SB' : 'OOP · BB'} — ${state.street}`, actions: legal.map((action) => action.label), rows };
}
