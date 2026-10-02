import {
  Card,
  cardRank,
  cardSuit,
  cardToStr,
  comboLabelToCombos,
  RANKS,
} from '@/lib/cards';
import type { ActionStrategy, ClassRow, ComboStrategy } from '@/lib/solver/client';

export const SUIT_SYMBOLS = ['♣', '♦', '♥', '♠'] as const;
export const SUIT_NAMES = ['Clubs', 'Diamonds', 'Hearts', 'Spades'] as const;

/**
 * Standard 4-color poker deck color definitions:
 * Clubs: Green
 * Diamonds: Blue
 * Hearts: Red
 * Spades: Dark / Charcoal
 */
export const SUIT_COLORS: Record<number, { text: string; bg: string; border: string }> = {
  0: {
    text: 'text-emerald-600 dark:text-emerald-400',
    bg: 'bg-emerald-50 dark:bg-emerald-950/40',
    border: 'border-emerald-500/30 dark:border-emerald-500/40',
  },
  1: {
    text: 'text-blue-600 dark:text-blue-400',
    bg: 'bg-blue-50 dark:bg-blue-950/40',
    border: 'border-blue-500/30 dark:border-blue-500/40',
  },
  2: {
    text: 'text-red-600 dark:text-red-400',
    bg: 'bg-red-50 dark:bg-red-950/40',
    border: 'border-red-500/30 dark:border-red-500/40',
  },
  3: {
    text: 'text-zinc-800 dark:text-zinc-200',
    bg: 'bg-zinc-100 dark:bg-zinc-800/60',
    border: 'border-zinc-400/40 dark:border-zinc-700',
  },
};

const RANK_NAMES: Record<string, string> = {
  A: 'Ace',
  K: 'King',
  Q: 'Queen',
  J: 'Jack',
  T: 'Ten',
  '9': 'Nine',
  '8': 'Eight',
  '7': 'Seven',
  '6': 'Six',
  '5': 'Five',
  '4': 'Four',
  '3': 'Three',
  '2': 'Deuce',
};

const PLURAL_RANK_NAMES: Record<string, string> = {
  A: 'Aces',
  K: 'Kings',
  Q: 'Queens',
  J: 'Jacks',
  T: 'Tens',
  '9': 'Nines',
  '8': 'Eights',
  '7': 'Sevens',
  '6': 'Sixes',
  '5': 'Fives',
  '4': 'Fours',
  '3': 'Threes',
  '2': 'Deuces',
};

export interface FormattedCombo {
  key: string;
  card0: Card;
  card1: Card;
  card0Str: string;
  card1Str: string;
  card0Rank: string;
  card0Suit: number;
  card1Rank: string;
  card1Suit: number;
  isBlocked: boolean;
  blockedCards: Card[];
  weight: number;
  actions: ActionStrategy[];
  ev: number;
  equity: number;
  hasData: boolean;
}

export interface HandClassBreakdown {
  label: string;
  category: 'suited' | 'offsuit' | 'pair';
  categoryLabel: string;
  friendlyName: string;
  totalCombos: number;
  activeCombos: number;
  blockedCombos: number;
  classEv?: number;
  classActions: ActionStrategy[];
  combos: FormattedCombo[];
}

export function describeHandClass(label: string): {
  category: 'suited' | 'offsuit' | 'pair';
  categoryLabel: string;
  friendlyName: string;
} {
  const r0 = label[0]?.toUpperCase() ?? '';
  const r1 = label[1]?.toUpperCase() ?? '';
  const isPair = r0 === r1;
  const isSuited = label.endsWith('s');

  if (isPair) {
    const plural = PLURAL_RANK_NAMES[r0] || `${r0}s`;
    return {
      category: 'pair',
      categoryLabel: 'Pocket Pair',
      friendlyName: `Pocket ${plural}`,
    };
  }

  const name0 = RANK_NAMES[r0] || r0;
  const name1 = RANK_NAMES[r1] || r1;

  if (isSuited) {
    return {
      category: 'suited',
      categoryLabel: 'Suited',
      friendlyName: `${name0}-${name1} Suited`,
    };
  }

  return {
    category: 'offsuit',
    categoryLabel: 'Offsuit',
    friendlyName: `${name0}-${name1} Offsuit`,
  };
}

export function buildHandClassBreakdown(
  label: string,
  row?: ClassRow,
  board: Card[] = []
): HandClassBreakdown {
  const { category, categoryLabel, friendlyName } = describeHandClass(label);
  const possibleCombos = comboLabelToCombos(label);
  const boardSet = new Set(board);

  const combosDataMap = new Map<string, ComboStrategy>();
  if (row?.combos_data) {
    for (const d of row.combos_data) {
      const k1 = `${d.card0}-${d.card1}`;
      const k2 = `${d.card1}-${d.card0}`;
      combosDataMap.set(k1, d);
      combosDataMap.set(k2, d);
    }
  }

  let activeCombosCount = 0;
  let blockedCombosCount = 0;

  const combos: FormattedCombo[] = possibleCombos.map(([c0, c1]) => {
    const blockedCards: Card[] = [];
    if (boardSet.has(c0)) blockedCards.push(c0);
    if (boardSet.has(c1)) blockedCards.push(c1);
    const isBlocked = blockedCards.length > 0;

    const matched = combosDataMap.get(`${c0}-${c1}`);
    const hasData = Boolean(matched);

    let weight = 0;
    let actions: ActionStrategy[] = [];
    let ev = 0;
    let equity = 0;

    if (matched) {
      weight = matched.weight;
      actions = matched.actions;
      ev = matched.ev;
      equity = matched.equity;
    } else if (!isBlocked && row?.actions) {
      // Fallback when solver data is at class level only
      weight = 1;
      actions = row.actions;
      ev = row.actions[0]?.ev ?? 0;
      equity = 0.5;
    }

    if (isBlocked) {
      blockedCombosCount++;
      weight = 0;
    } else if (weight > 0.001) {
      activeCombosCount++;
    }

    return {
      key: `${cardToStr(c0)}${cardToStr(c1)}`,
      card0: c0,
      card1: c1,
      card0Str: cardToStr(c0),
      card1Str: cardToStr(c1),
      card0Rank: RANKS[cardRank(c0)],
      card0Suit: cardSuit(c0),
      card1Rank: RANKS[cardRank(c1)],
      card1Suit: cardSuit(c1),
      isBlocked,
      blockedCards,
      weight,
      actions,
      ev,
      equity,
      hasData,
    };
  });

  return {
    label,
    category,
    categoryLabel,
    friendlyName,
    totalCombos: possibleCombos.length,
    activeCombos: activeCombosCount,
    blockedCombos: blockedCombosCount,
    classEv: row?.actions[0]?.ev,
    classActions: row?.actions ?? [],
    combos,
  };
}
