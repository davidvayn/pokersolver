import { cashRakeUnits, parseCashGameRules, requireMoneyUnits, type CashGameRules } from '@/lib/cash-game-rules';

export interface CashTerminal {
  readonly committedUnits: readonly [number, number];
  readonly button: 0 | 1;
  readonly reason: 'fold' | 'showdown';
  readonly outcome: 'player-zero' | 'player-one' | 'split';
  readonly boardCardsDealt: 0 | 3 | 4 | 5;
}

export interface CashSettlement {
  uncalledRefundUnits: [number, number];
  grossPotUnits: number;
  rakeUnits: number;
  netPotUnits: number;
  awardsUnits: [number, number];
  netPayoffUnits: [number, number];
}

/** Pure heads-up terminal settlement; partial practice reviews are not terminals. */
export function settleCashHand(rules: CashGameRules, terminal: CashTerminal): CashSettlement {
  parseCashGameRules(rules);
  if (!terminal || typeof terminal !== 'object' || Object.keys(terminal).length !== 5 ||
    !Array.isArray(terminal.committedUnits) || terminal.committedUnits.length !== 2 ||
    (terminal.button !== 0 && terminal.button !== 1) ||
    (terminal.reason !== 'fold' && terminal.reason !== 'showdown') ||
    !['player-zero', 'player-one', 'split'].includes(terminal.outcome) ||
    ![0, 3, 4, 5].includes(terminal.boardCardsDealt) ||
    (terminal.reason === 'showdown' && terminal.boardCardsDealt !== 5) ||
    (terminal.reason === 'fold' && terminal.outcome === 'split')) {
    throw new Error('Settlement requires a valid completed heads-up hand');
  }
  const committed = terminal.committedUnits;
  committed.forEach(requireMoneyUnits);
  requireMoneyUnits(committed[0] + committed[1]);
  const winner = terminal.outcome === 'player-zero' ? 0 : 1;
  if (terminal.reason === 'fold' && committed[winner] < committed[1 - winner]) {
    throw new Error('Fold winner cannot have the unmatched losing wager');
  }
  const matched = Math.min(...committed);
  const uncalledRefundUnits: [number, number] = [committed[0] - matched, committed[1] - matched];
  const grossPotUnits = 2 * matched;
  const rakeUnits = cashRakeUnits(rules, grossPotUnits, terminal.boardCardsDealt >= 3);
  const netPotUnits = grossPotUnits - rakeUnits;
  const awardsUnits: [number, number] = [0, 0];
  if (terminal.outcome === 'split') {
    awardsUnits[0] = Math.floor(netPotUnits / 2);
    awardsUnits[1] = Math.floor(netPotUnits / 2);
    awardsUnits[1 - terminal.button] += netPotUnits % 2;
  } else {
    awardsUnits[winner] = netPotUnits;
  }
  return {
    uncalledRefundUnits, grossPotUnits, rakeUnits, netPotUnits, awardsUnits,
    netPayoffUnits: [
      awardsUnits[0] + uncalledRefundUnits[0] - committed[0],
      awardsUnits[1] + uncalledRefundUnits[1] - committed[1],
    ],
  };
}
