import { parseCashGameRules, requireMoneyUnits, type CashGameRules } from '@/lib/cash-game-rules';
import { settleCashHand } from '@/lib/cash-settlement';
import type { CashHandLedger, HandState, Seat } from '@/lib/practice-types';

const SEATS: readonly Seat[] = ['button-small-blind', 'big-blind'];

export function cashUnitsFromBb(amount: number, rules: CashGameRules): number {
  const scaled = amount * rules.unitsPerBb;
  if (!Number.isFinite(scaled) || Math.abs(scaled - Math.round(scaled)) > 1e-8) {
    throw new Error('Cash wager must align to the pinned money units');
  }
  const units = Math.round(scaled);
  requireMoneyUnits(units);
  return units;
}

/** Raw sizing formulas only. Ledger amounts must use cashUnitsFromBb instead. */
export function quantizeCashSizingBb(amount: number, rules: CashGameRules): number {
  const scaled = amount * rules.unitsPerBb;
  if (!Number.isFinite(scaled) || scaled < 0 || scaled > Number.MAX_SAFE_INTEGER) {
    throw new Error('Cash sizing must be finite and within safe money units');
  }
  const whole = Math.floor(scaled);
  const fraction = scaled - whole;
  const up = Math.abs(fraction - .5) <= 1e-8
    ? rules.betRounding === 'half-up' || whole % 2 === 1
    : fraction > .5;
  return (whole + Number(up)) / rules.unitsPerBb;
}

export function initialCashLedger(depthBb: number, suppliedRules: CashGameRules): CashHandLedger {
  const rules = parseCashGameRules(suppliedRules);
  const depth = cashUnitsFromBb(depthBb, rules);
  requireMoneyUnits(depth * 2);
  if (depth <= rules.blindsUnits[1]) throw new Error('Cash stack must exceed the big blind');
  return {
    rules, potUnits: 0, houseRakeUnits: 0,
    stacksUnits: { 'button-small-blind': depth - rules.blindsUnits[0], 'big-blind': depth - rules.blindsUnits[1] },
    streetBetsUnits: { 'button-small-blind': rules.blindsUnits[0], 'big-blind': rules.blindsUnits[1] },
    committedUnits: { 'button-small-blind': rules.blindsUnits[0], 'big-blind': rules.blindsUnits[1] },
  };
}

/** Copy the integer ledger into display fields without rounding a second time. */
export function withCashLedger(state: HandState, cash: CashHandLedger): HandState {
  const scale = cash.rules.unitsPerBb;
  const asBb = (values: Record<Seat, number>): Record<Seat, number> => ({
    'button-small-blind': values['button-small-blind'] / scale,
    'big-blind': values['big-blind'] / scale,
  });
  return { ...state, cash, potBb: cash.potUnits / scale,
    stacksBb: asBb(cash.stacksUnits), streetBetsBb: asBb(cash.streetBetsUnits),
    totalCommittedBb: asBb(cash.committedUnits) };
}

export function collectCashStreet(state: HandState): HandState {
  const cash = state.cash;
  if (!cash) throw new Error('Missing cash ledger');
  return withCashLedger(state, { ...cash,
    potUnits: cash.potUnits + cash.streetBetsUnits['button-small-blind'] + cash.streetBetsUnits['big-blind'],
    streetBetsUnits: { 'button-small-blind': 0, 'big-blind': 0 },
  });
}

export function payCashWager(state: HandState, actor: Seat, amountToBb: number): HandState {
  const cash = state.cash;
  if (!cash) throw new Error('Missing cash ledger');
  const target = cashUnitsFromBb(amountToBb, cash.rules);
  const paid = target - cash.streetBetsUnits[actor];
  if (paid < 0 || paid > cash.stacksUnits[actor]) throw new Error('Cash wager exceeds the legal stack');
  return withCashLedger(state, { ...cash,
    stacksUnits: { ...cash.stacksUnits, [actor]: cash.stacksUnits[actor] - paid },
    streetBetsUnits: { ...cash.streetBetsUnits, [actor]: target },
    committedUnits: { ...cash.committedUnits, [actor]: cash.committedUnits[actor] + paid },
  });
}

export function settleCashState(state: HandState, winner: Seat | 'split', reason: 'fold' | 'showdown'): HandState {
  const cash = state.cash;
  if (!cash) throw new Error('Missing cash ledger');
  if (state.terminal || cash.houseRakeUnits !== 0) throw new Error('Cash hand has already been settled or stopped for review');
  const settlement = settleCashHand(cash.rules, {
    committedUnits: [cash.committedUnits['button-small-blind'], cash.committedUnits['big-blind']],
    button: state.button === 'button-small-blind' ? 0 : 1, reason,
    outcome: winner === 'split' ? 'split' : winner === 'button-small-blind' ? 'player-zero' : 'player-one',
    boardCardsDealt: state.board.length as 0 | 3 | 4 | 5,
  });
  const stacksUnits = { ...cash.stacksUnits };
  for (const [index, seat] of SEATS.entries()) {
    stacksUnits[seat] += settlement.awardsUnits[index] + settlement.uncalledRefundUnits[index];
  }
  return withCashLedger({ ...state, terminal: true, toAct: null, pendingActors: [], result: {
    reason, winner, potBb: settlement.grossPotUnits / cash.rules.unitsPerBb,
    netBb: { 'button-small-blind': settlement.netPayoffUnits[0] / cash.rules.unitsPerBb,
      'big-blind': settlement.netPayoffUnits[1] / cash.rules.unitsPerBb },
    cashSettlement: settlement,
  } }, { ...cash, stacksUnits, potUnits: 0,
    streetBetsUnits: { 'button-small-blind': 0, 'big-blind': 0 },
    houseRakeUnits: cash.houseRakeUnits + settlement.rakeUnits });
}

export function assertCashConservation(state: HandState): void {
  const cash = state.cash;
  if (!cash) throw new Error('Missing cash ledger');
  const amounts = [cash.potUnits, cash.houseRakeUnits, ...Object.values(cash.stacksUnits),
    ...Object.values(cash.streetBetsUnits), ...Object.values(cash.committedUnits)];
  amounts.forEach(requireMoneyUnits);
  const total = cash.stacksUnits['button-small-blind'] + cash.stacksUnits['big-blind'] + cash.potUnits
    + cash.streetBetsUnits['button-small-blind'] + cash.streetBetsUnits['big-blind'] + cash.houseRakeUnits;
  if (total !== cashUnitsFromBb(state.depthBb, cash.rules) * 2) throw new Error('Cash chip conservation failed');
  const mirrored = withCashLedger(state, cash);
  for (const seat of SEATS) {
    if (state.stacksBb[seat] !== mirrored.stacksBb[seat] || state.streetBetsBb[seat] !== mirrored.streetBetsBb[seat]
      || state.totalCommittedBb[seat] !== mirrored.totalCommittedBb[seat]) throw new Error('Cash ledger and display disagree');
  }
  if (state.potBb !== mirrored.potBb) throw new Error('Cash pot and display disagree');
}
