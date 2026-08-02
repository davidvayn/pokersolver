import { cardRank, cardSuit, RANKS, type Card } from '@/lib/cards';
import { evaluate, handCategory, HAND_CATEGORY_NAMES } from '@/lib/evaluator';
import type {
  HandResult,
  HandState,
  LegalAction,
  PracticeSettings,
  PracticeStreet,
  PublicAction,
  Seat,
} from '@/lib/practice-types';

const EPSILON = 1e-9;
const BIG_BLIND_BB = 1;

export interface CreateHandOptions {
  id?: string;
  modelVersion: string;
  depthBb: number;
  button: Seat;
  hero: Seat;
  random?: () => number;
}

export function otherSeat(seat: Seat): Seat {
  return seat === 'button-small-blind'
    ? 'big-blind'
    : 'button-small-blind';
}

export function seededRandom(seed: number): () => number {
  let value = seed >>> 0;
  return () => {
    value += 0x6d2b79f5;
    let next = value;
    next = Math.imul(next ^ (next >>> 15), next | 1);
    next ^= next + Math.imul(next ^ (next >>> 7), next | 61);
    return ((next ^ (next >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffledDeck(random: () => number): Card[] {
  const deck = Array.from({ length: 52 }, (_, card) => card);
  for (let index = deck.length - 1; index > 0; index--) {
    const swap = Math.floor(random() * (index + 1));
    [deck[index], deck[swap]] = [deck[swap], deck[index]];
  }
  return deck;
}

function draw(deck: Card[]): Card {
  const card = deck.pop();
  if (card === undefined) throw new Error('The deck is empty');
  return card;
}

function roundMoney(value: number): number {
  return Math.round(value * 1000) / 1000;
}

export function totalPotBb(state: HandState): number {
  return roundMoney(
    state.potBb +
      state.streetBetsBb['button-small-blind'] +
      state.streetBetsBb['big-blind']
  );
}

export function createHand(options: CreateHandOptions): HandState {
  if (!Number.isFinite(options.depthBb) || options.depthBb <= BIG_BLIND_BB) {
    throw new Error('Effective stack must be larger than the big blind');
  }
  const random = options.random ?? Math.random;
  const deck = shuffledDeck(random);
  const holeCards: HandState['holeCards'] = {
    'button-small-blind': [draw(deck), draw(deck)],
    'big-blind': [draw(deck), draw(deck)],
  };
  const depth = roundMoney(options.depthBb);
  return {
    id:
      options.id ??
      `hand-${Date.now().toString(36)}-${Math.floor(random() * 0xffff).toString(36)}`,
    modelVersion: options.modelVersion,
    depthBb: depth,
    button: options.button,
    hero: options.hero,
    street: 'preflop',
    holeCards,
    board: [],
    deck,
    potBb: 0,
    stacksBb: {
      'button-small-blind': roundMoney(depth - 0.5),
      'big-blind': roundMoney(depth - 1),
    },
    streetBetsBb: {
      'button-small-blind': 0.5,
      'big-blind': 1,
    },
    totalCommittedBb: {
      'button-small-blind': 0.5,
      'big-blind': 1,
    },
    toAct: 'button-small-blind',
    pendingActors: ['button-small-blind', 'big-blind'],
    lastFullRaiseBb: 1,
    raiseReopened: true,
    actionHistory: [],
    terminal: false,
    result: null,
  };
}

function maxStreetBet(state: HandState): number {
  return Math.max(
    state.streetBetsBb['button-small-blind'],
    state.streetBetsBb['big-blind']
  );
}

export function toCallBb(state: HandState, seat = state.toAct): number {
  if (!seat) return 0;
  return roundMoney(
    Math.max(0, maxStreetBet(state) - state.streetBetsBb[seat])
  );
}

export function engineLegalActions(state: HandState): LegalAction[] {
  if (state.terminal || !state.toAct) return [];
  const seat = state.toAct;
  const stack = state.stacksBb[seat];
  const toCall = toCallBb(state, seat);
  const actions: LegalAction[] = [];
  if (toCall > EPSILON) {
    actions.push({ id: 'fold', kind: 'fold', label: 'Fold' });
    actions.push({
      id: stack <= toCall + EPSILON ? 'call-all-in' : 'call',
      kind: 'call',
      label: stack <= toCall + EPSILON ? `Call ${stack.toFixed(1)}bb` : `Call ${toCall.toFixed(1)}bb`,
    });
  } else {
    actions.push({ id: 'check', kind: 'check', label: 'Check' });
  }

  const allInTo = roundMoney(state.streetBetsBb[seat] + stack);
  if (stack > toCall + EPSILON) {
    actions.push({
      id: 'all-in',
      kind: 'all-in',
      label: `All-in ${allInTo.toFixed(1)}bb`,
      amountToBb: allInTo,
    });
  }
  return actions;
}

function assertActionLegal(state: HandState, action: LegalAction): void {
  if (state.terminal || !state.toAct) throw new Error('Hand is already complete');
  const actor = state.toAct;
  const stack = state.stacksBb[actor];
  const current = state.streetBetsBb[actor];
  const maximum = roundMoney(current + stack);
  const toCall = toCallBb(state, actor);
  const highest = maxStreetBet(state);

  if (action.kind === 'fold' && toCall <= EPSILON) {
    throw new Error('Cannot fold when checking is available');
  }
  if (action.kind === 'check' && toCall > EPSILON) {
    throw new Error('Cannot check while facing a bet');
  }
  if (action.kind === 'call' && toCall <= EPSILON) {
    throw new Error('Cannot call without a wager to match');
  }
  if (action.kind === 'fold' || action.kind === 'check' || action.kind === 'call') {
    return;
  }

  const target = action.amountToBb;
  if (!Number.isFinite(target) || target === undefined) {
    throw new Error('Betting actions require an amount-to value');
  }
  if (target <= highest + EPSILON || target > maximum + EPSILON) {
    throw new Error('Bet amount is outside the legal stack range');
  }
  const raiseSize = target - highest;
  const isAllIn = Math.abs(target - maximum) <= EPSILON;
  const minimumRaise = toCall > EPSILON ? state.lastFullRaiseBb : BIG_BLIND_BB;
  if (raiseSize + EPSILON < minimumRaise && !isAllIn) {
    throw new Error('Bet is smaller than the minimum full raise');
  }
  if (action.kind === 'bet' && toCall > EPSILON) {
    throw new Error('Use raise while facing a wager');
  }
  if (action.kind === 'raise' && toCall <= EPSILON) {
    throw new Error('Use bet when no wager is outstanding');
  }
  if (action.kind === 'all-in' && !isAllIn) {
    throw new Error('All-in amount must use the actor’s full stack');
  }
  if (state.raiseReopened === false) {
    throw new Error('A short all-in did not reopen raising');
  }
}

function collectStreetBets(state: HandState): HandState {
  const added =
    state.streetBetsBb['button-small-blind'] +
    state.streetBetsBb['big-blind'];
  return {
    ...state,
    potBb: roundMoney(state.potBb + added),
    streetBetsBb: { 'button-small-blind': 0, 'big-blind': 0 },
  };
}

function dealNextStreet(state: HandState): HandState {
  const deck = [...state.deck];
  draw(deck); // burn
  const board = [...state.board];
  let street: PracticeStreet;
  if (state.street === 'preflop') {
    street = 'flop';
    board.push(draw(deck), draw(deck), draw(deck));
  } else if (state.street === 'flop') {
    street = 'turn';
    board.push(draw(deck));
  } else if (state.street === 'turn') {
    street = 'river';
    board.push(draw(deck));
  } else {
    throw new Error('Cannot advance past the river');
  }
  const first = otherSeat(state.button);
  return {
    ...state,
    deck,
    board,
    street,
    toAct: first,
    pendingActors: [first, state.button],
    lastFullRaiseBb: BIG_BLIND_BB,
    raiseReopened: true,
  };
}

function awardPot(
  state: HandState,
  winner: Seat | 'split',
  reason: HandResult['reason']
): HandState {
  const collected = collectStreetBets(state);
  const pot = collected.potBb;
  const stacks = { ...collected.stacksBb };
  if (winner === 'split') {
    stacks['button-small-blind'] = roundMoney(
      stacks['button-small-blind'] + pot / 2
    );
    stacks['big-blind'] = roundMoney(stacks['big-blind'] + pot / 2);
  } else {
    stacks[winner] = roundMoney(stacks[winner] + pot);
  }
  const result: HandResult = {
    reason,
    winner,
    potBb: pot,
    netBb: {
      'button-small-blind': roundMoney(
        stacks['button-small-blind'] - state.depthBb
      ),
      'big-blind': roundMoney(stacks['big-blind'] - state.depthBb),
    },
  };
  return {
    ...collected,
    potBb: 0,
    stacksBb: stacks,
    toAct: null,
    pendingActors: [],
    terminal: true,
    result,
  };
}

function runBoardToRiver(state: HandState): HandState {
  let next = state;
  while (next.street !== 'river') next = dealNextStreet(next);
  return next;
}

function settleShowdown(state: HandState): HandState {
  const river = state.street === 'river' ? state : runBoardToRiver(state);
  const buttonScore = evaluate([
    ...river.holeCards['button-small-blind'],
    ...river.board,
  ]);
  const bigBlindScore = evaluate([
    ...river.holeCards['big-blind'],
    ...river.board,
  ]);
  const winner: Seat | 'split' =
    buttonScore === bigBlindScore
      ? 'split'
      : buttonScore > bigBlindScore
        ? 'button-small-blind'
        : 'big-blind';
  const settled = awardPot(river, winner, 'showdown');
  const winningScore = Math.max(buttonScore, bigBlindScore);
  return {
    ...settled,
    result: settled.result
      ? {
          ...settled.result,
          winningHand: HAND_CATEGORY_NAMES[handCategory(winningScore)],
        }
      : null,
  };
}

function finishBettingRound(state: HandState): HandState {
  const collected = collectStreetBets(state);
  if (
    collected.stacksBb['button-small-blind'] <= EPSILON ||
    collected.stacksBb['big-blind'] <= EPSILON
  ) {
    return settleShowdown(collected);
  }
  if (collected.street === 'river') return settleShowdown(collected);
  return dealNextStreet(collected);
}

export function applyAction(state: HandState, action: LegalAction): HandState {
  assertActionLegal(state, action);
  const actor = state.toAct as Seat;
  const opponent = otherSeat(actor);
  const beforeHighest = maxStreetBet(state);
  const beforeBet = state.streetBetsBb[actor];
  let paid = 0;
  let amountTo = beforeBet;
  let next: HandState = {
    ...state,
    stacksBb: { ...state.stacksBb },
    streetBetsBb: { ...state.streetBetsBb },
    totalCommittedBb: { ...state.totalCommittedBb },
    pendingActors: [...state.pendingActors],
    actionHistory: [...state.actionHistory],
  };

  if (action.kind === 'fold') {
    const publicAction: PublicAction = {
      id: `${state.id}-${state.actionHistory.length}`,
      actor,
      street: state.street,
      kind: action.kind,
      label: action.label,
      amountBb: 0,
      potAfterBb: totalPotBb(state),
    };
    next.actionHistory.push(publicAction);
    return awardPot(next, opponent, 'fold');
  }

  if (action.kind === 'call') {
    paid = Math.min(toCallBb(state, actor), state.stacksBb[actor]);
    amountTo = roundMoney(beforeBet + paid);
  } else if (action.kind === 'bet' || action.kind === 'raise' || action.kind === 'all-in') {
    amountTo = roundMoney(action.amountToBb as number);
    paid = roundMoney(amountTo - beforeBet);
  }

  next.stacksBb[actor] = roundMoney(next.stacksBb[actor] - paid);
  next.streetBetsBb[actor] = amountTo;
  next.totalCommittedBb[actor] = roundMoney(
    next.totalCommittedBb[actor] + paid
  );

  const aggressive =
    action.kind === 'bet' || action.kind === 'raise' || action.kind === 'all-in';
  if (aggressive && amountTo > beforeHighest + EPSILON) {
    const raiseSize = roundMoney(amountTo - beforeHighest);
    const fullRaise = raiseSize + EPSILON >= state.lastFullRaiseBb;
    if (fullRaise) {
      next.lastFullRaiseBb = raiseSize;
    }
    next.raiseReopened = fullRaise;
    next.pendingActors = [opponent];
  } else {
    next.pendingActors = next.pendingActors.filter((seat) => seat !== actor);
  }

  const publicAction: PublicAction = {
    id: `${state.id}-${state.actionHistory.length}`,
    actor,
    street: state.street,
    kind: action.kind,
    label: action.label,
    amountBb: paid,
    amountToBb: aggressive ? amountTo : undefined,
    potAfterBb: roundMoney(
      next.potBb +
        next.streetBetsBb['button-small-blind'] +
        next.streetBetsBb['big-blind']
    ),
  };
  next.actionHistory.push(publicAction);

  if (next.pendingActors.length === 0) return finishBettingRound(next);
  next.toAct = next.pendingActors[0];
  return next;
}

export function isPreflopRoundComplete(
  previous: HandState,
  next: HandState
): boolean {
  return (
    previous.street === 'preflop' &&
    (next.street !== 'preflop' || next.terminal)
  );
}

export function modeStopsAfterAction(
  settings: PracticeSettings,
  previous: HandState,
  next: HandState
): boolean {
  if (settings.mode === 'preflop') return isPreflopRoundComplete(previous, next);
  if (settings.mode === 'postflop') return true;
  return next.terminal;
}

export function stopForReview(
  state: HandState,
  reason: 'preflop-complete' | 'review-complete'
): HandState {
  if (state.terminal) return state;
  return {
    ...state,
    toAct: null,
    pendingActors: [],
    terminal: true,
    result: {
      reason,
      winner: null,
      potBb: totalPotBb(state),
      netBb: { 'button-small-blind': 0, 'big-blind': 0 },
    },
  };
}

export function canonicalPolicyState(state: HandState, actor = state.toAct): string {
  if (!actor) throw new Error('A terminal state has no policy actor');
  const cards = [...state.holeCards[actor]].sort((a, b) => a - b).join(',');
  const board = [...state.board].sort((a, b) => a - b).join(',');
  const history = state.actionHistory
    .map((action) =>
      [action.street, action.actor, action.kind, action.amountToBb ?? action.amountBb]
        .join(':')
    )
    .join('/');
  return [
    'hu-cash-v1',
    state.modelVersion,
    state.depthBb.toFixed(3),
    actor,
    state.street,
    cards,
    board,
    state.potBb.toFixed(3),
    state.streetBetsBb['button-small-blind'].toFixed(3),
    state.streetBetsBb['big-blind'].toFixed(3),
    state.stacksBb['button-small-blind'].toFixed(3),
    state.stacksBb['big-blind'].toFixed(3),
    history,
  ].join('|');
}

export async function canonicalPolicyHash(
  state: HandState,
  actor = state.toAct
): Promise<string> {
  const bytes = new TextEncoder().encode(canonicalPolicyState(state, actor));
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
}

export function handBucket(cards: [Card, Card]): string {
  const [first, second] = cards;
  const firstRank = cardRank(first);
  const secondRank = cardRank(second);
  if (firstRank === secondRank) return `${RANKS[firstRank]}${RANKS[firstRank]}`;
  const high = Math.max(firstRank, secondRank);
  const low = Math.min(firstRank, secondRank);
  return `${RANKS[high]}${RANKS[low]}${cardSuit(first) === cardSuit(second) ? 's' : 'o'}`;
}

export function assertChipConservation(state: HandState): void {
  const total =
    state.stacksBb['button-small-blind'] +
    state.stacksBb['big-blind'] +
    totalPotBb(state);
  if (Math.abs(total - state.depthBb * 2) > 0.002) {
    throw new Error(`Chip conservation failed: ${total}bb`);
  }
}
