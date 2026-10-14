import { advanceHandStreet, applyAction, createHand, seededRandom, stopForReview } from '@/lib/practice-engine';
import { neuralLegalActions } from '@/lib/neural-policy';
import type { Card } from '@/lib/cards';
import type { ActionAbstraction, HandState, LegalAction, PracticeStreet, Seat } from '@/lib/practice-types';

export interface StudyConfig {
  modelVersion: string;
  depthBb: number;
  actionAbstraction: ActionAbstraction;
}
export interface StudyChoice { street: PracticeStreet; actor: Seat; action: LegalAction }
export interface StudyDecision { state: HandState; actions: LegalAction[]; selected: LegalAction | null }
export interface StudyLine {
  state: HandState;
  decisions: StudyDecision[];
  waitingForCards: number | null;
  unappliedChoices: number;
}

function publicRoundComplete(state: HandState): HandState {
  if (state.street === 'river') return stopForReview(state, 'review-complete');
  return { ...state, toAct: null, pendingActors: [] };
}

function advanceSelectedBoard(state: HandState, board: Card[]): HandState {
  let next = state;
  while (!next.terminal && !next.toAct) {
    const count = next.street === 'preflop' ? 3 : next.street === 'flop' ? 4 : 5;
    if (board.length < count) break;
    next = advanceHandStreet(next, board.slice(0, count));
    if (Object.values(next.stacksBb).some((stack) => stack <= 1e-9)) next = publicRoundComplete(next);
  }
  return next;
}

/** The same rules and modeled action generator as Practice, with manual public cards. */
export function replayStudyLine(config: StudyConfig, board: Card[], choices: readonly StudyChoice[]): StudyLine {
  if (board.length > 5 || new Set(board).size !== board.length ||
      board.some((card) => !Number.isInteger(card) || card < 0 || card >= 52)) {
    throw new Error('The study board must contain at most five unique cards');
  }
  let state = createHand({ id: `study-${config.modelVersion}-${config.depthBb}`,
    modelVersion: config.modelVersion, depthBb: config.depthBb,
    button: 'button-small-blind', hero: 'button-small-blind', random: seededRandom(42) });
  // Private placeholders satisfy the dealt-state transport. Public-range
  // queries export every compatible combination and never use these to pick
  // a showdown winner. Keep them disjoint from the entire selected board.
  const legalDeck = Array.from({ length: 52 }, (_, card) => card).filter((card) => !board.includes(card));
  const holes = legalDeck.slice(0, 4);
  state = { ...state, holeCards: { 'button-small-blind': [holes[0], holes[1]], 'big-blind': [holes[2], holes[3]] },
    deck: Array.from({ length: 52 }, (_, card) => card).filter((card) => !holes.includes(card)) };
  const decisions: StudyDecision[] = [];
  let applied = 0;
  for (const choice of choices) {
    state = advanceSelectedBoard(state, board);
    if (!state.toAct || state.terminal) break;
    const actions = neuralLegalActions(state, config.actionAbstraction);
    const action = actions.find((candidate) => candidate.id === choice.action.id);
    if (state.street !== choice.street || state.toAct !== choice.actor || !action ||
        action.kind !== choice.action.kind || action.amountToBb !== choice.action.amountToBb) {
      throw new Error('The saved study action is not modeled at this point in the hand');
    }
    decisions.push({ state, actions, selected: action });
    state = applyAction(state, action, { onRoundComplete: publicRoundComplete });
    applied++;
  }
  state = advanceSelectedBoard(state, board);
  if (state.toAct && !state.terminal) decisions.push({ state, actions: neuralLegalActions(state, config.actionAbstraction), selected: null });
  return { state, decisions, unappliedChoices: choices.length - applied,
    waitingForCards: state.terminal || state.toAct ? null : state.street === 'preflop' ? 3 : state.street === 'flop' ? 4 : 5 };
}

export function studyDecisionKey(state: HandState): string {
  // A public matrix is independent of the transport's private placeholders.
  return JSON.stringify([state.modelVersion, state.depthBb, state.street, state.board,
    state.actionHistory.map((action) => [action.actor, action.kind, action.amountToBb ?? null])]);
}
