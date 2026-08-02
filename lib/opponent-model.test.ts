import { describe, expect, it } from 'vitest';
import {
  buildOpponentModel,
  DEFAULT_OPPONENT_ADAPTATION,
  OPPONENT_PROFILE_FEATURE_COUNT,
} from '@/lib/opponent-model';
import type {
  ActionKind,
  PracticeDecisionRecord,
  PracticeHandRecord,
} from '@/lib/practice-types';

function decision(index: number, kind: ActionKind): PracticeDecisionRecord {
  return {
    id: `decision-${index}`,
    handId: `hand-${index}`,
    answeredAt: index + 1,
    responseMs: 500,
    modelVersion: 'test-v1',
    mode: 'full-hand',
    depthBb: 20,
    street: 'flop',
    position: 'button-small-blind',
    handBucket: 'AKs',
    facingAction: 'Bet 2bb',
    stateHash: 'a'.repeat(64),
    board: [0, 1, 2],
    heroCards: [50, 46],
    chosenAction: { id: kind, kind, label: kind },
    policyActions: [],
    chosenActionEvBb: 0,
    bestActionEvBb: 0,
    evLossBb: 0,
    grade: 'optimal',
    confidence: 'high',
    lowConfidence: false,
  };
}

function hand(records: PracticeDecisionRecord[]): PracticeHandRecord {
  return {
    id: 'profile-hand',
    startedAt: 1,
    completedAt: 1_000,
    modelVersion: 'test-v1',
    mode: 'full-hand',
    depthBb: 20,
    button: 'button-small-blind',
    hero: 'button-small-blind',
    heroCards: [50, 46],
    opponentCards: [0, 1],
    board: [2, 3, 4, 5, 6],
    actions: [],
    decisions: records,
    result: {
      reason: 'showdown',
      winner: 'button-small-blind',
      potBb: 4,
      netBb: { 'button-small-blind': 2, 'big-blind': -2 },
    },
  };
}

describe('local opponent model', () => {
  it('keeps the exploit response off until enough stable evidence exists', () => {
    const profile = buildOpponentModel(
      [hand(Array.from({ length: 49 }, (_, index) => decision(index, 'fold')))],
      'adaptive-exploitative'
    );
    expect(profile.features).toHaveLength(OPPONENT_PROFILE_FEATURE_COUNT);
    expect(profile.reason).toBe('insufficient-evidence');
    expect(profile.responseWeight).toBe(0);
  });

  it('caps adaptation after a sufficiently large stable sample', () => {
    const profile = buildOpponentModel(
      [hand(Array.from({ length: 300 }, (_, index) => decision(index, 'fold')))],
      'adaptive-exploitative'
    );
    expect(profile.confidence).toBe(1);
    expect(profile.responseWeight).toBe(
      DEFAULT_OPPONENT_ADAPTATION.maximumResponseWeight
    );
    expect(profile.version).toMatch(/^local-opponent-profile-v1-/);
  });

  it('rejects unstable evidence and honors an explicit baseline selection', () => {
    const changing = Array.from({ length: 300 }, (_, index) =>
      decision(index, index < 150 ? 'fold' : 'raise')
    );
    const adaptive = buildOpponentModel([hand(changing)], 'adaptive-exploitative');
    const baseline = buildOpponentModel([hand(changing)], 'baseline');
    expect(adaptive.reason).toBe('unstable-evidence');
    expect(adaptive.responseWeight).toBe(0);
    expect(baseline.reason).toBe('baseline-selected');
    expect(baseline.responseWeight).toBe(0);
  });
});
