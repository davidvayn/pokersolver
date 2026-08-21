import { describe, expect, it } from 'vitest';
import {
  adaptiveGroups,
  chooseAdaptiveGroup,
  nextHeroSeat,
  postflopStreetForHand,
  sanitizePracticeSettings,
  structuralSettingsChanged,
} from '@/lib/practice';
import {
  applyAction,
  assertChipConservation,
  canonicalPolicyHash,
  createHand,
  engineLegalActions,
  handBucket,
  seededRandom,
  totalPotBb,
} from '@/lib/practice-engine';
import {
  gradePolicyFrequency,
  gradePolicyChoice,
  practiceActionChoices,
  validatePolicyNode,
} from '@/lib/practice-grading';
import type {
  PolicyNode,
  PracticeDecisionRecord,
} from '@/lib/practice-types';
import { DEFAULT_PRACTICE_SETTINGS } from '@/lib/practice-types';

function hand(seed = 1, depthBb = 20) {
  return createHand({
    id: `hand-${seed}`,
    modelVersion: 'test-v1',
    depthBb,
    button: 'button-small-blind',
    hero: 'button-small-blind',
    random: seededRandom(seed),
  });
}

describe('heads-up hand engine', () => {
  it('deals unique cards, posts blinds, and is deterministic', () => {
    const first = hand(42);
    const second = hand(42);
    expect(first).toEqual(second);
    expect(first.stacksBb).toEqual({
      'button-small-blind': 19.5,
      'big-blind': 19,
    });
    expect(totalPotBb(first)).toBe(1.5);
    expect(first.toAct).toBe('button-small-blind');
    const cards = [
      ...first.holeCards['button-small-blind'],
      ...first.holeCards['big-blind'],
      ...first.deck,
    ];
    expect(new Set(cards).size).toBe(52);
    assertChipConservation(first);
  });

  it('gives the big blind its option after a limp and advances streets', () => {
    const initial = hand();
    const limped = applyAction(initial, {
      id: 'call',
      kind: 'call',
      label: 'Call 0.5bb',
    });
    expect(limped.toAct).toBe('big-blind');
    const flop = applyAction(limped, {
      id: 'check',
      kind: 'check',
      label: 'Check',
    });
    expect(flop.street).toBe('flop');
    expect(flop.board).toHaveLength(3);
    expect(flop.toAct).toBe('big-blind');
    expect(totalPotBb(flop)).toBe(2);
    assertChipConservation(flop);
  });

  it('allows the big blind to raise its option after a limp', () => {
    const limped = applyAction(hand(), {
      id: 'call',
      kind: 'call',
      label: 'Call 0.5bb',
    });
    const raised = applyAction(limped, {
      id: 'raise-3',
      kind: 'raise',
      label: 'Raise to 3.0bb',
      amountToBb: 3,
    });
    expect(raised.toAct).toBe('button-small-blind');
    expect(raised.streetBetsBb['big-blind']).toBe(3);
    expect(() =>
      applyAction(limped, {
        id: 'bet-3',
        kind: 'bet',
        label: 'Bet to 3.0bb',
        amountToBb: 3,
      })
    ).toThrow('Use raise when a street wager already exists');
    assertChipConservation(raised);
  });

  it('validates raises, settles folds, and conserves chips', () => {
    const opened = applyAction(hand(), {
      id: 'raise-2.5',
      kind: 'raise',
      label: 'Raise 2.5bb',
      amountToBb: 2.5,
    });
    expect(opened.stacksBb['button-small-blind']).toBe(17.5);
    const folded = applyAction(opened, {
      id: 'fold',
      kind: 'fold',
      label: 'Fold',
    });
    expect(folded.terminal).toBe(true);
    expect(folded.result?.winner).toBe('button-small-blind');
    expect(folded.result?.potBb).toBe(3.5);
    expect(folded.result?.netBb).toEqual({
      'button-small-blind': 1,
      'big-blind': -1,
    });
    assertChipConservation(folded);
  });

  it('runs out all-ins and reveals a settled showdown', () => {
    const shoved = applyAction(hand(7), {
      id: 'all-in',
      kind: 'all-in',
      label: 'All-in 20bb',
      amountToBb: 20,
    });
    const called = applyAction(shoved, {
      id: 'call',
      kind: 'call',
      label: 'Call 19bb',
    });
    expect(called.terminal).toBe(true);
    expect(called.board).toHaveLength(5);
    expect(called.result?.reason).toBe('showdown');
    expect(called.result?.winningHand).toBeTruthy();
    expect(called.result?.potBb).toBe(40);
    assertChipConservation(called);
  });

  it('offers only generic legal actions and rejects undersized non-all-in raises', () => {
    expect(engineLegalActions(hand()).map((action) => action.kind)).toEqual([
      'fold',
      'call',
      'all-in',
    ]);
    expect(() =>
      applyAction(hand(), {
        id: 'bad',
        kind: 'raise',
        label: 'Raise 1.5bb',
        amountToBb: 1.5,
      })
    ).toThrow('minimum full raise');
  });

  it('creates stable SHA-256 state hashes and canonical hand buckets', async () => {
    const state = hand(9);
    const first = await canonicalPolicyHash(state);
    const second = await canonicalPolicyHash(state);
    expect(first).toMatch(/^[a-f0-9]{64}$/);
    expect(second).toBe(first);
    expect(handBucket([48, 49])).toBe('AA');
    expect(handBucket([51, 47])).toBe('AKs');
    expect(handBucket([51, 46])).toBe('AKo');

    const parityState = {
      ...state,
      modelVersion: 'test-v1',
      depthBb: 20,
      street: 'preflop' as const,
      holeCards: {
        'button-small-blind': [51, 47] as [number, number],
        'big-blind': [0, 1] as [number, number],
      },
      board: [],
      potBb: 0,
      stacksBb: { 'button-small-blind': 19.5, 'big-blind': 19 },
      streetBetsBb: { 'button-small-blind': 0.5, 'big-blind': 1 },
      toAct: 'button-small-blind' as const,
      actionHistory: [],
    };
    expect(await canonicalPolicyHash(parityState)).toBe(
      'b61126532572af5ab17edbac4fc4a5a9976be22cec96813bff5c2fce64202ccb'
    );
  });
});

describe('EV grading and settings', () => {
  const node: PolicyNode = {
    stateHash: 'a'.repeat(64),
    bestActionId: 'raise',
    bestActionEvBb: 0.4,
    actions: [
      {
        id: 'fold',
        kind: 'fold',
        label: 'Fold',
        probability: 0.2,
        evBb: 0.1,
        standardErrorBb: 0.01,
        confidence: 'high',
      },
      {
        id: 'raise',
        kind: 'raise',
        label: 'Raise',
        amountToBb: 2.5,
        probability: 0.8,
        evBb: 0.4,
        standardErrorBb: 0.03,
        confidence: 'low',
      },
    ],
  };

  it('grades against policy frequency while preserving EV estimates', () => {
    expect(gradePolicyFrequency(0.4, 0.4)).toBe('perfect');
    expect(gradePolicyFrequency(0.32, 0.4)).toBe('excellent');
    expect(gradePolicyFrequency(0.2, 0.4)).toBe('good');
    expect(gradePolicyFrequency(0.1, 0.4)).toBe('inaccuracy');
    expect(gradePolicyFrequency(0.04, 0.4)).toBe('mistake');
    expect(gradePolicyFrequency(0.01, 0.4)).toBe('blunder');
    expect(gradePolicyFrequency(0, 0.4, false)).toBe('blunder');
    expect(gradePolicyChoice(node, 'fold')).toMatchObject({
      evLossBb: 0.30000000000000004,
      chosenActionProbability: 0.2,
      bestActionProbability: 0.8,
      grade: 'inaccuracy',
      lowConfidence: false,
    });
    expect(gradePolicyChoice(node, 'raise')).toMatchObject({
      evLossBb: 0,
      grade: 'perfect',
      lowConfidence: true,
    });
    expect(gradePolicyChoice(node, 'missing')).toMatchObject({
      chosenActionProbability: 0,
      grade: 'blunder',
      confidence: 'unavailable',
    });
    expect(validatePolicyNode(node)).toEqual([]);
  });

  it('offers only the two highest-frequency actions without revealing rank by order', () => {
    const choices = practiceActionChoices([
      { id: 'fold', probability: 0.3 },
      { id: 'call', probability: 0.1 },
      { id: 'raise', probability: 0.6 },
    ]);
    expect(choices).toEqual([
      { id: 'fold', probability: 0.3 },
      { id: 'raise', probability: 0.6 },
    ]);
  });

  it('accepts explicitly low-confidence EVs without invented uncertainty', () => {
    const lowConfidenceNode: PolicyNode = {
      ...node,
      actions: node.actions.map((action, index) =>
        index === 0
          ? { ...action, standardErrorBb: null, confidence: 'low' }
          : action
      ),
    };
    expect(validatePolicyNode(lowConfidenceNode)).toEqual([]);
    expect(
      validatePolicyNode({
        ...lowConfidenceNode,
        actions: lowConfidenceNode.actions.map((action, index) =>
          index === 0 ? { ...action, confidence: 'high' } : action
        ),
      })
    ).toContain('Invalid action EV data for fold');
  });

  it('falls back safely for malformed persisted settings and alternates seats', () => {
    expect(sanitizePracticeSettings({ mode: 'bogus', depthBb: 999 }).mode).toBe(
      'full-hand'
    );
    expect(nextHeroSeat('alternate', 0)).toBe('button-small-blind');
    expect(nextHeroSeat('alternate', 1)).toBe('big-blind');
  });

  it('rotates evenly through every selected postflop street', () => {
    const streets = ['flop', 'river'] as const;
    expect(postflopStreetForHand([...streets], 0)).toBe('flop');
    expect(postflopStreetForHand([...streets], 1)).toBe('river');
    expect(postflopStreetForHand([...streets], 2)).toBe('flop');
    expect(postflopStreetForHand([], 1)).toBe('turn');
  });

  it('queues structural table changes but treats a decision goal as run metadata', () => {
    expect(
      structuralSettingsChanged(DEFAULT_PRACTICE_SETTINGS, {
        ...DEFAULT_PRACTICE_SETTINGS,
        depthBb: 50,
      })
    ).toBe(true);
    expect(
      structuralSettingsChanged(DEFAULT_PRACTICE_SETTINGS, {
        ...DEFAULT_PRACTICE_SETTINGS,
        decisionGoal: 25,
      })
    ).toBe(false);
  });
});

describe('adaptive sampling', () => {
  function record(id: string, loss: number, bucket: string): PracticeDecisionRecord {
    return {
      id,
      handId: id,
      answeredAt: Number(id.replace(/\D/g, '')) || 1,
      responseMs: 100,
      modelVersion: 'test',
      mode: 'full-hand',
      depthBb: 20,
      street: 'flop',
      position: 'button-small-blind',
      handBucket: bucket,
      facingAction: 'check',
      stateHash: 'a'.repeat(64),
      board: [],
      heroCards: [0, 1],
      chosenAction: { id: 'check', kind: 'check', label: 'Check' },
      policyActions: [],
      chosenActionEvBb: 0,
      bestActionEvBb: loss,
      evLossBb: loss,
      grade: loss === 0.5 ? 'mistake' : 'good',
      confidence: 'high',
      lowConfidence: false,
    };
  }

  it('uses only the latest 200 decisions and keeps a 30% authentic branch', () => {
    const records = Array.from({ length: 210 }, (_, index) =>
      record(`r${index + 1}`, index === 209 ? 0.5 : 0.01, index === 209 ? 'AA' : '72o')
    );
    const groups = adaptiveGroups(records);
    expect(groups.reduce((sum, group) => sum + group.attempts, 0)).toBe(200);
    expect(groups[0].handBucket).toBe('AA');
    expect(chooseAdaptiveGroup(groups, () => 0.8)).toBeNull();
    expect(chooseAdaptiveGroup(groups, () => 0)).not.toBeNull();
  });
});
