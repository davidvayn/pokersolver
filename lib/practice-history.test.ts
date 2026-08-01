import 'fake-indexeddb/auto';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  clearPracticeHistory,
  loadPracticeHands,
  PRACTICE_DB_NAME,
  savePracticeHand,
} from '@/lib/practice-history';
import { analyzePractice } from '@/lib/practice-stats';
import type {
  PracticeDecisionRecord,
  PracticeHandRecord,
} from '@/lib/practice-types';

function decision(id: string, loss: number | null, lowConfidence = false): PracticeDecisionRecord {
  return {
    id,
    handId: `hand-${id}`,
    answeredAt: Number(id.replace(/\D/g, '')) + 1,
    responseMs: 800,
    modelVersion: 'test-v1',
    mode: 'full-hand',
    depthBb: 20,
    street: 'flop',
    position: 'button-small-blind',
    handBucket: 'AKs',
    facingAction: 'check',
    stateHash: 'a'.repeat(64),
    board: [0, 1, 2],
    heroCards: [50, 46],
    chosenAction: { id: 'check', kind: 'check', label: 'Check' },
    policyActions: [],
    chosenActionEvBb: loss === null ? null : 0,
    bestActionEvBb: loss,
    evLossBb: loss,
    grade: loss === null ? 'ungraded' : loss > 0.25 ? 'blunder' : 'good',
    confidence: lowConfidence ? 'low' : loss === null ? 'unavailable' : 'high',
    lowConfidence,
  };
}

function hand(id: string, decisions: PracticeDecisionRecord[]): PracticeHandRecord {
  return {
    id,
    startedAt: 1,
    completedAt: Number(id.replace(/\D/g, '')) + 2,
    modelVersion: 'test-v1',
    mode: 'full-hand',
    depthBb: 20,
    button: 'button-small-blind',
    hero: 'button-small-blind',
    heroCards: [50, 46],
    opponentCards: [0, 1],
    board: [2, 3, 4, 5, 6],
    actions: [],
    decisions,
    result: {
      reason: 'showdown',
      winner: 'button-small-blind',
      potBb: 4,
      netBb: { 'button-small-blind': 2, 'big-blind': -2 },
    },
  };
}

function deleteDatabase(): Promise<void> {
  return new Promise((resolve) => {
    const request = indexedDB.deleteDatabase(PRACTICE_DB_NAME);
    request.onsuccess = () => resolve();
    request.onerror = () => resolve();
    request.onblocked = () => resolve();
  });
}

beforeEach(async () => {
  vi.stubGlobal('window', {
    dispatchEvent: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  });
  await deleteDatabase();
});

afterEach(async () => {
  await deleteDatabase();
  vi.unstubAllGlobals();
});

describe('fresh IndexedDB practice history', () => {
  it('stores, orders, and clears complete hand records', async () => {
    expect(await savePracticeHand(hand('h1', [decision('d1', 0.1)]))).toBe(true);
    expect(await savePracticeHand(hand('h2', [decision('d2', 0.2)]))).toBe(true);
    expect((await loadPracticeHands()).map((record) => record.id)).toEqual([
      'h2',
      'h1',
    ]);
    expect(await clearPracticeHistory()).toBe(true);
    expect(await loadPracticeHands()).toEqual([]);
  });

  it('ignores malformed writes instead of contaminating the fresh schema', async () => {
    expect(await savePracticeHand({ id: 'broken' } as PracticeHandRecord)).toBe(false);
    expect(await loadPracticeHands()).toEqual([]);
  });

  it('aggregates EV loss, ungraded decisions, confidence, and costly records', () => {
    const records = [
      hand('h1', [decision('d1', 0.4, true), decision('d2', 0.1)]),
      hand('h2', [decision('d3', null)]),
    ];
    const stats = analyzePractice(records);
    expect(stats.hands).toBe(2);
    expect(stats.decisions).toBe(3);
    expect(stats.gradedDecisions).toBe(2);
    expect(stats.averageEvLossBb).toBeCloseTo(0.25);
    expect(stats.totalEvLossBb).toBeCloseTo(0.5);
    expect(stats.lowConfidencePercentage).toBeCloseTo(1 / 3);
    expect(stats.recentCostly[0].id).toBe('d1');
  });
});
