import { describe, expect, it } from 'vitest';
import {
  analyzePractice,
  summarizePracticeDecisions,
} from '@/lib/practice-stats';
import type {
  PracticeDecisionRecord,
  PracticeGrade,
  PracticeHandRecord,
} from '@/lib/practice-types';

const NOW = new Date(2026, 8, 3, 12).getTime();
const DAY = 24 * 60 * 60 * 1_000;

function decision({
  id,
  dayOffset,
  grade,
  loss,
  responseMs,
  lowConfidence = false,
}: {
  id: string;
  dayOffset: number;
  grade: PracticeGrade;
  loss: number | null;
  responseMs: number;
  lowConfidence?: boolean;
}): PracticeDecisionRecord {
  const answeredAt = NOW + dayOffset * DAY;
  return {
    id,
    handId: `hand-${id}`,
    answeredAt,
    responseMs,
    modelVersion: 'stats-test-v1',
    mode: dayOffset % 2 ? 'push-fold' : 'full-hand',
    depthBb: dayOffset % 2 ? 10 : 20,
    street: dayOffset % 2 ? 'preflop' : 'flop',
    position: dayOffset % 2 ? 'big-blind' : 'button-small-blind',
    handBucket: dayOffset % 2 ? 'pairs' : 'suited broadway',
    facingAction: dayOffset % 2 ? 'shove' : 'check',
    stateHash: id.padEnd(64, 'a'),
    board: dayOffset % 2 ? [] : [0, 5, 10],
    heroCards: [50, 46],
    chosenAction: { id: 'call', kind: 'call', label: 'Call' },
    policyActions: [],
    chosenActionEvBb: loss === null ? null : 0,
    bestActionEvBb: loss,
    evLossBb: loss,
    grade,
    confidence: lowConfidence ? 'low' : loss === null ? 'unavailable' : 'high',
    lowConfidence,
  };
}

function hand(record: PracticeDecisionRecord): PracticeHandRecord {
  return {
    id: record.handId,
    startedAt: record.answeredAt - 10_000,
    completedAt: record.answeredAt + 2_000,
    modelVersion: record.modelVersion,
    mode: record.mode,
    depthBb: record.depthBb,
    button: 'button-small-blind',
    hero: record.position,
    heroCards: record.heroCards,
    opponentCards: [0, 1],
    board: record.board,
    actions: [],
    decisions: [record],
    result: {
      reason: 'showdown',
      winner: record.position,
      potBb: 4,
      netBb: { 'button-small-blind': 2, 'big-blind': -2 },
    },
  };
}

describe('practice analytics', () => {
  const decisions = [
    decision({ id: 'd1', dayOffset: 0, grade: 'perfect', loss: 0, responseMs: 1_200 }),
    decision({ id: 'd2', dayOffset: -1, grade: 'good', loss: 0.1, responseMs: 3_000 }),
    decision({ id: 'd3', dayOffset: -3, grade: 'mistake', loss: 0.4, responseMs: 7_500, lowConfidence: true }),
    decision({ id: 'd4', dayOffset: -4, grade: 'blunder', loss: null, responseMs: 12_000, lowConfidence: true }),
  ];

  it('uses one decision summary for the practice rail and all-time dashboard', () => {
    const summary = summarizePracticeDecisions(decisions);
    expect(summary).toMatchObject({
      decisions: 4,
      gradedDecisions: 3,
      strongDecisions: 2,
      strongDecisionPercentage: 0.5,
      gradedCoveragePercentage: 0.75,
      lowConfidencePercentage: 0.5,
      averageResponseMs: 5_925,
    });
    expect(summary.averageEvLossBb).toBeCloseTo(1 / 6);
    expect(summary.totalEvLossBb).toBeCloseTo(0.5);
  });

  it('builds calendar, quality, scatter, streak, and response-time visual data', () => {
    const stats = analyzePractice(decisions.map(hand), NOW);
    expect(stats.dailyTrend).toHaveLength(21);
    expect(stats.activity).toHaveLength(84);
    expect(stats.dailyTrend.at(-1)).toMatchObject({
      decisions: 1,
      graded: 1,
      averageEvLossBb: 0,
      strongDecisionPercentage: 1,
    });
    expect(stats.activeDays).toBe(4);
    expect(stats.currentStreakDays).toBe(2);
    expect(stats.longestStreakDays).toBe(2);
    expect(stats.decisionsPerHand).toBe(1);
    expect(stats.averageHandDurationMs).toBe(12_000);
    expect(stats.gradeDistribution.map(({ grade, decisions: count }) => [grade, count])).toEqual([
      ['perfect', 1],
      ['excellent', 0],
      ['good', 1],
      ['inaccuracy', 0],
      ['mistake', 1],
      ['blunder', 1],
    ]);
    expect(stats.byResponseTime.map((bucket) => bucket.label).sort()).toEqual([
      '10s+',
      '2–5s',
      '5–10s',
      'Under 2s',
    ]);
    expect(stats.byAction).toHaveLength(1);
    expect(stats.byAction[0].label).toBe('Call');
    expect(stats.decisionPoints.map((point) => point.id)).toEqual(['d3', 'd2', 'd1']);
  });

  it('returns stable empty-series data without invented performance', () => {
    const stats = analyzePractice([], NOW);
    expect(stats.decisions).toBe(0);
    expect(stats.averageEvLossBb).toBeNull();
    expect(stats.strongDecisionPercentage).toBe(0);
    expect(stats.dailyTrend.every((day) => day.decisions === 0)).toBe(true);
    expect(stats.gradeDistribution.every((grade) => grade.decisions === 0)).toBe(true);
  });
});
