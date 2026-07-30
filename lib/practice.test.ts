import { describe, expect, it } from 'vitest';
import { CHARTS } from '@/data/preflop/ranges';
import {
  analyzePractice,
  createPracticeQuestion,
  generatePracticeQuestions,
  recordPracticeAnswer,
  type PracticeRecord,
} from '@/lib/practice';
import { TABLE_FORMATS } from '@/lib/positions';
import type { PreflopChart } from '@/data/preflop/ranges';

describe('practice questions', () => {
  it('respects format, category, and position rules', () => {
    const questions = generatePracticeQuestions(
      {
        seats: 2,
        scenarioId: 'curated-2-max-100bb',
        categories: ['RFI'],
        positions: ['BTN'],
        questionCount: 12,
      },
      () => 0.5
    );

    expect(questions).toHaveLength(12);
    expect(
      questions.every(
        (question) =>
          question.seats === 2 &&
          question.category === 'RFI' &&
          question.hero === 'BTN'
      )
    ).toBe(true);
  });

  it('only exposes formats with dedicated chart data', () => {
    expect(TABLE_FORMATS.map((format) => format.seats)).toEqual([2, 6, 9]);
  });

  it('samples every 9-max position from dedicated full-ring charts', () => {
    const questions = generatePracticeQuestions(
      {
        seats: 9,
        scenarioId: 'curated-9-max-100bb',
        categories: ['RFI', 'vs-RFI'],
        positions: [
          'UTG',
          'UTG1',
          'MP',
          'LJ',
          'HJ',
          'CO',
          'BTN',
          'SB',
          'BB',
        ],
        questionCount: 18,
      },
      () => 0.37
    );

    expect([...new Set(questions.map((question) => question.hero))].sort()).toEqual(
      ['BB', 'BTN', 'CO', 'HJ', 'LJ', 'MP', 'SB', 'UTG', 'UTG1']
    );
    expect(
      questions.every(
        (question) =>
          question.seats === 9 && question.chartId.startsWith('9max-')
      )
    ).toBe(true);
  });

  it('balances selected positions instead of overweighting chart-rich seats', () => {
    const questions = generatePracticeQuestions(
      {
        seats: 6,
        scenarioId: 'curated-6-max-100bb',
        categories: ['RFI', 'vs-RFI'],
        positions: ['UTG', 'MP', 'CO', 'BTN', 'SB', 'BB'],
        questionCount: 30,
      },
      () => 0.42
    );
    const counts = new Map<string, number>();
    for (const question of questions) {
      counts.set(question.hero, (counts.get(question.hero) ?? 0) + 1);
    }

    expect([...counts.keys()].sort()).toEqual(
      ['BB', 'BTN', 'CO', 'MP', 'SB', 'UTG']
    );
    expect(Math.max(...counts.values()) - Math.min(...counts.values())).toBeLessThanOrEqual(1);
    expect(
      questions.filter((question) => question.recommendedAction === 'Fold')
        .length
    ).toBeLessThan(questions.length * 0.7);
  });

  it('derives the recommended action from chart frequencies', () => {
    const chart = CHARTS.find((candidate) => candidate.id === 'rfi-UTG');
    expect(chart).toBeDefined();

    const premium = createPracticeQuestion(chart!, 'AA', 6, 'premium');
    const weak = createPracticeQuestion(chart!, '72o', 6, 'weak');

    expect(premium.recommendedAction).toBe('Raise');
    expect(weak.recommendedAction).toBe('Fold');
  });

  it('grades the primary action instead of rewarding a minor mix', () => {
    const mixedChart: PreflopChart = {
      id: 'mixed',
      title: 'Mixed',
      hero: 'BTN',
      category: 'RFI',
      formats: [6],
      actions: [{ name: 'Raise', color: 'red', range: 'AKs:0.02' }],
    };
    const question = createPracticeQuestion(mixedChart, 'AKs', 6, 'mixed');

    expect(question.recommendedAction).toBe('Fold');
    expect(question.strategy).toContainEqual({
      action: 'Raise',
      frequency: 0.02,
    });
    expect(question.correctActions).toEqual(['Fold']);
    expect(recordPracticeAnswer(question, 'Raise', 100).correct).toBe(false);
  });

  it('records answers and computes useful breakdowns', () => {
    const chart = CHARTS.find((candidate) => candidate.id === 'rfi-BTN')!;
    const question = createPracticeQuestion(chart, 'AA', 6, 'one');
    const records: PracticeRecord[] = [
      recordPracticeAnswer(question, 'Raise', 1200, Date.UTC(2026, 6, 22)),
      recordPracticeAnswer(question, 'Raise', 900, Date.UTC(2026, 6, 23)),
      recordPracticeAnswer(question, 'Fold', 1500, Date.UTC(2026, 6, 23) + 1),
    ];

    const stats = analyzePractice(records, Date.UTC(2026, 6, 23, 12));

    expect(stats.total).toBe(3);
    expect(stats.correct).toBe(2);
    expect(stats.accuracy).toBeCloseTo(2 / 3);
    expect(stats.averageResponseMs).toBe(1200);
    expect(stats.streakDays).toBe(2);
    expect(stats.byPosition[0]).toMatchObject({
      key: '6:BTN',
      label: '6-max · BTN',
      attempts: 3,
      correct: 2,
    });
    expect(stats.byFormat[0]).toMatchObject({
      key: '6',
      label: '6-max',
      attempts: 3,
    });
  });

  it('uses display labels for full-ring positions in stats', () => {
    const chart = {
      ...CHARTS.find((candidate) => candidate.id === 'rfi-UTG')!,
      hero: 'UTG1' as const,
    };
    const question = createPracticeQuestion(chart, 'AA', 9, 'full-ring');
    const stats = analyzePractice([
      recordPracticeAnswer(question, 'Raise', 100),
    ]);

    expect(stats.byPosition[0]).toMatchObject({
      key: '9:UTG1',
      label: '9-max · UTG+1',
    });
  });

  it('requires equal 20-answer windows before reporting a trend', () => {
    const chart = CHARTS.find((candidate) => candidate.id === 'rfi-BTN')!;
    const question = createPracticeQuestion(chart, 'AA', 6, 'trend');
    const makeRecords = (count: number) =>
      Array.from({ length: count }, (_, index) =>
        recordPracticeAnswer(
          question,
          index < 20 ? 'Raise' : 'Fold',
          100,
          Date.UTC(2026, 6, 23) - index
        )
      );

    expect(analyzePractice(makeRecords(21)).trend).toBe(0);
    expect(analyzePractice(makeRecords(40)).trend).toBe(1);
  });

  it('expires a streak when the latest practice day is stale', () => {
    const chart = CHARTS.find((candidate) => candidate.id === 'rfi-BTN')!;
    const question = createPracticeQuestion(chart, 'AA', 6, 'streak');
    const records = [
      recordPracticeAnswer(question, 'Raise', 100, Date.UTC(2026, 0, 1)),
      recordPracticeAnswer(question, 'Raise', 100, Date.UTC(2026, 0, 2)),
    ];

    expect(
      analyzePractice(records, Date.UTC(2026, 6, 23, 12)).streakDays
    ).toBe(0);
  });
});
