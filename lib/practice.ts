import {
  CHARTS,
  type ChartActionName,
  type PreflopChart,
} from '@/data/preflop/ranges';
import {
  comboKey,
  comboLabelToCombos,
  handClassLabel,
  parseRange,
} from '@/lib/cards';
import {
  formatForSeats,
  positionLabelForSeats,
} from '@/lib/positions';
import type { Position, TableSeats } from '@/lib/positions';

export type PracticeAction = 'Fold' | ChartActionName;
export type PracticeCategory = PreflopChart['category'];

export interface PracticeRules {
  seats: TableSeats;
  categories: PracticeCategory[];
  positions: Position[];
  questionCount: number;
}

export interface ActionFrequency {
  action: PracticeAction;
  frequency: number;
}

export interface PracticeQuestion {
  id: string;
  chartId: string;
  category: PracticeCategory;
  seats: TableSeats;
  hero: Position;
  villain?: Position;
  handClass: string;
  options: PracticeAction[];
  strategy: ActionFrequency[];
  correctActions: PracticeAction[];
  recommendedAction: PracticeAction;
}

export interface PracticeRecord {
  id: string;
  answeredAt: number;
  chartId: string;
  category: PracticeCategory;
  seats: TableSeats;
  hero: Position;
  villain?: Position;
  handClass: string;
  chosenAction: PracticeAction;
  recommendedAction: PracticeAction;
  correct: boolean;
  responseMs: number;
}

export interface StatBreakdown {
  key: string;
  label: string;
  attempts: number;
  correct: number;
  accuracy: number;
}

export interface PracticeStats {
  total: number;
  correct: number;
  accuracy: number;
  averageResponseMs: number;
  streakDays: number;
  trend: number;
  byFormat: StatBreakdown[];
  byPosition: StatBreakdown[];
  byCategory: StatBreakdown[];
  byAction: StatBreakdown[];
  weaknesses: StatBreakdown[];
  recent: PracticeRecord[];
}

export const DEFAULT_PRACTICE_RULES: PracticeRules = {
  seats: 6,
  categories: ['RFI', 'vs-RFI'],
  positions: ['UTG', 'MP', 'CO', 'BTN', 'SB', 'BB'],
  questionCount: 10,
};

export function handClasses(): string[] {
  const labels: string[] = [];
  for (let row = 0; row < 13; row++) {
    for (let column = 0; column < 13; column++) {
      labels.push(handClassLabel(row, column));
    }
  }
  return labels;
}

export function chartsForPractice(rules: PracticeRules): PreflopChart[] {
  return CHARTS.filter(
    (chart) =>
      chart.formats.includes(rules.seats) &&
      rules.categories.includes(chart.category) &&
      rules.positions.includes(chart.hero)
  );
}

export function createPracticeQuestion(
  chart: PreflopChart,
  handClass: string,
  seats: TableSeats,
  id: string
): PracticeQuestion {
  const combos = comboLabelToCombos(handClass);
  const frequencies = new Map<PracticeAction, number>();
  let acted = 0;

  for (const action of chart.actions) {
    const range = parseRange(action.range);
    const frequency =
      combos.reduce(
        (sum, [first, second]) =>
          sum + (range.get(comboKey(first, second)) ?? 0),
        0
      ) / combos.length;
    const normalizedAction: PracticeAction = action.name;
    frequencies.set(normalizedAction, frequency);
    acted += frequency;
  }

  frequencies.set('Fold', Math.max(0, 1 - acted));

  const preferredOrder: PracticeAction[] =
    chart.category === 'RFI'
      ? ['Fold', 'Raise']
      : ['Fold', 'Call', '3-bet'];
  const options = preferredOrder.filter(
    (action) => action === 'Fold' || frequencies.has(action)
  );
  const strategy = options.map((action) => ({
    action,
    frequency: frequencies.get(action) ?? 0,
  }));
  const recommendedAction = strategy.reduce((best, current) =>
    current.frequency > best.frequency ? current : best
  ).action;
  const correctActions = strategy
    .filter((item) => item.frequency > 0.01)
    .map((item) => item.action);

  return {
    id,
    chartId: chart.id,
    category: chart.category,
    seats,
    hero: chart.hero,
    villain: chart.vs,
    handClass,
    options,
    strategy,
    correctActions,
    recommendedAction,
  };
}

function shuffle<T>(values: T[], random: () => number): T[] {
  for (let index = values.length - 1; index > 0; index--) {
    const swapIndex = Math.floor(random() * (index + 1));
    [values[index], values[swapIndex]] = [values[swapIndex], values[index]];
  }
  return values;
}

export function generatePracticeQuestions(
  rules: PracticeRules,
  random: () => number = Math.random
): PracticeQuestion[] {
  const charts = chartsForPractice(rules);
  if (charts.length === 0) return [];

  const seed = Date.now().toString(36);
  const makeChartBucket = (chart: PreflopChart) => {
      const byAction = new Map<PracticeAction, PracticeQuestion[]>();
      for (const handClass of handClasses()) {
        const question = createPracticeQuestion(
          chart,
          handClass,
          rules.seats,
          `${seed}-${chart.id}-${handClass}`
        );
        const bucket = byAction.get(question.recommendedAction) ?? [];
        bucket.push(question);
        byAction.set(question.recommendedAction, bucket);
      }
      for (const bucket of byAction.values()) shuffle(bucket, random);
      return {
        byAction,
        actionOrder: shuffle([...byAction.keys()], random),
        actionIndex: 0,
      };
    };
  const chartsByPosition = new Map<Position, PreflopChart[]>();
  for (const chart of charts) {
    const positionCharts = chartsByPosition.get(chart.hero) ?? [];
    positionCharts.push(chart);
    chartsByPosition.set(chart.hero, positionCharts);
  }
  const positionBuckets = shuffle(
    [...chartsByPosition.values()].map((positionCharts) => ({
      chartBuckets: shuffle(positionCharts.map(makeChartBucket), random),
      chartIndex: 0,
    })),
    random
  );

  const questions: PracticeQuestion[] = [];
  while (questions.length < rules.questionCount) {
    let added = false;
    for (const positionBucket of positionBuckets) {
      let positionAdded = false;
      for (
        let chartOffset = 0;
        chartOffset < positionBucket.chartBuckets.length;
        chartOffset++
      ) {
        const chartIndex =
          (positionBucket.chartIndex + chartOffset) %
          positionBucket.chartBuckets.length;
        const chartBucket = positionBucket.chartBuckets[chartIndex];

        for (
          let actionOffset = 0;
          actionOffset < chartBucket.actionOrder.length;
          actionOffset++
        ) {
          const actionIndex =
            (chartBucket.actionIndex + actionOffset) %
            chartBucket.actionOrder.length;
          const action = chartBucket.actionOrder[actionIndex];
          const bucket = chartBucket.byAction.get(action);
          const question = bucket?.pop();
          if (!question) continue;

          questions.push(question);
          chartBucket.actionIndex =
            (actionIndex + 1) % chartBucket.actionOrder.length;
          positionBucket.chartIndex =
            (chartIndex + 1) % positionBucket.chartBuckets.length;
          added = true;
          positionAdded = true;
          break;
        }
        if (positionAdded) break;
      }
      if (questions.length >= rules.questionCount) break;
    }
    if (!added) break;
  }

  return questions;
}

export function recordPracticeAnswer(
  question: PracticeQuestion,
  chosenAction: PracticeAction,
  responseMs: number,
  answeredAt = Date.now()
): PracticeRecord {
  return {
    id: `${question.id}-${answeredAt}`,
    answeredAt,
    chartId: question.chartId,
    category: question.category,
    seats: question.seats,
    hero: question.hero,
    villain: question.villain,
    handClass: question.handClass,
    chosenAction,
    recommendedAction: question.recommendedAction,
    correct: question.correctActions.includes(chosenAction),
    responseMs: Math.max(0, Math.round(responseMs)),
  };
}

function breakdown(
  records: PracticeRecord[],
  keyFor: (record: PracticeRecord) => string,
  labelFor: (record: PracticeRecord) => string
): StatBreakdown[] {
  const groups = new Map<
    string,
    { label: string; attempts: number; correct: number }
  >();

  for (const record of records) {
    const key = keyFor(record);
    const current = groups.get(key) ?? {
      label: labelFor(record),
      attempts: 0,
      correct: 0,
    };
    current.attempts++;
    if (record.correct) current.correct++;
    groups.set(key, current);
  }

  return [...groups.entries()]
    .map(([key, value]) => ({
      key,
      ...value,
      accuracy: value.attempts ? value.correct / value.attempts : 0,
    }))
    .sort((first, second) => second.attempts - first.attempts);
}

function localDay(timestamp: number): number {
  const date = new Date(timestamp);
  return new Date(
    date.getFullYear(),
    date.getMonth(),
    date.getDate()
  ).getTime();
}

function calculateStreak(records: PracticeRecord[], now: number): number {
  const days = [...new Set(records.map((record) => localDay(record.answeredAt)))].sort(
    (first, second) => second - first
  );
  if (days.length === 0) return 0;

  const today = localDay(now);
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  if (days[0] !== today && days[0] !== yesterday.getTime()) return 0;

  let streak = 1;
  let cursor = new Date(days[0]);
  for (let index = 1; index < days.length; index++) {
    cursor.setDate(cursor.getDate() - 1);
    if (days[index] !== cursor.getTime()) break;
    streak++;
  }
  return streak;
}

function accuracy(records: PracticeRecord[]): number {
  if (records.length === 0) return 0;
  return records.filter((record) => record.correct).length / records.length;
}

export function analyzePractice(
  records: PracticeRecord[],
  now = Date.now()
): PracticeStats {
  const ordered = [...records].sort(
    (first, second) => second.answeredAt - first.answeredAt
  );
  const correct = records.filter((record) => record.correct).length;
  const recentWindow = ordered.slice(0, 20);
  const previousWindow = ordered.slice(20, 40);
  const byFormat = breakdown(
    records,
    (record) => String(record.seats),
    (record) => formatForSeats(record.seats).label
  );
  const byPosition = breakdown(
    records,
    (record) => `${record.seats}:${record.hero}`,
    (record) =>
      `${formatForSeats(record.seats).label} · ${positionLabelForSeats(
        record.hero,
        record.seats
      )}`
  );
  const byCategory = breakdown(
    records,
    (record) => record.category,
    (record) => (record.category === 'RFI' ? 'Raise first in' : 'Facing a raise')
  );
  const byAction = breakdown(
    records,
    (record) => record.recommendedAction,
    (record) => record.recommendedAction
  );
  const eligibleWeaknesses = [
    ...byFormat,
    ...byPosition,
    ...byCategory,
    ...byAction,
  ].filter((item) => item.attempts >= 3);

  return {
    total: records.length,
    correct,
    accuracy: accuracy(records),
    averageResponseMs: records.length
      ? Math.round(
          records.reduce((sum, record) => sum + record.responseMs, 0) /
            records.length
        )
      : 0,
    streakDays: calculateStreak(records, now),
    trend:
      recentWindow.length === 20 && previousWindow.length === 20
        ? accuracy(recentWindow) - accuracy(previousWindow)
        : 0,
    byFormat,
    byPosition,
    byCategory,
    byAction,
    weaknesses: eligibleWeaknesses
      .sort(
        (first, second) =>
          first.accuracy - second.accuracy ||
          second.attempts - first.attempts
      )
      .slice(0, 3),
    recent: ordered.slice(0, 8),
  };
}
