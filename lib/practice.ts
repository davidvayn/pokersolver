import type { PreflopChart } from '@/data/preflop/ranges';
import {
  defaultScenarioForSeats,
  PREFLOP_SCENARIOS,
  scenarioSnapshot,
  type PreflopScenario,
} from '@/data/preflop/catalog';
import {
  comboKey,
  comboLabelToCombos,
  handClassLabel,
  parseRange,
} from '@/lib/cards';
import type { Position, TableSeats } from '@/lib/positions';
import type {
  PracticeAction,
  PracticeCategory,
  PracticeQuestion,
  PracticeRecord,
  PracticeRules,
} from '@/lib/practice-types';

export { analyzePractice } from '@/lib/practice-stats';
export type {
  ActionFrequency,
  PracticeAction,
  PracticeCategory,
  PracticeQuestion,
  PracticeRecord,
  PracticeRules,
  PracticeStats,
  StatBreakdown,
} from '@/lib/practice-types';

export const DEFAULT_PRACTICE_RULES: PracticeRules = {
  seats: 6,
  scenarioId: 'curated-6-max-100bb',
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
  const scenario = PREFLOP_SCENARIOS.find(
    (candidate) =>
      candidate.id === rules.scenarioId && candidate.seats === rules.seats
  );
  if (!scenario) return [];
  return scenario.charts.filter(
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
  id: string,
  scenario: PreflopScenario = defaultScenarioForSeats(seats)
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

  const options = [
    'Fold' as const,
    ...chart.actions.map((action) => action.name),
  ].filter((action, index, values) => values.indexOf(action) === index);
  const strategy = options.map((action) => ({
    action,
    frequency: frequencies.get(action) ?? 0,
  }));
  const recommendedAction = strategy.reduce((best, current) =>
    current.frequency > best.frequency ? current : best
  ).action;
  const correctActions = [recommendedAction];

  return {
    id,
    chartId: chart.id,
    category: chart.category,
    seats,
    hero: chart.hero,
    villain: chart.vs,
    scenario: scenarioSnapshot(scenario),
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
  const scenario = PREFLOP_SCENARIOS.find(
    (candidate) =>
      candidate.id === rules.scenarioId && candidate.seats === rules.seats
  );
  if (!scenario) return [];

  const seed = Date.now().toString(36);
  const makeChartBucket = (chart: PreflopChart) => {
      const byAction = new Map<PracticeAction, PracticeQuestion[]>();
      for (const handClass of handClasses()) {
        const question = createPracticeQuestion(
          chart,
          handClass,
          rules.seats,
          `${seed}-${chart.id}-${handClass}`,
          scenario
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
    scenario: question.scenario,
    handClass: question.handClass,
    chosenAction,
    recommendedAction: question.recommendedAction,
    correct: question.correctActions.includes(chosenAction),
    responseMs: Math.max(0, Math.round(responseMs)),
  };
}
