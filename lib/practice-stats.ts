import type {
  EvBreakdown,
  PracticeDecisionRecord,
  PracticeGrade,
  PracticeHandRecord,
  PracticeStats,
  PracticeTrendPoint,
} from '@/lib/practice-types';

const DAY_MS = 24 * 60 * 60 * 1_000;
const TREND_DAYS = 21;
const ACTIVITY_DAYS = 84;
const STRONG_GRADES = new Set<PracticeGrade>(['perfect', 'excellent', 'good']);
const GRADE_ORDER: PracticeGrade[] = [
  'perfect',
  'excellent',
  'good',
  'inaccuracy',
  'mistake',
  'blunder',
];

export interface PracticeDecisionSummary {
  decisions: number;
  gradedDecisions: number;
  strongDecisions: number;
  strongDecisionPercentage: number;
  gradedCoveragePercentage: number;
  averageEvLossBb: number | null;
  totalEvLossBb: number;
  lowConfidencePercentage: number;
  averageResponseMs: number;
}

function mean(values: number[]): number | null {
  if (values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function breakdown(
  decisions: PracticeDecisionRecord[],
  keyFor: (record: PracticeDecisionRecord) => string,
  labelFor: (record: PracticeDecisionRecord) => string
): EvBreakdown[] {
  const groups = new Map<
    string,
    { label: string; records: PracticeDecisionRecord[] }
  >();
  for (const record of decisions) {
    const key = keyFor(record);
    const current = groups.get(key) ?? { label: labelFor(record), records: [] };
    current.records.push(record);
    groups.set(key, current);
  }
  return [...groups.entries()]
    .map(([key, group]) => {
      const losses = group.records
        .map((record) => record.evLossBb)
        .filter((loss): loss is number => loss !== null);
      const lowConfidence = group.records.filter(
        (record) => record.lowConfidence
      ).length;
      return {
        key,
        label: group.label,
        decisions: group.records.length,
        graded: losses.length,
        averageEvLossBb: mean(losses),
        totalEvLossBb: losses.reduce((sum, loss) => sum + loss, 0),
        lowConfidencePercentage: group.records.length
          ? lowConfidence / group.records.length
          : 0,
      };
    })
    .sort(
      (first, second) =>
        (second.averageEvLossBb ?? -1) - (first.averageEvLossBb ?? -1) ||
        second.decisions - first.decisions
    );
}

function averageLoss(records: PracticeDecisionRecord[]): number | null {
  return mean(
    records
      .map((record) => record.evLossBb)
      .filter((loss): loss is number => loss !== null)
  );
}

function localDayOrdinal(timestamp: number): number {
  const date = new Date(timestamp);
  return Date.UTC(date.getFullYear(), date.getMonth(), date.getDate());
}

function dayKey(timestamp: number): string {
  const date = new Date(timestamp);
  return [
    date.getFullYear(),
    String(date.getMonth() + 1).padStart(2, '0'),
    String(date.getDate()).padStart(2, '0'),
  ].join('-');
}

function dayLabel(timestamp: number): string {
  return new Intl.DateTimeFormat('en', {
    month: 'short',
    day: 'numeric',
  }).format(timestamp);
}

function dailySeries(
  decisions: PracticeDecisionRecord[],
  days: number,
  now: number
): PracticeTrendPoint[] {
  const byDay = new Map<string, PracticeDecisionRecord[]>();
  for (const decision of decisions) {
    const key = dayKey(decision.answeredAt);
    const current = byDay.get(key) ?? [];
    current.push(decision);
    byDay.set(key, current);
  }
  const end = new Date(now);
  end.setHours(0, 0, 0, 0);
  return Array.from({ length: days }, (_, index) => {
    const day = new Date(end);
    day.setDate(end.getDate() - (days - index - 1));
    const timestamp = day.getTime();
    const records = byDay.get(dayKey(timestamp)) ?? [];
    const graded = records.filter((record) => record.evLossBb !== null);
    const strong = records.filter((record) => STRONG_GRADES.has(record.grade));
    return {
      key: dayKey(timestamp),
      label: dayLabel(timestamp),
      timestamp,
      decisions: records.length,
      graded: graded.length,
      averageEvLossBb: averageLoss(graded),
      strongDecisionPercentage: records.length
        ? strong.length / records.length
        : null,
      averageResponseMs: records.length
        ? records.reduce((sum, record) => sum + record.responseMs, 0) /
          records.length
        : null,
    };
  });
}

function streaks(decisions: PracticeDecisionRecord[], now: number) {
  const active = [
    ...new Set(decisions.map((decision) => localDayOrdinal(decision.answeredAt))),
  ].sort((first, second) => first - second);
  let longest = 0;
  let run = 0;
  for (let index = 0; index < active.length; index += 1) {
    run = index > 0 && active[index] - active[index - 1] === DAY_MS ? run + 1 : 1;
    longest = Math.max(longest, run);
  }
  const today = localDayOrdinal(now);
  const mostRecent = active.at(-1);
  let current =
    mostRecent === today || mostRecent === today - DAY_MS ? 1 : 0;
  if (current) {
    for (let index = active.length - 1; index > 0; index -= 1) {
      if (active[index] - active[index - 1] !== DAY_MS) break;
      current += 1;
    }
  }
  return { activeDays: active.length, current, longest };
}

export function summarizePracticeDecisions(
  decisions: PracticeDecisionRecord[]
): PracticeDecisionSummary {
  const losses = decisions
    .map((record) => record.evLossBb)
    .filter((loss): loss is number => loss !== null);
  const strong = decisions.filter((record) => STRONG_GRADES.has(record.grade)).length;
  const lowConfidence = decisions.filter((record) => record.lowConfidence).length;
  return {
    decisions: decisions.length,
    gradedDecisions: losses.length,
    strongDecisions: strong,
    strongDecisionPercentage: decisions.length ? strong / decisions.length : 0,
    gradedCoveragePercentage: decisions.length ? losses.length / decisions.length : 0,
    averageEvLossBb: mean(losses),
    totalEvLossBb: losses.reduce((sum, loss) => sum + loss, 0),
    lowConfidencePercentage: decisions.length
      ? lowConfidence / decisions.length
      : 0,
    averageResponseMs: decisions.length
      ? decisions.reduce((sum, record) => sum + record.responseMs, 0) /
        decisions.length
      : 0,
  };
}

export function analyzePractice(
  hands: PracticeHandRecord[],
  now = Date.now()
): PracticeStats {
  const decisions = hands
    .flatMap((hand) => hand.decisions)
    .sort((first, second) => second.answeredAt - first.answeredAt);
  const summary = summarizePracticeDecisions(decisions);
  const recent = decisions.slice(0, 50);
  const previous = decisions.slice(50, 100);
  const recentLoss = averageLoss(recent);
  const previousLoss = averageLoss(previous);
  const byStreet = breakdown(decisions, (record) => record.street, (record) => record.street);
  const byStack = breakdown(
    decisions,
    (record) => String(record.depthBb),
    (record) => `${record.depthBb}bb`
  );
  const byPosition = breakdown(
    decisions,
    (record) => record.position,
    (record) =>
      record.position === 'button-small-blind' ? 'BTN / SB' : 'Big blind'
  );
  const byAction = breakdown(
    decisions,
    (record) => record.chosenAction.kind,
    (record) =>
      record.chosenAction.kind === 'all-in'
        ? 'All-in'
        : record.chosenAction.kind.charAt(0).toUpperCase() +
          record.chosenAction.kind.slice(1)
  );
  const byMode = breakdown(
    decisions,
    (record) => record.mode,
    (record) => record.mode.replace('-', ' ')
  );
  const bySeverity = breakdown(
    decisions,
    (record) => record.grade,
    (record) => record.grade
  );
  const byResponseTime = breakdown(
    decisions,
    (record) => {
      if (record.responseMs < 2_000) return 'under-2';
      if (record.responseMs < 5_000) return '2-5';
      if (record.responseMs < 10_000) return '5-10';
      return 'over-10';
    },
    (record) => {
      if (record.responseMs < 2_000) return 'Under 2s';
      if (record.responseMs < 5_000) return '2–5s';
      if (record.responseMs < 10_000) return '5–10s';
      return '10s+';
    }
  );
  const weaknesses = breakdown(
    decisions.slice(0, 200),
    (record) =>
      [record.street, record.position, record.depthBb, record.handBucket, record.facingAction].join('|'),
    (record) =>
      `${record.street} · ${record.position === 'button-small-blind' ? 'BTN / SB' : 'BB'} · ${record.handBucket} · ${record.facingAction}`
  )
    .filter((item) => item.decisions >= 2)
    .slice(0, 6);
  const streak = streaks(decisions, now);
  const dailyTrend = dailySeries(decisions, TREND_DAYS, now);
  const activity = dailySeries(decisions, ACTIVITY_DAYS, now).map((point) => ({
    key: point.key,
    label: point.label,
    timestamp: point.timestamp,
    decisions: point.decisions,
  }));

  return {
    hands: hands.length,
    ...summary,
    averageHandDurationMs: hands.length
      ? hands.reduce(
          (sum, hand) => sum + Math.max(0, hand.completedAt - hand.startedAt),
          0
        ) / hands.length
      : 0,
    decisionsPerHand: hands.length ? decisions.length / hands.length : 0,
    activeDays: streak.activeDays,
    currentStreakDays: streak.current,
    longestStreakDays: streak.longest,
    trendEvLossBb:
      recent.length === 50 &&
      previous.length === 50 &&
      recentLoss !== null &&
      previousLoss !== null
        ? recentLoss - previousLoss
        : null,
    dailyTrend,
    activity,
    gradeDistribution: GRADE_ORDER.map((grade) => {
      const count = decisions.filter((record) => record.grade === grade).length;
      return {
        grade,
        label: grade.charAt(0).toUpperCase() + grade.slice(1),
        decisions: count,
        percentage: decisions.length ? count / decisions.length : 0,
      };
    }),
    decisionPoints: decisions
      .filter(
        (record): record is PracticeDecisionRecord & { evLossBb: number } =>
          record.evLossBb !== null
      )
      .slice(0, 80)
      .reverse()
      .map((record) => ({
        id: record.id,
        label: `${record.handBucket} · ${record.street} · ${record.chosenAction.label}`,
        responseMs: record.responseMs,
        evLossBb: record.evLossBb,
        grade: record.grade,
      })),
    byStreet,
    byStack,
    byPosition,
    byAction,
    byMode,
    bySeverity,
    byResponseTime,
    weaknesses,
    recentCostly: decisions
      .filter((record) => record.evLossBb !== null)
      .sort((first, second) => (second.evLossBb ?? 0) - (first.evLossBb ?? 0))
      .slice(0, 12),
  };
}

export type {
  EvBreakdown,
  PracticeDecisionRecord,
  PracticeHandRecord,
  PracticeStats,
} from '@/lib/practice-types';
