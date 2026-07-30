import {
  formatForSeats,
  positionLabelForSeats,
} from '@/lib/positions';
import type {
  PracticeRecord,
  PracticeStats,
  StatBreakdown,
} from '@/lib/practice-types';

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
  const days = [
    ...new Set(records.map((record) => localDay(record.answeredAt))),
  ].sort((first, second) => second - first);
  if (days.length === 0) return 0;

  const today = localDay(now);
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  if (days[0] !== today && days[0] !== yesterday.getTime()) return 0;

  let streak = 1;
  const cursor = new Date(days[0]);
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
    (record) =>
      `${record.category}:${record.scenario?.openingSize.kind ?? 'legacy'}`,
    (record) =>
      record.category === 'RFI'
        ? record.scenario?.openingSize.kind === 'all-in'
          ? 'Push or fold'
          : 'Raise first in'
        : record.scenario?.openingSize.kind === 'all-in'
          ? 'Facing a shove'
          : 'Facing an open'
  );
  const byAction = breakdown(
    records,
    (record) => record.recommendedAction,
    (record) => record.recommendedAction
  );
  const byScenario = breakdown(
    records,
    (record) =>
      record.scenario?.scenarioId ??
      `legacy-curated-${record.seats}-max-100bb`,
    (record) =>
      record.scenario
        ? `${formatForSeats(record.seats).label} · ${record.scenario.effectiveStackBb}bb · ${
            record.scenario.provenance.source === 'offline-solver'
              ? 'Solver'
              : 'Curated'
          }`
        : `${formatForSeats(record.seats).label} · Legacy curated`
  );
  const eligibleWeaknesses = [
    ...byFormat,
    ...byPosition,
    ...byCategory,
    ...byAction,
    ...byScenario,
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
    byScenario,
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

export type {
  PracticeRecord,
  PracticeStats,
  StatBreakdown,
} from '@/lib/practice-types';
