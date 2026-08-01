import type {
  PracticeDecisionRecord,
  PracticeSettings,
  PracticeStreet,
  Seat,
} from '@/lib/practice-types';
import {
  DEFAULT_PRACTICE_SETTINGS,
  STREET_ORDER,
} from '@/lib/practice-types';

export * from '@/lib/practice-engine';
export * from '@/lib/practice-grading';
export type * from '@/lib/practice-types';

export const PRACTICE_SETTINGS_KEY = 'poker-lab-practice-settings-v3';

const MODES = new Set(['full-hand', 'preflop', 'postflop', 'push-fold']);
const FULL_DEPTHS = new Set([20, 50, 100]);
const PUSH_FOLD_DEPTHS = new Set([2, 3, 5, 8, 10, 12, 15, 20]);
const HERO_SEATS = new Set([
  'alternate',
  'button-small-blind',
  'big-blind',
]);
const DEAL_MODES = new Set(['authentic', 'adaptive']);
const GOALS = new Set(['continuous', 25, 50, 100]);
const POSTFLOP_STREETS = new Set(['flop', 'turn', 'river']);

export function sanitizePracticeSettings(value: unknown): PracticeSettings {
  if (!value || typeof value !== 'object') return DEFAULT_PRACTICE_SETTINGS;
  const candidate = value as Partial<PracticeSettings>;
  const postflopStreets = Array.isArray(candidate.postflopStreets)
    ? candidate.postflopStreets.filter(
        (street): street is Exclude<PracticeStreet, 'preflop'> =>
          POSTFLOP_STREETS.has(street)
      )
    : DEFAULT_PRACTICE_SETTINGS.postflopStreets;
  return {
    mode: MODES.has(candidate.mode ?? '')
      ? (candidate.mode as PracticeSettings['mode'])
      : DEFAULT_PRACTICE_SETTINGS.mode,
    depthBb: FULL_DEPTHS.has(candidate.depthBb ?? -1)
      ? (candidate.depthBb as PracticeSettings['depthBb'])
      : DEFAULT_PRACTICE_SETTINGS.depthBb,
    pushFoldDepthBb: PUSH_FOLD_DEPTHS.has(candidate.pushFoldDepthBb ?? -1)
      ? (candidate.pushFoldDepthBb as PracticeSettings['pushFoldDepthBb'])
      : DEFAULT_PRACTICE_SETTINGS.pushFoldDepthBb,
    postflopStreets:
      postflopStreets.length > 0
        ? postflopStreets
        : DEFAULT_PRACTICE_SETTINGS.postflopStreets,
    heroSeat: HERO_SEATS.has(candidate.heroSeat ?? '')
      ? (candidate.heroSeat as PracticeSettings['heroSeat'])
      : DEFAULT_PRACTICE_SETTINGS.heroSeat,
    dealMode: DEAL_MODES.has(candidate.dealMode ?? '')
      ? (candidate.dealMode as PracticeSettings['dealMode'])
      : DEFAULT_PRACTICE_SETTINGS.dealMode,
    decisionGoal: GOALS.has(candidate.decisionGoal ?? '')
      ? (candidate.decisionGoal as PracticeSettings['decisionGoal'])
      : DEFAULT_PRACTICE_SETTINGS.decisionGoal,
  };
}

export function loadPracticeSettings(): PracticeSettings {
  if (typeof window === 'undefined') return DEFAULT_PRACTICE_SETTINGS;
  try {
    return sanitizePracticeSettings(
      JSON.parse(localStorage.getItem(PRACTICE_SETTINGS_KEY) ?? 'null')
    );
  } catch {
    return DEFAULT_PRACTICE_SETTINGS;
  }
}

export function savePracticeSettings(settings: PracticeSettings): void {
  if (typeof window === 'undefined') return;
  try {
    localStorage.setItem(PRACTICE_SETTINGS_KEY, JSON.stringify(settings));
  } catch {
    // Settings persistence is helpful but must never block practice.
  }
}

export function structuralSettingsChanged(
  current: PracticeSettings,
  next: PracticeSettings
): boolean {
  return (
    current.mode !== next.mode ||
    current.depthBb !== next.depthBb ||
    current.pushFoldDepthBb !== next.pushFoldDepthBb ||
    current.heroSeat !== next.heroSeat ||
    current.dealMode !== next.dealMode ||
    current.postflopStreets.join(',') !== next.postflopStreets.join(',')
  );
}

export function nextHeroSeat(
  setting: PracticeSettings['heroSeat'],
  completedHands: number
): Seat {
  if (setting === 'button-small-blind' || setting === 'big-blind') return setting;
  return completedHands % 2 === 0
    ? 'button-small-blind'
    : 'big-blind';
}

export interface AdaptiveGroup {
  key: string;
  street: PracticeStreet;
  position: Seat;
  depthBb: number;
  handBucket: string;
  facingAction: string;
  attempts: number;
  averageEvLossBb: number;
}

export function adaptiveGroups(
  records: PracticeDecisionRecord[]
): AdaptiveGroup[] {
  const latest = [...records]
    .sort((first, second) => second.answeredAt - first.answeredAt)
    .slice(0, 200);
  const groups = new Map<
    string,
    Omit<AdaptiveGroup, 'averageEvLossBb'> & { totalLoss: number; graded: number }
  >();
  for (const record of latest) {
    const key = [
      record.street,
      record.position,
      record.depthBb,
      record.handBucket,
      record.facingAction,
    ].join('|');
    const current = groups.get(key) ?? {
      key,
      street: record.street,
      position: record.position,
      depthBb: record.depthBb,
      handBucket: record.handBucket,
      facingAction: record.facingAction,
      attempts: 0,
      totalLoss: 0,
      graded: 0,
    };
    current.attempts += 1;
    if (record.evLossBb !== null) {
      current.totalLoss += record.evLossBb;
      current.graded += 1;
    }
    groups.set(key, current);
  }
  return [...groups.values()]
    .map(({ totalLoss, graded, ...group }) => ({
      ...group,
      averageEvLossBb: graded ? totalLoss / graded : 0,
    }))
    .sort(
      (first, second) =>
        second.averageEvLossBb - first.averageEvLossBb ||
        first.attempts - second.attempts
    );
}

export function chooseAdaptiveGroup(
  groups: AdaptiveGroup[],
  random: () => number = Math.random
): AdaptiveGroup | null {
  if (groups.length === 0 || random() >= 0.7) return null;
  const maxAttempts = Math.max(...groups.map((group) => group.attempts), 1);
  const weights = groups.map(
    (group) =>
      0.01 +
      group.averageEvLossBb +
      (maxAttempts - group.attempts) / maxAttempts
  );
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  let roll = random() * total;
  for (let index = 0; index < groups.length; index++) {
    roll -= weights[index];
    if (roll <= 0) return groups[index];
  }
  return groups[groups.length - 1];
}

export function streetIndex(street: PracticeStreet): number {
  return STREET_ORDER.indexOf(street);
}
