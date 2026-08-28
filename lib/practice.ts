import type {
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
const OPPONENT_STYLES = new Set(['baseline', 'adaptive-exploitative']);
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
    opponentStyle: OPPONENT_STYLES.has(candidate.opponentStyle ?? '')
      ? (candidate.opponentStyle as PracticeSettings['opponentStyle'])
      : DEFAULT_PRACTICE_SETTINGS.opponentStyle,
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
    current.opponentStyle !== next.opponentStyle ||
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

export function postflopStreetForHand(
  streets: PracticeSettings['postflopStreets'],
  completedHands: number
): PracticeSettings['postflopStreets'][number] {
  const available = streets.length
    ? streets
    : DEFAULT_PRACTICE_SETTINGS.postflopStreets;
  const index = Math.abs(Math.trunc(completedHands)) % available.length;
  return available[index];
}

export function streetIndex(street: PracticeStreet): number {
  return STREET_ORDER.indexOf(street);
}
