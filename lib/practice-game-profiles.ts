import { HOME_GAME_RULES, NL25_STUDY_RULES, type CashGameRules } from '@/lib/cash-game-rules';

/** Requested training targets, deliberately separate from active manifests. */
export const REQUESTED_PRACTICE_STACK_DEPTHS_BB = [
  20, 40, 50, 100, 200, 1_000, 2_000,
] as const;

export interface OnlineCashReference {
  id: string;
  room: string;
  label: string;
  currency: 'USD';
  smallBlindCents: number;
  bigBlindCents: number;
  playersDealt: 2;
  anteBb: 0;
  rake: {
    rateBasisPoints: number;
    capCents: number;
    noFlopNoDrop: true;
    rounding: 'nearest-cent-half-to-even';
  };
  sourceUrl: string;
  checkedAt: string;
  modelStatus: 'requires-rake-aware-training';
}

export interface PracticeGameProfile {
  readonly id: string;
  readonly label: string;
  readonly rules: CashGameRules;
  readonly sourceUrl: string | null;
  readonly checkedAt: string | null;
  /** The public cash-game documentation did not verify an odd-cent rule. */
  readonly splitPotRuleProvenance: 'study-abstraction';
  readonly modelStatus: 'legacy-home-policies' | 'requires-rake-aware-training';
}

export const PRACTICE_GAME_PROFILES: readonly PracticeGameProfile[] = Object.freeze([
  Object.freeze({
    id: HOME_GAME_RULES.id, label: 'Home game', rules: HOME_GAME_RULES,
    sourceUrl: null, checkedAt: null, splitPotRuleProvenance: 'study-abstraction' as const,
    modelStatus: 'legacy-home-policies' as const,
  }),
  Object.freeze({
    id: NL25_STUDY_RULES.id, label: 'PokerStars NL25', rules: NL25_STUDY_RULES,
    sourceUrl: 'https://www.pokerstars.com/poker/room/rake/', checkedAt: '2026-10-06',
    splitPotRuleProvenance: 'study-abstraction' as const,
    modelStatus: 'requires-rake-aware-training' as const,
  }),
]);

export function practiceGameProfile(id: string): PracticeGameProfile | null {
  return PRACTICE_GAME_PROFILES.find((profile) => profile.id === id) ?? null;
}

// A mainstream room's published USD regular NLHE / two-player schedule, not
// a universal online rake or a trained policy. NL20 ($0.10/$0.20) is not listed
// in this schedule; do not interpolate or relabel NL25 as an exact NL20 game.
// Preflop all-ins that are called run out a board: no-flop-no-drop does NOT
// mean every hand with its final betting action preflop is free of rake.
// These study depths are not assertions about the room's permitted buy-ins.
export const DEFAULT_ONLINE_CASH_REFERENCE: OnlineCashReference = Object.freeze({
  id: NL25_STUDY_RULES.id,
  room: 'PokerStars',
  label: 'PokerStars NL25',
  currency: 'USD',
  smallBlindCents: NL25_STUDY_RULES.blindsUnits[0],
  bigBlindCents: NL25_STUDY_RULES.blindsUnits[1],
  playersDealt: 2,
  anteBb: 0,
  rake: Object.freeze({
    rateBasisPoints: NL25_STUDY_RULES.rake.rateBasisPoints,
    capCents: NL25_STUDY_RULES.rake.capUnits,
    noFlopNoDrop: true as const,
    rounding: 'nearest-cent-half-to-even' as const,
  }),
  sourceUrl: 'https://www.pokerstars.com/poker/room/rake/',
  checkedAt: '2026-10-06',
  modelStatus: 'requires-rake-aware-training',
});
