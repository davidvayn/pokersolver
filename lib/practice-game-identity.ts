import bundledManifests from '@/data/practice/full-hand-manifests.json';
import { canonicalCashGameRules, HOME_GAME_RULES, NL25_STUDY_RULES } from '@/lib/cash-game-rules';
import type { CashGameRules } from '@/lib/cash-game-rules';
import type { HandState, PracticeGameIdentity, PracticeHandRecord } from '@/lib/practice-types';

// Verified against crypto.subtle and the shared Rust/TypeScript fixtures.
export const HOME_RULES_SHA256 = '92121ddbdc970968be7cbb2e3519413b9c5d7f98a798d4500b55addb79f07f2d';
export const NL25_RULES_SHA256 = '6a31ca4bfbdf90213cd2bce3dcc0b75730bc2754bdc1fb593d789fd601a00908';
export const HOME_GAME_IDENTITY = Object.freeze({
  source: 'pinned' as const, profileId: HOME_GAME_RULES.id, rulesSha256: HOME_RULES_SHA256,
});
const UNKNOWN: PracticeGameIdentity = Object.freeze({ source: 'unresolved', profileId: null, rulesSha256: null });
const knownHomeVersions = new Set<string>([
  'hu-push-fold-v1',
  ...bundledManifests.filter((manifest) => manifest.abstraction.rake === 'none'
    && manifest.abstraction.blindsBb[0] === .5 && manifest.abstraction.blindsBb[1] === 1).map((manifest) => manifest.version),
]);

export function identityForRules(rules: CashGameRules): PracticeGameIdentity {
  const canonical = canonicalCashGameRules(rules);
  if (canonical === canonicalCashGameRules(HOME_GAME_RULES)) return HOME_GAME_IDENTITY;
  if (canonical === canonicalCashGameRules(NL25_STUDY_RULES)) {
    return { source: 'pinned', profileId: NL25_STUDY_RULES.id, rulesSha256: NL25_RULES_SHA256 };
  }
  throw new Error('Unregistered cash rules cannot be recorded as a known profile');
}

export function identityForHand(state: HandState): PracticeGameIdentity {
  return state.cash ? identityForRules(state.cash.rules) : HOME_GAME_IDENTITY;
}

export function isPracticeGameIdentity(value: unknown): value is PracticeGameIdentity {
  if (!value || typeof value !== 'object') return false;
  const identity = value as Partial<PracticeGameIdentity>;
  if (identity.source === 'unresolved') return identity.profileId === null && identity.rulesSha256 === null;
  if (identity.source !== 'pinned' && identity.source !== 'known-legacy-home') return false;
  return identity.profileId === HOME_GAME_RULES.id && identity.rulesSha256 === HOME_RULES_SHA256
    || identity.source === 'pinned' && identity.profileId === NL25_STUDY_RULES.id && identity.rulesSha256 === NL25_RULES_SHA256;
}

export function historicalGameIdentity(hand: Pick<PracticeHandRecord, 'modelVersion' | 'gameIdentity'>): PracticeGameIdentity {
  if (hand.gameIdentity) return isPracticeGameIdentity(hand.gameIdentity) ? hand.gameIdentity : UNKNOWN;
  return knownHomeVersions.has(hand.modelVersion)
    ? { ...HOME_GAME_IDENTITY, source: 'known-legacy-home' } : UNKNOWN;
}

export interface PracticeEvidenceScope { rulesSha256: string; depthBb: number }

export function handsForGame(hands: PracticeHandRecord[], scope: PracticeEvidenceScope): PracticeHandRecord[] {
  return hands.filter((hand) => hand.depthBb === scope.depthBb
    && historicalGameIdentity(hand).rulesSha256 === scope.rulesSha256);
}

/** Non-destructive metadata upgrade: no invented rake, payoffs or regrading. */
export function upgradeHistoryGameIdentity(hand: PracticeHandRecord): PracticeHandRecord {
  const identity = historicalGameIdentity(hand);
  return { ...hand, gameIdentity: identity, decisions: hand.decisions.map((decision) => ({
    ...decision, gameIdentity: decision.gameIdentity ?? identity,
  })) };
}
