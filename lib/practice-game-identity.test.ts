import { describe, expect, it } from 'vitest';
import { cashGameRulesDigest, HOME_GAME_RULES, NL25_STUDY_RULES } from '@/lib/cash-game-rules';
import { HOME_GAME_IDENTITY, HOME_RULES_SHA256, NL25_RULES_SHA256, historicalGameIdentity,
  identityForRules, isPracticeGameIdentity } from '@/lib/practice-game-identity';

describe('immutable practice game identities', () => {
  it('matches both stored identities to canonical rules hashes', async () => {
    expect(await cashGameRulesDigest(HOME_GAME_RULES)).toBe(HOME_RULES_SHA256);
    expect(await cashGameRulesDigest(NL25_STUDY_RULES)).toBe(NL25_RULES_SHA256);
  });

  it('never identifies changed rules or unfamiliar legacy models as Home', () => {
    expect(identityForRules(HOME_GAME_RULES)).toEqual(HOME_GAME_IDENTITY);
    expect(() => identityForRules({ ...NL25_STUDY_RULES,
      rake: { ...NL25_STUDY_RULES.rake, capUnits: 100 } })).toThrow('Unregistered');
    expect(historicalGameIdentity({ modelVersion: 'hu-20bb-unknown-experimental' }).source).toBe('unresolved');
    expect(historicalGameIdentity({ modelVersion: 'hu-push-fold-v1' }).source).toBe('known-legacy-home');
    expect(isPracticeGameIdentity({ source: 'pinned', profileId: NL25_STUDY_RULES.id, rulesSha256: HOME_RULES_SHA256 })).toBe(false);
    expect(isPracticeGameIdentity({ source: 'known-legacy-home', profileId: NL25_STUDY_RULES.id, rulesSha256: NL25_RULES_SHA256 })).toBe(false);
  });
});
