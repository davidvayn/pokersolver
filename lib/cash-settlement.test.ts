import { describe, expect, it } from 'vitest';
import fixtures from '@/data/practice/cash-settlement-fixtures.json';
import {
  HOME_GAME_RULES, NL25_STUDY_RULES, cashGameRulesDigest,
  canonicalCashGameRules, cashRakeUnits, parseCashGameRules,
  quantizeCashRaiseAmounts, roundMoneyRatio, type CashGameRules,
} from '@/lib/cash-game-rules';
import { settleCashHand, type CashTerminal } from '@/lib/cash-settlement';
import { PRACTICE_GAME_PROFILES, practiceGameProfile } from '@/lib/practice-game-profiles';

const rules: Record<string, CashGameRules> = { home: HOME_GAME_RULES, nl25: NL25_STUDY_RULES };

describe('cash rules and shared native fixtures', () => {
  for (const [name, identity] of Object.entries(fixtures.identities)) {
    it(`matches the frozen ${name} rules identity`, async () => {
      expect(canonicalCashGameRules(rules[name])).toBe(identity.canonical);
      expect(await cashGameRulesDigest(rules[name])).toBe(identity.sha256);
    });
  }

  it('freezes the registry and leaves online models explicitly untrained', () => {
    expect(Object.isFrozen(PRACTICE_GAME_PROFILES)).toBe(true);
    expect(Object.isFrozen(NL25_STUDY_RULES)).toBe(true);
    expect(Object.isFrozen(NL25_STUDY_RULES.rake)).toBe(true);
    expect(Object.isFrozen(NL25_STUDY_RULES.blindsUnits)).toBe(true);
    expect(NL25_STUDY_RULES.blindsUnits[0] / NL25_STUDY_RULES.unitsPerBb).toBe(0.4);
    expect(practiceGameProfile(NL25_STUDY_RULES.id)?.modelStatus).toBe('requires-rake-aware-training');
    expect(practiceGameProfile(NL25_STUDY_RULES.id)?.splitPotRuleProvenance).toBe('study-abstraction');
    expect(practiceGameProfile('unknown')).toBeNull();
  });

  it('changes identity for changed economics rather than inheriting a Home identity', async () => {
    const changed = parseCashGameRules({ ...NL25_STUDY_RULES,
      rake: { ...NL25_STUDY_RULES.rake, rateBasisPoints: 500 } });
    expect(await cashGameRulesDigest(changed)).not.toBe(await cashGameRulesDigest(NL25_STUDY_RULES));
    expect(() => parseCashGameRules({ ...HOME_GAME_RULES, rake: changed.rake })).toThrow();
  });

  it('rejects invalid, extended, and unsupported rules', () => {
    const invalid: unknown[] = [
      null, {}, { ...NL25_STUDY_RULES, schema: 'future' },
      { ...NL25_STUDY_RULES, id: 'bad|id' }, { ...NL25_STUDY_RULES, playersDealt: 6 },
      { ...NL25_STUDY_RULES, id: 'home-game-v1' },
      { ...NL25_STUDY_RULES, anteUnits: 1 }, { ...NL25_STUDY_RULES, blindsUnits: [10, 20] },
      { ...NL25_STUDY_RULES, unitsPerBb: 0 }, { ...NL25_STUDY_RULES, unitsPerBb: 25.5 },
      { ...NL25_STUDY_RULES, currency: 'EUR' }, { ...NL25_STUDY_RULES, betRounding: 'half-up' },
      { ...NL25_STUDY_RULES, unexpected: true },
      ...[10_001, -1, 0.5, Number.NaN].map((rateBasisPoints) => ({
        ...NL25_STUDY_RULES, rake: { ...NL25_STUDY_RULES.rake, rateBasisPoints } })),
      { ...NL25_STUDY_RULES, rake: { ...NL25_STUDY_RULES.rake, capUnits: 0 } },
      { ...NL25_STUDY_RULES, rake: { ...NL25_STUDY_RULES.rake, rounding: 'half-up' } },
    ];
    invalid.forEach((value) => expect(() => parseCashGameRules(value)).toThrow());
  });
});

describe('integer money rounding and sizing', () => {
  fixtures.rounding.forEach((fixture) => it(`rounds ${fixture.numerator}/${fixture.denominator} ${fixture.mode}`, () => {
    expect(roundMoneyRatio(fixture.numerator, fixture.denominator,
      fixture.mode as CashGameRules['betRounding'])).toBe(fixture.expected);
  }));
  fixtures.sizing.forEach((fixture) => it(`clamps and deduplicates ${fixture.rules} sizing`, () => {
    expect(quantizeCashRaiseAmounts(rules[fixture.rules], fixture.ratios as [number, number][],
      fixture.minimum, fixture.maximum)).toEqual(fixture.expected);
  }));
  it('rejects invalid ratios, unsafe numbers, and inverted bounds', () => {
    [[1, 0], [-1, 2], [1.5, 2], [Infinity, 2], [Number.MAX_SAFE_INTEGER + 1, 1]]
      .forEach(([numerator, denominator]) => expect(() => roundMoneyRatio(numerator, denominator)).toThrow());
    expect(() => quantizeCashRaiseAmounts(NL25_STUDY_RULES, [[10, 1]], 100, 50)).toThrow();
    expect(() => cashRakeUnits(NL25_STUDY_RULES, -1, true)).toThrow();
    // The percentage product exceeds safe Number multiplication but uses BigInt.
    expect(cashRakeUnits(NL25_STUDY_RULES, Number.MAX_SAFE_INTEGER, true)).toBe(50);
  });
});

describe('exact heads-up settlement', () => {
  fixtures.settlements.forEach((fixture) => it(fixture.name, () => {
    const settled = settleCashHand(rules[fixture.rules], fixture.terminal as unknown as CashTerminal);
    expect(settled).toEqual(fixture.expected);
    expect(settled.netPayoffUnits[0] + settled.netPayoffUnits[1] + settled.rakeUnits).toBe(0);
    expect(settled.awardsUnits[0] + settled.awardsUnits[1] + settled.rakeUnits).toBe(settled.grossPotUnits);
    expect(settled.grossPotUnits + settled.uncalledRefundUnits[0] + settled.uncalledRefundUnits[1])
      .toBe(fixture.terminal.committedUnits[0] + fixture.terminal.committedUnits[1]);
  }));

  it('rejects reviews, unfinished runouts, illegal fold outcomes, and overflow', () => {
    const base: CashTerminal = { committedUnits: [125, 125], button: 0, reason: 'showdown',
      outcome: 'player-zero', boardCardsDealt: 5 };
    const invalid = [
      { ...base, reason: 'preflop-complete' }, { ...base, reason: 'review-complete' },
      { ...base, unexpected: true },
      { ...base, boardCardsDealt: 0 }, { ...base, boardCardsDealt: 3 },
      { ...base, boardCardsDealt: 2 }, { ...base, button: 2 },
      { ...base, committedUnits: [-1, 125] }, { ...base, committedUnits: [1.5, 125] },
      { ...base, committedUnits: [125] },
      { ...base, committedUnits: [Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER] },
      { ...base, reason: 'fold', outcome: 'split' },
      { ...base, reason: 'fold', committedUnits: [50, 125] },
    ];
    invalid.forEach((value) => expect(() => settleCashHand(NL25_STUDY_RULES, value as CashTerminal)).toThrow());
  });

  it('conserves exact chips across many terminal amounts and both seats', () => {
    for (const profile of Object.values(rules)) {
      for (let committed = 1; committed <= 1200; committed += 7) {
        for (const button of [0, 1] as const) {
          for (const outcome of ['player-zero', 'player-one', 'split'] as const) {
            const result = settleCashHand(profile, { committedUnits: [committed, committed],
              button, reason: 'showdown', outcome, boardCardsDealt: 5 });
            expect(result.netPayoffUnits[0] + result.netPayoffUnits[1] + result.rakeUnits).toBe(0);
            expect(result.awardsUnits.every(Number.isSafeInteger)).toBe(true);
            expect(result.rakeUnits).toBeLessThanOrEqual(profile.rake.capUnits);
          }
        }
      }
    }
  });
});
