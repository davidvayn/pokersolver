import bundledRules from '@/data/practice/cash-game-rules.json';

export interface CashGameRules {
  readonly schema: 'hu-cash-rules-v1';
  readonly id: string;
  readonly currency: 'BB' | 'USD';
  /** Milli-bb for legacy Home game; cents for the USD study profile. */
  readonly unitsPerBb: number;
  readonly blindsUnits: readonly [number, number];
  readonly playersDealt: 2;
  readonly anteUnits: 0;
  readonly rake: {
    readonly rateBasisPoints: number;
    readonly capUnits: number;
    readonly noFlopNoDrop: boolean;
    readonly rounding: 'half-to-even';
  };
  readonly splitPotRule: 'first-left-of-button';
  readonly betRounding: 'half-up' | 'half-to-even';
}

function record(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function exactFields(value: Record<string, unknown>, keys: readonly string[]): boolean {
  return Object.keys(value).length === keys.length && keys.every((key) => Object.hasOwn(value, key));
}

export function requireMoneyUnits(value: unknown): asserts value is number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0) {
    throw new Error('Money amounts must be nonnegative safe integer units');
  }
}

/** Validates data and copies it into a deeply frozen, canonical rules value. */
export function parseCashGameRules(value: unknown): CashGameRules {
  if (
    !record(value) ||
    !exactFields(value, ['schema', 'id', 'currency', 'unitsPerBb', 'blindsUnits',
      'playersDealt', 'anteUnits', 'rake', 'splitPotRule', 'betRounding']) ||
    value.schema !== 'hu-cash-rules-v1' ||
    typeof value.id !== 'string' ||
    !/^[a-z][a-z0-9-]{0,95}$/.test(value.id) ||
    (value.currency !== 'BB' && value.currency !== 'USD') ||
    !Array.isArray(value.blindsUnits) || value.blindsUnits.length !== 2 ||
    value.playersDealt !== 2 || value.anteUnits !== 0 ||
    !record(value.rake) ||
    !exactFields(value.rake, ['rateBasisPoints', 'capUnits', 'noFlopNoDrop', 'rounding']) ||
    value.splitPotRule !== 'first-left-of-button' ||
    (value.betRounding !== 'half-up' && value.betRounding !== 'half-to-even')
  ) {
    throw new Error('Unsupported heads-up cash rules');
  }
  const { unitsPerBb, blindsUnits, rake } = value;
  if (typeof rake.noFlopNoDrop !== 'boolean' || rake.rounding !== 'half-to-even') {
    throw new Error('Unsupported rake rounding or eligibility');
  }
  requireMoneyUnits(unitsPerBb);
  requireMoneyUnits(blindsUnits[0]);
  requireMoneyUnits(blindsUnits[1]);
  requireMoneyUnits(rake.rateBasisPoints);
  requireMoneyUnits(rake.capUnits);
  if (
    unitsPerBb === 0 || blindsUnits[0] === 0 ||
    blindsUnits[0] >= blindsUnits[1] || blindsUnits[1] !== unitsPerBb ||
    rake.rateBasisPoints > 10_000 ||
    ((rake.rateBasisPoints === 0) !== (rake.capUnits === 0)) ||
    (value.currency === 'USD' && (value.id === 'home-game-v1' || value.betRounding !== 'half-to-even')) ||
    (value.currency === 'BB' && (
      value.id !== 'home-game-v1' || unitsPerBb !== 1000 ||
      blindsUnits[0] !== 500 || rake.rateBasisPoints !== 0 ||
      rake.noFlopNoDrop !== false || value.betRounding !== 'half-up'
    ))
  ) {
    throw new Error('Inconsistent heads-up cash rules');
  }
  return Object.freeze({
    schema: value.schema,
    id: value.id,
    currency: value.currency,
    unitsPerBb,
    blindsUnits: Object.freeze([blindsUnits[0], blindsUnits[1]] as const),
    playersDealt: value.playersDealt,
    anteUnits: value.anteUnits,
    rake: Object.freeze({
      rateBasisPoints: rake.rateBasisPoints,
      capUnits: rake.capUnits,
      noFlopNoDrop: rake.noFlopNoDrop,
      rounding: rake.rounding,
    }),
    splitPotRule: value.splitPotRule,
    betRounding: value.betRounding,
  });
}

export const HOME_GAME_RULES = parseCashGameRules(bundledRules.home);
export const NL25_STUDY_RULES = parseCashGameRules(bundledRules.nl25);

/** Metadata such as labels and source URLs does not change game identity. */
export function canonicalCashGameRules(value: CashGameRules): string {
  const rules = parseCashGameRules(value);
  return [
    rules.schema, rules.id, rules.currency, rules.unitsPerBb,
    ...rules.blindsUnits, rules.playersDealt, rules.anteUnits,
    rules.rake.rateBasisPoints, rules.rake.capUnits,
    rules.rake.noFlopNoDrop ? 1 : 0, rules.rake.rounding,
    rules.splitPotRule, rules.betRounding,
  ].join('|');
}

export async function cashGameRulesDigest(rules: CashGameRules): Promise<string> {
  const bytes = new TextEncoder().encode(canonicalCashGameRules(rules));
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, '0')).join('');
}

/** Exact integer-ratio rounding shared with the native implementation. */
export function roundMoneyRatio(
  numerator: number,
  denominator: number,
  rounding: CashGameRules['betRounding'] = 'half-to-even'
): number {
  requireMoneyUnits(numerator);
  requireMoneyUnits(denominator);
  if (denominator === 0 || (rounding !== 'half-up' && rounding !== 'half-to-even')) {
    throw new Error('Invalid money rounding ratio');
  }
  return Number(roundRatio(BigInt(numerator), BigInt(denominator), rounding));
}

function roundRatio(numerator: bigint, denominator: bigint, rounding: CashGameRules['betRounding']): bigint {
  const whole = numerator / denominator;
  const twiceRemainder = 2n * (numerator % denominator);
  return whole + (twiceRemainder > denominator ||
    (twiceRemainder === denominator && (rounding === 'half-up' || whole % 2n === 1n)) ? 1n : 0n);
}

/** Bounds come from the betting engine; this does not validate a betting tree. */
export function quantizeCashRaiseAmounts(
  rules: CashGameRules,
  requestedUnitRatios: ReadonlyArray<readonly [number, number]>,
  minimumToUnits: number,
  maximumToUnits: number
): number[] {
  parseCashGameRules(rules);
  requireMoneyUnits(minimumToUnits);
  requireMoneyUnits(maximumToUnits);
  if (minimumToUnits > maximumToUnits) throw new Error('Invalid raise bounds');
  return [...new Set(requestedUnitRatios.map((ratio) => {
    if (!Array.isArray(ratio) || ratio.length !== 2) throw new Error('Invalid sizing ratio');
    const rounded = roundMoneyRatio(ratio[0], ratio[1], rules.betRounding);
    return Math.min(maximumToUnits, Math.max(minimumToUnits, rounded));
  }))].sort((left, right) => left - right);
}

/** Internal percentage product may exceed Number's exact multiplication range. */
export function cashRakeUnits(rules: CashGameRules, grossPotUnits: number, flopDealt: boolean): number {
  parseCashGameRules(rules);
  requireMoneyUnits(grossPotUnits);
  if (typeof flopDealt !== 'boolean') throw new Error('Invalid flop eligibility');
  if (rules.rake.noFlopNoDrop && !flopDealt) return 0;
  const rounded = roundRatio(
    BigInt(grossPotUnits) * BigInt(rules.rake.rateBasisPoints), 10_000n, 'half-to-even'
  );
  return Number(rounded > BigInt(rules.rake.capUnits) ? BigInt(rules.rake.capUnits) : rounded);
}
