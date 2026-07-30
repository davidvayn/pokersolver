import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import type { RawPushFoldArtifact } from '@/data/preflop/artifacts/types';
import { cardRank, cardToStr } from '@/lib/cards';
import {
  compactPushFoldToScenario,
  pushFoldArtifactToScenario,
  validatePushFoldArtifact,
} from '@/lib/preflop-artifacts';

function handLabel(first: number, second: number): string {
  const ranks = '23456789TJQKA';
  const firstRank = cardRank(first);
  const secondRank = cardRank(second);
  if (firstRank === secondRank) return ranks[firstRank] + ranks[firstRank];
  const high = Math.max(firstRank, secondRank);
  const low = Math.min(firstRank, secondRank);
  const suited = first % 4 === second % 4;
  return `${ranks[high]}${ranks[low]}${suited ? 's' : 'o'}`;
}

function artifact(): RawPushFoldArtifact {
  const exact_combos = [];
  for (let high = 1; high < 52; high++) {
    for (let low = 0; low < high; low++) {
      const combo_key = (high * (high - 1)) / 2 + low;
      const premium = cardRank(high) === 12 && cardRank(low) === 12;
      exact_combos.push({
        combo_key,
        cards: [high, low] as [number, number],
        card_names: [cardToStr(high), cardToStr(low)] as [string, string],
        label: handLabel(high, low),
        small_blind: {
          fold: premium ? 0 : 1,
          shove: premium ? 1 : 0,
        },
        big_blind_vs_shove: {
          fold: premium ? 0 : 1,
          call: premium ? 1 : 0,
        },
      });
    }
  }

  return {
    schema_version: 1,
    artifact_id: 'test-hu-10bb',
    config_hash: 'test-config-hash',
    solver_version: '0.1.0',
    model: 'heads-up-push-fold-monte-carlo-v1',
    generated_at_unix_seconds: 1,
    payoff_convention: 'test',
    config: {
      small_blind_bb: 0.5,
      big_blind_bb: 1,
      effective_stack_bb: 10,
      iterations: 1,
      equity_samples: 1,
      seed: 1,
    },
    metrics: {
      profile_small_blind_ev_bb: 0,
      small_blind_best_response_ev_bb: 0,
      small_blind_ev_vs_big_blind_best_response_bb: 0,
      nash_conv_bb: 0,
      exploitability_bb: 0,
      small_blind_best_response_equity_interval_bb: { low: 0, high: 0 },
      small_blind_ev_vs_big_blind_best_response_equity_interval_bb: {
        low: 0,
        high: 0,
      },
      nash_conv_equity_interval_bb: { low: 0, high: 0 },
      equity_standard_error_upper_bound: 0,
      called_payoff_standard_error_upper_bound_bb: 0,
      compatible_deals: 1326 * 1225,
      training_equity_cache_entries: 1,
      evaluation_equity_cache_entries: 1,
      evaluation_seed: 2,
    },
    validation: {
      status: 'approximate',
      quality: 'advisory',
      validation_version: '1',
      note: 'test',
      checks: [
        'finite_metrics',
        'best_response_ordering',
        'strategy_probability_sums',
        'aces_shove_and_call_sanity',
        'exploitability_advisory',
      ].map((name) => ({
        name,
        passed: true,
        value: 0,
        threshold: 0,
        comparison: '<=',
      })),
    },
    strategies: { exact_combos, hand_classes: [] },
  };
}

describe('offline preflop artifacts', () => {
  it('validates every exact combo and adapts both decision points', () => {
    const raw = artifact();
    expect(validatePushFoldArtifact(raw)).toEqual([]);

    const scenario = pushFoldArtifactToScenario(raw);
    expect(scenario.id).toBe('test-hu-10bb');
    expect(scenario.openingSize).toEqual({ kind: 'all-in' });
    expect(scenario.charts.map((chart) => chart.actions[0].name)).toEqual([
      'All-in',
      'Call',
    ]);
    expect(scenario.charts[0].actions[0].range).toContain('AsAh');
  });

  it('rejects failed validation and malformed frequencies', () => {
    const raw = artifact();
    raw.validation.status = 'rejected';
    raw.strategies.exact_combos[0].small_blind = {
      fold: 0.75,
      shove: 0.75,
    };

    const errors = validatePushFoldArtifact(raw);
    expect(errors).toEqual(
      expect.arrayContaining([
        'Solver validation status is rejected',
        'Invalid action frequencies for combo 0',
      ])
    );
    expect(() => pushFoldArtifactToScenario(raw)).toThrow(
      'Rejected preflop artifact'
    );
  });

  it('accepts a real Rust-generated artifact', () => {
    const raw: unknown = JSON.parse(
      readFileSync(
        new URL(
          '../preflop-solver/artifacts/hu-push-fold-2bb.json',
          import.meta.url
        ),
        'utf8'
      )
    );

    expect(validatePushFoldArtifact(raw)).toEqual([]);
  });

  it('returns validation errors instead of throwing on malformed JSON', () => {
    expect(validatePushFoldArtifact(null)).toEqual(['Invalid artifact root']);
    expect(() =>
      validatePushFoldArtifact({ schema_version: 1 })
    ).not.toThrow();
    expect(validatePushFoldArtifact({ schema_version: 1 })).toEqual(
      expect.arrayContaining([
        'Missing solver config',
        'Missing solver metrics',
        'Missing solver validation',
        'Missing exact combo strategies',
      ])
    );
  });

  it('loads the compact browser catalog and rejects duplicate hand labels', () => {
    const compact = JSON.parse(
      readFileSync(
        new URL('../data/preflop/solved-scenarios.json', import.meta.url),
        'utf8'
      )
    );
    expect(compactPushFoldToScenario(compact[0]).effectiveStackBb).toBe(2);

    const malformed = structuredClone(compact[0]);
    malformed.hands[1][0] = malformed.hands[0][0];
    expect(() => compactPushFoldToScenario(malformed)).toThrow(
      'Rejected compact preflop scenario'
    );

    expect(() => compactPushFoldToScenario(null)).toThrow('invalid root');
    const missingHands = structuredClone(compact[0]);
    delete missingHands.hands;
    expect(() => compactPushFoldToScenario(missingHands)).toThrow(
      'Rejected compact preflop scenario'
    );
    const nonCanonical = structuredClone(compact[0]);
    nonCanonical.hands[0][0] = 'KAo';
    expect(() => compactPushFoldToScenario(nonCanonical)).toThrow(
      'Rejected compact preflop scenario'
    );
  });
});
