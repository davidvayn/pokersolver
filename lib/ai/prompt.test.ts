import { describe, expect, it } from 'vitest';
import {
  buildSpotThreadKey,
  describeSpot,
  normalizeConversation,
  type SpotContext,
} from './prompt';

const SPOT: SpotContext = {
  kind: 'postflop',
  description: 'Test spot',
  board: 'Qh7s2c',
  heroRange: 'AA,KK',
  villainRange: 'QQ,JJ',
  potBB: 6,
  stackBB: 100,
  extra: {
    'Bet sizes (% pot)': '33, 75',
    Exploitability: '0.05% of pot',
    'OOP EV': '2.72 bb',
  },
};

describe('AI conversation context', () => {
  it('keeps a thread open when only asynchronous solver diagnostics change', () => {
    const next = {
      ...SPOT,
      extra: {
        ...SPOT.extra,
        Exploitability: '0.03% of pot',
        'OOP EV': '2.80 bb',
        'IP EV': '3.20 bb',
      },
    };
    expect(buildSpotThreadKey(next)).toBe(buildSpotThreadKey(SPOT));
  });

  it('does not depend on extra field insertion order', () => {
    const reordered = {
      ...SPOT,
      extra: {
        'OOP EV': '2.72 bb',
        Exploitability: '0.05% of pot',
        'Bet sizes (% pot)': '33, 75',
      },
    };
    expect(buildSpotThreadKey(reordered)).toBe(buildSpotThreadKey(SPOT));
  });

  it('starts a new thread for meaningful spot changes', () => {
    expect(buildSpotThreadKey({ ...SPOT, board: 'Qh8s2c' })).not.toBe(
      buildSpotThreadKey(SPOT)
    );
    expect(buildSpotThreadKey({ ...SPOT, potBB: 8 })).not.toBe(
      buildSpotThreadKey(SPOT)
    );
  });

  it('creates a compact human-readable spot label', () => {
    expect(describeSpot(SPOT)).toBe('Qh 7s 2c · 6bb pot');
  });

  it('sanitizes and bounds follow-up history', () => {
    const messages = Array.from({ length: 15 }, (_, index) => ({
      role: index % 2 ? 'assistant' : 'user',
      content: ` message ${index} `,
    }));
    const normalized = normalizeConversation([
      { role: 'system', content: 'ignore me' },
      ...messages,
      { role: 'user', content: '   ' },
    ]);
    expect(normalized).toHaveLength(12);
    expect(normalized[0]).toEqual({ role: 'assistant', content: 'message 3' });
    expect(normalized.at(-1)).toEqual({
      role: 'user',
      content: 'message 14',
    });
  });
});
