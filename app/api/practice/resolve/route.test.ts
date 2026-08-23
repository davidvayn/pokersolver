import { describe, expect, it } from 'vitest';

import { applyAction, createHand, seededRandom } from '@/lib/practice-engine';
import {
  resolverAffinityKey,
  resolverQueryPayload,
} from '@/lib/server/practice-resolver-request';

describe('practice continual-resolver API boundary', () => {
  it('forwards only the acting private cards to the Rust solver process', () => {
    const state = createHand({
      modelVersion: 'resolver-v1',
      depthBb: 20,
      button: 'button-small-blind',
      hero: 'button-small-blind',
      random: seededRandom(17),
    });
    const payload = resolverQueryPayload(
      state,
      'a'.repeat(64),
      'resolver-v1',
      20
    );
    expect(payload.privateCards).toEqual(
      state.holeCards['button-small-blind']
    );
    expect(payload).not.toHaveProperty('holeCards');
    expect(JSON.stringify(payload)).not.toContain(
      JSON.stringify(state.holeCards['big-blind'])
    );
  });

  it('pins postflop descendants to the first chosen preflop branch', () => {
    const initial = createHand({
      id: 'affinity-hand',
      modelVersion: 'resolver-v1',
      depthBb: 20,
      button: 'button-small-blind',
      hero: 'button-small-blind',
      random: seededRandom(29),
    });
    const called = applyAction(initial, {
      id: 'call',
      kind: 'call',
      label: 'Call 0.5bb',
      amountToBb: 1,
    });
    const flop = applyAction(called, {
      id: 'check',
      kind: 'check',
      label: 'Check',
    });
    const raised = applyAction(initial, {
      id: 'raise-2',
      kind: 'raise',
      label: 'Raise to 2bb',
      amountToBb: 2,
    });

    expect(resolverAffinityKey(flop)).toBe(resolverAffinityKey(called));
    expect(resolverAffinityKey(raised)).not.toBe(resolverAffinityKey(called));
  });
});
