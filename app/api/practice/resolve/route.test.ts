import { describe, expect, it } from 'vitest';

import { createHand, seededRandom } from '@/lib/practice-engine';
import { resolverQueryPayload } from '@/lib/server/practice-resolver-request';

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
});
