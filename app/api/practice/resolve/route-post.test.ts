import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  modelForFullDepth: vi.fn(),
  query: vi.fn(),
}));

vi.mock('server-only', () => ({}));
vi.mock('@/lib/practice-models', () => ({
  modelForFullDepth: mocks.modelForFullDepth,
}));
vi.mock('@/lib/server/practice-solver-process', () => ({
  PRACTICE_RESOLVER_IDENTITY: {
    modelVersion: 'resolver-v1',
    networkSha256: 'a'.repeat(64),
    rangePolicySha256: 'b'.repeat(64),
    valueNetworkSha256: 'c'.repeat(64),
    preflopActionValuesSha256: 'd'.repeat(64),
  },
  practiceSolverProcess: () => ({ query: mocks.query }),
}));

import { POST } from '@/app/api/practice/resolve/route';
import { createHand, seededRandom } from '@/lib/practice-engine';

const stateHash = 'e'.repeat(64);

function resolverResponse(valueNetworkSha256 = 'c'.repeat(64)) {
  return {
    schema: 'hu-practice-continual-resolver-query-v1',
    requestId: 'rust-owned-request-id',
    stateHash,
    modelVersion: 'resolver-v1',
    depthBb: 20,
    networkSha256: 'a'.repeat(64),
    rangePolicySha256: 'b'.repeat(64),
    valueNetworkSha256,
    preflopActionValuesSha256: 'd'.repeat(64),
    maximumProbabilitySumError: 0,
    actions: [{ kind: 'fold', probability: 1 }],
  };
}

describe('practice continual-resolver POST route', () => {
  beforeEach(() => {
    mocks.modelForFullDepth.mockReset();
    mocks.query.mockReset();
    mocks.modelForFullDepth.mockReturnValue({
      version: 'resolver-v1',
      runtime: {
        kind: 'rust-continual-resolver-v1',
      },
    });
    mocks.query.mockResolvedValue(resolverResponse());
  });

  it('replays an exact pinned request and strips the non-acting private hand', async () => {
    const state = createHand({
      modelVersion: 'resolver-v1',
      depthBb: 20,
      button: 'button-small-blind',
      hero: 'button-small-blind',
      random: seededRandom(41),
    });
    const response = await POST(
      new Request('http://localhost/api/practice/resolve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          modelVersion: 'resolver-v1',
          depthBb: 20,
          stateHash,
          state,
        }),
      })
    );
    expect(response.status).toBe(200);
    expect(response.headers.get('cache-control')).toBe('private, no-store');
    const payload = mocks.query.mock.calls[0][0];
    expect(payload.privateCards).toEqual(
      state.holeCards['button-small-blind']
    );
    expect(payload).not.toHaveProperty('holeCards');
    expect(JSON.stringify(payload)).not.toContain(
      JSON.stringify(state.holeCards['big-blind'])
    );
  });

  it('fails closed when any loaded component identity drifts', async () => {
    const state = createHand({
      modelVersion: 'resolver-v1',
      depthBb: 20,
      button: 'button-small-blind',
      hero: 'button-small-blind',
      random: seededRandom(43),
    });
    mocks.query.mockResolvedValue(resolverResponse('f'.repeat(64)));
    const response = await POST(
      new Request('http://localhost/api/practice/resolve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          modelVersion: 'resolver-v1',
          depthBb: 20,
          stateHash,
          state,
        }),
      })
    );
    expect(response.status).toBe(503);
    await expect(response.json()).resolves.toMatchObject({
      error: 'The resolver response does not match its pinned manifest',
    });
  });
});
