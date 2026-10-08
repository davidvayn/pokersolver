import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  modelForFullHandIdentity: vi.fn(),
  query: vi.fn(),
  process: vi.fn(),
}));

vi.mock('server-only', () => ({}));
vi.mock('@/lib/practice-models', () => ({
  modelForFullHandIdentity: mocks.modelForFullHandIdentity,
}));
vi.mock('@/lib/server/practice-solver-process', () => ({
  practiceResolverIdentity: (manifest: { version: string; runtime: object }, depthBb: number) => ({
    ...manifest.runtime, modelVersion: manifest.version, depthBb,
  }),
  practiceSolverProcess: mocks.process,
}));

import { maxDuration, POST, runtime } from '@/app/api/practice/resolve/route';
import { createHand, seededRandom } from '@/lib/practice-engine';
import { NL25_STUDY_RULES } from '@/lib/cash-game-rules';

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
    mocks.modelForFullHandIdentity.mockReset();
    mocks.query.mockReset();
    mocks.process.mockReset().mockReturnValue({ query: mocks.query });
    mocks.modelForFullHandIdentity.mockReturnValue({
      version: 'resolver-v1',
      runtime: {
        kind: 'rust-continual-resolver-v1',
        networkSha256: 'a'.repeat(64), rangePolicySha256: 'b'.repeat(64),
        valueNetworkSha256: 'c'.repeat(64), preflopActionValuesSha256: 'd'.repeat(64),
      },
    });
    mocks.query.mockResolvedValue(resolverResponse());
  });

  it('uses the full Node runtime with the Hobby Fluid Compute duration ceiling', () => {
    expect(runtime).toBe('nodejs');
    expect(maxDuration).toBe(60);
  });

  it('rejects a cash ledger before invoking the Home resolver even with a Home version', async () => {
    const state = createHand({ modelVersion: 'resolver-v1', depthBb: 20,
      button: 'button-small-blind', hero: 'button-small-blind', cashRules: NL25_STUDY_RULES });
    const response = await POST(new Request('http://localhost/api/practice/resolve', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ modelVersion: 'resolver-v1', depthBb: 20, stateHash, state }),
    }));
    expect(response.status).toBe(400);
    expect(mocks.query).not.toHaveBeenCalled();
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
    expect(mocks.modelForFullHandIdentity).toHaveBeenCalledWith('resolver-v1',20);
    expect(mocks.process).toHaveBeenCalledWith(mocks.modelForFullHandIdentity.mock.results[0].value,20);
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

  it('does not accept impossible or non-finite probability error reports', async () => {
    const state = createHand({ modelVersion: 'resolver-v1', depthBb: 20,
      button: 'button-small-blind', hero: 'button-small-blind' });
    for (const error of [-1,NaN,Infinity]) {
      mocks.query.mockResolvedValue({ ...resolverResponse(),maximumProbabilitySumError:error });
      const response = await POST(new Request('http://localhost/api/practice/resolve', {
        method:'POST',body:JSON.stringify({modelVersion:'resolver-v1',depthBb:20,stateHash,state}),
      }));
      expect(response.status).toBe(503);
    }
  });

  it('does not switch a hand to another version or depth when its pinned model is unavailable', async () => {
    const state = createHand({modelVersion:'resolver-v1',depthBb:20,
      button:'button-small-blind',hero:'button-small-blind'});
    mocks.modelForFullHandIdentity.mockReturnValue(null);
    const response = await POST(new Request('http://localhost/api/practice/resolve', {
      method:'POST',body:JSON.stringify({modelVersion:'resolver-v1',depthBb:20,stateHash,state}),
    }));
    expect(response.status).toBe(404);
    expect(mocks.process).not.toHaveBeenCalled();
  });
});
