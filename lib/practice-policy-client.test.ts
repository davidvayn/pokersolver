import { describe, expect, it, vi } from 'vitest';
import { encodePolicyShard } from '@/lib/policy-codec';
import { createHand, seededRandom } from '@/lib/practice-engine';
import { buildOpponentModel } from '@/lib/opponent-model';
import {
  PolicyUnavailableError,
  PracticePolicyClient,
} from '@/lib/practice-policy-client';
import type { PolicyManifest, PolicyNode } from '@/lib/practice-types';

const manifest: PolicyManifest = {
  schemaVersion: 1,
  version: 'full-v1',
  model: 'test',
  label: 'Approximate GTO',
  subtype: 'full-hand',
  active: true,
  depthsBb: [20],
  generatedAt: '2026-01-01T00:00:00.000Z',
  stateSchema: 'test',
  shardSchema: 'binary-v1',
  runtime: { kind: 'binary-policy-shards-v1' },
  abstraction: {
    blindsBb: [0.5, 1],
    anteBb: 0,
    rake: 'none',
    actionSizing: 'test',
    cardAbstraction: 'test',
    recall: 'trajectory',
  },
  validation: { status: 'accepted', notes: [] },
};

const node: PolicyNode = {
  stateHash: 'ab12'.repeat(16),
  bestActionId: 'check',
  bestActionEvBb: 0,
  actions: [
    {
      id: 'check',
      kind: 'check',
      label: 'Check',
      probability: 1,
      evBb: 0,
      standardErrorBb: 0.01,
      confidence: 'high',
    },
  ],
};

describe('practice policy client', () => {
  it('pins one accepted version and caches immutable binary shards', async () => {
    const fetcher = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/models')) {
        return Response.json({ schemaVersion: 1, manifests: [manifest] });
      }
      return new Response(encodePolicyShard([node]), { status: 200 });
    }) as typeof fetch;
    const client = new PracticePolicyClient(fetcher);
    const pinned = await client.pinFullHandModel(20);
    expect(pinned.manifest.version).toBe('full-v1');
    expect(await client.lookup(pinned, node.stateHash)).toMatchObject({
      stateHash: node.stateHash,
    });
    expect(await client.lookup(pinned, node.stateHash)).toMatchObject({
      stateHash: node.stateHash,
    });
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it('pauses on unavailable depths and missing nodes instead of falling back', async () => {
    const fetcher = vi.fn(async (input: RequestInfo | URL) => {
      if (String(input).endsWith('/models')) {
        return Response.json({ schemaVersion: 1, manifests: [manifest] });
      }
      return new Response(null, { status: 404 });
    }) as typeof fetch;
    const client = new PracticePolicyClient(fetcher);
    await expect(client.pinFullHandModel(50)).rejects.toBeInstanceOf(
      PolicyUnavailableError
    );
    const pinned = await client.pinFullHandModel(20);
    await expect(client.lookup(pinned, 'cd34'.repeat(16))).rejects.toThrow(
      'Policy shard is missing'
    );
  });

  it('loads and queries neural policies through the injected worker boundary', async () => {
    const neuralManifest: PolicyManifest = {
      ...manifest,
      runtime: {
        kind: 'neural-deep-cfr-v1',
        artifactUrl: '/models/practice/full-v1/20bb.bin',
        artifactSha256: 'a'.repeat(64),
        stateFeatureSchema: 'hu-cash-trajectory-poker-aware-v4',
        actionFeatureSchema: 'hu-cash-legal-action-v1',
        opponentProfileSchema: 'local-opponent-profile-v1',
        actionAbstraction: {
          openSizesBb: [2, 2.5, 3],
          limpRaiseSizesBb: [3, 4, 5],
          threeBetSizesBb: [7.5, 9, 11],
          fourBetSizesBb: [18, 22, 26],
          deeperRaisePotFractions: [0.75, 1, 1.25],
          preflopRaiseCap: 4,
          flopBetPotFractions: [1 / 3, 0.75, 1.25],
          turnRiverBetPotFractions: [0.5, 1],
          postflopRaisePotFractions: [1],
          postflopRaiseCap: 1,
          includeAllIn: true,
        },
        adaptation: {
          minimumObservations: 50,
          fullConfidenceObservations: 250,
          maximumResponseWeight: 0.5,
        },
      },
    };
    const fetcher = vi.fn(async () =>
      Response.json({ schemaVersion: 1, manifests: [neuralManifest] })
    ) as typeof fetch;
    const executor = {
      load: vi.fn(async () => undefined),
      infer: vi.fn(async () => ({ node, trace: null })),
    };
    const client = new PracticePolicyClient(fetcher, executor);
    const pinned = await client.pinFullHandModel(20);
    const state = createHand({
      modelVersion: neuralManifest.version,
      depthBb: 20,
      button: 'button-small-blind',
      hero: 'button-small-blind',
      random: seededRandom(1),
    });
    await expect(
      client.lookupState({
        pinned,
        state,
        profile: buildOpponentModel([], 'baseline'),
        usage: 'grading',
      })
    ).resolves.toMatchObject({ node: { stateHash: node.stateHash } });
    expect(executor.load).toHaveBeenCalledOnce();
    expect(executor.infer).toHaveBeenCalledOnce();
  });
});
