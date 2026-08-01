import { describe, expect, it, vi } from 'vitest';
import { encodePolicyShard } from '@/lib/policy-codec';
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
});
