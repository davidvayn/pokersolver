import {
  decodePolicyShard,
  policyNodeFromShard,
} from '@/lib/policy-codec';
import type { PolicyManifest, PolicyNode } from '@/lib/practice-types';

export class PolicyUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'PolicyUnavailableError';
  }
}

export interface PinnedPracticeModel {
  manifest: PolicyManifest;
  depthBb: number;
}

type Fetcher = typeof fetch;

export class PracticePolicyClient {
  private manifests: PolicyManifest[] | null = null;
  private shardCache = new Map<string, Promise<PolicyNode[]>>();
  private fetcher: Fetcher;

  constructor(fetcher?: Fetcher) {
    // Browser-native fetch requires its global receiver. Tests normally pass a
    // plain mock, which is why keeping the unbound function here can otherwise
    // escape unit coverage and fail only in a real browser.
    this.fetcher = fetcher ?? globalThis.fetch.bind(globalThis);
  }

  async loadManifests(force = false): Promise<PolicyManifest[]> {
    if (this.manifests && !force) return this.manifests;
    const response = await this.fetcher('/api/practice/models', {
      cache: force ? 'no-store' : 'default',
    });
    if (!response.ok) throw new PolicyUnavailableError('Model manifest service is unavailable');
    const body: unknown = await response.json();
    if (!body || typeof body !== 'object' || !Array.isArray((body as { manifests?: unknown }).manifests)) {
      throw new PolicyUnavailableError('Model manifest response is invalid');
    }
    this.manifests = (body as { manifests: PolicyManifest[] }).manifests.filter(
      (manifest) =>
        manifest.active && manifest.validation?.status === 'accepted'
    );
    return this.manifests;
  }

  async pinFullHandModel(depthBb: number): Promise<PinnedPracticeModel> {
    const manifests = await this.loadManifests();
    const manifest = manifests.find(
      (candidate) =>
        candidate.subtype === 'full-hand' && candidate.depthsBb.includes(depthBb)
    );
    if (!manifest) {
      throw new PolicyUnavailableError(`No accepted full-hand model at ${depthBb}bb`);
    }
    return { manifest, depthBb };
  }

  private async shard(url: string): Promise<PolicyNode[]> {
    let pending = this.shardCache.get(url);
    if (!pending) {
      pending = this.fetcher(url, { cache: 'force-cache' }).then(async (response) => {
        if (!response.ok) throw new PolicyUnavailableError('Policy shard is missing');
        return decodePolicyShard(new Uint8Array(await response.arrayBuffer()));
      });
      this.shardCache.set(url, pending);
    }
    try {
      return await pending;
    } catch (error) {
      this.shardCache.delete(url);
      throw error;
    }
  }

  async lookup(
    pinned: PinnedPracticeModel,
    stateHash: string
  ): Promise<PolicyNode> {
    if (!/^[a-f0-9]{64}$/.test(stateHash)) {
      throw new PolicyUnavailableError('Canonical state hash is invalid');
    }
    for (const length of [4, 6]) {
      const prefix = stateHash.slice(0, length);
      const url = `/api/practice/policy/${encodeURIComponent(pinned.manifest.version)}/${pinned.depthBb}/${prefix}`;
      try {
        const node = policyNodeFromShard(await this.shard(url), stateHash);
        if (node) return node;
      } catch (error) {
        if (length === 6) throw error;
      }
    }
    throw new PolicyUnavailableError(
      'The pinned model has no policy node for this authentic state. The table is paused and the decision will not be scored.'
    );
  }

  retryShard(url: string): void {
    this.shardCache.delete(url);
  }
}
