import {
  decodePolicyShard,
  policyNodeFromShard,
} from '@/lib/policy-codec';
import { canonicalPolicyHash } from '@/lib/practice-engine';
import type { NeuralPolicyResult } from '@/lib/neural-policy';
import { NeuralPolicyWorkerClient } from '@/lib/neural-policy-worker-client';
import type { NeuralPolicyExecutor } from '@/lib/neural-policy-worker-protocol';
import type {
  HandState,
  OpponentModelSnapshot,
  PolicyManifest,
  PolicyNode,
} from '@/lib/practice-types';

export class PolicyUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'PolicyUnavailableError';
  }
}

export interface PinnedPracticeModel {
  manifest: PolicyManifest;
  depthBb: number;
  neuralReady?: true;
}

type Fetcher = typeof fetch;

export class PracticePolicyClient {
  private manifests: PolicyManifest[] | null = null;
  private shardCache = new Map<string, Promise<PolicyNode[]>>();
  private fetcher: Fetcher;
  private neuralExecutor: NeuralPolicyExecutor | null;

  constructor(fetcher?: Fetcher, neuralExecutor?: NeuralPolicyExecutor) {
    // Browser-native fetch requires its global receiver. Tests normally pass a
    // plain mock, which is why keeping the unbound function here can otherwise
    // escape unit coverage and fail only in a real browser.
    this.fetcher = fetcher ?? globalThis.fetch.bind(globalThis);
    this.neuralExecutor = neuralExecutor ?? null;
  }

  private neural(): NeuralPolicyExecutor {
    if (!this.neuralExecutor) this.neuralExecutor = new NeuralPolicyWorkerClient();
    return this.neuralExecutor;
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
    if (manifest.runtime?.kind === 'neural-deep-cfr-v1') {
      try {
        await this.neural().load({
          runtime: manifest.runtime,
          modelVersion: manifest.version,
          depthBb,
        });
        return { manifest, depthBb, neuralReady: true };
      } catch (error) {
        throw new PolicyUnavailableError(
          error instanceof Error
            ? error.message
            : 'The pinned neural artifact is unavailable'
        );
      }
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
    if (pinned.manifest.runtime?.kind === 'neural-deep-cfr-v1') {
      throw new PolicyUnavailableError(
        'A neural policy lookup requires the exact hand state'
      );
    }
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

  async lookupState(input: {
    pinned: PinnedPracticeModel;
    state: HandState;
    profile: OpponentModelSnapshot;
    usage: 'grading' | 'opponent';
  }): Promise<NeuralPolicyResult> {
    const runtime = input.pinned.manifest.runtime;
    if (runtime?.kind === 'neural-deep-cfr-v1') {
      if (!input.pinned.neuralReady) {
        throw new PolicyUnavailableError('The pinned neural weights are unavailable');
      }
      try {
        return await this.neural().infer({
          runtime,
          modelVersion: input.pinned.manifest.version,
          depthBb: input.pinned.depthBb,
          state: input.state,
          profile: input.profile,
          usage: input.usage,
        });
      } catch (error) {
        throw new PolicyUnavailableError(
          error instanceof Error ? error.message : 'Neural inference failed'
        );
      }
    }
    const stateHash = await canonicalPolicyHash(input.state);
    return {
      node: await this.lookup(input.pinned, stateHash),
      trace: null,
    };
  }

  retryShard(url: string): void {
    this.shardCache.delete(url);
  }
}
