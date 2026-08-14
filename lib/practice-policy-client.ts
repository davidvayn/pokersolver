import {
  decodePolicyShard,
  policyNodeFromShard,
} from '@/lib/policy-codec';
import { canonicalPolicyHash } from '@/lib/practice-engine';
import type { NeuralPolicyResult } from '@/lib/neural-policy';
import { neuralLegalActions } from '@/lib/neural-policy';
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
    if (manifest.runtime?.kind === 'rust-continual-resolver-v1') {
      return { manifest, depthBb };
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
    if (runtime?.kind === 'rust-continual-resolver-v1') {
      const stateHash = await canonicalPolicyHash(input.state);
      const response = await this.fetcher(runtime.endpoint, {
        method: 'POST',
        cache: 'no-store',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          modelVersion: input.pinned.manifest.version,
          depthBb: input.pinned.depthBb,
          stateHash,
          state: input.state,
        }),
      });
      if (!response.ok) {
        const body = (await response.json().catch(() => null)) as {
          error?: unknown;
        } | null;
        throw new PolicyUnavailableError(
          typeof body?.error === 'string'
            ? body.error
            : 'The pinned continual resolver is unavailable'
        );
      }
      const body = (await response.json()) as {
        stateHash?: unknown;
        modelVersion?: unknown;
        networkSha256?: unknown;
        rangePolicySha256?: unknown;
        valueNetworkSha256?: unknown;
        preflopActionValuesSha256?: unknown;
        actions?: unknown;
      };
      if (
        body.stateHash !== stateHash ||
        body.modelVersion !== input.pinned.manifest.version ||
        body.networkSha256 !== runtime.networkSha256 ||
        body.rangePolicySha256 !== runtime.rangePolicySha256 ||
        body.valueNetworkSha256 !== runtime.valueNetworkSha256 ||
        body.preflopActionValuesSha256 !==
          runtime.preflopActionValuesSha256 ||
        !Array.isArray(body.actions)
      ) {
        throw new PolicyUnavailableError(
          'The continual resolver response does not match the pinned model'
        );
      }
      const legal = neuralLegalActions(input.state, runtime.actionAbstraction);
      const raw = body.actions as Array<{
        kind?: unknown;
        amountToBb?: unknown;
        probability?: unknown;
        evBb?: unknown;
        standardErrorBb?: unknown;
        confidence?: unknown;
      }>;
      if (raw.length !== legal.length) {
        throw new PolicyUnavailableError(
          'The continual resolver returned a different legal action set'
        );
      }
      const actions = legal.map((action) => {
        const match = raw.filter((candidate) => {
          const kind =
            candidate.kind === 'all_in' ? 'all-in' : candidate.kind;
          const amountsMatch =
            action.amountToBb === undefined
              ? candidate.amountToBb === undefined || candidate.amountToBb === null
              : typeof candidate.amountToBb === 'number' &&
                Math.abs(candidate.amountToBb - action.amountToBb) <= 0.001;
          return kind === action.kind && amountsMatch;
        });
        if (match.length !== 1) {
          throw new PolicyUnavailableError(
            `The continual resolver could not match ${action.id}`
          );
        }
        const candidate = match[0];
        if (
          typeof candidate.probability !== 'number' ||
          !Number.isFinite(candidate.probability) ||
          candidate.probability < 0 ||
          !['high', 'low', 'unavailable'].includes(String(candidate.confidence)) ||
          !(
            (candidate.evBb === null && candidate.standardErrorBb === null) ||
            (typeof candidate.evBb === 'number' &&
              Number.isFinite(candidate.evBb) &&
              candidate.standardErrorBb === null) ||
            (typeof candidate.evBb === 'number' &&
              Number.isFinite(candidate.evBb) &&
              typeof candidate.standardErrorBb === 'number' &&
              Number.isFinite(candidate.standardErrorBb) &&
              candidate.standardErrorBb >= 0)
          )
        ) {
          throw new PolicyUnavailableError(
            `The continual resolver returned invalid data for ${action.id}`
          );
        }
        return {
          ...action,
          probability: candidate.probability,
          evBb: candidate.evBb as number | null,
          standardErrorBb: candidate.standardErrorBb as number | null,
          confidence: candidate.confidence as 'high' | 'low' | 'unavailable',
        };
      });
      const finite = actions.filter(
        (action): action is typeof action & { evBb: number } =>
          action.evBb !== null
      );
      const best = finite.reduce<(typeof finite)[number] | null>(
        (current, action) =>
          !current || action.evBb > current.evBb ? action : current,
        null
      );
      return {
        node: {
          stateHash,
          actions,
          bestActionId: best?.id ?? null,
          bestActionEvBb: best?.evBb ?? null,
        },
        trace:
          input.usage === 'opponent'
            ? {
                stateHash,
                modelVersion: input.pinned.manifest.version,
                profileVersion: input.profile.version,
                evidenceCount: input.profile.observations,
                confidence: input.profile.confidence,
                responseWeight: 0,
                componentModelVersion: input.pinned.manifest.version,
                baselineActions: actions.map(({ id, probability }) => ({
                  id,
                  probability,
                })),
                responseActions: actions.map(({ id, probability }) => ({
                  id,
                  probability,
                })),
                servedActions: actions.map(({ id, probability }) => ({
                  id,
                  probability,
                })),
              }
            : null,
      };
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
