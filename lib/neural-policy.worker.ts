/// <reference lib="webworker" />

import {
  inferNeuralPolicy,
  NeuralArtifactClient,
} from '@/lib/neural-policy';
import type {
  NeuralWorkerRequest,
  NeuralWorkerResponse,
} from '@/lib/neural-policy-worker-protocol';

const workerScope = self as unknown as DedicatedWorkerGlobalScope;
const artifacts = new NeuralArtifactClient(fetch.bind(globalThis));

workerScope.onmessage = async (event: MessageEvent<NeuralWorkerRequest>) => {
  const request = event.data;
  try {
    const artifact = await artifacts.load({
      runtime: request.runtime,
      modelVersion: request.modelVersion,
      depthBb: request.depthBb,
    });
    const result =
      request.type === 'infer'
        ? await inferNeuralPolicy({
            artifact,
            state: request.state,
            profile: request.profile,
            usage: request.usage,
          })
        : null;
    workerScope.postMessage({ id: request.id, ok: true, result } satisfies NeuralWorkerResponse);
  } catch (error) {
    workerScope.postMessage({
      id: request.id,
      ok: false,
      error: error instanceof Error ? error.message : 'Neural policy worker failed',
    } satisfies NeuralWorkerResponse);
  }
};
