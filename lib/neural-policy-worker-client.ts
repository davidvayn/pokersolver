'use client';

import type { NeuralPolicyResult } from '@/lib/neural-policy';
import type {
  NeuralPolicyExecutor,
  NeuralWorkerRequest,
  NeuralWorkerResponse,
} from '@/lib/neural-policy-worker-protocol';

export class NeuralPolicyWorkerClient implements NeuralPolicyExecutor {
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<
    number,
    {
      resolve: (result: NeuralPolicyResult | null) => void;
      reject: (error: Error) => void;
    }
  >();

  constructor(
    workerFactory: () => Worker = () =>
      new Worker(new URL('./neural-policy.worker.ts', import.meta.url), {
        type: 'module',
      })
  ) {
    this.worker = workerFactory();
    this.worker.onmessage = (event: MessageEvent<NeuralWorkerResponse>) => {
      const pending = this.pending.get(event.data.id);
      if (!pending) return;
      this.pending.delete(event.data.id);
      if (event.data.ok) pending.resolve(event.data.result);
      else pending.reject(new Error(event.data.error));
    };
    this.worker.onerror = () => {
      const error = new Error('Neural policy worker failed');
      for (const pending of this.pending.values()) pending.reject(error);
      this.pending.clear();
    };
  }

  private request(
    request: Omit<NeuralWorkerRequest, 'id'>
  ): Promise<NeuralPolicyResult | null> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ ...request, id } as NeuralWorkerRequest);
    });
  }

  async load(
    input: Parameters<NeuralPolicyExecutor['load']>[0]
  ): Promise<void> {
    await this.request({ type: 'load', ...input });
  }

  async infer(
    input: Parameters<NeuralPolicyExecutor['infer']>[0]
  ): Promise<NeuralPolicyResult> {
    const result = await this.request({ type: 'infer', ...input });
    if (!result) throw new Error('Neural worker returned no inference result');
    return result;
  }

  terminate(): void {
    this.worker.terminate();
    const error = new Error('Neural policy worker was terminated');
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
  }
}
