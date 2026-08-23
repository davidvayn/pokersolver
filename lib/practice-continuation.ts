import { canonicalPolicyState } from '@/lib/practice-engine';
import type { HandState } from '@/lib/practice-types';

export interface PreparedPracticeContinuation<Result, Progress> {
  promise: Promise<Result>;
  readonly latestProgress: Progress | undefined;
  subscribe(listener: (progress: Progress) => void): () => void;
}

/**
 * Keeps one exact continuation in flight for the current hand. The canonical
 * state includes the public action line, so alternate decisions cannot reuse a
 * solve prepared for a different branch.
 */
export class PracticeContinuationCache<Result, Progress = never> {
  private readonly entries = new Map<
    string,
    PreparedPracticeContinuation<Result, Progress>
  >();

  prepare(
    state: HandState,
    load: (report: (progress: Progress) => void) => Promise<Result>
  ): PreparedPracticeContinuation<Result, Progress> {
    const key = canonicalPolicyState(state);
    const existing = this.entries.get(key);
    if (existing) return existing;

    let latestProgress: Progress | undefined;
    const listeners = new Set<(progress: Progress) => void>();
    const report = (progress: Progress) => {
      latestProgress = progress;
      for (const listener of listeners) listener(progress);
    };
    const promise = Promise.resolve().then(() => load(report));
    const prepared: PreparedPracticeContinuation<Result, Progress> = {
      promise,
      get latestProgress() {
        return latestProgress;
      },
      subscribe(listener) {
        listeners.add(listener);
        if (latestProgress !== undefined) listener(latestProgress);
        return () => listeners.delete(listener);
      },
    };
    this.entries.set(key, prepared);
    void promise.catch(() => {
      if (this.entries.get(key) === prepared) this.entries.delete(key);
    });
    return prepared;
  }

  get(
    state: HandState
  ): PreparedPracticeContinuation<Result, Progress> | undefined {
    return this.entries.get(canonicalPolicyState(state));
  }

  clear(): void {
    this.entries.clear();
  }
}
