import { describe, expect, it, vi } from 'vitest';
import { cpus } from 'node:os';
import fullHandManifests from '@/data/practice/full-hand-manifests.json';
import type { PolicyManifest } from '@/lib/practice-types';

vi.mock('server-only', () => ({}));

import {
  PRACTICE_RESOLVER_IDENTITY,
  PracticeSolverPool,
  practiceResolverCommand,
  practiceResolverPoolSize,
  type PracticeResolverWorker,
} from '@/lib/server/practice-solver-process';

function option(args: string[], flag: string): string | null {
  const index = args.indexOf(flag);
  return index < 0 ? null : (args[index + 1] ?? null);
}

describe('pinned practice resolver process', () => {
  it('passes the complete manifest solver profile to Rust', () => {
    const manifest = (fullHandManifests as PolicyManifest[]).find(
      (candidate) =>
        candidate.version === PRACTICE_RESOLVER_IDENTITY.modelVersion
    );
    expect(manifest?.runtime?.kind).toBe('rust-continual-resolver-v1');
    if (manifest?.runtime?.kind !== 'rust-continual-resolver-v1') {
      throw new Error('the pinned resolver manifest is missing');
    }

    const { args } = practiceResolverCommand();
    const resolver = manifest.runtime.resolver;
    expect(option(args, '--dcfr-alpha')).toBe(
      String(manifest.runtime.dcfr.positiveRegretExponent)
    );
    expect(option(args, '--dcfr-beta')).toBe(
      String(manifest.runtime.dcfr.negativeRegretExponent)
    );
    expect(option(args, '--dcfr-gamma')).toBe(
      String(manifest.runtime.dcfr.strategyExponent)
    );
    expect(args.includes('--flop-resolver-deploy-solved-policy')).toBe(
      resolver.flopDeploySolvedPolicy
    );
    const streets = [
      ['flop', resolver.flopIterations, resolver.flopResolvedActor],
      ['turn', resolver.turnIterations, resolver.turnResolvedActor],
      ['river', resolver.riverIterations, resolver.riverResolvedActor],
    ] as const;

    for (const [street, iterations, resolvedActor] of streets) {
      expect(option(args, `--${street}-resolver-iterations`)).toBe(
        String(iterations)
      );
      expect(option(args, `--${street}-resolver-actor`)).toBe(
        resolvedActor === null ? null : String(resolvedActor)
      );
    }
  });

  it('shares one loaded model across micro-batched speculative branches', () => {
    expect(practiceResolverPoolSize()).toBe(1);
    const threads = Number(
      option(practiceResolverCommand().args, '--flop-resolver-threads')
    );
    expect(threads).toBeGreaterThanOrEqual(1);
    expect(threads).toBeLessThanOrEqual(Math.min(8, cpus().length));
  });

  it('can still dispatch simultaneous branch solves across an explicit pool', async () => {
    const releases: Array<() => void> = [];
    const calls: number[] = [0, 0];
    const workers = calls.map((_, index): PracticeResolverWorker => ({
      query: async () => {
        calls[index] += 1;
        await new Promise<void>((resolve) => releases.push(resolve));
        return index;
      },
      stop: async () => undefined,
    }));
    const pool = new PracticeSolverPool(workers);

    const first = pool.query({ branch: 'call' });
    const second = pool.query({ branch: 'raise' });
    await Promise.resolve();

    expect(calls).toEqual([1, 1]);
    releases.splice(0).forEach((release) => release());
    await expect(Promise.all([first, second])).resolves.toEqual([0, 1]);
  });

  it('keeps descendant queries on the worker that owns their cached subtree', async () => {
    const calls: number[] = [0, 0];
    const releases: Array<() => void> = [];
    const workers = calls.map((_, index): PracticeResolverWorker => ({
      query: async () => {
        calls[index] += 1;
        if (calls.reduce((sum, count) => sum + count, 0) <= 2) {
          await new Promise<void>((resolve) => releases.push(resolve));
        }
        return index;
      },
      stop: async () => undefined,
    }));
    const pool = new PracticeSolverPool(workers);

    const callRoot = pool.query({ street: 'flop' }, 'hand-1|call');
    const raiseRoot = pool.query({ street: 'flop' }, 'hand-1|raise');
    await Promise.resolve();
    releases.splice(0).forEach((release) => release());
    await expect(Promise.all([callRoot, raiseRoot])).resolves.toEqual([0, 1]);
    await expect(pool.query({ street: 'turn' }, 'hand-1|call')).resolves.toBe(
      0
    );

    expect(calls).toEqual([2, 1]);
  });
});
