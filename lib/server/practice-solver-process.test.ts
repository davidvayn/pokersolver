import { afterEach, describe, expect, it, vi } from 'vitest';
import { cpus } from 'node:os';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import fullHandManifests from '@/data/practice/full-hand-manifests.json';
import type { PolicyManifest } from '@/lib/practice-types';

vi.mock('server-only', () => ({}));
const { spawnMock, verifyMock } = vi.hoisted(() => ({ spawnMock: vi.fn(), verifyMock: vi.fn(async () => undefined) }));
vi.mock('node:child_process', () => ({ spawn: spawnMock }));
vi.mock('@/lib/server/practice-resolver-artifacts', () => ({verifyPracticeResolverArtifacts:verifyMock}));

import {
  PRACTICE_RESOLVER_IDENTITY,
  PracticeSolverPool,
  practiceResolverCommand,
  practiceResolverPoolSize,
  practiceResolverIdentity,
  practiceSolverProcess,
  stopPracticeSolverProcess,
  type PracticeResolverWorker,
} from '@/lib/server/practice-solver-process';
import { NL25_STUDY_RULES } from '@/lib/cash-game-rules';

function option(args: string[], flag: string): string | null {
  const index = args.indexOf(flag);
  return index < 0 ? null : (args[index + 1] ?? null);
}
afterEach(async () => { await stopPracticeSolverProcess(); vi.unstubAllEnvs(); verifyMock.mockReset().mockResolvedValue(undefined); });

function resolverFixture(): PolicyManifest {
  return structuredClone((fullHandManifests as PolicyManifest[]).find(
    m=>m.version===PRACTICE_RESOLVER_IDENTITY.modelVersion)!);
}

describe('pinned practice resolver process', () => {
  it('builds commands and identities from each qualified manifest/depth without a global version',()=>{
    const candidate = resolverFixture(); candidate.version = 'home-test-depth-40'; candidate.depthsBb = [40];
    const {args} = practiceResolverCommand(candidate,40);
    expect(option(args,'--model-version')).toBe(candidate.version);
    expect(option(args,'--effective-stack-bb')).toBe('40');
    expect(practiceResolverIdentity(candidate,40).depthBb).toBe(40);
    expect(()=>practiceResolverCommand(candidate,20)).toThrow('qualified');
    candidate.active = false;
    expect(()=>practiceSolverProcess(candidate,40)).toThrow('qualified');
  });

  it('does not permit a cash profile to reuse the legacy native runtime',()=>{
    const candidate = resolverFixture();
    candidate.cashGame = {rules:NL25_STUDY_RULES,rulesSha256:'a'.repeat(64)};
    expect(()=>practiceSolverProcess(candidate,20)).toThrow('qualified');
    expect(spawnMock).not.toHaveBeenCalled();
  });

  it('isolates model/config identities and bounds all model pools together',()=>{
    vi.stubEnv('PRACTICE_RESOLVER_THREADS','1');
    const first = resolverFixture();
    const original = practiceSolverProcess(first,20);
    expect(practiceSolverProcess(structuredClone(first),20)).toBe(original);
    const second = structuredClone(first); second.version = 'home-other-model';
    if (cpus().length < 2) {
      expect(()=>practiceSolverProcess(second,20)).toThrow('CPU loading budget');
      return;
    }
    expect(practiceSolverProcess(second,20)).not.toBe(original);
    const third = structuredClone(first); third.version = 'home-third-model';
    expect(()=>practiceSolverProcess(third,20)).toThrow('loading budget');
  });

  it('checks artifacts before spawning and allows retry after verification failure',async()=>{
    const pool = practiceSolverProcess();
    const initialSpawns = spawnMock.mock.calls.length;
    verifyMock.mockRejectedValue(new Error('canonical hash mismatch'));
    await expect(pool.query({fixture:'bad'})).rejects.toThrow('hash mismatch');
    await expect(pool.query({fixture:'retry'})).rejects.toThrow('hash mismatch');
    expect(verifyMock).toHaveBeenCalledTimes(2);
    expect(spawnMock).toHaveBeenCalledTimes(initialSpawns);
  });

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
    const actionGrid = JSON.parse(option(args,'--action-abstraction-json')!);
    expect(actionGrid.open_sizes_bb).toEqual(manifest.runtime.actionAbstraction.openSizesBb);
    expect(actionGrid.turn_river_bet_pot_fractions).toEqual(manifest.runtime.actionAbstraction.turnRiverBetPotFractions);
    expect(actionGrid.include_all_in).toBe(manifest.runtime.actionAbstraction.includeAllIn);
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
    expect(args.includes('--river-resolver-safe')).toBe(
      resolver.riverSafeResolving
    );
    expect(args.includes('--river-resolver-safe-maxmargin')).toBe(
      resolver.riverSafeMaxmargin
    );
    expect(option(args, '--river-resolver-safe-iterations')).toBe(
      resolver.riverSafeIterations === null
        ? null
        : String(resolver.riverSafeIterations)
    );
    expect(option(args, '--river-resolver-safe-actor')).toBe(
      resolver.riverSafeResolvedActor === null
        ? null
        : String(resolver.riverSafeResolvedActor)
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

  it('honors explicit higher whole-core budgets without oversubscribing process pools', () => {
    const cores = Math.max(1, cpus().length);
    try {
      vi.stubEnv('PRACTICE_RESOLVER_POOL_SIZE', '1');
      vi.stubEnv('PRACTICE_RESOLVER_THREADS', '10');
      expect(option(practiceResolverCommand().args, '--flop-resolver-threads')).toBe(
        String(Math.min(10, 16, cores))
      );
      vi.stubEnv('PRACTICE_RESOLVER_POOL_SIZE', '2');
      const budget = Math.max(1, Math.floor(cores / 2));
      for (const count of ['16', '128']) {
        vi.stubEnv('PRACTICE_RESOLVER_THREADS', count);
        const args = practiceResolverCommand().args;
        expect(option(args, '--flop-resolver-threads')).toBe(String(Math.min(16, budget)));
        expect(option(args, '--turn-resolver-threads')).toBe(String(Math.min(16, budget)));
      }
      for (const invalid of ['0', '-2', '3.5', 'bad']) {
        vi.stubEnv('PRACTICE_RESOLVER_THREADS', invalid);
        expect(option(practiceResolverCommand().args, '--flop-resolver-threads')).toBe(
          String(Math.min(8, budget))
        );
      }
    } finally {
      vi.unstubAllEnvs();
    }
  });

  it('opts into streaming and resolves a ready sibling without waiting for the batch', async () => {
    const child = Object.assign(new EventEmitter(), {
      stdin: new PassThrough(),
      stdout: new PassThrough(),
      stderr: new PassThrough(),
      killed: false,
      exitCode: null as number | null,
    });
    let writtenBatch: {
      streamResults: boolean;
      queries: Array<{ requestId: string; fixture: string }>;
    } | undefined;
    child.stdin.on('data', (bytes: Buffer) => {
      writtenBatch = JSON.parse(bytes.toString());
      const ready = writtenBatch!.queries.find((query) => query.fixture === 'fast')!;
      child.stdout.write(`${JSON.stringify({ requestId: ready.requestId, fixture: 'fast' })}\n`);
    });
    child.stdin.on('finish', () => {
      child.exitCode = 0;
      child.emit('exit', 0, null);
    });
    spawnMock.mockImplementationOnce(() => {
      queueMicrotask(() => child.emit('spawn'));
      return child;
    });

    const pool = practiceSolverProcess();
    let slowFinished = false;
    const slow = pool.query({ fixture: 'slow' }).then((value) => {
      slowFinished = true;
      return value;
    });
    const fast = pool.query({ fixture: 'fast' });
    try {
      await expect(fast).resolves.toMatchObject({ fixture: 'fast' });
      expect(slowFinished).toBe(false);
      expect(writtenBatch?.streamResults).toBe(true);
      expect(writtenBatch?.queries).toHaveLength(2);
      const pending = writtenBatch!.queries.find((query) => query.fixture === 'slow')!;
      child.stdout.write(`${JSON.stringify({ requestId: pending.requestId, fixture: 'slow' })}\n`);
      await expect(slow).resolves.toMatchObject({ fixture: 'slow' });
    } finally {
      await stopPracticeSolverProcess();
    }
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
