import 'server-only';

import { randomUUID } from 'node:crypto';
import { cpus } from 'node:os';
import path from 'node:path';
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import fullHandManifests from '@/data/practice/full-hand-manifests.json';
import type {
  ContinualResolverRuntime,
  PolicyManifest,
} from '@/lib/practice-types';

const MODEL_VERSION = 'hu-20bb-v102-consensus-continual-resolver-experimental';
const REQUEST_TIMEOUT_MS = 5 * 60 * 1000;
const MAX_STDERR_BYTES = 16_384;
const MAX_RESOLVER_PROCESSES = 2;
const MICRO_BATCH_DELAY_MS = 2;
const MAX_BATCH_QUERIES = 2;

const resolverManifest = (fullHandManifests as PolicyManifest[]).find(
  (manifest) => manifest.version === MODEL_VERSION
);
if (resolverManifest?.runtime?.kind !== 'rust-continual-resolver-v1') {
  throw new Error('The pinned continual resolver manifest is missing');
}
const resolverRuntime: ContinualResolverRuntime = resolverManifest.runtime;
const resolverArtifactFiles = resolverRuntime.artifactFiles;
for (const [kind, file] of Object.entries(resolverArtifactFiles)) {
  if (!/^[a-z0-9][a-z0-9.-]*\.json\.gz$/.test(file)) {
    throw new Error(`The pinned continual resolver ${kind} file is unsafe`);
  }
}

export const PRACTICE_RESOLVER_IDENTITY = {
  modelVersion: MODEL_VERSION,
  networkSha256: resolverRuntime.networkSha256,
  rangePolicySha256: resolverRuntime.rangePolicySha256,
  valueNetworkSha256: resolverRuntime.valueNetworkSha256,
  preflopActionValuesSha256: resolverRuntime.preflopActionValuesSha256,
} as const;

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (reason: Error) => void;
  timeout: ReturnType<typeof setTimeout>;
}

function configuredPath(environmentName: string, fallback: string): string {
  return path.resolve(process.env[environmentName] ?? fallback);
}

function resolvedActorArgs(
  flag: string,
  resolvedActor: 0 | 1 | null
): string[] {
  return resolvedActor === null ? [] : [flag, String(resolvedActor)];
}

export function practiceResolverPoolSize(): number {
  const configured = Number(process.env.PRACTICE_RESOLVER_POOL_SIZE);
  return Math.max(
    1,
    Math.min(
      MAX_RESOLVER_PROCESSES,
      Number.isInteger(configured) && configured > 0 ? configured : 1
    )
  );
}

export function practiceResolverCommand(): {
  executable: string;
  args: string[];
} {
  const root = process.cwd();
  const modelRoot = configuredPath(
    'PRACTICE_RESOLVER_MODEL_DIR',
    path.join(root, 'preflop-solver', 'models', 'practice')
  );
  const executable = configuredPath(
    'PRACTICE_RESOLVER_BIN',
    path.join(root, 'preflop-solver', 'target', 'release', 'preflop-solver')
  );
  const threadsPerProcess = Math.max(
    1,
    Math.min(8, Number(process.env.PRACTICE_RESOLVER_THREADS) || cpus().length)
  );
  return {
    executable,
    args: [
      'practice-policy-server',
      '--model-version',
      MODEL_VERSION,
      '--effective-stack-bb',
      '20',
      '--networks',
      path.join(modelRoot, resolverArtifactFiles.networks),
      '--range-policy',
      path.join(modelRoot, resolverArtifactFiles.rangePolicy),
      '--preflop-action-values',
      path.join(modelRoot, resolverArtifactFiles.preflopActionValues),
      '--flop-resolver-iterations',
      String(resolverRuntime.resolver.flopIterations),
      '--flop-resolver-averaging-delay',
      '0',
      '--flop-resolver-value-network',
      path.join(modelRoot, resolverArtifactFiles.flopValueNetwork),
      '--flop-resolver-threads',
      String(threadsPerProcess),
      ...resolvedActorArgs(
        '--flop-resolver-actor',
        resolverRuntime.resolver.flopResolvedActor
      ),
      '--turn-resolver-iterations',
      String(resolverRuntime.resolver.turnIterations),
      '--turn-resolver-averaging-delay',
      '0',
      '--turn-resolver-threads',
      String(threadsPerProcess),
      ...resolvedActorArgs(
        '--turn-resolver-actor',
        resolverRuntime.resolver.turnResolvedActor
      ),
      '--river-resolver-iterations',
      String(resolverRuntime.resolver.riverIterations),
      '--river-resolver-averaging-delay',
      '0',
      ...resolvedActorArgs(
        '--river-resolver-actor',
        resolverRuntime.resolver.riverResolvedActor
      ),
    ],
  };
}

export interface PracticeResolverWorker {
  query(
    payload: Record<string, unknown>,
    affinityKey?: string
  ): Promise<unknown>;
  stop(): Promise<void>;
}

class PracticeSolverProcess implements PracticeResolverWorker {
  private child: ChildProcessWithoutNullStreams | null = null;
  private starting: Promise<ChildProcessWithoutNullStreams> | null = null;
  private stdout = '';
  private stderr = '';
  private pending = new Map<string, PendingRequest>();
  private queued: Array<{
    child: ChildProcessWithoutNullStreams;
    requestId: string;
    request: Record<string, unknown>;
  }> = [];
  private flushTimer: ReturnType<typeof setTimeout> | null = null;

  private async start(): Promise<ChildProcessWithoutNullStreams> {
    if (this.child && !this.child.killed && this.child.exitCode === null) {
      return this.child;
    }
    if (this.starting) return this.starting;
    this.starting = new Promise((resolve, reject) => {
      const { executable, args } = practiceResolverCommand();
      const child = spawn(executable, args, {
        cwd: process.cwd(),
        env: process.env,
        stdio: ['pipe', 'pipe', 'pipe'],
      });
      const failed = (error: Error) => {
        this.starting = null;
        reject(
          new Error(
            `The pinned practice resolver could not start: ${error.message}`
          )
        );
      };
      child.once('error', failed);
      child.once('spawn', () => {
        child.off('error', failed);
        this.child = child;
        this.starting = null;
        this.attach(child);
        resolve(child);
      });
    });
    return this.starting;
  }

  private attach(child: ChildProcessWithoutNullStreams): void {
    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', (chunk: string) => {
      this.stdout += chunk;
      for (;;) {
        const newline = this.stdout.indexOf('\n');
        if (newline < 0) break;
        const line = this.stdout.slice(0, newline);
        this.stdout = this.stdout.slice(newline + 1);
        if (!line.trim()) continue;
        let response: unknown;
        try {
          response = JSON.parse(line);
        } catch {
          this.failAll('The pinned resolver returned malformed JSON');
          continue;
        }
        if (
          response &&
          typeof response === 'object' &&
          (response as { schema?: unknown }).schema ===
            'hu-practice-continual-resolver-batch-result-v1'
        ) {
          const results = (response as { results?: unknown }).results;
          if (!Array.isArray(results)) {
            this.failAll('The pinned resolver returned a malformed batch');
            continue;
          }
          for (const result of results) this.dispatch(result);
          continue;
        }
        this.dispatch(response);
      }
    });
    child.stderr.on('data', (chunk: string) => {
      this.stderr = (this.stderr + chunk).slice(-MAX_STDERR_BYTES);
    });
    child.once('exit', (code, signal) => {
      if (this.child === child) this.child = null;
      const detail = this.stderr.trim();
      this.stderr = '';
      this.stdout = '';
      this.failAll(
        `The pinned practice resolver stopped (${signal ?? code ?? 'unknown'})${
          detail ? `: ${detail}` : ''
        }`
      );
    });
  }

  private dispatch(response: unknown): void {
    const requestId =
      response && typeof response === 'object'
        ? (response as { requestId?: unknown }).requestId
        : null;
    if (typeof requestId !== 'string') {
      this.failAll('The pinned resolver response omitted its request ID');
      return;
    }
    const pending = this.pending.get(requestId);
    if (!pending) return;
    clearTimeout(pending.timeout);
    this.pending.delete(requestId);
    const error = (response as { error?: unknown }).error;
    if (typeof error === 'string') pending.reject(new Error(error));
    else pending.resolve(response);
  }

  private failAll(message: string): void {
    if (this.flushTimer) clearTimeout(this.flushTimer);
    this.flushTimer = null;
    this.queued = [];
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timeout);
      pending.reject(new Error(message));
    }
    this.pending.clear();
  }

  async query(payload: Record<string, unknown>): Promise<unknown> {
    const child = await this.start();
    const requestId = randomUUID();
    const request = { ...payload, requestId };
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(requestId);
        reject(new Error('The pinned practice resolver query timed out'));
      }, REQUEST_TIMEOUT_MS);
      this.pending.set(requestId, { resolve, reject, timeout });
      this.queued.push({ child, requestId, request });
      if (!this.flushTimer) {
        this.flushTimer = setTimeout(() => this.flush(), MICRO_BATCH_DELAY_MS);
      }
    });
  }

  private flush(): void {
    this.flushTimer = null;
    const queued = this.queued.splice(0, MAX_BATCH_QUERIES);
    if (this.queued.length > 0) {
      this.flushTimer = setTimeout(() => this.flush(), 0);
    }
    if (queued.length === 0) return;
    const child = queued[0].child;
    const batch = {
      schema: 'hu-practice-continual-resolver-batch-query-v1',
      requestId: randomUUID(),
      queries: queued.map(({ request }) => request),
    };
    child.stdin.write(`${JSON.stringify(batch)}\n`, (error) => {
      if (!error) return;
      for (const { requestId } of queued) {
        const pending = this.pending.get(requestId);
        if (!pending) continue;
        clearTimeout(pending.timeout);
        this.pending.delete(requestId);
        pending.reject(
          new Error(
            `The pinned practice resolver is unavailable: ${error.message}`
          )
        );
      }
    });
  }

  async stop(): Promise<void> {
    const child = this.child;
    if (!child || child.exitCode !== null) return;
    child.stdin.end();
    await new Promise<void>((resolve) => {
      child.once('exit', () => resolve());
    });
  }
}

export class PracticeSolverPool implements PracticeResolverWorker {
  private readonly loads: number[];
  private readonly affinities = new Map<string, number>();

  constructor(private readonly workers: PracticeResolverWorker[]) {
    if (workers.length === 0) {
      throw new Error('The practice resolver pool needs at least one worker');
    }
    this.loads = workers.map(() => 0);
  }

  async query(
    payload: Record<string, unknown>,
    affinityKey?: string
  ): Promise<unknown> {
    let workerIndex = affinityKey
      ? this.affinities.get(affinityKey)
      : undefined;
    if (workerIndex === undefined) {
      workerIndex = 0;
      for (let index = 1; index < this.workers.length; index++) {
        if (this.loads[index] < this.loads[workerIndex]) workerIndex = index;
      }
      if (affinityKey) {
        if (this.affinities.size >= 512) {
          const oldest = this.affinities.keys().next().value;
          if (oldest !== undefined) this.affinities.delete(oldest);
        }
        this.affinities.set(affinityKey, workerIndex);
      }
    }
    this.loads[workerIndex] += 1;
    try {
      return await this.workers[workerIndex].query(payload);
    } finally {
      this.loads[workerIndex] -= 1;
    }
  }

  async stop(): Promise<void> {
    await Promise.all(this.workers.map((worker) => worker.stop()));
  }
}

const globalResolver = globalThis as typeof globalThis & {
  __practiceSolverPool?: PracticeSolverPool;
};

export function practiceSolverProcess(): PracticeSolverPool {
  globalResolver.__practiceSolverPool ??= new PracticeSolverPool(
    Array.from(
      { length: practiceResolverPoolSize() },
      () => new PracticeSolverProcess()
    )
  );
  return globalResolver.__practiceSolverPool;
}

/** Release the long-lived child process in integration tests and shutdown hooks. */
export async function stopPracticeSolverProcess(): Promise<void> {
  const resolver = globalResolver.__practiceSolverPool;
  if (!resolver) return;
  await resolver.stop();
  delete globalResolver.__practiceSolverPool;
}
