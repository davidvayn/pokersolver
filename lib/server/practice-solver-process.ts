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

function command(): { executable: string; args: string[] } {
  const root = process.cwd();
  const modelRoot = configuredPath(
    'PRACTICE_RESOLVER_MODEL_DIR',
    path.join(root, 'preflop-solver', 'models', 'practice')
  );
  const executable = configuredPath(
    'PRACTICE_RESOLVER_BIN',
    path.join(root, 'preflop-solver', 'target', 'release', 'preflop-solver')
  );
  const threads = Math.max(
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
      '2',
      '--flop-resolver-averaging-delay',
      '0',
      '--flop-resolver-value-network',
      path.join(modelRoot, resolverArtifactFiles.flopValueNetwork),
      '--flop-resolver-threads',
      String(threads),
      '--turn-resolver-iterations',
      '2',
      '--turn-resolver-averaging-delay',
      '0',
      '--turn-resolver-threads',
      String(threads),
      '--river-resolver-iterations',
      '2',
      '--river-resolver-averaging-delay',
      '0',
    ],
  };
}

class PracticeSolverProcess {
  private child: ChildProcessWithoutNullStreams | null = null;
  private starting: Promise<ChildProcessWithoutNullStreams> | null = null;
  private stdout = '';
  private stderr = '';
  private pending = new Map<string, PendingRequest>();

  private async start(): Promise<ChildProcessWithoutNullStreams> {
    if (this.child && !this.child.killed && this.child.exitCode === null) {
      return this.child;
    }
    if (this.starting) return this.starting;
    this.starting = new Promise((resolve, reject) => {
      const { executable, args } = command();
      const child = spawn(executable, args, {
        cwd: process.cwd(),
        env: process.env,
        stdio: ['pipe', 'pipe', 'pipe'],
      });
      const failed = (error: Error) => {
        this.starting = null;
        reject(
          new Error(`The pinned practice resolver could not start: ${error.message}`)
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
        const requestId =
          response && typeof response === 'object'
            ? (response as { requestId?: unknown }).requestId
            : null;
        if (typeof requestId !== 'string') {
          this.failAll('The pinned resolver response omitted its request ID');
          continue;
        }
        const pending = this.pending.get(requestId);
        if (!pending) continue;
        clearTimeout(pending.timeout);
        this.pending.delete(requestId);
        const error = (response as { error?: unknown }).error;
        if (typeof error === 'string') pending.reject(new Error(error));
        else pending.resolve(response);
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

  private failAll(message: string): void {
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
      child.stdin.write(`${JSON.stringify(request)}\n`, (error) => {
        if (!error) return;
        clearTimeout(timeout);
        this.pending.delete(requestId);
        reject(new Error(`The pinned practice resolver is unavailable: ${error.message}`));
      });
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

const globalResolver = globalThis as typeof globalThis & {
  __practiceSolverProcess?: PracticeSolverProcess;
};

export function practiceSolverProcess(): PracticeSolverProcess {
  globalResolver.__practiceSolverProcess ??= new PracticeSolverProcess();
  return globalResolver.__practiceSolverProcess;
}

/** Release the long-lived child process in integration tests and shutdown hooks. */
export async function stopPracticeSolverProcess(): Promise<void> {
  const resolver = globalResolver.__practiceSolverProcess;
  if (!resolver) return;
  await resolver.stop();
  delete globalResolver.__practiceSolverProcess;
}
