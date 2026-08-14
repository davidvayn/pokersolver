import 'server-only';

import { randomUUID } from 'node:crypto';
import { cpus } from 'node:os';
import path from 'node:path';
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';

const MODEL_VERSION = 'hu-20bb-v102-consensus-continual-resolver-experimental';
const NETWORK_SHA256 =
  '310b9d1a39a3ecd6beff4ac99533a8ce5847dba05d9627b650a446c36e26b7c3';
const RANGE_POLICY_SHA256 =
  '7296e5a54cd0c310f5fd7dc126937b41131c54d00b0bc2c6807d7791c14772f0';
const VALUE_NETWORK_SHA256 =
  '2764959d5ddf004dc7ad9146a831250bbf2db2b3fcf86a3f3daf2cc51e458202';
const PREFLOP_ACTION_VALUES_SHA256 =
  '8369f6dde1f6de8380e0bc32cf54003524791478939741c9fc427b47d3efa70a';
const REQUEST_TIMEOUT_MS = 5 * 60 * 1000;
const MAX_STDERR_BYTES = 16_384;

export const PRACTICE_RESOLVER_IDENTITY = {
  modelVersion: MODEL_VERSION,
  networkSha256: NETWORK_SHA256,
  rangePolicySha256: RANGE_POLICY_SHA256,
  valueNetworkSha256: VALUE_NETWORK_SHA256,
  preflopActionValuesSha256: PREFLOP_ACTION_VALUES_SHA256,
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
      path.join(modelRoot, 'v57-seed7601-networks.json.gz'),
      '--range-policy',
      path.join(modelRoot, 'v102-seed20931-range-policy.json.gz'),
      '--preflop-action-values',
      path.join(modelRoot, 'v101-seed7601-preflop-action-values.json.gz'),
      '--flop-resolver-iterations',
      '2',
      '--flop-resolver-averaging-delay',
      '0',
      '--flop-resolver-value-network',
      path.join(modelRoot, 'v91-seed22501-turn-value.json.gz'),
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
}

const globalResolver = globalThis as typeof globalThis & {
  __practiceSolverProcess?: PracticeSolverProcess;
};

export function practiceSolverProcess(): PracticeSolverProcess {
  globalResolver.__practiceSolverProcess ??= new PracticeSolverProcess();
  return globalResolver.__practiceSolverProcess;
}
