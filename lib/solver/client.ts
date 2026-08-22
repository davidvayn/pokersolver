'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { Range } from '../cards';

export type SolverAlgorithm = 'cfr_plus';

export interface SolverInput {
  board: number[];
  oop: [number, number, number][]; // [card0, card1, weight]
  ip: [number, number, number][];
  pot: number;
  stack: number;
  bet_sizes: number[];
  raise_sizes: number[];
  iterations: number;
  algorithm: SolverAlgorithm;
  max_combos: number;
}

export interface ActionStrategy {
  action: string;
  freq: number;
  ev: number;
}
export interface ClassRow {
  class: string;
  combos: number;
  actions: ActionStrategy[];
}
export interface NodeStrategy {
  title: string;
  actions: string[];
  rows: ClassRow[];
}
export interface SolverResult {
  algorithm: SolverAlgorithm;
  iterations: number;
  exploitability_pct: number;
  oop_ev: number;
  ip_ev: number;
  pot: number;
  oop_combos: number;
  ip_combos: number;
  truncated: boolean;
  oop: NodeStrategy;
  ip: NodeStrategy;
  exploitability_history: number[];
  error?: string;
}

export function rangeToTriples(range: Range): [number, number, number][] {
  const out: [number, number, number][] = [];
  for (const [key, weight] of range) {
    if (weight <= 0) continue;
    const hi = Math.floor((1 + Math.sqrt(1 + 8 * key)) / 2);
    const lo = key - (hi * (hi - 1)) / 2;
    out.push([hi, lo, weight]);
  }
  return out;
}

let jobCounter = 1;
const SOLVE_TIMEOUT_MS = 45_000;

interface PendingSolve {
  resolve: (result: SolverResult) => void;
  timeout: ReturnType<typeof setTimeout>;
}

function errorResult(message: string): SolverResult {
  return { error: message } as SolverResult;
}

export function useSolver() {
  const workerRef = useRef<Worker | null>(null);
  const pending = useRef(new Map<number, PendingSolve>());
  const [running, setRunning] = useState(false);
  const [available, setAvailable] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const failPending = (message: string) => {
      setAvailable(false);
      setError(message);
      for (const request of pending.current.values()) {
        clearTimeout(request.timeout);
        request.resolve(errorResult(message));
      }
      pending.current.clear();
      setRunning(false);
    };

    try {
      const w = new Worker(new URL('./worker.ts', import.meta.url), {
        type: 'module',
      });
      w.onmessage = (e: MessageEvent<{ id: number; result: SolverResult }>) => {
        const request = pending.current.get(e.data.id);
        if (request) {
          clearTimeout(request.timeout);
          pending.current.delete(e.data.id);
          request.resolve(e.data.result);
          setRunning(pending.current.size > 0);
        }
      };
      w.onerror = (event) => {
        event.preventDefault();
        failPending(event.message || 'Solver worker failed to load');
      };
      w.onmessageerror = () => {
        failPending('Solver worker returned an unreadable response');
      };
      workerRef.current = w;
      setError(null);
      setAvailable(true);
      return () => {
        w.terminate();
        for (const request of pending.current.values()) {
          clearTimeout(request.timeout);
          request.resolve(errorResult('Solver worker stopped'));
        }
        pending.current.clear();
      };
    } catch (reason) {
      failPending(
        reason instanceof Error ? reason.message : 'Solver worker unavailable'
      );
    }
  }, []);

  const solve = useCallback((input: SolverInput): Promise<SolverResult> => {
    if (!workerRef.current) {
      return Promise.resolve(errorResult(error ?? 'Solver worker unavailable'));
    }
    setRunning(true);
    const id = jobCounter++;
    return new Promise<SolverResult>((resolve) => {
      const timeout = setTimeout(() => {
        pending.current.delete(id);
        setRunning(pending.current.size > 0);
        resolve(errorResult('Solver timed out. Try smaller ranges or reload.'));
      }, SOLVE_TIMEOUT_MS);
      pending.current.set(id, { resolve, timeout });
      try {
        workerRef.current!.postMessage({ id, input });
      } catch (reason) {
        clearTimeout(timeout);
        pending.current.delete(id);
        setRunning(pending.current.size > 0);
        resolve(
          errorResult(
            reason instanceof Error
              ? reason.message
              : 'Could not send the spot to the solver'
          )
        );
      }
    });
  }, [error]);

  return { solve, running, available, error };
}
