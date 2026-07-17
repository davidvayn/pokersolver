'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { Range } from '../cards';

export interface SolverInput {
  board: number[];
  oop: [number, number, number][]; // [card0, card1, weight]
  ip: [number, number, number][];
  pot: number;
  stack: number;
  bet_sizes: number[];
  raise_sizes: number[];
  iterations: number;
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

export function useSolver() {
  const workerRef = useRef<Worker | null>(null);
  const pending = useRef(new Map<number, (r: SolverResult) => void>());
  const [running, setRunning] = useState(false);
  const [available, setAvailable] = useState(false);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      const w = new Worker(new URL('./worker.ts', import.meta.url), {
        type: 'module',
      });
      w.onmessage = (e: MessageEvent<{ id: number; result: SolverResult }>) => {
        const resolve = pending.current.get(e.data.id);
        if (resolve) {
          resolve(e.data.result);
          pending.current.delete(e.data.id);
        }
      };
      workerRef.current = w;
      setAvailable(true);
      return () => w.terminate();
    } catch {
      setAvailable(false);
    }
  }, []);

  const solve = useCallback((input: SolverInput): Promise<SolverResult> => {
    if (!workerRef.current) {
      return Promise.resolve({ error: 'Solver worker unavailable' } as SolverResult);
    }
    setRunning(true);
    const id = jobCounter++;
    return new Promise<SolverResult>((resolve) => {
      pending.current.set(id, resolve);
      workerRef.current!.postMessage({ id, input });
    }).then((r) => {
      setRunning(false);
      return r;
    });
  }, []);

  return { solve, running, available };
}
