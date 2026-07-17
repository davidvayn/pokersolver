'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import type { EquityResult } from './compute';
import type { Range } from '../cards';
import { computeEquity } from './compute';

export interface PlayerSpec {
  hand?: [number, number];
  range?: Range;
}

let jobCounter = 1;

/**
 * Hook that runs equity jobs in a Web Worker (with a synchronous fallback if
 * workers are unavailable, e.g. during SSR or tests).
 */
export function useEquity() {
  const workerRef = useRef<Worker | null>(null);
  const pending = useRef(new Map<number, (r: EquityResult) => void>());
  const [running, setRunning] = useState(false);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      const w = new Worker(new URL('./worker.ts', import.meta.url));
      w.onmessage = (e: MessageEvent<{ id: number; result: EquityResult }>) => {
        const resolve = pending.current.get(e.data.id);
        if (resolve) {
          resolve(e.data.result);
          pending.current.delete(e.data.id);
        }
      };
      workerRef.current = w;
      return () => w.terminate();
    } catch {
      workerRef.current = null;
    }
  }, []);

  const run = useCallback(
    (
      players: PlayerSpec[],
      board: number[] = [],
      maxSamples = 200000
    ): Promise<EquityResult> => {
      setRunning(true);
      const id = jobCounter++;
      const seed = (id * 2654435761) >>> 0;

      const done = (r: EquityResult) => {
        setRunning(false);
        return r;
      };

      if (workerRef.current) {
        return new Promise<EquityResult>((resolve) => {
          pending.current.set(id, resolve);
          workerRef.current!.postMessage({
            id,
            players: players.map((p) => ({
              hand: p.hand,
              range: p.range ? Array.from(p.range.entries()) : undefined,
            })),
            board,
            maxSamples,
            seed,
          });
        }).then(done);
      }

      // Fallback: run on the main thread.
      return Promise.resolve(
        computeEquity(players, { board, maxSamples })
      ).then(done);
    },
    []
  );

  return { run, running };
}
