// Web Worker entry: runs equity computation off the main thread so the UI
// stays responsive during Monte Carlo / enumeration.

import { computeEquity, PlayerInput, seedRng } from './compute';
import { Range } from '../cards';

export interface EquityJob {
  id: number;
  players: {
    hand?: [number, number];
    // ranges are serialized as [key, weight][] because Map isn't structured-cloned nicely across some setups
    range?: [number, number][];
  }[];
  board: number[];
  maxSamples: number;
  seed: number;
}

self.onmessage = (e: MessageEvent<EquityJob>) => {
  const job = e.data;
  seedRng(job.seed || 1);

  const players: PlayerInput[] = job.players.map((p) => {
    if (p.hand) return { hand: p.hand };
    const range: Range = new Map(p.range ?? []);
    return { range };
  });

  const result = computeEquity(players, {
    board: job.board,
    maxSamples: job.maxSamples,
  });

  (self as unknown as Worker).postMessage({ id: job.id, result });
};
