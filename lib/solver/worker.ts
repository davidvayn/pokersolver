// Web Worker that runs the Rust→WASM postflop solver off the main thread.

import init, { solve } from './pkg/poker_solver_wasm.js';

let ready: Promise<unknown> | null = null;

function ensureReady() {
  if (!ready) {
    // The bundler emits the .wasm asset next to the glue and rewrites its URL;
    // calling init() with no argument uses that default path. Cross-origin
    // isolation headers are set in next.config.js.
    ready = init();
  }
  return ready;
}

export interface SolveRequest {
  id: number;
  input: unknown; // serialized to JSON for the Rust entry point
}

self.onmessage = async (e: MessageEvent<SolveRequest>) => {
  const { id, input } = e.data;
  try {
    await ensureReady();
    const json = solve(JSON.stringify(input));
    const result = JSON.parse(json);
    (self as unknown as Worker).postMessage({ id, result });
  } catch (err) {
    (self as unknown as Worker).postMessage({
      id,
      result: { error: (err as Error).message },
    });
  }
};
