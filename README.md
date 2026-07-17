# Poker Solver — Texas Hold'em

A local Texas Hold'em study tool: an accurate **equity calculator**, **preflop GTO charts**,
a **postflop solver** workspace, and optional **AI spot analysis** with your own API key.
Runs locally with `npm run dev` and deploys to Vercel unchanged.

## Features

- **Equity Calculator** — hand vs hand, hand vs range, range vs range. Exact enumeration
  when the runout is small, Monte Carlo otherwise, computed off the main thread in a Web
  Worker. Shows equity, win/tie split, and pot odds.
- **Preflop GTO Charts** — position-aware opening (RFI) and response ranges on a 13×13
  matrix with mixed-strategy frequency bars. Click a seat or press `1`–`9` to switch
  positions instantly; toggle 6-max / 9-max.
- **Postflop Solver** — real **Discounted CFR (CFR+)** solving, compiled from Rust to
  **WebAssembly** and run entirely in your browser (in a Web Worker). Configure OOP/IP
  ranges, board, pot, stacks, and bet/raise sizes; get GTO bet/check/call/fold
  frequencies and EVs per hand, with exploitability that visibly converges toward zero.
  Uses a single-street all-in-equity model (see below).
- **AI Analysis** — bring your own key (Anthropic Claude or OpenAI). Your key is stored
  only in `localStorage` and forwarded through a serverless route to the provider; it is
  never persisted server-side. Streams a natural-language read of the current spot.

## Getting started

```bash
npm install
npm run dev      # http://localhost:3000
npm test         # engine unit tests (cards, evaluator, equity)
npm run build    # production build (what Vercel runs)
```

## Deploying to Vercel

Import the repo into Vercel — no configuration needed. The AI route runs as an edge
function; all equity/solver compute happens client-side in the visitor's browser.

## Architecture

| Area | Path |
| --- | --- |
| Card model + range notation | `lib/cards.ts` |
| Hand evaluator (5–7 cards) | `lib/evaluator.ts` |
| Equity engine (enum + Monte Carlo) | `lib/equity/` |
| Position state | `lib/positions.ts`, `lib/store.ts` |
| Preflop chart data + rendering | `data/preflop/`, `lib/preflop.ts` |
| Solver interface (swappable) | `lib/solver/engine.ts` |
| AI provider adapters + prompt | `lib/ai/`, `app/api/ai/analyze/route.ts` |
| Core UI (matrix, table, cards, ranges) | `components/` |

The accuracy-critical code (`lib/cards.ts`, `lib/evaluator.ts`, `lib/equity/compute.ts`)
is covered by unit tests validating known spots (e.g. AA vs KK ≈ 82/18, exact river
enumeration, hand-vs-range).

## The WASM CFR solver

The solver is a self-contained **vectorized CFR+** implementation in `wasm/src/` (no
third-party solver dependency, so no AGPL obligations):

- `wasm/src/eval.rs` — Rust port of the hand evaluator.
- `wasm/src/lib.rs` — the solver: builds a discretized single-street bet/raise tree,
  precomputes a strategy-independent all-in-equity matrix over the runout, and runs CFR+
  to convergence. Exposes `solve(json)` via `wasm-bindgen`.

**Model.** It solves one street (flop or turn) with a configurable bet/raise tree. Any
line ending in a call or check-check is valued by the two hands' equity over the
remaining runout (the "all-in equity" simplification). This yields real GTO frequencies
and EVs and converges fast; it does not model separate per-street betting like a
full multi-street solver.

**Rebuilding the wasm** (only needed if you change the Rust):

```bash
cd wasm
wasm-pack build --release --target web --out-dir pkg
cp pkg/poker_solver_wasm.js pkg/*.d.ts ../lib/solver/pkg/
cp pkg/poker_solver_wasm_bg.wasm ../lib/solver/pkg/     # bundled by webpack
cp pkg/poker_solver_wasm_bg.wasm ../public/wasm/         # optional runtime copy
cargo test --release                                     # native correctness test
```

The compiled artifacts under `lib/solver/pkg/` are committed, so `npm run build` works
without the Rust toolchain. The solve runs client-side (in a Web Worker), so it works
both locally and on Vercel. `next.config.js` sets the `COOP`/`COEP` headers, leaving room
to add multithreading (`wasm-bindgen-rayon` + `SharedArrayBuffer`) later.
