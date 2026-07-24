# Poker Solver Agent Guide

## Scope and architecture

This repository is a local Texas Hold'em study application built with Next.js
15, React 19, TypeScript, Tailwind CSS, Zustand, and Vitest. It has four
user-facing areas:

- `app/preflop/`: bundled heads-up, 6-max, and 9-max preflop charts.
- `app/practice/`: configurable preflop decision drills.
- `app/stats/`: local practice-history analysis and weakness reporting.
- `app/solver/`: single-street postflop CFR+ solver run in a browser worker.
- `app/settings/` and `app/api/ai/analyze/`: client-side AI settings and the
  edge route that proxies streaming requests to Anthropic or OpenAI.

Keep computation and UI concerns separated. `lib/` contains poker-domain code,
state, worker clients, and integrations; `components/` contains reusable UI;
`data/preflop/` holds chart data; `wasm/` is the Rust source for the solver.
Use the `@/` TypeScript path alias for repository-root imports.

## Important implementation boundaries

- `lib/cards.ts` is the canonical card/range model. A card is an integer
  `rank * 4 + suit`, ranks run `2..A`, suits run `c,d,h,s`, and a `Range` is a
  `Map<comboKey, weight>`. Preserve these representations across UI, workers,
  TypeScript, and Rust boundaries.
- `lib/evaluator.ts`, `lib/equity/compute.ts`, and `lib/practice.ts` are
  accuracy-critical pure code. Add or update deterministic unit tests for
  changes here; seed the RNG with `seedRng()` when testing equity sampling.
- `data/preflop/ranges.ts` is the shared source for the preflop library and
  practice questions. Keep format metadata and action ranges valid and
  disjoint. Practice history is client-only localStorage data managed through
  `lib/practice-history.ts`.
- Browser work must stay off the main thread. `lib/equity/client.ts` /
  `worker.ts` and `lib/solver/client.ts` / `worker.ts` are paired transport
  layers. Serialize `Map`s as entry arrays across worker messages; do not pass
  browser-only APIs into server components.
- The solver accepts serialized combo triples in `lib/solver/client.ts` and
  JSON in `wasm/src/lib.rs`. Coordinate schema changes on both sides. The
  current model is a one-street all-in-equity simplification, not a full
  multi-street poker engine.
- The AI API route is Edge Runtime. API keys originate in browser `localStorage`
  and are forwarded only for the request. Do not persist, log, or expose them.
- `next.config.js` sets COOP/COEP headers needed by the WASM/worker setup.
  Retain equivalent headers when changing Next configuration or deployment.

## Commands

From the repository root:

```bash
npm install
npm run dev       # Next development server, fixed at http://localhost:3000
npm test          # Vitest: all *.test.ts files in the Node environment
npm run build     # production compile, type check, and Next validation
npm start         # serve a completed production build
npm run test:watch
```

Development output lives in `.next-dev`; production builds use `.next`. Do not
make them share a directory or move either directory while its process runs.

For every browser-facing change, start the development server and use the
installed `browser-use` CLI to exercise the affected routes. Inspect visual
layout at desktop and mobile viewport sizes, keyboard-accessible names, browser
console errors, and the changed interaction flows. Build output alone is not UI
validation.

Rust validation and regeneration are run from `wasm/`:

```bash
cargo test --release
wasm-pack build --release --target web --out-dir pkg
cp pkg/poker_solver_wasm.js pkg/*.d.ts ../lib/solver/pkg/
cp pkg/poker_solver_wasm_bg.wasm ../lib/solver/pkg/
cp pkg/poker_solver_wasm_bg.wasm ../public/wasm/
```

`npm run lint` currently invokes `next lint`, which opens interactive ESLint
setup because no ESLint configuration is committed. Do not use it as an
unattended check until the project adds that configuration; use `npm test` and
`npm run build` instead.

## Testing expectations

- Tests live adjacent to the TypeScript they cover and use `*.test.ts` with
  Vitest (`describe`, `it`, `expect`). The configured environment is Node.
- Cover known poker outcomes, range parsing/serialization, practice generation
  and scoring, and numerical invariants with tolerances where Monte Carlo is
  involved. Avoid brittle exact assertions for sampled equity.
- Run targeted tests during iteration, then `npm test` for TypeScript/domain
  changes. Run `cargo test --release` when touching `wasm/src/`.
- Run `npm run build` for app, worker, API-route, configuration, or generated
  WASM changes. It is the closest local analogue to the Vercel production build.

## TypeScript and UI conventions

- TypeScript is strict. Prefer explicit interfaces for worker payloads, solver
  results, and API bodies; use `import type` for type-only imports.
- Mark modules `'use client'` only when they use hooks, browser globals,
  Zustand hooks, or workers. App pages that compose only server-safe components
  can remain server components.
- Styling uses Tailwind plus CSS variables defined in `app/globals.css` and
  exposed through `tailwind.config.ts`. Reuse semantic tokens such as `bg`,
  `surface`, `border`, `fg`, `muted`, and poker action colors instead of adding
  hard-coded theme colors.
- Keep interfaces keyboard-accessible and responsive. Existing controls use
  `focus-visible` styles, semantic buttons, labels, and status/alert roles
  where applicable; preserve those patterns.
- Zustand stores in `lib/store.ts` and `lib/ui-store.ts` are intentionally
  small client-side state containers. Keep persisted data limited to the AI
  settings helper and theme preference.

## Generated and ignored files

- Never edit `lib/solver/pkg/poker_solver_wasm.*` by hand. They are generated
  by `wasm-pack`, but are committed so JavaScript builds work without Rust.
  Regenerate and commit them with the Rust source whenever the WASM interface
  or behavior changes.
- `public/wasm/poker_solver_wasm_bg.wasm` is the corresponding runtime copy;
  refresh it with the generated package.
- Do not commit `node_modules/`, `.next/`, `wasm/target/`, `wasm/pkg/`,
  `wasm/pkg-node/`, `out/`, `build/`, TypeScript build-info files, or local
  `.env*.local` files. These are ignored.
- `wasm/Cargo.lock` is committed. Update it only when Rust dependencies change.

## Before handing off

Keep changes narrowly scoped and preserve unrelated work in a dirty tree. For
behavioral changes, report the commands run and any limitations, especially
Monte Carlo variance, worker/browser-only behavior, or intentional solver-model
constraints.
