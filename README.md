# Poker Lab

A private Texas Hold'em study workspace for learning preflop ranges, drilling
decisions, tracking weaknesses, and running local postflop solves.

## Product areas

- **Preflop**: heads-up, 6-max, and 9-max opening and response charts on a
  13x13 hand matrix.
- **Practice**: an always-mounted heads-up cash table with exact-card dealing,
  push/fold drills, policy-frequency feedback, and fail-closed full-hand model
  loading.
- **Stats**: IndexedDB-backed EV-loss, confidence, response-time, breakdown,
  trend, and costly-decision analysis for the new practice history.
- **Postflop solver**: browser-worker CFR+ and full-width extensive-form
  fictitious play (XFP) implementations compiled from Rust to WebAssembly.
- **AI analysis**: optional provider-backed spot analysis using a key that stays
  in browser storage and is forwarded only for the current request.

The old equity-calculator route is intentionally removed. The underlying
evaluator and equity engine remain covered because solver and poker-domain
accuracy still depend on them.

## Run locally

```bash
npm install
npm run dev
npm test
npm run build
```

The development server normally runs at `http://localhost:3000`.

`npm run lint` is not suitable for unattended validation yet because the
repository does not have a committed ESLint configuration and `next lint`
opens an interactive setup flow.

## Landing page

The home route uses the selected Deck direction: a tactile playing-card scene
with direct links to Practice and the preflop range library.

## Architecture

| Area | Path |
| --- | --- |
| Card and range model | `lib/cards.ts` |
| Hand evaluator | `lib/evaluator.ts` |
| Equity engine | `lib/equity/` |
| Preflop chart data | `data/preflop/` |
| Practice state machine | `lib/practice-engine.ts` |
| Policy/grading model | `lib/practice-types.ts`, `lib/practice-grading.ts` |
| Practice settings and adaptive sampling | `lib/practice.ts` |
| Practice persistence | `lib/practice-history.ts` |
| Policy shard codec/client | `lib/policy-codec.ts`, `lib/practice-policy-client.ts` |
| Neural artifact/runtime | `lib/neural-policy.ts`, `scripts/policy/export-neural-artifact.mjs` |
| Exact continual-resolver runtime | `lib/server/practice-solver-process.ts`, `app/api/practice/resolve/route.ts` |
| Policy runtime API | `app/api/practice/` |
| Hosted-policy infrastructure/tools | `infra/`, `scripts/policy/` |
| Table formats and current spot | `lib/positions.ts`, `lib/store.ts` |
| Solver worker boundary | `lib/solver/` |
| Rust solver source | `wasm/src/` |
| AI integrations | `lib/ai/`, `app/api/ai/analyze/route.ts` |

The browser solver is a single-street all-in-equity model. The separate Rust
blueprint trainer under `preflop-solver/` models multi-street abstract heads-up
play with external-sampling DCFR and trajectory-recall information sets. Its
full-hand outputs remain advisory and hidden from Practice until two
independent seeds pass every validation and storage gate; Practice never
substitutes fabricated strategy when a model or shard is unavailable.

The current 20bb candidate is checked in as an inactive `Experimental
self-play` manifest. Its immutable gzip model bundle is loaded by one
long-lived Rust child process, and `/api/practice/resolve` verifies every
component hash before returning a policy. `npm run dev` and `npm run build`
compile that binary; Next's output trace includes the binary and model files.
This exact runtime therefore requires a Node host that permits child processes
and local read-only files. It does not require DynamoDB. DynamoDB remains an
optional backend for previously exported static policy/sample shards.

The bundled push/fold corpus remains available at 2/3/5/8/10/12/15/20bb. It
provides validated action frequencies plus deterministic, policy-consistent
counterfactual action-EV estimates. Called-action feedback carries the
conservative Monte Carlo error bound and is graded as low confidence.

## Rebuild WebAssembly

Run from `wasm/`:

```bash
wasm-pack build --release --target web --out-dir pkg
cp pkg/poker_solver_wasm.js pkg/*.d.ts ../lib/solver/pkg/
cp pkg/poker_solver_wasm_bg.wasm ../lib/solver/pkg/
cp pkg/poker_solver_wasm_bg.wasm ../public/wasm/
cargo test --release
```

Generated artifacts under `lib/solver/pkg/` are committed so JavaScript builds
do not require the Rust toolchain.
