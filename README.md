# Poker Lab

A private Texas Hold'em study workspace for learning preflop ranges, drilling
decisions, tracking weaknesses, and running local postflop solves.

## Product areas

- **Preflop**: heads-up, 6-max, and 9-max opening and response charts on a
  13x13 hand matrix.
- **Practice**: configurable random preflop spots generated from the bundled
  chart library.
- **Stats**: locally stored accuracy, trend, position, spot-type, and action
  analysis.
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
| Practice and stats model | `lib/practice.ts` |
| Practice persistence | `lib/practice-history.ts` |
| Table formats and current spot | `lib/positions.ts`, `lib/store.ts` |
| Solver worker boundary | `lib/solver/` |
| Rust solver source | `wasm/src/` |
| AI integrations | `lib/ai/`, `app/api/ai/analyze/route.ts` |

The current solver is a single-street all-in-equity model. It supports a
configurable bet/raise tree and produces converged strategy frequencies with
either CFR+ or XFP, but it does not model separate decisions across multiple
postflop streets. XFP is initialized with the same-iteration CFR+ strategy,
then trains by exact best-response self-play using realization-reach-corrected
behavioral averaging; its result includes cross-play EVs against that CFR+
baseline.

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
