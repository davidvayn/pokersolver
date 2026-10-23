# Solved-spot library: first vertical slice

## Scope

`/practice/spots` serves complete, offline-solved river games, separately from
experimental full-hand practice. Each game pins the board, initial equal stacks,
pot, declared exact-combo ranges, betting sizes, and complete continuation policy.
It does **not** certify a preflop strategy or full-game Hold'em exploitability.
The initial 50bb example uses illustrative ranges, explicitly disclosed in the UI.
No 50bb/100bb full-hand manifest is activated. Existing 20bb weights are untouched.

This changes the release unit from a universal neural policy to a bounded game
that can actually be solved and checked. No third-party solver code, licensed
solution corpus, neural continuation, distillation, paid compute or hosted
database is required for this slice.

## Solve and verify

- Source input: `data/solved-spots/river-50bb-paired.json`.
- Native module: `preflop-solver/src/blueprint/public_belief/solved_spot.rs`.
- Reuse existing exact-card-removal river DCFR with paired alternating updates;
  alpha=1.5, beta=0, gamma=2, discard first 10% for averaging.
- No hand bucketing. Every unblocked combo in a declared hand class starts with
  equal weight. Normalize each range and condition on private-card compatibility.
- Export only the frozen average policies and their action values, at every
  public decision node. No regret arrays or resumable training state are served.
- Run a separate direct-compatible-pair traversal of the exported float32 policy.
  It does not reuse the solver's card-marginal evaluator or vectorized response.
  The responder maximizes **after** integrating hidden opponent hands, never
  chooses a different action based on an unobserved opponent hand.
- Require the **sum** of both unilateral best-response gains ≤0.05bb per starting
  spot, every action-EV discrepancy ≤0.0001bb, probability sum error ≤1e-6, and
  payoff conservation residual ≤1e-5bb. No halving NashConv to clear this gate.
- A second implementation in TypeScript uses the website hand engine and
  evaluator to reproduce all saved EVs and both information-set best responses.
  Tests also exhaust every public branch, including off-policy user actions,
  and check chip conservation, complete transitions and settlement.

The independent checker still shares the Rust card representation and game-rule
implementation; the TypeScript check adds a separate rules/evaluator path. These
checks are strong evidence for this finite conditional game, not a formal proof
of the entire application or a full-game safety guarantee.

## Reproduction

```sh
npm run spots:build
npm run spots:verify
npx vitest run lib/solved-spots.test.ts components/practice/PracticeTable.test.ts
cargo test --release --manifest-path preflop-solver/Cargo.toml solved_spot
```

`spots:build` builds the native binary and runs the input with a 10-minute process
timeout. Failed convergence exports no serving artifact. A successful artifact
is named by its complete SHA-256 and written exclusively; existing bytes are
never overwritten. The small activation catalog points at those immutable bytes.
`spots:verify` checks source input, byte count, checksum and freshly recomputed
native verification against both artifact and catalog. Production builds run it.
Rebuilding the same source should produce the same artifact hash.

## Serving and practice

The catalog is bundled with the route. The browser downloads one immutable
JSON artifact, verifies SHA-256 and schema, and uses direct table lookups—no
online solver requests. Static hosting/CDN caching is sufficient. Parsing rounds
the JSON's shortest float32 decimals back to float32, then normalizes each row
in float64, matching the actual categorical distribution and native verification.
Missing, malformed or mismatched data pauses the table with Retry, never guesses.

The joint starting deal respects both ranges and exact card removal. Both seats
are playable. Seat changes apply on the next deal. Practice uses the existing
hand engine, cards, chip stacks, logs, settlement and accessible action controls.
Every modeled action is available, including zero-frequency user deviations;
opponent actions sample the complete frozen mix. Opponent cards remain hidden
until terminal review. Feedback reveals mixes and EVs only after the user acts.
Mobile Review decision jumps to the feedback section; keyboard focus is retained.
The complete river log is in the analyst rail rather than overlaid on the felt,
avoiding collisions with board cards on a narrow viewport.

EVs are net from the original hand start, including earlier commitments, following
the saved continuation thereafter. EV loss is best listed action EV minus chosen
action EV; a low-frequency mixed action is not automatically a mistake. Exact
enumeration has zero Monte Carlo SE, not zero numeric error or zero strategy gap.
An off-path node with zero compatible opponent reach has unavailable values and
is not graded. Per-hand feedback is session-only in this first slice; it is not
silently mixed into legacy full-hand training statistics.

## Next expansion boundary

Add distinct verified river situations before scaling breadth. Then publish a
complete turn-to-river tree, retaining one policy and posterior range evolution
through all descendants. Do not stitch independently re-solved streets together
and call that a full-hand equilibrium. Only introduce neural compression or a
database after measured artifact size/latency warrants it.

## First accepted artifact and checks (2026-10-22)

- 50bb, 12bb starting pot, Qs 9h 6c 2d 2s; 71 BTN / 63 BB combos.
- 2,000 paired DCFR rounds; 14 complete public decision nodes; 3,979 compatible
  private-card deals verified, with no sampling in the verifier.
- Summed best-response gain: **0.00017201993bb per starting spot** (gate ≤0.05).
- Maximum stored-action EV error: **0.00000189016bb** (gate ≤0.0001).
- Maximum stored probability-sum error: 3.73e-8; conservation residual 1.34e-15bb.
- 87,240 bytes; immutable artifact SHA-256:
  `8823f13de847f54cb1c7f9f03c1e3cafd90c216162e74b199acf341033e00292`.
- A repeated solve/export produced identical bytes and the same hash.
- `npm test`: 295 passed, 20 optional integrations skipped. Rust release suite:
  472 library + 9 binary + 15 CLI passed, 60 resource/fixture cases ignored.
- `npm run build` passes, including active-20bb integrity and fresh spot verification.
- Browser Use exercised decision → feedback → opponent bet → user raise → fold
  review, hidden-card/reveal behavior, keyboard focus, mobile feedback navigation,
  light/dark and reduced-motion. No horizontal overflow at 375/768/1024/1440px;
  no captured console/unhandled errors; mobile action buttons clear navigation.
  The artifact response retains immutable caching and COOP/COEP; no resolver API
  or neural policy request occurs in this route's practice flow.

This is an accepted **conditional river study**, not a release of the rejected
50bb full-hand neural policy, an arbitrary-board solver, or a full-game GTO claim.
