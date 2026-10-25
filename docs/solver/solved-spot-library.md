# Solved-spot library: first vertical slice

## Scope

`/practice/spots` serves complete, offline-solved river and turn-to-river games, separately from
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

- Source registry: `data/solved-spots/library.json`; each ID pins its own JSON input.
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

`spots:build` builds the native binary and runs each registered input with a 10-minute process
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

## Expansion (2026-10-24)

Five independent starting situations are now published locally. Each remains a
conditional study using disclosed illustrative ranges, not a newly activated
full-hand depth. All retain the unchanged ≤0.05bb **summed** best-response gate
and ≤0.0001bb action-EV discrepancy gate.

| Starting spot | Total response gain (bb/spot) | Artifact bytes |
| --- | ---: | ---: |
| 50bb paired river | 0.000172020 | 87,240 |
| 50bb ace-high dry river | 0.000293954 | 80,295 |
| 50bb three-heart river | 0.000100904 | 94,178 |
| 100bb connected river | 0.000037370 | 101,429 |
| 50bb turn-to-river | 0.008133890 | 2,092,523 |

The original paired-river artifact and its SHA-256 are unchanged. Registry,
catalog metadata, complete artifact bytes and native validation are checked
together; an empty/missing catalog cannot silently clear production validation.

### Complete turn game, not independent street resolves

The new `solved-turn-river-spot-v1` schema adds an explicit board at each node
and a river-card-to-node map at each non-all-in turn round closure. Existing
`solved-river-spot-v1` bytes remain readable and reproducible. A single joint
turn/river DCFR solve supplies **all** descendants. No river strategy is freshly
solved, substituted or predicted during playback. Bet sizes remain half-pot,
pot and all-in, with one subsequent raise; illegal stack-clipped sizes disappear
according to the existing rules, not arbitrary UI filtering.

The turn exercise starts on Qs 9h 6c 2d with a 32bb pot and 34bb behind at an
initial 50bb depth. Its deliberately narrow ranges each contain 14 unblocked
combos, producing 177 compatible pairs and 7,788 pair–river outcomes. Every pair
has exactly 44 possible rivers. The public chance maps contain all 48 non-board
cards; the dealt private hands remove four at playback. Turn response choices
integrate future river chance **before** maximizing. After a river is revealed,
the responder can condition on that public card, never on the opponent's cards.
All-in turn calls integrate river showdown outcomes too.

The bounded pilot at 128 rounds took 8.55 seconds, reached 0.064991070bb total
response gain, and was rejected without exporting a serving artifact. The same
configuration at 512 rounds took 25.70 seconds and reached 0.008133890bb; maximum
EV discrepancy is 0.00000329431bb. Peak memory footprint was about 204MB (maximum
resident set about 214MB). This is measured local performance for this narrow
tree—not a full-game training-time estimate or evidence for arbitrary ranges.
The production export repeats the solve and independently verifies the result.

Accepted turn artifact SHA-256:
`0f1a25bf2de1c4adcd471cfe6a4a084a17b386bbaa97542808e2c5fbafa91807`.

### Playback and verification additions

- Starting private cards still sample the exact joint ranges. The hidden deck
  is shuffled uniformly; turn strategies/EVs do not read the future deck.
- The shared hand engine draws the river, then the pinned chance edge selects
  its precomputed policy. Missing chance edges pause rather than re-solve.
- Opponent playback continues until the hero acts or the hand ends, including
  BB calling a BTN turn bet and then acting first again on the river.
- Both spot and seat changes apply on the next deal, with pending notices.
  Feedback and complete action history identify the street of each decision.
- Rust directly enumerates private pairs and river chance, including a separate
  structural pass over every public edge. TypeScript independently reproduces
  the whole-subgame response values and **every** served action EV.
- Tests cover every river artifact, every turn/river public node and all legal
  chance edges through the actual hand engine, chip conservation, card removal,
  no future-card leakage, missing chance maps and consecutive opponent actions.
- `npm test`: 321 passed, 20 optional integration cases skipped. Rust release:
  473 library + 9 binary + 15 CLI passed; 60 fixture/resource cases ignored.
  `npm run build` passes with fresh native verification of all five artifacts.
- Browser acceptance completed a turn decision, bet/call, river decision and
  all-in showdown through the shared engine. Opponent cards remained hidden
  through feedback and were revealed only in terminal review. Pending spot
  changes preserved the active hand; mobile feedback focus and visible keyboard
  focus worked under reduced motion. No horizontal overflow at
  375/768/1024/1440px in either theme, or at 812px landscape; 48px mobile action
  controls cleared the bottom navigation. No captured console/unhandled errors
  or resolver/neural API requests occurred. Artifact responses retained
  immutable caching and COOP/COEP headers. Native select options were exercised
  via a DOM change event because macOS menu selection did not respond to CDP
  keyboard events; native option-keyboard selection is not certified by this run.

The complete library is about 2.46MB uncompressed. Static immutable hosting still
suffices; no database, neural weights, paid compute or online solve was added.
The response bound is weighted from each spot's declared starting distribution,
not a pointwise equilibrium guarantee for every low-reach or off-path mix.
Next expansion should broaden the *declared range support* and starting turn
situations before attempting a full flop tree. This checkpoint does not solve
preflop, arbitrary flops, or prove full-game Hold'em exploitability.
