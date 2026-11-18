# Live 20bb Solver: matched turn–river policy tests

November 17, 2026. No production settings, models, or user interface changed.
The full flop benchmark was explicitly skipped.

## Matched design

Keep the [three original turn starts](live-solver-nash-2026-11-17.md), their
prior frozen flop check/check rows, both entry ranges, the modeled actions,
and all future cards. Use the same exact information-set-consistent response
evaluator on the frozen complete policy for every arm. Each rebuilt live
control reproduced its original profile values and both best responses.

- **Live-2:** shipped 20bb policy: two-update BB turn search; learned BTN;
  existing river routing and iteration budget.
- **Routed-32:** only BB turn update budget changes, 2 → 32. BTN and the
  river resolvers reconstruct beliefs using the modified turn action rows as
  they would in actual play; their models, algorithms, and budgets are unchanged.
- **Joint-2:** deploy both seats from one two-update joint turn–river solve,
  and retain its river rows. No new river solve or learned BTN policy within
  the subgame. This tests two routing changes together, not their individual
  effects. It does not claim safe subgame re-solving from the preflop root.

The test preserves exact card removal and evaluates play to terminal through
all 48 possible public rivers (44 compatible per private deal). Response
payoffs come from actual betting and showdowns, not the learned EV estimates.
NashConv is the sum of the two unilateral response gains, in bb per conditional
spot. Heads-up zero-sum exploitability is half. A lower number is better.

Run timing arms serially with rotated order and evaluator jobs under separate
three-worker resource guards. Cold turn timing starts after model load and
entry-belief reconstruction and includes native action-value calculation and
storing or indexing policy rows. River query timing uses the first legal
public card on the check/check turn line; the joint arm is a cached lookup.
These native timings exclude HTTP/browser overhead and are single local
observations per case/variant, not production percentiles.

## Original three spots

| Case | NashConv live | NashConv routed-32 | NashConv joint-2 | Cold turn live / routed-32 / joint-2 |
| --- | ---: | ---: | ---: | --- |
| Limped paired (2bb pot) | 5.9707 | 4.4189 | 7.5679 | 6.020 / 15.503 / 6.009s |
| Raised rainbow (5bb pot) | 6.4273 | 4.5434 | 7.9395 | 3.158 / 8.332 / 3.220s |
| Three-bet monotone (15bb pot) | 3.3087 | 2.1572 | 4.7301 | 0.815 / 2.549 / 0.831s |
| Mean NashConv / median time | **5.2356** | **3.7065** | **6.7459** | **3.158 / 8.332 / 3.220s** |

Routed-32 reduced NashConv by 29.2% and won all three cases; its median cold
turn cost increased 2.64×. Joint-2 increased NashConv by 28.8% and lost every
case despite making the tested river lookup much cheaper (median 7µs vs about
16ms live). The low-budget bilateral policy is not a strength improvement.
Mean exploitability was 2.6178 / 1.8532 / 3.3729bb in the same arm order.

Most of routed-32's improvement came from lowering BTN's absolute best-response
value against BB, from 4.1480bb to 2.7644bb on average. BB's absolute response
value against BTN moved from 1.0876bb to 0.9420bb. These are finer diagnostics
than comparing unilateral gains alone, because each gain also subtracts the
changed seat's self-play value. They do not certify a particular street or
outside-the-subgame intervention.

## Fresh-board follow-up

The three boards below were fixed before scoring the original trio. They reuse
the corresponding preflop action sequences, but reconstruct their own exact
entry ranges. Only the winner and the live control were tested. These are new
to this comparison, not claimed disjoint from every historical training board.

| Case | Flop → turn | NashConv live | NashConv routed-32 | Cold turn live / routed-32 |
| --- | --- | ---: | ---: | --- |
| Limped connected | 4c 5h 6c → Qd | 5.2756 | 4.0452 | 7.376 / 18.449s |
| Raised monotone | Ac 8c 3c → 9s | 5.5931 | 4.2388 | 3.191 / 11.317s |
| Three-bet paired | Js Jc 4d → 7h | 3.3173 | 2.1348 | 0.835 / 2.588s |
| Mean NashConv / median time | — | **4.7287** | **3.4729** | **3.191 / 11.317s** |

Routed-32 reduced mean NashConv by 26.6% and again won all three. Its median
cold turn cost increased 3.55×. The tested median river query stayed near
16ms under either routing budget. Mean exploitability was 2.3643 → 1.7365bb
per starting spot. With only these six cases, the consistent direction is
encouraging but not a full-game guarantee or a release validation.

Next strength/latency comparison: isolate 8 and 16 updates under unchanged
routing, then compare those budgets with 32 on fixed spots. Don't ship the
costlier budget by default based on this benchmark or treat the tested
two-update joint failure as a verdict on joint solving at higher budgets.

## Verification and reproduction

All 15 policy evaluations and 15 serial timing runs completed. The three old
controls exactly reproduced their earlier NashConv. Across all evaluations,
maximum entry range reconstruction delta was 1.74e-17, maximum absolute
zero-sum profile residual 7.26e-15bb, and maximum probability-sum residual
4.45e-16. Every arm traversed the same public decision count as its control
(5,584 / 2,990 / 776 in each cohort). The deterministic frozen-response parity
and betting-line backup tests passed in release mode. Native serving artifact
identity verification also passed. No full CI or full flop benchmark was run.

Runner: `preflop-solver/neural/run_matched_turn.py`, using the opt-in matched
test from the native library test binary. Build with `cargo test --release
--lib --no-run` from `preflop-solver/`; the measured binary used ThinLTO and
12 codegen units, with the same build for all variants. Start the runner with
`--binary`, `--original` (the completed live-turn benchmark directory) and
`--output` (a new local output directory). The same invocation resumes from
verified receipts; it rejects altered inputs or outputs. Only the conditional
turn prefixes, not entire flop trees, were computed for the fresh cases.

Original case inputs and exact control receipts remain unchanged at
`preflop-solver/neural/runs/live-solver-nash-20261117-c/`. The matched run
protocol and completed result JSON are retained under
`preflop-solver/neural/runs/matched-turn-20261117-a/`:

- Protocol SHA-256: `3a36f0bf061f9e10ee8623e9c5e1095a8b8f55f2b80f2e8d07fc9f04ca32f7d9`.
- Raw summary SHA-256: `710d9870b1376106d59f94cba1c641e20d86c3608eb94291362a415e4e889ca8`.
- Native benchmark binary SHA-256: `1378db0078df917f9c1a4753e696aa03655326d1da3dfdbff4a627f8c314a7b8`.
