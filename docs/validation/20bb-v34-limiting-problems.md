# 20bb v34 limiting-problem corrections

Status: research infrastructure improved; no model was activated.

This sequence addressed the measured evaluator-coverage, omitted-all-in,
continuation-oracle, and model-selection failures. It does not claim that the
current routed model is approximate GTO.

## Learned-response coverage

The full-game learned response now tries three imperfect-information-safe
keys in order: exact trajectory, fine observable backoff, and coarse observable
backoff. Both backoffs deliberately forget information and never observe the
opponent's cards. The coarse key retains made-hand category, draw flags, public
board texture, pot odds, SPR, actor, street, and the exact legal action set.

On the same v27 seed-7602 pilot with 2,000 training deals, 2,000 independent
evaluation deals, four rollouts per action, and a two-particle minimum:

| Metric | Fine backoff only | Fine + coarse |
| --- | ---: | ---: |
| Player 0 total lookup | 1.13% | 1.89% |
| Player 1 total lookup | 1.17% | 2.31% |
| Player 0 postflop lookup | 0.00% | 2.47% |
| Player 1 postflop lookup | 0.27% | 2.09% |
| Exploitability lower-bound point estimate | 0.0199bb/hand | 0.0281bb/hand |
| 99% confidence lower bound | 0 | 0 |

Coverage improved materially, but it remains far too low for a strong learned-
response audit and the confidence lower bound remains zero.

## Exact flop all-ins

The flop public-belief resolver now keeps the configured all-in action and
enumerates every compatible unordered turn-river runout. Exact equity matrices
are cached per flop for dense ranges. A dense two-iteration 4bb-pot resolve on
`2c 7d Th` took 13.78 seconds with eight threads, peaked at 88MB resident, and
had a `1.11e-15bb` projected zero-sum residual.

Resolver-leaf corpus generation no longer forces `include_all_in=false`.
Previous v31/v33 matched-resolver numbers omitted this branch and must not be
compared to new results without rerunning both baseline and candidate.

## Range-conditioned continuation cache

Continuation cache v2 can consume a frozen tabular preflop policy. For every
public preflop leaf it builds separate player reach factors from that player's
own hand class and public actions, applies exact flop blockers, and passes those
ranges into the flop resolver. A `1e-9` action-probability floor preserves
forced-deviation coverage without introducing opponent-card information.

Resolver-derived cache leaves retain both player-conditional CFVs. Legacy v1
rollout caches remain readable and use the original `[u0, -u0]` fallback. The
preflop evaluator now records both profile utilities, their sampled zero-sum
residual, and subtracts both profile values when computing NashConv.

The two-deal, 49-leaf-per-deal smoke used v31 seed 10902 for turn values and
v28 seed 8801 for preflop ranges. It produced 98/98 finite, stack-bounded
conditional leaf values under cache schema v2. The attached value uncertainty
was the measured `0.540214bb`, so 0% of leaves were falsely reported at or below
the `0.02bb` action-SE release threshold. The two-deal profile zero-sum residual
was `0.4973bb`; this tiny corpus is structural evidence only, not a strategic
evaluation.

Cache generation now uses a value-only resolver finish. It exports exactly the
same counterfactual vectors as the full diagnostic resolver while skipping five
unused best-response/control traversals:

| Two-deal exhaustive smoke | Full diagnostics | Value only |
| --- | ---: | ---: |
| Wall time | 1,176.31s | 359.45s |
| User CPU time | 5,756.67s | 1,818.73s |
| Peak resident memory | 286.6MB | 285.0MB |
| Artifact SHA-256 | `6ce7f516...f72898e` | identical |

This is a 3.27x wall-time improvement. A complete 2,652-deal exact-combo cycle
is still not an overnight CPU job; batched/GPU value inference or distributed
board jobs remain necessary before scaling this oracle.

## Downstream model selection

The v33 validator no longer lets tuning RMSE choose a research model. Every
candidate seed must have exactly three distinct matched resolver boards. The
seed with the lowest matched mean resolver exploitability wins, and promotion
still requires at least 2% mean improvement, improvement on two of three
boards, and all prediction-integrity gates. Missing or extra seed reports fail
closed.

## Verification and remaining gates

- Rust release: 67 library tests and 3 CLI tests passed.
- Python: 92 tests passed.
- Dense all-in and continuation cache artifacts were deterministic.
- Active manifests remain unchanged.

The limiting gates remain learned-response coverage/confidence, action-value
uncertainty, zero-sum convergence on an authentic continuation corpus,
downstream resolver exploitability, and ultimately an independent full-game
99% exploitability upper bound. The next training step should not be another
long fit to the old rollout cache. It should first make v2 continuation
generation fast enough for multiple independent range-conditioned corpora,
then run paired preflop solves and exact-all-in matched resolver selection.
