# 20bb v27 preflop solve and learned-response validation

Status: research sequence completed; v27 is rejected and not activated.

## Why the continuation oracle exists

The dedicated preflop solver ends when betting advances to the flop. A preflop
action still needs a terminal value, so the continuation oracle freezes the
v26 routed neural policy after each reachable preflop line, rolls the exact
deal to a fold or showdown, and caches player-zero utility. The tabular solver
can then revisit the finite preflop game without repeatedly invoking neural
postflop inference.

The oracle is not a solver and is not assumed to be GTO. It answers the more
limited question “what is this preflop line worth if v26 plays the remaining
streets?” Independent chance seeds and both frozen v26 seeds are required so a
tabular policy cannot be accepted merely for exploiting one finite rollout
corpus.

## Implemented controls

- The best response selects one action for every world in the same observable
  abstract information set. Its key contains the actor, the actor's 169-class
  private hand, and the complete public action history; opponent cards and the
  future board are excluded.
- Chance samples exact unique cards. One complete cycle balances all 1,326
  exact private combinations once in each seat, with a uniformly sampled
  compatible opponent and board. This balances exact-combo marginals; it is
  not an exhaustive uniform joint-deal enumeration.
- The continuation cache records bundle SHA-256 hashes, exact-combo cycles,
  per-leaf action-rollout standard errors, completeness, finite values, and
  stack bounds. Independent caches can be compared by observable hand class
  and public leaf, and compatible caches can be merged without recomputing
  neural rollouts.
- Preflop training uses alternating-traverser external-sampling DCFR with 5%
  explicit behavior-policy exploration and target/behavior importance
  correction. Lazy DCFR discounts are mathematically equivalent to eager
  global discounting while using constant memory.
- Only the frozen average policy is exported. Individual regrets and resumable
  solver state are not exported. Average-policy reach is retained as a scalar
  solely for reach-weighted validation and distillation.
- Distillation expands every 169-class teacher target over all 4, 6, or 12
  exact observable combinations. The teacher's reach is divided across those
  combinations, and paired datasets are loaded and trained sequentially to
  keep peak memory bounded.
- The whole-game learned response groups action rollouts by observable abstract
  information set, freezes one response, and evaluates it on independent
  deals. Only statistically supported actions are deployed; all noisy
  candidates and their uncertainty remain in the postflop resolver artifact.

## v26 preflop response result

The initial one-cycle caches used 2,652 deals, eight rollouts per public leaf,
and one frozen v26 seed each. Across the two v26 preflop policies and two
continuation corpora, information-set-consistent preflop exploitability ranged
from 1.060 to 1.149bb/hand:

| Frozen preflop policy | Oracle A | Oracle B |
| --- | ---: | ---: |
| v26 seed 5101 | 1.089 | 1.149 |
| v26 seed 5102 | 1.060 | 1.125 |

These are exact best-response values inside each sampled preflop game with
frozen v26 continuations. They are not whole-game exploitability estimates.
They nevertheless demonstrate a large preflop leak under both policy seeds and
both continuation samples.

## Why the first tabular pair was rejected

The one-cycle cache was complete and finite, but only 47.1% and 46.0% of its
individual leaf estimates had action-rollout standard error at or below
0.02bb. Independent group means differed by 4.58bb for player zero and 4.22bb
for player one when weighted by group sample count. That diagnostic does not
use strategic reach and therefore emphasizes rare high-variance lines, but it
correctly predicted material oracle overfit.

| Iterations | Training exploitability | Oracle-B exploitability | Cross-seed MAE | Primary agreement |
| ---: | ---: | ---: | ---: | ---: |
| 100,000 | 1.22–1.23 | 1.91–1.93 | 17.27% | 55.95% |
| 500,000 | 0.565–0.570 | 1.490–1.492 | 10.85% | 71.07% |
| 2,000,000 | 0.265–0.268 | 1.339–1.371 | 7.31% | 81.84% |

These preliminary cross-seed columns used square-root raw visit weighting
because average reach was not yet exported. They are convergence diagnostics,
not release-gate measurements. Subsequent artifacts export the importance-
corrected average reach and use it directly.

Two 2M policies trained on different one-cycle oracles disagreed by 18.05%
MAE, confirming that additional identical DCFR iterations would primarily fit
the finite continuation sample. This pair was stopped and cannot be used as a
v27 teacher.

## Stronger mixed-v26 continuation oracle

The replacement oracle pair uses both frozen v26 routed models, selecting one
model round-robin for each rollout. Each independent cache contains 13,260
deals (five complete exact-combo marginal cycles), four action rollouts per
leaf, and 649,740 leaf values. Merging the compatible caches produces 26,520
deals, ten cycles, and 1,299,480 leaf values. The merged cache SHA-256 is
`9677c61507baba284fa8b093df80ad1ade6d460c5a1a0633a0f4ca6969a7b1ad`.

The merged cache is complete and finite, but it is still noisy. Only 56.85% of
individual leaf-action values have standard error at or below 0.02bb. The
maximum public-history mean standard error is 0.119bb, and the maximum
information-group mean standard error is 2.400bb. Independent A/B
information-group means differ by 2.013bb for player zero and 2.036bb for
player one when weighted by raw group sample count. These diagnostics are not
reach-weighted release metrics, but they prohibit treating the oracle as an
exact continuation solver.

The stronger oracle also gives a fairer v26 baseline than the rejected
one-cycle experiment:

| Frozen preflop policy | Oracle A | Oracle B |
| --- | ---: | ---: |
| v26 seed 5101 | 0.495 | 0.484 |
| v26 seed 5102 | 0.466 | 0.467 |

All four lookups have 100% coverage. These values supersede the one-cycle
numbers for model comparison, but remain oracle-relative preflop estimates.

## Selected tabular teacher pair

On one five-cycle training cache, increasing paired DCFR solves from 2M to 10M
iterations reduced training exploitability from about 0.299bb/hand to
0.131–0.134bb/hand. Held-out exploitability improved only from about 0.632 to
0.572bb/hand, which exposed the finite-oracle floor and stopped the planned 50M
extension.

The two five-cycle caches were then merged, and the paired 10M solves were
retrained on the ten-cycle corpus. They reach 0.1321 and 0.1333bb/hand in the
merged sampled game. Their independent source-cache results are:

| Tabular teacher | Oracle A | Oracle B |
| --- | ---: | ---: |
| seed 7601 | 0.325 | 0.315 |
| seed 7602 | 0.326 | 0.318 |

The paired average policies contain 16,900 information sets each and pass the
configured stability gates: 4.437 percentage-point reach-weighted action MAE,
87.14% primary-action agreement, 88.48% tie-aware primary agreement, 0.074
percentage-point maximum aggregate action delta, and 100% lookup intersection.
Stability does not override the failed 0.05bb/hand exploitability target or the
continuation-value uncertainty.

## Neural distillation and payload boundary

Each selected teacher was expanded to 132,600 exact-private-card examples and
distilled from the corresponding v25 preflop student. At 5,000 optimization
steps the students achieve 4.60% and 4.36% reach-weighted teacher MAE, with
87.55% and 87.82% primary-action agreement. Their independent oracle results
retain most of the tabular improvement:

| Policy | Oracle A | Oracle B |
| --- | ---: | ---: |
| v27 student 7601 | 0.348 | 0.334 |
| v27 student 7602 | 0.348 | 0.333 |

The student weights are 876,050 bytes each, versus about 12MB of verbose JSON
for each tabular teacher. The 19MB compressed continuation oracle, 12MB
tabular teachers, 7.6MB distillation datasets, and 29MB routed evaluator JSON
bundles are offline research artifacts and must not be shipped to the browser.
Only one selected 856KiB preflop student belongs at the serving boundary. The
postflop component remains the separately frozen v26 student; the routed JSON
exists only to give Rust a deterministic evaluation interchange format.

## Routed full-hand validation

The paired v27 students were routed to their corresponding frozen v26 postflop
students and evaluated over 5,000 authentic trajectories per policy plus an
independent forced-deviation distribution. Every reached action received 16
counterfactual rollouts.

| Metric | v26 | v27 |
| --- | ---: | ---: |
| Authentic reach-weighted action MAE | 5.69% | 4.39% |
| Authentic primary agreement | 80.15% | 88.17% |
| Authentic maximum aggregate delta | 0.36% | 0.85% |
| Forced-deviation action MAE | 6.60% | 4.85% |
| Forced-deviation primary agreement | 77.81% | 87.39% |
| Forced-deviation maximum aggregate delta | 1.70% | 0.40% |

V27 passes the action-frequency, primary-agreement, aggregate-delta, lookup,
probability, and reach-weighting gates. Its action-EV uncertainty fails: only
22.77% of actions and 10.33% of decisions have standard error at or below
0.02bb, versus the required 95%, and the maximum observed standard error is
4.838bb. No qualifying full-game exploitability upper-bound certificate is
available. The routed result is therefore a stability improvement, not an
activation result.

## Whole-game learned response and postflop resolver

Four independent 5,000-training/10,000-evaluation-deal runs use four common-
random-number rollouts per action and a minimum of two range particles. The
response freezes one action per observable information set, retains separate
preflop and range-conditioned postflop artifacts, and falls back to baseline
play unless the chosen action beats the runner-up with positive approximate
99.5% gap confidence.

| Routed model | Point exploitability lower bound | Approx. 99% lower bound |
| --- | ---: | ---: |
| v26 seed 5101 | 0.0030 | 0 |
| v26 seed 5102 | 0.0075 | 0 |
| v27 seed 7601 | 0.0058 | 0 |
| v27 seed 7602 | 0.0119 | 0 |

Across responders, the evaluator learns 418–715 information sets but finds
only 45–80 confidence-supported actions. Preflop lookup coverage ranges from
0.84% to 3.54%, and postflop lookup coverage ranges from 0% to 0.10%. The
point estimates therefore reveal no statistically supported exploit, but the
zero confidence lower bounds and sparse coverage do not support equilibrium.
This is a valid learned-response lower-bound search, not an exploitability
upper-bound certificate.

## Release interpretation

Neither cross-seed agreement nor a failed learned response proves equilibrium.
The preflop response is exact only in the sampled preflop game, and the
whole-game learned response is a lower-bound red-team search rather than an
exploitability upper-bound certificate.

The sequence completed all six requested research stages, but v27 is rejected
and not activated. It improves materially on v26 in the dedicated preflop
oracle test and produces stable, compact students, yet remains far above the
0.05bb/hand preflop target. Continuation action-value uncertainty is also too
high, and the full-game learned response is too sparse to close the remaining
postflop gap. Activation remains fail-closed until a future candidate passes
the project-wide exploitability, cross-seed, lookup, probability, action-EV
uncertainty, storage, and browser gates.
