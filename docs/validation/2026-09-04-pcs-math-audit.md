# 20bb public-chance DCFR: mathematical and paid-compute audit

Date: September 4, 2026. Base: `4ef673f`. Solver: `0.3.0`; checkpoint schema: **5**.

Companion: [verified five-page research report](2026-09-04-pcs-research.pdf).

Follow-up: the [local checkpoint and response audit](2026-09-04-local-checkpoint-response-audit.md)
completes bounded local recovery and the tabular full-game response adapter.
Its results supersede this snapshot's "unmeasured" response status; neither
report certifies Approximate GTO or authorizes an unrestricted paid run.

## Decision

Keep the policy-preserving memory/evaluator optimizations. Use opt-in
`--integrate-terminal-actions` for the next **bounded accuracy-focused scaling
pilot**, not an unrestricted production run. Retain
`--opponent-checkdown-baseline` as a lower-memory research comparator. Neither
mode passes the release requirements, and no paid server was provisioned.

The user’s later roughly **0.50bb total full-game exploitability** preference
does not replace the separate 0.10bb root diagnostic threshold. Do not halve
the total gain by averaging the seats without explicitly changing conventions.
The user also allowed experimental release without exploitability certification;
that is not authorization to label an uncertified policy Approximate GTO.

## What changed and why

1. **Scale-invariant regret matching and positive support.** An absolute chip
   epsilon previously converted tiny nonzero regret weights to uniform play and
   excluded positive range/proposal support. For example, `[1e-12, 3e-12, 0]`
   must normalize to `[0.25, 0.75, 0]`, not a uniform distribution. Positive
   finite weights now normalize at their own scale, with overflow-safe
   rescaling. Chip-accounting tolerances remain unchanged. This fixes a real
   mathematical defect; it is not the source of the large policy improvement.
2. **Board-local linear terminal values.** Rank compatible private combos once
   per sampled board. Compute compatible/winning/tied mass using total and
   per-card marginals, with exact same-combo inclusion/exclusion. Evaluate only
   the traverser’s counterfactual values. Sparse/dense ranges, blocked hands,
   folds, ties, and both seats agree with the independent slower reference.
   This follows the structural idea in [Johanson et al., AAMAS 2012, Efficient
   Nash Equilibrium Approximation through Monte Carlo CFR](https://www.johanson.ca/publications/poker/2012-aamas-pcs/2012-aamas-pcs.html).
3. **Share immutable descriptors, not learning state.** Intern entire bucket
   trajectories and action-label slices. Regrets and average-strategy arrays
   stay independent. Rebuild sharing on checkpoint load. Named MessagePack
   serialization remains lossless. Deduplicate bucket-to-key work and strategy
   lookups within each public state.
4. **Bit-parallel card ranking.** Replace rank scans with bit operations; all
   8,192 possible rank masks are exhaustively checked against the old formulas.
5. **Exact terminal opponent integration.** Sum terminal branch CFVs exactly,
   then sample a continuation from the renormalized nonterminal proposal.
   Divide its range by the conditional proposal, not the original one. This
   removes terminal-action sampling noise but visits more continuation states.
6. **Research follow-up: stateless checkdown control variate.** Use legal
   check/call-to-showdown branch baselines on the already sampled board, then
   sample from the original opponent proposal. Terminal selections stop;
   continuation selections add an importance-corrected residual. No persistent
   baseline table, policy replacement, or hidden-information deployment rule is
   added. The identity follows [Davis, Schmid & Bowling, ICML 2020,
   Low-Variance and Zero-Variance Baselines](https://proceedings.mlr.press/v119/davis20a/davis20a.pdf),
   equation 4; this particular stateless range-vector design is our adaptation.

For uncorrected branch contributions `F[a]`, baselines `B[a]`, and original
proposal `q[a]`, the latter estimator is:

`sum(B) + (F_sampled - B[selected]) / q[selected]`.

The recursive child already includes `1/q` in the opponent range, so code adds
`child_cfv - B[selected]/q[selected]`: **never divide the child twice**. Exact
terminal baselines have zero residual. Unit tests enumerate the proposal’s
expectation for every combo, including deliberately wrong baselines. A wrong
baseline can increase variance despite remaining unbiased; no zero-variance
full-game claim is made.

## Measured results

All rows use full default sizing, alternating fixed DCFR, zero averaging delay,
20bb, seeds 26001/26002, 2,000 held-out deals, 256 root samples per class, and
2,000 action-value evaluation deals. Only preflop policy rows are exported;
training itself traverses every street. These are short diagnostics.

| Paired arm | Rounds | Root deviation A / B, bb | Infosets A / B, million | Max action MAE | Primary agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original PCS | 400 | 0.650 / 0.675 | 6.60 / 6.39 | 10.99% | 50.89% |
| Optimized ordinary PCS | 600 | 0.556 / 0.560 | 9.69 / 9.33 | 8.41% | 60.36% |
| Terminal integration | 400 | 0.463 / 0.486 | 10.01 / 9.39 | 8.90% | 61.54% |
| Terminal integration | 800 | 0.407 / 0.384 | 19.06 / 18.87 | 8.28% | 64.50% |
| Checkdown baseline | 400 | 0.587 / 0.567 | 6.96 / 6.58 | 11.01% | 52.66% |
| Checkdown baseline | 800 | 0.482 / 0.481 | 13.01 / 12.32 | 8.79% | 57.99% |

At equal 400-round budgets, integration reduces the two root estimates by
**28.7% / 28.0%**. Compared with the more expensive 600-round ordinary control,
its root estimates remain **16.6% / 13.2% lower**; not all stability metrics
win. The 600-round control’s aggregate action delta is 2.80%, versus 5.63%
for integration at 400. At 800 the checkdown alternative uses **31.8% / 34.7%
fewer states** than integration, but worse root values and primary agreement;
its aggregate delta is better (2.54% versus 5.27%).

Policy-preserving optimization alone reproduces seed 26001’s 400-round root
strategy exactly after lookup/card-evaluator changes, with elapsed time falling
from **47.82 to 35.96 seconds** and peak resident memory from **2.876 to
2.190GB**. Numeric fixes affect seed 26002 slightly; do not claim every
pre-fix artifact is bit-identical. The separate sharing/terminal-kernel pair
reduced RSS about 24–25% before the final evaluator speed improvement.

Apple M5, 16GiB, Rust 1.97.1; one training seed at a time. The 600-round control
took 49.71/48.01 user-CPU seconds. Integration at 400 took 57.13 user-CPU
seconds for seed A before the final bit-evaluator optimization. Some larger
runs overlapped compilation or experienced kernel/memory-compression overhead,
so these are **approximately comparable CPU budgets, not a controlled equal-wall
trial or cloud runtime prediction**. See the [machine-readable audit](2026-09-04-pcs-math-audit.json)
for every retained summary, comparison, binary/artifact digest, and timing.

## Remaining metrics: selected integration-800 pair

| Metric | Result | Existing target / interpretation |
| --- | --- | --- |
| Full-game exploitability, this exact policy | Unmeasured | User preference roughly 0.50bb total; not certified |
| Root local deviation | 0.407 / 0.384bb | <=0.10bb; fails |
| Max combo-weighted action MAE | 8.28% | <=5%; fails |
| Primary action agreement | 64.50% | >=85%; fails |
| Aggregate action delta | 5.27% | <=3%; fails |
| Median / p95 / max hand TV | 22.90 / 40.76 / 57.84% | <=20 / 35 / 65%; first two fail |
| Held-out unknown lookups | 11.51 / 11.68% | <=5% diagnostic; fails |
| Held-out untrained lookups | 1.87 / 1.84% | <=2.5%; passes this diagnostic |
| Forced-root continuation unknown / untrained | 16.84 / 16.81%; 2.80 / 2.86% | <=5% / <=2.5%; fail |
| Root trained combo coverage | 100% to roundoff | Passes root coverage only |
| Action-EV SE <=0.02bb, reach-weighted | 23.41 / 21.53% | >=95%; fails |
| Authentic + forced-deviation serving lookup coverage | Not release-audited | >=99.99%; cannot infer from root coverage |
| Quantized full-policy size and probabilities | Not exported/audited | <=20GB; not established by these small exports |

The full-game responder in `blueprint/response.rs` currently consumes a frozen
**neural** policy. Its historical scores cannot be assigned to these tabular
checkpoints. Root evaluation uses the same average-policy rollout algorithm
for all estimator modes; terminal integration changes training, not evaluation.
The root evaluator selects and reports the best action on the same samples,
which creates winner’s bias. Its nominal lower confidence field is descriptive,
not a selection-corrected bound. No existing threshold was relaxed to hide this.

## Why more iterations alone are not a sufficient plan

Regret minimization converges in the represented game; lossy buckets and missing
bet sizes can still hide exploitable full-game choices. Stronger abstract-game
convergence can even worsen full-game performance: see [Johanson et al.,
AAAI 2012, Finding Optimal Abstract Strategies](https://johanson.ca/publications/poker/2012-aaai-cfr-br/2012-aaai-cfr-br.pdf),
Figure 1. The current abstraction’s actual error floor is **unknown**, not
proven insurmountable.

A legal LBR’s expected winnings lower-bound best-response value. An upper
confidence interval on that responder’s winnings is **not** an upper bound on
exploitability. See [Lišý & Bowling, Equilibrium Approximation Quality of Current
No-Limit Poker Bots, 2016/AAAI 2017](https://arxiv.org/pdf/1612.07547).

More training does not automatically increase the separate fixed-budget
action-EV sample count. Sparse nodes can remain imprecise even when their
policy improves. Asymptotic Monte Carlo error typically falls only with the
square root of sample count under ordinary finite-variance assumptions;
neither doubling rounds nor reducing the mean EV error proves 95% coverage.

## Next decision, without another architecture detour

1. Continue the selected estimator to a bounded 1,600-round pair on a 128GB
   host, subject to confirmed pricing, authorization, resource preflight,
   checkpoint recovery, and an independent server-deletion mechanism.
2. Before a large multi-hour continuation, adapt the existing legal full-hand
   responder to the frozen tabular average policy/checkpoint. Separate action
   selection and evaluation deals. Test all streets, exact compatible opponent
   ranges, and complete terminal returns. Report gains as lower bounds.
3. If coverage/stability improves but responder gains do not, use the costly
   states to pilot CFR-BR-inspired exact-card turn/river responses and targeted
   bucket refinement. Do not blindly enlarge every bucket or add rounds.
4. Only if table growth remains limiting, generate exact turn/river targets for
   a small range-conditioned continuation model; validate resulting actions on
   perturbed as well as reached ranges. Low global RMSE is insufficient.

ReBeL/DeepStack motivate range-conditioned values, not a cheap wholesale
rewrite. [ReBeL, Brown et al., NeurIPS 2020](https://arxiv.org/pdf/2007.13544)
reports 90 eight-GPU DGX-1 machines for full-game data generation.
[DREAM, Steinberger et al., 2020](https://arxiv.org/pdf/2006.10410) permits
unbiased baselines but retains a regret-error term from imperfect advantage
fitting. [Brown & Sandholm, ICML 2017, Reduced Space and Faster Convergence
via Pruning](https://proceedings.mlr.press/v70/brown17a/brown17a.pdf) still
requires full early-game storage absent a warm start; it does not justify
deleting noisy negative sampled-DCFR regrets here.

## Reproduction and verification

For each seed `S` in `26001,26002`, use:

```bash
cargo build --release --manifest-path preflop-solver/Cargo.toml
preflop-solver/target/release/preflop-solver blueprint \
  --effective-stack-bb 20 --iterations 800 --max-information-sets 26000000 \
  --seed 26001 --averaging-delay 0 --public-chance-sampling \
  --integrate-terminal-actions --held-out-deals 2000 --held-out-seed 126001 \
  --root-deviation-samples 256 --root-deviation-seed 226001 \
  --action-value-deals 2000 --action-value-seed 326001 \
  --output /tmp/pcs-audit-seed26001.json.gz \
  --summary /tmp/pcs-audit-seed26001.summary.json
```

Adjust all four seeds together for seed B. Replace integration with
`--opponent-checkdown-baseline` for that comparator, or omit both for ordinary
PCS. Artifact prose gained an explicit selection-bias warning after the pilots;
new envelope hashes therefore need not equal the measured binary’s envelope.
Policy math is unchanged by that warning.

Rust release tests cover exact terminal/reference equality, numerical scale,
exhaustive rank masks, estimator expectation, checkdown accounting, shared
storage independence, all three estimator modes’ deterministic checkpoint
resume, and old-schema/mode-change rejection. Cloud-runner tests pin both
research flags in CLI commands, fingerprints, summaries, and resume lineage.
No browser/API behavior or serving model was changed.

Final checks: **189 Rust library tests + 3 CLI integration tests passed**;
**15 cloud-runner unit tests passed**; `cargo fmt --check`, release build, and
`git diff --check` passed. A real local runner pair completed four rounds with
full postflop export, then resumed to eight. A separate fresh eight-round pair
produced byte-identical gzip artifacts for both seeds:

- 26001: `49739b5e5eef32713eba14e55e4329cb354320f0b13df72a53877726451b4b8c`
- 26002: `029d576724209dcac2321d907ed64709ba91e470fe6bee8ba4a723db7517279b`

Those tiny runs verify orchestration, immutable flags, exact resume, and export
integrity only; they add no meaningful policy-quality evidence. The PDF's five
pages were rendered and visually inspected, with text bounds and links checked.
