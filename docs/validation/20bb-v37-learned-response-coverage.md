# 20bb v37 learned-response coverage audit

Status: evaluator coverage improved, response selection rejected, no model was
activated.

This sequence tested whether the full-game learned-response lower-bound audit
could generalize beyond exact/fine/coarse hash lookups without observing hidden
information. It also tested the learned response at a larger independent deal
budget. The result isolates action-selection overfit as the remaining blocker.

## Strategic observable backoff

The new fourth lookup layer retains only information available to the
responder:

- street and acting player;
- the responder's current made-hand category;
- categorical pot-odds and SPR bands; and
- normalized legal-action shape such as `fold`, `call`, `raise_1`, and
  `all_in`.

It deliberately forgets public-board texture, draw flags, prior public
trajectory, and absolute bet amounts. It never uses the opponent's cards.
Exact, fine, and coarse decisions remain ahead of it in the lookup hierarchy.
Regression tests verify hidden-card invariance, public-texture generalization,
and equivalent action shapes at different absolute amounts.

On the matched v27 seed-7602 pilot with 2,000 training deals, 2,000 independent
evaluation deals, four rollouts per action, a two-particle minimum, and root
seed 992701:

| Metric | Fine + coarse | + strategic | Relative change |
| --- | ---: | ---: | ---: |
| Player 0 total lookup | 1.89% | 3.24% | 1.71x |
| Player 1 total lookup | 2.31% | 3.66% | 1.58x |
| Player 0 postflop lookup | 2.47% | 6.30% | 2.55x |
| Player 1 postflop lookup | 2.09% | 4.12% | 1.97x |

The strategic postflop layer contained only 96 and 87 learned buckets and
averaged approximately 11 and 15 training particles per bucket. This justified
a larger audit; it did not establish strategy quality.

## Scaled audit and one-step correction

At 10,000 training and 20,000 independent evaluation deals with root seed
992702, total lookup reached 4.63% and 9.71%, while postflop lookup reached
9.01% and 13.22%. Both fixed learned responses nevertheless regressed against
the frozen policy:

| Responder | Estimated gain | Standard error | One-sided 99.5% lower bound |
| --- | ---: | ---: | ---: |
| Player 0 | -0.016054bb | 0.018464bb | -0.063613bb |
| Player 1 | -0.029392bb | 0.022481bb | -0.087298bb |

The evaluator method said `one_step`, but the implementation previously used
every matching learned action in a hand. It now takes at most the first
confident learned action and then returns to the frozen policy, matching the
baseline-continuation rollouts used for training. The matched scaled rerun was
effectively unchanged (`-0.016054bb`, `-0.029617bb`), so compounding was not the
primary failure.

## Granularity ablation

The evaluator now accepts an explicit maximum response granularity. A matched
2,000/2,000 ablation on root seed 992701 measured:

| Maximum layer | Player 0 gain | Player 1 gain | Lower-bound point estimate |
| --- | ---: | ---: | ---: |
| Exact | +0.038750bb | +0.010000bb | 0.024375bb/hand |
| Fine | +0.039750bb | -0.017000bb | 0.019875bb/hand |
| Coarse | +0.056250bb | -0.027584bb | 0.028125bb/hand |
| Strategic | +0.082750bb | -0.022500bb | 0.041375bb/hand |

Every 99% combined lower confidence bound remained zero. The small exact-only
positive sign was frozen before a fresh larger audit. On root seed 992703 with
10,000/20,000 deals it failed to reproduce:

| Responder | Estimated gain | Standard error | Total lookup |
| --- | ---: | ---: | ---: |
| Player 0 | -0.019742bb | 0.007872bb | 0.68% |
| Player 1 | -0.014796bb | 0.009922bb | 1.50% |

The thousands of candidate information sets are selected from noisy action
rollouts using per-decision confidence tests. This creates a multiple-selection
problem: even nominally confident action gaps can regress on independent
deals. More aggressive backoff increases coverage but also increases
distribution mismatch.

## Decision

Exact-only is now the CLI default. Fine, coarse, and strategic layers remain
available only as explicit research ablations. Negative learned-response gains
are clamped to zero only when composing the exploitability lower-bound point
estimate; they are retained verbatim in the report and are never treated as
GTO evidence.

The evaluator now adds a third, disjoint calibration phase between response
training and final evaluation. A player's whole frozen response is deployed
only when its calibration gain has a strictly positive finite one-sided 99.5%
lower bound. Otherwise final evaluation uses the frozen baseline policy, keeps
the failed calibration evidence, and claims exactly zero gain.

A fresh 2,000-training/2,000-calibration/2,000-evaluation exact-only smoke on
root seed 992704 rejected both responders. Calibration gains were
`-0.002750bb` and `-0.061167bb`, with lower bounds `-0.054540bb` and
`-0.153001bb`. Final evaluation deployed neither response and reproduced
exactly zero gain and standard error for both players. This validates the
fail-closed split; it does not make the attack strong enough.

The next learned-response design should confirm individual candidate
deviations with multiple-selection control before composing a whole response.
Until then:

- learned-response coverage/confidence remains a blocker;
- the lower-bound audit has not found a validated exploit;
- failure to find an exploit does not upper-bound exploitability; and
- the historical clairvoyant approximately 5.29bb result is diagnostic only;
  the v45 audit found that its pre-river all-in settlement was not suitable for
  a rigorous chance-sampling certificate, and the corrected successor still
  fails the 0.10bb release gate.

No active manifest or frozen research policy was changed.
