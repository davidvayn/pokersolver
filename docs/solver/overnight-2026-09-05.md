# September 5 local policy-improvement sequence

User request: work overnight on remaining weaknesses; use short pilots and
research before long training; commit/push major milestones. First target is
all existing quality metrics plus full-game exploitability below 0.50bb/hand,
then continue toward 0.05bb/hand. Do not turn a restricted-response lower bound,
rejected-response zero, or fixed-opponent payoff gain into an upper certificate.
The expected user return is approximately 18:30 UTC. The local 16GiB machine
remains the only authorized compute; no deployment or paid resources.

## Existing quality gaps (not additional gates)

The retained 800-round run manifest already records failing preflop root
stability: maximum combo-weighted per-action MAE 8.2816 percentage points
(target 5), maximum aggregate action-frequency delta 5.2719 points (target 3),
minimum primary-action agreement 64.497% unweighted / 66.817% combo-weighted
(target 85%). Root coverage is complete for all 169 classes per seed, legal
action sets match, and maximum probability-sum error is 3.33e-16 (passing).
The routed candidate has the same preflop action frequencies, so routing the
later streets has not fixed these stability gaps.

The source table's existing local action-value evaluation reports only 21.530%
minimum standard-error coverage, versus the intended 95% precision target.
That is a source-table diagnostic, not a new precision measurement of the
routed full-hand candidate. Likewise its mean root-local-deviation gain of
0.395723bb is not total full-game exploitability. Missing flop rows, uncertainty,
preflop stability, and full-game qualification remain unresolved; do not
present a restricted-response zero as all metrics passing.

## Preserved milestone

`e25ec44` was committed and pushed to `origin/main`. It contains the bounded
checkpoint/response tooling, paired-estimator corrections, connected turn/river
pilots, and the confirmed terminal-flop payoff improvement. See
[the preceding results](tabular-turn-pilot.md#continuing-policy-action-work-flop-corrections).
Unrelated `public/reports/`, `scripts/solver-review.test.mjs`, and the loose
`blueprint-artifact.json` were excluded and left untouched.

## Pilot 1: suit-consistent buckets

The card-feature estimator seeded its sampling from literal suit labels.
`cargo test --release rollout_buckets_do_not_split_suit_isomorphic_observations`
reproduced a split between equity buckets 3 and 2 after a pure suit relabeling.
The minimized observation has hero cards `[31,14]` and turn board
`[11,40,45,44]`. No opponent cards or unrevealed board are needed to reproduce
the classification issue. Suit symmetry is strategically irrelevant; see
[Waugh's primary hand-isomorphism paper](https://www.cs.cmu.edu/~waugh/publications/isomorphism13.pdf).
We use the existing visible-card suit-signature map, not a new implementation
of the paper's complete indexing algorithm.

Implemented opt-in `--canonical-suit-buckets`: canonicalize the known cards
and sampling deck, retaining exact card removal, betting abstraction, and
trajectory recall. Its identity is serialized and fingerprinted; a legacy
checkpoint cannot be resumed with the changed abstraction. The default stays
legacy-compatible. All 24 suit permutations are tested across all streets.

A second failing regression proved the per-deal bucket cache ignored
abstraction settings. Its key now includes those settings. The legacy-mode
eight-round seeded artifact is byte-identical across the old/new executable:
SHA-256 `93c85c4c86ca1b4a5b9203fba1d0bb3099888b785cacb00407ba2c9e900cead6`.
This verifies a small replay, not an unperformed full-size replay.

Both 400-round pairs completed: seeds 27001/27002, fixed DCFR, zero averaging
delay, public-chance sampling, terminal-action integration, full default action
grid. Held-out / root-deviation-per-class / action-value budgets: 4,000 / 64 /
1,000. Workers ran sequentially, with a 6GiB sampled footprint stop, 20-minute
time stop, and 20GiB disk reserve. No stop fired.

| Seed | Mode | Nodes | Held-out unknown | Held-out untrained | Root local gain (SE), bb |
| --- | --- | ---: | ---: | ---: | --- |
| 27001 | Legacy control | 9,064,421 | 11.410% | 1.508% | 0.74516 (0.05233) |
| 27001 | Suit canonical | 8,319,321 | 12.705% | 1.578% | 0.76742 (0.05213) |
| 27002 | Legacy control | 9,847,604 | 11.758% | 1.830% | 0.82207 (0.05378) |
| 27002 | Suit canonical | 8,533,620 | 12.861% | 1.806% | 0.74719 (0.05130) |

These held-out trajectories follow each policy; they are not a matched fixed
trajectory coverage experiment. Root local gains are not full-game
exploitability. Node counts decrease 8.2% / 13.3%, but the policy-quality
screen is mixed. **No longer run was triggered and no canonical policy was
selected as a quality winner.** Keep this as a tested optional abstraction,
not a reason to discard the preceding 800-round candidate.

Local records:
`preflop-solver/neural/runs/local-suit-20260905-control400/run-manifest.json`
and `local-suit-20260905-canonical400/run-manifest.json`.
Control runtime: 244.161 / 276.312 seconds; sampled footprint:
2,968,979,952 / 3,216,263,808 bytes. Canonical: 270.272 / 286.356 seconds;
2,725,939,480 / 2,793,441,608 bytes. Concurrent builds preclude isolated
runtime-speedup claims. Canonical binary:
`11a627aab735fcf2267cce725308c1747244fa181d938a727e00dbc6c65a6b2f`.

## Pilot 2: compare response actions to the actual baseline

The old learner required a confidently unique best action. That is a different
question from whether an action beats the profile being attacked. A regression
reproduced rejection when two actions tie for best but the baseline mixes in
losing actions. Conversely, a coarsened response can have a clearly best average
action while losing to a baseline that distinguishes the underlying situations.

The learner now records paired `Q(action) - sum_a policy(a) Q(a)` observations,
using the actual profile at each sampled state. Welford moments preserve the
common-random covariance. New response admission uses the selected action's
positive one-sided normal-approximation 99.5% advantage lower bound. The old
runner-up margins and `low_confidence` rank flags remain visible; legacy
reports without the new `response_advantage` data retain their old meaning.
The new method string distinguishes the estimator/admission version.

This changes neither the independent positive 99.5% calibration requirement
nor the independent holdout, and it does not edit the defender's policy.
Exploitability certification is still unresolved. A stronger attack may expose
larger leaks; that is useful diagnostic evidence, not a model regression.
The [primary LBR paper](https://arxiv.org/html/1612.07547v1) likewise treats
approximate responses as lower-bound evidence, not equilibrium certification.

The new regressions and full suite pass: 217 Rust library tests, 6 CLI tests,
and 32 Python runner/resource tests. The preserved milestone's remote CI also
passed (GitHub Actions run 33955548073).

`preflop-solver/neural/runs/local-advantage-20260905-pair1/cohort.json` is
**complete**, using the retained 800-round seeds 26001/26002, joint turn/river
four iterations, and the 25% terminal-flop correction with 2,048 equity samples.
Response budgets per seat: 512 training / 2,000 calibration / 2,000 independent
holdout hands, four action rollouts, minimum four particles, offset 2000000.
Two shared-table workers; sequential seeds; 45-minute / 7.5GiB sampled-footprint
stop per seed and 20GiB disk reserve. The runner froze the executable and pins
source/output hashes. Neither seed hit a resource stop. All four responses
were rejected by calibration; reported deployed zeros remain inconclusive.

| Seed | Responder | Calibration gain, bb | SE | One-sided 99.5% lower bound |
| --- | --- | ---: | ---: | ---: |
| 26001 | BTN/SB | 0.0764165 | 0.0743797 | -0.1151730 |
| 26001 | BB | -0.0091250 | 0.0553080 | -0.1515889 |
| 26002 | BTN/SB | 0.1360420 | 0.0758490 | -0.0593322 |
| 26002 | BB | -0.0248330 | 0.0523568 | -0.1596953 |

On seed A's saved flop rows, seven actions pass the new local advantage test
versus four under the old runner-up test. This did not establish a profitable
full-hand response. Do not trigger a longer identical run from these results.
Held-out source flop unknown/untrained fractions are 30.933% / 8.976% for A
and 26.576% / 9.014% for B. Preflop source lookup is complete; zero source
lookups on routed turn/river are not an independent coverage certificate.

Runtime: 773.911 / 1,222.049 seconds. Sampled physical footprint:
6,693,573,744 / 6,632,494,144 bytes. Concurrent builds/tests affect runtime.
Frozen executable SHA-256:
`606e6bb50d5f77a8286a2bf47c560127cccb08c6ca4e3d4474c24c9f6148a9f3`.
Output A: `f0749d040873ac658450b9403e87c66bf1afa2b9bb243910dbdf3307db7dad37`;
B: `8ad9c8ccf3dce635d122a6bdb17db2018452db39af13c954f2a872c6006feed5`.
Output hashes were independently rechecked. Commit `e123bad` is pushed and
its remote CI passed (run 33957223213).

## Pilot 3: frozen average-mass completion for missing flop rows

Implemented opt-in `--flop-backoff-minimum-visits N` and
`--flop-backoff-weight W`. Pool trained flop average-policy accumulators by
current hand bucket, board bucket, actor, and exact public betting history;
forget only the private preflop bucket for this experimental completion.
Keep the original DCFR average mass weights, not equal weights per row.
Exact trained entries retain precedence. Insufficient support retains the
explicit baseline completion. Borrowed matches are counted separately and
never relabeled as newly trained exact coverage. The support count is averaging
contributions, not independent effective samples or an EV-confidence bound.

This is a generalization hypothesis, not a new perfect-recall solve or a
safe-resolving theorem. [Waugh et al.'s primary imperfect-recall study](https://www.cs.cmu.edu/~waugh/publications/sara09.pdf)
motivates empirical testing of reduced private-history dependence, while
explicitly warning that the standard theoretical guarantees do not transfer.
We do not change the original training abstraction or its checkpoint.

`preflop-solver/neural/runs/local-pooling-20260905-pair1/cohort.json` is
**complete**. Minimum eight averaging contributions, full pooled weight only
at missing/untrained supported flop rows; joint-four turn/river and the
25% terminal-flop correction are preserved. Two retained opponents per seed:
the earlier baseline attack and Pilot 2's raw diagnostic attack. Each opponent
keeps its original continuation and calibration status. Use 1,024 fresh hands
per seat/opponent, seed offset 2200000, sequential seeds, two shared-table
workers, 45-minute / 7.5GiB footprint stop and 20GiB disk reserve. This uses
paired realized rollouts, not the terminal-only exact estimator, because the
candidate can change nonterminal flop decisions. No second full-size table
may run alongside another full-size worker.

The policy screen was mixed: four positive and four negative point estimates;
every individual 99% interval crosses zero. Only 2–6 of 1,024 hand payoffs
changed per comparison. **Do not select the pooled policy or extend this arm.**
The unchanged terminal-corrected candidate remains the retained policy.

| Seed | Older opponent BB/SB defender gains, bb/hand | New raw opponent BB/SB defender gains |
| --- | --- | --- |
| 26001 | -0.015625 / +0.012939 | -0.018555 / +0.064290 |
| 26002 | +0.006673 / -0.054851 | +0.024902 / -0.023926 |

The labels denote defender seats, in report responder order 0 then 1. The
underlying report preserves every comparison's standard error, interval, and
opponent calibration status. Seed A had 154,046 pooled rows with the required
support. Pooled matches accounted for 182/415 eligible source-missing queries
on A and 175/399 on B; matching a borrowed row did not establish useful policy
improvement. Runs took 197.862 / 245.291 seconds, peaking at 7,100,568,976 /
7,135,237,520 sampled physical-footprint bytes without resource stops.
Frozen binary:
`4aa263d088218d50a95edc96ee9fc9a3a1bc4016c09b104f64cc4584d7f6839c`.
Output A: `d8cf739c4ec67d49456857cb053255a919792e23794dd3bc8349ee37ac279606`;
B: `5ab6da8e798d69a1d7d28f578e56f40827ae914c6fc7b2c8c244c26804618bea`.

## Next attack pilot: remove redundant terminal-runout noise

Inspection showed that four action rollouts after a terminal flop call repeat
the same dealt runout. They do not add four independent boards. New optional
`--response-terminal-expectations` computes offline terminal action labels over
all 990 legal flop runouts or 44 turn rivers. Nonterminal actions and preflop
all-ins retain the existing rollout path. This averages chance, not hidden
opponent information: the learner still aggregates hidden hands at observable
response keys; the defender never sees the offline evaluator's cards.
Calibration and independent evaluation are unchanged. The report explicitly
records the changed training-label method. This is conditional-expectation
variance reduction, not a full AIVAT implementation or equilibrium proof.

A test demonstrates that the old four-rollout call label is -20bb on one
sampled runout and +20bb on another; the new exact label is identical across
those future-card replacements. Other tests check exact payout means, zero sum,
legal card removal, trained-row precedence, terminal-correction preservation,
and deterministic shared-worker completion counts. All 222 Rust release
library tests, 6 CLI tests, and 32 Python resource/runner tests pass; release
build and whitespace checks pass. Neither new option is a website activation.
These changes were committed/pushed as `45424d7`, whose CI passed (33958864025).

`preflop-solver/neural/runs/local-exact-terminal-20260905-pair1/cohort.json`
is **complete**, against the retained non-pooled 800-round profile. It used
Pilot 2's identical 512/2,000/2,000 sample budgets, rollout/particle settings,
and offset 2000000, but exact terminal training values and four shared-table
workers. The frozen binary is `4aa263d...` above. This intentionally reuses the
pilot's random corpora for a controlled training-label comparison; it is not
an untouched final qualification set. Normal independent phase domains and
calibration acceptance still apply within the experiment. Both seeds are
sequential with 45-minute / 7.5GiB footprint stops and a 20GiB disk reserve.
No stop fired. All four responses again failed calibration:

| Seed | Responder | Calibration gain, bb | SE | One-sided 99.5% lower bound |
| --- | --- | ---: | ---: | ---: |
| 26001 | BTN/SB | 0.0372915 | 0.0588113 | -0.1141963 |
| 26001 | BB | 0.0542500 | 0.0355190 | -0.0372408 |
| 26002 | BTN/SB | 0.1360420 | 0.0758490 | -0.0593322 |
| 26002 | BB | -0.0095830 | 0.0653927 | -0.1780235 |

Exact terminal labels remove a demonstrated source of terminal-label noise,
but this pair does not demonstrate stronger overall attacks or a better
defender. It does not justify a longer identical run. Runtimes were 582.207 /
590.919 seconds; sampled peak footprints 7,399,298,640 / 7,542,953,648 bytes.
Four shared-table workers therefore completed these full-size pilots within
the existing 7.5GiB sampled stop, not a guarantee about future peaks.
Output A: `a689feb7cf4380261ada92ef2130f6b5b07d5690088263dc517df6d1b68140a9`;
B: `e7dd9f33a6faf3f730af54270a169ccd3f0a2ff7e4d23a702b11ea69aafc6de4`.
Both this pair's and the pooling pair's output hashes were independently
rechecked against their manifests.

A subsequent postflop-only training option is implemented and verified. It keeps authentic
preflop sampling but skips expensive preflop counterfactual-label generation,
allocating the training budget to the remaining streets. In Pilot 2, A's
postflop response lookup coverage was only 16.081% / 9.659% by seat. BB had
no accepted preflop rows at all, despite spending computation generating them.
This motivates a more focused critic, not a claim that preflop explains every
calibration failure or that a postflop-only attack certifies full-game play.
The regression confirms unchanged authentic trajectories and identical
postflop labels versus the full training pass. All 223 Rust library tests,
6 CLI tests, and 32 Python runner/resource tests pass; release build passes.

## Completed focused postflop and density pairs

`preflop-solver/neural/runs/local-postflop-response-20260905-pair1/cohort.json`
is **complete**. It retains joint-four turn/river, 25% terminal correction, and
the non-pooled 800-round source. New offset 2400000; per-seat budgets are
4,000 training / 4,000 calibration / 4,000 evaluation hands, four rollouts,
minimum four particles, exact terminal labels, and `--postflop-response-only`.
Four shared-table workers, sequential seeds, 45-minute / 7.5GiB sampled stop,
20GiB disk reserve. Frozen binary:
`96bf53f14de9978de117b64bcc4f9c3ed5fc503fd6a8137c41a086c2ebb6a127`.
This reallocates label-generation work toward postflop; it is not a complete
best-response algorithm or a claimed exploitability upper bound.

Both BTN/SB responses passed calibration and earned positive independent
holdout gains. The BB responses failed calibration; their deployed zeros
remain inconclusive, not evidence of zero exploitability.

| Seed | Seat | Calibration gain | Calibration 99.5% lower | Holdout gain | Holdout SE | Holdout 99.5% lower |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 26001 | BTN/SB | 0.18054125 | 0.04434942 | 0.14910475 | 0.05157395 | 0.01625906 |
| 26001 | BB | 0.09758375 | -0.00238066 | rejected | — | — |
| 26002 | BTN/SB | 0.21062475 | 0.08910018 | 0.18147950 | 0.04806679 | 0.05766765 |
| 26002 | BB | 0.15929000 | -0.00489311 | rejected | — | — |

These are restricted-response **lower-bound evidence of postflop leakage**,
not a full-game upper bound or proof that either numerical release target
passes. Accepted-seat postflop lookup coverage on holdout was 35.232% /
19.966%, with 18 / 19 admitted information sets out of 94 / 110 learned.
The stronger critic found a weakness; the defender itself is unchanged.
This pair changed both the training allocation and sample budgets, so it
does not isolate the contribution of each change.

Runtimes: 1,288.697 / 1,557.786 seconds. Sampled peak physical footprints:
7,557,142,192 / 7,571,576,496 bytes; neither resource stop fired.
Verified output hashes:
A `680aa8640087463c117d9f464a3f73e3de6e54ad1316d77d0d2df2c6e5b4d8a0`;
B `2f40036b18d15be1280b59ad4bc007ee704f089d9ea20ce6e7dc11036d4be374`.

The resource-guarded blueprint runner now supports `--potential-bins`, with
default three retaining the legacy fingerprint and command identity. The
setting is pinned in new fingerprints, resume checks, and native summaries.
Tests first failed on all three missing checks, then passed after implementation.
Native CLI tests verify the actual non-default artifact and summary values.

The separate policy-density screen started after the full-size response pair
finished: a fresh 400-round pair, seeds 27001/27002, `--potential-bins 1`,
against the retained 400-round control pair. Keep the full betting grid,
trajectory recall, 10 equity bins, draw features, future category, and legacy
suit mode unchanged. Only the coarse improvement-probability dimension changes.
This does lose some information; it is a hypothesis about the memory/learning
density tradeoff, not a claim that potential information is useless.
[Ganzfried and Sandholm's primary abstraction paper](https://www.cs.cmu.edu/~sandholm/potential-aware_imperfect-recall.aaai14.pdf)
motivates measuring such tradeoffs and warns that finer abstraction does not
monotonically guarantee better full-game play. Do not call this pilot an
implementation of their clustering algorithm. Use the same 4,000 held-out /
64 root-deviation-per-class / 1,000 action-value budgets as the control,
sequential workers, 6GiB / 20-minute stops and 20GiB disk reserve. A longer
stage requires an actual promising policy/resource screen; no such stage is
authorized by a node-count decrease alone. The density pilot is **complete** at
`preflop-solver/neural/runs/local-potential1-20260905-pair400/run-manifest.json`.
It uses the focused response pair's frozen executable (`96bf53f...`), not
the subsequently rebuilt mutable target. Its fingerprint matches the prior
dry run: `1084a493c2423dc83fc05d161d2743b83054dbf3b548442a50d572570ca17e04`.

| Seed | Information sets | Root deviation gain (SE), bb | Held-out unknown | Held-out untrained |
| --- | ---: | ---: | ---: | ---: |
| 27001 | 8,100,626 | 0.8517721 (0.0539517) | 12.0083% | 1.1575% |
| 27002 | 8,757,278 | 0.8216679 (0.0556972) | 11.6714% | 1.7701% |

Compared with the same-seed 400-round control, node counts fall 10.633% /
11.072%, but mean root-deviation gain worsens from 0.7836192 to 0.8367200bb.
Cross-seed per-action MAE improves from 9.7002% to 8.0784%, while aggregate
action delta worsens from 6.8545% to 7.0348% and primary agreement falls
from 50.8876% to 49.1124% (unweighted). These remain failed stability gates.
Minimum source action-EV precision coverage improves only from 18.3825%
to 19.5257%, far below 95%; this is not a routed-profile precision result.
The noisy two-seed screen does not establish a policy improvement, so
**potential-bins=1 is not selected and gets no longer run**.

Runtimes 256.389 / 238.232 seconds; sampled peak footprints 2,650,802,408 /
2,858,502,544 bytes. No resource stop fired. Checkpoint hashes:
A `d9a922c2ad53befe2d6d21e2d44c02bd55312f31857322e220c087e443fb4d48`;
B `adff4649e77077c7c30fbc08af3fa6e0c050856544f2375b96abbadf233032b4`.

## Saved flop correction composition regression

Before testing the accepted response's nonterminal flop actions as a bounded
defender correction, a deterministic 71-hand panel exposed a comparison bug:
even a **zero-weight** saved-action patch could change a payoff from +20bb to
-4bb, because it replaced the control's terminal correction. This was not a
training or RNG change. Pooling mode already preserved that correction;
saved-action mode did not.

The fix composes saved nonterminal corrections with the existing terminal
rule. All-terminal decisions retain that rule, including abstention on noisy
equity; coarse saved decisions cannot override it. The original terminal-only
pilot remains a separate explicit replacement. Reports disclose the inherited
terminal options. This preserves a rule, not a theorem of safe resolving:
earlier changed actions can still change later action-conditioned ranges.
The zero-weight end-to-end regression and direct terminal/nonzero-weight
tests cover this seam. All 224 Rust release library tests, 6 CLI tests, and
32 Python runner/resource tests pass. The release build and whitespace check
pass. No saved-action policy has been selected or activated.

The fix and completed postflop response milestone were pushed as `a865cef`;
CI passed (33962503664).
`preflop-solver/neural/runs/local-saved-flop-20260905-pair1/cohort.json`
was **resource-stopped**: calibrated-seat saved flop decisions, 25% blend, inherited
terminal correction, original 800-round sources, 1,024 fresh hands per seat
per opponent, offset 2600000, four shared-table workers, sequential seeds,
30-minute / 7.5GiB stops. Opponents are the retained `local-turn-20260904-pair2`
baseline and today's focused postflop responses. Binary:
`e58db45601509c3ed6d82d7e33d03f770f61f283cef1e1b270109810a05fb4e8`.
Only BTN/SB proposal seats calibrated, so this pilot does not directly change
BB's action rule or claim to repair the BB leak detected by BTN/SB attacks.
It tests whether those supported BTN/SB actions improve the joint profile's
other seat against existing BB challenges. Raw paired rollouts are used,
not the terminal-only conditional estimator.

Seed A's four-worker run hit the unchanged 7.5GiB sampled memory stop after
144.530 seconds, at 8,060,131,376 bytes (limit 8,053,063,680). Exit -15;
seed B did not start and no policy-quality outcome was selected from this
failed run. A retry is **complete** at
`preflop-solver/neural/runs/local-saved-flop-20260905-pair1-two-workers`.
It uses the failed cohort's identical frozen executable, candidate, opponents,
and chance seeds, but **two** workers. This is a resource-only retry, not a
fresh statistical confirmation. Keep the memory guard unchanged and use
two workers for this full-size multi-policy panel unless measured headroom
supports more.

Two-worker runtimes were 225.597 / 270.861 seconds; sampled peak footprints
6,675,862,736 / 6,666,835,272 bytes. Both completed without stops. Saved
proposal rows were 16 / 17 for BTN/SB and zero for BB. Payoff improvements
(bb/hand), in panel order: old opponent versus BB, old opponent versus BTN/SB,
focused opponent versus BB, focused raw BB opponent versus BTN/SB:

| Seed | Old / BB | Old / BTN-SB | Focused / BB | Focused / BTN-SB |
| --- | ---: | ---: | ---: | ---: |
| 26001 | 0.00976563 | 0.00537109 | 0 | 0.09326172 |
| 26002 | 0.01220703 | -0.01513672 | 0 | -0.01367188 |

Every nonzero comparison's 99% interval includes zero. No payoff changed
against either seed's accepted focused BTN/SB attack. Changes against old
BTN/SB opponents can arise from the changed action-conditioned range rule;
the direct saved decisions are only for BTN/SB. **No saved-action correction
is selected and this pair gets no larger confirmation run.** Verified outputs:
A `0c0aa1922f55317e08af2220450c9f07022cbf8fda667b4b1bef58e7bec8daac`;
B `cab2d50984854fc27d921904c7d42076a98b3bd328ee5ef66ebdcc129eead12f`.

The next separate density hypothesis is the existing compact serving grid,
removing only 4bb/5bb opens while restoring three potential bins. This follows
the original project plan and requires no new solver mechanism. Use the same
400-round paired control budgets before considering any longer training.
[Sandholm's abstraction survey](https://cdn.aaai.org/ojs/9757/9757-13-13285-1-2-20201228.pdf)
explains why smaller abstractions can aid computation but neither finer nor
coarser abstractions guarantee better original-game play. A compact-grid
deviation metric excludes the removed opens and cannot by itself demonstrate
lower exploitability in the wider game. No off-grid action mapping or silent
fallback is added. This pilot is **complete** at
`preflop-solver/neural/runs/local-compact-20260905-pair400/run-manifest.json`,
using frozen binary `96bf53f...`, 6GiB/20-minute stops, and sequential seeds.

Compact-grid results:

| Seed | Information sets | Root deviation gain (SE), bb | Held-out unknown | Held-out untrained |
| --- | ---: | ---: | ---: | ---: |
| 27001 | 8,360,930 | 0.6418617 (0.0486548) | 13.5986% | 1.6727% |
| 27002 | 8,745,571 | 0.7802392 (0.0520759) | 12.2339% | 1.7532% |

Mean restricted-grid root gain is 0.7110505bb, but it excludes the removed
deviations. Cross-seed aggregate action delta is 8.8618%, per-action MAE
12.0859%, and primary agreement 54.4379% (59.1252% combo weighted): all
stability gates still fail. Per-action MAE also averages over fewer actions
than the wide grid, so its absolute change is not an apples-to-apples policy
comparison. Minimum action-EV precision coverage is 21.7575%. The coverage and
stability screen does not justify a larger compact-grid run; **not selected**.
Runtimes 232.163 / 258.446 seconds; sampled peak footprints 2,742,946,096 /
2,865,875,344 bytes; no stops. Checkpoint hashes:
A `76e5bf653174267a61f7f5e59b7683c2c0c6e694adbff6f6f2f930a5e6b2dda7`;
B `02564eb27d39f1e7b7a7afb96aa5c12783fb518b50d568db32eed032140dd5fa`.
Fingerprint: `3b9fd05a2de785fdcb14b2cfd8dd67a42d2a6e170c1b31d05e42864934616a50`.

## Frozen-response recheck without retraining

The focused pair's BB calibration lower bounds were -0.00238 / -0.00489bb
despite positive point estimates. Repeating its expensive training pass would
produce the same frozen response; a bounded new assessment can instead test
whether those unchanged choices generalize. A new native command,
`full-game-response-check`, reuses the original report's exact learned rows
and inherits every profile/training setting. It verifies the checkpoint,
depth, iteration count, response method, actor identities, finite row values,
and source profile. It rejects old seed reuse, chained recheck inputs,
training/profile overrides, and overwriting an output. New calibration and
holdout phase domains remain disjoint. Old payoffs are not pooled in, no
training coverage is fabricated, and report provenance records both the
original report hash/training seed and the new assessment seed. Flop panels
also reject reuse of either seed.

The existing guarded runner accepts `--arms recheck --recheck-responses <csv>`;
input hashes are checked again against returned provenance. This is a compute
reuse feature, **not a stronger response algorithm or policy improvement**.
Because the experiment is chosen after inspecting prior results, it is an
adaptive diagnostic, not a multiple-testing-adjusted release certificate.
Keep all earlier failed calibration results. Do not keep rechecking until a
favorable sample happens to pass.

Verification: all 225 Rust release library tests, 7 CLI tests, and 33 Python
runner/resource tests pass. Tests cover exact learned-row preservation,
profile inheritance, worker determinism, seed and method rejection, checkpoint
mismatch, and ambiguous/chained runner inputs. Planned one-time follow-up:
the original focused responses, **8,000 fresh calibration / 8,000 fresh holdout
hands per seat**, offset 2800000, four shared-table workers, 30-minute/7.5GiB
stops, sequential seeds. It is **complete** at
`preflop-solver/neural/runs/local-postflop-recheck-20260905-pair1/cohort.json`.
No second recheck is scheduled. Release build and whitespace checks also pass.

The recheck implementation and completed density/correction pilots were
committed and pushed as `dcd55f4`; CI passed (33963811748). Active binary:
`991db6cbf5c86c79698b5c8e8ff60920bb72d965f3ba26704c3db929e014fd4f`.
Seed A is complete: both responses passed fresh calibration. Calibration
gains were 0.215530625 / 0.120656125bb, with 99.5% lower bounds
0.115929414 / 0.054096965bb. Independent holdout gains were
0.258166625 (SE 0.040220500; lower 0.154565482) and
0.156583375 (SE 0.029665678; lower 0.080169653) bb/hand.
Their **seat-summed restricted-response gain is 0.41475bb/hand**. This is
stronger evidence of leakage in the unchanged defender, not a policy
regression or an exploitability upper-bound pass. Runtime 606.899 seconds;
sampled peak footprint 7,552,063,152 bytes; no stop. Output A:
`b33d4970543c63b0e4c2c5950d8ec270a2ea12b81c9f2f1b87e0a6c7fec63cce`.
Seed B also completed with both responses passing calibration: gains
0.256062125 / 0.166977625bb, SE 0.036316578 / 0.037707158,
99.5% lower bounds 0.162516819 / 0.069850423. Independent holdout gains:
BTN/SB **0.240406250** (SE 0.035801115; lower 0.148188688) and
BB **0.084977250** (SE 0.038441634; lower **-0.014041838**) bb/hand.
The BB holdout result is therefore inconclusive at this confidence level,
despite passing the separate calibration gate. Seat-summed gain is
**0.3253835bb/hand**. Runtime 912.662 seconds; sampled peak footprint
7,564,236,464 bytes; no stop. Output B:
`d1063cb7cf54cdfd6b517cee3f804dd7a769f851df5bc05dedbdb995192791d0`.
Both output hashes were independently verified. Earlier calibration failures
remain preserved. There is no claim that an exploitability upper bound passes,
nor that the unchanged defender improved. No second recheck is planned.

The next direct policy pilot is the already-supported **50% terminal-flop
range correction versus the retained 25% correction**, leaving every other
setting and the original full open grid unchanged. This directly changes BB
call/fold decisions targeted by the newly accepted BTN/SB attack, unlike the
unselected saved-action patch. Compare against both old and focused retained
opponents with 1,024 fresh hands per seat/opponent, offset 3000000, two
workers, exact terminal-action/runout payoff integration, and unchanged
30-minute/7.5GiB stops. A larger confirmation requires consistent promising
results. The range calculation is still against a frozen profile, not a
minimax safety guarantee. This pilot is **complete** at
`preflop-solver/neural/runs/local-terminal50-20260905-pair1/cohort.json`,
using the recheck cohort's frozen `991db6c...` executable and the rechecked
reports as proposals/focused opponents. Their underlying learned opponents
are unchanged from the original focused reports, not additional diverse
opponents; do not include both copies in a panel.

All eight comparisons improved with individual (not family-adjusted) 99%
intervals above zero:

| Seed | Old / BB | Old / BTN-SB | Focused / BB | Focused / BTN-SB |
| --- | ---: | ---: | ---: | ---: |
| 26001 | 0.08114156 | 0.03789919 | 0.06866995 | 0.07953758 |
| 26002 | 0.09082556 | 0.05555198 | 0.08465559 | 0.05519603 |

These are paired defender payoff improvements in bb/hand, **not measured
full-game exploitability reductions**. Runtimes 162.036 / 187.770 seconds;
sampled peak footprints 6,577,755,128 / 6,728,586,352 bytes; no stops.
Verified outputs:
A `2249daf10c5284724e55500d48450f11e8179ee4f17fe5bb659ec8333a9b7080`;
B `088fe9abb2ae9692c9f1b6e0dcea5d7361e75839bf9432c5b2a1d81e4cbd6cb4`.
The terminal-50 candidate has earned a fresh **4,096-hand per seat/opponent
confirmation**, offset 3200000, otherwise identical settings. This is now
**complete** at `preflop-solver/neural/runs/local-terminal50-20260905-confirm2`,
after the short both-seat saved-action comparison completed. The best
retained native research profile now uses **terminal weight 0.50**, with the
original full-grid 800-round checkpoints and joint-four turn/river solver.
This does not activate a website model or label it Approximate GTO.

Fresh confirmation gains (bb/hand; 4,096 hands per seat/opponent):

| Seed | Old / BB | Old / BTN-SB | Focused / BB | Focused / BTN-SB |
| --- | ---: | ---: | ---: | ---: |
| 26001 | 0.08422201 | 0.05816696 | 0.07594549 | 0.07276139 |
| 26002 | 0.07582034 | 0.05444097 | 0.09253297 | 0.06316029 |

Every individual 99% interval is positive; the smallest lower endpoint is
0.03815890bb. These remain fixed-opponent payoff gains, not an exploitability
upper bound or proof of lower optimal-response exploitability. Runtimes
374.918 / 521.747 seconds; sampled peak footprints 6,901,896,400 /
6,859,068,600 bytes; no stops. Both outputs were hash-verified:
A `192642ae1a9227845886f4db531d6ca122cd1c3deb1c06d964cc6a4e5718507f`;
B `bbfb7fbdc1c11357ccfaf3888aa5bbf5e46861ddbd03bf0838af30fddaeb28c9`.

Now both proposal seats pass calibration, a separate 25% **both-seat** saved
flop pilot is also warranted against the same retained 25% terminal control.
It is a different candidate from the rejected BTN/SB-only patch: BB has 16 /
31 supported flop rows across the seeds, most nonterminal. Use 1,024 fresh
hands per seat/opponent, offset 3100000, raw paired rollout payoffs, two workers,
and unchanged resource stops. Seed B's BB holdout uncertainty stays disclosed.
This comparison is **complete** at
`preflop-solver/neural/runs/local-saved-both-20260905-pair1/cohort.json`, frozen
binary `991db6c...`. Do not combine these two policy changes
or allocate longer confirmation runs before the short pilots justify it.

Both-seat saved-action results, same panel order as above:

| Seed | Old / BB | Old / BTN-SB | Focused / BB | Focused / BTN-SB |
| --- | ---: | ---: | ---: | ---: |
| 26001 | 0.02539063 | 0.00903125 | 0.05257031 | 0.02327441 |
| 26002 | -0.02669238 | -0.02718164 | -0.04736426 | -0.02514648 |

All eight 99% intervals include zero. Opposite directions across seeds do
not justify a larger run; **not selected**, with no new confidence-estimator
or resampling effort scheduled to rescue this result. Runtimes 218.448 /
275.013 seconds; sampled peak footprints 6,782,243,976 / 6,992,172,336 bytes;
no stops. Verified outputs:
A `b3799d7aff552d72b0477469948703505e3daf0d8162139a034855ad738e9e5f`;
B `30d14a781c79c3793f5219f0f4076abb432e34f9f98a76c9ec6c9fe76c6cc9f5`.
The prior benchmark documentation commit `aa11300` also passed CI (33964920732).

Following the successful confirmation, genuinely fresh postflop-only
responses are now training against terminal-50: 4,000 training / 8,000
calibration / 8,000 holdout hands per seat, four rollouts, minimum four
particles, exact terminal labels, offset 3400000, four shared-table workers,
45-minute/7.5GiB stops, sequential seeds. This keeps the focused critic's
training budget instead of weakening it to manufacture lower measured gain.
It is **complete** at
`preflop-solver/neural/runs/local-terminal50-response-20260905-pair1/cohort.json`,
using frozen binary `991db6c...`. Calibration rejection is still inconclusive,
not a low-exploitability win. The later all-street qualification must include
preflop deviations too; this focused critic alone cannot qualify the full game.
No additional
memory rewrite, abstraction variant, new response-key layer, or neural
distillation is currently selected; finish these concrete policy pilots first.

Fresh terminal-50 responses completed with unchanged training/sample budgets:

| Seed / response seat | Calibration gain (SE), bb | Calibration 99.5% lower | Holdout gain (SE), bb | Holdout 99.5% lower |
| --- | ---: | ---: | ---: | ---: |
| 26001 / BTN-SB | 0.19389575 (0.03397279) | 0.10638763 | 0.12553100 (0.03557697) | 0.03389079 |
| 26001 / BB | 0.01659325 (0.02738971) | -0.05395796 | Not deployed | Inconclusive |
| 26002 / BTN-SB | 0.17308325 (0.03485985) | 0.08329023 | 0.12979163 (0.03252198) | 0.04602054 |
| 26002 / BB | 0.11820925 (0.03274050) | 0.03387531 | 0.10661550 (0.03491093) | 0.01669091 |

Seed B's two accepted responses have seat-summed holdout gain **0.236407125
bb/hand**. Seed A's report sums to 0.125531, but its rejected BB response's
zero is not a quality result. Both accepted BTN/SB attacks still show positive
held-out leakage. Freshly trained responses and fresh samples are not a
paired comparison with the earlier terminal-25 benchmark; do not subtract
the headline totals and claim a certified exploitability decrease. No
repeated calibration retry is scheduled. Postflop holdout response coverage:
29.3762% for A's accepted seat, and 27.8395% / 29.5429% for B. These remain
restricted critics, with no full-game upper-bound qualification.

Runtimes 1,667.424 / 2,054.331 seconds; sampled peak footprints
7,596,807,880 / 7,748,900,672 bytes; no resource stops. Both hashes verified:
A `c8079b4ef8d748c07f1306670b65f7de20ccd0d8cd319d065e33f76b452c52d7`;
B `0c7d6c1ad3088ad8075903bef23a58a8c895174d3ca21879d3704439a26fbcb7`.

## Full terminal-weight experiment support

The terminal-only correction now accepts weights through 1.0, while saved
nonterminal corrections retain their 0.50 cap. Defaults remain unchanged at
0.25, and the retained research candidate remains the independently confirmed
0.50 profile. No validation threshold or active website model changes.

This tests a specific remaining policy limitation: a partial blend retains
some baseline fold/call probability even where sampled range equity gives a
confident preference. Terminal decisions cannot precede another action, so
changing only this blend leaves earlier action probabilities and the range
likelihood calculation unchanged. Against an identical frozen opponent,
conditional payoff is affine in the blend weight. A deterministic paired
test verifies that moving from 0.25 to 1.0 gives exactly three times the
per-hand payoff change of moving from 0.25 to 0.50, with unchanged controls
and at least one exercised correction. The test failed under the old weight
limit before the implementation change. Bounds tests reject invalid and
nonfinite probabilities and keep full-weight saved nonterminal patches invalid.

This is **not a minimax guarantee**: equity uses a frozen opponent profile,
and an adapting opponent can change which terminal calls/folds are profitable.
After the terminal-50 response pair finishes, a short 1,024-hand per
seat/opponent comparison may test weight 1.0 against weight 0.50 and the new
attacks. Any larger run or research-candidate selection must follow that
screen, not the fixed-opponent linearity calculation alone. No full-weight
pilot had started when this implementation was committed.

Verification: 226 Rust release library tests, 7 CLI tests, 34 Python
runner/resource tests, release build, and whitespace checks pass. New native
binary SHA-256:
`804d9f83fc7d99ee49f1f90087b8a45ee3f28a31b01dff722324be38c82fff7e`.
The terminal-50 response run completed using its original frozen `991db6c...`
executable and unchanged configuration. Implementation commit `25f00c5`
was pushed and passed CI (33968621983).

The full-weight short comparison is now **complete** at
`preflop-solver/neural/runs/local-terminal100-20260905-pair1/cohort.json`:
weight 1.0 versus the terminal-50 control pinned by the new proposal reports;
1,024 hands per seat/opponent, offset 3600000, 2 workers, 2,048 equity
samples, exact terminal payoff integration, sequential checkpoint seeds,
30-minute/7.5GiB stops. Opponents are the old baseline, rechecked terminal-25
focused responses, and newly trained terminal-50 responses. The uncalibrated
A/BB attacker remains explicitly a raw diagnostic challenge, not a certified
response. No gate is relaxed and no full-weight profile is selected yet.

All twelve comparisons have positive individual 99% intervals:

| Seed | Old / BB | Old / BTN-SB | Terminal-25 / BB | Terminal-25 / BTN-SB | Terminal-50 / BB | Terminal-50 / BTN-SB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 26001 | 0.18945161 | 0.09655398 | 0.08379444 | 0.14159263 | 0.16038898 | 0.13617210 |
| 26002 | 0.15734316 | 0.10453812 | 0.11675794 | 0.11972231 | 0.07066245 | 0.08514126 |

Units are paired defender payoff improvement in bb/hand over terminal-50,
not exploitability reduction. The smallest individual 99% lower endpoint
is 0.01321501bb. These intervals are not family-adjusted. The final A
comparison uses the explicitly uncalibrated raw diagnostic opponent.
Runtimes 207.132 / 293.361 seconds; sampled peak footprints
6,785,045,664 / 6,760,600,712 bytes; no stops. Both hashes verified:
A `f83bec143dc851c9212f4456445c15922d9750c0c6494419fe622f8a93d19f0d`;
B `7dedfda64707e1a7c1929ec252a8f83a1725c4d9b4175869d3e65c5c914157eb`.

The full-weight candidate earned a genuinely fresh **all-street** response
pair, now **complete** at
`preflop-solver/neural/runs/local-terminal100-full-response-20260905-pair1`.
This deliberately includes preflop deviations rather than treating the focused
postflop critic as a full-game qualification. It uses the verified frozen
`804d9f8...` binary, original 800-round checkpoints, joint-four turn/river,
terminal 1.0 / 2,048 samples, 4,000 training / 8,000 calibration / 8,000
holdout hands per seat, four action rollouts, minimum four particles, exact
postflop terminal labels, and fresh offset 3800000. Three shared-table workers
leave more memory headroom than the previous four-worker training pair;
seeds remain sequential. Memory/disk stops remain 7.5GiB / 20GiB. The time
stop is 90 minutes per seed to allow the additional preflop continuation
labels; this is a compute budget change, not a validation-gate change.
Postflop training hands are not replaced by preflop hands: the same authentic
trajectories now also receive preflop action labels. Terminal-50 remains the
previously confirmed fallback; full weight is under fresh attack assessment,
not activated or labeled Approximate GTO. No distillation is running.

Targeted research check while that run trains: the
[primary LBR paper](https://arxiv.org/html/1612.07547v1) shows that greedy
early-street deviations can earn less than delayed attacks, despite offering
more actions. Therefore an all-street learned critic is a different challenge,
not automatically a stronger one; retain focused postflop evidence as well.
Its Bayesian range update supports exact-card terminal decisions, but a
frozen-range response is not a minimax certificate.

Read-only inspection of the old baseline reports found one supported
preflop terminal fold/call row in seed A and nine in seed B, each with just
4–6 training particles under the old noisy estimator. This count includes
both `call` and `call_all_in` labels; repeated abstraction layers are not
independent evidence. These do not justify
extending the terminal patch to preflop yet. No such extension is implemented
or scheduled; inspect the fresh all-street action records first.

The completed terminal-50 response milestone (`873267a`, CI 33969469385)
and full-weight paired pilot (`a107958`, CI 33969879393) were both pushed
and passed remote CI.

All-street seed A is complete. Both responses failed
calibration: gains 0.044375 (SE 0.026450459, 99.5% lower -0.023756868) and
0.0847075 (SE 0.044002075, lower -0.028634335) bb/hand. Neither was deployed
in holdout; their reported zeros are **inconclusive**, not low-exploitability
wins. No recheck is scheduled. Runtime 3,774.244 seconds; sampled footprint
7,210,751,448 bytes; no stop. Verified output A:
`f64b0ef6d21d3439938a6f60ee9dc6cb874987e36c958f5865a15bc843e3d000`.

Seed B is also complete. The SB response passed calibration: gain
0.123833625bb, SE 0.039925169, one-sided 99.5% lower 0.020993204bb. Its
independent holdout gain is **0.16775bb/hand**, SE 0.039577156, lower
0.065806002bb. It uses 77 supported response rows out of 966 learned rows;
holdout response lookup coverage is 6.849% preflop / 31.263% postflop.
The BB response narrowly failed calibration: gain 0.117103125bb,
SE 0.045915691, lower -0.001167859bb. It was not deployed in holdout;
its zero is inconclusive. No recheck or larger identical run is scheduled.
Runtime 4,461.491 seconds; sampled footprint 7,145,674,224 bytes; no stop.
Verified output B:
`00a79d8a3fb97b16e4def837e5846936de8146c3096bd712c4faa6fa9638c33d`.
Neither seed has two accepted responses, so the pair is not eligible for a
complete two-seat response comparison or a below-0.50 exploitability claim.
Do not subtract its lower-looking totals from the earlier focused attacks:
the learned responses, training scope, and random corpora differ.

## Conditional preflop chance averaging

The new all-street records expose a specific label-sampling weakness. For
example, Q9o after limp/jam has four training particles, call value -20bb,
and a selected-gain SE of zero. Its three matching key layers repeat the same
evidence. This does **not** establish that folding Q9o is optimal against the
actual range: the preflop terminal samples can all share losing runouts.

A minimized native regression fixes AA versus KK and the preflop betting
line, changes just one unrevealed board card, and requests 64 action rollouts.
It reproduced call values -20 versus +20bb; the information-set identity and
known settlement controls passed. The cause is that the old preflop Q loop
replays the same predealt board on every rollout. It remains unbiased across
authentic independent deals, but more action rollouts do not average this
source of chance noise. Do not claim a bias correction or hidden-card leak.

Implemented opt-in `--response-preflop-runouts`: sample a fresh uniform legal
board per preflop rollout, conditional on both **offline training** holdings,
and share that board across all candidate actions. A separate seeded chance
domain preserves paired action RNG streams and the original authentic hand
trajectory. All preflop continuations, not just immediate all-ins, use the
paired board samples; postflop labels remain byte-identical. Default behavior
is unchanged. The report flag and method suffix pin the new estimator;
retained-response checks inherit it and reject silent method upgrades.
Postflop-only no-ops and allocations beyond 4,096 runouts are rejected.

Verification: 228 Rust release library tests, 8 CLI tests, 35 Python tests,
release build, and whitespace checks pass. Tests cover legal unique cards,
determinism, the reproduced board-reuse symptom, legacy values, unchanged
authentic postflop records, serial/three-worker parity, provenance inheritance,
and invalid options. A separate two-round/245-node fixture with 32 training
and 16 calibration/holdout hands yields byte-identical old/new **default**
reports: SHA-256
`aec78edf8e4764525d15ed29aae19b2c4d4a739d60ad94948a518a24523fe919`.
The enabled option changes preflop labels while preserving checkpoint identity.
This tiny replay is not a full-size qualification. Temporary comparison
artifacts are retained at `/tmp/poker-preflop-runouts.MUEL5n`.

Verified new binary:
`0c66f00a6ede193a94201f9c7e00a89edef241fedb52418b91c4b37d907ddacd`.
The completed all-street pair remained frozen on `804d9f8...` with this option
off. The implementation was committed/pushed as `14a54fb`; remote CI
33974273771 passed. No large conditional-runout pilot has started. This
change does not edit the defender,
lower a gate, fix all small-sample confidence issues, or establish a stronger
learned response by itself.

Next selected experiment is a short matched control/resampled pair, not
another long qualification run: original 800-round seeds, joint-four,
terminal 1.0 / 2,048 samples, 1,024 training / 2,000 calibration / 2,000
holdout hands per seat, four action rollouts, exact postflop terminal labels,
minimum four particles, three workers, offset 4000000. Both arms use the
same source checkpoints, executable and phase seeds; only conditional
preflop runouts differ. Separate immutable cohorts preserve each result.
Seeds and arms run sequentially with 30-minute / 7.5GiB sampled-footprint
stops and a 20GiB disk reserve. Compare runtime, preflop label uncertainty
and independent response gains; more admitted rows alone is not success.
Small-sample rejected responses remain inconclusive, and this pilot alone
will not qualify or activate the defender.

The control arm is complete at
`preflop-solver/neural/runs/local-preflop-runouts-20260905-control1`.
Seed A completed in 1,123.372 seconds, sampled footprint 7,145,854,400 bytes,
without a resource stop. Both calibration responses were rejected:
SB gain 0.1175835bb, SE 0.072647709, lower -0.069544598bb;
BB gain 0.01925bb, SE 0.013600357, lower -0.015782197bb.
These zeros are inconclusive. Seed B also rejected both responses:
SB gain -0.0459585bb, SE 0.083104502, lower -0.260021510bb;
BB gain 0.028833bb, SE 0.023525942, lower -0.031765811bb.
B runtime 1,305.021 seconds, sampled footprint 7,210,178,032 bytes, no stop.
Both control output hashes were independently verified:
A `198424be03573d4a7b026aaa1170f3747dca36732776da2b76e782dbc10b91e1`;
B `dde06e31df9a892b7a7e445546a87ea2ca134af4709f0db802f2f5d4f6612061`.
The resampled arm is complete at
`preflop-solver/neural/runs/local-preflop-runouts-20260905-resampled1`.
Configuration comparison verifies identical frozen executable SHA, seeds and
budgets: only the preflop-runout flag differs (besides input/output paths).
It uses the control's frozen `0c66f00...` executable, not the newly rebuilt
`target/release` binary. Concurrent native builds mean the control wall times
are not isolated performance benchmarks. Commit `02798b7`, recording the
preceding full-weight assessment, was pushed and passed CI 33982696003.

## Bounded response-worker load balancing

While the control ran, live CPU readings repeatedly moved between roughly
100% and 300% utilization with three configured workers. A production-seam
test held hand zero busy and checked whether an idle worker could process
hand one. It failed in 1.01 seconds on a full 32-hand wave, then failed after
minimization to two hands. The same test passes with dynamic assignment.

The cause is static 16-hand worker chunks: an idle worker cannot take a hand
behind another worker's expensive solve. Replace only chunk assignment with
an atomic next-hand index inside the existing bounded wave (at most
16 * workers hands). Each worker still owns its resolver cache and shares
the immutable table; sampled deals and per-hand seeds remain fixed. Sort
completed hand results by their original index before any accumulation.
No unbounded queue, extra table, new policy, or new dependency is introduced.

The reproduced idle-worker test passes, and all 229 release library tests
pass. The joint turn/river test now includes three workers and still requires
byte-identical policies, decision/coverage counts and protection diagnostics.
Its artificial repeated-board fixture can cause duplicate cache solves under
dynamic assignment: the real solve count remains reported, is bounded in
the test, and is excluded alongside wall time from policy-parity comparison.
This is operational cache work, not a changed coverage or quality gate.
Full `cargo test --release` passes: 229 library tests and 8 CLI tests.
Old/new executable comparison on the two-round/245-node fixture also passes:
32 training / 16 calibration / 16 holdout hands, three workers, conditional
preflop runouts enabled, same seed 452. Five independent new-process reports
are byte-identical to the old frozen executable's report, SHA-256
`183ff8c220714067e8777c49122adcbd9ea393535a37ab9eef4257e685d8576c`.
These small replay artifacts remain at `/tmp/poker-work-queue.EHFOlN`;
the joint-resolver regression separately checks policy/coverage parity.
Verified new executable:
`2f023b53fe61fdcba4bd3efc02389650dcab063a11804300fde7bb637b4be370`.
No full-size training speedup is claimed yet; the running sampling experiment
remains on its frozen scheduler. No debug instrumentation or dependencies
were added, and no model or release gate was changed.
The scheduler milestone was committed/pushed as `a9346f6`; remote CI
33984490992 passed. Its scheduler is deliberately not used by either arm
of the ongoing conditional-runout comparison.

## Conditional-runout pilot: matched results

Resampled seed A completed in 1,042.046 seconds, sampled footprint
7,133,779,344 bytes, no stop. Verified report SHA-256:
`172b5df2eb817bb02d9128abdf1bd4d072c304860b9b1c1fb883231a70c2e0d1`.
Neither A response passed calibration:
SB gain 0.0285835bb, SE 0.060547640, lower -0.127376886bb;
BB gain 0.03475bb, SE 0.025911762, lower -0.031994277bb.
This is not a stronger accepted response or a defender improvement.

Matched-record comparison confirms identical preflop information-set keys,
action labels, sample counts, hand buckets and public histories (461 SB /
72 BB rows, including repeated abstraction layers). All preflop Q rows
changed; both complete postflop resolver records are exactly identical.
Considering only exact-trajectory rows, the median non-fold action standard
error drops 3.320618 -> 2.110785bb for SB and 5.858963 -> 3.112920bb for BB.
These are diagnostic medians over training labels, not reach-weighted served
EV-precision gates. Locally supported exact rows change 11 -> 9 SB and
2 -> 5 BB; higher counts are not themselves a quality win. The comparison
supports variance reduction on this seed, not accurate enough action values
or a longer identical run.
The control-results commit `acef0e5` passed CI 33984986143.

Resampled seed B also completed without a stop: 1,116.871 seconds,
7,147,607,464 bytes sampled footprint. Verified report SHA-256:
`5510e2261fafab928460d5b48ca704a66a4e07ff8a1b703c6b3315a9b2d482d6`.
Both responses were rejected: SB gain -0.0009165bb, SE 0.073673915,
lower -0.190687929bb; BB gain 0.019333bb, SE 0.021526814,
lower -0.036116398bb. No accepted holdout gain is established in either seed.

B preserves the same 498 SB / 45 BB preflop keys and sample counts; Q values
change in 497 / 45 rows. Both postflop resolver records remain exactly equal
to control. Among 123 / 8 exact rows, median non-fold label SE changes
3.727280 -> 2.250386bb for SB, but **2.464595 -> 2.946186bb for BB**.
Locally supported exact rows change 18 -> 9 SB and 3 -> 0 BB. Sparse-node
sample SE is not a ground-truth accuracy measurement, and fewer locally
supported actions need not be worse if old evidence was overconfident.

Decision: retain the opt-in sampler and its correctness regressions, but do
not claim a stronger response or defender. SB variance improvement repeats;
BB is mixed. No larger identical response run or calibration recheck follows.
This completed experiment does not change any release metric or active model.

## Next bounded policy-training comparison

Targeted source review revisited the existing
[checkdown-baseline results](../validation/2026-09-04-pcs-math-audit.md)
and the primary [Davis et al. baseline framework](https://proceedings.mlr.press/v119/davis20a.html).
The framework does not give our stateless checkdown approximation its
predictive-baseline zero-variance guarantee. At 800 rounds this implementation
used 32–35% fewer states but had weaker root diagnostics; additional rounds
within similar storage are a hypothesis to test, not a promised win.

Selected short test, now complete at
`preflop-solver/neural/runs/local-checkdown-20260905-pair600`:
fresh training with the existing checkdown-baseline option, 600 rounds per
seed 26001/26002, versus the preserved 400-round terminal-integration
controls in `local-pcs-20260904-checkpoint400`. Both retain full sizing,
legacy suit buckets, potential bins three, fixed DCFR 1.5/0/2 and zero
averaging delay. Held-out / root-per-class / action-value budgets and seeds
match the controls (2,000 / 256 / 2,000). This is an approximately
table-size-matched development screen using historical controls, not an
equal-round/equal-wall-time or untouched final qualification comparison.
The 800-round integration candidate remains unchanged.

Use one training process at a time, a 12M-state cap, 6GiB sampled-footprint
stop, 20-minute stop per seed and 20GiB disk reserve. Checkpoints retain
all-street training; the diagnostic artifact exports only preflop. The
verified `2f023b5...` binary is copied to immutable
`runs/local-checkdown-20260905-solver`. No new training math, model interface,
abstraction, or gate is added. Require a useful policy-quality result before
considering further scaling; earlier equal-round evidence was not a win.

Both 600-round seeds finished without a stop. Checkpoint hashes were
independently verified. The measured comparison is:

| Metric | Integration 400 control | Checkdown 600 candidate |
| --- | ---: | ---: |
| Infosets A / B, million | 10.014 / 9.391 | 10.171 / 9.535 |
| Root local gain A / B, bb | 0.463259 / 0.486139 | 0.514125 / 0.505366 |
| Mean root local gain, bb | 0.474699 | 0.509745 |
| Maximum action MAE | 8.902% | 9.806% |
| Primary agreement, unweighted | 61.538% | 56.213% |
| Primary agreement, combo-weighted | 63.952% | 58.824% |
| Maximum aggregate action delta | 5.633% | 3.983% |
| Maximum held-out unknown fraction | 14.354% | 13.900% |
| Minimum action-EV SE coverage | 20.932% | 17.835% |

The apparent aggregate coverage improvement is not a matched fixed-trajectory
coverage test; seed B's unknown fraction actually rises 12.649% -> 13.482%.
Root local gain is not full-game exploitability. Both candidate root point
estimates worsen; no paired per-deal significance claim is made. Aggregate
action delta improves but remains above 3%; other stability and precision
checks worsen. **Reject this candidate; no larger pure-checkdown run follows.**
All 169 classes, compatible legal actions and probability sums still pass.

Candidate runtimes A/B: 276.363 / 269.204 seconds. Sampled footprints:
3,350,547,192 / 3,146,451,536 bytes. Checkpoints:
A `53e90dee4bf8d7d02e3725eb3aa587cf943e02ff6e324d599a365c88105ce7fb`;
B `236683e627b1bd6e2725590417c07760aa8278bcc275502a39fab1708e4faffd`.
Canonical artifact hashes:
A `d3129b28e855c588097a18faf7d7994aca57608338287884e9d4e3d1c3708111`;
B `2b5215fc2bbda99f31072fd5af3374074613ff85b7a36fe94138ffcda5c62976`.
The frozen binary and full reports remain unchanged. No solver process from
these completed cohorts remains running.

Next selected implementation: an explicit opt-in **streetwise estimator**,
using the existing exact terminal-integration branch through preflop/flop and
the existing checkdown-baseline branch on turn/river. Both components already
have unbiased-estimator tests; composing them by public street still needs
dedicated tests, resume/configuration pinning and legacy replay verification.
The hypothesis is to preserve early-street accuracy while spending fewer
states in late streets, which the retained research profile already resolves
online. It is not a demonstrated policy improvement or zero-variance method.
Do not change the game, default estimator, current 800-round checkpoints,
serving model, or validation gates.

After verification, use a short paired 400-round screen against the preserved
integration-400 controls before any larger run. Disk free is about 22GiB. Permission was asked
asynchronously to delete only the two newly generated rejected-checkdown
checkpoints (about 2.4GB), preserving reports, exports, frozen executable and
all original/best checkpoints. No files have been deleted; do not treat the
unanswered question as approval. Coding/tests can proceed independently of
that retention choice, and the 20GiB disk reserve remains enforced.

## Streetwise estimator implementation

Implemented opt-in `--streetwise-opponent-estimator`. The opponent estimator
selects the unchanged terminal-integration branch on preflop/flop and the
unchanged checkdown-control-variate branch on turn/river. Traverser enumeration,
regret updates, average-policy weighting and the game abstraction stay unchanged.
Selection depends only on public street. The expectation-preserving composition
is our design inference from the existing estimator identities, not a published
claim that this particular hybrid has lower variance or exploitability.

The mode requires public-chance sampling and is exclusive with the two older
variance options. Config, CLI summary, runner fingerprint and artifact training
identity pin it. Its checkpoints use schema 6 so older readers cannot silently
ignore the new flag; existing modes still write schema 5. The inference-only
reader validates both schema/mode combinations and rejects mismatches. Do not
resume a schema-5 training run into this mode.

Verification: 232 release library tests and 9 CLI tests pass, including the
previously failing all-street branch/terminal-CFV regression, deterministic
interrupted/uninterrupted training and checkpoint reader rejection. The relevant
32 Python runner/resource/pilot tests pass. An additional, unrelated neural
release test module could not import under system Python because NumPy is absent;
no dependency was installed and it is not counted as passing.

Actual two-round old/new executable comparisons preserve both artifact and
summary bytes for every older PCS mode. Artifact SHA-256 values:

- Ordinary sampling: `0dbd2fa30b6f2f6023e5eab9ebab1f8af4d843224737c6554bf873659f38689c`.
- Terminal integration: `c602ffbb37a2a2c8d6b787051bdafe5749ea4ba2c03905f9b133a4c7a30361d3`.
- Checkdown baseline: `c160cf62c61b85085ee9d382ede30964c98af338f7a2fcd28541aa644105377a`.

Temporary fixtures are at `/tmp/poker-streetwise-check.0CSwdE`. The real new-mode
checkpoint has schema 6; the old frozen `2f023b5...` executable rejects it as
unsupported. This is a small correctness replay, not policy qualification.
The preceding rejected-pilot milestone `f7fb41e` passed CI 33987662104.

To proceed without deleting any checkpoint, the paired 400-round screen will
omit only **new resumable checkpoint writes**. It still trains all streets and
performs the same internal held-out/root/action-EV evaluation, retaining summaries,
diagnostic preflop exports, command/configuration hashes and a frozen executable.
Use the existing resource guard and summary/artifact validators. If promising,
reproduce the deterministic run before saving a full checkpoint for later routed
evaluation; the diagnostic preflop export cannot serve the full-hand model.
This saves disk, not training memory, and no completed policy result is claimed yet.

## Streetwise 400-round screen: rejected

The implementation milestone `17636b3` was committed/pushed. Frozen executable
SHA-256: `58f2437e4a58450d01f8e5c580b295030526b7e609cb4f6f51d1d2c4f089bc76`.
The screen is complete at
`preflop-solver/neural/runs/local-streetwise-20260905-screen400/screen.json`.
Both seeds use the integration-400 control commands with only the estimator,
binary/output paths and omission of checkpoint writes changed. The original
800-round source checkpoints and previously rejected checkpoints are untouched.

| Metric | Integration 400 control | Streetwise 400 candidate |
| --- | ---: | ---: |
| Infosets A / B, million | 10.014 / 9.391 | 8.944 / 8.687 |
| Root local gain A / B, bb | 0.463259 / 0.486139 | 0.482961 / 0.492458 |
| Mean root local gain, bb | 0.474699 | 0.487710 |
| Maximum action MAE | 8.902% | 9.368% |
| Primary agreement, unweighted | 61.538% | 47.929% |
| Primary agreement, combo-weighted | 63.952% | 51.735% |
| Maximum aggregate action delta | 5.633% | 7.080% |
| Maximum held-out unknown fraction | 14.354% | 14.397% |
| Minimum action-EV SE coverage | 20.932% | 20.266% |

State savings are 10.69% / 7.49%, but policy consistency deteriorates and both
root-gain point estimates worsen. Their standard errors are 0.022968 / 0.022706bb;
no paired per-deal significance claim is made. **Reject the candidate; do not
extend this hybrid or save a new full checkpoint.** It has not improved full-game
exploitability. All 169 root classes, legal-action compatibility and probability
sums still pass. Preserve the existing integration-800 research source.

Runtimes A/B excluding checkpoint writes: 44.591 / 42.566 seconds. Sampled
physical footprints: 2,936,850,880 / 2,859,665,784 bytes. No resource stop fired.
Do not compare those times directly with controls that wrote full checkpoints.
Both canonical artifact hashes were retained:
A `8e46d7d2f3a2122d91b508041abd409dd0b3f4f833f372f23fc026dfa41c8fd4`;
B `e6222b8b648f9d807533a2a5de187aac303f330faf1feace9e3b83e6ef8be2fa`.
Compressed output hashes were independently verified:
A `035f8c1a5232f79afd63e084835f4fbcd70ee185dee0d00104b41d213f5968d3`;
B `d4848a04b0d409f1f35828814bdf5e8a6ac8bb199f3cb302f5d4e9b9b65a8f98`.

The one-off supervisor initially omitted `export_postflop_strategies=False` from
its validator namespace. Seed A had already exited successfully and written both
outputs when validation raised an AttributeError. Preserve `pilot-original.py`
and its recorded hash; the corrected supervisor validates and reuses A's exact
command/output/resource record, then runs B once. No solver process was restarted,
training modified, or failed evaluation counted as passing. `screen.json` records
both script identities and this recovery. Both solver processes are now terminal.

The pure-checkdown and hybrid screens now both argue against sacrificing
terminal-integration accuracy for fewer visited states. Further work should not
repeat either arm with a longer budget merely to recover the lost consistency.

## Common-sample counterfactual training pilot

The previous screen's remote CI passed (33988945311). The earlier implementation
CI 33988647857 was superseded/cancelled, not a separate passing run.

Research/code review first checked whether tiny shared-action probabilities
cause unbounded counterfactual values. For the existing proposal
`q[a] = sum_h(r[h] * sigma[h,a]) / sum_h(r[h])`, the corrected child range has
`sum_h(r[h] * sigma[h,a] / q[a]) = sum_h(r[h])`. After exact terminal integration,
the conditional proposal multiplies this total by the remaining continuation
mass, at most one. The code's regression now explicitly checks this mass
identity. Large per-combo `sigma/q` alone is therefore not evidence of exploding
aggregate range values. No proposal floor, clipping or changed action weights
were added on that unsupported premise.

The next hypothesis targets **cross-action covariance**, not range-mass scale.
[Lee et al., UAI 2020](https://proceedings.mlr.press/v124/lee20a/lee20a.pdf)
describe common random numbers for rollout comparisons: the variance of a
difference includes `-2 Cov(X,Y)`. Positive covariance helps; negative covariance
hurts. This is a general simulation technique, not their experiment on our CFR
implementation. [Gibson et al., AAAI 2012](https://ojs.aaai.org/index.php/AAAI/article/view/8241/8100)
relate bounded unbiased counterfactual estimates and their variance to sampled
regret minimization. Neither reference certifies this candidate's exploitability.
Owen's chapter appeared in search, but direct PDF retrieval was denied; the
accessible primary rollout paper supplies the formula used here.

For this pilot, implemented experimental `--coupled-traverser-samples`: at a traverser node, draw one
fresh continuation seed and give each sibling counterfactual action its own
clone of that stream. Opponent nodes still use their exact reached-range
proposal, terminal integration and importance correction. Along a path, future
samples remain fresh; sibling public histories are disjoint. The adaptation
preserves conditional marginal sampling expectations while changing correlations.
It requires the existing integrated PCS/trajectory-recall mode. It does not
change evaluation sampling, the game, probability policies, neural weights,
chance-card removal, or the default recurrence.

Config, summary, training hash, runner fingerprint and resume compatibility pin
the option. Coupled checkpoints use schema 7; schemas 5 and 6 remain unchanged.
Verification: 234 Rust release library tests, 10 CLI tests and 33 Python
runner/resource/pilot tests pass. Tests cover exact terminal CFVs/regrets,
explicit configuration, mode/schema rejection, and deterministic resume in all
five experimental modes. Small old/new artifact **and summary** comparisons are byte-identical
for ordinary PCS, terminal integration, checkdown and streetwise modes. Fixtures:
`/tmp/poker-coupled-check.Lf3nFv`; the first three artifact hashes match the prior
streetwise replay, and streetwise hashes to
`9e405461f1679043cfe854a8628ba7ffd43c8db9bfc10e3bdd4ae4062c215ead`.
The preceding executable rejects an actual coupled schema-7 checkpoint. These
are correctness checks, not a demonstrated variance or policy-quality win.

Frozen pilot executable:
`dae25592c5bc44716610cd233582015d356537c0eacfcc46ead656e685a7f77d`.
The checkpoint-free development screen runs sequential seeds 26001/26002 for
400 rounds, full default sizing and legacy buckets, fixed DCFR 1.5/0/2, zero
averaging delay, and matched 2,000 / 256 / 2,000 evaluation budgets. It retains
the 12M-state cap, 6GiB sampled-footprint stop, 20-minute limit and 20GiB disk
reserve. Directory: `preflop-solver/neural/runs/local-coupled-20260905-screen400`.
Only the training coupling and checkpoint-write omission differ from the
preserved integration-400 control, besides executable/output paths. No long run
or source-policy replacement is authorized by the implementation tests alone.

### Completed paired result and disposition

Both 400-round seeds completed, without resource stops or full checkpoint writes.

| Metric | Integration 400 control | Coupled 400 candidate |
| --- | ---: | ---: |
| Infosets A / B, million | 10.014 / 9.391 | 9.388 / 8.811 |
| Root local gain A / B, bb | 0.463259 / 0.486139 | 0.446816 / 0.500419 |
| Mean root local gain, bb | 0.474699 | 0.473617 |
| Maximum action MAE | 8.902% | 8.637% |
| Primary agreement, unweighted | 61.538% | 55.621% |
| Primary agreement, combo-weighted | 63.952% | 56.712% |
| Maximum aggregate action delta | 5.633% | 4.081% |
| Maximum held-out unknown fraction | 14.354% | 13.019% |
| Minimum action-EV SE coverage | 20.932% | 21.687% |

This is mixed evidence, not a useful overall policy winner. Root gains move in
opposite directions across seeds, and the mean is essentially unchanged; primary
agreement worsens. Reduced frequency error/unknown fraction and a small EV-precision
increase do not establish lower exploitability. These held-out trajectories follow
each policy and are not identical fixed-trajectory coverage tests. **Reject the
candidate; no longer coupling run or full-checkpoint reproduction follows.**

Unlike the prior streetwise implementation, this unsuccessful coupling option is
**not retained in the supported trainer/runner**. Its source diff is archived next
to the immutable executable, driver, summaries and exports in the local cohort:
`experimental-source.patch`, SHA-256
`92d13be05552894c5b532e3c610c79e693ab9a202aa9879ad4e8d80a2d698a60`.
It applies to `dd09cfd`; use that archived version for reproduction, not the current
runner. Schema 7 identifies only this archived coupled experiment and must not be
reused for an unrelated future recurrence. The small range-mass regression remains;
the normal training recurrence, config and schemas are unchanged from `dd09cfd`.

Runtimes without checkpoint writes: 41.022 / 38.983 seconds. Sampled footprints:
3,058,616,864 / 2,874,378,640 bytes. Root gain SE: 0.022401 / 0.022383bb.
Canonical artifact hashes:
A `fa41ad7a0eec4c8c9173652150eccfe485738b96309821ce31c353a922c0a8b6`;
B `4503cfe26c6569955ad208217c9744c60612d6aa8a0419807d4e29efb50d5307`.
Independently verified compressed output hashes:
A `c6da3f672a3eabae8969b464f19254c2340872ddada86e4b9dd74f586e9784b0`;
B `862942ff16ab14d220a7d827d4e650a57b0fef9e6d6ef0f0cb86a4e0fcf79e3b`.
Neither worker remains live. No source checkpoint, retained response, website model,
gate, or user file was changed or deleted.

After removing the unsuccessful experimental option, the supported worktree
passes 232 release library tests, 9 CLI tests, 32 Python runner/resource/pilot
tests and the release build. Its release executable is byte-identical to the
preceding `58f2437...` binary. The only retained native edit from this experiment
is the proposal/range-mass regression assertion; there is no new serving policy.

The three estimator screens do not support extending a new sampler. Next focus
returns to unresolved flop policy actions: inspect the existing public-belief
`solve_flop` / `solve_flop_public_chance_vector_mvp` continuation requirements and
the tabular turn/river adapter before selecting a bounded flop experiment.
This is not a scheduled new oracle/network rewrite, and no flop run has started.

## Fixed-flop sampled subgame proposal

Implemented `public_belief::sampled_flop::solve`: a pure, explicitly research-only
fixed-flop solve using the existing trajectory-recall PCS/DCFR traversal. It
trains through terminal outcomes without a neural leaf oracle and returns only
the current root's frozen average policy. It is not a complete continuation
policy, safe continual resolver, action-EV grader, or full-game certificate.
The normal full-hand training and serving paths do not call it.

The sampled future board proposal is uniform over 49*48 ordered turn/river
pairs. For a fixed compatible private pair, actual chance is uniform over
45*44 pairs. Future-blocked range weights become zero without renormalization;
the opponent CFV receives `(49*48)/(45*44)` exactly once. Zero joint-mass
proposals contribute zero rather than being resampled. Exact chance-mass/fold
integration, deterministic output, card uniqueness, legal normalized rows,
resource/invalid-input rejection and a known profitable all-in call are tested.
Any required root combo lacking both regret and average updates fails closed.

The research direction is motivated by the previously inspected
[MCCR paper](https://www.ifaamas.org/Proceedings/aamas2019/pdfs/p224.pdf): sample
from a current public subgame rather than train every possible full-game root.
This module does **not** implement that paper's full-game protection/statistics
and must not inherit its convergence claim. It retains the supplied game's
betting and card abstractions; only the local training controls are reset.

### Bounded cost measurement

First cohort: `local-sampled-flop-20260905-cost1`. The 4bb-pot probe reached its
500,000-information-set guard after 1.81 seconds (sampled footprint 165,003,744
bytes); the test failed and the other two pots were not attempted. No memory
or time safety stop fired. Frozen test executable:
`5fea8bfb9054a3643f5578d0c408ec59ab09f2ab2d9f1ff445faf9f9a888f94b`.

Second cohort: `local-sampled-flop-20260905-cost2`. Explicitly increased only
the experimental node cap to 2M, keeping the 2GiB sampled-memory stop, 120-second
overall stop and 20GiB disk reserve. All three 20bb/default-abstraction roots
completed at 32 iterations, uniform flop ranges, board `[48,21,2]`:

| Pot, bb | Seed | Information sets | Seconds | Trained root combos | Minimum averaging contributions |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 83001 | 881,173 | 3.484 | 1,176 | 12 |
| 10 | 83002 | 592,142 | 2.613 | 1,176 | 10 |
| 20 | 83003 | 391,837 | 2.354 | 1,176 | 10 |

Whole-worker sampled peak: 293,569,064 bytes; elapsed 8.658 seconds; exit 0;
no stop. Concurrent compilation means these are observed costs, not isolated
speedup measurements. Averaging contributions are not independent effective
sample counts or a quality gate. No checkpoints were written. Frozen test
executable `2d705f2e2a60639879c21fb1c4bb5f3636dcb74daf155a8249289a7e904aef98`;
log SHA `94dd1a89c1ddbdaa8e6772716a90d1ff8b859a75d52469525c1b3fd2d141de02`.

The module-only work passed 237 release library tests (one explicitly ignored
cost probe), nine CLI tests and the release build. Two additional test-only
pilot checks subsequently passed: conditional private/runout sampling and zero
paired gain for identical root policies. The full suite after that addition
has not yet been rerun.

### Conditional root-action pilot

`local-sampled-flop-20260905-rootpair1` is now complete; results are below.
Its test-only driver loads original 800-round checkpoints sequentially, retains
terminal-flop weight 0.5 and joint-four turn/river, and collects one authentic
nonterminal flop decision per seat. For each public root it reconstructs ranges
from the frozen observed action frequencies, then trains two independent
32-round proposals (84001/84002, 2M-state cap). All private hands and future
cards are absent from the proposal input.

Both proposed mixes are evaluated on the same 128 independent conditional
private-pair/runout draws; initial actions are enumerated and continuation
streams are shared. Every later decision uses the unchanged retained profile,
including its original turn-range reconstruction. This intentionally measures
**one-decision conditional payoff**, not a newly integrated multi-street policy:
the local training continuation differs from the evaluation continuation. A
positive result would justify testing integration, not model activation or an
exploitability claim. The two exact roots per source are not population coverage
or an untouched full-game test. Each source has a 30-minute/7.5GiB stop and a
20GiB disk reserve; no additional full-size worker runs concurrently.

### Accidental loose-artifact overwrite

An operator call to the old frozen executable with `blueprint --help` was
mistakenly assumed read-only. That subcommand ignores this flag and instead
completed a default run, atomically replacing the previously untracked root
`blueprint-artifact.json`. The attempted process stop arrived after completion.
The overwrite was disclosed immediately; no original content hash was recorded,
so byte identity with the preceding loose artifact cannot be established.
No duplicate was found in the repository or `/tmp`; the listed local snapshots
were OS-update snapshots, not an identified recovery copy. The preceding loose
artifact is **not restored**. Current file SHA:
`f9ce87ffe648b90b6ae6b5b10ffe4ccb59c3ee57cae0d4292708a1b8755a65b2`
(33,055,215 bytes). Leave it untracked, do not count it as an experiment, and do
not claim it was preserved. The original source checkpoints, serving model,
reports, and other unrelated files were not overwritten. Read CLI source/help
text directly for subsequent diagnostics; all intentional runs specify new
explicit output paths. No files have been deleted.

### Completed root-pair result and next experiment

Both source checkpoints completed without resource stops. The root collector
selected the same visible flop `[42,28,21]` for both sources, with different
preflop opens and flop bet sizes. These are therefore **not four independent
flop textures**. Each row below compares the two proposal training seeds
84001 / 84002 against its unchanged retained continuation:

| Source | Decision | Conditional gain, bb (seed 1 / seed 2) | SE, bb (seed 1 / seed 2) |
| --- | --- | ---: | ---: |
| 26001 | SB facing 4.5bb bet, 10.5bb pot | +2.749064 / +2.234088 | 0.432734 / 0.417405 |
| 26001 | BB opening action, 6bb pot | -0.245809 / -0.228770 | 0.348317 / 0.424942 |
| 26002 | SB facing 1.667bb bet, 6.667bb pot | +0.543443 / +0.305331 | 0.433885 / 0.426486 |
| 26002 | BB opening action, 5bb pot | +0.186164 / +0.719841 | 0.372645 / 0.402985 |

Only source A's facing-bet root has positive individual normal-approximation
99% intervals for both proposal seeds (lower endpoints +1.634416 / +1.158925bb).
Every other interval crosses zero. No family correction was applied. These are
128 conditional deal samples per root, not full-hand win rates or an
exploitability improvement. There is evidence worth following up at a facing-bet
decision, but **no broad flop-policy winner and no activation**.

Actual authentic-range proposal costs were 0.974–2.301 seconds and
84,845–430,280 information sets at 32 iterations. Every proposal served all
1,176 board-compatible root combos. Whole-worker runtimes including checkpoint
loading/evaluation: 135.727 / 202.800 seconds; sampled footprints:
5,952,148,112 / 5,909,303,832 bytes. Only one source was loaded at a time.
Frozen test executable:
`f424a838840824cf8a3e4ae019d8d5d64c2e8bd63a9f9dfb3afecb080f04d442`.
Independently checked log hashes:
A `940eb0c3c12a540eb4bf079ed70249c57e71dd2426c1ba211545f4a8d4328999`;
B `86cc0b8fd9c030a03132eb0f8616990fe38642111b2a03071ec776502f210df8`.
The local manifest retains each public line, input/policy hashes and all gains
and intervals. Neither worker remains live.

After adding the pilot, full `cargo test --release` passes 239 library tests
and nine CLI tests; the two expensive development probes are explicitly ignored
by default and were run separately above. `cargo build --release` passes; binary
SHA `1f44c0c784c32a43da9c8874c9ebdad77f9759fbd4679ed3510a0c0f9a74a248`.
A fresh two-round integrated-PCS replay matches the preceding frozen executable's
artifact **and summary bytes**, artifact SHA
`c602ffbb37a2a2c8d6b787051bdafe5749ea4ba2c03905f9b133a4c7a30361d3`;
fixtures `/tmp/poker-sampled-flop-replay.RtODeQ`. This is not a full-size
deterministic replay. The 25 existing cloud-runner/resource-guard Python tests
also pass. No TypeScript/browser/serving code changed.

Next bounded action work: retain the exact public root inputs and per-sampled-
hand action-value vectors in the development pilot so subsequent short training
comparisons can reuse the **same frozen evaluation** instead of repeatedly
loading the full checkpoint and resolving the same continuations. The current
logs retain policy hashes, not reconstructible proposal rows/value vectors, so
do not claim such a cache already exists. Use it for a controlled short 32-vs-128
iteration comparison and additional fresh flop textures, including both seats
facing bets. Development reuse is not independent validation; any selected
policy still needs a genuinely fresh full-hand challenge. Do not start a long
run from this one-texture result, rewrite the neural oracle, or change serving
behavior or gates. All quality gaps at the top of this document remain open.

## Frozen root-action samples and short training comparison

Milestone `ed40290` passed remote CI 33991800881, including native release,
TypeScript/application build and existing production-resolver checks. This is
implementation verification, not policy qualification.

The test-only root pilot now separates expensive conditional action sampling
from inexpensive scoring of new root policies. A MessagePack cache retains the
exact public state/ranges, source checkpoint hash and continuation settings,
evaluation seed, sampled acting-hand combo, baseline mix and every action's
realized continuation payoff. It contains no neural value predictions and is
not read by serving. The existing direct rollout comparison remains an
independent regression reference: cached means and SEs match it exactly for
several distinct root policies. MessagePack round-trip preserves all floats;
validation rejects illegal line/accounting, malformed probabilities, nonfinite
or out-of-bounds payoffs, invalid combos and truncated data. A new proposal's
paired difference against the first proposal is accumulated per sample, retaining
their covariance rather than combining unrelated standard errors.

Cache collection and proposal export require explicit directories and use
`create_new`; existing or interrupted output files cannot be overwritten.
Each cache/proposal is hash-pinned by the external supervisor. The original
800-round checkpoints, default solver and serving model remain unchanged.

`local-sampled-flop-20260905-cachepair1` is running. It collects two distinct
flop boards within each of four strata (seat SB/BB crossed with facing a bet /
not facing a bet), eight roots per original checkpoint. Roots come from the
retained profile's authentic public lines with fresh collection seed 85003;
all-terminal decisions are excluded because the experiment targets nonterminal
flop play. Different strata may share a board, so eight roots do not imply
eight independent textures. Cache evaluation uses 128 conditional deals/root,
seeds `85004 + 100*stratum + index`. This is a stratified **development** sample,
not reach-weighted full-game validation.

The two checkpoints are processed sequentially under 7.5GiB/30-minute sampled
stops and the unchanged 20GiB disk reserve. Once both caches are complete and
verified, the full tables are no longer needed: two small workers compare
32 versus 128 iterations using independent training seeds 85001/85002 and the
same frozen cache for every comparison. Each worker has a 2GiB/180-second stop;
each proposal retains the 2M-information-set cap. A cap failure is explicitly
recorded as an incomplete comparison, never a uniform fallback or a winning
zero. Policies and exact inputs are retained for later inspection/reproduction.
No additional sampling run is performed for the longer proposal. Any later
selection from this cache still needs independent fresh evaluation; cache reuse
must not be presented as a new holdout or exploitability evidence.

Research cross-check while collecting the cache:
[Brown/Sandholm 2017](https://papers.neurips.cc/paper/6671-safe-and-nested-subgame-solving-for-imperfect-information-games.pdf)
explains why conditional subgame gains do not alone protect the full game;
counterfactual best-response values include unreached opponent information sets.
This supports keeping our root-action diagnostic separate from full-game claims.

A newer [Kubíček/Lisý/Sandholm preprint (January 2026)](https://arxiv.org/html/2601.17131v1#S7.SS2)
identifies another possible issue in **gadget-based** solving: different gadget
equilibria can perform differently in the original game, particularly where the
gadget opponent never enters. Its CFR modification perturbs auxiliary gadget
choices toward a positive prior and transforms the regrets consistently; it is
not a floor on every poker action. The paper tests small benchmark games and
thousands of CFR iterations, not our sampled 20bb solver. Our current candidate
uses unconstrained joint-turn solving and the sampled flop has no gadget, so
this is **not an established explanation or immediate fix for the current
results**. Do not add blanket probability floors, claim its reported improvement
for this model, or interrupt the paired experiment to build a new resolver.

### Avoiding repeated continuation solves without changing samples

Source A finished all eight caches in 813.146 seconds, with a sampled peak of
6,200,840,944 bytes and no stop; source B is still collecting. A's logs contain
1,394 turn solve calls for 635 distinct public roots across all caches. Within
each cache separately the distinct counts sum to 731, showing repeated solves
can be removed without retaining a large multi-root policy table.

Implemented a separate test-only grouped collector. It samples the same deals
and traverses the same flop actions, stops each nonterminal request at turn
entry, groups by visible turn board/public history, and carries each request's
unchanged RNG into the existing rollout. The resulting action values are put
back in their original sample/action slots. It still uses the existing single-
generation turn cache; no unbounded policy cache or changed strategy is added.
A real small turn/river regression verifies identical cache bytes, positive
solve counts and strictly fewer actual solves. The original direct collector
remains available as a reference.

The ongoing paired run stays on its immutable `d8d769a...` test executable and
does not use grouping. A separate, not-yet-run full-size check is prepared at
`local-sampled-flop-20260905-grouped1`: reconstruct A's first frozen cache and
require exact equality of every sample and all serialized bytes. Its supervisor
refuses to overlap the earlier pipeline. Frozen check executable:
`e0c11c4b01b50860a373838d8ead16158075a7cfb82a476047e3f5da767e9f64`.
Until that check runs, no full-size speedup or parity result is claimed.

Current verification: 241 release library tests, nine CLI tests and the release
build pass; five expensive development probes are explicitly ignored by the
default suite. The production executable remains byte-identical to `1f44c0c...`.

### Completed cached pair and full-size grouping check

The preceding pipeline is now **complete**, not still collecting. All 16 roots
finished both independent training seeds at both 32 and 128 iterations: 64
retained proposal artifacts, with no node-cap failures or resource stops.
All 80 cache/policy hashes and all 18 worker-log hashes were independently
rechecked. Manifest SHA:
`248fd69318bf2b7353e5e8c6667cafe3fcc5b32ce19dd14de801e6eefefc529d`.
Source A/B cache workers took 813.146 / 866.275 seconds, sampled peaks
6,200,840,944 / 6,130,635,456 bytes. Their log hashes are
`83b556821816bd46c74011c82914741c717aed112fd2a621ca1a78a5d78436fc` /
`d07fe077d6eaa63af0dc044421dfee74fb36f60d8814d1e315f4650445dd14ef`.
Offline proposal training summed to 293.227 worker-seconds using two workers;
maximum sampled footprint per worker was 490,865,456 bytes. Whole-pipeline
wall time was 1,835.936 seconds. Those timings are not isolated throughput
benchmarks.

Conditional gains at 128 iterations and paired changes versus 32 are below
(training seeds 85001 / 85002). These are bb per sampled root decision, **not
bb per full hand**, and are not reach-weighted. Strata 0/1 are SB not-facing /
facing; 2/3 are BB not-facing / facing. Indices distinguish the two roots.

| Source / stratum / index | 128 gain over retained, seed 1 / 2 | 128 minus 32, seed 1 / 2 |
| --- | ---: | ---: |
| A / 0 / 0 | +0.252 / +0.220 | +0.199 / -0.084 |
| A / 0 / 1 | +0.232 / +0.490 | +0.089 / -0.153 |
| A / 1 / 0 | +1.330 / +1.488 | +0.392 / +1.242 |
| A / 1 / 1 | +4.396 / +4.394 | +0.009 / +0.209 |
| A / 2 / 0 | +1.168 / +1.311 | +0.137 / +0.608 |
| A / 2 / 1 | -0.252 / -0.160 | +0.063 / +0.410 |
| A / 3 / 0 | +0.071 / +0.491 | -0.024 / +0.370 |
| A / 3 / 1 | +2.158 / +2.074 | +0.004 / +0.735 |
| B / 0 / 0 | +0.440 / +0.805 | +0.055 / +0.436 |
| B / 0 / 1 | +0.336 / +0.467 | +0.673 / -0.517 |
| B / 1 / 0 | +0.441 / +0.532 | +0.082 / -0.005 |
| B / 1 / 1 | -0.271 / -0.132 | -0.045 / +0.070 |
| B / 2 / 0 | +0.037 / -0.269 | -0.443 / +0.088 |
| B / 2 / 1 | +0.537 / +0.438 | +0.132 / +0.758 |
| B / 3 / 0 | +2.434 / +2.516 | -0.292 / +0.910 |
| B / 3 / 1 | -0.671 / -0.179 | -0.887 / -0.035 |

Three roots (A/1/1, A/3/1, B/3/0) have positive individual normal-approximation
99% intervals against the retained policy for both seeds at both lengths.
These intervals are not family-adjusted. Increasing iterations improves the
point estimate in 22/32 comparisons, but only two paired individual 99%
intervals are positive and one negative; the other 29 cross zero. The seeds
share each root's frozen action samples, so do not pool them as independent
evaluation evidence. **No broad winner, longer identical run, model activation,
or new full-game exploitability claim follows.**

`local-sampled-flop-20260905-grouped1` also completed without a stop. Every
sample and serialized byte of A/0/0 matched the frozen original cache. Actual
turn solves decreased from 215 to 115 (46.5% fewer); evaluation time was
85.083 seconds versus the earlier 198.133 seconds. The measured wall-time
comparison was not isolated from other host work; byte parity and solve-count
reduction are the stronger evidence. Whole worker: 154.380 seconds, sampled
peak 6,200,742,640 bytes. Log SHA:
`1e84a676a8031f799fd4296c56156bb1ab047a3a0a813ad26bd921fdf850b9e4`.
Future development-cache collection now uses grouping; the completed pipeline
and original direct regression reference remain unchanged. No serving code
or probability changed.

### Action-level diagnosis before another policy change

Read-only inspection of the preserved MessagePack rows and samples used
`uv run --no-project --with msgpack==1.2.2 python`; no project dependency was
added. Reconstructed gains and paired changes match every inspected manifest
value within 1e-10. Attribution centers each sample's Q vector on the reference
policy value before multiplying by the probability change; it is a diagnostic
decomposition, not a new action-EV accuracy estimate.

The negative B/3/1 comparison (85001) is reproducible from the frozen manifest:
128-minus-32 = -0.887024bb, individual 99% interval [-1.719393, -0.054656].
Its sampled call frequency rises 20.4% -> 36.9%, while ordinary raises fall
18.7% -> 5.9%. The call component accounts for -0.805bb of the paired change.
The other training seed's paired change is -0.034696bb and inconclusive. This
identifies the action shift, not yet why training chose it; a fresh conditional
sample is needed before attributing a persistent model defect.

The largest positive root A/1/1 starts with a 50/50 fold/call baseline on the
sampled hands; the 128 proposals call 98.5% / 97.9%. A/3/1 instead improves
mainly by folding more and reducing all-in raises. B/3/0 improves hand-dependent
fold/call allocation despite only modest aggregate frequency change. There
is **no single blanket fold/call correction** supported across these roots.

Next action work should distinguish evaluation noise from the known training /
deployed-continuation mismatch at the negative call root before integrating or
lengthening this proposal. The existing cache makes action inspection cheap;
do not rerun the complete cache collection merely to inspect the same rows.

### Fresh recheck changes the apparent call-regression diagnosis

Milestone `920bd049358f5f44e596bb34b8b9eb9d812e3f56` is pushed; remote CI
33994258109 passed. It includes the completed cached pair and exact grouping
optimization, not a newly activated poker model.

`local-sampled-flop-20260905-recheck1` is complete. It re-evaluated the four
already-frozen B/3/1 proposals on 1,024 fresh conditional deals, seed 86004;
no retraining or action-frequency adjustment occurred. The root was selected
from development results, so this is fresh sampling at a selected root, **not
an untouched-root or full-game validation**.

| Training seed | 32 gain over retained (SE), bb | 128 gain over retained (SE), bb | 128 minus 32, individual 99% interval |
| --- | ---: | ---: | --- |
| 85001 | +0.915863 (0.167384) | +0.678016 (0.218439) | -0.237848 [-0.524300, +0.048604] |
| 85002 | +0.728138 (0.171197) | +1.015516 (0.202552) | +0.287378 [+0.066131, +0.508624] |

All four gains against the retained policy have positive individual 99%
intervals. The earlier negative 128-vs-32 interval does not reproduce in the
fresh sample; the other training seed now favors 128. Thus the earlier
call-attribution decomposition described that particular sample correctly,
but did **not** establish a persistent harmful-call defect. Do not hard-code
a call reduction, claim that the defect was fixed, or select a universal
training length from these results. This is why the diagnostic stopped short
of a policy mutation until the fresh recheck.

Whole worker: 131.753 seconds, sampled peak 5,896,229,400 bytes, no stop.
Grouped evaluation: 58.095 seconds / 98 unique turn roots. Binary SHA:
`a0eba1aa35645eb76cd8a4ef581c3dc9d69e27d62b8db06d3512c936485f0d68`;
fresh cache SHA:
`4c44bb25c7c8046033e544cfa3fb4eac90f2a2fa3407015b5e6717377dd89aaf`;
log SHA `2ccdd75da8ef1de2e59dc3522581644c2481f9d7eb3842150fd893505b2a5b54`.
The read-only recheck driver validates each proposal hash against the frozen
earlier manifest and creates a new cache; it cannot overwrite the old one.

### Full-hand integration experiment, not another isolated action correction

The next experiment routes every nonterminal flop decision through a bounded
32-iteration sampled solve. The existing terminal-flop rule and joint-four
turn/river solver remain in place. A test-only adapter at `FlopPatch` makes
actual play and all later public-range reconstruction call the **same** new
flop probabilities. The optional adapter is excluded from production builds.
There is no website activation, new training export, permanent policy table,
or baseline fallback after a failed sampled solve.

The policy is a deterministic function of the original checkpoint, training
seed, visible flop and public action history. Earlier flop prefixes can be
resolved recursively during range reconstruction; no actual opponent holding
or future community card enters those solves. Cached rows are bounded to one
flop board and 64 public histories. Cache eviction changes computation only.
The source model remains shared read-only; each local training table is dropped
after exporting its root average policy.

Research cross-check: [DeepStack's continual-resolving description](https://arxiv.org/html/1701.01724v2)
updates its own range with the probabilities it actually played, and separately
maintains opponent counterfactual-value constraints. The new adapter addresses
the probability/range consistency part, **not** that full safety construction.
[Brown, Sandholm and Amos 2018](https://noambrown.github.io/papers/18-NIPS-Depth.pdf)
also explain why fixed continuation values are insufficient for equilibrium
robustness and study opponent continuation choices at depth limits. Our cached
Q samples are not that construction or a substitute for its requirements.

The known mismatch between local sampled self-play training and the retained
turn/river continuation is not fully removed by consistent posterior updates.
The full-hand pilot therefore tests the combined behavior empirically rather
than assuming safe composition from the local-root gains. It uses 64 paired
hands per seat and independent solve seeds 87001/87002 for each retained source,
fresh evaluation seed 87004, sequential checkpoint workers and the unchanged
7.5GiB / 20GiB guards (30-minute cap per source). This is a bounded integration
and fixed-opponent payoff screen, not a full-game exploitability certificate.

### Completed full-hand routing result

`local-sampled-flop-20260905-fullhand1` is complete: 512 paired hand comparisons
across two source checkpoints, two independent resolve seeds and both seats.
All eight cases completed without resource stops or missing-policy fallbacks.
The sample exercised every street overall, but not every seat/seed case reached
the river. `candidateStreetDecisionVisits` counts both players' decisions in
the candidate-versus-control game, not only the candidate's decisions.

| Source / solve seed | SB payoff gain (SE), bb/hand | BB payoff gain (SE), bb/hand |
| --- | ---: | ---: |
| A / 87001 | -0.062500 (0.456368) | -0.364578 (0.364578) |
| A / 87002 | +0.039063 (0.626615) | -0.419266 (0.367797) |
| B / 87001 | +0.445313 (0.474958) | -0.046875 (0.046875) |
| B / 87002 | +0.070313 (0.062863) | -0.164063 (0.095181) |

Every individual 99% interval crosses zero. BB point estimates are negative
in all four cases, but these are only 64 hands per case and share their chance
streams across resolve seeds; no pooled significance or causal strategy defect
is established. The earlier local-root gains have **not** established a
full-hand strength improvement. Do not promote this experiment, claim lower
exploitability, or increase training iterations on the assumption it won.

Source A/B worker times were 193.079 / 206.215 seconds, sampled peak physical
footprints 6,432,838,600 / 6,380,835,736 bytes. Maximum observed local solve was
654,909 information sets under the 2M cap. Whole-pipeline wall time was
400.998 seconds. Frozen executable SHA:
`3e3095e82b824f1b4d16e3d90bfd86077d93b710f2aec3b40730c654fe92c73e`;
manifest SHA:
`d59ca540710abcbc27446e356cc10ec955a7ac2fbcf8dc5272e9c9509142ffb4`.
Worker logs independently verified:
A `f486a21ca8a5a01562e3828b578a5838f65c5bd59fb1b30dcdb600afeaf7ae07`;
B `311bf098d901dfa2f6d13392e1c8afc432c28c83bf856f22752e656160cf5fb3`.
No worker remains live.

Final native verification passes 243 release library tests and nine CLI tests;
seven explicit development probes are ignored by the ordinary suite. The new
regressions test hidden-card/future-card independence, deterministic cache
eviction, serving/replay probability parity, and the actual turn solver's range
reconstruction after resolved flop actions. `cargo build --release` passes.
Production binary SHA is now
`3076ab8fbfe56cce996a3b1054429df7fbe1251919d5baafa57087e32f6ccb39`;
the test adapter is excluded from it. A fresh explicit-path two-round native
replay matches both the prior artifact and summary bytes, artifact SHA
`c602ffbb37a2a2c8d6b787051bdafe5749ea4ba2c03905f9b133a4c7a30361d3`.
Replay directory: `/tmp/poker-flop-routing-replay.oQIaFA`. This is not an
unperformed full-size replay. No browser-facing change was made.

Next policy work: use the fully specified experimental full-hand policy in
the existing full-game response evaluator, starting with a bounded interface /
cost check before a larger independent challenge. Do not keep selecting
policies from fixed-opponent payoff gains or fitting a call/fold rule to these
64-hand samples. Response calibration must still reject unprofitable critics
without interpreting their deployed zero as an exploitability upper bound.
The original preflop consistency, precision, coverage and full-game quality
requirements remain open. The managed goal remains active.
