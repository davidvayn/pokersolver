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
overall stop and 20GiB disk reserve. All three 100bb/default-abstraction roots
completed at 32 iterations, uniform flop ranges, board `[48,21,2]`:

Depth correction, verified during the exact-terminal training pilot below:
this paragraph originally said 20bb. The archived `cost_probe` in `ed40290`
uses `BlueprintConfig::default()`, which is 100bb in that same commit. These
standalone timings must not be treated as 20bb cost evidence. Full-profile
experiments cloning the explicitly checked 20bb source are a separate scope.

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

### Completed connected-policy response cost probe

`local-sampled-flop-20260905-responseprobe1` completed successfully in 401.755
seconds, peak sampled physical footprint 6,413,128,552 bytes, without a
resource stop. It attacked the actual experimental source-A / resolve-seed
87001 profile, not the legacy checkpoint factory's unpatched policy. Budgets
were 16 training, 32 calibration and 32 fresh raw holdout hands per seat,
two action rollouts and minimum four observations per response group.

Both saved critics contain **zero supported decisions**. Both calibration
and raw independent holdout consequently report identity gains of zero;
neither qualified. This is an unsuccessful critic budget, not an exploitability
pass, nor evidence that the connected policy fixed the previous weaknesses.
Do not repeat another 16/32 probe or hide rejected critics behind zero totals.
Frozen executable SHA:
`b1f9ff4b6e298a2eb33fa2bfb904d852ea227327ecc4ad65fe586042f8a27291`;
log SHA `c0391f0797fce4f3aa0977b4cccc144a0b446c2a2125562403599bc34d5d5dad`.
Saved seat-0/seat-1 response hashes:
`df94739e235c3452a91d6bd73d76713b4bdc4c1a05e7680dd4adcaee3503baa1` /
`5d890af571d93ee294ea0a70a1ef09defb8329ea7008dc28e39bd4f0ee04db60`.
Neither worker remains live. Previous milestone `67aad96` passed CI
33995739284.

The diagnosing-bugs loop first reproduced the empty-critic symptom directly
from the frozen report. A cheap authentic-trajectory replay then counted the
same four layers of response keys without generating counterfactual action
labels. A deterministic regression compares those keys against the real
collector, for both seats and all-street/postflop-only settings. This is
sampling-support diagnosis, not a change to the response confidence rules.
No permanent dependency, grouping coarsening or reduced particle minimum was
introduced. Initial source-A / SB counts reproduce zero qualifying groups at
16 hands (maximum support two); 128 hands support only one preflop group per
layer and no postflop groups. At 512 hands the strategic layer supports
70 preflop / 8 flop / 4 turn / 0 river groups. Finer postflop layers still have
no supported groups. The full bounded census is recorded separately below
when terminal; these prefix observations are not a completed-census claim.

The census `local-sampled-flop-20260905-support1` was subsequently stopped
intentionally after the diagnostic had changed the next action. It completed
the above 16/128/512-hand SB prefixes, not 1,024 hands or the BB census. Its
immutable supervisor manifest correctly says `failed`, exit -15, rather than
complete; the parent saved all completed events before its completion
assertion failed. Runtime 741.625 seconds, peak 6,400,938,880 bytes, no
memory/disk/time guard violation. Binary SHA:
`bf06cfee89778664d4005bbc8a8dc63b51001580815d50142445deb2abc0fe6b`;
log SHA `b3a253ac963f25697880fa4255983724051de3c8ea5564ed29e703ec975d49ad`.
No larger learned-response budget was launched from these sparse counts.

### Direct endgame response experiment

The old joint-16 endgame run was stopped for rollout-evaluator cost, not
because joint-four converged (`tabular-turn-pilot.md`, pair2/pair3). Instead
of training another sparse critic at an inadequate budget, the next paired
policy experiment uses the existing exact turn/river best-response traversal.
It loads the **actual exported f32 policy**, validates the complete tree, and
computes information-set-consistent responses over every compatible private
holding, remaining river and abstract betting continuation. No retraining,
regrets or neural value estimates enter the frozen-policy evaluation.

The intended comparison is four versus 16 versus 64 joint iterations at
identical authentic turn roots, reached through the experimental sampled-flop
profile. It counts hands ending before a live turn as zero, rather than
reporting a conditional root average as bb per full hand. Both individual
seat gains and their sum/half-sum are explicitly distinguished. A response
limited to the turn/river cannot establish a full-game upper bound: earlier
deviations remain untested, and lower tail leakage need not mean lower
unrestricted exploitability.

This follows the distinction in the [primary LBR study](https://arxiv.org/html/1612.07547v2):
a legal restricted attack supplies lower-bound evidence, and delaying attacks
can expose weaknesses missed by greedy earlier deviations. Here the existing
exact endgame traversal replaces that paper's heuristic continuation in this
limited suffix; it does not make our evaluator an unrestricted best response.
No extra release gate or model activation follows from this experiment.

The bounded exact-tail pair is now launched at
`local-sampled-flop-20260905-tailpair1`: sequential original A/B sources,
sampled-flop seeds 87001/87002, common fresh prefix seed 89004, 128 full-hand
prefixes per source/solve seed. Each live turn root compares 4/16/64 joint
iterations, preserving the preflop/flop prefix policy. Source guards are
7.5GiB physical footprint, 20GiB disk reserve and 30 minutes. Frozen public
root/range snapshots and each policy hash are recorded for reuse without
another expensive prefix replay. This paragraph records a running experiment,
not completed comparisons. Both exact-response and actual-policy snapshot
regressions passed before launch. The audit takes ownership of the one-root
cache and drops each evaluated policy before allocating the next, avoiding
retention of duplicate complete endgame tables.

Pre-pilot verification: 247 release library tests and nine CLI tests pass;
ten explicit development experiments are ignored by the ordinary suite.
The release build also passes. A new explicit-path two-round production
replay in `/tmp/poker-tail-response-replay.5cN1gG` matches both previous
artifact and summary bytes. Artifact SHA remains
`c602ffbb37a2a2c8d6b787051bdafe5749ea4ba2c03905f9b133a4c7a30361d3`;
production executable SHA:
`8e52135289bd07c31c43678b532908a3fa1d6bee2202a5f5ee6fc462a9d4c787`.
Frozen exact-tail experiment executable:
`9d7aa81af625b8fd27867da79e45c46c6752810bcacce5fe48380ef07aa23cea`.
All new experiment entry points and the policy snapshot helper are test-only;
no production action mix, serving artifact, browser UI or release gate changed.

### Exact-tail result, memory-separated completion, and retained candidate

Implementation milestone `1c5241724ffbec724dcb3e897abcf4c1b132760d` was
pushed and passed CI 33997425315. The original combined-memory pair then
completed source A and source B's first solve-seed set, but **did not finish**:
the second B set hit the 7.5GiB physical-footprint stop at hand index 29.
Observed peak was 8,065,292,864 bytes versus the 8,053,063,680-byte limit.
The original manifest remains `failed`; it was not relabeled complete and
the memory budget was not raised. Source A's completed time/peak were
491.872 seconds / 8,022,546,984 bytes; B stopped at 352.298 seconds. Their
log hashes are `5a016c7dce35b27b051add9f16b718c396973860b1c85c08ccff69098a56aa0c`
and `3fe2dbc379f102c336bf555b084511f81a363747a4c9cc83658dc1cd6ab089b8`.
Frozen failed-parent manifest SHA:
`8756c599c4cf5534c8a3ec5f8df7c9573a6e254dc8ba63e518475611ea14d20c`.

`local-sampled-flop-20260905-tailrecover1` **completed the missing portion**
without rerunning the three completed sets. It first solved the saved large
A/87001/7 public root in a standalone process. All three policy hashes and
every exact response value matched the original report exactly. Peak physical
footprint was only 535,118,712 bytes, runtime 29.327 seconds. This proves
the separated calculation at that root, not an unperformed whole-run byte
comparison. No full checkpoint is loaded by the standalone endgame worker.

The recovery then collected only B/87002 hand indices 29..127 (99 prefixes),
without constructing endgame policies. A cold replay produced byte-identical
public-state/range input for the interrupted index 29. Collection took 156.785
seconds / 5,814,538,704-byte peak. Once the checkpoint process exited, two
independent workers solved the 13 remaining turn-root inputs under separate
2GiB / 180-second guards. Maximum per-root peak was 543,638,416 bytes; all
jobs passed. The full recovery took 313.778 seconds. Completed prefixes,
including those not reaching a turn, were combined exactly once with the
preserved parent results. No missing result was replaced by zero.

Final paired data: four sets of 128 authentic full-hand prefixes, two source
checkpoints times two flop-solve seeds. There are 512 prefix executions and
46 live turn-root encounters; the common 128-deal chance stream is reused
across sets, **not 512 independent evaluation deals**. Within each set,
4/16/64 endgame iterations share the identical public prefix and ranges.

The following numbers are **sums of the two seats' turn/river response gains,
in bb per full hand**, counting zero only for hands ending before a live turn.
They are not half-seat averages and not unrestricted full-game exploitability.

| Source / flop seed | Turn encounters | 4 iterations | 16 iterations | 64 iterations | 64-minus-4 individual 99% interval |
| --- | ---: | ---: | ---: | ---: | --- |
| A / 87001 | 11 | 0.408226 | 0.069611 | 0.006785 | [-0.715844, -0.087039] |
| A / 87002 | 9 | 0.282612 | 0.052651 | 0.005156 | [-0.542815, -0.012096] |
| B / 87001 | 13 | 0.376708 | 0.058354 | 0.006206 | [-0.671948, -0.069055] |
| B / 87002 | 13 | 0.431623 | 0.068045 | 0.007434 | [-0.745046, -0.103333] |

All 46 encountered roots have lower seat-summed loss at 64 than at four.
Full-hand-weighted tail loss drops 98.18–98.35% across sets. Every displayed
paired individual normal-approximation 99% interval is negative; these are
not family-adjusted intervals. An independent read-only reconstruction of all
512 prefix records verified the reported means and standard errors within
1e-12. All 15 recovery worker log hashes and all referenced old/new public
root input hashes were verified. Recovery manifest SHA:
`5a8dd26f8ceffeef832fa2d1e3b37de7b03315114672c4f7edc85180f036af16`;
frozen recovery executable SHA:
`582704c0a226bc87d8e07454c76e38a50b5b02cb6789953a49b37d458af1022a`.
The original failed run, checkpoints and completed logs remain untouched.

Retain **64 joint turn/river iterations as the next experimental endgame
candidate**, with the existing 32-iteration sampled flop and terminal weight
0.50. `profile_with_turn_iterations(table, seed, 64)` now constructs that
complete full-hand profile. The old `profile(table, seed)` deliberately stays
at four iterations to preserve archived control drivers. A subsequent response
configuration must also name 64 iterations; do not accidentally use the old
factory or describe a four-iteration evaluation as the new candidate.
Tests verify actual turn **and river** queries against the exported policy
for both four and 64 iterations, and compare the input-only public-root
collector against the actual runtime generation's input.

This is a substantial demonstrated improvement to the late-street policy,
not an unrestricted full-game winner or an Approximate GTO certificate.
Next action work should challenge the preflop/flop choices against this
stronger continuation, with a range-aware exact-card opponent rather than
another 16-hand sparse learned critic. Preserve the full-game objective;
do not spend another long run shaving an already-small tail-only number or
treat this suffix's low loss as passing the earlier-street/full-game gates.
The preflop stability, full-hand coverage, and action-EV precision gaps at the
top of this document remain unresolved. No source preflop probabilities or
browser-facing model were replaced.

Final verification after the memory split and candidate wiring: 247 release
library tests plus nine CLI tests pass; 12 explicit research probes are ignored
by the normal suite. Release build and whitespace checks pass. A fresh
explicit-path two-round production replay in
`/tmp/poker-tail-split-replay.bY9zaA` matches both prior artifact and summary
bytes, artifact SHA still `c602ffbb37a2a2c8d6b787051bdafe5749ea4ba2c03905f9b133a4c7a30361d3`.
Production executable SHA is
`4802462816308f3fd84bddf284345593036e95862a05fa2ac1c919290a95d69a`.
All sequence-owned workers have completed or were explicitly stopped as
recorded above; none remains live. The managed goal remains active.

### Exact-card all-street LBR challenge of the stronger continuation

The next attack is an online exact-card Bayesian local best response, following
the [primary LBR method](https://arxiv.org/html/1612.07547v2), rather than another
small learned critic with no supported decision groups. Its opponent belief
starts from the legal deal prior, removes visible-card collisions, and updates
only from the defender's actual observed action likelihoods. Responder actions
do not incorrectly reweight that opponent belief. Policy queries contain only
the queried actor's cards and the public board/history, never the actual other
holding or unrevealed runout. Invalid policies and impossible posteriors fail
explicitly, without a replacement mix or range.

The heuristic compares every legal action using checkdown equity and the
defender's immediate fold probabilities. Early-street equity uses 16 seeded
runouts per positive-weight opponent combo; turn equity enumerates all 44
compatible rivers and river equity is exact. The same per-combo equity vector
is reused across candidate actions. Raises are approximated as calls only in
the heuristic: actual hand play samples all legal defender actions. Recorded
heuristic action values are not served action EVs or grading confidence data.
This is a legal restricted attack and supplies lower-bound evidence, not a
full-game exploitability upper bound or a replacement release criterion.

New regressions cover Bayesian weights/card removal, exact river value bets
and losing calls, hidden-deal-independent equity samples, malformed-policy
rejection, deterministic complete hands with decisions on all four streets,
and covariance-preserving paired gain estimates. Per-hand experimental cache
clearing drops prior endgame policies and dense equity caches while retaining
the source checkpoint and cumulative diagnostics. The existing snapshot test
verifies unchanged river probabilities after clearing/reconstruction for both
four and 64 iterations. No production model or browser policy was activated.

Pre-pilot verification passes 252 release library tests and nine CLI tests;
13 explicit research probes are ignored by the ordinary suite. Release build
passes. The explicit-path two-round production replay at
`/tmp/poker-exact-lbr-replay.G72S7u` matches both previous artifact and summary
bytes; artifact SHA remains
`c602ffbb37a2a2c8d6b787051bdafe5749ea4ba2c03905f9b133a4c7a30361d3`.
Production executable SHA:
`2feccf5806edc6165937d1533203ead37224ec651580adcb0d78f9b0388b7ce5`.

`local-sampled-flop-20260905-lbr1` is launched, not yet a completed result.
It pins source A / resolve seed 87001, 32 flop iterations, terminal weight
0.50 and 64 joint turn/river iterations; LBR seed 90001, evaluation seed 90004.
Eight calibration and 24 independent raw holdout deals each run both attacker
seats against the same baseline deal. The seat-summed estimator preserves
within-deal covariance and cancels the baseline exactly. Raw holdout outcomes
remain visible even when the existing calibration rule rejects a seat; these
small normal-approximation intervals are cost-pilot diagnostics, not a
certificate. The guard is 7.5GiB physical footprint, 20GiB free disk reserve,
and 1,800 seconds. Frozen test executable SHA:
`6ae3c7064572686dfaaa2b2cab2c7b70f00eb2d1edfcfcdb032c49fe90b9d6a2`.

The cost pilot subsequently **completed** all eight calibration and 24 raw
holdout deals, both seats, in 572.819 seconds including checkpoint loading.
Peak physical footprint was 6,374,822,760 bytes (5.937GiB); no guard fired.
The supervisor independently reconstructs the paired sums and standard errors
from all completed per-hand events. Both seats failed the small calibration
block; neither is relabeled qualified based on its later raw holdout.

| Raw independent holdout metric | Mean bb/full hand | SE | Individual normal 99% interval |
| --- | ---: | ---: | --- |
| BTN/SB unilateral gain | 3.456636 | 1.426346 | [-0.217388, 7.130660] |
| BB unilateral gain | 0.991898 | 1.869497 | [-3.823607, 5.807403] |
| Sum of both gains | 4.448534 | 1.908638 | [-0.467791, 9.364859] |
| Half-sum of both gains | 2.224267 | 0.954319 | [-0.233896, 4.682430] |

Actual holdout attacker decisions by preflop/flop/turn/river were
33/7/2/0 for BTN/SB and 17/18/7/1 for BB. This attack is nonempty and exercises
the combined model, unlike the earlier unsupported learned critic. The
positive but noisy point estimates warrant an unchanged-rule independent
larger challenge, not a low-exploitability claim or a hand-fitted repair.
Keep source A/B paired and use fresh calibration/holdout chance domains.
No defender action, source checkpoint, or gate changed in this milestone.

A one-second live macOS sample at
`/tmp/poker-lbr1-policy-cost.sample.txt` captured a normal turn/river solve.
It does not establish that repeated query overhead is a bug. The diagnostic
skill's measurement-first loop therefore did not justify a speculative
performance rewrite; any optimization needs a repeatable isolated cost/parity
check first. The completed pilot worker and supervisor have both exited.

The completed LBR implementation/result milestone is committed and pushed as
`0097deebad92309067ac9298e4a500378e3bbf24`. Its first push timed out; an
authoritative remote check showed the old head, and the subsequent push
succeeded. Cost-pilot final manifest SHA:
`0a842a443d046730a295c3c8b331494314ff440c50f6d75b36bf1bbaffdfc0ad`;
log SHA `92188f3a73620ecec969dcd1ea71b0e3780c7f9139365acbffe934556602f4f6`.

The next independent challenge keeps the same LBR seed/rule, defender profile,
and calibration threshold, increasing to 64 calibration / 128 raw holdout
deals per source with fresh evaluation seed 91004. Sources A/B run
sequentially; common chance across sources supports paired comparisons but
does not double the independent-deal count. The original 8/24 cost entry
point retains its original seed and budgets. The new guarded driver is
`local-sampled-flop-20260905-lbrpair1/run.py`, with a 7.5GiB physical-footprint
limit, 20GiB disk reserve and 5,400-second cap per source. Expected pair cost
from the small pilot is roughly 1.5–2 hours, not a completion guarantee.
This paragraph records the planned independent challenge; consult the live
manifest for launch/completion and do not describe it as a quality pass.

The pair is now **launched**, with source A verified live first; the supervisor
starts B only after A completes and passes its event-integrity checks. Frozen
test executable SHA:
`f5a3c9775e6827c9d2c5f383ebc879688f58a70a0ef0fa8c5c1dcc9fd111e50c`;
runner SHA `5ca111c0a61274480cfcdcd75572f1fb0bbce9eafcce9fc9fe4f9c9bdd7f4688`.
The refactored driver passes all 252 library and nine CLI release tests;
14 explicit research entries are ignored in the normal suite. Milestone CI
34000465744 is in progress, not yet a recorded pass. No pair results are
available at this launch checkpoint. The managed goal remains active.

### Reuse terminal-flop public likelihoods without changing the policy

CI 34000465744 for `0097dee` subsequently passed. The independent A/B pair
continues on its original immutable executable; none of the following cache
changes enters that running experiment.

The diagnosing-bugs loop reproduced repeated terminal-flop range work with
`cargo test --release terminal_panel_reuses_public_likelihoods_across_hero_holdings -- --nocapture`.
The test queries the real terminal-correction path for two different hero
holdings on the same public line. Before the fix it failed twice: the second
holding repeated 2,162 public policy likelihood queries, although only 182
entries could be newly unblocked (91 at each of two opponent decisions).
The minimized execution took 0.016732 / 0.015882 seconds, without loading a
large checkpoint. These are observations, not isolated performance estimates.

A bounded cache now memoizes the selected action's likelihood for each exact
opponent combo/public board/history. It does **not** memoize a hero-conditioned
range, reorder its accumulation, alter blocker removal or normalization,
change the equity samples, or substitute probabilities. Lazy cells only query
entries the original code would request. Each patch owns its cache; weak table
and optional backoff identities prevent cross-source reuse without retaining
checkpoints. A differently configured caller follows the uncached path. The
32-row limit holds about 0.7MiB of cells plus keys; eviction preserves live row
references, and no cache mutex is held during nested policy queries.

The same regression passes with 2,162 cold queries and **182** second-holding
queries (91.6% fewer for that second holding), observed total 0.009307 seconds.
The deterministic work-count improvement is established; a complete LBR
runtime speedup is not yet measured. Additional tests compare cached and
uncached normalized ranges with exact equality for both seats, four boards,
and both terminal-only and sampled-flop policies. Cold/warm action mixes match
exactly. Cache tests verify board/action/source separation, bounded retention,
live-row eviction safety, and no extra strong checkpoint owner.

Final local verification: 255 release library tests and nine CLI tests pass,
14 research probes ignored; release build and whitespace checks pass. The
two-round production artifact and summary at
`/tmp/poker-terminal-cache-replay.6ExvUJ` match the previous replay bytes;
artifact SHA remains
`c602ffbb37a2a2c8d6b787051bdafe5749ea4ba2c03905f9b133a4c7a30361d3`.
Production executable SHA:
`aa6bf3c5e983efa93aae5baa011fd86949e5f41dbbeb8d82d1eea6c5aa83c3e1`;
test executable SHA:
`4aabb3880ab21c75024554c3548faf4f8852b97d96f26c4d763f3dbef8b24df6`.
This small production replay does not exercise the full-size LBR adapter.
Once the original pair completes, replay the original 8/24 cost pilot with
this optimization and compare all decision/action/value outputs before
claiming whole-evaluation parity or speedup. Keep only one full checkpoint
loaded at a time. Concurrent local builds affect running-pair wall times.

Source A's completed 64-hand calibration in the ongoing pair is unqualified:
BTN/SB gain -1.018324bb (SE 1.202154), BB -0.035743bb (SE 0.821065),
half-sum -0.527033bb (SE 0.651692). Preserve the raw independent holdout and
finish both sources; this is neither a complete pair nor zero exploitability.
The original positive 24-hand point estimate has not been replicated by this
calibration block. Do not tune the running attack to it.

If the completed all-street attack remains uninformative, the primary
[LBR experiments](https://arxiv.org/html/1612.07547v2) motivate a separately
frozen delayed-attack pilot: greedy early bets can miss later exploitable
decisions. That is a candidate next diagnostic, not evidence of a weakness
already isolated here, and its gains would still be lower-bound evidence.
No defender policy change or extra release gate is introduced by that option.

### Prepared delayed-attack control; original pair still frozen

Cache milestone `eec4b16f8ddc068617ea766b2dfe0616f8aab2aa` is pushed and
passed CI 34001386919. The full-size cache parity runner is prepared at
`local-sampled-flop-20260905-lbrcacheparity1/run.py`, with frozen executable
`4aabb3880ab21c75024554c3548faf4f8852b97d96f26c4d763f3dbef8b24df6`.
It refuses to start before the original paired workers finish, verifies the
old cost-pilot manifest/log hashes, and compares every semantic event exactly,
excluding only time and work-count diagnostics. This has not run yet; do not
claim full-size parity or speedup from the prepared script.

The separate delayed-attack implementation follows the baseline policy until
the flop, then uses the same exact-card LBR rule. Unlike the cited paper's
forced early check/call experiments, this adaptation preserves the authentic
preflop distribution and makes preflop-terminal hands exact paired identities.
It does not condition the opponent belief on the hero's baseline actions.
The first public street on which attacks begin is explicitly reported for
the new variant; archived all-street entry points retain their original
budgets, seeds and event semantics. A disabled-attack regression verifies
exact baseline payoffs and random-draw alignment; the delayed regression
verifies identical preflop histories, only postflop attacks, deterministic
replay, and unchanged all-street wrapper behavior. An initial test import-path
compile error was corrected before the tests ran.

The prepared ignored entry is
`lbr::pilot::delayed::sampled_profile_delayed_lbr_pilot`: 32 calibration / 64
independent raw holdout deals, fresh evaluation seed 92004, unchanged LBR seed
90001 and defender profile. This is not launched while the original pair or
its subsequent cache parity is outstanding. Do not tune it to partial holdout
outcomes or treat baseline-identical preflop terminals as rejected-critic
zeros. All 256 release library tests and nine CLI tests pass after the helper
change (two test threads to limit local pressure); 15 research entries are
ignored. The original full-size worker is still running its older frozen
binary; these changes cannot alter its action rule or results.

The delayed prototype's tested executable is frozen at
`local-sampled-flop-20260905-lbrdelay1/frozen-tests`, SHA
`9e0ed687bf9a583c00aa001069ece73272cf2e516395eb3c74d9498b7675c925`.
There is no delayed full-size worker or result yet. Its production executable
is byte-identical to the cache milestone (`aa6bf3c5...`); the new attack path
is test-only.

### Independent all-street challenge: source A complete, B running

Source A completed 64 calibration and 128 raw holdout hands, both seats,
in 2,070.010 seconds including load, peak physical footprint 6,378,869,608
bytes, no resource stop. Its worker exited before source B was launched.
Source-A immutable log SHA:
`fff3cb2730b54c34f49a46ba6e502cb0ecf440b988eb89578c1ca796f8bd1c05`.
Both seats remained unqualified by calibration. Raw independent holdout:

| Metric | Mean bb/full hand | SE | Individual normal 99% interval |
| --- | ---: | ---: | --- |
| BTN/SB unilateral gain | -0.102528 | 0.648449 | [-1.772821, 1.567765] |
| BB unilateral gain | -0.024061 | 0.697207 | [-1.819948, 1.771826] |
| Seat sum | -0.126589 | 0.852173 | [-2.321641, 2.068464] |
| Half-seat sum | -0.063294 | 0.426087 | [-1.160821, 1.034232] |

Attacker decision counts were 171/31/4/1 (BTN/SB) and 71/48/11/1 (BB),
ordered preflop/flop/turn/river. An independent read-only verifier checked all
192 hands / 384 seat records, each paired sum and baseline cancellation,
finite/bounded heuristic values, selected-action argmax, payout bounds, and
means/SEs against the report within 1e-12. It also verified the completed log
hash. This does not make the heuristic action values true action EVs.

The small pilot's positive point estimate did not replicate on this source's
larger independent sample. The unchanged defender has not improved or worsened
because its evaluator received more samples. The restricted attack remains
uninformative; no exploitability gate passes from its near-zero/negative raw
mean, and rejected-response zeros remain prohibited. Source B is verified
live under the original supervisor and frozen executable. Finish it before
the cache parity replay, then assess the separately frozen delayed-attack
pilot. The combined pair is not yet complete; the managed goal remains active.

### Completed independent pair; delayed-attack preparation

Source B subsequently completed its 64 calibration and 128 raw holdout hands,
both seats, in 1,907.693 seconds including load; peak physical footprint was
6,303,798,072 bytes. Neither source hit a resource stop. The original workers
and supervisor have exited; the combined pair completed in 3,979.507 seconds.
Source-B immutable log SHA:
`f1fcdc73b94f30cd6c7cd6f447014b25a391826e16dae176313d29d14c9b1b27`;
completed pair manifest SHA:
`e89bb6f02d2da90c4d7597f4bd17b622880719de9f43028387fa2527e104f86d`.

B calibration remains unqualified for both seats: SB -0.837521bb
(SE 1.100612), BB 0.453238bb (SE 1.146704). Raw independent holdout:

| Metric | Mean bb/full hand | SE | Individual normal 99% interval |
| --- | ---: | ---: | --- |
| BTN/SB unilateral gain | 1.168938 | 0.738323 | [-0.732857, 3.070733] |
| BB unilateral gain | -0.354778 | 0.602139 | [-1.905786, 1.196230] |
| Seat sum | 0.814160 | 0.860350 | [-1.401954, 3.030273] |
| Half-seat sum | 0.407080 | 0.430175 | [-0.700977, 1.515137] |

B holdout attack decisions were 174/32/6/1 (SB), 83/39/14/8 (BB), ordered
preflop/flop/turn/river. An independent read-only audit verified both log
hashes, runner/binary hashes, all 384 hands / 768 seat records, baseline
cancellation, finite and bounded heuristic values, selected-action argmax,
and every summary mean/SE within 1e-12. It also checked exact common deal and
action-seed records across A/B. The shared holdout stream contains **128
independent deals**, not 256. Both sources failed attacker calibration, and
all individual raw gain intervals include zero. The old cost pilot's large
positive point estimate did not become a reproducible positive lower bound.
No exploitability gate passes and no defender policy changed in this pair.
Do not spend another unchanged large all-street attack run on these results.

The prepared full-size cache replay is now launched against the original
8/24 cost probe, after verifying the completed pair and absence of its live
workers. Its result is still pending; exact semantic parity and whole-run
speedup are not yet established.

The delayed pair's guarded runner is prepared and syntax-checked at
`local-sampled-flop-20260905-lbrdelay1/run.py`, SHA
`da12bf7f4cf74d80ada7a84bcbc627d2cf04edf9104eea6abc92ce10cacd3c59`.
It requires completed original-pair and cache-parity manifests, refuses live
full-size experiment workers or an existing output, pins the tested executable
and both source checkpoints, and runs A/B sequentially. Each source has a
1,800-second / 7.5GiB physical-footprint stop and 20GiB disk reserve. It verifies
the delayed scope, exact configuration, unique legal deals, complete paired
records, baseline cancellation, action argmax, absence of preflop attacks,
unchanged payoffs without intervention, summary estimates and common A/B
chance. Preparing this runner is not a delayed-pilot launch or quality result.

### Research boundary for eventual full-game qualification

The [dynamic zero-sum information-relaxation analysis](https://arxiv.org/html/1405.4347)
explains why fixing an imperfect-information opponent produces a POMDP best
response, and why a legal approximate response supplies the wrong bound
direction for certifying low exploitability. Dual-feasible information
penalties can bound the optimum from above instead. The
[POMDP treatment](https://martin-haugh.github.io/files/Research/POMDP_IR_March_2019.pdf)
constructs such penalties from conditional expectations and requires solving
the relaxed inner optimization; it explicitly warns that direct belief-state
inner problems are generally intractable beyond small cases. This is not a
demonstrated practical Hold'em certificate. Applying it here would require
valid conditional expectations, justified inner-solve bounds, small-game
verification and a measured full-size cost; heuristic values cannot simply be
subtracted from the existing clairvoyant evaluator and called a valid bound.
No such evaluator is implemented or added as a new gate in this milestone.
Finish cache parity and the narrowly scoped delayed pilot before considering
a different evaluation architecture.

### Full-size cache parity passes; delayed pair launched

The cache replay completed all eight calibration and 24 holdout hands, both
seats, with **195/195 semantic events exactly equal** to the original probe.
This includes configuration, exact deals and seeds, baseline payoffs, selected
actions, entire histories, heuristic values, gains, summary estimates and
calibration flags; only elapsed time and diagnostic work counts are excluded.
An independent read-only audit rechecked every event, completed log/binary/
runner hashes, and the pinned prerequisite pair hash. Parity manifest SHA:
`dfb71148eea06e8e4e614e02469f0f373155bb2da8322da9e2b1c8ad5f1a535e`;
log SHA `4e9fc9cdfed0dfb237640037c9ba123947fd31305996352cd7dd74b09e4db498`.

Observed worker time including load is 297.519 seconds versus 572.819 seconds
before caching (48.06% less time, approximately 1.93x throughput for this
single replay). Peak physical footprint is 6,349,017,936 bytes versus
6,374,822,760 bytes; neither run hit a stop. These wall times are not an
isolated repeated benchmark: the original run overlapped local builds.
The exact semantic parity and deterministic duplicate-query reduction are
established, but the whole-time difference cannot all be attributed to the
cache from this comparison alone. This changes cost, not policy quality.

The earlier completed independent pair and tested delayed-attack milestone
are committed and pushed as `372a4727ac1ded207077cc9a7850ce477de3a6d7`.
CI 34003540542 is running, not yet a recorded pass. After the parity worker
exited, `local-sampled-flop-20260905-lbrdelay1/run.py` launched source A with
its already frozen executable and limits. No delayed result is available
yet. Source B may start only after A finishes and its records pass integrity
checks. No model has been activated or relabeled.

### Completed delayed postflop pair and next evidence-driven action

CI 34003540542 for `372a472` passed. The delayed pair subsequently completed
both sources without a stop: A 338.366 seconds / 6,362,141,544-byte peak,
B 502.788 seconds / 6,340,039,504-byte peak. Total pair time was 842.861 seconds.
Both workers and their supervisor exited. Completed manifest SHA:
`b7571c3cc882dadce2ad2f8e2f5fc97705269d181f9cb49e5fa07ec626c84ea5`;
A log `8e1892412625744e4185c17ba99cf7fd177ee40f537c96cbda70a716135d4c3c`;
B log `897d70689bbaf27ed8cfe63e072134347056499a07aa2f3ee11e8f3a847b5dd2`.

Both seats in both sources failed the existing calibration rule. Calibration
contained only three hands with a postflop attack for A and six for B, out
of 32 authentic hands each. The independent holdout contained 11 attacked
hands per source out of 64. Preflop-terminal hands are exact baseline
identities, not rejected-response values replaced by zero.

| Source / raw holdout metric | Mean bb/full hand | SE | Individual normal 99% interval |
| --- | ---: | ---: | --- |
| A BTN/SB gain | 0.921870 | 0.599993 | [-0.623609, 2.467349] |
| A BB gain | 0.204551 | 0.655433 | [-1.483734, 1.892835] |
| A seat sum | 1.126420 | 0.551878 | [-0.295123, 2.547964] |
| A half-seat sum | 0.563210 | 0.275939 | [-0.147561, 1.273982] |
| B BTN/SB gain | 0.247085 | 0.341599 | [-0.632815, 1.126985] |
| B BB gain | 0.720013 | 0.464044 | [-0.475286, 1.915312] |
| B seat sum | 0.967098 | 0.459511 | [-0.216523, 2.150719] |
| B half-seat sum | 0.483549 | 0.229755 | [-0.108262, 1.075360] |

Holdout attack decisions by preflop/flop/turn/river were A: SB 0/11/1/0,
BB 0/14/1/0; B: SB 0/11/4/3, BB 0/12/1/1. The positive point estimates are
directional evidence only: all intervals include zero, neither source has
a qualified attacker, and these are not upper bounds. In particular B's
0.483549 point estimate does **not** pass the 0.50 full-game gate. The two
sources share the same 64 independent holdout deals; do not pool them as
128 independent hands or compare this new chance stream to the all-street
pilot as a paired policy improvement.

An independent read-only audit checked all 192 hands / 384 seat records,
both logs, runner/binary and prerequisite hashes, the complete common-chance
stream, baseline cancellation, absence of preflop interventions, zero gain
without any intervention, bounded finite values, selected-action argmax, and
all summary estimates within 1e-12. This establishes experiment integrity,
not equilibrium quality. The defender remains unchanged.

The saved development traces identify concrete decisions to replay next.
For example B holdout index 3 / seat 0 has own cards `[34,21]`, flop
`[27,2,9]`, limp/check preflop, then BB shoves 19bb into 2bb; the attacker
folds. A index 29 / seat 0 shoves after limp/check and a flop check on
`[30,35,4]`. A/B index 62 / seat 1 shoves an 18bb-pot flop on `[8,42,32]`.
Their large realized paired gains are not exact counterfactual action EVs.
Treat these now-inspected outcomes as development inputs only; never recycle
them as untouched validation or infer a policy repair from winnings alone.

The diagnosing-bugs discipline currently pauses speculative policy edits:
there is no minimized, fast, red-capable reproduction proving the source of
these remaining gains. The next concrete action is to replay the captured
terminal flop fold/call case, preserve its actual public posterior and policy
mix, and measure conditional action values with controlled/exact runouts.
If expected loss is reproduced, minimize that real path before a scoped policy
change, then evaluate the **combined full-hand** candidate on fresh paired
deals. If it is only realized chance noise, do not manufacture a fix. The
known mismatch between local flop-training continuation and served endgame
is not yet a proven cause, and the already-small turn/river suffix metric
does not justify another tail-only optimization loop.

Additional primary research considered
[DCFR+ and predictive DCFR+](https://arxiv.org/html/2404.13891v2).
The paper reports DCFR+ strongest on its large HUNL subgame, while predictive
variants excelled in other settings. Its regret clipping/discounting is a
possible separate optimizer experiment, not evidence of improvement in this
sampled solver. The full-tree experimental findings are not a sampled-CFR
performance guarantee; no optimizer, checkpoint schema, or extra gate was
changed here. Prioritize the captured policy-action diagnosis before adding
another optimizer or neural architecture.

Current gaps remain the preflop stability/precision figures at the top of
this document, full-hand coverage qualification, and a defensible full-game
exploitability bound below 0.50 then 0.05bb/hand. No unvalidated activation,
paid compute, gate relaxation, or additional long training run occurred.

### Reproduced terminal bad calls; explicit full-weight candidate

CI 34004697297 for `06132fe` passed. The diagnosing-bugs loop now has a
concrete reproduced policy-action loss, not just a noisy large winning hand.
The development capture replays delayed-pilot source B / holdout index 3 /
seat 0, reconstructing the actual opponent posterior from the original
20bb/800-round checkpoint and current 32-flop/64-turn-river profile. Its
16-runout LBR values exactly match the archived trace `[-1,-9.78626961397315]`.
It then enumerates all 990 legal remaining boards for each compatible holding:
1,070,190 showdowns, with no hidden actual opponent cards or future board
used to choose an action.

The full-checkpoint measurement and a minimized 339-row frozen-average subset
match **exactly** in action mix, posterior, sampled LBR values and enumerated
values. The compact MessagePack fixture is 159,594 bytes, contains no regrets
or resumable state, and is retained under `preflop-solver/tests/fixtures/` with
provenance. Fixture SHA:
`4caf41a368b79a80ebdf576e792ed4cd5d1cc0203bde709428018904d0842dc9`.
The source terminal row has seven averaging contributions and equal sums
`[234.47321012757615,234.47321012757615]`; it is an existing 50/50 row, not
an absent-node lookup in this particular case.

`local-sampled-flop-20260905-terminalcapture1` completed in 69.996 seconds,
peak physical footprint 5,806,887,400 bytes, no resource stop. Its frozen
executable SHA is
`9555db8559ed3c606af47c226c15778475ec8cd442393a0e218eee2dc98011d4`,
runner SHA `a51d767210636af63f5ae411fd9906086d76ceff1640ee9d5533f41edb20e458`,
and log SHA `ce55cbcf90f5e0321fcf6789664abe4b82df5a2ceb93fec8e08ec91425510caf`.

The fast, unattended red command is the captured frozen executable's
`terminal_capture::replay_terminal_flop_conditional_loss` test, with
`POKER_TERMINAL_CAPTURE_DIR` pointing at the capture directory. It failed
identically twice (2.215 / 2.189 seconds), reporting conditional loss
**2.2327421425686964bb**, beyond the existing 0.05bb decision tolerance.
Fold EV is -1bb, enumerated call EV is -9.930968570274786bb, but the composed
policy folds/calls 75%/25%. This is a conditional decision value against the
frozen public posterior, not a per-full-hand exploitability estimate.

Three ranked hypotheses were declared before the controlled probes: the
blend retains bad calls; the corrector and evaluator disagree about ranges;
or equity sampling changes the selected action. Results on the same fixture:

| Equity samples | Blend weight | Corrector selection | Fold/call mix | Conditional loss, bb |
| ---: | ---: | --- | --- | ---: |
| 128 | 0.5 | Fold | 75% / 25% | 2.232742 |
| 2,048 | 0.5 | Fold | 75% / 25% | 2.232742 |
| 16,384 | 0.5 | Fold | 75% / 25% | 2.232742 |
| 2,048 | 1.0 | Fold | 100% / 0% | 0 |

The actual corrector posterior agrees with the independent Bayesian replay
to maximum absolute difference 2.0816681711721685e-17. The corrector already
selects fold at every tested sample budget. Thus the retained partial blend,
not a misclassified equity sign or mismatched range, explains this loss.
The exact same replay with explicit full weight passes in 2.274 seconds.
The frozen old executable remains red; no old artifact or gate was changed.

A normal regression now checks both control and explicit candidate through
the real composed policy path. All **257 library and nine CLI release tests**
pass; 20 explicit development/research entries are ignored. Release build and
whitespace checks pass. Production executable SHA is
`5ca2ad865f6cf4686b22d163965bf6e64b94564aff49079575480fc90651841e`.
The explicit two-round production replay at
`/tmp/poker-terminal-candidate-replay.Jm0tRQ` preserves both artifact and summary
bytes; artifact SHA remains
`c602ffbb37a2a2c8d6b787051bdafe5749ea4ba2c03905f9b133a4c7a30361d3`.
This tiny production check does not certify the full-size research candidate.

The parameter seam is test-only: archived profile factories still use
terminal weight 0.5 / 2,048 samples, and their event metadata remains correct.
Only explicit candidate entries request weight 1.0. The terminal correction's
existing confidence abstention remains unchanged. Earlier full-weight work
already improved the older profile against fixed opponents; it did not
qualify an exploitability upper bound and was not the default in the newer
sampled-flop/64-turn-river combination. This is a controlled integration
comparison on that combination, not a new optimizer or claimed Nash fix.

`local-sampled-flop-20260905-terminalweightpair1/run.py` is now launched with
frozen executable
`6e5fb3b00e0f2bdcca1a12f135b76dbc436654a72e9b601f63c66e0f3a1771cf`
and runner SHA
`7af92aad2fb35c0404e2d83ff7c81017bc066262cf5fe53c1f8631ff4f520fda`.
First it verifies the candidate against the original full checkpoint. Then
it compares control/candidate sequentially for both source seeds: 32
calibration and 96 raw holdout hands per variant, fresh seed 93004, the same
exact-card delayed-LBR rule, 32 flop iterations and 64 turn/river iterations.
Only terminal blend weight changes. All variants share chance for paired
deltas, not extra independent hands. Each full-size worker has the existing
7.5GiB / 20GiB disk-reserve guard and a 1,800-second challenge cap; the initial
single-case replay has a 600-second cap. Prior inspected cases are not reused
as validation deals. This paragraph records launch, not completed full-game
results. No research candidate or website model has been activated.

The initial full-checkpoint replay subsequently passed in 70.464 seconds,
peak physical footprint 5,838,442,960 bytes, no stop. It reproduces the old
75/25 mix and 2.2327421425686964bb conditional loss, and the explicit candidate's
100/0 mix with zero conditional loss, against the same enumerated posterior.
Full-replay log SHA:
`ddb386449d00dc0846b21823a62b30fa62876fda1c565c017f7745a81449df67`.
The supervisor then launched source A's fresh control challenge; paired
policy-quality results are still pending. This confirms the original
unminimized decision replay, not a full-game exploitability improvement.

### Terminal-action estimator prepared; inference memory census

Milestone `50aca613b60e4d5ddad2b6bd76b9ab5d75e56235` is pushed and its remote
CI 34006103776 passed. The fresh terminal-weight pair remains frozen while a
separate development replay is prepared. This replay does not rerun an attack,
select different actions, change a defender, or replace an original result.
It reconstructs each archived terminal history, verifies its original payout,
and, only when the final defender flop decision has two terminal actions,
integrates both outcomes using the actual frozen policy probabilities and
the existing exact legal-runout evaluator. Eligibility is determined before
the sampled action; folds with a nonterminal alternative and the attacker's
own actions are not substituted. Strategy queries still receive only own
cards and the visible board. Invalid histories, payout mismatches, malformed
probabilities, and observed zero-probability actions fail closed.

This is ordinary terminal-action Rao-Blackwellization, consistent with the
known-strategy/terminal-observation principles discussed by
[Burch et al.](https://poker.cs.ualberta.ca/publications/aaai18-burch-aivat.pdf),
not a complete AIVAT implementation. The
[Kim/Sandholm heuristic-pathology paper](https://arxiv.org/html/2605.14261v1)
warns against fitting a correction on the same evaluation outcomes. Here no
value model, coefficient, inverse-variance weight, or significance target is
fitted: terminal payouts are determined by the game rules. The original raw
results remain available. Removing individual action noise does not guarantee
a smaller *paired* standard error; that must be measured. Reanalyzing these
same deals adds no independent evidence and cannot turn a restricted response
into an exploitability upper bound or override its original calibration.

The two normal regressions verify both seats, exact weighted-mean preservation,
zero expected correction across both terminal actions, independence of hidden
future cards, exclusion of noneligible folds/attacker actions, and rejection
of invalid records. All **259 library and nine CLI release tests** pass;
21 explicit research entries are ignored. The release build passes and the
production executable remains byte-identical:
`5ca2ad865f6cf4686b22d163965bf6e64b94564aff49079575480fc90651841e`.
These are research-only source changes, with no browser or serving change.

Prepared but not yet launched: `local-sampled-flop-20260905-terminalmarginal1`.
Its frozen test executable SHA is
`ee379af44b80393bee57109a5e3c3492f3ad26247b11b394427db67252e49e40`;
runner SHA at preparation was
`534dda15d1ef8ed745e8efa15eb16b2bd2b0e3396ca522f5ac3d583f8b89e6a8`.
The runner requires a completed, hash-pinned original pair, checks original
log/source hashes, loads one full checkpoint at a time, and retains the
7.5GiB / 20GiB disk-reserve guards with a 900-second cap per source.
It will report original and marginalized seat-summed outcomes separately,
including covariance-preserving per-deal control/candidate differences.

A separate read-only streaming census completed both original checkpoints
without materializing another inference table or altering any artifact.
It pins the original source hashes and its counts match both original
800-round summaries. Results in `local-sampled-flop-20260905-inference-census1`:

| Source | Preflop rows | Flop rows | Turn rows | River rows | Peak physical bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 16,900 | 3,884,147 | 7,441,993 | 7,721,262 | 19,644,824 |
| B | 16,900 | 3,817,540 | 7,413,434 | 7,623,642 | 20,054,424 |

Runtime was 48.712 / 48.955 seconds, each below its 300-second / 256MiB
streaming limits. Census SHA:
`8ab4e8726b7b532b0eaca400da2d739ecada541b64bd15923bac15f9ce2d2251`.
Although roughly 79.5% / 79.7% of rows are on the later streets, they cannot
simply be removed: `TabularTurnPolicy::resolved_strategy` still uses original
source probabilities for forced zero-entry hands, and the safe arm needs its
original anchors. No filtered model, invented completion, storage deletion,
or claimed inference-parity result follows from this census. Preserving that
exception is necessary before any future memory-saving representation could
enable more parallel policy trials. This investigation adds no release gate.

### Completed fresh terminal-weight pair; exact replay launched

`local-sampled-flop-20260905-terminalweightpair1` completed every planned job
without a resource stop. The source A/B control/candidate workers took
475.229 / 505.657 / 1,237.152 / 1,237.619 seconds. Their peak physical footprints
were 6,358,733,672 / 6,357,947,240 / 6,366,286,672 / 6,339,122,000 bytes.
Whole-pipeline time including the initial full-checkpoint case replay was
3,530.467 seconds. Source B traversed more expensive later-street lines;
these timings are not an isolated hardware or estimator benchmark.

| Source | Control holdout half-seat gain | Candidate holdout half-seat gain | Paired candidate-minus-control (SE), bb/hand | Individual normal 99% interval |
| --- | ---: | ---: | --- | --- |
| A | 0.096645 | -0.163403 | -0.260048 (0.176149) | [-0.713777, +0.193680] |
| B | 0.562916 | -0.119292 | -0.682208 (0.280931) | [-1.405839, +0.041424] |

These are **half the two seats' summed restricted-attack gains**, not an exact
Nash gap or certified full-game exploitability. For seat sums, multiply both
means and standard errors by two. Both sources' point estimates favor the
full-weight correction, but both paired intervals cross zero. Calibration
rejects both seats for all four profiles; the raw negative candidate gains
do not certify zero exploitability. Of 96 holdout hands, source A changes
five seat sums (four lower, one higher); B changes ten (nine lower, one higher).
This is encouraging directional evidence with a small affected sample, not
a qualified global winner or a reason for a longer unchanged training run.

The independent read-only audit checks all 512 hand executions / 1,024 seat
records, exact deals and action/chance seeds, configurations and scope, original
source/log/binary/runner hashes, legal bounded action-value argmax, baseline
cancellation, no-intervention identity, stage counts, calibration flags,
means, SEs and paired intervals. All pass. There are **96 independent holdout
deals**, reused across sources and variants, not 384. The audit initially
exposed its own parser assumption: Rust's harness prefixes the first JSON
event with the test name. Reading from the opening brace preserves that event
and makes the complete log/manifest comparison pass. No original log, runner,
result or statistical estimate was edited to fix this audit issue.

Completed pair manifest SHA:
`c9d5b168fd3b472b602c4b4dfea9794f2eaac0e54fedc8fd99cfc6d3c2b7eeda`.
Worker log hashes (A control, A candidate, B control, B candidate):

- `2413901aa36d60dcb85875fdcfc6a1bed0839e986183f63354e282b1ac444d1d`
- `58ebd1d87cdad8187626d82a7d6d3df09a111765a121019a695cf13ad354fd55`
- `c3b0b1ac06c9964302ea974e33e20497e5d422c242b8f0d0df3d048e0836d4db`
- `e7e91893c6d6bdce0f24b72b529a88e5cdcce7c6b7fc7061ea1ee0290b3b2867`

After confirming the original supervisor and all its workers had exited,
`local-sampled-flop-20260905-terminalmarginal1` was launched against that exact
completed manifest. Before launch, its parser was also corrected to retain
Rust's first prefixed JSON event. The actual frozen runner SHA is therefore
`5d5bb2889ba19bbc26cf8702e45750c8cba1a5ac2e6a86cc305a78e9ad542a51`;
the executable remains the previously recorded `ee379af4...`. This is a
separate estimator replay with both original policies and all old records
preserved, not a fresh validation sample or additional training. Results are
pending at launch. The existing preflop consistency, action-EV precision,
full-hand coverage and full-game exploitability requirements remain open.

### Completed exact terminal-action replay

`local-sampled-flop-20260905-terminalmarginal1` is now **complete**, with no
live worker or resource stop. A/B workers took 123.160 / 162.498 seconds,
with peak physical footprints 5,797,564,928 / 5,736,927,696 bytes. Total
pipeline time was 287.327 seconds. This reused all existing trajectories;
it did not rerun the expensive LBR searches or create new evaluation deals.

| Source | Marginalized control half-seat gain (SE) | Marginalized candidate half-seat gain (SE) | Paired candidate-minus-control (SE) | Individual normal 99% interval |
| --- | --- | --- | --- | --- |
| A | +0.174586 (0.126540) | -0.163403 (0.195696) | -0.337989 (0.149054) | [-0.721926, +0.045948] |
| B | +0.334385 (0.125366) | -0.078664 (0.123265) | -0.413050 (0.131560) | [-0.751925, -0.074174] |

All table values are bb/full-hand **half-seat sums**. Source A integrates
12 control / seven candidate holdout terminal decisions; B integrates 18 /
eight. Candidate terminal choices are often already deterministic, explaining
why removing their final action draw can leave an outcome unchanged. Paired
SE falls from 0.176149 to 0.149054 for A and from 0.280931 to 0.131560 for B.
Both mean differences still favor full weight. B's individual interval is now
entirely negative; A's still includes zero. This is a development reanalysis
of the same 96 shared holdout deals, **not an independent confirmation**.
Original calibration flags stay rejected, and no negative response payoff is
converted to zero exploitability. No full-game upper bound has been obtained.

An independent read-only audit reconstructs all 1,024 seat assessments and
512 hand sums, matches original observations, verifies exact action weighting,
player signs, eligible/unchanged records, stage counts, original input and
output hashes, means, standard errors, and covariance-preserving paired
intervals. All pass within 1e-12. JSON reparse/serialization can change an
archived floating payout by at most 1.7763568394002505e-15 in these records;
this was explicitly checked, not treated as a strategy change. Original
artifacts and original statistical results remain immutable.

Replay manifest SHA:
`af2194880cd9423789b2e32b6f9ac4db6f033d0bdcf9ab6dd377fc20df4ac3a3`.
A/B log hashes:
`9acb4947ab5a14d858fa9786e9ca4f8d9b6bc603cf7db59b30f02d360f645837` /
`220ae01d0ce5a7856223749bd65c52c74757e027f8785d86f403c799ca99a2e8`.

Disposition: the full-weight terminal correction remains a promising explicit
research candidate, not an activated model or a proven global winner. The
concrete bad-call reproduction is fixed by the candidate, and both paired
sources support the same direction, but a fresh confirmation must freeze the
terminal-action estimator before seeing its data. Do not repeat the old
32-vs-128 flop-iteration experiment, declare approximate GTO from this LBR
result, or substitute these payoff SEs for the separate action-EV grading
precision requirement. No new release gate or paid compute was introduced.

### Shared-table parallel confirmation: full-size parity passed

`local-sampled-flop-20260905-terminalparallel1/parity` completed both original
sources. The new research-only `pilot/confirmation.rs` runs the control and
candidate in two threads, each with its own policy and caches, sharing one
immutable full inference table. Source A/B still run sequentially; no source
rows are discarded and no second full checkpoint is resident. Both workers
retain the original per-hand cache clearing and chance/action streams. A
worker error stops the panel without inventing a replacement result.

The small-game regression compares all serial and parallel semantic records,
including actual eligible terminal integration, and checks cooperative failure.
The full-size parity stage then uses the archived evaluation-seed-93004
prefix: eight calibration and 24 holdout deals per source and variant. All
raw baselines, attack actions/histories, cards/seeds, and exact terminal
assessments reproduce the original serial pair and marginal replay within
1e-12 (discrete fields exactly). Calibration is recomputed on the shorter
eight-deal prefix, not copied from the original 32-deal calibration.

| Source | Worker seconds | Sampled peak physical bytes | Serial parity |
| --- | ---: | ---: | --- |
| 26001 | 156.871 | 6,155,015,016 | pass |
| 26002 | 185.258 | 6,594,745,480 | pass |

Total pipeline time was 343.866 seconds. Both workers exited successfully
without reaching the 900-second, 7.5GiB physical-footprint or 20GiB free-disk
reserve stops. An independent read-only audit rechecked all 128 hand executions
and 256 seat records against the original serial artifacts, log/source hashes,
common deals, terminal sums, and paired means/SEs/99% intervals. There are only
24 shared holdout deals here; this is implementation parity, not new policy
quality evidence or an isolated serial-versus-parallel speed benchmark.

Frozen executable SHA:
`bb00d8cb4694c1acea5060955469ec53ae46d1ebe321de36ed0c3f2a462ace74`.
Frozen runner SHA:
`1500ba6a2eddef17ff5268a0f075fc7dd3519e7d4fb553258d446db5782b32ba`.
Completed parity manifest SHA:
`be7a6961c8019d0595e7ecf5e6516a210b4c4f205d71f7dbb802502a2c6174a9`.
A/B worker log SHAs:
`48e693efbf0463d701d4100f4b83fd47f68f2ee015802b745b7af87a64075be2` /
`9a45b000831c442b20edf89ee1bbc63146c340c1355fc44b26294e87c288caf1`.

The same frozen executable/runner has a separately selected `fresh` stage:
64 calibration and 256 holdout deals, chance seed **96004**, distinct from
the archived development data and the unit test's seed 95004. Only the
terminal weight differs (0.5 versus 1.0); both arms keep 32 sampled-flop
iterations, 64 joint turn/river iterations, 2,048 terminal equity samples,
policy seed 87001, and delayed-flop LBR seed 90001 with 16 early runouts.
The exact final-action estimator and fixed budget were frozen before viewing
this new data. Do not stop early for an apparent win, pool the shared deals
as independent observations, or tune either arm from interim outcomes.

Fresh launch requires the completed hash-matching parity record and no live
full-size worker. It retains the two-thread/one-table design, a 7.5GiB sampled
physical-footprint stop, a 20GiB free-disk reserve and a 5,400-second cap per
source. Failure preserves the stage and requires inspection, not an automatic
restart. Completion yields a paired restricted-response comparison, **not**
a full-game exploitability upper bound; raw response calibration remains
visible and is not replaced by marginalized scores or rejected-response zeros.
No serving model, browser code, release gate or paid resource is changed.

Verification before launch: `cargo test --release -j 1 --quiet --
--test-threads=2` passes **260 library and nine CLI tests**, with 22 explicit
research entries ignored (73.43 / 0.59 seconds). `cargo build --release -j 1
--quiet` and `git diff --check` pass. The production executable is still
byte-identical to the prior milestone:
`5ca2ad865f6cf4686b22d163965bf6e64b94564aff49079575480fc90651841e`.
Browser/npm checks are not rerun locally for this research-only change.

After both parity workers and the regression test process exited, the fresh
stage was explicitly launched from `preflop-solver/`:

```bash
python3 neural/runs/local-sampled-flop-20260905-terminalparallel1/run.py fresh bb00d8cb4694c1acea5060955469ec53ae46d1ebe321de36ed0c3f2a462ace74
```

Initial source A worker PID: 30047. The authoritative stage record is
`neural/runs/local-sampled-flop-20260905-terminalparallel1/fresh/manifest.json`.
This launch is pending evidence, not a completed comparison; retain the
fixed budget and inspect the actual process and resource record before any
next action. The previous terminal correction remains experimental.

### Fresh confirmation: source A complete, source B live

The shared-table milestone was committed/pushed as `0ae59c3`; remote `main`
was verified at the same SHA. CI **34010081484 passed**, including the Linux
release/application pipeline. No unrelated loose artifact, report directory
or review test was staged.

The fixed fresh stage completed source A (26001) successfully: 2,196.873
seconds, sampled peak physical footprint 6,892,508,416 bytes, no resource stop.
Its control and candidate each completed all 64 calibration and 256 holdout
deals. Source A's process exited before source B (26002) started, initial
PID 32163. Source B and the complete pair are **pending** at this checkpoint.

| Source A holdout metric, bb/full hand (half-seat sums) | Control | Candidate |
| --- | ---: | ---: |
| Raw sampled restricted-attack gain | 0.215530 | 0.023454 |
| Exact-terminal marginalized restricted-attack gain | 0.208727 | 0.024396 |
| Marginalized gain SE | 0.121439 | 0.134394 |

The paired marginalized candidate-minus-control estimate is **-0.184331bb**,
SE **0.097062**, individual normal 99% interval **[-0.434345, +0.065683]**.
This fresh holdout points in the same direction as the development result,
but does not exclude zero at the stated confidence. Calibration's paired
estimate points the other way (+0.148092bb, SE 0.182493); both seats in both
source-A profiles still fail raw response calibration. Do not convert the
candidate's small observed response gain into a full-game upper bound,
declare the pair won before B completes, or extend/tune from partial scores.

The independent read-only source-A audit checks all **640 hand executions /
1,280 seat records**, 320 shared phase-specific deals, native-log/manifest
agreement, frozen binary/runner/parity hashes, legal action-value selections,
exact common cards/seeds, baseline cancellation and no-intervention identity,
terminal probability weighting/signs, phase counts and retained calibration,
and raw/marginal/paired means, SEs and intervals. All pass within 1e-12.
Eligible terminal decisions integrated: control 12 calibration / 48 holdout;
candidate five / 27. The holdout has **256 shared independent deals**, not
512 because both policy variants executed them.

Completed source-A stdout SHA:
`936d70131bb4ca159eb28052966009aefa3f18a0f97c0f6c2953042c7c454b23`.
Its separate diagnostic log SHA:
`9e9e477a655865dde39859a7f077a33f11883fbee60ced014e678a4c8fa6df2c`.
The fresh stage's manifest is still mutable while B runs; no final manifest
hash or completed-pair result is claimed. Disk availability fell from about
21.35GiB to 20.8GiB during A, while its logs were only about 608KiB at the
resource check. No files or user processes were removed, and the original
20GiB reserve remains enforced for B.

### Completed fresh confirmation: mixed result, no promotion

`local-sampled-flop-20260905-terminalparallel1/fresh` is now **complete**.
Both source processes and the supervisor have exited, and the retained exec
session returned exit 0. Source B took 2,043.351 seconds and peaked at
6,937,859,352 physical bytes. Total pipeline time was **4,242.013 seconds**
(70.7 minutes). Neither source reached a resource stop; the final disk check
showed about 20.68GiB free. No checkpoint or unrelated user file was removed.

| Source | Control marginalized holdout gain (SE) | Candidate marginalized holdout gain (SE) | Paired candidate-minus-control (SE) | Individual normal 99% interval |
| --- | --- | --- | --- | --- |
| A / 26001 | +0.208727 (0.121439) | +0.024396 (0.134394) | -0.184331 (0.097062) | [-0.434345, +0.065683] |
| B / 26002 | +0.125005 (0.068607) | +0.217296 (0.112076) | +0.092291 (0.083345) | [-0.122391, +0.306974] |

All table values are bb/full-hand **half-seat sums**, not total exploitability.
Multiply means, SEs and interval endpoints by two for the seat-summed scale.
Raw source-B holdout gains are +0.137597 control / +0.222115 candidate on the
same half-seat scale. Source B's fresh point estimate reverses its development
result. Neither source's paired interval excludes zero, and both seats in all
four profiles remain rejected by raw response calibration. B's calibration
paired estimate is +0.040395bb (SE 0.097769); no calibration result is replaced
by a marginalized score or a zero certificate.

Disposition: **the full-weight patch did not confirm as a cross-source
improvement**. Preserve it as an explicit research option and preserve the
deterministic bad-call reproduction, but keep the existing 0.5-weight control
as the comparison baseline. Do not activate the candidate, pool development
and confirmation deals into a new passing claim, or run a longer unchanged
terminal-weight comparison merely to seek significance. This result leaves
the preflop consistency, routed action-EV precision, full-hand coverage and
full-game exploitability requirements unresolved.

The independent full-pair audit checks all **1,280 hand executions / 2,560
seat records** against native logs and the completed manifest, verifies
original source, frozen executable/runner, prerequisite-parity and output
hashes, exact common deals across both sources and weights, all phase counts,
legal bounded action-value selections, baseline cancellation/no-intervention
identity, exact terminal weighting and signs, calibration retention, and raw,
marginal and paired means/SEs/intervals. All pass within 1e-12. There are **256
independent holdout deals reused across the four profiles**, not 1,024.
Source B integrates eight calibration / 40 holdout terminal decisions in the
control and six / 23 in the candidate.

Of the 256 holdout deal-level differences, A changes 36 (24 lower, 12 higher)
and B changes 28 (15 lower, 13 higher). The largest adverse B difference is
index 240, +15.666035bb on the half-seat-sum scale. Its native records show
the seat-0 attacker switching from a flop shove to a call followed by turn
and river decisions; the other attack also changes its terminal call mix.
The candidate's river payoff remains a sampled runout outcome, whereas the
control's terminal flop payoff was exactly integrated. This is not a software
bug reproduction or evidence that one particular river action is wrong.
The diagnosing-bugs discipline therefore stops short of inventing a
"candidate must win this deal" regression: a deterministic path difference
alone does not establish a population policy defect or explain the entire
mixed result. No hidden opponent cards are supplied to policy decisions.

Completed fresh manifest SHA:
`e60bf5a9c36c851afec587caaa92e140d1577aefeb8c0f707447fd31bd429abc`.
Source-B stdout SHA:
`847a6de33cb84e7ccf4d197a1f33b1dbfc0d37a77a46cd5ada781412c7f0056b`.
Source-B diagnostic SHA:
`e3ce7e658692d8aa8c794e802bdfa214592852406b5110e0dafc10def671a2f2`.
Executable and runner remain the parity/fresh frozen hashes recorded above.
This completed-result update changes documentation only; the 269-test release
suite and successful CI for `0ae59c3` cover the unchanged implementation.

### Exact flop all-in chance inside training: controlled pilot passed

Following the mixed terminal-weight confirmation, implemented an explicit
research training alternative, `sampled_flop::solve_with_exact_terminals`.
Instead of modifying a served probability mix after solving, this changes
the terminal counterfactual values used by the flop DCFR updates. A showdown
ending on the flop uses all 990 legal unordered turn/river runouts for each
compatible private pair. Nonterminal continuations still sample future cards;
folds and terminals on other streets retain the old evaluator.

The motivation is the sampling-variance problem discussed by
[Davis, Schmid and Bowling (ICML 2020)](https://proceedings.mlr.press/v119/davis20a.html).
This implementation does not claim their predictive-baseline theorem. Its
specific expectation argument keeps the existing sampled-future card masks,
raw reach weights, and `(49*48)/(45*44)` chance correction. For each fixed
compatible private pair, the mask admits 45*44 of the 49*48 proposals, and
the substituted payoff is the exact conditional all-in mean. No future cards
enter a flop decision and no reach is renormalized after masking. The existing
opponent-action importance correction remains unchanged. This does not prove
lower variance at every upstream node or faster full-game convergence.

`range_vector/flop_terminal.rs` recovers integer showdown counts from the
existing dense equity matrix, checks its exact 1/1980 lattice, blocked entries,
finite probabilities and mirrored zero-sum consistency, and stores u16 counts.
Thus f32 matrix rounding does not remain in the exact terminal CFVs. New mode
has a distinct schema and input hash; ordinary `solve` retains its original
identity and behavior. No training checkpoint format, production setting,
website, neural model, action-EV grade or active policy is changed.

Four new mathematical/behavioral tests pass: exhaustive single-private-pair
mean preservation and lower terminal variance over all 49*48 proposals;
raw-reach scaling, weighted zero-sum and unchanged folds/other streets;
invalid/missing/off-lattice/asymmetric matrix rejection; and deterministic
separately identified root policies without changing the control. A fifth
normal test pins the pilot's depth explicitly. Complete release results:
**265 library + nine CLI tests pass**, 23 research entries ignored, 74.18 /
0.63 seconds. The release native binary is
`2d93437e9b11b08386ac3f74a7fc793f530477f5ed11bd1ce7f1d1de50e09f97`.
Default two-round CLI artifact and summary bytes match the archived executable
for both ordinary PCS and terminal-action-integrated PCS. This is a small
artifact compatibility check, not a byte-identical executable claim.

The initial `local-sampled-flop-20260905-exactterminaltrain1` startup failed
before training: its explicit assertion found the unoverridden library default
100bb instead of 20bb. The diagnosing-bugs loop reproduced that failure twice,
verified matching binary hashes and no config override, and corrected only
the pilot config. The original failed run remains untouched. The corrected
new-directory run `local-sampled-flop-20260905-exactterminaltrain2` completed
all **24 native solves** at 20bb, 32 iterations and the unchanged action grid:
three terminal boards times two seeds times two methods, plus three nonterminal
pot sizes times two seeds times two methods. Its worker took **24.392 seconds**,
sampled peak physical footprint **185,139,728 bytes**, exit zero, no 2GiB /
300-second / 20GiB-free-disk stop.

The terminal roots use uniform public flop ranges, limp/check then BB shove,
and the full exact compatible-opponent/runout expectation for fold/call loss:

| Flop card IDs | Seed | Sampled training loss, bb/decision | Exact-terminal training loss | Reduction |
| --- | ---: | ---: | ---: | ---: |
| 48,21,2 | 97001 | 0.621242 | 0.002697 | 99.57% |
| 48,21,2 | 97002 | 0.468291 | 0.004077 | 99.13% |
| 27,2,9 | 97001 | 0.465239 | 0.000430 | 99.91% |
| 27,2,9 | 97002 | 0.435429 | 0.000361 | 99.92% |
| 42,34,25 | 97001 | 0.507589 | 0.002633 | 99.48% |
| 42,34,25 | 97002 | 0.485657 | 0.002728 | 99.44% |

These are six paired local terminal decisions, not authentic full-hand reach,
untouched board holdouts, independent per-combo samples or full-game
exploitability. The exact evaluation covers a single terminal action choice,
not adversarial nonterminal continuations. Do not promote from this result.

Nonterminal 20bb flop root costs, averaged over seeds 97101/97102, were
2.206 -> 2.375 seconds (4bb pot), 1.261 -> 1.424 (10bb), and 0.711 -> 0.788
(20bb). These observed increases of 7.7%, 12.9% and 10.9% use an already warmed
process-local equity cache; they are not isolated cold-start benchmarks.
No nonterminal payoff improvement was measured by this cost screen.

The native pilot passed, but its Python post-check then raised `KeyError`:
it expected `action_values_bb: null`, whereas `PublicBeliefStrategy` explicitly
omits that optional field when absent. A separate recovered audit, without
rerunning training or rewriting the failed runner manifest, validates that
actual schema. Its first label assertion was also corrected from `call` to
the legal `call_all_in` after checking the saved 20bb history and engine label.
The final audit verifies all 24 policies, complete board/seed/method coverage,
native output and binary/runner hashes, control artifact parity, limits, and
all 1,176 legal / 150 blocked rows per root. Maximum exported probability-sum
error is **8.0e-8**. EV estimates remain absent; this is not EV precision passing.

Frozen test binary: `d37f77ec5bce82588d86bd4458935cfc6335d0a5e02103a406e0f9bcf8246b63`.
Frozen runner: `7c3c8f50f09c37bb6b700b010ee6be6d7159312507be7d57ab5a2696a61197ef`.
Preserved post-check-failed manifest: `30dd9905326c54d4998276223b77cdfcac5d527145fe5c18a8a29802e0b51c4a`.
Recovered auditor: `c106ea8d81bb7f32eb4fb5e8f8d73c3efdc5c2c19e5a35765ed720a13b7ec6dc`.
Completed `pilot/recovered-audit.json`: `7e0ca51920508480a93384e232858375128512e09d0434c06d899dac577c5365`.

Disposition: the consistent local loss reduction and bounded cost justify a
short full-profile comparison of sampled versus exact-terminal training,
keeping terminal serving weight 0.5 and the 64-iteration turn/river continuation
fixed. That routed comparison has not yet run. Preflop consistency, routed EV
precision, full-hand lookup coverage and full-game exploitability remain open.

### Exact-terminal training routed into a short full-profile comparison

The completed local-training milestone was committed/pushed as `e0fdae9`;
remote `main` was verified at that commit. New research-only routing adds
`FlopResolve::new_with_exact_terminal_chance` and an explicit training-option
profile constructor. Both arms still serve the 0.5/2048 terminal correction,
32-iteration nonterminal flop roots, and 64-iteration joint turn/river policies.
Only the candidate's nonterminal flop training uses the exact all-in payoff.
Thus the earlier six terminal-root improvements are not silently substituted
for the terminal serving policy. Preflop source weights remain untouched.

The new routed-policy test exercises public-range replay, hidden-card/future-card
invariance and cache-order independence with exact training enabled. A separate
small paired-run test checks serial/parallel equality, unchanged control, both
0.5 serving weights, variant identity and cooperative failure. The latter is
a scheduling/interface test, not evidence of improved full-hand play. The old
confirmation path retains its records when exact training is disabled; new
candidate records carry `exactTerminalTraining: true`. In this explicitly
declared pilot schema an absent tag means the sampled-training control.

Full release verification passes **267 library and nine CLI tests**, with
24 research entries ignored (76.52 / 0.59 seconds); `git diff --check` passes.
These routing modules are test-only. The native production binary remains
`2d93437e9b11b08386ac3f74a7fc793f530477f5ed11bd1ce7f1d1de50e09f97`;
no website or active-model change is made.

The fixed full-size pilot is `local-sampled-flop-20260905-exactterminalprofile1`:
source 800-round seeds 26001/26002 sequentially, two workers sharing one full
immutable source at a time, **16 calibration and 64 holdout deals**, evaluation
seed **98004**, policy seed 87001, delayed-flop LBR seed 90001/16 early runouts.
Both sources/variants reuse the same held-out deals: 64 independent holdout
deals, not 256. This budget and exact-terminal evaluation method were selected
before viewing its results. No interim tuning, automatic extension or promotion.
The outcome is a paired restricted-response comparison, not an unrestricted
full-game exploitability upper bound. Raw calibration remains separately visible.

Frozen test executable:
`7df5e9e0cf88bed4a507d21acf7dda36b519684eff95b410480a89bb489d927b`.
Frozen runner:
`36ba9f30d088cf7d464903887b4898e17ebb254f1151a3e8fe65ee9064e49de6`.
The runner requires the hash-matching recovered local-pilot audit, refuses an
existing output directory or another live full-size worker, verifies source
hashes and retains a **7.5GiB sampled physical-footprint stop, 20GiB free-disk
reserve and 1,200-second cap per source**. Failures preserve outputs and require
inspection; no automatic restart or missing-result zero is allowed.

After the test process exited, launched once from `preflop-solver/`:

```bash
python3 neural/runs/local-sampled-flop-20260905-exactterminalprofile1/run.py 7df5e9e0cf88bed4a507d21acf7dda36b519684eff95b410480a89bb489d927b
```

Source A initial worker PID: **37044**, retained exec session **90282**.
The process was verified live after loading the checkpoint, using both policy
workers and executing flop/turn decisions. Authoritative mutable record:
`neural/runs/local-sampled-flop-20260905-exactterminalprofile1/pair/manifest.json`.
Results are **pending**, not a completed or passing model comparison.

### Full-profile exact-training screen complete: do not promote

The fixed pair completed successfully in **955.631 seconds (15.9 minutes)**.
Source A took 452.867 seconds with sampled peak physical footprint
6,728,455,376 bytes; B took 501.041 seconds with peak 6,434,559,160 bytes.
Both workers exited zero with no resource stops. The supervisor and source
PIDs exited, and exec session 90282 returned zero. About 20.29GiB remained
free; no files were removed. CI **34015092485 passed** for the preceding
`e0fdae9` implementation milestone.

| Source | Sampled control holdout gain (SE) | Exact-training candidate gain (SE) | Paired candidate-minus-control (SE) | Individual normal 99% interval |
| --- | --- | --- | --- | --- |
| A / 26001 | -0.008247 (0.096100) | +0.404016 (0.193438) | +0.412263 (0.231230) | [-0.183345, +1.007872] |
| B / 26002 | +0.001966 (0.207939) | +0.311592 (0.169734) | +0.309627 (0.244416) | [-0.319948, +0.939201] |

All values are **exact-terminal-marginalized restricted-attack gains,
bb/full hand on the half-seat-sum scale**, not unrestricted exploitability or
an upper bound. The negative control estimate is sampling evidence, not
negative exploitability. Both seats of all four profiles fail raw calibration;
that rejection is preserved and never replaced with a passing zero. Raw
half-seat holdout gains are A +0.001063 -> +0.474742 and B -0.132753 ->
+0.319287. Paired calibration estimates are A -0.015625 (SE 0.058044) and
B +0.161379 (SE 0.161379), also not significant at the displayed confidence.

**Disposition: do not promote or extend this unchanged candidate.** Despite
the six strong local terminal-decision results, both full-profile holdout
point estimates move in the adverse direction and neither paired interval
excludes zero. This is not proof of a statistically established regression,
but it supplies no reason to scale this configuration or call it a full-game
improvement. Keep the default sampled-flop research control, preserve the
explicit exact-terminal option and all artifacts, and leave serving unchanged.
Do not pool this development screen with the earlier uniform-range terminal
pilot to manufacture a global result.

On the shared 64 holdout deals, A changes nine payoff records (two lower,
seven higher) and eight attack-history pairs; B changes ten payoff records
(three lower, seven higher) and ten history pairs. This confirms that actual
full-profile action paths changed, rather than only an unused configuration
flag. Individual changed-deal outcomes are not software-bug regressions or
proof that one action is intrinsically wrong.

The native runner checks and an independently implemented read-only JS audit
both pass. The latter reconstructs all **320 profile hand records / 640 attack
seat records**, exact matching native-log/manifest events, source and variant
configuration, frozen binary/runner/log hashes, phase counts, shared unique
cards/seeds, legal heuristic-maximizing actions, baseline cancellation,
no-intervention identity, integrated terminal probabilities/values/signs,
raw calibration flags and raw/marginal/paired estimates and intervals within
1e-12. There are **64 independent holdout deals**, reused across four profiles.

Completed manifest SHA:
`4c5e1686459894900308fc9112a1540d3483b42c8a13ff647b1a438c2c30dc39`.
Read-only `audit.mjs` SHA:
`cec522002cb5ddc637a04b932e59e41689bb3dea38f55a8b234b5bf636ff861c`.
A stdout / diagnostic hashes:
`7c4a52c62923101eb922bd38adf07290e3da7d06534457d92aec2d4ac925e6b7` /
`c11ba5e4f2958fe253ccfe9f39b03f6966bd639df51580dd216df3dc7502b699`.
B stdout / diagnostic hashes:
`af9d4f5bb8e6c5f924f3a8c73e3ee60778aec486661d03979a772ab3e5b56e20` /
`a0623bcd32d793f6f18b7ddce5e29e7ec0274e3f8fd6f85eae8ea1a0e61eac19`.

The next policy-action work should investigate the already identified
flop-training/served-continuation mismatch with controlled public-root
comparisons, not repeat this unchanged exact-training pair with more hands or
iterations. The mismatch is an architectural limitation and a hypothesis for
this failed transfer, not a diagnosed cause established by these outcomes.
All unresolved preflop, routed action-EV, coverage and full-game qualification
requirements remain unchanged; this goal is not complete.

### Continuation diagnosis: sparse local averages prevent a faithful EV comparison

The full-profile screen milestone `601c873` was pushed and CI **34016160392
passed**. The next action was a tight diagnostic, not another longer training
pair. `sampled_flop` now has an internal read-only observer at the end of its
unchanged training traversal. The ordinary caller uses a no-op observer. A
test-only `continuation` module captures frozen average probabilities and
update counts in memory; it exposes no regrets, resumable state, fallback
actions or deployable model. It refuses a row with no average contributions
or no regret updates, matching the local root export's trained-row requirement.

The diagnostic uses **one reused development root**, source A's saved first
BB flop decision after limp/check, board `[24,14,5]`, 20bb, 2bb pot. Only its
public state, game configuration and range vectors are reused, not its old
four-iteration-turn payoff samples. The root is before any flop action, so
those ranges depend only on the unchanged preflop source. Training uses the
actual routed seed formula, `87001 XOR first_u64_le(SHA256(board, history))`,
yielding **3454475668975051736**, and 32 iterations. Each method enumerates all
four first actions on the same 64 conditional private/runout draws (seed99004),
then attempts to follow its own frozen local average continuation.

| Local training | Complete action rollouts | Missing infoset | No average contribution | Average exists, no regret update | Unscored total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sampled | 189 / 256 | 43 | 13 | 11 | 67 |
| Exact flop terminal | 155 / 256 | 57 | 21 | 23 | 101 |

Most failures occur at turn entry; one per method occurs on the flop. These
are attempted conditional action rollouts, not independent full hands,
reach-weighted full-hand lookup coverage or an additional release gate.
Average-only rows do contain an average distribution but are explicitly
untrained under this inspection's stated rule; they are distinguished from
missing rows and absent averages. No interrupted rollout is assigned an EV,
zero loss, uniform continuation, extrapolated policy or passing grade.

The intentionally strict native command returns **101**, reporting
`left: 168, right: 0` for complete-continuation availability. Its supervisor
records `diagnostic_complete_continuation_unavailable`, not a model pass.
The first bounded run took **5.961 seconds**, sampled peak **133,693,896 bytes**,
without a 2GiB / 120-second / 20GiB-disk stop. The same two event records are
reproduced exactly with the final executable in 5.947 seconds, peak136,200,672
bytes. This makes the negative diagnostic repeatable without loading a full
checkpoint.

Minimized reproductions need only own cards `[35,26]`, visible turn board
`[24,14,5,50]`, and their public action histories. After the sampled profile's
flop check/bet2.5/call, the actual turn key exists with average contributions
but zero regret updates. After the exact-training profile's
check/bet2.5/raise9.5/call, the actual key exists with regret updates but zero
average contributions. The normal regression checks both cases and changes
the synthetic opponent holding and hidden river; the keys and diagnoses remain
identical. These are two different continuation histories, not a paired
same-turn-state payoff comparison. The fixture contains only the frozen public
root and its source identity, not the old sampled payoffs or training state.

The diagnosing-bugs process therefore rules out a key mismatch or hidden-card
dependency for these two failures, but does **not** diagnose them as the cause
of the previous full-profile attack-gain result. The proposed comparison of
trained-plan versus served-plan action EVs was not completed: the trained
average continuation is insufficiently populated to score it under the stated
no-fallback contract. Neither root action ranking nor full-game improvement
is claimed from this diagnostic.

Research cross-check: [OpenSpiel's external-sampling implementation](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/algorithms/external_sampling_mccfr.cc)
also places its simple average updates on the non-traverser's nodes and has a
separate full-average traversal option. A finite-budget average-only or
regret-only row is therefore not proof that our standard averaging formula is
wrong. Do not add unweighted extra average updates merely to fill the table.
The newer [CCS-MCCFR preprint](https://arxiv.org/html/2607.27035v1) likewise warns
that fixed marginal chance correctness is not the same as conditional
unbiasedness under adaptive strategies, and its four HUNL endgame comparisons
do not establish a significant benefit. This does not justify blindly coupling
traversers' chance samples or adding another long sampler experiment here.

Verification: **269 library + nine CLI release tests pass**, 25 explicit
research entries ignored, 80.18 / 0.59 seconds. The minimized diagnostic test
takes 5.68 seconds and records the known limitation; it does not pretend to fix
it. Private diagnostic interfaces are scoped to `blueprint`, with no new
compiler warnings. The final native executable is
`acbb92b34c47ae09354dbeb50f5f5357a08f24178faf7485363e7dd86e97e662`.
Its two-round ordinary PCS and terminal-action-integrated artifact and summary
bytes equal the previously pinned reference outputs. No byte-identical native
binary or large-run parity claim is made. `git diff --check` passes. No browser
code or serving policy changed, so no local browser/npm run was performed.

Original diagnostic directory: `local-sampled-flop-20260905-continuationprobe1`.
Its frozen executable SHA:
`4a4a9af2d07832898cf112a4c285b31802a7afa8e353e02684b1b3d6501bce4c`.
Completed negative diagnostic manifest:
`710a1e503d79687525d65e7a883bbb8b41bcc53db25b762800773ab8e041a16a`.
Original public cache:
`6d796cb81412c2740d38cffc9d78a593cd05e0e78c0fce30030c6313adb1c1c2`.
Committed public-only JSON fixture:
`6d4c437507c5083d4930c9fea471660477a73f166cec680490c488f9c2cce260`.
Final frozen test executable:
`1391bd3475c76dc59ab5462534ab8fd646f32821a882a6f2d946dcd8a4cef243`.
`local-sampled-flop-20260905-continuationverify1` runner:
`eb4a2c0b23702414d75a16dc7bd5fb060365f0265f638626fa17d311fef5e233`.
Completed compatibility/reproduction manifest:
`1602da17451491582164dc995d8f521ce3b015e24a4ae9432f1d0434334bc655`.

Next concrete policy-action direction: construct range-conditioned turn
boundary targets with the existing complete 64-iteration joint turn/river
solver. Start from this compact public flop root and replay the actual routed
nonterminal flop policies, updating every acting-player combo likelihood after
each action, before blocking the revealed turn. This can avoid loading a full
preflop checkpoint for every target: the root already pins the preflop
posterior, and every flop action on a path reaching the turn uses the sampled
nonterminal resolver, not the terminal correction. First verify this rooted
replay against the existing full-profile range reconstruction, then build a
small boundary-value pilot. Do not use the sparse training averages or the
old cache's four-iteration-turn action payoffs as substitutes for those targets.
No such replay/value generator is implemented by this diagnostic milestone.
