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

The full-weight candidate has earned a genuinely fresh **all-street** response
pair, now **active** at
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
