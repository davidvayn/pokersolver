# September 5 local policy-improvement sequence

User request: work overnight on remaining weaknesses; use short pilots and
research before long training; commit/push major milestones. First target is
all existing quality metrics plus full-game exploitability below 0.50bb/hand,
then continue toward 0.05bb/hand. Do not turn a restricted-response lower bound,
rejected-response zero, or fixed-opponent payoff gain into an upper certificate.
The expected user return is approximately 18:30 UTC. The local 16GiB machine
remains the only authorized compute; no deployment or paid resources.

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
**active**. Minimum eight averaging contributions, full pooled weight only
at missing/untrained supported flop rows; joint-four turn/river and the
25% terminal-flop correction are preserved. Two retained opponents per seed:
the earlier baseline attack and Pilot 2's raw diagnostic attack. Each opponent
keeps its original continuation and calibration status. Use 1,024 fresh hands
per seat/opponent, seed offset 2200000, sequential seeds, two shared-table
workers, 45-minute / 7.5GiB footprint stop and 20GiB disk reserve. This uses
paired realized rollouts, not the terminal-only exact estimator, because the
candidate can change nonterminal flop decisions. No second full-size table
may run alongside it.

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
