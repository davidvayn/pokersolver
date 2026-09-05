# Connected tabular turn/river pilot

Research sequence started September 4, 2026; **bounded sequence complete;
full-hand quality comparison inconclusive**. The frozen
800-round pair and its existing response audit remain the baseline. No website
model is changed by these native experiments.

## Design

`full-game-lbr --tabular-checkpoint PATH --tabular-turn-iterations N` retains
the exact frozen preflop/flop policy and replaces the complete continuation
from the public turn root. It reconstructs exact-combo entry ranges by replaying
each player's own observed action probabilities and removes only revealed board
cards. Actual opponent cards and unrevealed river cards never enter the solve.

The ordinary comparator (`--tabular-turn-unconstrained`) jointly solves turn
and river with all compatible river chance outcomes. The default protection
path freezes the blueprint's complete turn/river subtree, then replaces each
seat's complete strategy sequentially using opponent best-response-CFV opt-outs
and the existing finite-iteration projection. Neither arm switches back to an
independent river solve or uses an adversarial gadget opponent as its policy.

The root's player-to-act is not changed to resolve the other seat. The gadget
precedes that root, as in the repository's existing bilateral flop solve.
Approximate projection tolerance remains 0.05bb per protected opponent hand;
this is **not** an exact no-regression theorem or full-game upper certificate.

Relevant primary literature:

- [Safe and Nested Subgame Solving](https://arxiv.org/html/1705.02955v3):
  sections 4–6 distinguish isolated range solves from boundary-protected ones
  and explain what is lost when boundary values are estimated.
- [Depth-Limited Solving](https://arxiv.org/html/1805.08195v1): continuation
  values must account for opponent strategy choices, not treat a poker state
  as having a single strategy-independent scalar value.

## Correctness and memory boundaries

- Cache identity is the exact public turn board and complete root history.
  The active generation contains complete turn and river rows, with the river
  chance card included in descendant keys. Eviction changes runtime only:
  reconstruction of that public root is deterministic.
- Missing descendants, mismatched grids and positive-entry zero-probability
  rows fail closed. A max-row stop rejects the whole solve, never a partial
  policy. Use the external worker guard as well: the native row limit alone
  does not bound intermediate solve memory.
- The frozen trunk may give some private hands exactly zero entry reach.
  When one protected seat is replaced, its zero-entry rows retain the original
  blueprint policy so the other seat's all-opponent-hands gadget has a complete
  anchor. This preserves positive-entry strategies and the protected CFVs.
  The unconstrained comparator counts any queried zero-entry-hand blueprint
  completion separately; such hands are not presented as solved recommendations.
- Card-bucket calculations are reused within a public board when freezing the
  anchor. Mutable regret/average accumulators are never shared across nodes.
- Evaluation uses a separate inference-only checkpoint reader. It decodes
  exact f64 average accumulators and descriptors into smaller nodes, without
  allocating regrets or resumable state, and interns metadata during decoding.
  It validates policy identity, dimensions, finite nonnegative accumulators
  and history references. The unchanged training reader still validates the
  resumable training state. Saved checkpoints are never modified, and this
  inference-only type cannot be resumed or exported as a training checkpoint.
- Existing sparse source-policy completion remains explicit and offline-only.
  Source lookup counts and resolver decision counts are separate. A zero source
  lookup count on a resolved street is not independently a coverage certificate.

## Guarded paired runner

From the repository root:

```bash
python3 preflop-solver/neural/tabular_turn_pilot.py \
  --binary preflop-solver/target/release/preflop-solver \
  --checkpoint-stage preflop-solver/neural/runs/local-pcs-20260904-streamed800 \
  --output-dir preflop-solver/neural/runs/NEW-PILOT-DIRECTORY \
  --arms baseline,joint:4 --seeds 26001,26002 \
  --seed-offset 900000 --response-workers 2 \
  --training-deals 2000 --calibration-deals 5000 --evaluation-deals 5000 \
  --rollouts-per-action 4 --minimum-range-particles 4 --max-worker-minutes 180 \
  --max-worker-memory-gib 7.5
```

This reproduces the screening budget, not a release evaluation or a
recommendation to rerun an inconclusive pair unchanged. The earlier 256-hand
budget failed to calibrate the controls; the 16-iteration arm exceeded the
serial cost screen. The protected arm needs its own cost-qualified design,
not an unattended addition to this command. The runner freezes
its binary, pins checkpoint digests, runs both seeds of each arm before the
next arm, and stops on the first worker/resource failure. It reserves 20GiB
disk, runs one checkpoint process at a time, and refuses to overwrite an existing cohort. SIGINT or
SIGTERM propagates to the owned worker. It does not start any paid resource.

Response training, calibration and evaluation deal streams are disjoint.
Train a new responder against each candidate. A responder rejected by
calibration contributes no claimed gain and makes that result inconclusive;
do not count rejection as evidence of low exploitability. Compare total signed
seat-summed response gains, not the legacy half-sum fields.

## Initial integration probes

- `local-turn-20260904-cost`: exact checkpoint, seed 26001, 2 protected
  iterations, 4/2/8 training/calibration/evaluation deals. Failed on an
  incomplete zero-entry anchor row, not a strategy-quality gate. The diagnosis
  regression reproduces this with a preflop BB hand having zero check reach.
- `local-turn-20260904-cost-fixed`: the same probe after preserving zero-entry
  rows and adding board-local reuse. Three protected roots completed; the
  larger fourth root hit the 7.5GiB guard at 8,148,195,448 sampled bytes.
  No checkpoint or score was accepted. This prompted discarding training-only
  allocations in the evaluation copy, not raising the memory stop.
- `local-turn-20260904-cost-reclaimed`: dropping those allocations after
  loading still stopped on the larger subtree at 8,092,784,688 sampled bytes.
  The next reader omits training allocations at decode time and uses a smaller
  node representation; it does not rely on reclaiming a fragmented live heap.

- `local-turn-20260904-inference-parity`: both complete 20,000/5,000/20,000
  response evaluations reproduced the original JSON bytes exactly using the
  compact reader. Sampled peaks were 5,741,121,880 and 5,691,232,600 bytes.
  This verifies the allocation change alone did not change the average policy
  or the original response results (0.8051452 / 0.7332198 total bb/hand).
- `local-turn-20260904-cost-compact`: the original failing protected probe
  completed in 124.802 seconds, including checkpoint loading; peak
  6,405,460,840 bytes. All four roots completed; maximum opponent-CFV excess
  0.048828125bb. Its tiny rejected responders are **not** quality evidence.
- `local-turn-20260904-pair1`: baseline responders trained on only 256 deals
  failed calibration for both seats of both seeds. Stopped the candidate arm
  early instead of treating the resulting zero claimed gains as a win. The
  interrupted joint worker produced no accepted evaluation. Its 60 attempted
  public roots were all distinct, so a larger cache would not remove that cost.
- `local-turn-20260904-budget2000`: increasing training to 2,000 deals and
  calibration to 5,000 still calibrated only one seat per seed. Not eligible
  for a two-seat policy comparison.
- `local-turn-20260904-budget2000-paired`: after the paired-error correction,
  both seats calibrated on both seeds. Held-out total gains were 0.441883 and
  0.4805994bb/hand. These use a weaker 2,000-hand responder-training budget than
  the old 20,000-hand audit; the lower numbers do **not** show a stronger poker
  policy. They establish an affordable baseline for comparing candidates with
  the same evaluator and budgets.
- `local-turn-20260904-pair2`: reproduced both new baseline result files
  byte-for-byte, then cost-screened the joint 16-iteration arm. Stopped that
  worker at 517.259 seconds (peak 6,205,657,816 bytes): its first 128-deal
  training block was still incomplete, projecting beyond the 180-minute cap
  for training both responders alone. No candidate score was produced.
- `local-turn-20260904-pair3`: joint 4-iteration pair, seed A complete. Used the
  identical frozen executable and checkpoint identities as pair2, plus the
  same 2,000/5,000/5,000 budgets, rollout/range-particle settings and evaluation
  seeds. Pair2's completed baselines are its controls; do not rerun or substitute
  the 256-hand rejected controls. Only resolver iterations changed from the
  interrupted cost screen. No protected full-hand win has been measured yet.

## Response-budget diagnosis

The responder uses common-random action rollouts, but its previous action-gap
error calculation added two marginal variances as if those samples were
independent. A regression with noisy common payoffs and an exactly constant
action advantage reproduced an erroneous low-confidence label. The accumulator
now measures paired action differences directly with a Welford variance update.
An anticorrelated control still reports uncertainty. Per-action EV errors and
the disjoint 99.5% calibration gate are unchanged. This affects the responder,
not the frozen poker policy; compare candidates against a new baseline using
the same revised evaluator, not against an older responder's score.

The runner now stops a baseline-plus-candidates cohort before candidate work if
either baseline seat fails calibration. Missing seats, nonfinite statistics or
negative error estimates are ineligible too. There is no promotion logic.

## Exact terminal-work optimization

A five-second CPU sample of the frozen four-iteration worker showed substantial
time in terminal card-marginal and showdown-vector calculations, not disk I/O.
Alternating CFR training only consumes the traverser's values; a best-response
pass only consumes the responding player's values. These two paths now compute
only that player's terminal vector. Ordinary two-player profile evaluation is
unchanged. No actions, iterations, chance outcomes, accumulation weights or
probability arithmetic were changed.

A test-only reference keeps the original both-player calculation. Differential
tests compare every regret/average accumulator, serialized policy, exact
conditional BR value and safety excess for dense/sparse ranges, the full 20bb
grid, both protected seats, and the river-refinement/CFR+ path.

`local-turn-20260904-terminal-parity/benchmark.json` records two order-reversed
comparisons per root. All complete root-value/response-metric outputs match byte
for byte between executables. Mean wall times (including evaluation/output):

| Root | Original | Optimized | Reduction |
| --- | ---: | ---: | ---: |
| 2bb pot, 16 iterations | 9.050s | 6.885s | 23.9% |
| 5bb pot, 16 iterations | 4.889s | 3.605s | 26.3% |
| 2bb pot, 4 iterations | 3.895s | 2.857s | 26.6% |

These are local cost measurements, not policy-quality improvements. Pair3
seed A finished on its frozen pre-optimization executable; seed B used the
verified optimized executable with its own pinned provenance.

The completed pair below does not establish a full-hand policy improvement.
Only after a full-hand improvement is supported should we either extend the
winning solve or, if its runtime is too expensive, generate targets for a small
range-conditioned continuation model. A neural model must preserve the action
improvement on new/perturbed ranges; lower fitting loss alone is insufficient.

## Shared-table CPU workers

`--response-workers 2` enables two CPU-parallel Rust threads during responder
training, calibration, and independent evaluation. The default is one; accepted
counts are 1–4, only for tabular checkpoints. Each worker shares the immutable
inference table through `Arc` and owns its range-solver cache and counters.
Separate full-table OS processes are not needed on the 16GiB laptop.

The main thread samples the original deterministic deal stream. Bounded waves
contain 16 deals per worker. Workers return raw ordered observations; the main
thread replays those observations in the original deal/decision order, retaining
the exact floating-point accumulation order instead of merging partial means.
Calibration/evaluation retain their disjoint seeds and paired rollout streams.
Source coverage is summed as exact integer counts; solve cache/timing statistics
are combined separately. No runtime fallback switches to an incomplete policy.

Serial reports keep their existing JSON representation. Parallel reports add
`response_workers` for provenance. Comparing results may normalize this field
and resolver timing/cache-work diagnostics, but never policy values, payoff
statistics, confidence, or source-query coverage. Unit checks cover 1/2/4 workers,
partial waves, full response reports, and connected protected/unprotected paths.
Release verification: `cargo test --release -- --test-threads=2` passed all
201 library and 5 CLI tests; the focused runner/resource suite passed 30 Python
tests. `cargo fmt --check`, `cargo check --tests`, and `git diff --check` passed.
Both full-size baseline reports match pair2 after normalizing only the worker
count (including a canonical JSON comparison retaining signed zero). The
full-size connected 32/16/16-deal probe also matches in every policy/result and
source-coverage field; only worker count and cache-work/timing diagnostics are
excluded from that comparison. It is a resource/parity probe, not a quality run.

| Connected probe | Wall time including load | Sampled physical footprint |
| --- | ---: | ---: |
| One worker | 110.333s | 6,131,143,360 bytes |
| Two workers | 95.814s | 6,306,599,736 bytes |

This is a single matched cost pair (13.2% lower total wall time), not a universal
speedup claim. The checkpoint load remains serial and dominates such a short
probe. Both workers share one table, and the 7.5GiB guard is unchanged. Four
workers have unit parity coverage but no full-size memory/throughput validation.

Pair3 seed A was preserved on its frozen serial executable until it finished.
The one-off `local-turn-20260904-parallel-handoff.py` waited for A's completed,
digest-checked output, stopped only the owned old runner, and verified that its
solver exited before loading another full-size table. It then compared both
baseline reports to pair2, measured serial/two-worker connected parity and cost,
and started seed B only after those checks passed. Its status is recorded separately;
it does not turn an interrupted or ineligible result into a quality score.

Seed A completed in pair3 with result SHA
`e711ce7c89325c82a56809e6fd810b5adc6196a7632f296254cff1fc630a6a42`.
Seat 0 failed calibration (0.0241664bb gain, SE 0.0291192); its zero deployed
holdout gain cannot rank the policy. Seat 1 calibrated and measured
0.1761668bb (SE 0.0587053), versus 0.1949834bb (SE 0.0598998) for its control.
That small seat-1 difference is not a confirmed improvement. The two-seat
comparison is ineligible, not a claimed 0.441883→0.176167 exploitability win.

Seed B completed in `local-turn-20260904-pair3-parallel-b`, with two workers,
the original pair3 budgets/streams, and verified performance-only build
`cdbabf150044c292325392b19e8e263c8bd8fbb61f2ff760bb9f1cc4742956f0`.
The old runner was stopped only after A's completed result was recorded; its
briefly started serial B was interrupted during setup and is not a result.

Seed B's result SHA is
`3b58ec408806b8837eb9687eee02107db0647e2f34956d5933d02f5a04f10a29`.
It also rejected seat 0 on calibration (gain 0.1152998bb, SE 0.0493699,
99.5% lower bound -0.0118687). Calibrated seat 1 measured 0.1592998bb
(SE 0.0476341), versus its control's 0.2188664bb (SE 0.0581619).
Its total deployed 0.1592998bb is not a complete two-seat quality score.

The pair does not qualify for selecting a full-hand winner. Do not promote the
candidate or start the conditional neural-distillation/longer-training stages
from the artificially smaller deployed totals. No release metrics were relaxed.
Seed A took 4,334.757s (peak 6,220,731,144 bytes); parallel optimized seed B took
2,622.295s (peak 6,721,262,728 bytes). Different seed policies/deal streams mean
that ratio is not an isolated performance benchmark; use the matched cost probe.

## Completed protection check and handoff

`local-turn-20260904-protected-pair` completed the final bounded protected probe
on both seeds with the verified optimized executable. Budgets were 4/2/8 hands
per seat, two iterations/rollouts, minimum two particles, and offset 700000.
This is integration/cost evidence, not a qualified full-hand comparison.

Seed A completed four roots in 133.606s, peak 6,279,123,792 bytes, maximum local
opponent-CFV excess 0.048828125bb. Seed B completed eleven roots in 146.107s,
peak 6,116,774,568 bytes, maximum excess 0.049509162bb. Both stayed under the
0.05bb local tolerance and 7.5GiB memory stop. Both had a minimum retained
proposal weight of 0.015625; this is a strongly constrained replacement at
those stages, not evidence that the new policy can replace the full baseline.
Both tiny probes rejected their responders and provide no quality score.

All sequence-owned workers have finished. The conditional continuation student
and longer policy-training stages were not triggered, because the full-hand
pair did not qualify a winner. No model was promoted and no gate was relaxed.
See the [completed audit](../validation/2026-09-04-connected-turn-pilot.md) and
its [machine-readable evidence](../validation/2026-09-04-connected-turn-pilot.json).

## Continuing policy-action work: flop corrections

The user requested continued policy improvement after the inconclusive pair;
do not treat completion of that old audit as completion of the current task.
No distillation is part of this continuation.

A full-hand identity-intervention regression reproduced another avoidable
source of comparison noise: a forced response skipped the random action draw,
shifting all later draws even when it selected the baseline's identical action.
The response now consumes that draw. The method string records the change;
old sampled reports remain immutable and must not be treated as same-estimator
controls for new response scores.

`tabular-flop-pilot` / `neural/flop_patch_pilot.py` compare a changed defender
against retained opponents on paired fresh hands. Importantly, an opponent's
original continuation policy remains frozen too; it is not replaced by the
candidate's completion. Reports retain raw per-hand paired payoffs and signed
improvements. Rejected old responses are explicitly marked diagnostic, never
converted into an apparent zero-exploitability result. Every worker shares one
immutable checkpoint; the runner pins input/output/executable hashes and keeps
the 7.5GiB memory and 20GiB disk-reserve stops.

`local-flop-20260904-pair1` completed both seeds, 1,024 hands per seat/opponent,
offset 1100000, two workers, and a 25% blend at the first confident flop
opportunity. Only the calibrated BB proposal seats were changed. Against the
old baseline / connected retained opponent respectively, corrected BB payoff
changes were +0.046631 / -0.078857bb (seed A) and +0.062337 / +0.004150bb
(seed B). Every interval crosses zero. Just 6–20 paired hands per corrected-seat
comparison changed payoff. This sparse patch is **not a selected winner**.
Elapsed times were 248.067 / 304.039 seconds; sampled peak footprints were
7,010,932,016 / 6,980,932,912 bytes. Frozen binary:
`01111143ab318b94b0bcbf79948213ebe02deea017b3ae91878fd8ec284c4be0`.

The next, materially different candidate uses `--all-in-samples 2048`: terminal
flop call/fold correction from exact hero cards, blocker-correct opponent ranges
conditioned on its own observed actions, and independent legal runout samples.
Only an advantage outside a two-sided 99.5% Hoeffding equity margin changes the
mix. Uncertain/zero-reach lines retain the explicit original baseline; no
uniform posterior is invented. This is a response to the frozen opponent model,
not an arbitrary-opponent safety theorem. Turn/river solving stays at four
iterations; terminal flop actions do not require extra continuation solving.

`local-flop-20260904-allin-pair2` completed both seeds: 2,048 fresh
hands per seat/opponent, offset 1200000, the same 25% blend, two workers, and a
30-minute per-seed cap. All eight paired point estimates were positive, but all
individual 99% intervals crossed zero. Ordered as baseline-opponent BB/SB,
connected-opponent BB/SB, gains were 0.092000 / 0.000977 / 0.020589 / 0.055908bb
for A and 0.102417 / 0.066813 / 0.073039 / 0.006104bb for B. This is encouraging,
not a release qualification. Runtime was 468.131 / 526.530 seconds, sampled
physical footprint 7,024,366,896 / 7,051,285,832 bytes, with no resource stop.
Binary: `efb82a2932c18959a068e5a86ee7b6cbcd597786be455ae8001999a9159fcd35`.

`--integrate-terminal` leaves that policy unchanged but averages over the final
fold/call mix and all 990 legal flop runouts for payout assessment. The two
profiles are identical until this terminal-only correction, so their common
prefix needs to be played only once. Hidden cards are used only by the payoff
evaluator, never by the policy. This is conditional-mean/Rao-Blackwell scoring,
not a full implementation of [AIVAT](https://arxiv.org/abs/1612.06915).
It is prohibited for nonterminal/saved-action corrections. Explicit exhaustive
runout and ordered serial/parallel regression tests validate the integration.

`local-flop-20260904-allin-confirm3` completed the confirmation: unchanged 25%
terminal correction with 2,048 equity samples, 4,096 **fresh** hands per
seat/opponent, offset 1300000, integrated terminal scoring, and two workers.
All eight individual 99% paired improvement intervals exclude zero:

| Seed | Retained opponent | BB gain (99% interval), bb/hand | SB gain (99% interval), bb/hand |
| --- | --- | --- | --- |
| 26001 | Baseline response | 0.07552 [0.05690, 0.09414] | 0.06513 [0.04801, 0.08224] |
| 26001 | Connected response | 0.05914 [0.04277, 0.07551] | 0.05928 [0.04292, 0.07564] |
| 26002 | Baseline response | 0.08366 [0.06500, 0.10231] | 0.05512 [0.03785, 0.07238] |
| 26002 | Connected response | 0.07051 [0.05340, 0.08761] | 0.05260 [0.03642, 0.06878] |

These measure defender payoff improvements against frozen retained opponents,
**not a drop of that size in full-game exploitability**. The connected SB
attackers were originally rejected on calibration and remain labeled raw
diagnostic opponents. Both seeds' improvements are supported against the old
calibrated opponents too. No new opponent was trained for these confirmations.

Confirmation binary:
`a5fc3ed9c30aec7eff5820e9cd00a7211d9c6ac3db5cd2274781f94c8d109638`.
Result hashes: A `7b8785eda7c2f6cca86c0898da17d9f1e8608409e0a864660966251604dac3a7`,
B `84ffa86e4864b6b443871ec1db60e12dfa37f7a3b04c6a1fe2ff383e336825d6`.
Runtime 521.398 / 578.823 seconds; sampled peaks 6,865,900,752 / 6,878,123,264
bytes. Different corpora/budgets and concurrent builds mean these are not an
isolated throughput benchmark against earlier cohorts.

`full-game-lbr --terminal-flop-samples 2048 --terminal-flop-weight 0.25` now
loads this same candidate for fresh response training. The options are pinned
in `terminal_flop` and the source-kind label; absent options preserve the old
report shape and policy. Fixed-panel opponents loaded from such future reports
also retain their terminal correction as part of their frozen completion.

`local-flop-20260904-fresh-response4` **completed**: both seeds, 2,000 response
training / 5,000 calibration / 5,000 held-out hands per seat, four rollouts,
minimum four particles, two workers, offset 1500000, joint turn/river four
iterations, and the frozen 25% terminal correction. Neither seed hit its
90-minute / 7.5GiB / 20GiB disk-reserve stop. All four responses failed the
positive 99.5% calibration-lower-bound requirement:

| Seed | Attacking seat | Raw calibration gain, bb/hand | Standard error | 99.5% lower bound |
| --- | --- | ---: | ---: | ---: |
| 26001 | SB | 0.071417 | 0.031763 | -0.010399 |
| 26001 | BB | 0.040483 | 0.019225 | -0.009038 |
| 26002 | SB | 0.017733 | 0.023003 | -0.041519 |
| 26002 | BB | 0.067167 | 0.040251 | -0.036514 |

Consequently both deployed holdout totals are zero **by rejection**, not a
zero-exploitability result. This is an inconclusive restricted-response audit,
not evidence of passing a full-game gate. Its scores cannot be compared directly
to old aggregates from the pre-aligned-draw evaluator. The learned responses
remain available as explicitly unvalidated opponents in a separate payoff test.

The underlying table still has a material flop-coverage weakness: independent
evaluation lookups were 28.677% unknown / 8.969% untrained for A and 27.975% /
8.295% for B. These are source-table diagnostics, not complete composed-policy
coverage; turn and river normally use the connected resolver instead. Preflop
had no unknown or untrained lookups in this corpus. The terminal correction
does not eliminate the broader flop-coverage problem.

Frozen binary: `48c88f6e9df394606099ba990514fad1940ab24b5abe8c66958f0fd7cebb3f6d`.
Results: A `98e2437cdbf94a4d2a4fbf986db62177a63e82a0f9652d88ed7faab0fd4d8d50`,
B `6a7910394e262516f6312b45fc446ad7ed708441c7720abd1cefc4c3d7ae7e42`.
Elapsed 2,488.006 / 3,042.581 seconds; sampled peak physical footprint
6,705,714,264 / 6,664,344,640 bytes. Both workers have exited.

The bounded follow-up, `local-flop-20260905-fresh-confirm5`, **completed** on
the same frozen binary, comparing the
unchanged correction against the old uncorrected joint-four-iteration control,
using response4's newly trained opponents and a separate offset 1700000. It
keeps weight 0.25 and 2,048 equity samples, evaluates 4,096 fresh hands per seat
and seed with terminal conditional-mean scoring, and uses two shared-table
workers with the same 7.5GiB / 20GiB resource stops (30 minutes per seed).
Both raw opponents are retained even if calibration rejects them; their
original calibration flags and corrected completion policies remain pinned.
This is a paired payoff test against adapted opponents, not an exploitability
certificate. It started only after response4 released its full-size table.

All four individual normal-approximation 99% paired intervals exclude zero:

| Seed | BB defender gain (99% interval), bb/hand | SB defender gain (99% interval), bb/hand |
| --- | --- | --- |
| 26001 | 0.05937 [0.04326, 0.07548] | 0.04541 [0.03056, 0.06027] |
| 26002 | 0.06123 [0.04506, 0.07740] | 0.06776 [0.04947, 0.08604] |

The opponents include both their retained deviations and their original
corrected joint-turn/river completion. All four remain explicitly marked
uncalibrated. This confirms payoff improvement against these new diagnostic
opponents, not low exploitability against arbitrary or optimal responses.
Together with confirm3, all twelve comparisons across 49,152 fresh comparison
hands have positive individual 99% intervals. No multiple-comparison-adjusted
or equilibrium-certification claim is implied.

Result hashes: A `6ab79770d524ab169f0941f1fa28664541d84549c0e70aa67ac8ac7a498f3b87`,
B `ce283ca387c4ccee31dde0972dfbfe1497d080a3ef1588a53b843b1e6f38c81f`.
Both output and frozen-binary hashes were independently rechecked against the
cohort manifest. Runtime was 351.391 / 360.499 seconds; sampled peak physical
footprint 6,838,097,080 / 6,792,336,544 bytes. No resource stop occurred.

The bounded continuation is complete and no sequence-owned workers remain.
Keep the 25% range-conditioned terminal-flop correction as an experimentally
improved native candidate; do not promote it as Approximate GTO. The next
substantive policy problem is the broader missing/untrained flop strategy,
not distilling the current policy or repeating the same rejected response
scores. Fresh-response calibration and full-game exploitability qualification
remain unresolved. The source 800-round checkpoints were not changed.

Latest verification: 211 Rust release library tests, 6 CLI tests, and 32 Python
runner/resource tests pass. No distillation, source checkpoint overwrite,
website model promotion, paid compute, commit/push, or gate relaxation occurred.
