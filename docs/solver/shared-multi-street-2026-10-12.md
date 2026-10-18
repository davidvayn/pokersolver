# Shared multi-street engine and deeper-stack pilots

## Implementation sequence

1. Use the existing full-hand hand engine for both Practice and Solver. Solver
   selects public board cards; Practice retains authentic dealing and settlement.
2. Extend the existing pinned native resolver query with an optional complete
   public-range strategy. Export the same cached probabilities and action values
   used by Practice, not a separate one-street solve or made-up missing values.
3. Replace the Solver page's single-street WASM execution path with a study-line
   controller. Keep decision cards across streets and board edits, carry real
   stacks/pot/history forward, and condition ranges through the native trajectory.
4. Verify preflop, flop, turn, river, folds, all-ins, manual board entry, rewinding,
   loading/errors, terminal retention, and Solver/Practice policy parity. Preserve
   the existing action-card and combo layout and explicitly expose uncertainty.
5. Start bounded, independent 50bb and 100bb local training pilots using the
   depth-aware trainer and a pinned, spread-out rake-free action abstraction. Do not train
   from mislabeled 20bb targets, modify active weights, or activate pilot models.

## Acceptance constraints

- Only installed policy depths are selectable for live study. 50bb/100bb remain
  unavailable until independent training and validation support them.
- Existing 20bb policy is experimental; full-game exploitability has not passed.
- Manual board edits retain the chosen multi-street action trajectory. A missing
  required board card pauses the line rather than clearing or guessing it.
- Public range study cannot settle an unknown private showdown winner.
- No new rake work, paid compute, AWS resources, or scheduled automation.

## Pilot grid

At most three regular sizes per decision, with all-in separate. Opens:
2/3.5/5bb; limp raises: 3/5/7bb; three-bets: 8/12/16bb; four-bets: 20/30/45bb.
Deeper raises use 0.5/1/2 pot fractions. Existing distinct flop bets
(1/3, 3/4, 5/4 pot), turn/river bets (1/2, 1 pot), and one pot-sized postflop
raise are retained. Stack-clipped duplicate all-ins are deduplicated by the
hand engine. This reduces the five-open grid without claiming that abstraction
reduction itself decreases full-game exploitability.

The explicit grid is hashed into the Python run identity, passed to Rust
sampling on every round, checked against returned shard metadata, and exported
in the model. Existing Home resumes omit the new optional field to retain
their hashes. The live 20bb policy is unchanged: filtering its probabilities
would not be equivalent to training in this smaller game.

The supervisor runs two jobs concurrently, two independent seeds per depth,
up to three rounds with a 12-minute soft budget, 30-minute hard limit per seed,
and 3.5GiB process-family CPU RSS limit. GPU/unified-memory usage is not included
in RSS; the small reservoirs and two-job limit are intentional on this 16GB Mac.

## Verification and pilot checkpoint

- TypeScript: `npm test` — 286 passed, four optional native integration tests
  skipped. `npm run build` passed, including frozen serving-artifact hashes.
- Rust: library release suite — 457 passed, 59 guarded research tests ignored;
  CLI release suite — nine passed. Legacy WASM release suite — six passed;
  regenerated package/runtime WASM copies have matching SHA-256 hashes.
- Python: `python -m unittest test_depth_pilots test_train` — 26 passed.
- Real browser: complete eight-decision preflop/flop/turn/river line; pot/stacks
  maintained; missing river removal/restoration retained all cards/actions;
  terminal deselection retained the matrix/combos. Fold, all-in runout, blocked
  policy request with Retry recovery, keyboard focus, picker Escape, and
  Practice's full-hand all-in/showdown also checked. Responsive layouts at
  375/768/1024/1440px, light/dark and reduced motion; no page-wide horizontal
  overflow or application console exceptions observed. The decision rail
  scrolls the newly active card into view.

All four three-round pilots completed: 50bb seeds 50121/50122 and 100bb seeds
100121/100122. Each used 64 traversals per round; this is a pipeline smoke
pilot, not adequate strategy training. Final held-out action-value RMSE was
18.978/19.085bb (50bb) and 37.022/41.991bb (100bb). These sampled-target
diagnostics do not measure exploitability, EV confidence, or cross-seed
equilibrium convergence. No new depth or weights were activated.
The four final binary artifacts each have 660,868 finite parameters and are
about 2.65MB. Re-encoding their JSON sources produced identical binary bytes;
each exported grid has at most three regular sizes and the correct depth.

Artifacts/logs are under
`/Users/davidvayntrub/pokersolver-training/shared-depth-pilots-2026-10-12`.
One sampler launch for seed 100122 stalled before Rust execution: `sample`
showed only `_dyld_start`, near-zero footprint and no CPU/shard progress, while
a fresh binary `--help` invocation succeeded. Terminating that exact child and
retrying the unchanged checkpoint completed all three rounds in about ten
seconds. No poker algorithm/seed/grid change was made. A deterministic unit
test cannot reproduce this macOS loader incident; tests instead cover retry
history and preserving the other completed seeds. `--only-seed` supports that
bounded recovery and the result retains the failed attempt.

Next policy work is a matched, larger pilot and independent full-hand response
evaluation, including responses with omitted/off-grid sizes. Fewer actions can
improve throughput within the abstraction, but do not prove lower real-game
exploitability. The existing 20bb policy also needs matching retraining before
this spread-out grid can replace its five-open action set.

## Full run — 2026-10-13

User authorized the longer run after the smoke pilots. The controller is
`preflop-solver/neural/run_depth_full.py`; output is
`/Users/davidvayntrub/pokersolver-training/shared-depth-full-2026-10-13`.
Two independent seeds per depth (50bb: 50121/50122; 100bb: 100121/100122),
eight hours each, two jobs concurrently: roughly sixteen hours total plus
checkpoint/startup overhead. This run does not retrain or replace live 20bb.

Fresh run directories are required because the smoke configuration differs.
The full profile uses 400 traversals and 100 optimizer steps per round,
100,000-record reservoirs, 1,024-sized batches, 256/128 preflop/flop runout
samples and four independent action-value rollouts. Architecture and spread-out
action grid are unchanged. Later-street chance remains sampled. Snapshots are
exported every 100 rounds and on normal completion, not every round; consumed
shards and superseded checkpoints use the trainer's existing retention rules.

The Rust sampler is copied to the output directory and SHA-256-pinned in each
run configuration, independent of later website rebuilds. Python source and
exporter hashes are recorded; changed pinned inputs prevent queued seeds from
starting with different code. `--sampler-binary`/`--sampler-sha256` preserve the
legacy run identity when omitted. Resume budgets subtract the complete recorded
round durations rather than granting another eight hours.

Guards stop owned workers on non-finite metrics, truncated sample output,
3.5GiB CPU RSS, less than 25GiB disk free, thirty minutes without a checkpoint,
or the remaining time budget plus a thirty-minute hard grace. Worker cleanup
tracks descendant births so an orphaned sampler is not leaked and a reused PID
is not targeted. These guards catch execution/data-integrity problems; they
are not an exploitability or learning-quality evaluator. Normal time-budget
completion checkpoints the last round; forced termination can lose in-flight
work and requires a checkpoint/replay audit before resuming.

`caffeinate` prevents idle sleep only while the controller is alive. Closing
the lid or powering off can still suspend training. Keep the Mac plugged in
and lid open. No cloud billing or automatic model promotion is involved.

Preflight: native release build and 37 Python tests passed. Both first seeds
completed initial full-profile updates/checkpoints with finite metrics and
untruncated records. This verifies that training actually runs; early cold-start
errors are not evidence of equilibrium strength.

## Release audit — 2026-10-14: rejected, not activated

All four seeds completed their eight-hour training budgets normally. The 50bb
seeds finished 465/462 rounds (186,000/184,800 traversals); 100bb finished
366/370 rounds (146,400/148,000 traversals). Training completion did not satisfy
the existing release criteria. Neither depth is enabled, no winning seed was
selected, and the active experimental 20bb manifest/weights are unchanged.

The held-out authentic evaluation used an equal mixture of both frozen policies
over exact-card trajectories, with separate uniform-policy forced deviations.
Every evaluation uses the exact trained stack depth and spread-out action grid;
the validator now checks returned game metadata and can pin the sampler binary
by SHA-256. The native certificate entry point also accepts the explicit grid.
This fixes evaluation identity, not poker strategy or measured exploitability.

| Measured criterion | Existing gate | 50bb | 100bb |
| --- | --- | --- | --- |
| Cross-seed action-frequency MAE | at most 5 percentage points | 7.89 — fail | 7.42 — fail |
| Primary-action agreement | at least 85% | 73.49% — fail | 76.00% — fail |
| Maximum aggregate action delta | at most 3 percentage points | 2.28 — pass | 2.13 — pass |
| Authentic decisions evaluated | sample size, not a gate | 17,207 | 16,126 |
| Forced-deviation decisions | sample size, not a gate | 6,743 | 6,944 |

Observed inference coverage and normalized probabilities passed on both sampled
distributions. This is not exhaustive coverage or equilibrium proof. Preflop
primary-action agreement was especially weak (66.33%/69.37%). All four final
binary artifacts have finite parameters, correct grids/depths, and identical
bytes after source re-encoding; together they occupy 10,585,386 bytes. This is
not a total hosted-storage audit of a future sample/oracle serving corpus.

A separate independent EV reference pass used 128 traversals per reach dataset
and 64 rollouts per action. Only 5.06%/2.53% of evaluated decisions met the
0.02bb all-action reference-SE threshold, versus the existing 95% requirement.
The exported value heads' own policy-action-weighted predicted precision
coverage was also only 36.13%/30.07% (50bb seeds) and 34.73%/33.72% (100bb).
Those predictions describe Monte Carlo target noise, not calibrated neural
approximation error. More reference samples alone would not resolve the
cross-seed failures or validate learned action-EV accuracy.

Full-game exploitability and its 99% upper bound were **not measured** in this
audit: decisive normal-gate failures already reject both candidates. No numeric
exploitability improvement, regression, or pass is claimed. The existing
0.50bb/hand release thresholds were not changed or waived. Off-grid responses
and full-hand evaluation remain necessary for any future accepted candidate.

The machine-readable [rejection decision](depth-release-audit-2026-10-14.json)
records measured results, artifact/report hashes, evaluation budgets, and
unmeasured gates. The original reports and weights remain in the full-run
output directory, outside Git. Older reports' false exploitability booleans
must not be read as a measured failure: their certificates are absent/null.
The validator now distinguishes unmeasured gates from measured failures and
reports release eligibility without activating models.

Final verification: 47 Python tests passed (including a tiny native certificate
transport test, not a strength certificate); nine Rust CLI release tests
passed; 290 TypeScript tests passed with four unrelated optional native tests
skipped, including four real-artifact full-hand compatibility tests; production
build passed. No browser-facing implementation changed in this audit.

## Depth repair — 2026-10-15

The user authorized the following implementation and short matched comparisons,
not another unconditional eight-hour run or activation of rejected exports.

1. **Frozen-policy EV qualification.** `neural-frozen-values` accepts sampled
   public decision records and exact depth/grid-bound policy weights. It replays
   the complete legal history, computes opponent-card posterior weights from
   observed action likelihoods, removes known cards, and samples independent
   opponent/future-board blocks. Within each block it uses shared continuation
   random streams across actions. SE is computed across outer block means, not
   just repeated actions with a fixed hidden deal. Complete runouts settle
   exactly. This is an offline evaluator, not a change to the live resolver.
2. **Actual regret variance.** `--average-regret-rollouts` reuses the existing
   independent action-value samples in the instantaneous regret targets. The
   default sampler remains unchanged; extra samples previously only improved
   EV-head labels. The primary traversal return and visitation accounting stay
   unchanged. Compare the existing control-variate baseline against scale zero
   rather than assuming that an inaccurate baseline must reduce variance.
3. **Matched training-rule comparison.** `--regret-training
   retained_instantaneous` retains bounded raw-regret reservoirs, uses linear
   iteration-weighted losses, and refits the regret networks from deterministic
   fresh initialization each round. It does not recursively bootstrap previous
   predicted cumulative regrets. This opt-in is a different training algorithm,
   with a different pinned identity; old checkpoints cannot silently resume in
   it. No claim that the alternative is superior is justified without pilots.
4. **Correct averaging reference.** `--retain-policy-snapshots` saves the policy
   actually used before each sampling round, including the initial uniform
   policy. It records separate player weights for alternating traversals.
   Snapshot queries multiply these weights by each policy's own action reach
   along the observed history. Arithmetic probability/logit ensembles and
   post-update-only checkpoints are not equivalent. This exact pilot history is
   bounded to 128 rounds and stays outside serving manifests.
5. **Small independent response checks.** A local deviating action is selected
   using the first half of independent hidden blocks, then measured on the
   other half. Negative held-out gains are retained, not clipped into artificial
   improvement. These street-stratified local deviations are not bb/hand
   exploitability, best-response certificates, or evidence of full-game safety.

`preflop-solver/neural/depth_repair.py` performs these probes using a copied,
SHA-256-pinned native binary. Four profiles each use two independent seeds:
control, averaged regrets, retained raw regrets plus averaging, and averaged
regrets with the baseline disabled. The action grid stays fixed. Paired
comparisons use the same held-out control-policy reach and forced-deviation
corpora. Configurations match rounds/traversals/optimizer steps; actual wall
time is reported separately. At most two training processes run concurrently,
with per-seed time/RSS/disk guards and rejection of truncated/nonfinite data.

Frozen EV refits are separate from policy training. Train/calibration/test
trajectory families are disjoint. The stored residual interval includes
reference noise and is explicitly not called sampling SE or a conditional
error guarantee. Tiny pilot fits do not qualify a depth for release.

The initial native integration test exposed JSON `50` versus `50.0` identity
comparison. Typed numeric/grid normalization fixed it; wrong depth/grid and
invalid betting snapshots still fail closed. No strategy run started during
that failed preflight. Its output was retained under
`/Users/davidvayntrub/pokersolver-training/depth-repair-2026-10-15`.
The corrected matched experiment is under the sibling directory
`depth-repair-2026-10-15-v2`.

Research basis: [Deep CFR](https://proceedings.mlr.press/v97/brown19b/brown19b.pdf)
studies retained sampled regrets and regret-network refitting;
[SD-CFR](https://arxiv.org/abs/1901.07621) derives own-reach-weighted snapshot
averaging; [low-variance baselines](https://proceedings.mlr.press/v119/davis20a.html)
explain unbiased control-variate variance reduction. These papers motivate
controlled experiments; their results do not establish strength of our variant.

### Completed pilot decisions

The four-profile screen completed eight independent 12-round runs (64
traversals and 80 optimizer steps per round), with 1,597 common held-out
authentic decisions. These cold-start results are not a comparison against the
eight-hour checkpoints:

| Profile | Cross-seed frequency MAE | Primary-action agreement |
| --- | --- | --- |
| Existing control | 14.62 percentage points | 42.17% |
| Averaged regret targets | 12.93 | 55.89% |
| Retained raw regrets + averaging | 14.19 | 50.58% |
| Averaging, baseline disabled | 13.50 | 47.45% |

Averaging is the best directional stability result, not a release candidate.
The original 5-point/85% gates remain unmet. The initial local response probes
were too noisy to justify a longer run.

The isolated compression follow-up used 1,597 decisions, whole-trajectory
train/tuning/test splits, and 100/400-step fits selected only on tuning data.
For the averaged-regret pair, held-out student/reference MAE improved from
8.70 to 7.11 points and 7.80 to 6.55 points. However, polished students did not
improve cross-seed consistency (12.97-point MAE, 53.41% primary agreement).
Do not mistake better teacher fitting for improved poker strength or promote
these students.

The final terminal-response integration ablation reran the averaged control
and added an integrated pair, on one new pinned binary. Both control seeds
reproduced all 660,868 parameter values exactly; binary identities intentionally
changed because the sampler SHA is part of the model version. A deterministic
regression also checks that integration preserves primary chance draws,
visited states, and average-policy records while changing estimates only.
The sampler integrates only nodes whose every response is terminal; it retains
the configured all-in runout evaluation and never closes an open betting tree.

This follow-up used 1,428 common decisions and 256 independent hidden blocks
per local response root, eight times the first screen's budget. Integration
worsened MAE (14.07 vs 12.73 points), primary agreement (48.39% vs 58.51%),
and held-out local response gains (2.02/4.68bb vs 1.55/1.19bb, with respective
SEs 0.37/0.40 and 0.45/0.39bb). Those gains are per stratified sampled decision,
not bb/hand exploitability. Reject this option as a default at this budget;
local estimator variance reduction did not translate into better learned play.

All four old frozen exports also received full-hidden-scenario probes and
separate EV-head refits. Nominal 95% interval coverage was only 61.1%/70.6%
(50bb) and 66.7%/43.2% (100bb). The tiny frozen refits still require wide empirical
intervals (12.4–22.6bb half-widths), not precise grading claims. These small,
noisy, stratified probes reject qualification; they do not estimate global
prediction accuracy or qualify authentic reach-weighted release coverage.

`validate_seeds.py` now emits schema v12. Its unchanged 0.02bb/95% EV threshold
requires an artifact-bound frozen information-set precision/calibration receipt
through `--frozen-ev-qualification`. Old conditional-rollout SE is retained as
a diagnostic, not release evidence. Wrong hashes/depth/grid fail closed;
stratified pilot receipts are explicitly ineligible. An actual CLI replay
confirmed both old 50bb exports remain rejected and exploitability remains
unmeasured. The existing experimental 20bb manifest is not recertified or
changed by this tooling.

See the [machine-readable audit](depth-repair-audit-2026-10-15.json) for exact
results and report hashes. Carry averaged regret targets forward only for a
subsequent matched policy-quality experiment. Do not enable the other options
or start another eight-hour run on these results. No new depth was released.

The guarded full-run supervisor now accepts `--average-regret-rollouts` as an
explicit opt-in, pins that estimator in its plan, and checks sampler support
before launching. Legacy defaults/resume identities are preserved where the
new flags are omitted; changed source hashes still require a fresh run directory.
This is preparation only, not authorization or evidence for another long run.

Final checks: 65 Python tests passed with real native transport/estimator probes;
463 Rust library release tests and nine CLI tests passed, with 59 guarded
research tests ignored; 290 TypeScript tests passed, with four optional native
tests skipped; `npm run build` passed and reverified all active 20bb artifact
hashes. No browser-facing code or active model manifest changed.

## Longer matched estimator comparison — 2026-10-17

The user authorized a longer 50bb comparison, a conditional 100bb replication,
and subsequent validation, with major checkpoints committed and pushed.
`run_matched_depth_repair.py` pins the implementation and predeclares the
following screen before any training results are observed:

- Original versus averaged regret targets, each with seeds 51701/51702.
- 64 rounds, 256 traversals/round (16,384 traversals/seed), 160 optimizer
  steps/round. Same network, action grid, replay, baseline, and all other
  training parameters. This is a longer cold-start comparison, not a resume or
  a claim to outperform the completed eight-hour checkpoints.
- Two concurrent training processes, at most 35 minutes per run plus a
  two-minute hard-stop allowance, 3.5GiB RSS/owned process family, and at least
  20GiB free disk. Truncated/nonfinite records reject the experiment.
- Two independent held-out evaluation seeds. Each evaluation batch mixes
  authentic trajectories from all four frozen policies equally, giving both
  estimators exactly the same comparison distribution. Whole-hand family IDs
  are namespaced across independent corpora.
- 32 distinct street-stratified response roots, 512 independent hidden
  opponent/runout blocks/root, two inner continuations/block. Action selection
  and measurement use disjoint halves. Both configurations see the same roots
  and random seeds, but their own posterior and continuation policy.
- Proceed to the same 100bb comparison only if frequency MAE decreases and
  primary agreement increases on **both** evaluation batches, both training
  seeds reduce their held-out local response gain, and a paired-root bootstrap
  has a negative one-sided 95% upper delta. Seed pairs remain grouped when
  resampling whole roots. This is an exploratory conditional screen, not a
  full-game exploitability confidence bound or an across-training-seeds proof.
- Run the ordinary independent release validator on both frozen seed pairs.
  Missing authentic EV qualification or a missing full-game certificate stays
  unmeasured. Expensive strength qualification is not justified by a candidate
  already failing the measured stability/precision checks. No activation or
  automatic eight-hour extension is built into this runner.

Initial checks: 68 Python tests passed, including the explicit native
transport/estimator tests and the new matched-budget/promotion-guard tests.
The 50bb run is saved outside Git under
`/Users/davidvayntrub/pokersolver-training/depth-repair-longer-2026-10-17-50bb`.
Its output records frozen source/binary hashes, checkpoints, reference samples,
the paired response analysis, and ordinary validation reports.

### Checkpoint recovery during the comparison

Foreground CPU/GPU contention substantially slowed late updates. Both controls
had already completed 64 rounds. The averaged seeds received a supported
graceful stop request and finished/checkpointed rounds 60/57; no sampler was
killed mid-update. The incomplete attempt remains preserved, including its
source hashes and summary hash. This is a scheduling interruption, not a
measured failure of the estimator.

`--resume-existing` now resumes only the missing updates and skips completed
runs without retraining them. It requires the same depth, seeds, sampler,
iteration budget and predeclared evaluation settings; learning/evaluation
source changes are rejected. Only the two scheduling drivers may change during
recovery, and the old summary is retained under its content hash. Recovery is
bounded to two attempts with the same per-invocation time/RSS/disk guards.
The fixed 64-round/16,384-traversal target did not increase. Wall times from
this experiment must not be used to rank estimator efficiency because of the
external contention and checkpoint overhead.

The real native regression compares three uninterrupted updates against two
updates followed by a checkpoint resume: all exported parameters match exactly.
Reopening an already-complete run leaves its update history unchanged. Seven
new matched-run tests and the 65 existing Python/native checks passed.

### Completed 50bb comparison; conditional 100bb replication

All four runs completed exactly 64 rounds and 16,384 traversals. The recovery
did not alter the learner or add updates. On both fixed common held-out
distributions, averaged regret targets improved stability:

| Batch | Control frequency MAE | Averaged MAE | Control primary agreement | Averaged agreement |
| --- | --- | --- | --- | --- |
| 1 (3,115 decisions) | 8.63 points | 8.19 points | 64.14% | 72.24% |
| 2 (3,119 decisions) | 8.89 points | 7.93 points | 62.99% | 72.87% |

Independent held-out local response gains decreased in both seed pairs:
1.181/1.092bb to 0.789/0.862bb **per sampled decision**, not per complete hand.
The pooled difference is -0.3113bb, with a paired-root one-sided 95% bootstrap
upper delta of -0.0062bb. That only narrowly satisfies the predeclared screen.
It is promising conditional evidence across these 32 roots and two training
seeds, not proof of a full-game exploitability decrease or a release pass.

The ordinary independent validator still rejects the averaged 50bb pair:
7.86-point frequency MAE versus the 5-point limit, and 72.75% primary agreement
versus 85%. Maximum aggregate action delta (0.97 points), sampled authentic
and forced lookup coverage, probability sums, and artifact integrity pass.
This ordinary validator uses a separate seed and the pair's own reach mixture,
not the four-policy common distribution in the table.

EV feedback is also not qualified. On the small stratified frozen probes,
nominal 95% intervals covered references only 37.9%/50.3%; only 5.5% of
reference action estimates met the 0.02bb SE threshold. Those action-level
diagnostics are not the authentic reach-weighted decision qualification gate.
No valid EV receipt or full-game exploitability certificate was generated;
both remain unmeasured release gates. Strength qualification is not claimed
from local response gains or cross-seed stability.

Because the **screen** passed, the authorized 100bb replication has started
with the same frozen implementation, two seeds, action grid, and fixed
64x256/160 budget. Its per-invocation wall-time guard is explicitly 60 minutes
to account for the observed contention and deeper tree; iteration counts and
model architecture are unchanged. No paid compute or serving promotion is
involved. The final 50bb reports/weights remain outside Git; their hashes and
the decision are recorded in
[the 50bb audit](depth-repair-longer-50bb-2026-10-17.json).
