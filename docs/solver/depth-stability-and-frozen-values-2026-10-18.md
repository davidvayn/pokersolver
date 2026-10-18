# Depth stability and policy-pinned EV repair — 2026-10-18

## Scope and predeclared decisions

Keep the game, action grid, gates, and active experimental 20bb artifacts unchanged.
Do not start a longer run or activate 50bb/100bb. First distinguish disagreement
in the learned regret-policy history from average-policy compression error. Fix
the serving-value target contract independently of policy training.

The short upstream-learning comparison uses 50bb, seeds 51701/51702, 24 rounds,
128 traversals/round, and 240 fitting steps/round for both arms. Compare the
existing bootstrapped DCFR+ learner with the existing retained-instantaneous
linear-CFR learner and fresh per-round regret refits. This is an algorithm
comparison, not an isolated discount-exponent ablation. The network, replay
capacity, action grid, four-rollout regret estimator and baseline scale stay
fixed. The earlier 12-round/80-step retained experiment was not positive; this
better-fitted replication is not evidence that it will now work.

Use two common held-out reach mixtures and independent selection/evaluation
blocks for local response gains. Require better MAE and primary agreement on
both batches, lower local gain in both seeds and a negative paired-root 95%
upper delta before recommending an extension. Local gains remain bb/decision,
not full-game exploitability. No automatic extension is built in.

## Isolation on existing 64-round checkpoints

`isolate_depth_stability.py` samples 128 IID decisions per saved common mixture,
preserving duplicate draws and existing whole-hand namespaces. Native inference
evaluates the exact retained pre-update policy history with player-specific
iteration weights and own-reach weighting. No model is retrained in this check.

| Depth / batch | Export cross-seed MAE | Direct-average cross-seed MAE | Export primary agreement | Direct-average agreement |
| --- | --- | --- | --- | --- |
| 50bb / 1 | 7.51 points | 8.00 points | 78.91% | 73.44% |
| 50bb / 2 | 6.95 points | 8.21 points | 75.00% | 67.97% |
| 100bb / 1 | 6.98 points | 8.15 points | 62.50% | 60.94% |
| 100bb / 2 | 7.12 points | 8.21 points | 73.44% | 69.53% |

Student-to-own-average errors are 4.97–5.71 points. These errors are NOT additive.
Removing compression does not remove cross-seed disagreement: upstream policy
histories already differ. Compression is not the sole explanation, and another
distillation pass is not the justified next policy experiment. Agreement alone
also cannot distinguish harmful differences from strategically equivalent mixes.
The bounded 128-draw checks are diagnostics, not replacement release validation.

Saved roots and hashes are under
`/Users/davidvayntrub/pokersolver-training/depth-stability-isolation-2026-10-18-{50,100}bb-v2`.
The initial empty output directories record an early namespace-check failure;
the corrected sampler preserves already-namespaced IDs without changing them.

## EV target contract and reproducible fix

The existing value replay contains returns under changing regret-matching
policies. The exported playing policy is the learned average policy. The old
uncertainty label is continuation-action sampling noise conditional on a sampled
deal, not total information-set sampling uncertainty or prediction error.

New training exports explicitly mark those values as historical-policy training
control variates, unmatched to serving and ineligible for release qualification.
Existing frozen artifacts are never rewritten. A regression first failed on the
missing contract and now verifies the actual export, not a mock metadata helper.

`frozen_policy_values.py` implements a separate, artifact-bound fitting path:

1. Verify the source metadata and FP32 parameter bytes against the immutable
   binary and descriptor. Verify the reference input, policy-network and game
   hashes, policy-only routing, and per-decision state/action alignment. Native
   FP32 cash-amount round trips are checked exactly at FP32 precision, not with a
   loose poker-state tolerance.
2. Accept only full hidden-opponent/future-board frozen-policy references for
   that exact exported policy. Historical training replay cannot supply labels.
3. Split whole trajectory families into train/tuning/calibration/test partitions.
   Preserve duplicate decisions in their original family. Test families never
   select step counts or uncertainty widths.
4. Initialize a separate mean network from the original mean, retaining its
   architecture. Compare 400 and 1,600 fixed-policy fitting steps on tuning data.
   The unchanged training head is a control, never a step-zero candidate that
   could be relabeled as a policy-pinned fit.
5. Export a one-output, bb-unit value sidecar bound to the policy artifact hash.
   Calibration supplies an empirical residual envelope, explicitly NOT sampling
   SE. No sidecar can claim precision qualification or authorize activation.

Four matched fits reuse the existing 128-decision, 2,048-hidden-block references.
This avoids collecting the same expensive labels again. Before/after errors use
the SAME untouched test families, unlike the earlier separate-refit summaries.
They are pilot point estimates against noisy references, not confidence proofs.

| Depth / seed | Selected steps | Tuning improved? | Test RMSE before → after (bb) | Centered action error before → after (bb) |
| --- | --- | --- | --- | --- |
| 50bb / 51701 | 400 | yes | 5.675 → 4.318 | 3.425 → 2.815 |
| 50bb / 51702 | 400 | no | 4.706 → 5.819 | 3.382 → 3.694 |
| 100bb / 51701 | 1,600 | yes | 8.240 → 5.657 | 5.246 → 3.899 |
| 100bb / 51702 | 1,600 | no | 8.601 → 6.552 | 6.587 → 4.442 |

Do not select 100bb seed 51702 because its test error happened to improve: its
predeclared tuning comparison failed. Neither depth has a two-seed validated
value improvement. Residual half-widths remain 6.55–16.40bb, far from precise
EV-loss grading. The previous independent reference-precision failures remain.
Correct target identity does not make these small fits accurate automatically.

Final reports/sidecars are under
`/Users/davidvayntrub/pokersolver-training/frozen-policy-values-2026-10-18-{depth}bb-{seed}-v3`.
Earlier exploratory outputs are retained, not substituted for final evidence.

## Offline variance-reduction pilot

The opt-in native `--integrate-terminal-responses` evaluates the exact policy
expectation only where EVERY child is terminal. It never closes open betting
subtrees, alters the trained policy, leaks future cards to action selection, or
removes outer hidden-scenario variance. Complete sampled boards still determine
showdown payoffs. A native regression verifies exact terminal expectations,
nonzero legacy terminal-action noise, and preservation of open trees.

`probe_frozen_value_precision.py` compares sampled and integrated references on
the same two frozen 50bb policies, 16 independent street-stratified roots, 512
hidden blocks, two inner rollouts and common seeds. Evaluation order reverses
for seed two. Wall-time ratios may include host contention and are not isolated
performance claims. This read-only evaluator experiment is distinct from the
previously rejected terminal-integration TRAINING variant.

Both seeds completed. Mean action sampling variance decreased by 12.94%/12.79%.
Median worst-action SE decreased from 1.127 to 1.020bb and 1.141 to 1.051bb.
Observed variance-times-wall-time ratios were 0.674/0.790, under host contention;
these are not controlled speedup estimates. Both estimators still qualified
0/16 sampled decisions. Terminal-action integration helps but does not resolve
hidden-card/runout noise. It stays opt-in and is not silently enabled in training.

## Checkpoint verification

- Targeted Python/native checks: 69 passed, including actual native sampling,
  deterministic resume and chunked qualification tests.
- Rust release: 464 library, nine binary and 15 CLI tests passed; 59 explicitly
  resource-guarded library experiments ignored.
- TypeScript: 286 passed, 16 optional integration cases skipped.
- `npm run build`: passed and reverified the unchanged active 20bb artifacts.
- Old/new default evaluator: identical probabilities, raw hidden blocks, means
  and SE on a pinned native differential fixture; the new estimator is opt-in.
- Full Python discovery: 768 cases, four errors and five skips. Two errors are
  OS permission failures in unchanged cash process-group cleanup tests; two
  require an absent ignored `runs/20bb-v26-routed-seed5101.json` legacy fixture.
  Those modules were not edited. Do not report the entire Python suite green.

The short matched policy pilot remains running at this implementation checkpoint.
Machine-readable metrics and external report hashes are in
[the checkpoint audit](depth-target-repair-audit-2026-10-18.json).

## Research basis

- [Deep CFR](https://proceedings.mlr.press/v97/brown19b.html): neural regret and
  average-strategy approximation are distinct learning problems.
- [Single Deep CFR](https://arxiv.org/abs/1901.07621): motivates checking a direct
  historical average before attributing policy errors to its compressed head.
- [Low-variance baselines](https://proceedings.mlr.press/v119/davis20a.html):
  motivates measuring variance reduction rather than assuming more samples
  alone will make an inefficient estimator practical.

No cross-seed, RMSE, calibration, or local-response statistic here is a full-game
exploitability certificate. The <=0.50bb/hand mean/99%-upper release gates are
unchanged and remain unmeasured for these candidates.
