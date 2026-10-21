# Home v5 bootstrap, generalization, and EV variance

## Scope and ordered decisions

Follow-up to `card-memory-and-label-repair-2026-10-19.md`. Keep correct public
card-arrival memory. Do not change active 20bb weights, action grids, rake,
release gates, or old artifacts. No long/paid training is authorized by this
bounded diagnostic sequence. Run 100bb only after a credible 50bb improvement.

1. Complete fixed-target 240/480/960-step fits on actual v5 replay, separately
   for both regret networks and the average-policy network. Compare full recall
   with masked arrivals, using identical initialization/targets/minibatches.
2. Compare bootstrap priors with the actual Rust traversal scorer; repair a
   demonstrated inference inconsistency behind a new, hash-pinned opt-in flag.
3. Run a short two-seed, single-factor 50bb policy comparison. Require stability
   improvement on both independent batches and credible lower local-response
   gains before conditional 100bb replication. Local gains are not exploitability.
4. Independently test bounded action expectations for frozen-policy EV labels.
   Judge variance and cost, not merely more work per reference. Do not refit a
   neural mean head against labels that remain inadequate and call it qualified.

## Fixed-target finding: additional regret fitting overfits

The new probe opens the original replay read-only and checks exact file byte
lengths against 716/820 state features. The old raw-regret probe cannot simply
be reused with a hard-coded 716-column memmap: a larger file can be silently
reinterpreted with incorrect row strides. No original replay is rewritten.

Both ablation arms use a common split grouped by the legacy projection, so
histories collapsed by the masked-arrival arm cannot leak across its holdout.
This is an information-set holdout, **not** whole-hand-family independence;
the old replay has no family IDs. Train/held-out metric subsets are fixed and
capped at 2,048 decisions for bounded cost. Budgets are nested checkpoints of
the same fitting stream, not independently randomized runs.

Full-recall regret weighted MSE, normalized stack-fraction squared:

| Frozen replay | Train, 240 → 960 | Held-out, 240 → 960 |
| --- | --- | --- |
| 51701 / p0 | .01109 → .00279 | .33186 → .36112 |
| 51701 / p1 | .01567 → .00237 | .12403 → .13111 |
| 51702 / p0 | .01809 → .00228 | .20274 → .23945 |
| 51702 / p1 | .05476 → .01118 | .20009 → .24531 |

All four training errors improve while all four held-out errors worsen.
Masked arrivals do not consistently remove the problem. This supports a
generalization/noisy-target bottleneck, **not** a blanket recommendation for
four times more regret fitting or reverting correct recall. It does not isolate
sample sparsity, bootstrap bias, and shared-street interference from each other.

The average-policy network is different: held-out weighted KL improves at 480
steps (.20221 → .19764 and .19072 → .17795). A new optional
`--average-policy-steps-per-round` permits a separate fitting budget without
changing regret/value updates. It is not enabled in the CPU-prior policy pilot.

Output: `pokersolver-training/home-v5-fit-2026-10-20-v2.json`.
SHA-256: `562d411301f958ffd29f352609dff32bf640e6e581d78eaa302ddfa9ecd66ec3`.
The initial report is preserved as exploratory; v2 fixes the common ablation
split. These historical bootstrapped targets are not comparable numerically
with the earlier retained-instantaneous-regret diagnostic.

## Confirmed numerical inconsistency

`probe_home_bootstrap.py` passes real saved weights and grouped poker features
through an explicitly enabled Rust test seam calling the **actual**
`DenseScorer::score_state_actions`. This is not a NumPy reimplementation. Both
MLX inference paths receive exactly the same float32 features/parameters.
Training-state/replay hashes are checked before and after this read-only probe.

| Encoding / seed | GPU-vs-native maximum, p0 / p1 (bb) | CPU-vs-native maximum, p0 / p1 (bb) |
| --- | --- | --- |
| v5 / 51701 | .18927 / .13057 | .0000298 / .0000238 |
| v5 / 51702 | .26959 / .21055 | .0000238 / .0000358 |
| v4 / 51701 | .18506 / .17231 | .0000402 / .0000358 |
| v4 / 51702 | .22563 / .13366 | .0000238 / .0000268 |

Home bootstrapping recycled GPU prior predictions into cumulative-regret
targets although traversal policies use native FP32 arithmetic. The mismatch
is real and material relative to the .02bb EV precision target, but these are
regret-prior differences, **not** measured EV errors or exploitability. The
problem also exists in v4, so it is not proof of why v5 specifically worsened.

`--bootstrap-cpu-prior` now opts Home training into CPU FP32 prior evaluation;
optimization remains on the GPU. It is part of the run configuration hash and
therefore cannot silently resume/relabel an old GPU-prior run. Existing flags,
configuration hashes, default 20bb training, and serving remain unchanged.
CPU is close to native, not claimed bit-identical. Regression tests exercise
`ingest_records` and actual uninterrupted-versus-resumed v5 training.

Output: `pokersolver-training/home-bootstrap-parity-2026-10-20-v2/summary.json`.
SHA-256: `fd9d416b6fd613c4c227793d5215174cd0592a9d4f99b17e6eead81bcf4a25fa`.
An initial partial report is retained; v2 completes both seeds and both encodings.

### Hardware-level cause and a second isolated treatment

The host is Apple M5 with MLX 0.32.0. MLX's
[precision documentation](https://ml-explore.github.io/mlx/build/html/usage/precision.html)
explains that float32 matrix operations can use reduced-precision hardware
paths unless `MLX_ENABLE_TF32=0` is set at process launch. Merely keeping tensor
dtypes float32 does not prevent this. CPU-prior correction alone does not change
the GPU arithmetic used to fit all networks.

Repeating the **same frozen-weight/native-scorer** experiment in a fresh process
with that setting reduces maximum GPU mismatch from .26959bb to .00008345bb
across the four v5 heads. This isolates a hardware precision source of the
mismatch, not a convergence or exploitability improvement.

The full-FP32 fixed-replay probe still worsens regret holdout error when going
from 240 to 960 steps in all four networks. Thus the memorization/noisy-target
finding survives eliminating the large arithmetic discrepancy. The mean-policy
fit still prefers a moderate budget; no claim that precision alone repairs the
remaining training-data/generalization bottleneck.

Outputs under `pokersolver-training/`:

- `home-bootstrap-parity-full-fp32-2026-10-20/summary.json`, SHA-256
  `0bebed4292eb313cbf3a295784bd11c661338b54a84a936781e421dd8faf0f8f`.
- `home-v5-fit-full-fp32-2026-10-20.json`, SHA-256
  `f4ba18bb16054760e6697c0887aa84c6a18e3cd25b5485a57b2f9d4f475156dd`.

## Matched policy pilot

`home-cpu-prior-pilot-2026-10-20-50bb`: v5 GPU-prior control versus v5 CPU-prior,
seeds 51701/51702; 24 rounds × 128 traversals, 240 steps/head/round, batch 512,
256/128 hidden layers, four action-value rollouts, unchanged frozen sampler and
game grid. Two training processes have explicit time/RSS/free-disk guards.

Evaluation uses two equal-policy-mixture batches and 32 disjoint,
street-stratified response roots. Response selection and scoring use disjoint
hidden blocks. The driver preserves source/binary/config identities and records
ordinary independent release-validator failures rather than promoting a pilot.

The original pilot completed but used the old GPU audit path for stability.
`audit_home_precision.py` preserves that report, reuses the identical corpora,
and recomputes the measurements with CPU FP32. All four policies were also
queried through the actual frozen native policy: maximum CPU/native action
probability difference is 7.25e-7 on the 32 response roots. This measurement
correction is not counted as a policy improvement.

Native-aligned stability on the same fixed equal-policy mixtures:

| Metric | Batch 1: GPU-prior → CPU-prior | Batch 2: GPU-prior → CPU-prior | Gate |
| --- | --- | --- | --- |
| Frequency MAE | 12.900 → 11.829pp | 12.876 → 11.749pp | ≤5pp |
| Primary agreement | 47.357 → 53.054% | 45.682 → 53.915% | ≥85% |
| Max aggregate delta | 6.880 → 2.207pp | 7.262 → 2.204pp | ≤3pp |

Aggregate action stability now passes these sampled comparisons, but the other
two stability criteria do not. The independently sampled release validator
also rejects both pairs. The unchanged native local-response results are:

- 51701: 1.59355 → 1.29654bb/decision (improves).
- 51702: 1.64359 → 1.74162bb/decision (worsens).
- Pooled delta −.09949; one-sided 95% whole-root bootstrap upper delta +.26537.

Thus the preregistered improvement screen fails; no CPU-prior-only weights are
promoted, and no 100bb replication is justified by this result. All eight saved
networks (four heads × two seeds) in the v5 GPU controls reproduce the
previous day's controls bit-for-bit. Neither optional fitting controls nor
CPU-prior code silently changed legacy training.

Reports under `pokersolver-training/`:

- `home-cpu-prior-pilot-2026-10-20-50bb/summary.json`, SHA-256
  `7d9dbe42e98f1ce4469d952656e10ea7cf9252c365d246532578a8f0435bcfd7`.
  Original source snapshots match every recorded source hash.
- `home-cpu-prior-native-audit-2026-10-20-50bb/summary.json`, SHA-256
  `ed8b5be1e5d44a1eb0e44b74d58878ce36b84425576350fd1d7242974fdf9103`.

The second isolated comparison, `home-full-fp32-pilot-2026-10-20-50bb`, keeps
CPU priors in both arms and changes **only GPU fitting precision**. Existing
frozen CPU-prior controls are reused without retraining, resuming, or modifying
them. Their configuration fingerprints and immutable artifact hashes are
verified against the completed first experiment, whose original source hashes
are retained separately from the new treatment's source hashes. All other
configuration fields must match. New ordinary validation uses CPU inference.
The launch math mode is now part of the run identity; incompatible resumes
fail before changing checkpoint state.

The full-FP32 treatment also completed and failed its predeclared screen:

| Metric | Batch 1: CPU-prior control → full-FP32 fit | Batch 2: control → full-FP32 fit |
| --- | --- | --- |
| Frequency MAE | 11.855 → 12.928pp | 11.518 → 12.369pp |
| Primary agreement | 52.096 → 50.890% | 54.664 → 50.309% |
| Max aggregate delta | 2.147 → 4.937pp | 2.342 → 4.837pp |

Local-response gain increases in both seed point estimates: 1.54976 → 1.94726
and 1.54851 → 1.63860bb/decision. Pooled delta +.24380; one-sided 95% bootstrap
upper delta +.73122. These weights are rejected, not installed. This is evidence
against promoting this short treatment, not a universal claim that full-FP32
optimization is worse. More accurate matrix arithmetic is not equivalent to
more accurate learned strategy at a fixed, noisy sample budget.

The two comparisons use different fixed reach mixtures/roots. Compare arms
within each experiment, not point estimates across their different roots.

Full-FP32 reports under `pokersolver-training/`:

- `home-full-fp32-pilot-2026-10-20-50bb/summary.json`, SHA-256
  `ccecd27d849cbb85e69877dede50a5f692bb6d321894623666a410bfd59aac06`.
- `home-full-fp32-native-audit-2026-10-20-50bb/summary.json`, SHA-256
  `8b331a3f91759971cde2979f51c2659fd178302a3e4b60be334231080f9de610`.
  All four CPU/native probability differences are below 8.90e-7. The final
  audit also evaluates value heads on CPU and verifies JSON metadata/parameter
  bytes against the immutable binary, rather than trusting its descriptor alone.

The separate, native-aligned CPU-prior validation sample records MAE **11.604pp**,
primary agreement **52.952%**, aggregate delta **1.717pp**. Aggregate stability,
sampled lookup/probability validity, reach weighting and seed independence pass;
MAE and primary agreement fail. Frozen served-EV confidence and full-game
exploitability are **unmeasured for these new policies**, not measured passes or
measured .50bb/hand upper bounds. The EV estimator pilot below uses older v5
policies and cannot qualify either new pair.

## Frozen-policy action-expectation pilot

`--action-expectation-depth 0..2` sums a bounded prefix of future actions using
the actual frozen policy probabilities. Descendants receive separately seeded
continuation streams; no independence between action columns is assumed.
Opponent private cards and future boards
remain sampled in independent outer blocks. Policies still see only their own
information sets. This is conditional action integration, not an exact
open-street information-set value or zero sampling error.

Depth zero delegates directly to the original rollout with the original seed.
Exact river/closed-action labels are unchanged; their estimator tags remain
distinct. Chunked and single-call sampling retain the same global root seeds.
The frozen-value verifier accepts only bounded integer prefix depths and still
binds labels to the actual immutable exported average policy, not the regret
policy, historical value head, or a new trained policy under an old URL.

`run_home_ev_variance.py` compares depth zero/one on both old v5 average policies,
16 disjoint street-stratified roots, 128 hidden blocks, one inner rollout.
It requires at least 5% lower sampled-action variance **and** lower
variance-times-wall-time in both seeds before considering a larger label run.
Host contention means wall times are exploratory, not a controlled speed claim.
This probe is not the authentic-reach 95%-of-decisions confidence gate.

Completed 128-block results:

| Seed | Mean sampled-action variance, before → after (bb²) | Reduction | Runtime ratio | Variance × runtime ratio |
| --- | --- | --- | --- | --- |
| 51701 | 994.80 → 588.45 | 40.85% | 2.42 | 1.43 |
| 51702 | 1040.12 → 718.86 | 30.89% | 1.39 | .96 |

All exact values/constants and frozen policy probabilities remain identical.
All-action reference precision remains 25% in both seeds. The small screen
does not justify a larger label run or a neural refit. Timing is confounded by
concurrent tests/build/training, so the cost result is not a controlled claim
that this estimator is slower or faster. Conditional variance reduction is
real on these retained blocks; adequate .02bb confidence is not established.

Output: `home-open-action-variance-2026-10-20-50bb/summary.json`, SHA-256
`2413730d507a8ce4adfb4bedf2dcf7a14d6ddaf78daddb9a42f1253b7c982a08`.

## Research basis and limitations

- [Deep CFR](https://proceedings.mlr.press/v97/brown19b/brown19b.pdf) distinguishes
  sample budget, network fitting, and final average-policy approximation. Its
  function-approximation bound is not a certificate for this custom bootstrapped
  DCFR+ learner. In particular, our fixed-target overfitting result argues against
  assuming more optimization steps must help.
- [Low-Variance and Zero-Variance Baselines](https://proceedings.mlr.press/v119/davis20a.html)
  motivates conditional expectations/control variates. Our bounded prefix method
  removes selected action draws only; it is not the paper's full predictive
  baseline algorithm and makes no zero-variance claim.

Full-game exploitability and calibrated served EV confidence must still be
measured on any accepted combined candidate. Stable policies alone are not an
equilibrium proof; low sampling SE alone is not low neural prediction error.

## Completed verification and next bounded decision

- 66 focused Python/native cases pass, including full-width replay, strict source
  identity, CPU scoring, sampling contracts and chunk parity. Full-FP32 resume
  was also run explicitly: uninterrupted/resumed parameters are bit-identical,
  and changing math mode refuses resume without changing the run state. This is
  not a claim that full Python unittest discovery is clean.
- Rust release tests: 468 library + 9 binary + 15 CLI tests pass; 60
  resource/fixture-gated cases remain ignored in the normal suite. The explicit
  real-weight Rust inference seam was run for all four frozen v4/v5 controls and
  both full-FP32 v5 probes.
- `npm test`: 287 pass, 20 optional cases skipped. All four actual native serving
  integration cases then pass separately; `npm run build` and active-artifact
  verification pass. No UI, browser worker, WASM, active manifest or 20bb weights
  were changed, so no browser-facing redesign is claimed.

The bounded diagnostic sequence is complete; its 100bb/long-run/activation
preconditions did not pass. Keep the CPU-prior repair and optional explicit math
mode, not the rejected policy weights. Do not merge histories again to make a
stability number look better. Do not multiply fitting steps on the tiny noisy
bootstrapped replay: the held-out regression persisted even with full FP32.

The next useful policy experiment is more independent traversal data per update
at a fixed fitting budget, with CPU priors and native-aligned evaluation. A
warm-start fitting/gradient ablation can distinguish replay sparsity and
cross-street interference; the current cold-fit probe does **not** isolate those
causes or reproduce the original Adam state. For EVs, the remaining open-street
posterior/runout variance needs a stronger policy-consistent control variate or
conditional integration; another fit to these imprecise labels is not a repair.
No paid compute is required for those next diagnostics, and no claim is made
that additional compute alone will pass the release gates.
