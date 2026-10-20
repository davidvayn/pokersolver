# Card-memory, fitting-budget, and frozen EV-label repair

## Scope

User-authorized sequence: fix public card-arrival memory, isolate fitting
budget, improve frozen-policy EV labels, run short matched 50bb pilots, and
replicate at 100bb only if the 50bb improvements hold. No changes to active
20bb serving, rake, action grids, release thresholds, or immutable old weights.

## Card-arrival collision and repair

The Home v4 encoder merges these turn histories with the same betting line:
`[34,47,49,22]` and `[22,47,49,34]`. The current board set is identical but
the flops differ, changing the public information on which earlier decisions
and ranges depend. This is a reproducible representation defect, not yet a
measurement of how much playing strength is lost.

The opt-in Home v5 state schema `hu-home-card-arrival-recall-v5` retains all
716 legacy features and adds 52 turn-arrival and 52 river-arrival indicators.
Current-board membership plus these arrivals reconstructs earlier boards;
internal flop order remains irrelevant. No future card is encoded. Python,
Rust, the binary exporter, and TypeScript agree on 820 state / 829 state-action inputs.
Old v4 remains the default and unchanged. A schema/width mismatch fails closed.
New exports use `v14-recall`, never an existing immutable model URL.

For matched initialization, insert 104 zero first-layer columns before action
features, preserving the shared weights and biases. Predictions match within
floating-point tolerance at initialization. Training and sampled continuation
RNG schedules are otherwise unchanged. `--card-arrival-memory` is required by
both trainer and native sampler; a mismatched policy is rejected.

Regression: `test_home_arrival_memory_separates_turn_histories_without_changing_legacy`
was red before implementation (new encoder unavailable), then passed. Native
and browser tests also cover history separation, flop permutations, legacy
compatibility, and no future-card leakage.

## Fixed-target fitting budget (completed)

Readonly replay from the rejected 24-round retained-regret pilot; compare 240
and 960 steps from identical fresh weights, with the trainer's neutral texture
initialization. Optimizer, learning rate, minibatch stream, historical targets,
and network sizes are fixed. Duplicate encoded information sets share a split.
Original replay is opened with `mmap` mode `r`, never modified.

| Seed / player | Stored decisions | Held-out weighted MSE, 240 → 960 | Reduction | Target-mix MAE, 240 → 960 |
| --- | ---: | --- | ---: | --- |
| 51701 / 0 | 20,632 | .09056 → .08195 | 9.51% | 35.62 → 33.31pp |
| 51701 / 1 | 31,269 | .10525 → .06671 | 36.62% | 37.31 → 29.83pp |
| 51702 / 0 | 21,182 | .10011 → .08083 | 19.26% | 35.88 → 32.12pp |
| 51702 / 1 | 26,617 | .09496 → .06885 | 27.50% | 36.20 → 30.66pp |

Underfitting is a contributing issue: more fitting helps every saved network.
But sizeable held-out target disagreement remains. This is not a CFR-loop
experiment, cross-seed stability test, or evidence of reduced exploitability.
Saved replay lacks whole-hand family IDs, so this is an information-set holdout,
not a family-independent release evaluation. It does not justify blindly
restarting a long retained-regret run or comparing its raw MSE with bootstrapped
cumulative-regret MSE.

Output: `/Users/davidvayntrub/pokersolver-training/regret-fit-budget-2026-10-19-v2.json`.
SHA-256: `f08cc5420a0312f7a488866fa8b1adc1e39060dfbf1712dd68995577a6d8c863`.
The initial exploratory report is preserved separately; v2 corrects the metric
to match the grouped, per-decision weighted training objective and initialization.

## Frozen EV labels

`neural-frozen-values --exact-river-values` enumerates every blocker-compatible
opponent combo with its frozen-policy posterior weight and integrates the
complete finite modeled river betting tree. Policies see only their own
information sets. Zero-probability branches contribute zero; no open chance
subtree is replaced by a showdown shortcut. A two-million-node per-decision
guard fails explicitly rather than silently returning an approximate label.

Exact river records declare their estimator, node count, zero Monte Carlo SE,
and **no independent sampled blocks**. Other streets retain the existing
outer-posterior/future-board block estimator and identical sampling streams.
Zero sampling error is conditional on the frozen modeled game/posterior; it is
not zero neural prediction error, full-game exploitability, or a confidence
guarantee for unknown real opponents. Frozen mean fits retain separate empirical
prediction-residual confidence and remain ineligible for serving.

## Predeclared short policy comparison

`run_regret_learning_pilot.py --candidate recall`: two independent seeds,
24 rounds × 128 traversals, 240 steps/head/round, batch 512, 256/128 hidden
layers, 40k replay. Both arms use averaged regret rollouts; only input memory
differs. Fixed equal-policy reach mixtures on two evaluation batches prevent
different visited distributions from masquerading as a stability improvement.
32 independent, street-stratified hand-family roots use 512 hidden blocks;
response actions are selected on one half and evaluated on the other.

Conditional follow-up requires lower frequency MAE and higher primary agreement
on **both** batches, lower local response gain in **both** seeds, and negative
one-sided 95% whole-root bootstrap upper delta. The driver checks source/binary
identities before a 100bb replication. These are pilot screening criteria, not
the release gates or a full-game exploitability certificate.

## Completed estimator checks

The follow-up `--exact-closed-values` also integrates tractable postflop
all-in branches: 44 possible rivers per private pair on the turn and 990
unordered turn/river pairs on the flop. Unordered pairs are valid only because
these branches have no remaining decisions. Opponent actions are weighted by
their exact frozen-policy frequencies and posterior, not a uniform caller
range. Open branches and preflop all-in runouts remain sampled. A separate
two-million-runout guard is per decision, with no partial result on exhaustion.

Mixed decisions explicitly mark each exact versus sampled action. Fully exact
decisions contain no sampled blocks; sampled columns remain bit-identical to
the control. Companions preserve input/network bytes and hashes for consumption
by `frozen_policy_values.py`. Both `frozen_reference` and the chunked helper
accept the opt-in exact flags; chunking preserves the global RNG schedule.

Two unchanged 64-round 50bb policies, 16 street-stratified decisions and 512
hidden blocks each, give:

| Frozen seed | Mean action sampling variance, sampled → exact (bb²) | Reduction | Actions with reference SE ≤.02bb | Decisions with all actions ≤.02bb |
| --- | --- | ---: | --- | --- |
| 51701 | .80962 → .50762 | 37.30% | 5.33% → 37.33% | 0% → 25% |
| 51702 | .80194 → .50636 | 36.86% | 5.33% → 37.33% | 0% → 25% |

Each policy check integrates 4,462,920 closed-branch runouts and four complete
river posterior/action trees (37,620 nodes each). All 24 nondeterministic
sampled-versus-exact action comparisons per seed fell inside their old nominal
95% sampling intervals. This is a differential sanity check on 16 stratified
decisions, **not** a new calibration guarantee or authentic-reach qualification.
All still-sampled action columns are exactly unchanged. Median decision maximum
SE remains .887/.762bb, far above .02bb. There is no claim that a neural mean
head's prediction accuracy improved by 37%; this measures reference variance.
Wall times were host-contended and do not establish a speedup.

Outputs under `pokersolver-training/`:

- `exact-river-labels-2026-10-19-50bb/summary.json`, SHA-256
  `d83b9a532d5bd7cadfb9c9665101f91541b8fab5c96c7586fd9fc6fcaf7d7640`.
- `exact-closed-labels-2026-10-19-50bb/summary.json`, SHA-256
  `3683511957dcc5eea528714d10c8247ba9f2597d4021fba4bd741bfbec061138`.

All eight sampled/exact label sets were verified against the original binary
policies, exact source weights, game abstraction, input hashes, and estimator
contracts. They remain research data, not serving receipts.

## Completed 50bb policy comparison: reject these weights

All four cold runs completed the predeclared 24 × 128 traversal / 240-step
budget. Both legacy control parameter arrays are bit-identical to the prior
day's controls, including the value/response heads. Metadata/version hashes
correctly differ because the sampler identity changed. This confirms that the
new opt-in features did not silently alter the old learner.

| Metric | Batch 1: v4 → v5 recall | Batch 2: v4 → v5 recall | Release requirement |
| --- | --- | --- | --- |
| Frequency MAE | 10.66 → 12.88pp | 10.59 → 12.37pp | ≤5pp |
| Primary agreement | 60.13 → 46.16% | 56.31 → 46.43% | ≥85% |
| Max aggregate action delta | 1.89 → 6.78pp | 1.59 → 7.26pp | ≤3pp |
| Sampled lookup coverage | 100 → 100% | 100 → 100% | ≥99.99% |

The dominant aggregate disagreement is in all-in/fold frequencies; it is not
just argmax sensitivity to near-tied actions. Preflop, which has zero arrival
features, also becomes less stable after training. The experiment demonstrates
that correcting the representation alone does not fix the learning loop at
this budget. Underfitting, increased sample complexity, shared-network training,
and bootstrap propagation remain hypotheses for the quality regression, not
causes isolated by this one comparison.

Information-set-consistent local response gains (bb/**decision**, not bb/hand):

- Seed 51701: 1.51831 → 1.35096 (improves).
- Seed 51702: 1.10612 → 1.34869 (worsens).
- Pooled: 1.31221 → 1.34982; delta +.03761, one-sided 95% whole-root
  bootstrap upper delta +.40341. No credible paired improvement; this is not
  a statistically established regression across training seeds either.

Both ordinary independent release validations also reject the weights. Valid
probability sums, sampled coverage, and seed independence pass. Frequency MAE,
primary agreement, and the recall arm's aggregate action deltas fail. Frozen
served-EV confidence and full-game exploitability remain unmeasured for these
new policies, rather than being certified by the local response test.

**100bb was not started:** the predeclared 50bb improvement condition failed.
The representation correction stays available for controlled future experiments;
these weights are not promoted. This short cold comparison must not be presented
as degradation of the older 64-round candidates or the active 20bb model.

Output: `card-memory-pilot-2026-10-19-50bb/summary.json`, SHA-256
`17ba09381601e5429d69e49783e4ba83de033b31e8115ee91a7b4c5824a86bf2`.
The compact audit beside this document retains complete summary metrics,
provenance, artifact hashes, and rejection decisions without committing weights.

## Verification and bounded next decision

- `npm test`: 287 passed; 20 optional integration cases skipped in the normal
  suite. All four new artifact tests then ran explicitly, covering complete
  passive/raised/all-in hands in both seats for both pairs.
- `npm run build`, production artifact and trace checks passed; the active
  20bb artifact hashes and sole available depth are unchanged.
- Rust release tests: 491 passed across library/binary/CLI, 59 resource-guarded
  library tests ignored. New exact-label and recall tests pass.
- Focused Python/native suites pass, including v5 deterministic checkpoint
  resume, feature hashes, exact estimator contracts, and chunk RNG parity.
  This is not a claim that full Python unittest discovery is clean.
- Nine policy-tool, eleven practice-tool, one native-audit, and all four
  actual resolver integration tests passed.
- Browser smoke checks: completed Practice hand review; Solver preflop-to-flop
  line; visible 2px keyboard focus; 375/768/1024/1440px without horizontal
  overflow; dark/light and reduced-motion layouts inspected. No application
  console errors. The only observed failed resource was the existing favicon
  404. The temporary tab was closed and the original tab/theme restored.
- Implementation checkpoint `04f3d70` was pushed to main and its GitHub CI passed.

No long training run, 100bb replication, paid compute, rake work, activation,
or gate relaxation was performed. The useful next policy test is a bounded,
budget-scaled fit comparison on the new representation, with mean-label noise
kept separate from regret/average-policy fitting. The old retained-replay probe
supports testing more fitting, but does not prove it will repair CFR stability.
Open-street label variance and neural calibration still need independent work
before either 50bb or 100bb can be released.
