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
Rust, the binary exporter, and TypeScript agree on 820 state / 829 action inputs.
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

Results and final verification are appended after the experiments finish.
