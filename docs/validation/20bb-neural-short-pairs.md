# 20bb neural short-pair validation

Status: experimental, rejected, and not activated.

## Method

All comparisons below use two independent training seeds. The original reports
use 1,000 deterministic evaluation traversals per corpus (`seed=9001`); the
street-routed reports use the validator's pinned `0x8A5CD789` seed. The final
candidate was also checked with 2,000 traversals at the previously unseen
`314159265` seed. The evaluator samples
pure exact-card trajectories, samples both players once at every decision,
preserves repeated visits, and gives equal total mass to the two frozen-policy
corpora. The forced-deviation corpus samples both players uniformly. This
replaced an earlier external-sampling-tree calculation that double-counted
sampled opponent reach and deduplicated legitimate repeat visits; those earlier
reports are not release evidence.

The release stability gates are reach-distribution action-frequency MAE at most
5%, exact primary-action agreement at least 85%, and maximum aggregate action
delta at most 3%. Cross-seed stability is not equilibrium proof. Exploitability
and action-EV standard-error gates remain unavailable and therefore fail
closed.

Short-pair continuation also reports a deliberately weaker research-only bar:
MAE at most 6%, primary-action agreement at least 80%, and maximum aggregate
delta at most 4%, with valid probabilities and at least 99.99% lookup coverage.
This can justify another bounded experiment but can never activate a model or
replace any release gate.

## Corrected results

| Pair/checkpoint | Traversals per seed | MAE | Primary agreement | Max aggregate delta | Forced MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v5 instantaneous-regret baseline, round 50 | 20,000 | 12.65% | 60.62% | 3.75% | 12.82% |
| v6 Deep DCFR+, round 10 | 4,000 | 10.54% | 60.02% | 5.68% | 11.04% |
| v6 Deep DCFR+, round 25 | 10,000 | 8.23% | 70.34% | 3.51% | 8.65% |
| v6 Deep DCFR+, round 50 | 20,000 | 7.74% | 74.43% | 2.92% | 7.95% |
| v6 Deep DCFR+, round 80 | 32,000 | 7.28% | 75.60% | 1.92% | 7.68% |
| v6, 400 optimizer steps, round 10 | 4,000 | 12.50% | 47.65% | 1.75% | 12.78% |
| v6, 800 traversals/round, round 25 | 20,000 | 8.61% | 72.19% | 8.05% | 8.94% |
| v7 texture, no baseline, round 10 | 4,000 | 9.68% | 56.64% | 3.85% | 10.40% |
| v7 texture, no baseline, round 25 | 10,000 | 8.40% | 62.97% | 2.45% | 8.83% |
| v7 texture + raw-bb baseline, round 10 | 4,000 | 10.52% | 60.51% | 2.51% | 11.14% |
| v7 texture + raw-bb baseline, round 25 | 10,000 | 8.44% | 62.48% | 1.51% | 9.04% |
| v8 normalized baseline, round 10 | 4,000 | 9.98% | 55.55% | 5.48% | 10.49% |
| v8 normalized baseline, round 25 | 10,000 | 8.02% | 66.73% | 3.31% | 8.72% |
| v9 zero-init texture + normalized baseline, round 10 | 4,000 | 9.42% | 59.70% | 1.15% | 9.94% |
| v9 zero-init texture + normalized baseline, round 25 | 10,000 | 7.96% | 69.70% | 4.14% | 8.25% |
| v9 zero-init texture + normalized baseline, round 40 | 16,000 | 7.31% | 74.11% | 3.77% | 7.85% |
| v9 zero-init texture + normalized baseline, round 50 | 20,000 | 6.97% | 75.05% | 1.52% | 7.29% |
| v11 importance-corrected street replay, round 10 | 4,000 | 9.86% | 57.94% | 3.99% | 10.56% |
| v11 importance-corrected street replay, round 25 | 10,000 | 8.20% | 67.46% | 2.76% | 8.77% |
| v12 four-sample value targets, round 10 | 4,000 | 9.48% | 54.93% | 2.49% | 10.56% |
| v12 four-sample value targets, round 25 | 10,000 | 7.73% | 68.88% | 1.06% | 8.18% |
| v13 zero control-variate baseline, round 10 | 4,000 | 9.63% | 59.48% | 1.23% | 10.29% |
| v13 zero control-variate baseline, round 25 | 10,000 | 8.57% | 63.21% | 3.34% | 9.00% |
| v14 512/256 hidden layers, round 10 | 4,000 | 8.93% | 63.45% | 2.49% | 9.67% |
| v14 512/256 hidden layers, round 25 | 10,000 | 8.50% | 66.14% | 4.65% | 9.46% |
| v15 512/256, learning rate 0.0003, round 10 | 4,000 | 10.08% | 60.15% | 2.19% | 10.46% |
| v16 512/256, linear decay, round 25 | 10,000 | 7.85% | 64.63% | 3.92% | 8.62% |
| v9 r50 preflop + v16 r25 postflop | 20,000 / 10,000 | 6.86% | 76.29% | 1.45% | 7.18% |
| v9 r50 preflop + v16 r50 postflop | 20,000 each | 6.39% | 78.45% | 1.02% | 7.00% |
| v17 r60 preflop + v16 r50 postflop | 24,000 / 20,000 | 6.29% | 78.76% | 0.42% | 6.93% |
| same frozen hybrid, independent 2,000-traversal corpus | 24,000 / 20,000 | 6.31% | 78.09% | 0.64% | 6.99% |
| v17 r70 preflop + v16 r50 postflop | 28,000 / 20,000 | 6.31% | 77.82% | 0.80% | 6.95% |

Deep DCFR+ substantially improves stability over v5, and the standard v6
schedule is better than either increasing optimizer steps or doubling samples
per outer round at a fixed traversal budget. However, progress from round 50 to
80 is too small to project a pass from more identical iterations. The round-80
street breakdown remains weakest after preflop: preflop MAE is 5.95%, while
flop, turn, and river are 9.66%, 10.09%, and 9.08% respectively.

## Poker-aware feature and variance-reduction pair

The next pilots implemented a shared 716-value v4 state schema in Rust, Python,
and TypeScript. The 64 new suit-invariant values describe made hands, draws,
rank and suit multiplicity, straight windows, and board wetness while retaining
the exact-card representation. A learned scalar action-value scorer supplies
the unbiased action-dependent control variate
`sum(policy * baseline) + sampled_value - sampled_baseline` at sampled opponent
nodes.

The v7 ablation shows that the new features alone and the raw-big-blind baseline
were effectively tied at round 25; the baseline did not provide the expected
gain. Inspection found that the value head was fitting raw values as large as
20bb with a unit-scale Huber loss while the advantage heads were normalized.
v8 trained scalar values in effective-stack fractions and converted them back
to big blinds only at the traversal and artifact boundaries. That improved the
round-25 pair, but randomly initialized new feature columns still changed the
starting function and hurt early agreement. v9 zero-initializes exactly those
new first-layer columns. Its round-10 result is the strongest early pair and
its round-40 MAE is better than the old v6 round-50 result at 20% fewer
traversals, while agreement is nearly equal. At round 50 it beats the old v6
round-50 checkpoint on all three authentic-trajectory stability metrics. It
also beats the old v6 round-80 checkpoint on MAE and aggregate delta with
37.5% fewer traversals, although primary agreement remains 0.55 percentage
points lower.

This is evidence of a more sample-efficient training path, not a release pass.
Postflop MAE remains materially higher than preflop, overall MAE is above the
5% release target, primary agreement is 75.05% rather than 85%, and the
value-head held-out RMSE remains noisy around 9–10bb. The exploitability and
action-EV uncertainty gates are still unimplemented and therefore fail closed.

## Street-stratified replay pair

The first binary pilot proposed 50% preflop and 50% postflop replay. Live
reservoir telemetry showed that external-sampling training was already about
65% postflop, so this would have reduced rather than increased postflop
exposure. That pilot was stopped cleanly at round 9 and was not used as
validation evidence.

The corrected v11 pair measured the v9 replay reservoirs at approximately
26%/28%/25%/20% preflop/flop/turn/river, with larger player/round swings in the
current advantage reservoirs. It proposed 25% from each street in every
minibatch and multiplied each sample by its empirical street probability over
its realized proposal probability. Deterministic resume produced byte-identical
artifacts, and unit tests recover the original empirical objective exactly.

At round 25, v11 improved maximum aggregate action delta from 4.14% to 2.76%
and made small flop/turn gains, but overall MAE regressed from 7.96% to 8.20%,
primary agreement from 69.70% to 67.46%, and forced-deviation MAE from 8.25% to
8.77%. River primary agreement also fell sharply. Reject this proposal as the
default and do not extend the pair. The mechanism remains an explicit research
flag; authentic replay remains the trainer default.

## Value-target, control-variate, and capacity pairs

The v12 pair averaged four independent external samples into each traverser
action-value target. It reused the primary traversal value and derived the
three extra random streams from the canonical state, action, deal, iteration,
and sample index. A deterministic unit test confirms that increasing the value
sample count changes only `action_values_bb` under fixed networks; it cannot
advance the primary traversal RNG or directly change regret and average-policy
records. Deterministic stop/resume also produced byte-identical artifacts.

At round 25, v12 narrowly improved the v9 MAE from 7.96% to 7.73% and aggregate
delta from 4.14% to 1.06%, but agreement fell from 69.70% to 68.88%. Its
forced-deviation MAE improved only from 8.25% to 8.18%. The fourfold traversal
cost did not produce a consistent paired gain and the pair missed the relaxed
research bar, so one-sample value targets remain the default.

The v13 pair set the learned action-dependent control-variate scale to zero
while continuing to train the value head for calibration. It regressed every
round-25 local-stability measure relative to v9: MAE reached 8.57%, agreement
63.21%, and forced-deviation MAE 9.00%. This supports retaining the 0.5 baseline
scale, although it does not show that the value estimator is already well
calibrated.

The v14 pair doubled hidden-layer widths from 256/128 to 512/256. At round 10 it
was the first new pilot to improve all three local metrics over v9: MAE 8.93%,
agreement 63.45%, and forced-deviation MAE 9.67%. The gain did not persist.
Round-25 MAE regressed to 8.50%, agreement to 66.14%, forced-deviation MAE to
9.46%, and the 4.65% aggregate delta missed even the relaxed pilot bar. Larger
artifacts also grew from about 2.65MB to 6.08MB. Do not retain the wider model
at the current learning-rate schedule.

The v15 follow-up lowered the wider model's constant AdamW learning rate from
0.001 to 0.0003. Its predeclared round-10 early-stop comparison was worse than
both v9 and v14: 10.08% MAE, 60.15% agreement, and 10.46% forced-deviation MAE.
Both processes accepted `SIGINT`, finished and checkpointed their in-flight
rounds, and stopped at rounds 13 and 12 without truncation. Do not extend this
pair. A future optimization-schedule experiment would need warmup or decay
rather than simply lowering the constant rate.

At this stage, retain v9 as the leading short-run configuration, but do not promote it or
extrapolate a production pass from cross-seed stability. None of v11 through
v15 clears the research-only continuation bar, much less the release gates.
The next paired pilot should test optimization stability rather than another
value-target or replay change and compare against the exact v9 checkpoints
before spending 8–12 hours on independent seeds.

## Linear-decay and street-routed pairs

The v16 pair retained the 512/256 model's 0.001 learning rate through round 10
and decayed it linearly to 0.0003 at round 25. Its round-10 parameters were
bit-identical to v14, and deterministic interrupted/resumed smoke runs produced
byte-identical artifacts. Decay improved v14's round-25 MAE, aggregate delta,
and forced-line MAE, but primary agreement remained weak. Street breakdowns
showed the wider network was substantially more stable postflop while the
256/128 v9 network was more stable preflop.

The evaluator therefore gained an optional exact street route: the primary
network is used preflop and a separately validated network is used on flop,
turn, and river. Rust generates each composite policy's authentic reachable
trajectories with the routed networks; Python uses the same route for frequency
comparison. This is not a union-support approximation. v9 round 50 plus v16
round 50 improved all headline stability measures over either early component,
reaching 6.39% MAE, 78.45% agreement, and a 1.02-point maximum aggregate delta.

A constant-rate v9 continuation increased agreement to 79.42% at round 70 but
worsened MAE to 6.78% and forced-line MAE to 7.50%. v17 reproduced every one of
v9's 660,868 round-50 artifact parameters exactly, then decayed the narrow
network from 0.001 at round 50 toward 0.0003 at round 70. Round 60 was the best
checkpoint: the routed hybrid reached 6.29% MAE, 78.76% agreement, a 0.42-point
aggregate delta, and 6.93% forced-line MAE. The independent 2,000-traversal
corpus confirmed 6.31%, 78.09%, 0.64 points, and 6.99%, respectively. Further
decay to round 70 reduced agreement, so round 60 is frozen as the leading
research checkpoint.

This hybrid still misses even the relaxed 6%/80% continuation bar and all
production equilibrium/EV gates remain fail-closed. The browser artifact and
practice runtime now implement the frozen schema-2 street route. Actual seed
4501 and 4502 composites exported twice to byte-identical 8,720,729-byte
artifacts. The browser selects the narrow round-60 component preflop and the
wide round-50 component postflop for baseline policy, exploit response, and
action value together.

Fresh v14 paired smoke runs also completed with measured per-action standard
errors. A two-traversal evaluation reached six decisions with valid probability
sums and complete lookup coverage. Only 16.7% of those tiny-sample decisions
had every action at or below 0.02bb, demonstrating that the new gate fails
closed rather than accepting the former fixed uncertainty target. This smoke
corpus is pipeline evidence only, not a quality estimate.

The real pair still needs a predeclared rollout-count pilot, a genuine
full-game exploitability upper-bound method, and an end-to-end browser
acceptance pass before launch. Sparse advantage snapshots preserve an
SD-CFR-style teacher comparison at artifact rounds so the final average-policy
network can be challenged without rerunning training. Do not activate this
candidate.

Every browser artifact in this table remains `training_not_activated`.
`data/practice/full-hand-manifests.json` remains empty.
