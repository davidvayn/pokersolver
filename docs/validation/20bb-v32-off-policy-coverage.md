# 20bb v32 off-policy coverage pilot

## Decision

**Rejected for activation.** The v32 research sequence improves the frozen
32-state turn-value holdout enough to unlock a matched resolver audit, but it
does not provide the independent full-game exploitability upper bound required
for publication. The active manifest and served model remain unchanged.

Large target, weight, parity, and resolver artifacts remain in ignored
`preflop-solver/neural/runs/v32-coverage/` paths. The checked-in candidate
summary records the selected-weight hash, measured gates, and fail-closed
decision.

## Research choice

[DeepStack](https://arxiv.org/abs/1701.01724) established the use of learned
counterfactual values at depth-limited leaves. The
[Supremus](https://arxiv.org/abs/2007.10442) implementation study reports
bottom-up river-to-turn-to-flop training, Huber loss, and values expressed as a
fraction of the pot. Those results motivated retaining v31's pot-normalized
Huber target and testing a larger value network as a controlled alternative.

[ReBeL](https://arxiv.org/abs/2007.13544) is more directly relevant to the
remaining coverage problem: its value network is trained on public belief
states reached by self-play search, with random-action exploration to improve
off-policy coverage. Its reported failure with arbitrary random beliefs is why
this pilot does not synthesize unconstrained ranges. It samples one exploring
player per trajectory, but recomputes both players' exact public beliefs from
the frozen policy's original action likelihoods. Lines with zero exact frozen
reach are skipped.

This is a value-model pilot, not an independently solved subgame. That
distinction matters because
[safe nested solving](https://papers.nips.cc/paper_files/paper/2017/hash/7fe1f8abaad094e0b5cb1b01d712f708-Abstract.html)
shows why subgame results cannot generally be treated as independent of the
trunk strategy. Downstream resolver improvement is therefore required here,
and even that remains separate from the full-game release certificate.

## Implemented experiment

The Rust target generator now supports `--exploration` and
`--minimum-pot-bb`. With epsilon `0.2`, the acting explorer samples from
`0.8 * frozen policy + 0.2 * uniform legal actions`; the explorer alternates
between players across accepted states. Exact card removal and exact
per-player reach-factor beliefs still use the frozen policy, not the behavior
mixture. Target metadata records the explorer, epsilon, and public action line.

The 12-state supplement targets pots of at least 8bb and contains six states
for each explorer. Its standalone dataset is deliberately rejected because it
is smaller than the 64-state release-corpus minimum. When combined with the
already accepted 128-state primary corpus, all 140 targets are independently
revalidated and all turn boards must remain at least 95% distinct. Supplemental
indices are appended to training only; the original tuning and 32-state
holdout are immutable.

The supplement measured:

| Label diagnostic | Result | Gate |
| --- | ---: | ---: |
| Maximum river abstract exploitability | 0.018532bb/hand | at most 0.05 |
| Maximum belief-replicate total variation | 0.099970 | at most 0.15 |
| Minimum effective sample size | 1093.14 of 4096 | at least 409.6 |
| Maximum absolute zero-sum residual | 2.18e-8bb | at most 1e-7 |
| Distinct boards after combination | 140 of 140 | at least 95% |

`--split-seed` now pins train/tuning/holdout membership independently from
network seeds. `--supplemental-dataset` accepts validated training-only
supplements, and `--variant-set range-only` avoids rerunning the already
established range-input ablation during architecture research.

## Capacity pilot

The larger control uses 128/128/64 GELU context and query towers with a
128/64 GELU head. Both Python and Rust support the exported dense tower and
fast GELU activation. Runtime parity passed with maximum error
`1.01744e-5bb`, but paired mean holdout RMSE was `0.557526bb`, 3.20% worse
than v31 compact's `0.540214bb`. Cross-seed correlation was `0.998526`.
Capacity alone was rejected, and the compact architecture was retained.

## Paired compact results

Seeds 11401 and 11402 trained on the combined 140-state corpus. Seed 11401
was selected by tuning RMSE only (`0.332686bb`); the holdout did not influence
selection.

| Measurement | v31 compact | v32 compact | Change | Gate |
| --- | ---: | ---: | ---: | --- |
| Paired mean holdout RMSE | 0.540214bb | 0.486965bb | 9.86% better | Pass: at least 5% |
| Small-pot RMSE | 0.215392bb | 0.224880bb | 4.40% worse | Pass: no more than 5% worse |
| Medium-pot RMSE | 0.803120bb | 0.730631bb | 9.03% better | Pass: at least 5% |
| Large-pot RMSE | 1.573794bb | 1.351428bb | 14.13% better | Pass: at least 5% |

Cross-seed prediction correlation is `0.997434`. Python and Rust inference on
all 32 untouched states has maximum absolute error `6.90422e-6bb`, passing the
`0.0001bb` parity gate. Absolute RMSE still exceeds the historical `0.25bb`
release-oriented value target; these measurements only authorize the matched
resolver experiment.

## Matched resolver evaluation

The audit uses the identical v31 configuration: a 4bb flop pot, actor 1,
100 iterations, averaging after iteration 10, no abstract all-in branch, and
the predeclared dry-low, dry-high, and monotone boards. Results are recorded in
the candidate summary.

| Board texture | v31 | v32 | Result |
| --- | ---: | ---: | --- |
| Dry low, 2c 7d Th | 0.540310bb/hand | 0.724564bb/hand | v32 34.10% worse |
| Dry high, As Kd 7c | 0.900115bb/hand | 0.787130bb/hand | v32 12.55% better |
| Monotone, 9h Th Jh | 0.694163bb/hand | 0.703217bb/hand | v32 1.30% worse |
| Mean | 0.711529bb/hand | 0.738304bb/hand | v32 3.76% worse |

The matched resolver-improvement gate therefore fails. Regardless of the local
result, the solver remains an approximate depth-limited diagnostic and not a
full-game exploitability bound.

The selected-seed texture audit helps localize the mismatch. The v32 holdout
RMSE is still `0.548546bb` on disconnected turns, `0.570945bb` on paired
turns, and `0.579899bb` on rainbow turns. These are better than the matched
v31 selected-seed values, yet the resolver regresses. Ordinary held-out turn
states therefore do not weight errors like the resolver's own counterfactual
leaf distribution.

## Reproduction

From `preflop-solver`:

```sh
target/release/preflop-solver turn-pbs-self-play-targets \
  --networks neural/runs/20bb-v26-routed-seed5101.json \
  --states 12 --range-particles 4096 --belief-replicates 2 \
  --river-iterations 200 --river-averaging-delay 20 \
  --exploration 0.2 --minimum-pot-bb 8 --seed 11301 --threads 10 \
  --checkpoint-dir neural/runs/v32-coverage/off-policy-checkpoints \
  --output neural/runs/v32-coverage/off-policy-turn-targets-12-seed11301.json

.venv-neural/bin/python neural/train_public_value_network.py \
  --dataset neural/runs/v30-public-belief/turn-targets-self-play-exact-128-seed10701.json \
  --supplemental-dataset neural/runs/v32-coverage/off-policy-turn-targets-12-seed11301.json \
  --output-dir neural/runs/v32-coverage/off-policy-compact-140-paired \
  --architecture compact --value-normalization pot --variant-set range-only \
  --steps 3000 --batch-size 6 --seeds 11401,11402 \
  --split-seed 10901 --holdout-start-index 64
```

The fail-closed composition is implemented in
`neural/validate_v32_coverage.py`. Prediction improvements merely make the
resolver audit eligible. It never permits activation without the independent
one-sided 99% full-game exploitability upper bound.

## Next boundary

The experiment isolates two facts: targeted belief-state coverage helps
ordinary held-out labels more than a larger network, but ordinary label RMSE
is not yet a reliable proxy for resolver quality. The next experiment should
sample and solve the resolver's own counterfactual turn-leaf beliefs, weight
validation by resolver leaf reach, and preserve a second authentic-state
holdout to detect overfitting. Generic corpus scaling and another capacity
increase are not justified by these results.
