# 20bb v28 overnight solve and range-value pilot

Status: the overnight sequence completed. The v28 preflop candidate was rejected
and not activated. The follow-on range-conditioned value model passed its
research-pilot confirmation criteria, but is not release eligible and is not
connected to the web app.

## What ran overnight

The run built four independent continuation corpora against the frozen paired
v26 routed policies. Each corpus contains 26,520 exact deals, ten complete
exact-combo marginal cycles, 49 reachable flop histories, 16 continuations per
leaf, and 1,299,480 leaf values. T1 and T2 were training-only, V was used for
candidate selection, and H was opened only after the preflop candidate was
selected. The cache phase ran from approximately 03:19 to 10:13 PDT; the
resumable solver and gate pipeline completed at 10:48 PDT.

| Corpus | Seed | Deals | Leaf values | Action-value SE <= 0.02bb | Maximum history-mean SE |
| --- | ---: | ---: | ---: | ---: | ---: |
| T1 | 8309 | 26,520 | 1,299,480 | 35.98% | 0.119bb |
| T2 | 8317 | 26,520 | 1,299,480 | 35.78% | 0.119bb |
| T1+T2 | 24,719 | 53,040 | 2,598,960 | 35.88% | 0.084bb |
| V | 8329 | 26,520 | 1,299,480 | 35.85% | 0.119bb |
| H | 8353 | 26,520 | 1,299,480 | 36.05% | 0.119bb |

Every cache is complete, finite, stack-bounded, and atomically written. The
merged T1+T2 maximum information-group mean standard error is still 1.678bb.
This uncertainty is much larger than the release requirement and is central to
the result below.

## Paired DCFR result

Two independent DCFR seeds, 8801 and 8802, were trained on the merged T1+T2
cache. A split-cache 10M diagnostic first confirmed that independent finite
continuation corpora produced substantially different policies. Training both
seeds on the merged corpus improved stability, and extending the selected
configuration from 10M to 100M iterations improved both stability and V
exploitability.

| Candidate | V exploitability | Cross-seed MAE | Primary agreement | Maximum aggregate delta |
| --- | ---: | ---: | ---: | ---: |
| Split T1/T2, 10M | 0.349 / 0.347bb | 10.30% | 69.63% | 0.404% |
| Merged, 10M | 0.333 / 0.330bb | 4.52% | 84.50% | 0.162% |
| Merged, 100M | 0.296 / 0.296bb | 2.34% | 92.62% | 0.210% |

The 100M pair contains 16,900 information sets per seed and reaches training-
corpus exploitability of 0.03473 and 0.03468bb/hand. Its 100% lookup coverage,
2.34 percentage-point action-frequency MAE, 92.62% primary agreement, 93.30%
tie-aware agreement, and 0.21 percentage-point maximum aggregate delta pass
the stability gates.

The selected pair then measured 0.32458 and 0.32774bb/hand on H. These are
information-set-consistent best-response estimates inside sampled preflop games
whose postflop values come from frozen v26 rollouts. They are not full-game
exploitability certificates. Even in that limited game, however, they are far
above the 0.05bb/hand activation threshold.

The planned 300M extension was skipped. The 10M-to-100M V improvement was real,
but the worst V result remained above the predeclared continuation threshold,
and H exposed a sizeable generalization gap. More iterations on the same noisy
finite target corpus were therefore unlikely to close the release gap.

## Fail-closed release decision

| Gate | Result |
| --- | --- |
| Exploitability estimate <= 0.05bb/hand | Failed: 0.325–0.328bb/hand on H |
| One-sided 99% upper bound <= 0.10bb/hand | Failed: evidence unavailable |
| Cross-seed action-frequency MAE <= 5% | Passed: 2.34% |
| Primary-action agreement >= 85% | Passed: 92.62% |
| Maximum aggregate action delta <= 3% | Passed: 0.21% |
| Lookup coverage >= 99.99% | Passed: 100% |
| Probability validity | Passed |
| Action-EV SE <= 0.02bb for >= 95% | Failed: qualifying evidence unavailable; cache diagnostic is about 36% |
| Projected policy storage <= 20GB | Passed: 5,147,292 bytes |

Both compact policies are 2,573,646 bytes. Their SHA-256 hashes are
`c8045556fbd9940aa09ccdd5d45192abab706950b545150662cda0c845b65af8`
and
`a3ef857754392c3da44e4a497ec9644c8863ab55f2cb1d3526b0e1a7d4b3c10b`.
They remain offline research artifacts. No manifest was activated, no fallback
policy was substituted, and no runtime database was added.

## Range-conditioned value-oracle pilot

Because the preflop gates failed, the next stage tested whether a compact value
function could learn the noisy flop continuation surface. The leakage boundary
is explicit:

- Inputs include the acting player's exact hole cards, exact flop, public
  preflop line, pot/stacks, acting player, exact 169-class hand, the production
  64-feature board texture, and hero/opponent 169-class Bayesian ranges.
- Opponent range construction removes the hero cards and flop exactly. The
  public hero range removes the flop. Public actions are replayed through the
  frozen average 100M preflop policy to update both ranges.
- Opponent exact cards, turn, and river are target-only and never enter the
  model. A deterministic leakage test changes only opponent cards and verifies
  that the complete input vector is unchanged.
- Training samples 80% from authentic policy reach and 20% uniformly from
  meaningful flop leaves. Preflop all-in runouts are excluded.
- Every trial has an identical no-range control with the same architecture,
  seeds, optimizer, sampled examples, and targets; only its two range channels
  are zeroed.

Six short paired pilots isolated the main failure modes rather than committing
to another long run prematurely.

| Pilot | Main change | Range grouped RMSE | No-range grouped RMSE | Cross-seed correlation | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | Flat, Huber, raw rollout | 2.727bb | 2.721bb | 0.878 | Reject |
| 2 | Range tower, MSE, bounded | 2.318bb | 2.359bb | 0.803 | Reject: unstable |
| 3 | 8,000 steps | 2.432bb | 2.406bb | 0.689 | Reject: overfit |
| 4 | Exact class + 64 textures | 2.396bb | 2.377bb | 0.808 | Reject |
| 5 | Texture+hand conditional target | 1.793bb | 1.781bb | 0.835 | Reject: sparse buckets |
| 6 | Coarse texture conditional target | 0.803bb | 0.775bb | 0.971 | Freeze for H |

Pilot 1 showed that Huber loss was learning a median-like target under an
extreme, multimodal rollout distribution. MSE corrected the objective but did
not fix instability. Longer training worsened held-out performance. Adding the
exact hand class and board features improved representation, while the first
conditional-mean target remained too sparse: it created 25,633 training
buckets and 18,489 V buckets. Coarsening only the target to public texture
produced 2,012 training buckets and 1,820 V buckets and materially improved
cross-seed consistency.

After pilot 6, a non-inferiority research criterion was declared before using
the value model on H: grouped RMSE no more than 5% worse than the paired
no-range control, grouped RMSE no greater than 1bb, and range-model cross-seed
prediction correlation at least 0.95. The four frozen pilot-6 weight files were
then loaded without retraining or tuning for H evaluation.

| Metric | V | H |
| --- | ---: | ---: |
| Range ensemble RMSE | 2.597bb | 2.504bb |
| Range grouped RMSE | 0.803bb | 0.697bb |
| Range grouped correlation | 0.876 | 0.905 |
| Range cross-seed correlation | 0.971 | 0.970 |
| No-range grouped RMSE | 0.775bb | 0.692bb |
| Range relative to no-range | 3.57% worse | 0.69% worse |

H passes the declared research criterion. It also shows that the explicit range
vectors are effectively redundant here: under one pinned public policy they
are deterministic transformations of the public line and visible blockers, so
the rest of the network can infer nearly the same information. The range model
slightly improves raw H RMSE but slightly worsens grouped H RMSE. This is useful
architecture evidence, not a claim that range conditioning is unnecessary in
a resolver where policies change or counterfactual ranges branch.

The underlying targets remain noisy. Mean reported target standard error is
1.509bb on V and 1.516bb on H; only 0.161% and 0.141% respectively are at or
below 0.02bb. The reported zero noise-adjusted RMSE occurs because the estimated
target variance exceeds prediction MSE; it must not be interpreted as perfect
prediction. More continuation rollouts, variance reduction, or a resolver-
generated value target are required before these weights could grade actions.

## Artifact and serving boundary

The tracked
`preflop-solver/neural/20bb-v28-research-candidate.json` records the compact
policy and value-weight hashes, all gate outcomes, and rejection reasons. The
large caches, tabular policies, reports, and MLX weights remain ignored offline
artifacts under `preflop-solver/neural/runs/`.

This stage does not need a hosted database. A future browser runtime could ship
small immutable neural weights as versioned static assets. A database or shard
store becomes relevant only if a future resolver serves a large tabular policy,
sample corpus, or frequently updated opponent-specific state.

## Reproduction

From `preflop-solver`, the resumable sealed pipeline is:

```bash
.venv-neural/bin/python neural/run_v28_overnight.py
```

The final frozen-weight value evaluation, run from `preflop-solver/neural`, is:

```bash
../.venv-neural/bin/python train_range_value_oracle.py \
  --train-cache-a runs/v28-overnight/cache-t1.json.gz \
  --train-cache-b runs/v28-overnight/cache-t2.json.gz \
  --validation-cache runs/v28-overnight/cache-v.json.gz \
  --holdout-cache runs/v28-overnight/cache-h.json.gz \
  --policy-a runs/v28-overnight/merged-seed8801-r100000000.json \
  --policy-b runs/v28-overnight/merged-seed8802-r100000000.json \
  --output-dir runs/v28-range-oracle-pilot6-h-final \
  --load-weights-dir runs/v28-range-oracle-pilot6-coarse-short \
  --steps 2000 --batch-size 2048 --evaluation-samples 200000 \
  --learning-rate 0.0003 --hidden-sizes 128,64 --seeds 9901,9902 \
  --architecture range_tower --loss mse --bounded-output \
  --weight-decay 0.001 --target-mode texture_group_mean \
  --target-bucketing texture --minimum-range-relative-improvement -0.05 \
  --minimum-cross-seed-prediction-correlation 0.95 \
  --maximum-grouped-rmse-bb 1.0
```

The next useful solver experiment is not a longer fit to the same corpus. It is
a range-conditioned postflop action resolver or counterfactual-value generator
with substantially lower-variance targets, followed by independent full-game
learned-response evaluation and the original fail-closed release gates.
