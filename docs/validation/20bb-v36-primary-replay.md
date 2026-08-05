# 20bb v36 authentic-primary replay pilot

Status: rejected; v31 remains the frozen research baseline and no active
manifest was modified.

This pilot tested whether the v33 regression came from supplemental
resolver/off-policy rows displacing authentic public-belief examples in tiny
training batches. It also corrected a validation hole: a downstream winner
may no longer be selected when the second independent seed disagrees.

## Training correction

The value trainer now stratifies each mini-batch by pot band while reserving a
declared minimum fraction for the primary authentic corpus. Supplemental rows
may fill the remaining slots but cannot crowd the primary corpus out. The
sampling split is deterministic from the training seed and safely falls back
when a band is absent from one source.

The v36 pair used:

- the accepted 128-state authentic v30 corpus, SHA-256
  `d4c0b6a6072cb88494e18ec2c46311beaf6cb202baf5cf322669b20573dbc366`;
- the 12-state v32 off-policy training supplement, SHA-256
  `efa5fd4318110386f788135a473307390e85ff4986cbb2f74bab48701bd783e0`;
- the 18-state v33 resolver-leaf training supplement, SHA-256
  `bb984d1928502d7ab67f7e9fcba07170ea1807cad3381b0fdf45a41ec3abf3e2`;
- a 50% authentic-primary batch floor, compact range model, pot
  normalization, batch size 6, 3,000 maximum steps, split seed 10901, and
  model seeds 11901 and 11902.

The combined training dataset SHA-256 was
`d0835f589f8939fbaa4cb132cc9c451c2922466b1346536a0808963058dd4228`.
All supplemental rows remained training-only. The frozen source policy
SHA-256 was
`c78397af5900b3409d3dfc911fce56075cb54ce860c38fc2a1459fe5d56df948`.

## Authentic holdout

The 32 matched authentic states were untouched and identical to the v31
evaluation split.

| Metric | v31 paired mean | v36 paired mean | Relative change |
| --- | ---: | ---: | ---: |
| Overall RMSE | 0.540214bb | 0.483085bb | 10.58% better |
| Small-pot RMSE | 0.215392bb | 0.236018bb | 9.58% worse |
| Medium-pot RMSE | 0.803120bb | 0.678820bb | 15.48% better |
| Large-pot RMSE | 1.573794bb | 1.371459bb | 12.86% better |

Cross-seed prediction correlation was `0.996664`. Seed 11901 trained through
step 3,000 with best tuning RMSE `0.320770bb`; seed 11902 selected step 1,500
with tuning RMSE `0.322359bb` and early-stopped at step 2,000. Rust/Python
parity on the exact holdout had maximum absolute error
`0.00000629948bb`, below the `0.0001bb` gate.

The directional improvement is real, but `0.483085bb` fails the newly
explicit absolute authentic RMSE ceiling of `0.25bb`. The selected seed-11901
weights had SHA-256
`de3a53682738afb8ecd027d8b2da09a0bd4cc4b48be1721faba88231c2ed7b20`.

## Untouched resolver-leaf evaluation

The nine-state resolver-leaf corpus remained disjoint from all training
components. Its SHA-256 was
`3f4382d823b60ecaa55112a2477cf9fd5e2ebf5cf8738c8104bba5279eba3869`.

| Model | Reach-weighted RMSE | Relative to v31 | Absolute player bias |
| --- | ---: | ---: | ---: |
| v31 seed 10902 | 1.070261bb | baseline | 0.171208bb |
| v36 seed 11901 | 0.883031bb | 17.49% better | 0.238383bb |
| v36 seed 11902 | 0.919554bb | 14.08% better | 0.259895bb |

The explicit per-player signed metric is important. Zero-sum projection makes
an aggregate signed error appear almost exactly zero even when one player's
values are systematically low and the other's are systematically high. v36
improves RMSE while worsening this resolver-reach-weighted bias, another reason
not to promote it from a single aggregate score.

## Exact-all-in matched resolver audit

All candidate and baseline resolves used the same three predeclared boards,
uniform public flop ranges, a 4bb pot, actor 1, the exact all-in branch, and
the same iteration count within each comparison.

At 10 iterations both candidate seeds looked directionally positive. The more
converged 30-iteration comparison reversed that conclusion for seed 11902:

| Model | Mean exploitability | Boards improved | Relative to v31 |
| --- | ---: | ---: | ---: |
| v31 seed 10902 | 0.458647bb/hand | baseline | baseline |
| v36 seed 11901 | 0.429858bb/hand | 2 of 3 | 6.28% better |
| v36 seed 11902 | 0.478041bb/hand | 1 of 3 | 4.23% worse |

This is not sufficient cross-seed agreement. The validator now requires every
candidate seed to have a non-negative matched mean improvement and improve at
least two of three boards, in addition to choosing a winner with at least 2%
mean improvement. The new gate prevents selection from cherry-picking seed
11901.

## Decision and next experiment

The strict v36 composition fails:

- absolute authentic holdout RMSE;
- matched resolver cross-seed agreement;
- model-selection eligibility; and
- the still-unmeasured independent one-sided 99% full-game exploitability
  upper bound.

The v36 fit is therefore stopped at the paired pilot. A 100-iteration audit
would characterize a candidate that already fails two upstream gates and is
not the highest-value use of compute. The next experiment expands the
immutable, exact range-conditioned authentic corpus from 128 to 256 states,
preserves 64 validation states sampled only from the newly generated half, and
repeats paired training. It may
extend to 512 only if the new holdout and cross-seed resolver results show
meaningful progress toward the absolute gate.

## Verification

- All 97 Python tests passed after the replay, signed-bias, absolute-RMSE, and
  cross-seed gates were added.
- Source validation and Rust/Python parity remained accepted.
- Every compared resolver included exact all-ins and had finite values with
  projected zero-sum residuals below `7e-13bb` before projection.
- Active manifests remain unchanged.
