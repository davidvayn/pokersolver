# 20bb v42/v43 high-pot replay

Status: near-pass rejected; no active manifest was modified.

## Independent high-pot labels

The v42 target job used frozen-v26 one-player epsilon exploration with exact
frozen-policy reach-factor beliefs. It generated 64 independent states at a
minimum 8bb total pot: 35 states in the trainer's large band and 29 in its
medium band. Dataset SHA-256 is
`cd762cbac403fa0325e9ba37fb55d77c340627e055521b0bd5138970a1841de9`.

Every source gate passed:

| Source metric | Result | Gate |
| --- | ---: | ---: |
| Maximum river exploitability | 0.022262bb/hand | at most 0.05 |
| Maximum belief-replicate TV | 0.118400 | at most 0.15 |
| Minimum effective sample size | 949.91 of 4096 | at least 409.6 |
| Belief replicates | 2 | at least 2 |

The corpus was appended to training only. The 64-state authentic validation
set, split seed, source policy, and all prior supplements remained unchanged.

## Paired training controls

At batch size 6 and the default loss, seeds 12901/12902 measured
`0.293238bb` and `0.268178bb` holdout RMSE. Increasing the raw-bb auxiliary
from 0.25 to 1.0 was unstable (`0.253114bb` and `0.334266bb`) and rejected.

Batch size 12 gives every update two authentic-primary and two supplemental
draws in each available pot band. It reduced the paired mean to `0.245799bb`
with cross-seed correlation `0.998926`:

| Metric | Seed 13201 | Seed 13202 |
| --- | ---: | ---: |
| Overall RMSE | 0.254874bb | 0.236723bb |
| Small-pot RMSE | 0.165528bb | 0.146054bb |
| Medium-pot RMSE | 0.322575bb | 0.328326bb |
| Large-pot RMSE | 0.532176bb | 0.489721bb |

The old mean-only composition would accept this pair. The absolute gate now
requires every independent seed to be at or below `0.25bb`; seed 13201 fails.
This closes another best-seed/averaging selection hole. A regression test pins
the all-seed behavior.

The v42 weight SHA-256 values are
`79fbb6fcf29dfdb54130f31100dc344aa72e62f0a826e595bdd14e3e61ddf21b`
and `761015c795db67a540ec9e602520d152b37b8d88165abeeed6c2aea4887a2129`.

## Extended pair

The v43 pair increased the ceiling to 5,000 steps and early-stopping patience
to 20. Both seeds retained batch size 12 and all v42 data.

| Metric | Seed 13301 | Seed 13302 | Paired mean |
| --- | ---: | ---: | ---: |
| Overall RMSE | 0.251217bb | 0.237755bb | 0.244486bb |
| Small-pot RMSE | 0.171180bb | 0.136423bb | 0.153801bb |
| Medium-pot RMSE | 0.301561bb | 0.279712bb | 0.290637bb |
| Large-pot RMSE | 0.517645bb | 0.545094bb | 0.531370bb |

Cross-seed prediction correlation was `0.998793`. Seed 13301 misses the
all-seed ceiling by `0.001217bb`, so the pair remains rejected. Its weight
SHA-256 values are
`179ab0c1123394e6ce129287f58455403f2b5503ef432480b6b7cb7839f1ed13`
and `4fcad2492ee718445800df8124cc4c2dc5cf21a6251a43a3fbb4e3b27e8f5985`.

## Next gate

The result is meaningful but upstream only. A second 64-state supplement is
being generated with a minimum 16bb total pot so every accepted state belongs
to the large band. The authentic holdout remains frozen. Even after both seeds
pass the value ceiling, exact Python/Rust parity and paired exact-all-in
resolver improvement are required before continuation-oracle or activation
work can use the candidate.
