# 20bb v40/v41 board-relative value features

Status: substantial measured improvement; absolute value gate still failed and
no active manifest was modified.

These paired pilots retained v39's immutable pot-stratified split and isolated
two representation defects in the range-conditioned turn value network.

## v40 board-relative beliefs

The v1 context represented each range mainly as 169 preflop hand classes.
That loses strategically relevant information about how a range interacts
with the public board. Feature schema v2 adds suit-invariant, board-relative
summaries for both ranges:

- current made-hand category mass;
- strength-percentile histograms;
- expected river-category and improvement mass; and
- query-specific opponent distributions after exact private-card removal.

The query-specific features use only public beliefs and the queried player's
own private hand. They do not observe the opponent's cards. A global suit
permutation produces exactly permuted query rows and identical context rows.

| Metric | v39 wide v1 | v40 wide v2 | Change |
| --- | ---: | ---: | ---: |
| Paired overall RMSE | 0.466657bb | 0.414046bb | 11.27% better |
| Small-pot RMSE | 0.306402bb | 0.267501bb | 12.70% better |
| Medium-pot RMSE | 0.532437bb | 0.429644bb | 19.31% better |
| Large-pot RMSE | 1.003605bb | 0.924068bb | 7.93% better |

Cross-seed prediction correlation was `0.997864`. The seed-12601 and
seed-12602 weights have SHA-256 values
`9509146253ce4258439eef88ccfe9399517b15529a3499a9a7cb6cc1224c871d`
and `7bd3d90f82e53b2273720e208a875d534eea9d02464568b7fe8e810c0d8247b3`.

## v41 exact turn-runout baseline

The old residual baseline compared current six-card turn strength. It did not
enumerate the river, despite being described as showdown equity. Schema v3
replaces that scalar with exact, range-conditioned turn all-in equity:

- all 44 river cards compatible with each pair of private hands are included;
- both queried-card and river-card removal are exact;
- ties contribute one half; and
- the result remains invariant under a global suit permutation.

On untouched near-all-in state 130, this correction alone reduced the raw
baseline RMSE from `2.354538bb` to `0.806714bb`. Across the combined corpus the
raw baseline RMSE fell from `1.390207bb` to `1.050360bb`.

| Metric | v40 wide v2 | v41 wide v3 | Change |
| --- | ---: | ---: | ---: |
| Paired overall RMSE | 0.414046bb | 0.351024bb | 15.22% better |
| Small-pot RMSE | 0.267501bb | 0.224465bb | 16.09% better |
| Medium-pot RMSE | 0.429644bb | 0.342862bb | 20.20% better |
| Large-pot RMSE | 0.924068bb | 0.799672bb | 13.46% better |
| Reach weight within 0.25bb | 76.1% | 80.6% | +4.6 points |

Cross-seed prediction correlation improved to `0.998860`. The v41 weights
have SHA-256 values
`2449ddce3f51371ada9eb7c4c03911dfcf15771604f1a76f604be9ed946c1e3b`
and `89587496ca4cd8706982fe74b7a078abc8630dbfb90e8ba41097c19b5552ffb8`.

Exact Python preprocessing was parallelized deterministically across local
processes. This changes feature-construction throughput only; state order,
features, targets, and model seeds remain pinned. Serial/parallel equality is
covered by a regression test.

Rust serving now accepts feature schemas v2 and v3. It computes v3 runout
equity from a blocker-exact, board-keyed `u8` equity-unit matrix, so repeated
range queries reuse the expensive public-card calculation without storing a
full floating-point matrix. Three matched authentic states, including two
large-pot states, had maximum Python/Rust prediction error
`0.00000449328bb`, below the `0.0001bb` parity gate. A cold one-state CLI
prediction took 0.12 seconds; resolver calls reuse the cached matrix.

## Decision

Both pilots are rejected by the `0.25bb` paired authentic holdout ceiling.
v41 makes small pots pass and leaves medium pots close, but large pots remain
the dominant blocker. The next experiment adds an independent 64-state
large-pot off-policy training corpus while preserving the v39 validation set.
No resolver audit or model activation is permitted before the upstream gate
passes for both seeds.

All 100 Python tests, 72 Rust release tests, and three Rust CLI tests passed.
