# 20bb v39 pot-stratified value split

Status: directional improvement only; rejected for activation.

The three-way value-model split now stratifies the authentic primary corpus by
pot band. It remains deterministic, keeps the 64-state validation set entirely
inside newly generated indices 128--255, and keeps every supplemental corpus
training-only. For the 256-state corpus the split is:

| Pot band | Train | Tuning | Validation |
| --- | ---: | ---: | ---: |
| Small | 133 | 21 | 49 |
| Medium | 20 | 3 | 8 |
| Large | 13 | 2 | 7 |

The report records `potStratifiedSplit` and these counts, preventing a future
run from silently selecting a checkpoint without representation from every
available pot band. A regression test pins deterministic membership,
disjointness, the new-half holdout boundary, and all-band representation.

## Paired wide-network result

Seeds 12401 and 12402 used the accepted 256-state authentic corpus, both
validated training supplements, a 50% primary replay floor, the wide ReLU
architecture, pot normalization, batch size 6, and at most 3,000 steps.

| Metric | Seed 12401 | Seed 12402 | Paired mean |
| --- | ---: | ---: | ---: |
| Overall RMSE | 0.458805bb | 0.474510bb | 0.466657bb |
| Small-pot RMSE | 0.285417bb | 0.327387bb | 0.306402bb |
| Medium-pot RMSE | 0.541187bb | 0.523688bb | 0.532437bb |
| Large-pot RMSE | 1.009764bb | 0.997446bb | 1.003605bb |
| Best tuning RMSE | 0.422979bb | 0.434706bb | 0.428842bb |

Cross-seed prediction correlation was `0.997619`. The weights SHA-256 values
were `9c6384ef05fe118211ef1834419c4007c3fc752627c899ac7afd11876c56a304`
and `606d852b5280cb1560f090fa4539c15c586b1b73bf8729ce0466ed2ba295383f`.

On the corrected holdout, the untouched v36 pair measured `0.510131bb` mean
RMSE. v39 improves that valid baseline by 8.52%, and lowers paired large-pot
RMSE from `1.215921bb` to `1.003605bb`. It nevertheless exceeds the absolute
`0.25bb` ceiling, both seeds exceed that ceiling, and large-pot error is still
roughly four times the target. No resolver audit or active-manifest change is
permitted.

The next controlled pilots must retain this split while testing richer
board-relative range features and additional independent large-pot targets.
Capacity alone is not a release path.

## Verification

All 98 Python tests passed. The active model remains unchanged.
