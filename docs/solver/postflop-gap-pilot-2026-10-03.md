# Matched postflop leaf pilot and fixed-belief reference check

Date: October 3, 2026 (PDT). Status: research only; no model promotion.

The September 21 20bb benchmark and both retained value networks remained
frozen. On two selected development boards, each with two independent chance
seeds, the pilot changed only the flop leaf-value source: learned versus native.
Both arms used 32 flop updates, the same seed stream, the same action grid,
native64 played turn/river continuations, all 49 legal turn cards, and the same
frozen-policy response evaluator. Every candidate passed an independent
JavaScript flop accounting/backup audit. The complete pilot manifest SHA-256 is
`b79cce8a8d47a2a99fe94661c2da44c0f98614a26aeef129f94c191c6d6d341a`;
local ignored artifacts are under
`preflop-solver/neural/runs/local-postflop-gap-20261002-matched32-a/`.

| Selected root | Seed | Learned32 response gain | Native32 response gain | Native improvement |
| --- | ---: | ---: | ---: | ---: |
| Limped paired, 2bb starting pot | 100101 | 0.5195bb | 0.1876bb | 0.3319bb |
| Limped paired | 100102 | 0.4857bb | 0.2461bb | 0.2396bb |
| Single-raised high rainbow, 5bb starting pot | 100101 | 0.8783bb | 0.3040bb | 0.5744bb |
| Single-raised high rainbow | 100102 | 0.9414bb | 0.2990bb | 0.6424bb |

The equal seed/root mean falls from **0.7062bb to 0.2592bb**, an improvement
of **0.4471bb**. Equal weighting of the four within-root percentages falls
from **21.66% to 8.44% of starting pot** (13.23 percentage points). All four
paired directions favor native leaves. The selected boards were already used
for diagnosis; this is a causal development pilot, **not untouched validation,
full-game exploitability, or an Approximate GTO certificate**. Even native32
is far above the descriptive sub-1%-of-pot research objective. Native32 policy
construction took roughly 20–33 minutes per candidate versus 4–5 seconds for
learned32, so directly serving native-leaf re-solves is not the proposed web
route.

The first limped seed's root action audit illustrates the policy difference:
at BB's initial flop decision, native32 checked 53.5% with 0.1025bb
root-weighted local loss; learned32 checked 40.4% with 0.3896bb local loss.
These overlapping local losses are not additive exploitability.

## Is the native64 target itself good enough?

Two exact turn beliefs from the frozen learned128 single-raised candidate were
held fixed on turn card 36: after check/check and after check, small bet, call.
Only the native turn/river update budget changed. The compiled probe, candidate,
public-history/range hash, and per-budget label files are pinned in the local
manifests. Inference was not substituted for a native label.

| Fixed belief | Budget comparison | Reach-weighted conditional-value RMSE by seat | Native policy response gain by seat, lower → higher budget |
| --- | --- | ---: | ---: |
| Check/check | 64 → 256 | 0.2411 / 0.0739bb | 0.0886/0.1073 → 0.0136/0.0125bb |
| Check/check | 256 → 1,024 | 0.0326 / 0.0086bb | 0.0136/0.0125 → 0.0018/0.0016bb |
| Small bet/call | 64 → 256 | 0.1187 / 0.0589bb | 0.0435/0.0565 → 0.0034/0.0038bb |
| Small bet/call | 256 → 1,024 | 0.0087 / 0.0104bb | 0.0034/0.0038 → 0.0003/0.0003bb |

The 64/256 check/check label was rerun independently and its non-timing
contents matched exactly at each budget. Both native64 labels are materially
different from native256 on these selected beliefs. Native256 is substantially
more stable, though the check/check seat-0 drift to native1024 is still
0.0326bb. This is evidence to preflight 256-budget training labels and
consider 1,024 for costly sensitive beliefs; it is **not** a population error
bound, action-EV precision result, or proof that native1024 is exact.

Next, measure the independent flop-search axis with matched learned128 versus
learned256 policies. Then choose a training-only search/deviation-belief corpus
and label budget based on both factors, fit paired students, and re-evaluate
their actual frozen responses before any full-hand integration. The current
20bb web policy and release gates were not changed.
