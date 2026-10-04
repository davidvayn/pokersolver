# Postflop update and native-label pilots

Date: October 3, 2026 (PDT). Status: research only; no model promotion.

## Matched flop-update comparison

The September 21 benchmark's 128-update learned-leaf candidates were the
controls. A rebuilt binary first reproduced a frozen 128-update candidate
byte-for-byte. The only candidate change was 256 flop updates; model weights,
chance seeds, legal tree, native64 played turn/river continuations, all 49
legal turn cards, and the independent JavaScript flop backup audit remained
fixed. Four deliberately selected development/control roots were run with
both retained seeds. The two completed manifest SHA-256 hashes are
`e3d3f2ba15f8ea85dac0a064b8e52863dbaca8760e14d97698e92fe5528afce9`
and `79da5ceb3127a9bc9e0616981f8d3b759e82b3d0b7398473b336d74079a93b35`.
Local ignored artifacts are under `runs/local-flop-update-20261003-compare-a/`
and `runs/local-flop-update-20261003-confirm-a/` in `preflop-solver/neural`.

| Root | Seed 100101 improvement | Seed 100102 improvement |
| --- | ---: | ---: |
| Limped paired | −0.00542bb | +0.01391bb |
| Single-raised high rainbow | +0.10946bb | +0.09429bb |
| Three-bet monotone | +0.00849bb | +0.04618bb |
| Three-bet high-rainbow control | +0.04080bb | +0.01008bb |

Across these eight selected cases, equal-case mean conditional response gain
fell from **0.41214bb to 0.37242bb** (improvement **0.03972bb**), or from
**8.277% to 7.627%** of starting pot (0.650 percentage points). Seven of
eight comparisons improved, but one limped seed regressed. This is credible
evidence of a modest finite-flop-update effect, not a universal convergence
curve, a full-game exploitability estimate, or a reason to serve this candidate.
It remains far above the descriptive sub-1%-of-pot postflop objective on these
selected roots. A 512-update pilot should be considered only after the
continuation-value intervention is compared under the same evaluator.

## Matched native-label control

Training-only search-belief roots 2, 3, and 4 each produced identical
proposer policies and sampled public beliefs at native64 and native256 label
budgets. The resulting corpora each contain the retained 508-state prefix plus
the same 107 added states; the 69 tuning and 72 holdout states remain frozen.
Only the new native value labels differ. The native256 corpus SHA-256 is
`38f329f0863354bd5da6dec6ca1aa4826cfd7ff5f1113aa72e99b5d749afc3cd`;
the matched native64 corpus SHA-256 is
`2efb28692ffea119e65894a8748677cb2dda270a4571ae79d3d786d89bedfa73`.

| Training root | Added states | Reach-weighted 64→256 value RMSE, seats 0 / 1 |
| --- | ---: | ---: |
| 2 | 35 | 0.02834 / 0.01456bb |
| 3 | 36 | 0.07795 / 0.05464bb |
| 4 | 36 | 0.12250 / 0.08913bb |

The values drift materially on these sampled beliefs, especially root 4.
This identifies finite-reference sensitivity but does **not** by itself show
that a student trained on the 256-budget labels makes better decisions.

## Paired students and actual policy response

Two independent students per label budget used the same retained wide
architecture, 600-step cap, objective, split, and seeds 10601/10602. Both
615-state pairs passed Rust/Python prediction parity on every state (maximum
absolute error below 0.000007bb). Frozen authentic holdout RMSE was
1.4345/1.4665bb with native256 labels and 1.4304/1.4587bb with native64
labels. This regression diagnostic slightly favors the control but is **not**
the policy-selection metric. The student manifest SHA-256 hashes are
`6a230119ea503fd5140b870dc7995f8ac63f0fc1c5cf42b1aa5dec557fab6148`
and `e79707c7f11f435af8fd5f120e1a5c2395109bcb6716a767a4f021de9834f971`.

Each pair was then re-solved on the same two selected development roots and
seeds at **128 learned-leaf flop updates**, native64 played continuations,
all 49 legal turns, and the independent JavaScript audit. Only the model
weights changed. The completed response manifest SHA-256 hashes are
`e238057c74fb0832c4ea0fe7d8078bd1d62bfb0fdc7309c3eb83db47b9c14e58`
and `0cc2643ab91c8c8185382d915c41e5da3c66008335f91a673945bdd0134cf193`.

| Selected root | Seed | Frozen response gain | 64-label student | 256-label student |
| --- | ---: | ---: | ---: | ---: |
| Limped paired | 100101 | 0.22863bb | 0.23133bb | 0.23129bb |
| Limped paired | 100102 | 0.40484bb | 0.32578bb | 0.32161bb |
| Single-raised high rainbow | 100101 | 0.71709bb | 0.40889bb | 0.33133bb |
| Single-raised high rainbow | 100102 | 0.54173bb | 0.48652bb | 0.52795bb |

Equal-case mean gain fell from **0.47307bb frozen** to **0.36313bb with
64-label training** or **0.35305bb with 256-label training**. The respective
mean improvements are **0.10994bb / 2.771 percentage points of starting pot**
and **0.12003bb / 3.005 percentage points**. Both arms improve both
single-raised seeds, one limped seed, and slightly regress the other limped
seed. The 256-versus-64 advantage on single-raised seed 100101 is 0.07756bb,
but it reverses by 0.04143bb on seed 100102. Thus extra search-belief
training data is promising; the *additional* native256 label budget is not
yet a stable advantage. These are selected development roots, not untouched
validation or full-game exploitability. Neither pair is release-ready.

The same two student pairs were subsequently evaluated on the declared
`three-bet-monotone` development root and `three-bet-high-rainbow` lower-gap
control. Their completed manifest SHA-256 hashes are
`0375d5d5ccdd74f8f154486155506ef12b18a469cbfd171a97c7a995dabe8996`
and `a0abb78df7c1b8b36169c6b0332d8400da6f1fda7154ecdb3917dc095a917088`.

| Selected root | Seed | Frozen response gain | 64-label student | 256-label student |
| --- | ---: | ---: | ---: | ---: |
| Three-bet monotone | 100101 | 0.37677bb | 0.29785bb | 0.28343bb |
| Three-bet monotone | 100102 | 0.49655bb | 0.39542bb | 0.39407bb |
| Three-bet high-rainbow control | 100101 | 0.27834bb | 0.34492bb | 0.35101bb |
| Three-bet high-rainbow control | 100102 | 0.25318bb | 0.27965bb | 0.27298bb |

Across all eight selected seed/root cases, equal-case mean response gain fell
from **0.41214bb frozen** to **0.34629bb with 64 labels** or **0.33921bb with
256 labels**. Those are mean improvements of 0.06585bb and 0.07293bb,
respectively. **Neither data-only arm passes the promotion rule:** both
regress the lower-gap high-rainbow control on *both* seeds, by 0.0198–0.0727bb
depending on arm and seed. The 256-label budget does not correct that failure;
its small overall mean edge over 64 labels also reverses between the two
single-raised seeds. These student artifacts remain research controls, not
website weights or Approximate GTO models.

Next: diagnose the costly action rankings on the failing control and a
successful root using saved native packets, then test a **separate**
decision-directed objective or targeted coverage adjustment at the same
corpus/training budget. Do not escalate to a 512-update or paid training run
as if the data-only student were already safe. Fresh-board confirmation and
combined full-hand evaluation remain gated on a candidate without material
development/control regressions.

## Why the control failed: frozen-policy action audit

The completed student candidates were re-audited with single-node local
deviations; this reuses their already saved native turn packets and does not
re-solve or change either policy. On `three-bet-high-rainbow`, seed 100101,
the largest local loss is BTN's decision after BB checks:

| Frozen candidate | Check / smallest bet / all-in reach-weighted mix | Root-weighted local loss |
| --- | ---: | ---: |
| Retained policy | 46.3% / 19.2% / 32.8% | 0.3645bb |
| 64-label student | 54.9% / 18.1% / 25.1% | 0.4954bb |
| 256-label student | 55.3% / 17.9% / 24.8% | 0.5069bb |

The student shifts toward checking and away from all-in at this decision,
where the native local deviation loss rises. On the **student's own**
frozen-policy beliefs, the 256-label model agrees with the native best action
on only 44.7% of reached hands; its predicted policy-deviation loss is
0.1337bb against 0.5872bb measured from native continuations, and its
chance-integrated action-contrast RMSE is 0.7137bb. This is direct evidence
that its action ranking/scale is unreliable at the costly control node. It is
not an additive attribution of full response gain, and native64 remains a
finite-budget reference. The pinned diagnostic JSON SHA-256 is
`385ed78eb957084af78bb08e21463eb47c3c56b7f4771ed38adbf1f6af71a53a`.

The successful `single-raised-high-rainbow` seed 100101 shows the other side:
after BB checks, the 256-label student bets the smallest size 88.4% instead
of 79.3% in the retained policy; native root-weighted local loss at that node
falls from 0.8773bb to 0.2634bb. Its diagnostic JSON SHA-256 is
`55c3d0c953416183d1006ddfe7f79b6cad5c718eef9aea488a379a9ce464c4ea`.

A separate cross-model probe on the *retained* policy's unchanged beliefs
found lower top-node ranking loss for the new high-label model on control
seed 100101 (about 0.1704bb versus the retained model's 0.2423bb reported
earlier). Yet the new policy's own response worsens. That contrast makes
on-policy belief distribution and action-contrast calibration worth testing;
an off-policy value or RMSE improvement is insufficient. The cross-model
probe manifest SHA-256 is
`a459d77bd27740476ffe68bb5174be9bb3396b751267fcaf0a7eeee528a34715`.
