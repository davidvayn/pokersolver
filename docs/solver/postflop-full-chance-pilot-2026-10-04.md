# Full-chance continuation of the action-contrast pilot

Status: **full-chance data and conditioning passed; paired student fits underway**.
No policy improvement, full-game exploitability result, or model promotion
is claimed before the actual-policy screens.

The [sampled pilot](postflop-action-contrast-pilot-2026-10-04.md) was inconclusive:
two eight-turn blocks changed action rankings by much more than the planned
0.02bb useful effect. The user authorized continuing with the focused exact
chance experiment in [Section 10 of the plan](postflop-decision-pilot-plan.md).

## What changed

Complete all 49 public turn cards for the same frozen training snapshots, at
unchanged native64. Reuse the previous 16 native64 packets per family; solve
only the missing 33. Complete prefix branches, exact blocker/range accounting,
profile-consistent support and the finite continuation contract remain fixed.
No sampling correction is replaced by a naive `1/49`: the existing raw-CFV
`1/45` convention and exact private-card removal are preserved.

The collector uses immutable inputs, unique native64 turn receipts, two guarded
workers per family, a measured 1.5x projected-cost margin and a 45-minute cap.
Only after observing <1GiB packet footprints and 55% free system memory were
roots 3/4 run together: four native workers TOTAL, not two four-worker pools.
Two feature processes prepare the completed root-2 shard under a separate
4GiB parent-worker guard and the same system memory-pressure stop. This is
deterministic preprocessing, not training. All work preserves the 20GiB disk
reserve; the source/native/WASM playing code is unchanged.

The new joined data decision requires all three original training families,
49 unique native64 turns per family, complete matching calibration shards,
consistent authentic support and the measured native64/256 teacher sentinels.
It permits only a paired fit preflight, not promotion or release. Each family's
245/441/441 states stay in its own bounded shard; the frozen 615-state corpus
and its 640-state/256MiB bounds are not enlarged or bypassed.

## Completed root-2 result

The three-bet reference reused 16 turns and solved the remaining 33 in
**102.27 seconds**, including backup/diagnosis and writing the calibration shard.
The saved-timing projection was 142.61 seconds. Memory pressure stayed normal;
both decision groups retain essentially 100% authentic profile support.

Below, loss means the reach-weighted value lost *on the full49/native64 target*
by choosing the sampled target's best action. It is not loss against a perfectly
solved game, a statistical confidence interval, or full-hand exploitability.

| Decision | Original eight A | Reserved eight B | Combined sixteen |
| --- | ---: | ---: | ---: |
| Initial BB | 0.10050bb | 0.10590bb | 0.08836bb |
| BTN after check | 0.10249bb | 0.08625bb | 0.02914bb |

The measured native64/256 sentinel changes no best actions in either group;
the same one-turn upgrade inside full49 also changes no best actions. Its
all49 contrast RMS effect is 0.000146/0.000821bb. That limited sentinel is not
a proof that every native64 continuation is converged. It does show the
observed sampling problem is much larger here, and even sixteen public cards
do not adequately determine these particular rankings.

The actionable correction is complete chance coverage for this small training
bundle set, not a larger unchanged student fit. The actual payoff must still
be measured using C0/C1's resulting policies, not this target-data diagnostic.

### Other families and joined data decision

Root 3 completed in 2,606.02 seconds (43.43 minutes); root 4 completed in
2,724.98 seconds (45.42 minutes). These wall times include final target
validation, diagnosis and compression, not only native solving. The projection
underestimated the cost while the larger families and preprocessing overlapped.
No native worker or memory-pressure guard failed.

| Root / decision | Original eight A loss | Reserved eight B loss | Combined sixteen loss |
| --- | ---: | ---: | ---: |
| 3 / Initial BB | 0.03367bb | 0.02581bb | 0.01421bb |
| 3 / BTN after check | 0.03480bb | 0.04098bb | 0.02914bb |
| 4 / Initial BB | 0.00709bb | 0.01527bb | 0.00230bb |
| 4 / BTN after check | 0.01249bb | 0.00543bb | 0.00475bb |

All three shards are complete and preserve essentially 100% authentic
profile-consistent support. The maximum native64/256 sentinel ranking loss is
0.0169985bb; the maximum effect of its one-turn upgrade inside full49 is
0.00001312bb. The joined screen permits a paired conditioning preflight under
the predeclared 0.02bb data-sensitivity rule. This is NOT an equilibrium gate,
nor proof that every native64 teacher value is converged.

The root-4 collector exposed a timer-boundary bug: it checked the deadline
after native collection, but not after final verification/serialization. Native
collection finished within 45 minutes; file receipt timings indicate roughly
100.8 seconds of post-label finalization, producing a 25-second total overrun.
The original immutable receipt remains unchanged and this overrun is reported,
not hidden. Future collectors explicitly separate the 45-minute native label
cap from a bounded three-minute finalization phase and fail closed if either
phase stops. This correction does not redo or change the verified native labels.

## Fitting safeguards

The student runner can build deterministic features using one or two processes,
and requires all49 in its affine backup when the manifest declares all49.
Before any fitting, a training-only component-gradient preflight resolves the
contrast weight and measures complete-family gradient time. Cadence is chosen
only from that cost, never a policy/holdout score: try 4, then 8, then 20 updates
per bundle addition. Each choice preserves equal family counts at step 600
and is identical for C0/C1. Stop if none fits the remaining two-hour cap.
Fixed step-600 weights still decide the comparison; no extra optimizer updates,
response-selected intermediate snapshots, or changed architecture are added.

Conditioning completed in **1,056.32 seconds** (17.61 minutes), including the
two large new feature caches. Peak parent footprint was **3,091,385,752 bytes**
(2.88GiB), with normal system pressure. Duplicate parsed primary/reference
sources and decoded leaf packets are released after their split/ordering checks;
the complete validated calibration source and all metadata remain intact.
All hashes/splits and mathematical inputs are unchanged by this memory cleanup.

The complete-family gradient times were 2.3294 / 3.1223 / 2.4806 seconds for
roots 2/3/4. Cadence 4 is affordable without reduction, projecting **5,394.7
seconds** for matched fitting and exhaustive parity with the declared 1.5x
margin. Resolved contrast weight is **1.99357696**, yielding the intended
25% gradient-norm fraction without hitting cap 10. These are training-only
conditioning quantities, not selected using policy or holdout outcomes.

The fitting stage uses a new immutable output directory and reuses the completed
feature caches. Only one MLX fitting process runs at a time. C0/C1 each use
seeds 10601/10602, fixed 600 steps, 150 bundle updates, and equal family counts.
The final four models require independent NumPy/native exhaustive prediction
parity before their full49 own-policy control/monotone screens.

## Immutable root-2 evidence

The 141-test targeted suite passed before the finalization correction; its new
timer regression plus the focused student/data tests also pass. Tests include
cache-receipt identities, full-chance
diagnostic direction, finite-teacher stopping, equal-family cadence selection,
and the prior affine/gradient/parity fixtures. No Rust source or generated WASM
changed in this continuation; the previous native release suite remains the
applicable source verification. Full C0/C1 operational verification is in progress.

- Source complete 8+8 blocks: `runs/local-action-bundles-20261004-sensitivity-a/manifest.json`,
  SHA-256 `f15d26a0ec566ea136d2f4de181299471727089cc4b81c9710d9d8c36af592b6`.
- Full49 root-2 reference: `runs/local-action-bundles-full49-20261004-root2-a/manifest.json`,
  SHA-256 `44680b3f60dfdae1606b987c59e2e743153d4121ed978bcda19e69131a463ab0`.
- Root-2 calibration shard: `runs/local-action-bundles-full49-20261004-root2-a/calibration.json.gz`,
  SHA-256 `1d5da2190fd3512230c8c7efac973e7e455c0febae134f8cfb405cb37aef5105`.
- Root-3 reference: `runs/local-action-bundles-full49-20261004-root3-a/manifest.json`,
  SHA-256 `831e701ac02123589c306f822f67d3b16706d727ea41ba0cc2083d52ae357559`.
- Root-4 reference: `runs/local-action-bundles-full49-20261004-root4-a/manifest.json`,
  SHA-256 `6fefdd0a2f31a3ef730c65981b37d4bbfc5bb4dc0efcceb42e079ded7ded1cb1`.
- Joined full-chance TRAIN set: `runs/local-action-bundles-full49-20261004-set-a/manifest.json`,
  SHA-256 `ca71414f6a451c418761e42ea43175543291084aa3a2e9a99a82d74cf1b5bfcc`.
- Training-data decision: `runs/local-action-bundles-full49-20261004-set-a/decision.json`,
  SHA-256 `8a29b062f40277a38d93bdc5a2b29204ed990dbbdd57a4b490d6d9288657bd60`.
- Conditioning: `runs/local-full49-contrast-conditioning-20261004-a/manifest.json`,
  SHA-256 `10986c44829eaa4cb6744d0f0ade1b91d99af59e0571ea55524c0b50cef09922`.
- Active matched student stage: `runs/local-full49-contrast-students-20261004-a/manifest.json`.
  Do not pin/hash its mutable running manifest as a completed pair.

All run paths are relative to `preflop-solver/neural/`. The website, unrelated
dirty benchmark work, and global exploitability gates are outside this reference
milestone. Subsequent actual-policy comparison results must be appended before
calling the paired experiment successful.
