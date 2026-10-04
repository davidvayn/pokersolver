# Full-chance continuation of the action-contrast pilot

Status: **root 2 reference complete; roots 3/4 and conditioning in progress**.
No student fit, policy improvement, full-game exploitability result, or model
promotion is claimed by this reference milestone.

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

## Immutable root-2 evidence

141 targeted Python tests pass, including cache-receipt identities, full-chance
diagnostic direction, finite-teacher stopping, equal-family cadence selection,
and the prior affine/gradient/parity fixtures. No Rust source or generated WASM
changed in this continuation; the previous native release suite remains the
applicable source verification. The full C0/C1 operational fits remain pending.

- Source complete 8+8 blocks: `runs/local-action-bundles-20261004-sensitivity-a/manifest.json`,
  SHA-256 `f15d26a0ec566ea136d2f4de181299471727089cc4b81c9710d9d8c36af592b6`.
- Full49 root-2 reference: `runs/local-action-bundles-full49-20261004-root2-a/manifest.json`,
  SHA-256 `44680b3f60dfdae1606b987c59e2e743153d4121ed978bcda19e69131a463ab0`.
- Root-2 calibration shard: `runs/local-action-bundles-full49-20261004-root2-a/calibration.json.gz`,
  SHA-256 `1d5da2190fd3512230c8c7efac973e7e455c0febae134f8cfb405cb37aef5105`.

All run paths are relative to `preflop-solver/neural/`. The website, unrelated
dirty benchmark work, and global exploitability gates are outside this reference
milestone. Subsequent actual-policy comparison results must be appended before
calling the paired experiment successful.
