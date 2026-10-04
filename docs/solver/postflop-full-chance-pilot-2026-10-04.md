# Full-chance continuation of the action-contrast pilot

Status: **paired pilot complete; C0 and C1 rejected for material rainbow regressions**.
Full-chance data, fitting and export parity passed, and frozen training rankings
improved. Neither arm earns model promotion. No full-game exploitability
improvement or Approximate GTO qualification is claimed.

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
All four fixed-step models passed independent NumPy/native exhaustive prediction
parity over all 615 states and both 1,326-hand vectors. The complete matched
stage took **1,674.27 seconds** (27.90 minutes), below its two-hour cap. Maximum
prediction discrepancy across the four models was **0.00000584bb**. Both arms
retain 600 optimizer steps, 150 bundle updates and 50 updates per family.

### Frozen training-decision probe

A read-only probe uses the already verified feature caches and exact all49
affine backups. It compares C0/C1's predicted action differences on the same
six frozen TRAIN decision groups; it neither updates weights nor evaluates
the policies those weights produce. Off-support holdings are excluded exactly
as in training. Common action-value offsets do not affect its contrast/ranking
results, and deterministic tests cover this and invalid/empty support.

Across six groups and two seeds, the equal-group mean of each group's
reach-weighted best-action loss decreased from **0.08936bb (C0)** to
**0.05130bb (C1)**, a **42.6% reduction** on these training targets. This is
not an authentic full-hand average, fresh-board result, EV-confidence estimate
or exploitability improvement. The decisive test is each student's actual
policy against the frozen response evaluator, completed below.

The final probe completed in 78.01 seconds with a 1.45GiB worker peak and
normal system pressure. Repeating after strengthening its provenance capture
reproduced all numerical results exactly; original receipts remain immutable.
It requires existing hash-verified caches, complete matching pairs and all49
groups, validates target/feature ordering, and pins model, data, helper-source
and cache-metadata identities. It creates no new labels or optimizer updates.

### Completed actual-policy screens

C0 completed four cases in **799.83 seconds** (13.33 minutes), using four
native packet workers TOTAL. Every case integrates all 49 public turns with
unchanged 128 flop updates/native64 played continuations and the independent
JS backup audit. C1 then completed the identical screen in **626.42 seconds**
(10.44 minutes). Lower half-summed restricted response gain is better.

| Case / solver seed | Retained model | C0 full49 calibration | C1 plus contrast |
| --- | ---: | ---: | ---: |
| Three-bet high-rainbow / 100101 | 0.278335bb | 0.395050bb | 0.418741bb |
| Three-bet high-rainbow / 100102 | 0.253183bb | 0.330574bb | 0.301359bb |
| Three-bet monotone / 100101 | 0.376774bb | 0.286401bb | 0.278278bb |
| Three-bet monotone / 100102 | 0.496554bb | 0.329069bb | 0.342402bb |

C0 is **rejected** under the predeclared screen: rainbow regressions are
0.116714/0.077391bb, exceeding the 0.01bb material-regression rule. Its mean
retained improvement is only 0.015938bb (below 0.02bb), and seed 100101's
paired mean regresses both retained and R0. Better monotone cases must not hide
this known failure.

C1 is also **rejected**: rainbow regressions are 0.140406/0.048176bb.
Its 0.016017bb equal-case mean improvement over retained is below the
predeclared 0.02bb useful-effect screen. C1 minus C0 improvement averages just
**0.000078bb**, with opposing paired seed means (-0.007785/+0.007941bb).
The contrast objective improved the frozen training decisions but did not
establish an actual-policy benefit. These are conditional development screens,
not global exploitability estimates, statistical confidence bounds or new
release gates. Both seed/board regressions remain explicit.

### Own-policy and same-board retained-policy diagnosis

Reuse the completed native packets, the existing hash-pinned native prediction
probe and independent JS action-value backup; perform **no new native solves**.
For seed 100101 on high-rainbow, compare each model's values with native
profile values under its own frozen policy. Then hold the board and retained
policy fixed, applying C0/C1 as alternative value models. This prevents
attributing every failure to the changed policy's ranges. The two diagnosis
stages took 7.46/7.21 seconds, with normal memory pressure.

The largest actionable decision mismatch is BTN after BB checks:

| Value model / frozen policy | Native best agreement | Native loss from predicted best | Contrast RMSE |
| --- | ---: | ---: | ---: |
| C0 / C0 own policy | 45.06% | 0.33766bb | 0.83428bb |
| C1 / C1 own policy | 44.21% | 0.38796bb | 0.88695bb |
| C0 / retained policy, same board | 24.48% | 0.40189bb | 0.70429bb |
| C1 / retained policy, same board | 26.28% | 0.37942bb | 0.74646bb |

On C1's own after-check node, the frozen policy's local deviation loss is
**0.69696bb under native values**, versus **0.14028bb under its predicted
values**. These are local conditional diagnostics, not summed full-hand
exploitability. All-in-only branches have zero contrast discrepancy because
their payoff is evaluated exactly, a useful internal control.

One concrete own-policy example is **Ks5d**: predicted values favor check
(5.17854bb) over bet-to-5bb (4.80282bb); native values favor bet-to-5bb
(5.61651bb) over check (4.59933bb), reversing a **1.01718bb** action gap.
This is not a uniform-strategy display issue or merely an absolute-value offset.

**Inference:** continuation action-value generalization outside the three
TRAIN families is the clearest remaining bottleneck here. The same-board
retained-policy probe shows changed own-policy ranges are not the sole cause.
It does not isolate board coverage from every belief/future-policy effect or
prove all native64 continuations converged. The finite teacher and safety
limitations remain. A larger unchanged iteration run is not justified by
these results.

### Decision and next bounded intervention

Do not expand these rejected students to expensive roots, deploy them, or buy
compute to scale this unchanged fit. The pilot boundary specified in the plan
is complete, including its failure diagnosis; success-conditional broad
confirmation, scaling and website integration were not triggered.

The next relevant pilot should improve **coverage of continuation action
contrasts across new TRAIN board families and search beliefs**, not repeatedly
fit these three frozen families. Start with a small stratified TRAIN family
block spanning three-bet/high-card textures, retaining all existing evaluation
families unchanged. Capture early/middle/late and final-average search beliefs
from current proposers. First compare same-board/known-policy predictions with
own-policy predictions against reused or bounded native references; expand
complete all49 bundles only where that diagnosis identifies useful missing
supervision. Preserve old-data replay, equal C0/C1 work, fixed checkpoints and
the existing regression screen. Do not use consumed evaluation families as
training or count them as fresh confirmation. Re-estimate cost before labeling:
the limped full49 references cost roughly 45 minutes each, not root 2's two
minutes. A safe-continuation redesign remains conditional on evidence of a
continuation-contract problem rather than assumed from this surrogate failure.

Artifacts for this continuation, including the entire shared feature cache
(a conservative overcount of newly generated data), occupy approximately
**3.02GiB**. Free disk remained about **119GiB**, above the 20GiB reserve.
Peak fitting footprint was **2.78GiB**; packet workers were about 0.13GiB
maximum on the previous measured screen. No memory/time guard stopped either
student or policy stage; no paid compute was used.

## Verification and immutable evidence

The complete **144-test targeted suite passed**, including the finalization
correction and two frozen-probe tests. Tests include
cache-receipt identities, full-chance
diagnostic direction, finite-teacher stopping, equal-family cadence selection,
and the prior affine/gradient/parity fixtures. No Rust source or generated WASM
changed in this continuation; the previous native release suite remains the
applicable source verification. C0/C1 fitting/export, full49 actual-policy
screens, deterministic frozen-probe comparison and targeted native/JS action
diagnoses all completed. No frontend or runtime route changed, so no browser,
TypeScript build or generated-WASM check is claimed for this milestone.

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
- Completed matched student stage: `runs/local-full49-contrast-students-20261004-a/manifest.json`,
  SHA-256 `c9243ed6df6b11c4de2ae59594a628746259d1322390b82edf57be394a2d558e`.
- Completed C0 pair: `runs/local-full49-contrast-students-20261004-a/C0/manifest.json`,
  SHA-256 `510189f90f4f974351c30772c72bc8ca34b36e511e56a2dfa4b1035391905790`.
- Completed C1 pair: `runs/local-full49-contrast-students-20261004-a/C1/manifest.json`,
  SHA-256 `78b4b531af3d27ce3a1a69d7355b933e4bb6514e61532a7d5b8ece90cebf51c3`.
- Frozen training probes: `runs/local-full49-contrast-frozen-probes-20261004-b/analysis/manifest.json`,
  SHA-256 `4b2ca214c681b65dfec63b4ba7aac20b4cc50a2dd2413d25a5c52b2d674b849c`.
- C0 actual-policy screen: `runs/local-full49-contrast-response-20261004-C0-a/manifest.json`,
  SHA-256 `fd0539dc3d192d9d6c93f784a3d2a2a7a35ba54bf6a10a305d4af4584e560de5`.
- C1 actual-policy screen: `runs/local-full49-contrast-response-20261004-C1-a/manifest.json`,
  SHA-256 `e3e5d4dda283f3b19e8d0c0b7dd3a57281586d5df1e9b4d5f4d5bd2c702a79c3`.
- C0 policy-screen decision: `runs/local-full49-contrast-score-20261004-C0-a/manifest.json`,
  SHA-256 `68ae2f79d9a22c4a2fd93c6ef4c3fa097fb153aa27e47d337735d02e0458f207`.
- C1 policy-screen decision: `runs/local-full49-contrast-score-20261004-C1-a/manifest.json`,
  SHA-256 `5029afb324a5c6b39277c865c60b1b711a6bd9a0de5ff1af93ba8b40f97a2c51`.
- Own-policy diagnosis: `runs/local-full49-contrast-own-policy-diagnosis-20261004-a/manifest.json`,
  SHA-256 `cef3c3374c418f6a48b7006c4e856c76de9be4a57c0ab4bdca5ed718c3b2975f`.
- Same-board retained-policy diagnosis: `runs/local-full49-contrast-retained-policy-diagnosis-20261004-a/manifest.json`,
  SHA-256 `c65ee3fe6b0fcb8f67506858c70e49737825255e7b7e1abbc34a2608324f6d86`.

All run paths are relative to `preflop-solver/neural/`. The website, unrelated
dirty benchmark work, and global exploitability gates are outside this pilot
milestone. No weights were activated and neither failed arm is called successful.
