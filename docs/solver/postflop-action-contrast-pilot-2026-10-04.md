# Coherent action-contrast pilot

Status: **data pilot inconclusive; no student fit or promotion**. This is the
conditional Stage 2 of [the decision-pilot plan](postflop-decision-pilot-plan.md),
following the [rejected replay pair](postflop-replay-pilot-2026-10-03.md).

## Correctness and target contract

The export is a test-only native seam, not a changed playing solver. It records
every legal frozen-prefix branch, exact reaches, renormalized frozen policies,
and native fold/all-in CFVs. Complete turn label packets include their exact
public states, conditional values, profile/BR raw CFVs, and continuation hashes.

The Python affine backup integrates all common sampled public turns before
forming action contrasts. It applies `49/K/45` chance factors, opponent blocker
mass once, later own-action probabilities but not own reach, and parent
conditional-value normalization. It never differentiates through CFR or argmax.
Topology, leaf ordering, exact range reconstruction, card blockers, candidate
identity and complete branches are checked before fitting.

Existing calibration labels are profile values for positive-own-reach holdings
and BR completions for zero-own-reach holdings. Actual response evaluation uses
the frozen profile everywhere. These contracts are intentionally different.
Calibration retains the completion targets and bounded counterfactual coverage;
the auxiliary masks holdings that lack consistent profile support at a needed
leaf. True ranges stay unchanged. This is not evidence that the existing
evaluation accidentally used BR completions.

The rebuilt binary reproduced the retained 128-update high-rainbow candidate
byte-for-byte. The new affine backup matches cached full-49 native/JavaScript
action EVs within **2.8422e-14bb**, including weighted action means and costly
holdings at BB's initial decision and BTN after check. Essentially all authentic
reach is support-consistent on that control. Thus the off-support distinction
does not explain its regression on authentic holdings.

The blocker fixture initially caught a missing hero-turn mask in the new code.
Opponent-compatible mass alone cannot detect that hero contains the drawn turn;
the affine coefficient now explicitly zeroes these impossible queries. The
regression test passes. Unequal masses, own-reach exclusion, exact terminal
terms, chance integration, leaf completeness, duplicate turns, finite-difference
gradients and full-graph/two-pass MLX gradient parity are tested.

## Bounded labels and matched objective

Frozen training families are original roots 2/3/4, excluding tuning/holdout and
all consumed benchmark families. Eight shared sampled turns per family produce
40/72/72 complete native64 labels. One complete native256 sentinel turn and one
disjoint native64 sensitivity turn per family diagnose label/chance sensitivity.
The remaining disjoint seven turns are reserved for a single bounded expansion
if needed. No noisy isolated reservoir labels are treated as action groups.

The two-turn expensive-family preflight took **98.42 seconds** for 18 labels.
The 1.5x-margin projection for all initial labels/sentinels was **2,452 seconds**
(40.9 minutes), below the 90-minute cap. Two native workers, 2GiB per-process
guards, system memory-pressure checks and a 20GiB disk reserve remain active.
Data are separate bounded calibration shards; the frozen 615-state corpus and
its 640-state/256MiB guards are not changed.

C0 and C1 use the same ordinary batches, bundle calibration, family schedule,
seeds 10601/10602, architecture, optimizer and 600 update budget. A bundle
gradient is added every fourth step, then the optimizer updates once. C1 alone
adds calibrated, depth-normalized all-pairs Huber contrast supervision. A
training-only component-gradient preflight resolves one coefficient, targeting
25% of the value-gradient norm with cap 10. This coefficient is not selected
using development responses. Fixed step-600 weights decide the comparison;
200/400 snapshots are diagnostics, not response-selected winners.

Complete leaf vectors are microbatched without splitting the zero-sum projection.
The first pass computes chance-integrated Q; the second accumulates calibration
and affine VJPs at unchanged weights. The independent NumPy export verifier can
reuse byte-verified deterministic features, avoiding repeated feature generation
without replacing its dense/projection calculation by the MLX forward pass.

## Completed data pilot and decision

Initial labeling/sentinels took **1,187.7 seconds**. The reserved disjoint
eight-turn block added **651.3 seconds**, giving **30.65 minutes** combined,
well below the 90-minute label cap. Both stages completed without resource
stops; system pressure stayed normal. All six groups retain essentially 100%
profile-consistent authentic support. Native label artifacts use about 162MiB.

The like-for-like eight-turn blocks still change important action rankings:

| Training root | Decision | Contrast RMS difference | Loss on block A from choosing block B's best action |
| --- | --- | ---: | ---: |
| 2, three-bet | Initial BB | 0.86956bb | 0.31427bb |
| 2, three-bet | BTN after check | 1.48654bb | 0.24175bb |
| 3, limped | Initial BB | 0.35974bb | 0.01967bb |
| 3, limped | BTN after check | 0.33892bb | 0.03506bb |
| 4, limped | Initial BB | 0.46905bb | 0.06022bb |
| 4, limped | BTN after check | 0.31988bb | 0.03581bb |

These are disagreement diagnostics between two finite training samples, not
loss against an exact target, standard errors, or exploitability. Block B is
not a newly held-out board family. The large three-bet disagreement nevertheless
makes an intended 0.02bb policy benefit difficult to interpret with this data.
The initial eight-versus-one check had even larger differences, but was not
used as the decisive comparison because its sample counts were unequal.

On the measured native64/256 sentinel turns, contrast RMS differences are
0.00716–0.11380bb and action-selection disagreements cost 0–0.01700bb.
Upgrading that one turn within the eight-turn target changes rankings by at
most 0.000559bb loss. Those limited sentinels do not prove every teacher is
converged, but chance coverage is the larger observed problem here. Best-action
selection is nonlinear; unbiased raw CFV inclusion correction does not make
eight-card action rankings precise. Large three-bet payoffs magnify this noise.

**Decision:** stop this data arm as inconclusive, as specified in Section 6B
of the plan. Do not fit four students, sweep contrast coefficients, claim an
objective improvement, run expensive response cases, or change website weights.
Both the original block and the planned reserved-block expansion are preserved.
The machine-readable data decision is pinned to both the original training
manifest and the extended sensitivity manifest. The student controller requires
a hash-verified, fit-ready data decision before any fitting stage; the actual
inconclusive receipt was rejected by a CLI smoke check before creating output
or allocating features. This is a research-data screen, not a new release gate.

A no-optimizer-update conditioning preflight was cancelled after the data
decision, at 720.4 seconds and approximately 2.92GiB peak footprint. Its complete
primary deterministic feature cache is retained (about 914MiB); incomplete
entries are never accepted without matching metadata and array hashes. No
coefficient was resolved and no actual C0/C1 student was fitted. An initial
preflight stopped in 3.6 seconds on a split-setting error, then the runner was
corrected to use the already established training-refresh contract while
protecting byte-identical tuning/holdout labels. That fix is regression-tested.

## Verified implementation and conditional next action

134 targeted Python tests passed, including teacher-sensitivity qualification,
student split-contract preservation, and fail-closed fitting. Native
release tests passed: 354 unit tests and 9 CLI tests (56 research tests deliberately
ignored unless explicitly invoked with guarded, pinned artifacts). Two-pass
gradients, fixed-final checkpoint scheduling and independent cached-feature
prediction parity are verified on deterministic fixtures; the complete 600-step
C0/C1 operational pipeline remains unexercised. No full-game exploitability
improvement has been established and no website weights changed.

The next focused data experiment should establish reliable chance integration
on the cheapest problematic three-bet training family: reuse the existing 16
turns, measure cost for the remaining 33, and compare sampled contrasts with an
all49 native64 reference. Alternatively, test a pinned learned control variate
whose complete chance mean is integrated before the sampled native correction;
verify unbiased accounting and measured ranking-variance reduction before use.
Do not assume a naive `1/49`, different cards per action, or more unchanged
student steps will resolve the issue. This is a proposed next experiment,
**not a run launched during this pilot**. Only after target quality and the
fit-cost preflight are satisfactory should C0/C1 proceed to their own-policy
full-49 screens and the plan's conditional expansion.

Local immutable evidence:

- Cached affine parity: `runs/local-action-bundle-parity-20261004-b/manifest.json`,
  SHA-256 `db1c6270b5aed9cc6c9be5cebd5cd00b87a3c39ded13a0a74d8a7e3e08dde50b`.
- Retained-policy reproduction: `runs/local-flop-update-20261004-bundlebinary-preflight-a/manifest.json`,
  SHA-256 `066c6a4503af228e9509f9ea76c67322a3627364aa1b356501912ec3607697d0`.
- Immutable binary: `runs/local-compact-binaries/preflop-solver-086305216c9d5b332cedafc5f3cf1ff183dcbcbd57b9027a3eae11060a14e0fb`.
- Initial labels: `runs/local-action-bundles-20261004-a/manifest.json`,
  SHA-256 `ad0c914cc3c00c523e9a2e19c43e71862b6f5990dc6ad2d923400495e28a62e0`.
- Complete reserved block: `runs/local-action-bundles-20261004-sensitivity-a/manifest.json`,
  SHA-256 `f15d26a0ec566ea136d2f4de181299471727089cc4b81c9710d9d8c36af592b6`.
- Eight-versus-eight diagnosis: `runs/local-action-bundles-20261004-sensitivity-a/quality.json`,
  SHA-256 `7fc6b826c63324da518598dc969e578929a4d6d0b68c3656cf5e3fc991e443e6`.
- Qualified diagnosis (original diagnosis remains unchanged):
  `runs/local-action-bundles-20261004-sensitivity-a/quality-qualified.json`,
  SHA-256 `08951e0735107251ce15ae0aa3ab5ca9ce6c928e6b479d42e893ddccf59efe76`.
- Inconclusive data decision: `runs/local-action-bundles-20261004-sensitivity-a/decision.json`,
  SHA-256 `cfdf0c681fa0629737f83a1ae6926025ccff3a6256aafecb1bb3be39c5c56f0d`.

Verification commands (from `preflop-solver/neural/` unless noted):

```bash
../.venv-neural/bin/python -m unittest test_action_bundle_quality test_action_contrast_students test_action_contrast_dataset test_action_contrast_loss test_validate_public_value_parity test_train_public_value_network test_native_value_dataset test_mixed_native_reference test_native_replay test_student_value_pilot test_decision_pilot_screen
# From preflop-solver/:
cargo test --release -- --test-threads=2
```

No frontend or WASM runtime behavior was changed; browser acceptance and an
app build are not claimed for this native research-only milestone.

All run paths above are relative to `preflop-solver/neural/`. The website and
unrelated dirty benchmark changes are outside this pilot's scope.
