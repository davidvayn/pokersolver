# Native replay-balancing pilot

Date: October 3, 2026 (PDT). Status: complete, **rejected**; research only.

This executes [the bounded decision-pilot plan](postflop-decision-pilot-plan.md).
No website weights, serving route, action abstraction, or native solver changed.

## Cached screen calibration

Reused the frozen response candidates, all49 native turn packets and completed
independent backup audits. SHA-256 verification passed; no native solves or
new labels were needed. Calibration completed in approximately 0.48 seconds.
Its immutable local artifact is
`preflop-solver/neural/runs/local-decision-calibration-20261003-a/manifest.json`.

| Cached arm | Mean improvement across the cheap four cases | Screen result |
| --- | ---: | --- |
| Native64-label students | +0.02175bb | Rejected: both high-rainbow control seeds regress |
| Native256-label students | +0.02584bb | Rejected: both high-rainbow control seeds regress |
| 256 flop updates, retained weights | +0.02639bb | Promising: both paired means improve, no material control regression |

This is retrospective calibration on consumed three-bet roots, not proof of
future generalization. Fixed old-policy value probes are insufficient: the
saved native256 student's own control diagnosis has 0.13366bb predicted local
policy loss versus 0.58716bb native loss, and 0.71374bb contrast RMSE.

## Implemented intervention

The 615-state native64 corpus remains unchanged. Replay validates an exact
508-state target AND capture-manifest prefix against its pinned original
artifact, then partitions only training rows. The 69 tuning and 72 holdout
states remain unchanged and cannot enter replay batches.

Every eight-example batch contains seven retained training examples and one
appended search example. Pot quotas remain 3 small / 3 medium / 2 large. The
appended slot rotates through all eight positions, preserving the same pot
mixture and avoiding starvation of large-pot appended examples. Actual draws
and any empty-band fallbacks are reported separately from poker ranges.

Replay is opt-in. The disabled path calls the original sampler and consumes
the same NumPy RNG sequence, verified through the actual MLX trainer. Legacy
supplemental datasets remain forbidden for native training.

## Completed paired fit

Artifact directory: `preflop-solver/neural/runs/local-native-replay875-20261003-a/`.
Seeds 10601/10602, wide architecture, exact-runout feature schema, original
objective/optimizer, 600-step cap, and original family split are fixed.
Only sampling changes. No warm-start and no new native labels.

Both seeds completed 600 steps. The unchanged tuning rule selected step 550
for seed 10601 and step 400 for seed 10602 (R0 selected 600 for both).
Each fit drew exactly 4,200 retained and 600 appended examples with no
empty-band fallbacks. Retained small/medium/large draws were 1,575/1,575/1,050;
appended draws were 225/225/150. The intended intervention therefore occurred.

Fitting took 404.6 seconds with a measured 3.31GiB peak footprint. Fitting
plus exhaustive native/Python parity took 1,257.5 seconds. Maximum differences
were 0.000006172bb and 0.000005756bb, below the 0.0001bb tolerance. Authentic
holdout RMSE was 1.4564bb/1.5093bb; it did not improve over R0 and was not used
to substitute for the actual policy test.

## Actual-policy screen and decision

Both seeds were evaluated at 128 flop updates, native64 continuations, and
all 49 turns, with the independent backup audit. The four cases took 773.6
seconds. Lower conditional response gain is better.

| Root | Solver seed | Retained gain | R0 gain | Replay gain | Improvement vs retained |
| --- | ---: | ---: | ---: | ---: | ---: |
| Three-bet high-rainbow | 100101 | 0.278335bb | 0.344916bb | 0.296841bb | −0.018506bb |
| Three-bet high-rainbow | 100102 | 0.253183bb | 0.279652bb | 0.285081bb | −0.031898bb |
| Three-bet monotone | 100101 | 0.376774bb | 0.297848bb | 0.340417bb | +0.036356bb |
| Three-bet monotone | 100102 | 0.496554bb | 0.395416bb | 0.432340bb | +0.064214bb |

Mean improvement versus retained is only 0.012542bb. Both control regressions
exceed the predeclared 0.01bb limit. Replay also loses the second paired mean
versus R0 by 0.021177bb. **Reject R1; do not run the expensive roots or promote
weights.** Balancing exposure alone did not fix action-relevant errors. Proceed
to the separately controlled coherent action-contrast pilot, starting with
backup/target-contract tests. These are conditional postflop diagnostics, not
full-game exploitability measurements.

Immutable local manifests (SHA-256):

- Fit: `runs/local-native-replay875-20261003-a/manifest.json`,
  `89dc9b465cd1d8fa756aad4a703850cd2420f2e8359d8c24c0bf0fba05144c9a`.
- Responses: `runs/local-native-replay-response-20261003-cheap-a/manifest.json`,
  `61cf175ef3b49131a8a252715e893480731fb1d488ddbfe0a3de0ad95b6db11b`.
- Decision: `runs/local-decision-replay875-score-20261003-a/manifest.json`;
  its pinned inputs are verified by `decision_pilot_screen.py` before scoring.

## Verification

`python -m unittest test_native_replay test_train_public_value_network
test_native_value_dataset test_mixed_native_reference test_decision_pilot_screen
test_student_value_pilot`: **74 tests passed**. Existing constant-vector
correlation fixtures emit NumPy warnings; no tests failed.

`git diff --check` passed. Free disk was 124GiB and macOS memory-pressure level
was normal at launch. There are no frontend or Rust behavior changes, so an
app build and native rebuild are not required for this milestone.
