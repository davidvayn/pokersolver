# Native replay-balancing pilot

Date: October 3, 2026 (PDT). Status: in progress; research only.

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

## Pair running

Artifact directory: `preflop-solver/neural/runs/local-native-replay875-20261003-a/`.
Seeds 10601/10602, wide architecture, exact-runout feature schema, original
objective/optimizer, 600-step cap, and original family split are fixed.
Only sampling changes. No warm-start and no new native labels.

After fitting and parity verification, test the new policies first on
three-bet-high-rainbow and three-bet-monotone, both solver seeds, at 128 flop
updates with native64 continuations and all 49 turns. Compare against both
the retained policy and matched native64 students. Advance, reject, or mark
inconclusive under the predeclared rules before any expensive root expansion.

## Checks so far

`python -m unittest test_native_replay test_train_public_value_network
test_native_value_dataset test_mixed_native_reference test_decision_pilot_screen
test_student_value_pilot`: **74 tests passed**. Existing constant-vector
correlation fixtures emit NumPy warnings; no tests failed.

`git diff --check` passed. Free disk was 124GiB and macOS memory-pressure level
was normal at launch. There are no frontend or Rust behavior changes, so an
app build and native rebuild are not required for this milestone.
