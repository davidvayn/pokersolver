# Retained initialization and numerical precision pilots

Research only. No serving weights, game abstraction or website behavior changed.
This implements Section 12 of [the decision plan](postflop-decision-pilot-plan.md).

## Initialization-only comparison: completed, rejected

Reuse the prior immutable six-family scratch C1 pair rather than fit it again.
Warm-start seeds 10601/10602 from their corresponding retained benchmark
networks. Keep all 615 primary states, frozen 474/69/72 split, six bounded TRAIN
bundles, 600 fixed steps, cadence 4, 25 bundle updates/family/seed, optimizer,
learning-rate schedule and contrast coefficient unchanged. The optimizer is
fresh: this is weight transfer, not resumed optimizer or CFR state.

The importer rejects incorrect schema, feature/value contracts, seeds, layer
shapes, activations and nonfinite values before mutating any weights. Its
CPU full-float32/independent-NumPy comparison checks both players and every
private combination on two frozen TRAIN states before fitting. Finished
students additionally pass independent NumPy/native comparison on all 615
states. Retained input networks are research-only, not accepted GTO policies.

Two preflight attempts stopped before training. They exposed a comparison of
different train/serving wrappers and MLX's reduced-precision default matmul.
The final import check uses the same bounded serving wrapper on CPU; the
training device/precision remains unchanged for this initialization-only arm.
No failed preflight was scored as an experiment.

| Measure | Seed 10601 | Seed 10602 |
| --- | ---: | ---: |
| Selected final step | 600 | 600 |
| Authentic holdout value RMSE | 1.197413bb | 1.256860bb |
| Maximum full-615-state NumPy/native difference | 0.000005824bb | 0.000005204bb |

Fit plus parity took 239.8 seconds, fitting worker 196.9 seconds; peak sampled
footprint was 3.57GiB. Guards did not stop the successful stage.

Actual policy: 128 flop updates, native64 continuations, all 49 turns, two native
workers and an independent JavaScript backup audit. First consumed control:

| High-rainbow, solver seed 100101 | Conditional response gain |
| --- | ---: |
| Retained | 0.278335bb |
| Frozen scratch six-family C1 | 0.350730bb |
| Retained-initialized C1 | 0.359659bb |

Warm-start regresses versus retained by 0.081324bb, over the fixed 0.01bb limit.
**Reject and stop automatically after the first completed audited failure.**
Other cases and paired means remain unmeasured. Better surrogate value RMSE
did not establish improved play or any full-game exploitability reduction.

Immutable local artifacts, under `preflop-solver/neural/`:

- Fit: `runs/local-retained-contrast-students-20261004-c/manifest.json`,
  SHA-256 `1b674c8c14d577800b0e8de994bfd025ca2b7bf27aced6e4034dc7f3af5b7bb6`.
- Exported pair: same stage's `C1/manifest.json`,
  `f1de58d00966320a827be621a582e36c8044131ac8216efe9a92700cbd7d3093`.
- Rejected response: `runs/local-retained-contrast-response-20261004-c/manifest.json`,
  `d63c86c2e1a039c55cd34a1c04ff032c055bd9295898127ebec074bf4c503616`.

## Isolated full-float32 comparison: running

The retained network on cached state 0 differs from independent NumPy by
0.015009bb under default MLX GPU inference. Errors exist before projection:
context embedding 0.0002383, query embedding 0.0007181 and residual head
0.0008790 (normalized units). This observation does not explain the entire
policy regression: that input has zero joint reach and is not an actual-policy
control. It does expose a real arithmetic mismatch worth isolating.

[MLX documents reduced-precision float32 matrix multiplication](https://ml-explore.github.io/mlx/build/html/usage/precision.html)
and the launch-time `MLX_ENABLE_TF32=0` switch. The next pair changes only that
training arithmetic relative to the completed warm pair. Inputs, retained
weights, loss, coefficient, batch draws, schedule and checkpoint stay frozen.
Its imported GPU prediction must match NumPy before fitting; no tolerance is
relaxed. Worker environment records the switch before MLX imports. The same
full-615 export verification and automatic control-first rejection apply.

If precision alone fails, proceed to a TRAIN-only frozen-output retention
constraint, then representation/target diagnosis. [Learning without Forgetting](https://arxiv.org/abs/1606.09282)
supports preserving prior outputs as a regularizer; it does not make those
outputs ground truth or provide poker re-solving safety. The prior
sampling-only replay experiment already failed and is not being repeated.

## Verification so far

63 targeted Python tests pass, including exact weight/forward round trips,
fail-before-mutation cases, clipping-aware native wrapper parity, frozen fit
settings, original trainer paths, action-bundle gradients, coverage and early
rejection. Existing constant-vector correlation fixtures emit expected NumPy
warnings. `git diff --check` passes. Native and browser code are unchanged;
native inference and policy evaluation use the hash-qualified existing binary.
