# Frozen TRAIN objective diagnostics

October 5, 2026 (PDT). Research only; serving weights remain unchanged.

The function-preserving pooled pilot improved held-out value regression but
failed actual-policy screening. Before another fit, two small diagnostics
tested specific explanations. Neither supported its proposed intervention.

## Gradient interference: projection not supported

Both frozen serving-aligned seeds were inspected at steps 0/200/400/600.
The same 615-state primary corpus, 474-state TRAIN split, six native bundles
and 366 original positive-joint TRAIN states were pinned. No fitting, new
labels or evaluation scores were used. The combined active-bundle learning
gradient includes primary calibration, mean bundle calibration and weighted
contrast, but excludes the separate soft-retention term.

All eight combined cosines with original authentic calibration are positive
(0.315–0.965). None passes the predeclared <-0.2 conflict criterion. Original
authentic loss fell 45.2%/40.2%, rather than worsening >=5% in both seeds.
Do not implement gradient projection from this evidence. Individual contrast
directions sometimes conflict, and these aggregate snapshots do not exclude
regional/stochastic interference. Raw-gradient alignment is not an AdamW-step
or poker-performance guarantee.

The probe took 112.7s, with 3.840GiB sampled worker footprint, under its 6GiB
guard. [A-GEM](https://arxiv.org/abs/1812.00420) and
[PCGrad](https://arxiv.org/abs/2001.06782) motivated testing interference;
neither establishes safety for this resolver.

## Model-induced beliefs: aggregation not supported

All three pre-registered TRAIN families 100/101/102 were used, without choosing
from scores. Each generated 16 native64 labels from stratified early, middle,
late and final-average beliefs under the corresponding current aligned model.
Roots, trunk/sampling seeds, 128 flop updates and label count remained matched
to the older proposer. Both old and new captures were evaluated with the same
current model, actual native inference and independent NumPy parity.

| TRAIN root | Old-belief authentic RMSE | Current-belief authentic RMSE | Change |
| --- | ---: | ---: | ---: |
| 100 | 0.768234bb | 0.781664bb | +1.7% |
| 101 | 1.030280bb | 1.037920bb | +0.7% |
| 102 | 0.732244bb | 0.723992bb | -1.1% |

Zero families reach the predeclared >=10% increase; two were required.
**Do not launch a dataset-aggregation fit from this screen.** Four samples per
band are a small diagnostic, not a comprehensive coverage or action-ranking
test. [DAgger](https://arxiv.org/abs/1011.0686) motivates learner-induced state
labeling; its sequential supervised-learning guarantees do not transfer to
adversarial poker or unsafe re-solving.

Current native label mean response residuals are 0.005367/0.004471/0.007465bb;
maximum residuals are 0.027657/0.032797/0.038312bb. These finite-budget labels
have far smaller residual response gains than the model's value RMSE, but
the metrics are different; subtracting them would be invalid. All six
NumPy/native comparisons passed, maximum error 0.000005173bb. New captures
verified policy-observation parity on every family. The old capture protocol
verified that separately only on its first family; it is not retroactively
claimed for the others.

The 48-label probe took 147.2s. Two native workers, separate 2GiB native/6GiB
feature-analysis guards, pressure stops and 20GiB disk reserve stayed active.
No new fit, policy promotion or full-game exploitability improvement occurred.

Artifacts under `preflop-solver/neural/runs/`:

- `local-training-gradient-conflict-20261005-a/manifest.json`:
  `65e3a833cdb2e3ae6c0b19b6e81f4b3a89112da9dcd93c0be13a635568f9d616`.
- Its `analysis.json`:
  `398170eb277af8789d6c733ac53c4a44aaff260c3ce690274515f8948b77ac07`.
- `local-on-policy-calibration-probe-20261005-a/manifest.json`:
  `290b7c7ec767c0edbec31563c2f873cc5069e34d7adf4e5ebf4e0cd48900c121`.

Next: inspect how the existing pairwise contrast objective allocates gradient
between correct and wrong action orderings on frozen TRAIN data. Retain
native-value calibration and the rejecting actual-policy screen; neither
supervised ranking nor value RMSE can substitute for exploitability.
