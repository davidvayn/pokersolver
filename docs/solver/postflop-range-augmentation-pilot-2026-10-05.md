# Function-preserving range augmentation pilot

October 5, 2026 (PDT). Completed; rejected. No serving/model activation.

The opt-in wide-to-wide-pooled transfer adds own/opponent learned range inputs
without changing the initial function. Both encoder towers and the final head
are copied exactly; the expanded first head copies context/query columns and
zeros its new 128 pooled columns. Source v4 wide/v3/20bb metadata, dimensions,
seed and finite parameters are validated before mutation. Unknown transfers
and unsupported destinations fail closed. Tests confirm nonzero gradients
into the initially zero columns and active pooled NumPy/MLX serving parity.

All-615 initial NumPy predictions matched the source exactly on both seeds;
actual native v5 maximum errors were 0.000005937bb/0.000005602bb. The largest
TRAIN-family preflight projected 994 seconds, under the two-hour cap.

Only representation changed from the completed serving-aligned control.
Retained initial weights, fresh optimizer, two seeds, frozen split, six native
bundles, 600 fixed steps, full float32, cadence/chunk 4, contrast coefficient
1.883306130920223 and retention coefficient 0.42899030580264763 stayed fixed.
Reference predictions and anchor draws still come from the unchanged wide
model. Fit plus both preflights/parity took 377.9 seconds with 4.074GiB peak
fit footprint. Finished all-615 native errors were
0.000006195bb/0.000005768bb. Holdout RMSE improved from
1.205253bb/1.259385bb to 1.164554bb/1.220335bb.

The first completed policy control nevertheless worsened:

| Model | High-rainbow seed-100101 conditional response gain |
| --- | ---: |
| Retained | 0.278335bb |
| Serving-aligned matched control | 0.343412bb |
| Function-preserving range augmentation | 0.352513bb |

The control regressed retained by 0.074178bb, over the unchanged 0.01bb
limit. The 141-second response screen rejected it and automatically stopped.
Remaining cases and paired means are unmeasured. No full-game exploitability
improvement is established by this conditional-root test.

Cached own-policy diagnosis took 3.69 seconds, with zero new native solves.
After BB checks, native local deviation loss is 0.560726bb versus 0.135293bb
predicted, ranking loss 0.265914bb and contrast RMSE 0.736777bb. All-in-only
branches still match exactly. These overlapping local errors are not additive
exploitability. Better held-out regression did not fix action-relevant errors.
The TRAIN bundles include four three-bet and two limped roots, so simple
absence of three-bet data is not the explanation; six families remain narrow.

Artifacts under `preflop-solver/neural/runs/`:

- Fit `local-pooled-transfer-students-20261005-a/manifest.json`:
  `a259cae81bec509372da18fc58a9f421c6ecc9c9955a1e3fc32f2b34b00c4a2a`.
- Pair `local-pooled-transfer-students-20261005-a/C1/manifest.json`:
  `624c1fe1dab81cae3ad65a67583b818a4aa5a1bd46752fd3224cb1831e543027`.
- Response `local-pooled-transfer-response-20261005-a/manifest.json`:
  `cc3f7236e2cd0da8cc290c3d45e38423a7ba837937bdcad2a089c8bfd5b469d1`.
- Cached diagnosis `local-pooled-transfer-own-policy-diagnosis-20261005-a/manifest.json`:
  `5a3939cbb2777cf835a99470f3bcb2ee2850cb15ccc4d34b4075758347918d52`.

77 targeted Python tests passed, including existing trainer paths. No native
or browser source changed; the qualified binary independently verified both
initial and finished exports and supplied the audited frozen response.

Research: [Net2Net](https://arxiv.org/abs/1511.05641) motivates preserving a
starting function when expanding a model; [Deep Sets](https://arxiv.org/abs/1703.06114)
motivates learned set aggregation. Neither predicts this poker pilot's outcome
or provides an equilibrium guarantee. Reject this arm and diagnose TRAIN-only
gradient interference before another fit; do not scale an unchanged failure.
