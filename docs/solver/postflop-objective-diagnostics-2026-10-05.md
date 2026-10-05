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

## Pairwise allocation: ranking fit not supported

The exact Huber derivative was decomposed into correctly ordered, inverted
and near-tie pairs, and reconstructed to <=1e-12 before the same affine and
bounded-serving VJP. Both frozen seeds were measured at initialization and
step 600 on all twelve TRAIN decisions. Equal-group native ranking loss
fell from 0.215331/0.198206bb to 0.063204/0.069243bb.

Correct-order pairs receive 75.54%/73.62% of the endpoint pairwise derivative
magnitude. The requiring-both-seeds >=75% screen fails; do not lower it after
seeing the numbers. A capped-margin ranking direction has parameter-gradient
cosine -0.052/-0.068 with the existing contrast direction, but that does not
establish improvement. No ranking fit was launched. These derivative shares
are before the affine/network Jacobians, not parameter-gradient norms.

The allocation manifest SHA-256 is
`1e500e62729774b26c2dc2f7a46673518e1655f5af35cc1cb4b72509f62e4910`,
under `local-decision-gradient-allocation-20261005-a/`. Five new unit tests
cover decomposition, near ties, actual finite differences, offset invariance
and fail-closed screening.

Native inference was also inspected for reach-scale invariance. It already
normalizes both raw ranges before feature construction; the qualified binary's
`native_value_contract_preserves_future_bets_and_zero_own_reach` test passed.
Do not add redundant normalization or scale augmentation.

[Exploitability Descent](https://arxiv.org/abs/1903.05614) requires policy
optimization against best responses, unlike this supervised ranking proposal.
[Value Functions for Depth-Limited Solving](https://arxiv.org/abs/1906.06412)
reports limited benefit from its explored alternative losses. Our next
intervention addresses the known accurate-native-leaf computation cost with
a matched smaller-inner-budget comparison, retaining native64 evaluation.
No full-game improvement or serving qualification follows from these probes.

## Native construction cost and allocation overhead

The first native4/32 construction stopped at 43.943s under its original
2.5GiB physical-footprint guard. A separate 4GiB-only retry completed in
205.690s, versus the pinned native64/32 control's 1956.564s (9.51x faster).
Sampled peak footprint was 3,772,042,816 bytes. The same board, ranges, game,
seed, 32 flop updates and one turn proposal per iteration were retained.
This is construction speed, not playing strength. The projected complete
two-root/two-seed response stage was 13,842.9s, exceeding the declared 7200s
cap. No quality score or paired mean was produced by that cost pilot.

A live memory snapshot showed large unused allocator regions. A three-second
CPU sample also found policy serialization/hashing among the repeated work.
Native policy hashing now streams the same JSON bytes through a 64KiB buffer,
rather than materializing a complete second serialized policy. No regret,
average, f32 probability or value calculation changes. A serial frozen
native64 turn replay matched the cached packet byte-for-byte before and
after the change. Sampled footprint fell from 621,052,984 to 335,709,000 bytes
(46%); time was 54.473/53.959s, not a meaningful measured speed gain. One
packet does not establish the savings on every root or long construction.

The first staged quality launch failed before solving because Python JSON
formatting changed the candidate byte identity. Probabilities had not changed,
but the native reader correctly refused the noncanonical payload. A guarded
Rust exporter now changes only the explicit response budget, verifies both
canonical round trips, and restores the original training bytes exactly when
that field is removed. The regression test rejects pretty JSON, wrong source
hashes, non-stronger budgets and excessive budgets. The identity check was
not relaxed. The raw candidate is still immutable; a new canonical evaluation
candidate uses native64 played continuations.

A staged first-control evaluation is running across all49 turns with the
independent JavaScript audit and unchanged cost/quality rejection rules.
Its runner supports identity-matching interrupted-job recovery. Passing one
control cannot qualify the other seed/root or full-game exploitability.

Artifacts under `preflop-solver/neural/runs/`:

- Completed cost retry `local-native-inner-budget-20261005-b/manifest.json`:
  `7d5fe82701affb37b180597e314ef034021288ce68ed7db22c16d6843860850c`.
- Immutable native4 training candidate:
  `ea79c625efad2e21cc0bb15cd5b96a55828489df135776026ae077c3fcf57ebb`.
- First streaming-hash packet preflight manifest:
  `ce24be53c932158483a5b83c689e59353a6c62730e9aee846e56b5fd7ad0b83c`.
- All three cached/buffered/streamed control packets:
  `25a6bd363f1edd8b076919299287827dca8f525ff407c9797e1ecd3ba39ff0cd`.
- Canonical native4-training/native64-evaluation candidate:
  `bf22f7963d582cb8d00add6eed97b2e3d75e03f363f3d48efdc39cae92bb7723`.
- Rebuilt canonical-export binary's identical packet proof:
  `7371d6593ce2c43be2a9edfe88abd42f5728a8be101d866c62e3b33f18efd235`,
  under `local-native-streaming-hash-20261005-a/canonical-packet-preflight/`.

Verification before the export addition: `cargo test --release --
--test-threads=1` passed 355 unit and nine CLI tests, with 56 explicit research
jobs skipped. The new canonical-export test and 46 targeted Python tests also
passed. The expanded native continuation suite passed all40 runnable tests,
with 13 explicit research jobs skipped.
