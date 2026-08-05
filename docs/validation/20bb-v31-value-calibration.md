# 20bb v31 value-calibration sequence

## Decision

**Rejected for activation.** The v31 pilot improves turn-value calibration but
does not meet the predeclared medium-pot, absolute holdout, or downstream
resolver gates. The active manifest and served model remain unchanged.

The 512-state corpus and exact low-SPR/all-in hybrid were conditional steps.
They were deliberately not run after their prerequisites failed. This is a
fail-closed sequencing decision, not a missing measurement represented as a
success.

Large weights and resolver reports remain in ignored
`preflop-solver/neural/runs/v31-calibration/` paths. The checked-in candidate
summary retains the measurements and SHA-256 identifiers needed to audit the
decision.

## Implemented calibration architecture

The v4 shared-combo value network replaces the fixed 5bb bounded residual with
a linear residual expressed in a state-dependent unit. The selected `pot`
normalization uses total invested chips, with a 1bb floor. The independent
`payoff-exposure` control uses the maximum legally exposed stack amount.
Runtime inference reconstructs exact big-blind values before the existing
reach-weighted zero-sum projection.

Training now combines:

- equal loss mass for every public state while preserving exact within-state
  combo reach;
- batches stratified across every available small, medium, and large pot band;
- a robust Huber loss in normalized units;
- a depth-normalized raw-big-blind auxiliary loss; and
- an unbounded linear final residual head.

The exported schema is `hu-public-belief-combo-value-network-v4`. Rust accepts
both v3 and v4 artifacts, validates the new normalization field, and applies
the same dynamic scale as Python. Historical v3 behavior remains unchanged.

Normalization and seed selection use only the disjoint tuning split. Untouched
holdout results are evaluation data, never selection data.

## Paired 128-state results

Pot normalization won the fixed tuning comparison:

| Normalization | Paired mean tuning RMSE |
| --- | ---: |
| Pot | 0.354372bb |
| Payoff exposure | 0.413391bb |

Seed 10902 was selected within the pot pair with tuning RMSE `0.353724bb`.
Its exported weights have SHA-256
`2ec8e8d2af6704e5e53f98a182b19d5cb1836137a6dff3b34cfff94b7b569b52`.

On 32 untouched states from the newly generated half of the 128-state corpus,
the paired pot models measured `0.540214bb` mean range RMSE versus
`1.132942bb` without ranges, a 52.32% relative improvement. Cross-seed
prediction correlation was `0.997032`. The absolute `0.25bb` holdout gate
still failed.

### Matched-holdout correction

The original table below copied v30 band metrics from v30's own report. Because
the historical split was tied to each run's model seed, those v30 states were
not identical to v31's holdout. Those cross-report percentages therefore are
retained only as the original sequencing record and must not be used as a
comparative claim.

A subsequent audit evaluated the tuning-selected v30 and v31 artifacts on the
same 32 v31 states. It measured `0.615419bb` for v30 and `0.548175bb` for v31,
an overall 10.93% improvement. The matched band results were:

| Pot band | v30 selected seed | v31 selected seed | Improvement |
| --- | ---: | ---: | ---: |
| Small | 0.265911bb | 0.218075bb | 17.99% |
| Medium | 0.907035bb | 0.818053bb | 9.81% |
| Large | 1.766130bb | 1.593880bb | 9.75% |

This correction strengthens the rejection: none of the matched bands reaches
the original 25% calibration target. New training now pins `--split-seed`
independently from model initialization so this ambiguity cannot recur.

Original, unmatched cross-report sequencing table:

| Pot band | v30 RMSE | v31 paired mean RMSE | Relative improvement | Gate |
| --- | ---: | ---: | ---: | --- |
| Small, at most 3.5bb each | 0.274315bb | 0.215392bb | 21.48% | Pass: no regression |
| Medium, 4–7.5bb each | 0.949079bb | 0.803120bb | 15.38% | Fail: required 25% |
| Large, at least 10.5bb each | 2.115362bb | 1.573794bb | 25.60% | Pass |

Independent Python and Rust inference matched across all 32 untouched states
with maximum absolute error `7.8933e-6bb`, passing the `0.0001bb` parity gate.

## Increased resolver iterations and board coverage

The matched resolver audit used a 4bb flop pot, 100 iterations, averaging after
iteration 10, and the same three predeclared board textures for v30 and v31.

| Texture | Board | v30 exploitability | v31 exploitability | Result |
| --- | --- | ---: | ---: | --- |
| Dry low | 2c 7d Th | 0.572929bb/hand | 0.540310bb/hand | v31 5.69% better |
| Dry high | As Kd 7c | 0.548429bb/hand | 0.900115bb/hand | v31 64.13% worse |
| Monotone | 9h Th Jh | 0.686771bb/hand | 0.694163bb/hand | v31 1.08% worse |
| Mean | — | 0.602710bb/hand | 0.711529bb/hand | v31 18.06% worse |

On the dry-low board, increasing v31 from 20 to 100 iterations improved its
result from `0.697589` to `0.540310bb/hand`. The wider audit shows that this is
not a general fix: aggregate v31 resolving regresses and remains far above the
`0.05bb/hand` research gate. The current pilot abstraction also intentionally
omits exact all-in branches.

## Independent adversarial evaluation

A fresh learned-response probe evaluated the pinned frozen v26 routed policy,
whose SHA-256 exactly matches the v31 source-policy identifier. It used 2,000
response-training deals and 4,000 independent evaluation deals. The estimated
exploitability lower bound was `0.00475bb/hand`; its approximate 99% lower
confidence bound was zero.

This result is inconclusive. Minimum resolver lookup coverage was only
`0.000883`, postflop coverage was zero, and only two information sets were
confident. A learned response can reveal a leak but cannot certify its absence.
It is never the independent one-sided 99% exploitability upper bound required
for release.

## Reproduction

From `preflop-solver`:

```sh
.venv-neural/bin/python neural/train_public_value_network.py \
  --dataset neural/runs/v30-public-belief/turn-targets-self-play-exact-128-seed10701.json \
  --output-dir neural/runs/v31-calibration/pot-128-paired \
  --architecture compact --value-normalization pot \
  --steps 3000 --batch-size 6 --seeds 10901,10902 \
  --holdout-start-index 64

.venv-neural/bin/python neural/train_public_value_network.py \
  --dataset neural/runs/v30-public-belief/turn-targets-self-play-exact-128-seed10701.json \
  --output-dir neural/runs/v31-calibration/payoff-128-paired \
  --architecture compact --value-normalization payoff-exposure \
  --steps 3000 --batch-size 6 --seeds 10901,10902 \
  --holdout-start-index 64

.venv-neural/bin/python neural/validate_public_value_parity.py \
  --dataset neural/runs/v30-public-belief/turn-targets-self-play-exact-128-seed10701.json \
  --model neural/runs/v31-calibration/pot-128-paired/turn-value-range-seed10902.json \
  --state-indices 65,69,70,72,73,75,76,77,81,84,85,86,88,91,92,94,96,99,104,107,109,110,111,112,116,117,118,121,122,124,125,126 \
  --solver target/release/preflop-solver \
  --output neural/runs/v31-calibration/pot-128-paired/parity-holdout.json
```

The fail-closed composition is implemented in
`neural/validate_v31_calibration.py`. It checks paired tuning selection,
pot-band deltas, parity, matched resolver inputs, the independent response
artifact, conditional sequencing, and the still-missing release upper bound.

## Next research boundary

Generic corpus scaling is not the next justified action. The next pilot should
target board-conditional value generalization and increase the adversarial
evaluator's postflop coverage on frozen artifacts. Only a model that improves
both held-out value bands and matched multi-board resolving should unlock the
512-state corpus; exact low-SPR/all-in implementation remains one gate later.
