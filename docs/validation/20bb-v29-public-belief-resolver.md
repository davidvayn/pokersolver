# 20bb v29 public-belief resolver sequence

## Decision

**Rejected for activation.** No model manifest, browser policy, or hosted policy
was changed. The sequence produced a useful exact river solver and showed that
range-aware depth-limited resolving is materially better than a range-blind
control, but the turn value network does not generalize and the experimental
preflop bridge overfits its two-deal corpus.

The fail-closed result is summarized in
`preflop-solver/neural/20bb-v29-research-candidate.json`. Large generated
targets and weights remain under the ignored `neural/runs/` workspace.

## Architecture implemented

1. Neural external sampling retains unbiased action-dependent baseline
   correction and can now enumerate all 44 legal river cards at a
   turn-to-river transition with `--enumerate-turn-river-chance`.
2. `public_belief.rs` represents both players with exact 1,326-combination
   range vectors and removes board and opponent blockers exactly.
3. River subgames use alternating vectorized DCFR. There is no sampled private
   deal or future chance card. The evaluator computes an exact best response
   inside the configured river betting abstraction.
4. Turn targets enumerate every public river card and consume solved river
   per-hand counterfactual values. Their Monte Carlo target SE is zero.
5. The paired MLX trainer emits full 2,652-value vectors, uses a separate exact
   range tower, splits by public state, and compares range/no-range seeds on
   identical holdouts. Exact suit permutations are available as a grouped
   ablation and cannot cross the state split.
6. The flop pilot enumerates every turn card, projects learned leaf values to
   zero sum, runs alternating DCFR, and compares the resolved policy with an
   unresolved uniform control. It refuses to silently omit all-in: callers
   must explicitly use the research-only no-all-in tree, which keeps release
   validation rejected.
7. The preflop bridge reconstructs a zero-sum exact-deal utility from both
   players' conditional flop CFVs and carries measured network uncertainty
   into every continuation. Leaf jobs run in parallel.
8. `validate_v29_public_belief.py` composes river, target, value, resolver,
   preflop, and full-game learned-response evidence. A learned response remains
   a lower-bound red team and cannot satisfy the missing exploitability
   upper-bound gate.

This is a small DeepStack/ReBeL-style research path, not a claim that those
systems were reproduced. The relevant design basis is
[DeepStack](https://www.science.org/doi/10.1126/science.aam6960),
[ReBeL](https://arxiv.org/abs/2007.13544), and
[VR-MCCFR](https://arxiv.org/abs/1809.03057). CFR equilibrium reasoning still
comes from [CFR](https://papers.nips.cc/paper_files/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html),
[MCCFR](https://papers.nips.cc/paper_files/paper/2009/hash/00411460f7c92d2124a67ea0f4cb5f85-Abstract.html),
and [DCFR](https://ojs.aaai.org/index.php/AAAI/article/view/4007).

## Results

### Exact river solver

On `2c 7d Th Js Ac`, a 4bb pot, and uniform legal ranges:

| Iterations | Exact abstract exploitability |
| ---: | ---: |
| 20 | 0.387835bb/hand |
| 200 | 0.009319bb/hand |
| 500 | 0.002872bb/hand |
| 2,000 | 0.000445bb/hand |

At 200 iterations the zero-sum residual was `3.33e-16` and maximum probability
sum error was `3.33e-16`. Two independent command invocations produced the
same SHA-256:
`280872edb7ec2be7c9662b4c1ad63aba80647b466107057f6621a422988cb008`.

The eight-state turn corpus used 200 iterations for each of 384 river solves.
Its worst river exploitability was `0.029345bb/hand`; worst turn target
zero-sum residual was `7.77e-9bb`.

### Public-belief distributions

- Synthetic range-diversity pilot: eight states, rejected because it was not
  drawn from frozen-policy self play.
- Authentic frozen-v26 self-play pilot: two states, 256 likelihood-weighted
  particles per state, minimum effective sample size `87.84`, and maximum
  river exploitability `0.021002bb/hand`. It is rejected because release needs
  far denser particles plus an independent belief replicate.

### Paired turn value networks

The best eight-state unaugmented pair measured:

| Metric | Range | No range |
| --- | ---: | ---: |
| Mean weighted holdout RMSE | 1.963927bb | 2.300256bb |
| Relative range improvement | 14.62% | — |
| Range cross-seed prediction correlation | 0.882133 | — |

Exact ranges are useful, but neither the `0.25bb` research RMSE gate nor the
`0.95` seed-correlation gate passed. Twenty-four exact suit permutations per
state raised range seed correlation to `0.951760` but worsened RMSE to
`3.280167bb`; eight distinct rank/board textures are the bottleneck.

### Flop resolver ablation

On `2c 7d Th`, a 4bb pot, 20 resolver iterations, and the same learned leaf
family:

| Variant | Depth-limited exploitability | Unresolved control | Improvement |
| --- | ---: | ---: | ---: |
| Exact-range network | 0.014052bb/hand | 1.192320bb/hand | 98.82% |
| Range-blind network | 0.254136bb/hand | 1.264124bb/hand | 79.90% |

This passes the paired research ablation, not release. The source network is
rejected and the pilot has no flop all-in branch. Learned leaves needed as much
as `8.594166bb` of zero-sum correction, further evidence of network error. The
exported range-aware root CFVs have a `2.78e-17bb` zero-sum residual after the
explicit projection; that accounting correction does not make the value model
accurate.

### Resolver-derived preflop solve

The bridge intentionally retained the turn network RMSE (`1.932344bb`) as the
uncertainty on every continuation. Consequently, zero of 98 leaf values passed
the `0.02bb` precision threshold.

Two one-million-iteration DCFR seeds looked stable on the two-deal research
game:

- action-frequency MAE `0.012786`
- primary action agreement `0.999975`
- lookup intersection `1.0`
- in-corpus exploitability about `0.00003bb/hand`

The independent H corpus rejected both seeds:

| Seed | H exploitability | H lookup coverage |
| ---: | ---: | ---: |
| 9801 | 2.579659bb/hand | 0.8247% |
| 9802 | 2.579502bb/hand | 0.8296% |

This is direct evidence that the apparent in-corpus convergence and cross-seed
agreement were overfit, not equilibrium proof.

### Independent full-game learned response

The fresh v26 red-team run used 5,000 training and 5,000 evaluation deals. Its
approximate exploitability lower bound was `0.001bb/hand`; the approximate 99%
lower confidence bound was zero. Coverage was insufficient: overall lookup was
about `0.54%` for player zero and `0.12%` for player one, with essentially zero
postflop coverage. It does not find a supported leak, but it cannot certify the
model.

## What remains before a long run

1. Generate at least hundreds of distinct frozen-policy self-play public
   states for a structural pilot, then scale to the much larger corpus needed
   by a full-vector network. Track independent particle-replicate range error.
2. Add exact vectorized flop all-in equity/runout leaves and restore the full
   serving action abstraction.
3. Train on authentic public beliefs with grouped board holdouts. Require the
   range/no-range advantage, absolute RMSE, and cross-seed gates together.
4. Rebuild a balanced multi-cycle continuation corpus. A two-deal bridge is
   only an interface test.
5. Improve full-game LBR/learned-response postflop coverage and add a genuinely
   independent exploitability upper-bound evaluator. Until then activation is
   impossible even if all lower-bound red teams are quiet.

## Reproduction

```sh
cd preflop-solver
cargo test --release

target/release/preflop-solver river-pbs-solve \
  --board 2c,7d,Th,Js,Ac --pot-bb 4 \
  --iterations 200 --averaging-delay 20 --output river.json

target/release/preflop-solver turn-pbs-self-play-targets \
  --states 2 --range-particles 256 --river-iterations 200 --threads 8 \
  --networks neural/runs/20bb-v26-routed-seed5101.json \
  --output turn-self-play.json

.venv-neural/bin/python neural/train_public_value_network.py \
  --dataset turn-self-play.json --output-dir turn-value \
  --steps 3000 --seeds 9701,9702
```
