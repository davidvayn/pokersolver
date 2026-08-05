# 20bb v30 exact-belief shared-combo pilot

## Decision

**Rejected for activation unless every gate below is later measured as passed.**
This sequence is a research candidate layered on frozen v26. It does not modify
the active manifest, browser policy, or hosted model. In particular, a good
turn-value holdout score is not a full-game exploitability certificate.

Large targets and weights remain in ignored `preflop-solver/neural/runs/` paths.
The checked-in candidate summary and this document contain the reproducible
measurements needed to explain why activation remained blocked.

## Corrections and architecture

### Exact public beliefs

The v29 particle sampler accumulated each sampled deal's joint action-line
likelihood into both players' hand marginals. That object is not what the
public-belief CFR solver consumes: the solver expects an independent reach
factor for each player and applies exact hand compatibility itself. Feeding it
posterior marginals applies blocker effects twice.

v30 enumerates all 1,128 board-legal exact combinations for each player. It
replays the public line and multiplies only actions taken by that player. This
is exact for the frozen policy because its information-state encoder uses the
acting player's cards and public state, never the opponent's cards or an
unseen river. The product of the two reach factors, masked by exact card
compatibility, is the correct joint posterior represented by the solver.

Two independently rotated, stratified 4,096-particle estimates are retained as
diagnostics. They are compared with the exact reach factors and with one
another; they are no longer training truth. Each solved state is written
atomically to a SHA-256-fingerprinted checkpoint before the full corpus is
assembled.

### Shared combo value network

The v29 network used a 2,652-coordinate dense output layer. It had to relearn
the same poker relationship separately for every exact hand, was not
structurally suit equivariant, and exported a 37MB JSON model in the pilot.

v30 evaluates every `(target player, exact combo)` with the same query tower.
Its features are invariant to all 24 global suit permutations by construction:

- board rank counts and sorted suit counts;
- exact hand ranks, board-to-hole-suit relationships, made-hand category,
  strength percentile, and exact one-card category potential;
- relative actor and investment features;
- own/opponent 169-class range summaries;
- local exact-combo, card, rank, suit, and blocker-compatible range masses;
- range-conditioned current-showdown equity.

The value is represented as a poker baseline plus a bounded learned residual.
The exact-range variant uses range-conditioned current-showdown equity; the
range-blind control uses structural hand-strength percentile. An explicit
reach-weighted projection makes the two exported CFV vectors zero sum. The
compact model is about 684KB as JSON, roughly 54 times smaller than the v29
37MB pilot model. A wider model remains available as an accuracy ablation.

Training uses disjoint train, tuning, and untouched grouped public-state
holdouts. The tuning set selects an early-stopping checkpoint; reported gates
are measured only on the untouched holdout. Exact suit augmentation defaults
to one because all 24 permutations are identical observations under the new
feature construction.

The architecture-selection rule was fixed before the authentic corpus
finished: select the wide network only if its mean range-variant tuning RMSE is
at least 5% lower than the compact network's; otherwise select compact for its
smaller serving and resolver cost. Within the selected architecture, choose
the seed with the lower tuning RMSE. Untouched-holdout RMSE is never used to
select either architecture or seed.

### Runtime parity and resolver performance

Rust supports both the legacy v2 full-vector schema and the v3 shared-combo
schema. `turn-pbs-value-predict` exposes a deterministic parity surface. The
Python verifier reconstructs exported dense inference independently and
compares all 2,652 values with Rust.

Static poker features are cached by turn board with a bounded 128-board cache.
Flop leaf evaluation parallelizes the 49 public turn cards while the preflop
batch bridge keeps one inner resolver thread to avoid nested oversubscription.
On the cold `2c 7d Th` two-iteration pilot, ten-thread leaf evaluation reduced
wall time from 142.63s to 27.67s and reproduced the single-thread CFVs and
metrics to floating-point precision.

## Pilot gates

The fail-closed composer in `validate_v30_public_belief.py` requires all of the
following research evidence:

1. at least 64 authentic frozen-policy turn states;
2. exact reach-factor beliefs plus two 4,096-particle diagnostics;
3. maximum exact-combo belief total variation at most `0.15`;
4. every river abstraction at most `0.05bb/hand` exploitability;
5. turn target zero-sum residual at most `1e-7bb`;
6. range-network untouched-holdout RMSE at most `0.25bb`;
7. at least 2% improvement over the range-blind control;
8. range-network cross-seed prediction correlation at least `0.95`;
9. Python/Rust inference maximum error at most `0.0001bb`;
10. paired flop resolver evidence with exact all-in branches; and
11. an independent full-game one-sided 99% exploitability upper bound at most
    `0.10bb/hand`.

Items 10 and 11 remain activation blockers even if the value pilot passes.
The learned leaf game's internal best response is only an evaluation of that
approximate depth-limited game.

## Interim structural pilot

On the earlier eight-state synthetic stress corpus, the flat v29 range model
measured `1.963927bb` holdout RMSE and `0.882133` cross-seed correlation. The
first shared head without poker features was worse (`3.512415bb`) but showed an
8.44% range advantage. Adding exact poker features improved RMSE to
`2.360686bb`, the range advantage to 21.88%, and correlation to `0.984483`.

Adding the explicit poker baseline and zero-sum projection reduced matched
wide-model RMSE to approximately `0.89bb`. The compact model measured
approximately `0.91bb` on the same split while reducing export size and
resolver arithmetic substantially. These are useful architecture ablations,
not acceptance evidence: the source corpus is synthetic and contains only
eight turn boards.

The first exported v3 parity smoke test measured `1.43198e-6bb` maximum error
between independent Python and Rust inference, passing the `0.0001bb` gate.

## Reproduction

```sh
cd preflop-solver

cargo build --release

target/release/preflop-solver turn-pbs-self-play-targets \
  --networks neural/runs/20bb-v26-routed-seed5101.json \
  --states 128 --range-particles 4096 --belief-replicates 2 \
  --river-iterations 200 --river-averaging-delay 20 --seed 10701 --threads 10 \
  --checkpoint-dir neural/runs/v30-public-belief/target-checkpoints \
  --output neural/runs/v30-public-belief/turn-targets-128.json

.venv-neural/bin/python neural/train_public_value_network.py \
  --dataset neural/runs/v30-public-belief/turn-targets-128.json \
  --output-dir neural/runs/v30-public-belief/compact-128-paired \
  --architecture compact --steps 3000 --seeds 10801,10802 \
  --holdout-start-index 64

.venv-neural/bin/python neural/validate_public_value_parity.py \
  --dataset neural/runs/v30-public-belief/turn-targets-128.json \
  --model neural/runs/v30-public-belief/compact-128-paired/turn-value-range-seed10801.json \
  --state-indices 64,65,66,67,69,70,72,76,77,78,79,81,82,85,90,91,93,94,95,96,104,106,107,109,111,112,113,115,121,122,123,127 \
  --solver target/release/preflop-solver \
  --output neural/runs/v30-public-belief/compact-128-paired/parity-holdout.json
```

The final measurements below come only from completed artifacts; a partial
checkpoint directory is never treated as an accepted corpus.

## Completed results

The first 64-state corpus passed all source gates and reproduced byte for byte
after cached-checkpoint assembly was made immutable. Compact paired training
measured `0.496425bb` mean range holdout RMSE, 35.47% improvement over the
range-blind control, and `0.996122` cross-seed correlation. The wide network
did not earn selection: its mean range tuning RMSE was approximately
`0.478751bb` versus compact's `0.464269bb`, while its JSON weights were 1.6MB
instead of 683KB.

Because the 64-state result materially improved on the earlier structural
pilot and passed the range/correlation gates, the corpus was extended to 128
states. The completed source artifact has SHA-256
`d4c0b6a6072cb88494e18ec2c46311beaf6cb202baf5cf322669b20573dbc366`
and reproduced byte for byte from immutable checkpoints. All 128 boards are
distinct. Source validation measured:

- maximum belief diagnostic TV `0.128834`;
- minimum belief ESS `1362.41` from two 4,096-particle replicates;
- maximum river exploitability `0.035762bb/hand`; and
- maximum target zero-sum residual `2.0923e-8bb`.

For the 128-state run, 32 untouched holdout states were restricted to newly
generated indices 64–127; none came from the already observed 64-state pilot.
The selected compact range seed was 10801 by tuning RMSE. On this genuinely
unseen holdout, paired mean range RMSE was `0.784731bb`, versus
`1.377465bb` without ranges (43.03% improvement), with `0.998547` cross-seed
correlation. Independent Python/Rust inference across all 32 holdout states
had maximum error `5.33069e-6bb`, passing the `0.0001bb` gate.

The larger holdout exposed the principal weakness. Selected-seed RMSE was
`0.274315bb` for states with at most 3.5bb invested per player,
`0.949079bb` for 4–7.5bb, and `2.115362bb` for the sampled 10.5–18.33bb states.
Correlation stayed high, so the network learned ordering but not value
magnitude as pot size increased. This points to pot/stack-normalized residual
targets and pot-stratified sampling or loss balancing, not more generic steps
or a wider dense head.

On the matched `2c 7d Th`, 4bb-pot, 20-iteration flop pilot, range conditioning
improved depth-limited exploitability from `0.789528` to `0.650617bb/hand`.
The range resolver also improved 58.49% over its uniform baseline, and leaf
zero-sum residual after projection was `5.55e-17bb`. These are useful
directional results but remain far above the `0.05bb/hand` gate. The pilot
also deliberately omits all-in branches.

The final fail-closed report rejects activation on turn RMSE, depth-limited
flop exploitability, full all-in action coverage, and the absent independent
full-game exploitability upper bound. No manifest or serving model was
modified. Exact all-in implementation is deferred until a revised value model
passes its support gate.
