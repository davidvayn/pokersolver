# 20bb v33 resolver-leaf conditioning pilot

## Decision

**Rejected for activation and rejected as the research default.** The strongest
v33 pair improves the matched authentic holdout by 13.94% and the untouched
resolver-reach-weighted leaf RMSE by 14.52%, but regresses the authentic
small-pot band by 12.98%. A second, less aggressive pair also regresses all
three matched resolvers. The pinned v31 model therefore remains the research
selection, and the active manifest remains unchanged.

No local value-model or depth-limited resolver measurement substitutes for the
missing independent one-sided 99% full-game exploitability upper bound. Large
datasets, weights, and resolver outputs remain under ignored
`preflop-solver/neural/runs/v33-resolver-leaves/` paths. The checked candidate
summary records hashes, measurements, and the fail-closed decision.

## Why this experiment

The v32 pilot showed that generic off-policy coverage improved an ordinary
turn-value holdout but made the matched resolver worse. Its next boundary was
therefore to train on the resolver's own counterfactual leaf distribution,
while keeping both an authentic holdout and a disjoint resolver-leaf holdout.
This follows the public-belief value-function direction used by DeepStack and
ReBeL, but remains a controlled local pilot rather than a claim that the
trunk/subgame interaction is solved. See the research rationale and primary
references in [the v32 report](20bb-v32-off-policy-coverage.md).

## Resolver-leaf capture

The Rust flop resolver can now freeze its average policy and recursively
capture every positive-reach turn leaf. At each public action it multiplies
the acting player's exact 1,326-combo reach by the frozen average action
probability. At the turn it applies exact blocker removal and records

```text
leaf probability = compatible joint action reach / root joint reach / 45
```

along with the root flop, complete public action history, actor, investment,
exact normalized ranges, and source-network hash. Each selected leaf is then
labelled by exact river enumeration with independently solved abstract river
games. No synthetic ranges or uniform fallback are used.

Sampling is deterministic and reach-weighted. Each root contributes one
small-, medium-, and large-pot leaf before any additional samples, and selected
turn boards must be distinct. The generator supports resumable, input-hashed
label checkpoints. A Rust regression test verifies frozen average-strategy
reach, the chance factor, and exact turn-card blockers.

Multithreaded resolver inference can vary below persisted precision. Before a
leaf is sampled or hashed, ranges are therefore canonicalized to their stored
`f32` precision and reach probabilities to a `1e-10` lattice. Without that
boundary, sub-ULP reach drift caused false checkpoint misses and up to
`0.00037bb` label drift after unnecessary re-solves. Two concurrent full
replays of each final corpus are byte-identical.

## Disjoint corpora

The capture resolver is pinned to v31 seed 10902, weights SHA-256
`2ec8e8d2af6704e5e53f98a182b19d5cb1836137a6dff3b34cfff94b7b569b52`,
and source-policy SHA-256
`c78397af5900b3409d3dfc911fce56075cb54ce860c38fc2a1459fe5d56df948`.
Both corpora use 20 resolver iterations, averaging after iteration 2, and
200-iteration river labels averaging after iteration 20.

| Corpus | Root flops | Captured leaves | Selected | Mean captured probability mass | SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| Training | 6 | 3,234 | 18 | 0.640413 | `bb984d1928502d7ab67f7e9fcba07170ea1807cad3381b0fdf45a41ec3abf3e2` |
| Evaluation | 3 | 1,617 | 9 | 0.650318 | `3f4382d823b60ecaa55112a2477cf9fd5e2ebf5cf8738c8104bba5279eba3869` |

Training roots are `2d 8c Qh`, `4s 9d Kc`, `5h 6h 7h`, `Ac Ad 6s`,
`8c 8d Jh`, and `3c Ts Qs`. Evaluation roots are `3d 8h Qc`,
`Kh Ks 5d`, and `6s 7s 9s`; none is a training root. The original matched
resolver boards are a third disjoint set.

Each training root contributes exactly three labels, giving six labels in
each pot band. The evaluation corpus has three labels per band. Training label
quality measured a maximum abstract river exploitability of
`0.009598bb/hand` and maximum absolute zero-sum residual of `1.44e-8bb`.
Evaluation measured `0.013507bb/hand` and `1.45e-8bb`. Both datasets are
accepted by their label gates.

## Paired students

All variants retain the v31 compact GELU network, pot-normalized Huber loss,
pinned split seed 10901, and untouched 32-state authentic holdout. Resolver
supplements are appended to training only. Two independent network seeds are
trained for every configuration; the selected seed is chosen by tuning RMSE.

| Pair | Training supplement | Mean authentic RMSE | Change from v31 | Leaf reach RMSE | Change from v31 |
| --- | --- | ---: | ---: | ---: | ---: |
| v31 baseline | none | 0.540214bb | — | 1.070258bb | — |
| Resolver full weight, preliminary | 18 leaves | 0.530314bb | 1.83% better | 0.859521bb | 19.69% better |
| Resolver weight 0.35, preliminary | 18 leaves | 0.514956bb | 4.68% better | 0.936644bb | 12.48% better |
| Fused, selected v33 | 12 off-policy + 18 leaves | 0.464882bb | 13.94% better | 0.914899bb | 14.52% better |

The first two rows were completed before the persistence canonicalization was
added; they use the same selected states and remain only exploratory evidence.
The final fused pair was retrained from the byte-stable corpus. It uses seeds
11801 and 11802. Their prediction correlation is `0.996256`; seed 11802 wins
by tuning RMSE. Its Rust/Python inference parity maximum is `9.7789e-6bb`,
below the `0.0001bb` gate. The selected weights hash is
`60b4af1ad210c23804c422e9fda124babd14bd8c573e187632db74943a580e70`.

Matched authentic pot bands expose the tradeoff:

| Pot band | v31 paired mean | Fused paired mean | Change | Gate |
| --- | ---: | ---: | ---: | --- |
| Small | 0.215392bb | 0.243352bb | 12.98% worse | Fail: no more than 10% worse |
| Medium | 0.803120bb | 0.686863bb | 14.48% better | Pass: no more than 5% worse |
| Large | 1.573794bb | 1.239399bb | 21.25% better | Pass: no more than 5% worse |

The fused model is therefore ineligible for the matched resolver audit. Not
running that conditional step prevents a favorable board draw from overriding
the predeclared authentic-distribution guardrail.

## Downstream resolver check

The preliminary weight-0.35 pair was already undergoing the predeclared
three-board, 100-iteration diagnostic when its small-pot result completed.
That audit is retained as negative evidence, not used to make the canonical
fused pair eligible.

| Board texture | v31 | Weight-0.35 v33 | Change |
| --- | ---: | ---: | ---: |
| Dry low, `2c 7d Th` | 0.540310bb/hand | 0.668023bb/hand | 23.64% worse |
| Dry high, `As Kd 7c` | 0.900115bb/hand | 1.184168bb/hand | 31.56% worse |
| Monotone, `9h Th Jh` | 0.694163bb/hand | 0.910744bb/hand | 31.20% worse |
| Mean | 0.711529bb/hand | 0.920978bb/hand | 29.44% worse |

This rejects the hypothesis that lower untouched leaf RMSE alone guarantees a
better resolver. Reach weighting is necessary but still incomplete: errors
also differ in strategic direction and alter the resolver policy that creates
future leaf reaches.

## Reproduction

From `preflop-solver`, generate the disjoint training corpus with:

```sh
target/release/preflop-solver flop-pbs-leaf-targets \
  --effective-stack-bb 20 \
  --boards '2d,8c,Qh;4s,9d,Kc;5h,6h,7h;Ac,Ad,6s;8c,8d,Jh;3c,Ts,Qs' \
  --states-per-board 3 --pot-bb 4 --actor 1 \
  --resolver-iterations 20 --resolver-averaging-delay 2 \
  --river-iterations 200 --river-averaging-delay 20 \
  --seed 11501 --threads 10 \
  --value-network neural/runs/v31-calibration/pot-128-paired/turn-value-range-seed10902.json \
  --checkpoint-dir neural/runs/v33-resolver-leaves/train-checkpoints \
  --output neural/runs/v33-resolver-leaves/resolver-leaf-train-18-seed11501.json
```

Generate the evaluation corpus with seed 11502 and the three evaluation roots,
then train the selected fused pair:

```sh
.venv-neural/bin/python neural/train_public_value_network.py \
  --dataset neural/runs/v30-public-belief/turn-targets-self-play-exact-128-seed10701.json \
  --supplemental-dataset neural/runs/v32-coverage/off-policy-turn-targets-12-seed11301.json \
  --supplemental-dataset neural/runs/v33-resolver-leaves/resolver-leaf-train-18-seed11501.json \
  --output-dir neural/runs/v33-resolver-leaves/fused-offpolicy-leaf-158-paired \
  --architecture compact --value-normalization pot --variant-set range-only \
  --steps 3000 --batch-size 6 --seeds 11801,11802 \
  --split-seed 10901 --holdout-start-index 64
```

`neural/validate_v33_resolver_leaf.py` composes the immutable holdout,
evaluation-corpus hash, parity, conditional resolver, and full-game gates. Its
checked report selects v31 and never activates v33.

## Next boundary

More generic labels are not justified. The next pilot should make training
resolver-aware without sacrificing authentic small-pot calibration: use a
multi-objective sampler or loss that preserves a fixed authentic quota per pot
band, and evaluate strategic signed-error sensitivity rather than scalar RMSE
alone. Any successor must first pass the same disjoint authentic and
resolver-leaf gates, then improve at least two of the three matched resolvers,
before investing in a broader full-game response evaluator.
