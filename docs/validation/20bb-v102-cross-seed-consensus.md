# 20bb v102 cross-seed consensus

Date: 2026-08-14

## Result

The paired 20bb range-conditioned policies now pass every declared exact
cross-seed action gate on the two untouched v82 release holdouts. This closes
the policy-stability blocker only. It is not an exploitability certificate and
does not activate the full-hand model.

| Metric | Full holdout result | Gate | Status |
| --- | ---: | ---: | --- |
| Reach-weighted action-frequency MAE | 4.9387% | <=5% | Pass |
| Primary-action agreement | 87.8941% | >=85% | Pass |
| Maximum aggregate-action delta | 1.4568% | <=3% | Pass |
| Policy lookup coverage | 100% | >=99.99% | Pass |
| Maximum probability-sum error | 4.44e-16 | <=1e-6 | Pass |

The comparison evaluated 5,997 authentic held-out public states and
14,053,707 exact private-combo action probabilities. Conditional MAE was
4.0252% on the flop, 6.5370% on the turn, and 4.2876% on the river. Turn is
still the weakest street, but the declared aggregate gate is reach weighted
and passes without changing its bound.

## Method and leakage controls

`train_range_policy_consensus.py` forms equal-probability consensus targets
from the two independently trained round-8 parents on bounded training
partitions. Each student starts from its own parent and uses a distinct
optimizer seed. The release holdout public-state identities are excluded from
the training partitions before any target is formed.

The accepted pair used two conservative stages:

1. 25 steps from the round-8 parents with 256 records per teacher at a 1e-6
   learning rate.
2. 20 steps from those two students toward their new midpoint at a 7.5e-7
   learning rate.

The final student artifact hashes are:

- seed 20931: `7296e5a54cd0c310f5fd7dc126937b41131c54d00b0bc2c6807d7791c14772f0`
  (`preflop-solver/models/practice/v102-seed20931-range-policy.json.gz`)
- seed 20932: `6a97155c767a4bb4beab5ff4e792965e58780be4e904601cc37c7c6555b1e1f1`
  (`preflop-solver/models/validation/v102-seed20932-range-policy.json.gz`)

The full comparison used the original independent heldout corpora with hashes
`d981edfe0435bc00870612673396871d54347c4e10456327ef74a25911444933`
and `f84269d93e973d23954a52ef9196dc0654991d24988825672dc68950ecbb00c7`.
Small deterministic subsets were used only to reject weak pilots; only the
full exact Rust result above is accepted.

The Rust evaluator now reports street contributions that reconcile exactly to
the aggregate metric and evaluates the two independent corpora concurrently.
Parallel evaluation changes wall-clock time only; probability inference,
weights, and gate definitions are unchanged.

## Remaining release blocker

Action-EV sampling precision still fails. The Rust report previously counted a
whole decision as failed when any legal action was noisy, while the release
validator weights each action by node reach and its served policy frequency.
Using the declared weighting raises the measured ten-cycle preflop coverage
from a misleading 0% to about 30.38%, still far below the required 95% at
0.02bb standard error. Learned-continuation approximation remains a separate
low-confidence warning and is not sampling error.

Exploitability is explicitly deferred for the current check. The practice
manifest therefore remains inactive and fail closed.
