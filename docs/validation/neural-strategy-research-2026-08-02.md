# Neural poker strategy research — 2026-08-02

## Decision

Keep the frozen Deep DCFR+ baseline and confidence-capped response architecture,
serve the empirically stronger preflop/postflop components through one routed
artifact, train uncertainty from sampled action values, and retain sparse
advantage snapshots for an offline direct-average-policy challenger.

No model is activated by this work. The full-hand registry remains empty.

## Primary-source findings

- [Deep CFR](https://proceedings.mlr.press/v97/brown19b.html) assigns external-
  sampling strategy-memory samples equal weight within an iteration and the
  iteration number across iterations. The existing trainer already follows
  that rule; multiplying its stored reach field into the loss would have been
  an incorrect double weighting. The paper also notes that retaining a subset
  of prior advantage networks can reduce final policy-network approximation
  error.
- [Single Deep CFR](https://arxiv.org/abs/1901.07621) removes the separately
  trained average-policy network and reports lower approximation error and
  better poker exploitability. Direct SD-CFR inference is not yet a browser
  replacement here because grading requires a complete frequency mix at an
  arbitrary reached information set. Sparse, hashed teacher snapshots preserve
  a matched offline comparison without increasing the served artifact.
- [DREAM](https://arxiv.org/abs/2006.10410) contributes learned advantage
  baselines for model-free regret minimization. This project has an exact game
  simulator and already uses an action-dependent control variate. The valuable
  local change is to train and validate that value/uncertainty path from real
  independent rollouts, not to import DREAM's model-free machinery.
- [ReBeL](https://proceedings.neurips.cc/paper/2020/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html)
  combines self-play with public-belief search and has a two-player zero-sum
  convergence basis. It is a different online-search architecture, not a
  low-risk optimization of the current bounded trainer, so it is deferred.
- [Fictitious Self-Play](https://proceedings.mlr.press/v37/heinrich15.html) is a
  sound sample-based framework, but the repository's matched Deep CFR/NFSP
  prototype favored Deep CFR under the available budget. NFSP remains outside
  the serving architecture.

## Implemented evidence

- Schema-2 routed artifacts compose two schema-1 components only after exact
  compatibility checks. The actual seed candidates are 8,720,729 bytes each
  and reproduce byte-for-byte across repeated exports.
- Rust trajectory evaluation records per-action means and sample standard
  errors from independent deterministic rollouts. Python trains the uncertainty
  head from those values and release validation requires all actions at a
  reached decision to meet the 0.02bb threshold.
- Two independent one-round v14 smoke runs completed, exported, and passed
  probability, integrity, and coverage checks. Their deliberately tiny two-
  rollout corpus failed the confidence threshold, as it should.
- Advantage snapshots are retained only at configured artifact rounds and are
  excluded from browser artifacts. Each snapshot manifest records both hashes,
  its traversal count, and its DCFR strategy weight.

## Long-run readiness

The short paired rollout comparison, frozen v14 launch plan, bounded-memory
preflight, deterministic parallel exploitability certificate, and pre-run
browser matrix are complete. Four action-value rollouts are selected because
they improved four of six paired stability metrics, materially reduced both
aggregate-delta checks, and are the smallest tested count that produces a real
sample-standard-error target.

The remaining work is downstream of training rather than a prerequisite to
starting it:

1. Run the two fresh independent composite seeds from the checked-in plan.
2. Compare the final average-policy network against the sparse direct-policy
   teacher on identical held-out corpora, then retain only the better validated
   serving approach.
3. Run the full post-training gates, compose the selected seed, and complete
   full-hand/preflop/postflop browser acceptance before adding an active
   manifest.

The certificate is a true upper bound because it enlarges the responder's
information and then solves every betting action exactly; it is not a learned
approximate response. Its one-sided Hoeffding chance bound and exact-card deals
are deterministic across thread counts. The information relaxation is
conservative enough that a good policy may still fail. That outcome keeps the
depth hidden and does not justify relabeling a lower bound as release evidence.
