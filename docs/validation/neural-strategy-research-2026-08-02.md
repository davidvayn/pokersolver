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

## Launch readiness (pre-run record)

The short paired rollout comparison, frozen v14 launch plan, bounded-memory
preflight, deterministic parallel exploitability certificate, and pre-run
browser matrix are complete. Four action-value rollouts are selected because
they improved four of six paired stability metrics, materially reduced both
aggregate-delta checks, and are the smallest tested count that produces a real
sample-standard-error target.

At launch time, the remaining work was downstream of training rather than a
prerequisite to starting it:

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

## Long-run outcome and temporal distillation

The paired 20bb run completed 310 narrow rounds for each seed. Round 250 was
stronger than round 310 on an independent 5,000-trajectory comparison, so it
is the protected preflop checkpoint. The initial wide stage completed 133
rounds per seed and selected round 100. An authorized extension was stopped
cleanly after rounds 150 and 200 both regressed against round 100 on broad
held-out comparisons; the atomic runs remain resumable at rounds 208 and 205.
Continuing the same updates was therefore rejected as non-productive compute.

Temporal logit averaging was then tested without changing the game. An 80/20
round-250/300 preflop teacher improved all six authentic/forced stability
measures, and an 80/20 round-100/200 postflop teacher improved the routed
candidate. Each teacher was distilled independently into a single frozen
network per seed. On a fresh 5,000-trajectory confirmation seed, the combined
students improved over the protected round-250/100 route as follows:

| Distribution | Metric | Protected | Distilled |
| --- | --- | ---: | ---: |
| Authentic reach | action-frequency MAE | 0.06324 | 0.06119 |
| Authentic reach | primary-action agreement | 78.64% | 79.34% |
| Authentic reach | maximum aggregate delta | 0.00886 | 0.00793 |
| Forced deviation | action-frequency MAE | 0.07129 | 0.06900 |
| Forced deviation | primary-action agreement | 76.28% | 77.24% |
| Forced deviation | maximum aggregate delta | 0.01193 | 0.01066 |

The validator hashes paired frozen-weight overrides and evaluates their actual
reach rather than treating the original checkpoint artifacts as the candidate.
This result is a genuine stability improvement, but it still fails the 0.05
MAE and 85% primary-agreement release gates. It remains hidden and must not be
described as Approximate GTO.

### Probability-space challenger

Two follow-up training branches were stopped under the same predeclared rule.
Persistent instantaneous-advantage replay improved from rounds 10 to 20 but
regressed broadly at round 30. Denser 1,600-traversal rounds reduced MAE only
slightly from rounds 5 to 10 while authentic agreement fell 5.8 percentage
points. Neither branch replaced the protected checkpoints, and their unhelpful
trainer variants were not retained.

Averaging checkpoint policies in probability space was consistently stronger
than averaging logits. The frozen v26 route uses independently distilled
students with these teachers:

- preflop: equal probability mix of narrow rounds 150/200/250/300;
- postflop: equal probability mix of wide rounds 50/100/150/200.

On a third untouched 5,000-trajectory seed, v26 improved over the v25 preflop
probability student paired with the previous postflop student:

| Distribution | Metric | v25 route | v26 route |
| --- | --- | ---: | ---: |
| Authentic reach | action-frequency MAE | 0.05873 | 0.05688 |
| Authentic reach | primary-action agreement | 79.63% | 80.15% |
| Authentic reach | maximum aggregate delta | 0.00349 | 0.00362 |
| Forced deviation | action-frequency MAE | 0.06779 | 0.06601 |
| Forced deviation | primary-action agreement | 77.67% | 77.81% |
| Forced deviation | maximum aggregate delta | 0.01649 | 0.01700 |

This clears the relaxed 0.06 MAE / 80% agreement research-pilot thresholds on
authentic reach. It does not clear the release thresholds. Preflop remains the
bottleneck at 0.05356 MAE and 75.60% primary agreement; authentic postflop
street agreement ranges from 84.81% to 86.75%.

The paired frozen files are deliberately kept outside the active registry.
Their byte counts and SHA-256 hashes are:

| Route | Seed | Bytes | SHA-256 |
| --- | ---: | ---: | --- |
| preflop | 5101 | 876,050 | `bede19e230cda7dbf30a4a39df9e9393d12d2090c921d993e6d67d3f53a18157` |
| preflop | 5102 | 876,050 | `6c02fc99c1b5c65160bde4fdc1d708df0a51c8c254a820a6eacc82a2ccbc4b3b` |
| postflop | 5101 | 2,013,717 | `cbc7135f6c4aa38232d511a71d05db1c0f6c5aa0fcd8b58e799c1fbc311eb427` |
| postflop | 5102 | 2,013,717 | `fba9ae94b3af651a0057cd0e20c91de0c1be0985821b33db5470f2a95c97d07f` |

### Exploitability scout and stop decision

The v26 pair was evaluated with the actual conservative certificate at 5,000
complete deals per seed and 99.5% confidence per candidate. Each candidate
expanded 87,700,000 exact betting-tree nodes. The results were:

| Seed | Sample mean | Standard error | Hoeffding margin | Upper bound |
| ---: | ---: | ---: | ---: | ---: |
| 5101 | 4.83081bb | 0.03443bb | 0.46036bb | 5.29117bb |
| 5102 | 4.84144bb | 0.03485bb | 0.46036bb | 5.30180bb |

The responder observes both private hands and the complete board. This makes
the result a mathematically valid upper bound on imperfect-information
exploitability, but also introduces a large value-of-perfect-information gap.
It must not be reported as a 4.84bb estimate of true exploitability. Increasing
the deal count can shrink the chance margin but cannot remove the observed
relaxed-response mean. Consequently, the 125,000-deal release certificate was
not launched and further identical neural iterations were stopped.

The next architecture prerequisite is a substantially tighter evaluator that
preserves responder information sets, followed by a dedicated preflop solve or
resolver if that evaluator confirms the preflop weakness. The checked-in v26
research manifest remains rejected and the active registry remains empty.
