# Cheap, decision-relevant postflop pilots

Date: October 3, 2026 (PDT).
Status: **Replay, full-chance and TRAIN-coverage student pairs rejected; bounded coverage intervention complete.**
No model has been promoted. Earlier execution is recorded in
[the replay pilot report](postflop-replay-pilot-2026-10-03.md) and
[the action-contrast report](postflop-action-contrast-pilot-2026-10-04.md).
The reserved chance block exposed action-ranking instability larger than the
intended improvement. Per Section 6B, no C0/C1 fit or conditional expansion
was started at that boundary. The subsequent all49 pilot improved frozen
training rankings but failed the actual-policy screen. The user authorized
continuing on October 4; Section 11 now tests broader TRAIN board/search
coverage with that reliable chance integration, not unchanged iteration scaling.

This refines Step 3 of [the postflop gap plan](postflop-gap-implementation-plan.md).
It specifies what to implement, how to avoid another expensive false positive,
and what to do after a promising pilot. Results and artifact identities are in
[the completed update/label report](postflop-gap-update-label-pilot-2026-10-03.md).

## 1. Decision and implementation clarifications

The first experiment should be **replay balancing**, using already generated
labels. If that does not protect the control while retaining useful gains,
test a **separate, chance-integrated action-contrast objective**. Do not start
by changing the architecture, doubling iterations, and improving every label
simultaneously.

The evidence supports an action-value/distribution problem, not a proven
single cause. The 64-label and 256-label students lowered mean response gain
on eight selected cases from 0.41214bb to 0.34629bb and 0.33921bb, respectively.
But both worsened both seeds of the three-bet high-rainbow control. Extra
label computation did not fix that regression. On the new student's own
control beliefs, predicted local policy loss was 0.1337bb while native local
loss was 0.5872bb. An improvement on the old policy's beliefs did not predict
the new policy's performance.

These numbers are conditional postflop response gains under a finite native
continuation reference. They are **not full-game bb/hand exploitability**.

| Clarification | Implementation decision |
| --- | --- |
| Are more labels automatically the answer? | No. Reuse the 615-state native64 corpus first; stronger labels added little stable policy benefit in the completed comparison. |
| Does the existing replay flag protect the old data? | Not here. All 615 states currently count as primary, so the supplemental set is empty. Moreover, native training explicitly rejects legacy supplemental datasets. Add a native provenance partition inside the existing family-split corpus; do not bypass that rejection. |
| Can the 107 isolated turn labels directly train flop action rankings? | No. A contrast requires a coherent frozen prefix and every needed action branch, with a shared chance sample and correct card removal. The existing reservoir samples do not contain those complete groups. |
| Should we train the network to copy a single best action? | No. Continue predicting calibrated turn values. Add supervision on backed-up action EV differences; preserve the legal action tree and mixed-policy solver. |
| Should “check more” be a target? | No. More checking accompanied the control regression but is not a general poker rule. Train hand-specific values, not aggregate desired frequencies. |
| Should we unroll and differentiate through CFR? | Not for this pilot. Freeze the prefix, policies, ranges, and chance coefficients. The differentiable portion is the value network followed by an affine backup. |
| Can a cached native vector be reused at a new range? | Only if the complete input identity matches. New candidate policies generally create new beliefs, requiring new native packets. |
| What decides success? | Native-evaluated actions and actual candidate-policy response gains, including the known control. RMSE, KL, and lower training loss are diagnostics, not substitutes. |

## 2. Research basis—and what it does not prove

DeepStack separates value approximation error from finite solving error in
its theoretical analysis. That motivates fixing action-relevant leaf errors
before buying unchanged iteration scaling. Its safe re-solving assumptions
do not automatically hold in this repository; its paper also distinguishes
the sparse-action implementation from the theoretical guarantee.
[DeepStack, Theorem 1 and sparse lookahead](https://arxiv.org/html/1701.01724v3).

ReBeL samples beliefs encountered during search iterations, including
one-player exploration. This supports testing current-search and unilateral
deviation coverage rather than fitting only final-average beliefs. Our
existing capture already samples early/middle/late and final beliefs; the
missing addition here is coherent decision groups and their use in training,
not a newly invented range-conditioned network.
[ReBeL, Section 5.2](https://arxiv.org/html/2007.13544v2).

Decision-focused learning provides evidence, outside poker, that prediction
accuracy need not track downstream decision quality. This motivates the
contrast auxiliary; it does **not** establish that our proposed objective
reduces poker exploitability. The paired policy experiments must establish
that locally before expansion.
[Wilder, Dilkina, and Tambe](https://arxiv.org/abs/1809.05504).

## 3. Freeze the experiment before implementing expensive work

1. Pin the September benchmark protocol/manifest, retained model pair,
   native64 student pair, game/action schema, source hashes, and rebuilt
   binary. Preserve every existing artifact. A rebuilt binary must reproduce
   a retained candidate before its new candidates are compared.
2. Reuse the native64 merged corpus:
   `runs/local-search-label64-20261003-matched-a/corpus.json.gz` (paths in this
   document are relative to `preflop-solver/neural/`). Its SHA-256 is
   `2efb28692ffea119e65894a8748677cb2dda270a4571ae79d3d786d89bedfa73`.
   The native64 student manifest SHA-256 is
   `e79707c7f11f435af8fd5f120e1a5c2395109bcb6716a767a4f021de9834f971`.
3. Keep the original 508-state prefix, its training/tuning/holdout membership,
   and all label values unchanged. The 107 appended states are training-only.
   The frozen split is 474 train, 69 tuning, 72 holdout states.
4. Retain the wide architecture, feature schema
   `rank-suit-invariant-combo-query-v3`, prediction contract
   `native-turn-cfv-full-stack-v1`, payoff-exposure normalization, value loss,
   optimizer, batch size 8, and paired training seeds 10601/10602. Use the
   existing solver pair 100101/100102. Do not select one lucky seed.
5. Keep candidate flop updates at 128, native played continuations at 64,
   and final response aggregation over all 49 public turn cards. The existing
   256-flop-update arm remains a separate compute comparator.
6. Declare the four consumed development/control roots and all twelve
   September benchmark families ineligible for training or fresh confirmation.
   New training bundles must also avoid the frozen tuning/holdout families.
   Split by suit-canonical flop family, including all turns, histories, and
   iterations—not by individual hand rows.
7. Record a resource projection, stage limits, advancement rules, and output
   directory in a manifest before starting each stage. An incomplete or
   resource-stopped stage is inconclusive, not a passing result.

## 4. Stage 0: establish an inexpensive but predictive screen

Do this before new labeling or a new student pair.

1. Reuse saved policies, native packets, terminal equities, and prediction
   probes for the retained, rejected native64/native256, and 256-update arms.
   Recompute summary diagnostics only; do not regenerate their native packets.
2. Check whether proposed fixed-belief screens would have caught the two
   known failed students. Include BOTH old-policy beliefs and each student's
   own frozen-policy beliefs. Record native loss from the predicted-best
   action, policy-deviation loss, and action contrasts at the costly nodes.
3. Order subsequent actual policy tests by measured cost: first
   `three-bet-high-rainbow` (known failure), then `three-bet-monotone`
   (known improvement), both seeds. Previous complete four-case confirmation
   stages took roughly 10–15 minutes per arm; these are much cheaper than
   the limped/single-raised evaluation stages, which took roughly two hours
   under the measured machine load.
4. Use the existing **full-49-turn response evaluator** for this small screen.
   Do not implement a new sampled-turn exploitability evaluator merely to
   make the first pilot cheaper. Four carefully selected exact cases plus
   cached diagnostics are inexpensive and directly test the previous failure.
5. Keep fixed-belief probes as explanatory screens, not promotion gates by
   themselves. A candidate must generate its own policy and be scored on
   its own reconstructed beliefs. Never relabel an alternative-value probe
   on an unchanged policy as that alternative model's response gain.

These four cases test a known weakness and a known success, not generalization.
They predict whether repeating the earlier failure is likely; they cannot
predict full-game performance with statistical certainty.

**Compute target:** 5–20 minutes; hard cap 30 minutes. No new native labels.
If cached identities are missing, repair the manifest/cache inventory before
running new fits; do not silently compare mismatched policies.

## 5. Stage 1: paired native replay-balancing pilot

### Minimal implementation

1. Add an explicit, hash-checked native replay boundary of 508 states, derived
   from the retained-prefix provenance. After `family_split`, partition only
   training rows into retained-prefix and appended-search groups. Assert that
   tuning/holdout rows cannot enter either sampler.
2. Make the intervention real, rather than choose a nearly identical replay
   target. The training-state counts by pot band are:

   | Band | Retained | Appended | Retained fraction within band |
   | --- | ---: | ---: | ---: |
   | Small | 82 | 39 | 67.8% |
   | Medium | 91 | 26 | 77.8% |
   | Large | 194 | 42 | 82.2% |

   The current eight-slot sampler allocates 3/3/2 slots to these bands,
   yielding about 75.1% retained examples overall. A 75% replay target would
   therefore barely change total exposure. Instead reserve 7 retained and
   1 appended slot (87.5%/12.5%). Keep the SAME 3/3/2 pot-slot schedule, and
   rotate which of its eight positions receives the appended example over
   eight successive batches. This preserves the pot mixture while increasing
   replay in every band. The present fixed preferred-slot ordering could
   otherwise starve the appended large-pot examples. Log actual group/pot
   counts and every explicit empty-band fallback.
3. Keep sampling weights separate from true ranges and projection weights.
   Replay balancing changes the loss distribution, never the public belief.
4. The disabled feature must preserve the current native training path and
   RNG consumption. Add tests for zero-feature equivalence, provenance
   identity, split exclusion, empty-band fallback, and long-run band balance.

### Run and comparison

- Reuse the completed native64 pair as R0, after verifying no-op equivalence.
  Fit one new R1 pair from the same random initializations with only replay
  sampling changed. No warm-start or architecture change is needed.
- Cap at 600 steps. The R0 students both completed and selected step 600.
  Retain the existing tuning-checkpoint rule for this sampling experiment,
  and record actual steps/checkpoints so early stopping is not hidden.
- First run cached value/action probes and representative Rust/Python parity.
  Then run the cheap full-49 control/monotone screen on the new R1 policies.
  Perform exhaustive corpus parity once before an arm advances; do not repeat
  a forty-minute verification pipeline for every intermediate snapshot.
- Compare R1 against R0 to measure the sampling intervention, and against the
  retained policy to ensure it has not merely become less bad than R0.

**Compute target:** approximately 1–1.5 hours including fitting, verification,
and the four-case screen; hard cap 2 hours, recalibrated by the first receipt.
The completed pair's fit alone took about 16 minutes and fit plus exhaustive
parity took about 39 minutes. Those are measurements, not guaranteed runtimes.

**Branch:** if replay balancing protects the control and retains credible
policy improvement, proceed directly to Stage 3. If it fails or is marginal,
preserve its diagnosis and proceed to Stage 2; do not sweep many replay ratios.

## 6. Stage 2: coherent action-contrast pilot, only if needed

### 2A. Specify the exact value/backup contract

An example is BTN choosing check, small bet, larger bet, or all-in after BB
checks. For each BTN holding, compare those action values **after integrating
future cards**, while holding the following prefix policy and continuation
contract fixed. Do not choose a different flop action separately for each turn.

For a frozen decision group, define the affine backup

```text
Q_theta(h, a) = exact_terminal_term(h, a)
                + sum_l backup_coefficient(h, a, l) * EV_theta(l, h)
Delta_theta(h, a, b) = Q_theta(h, a) - Q_theta(h, b)
```

Here a leaf `l` identifies its exact public history, turn, frozen ranges, and
continuation settings. The coefficients include the appropriate later policy,
chance, opponent-compatible mass, and conversion to the parent's conditional
EV. Reuse the native/JavaScript backup as the semantic oracle; do not invent
a second normalization convention.

- The network exports conditional EV. Multiply compatible opponent mass
  exactly once to obtain raw CFVs. Own reach is a loss weight, not an extra
  CFV multiplier. Preserve the existing zero-sum projection and exposure scale.
- The current all-49-card backup sums raw packets with factor `1/45`, because
  both private hands remove four additional cards. Blocked-hand outputs are
  zero and opponent blocking is already inside each packet. Replacing this
  with a naive `1/49` is incorrect.
- For a uniformly selected subset of K public turn cards, use the corresponding
  inclusion correction `49/K` on those raw `1/45` contributions. Use the SAME
  subset for every action in a group. Do not renormalize each action over its
  sampled surviving cards. Such a subset estimates values; nonlinear ranking
  and regret metrics can still be biased, so it is not a release evaluator.
- Training targets are native **profile** values for the declared continuation,
  not best-response values mixed into a self-play value target. Best-response
  values remain evaluation diagnostics.
  Implementation clarification: the existing calibration contract deliberately
  completes zero-own-reach holdings by best response. Preserve that contract
  and its counterfactual calibration coverage. The contrast auxiliary uses
  profile labels only for holdings with positive own support at every
  contributing leaf; report excluded authentic reach. Do not feed contradictory
  off-support profile targets into the same value head or floor the ranges.
  The cached high-rainbow control has essentially 100% consistent authentic
  support, so this distinction alone does not explain its observed regression.
- Prefix policies/ranges are frozen during differentiation. No hidden future
  card or opponent holding may select an earlier action. Include every legal
  action even when its current frequency is small.

Prove the backup/gradient on a tiny two-chance, two-action imperfect-information
fixture first. Test card blockers, unequal masses, zero-own-reach legal hands,
exact fold/all-in terms, leaf ordering, chance integration before maximization,
and finite-difference gradients. Match a cached full-49 native/JS diagnostic
before paying for new groups.

### 2B. Build a bounded, training-only bundle set

1. Choose one frozen proposer snapshot per eligible training family, initially
   using the already pinned roots 2/3/4 if their split/provenance checks pass.
   Their complete live turn-leaf counts are 5/9/9. These training roots are
   separate from the named development/control boards.
2. Export complete prefix strategies and leaf-input identities. Capture initial
   BB and BTN-after-check decision groups from the same snapshot where legal;
   share leaf labels between overlapping groups rather than solving twice.
3. Start with one predeclared random block of 8 public turns per family. This
   requires at most `8 * (5 + 9 + 9) = 184` native64 leaf labels for these
   snapshots, not thousands of arbitrary samples. Confirm all required leaves
   exist; do not fill missing branches with zeros or model predictions.
4. Reserve a disjoint second 8-turn block for sensitivity checking. Expand to
   it only within the stage budget if the first block's action contrasts are
   unstable. It remains in the same TRAIN family, not independent board
   validation. Record its role explicitly.
5. Before fitting, compare 64/256 native labels on a bounded sentinel set of
   complete branches, including small-pot and control-like check/bet lines.
   Inspect whether teacher drift changes important backed-up contrasts. Use
   1024 only on a genuinely unresolved sentinel within budget, not universally.
   A few stable sentinel values do not prove all targets are exact.
6. If teacher/sampling sensitivity is comparable to the intended improvement,
   mark this data pilot inconclusive. Increase the targeted reference/chance
   budget once if affordable; otherwise stop this arm. Do not fit increasingly
   confident rankings to unresolved targets.

Store bundles separately from the frozen 615-state corpus, with hashed input
references, family, snapshot/iteration, history, actions, turns/inclusion
weights, true ranges, reach weights, native budget, and terminal terms. Do not
silently raise or bypass the current 640-state/256MiB corpus guards. Stream
bounded bundle shards and cache features only under their full identities.

**Compute target:** 30–75 minutes; hard cap 90 minutes for new labeling and
sensitivity checks combined. First time a 2-turn complete bundle on the most
expensive selected family; project remaining labels with a 1.5× margin.
The earlier 107-label native64 capture cost about 16 minutes versus about
57 minutes of worker time at native256. Complete bundles can differ in cost;
use the new receipt, not those averages alone.

### 2C. Implement a calibrated contrast auxiliary

Use calibrated value loss plus reach-weighted Huber loss on all unique action
pairs, with contrasts in consistent depth-normalized bb units. Average within
each decision group, then across families/pot bands; a family with more leaves
must not acquire more weight merely by being larger. Preserve bounded
counterfactual coverage for legal zero-own-reach holdings; those training
weights must not modify the ranges used in the backup.

Do not backpropagate through `argmax`, prescribe a pure action, or penalize
absolute player bias again. Fix one coefficient before the paired run: use a
training-only gradient preflight to scale the auxiliary to approximately 25%
of the calibrated-value gradient norm, with a finite declared upper cap.
Record the resolved coefficient and component norms. This is a conditioning
choice, not a result selected against development responses.

Keep the memory bound by microbatching complete leaf vectors. If one group's
graph is too large, first accumulate Q without gradients, then compute the
loss derivative with respect to Q and accumulate per-chunk vector-Jacobian
products at unchanged weights. Update the optimizer ONCE after every chunk.
Test equality with the single-graph gradient. Updating after each chunk would
not optimize the chance-integrated contrast loss.

### 2D. Match the control properly

Fit exactly two arms, two independent seeds each, same initializations and
600 optimizer-update cap:

| Arm | Ordinary value batches | Bundle value calibration | Contrast auxiliary |
| --- | --- | --- | --- |
| C0 | Same frozen corpus and sampling | Same new bundle states/cadence | Off |
| C1 | Identical to C0 | Identical to C0 | On |

Both arms see the same leaf labels, family schedule, minibatches, and bundle
forward passes. Add a bundle update every fourth optimizer step, combining
its calibration gradient with that step's ordinary gradient in BOTH arms;
only C1 adds the contrast gradient. This avoids attributing extra data or
extra optimizer updates to the new objective. Compare C0 to R0 separately
to identify the coverage effect.

Use fixed step-600 weights for the primary C0/C1 comparison. Record steps
200/400 as failure/learning-curve diagnostics, not additional response-selected
winners. Disable RMSE-only early checkpoint selection in BOTH arms for this
comparison: it could discard a checkpoint whose decisions improved. Still
stop on non-finite loss/gradients or a resource limit. No warm-start is needed.

Perform parity checks, cached contrast probes, then the same full-49 cheap
control/monotone policy screen. Run one MLX fit process at a time on the 16GB
machine. **Compute cap:** 2 hours for both fitted pairs and verification,
then 1 hour for their small actual-policy screens. Re-estimate after preflight;
if the new gradient workload cannot fit, reduce bundle cadence equally in
both arms BEFORE starting, rather than silently giving C1 less computation.

## 7. Advancement, rejection, and inconclusive outcomes

These are pragmatic experiment rules, **not new Approximate GTO release gates**.
For the initial four-case exact screen:

1. Require valid probabilities, finite outputs, no missing nodes, and maximum
   Rust/Python prediction discrepancy below the existing 0.0001bb tolerance.
2. Compare actual native response gains to both the matched experimental
   control and the retained model. A promising arm should improve the
   equal-case mean by at least 0.02bb versus retained, with positive mean
   improvement for each seed pairing. This is a useful-effect threshold,
   not a confidence interval or a convergence theorem.
3. Do not advance an arm that increases either high-rainbow control seed's
   gain by more than 0.01bb versus retained. This catches the already observed
   0.0198–0.0727bb regressions. Apply the same material-regression check to
   the monotone case rather than hide a new weakness in the mean.
4. For an objective claim, C1 must also beat C0 in both paired means, with
   action-contrast improvement on the costly nodes. If C0 wins and passes
   the retained-policy checks, advance C0 as a coverage improvement, not
   an action-loss success. If replay R1 succeeds, do not implement C solely
   to add another experiment.
5. Borderline seed disagreement, smaller effects, reference sensitivity, or
   an exhausted compute cap are **inconclusive**. Add one predeclared,
   independent board block if the effect justifies it; do not repeatedly
   rerun seeds until one passes. A clear regression rejects the arm.

## 8. Stage 3: conditional expansion after a promising pilot

1. **Complete the four-root development comparison.** Evaluate the winning
   arm on limped-paired and single-raised-high-rainbow, both seeds, with all
   49 turns. Keep the same 128/native64 protocol and independent JS audit.
   Apply the mean/seed/control rules to all eight cases. Expensive roots now
   run only for an arm that already passed the known cheap failure test.
2. **Confirm family generalization.** Freeze a new twelve-family set spanning
   the three pot types and four textures, excluding every consumed benchmark,
   training, tuning, holdout, and pilot family. Generate retained controls and
   candidate cases under identical settings. Treat the BOARD FAMILY, not
   seeds/private holdings, as the independent unit. Report paired means,
   seed and pot breakdowns, tails, and uncertainty in bb and percent of pot.
   Run the twelve consumed benchmark cases as regression evidence separately.
3. **Scale only the demonstrated intervention.** If replay alone wins, retain
   it without a new network head. If coherent supervision wins, expand from
   three training families to roughly twelve, including early/middle/late
   search snapshots and one-player-deviation beliefs. Add labels where native
   action contrasts/coverage remain weak; retain old-data replay. Re-run the
   cheap controls before broad evaluation after each substantial dataset change.
4. **Measure an actual scaling curve.** Compare the accepted configuration
   at 128/256 flop updates on a small matched block; extend to 512, then 1024
   only if native-evaluated policy gains continue to improve. Separately test
   600/1200 fitting updates if the learning curve suggests underfitting.
   Do not mix both changes in one claimed causal test.
5. **Use paid compute only for measured useful work.** Estimate per-family
   labeling, independent seed fitting, packet evaluation, RAM, and throughput
   from receipts. Buy parallel board/seed jobs if they are the obstacle; a
   serial regret chain does not become sixteen times faster from sixteen cores.
   No paid server provisioning is part of this planning request.
6. **Evaluate the pinned combined full-hand model.** Route retained preflop
   plus accepted postflop weights and evaluate complete hands against existing
   information-set-consistent responders. Report the 0.50bb/hand target and
   later 0.05 aspiration with the summed/half-summed convention and responder
   scope. A restricted/learned response is a lower-bound diagnostic, not a
   certified global exploitability upper bound.
7. **Qualify serving and integrate only accepted weights.** Recheck coverage,
   probability validity, action-EV uncertainty, pinned-version behavior,
   inference parity, latency, and missing-model handling for the actual route.
   Integrate the same model/action/continuation contract into the website;
   run the applicable TypeScript, build, Rust, and desktop/mobile browser
   acceptance checks. Do not introduce a database or UI overhaul for this pilot.

If C0/C1 both improve frozen-belief values but fail on their own policies,
trace the identified range/boundary shift. The next bounded option is refreshed
current-search decision bundles, not a larger unchanged fit. Only if that
trace identifies a continuation-contract problem should the existing plan's
safe-resolving or coherent multi-valued continuation pilot begin. Do not build
both speculatively or describe approximate teacher values as safe bounds.

## 9. Resources, scheduling, and implementation order

The Stage 0–1 replay test should cost roughly 1–2 hours of local compute, not
an overnight run. The optional Stage 2 is bounded at 4.5 hours of additional
compute before a decision. These exclude implementation time and conditional
broad confirmation; the first receipts may invalidate the estimates.

- Reserve at least 20GiB free disk. Cap this pilot's new artifacts at 10GiB;
  shared immutable packets/features should be referenced, not copied per arm.
- Keep fitting below the existing 6GiB per-process guard; previous peak was
  about 3.4GiB. Start at two native packet workers, increase to four only after
  memory and throughput measurements. One labeling controller must account
  for its internal workers—do not stack independent four-worker pools blindly.
- Keep at least 4GiB system headroom and honor the existing memory-pressure
  guard. Pause/stop on critical pressure, disk reserve violation, or manifest
  identity drift. Preserve complete artifacts and mark interrupted jobs.
- Check early receipts for failures/throughput. Once healthy, wait roughly
  half the revised remaining duration, then re-estimate near completion.
  Sleeping saves orchestration usage; stage timeouts/guards must remain active.
- Resume only complete hash-matching jobs. Do not append to an exported
  average policy as if it contained CFR regrets or optimizer state.

Implement in this order, with targeted tests at each seam:

| Order | Existing location / proposed addition | Deliverable |
| --- | --- | --- |
| 1 | `native_action_value_probe.mjs`, `native_action_diagnostics.mjs`, saved response manifests | Cached screen calibration and cost-ordered case list; no new evaluator |
| 2 | `train_public_value_network.py`, `run_native_value_student.py`, adjacent tests | Native provenance replay mask, balanced sampler, explicit report fields, R0 no-op parity |
| 3, only if needed | Proposed `action_contrast_dataset.py` and tests; `flop_pilot` frozen-prefix export seam | Complete hashed groups, card/chance/affine-backup parity, bounded native label capture |
| 4 | Proposed `action_contrast_loss.py` and tests; trainer/student runner | Chance-integrated loss, exact microbatch gradients, C0/C1 scheduling and checkpoint contract |
| 5 | `run_student_value_pilot.py` and a small proposed decision-pilot controller | Cheap control-first policy evaluation, stage budgets, paired rules, immutable receipts |
| 6, after success | Existing broad benchmark/full-hand route, then serving integration | Fresh-family confirmation, measured scaling, combined-model and website acceptance |

The primary checkout currently contains unrelated dirty benchmark/resource
changes. Preserve them; coordinate with their owner rather than editing that
work opportunistically. Commit only scoped milestones. Use a clean checkout
to reconcile/push over newer remote commits; never force-push the divergent
primary branch or discard its dirty files.

**Deliverable at the pilot boundary:** one compact report saying advance,
reject, or inconclusive; paired actual-policy numbers; known-regression checks;
action diagnosis; compute/RAM/disk costs; and immutable artifact identities.
No guaranteed convergence time and no promotion based on a better surrogate.

## 10. Focused continuation after the inconclusive chance pilot

October 4 continuation: first complete original TRAIN root 2 (the cheap
three-bet family) at unchanged native64. Reuse all 16 hash-verified native64
turns from the two previous blocks and solve only the remaining 33. Keep its
frozen policies, ranges, legal action tree, and continuation contract identical.
Use two guarded workers and a 45-minute cap; project cost from the saved turn
receipts before starting. Do not add new boards or optimizer updates yet.

Compare the original 8/8/16-card action rankings to the all49 reference using
the same authentic parent reach. Compare the existing native64/256 sentinel
both alone and as a one-turn upgrade inside the all49 target. Exact integration
removes sampled public-turn noise, not native64 continuation error, abstraction
error, or exploitable play. Report those distinctions explicitly.

If sampled rankings materially disagree with all49 while the measured teacher
sentinel is comparatively stable, the next actionable change is full-chance
training bundles for the same three TRAIN families, stored as separate bounded
calibration shards. Before C0/C1, require complete coverage, consistent support,
finite targets, hash/parity checks and a measured fitting-cost preflight. Both
arms must integrate the same complete chance set and see identical calibration
data. Reduce their bundle cadence equally *before* fitting if needed for the
existing resource cap. If teacher drift or costs are the blocker instead,
preserve the evidence and choose a targeted reference or validated chance
control-variate pilot; do not launch another unchanged fit.

### October 4 completion and branch decision

The [full-chance pilot report](postflop-full-chance-pilot-2026-10-04.md)
records completion of all49 references, guarded conditioning, both fixed-step
student pairs, exhaustive export parity, cached TRAIN contrast probes and both
full49 actual-policy screens. C1 reduces frozen TRAIN ranking loss by 42.6%,
but neither arm passes the actual-policy regression screen. C1's mean benefit
versus C0 is only 0.000078bb with opposing seed means; both regress high-rainbow
versus retained. **Reject both; do not trigger Stage 3 expansion/integration.**

Targeted own-policy and same-board retained-policy probes localize a large
continuation action-value mismatch at BTN after BB checks. It persists under
retained-policy ranges, so pure own-policy range drift is insufficient as an
explanation. The next bounded intervention should broaden TRAIN board/search
belief contrast coverage, retaining frozen evaluation families and old-data
replay. First diagnose that small block with reused/bounded native references;
do not repeat an unchanged fit, silently relax the screen, or buy compute
before establishing a useful effect. The pilot boundary is complete; broad
full-hand/serving qualification remains conditional on an accepted policy.

## 11. Bounded TRAIN coverage intervention (October 4)

1. Freeze three new TRAIN families from an authentic 128-root bank (seed
   2026100417): first two unpaired Q-or-higher rainbow and first two-tone
   five-leaf three-bet families, in bank order. Exclude every family in the
   frozen 615-state corpus and all twelve benchmark roots, before inspecting
   any model errors. Bank indices 22, 76, 115 become TRAIN IDs 100, 101, 102.
2. Use the rejected C1 pair only as an exploratory proposer, never active play.
   Capture 16 native64-labeled queries per root from early/middle/late search
   and final-average beliefs at 128 flop updates. Check observation/no-
   observation policy parity on the first root. Export its complete final
   prefix and label all 49 legal public turns at native64; one native256
   sentinel per root checks measured teacher drift.
3. Keep the original 615 corpus/split unchanged. Store each new family as a
   separate 261-state calibration shard (245 final-prefix leaf labels plus
   16 search queries), retaining existing 640-state/256MiB decoded guards.
   Zero-pad the affine contrast coefficients for the extra queries: search
   calibration cannot silently alter frozen action targets.
4. Diagnose continuation/action-ranking errors on these new training inputs.
   Only trustworthy complete targets can proceed to fitting; a finite
   sentinel loss comparable to the 0.02bb intended benefit is inconclusive.
   Native64/256 stability is not an exploitability certificate.
5. If justified, join the three old and three registered new TRAIN shards.
   Run matched C0/C1, seeds 10601/10602, at 600 fixed steps. Both arms receive
   identical calibration, primary replay, architecture, optimizer and features;
   only the contrast auxiliary differs. Resource-only cadence changes must
   preserve equal counts for all six families (cadence 4 gives 25 each).
   TRAIN-only gradient conditioning may change its coefficient; comparison
   against older pilots is therefore diagnostic, not a pure coverage estimate.
6. Verify full 615-state independent NumPy/native export parity, probe frozen
   TRAIN decisions, then run the same four cheap actual-policy controls using
   all49/native64 responses. Advance only on actual playing benefit without
   known-regression failures. Otherwise reject/inconclusive, diagnose, and
   preserve the retained model. Full-hand and serving work remain conditional.

Resource limits: two native workers total, 2GiB per worker, 90-minute labeling
cap with a separate three-minute finalization allowance; first two turns per
family establish cost before complete labeling. New artifacts stay below
10GiB with 20GiB free disk reserve. Fit one MLX process at a time within 6GiB
and two hours including conditioning/parity. Honor system memory pressure.
No paid compute, UI changes, or model promotion in this intervention.

### Section 11 completion

[The coverage report](postflop-training-coverage-pilot-2026-10-04.md) records
all three complete native label shards, old-model diagnosis, both matched
600-step pairs, full-615-state export parity, and six-family frozen probes.
New C1 reduces ranking loss on the new TRAIN block by 57.5% and improves the
first known policy control from the prior C1's 0.418741bb to 0.350730bb.
However, retained is 0.278335bb on that control. C0 and C1 both exceed the
0.01bb known-case regression limit, so remaining evaluations were stopped
without scoring partial cases or inventing paired means. **Reject both; no
conditional full-hand/serving expansion.** All pilot processes have stopped.

The same remaining continuation-ranking problem is measurable from cached
own-policy packets. Old TRAIN accuracy also worsened with the six-family
schedule. Next distinguish representation/target mismatch from retention:
hash-matched input diagnostics, then a bounded retained-weight fine-tune with
protected old replay versus the from-scratch control. This proposed next pilot
is not started, not a paid-run recommendation, and not a guaranteed fix.
Use the new optional control-first automatic rejection to avoid spending the
rest of a screen after a completed audited case already fails its fixed limit.
