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

## 12. Retained-initialization isolation (October 4, in progress)

First change only initialization: warm-start seeds 10601/10602 from the
retained benchmark pair with a fresh AdamW optimizer. Reuse Section 11's
immutable scratch C1 pair as the matched control; do not repeat unchanged fits.
Keep its exact six bounded TRAIN bundles, frozen 474/69/72 split, 600 final
steps, batch 8, 0.0003-to-0.00003 learning rate, cadence 4, chunk 4 and original
TRAIN-conditioned contrast coefficient 1.883306130920223. Do not select a
checkpoint or recondition its coefficient using response scores.

Validate the wide/v3/full-stack/payoff-exposure JSON contract and all parameter
dimensions/finite values before transferring any weights. Compare imported
MLX predictions against independent NumPy on two frozen TRAIN states, every
private combo and both players; validate the finished students against native
inference on all 615 states. No optimizer state or regrets are imported.
The retained sources themselves are research-only, not accepted GTO models.

Run the consumed high-rainbow/monotone four-case actual-policy control first,
stopping after any independently audited >0.01bb regression. No positive
partial case can accept a policy. If useful and all controls pass, proceed to
the existing broader qualification sequence. If initialization alone fails,
test a TRAIN-only retained-prediction protection constraint, distinguish it
from the already failed replay-only intervention, then diagnose representation
or target limitations. Advance based on actual policy outcomes, not RMSE alone.

Use one MLX process within 6GiB and a two-hour fit/parity cap; at most two native
workers, system-pressure stops and a 20GiB free disk reserve. No paid compute,
serving-route changes or automatic model promotion. Continue to the next
supported intervention without awaiting user permission for each local pilot.

### Numerical follow-up and research basis

The new import preflight exposed an additional, measured numerical mismatch:
with MLX 0.32.0's default GPU arithmetic, a retained-network serving prediction
differs from independent NumPy by 0.015009bb on a cached input. Tower errors
are already present before the value projection. [MLX's precision documentation](https://ml-explore.github.io/mlx/build/html/usage/precision.html)
explains that float32 matrix multiplication can use reduced precision, and
documents `MLX_ENABLE_TF32=0` as the full-float32 switch. Do not silently change
training precision inside an initialization-only comparison. Verify import on
full-float32 CPU, preserve the prior training arithmetic, then compare a
separately pinned full-float32 GPU fit with the same warm-start arm. Measure
parity and throughput before committing to the same fixed 600-step schedule.
Numerical agreement is necessary bookkeeping, not an exploitability result.

If old behavior still degrades, a frozen-output retention constraint is an
adaptation of [Learning without Forgetting](https://arxiv.org/abs/1606.09282),
not a new source of poker truth: penalize value drift on original TRAIN-only
inputs while retaining native labels for learning. This differs from the
already rejected sampling-only replay pair. It can preserve old errors as well
as old strengths, so only native actual-policy results justify advancement.
[DeepStack](https://arxiv.org/abs/1701.01724) motivates accurate range-conditioned
continuation values; its continual-re-solving guarantees do not transfer to
our current unsafeguarded route merely because a local fit improves.

## 13. Frozen-output retention constraint (October 4, in progress)

Both initialization and full-float32 arms failed the first audited control.
Full precision improved that one conditional response gain from 0.359659bb to
0.347476bb, but retained remains 0.278335bb. Stop both screens; there are no
paired means or new full-game exploitability results. Keep full float32 for
the next comparison because it removes the demonstrated arithmetic mismatch.

Change only frozen-output retention relative to the completed full-float32
pair. Anchor original-prefix (`group < 508`) TRAIN states with positive joint
reach; exclude tuning/holdout, appended states and all new board shards. Freeze
the corresponding retained network's normalized outputs. Apply the same
Huber/depth auxiliary form using authentic joint-reach weights, not a new
solver target. Do not replace native labels or anchor zero-joint states.

Use a separate deterministic RNG so the existing training draws stay identical.
Add an eight-state anchor gradient every four steps, combined into the existing
single optimizer update: 150 anchor updates and 1,200 draws per fixed 600-step
seed. The six native calibration/contrast bundles still receive 25 updates
each. Do not retain the reference network as a live trainable submodule.

Before fitting, choose one common coefficient from frozen TRAIN gradients at
the full-float32 seed-10601 step-200 checkpoint: 0.5 times the learning-gradient
norm divided by the anchor-gradient norm, capped at 32. Conditioning includes
the ordinary loss and equal-family mean calibrated contrast gradients; no
response scores, holdout errors or checkpoint selection enter this rule. Pin
the checkpoint and resulting settings before fitting either seed.

Run full-corpus parity and the same automatically rejecting actual-policy
controls. If useful, expand the policy screen; otherwise reject the retention
arm and investigate a function-preserving richer range representation. No
automatic promotion or longer unchanged training run. Existing local resource
caps apply; no paid compute. This constraint tests retention, not equilibrium
safety, and may preserve the retained model's own mistakes.

## 14. Align contrast optimization with bounded serving (October 4)

Section 13 completed all fits/parity but failed the first audited control:
0.350971bb versus retained 0.278335bb. Do not tune its coefficient against
that response or repeat an unchanged fit. The subsequent hash-matched TRAIN
probe measured optimization/serving action-contrast RMSE as high as 0.198796bb,
authentic value differences up to 0.636126bb, and local ranking changes.
Serving occasionally improves rankings, so this is a contract discrepancy,
not proof that it explains the policy failure.

1. Expose the unchanged network's raw pre-projection values without altering
   its existing value-calibration forward or exported inference contract.
2. Implement the same legal-card mask, +/-20bb clipping, native 1e-9 joint
   denominator and bounded zero-sum correction used by serving. Vectorize the
   80-step bisection over a complete family to avoid thousands of Python loops.
3. Use its implicit derivative with respect to raw values, including both
   clips and the cross-player dependence of the weighted target. Bisection's
   discrete branches themselves are not a useful autodiff program. Verify
   native forward parity, zero/tiny joint cases, directional finite differences
   and the real two-pass network-parameter gradient before fitting.
4. Change only the contrast forward/VJP relative to Section 13. Retain all
   ordinary/calibration losses, data/split, architecture, seeds, full float32,
   600 fixed steps, native bundle cadence and coefficients. Reuse its frozen
   retention settings (0.42899030580264763); do not recondition them. Raw value
   calibration remains active so a clipped auxiliary cannot prevent recovery
   of oversaturated predictions.
5. Cost/parity-preflight the largest TRAIN family at retained initialization.
   Require complete private-vector agreement within 0.0001bb and a projected
   pair below two hours. Do not quietly lower counts or alter the matched game.
6. Verify all-615 export parity, then the same control-first actual-policy
   screen with automatic rejection. Verify the original training/serving
   mismatch is removed *for the new contrast forward*, not by pretending the
   intentionally unbounded calibration forward changed.

Keep the two-native-worker, 6GiB fitting, pressure and disk guards. No paid
compute, serving changes or automatic activation. If alignment alone fails,
use targeted own-policy diagnostics before choosing a range-representation or
target-coverage intervention; isolated correct gradients cannot certify GTO.

## 15. Function-preserving range augmentation (October 5)

Section 14 fixed contrast-forward parity (maximum 0.0000001413bb), but its first
control regressed retained by 0.065077bb and was rejected. Cached own-policy
replay isolates the large after-check ranking error: 0.269730bb local ranking
loss and 0.748897bb contrast RMSE; all-in branches match exactly. This motivates
a representation test, not another unchanged fit. It does not prove that
representation rather than labels/coverage is the dominant cause.

1. Add an explicit research-only wide-to-wide-pooled transfer. Validate the
   original v4 wide/v3/20bb contract completely before mutation. Keep both
   encoder towers, biases and final head identical. Expand the first head's
   128 inputs to 256: copy context columns to 0:64 and query columns to
   192:256, with new own/opponent pooled columns 64:192 exactly zero. This
   preserves the original function, with newly available trainable range
   inputs. Unsupported transforms must fail closed.
2. Prove raw/training/serving forward equivalence, zero added columns, nonzero
   gradients into those columns, and independent NumPy/native v5 export parity
   before fitting. Native already supports this research v5 contract. Use
   joint-reach-weighted pooling with the existing native 1e-9 denominator.
3. Change only this representation relative to the completed aligned arm:
   frozen retained source weights, seeds, data/split, six bundles, coefficients
   1.883306130920223 and 0.42899030580264763, full float32, fresh AdamW, 600
   fixed steps, cadence/chunk 4 and identical reference/anchor sampling.
   Preserve the non-pooled reference network; never train on consumed controls.
4. Preflight the largest TRAIN family and full-615 initial exported v5 parity;
   require <=0.0001bb and projected two-seed fitting/parity <=2h, 6GiB worker,
   two-native-worker response and existing memory/disk guards.
5. Run full-615 finished parity and control-first actual-policy screening. The
   retained regression limit remains 0.01bb; a first failed completed case
   stops remaining work. If all matched controls improve credibly, expand to
   the broader frozen screen before any full-hand qualification. No promotion
   from value RMSE or an isolated conditional root.
6. If this fails, use the resulting frozen replay to discriminate label/profile
   mismatch and off-policy belief coverage; do not repeat earlier scratch
   pooling or increase unchanged iterations automatically.

Research: [Net2Net](https://arxiv.org/abs/1511.05641) motivates function-preserving
transfer; [Deep Sets](https://arxiv.org/abs/1703.06114) motivates learned
permutation-invariant aggregation. This weighted finite representation and
zero-column transfer are adaptations, not the papers' poker results or GTO
guarantees. Earlier scratch pooling gave mixed facing gains and worse root
rankings; retaining the starting function is the specific new hypothesis.

## 16. Diagnose interference before gradient projection (October 5)

Section 15 completed and failed the first control (0.352513bb versus retained
0.278335bb), despite improved holdout value RMSE. A-GEM/PCGrad are plausible
only if the relevant objectives interfere. Measure gradients on both frozen
aligned-model seeds at steps 0, 200, 400 and 600, on TRAIN only. Compare primary,
mean bundle calibration, weighted contrast and their combined active-bundle
direction with the 366 original positive-joint states' authentic native-value
gradient. Include ordinary calibration weighting as a separate diagnostic.
No new fit, labels, evaluation scores or checkpoint selection.

Result: all eight combined cosines are positive; no checkpoint meets the
predeclared <-0.2 conflict screen. Original authentic TRAIN loss decreases
45.2%/40.2%. The requiring-both-seeds-degrade-by-5% criterion fails as well.
**Do not implement or fit projection from this evidence.** Some individual
contrast gradients conflict, but the combined direction does not. This does
not exclude stochastic or region-specific interference. Raw-gradient geometry
is also not an AdamW-step guarantee; any later constraint would need to inspect
the actual preconditioned/momentum update rather than promise safety from raw
gradient projection.

Sources: [A-GEM](https://arxiv.org/abs/1812.00420),
[PCGrad](https://arxiv.org/abs/2001.06782). Probe artifacts:
`preflop-solver/neural/runs/local-training-gradient-conflict-20261005-a/`.

## 17. Test policy-induced calibration shift before dataset aggregation

Earlier frozen-belief probes improved rankings at retained beliefs while the
new policy's own response worsened. The new models' improved TRAIN/holdout
regression but persistent own-policy ranking error makes induced-belief shift
a live hypothesis. [DAgger](https://arxiv.org/abs/1011.0686) motivates relabeling
learner-induced states; its supervised sequential-learning guarantees do not
transfer directly to adversarial poker or unsafe re-solving.

1. Pin the completed aligned wide pair, existing three registered TRAIN roots
   100/101/102, their old 16-state captures, root hashes, trunk/sampling seeds,
   native64 labels and 128 flop updates. Use every registered family; no
   score-based choice or reuse of control/holdout boards.
2. Generate only 16 new stratified early/middle/late/final-average native labels
   per family with the corresponding current model. Preserve the exact root,
   seed stream and counts. Verify policy-observation parity. Native labels,
   not model predictions, remain truth for this finite-budget test.
3. On both old and new captures, evaluate the same current predictor using the
   actual native inference and independent NumPy parity. Report equal-state
   authentic RMSE, per-band error, and native label response residuals. These
   sampled leaf errors do not establish action ranking or exploitability.
4. Use a 30-minute stage cap, 2GiB native and 6GiB analysis workers, 20GiB disk
   reserve and memory-pressure stop. Time the first family and reject a cost
   projection above the remaining cap; do not silently reduce labels.
5. Dataset replacement becomes worth a matched pilot only if error on current
   induced states exceeds the old-distribution error by >=10% in at least two
   of three family/seed pairs. Inspect native-label residuals before blaming
   the model. If the screen fails, do not launch a DAgger-style fit merely
   because data aggregation is fashionable.
6. If supported, replace (do not duplicate) the 16 calibration-only extra
   states per new-family bundle, preserving its original all49 affine targets.
   Pin the foreign proposal identity and exact root independently; new states
   have zero affine coefficients and cannot be mistaken for the original
   profile's action targets. Preserve counts, data split, wide architecture,
   weights/optimizer initialization, frozen coefficients and 600-step budget.
   Require finished export parity and the same rejecting policy screen before
   broader confirmation. Mixed-policy calibration provenance must be explicit.

Completed in 147.2s, with all 48 new labels and all six independent/native
prediction comparisons. Current/old authentic RMSE changes are +1.7%, +0.7%
and -1.1%; zero families reach the +10% criterion. Dataset replacement is
**not supported**, so no aggregation fit follows. Current native label mean
response residuals are 0.00447–0.00746bb, versus model RMSE 0.724–1.038bb;
these are different metrics, not an error decomposition or proof of exact
labels. See `postflop-objective-diagnostics-2026-10-05.md`.

## 18. Inspect decision-gradient allocation before changing the auxiliary

Two cheap diagnostics failed their prespecified intervention screens. Do not
force projection or dataset aggregation anyway. The remaining learned-value
error is large and existing affine TRAIN decisions still have wrong rankings.
The next hypothesis is objective allocation, not longer unchanged training.

1. Pin all six existing TRAIN bundles and both aligned-model seeds at retained
   initialization and the fixed 600-step endpoint. No evaluation boards,
   checkpoint selection, native labels or fitting.
2. Decompose the exact existing unique-pair Huber derivative into already
   correctly ordered pairs, inverted pairs, and near-tie pairs. Retain the same
   authentic/profile-consistent weights, complete chance integration, 20bb
   normalization and bounded-serving VJP. Check that components reconstruct
   the original derivative exactly. Measure pairwise derivative allocation and
   native ranking losses, not only value RMSE.
3. At the endpoint, compare the original contrast parameter gradient with a
   bounded-margin, native-gap-aware ranking direction. Do not call a ranking
   auxiliary exploitability descent: it lacks best-response policy gradients
   and game-wide guarantees. Near ties below 0.05bb provide no ranking signal;
   keep ordinary native-value calibration to preserve EV and indifference.
4. A matched auxiliary pilot is warranted only if both seeds retain >=0.03bb
   equal-group TRAIN ranking loss and >=75% of the exact pairwise Huber
   derivative magnitude comes from already correctly ordered pairs. Inspect
   parameter-gradient agreement too; cancellation/representation may defeat
   the allocation hypothesis. A failed screen means no ranking fit.
5. If supported, freeze the alternative auxiliary and coefficient from TRAIN
   gradient norms before fitting; change only that auxiliary relative to the
   completed aligned control. Preserve weights, optimizer, primary/bundle
   data, split, seeds, 600 steps, retention, precision and cadence. Verify the
   derivative numerically and by an actual network VJP, full export parity,
   then the same automatic-rejection actual-policy response screen.

Research: [Exploitability Descent](https://arxiv.org/abs/1903.05614) optimizes
policies against best responses. It helps distinguish actual worst-case policy
optimization from this proposed supervised ranking surrogate; its convergence
claims do not apply to a leaf-value loss. No automatic cloud spending or
website/model activation is authorized.

Section 18 completed: endpoint TRAIN ranking loss is 0.063204/0.069243bb,
down from 0.215331/0.198206bb. Correctly ordered pairs receive 75.54%/73.62%
of pairwise derivative magnitude. The requiring-both-seeds >=75% screen fails;
**no ranking-auxiliary fit follows**. Original/margin parameter-gradient
cosines are -0.052/-0.068 at the endpoints, but different directions do not
prove policy benefit. The range-scale invariance hypothesis was also checked:
native inference already normalizes raw ranges and its qualified Rust test
passes; no augmentation/normalization patch is needed.

## 19. Measure a cheaper native-continuation construction budget

Native64 leaves have the strongest measured actual-policy effect (selected
32-update root mean 0.7062bb learned versus 0.2592bb native), but 20–33 minutes
per construction limits offline coverage. Test the existing native4 budget
before building a larger neural architecture or buying compute. This is an
offline generator cost/quality comparison, not a proposal to serve native4.

1. Pin the completed October 2 matched32 comparison, original benchmark roots,
   binary, input hashes and cached native64/learned32 candidates/responses.
   Use the same two predeclared roots and both chance seeds, in original order.
   No new fit, parameter tuning or untouched-validation claim.
2. Construct native4 candidates at the same 32 flop updates and one leaf worker.
   Change only the inner continuation update budget. Preserve an immutable
   raw training candidate. Export a separate evaluation candidate with explicit
   response_turn_iterations=64, proving all policy probabilities unchanged.
   Construction and played/evaluation budgets must never be conflated.
3. First time the limped seed-100101 construction; retain the existing 2.5GiB
   serial worker guard. Target <=20% of its cached native64 construction time.
   Project the remaining complete comparison with margin and a two-hour cap;
   stop if it cannot fit. Do not silently reduce chance/evaluation counts.
4. Evaluate each completed policy using native64 played continuations, all49
   exact turns and the independent JavaScript accounting/backup audit. Limit
   packet concurrency to two workers, retain pressure/disk stops, pin every
   artifact and source. No score from partial packets.
5. Stop after an audited case if its gain exceeds native64 by >0.05bb or retains
   less than half the native-versus-learned32 improvement. Advance only after
   all four complete comparisons and cost/quality checks pass. This pragmatic
   generator screen is not an Approximate GTO release gate. If rejected, use
   sampled CPU attribution to identify reusable work in the accurate native
   solver rather than scaling failed student fits.

Research: [Value Functions for Depth-Limited Solving](https://arxiv.org/abs/1906.06412)
ties useful depth-limited play to suitable value functions and reports limited
benefit from its explored loss variations. Together with this repository's
matched native-versus-learned result, it motivates addressing accurate-target
cost directly. It does not guarantee a four-update continuation is sufficient.

The first native4 construction was interrupted at 43.943s by the original
2.5GiB physical-footprint guard (sampled peak 2,703,821,728 bytes). No policy or
quality score exists from that attempt. An explicitly separate memory-only
retry permits 4GiB for the one serial constructor, leaving the same two
1.5GiB evaluation workers, system-pressure/disk stops and unchanged cost and
quality thresholds. The original failure remains immutable. A three-second
macOS CPU sample attributes time to repeated training/profile walks, compatible
mass calculations, policy serialization and SHA-256; this is not evidence of
a leak, nor a measured whole-run percentage. No target/algorithm changes are
made by this retry.

The retry completed in 205.690s (9.51x faster construction) at 3.513GiB
sampled footprint. The projected full response comparison was 13,842.9s,
so it stopped at the two-hour stage cap without a quality score. Do not
confuse passing construction speed with passing the generator pilot.

## 20. Reduce native allocation overhead, then stage the quality screen

1. Preserve the completed native4 raw candidate and all failed-attempt records.
   Inspect one owned CPU/memory snapshot rather than increasing iteration
   counts or launching another unsupported student fit.
2. Stream the exact native policy JSON into SHA-256 with a bounded 64KiB
   buffer. Leave all mathematical operations, policy rows, export rounding and
   best-response accounting unchanged. Compare the new hash with buffered
   export bytes, then replay a pinned native64 turn serially before/after and
   require complete byte parity with the cached original control.
3. Separate native canonical export from Python manifest writing. A new
   response-budget candidate must round-trip to its exact Rust identity and
   recover every original training byte when the override is removed. Reject
   cosmetic JSON identity drift; never bypass the native reader's check.
4. Stage only the original limped seed-100101 quality control first. Reuse the
   completed native4 construction, explicitly export native64 evaluation,
   and solve all49 turns with two guarded workers plus the independent JS
   audit. Retain the two-hour cap and original >0.05bb/half-benefit rejection
   rules. Resume only identical interrupted jobs. This first control does
   not establish a paired mean or full-game quality.
5. Reject on a complete failed quality screen. If it passes, use actual new
   packet timings to plan the remaining seed/root comparisons; do not expand
   a run from a stale optimistic estimate. Require all four before accepting
   a generator budget, and separately validate any subsequently fitted model.

Step 2's first frozen packet passed exact parity, using 621MB buffered versus
336MB streamed footprint. Time was essentially unchanged (54.5/54.0s). The
first step-4 launch caught noncanonical Python candidate JSON before solving;
step 3 now has a Rust exporter and a regression test. This is a research
controller fix, not a website or solver-policy improvement. The updated
binary's packet parity passed and the complete first-control screen is running.
