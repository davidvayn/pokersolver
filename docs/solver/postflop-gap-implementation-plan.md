# Implementation plan: close the postflop policy gap

Date: September 22, 2026 (PDT).
Status: **Planning only. No new training, solver implementation, or deployment authorized by this document.**

## 1. What the evidence actually identifies

The immediate priority is **flop decision quality and the future values used to choose those decisions**, not another river-only rewrite or an unchanged overnight training run.

The completed benchmark covers twelve fresh 20bb postflop roots, two retained value-model/solver-seed pairs, 128 flop updates, and native 64-update turn/river continuations. Its mean conditional exploitability is **7.7427% of the starting pot**, or **0.334365bb per selected postflop root**. It is not full-game bb/hand exploitability and did not evaluate the deployed website model.

Read-only analysis of the existing audit outputs adds an important localization:

| Pot type | Full postflop response gain | Response changing only flop actions |
| --- | ---: | ---: |
| Limped | 0.261890bb | 0.205359bb |
| Single-raised | 0.389425bb | 0.346009bb |
| Three-bet | 0.351779bb | 0.333421bb |
| Equal-weight mean | **0.334365bb** | **0.294930bb** |

An attacker who changes flop actions but then follows the frozen continuation already recovers **88.2% of the mean bb gap**. The corresponding mean percentage-of-pot figures are 7.7427% and 6.4703%; the 88.2% ratio uses bb, not percentages. This is a restricted-response comparison, **not an additive attribution of error to streets**. Bad continuation policies can still make earlier decisions wrong.

Specific reproducible symptoms:

- `single-raised-high-rainbow`: after BB checks, BTN checks only **1.944% / 1.528%** across the two seeds. A per-hand, single-node deviation against unchanged later play gains **0.9860 / 0.8019bb conditional on that node**. This does not mean every holding should check, nor that the replacement frequencies are equilibrium frequencies.
- `limped-paired`: BB's initial check frequency varies **66.33% / 93.95%**. Value-model seeds and chance seeds vary together, so their individual contributions to this instability are unidentified.
- The retained network was fit on **508 turn-state targets** (367 train, 69 tuning, 72 holdout), with a maximum 600 optimization steps. Its held-out authentic value RMSE was **1.4724 / 1.4608bb**. Each state contains many private-hand values; this is not merely 508 scalar labels. Neither the dataset count nor RMSE alone proves the cause of the policy gap.
- Native and independent JavaScript flop backups agree on the largest-bb case within **4.44e-16bb**. Both share native continuation values and terminal equities: this reduces the likelihood of a flop accounting error but does not independently validate the whole poker engine.

Small pots look worst in percentage terms partly because the denominator is smaller; single-raised pots have the largest average absolute bb loss. Prioritize both, not only the largest percentage.

### Why the commercial comparison looks so large

PokerCortex cites Deepsolver's historical **0.6%-of-pot mean over 430 spots**. Our 7.74% is clearly a poor result in our own game, but different ranges, bet trees, stacks, cases, and evaluators prevent a defensible claim that our app is a specific multiple weaker. No matching commercial strategy outputs were supplied. [PokerCortex methodology](https://poker-cortex.com/methodology).

The plausible technical explanation is a combination of inaccurate action-relevant continuation values, limited coverage, finite solving, and reconstruction at street boundaries. **Their causal shares have not been measured.** DeepStack's theoretical analysis separates a value-error term from a finite-iteration term: more iterations cannot remove arbitrary value-function error. That result motivates the experiments below; its guarantee does not automatically apply to our current resolver. [DeepStack, Theorem 1](https://arxiv.org/html/1701.01724v3).

## 2. Ranked hypotheses and discriminating tests

| Priority | Hypothesis | Prediction that would support it | Evidence against it |
| --- | --- | --- | --- |
| 1 | Learned continuation values rank flop actions incorrectly | Replacing learned values with native values at identical beliefs changes costly action rankings; a matched native-leaf solve improves actual response gain | Learned and native rankings agree, yet both produce the same losses |
| 2 | Flop search is under-converged or excessively noisy | Increasing only flop updates consistently reduces native-evaluated response gains across paired seeds | More updates reduce surrogate regret but leave actual gains flat or worse |
| 3 | Training coverage/objective misses decision-critical beliefs | Targeted, training-only action-contrast supervision improves native action loss and fresh-board response gains | Better regression scores do not improve decisions or responses |
| 4 | Finite native labels and street reconstruction give inconsistent targets | Higher-budget labels or a coherent continuation contract change the costly action values and improve the assembled policy | Labels stabilize and boundary-consistent policies retain the same gap |

Do not call any of these a confirmed root cause until its controlled test supports it. Separate value fitting error from error in the finite-budget native reference itself.

## 3. Step-by-step implementation sequence

### Step 1 — Reuse existing diagnostics to identify the expensive mistakes

1. Freeze the completed benchmark, model hashes, preflop ranges, game abstraction, and evaluation definition as the baseline. Never overwrite its policies or packets.
2. Declare these consumed boards development cases: `limped-paired`, `single-raised-high-rainbow`, and `three-bet-monotone`. Use `three-bet-high-rainbow` as a lower-gap regression control. They are deliberately selected, not a representative validation sample.
3. Reuse `audit_native_flop_response.mjs` and `native_action_diagnostics.mjs` to rank public nodes and holdings by root-reach-weighted local loss. Do not add local losses across overlapping nodes and label that sum exploitability.
4. Adapt the existing `run_native_action_value_probe.py` to accept benchmark cases. Export learned predictions on exactly the frozen beliefs used by the saved native packets; compare **action EV differences**, not just absolute value errors. Integrate future cards with exact card removal before choosing a flop action.
5. For the most costly decisions, capture a small number of actual search-iteration beliefs as well. Final-average beliefs alone cannot reveal whether the network misled earlier regret updates.
6. Cross the two retained network seeds with the two solver chance seeds on the weakest development root. This small 2×2 comparison separates model variation from sampling variation; do not select whichever seed wins.

**Deliverable:** one compact attribution report identifying the affected actions and whether the learned ranking, sampled optimization, or reference target disagrees. Reuse cached native data wherever the exact input identity matches.

### Step 2 — Separate iteration limits from continuation-value limits

Run one factor at a time, initially on the limped-paired and single-raised-high-rainbow roots, then confirm promising changes on the other two development cases.

**A. Flop-update test.** Keep weights, chance generator, averaging, legal tree, and played 64-update continuations fixed. Compare 128 with 256 flop updates. Evaluate every candidate against the same native response procedure. Extend to 512, then 1,024 only while paired response gains continue to improve. Preserve optimizer state for genuine continuation; an exported average policy is not a resumable training checkpoint.

**B. Leaf-value test.** Compare the existing learned and native leaf adapters with the same flop updates, seed stream, and native reference budget. Preflight native cost at eight updates. Use 128 updates if the bounded pilot fits; otherwise compare both arms at 32 and label the weaker test explicitly. Never compare native32 with learned128 and attribute the difference solely to leaf quality. Native values must be recomputed for current ranges: a cached CFV vector at an old range is not a valid replacement.

**C. Native-target quality test.** On a small fixed set of costly beliefs, compare native 64, 256, and, only if needed, 1,024 updates. Record each native policy's own conditional response gap and action-value drift. More iterations are not themselves proof of accurate labels. If the 64-update teacher is inaccurate enough to reverse relevant rankings, improve reference labels before fitting them more closely.

A and B retain the same played continuation budget to isolate flop changes. Raising the played continuation budget is a separate candidate change, not an allegedly stronger evaluator of the same fixed policy. A frozen policy's best-response evaluator must not silently replace that policy while scoring it.

**Decision:** choose the intervention supported by actual response gains. If A works, extend the unchanged promising solver. If B works and A plateaus, prioritize Step 3. If both native and learned arms remain poor, investigate native convergence and the boundary contract before buying more network training.

### Step 3 — Repair decision-relevant value errors, if Step 2 supports it

1. Preserve the retained weights as a control. Collect current-search and explicit one-player-deviation beliefs from training-only board families, using the existing belief-capture machinery. Cover small-pot checks, bets, calls, and raises identified in Step 1; do not simply append more examples of one favorable root.
2. Generate native targets with the quality budget selected in Step 2C. Keep finite-budget quality metadata and exact public/private-card semantics. Do not use the network's own predictions as independent native labels.
3. First test **data/label quality alone** with the retained architecture and objective. Then separately test a decision-directed auxiliary objective on the same data and training budget, so improvements can be attributed.
4. For the auxiliary, back up predicted leaf values through a frozen training prefix and compare chance-integrated action contrasts, such as `Q(call) - Q(fold)` or `Q(bet) - Q(check)`, with native contrasts. Keep a calibrated value loss alongside it. This is a proposed experiment, not a proven algorithmic improvement. Maintain opponent-reach scaling; never multiply a CFV by own reach a second time or give an earlier action knowledge of a future card.
5. Weight decision examples by authentic joint reach and meaningful native loss, with a bounded allocation for unilateral deviations. Retain legal zero-own-reach coverage without changing the underlying ranges.
6. Train paired students, check native/Python inference parity, then compare actual policies under the same response evaluation. Use tuning data for selection and new board families for confirmation. Lower RMSE, KL, or action disagreement alone is not a promotion.

ReBeL motivates sampling beliefs encountered during iterative search, including unilateral exploration; the repository already has this machinery. The proposed change is how coverage and action-relevant supervision are used, not a claim to have newly invented range conditioning or implemented ReBeL. [ReBeL, Section 5.2](https://arxiv.org/html/2007.13544v2).

**Deliverable:** a paired candidate with improved native action loss and conditional response gain, or an explicit rejection. Expand corpus size or architecture only after a sample/learning curve identifies the limiting factor.

### Step 4 — Address street-boundary inconsistency only where needed

If stronger native leaves do not translate into a good assembled policy:

1. Trace one high-loss line end to end: iteration ranges → predicted/native CFVs → frozen flop average → reconstructed turn policy → evaluated response. Compare current-policy versus average-policy beliefs and document where values or policies change.
2. Keep the existing card removal, true policy reaches versus estimator reaches, payoff units, and zero-reach completion contracts explicit. Reuse their regression tests; do not repeat already-fixed accounting work without a new failing case.
3. Choose **one** bounded structural pilot:
   - **Preferred when reconstruction is the issue:** carry opponent per-hand counterfactual bounds from the preceding policy and use a safe re-solving gadget. Validate on a small exactly solvable imperfect-information game, then the identified Hold'em lines. Approximate/self-play values are not automatically safe bounds, and protecting an already weak policy does not make it GTO. [CFR-D decomposition](https://ojs.aaai.org/index.php/AAAI/article/view/8810).
   - **Alternative when single continuation estimates remain brittle:** a small multi-valued continuation portfolio, allowing the opponent to select among coherent continuation strategies at information sets. Never maximize separately using the opponent's hidden cards or future chance outcomes. A small portfolio is a robustness experiment, not a full-game certificate. [Depth-Limited Solving for Imperfect-Information Games](https://proceedings.neurips.cc/paper_files/paper/2018/file/34306d99c63613fad5b2a140398c0420-Paper.pdf).

Do not implement both systems speculatively. Retain one only if its assembled-policy response gains improve at measured cost.

### Step 5 — Confirm generalization and decide whether paid compute is useful

1. Promote an experimental arm to confirmation only when the paired mean response gain improves across both seed pairings, without a material regression hidden in a pot-type/control breakdown. If seed results disagree, report inconclusive; add independent board evidence rather than pick the winner.
2. Re-run the twelve consumed spots as a regression suite, then a newly frozen twelve-family set spanning the same pot types and textures. Do not call the September 21 set untouched validation after tuning against it. Count boards, not correlated seeds or private hands, as independent sampling units.
3. Report mean, tail, and paired improvement in **both bb and percentage of starting pot**. Treat sub-1%-of-pot results as a descriptive postflop objective, not a newly imposed full-game release gate. Broad-action best response remains a later robustness check; do not narrow the action tree just to improve the score.
4. Buy compute only if measured update/label scaling reduces response gains and local throughput is the obstacle. Use it for independent board labeling, seed jobs, or demonstrated useful iteration extensions. More workers do not accelerate a serial regret chain automatically. No server provisioning is authorized by this plan.

**Deliverable:** an accepted research candidate plus its measured cost/quality curve, or a documented bottleneck requiring a different intervention. No promised convergence time or guaranteed cloud success.

### Step 6 — Evaluate the combined full-hand policy, then integrate it

1. Pin the accepted postflop candidate and the retained preflop policy into one complete-hand route. Test the actual assembled policy against existing information-set-consistent responders, including adaptive/full-hand deviations. Do not estimate whole-game exploitability by averaging these twelve conditional roots or by adding incompatible preflop and postflop scores.
2. Report the user's **0.50bb/hand target**, and later **0.05bb/hand aspiration**, using a clearly stated summed-versus-half-summed convention and responder scope. A response learner or LBR gives evidence of an exploitable lower bound, not proof of a global upper bound. A small observed gain cannot certify that every stronger attacker fails.
3. Before publishing an Approximate GTO claim, explicitly state which exploitability guarantee is actually supported. Keep remaining coverage, probability, EV-confidence, runtime consistency, and serving checks visible; they were not requalified by this benchmark.
4. Only after candidate acceptance, wire the same pinned weights, action grid, and continuation contract into the website. Verify full-hand, preflop, postflop, and push/fold routes as applicable. No unrelated UI, database, or model-family overhaul is needed to reduce the present gap.

## 4. Work that must not be repeated without new evidence

The history in [the existing research plan](../../plan.md) already records:

- Path A's 508→636 state extension improved RMSE but worsened paired conditional response gain by about **11.5%**. Do not repeat it unchanged.
- The per-player value-bias penalty worsened both seed responses. Do not treat it as the unimplemented action-contrast loss.
- Learned range pooling gave mixed responses and worse rankings on another root. Larger/pooled networks are not an evidence-backed first step.
- Two boards per shared preflop endpoint with fewer updates did not give reproducible paired improvement. That was a preflop sampling experiment, not a test of this plan's flop-value versus flop-update comparison.
- Exact preflop averaging, variance baselines, card-removal checks, and extensive response auditing already exist. Reuse them. No new preflop overhaul, NFSP rewrite, or unrelated metric project is part of the initial intervention.

## 5. Implementation locations, tests, and resource boundaries

- Diagnostics/orchestration: `preflop-solver/neural/run_postflop_benchmark.py`, `run_native_action_value_probe.py`, `audit_native_flop_response.mjs`, `native_action_diagnostics.mjs`.
- Solver/value seam: `preflop-solver/src/blueprint/public_belief/counterfactual_turn/flop_pilot.rs` and its `continuation.rs`, `search_beliefs.rs`, `value_targets.rs`, and `frozen_response/` modules.
- Fitting/contracts: `preflop-solver/neural/train_public_value_network.py`, `native_value_dataset.py` and adjacent tests.
- Tests must exercise actual callers: native/learned boundary parity, current-range cache identity, chance integration before maximization, action-contrast gradients, leakage-free board-family splits, frozen-policy evaluation, and any new bound propagation. Add only tests relevant to the intervention.
- Run targeted tests while implementing; run `cargo test --release` in `preflop-solver/` for native changes. Run `npm test`, `npm run build`, and desktop/mobile browser checks when the eventual website integration changes the app. A document-only commit does not need those builds.
- Cost-preflight each new expensive arm. Keep checkpointed/resumable jobs, sampled memory/pressure guards, and a 20GiB disk reserve. Six-way concurrency was measured for the completed evaluation, not for every future workload. Start native-training arms conservatively; choose worker count from measured peak memory and throughput.
- Stop at the declared pilot budget if responses plateau/regress; preserve artifacts and choose the next evidence-backed branch. Do not launch a long run because a surrogate metric alone improves.

## 6. Evidence and reproducibility

Baseline artifacts (local/ignored): `preflop-solver/neural/runs/local-postflop-benchmark-20260921-c/`.

- Protocol SHA-256: `093296987e627c94c0da8ffe0fd90f9959109b9f0cf3be76840a9641675277e6`.
- Completed manifest SHA-256: `9a7f178b88fb595235c22e83f8fb173c9413d8692da34f6990a8a0540faf0ad8`.
- Audit manifest SHA-256: `8905e4afdad674fc3869b5c92f9ea94d36123a1124e67888362a774e44f60208`.
- Flop-only table: for each of the 24 `final-verification-20260923/*/worker.log` JSON outputs, take `sum(flopOnlyRestrictedGainBb)/2`; average equally within each pot type and overall. Compare with `half_summed_gain_bb`. These are ratios of averages, not averages of per-case ratios.
- Node examples: replay the existing auditor with `{actionDiagnostics: true}` on the named candidates; no native solves or policy changes are needed.
- Training provenance: `runs/local-native-value-20260907-student-ace-coverage-a/manifest.json` and the September 7–9 results in `plan.md`.

Fast symptom replay, from `preflop-solver/neural`:

```bash
node --input-type=module - <<'JS'
import assert from 'node:assert/strict';
import { audit } from './audit_native_flop_response.mjs';
const p = 'runs/local-postflop-benchmark-20260921-c/jobs/single-raised-high-rainbow/100101';
const r = audit(`${p}/candidate.json`, `${p}/packets`, `${p}/equity.json`, `${p}/response.json`);
const percentPot = 100 * r.half_summed_gain_bb / 5;
console.log({ percentPot, gapBb: r.half_summed_gain_bb });
assert.ok(percentPot <= 1, 'Above the 1%-of-pot diagnostic comparison target');
JS
```

Replayed twice during planning: **14.3418587486%**, **0.7170929374bb**, the same expected failing verdict each time; approximately one second for both replays together. The 1% assertion demonstrates the observed symptom, not a claim that this is the user's full-game gate. Minimization stops at one existing frozen root: removing legal branches or chance packets would change the measured game. Causal intervention, fix, and regression-test phases are deliberately deferred because the request is for a plan only.

Only this plan is to be committed for the present request; pre-existing benchmark/resource and UI-test working-tree changes are outside this document-only commit. No training, schedule, policy, or website was changed during planning.
