# Rake aware practice implementation plan

Status on October 7, 2026: cash accounting is connected to the pure hand engine and native own-payoff training, turn/river solving, evaluation, and an offline query adapter. Short NL25 20bb research policies and continuation networks have been trained and tested. **No cash model qualifies for activation.** The production website and existing Home models remain rake-free; an unavailable cash profile cannot fall back to them.

## Current implementation and remaining qualification

Implemented on October 6 and 7, 2026:

- Frozen Home and NL25 study rules in `data/practice/cash-game-rules.json`, a read-only profile registry, and matching canonical SHA-256 identities in both languages. Existing Home serving hashes remain unchanged.
- Exact integer-unit percentage rounding, capped rake, cent-aligned legal sizes, unmatched-wager refunds, split-pot allocation, and separate own-player net payoffs. The actual cash hand ledger conserves player chips, outstanding pot, and house rake. Early drill review does not settle an unfinished pot.
- Nineteen shared terminal fixtures, eight rounding fixtures, two sizing fixtures, invalid-input checks, and deterministic conservation sweeps. The online odd cent goes to the first seat left of the button as an explicitly marked study abstraction, not a verified cash-room rule.
- Rules, stack depth, and action abstraction bound to native checkpoints, average-policy exports, cash queries, continuation datasets, and cache captures. Legacy Home hashes remain unchanged. A versioned cash state hash has shared TypeScript and Rust preflop, flop, and river fixtures.
- Cash state keys now use `hu-cash-v3`: sort the flop only, preserving turn/river arrivals. Cash neural inputs use 820 state features plus nine action features, with exact turn/river one-hots appended to the unchanged 716-feature Home encoding. This repairs public-card recall without leaking future cards. Cash traversal schema v3 pins terminal-action expectation integration; weights schema v3 pins the new feature width. Older cash data/weights are rejected rather than silently reused. Home keys, 725-input weights, and serving artifacts remain unchanged.
- Native tabular and neural traversals use the traverser's own terminal payoff. Exact blocker-conditioned turn/river values and independent house-ledger evaluation replace scalar zero-sum conversions. Actor-only neural baselines are disabled for cash rather than negated. Old Home range, preflop-oracle, safe-resolving, and browser-weight routes reject cash.
- Separate Python/native own-payoff turn continuation contracts: exact forced-checkdown baseline plus a learned betting correction, independent clipping, and no zero-sum projection. Frozen CPU inference agrees with native predictions within about 0.000007bb. GPU inference differed by up to 0.0185bb on the same frozen weights; GPU training remains enabled, but serving-parity measurements use CPU inference.
- Cash regret bootstrap predictions also use CPU FP32 arithmetic. On 899 actual grouped poker features, GPU and CPU frozen predictions differed by up to 0.053606bb, while CPU and NumPy differed by at most 0.000010bb. Recycling inconsistent prior predictions into DCFR targets is avoided; GPU optimization remains enabled. This behavior is part of the resume digest.
- An inactive native cash query adapter reconstructs opponent beliefs from public actions, pins weights/rules/depth, and never accepts hidden opponent cards or a future deck. It uses frozen average frequencies without online updates, paired Monte Carlo EVs for open branches, bounded exact closed-all-in enumeration on the flop/turn, and bounded exact policy-tree enumeration on the river. Each action reports its evaluation method. Zero sampling error is not a claim of zero policy or belief-model error.
- Cash neural opponent responses whose every legal child is terminal now integrate the frozen action mix and the traverser's own terminal payoffs. Average-policy records remain intact, open trees still sample, and Home behavior is unchanged. Early all-in runout budgets remain explicitly separate; exact action integration does not make sampled preflop/flop runouts exact. The estimator is bound to cash dataset metadata and the resume digest.
- A bounded native sampler now captures authentic fresh-turn public ranges from a pinned scalar average policy. Both seats' reaches are computed over every board-legal combo using their own prior public-action likelihoods; actual sampled hole cards and the future river are not inputs to that range boundary. Native/Python root hashes preserve exact f64 ranges, prior public actions, rules, depth, and abstraction. These are fresh-subgame references, not values of the frozen full-hand continuation or a safety guarantee.
- Continuation training can pin one board-family split independently of network initialization. Training-corpus extensions preserve the original tuning/holdout prefix, exclude those families from added training contexts, and retain validated parent datasets and their authentic reach provenance. Native replay's adjacent float representations of the same cent are compared as integer units, while actual unequal/subcent investments still fail.
- A profile-specific paired-pilot audit pins completed run configs, average weights, full-hand reports and causal response reports; checks independently accounted own gains/house rake; and distinguishes failed, passed-in-scope, and unmeasured gates. It cannot activate a model. It records actual training duration, not intended overnight hours.
- The Home server now selects a pinned version/depth, snapshots each manifest's worker configuration, passes its action abstraction explicitly to Rust, and verifies decoded artifact and compiled-sidecar hashes before startup. Model/process and total core budgets are bounded together. A wrong version/depth cannot fall back to whichever model shares its stack depth. The legacy runtime still rejects cash; this is preparation, not cash activation.
- Non-destructive IndexedDB identity upgrade, rules/depth-scoped evidence and weaknesses, structural game settings, accurate cash settlement review, and unavailable-profile guards. Home defaults and current two-choice probability grading remain unchanged.

Remaining work is substantial: authentic nonuniform continuation coverage, cash preflop/flop and routed continuations, a stable full-hand policy, sufficiently precise earlier-street EVs, complete serving gates, a qualified cash runtime, and full browser acceptance. Per-manifest worker routing is implemented for qualified Home models; the production resolver intentionally remains Home-only. Offline adapter tests are not website activation evidence.

Bounded pilot findings:

- Two NL25 seeds and two rake-off controls completed 32 rounds and 1,024 traversals each, in tens of seconds rather than eight hours. The split/resumed NL25 run reproduces uninterrupted average weights, sample shards, checkpoint tensors, RNG state, and optimizer arrays bit for bit.
- Cross-seed full-hand frequency MAE is about 11–12 percentage points against a 5-point gate; primary-action agreement is 49–56% against 85%. The matched rake-off control also lacks stability. A favorable result against four fixed deviations is not full-game exploitability.
- The CPU-bootstrap-only pair reduced MAE to 9.5–9.9 points and raised primary agreement to 57–62%, but did not improve the stronger response test in both seeds. The fresh recall-v5 pair further reduced MAE to 7.92–8.17 points and raised primary agreement to 65.2–66.2%. Its largest aggregate action delta is 3.61 points on seed one's trajectories and 2.36 on seed two's, against the 3-point gate. These are improvements, not passes. Changing input width changes seeded initialization, so the v4/v5 comparison is not a controlled causal feature ablation.
- Within the same recall-v5 runs, round 16 to 32 changes the optimistic finite sample-game total response estimate from 5.446 to 4.387bb in seed one (paired change -1.059, SE 0.333bb), but from 4.881 to 5.315bb in seed two (change +0.434, SE 0.198bb). There is no credible two-seed improvement supporting a long extension. These are eight outer samples with 32 scenarios per seat; they are not exact full-game exploitability measurements. The corresponding 99% upper bounds remain 47.31/48.23bb, not below 0.50bb.
- Recall-v5 controls with identical 0.4/1 blinds but no rake have MAE 8.36–8.53 points and primary agreement 60.6–62.1%, so the stability problem is not specific to the rake deduction. The new split/resume audit reproduces all 32 decompressed shards, average weights, checkpoint tensors, optimizer arrays, RNG state, and replay maps bit for bit. NL25 training consumed about 37 seconds per seed and under 172MB peak process RSS, excluding compilation and concurrent processes. No eight-hour training evidence is claimed.
- A stronger information-set-consistent, finite-chance-game response comparison at the same budget gives no credible paired improvement from round 3 to 32. Its conservative 99% bounds are far above the 0.50bb total release limit. No long policy-training extension or paid compute purchase is justified by these pilots yet.
- Native query Monte Carlo errors at 4,096 samples remain approximately 0.11–0.26bb across five fixed snapshots. Exact closed-all-in branches enumerate 1,070,190 compatible flop outcomes or 45,540 turn outcomes, and the fixed river query enumerates 43,560 policy-tree nodes. These exact action rows have zero sampling error and still carry low model confidence. Seven of 24 action rows meet 0.02bb in each seed's snapshot set; this is an unweighted diagnostic, not 95% reach-weighted gate coverage. On the same frozen weights, the other flop actions' sampled EVs and standard errors remain bitwise unchanged.
- The terminal-action-integration pair completes 32 rounds/1,024 traversals in about 21–22 measured training seconds per seed, with peak process RSS below 186MB. Cross-seed MAE remains 8.12–8.22 points, primary agreement is 67.6–67.7%, and largest aggregate deltas are 3.44/3.77 points. Stronger finite sample-game means are 5.227/5.027bb, with 99% bounds 48.15/47.95bb. Against recall-v5 at the same evaluation seed, changes are +0.840bb (paired SE 0.359) and -0.288bb (SE 0.457): no credible two-seed improvement. The estimator removes a proven variance source but is not evidence of better policy quality. Its split/resume audit reproduces all 32 decompressed shards, frozen weights, checkpoint/optimizer arrays, RNG state, and all 22 replay files exactly.
- At 32 rounds, increasing fresh traversals from 32 to 128 reduces cross-seed MAE to 7.09–7.21 points and raises primary agreement to 71.4–72.9%. Fresh 32-outer-sample response comparisons give optimistic finite sample-game means of 4.276/4.531bb, versus 5.029/4.868bb for the terminal-integration reference on the same seed. Paired changes are -0.753 (SE 0.309) and -0.337bb (SE 0.247): directional, but the second seed is uncertain.
- The bounded extension to 64 rounds/8,192 traversals reduces MAE further to 6.77–6.84 points; primary agreement is 73.3–75.1%. Aggregate deltas worsen to 5.08/5.80 points. Response means are 4.298/4.410bb: changes from 32 rounds are +0.021 (SE 0.094) and -0.121bb (SE 0.106), not meaningful improvement in both seeds. Do not extend this configuration blindly. Its qualification report fails stability and the 0.50bb bound; the largest research upper bound is 25.87bb, not an unrestricted certificate. Actual training consumed only 0.0508/0.0598 hours per seed.
- Six exact small-pot turn/river references at 64 updates have conditional total deviation gains of 0.10–0.15bb. Combined with six near-all-in references, paired 500-step continuation fits reduce held-out error from 0.61/0.69bb at 100 steps to 0.31/0.17bb. The corpus still contains only six uniform-range flop families, not authentic full-hand coverage. A separate 50-step accounting-penalty experiment worsened held-out accuracy in both seeds and was not adopted as a default.
- Six authentic public-range turn roots improve from 0.51–2.04bb conditional NashConv at eight updates to 0.006–0.123bb at 64. A separate 24-root, 64-update corpus has conditional gains of 0.00005–0.163bb (mean 0.0928), with maximum accounting residual below 1e-14bb. Neither measures full-game exploitability.
- On the 24-root corpus, two independent 500-step value fits share split seed 937 and the same five held-out board families. Selected checkpoints are at steps 20/30. Held-out RMSE is 2.025/1.480bb versus forced checkdown's 2.378bb; maximum aggregate own-payoff bias is 1.187/0.909bb. Native parity is within 0.000005bb. These are improvements over checkdown, not usable serving values. Earlier six-root and uniform fits used different holdouts per training seed and are not same-board paired comparisons.
- Matched 80-step fits reproduce the unregularized selected checkpoints. A 0.1 accounting penalty changes RMSE to 1.367/2.048bb and maximum bias to 2.190/0.819bb: not a reliable two-seed improvement, so it is not adopted.
- A separate 64-root authentic corpus has conditional gains of 0.00004–0.167bb (mean 0.0819), with accounting residual below 1.5e-14bb. Adding 63 of those roots to the original 24 preserves the original five tuning and five holdout board families and excludes a matching held-out family from new training rows. Both 80-step fits select step 80: held-out RMSE falls from the matched 24-root controls' 2.025/1.480bb to 1.137/1.039bb, while maximum aggregate own-payoff bias falls from 1.187/0.909bb to 0.523/0.565bb. Native parity is within 0.000005bb. This supports expanding authentic continuation coverage, not cash activation or a claimed full-game exploitability improvement; the error and accounting bias remain material.

Latest regression validation: 246 TypeScript tests passed with four existing integration skips; type checking and production build passed. The final combined native release suite passed 434 tests (415 library, eight binary, 11 CLI), with 60 intentionally ignored heavy tests. The affected Python suite passed 80 tests. A real native Home-worker smoke query passes with artifact verification and the manifest's explicit action grid. macOS authorization was successfully retried: browser control now works. Fresh 1440px dark, 375px reduced-motion, and 768px/1024px light smoke checks show Home rules, a disabled NL25 profile, accessible action/status names, visible keyboard focus, mobile analyst settings, and no horizontal overflow. The native Home decision route returns HTTP 200 and renders all-in feedback without relabeling Home as raked. Full mode-by-mode cash browser acceptance remains impossible without an accepted cash model; these smoke checks do not activate one. `/api/practice/models` still exposes only Home models.

The isolated publish checkout also passes 262 TypeScript tests (four existing integration skips), 430 native release tests (59 intentionally ignored heavy tests), 57 targeted cash/trainer Python tests, and a production build using locked Next.js 15.1.11 dependencies. It preserves the newer upstream solver workspace. The native count is lower than the dirty development checkout because the separate older routed-preflop experiment is deliberately not included in this checkpoint. Browser smoke testing also completes a Home all-in hand through showdown with no alert and HTTP 200 decision responses; this is not cash-mode acceptance.

## Reproducible local pilot configuration

Artifacts remain outside Git in `/Users/davidvayntrub/pokersolver-training/rake-pilots-2026-10-07.k3G5SR/`.

The recall reference pair is `recall-v5-nl25-seed-7201` and `recall-v5-nl25-seed-7202`; `recall-v5-nl25-seed-7201-uninterrupted` is a determinism audit, not an independent seed. Rake-off controls are `recall-v5-control-seed-7201` and `recall-v5-control-seed-7202`. The terminal-estimator pair is `terminal-integrated-v5-nl25-seed-7201` and `terminal-integrated-v5-nl25-seed-7202`, with its separate `-uninterrupted` audit. Each uses 20bb, 32 rounds, 32 alternating traversals per round, 64/32 hidden layers, batch 256, 20 optimization steps per round, reservoir 5,000, learning rate 0.001, baseline scale zero, two value rollouts, 64/32 preflop/flop runout samples, exact turn rivers, and the compact cent-aligned grid. `coverage-128-v5-nl25-seed-7201/7202` changes only traversals per round to 128. `state.json` records the complete config/digest and measured timings. The CLI now accepts all requested cash depths, but no deeper run or acceptance is implied.

Full-hand checks use 16,000 legal deals, seed 917, two threads and both trajectory distributions. Causal comparisons use eight outer deals, seed 919, two public branches per street, four conditional opponent samples per runout, and a strict 250,000-node limit per seat. Outputs are `recall-v5-full-hands-*.json` and `recall-v5-causal-*.json`. Native queries use seed 923, 4,096 rollouts for sampled branches, and separate binary/weights/rules/state hashes. A changed binary cannot reuse an earlier query capture.

The larger response recheck uses 32 outer samples and seed 929, with the same per-root chance budget and node cap. Frozen reports are `coverage-128-v5-qualification-r32.json` and `coverage-128-v5-qualification-r64.json`; the r32 audit was captured before those run states resumed to r64. The r64 average-policy hashes are `e2422abd6db6eb092f4663543f5eb5f23625ae9f8e44f0e29b4b1680c861d101` and `1a8d40d16444f0e861719342a49a0234b26358bb20899dedc81b1ba06d805f12`.

Authentic continuation captures use the frozen seed-7201 r32 policy, SHA-256 `06a021b50fbc1cc56b7c79496259c5bafd5bbf2d3e98b8d85503c24988a6e282`, not an accepted policy. `authentic-turn-roots-hashed-seed-931.json` supplies six roots; `authentic-turn-roots-24-seed-937.json` supplies 24. The 24-root dataset in `authentic-turn-labels-24x64/dataset.json` has hash `a9ba056990681ce7536f61a0ba0795a36d01cf301ef0edeb327aa94293bf99d4`. Its final label-report timings are cache-read audit timings, not original native solve timings; new label reports explicitly flag cache hits.

The usable paired comparison records are `authentic-turn-value-24x64-cent-fixed-split937-seed-7101/7102/report.json`. The earlier fits without `cent-fixed` failed native parity on the float/cent guard and have no completed acceptance report. A separate 64-root capture uses seed 941, 64 updates and two independent native label workers, capped at 2,000 sampled deals. Keep generated weights, datasets, source captures and logs outside Git.

The 87-context dataset is `authentic-turn-extended-training-fixed-holdout.json`, SHA-256 `b0995062e185b1df2f6d207c7082a6410ae2630652eb50883e38642067ce2263`. Its two parent datasets remain auditable; the original 24 rows are unchanged at the prefix. `authentic-turn-value-extended-80-step-split937-seed-7101/7102/report.json` pins split seed 937, the original holdout indices `[1, 2, 9, 13, 21]`, and frozen network hashes `9d11f8608956b6493e2f2ccf1e5bb72449899613b82fe4915cd03a25753d756a` / `545cf76afd1a62af28bc970bbd3e5d7864996862eba4715d7885f90a9fcac74b`. Full-game exploitability of these continuation students is unmeasured, and neither is active.

## Scope and selected rules

Keep the current rake-free practice game as **Home game**. Add **PokerStars NL25** as the first online study profile, using the published standard USD regular NLHE schedule for two players dealt into the hand. Additional rooms or stakes require separately verified profiles, not interpolated defaults.

| Rule | Home game | First online profile |
| --- | --- | --- |
| Blinds | 0.5bb and 1bb | $0.10 and $0.25, or 0.4bb and 1bb |
| Players and ante | Heads-up, no ante | Heads-up, no ante |
| Rake | None | 4.5%, capped at $0.50 or 2bb |
| Eligibility | Not applicable | No flop, no drop; called preflop all-ins run out a board and incur rake |
| Rake rounding | Not applicable | Nearest cent, half to even |

The online schedule is a snapshot checked on October 6, 2026. Caps depend on players dealt, not table capacity. NL20 is not listed in this USD schedule; do not present NL25 as an exact NL20 model. [PokerStars published rake rules](https://www.pokerstars.com/poker/room/rake/)

Requested study depths are **20, 40, 50, 100, 200, 1,000, and 2,000bb**. Here 1k and 2k mean stack depths, not NL1000 and NL2000 stakes. These targets do not assert that the room permits those buy-ins. Only depths with matching accepted artifacts become playable.

Preserve exact cards, alternating seats, equal starting stacks, current practice modes, and existing Home game policies. No multiway solver, tournament rules, rewards, rakeback, jackpot deductions, database migration to a hosted service, or paid compute purchase is included.

## Why this requires new models

The native solver currently evaluates player zero and negates that value for player one. Its continuation networks also project values to zero sum. With rake, the correct identity is:

```text
player zero net payoff + player one net payoff = minus house rake
```

Because rake varies with the line played, this is generally not a constant-sum game. Deducting rake only in the UI leaves training incentives, best responses, and action EVs wrong. Likewise, distributing rake equally between artificial zero-sum utilities changes the players' incentives.

CFR remains a useful candidate optimization procedure using each player's own payoff, but the standard two-player zero-sum average-policy equilibrium guarantee does not automatically apply to this game. The original result explicitly assumes zero sum. General-sum candidates require direct deviation evaluation; seed agreement alone is not equilibrium evidence. [Original CFR paper, Theorem 2](https://poker.cs.ualberta.ca/publications/NIPS07-cfr.pdf)

## Step 1 Freeze the game profile and money representation

- [x] Extend `lib/practice-game-profiles.ts` into an immutable rules registry. Include profile ID, rules schema, currency, blinds, players dealt, ante, rake rate, cap, eligibility, rounding, odd-chip rule, source, and verification date.
- [x] Define one canonical rules digest with matching TypeScript and Rust fixtures. Bind current artifacts, settings, hands, queries, samples, and caches to that digest. A schedule change creates a new profile version. Production cash artifacts/sample serving remain unavailable, not unpinned.
- [x] Keep Home game on its existing numeric rules and legacy hashes. Infer legacy Home rules only for explicitly recognized rake-free manifests; unknown profiles must fail closed.
- [x] Use integer cents for online wager amounts and settlement, deriving bb only for display and EV reporting. Use integer rational arithmetic for percentage rake, not floating-point `Math.round`.
- [x] For the online action grid, deterministically round requested sizing amounts to legal cents, clamp to valid minimum raises and stacks, and remove duplicates. Freeze this as a new betting abstraction, with actual amount labels rather than misleading unrounded labels. This sizing rule is the application's abstraction, not a claim about the room's preferred bet sizes.
- [x] Verify and freeze the split-pot odd-cent rule from a primary room source. If unavailable, explicitly document the chosen study abstraction before training; do not silently claim exact room settlement.

Primary files: `lib/practice-game-profiles.ts`, `lib/practice-types.ts`, `lib/practice-engine.ts`, `preflop-solver/src/blueprint.rs`, and `preflop-solver/src/practice_transport.rs`.

Done when both implementations agree on rules, cent-aligned legal actions, rounding, and canonical digest. In particular, NL25 must post **0.4bb**, not the Home game's 0.5bb, small blind.

## Step 2 Implement a pure settlement module

- [x] Add a settlement Module in TypeScript and Rust with a small Interface: immutable rules, committed amounts, outcome, and whether a flop was dealt in the completed hand. Return uncalled refunds, contestable gross pot, house rake, net awarded pot, awards, and both players' net payoffs.
- [x] Refund unmatched wagers before calculating the contestable pot. Deduct rake once at actual terminal settlement, cap it, and allocate the remaining pot according to the frozen split rule.
- [x] Treat preflop folds as unraked. Treat called preflop all-ins as raked after their board runout. A preflop-only or postflop-only practice review is not an actual pot settlement and must not charge hypothetical rake.
- [x] Track house rake separately so player stacks plus outstanding pot plus collected house rake equal initial chips. Net player payoffs include returned wagers and exclude the house's share.
- [x] Share deterministic JSON test vectors between TypeScript and Rust. Test folds on every street, all-in calls, checks to showdown, ties, uncalled shoves, cap boundaries, and half-cent rounding in both directions.

Concrete NL25 fixtures, with all amounts expressed in bb:

| Completed hand | Contestable pot | Rake | Net payoffs |
| --- | --- | --- | --- |
| Both invest 5, player zero wins | 10 | 0.44 | 4.56 and -5 |
| Both invest 20, called preflop all-in, player zero wins | 40 | 1.80 | 18.20 and -20 |
| Both invest 50, player zero wins | 100 | 2.00 cap | 48 and -50 |
| Postflop fold, commitments 10 and 2, player zero wins | 4 after refunding 8 | 0.16 | 1.84 and -2 |

Done when cent accounting is exact, payoff sums equal minus rake, and existing Home settlement fixtures remain unchanged.

## Step 3 Remove zero sum assumptions from native training

- [x] Introduce paired terminal payoffs and select the traverser's own utility in cash external-sampling, batched, public-chance, rollout, and best-response paths. Unsupported legacy continuation paths reject cash.
- [x] Audit `utility_p0`, `complete_runout_utility_p0`, scalar negations, and actor-perspective conversions in `blueprint.rs` and `blueprint/neural.rs`. Correct sampled action-value targets; disable the incompatible actor-only baseline for cash.
- [x] Update fold, exact all-in, turn, and river counterfactual values in `blueprint/public_belief.rs`; each player gets their own blocker-conditioned payoff weighted by the appropriate opponent reach.
- [x] Preserve deterministic chance sampling, perfect recall, and DCFR discounting. Do not bundle a new sampling algorithm or sizing experiment into the first rake implementation.
- [x] Version checkpoint and export schemas. Reject resumed regrets, averages, or labels belonging to another rules digest or abstraction. Old policy weights may be an explicitly labeled warm-start experiment, never evidence of a trained online model.

Done when small fully enumerable raked games agree with independently calculated values and information-set-consistent best responses for both seats. Home regressions must still pass.

## Step 4 Generate rake aware continuation values

- [x] Replace zero-sum projections and target acceptance tests for online profiles with two own-player net-value outputs. Keep the legacy zero-sum route for Home models.
- [ ] Regenerate continuation datasets, preflop teacher values, range policies, action EV tables, and student targets using the same rules and pinned continuation route that will actually serve decisions.
- [ ] Update `train_public_value_network.py`, `serving_value_projection.py`, `train_range_value_oracle.py`, and their native inference adapters together. A native-only fix must not leave Python training restoring zero sum.
- [x] Check aggregate consistency against expected house rake under the **same compatible joint belief distribution**. Do not demand that differently conditioned individual combo values sum to zero, or force a zero residual by shifting both players' values.
- [x] Reuse exact rank, blocker, and runout kernels where valid, but recompute payoff labels. Where cent split rules matter, retain win and tie information rather than relying only on a scalar equity.
- [x] Include rules, abstraction, model, board, ranges, stacks, and betting history in research continuation-cache identities. Never reuse Home payoff arrays as online values. Production cash continuation caches remain unavailable pending a matching accepted route.
- [x] Reject raked use of existing zero-sum safe-resolving or max-margin guarantees unless separately justified. A heuristic online resolver can be experimental, but must not acquire a safety claim by inheriting a Home flag.

Done when native and Python outputs agree, raked target provenance is complete, and feedback evaluates the actual served route. Sampling standard error must be distinguished from continuation-model bias.

## Step 5 Run cheap matched pilots before long training

- [x] First run exact small-game tests and a capped 20bb paired pilot: two independent seeds, identical budgets, held-out cards and histories, and frozen cent-aligned online abstraction. Both seeds have been evaluated; their failed gates do not satisfy the step's acceptance condition.
- [x] Include a rake-off control with the **same 0.4bb and 1bb blinds** to isolate rake's effect. Separately check existing Home policies for regressions; their 0.5bb small blind makes them a different game, not the matched control.
- [x] Start with at most 30 minutes per training pilot. Measure throughput, peak memory, artifact size, and full-hand evaluation cost before projecting a longer run. Use independent processes only where work is independent and fits the local 16GB memory budget.
- [ ] Compare trained candidates against the unadapted warm start **inside the same raked game**. Evaluate both seats' net deviation gains, legal action coverage, and action EV consistency across untouched boards and forced deviations. Lower raw winnings than Home are not by themselves policy deterioration.
- [ ] If both seeds show credible improvement without new accounting or serving failures, extend the unchanged configuration. Otherwise identify the failing assumption before increasing iteration counts or buying compute.
- [ ] Audit feature normalization, value scales, raise caps, and action-grid suitability before each deeper stack pilot. Do not extrapolate a 20bb policy to 1,000bb or 2,000bb.

Depth order: **20, 40, 50, 100, 200, 1,000, 2,000bb**. Each depth is independently trained, evaluated, and activated. Long runs begin only after pilot evidence supports their cost; no fixed completion time is promised.

Done when a 20bb configuration has reproducible raked improvement, recorded resource requirements, and a bounded follow-up run plan. Pilot artifacts remain inactive.

## Step 6 Evaluate the complete routed model

- [ ] For the pinned full-hand policy, report each seat's unilateral deviation gain in net bb per hand:

```text
gain i = value of best response i against the other pinned policy
         minus value of player i in the pinned policy pair
NashConv = gain zero + gain one
maximum deviation gain = max of the two gains
```

- [ ] Use information-set-consistent responses, exact card removal, and rake-aware settlement throughout. Report seat-specific baseline EV and expected rake as accounting diagnostics, not exploitability.
- [x] Freeze the metric definition and its mapping to historical zero-sum reports before comparing numbers. Do not divide the total by two simply to clear a threshold.
- [x] Treat learned-response and local-best-response results as restricted response evidence, not certified upper bounds on unrestricted exploitability. A 99% sampling interval does not bound errors from incomplete response search or learned continuation values.
- [ ] Retain the repository's current relaxed **0.50bb per hand total** release threshold for qualification, now with an explicit raked total-deviation definition. Do not silently reintroduce 0.05bb as a new mandatory gate; it remains an improvement target. Qualification requires an appropriate bound, not just a favorable restricted-response estimate.
- [ ] Rerun existing normal gates for each new profile and depth: cross-seed frequency MAE at most 5 percentage points; primary-action agreement at least 85%; aggregate action delta at most 3 percentage points; coverage at least 99.99%; valid raw and quantized probability sums; action EV standard error at most 0.02bb for at least 95% of reach-weighted served decisions; total projected storage at most the existing 20GiB code limit.
- [ ] Record two independent seeds and actual training duration. The existing validator checks 8 to 12 hours per seed; do not fill those fields with planned durations or promote short pilots. Keep normal confidence and provenance checks separate from equilibrium claims.

Done when a profile-specific report distinguishes passed, failed, deferred, and unmeasured checks. If equilibrium evidence is insufficient, follow the existing explicitly experimental pathway rather than labeling the model Approximate GTO. This plan does not waive normal serving gates.

## Step 7 Route by profile and pin all artifacts

- [ ] Extend manifest selection from depth alone to **profile digest, depth, model version**. Update `lib/practice-models.ts`, `lib/practice-policy-client.ts`, and API manifest responses without globally removing the current Home-only protection.
- [x] Replace the single hard-coded 20bb resolver identity in `lib/server/practice-solver-process.ts` with per-manifest worker configuration, artifact verification, and bounded loading. Match the request and response to the same profile and artifact hashes. Implemented for qualified Home runtime manifests; a separately qualified cash runtime is still required, not inferred from these workers.
- [ ] Update `app/api/practice/resolve/route.ts` and `preflop-solver/src/practice_transport.rs` together. Reject unknown profiles, mismatched blinds, malformed rules, wrong model hashes, incompatible states, and missing depth artifacts.
- [x] Add a versioned canonical state schema for raked hands. Preserve existing Home keys. Native cash queries, recorded evidence, and continuation cache identities are rules-pinned; production cash warm-up and sample serving remain unavailable rather than using Home keys.
- [ ] Pin rules and model version for the entire hand, including precomputed continuations. Missing artifacts pause practice with Retry; never substitute a Home model or uniform policy.
- [ ] Keep private hand queries private and existing worker computation off the main thread. Immutable public artifacts can keep versioned caching. A new database or permanent cloud credentials are not needed for adding these profiles.

Done when Home and NL25 can both have a 20bb manifest without ambiguous selection, cross-profile cache reuse, or artifact mixing.

## Step 8 Integrate practice settings and history

- [ ] Make game profile selection structural settings, applied after the current hand with a visible pending notice. Show only active depths for the selected profile; untrained targets remain disabled or absent.
- [ ] Show Home game as no rake. Show NL25 with actual blinds, rake percentage, cap, and study abstraction details. Display gross pot and net awarded pot distinctly in terminal review, with the rake deduction and uncalled refunds.
- [ ] Keep current probability-based grades and the two-choice interaction. Recalculate accompanying action EV and EV-loss estimates using rake-aware own-player values; do not silently replace the accepted grading design.
- [ ] Support full-hand, complete preflop-round, and sampled postflop modes only with matching artifacts and authentic replay histories. Keep existing push/fold as Home-only until a separately trained and validated raked subtype exists; called shoves must not inherit rake-free EVs.
- [x] Add rules identity and settlement ledger to hand and decision records. Perform a non-destructive IndexedDB schema upgrade in `lib/practice-history.ts`; classify known historical Home records, preserve unresolved legacy records, and never fabricate missing rake or regrade history.
- [x] Scope opponent evidence and weakness groups by game profile and depth; expose the profile in stats. Runtime cash adaptive sampling remains unavailable pending an accepted cash policy. Historical Home decisions must not become NL25 training evidence.
- [x] Keep bundled preflop charts and the standalone WASM solver explicitly on their existing assumptions unless separately extended. Selecting an online practice profile must not relabel their existing results as raked solutions. Curated chart rake assumptions are marked unverified, rather than falsely verified as rake-free.

Primary UI files: `app/practice/page.tsx`, `components/practice/PracticeTable.tsx`, `components/practice/AnalystRail.tsx`, and `app/stats/`.

Done when each supported mode uses the selected game's rules from dealing through feedback and history, while unavailable modes cannot start.

## Step 9 Verify and release incrementally

- [x] Add TypeScript, native Rust, and Python tests for rounding, caps, refunds, all-in eligibility, ties, cent-aligned action legality, paired values, rules hash parity, incompatible checkpoint rejection, migration, and profile-specific research queries. Accepted cash production-worker tests remain dependent on that runtime.
- [ ] Test mixed-profile and mixed-depth requests, missing artifacts, stale caches, model switching mid-hand, worker failures, and unavailable policies. Confirm interrupted decisions are not scored.
- [ ] Run targeted suites, `npm test`, `npm run build`, and `cargo test --release` in `preflop-solver/`, plus affected Python suites. If `wasm/src/` changes, run its release tests and regenerate the committed package and runtime copy; never edit generated WASM glue manually.
- [ ] Start the development server and exercise changed flows with the installed browser-use CLI at 375, 768, 1024, and 1440px. Check light and dark themes, reduced motion, keyboard focus, status announcements, pot and chip labels, mobile navigation clearance, console errors, and network responses.
- [ ] Release **NL25 20bb first**, only in its measured qualification category. Preserve Home defaults and rollback independently by profile and model version. Activate further depths only after their own reports pass; requested menu entries are not release evidence.

Done when the published manifest, selected profile, actual native route, action EVs, terminal ledger, and stored history all describe the same frozen game. Remaining unqualified depths stay unavailable.

## Implementation order and handoff

Complete Steps 1 to 4 before meaningful raked training. Steps 5 and 6 decide whether longer compute is warranted; Step 7 can be developed against test manifests while that evidence is gathered. Finish Steps 8 and 9 before public activation.

The first milestone is **correct shared accounting with unchanged Home behavior**, not a new model label. The second is **a credible paired NL25 20bb pilot**. The third is **one fully routed and validated online practice depth**. Do not expand to every stake, add unrelated infrastructure, or spend on cloud training to bypass a failed payoff or continuation implementation.
