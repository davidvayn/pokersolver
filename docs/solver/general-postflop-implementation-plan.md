# General postflop solver: implementation plan

Date: 2026-11-26 (America/Los_Angeles).

Status: **planning only; implementation has not started**. The current request
authorizes writing, committing and pushing this plan, not executing its phases.
Wait for a separate instruction before changing code, running solver experiments,
removing documentation or migrating user-facing behavior.

This is the forward plan for the newly agreed general study tool. Older neural,
rake and experiment plans are historical context, not parallel work queues.

## 1. Product contract

Build **B: a general two-player NLHE cash postflop study tool** first. Keep
**C: fast full-hand play** as a later consumer of its game and strategy interfaces.
Supporting C does not authorize another neural-training project during B.

- Users supply weighted private ranges, a flop/turn/river board, pot, stacks,
  valid current betting state, and supported action sizes. No new trained model
  is required merely to change stack depth.
- Solve the complete configured continuation through the river. Let users explore
  every modeled action and legal future public card, action by action.
- Initially expose up to three configurable regular sizes plus all-in, with
  seat/street-specific configuration. Do not bake that UI default, particular
  sizes or the old one-raise restriction into the solving mathematics.
- An added off-tree action revises the game and requires an explicitly revised
  solution; never pretend the old solution already covers it.
- Prioritize correctness, then waiting time, then memory efficiency. Aim for
  ten-second fresh solves and substantially faster navigation of completed ones.
- Browser-local WebAssembly workers are the public execution target. The same
  portable Rust mathematics must run natively for independent checks and profiling.
- First establish rake-free correctness. Give betting rules and payoff rules
  separate interfaces so later rake support does not require a second engine.
- Preserve the current UI, accepted solved-spot bytes/readers, validated drills,
  and existing experimental 20bb full-hand Practice during migration. Keep that
  old policy's actual qualification status honest and separate from new results.
- Full multiway games, tournament/ICM solving, general preflop solving, cloud
  infrastructure and new full-hand model training are outside this phase.

Two-player postflop pots can use ranges originating from heads-up or six-max.
The initial model does not infer folded players' hidden-card distributions or
silently claim to account for bunching.

## 2. Acceptance and reporting

### Strategy quality: the primary gate

For the **frozen strategy of the configured conditional game**, calculate:

`NashConv = sum over players [value of their information-set-consistent best response - their strategy-profile value]`.

- Default qualification: **NashConv <= 0.005 * starting pot**, in the same bb units.
  Examples: 0.05bb for a 10bb pot; 0.50bb for a 100bb pot.
- Offer an explicitly stricter target; record the requested and attained targets
  in each result. Do not lower either silently to meet a timer.
- Report NashConv in bb per configured spot and as a percentage of starting pot,
  together with both individual response gains.
- For rake-free two-player zero-sum games, report **exploitability = NashConv / 2**
  as a secondary readout, not a second independent acceptance gate.
- Once rake is supported, evaluate each player's actual payoff and retain the
  unhalved NashConv definition. Do not assume zero-sum CFR guarantees or the
  half-NashConv convention automatically carry over to variable rake.
- Qualification applies to the declared ranges, chance model and action tree,
  not unrestricted Hold'em or full-game bb/hand exploitability. A root-weighted
  bound is not a pointwise bound on every rare or off-policy continuation.
- Existing accepted saved studies retain their original absolute gates and
  immutable identities. Do not retroactively requalify them under a looser target.

Rules, legality, accounting, probability validity, information-set consistency,
and numerical EV checks remain mandatory. NashConv is not a substitute for those
checks. Freeze and evaluate the actual representation users consume, including
any normalization or quantization, rather than only a higher-precision ancestor.
Action EVs describe continuation under that frozen policy, not optimal-play labels.

### Waiting time: a measured goal, not an assumed capability

- Target **<=10 seconds for at least 90% of repeated fresh-result runs for each
  declared benchmark configuration on the user's Mac**. Do not hide a slow flop
  configuration behind aggregate results from fast rivers or cache hits.
- Include input preparation, worker startup/initialization when needed, tree
  preparation, training, required quality checks, and delivery of the usable
  decision. Publish the precise timer boundaries and asset-loading conditions.
- Report cold-start, warm-worker fresh solves and cached navigation separately.
  A cached answer is not evidence that a fresh solve meets the target.
- At ten seconds display **"Solver is taking longer than usual"** and continue
  automatically. A time warning does not cancel the job or mark it solved.
- Keep progress and explicit cancellation available. Errors, replacement jobs,
  component cleanup and cancellation must release actual computation, not just
  suppress a stale callback.
- The performance promise is initially Mac-specific. Other devices can work with
  conservative budgets without receiving an unsupported ten-second guarantee.

### Memory: a soft ceiling goal

- Aim below **2GB total solver working memory**, not merely regret-array size.
  Measure working arrays, caches, worker copies, temporary allocations, and peak
  overlap while freezing, evaluating and decoding results.
- This is a strong goal, not an absolute user prohibition on larger jobs. Estimate
  and disclose over-goal configurations; support an explicit bounded budget for
  larger jobs rather than silently allowing unbounded growth.
- Use conservative defaults and user controls when hardware information is absent
  or imprecise. Browser device-memory hints do not reveal reliable free RAM.
- Track controlled allocations and native/browser process measurements separately;
  WebAssembly linear-memory size alone is not total physical memory consumption.
- Offer explicit tree adjustments when a job cannot fit. Do not silently remove
  range combinations, change sizes, prune actions or alter the game.

## 3. Architecture and interfaces

The main flow is:

`GameSpec -> SolveSession -> StrategySnapshot -> action-by-action study/practice`

Names below are proposed implementation seams, not existing interfaces or a
mandate for a large generic framework.

### Game specification and rules

- Version the game schema and canonical game fingerprint. Preserve repository
  card encoding (`rank * 4 + suit`) and canonical combo keys across Rust and TS.
- Accept exact-combo weights, including suit-specific weights and zero-support
  hands; remove public-card blockers, normalize explicitly and reject a starting
  distribution with no compatible private deals.
- Declare the joint starting distribution: the supplied seat ranges conditioned
  on exact card compatibility, not arbitrary hidden correlations.
- Represent pot/dead money, stacks behind, current wagers, actor, minimum full
  raise and reopening status without inventing a fictitious preflop history or
  private placeholder cards. Support valid fresh-street and facing-bet roots.
- Specify whether each pot field includes live wagers, and canonicalize the total
  starting pot used by the quality gate without double-counting current bets.
- Specify chip precision, rounding, duplicate/clipped-size handling, uncalled
  returns and heads-up effective-stack accounting before relying on them.
- Define EV reference points explicitly. New study EVs can be incremental from
  the study root; legacy artifacts retain their original-hand EV convention.
  Include the basis in result metadata and adapt deliberately at the UI seam.
- Keep betting-tree configuration and payoff calculation separate from traversal
  and regret updates. Keep rules consistent between exploring and solving a game;
  retain an independent rules/evaluator path as a test oracle.

### Portable solving mathematics

- Extract a focused Rust module without neural-model loaders, training filesystem
  concerns or UI state. Native and WASM adapters use the same implementation.
- Start with the established exact-card, average-policy DCFR approach and reusable
  efficient evaluator/turn-river kernels. Check the algorithm on small games and
  against independent responses before pursuing speed.
- Avoid retaining a dense private-pair equity matrix for every ordered board.
  Use compact hand-strength/card marginals, bounded reusable kernels and explicit
  memory accounting. Do not just lift the narrow-flop support caps.
- Use contiguous storage and reusable scratch buffers. Evaluate f32 or scaled
  integer storage through numerical and response-gain checks before accepting it.
- Cache only genuinely identical continuation computations. Include board,
  exact ranges, wagers/stacks, betting and payoff rules, and relevant versioning
  in identities; distinguish an equity-kernel cache from a strategy cache.
- Apply suit symmetries only where the computation and any required range symmetry
  justify them. Preserve distinct histories and information sets where players
  can distinguish them; never merge turn/river strategy states just because a
  terminal board has the same unordered cards.
- Parallelize independent work with bounded workers and deterministic test modes.
  Do not multiply memory by cloning whole game trees per worker.
- Do not use learned leaves, separately grafted street policies or the old neural
  model as an undisclosed substitute for a complete configured B solution.

### Solve session and frozen strategy

- Use versioned messages for start, progress, cancellation, completion and error.
  Every response identifies its job and game fingerprint; stale jobs cannot
  update or consume the resources of a replacement indefinitely.
- Run computation in bounded steps so progress, quality checks and cancellation
  remain responsive; do not inherit the legacy synchronous worker's timeout-only
  behavior or the exact pilot's hard ten-minute export deadline.
- Publish a qualified immutable snapshot with game identity, average policy,
  legal actions, range/EV conventions and attained quality. Partial results, if
  inspectable, are clearly unfinished and not graded as a qualified answer.
- Explore descendants of that pinned strategy with correct posterior ranges and
  exact public chance. Do not re-solve independently at every click by default.
- Edits preserve the visible study line where meaningful, but revise game identity
  and invalidate affected results. Old results remain tied to their old inputs.
- Reuse current accepted artifact readers through explicit adapters. Do not
  overwrite accepted artifacts or silently reinterpret their formats.
- Keep future full-hand C consumption possible through the game/strategy seam.
  Do not build a speculative universal neural/plugin framework now.

## 4. Sequential implementation milestones

Every task remains unchecked. A successful milestone means its stated behavior
and evidence exist; a compile alone is not acceptance.

### M0. Freeze the contract and reference workload

- [ ] Record the current Mac CPU, RAM, OS, browser, build and thread settings.
- [ ] Map live callers, active model dependencies, accepted artifact readers and
      generated files before moving or deleting anything.
- [ ] Commit versioned fixture specifications, canonical inputs, expected rule
      outcomes, numerical tolerances and the prespecified timing matrix.
- [ ] Separate fast exhaustive correctness fixtures from performance/stress jobs.
- [ ] Record small existing reference outputs without rerunning the cancelled
      full-flop live-policy benchmark or starting another neural campaign.

Exit: reproducible contracts and fixtures; no user-facing behavior changes.

### M1. Extract the game specification and portable module

- [ ] Implement and validate independent root state, exact weighted ranges,
      canonical identities, betting configuration and rake-free payoff rules.
- [ ] Separate reusable native mathematics from blueprint/training dependencies.
- [ ] Establish native/WASM parity fixtures and shared message/result schemas.
- [ ] Add memory estimation and allocation instrumentation before broad trees.
- [ ] Keep the active old runtime and immutable corpus unchanged.

Exit: one small rules/solving module runs identically through native and WASM
adapters; invalid games fail explicitly rather than being repaired silently.

### M2. Complete a river vertical slice in a worker

- [ ] Solve both players' complete river betting game from user-supplied inputs.
- [ ] Freeze the average policy and measure information-set-consistent NashConv.
- [ ] Independently check terminal values, action EVs and every modeled branch.
- [ ] Implement progress, explicit cancellation, job identity and overrun warnings.
- [ ] Browse the qualified result action by action through the existing UI shell.

Exit: one end-to-end browser study, then broad weighted river ranges, passes
correctness and numerical gates with measured memory and latency.

### M3. Extend to a joint turn-to-river game

- [ ] Train and qualify one policy covering all modeled turn/river continuations.
- [ ] Verify chance normalization, card removal, all-in runouts, posterior ranges
      and both seats' actions, including consecutive actions across streets.
- [ ] Reuse exact compact kernels and bounded workers without hidden leaf models.
- [ ] Exercise narrow, broad and full legal range support; compare native/WASM.

Exit: accepted complete turn games, with every branch checked on bounded
fixtures and no independent river-policy stitching.

### M4. Extend to a joint flop-to-river game

- [ ] Replace the dense per-board equity layout before attempting full ranges.
- [ ] Start with small exact fixtures, then widen ranges and tree complexity in
      short measured pilots. Do not use narrow support as the general release.
- [ ] Preserve complete future decisions, exact removal and perfect recall.
- [ ] Measure total peak memory, initialization, traversal and best-response costs
      separately; investigate structural bottlenecks before adding iterations.
- [ ] Validate storage compression, exact caching and symmetry changes one at a
      time against the same frozen-policy and independent-check fixtures.

Exit: broad-range complete flop studies pass the quality target. Publish any
remaining speed/memory failures rather than removing hard cases from the suite.

### M5. Finish the general study workflow

- [ ] Add editable ranges, pot, stacks/current wagers, board and per-seat/street
      sizes to the preserved UI; expose limitations rather than hidden defaults.
- [ ] Preserve action cards, combo inspection, history, board navigation and
      completed-line review without clearing the whole workspace after a hand.
- [ ] Handle edited inputs, off-tree additions, missing/corrupt data and retries
      without stale results, double-applied actions or fabricated values.
- [ ] Make the new B path independent of the old 20bb manifest/depth restrictions.
- [ ] Keep full-hand Practice on its old model and saved studies on their accepted
      bytes until a separately validated migration is requested.

Exit: users can configure and explore complete studies; existing useful flows
remain available and distinguishable from newly qualified results.

### M6. Optimize and qualify the fixed workload

- [ ] Profile expensive operations and remove redundant computation first.
- [ ] Run matched short pilots for layout, cache, SIMD and parallel changes.
- [ ] Compare any external engine under identical inputs, quality definitions,
      compiler settings and hardware before claiming a clear improvement.
- [ ] Prefer our own mathematics; adopt dependencies/code only with a demonstrated
      benefit and an explicit compatible licensing decision. Do not silently
      change the project's MIT licensing because a reference is AGPL.
- [ ] Run repeated Mac/browser timing qualification only after cheap correctness
      and performance pilots justify its cost. Include all frozen suite cases.
- [ ] Report quality, per-case latency distribution, memory and coverage together.

Exit: demonstrate the agreed per-configuration timing goal, or document concrete
misses and their causes. A quality failure is never converted into a speed pass.

### M7. Consolidate documentation and retire obsolete paths

- [ ] Classify documentation as current guidance, accepted-result provenance,
      active protocol input, superseded narrative or unrelated work.
- [ ] Maintain a concise README/setup guide, domain/architecture reference, this
      current plan, and a compact accepted-results ledger/runbook.
- [ ] Extract useful findings, exact definitions, acceptance evidence, hashes,
      reproduction commands and current limitations before deleting old reports.
- [ ] Remove superseded experiment diaries/plans from the working tree; use Git
      history rather than a new giant archive or concatenated document.
- [ ] Keep machine-consumed protocol JSON, hash chains, fixtures and active model
      dependencies unless their callers are deliberately retired and verified.
- [ ] Remove unused implementation paths only after caller/usage checks and
      regression tests. Experimental does not mean unused: old Practice remains.
- [ ] Check links and reproduction references after consolidation.

Exit: one discoverable current direction, retained provenance and no broken
runtime/test dependencies. This cleanup is future implementation, not part of
the present plan-only commit.

## 5. Verification matrix

### Mathematical and rules coverage

- Exact cards, evaluator outcomes, blockers, compatible joint deals, weighted and
  suit-specific ranges; empty/invalid/incompatible distributions.
- Both seats; fresh streets and facing bets; every modeled size; minimum raises,
  short all-in reopening rules, repeated raises, clipped/duplicate targets,
  unmatched wagers, folds, ties, all-ins and every remaining street transition.
- Rainbow, two-tone, monotone, connected, paired and trips boards; turn/river
  arrivals and no access to hidden cards or future board information.
- Deterministic regret/averaging tests, small-game convergence, trajectory recall,
  profile values and information-set best responses that integrate hidden hands
  before maximizing. Never allow a responder to know the opponent's private hand.
- Probability normalization, payoff conservation and independent action-EV checks
  of the frozen representation. Preserve established numerical checks on accepted
  fixtures; document scale-aware tolerances rather than relaxing them after failure.
- Exhaustive action/chance traversal for bounded reference games, including
  off-policy actions and unavailable EVs at zero compatible opponent reach.
- Target complete reachable branch coverage of critical rules/session/codec
  modules, report uncovered branches and justify unreachable ones. Supplement
  with property/generated tests for larger spaces. Neither code coverage nor a
  finite fixture suite proves every possible user configuration correct.

### Performance workload and methodology

- Prespecify cases across flop/turn/river, stack/pot/SPR variation, narrow-to-full
  ranges, asymmetric/suit-specific weights, board textures and betting grids,
  including more than one raise where the configured tree permits it.
- Cover pairwise interactions and deliberate hard cases; do not promise an
  exhaustive Cartesian product over arbitrary continuous user inputs.
- Use cheap short repeats during pilots. For final timing qualification, use at
  least 30 fresh-result runs per case and report sample count, p50, empirical p90,
  maximum, failures and the fraction <=10 seconds. Report sampling uncertainty;
  a small empirical sample is not proof of a population percentile.
- Run latency trials serially without competing benchmark workers. Record Mac,
  browser, power/thermal context, thread count, build and cache state; keep the
  corpus and target quality fixed. Run independent throughput tests separately.
- Include full-range and memory-stress cases in reports even if they overrun or
  are refused under a selected budget. Report any declared support envelope.
- Measure navigation separately and ensure branch changes do not trigger hidden
  expensive recomputation when a pinned continuation already exists.

### Transport, UI and regressions

- Native/WASM card, action, game-key and result parity; immutable snapshot
  identities; malformed messages; stale jobs; worker errors; true cancellation;
  warnings followed by successful completion; resource cleanup and retries.
- Progress and controls remain responsive while computing. Input edits cannot
  display an old result as a solution for the new game.
- Headless Chrome only: isolated temporary profile, `--headless=new --mute-audio`,
  loopback-only CDP, no personal browser tabs; close task-owned instances after QA.
- Exercise desktop/mobile widths 375/768/1024/1440, light/dark, reduced motion,
  keyboard focus, accessible statuses, complete action flows and console/network
  inspection. Mobile UI parity is distinct from the Mac timing qualification.
- Regression-check accepted solved artifacts, full-hand Practice, drills and
  relevant history/stats behavior without changing old model identities.
- Use targeted tests while iterating, then appropriate Rust release tests,
  `npm test`, `npm run typecheck`, and `npm run build` for affected milestones.
  Regenerate committed WASM outputs with source changes; never edit them by hand.
- Keep full CI PR/manual-only. Do not trigger full CI merely to commit a milestone.

## 6. Guardrails and later work

- Keep milestones small and reviewable; commit/push major verified improvements
  when authorized. Preserve unrelated dirty worktrees and user edits.
- No paid compute, hosted database, dependency/license migration or cloud service
  is a prerequisite for B. Obtain new authority if one becomes necessary.
- No new neural fitting/distillation, broad corpus expansion or historical
  experiment reruns merely because those tools already exist.
- If credible pilots cannot meet quality plus resource goals, report the measured
  bottleneck and compare substantive algorithm/layout alternatives. Do not conceal
  it with more iterations, smaller undisclosed ranges or an old-policy fallback.
- Later rake work must validate actual payoffs, returned bets, cap/rounding rules
  and the correct response metric before exposing a qualified raked solve.
- Later C work must establish preflop and full-game policy quality plus any
  re-solving safety requirements. B's conditional-game NashConv does not certify
  a composed full-hand opponent or promise it will be fast automatically.

## 7. Evidence and references

Current code seams to inspect again before implementation:

- [Live study controller](../../components/solver/useSolverStudy.ts) and
  [public study replay](../../lib/solver/study-engine.ts).
- [Hand rules](../../lib/practice-engine.ts),
  [modeled actions](../../lib/neural-policy.ts), and
  [pinned full-hand manifest](../../data/practice/full-hand-manifests.json).
- [Native public-belief solving](../../preflop-solver/src/blueprint/public_belief.rs)
  and [saved-game wrapper](../../preflop-solver/src/blueprint/public_belief/solved_spot.rs).
- [Exact-flop allocation](../../preflop-solver/src/blueprint/public_belief/solved_spot/exact_flop/tree.rs)
  and [independent response check](../../preflop-solver/src/blueprint/public_belief/solved_spot/exact_flop/check.rs).
- [Accepted library contract](solved-spot-library.md),
  [shard format](flop-shards.md), and
  [matched live turn benchmark](matched-turn-2026-11-17.md).

The current narrow exact-flop implementation retains a dense f64 private-pair
equity table for each of 2,402 ordered board contexts. At 1,176 starting combos
per seat, that layout alone projects to 26,575,266,816 bytes (about 24.75GiB),
before betting arrays. This is allocation arithmetic, not a measured full-range
run, and is why merely lifting its 32-combo/512MiB accumulator guards is not a plan.

Primary external references, not imported code or evidence of our performance:

- [CFR](https://papers.nips.cc/paper_files/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html)
  and [DCFR](https://arxiv.org/abs/1809.04040).
- [OpenSpiel response/NashConv definitions](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/exploitability.py).
- [Rust postflop reference](https://github.com/b-inary/postflop-solver) and
  [its browser implementation](https://github.com/b-inary/wasm-postflop), both with
  licensing constraints to review before adoption; their results are not ours.

Resume instruction: **do not begin M0 or any later milestone until the user
explicitly asks to start implementation.** This document and its push are the
only deliverables authorized for the current turn.
