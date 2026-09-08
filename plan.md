# Native full-hand policy improvement plan

Updated: 2026-09-08. Status: **Step 3: late-refresh128 training complete; matched response captures underway**.
The 32 -> 128 matched response comparison is complete for both solver seeds.
Response gains WORSENED on every seed/board comparison: **0.71960 -> 0.86079bb**
and **0.93397 -> 1.47283bb**. The predeclared condition for 512/1,024 is false.
The failed scaling jobs and bounded target-consistency replay are complete.
The isolated played-profile-target pilot improves both seeds on both development
evaluation boards: **0.71960 -> 0.42267bb**, **0.93397 -> 0.88371bb**.
Its matched 128-update response comparison is complete and REGRESSES:
**0.42267 -> 0.84951bb**, **0.88371 -> 1.05051bb**. No512/1024.
The paired 32-update average-support pilot and recovered response evaluations
are complete. Both seeds regress on BOTH development evaluation boards:
**0.42267 -> 0.76001bb**, **0.88371 -> 0.96470bb**. Do not extend this pilot.
No paid compute or website change.
The overall release plan is NOT complete. A read-only replay reproduced the
retained policies byte-for-byte and exposed large sampled opening-update jumps.
It also revealed an extra first-update discount when the initial sweep creates
nodes. That accounting bug is fixed and verified (342 release tests pass).
The fixed32 response pair is complete: **0.42267 -> 0.50590bb** and
**0.88371 -> 0.87580bb**. Keep the correctness fix, not a claim of stronger play.
That discount-only result did not support a128/512 extension. The subsequent
all-state, actor-specific fixed importance allocation completed paired32 and
improved both seed means against the corrected reference (details below).
Its unchanged checkpointed128 pair and response comparison completed and regressed.
The subsequent LCFR32 improves on importance32, but LCFR128 also regresses on
both seeds/both boards. No512/1024 or paid scaling is supported. See the current
sampling-drift diagnosis below; do not repeat completed comparisons.
The target diagnosis and corrected32/corrected128 comparisons are complete.
Do not repeat them. Neither result proves an asymptotic convergence ceiling or GTO qualification.

Latest user direction: compare actual 128-update response gains against 32;
extend the unchanged configuration to **512, then 1,024** only if improvement is
credible across seeds/evaluation boards. Ignore the friend's run for now.
Prefer sparse ETA-based checks after early health verification; existing worker
memory/time/disk guards remain active continuously. Paid compute requires a
throughput/cost recommendation and separate provisioning authorization.
The fixed-checkdown control converges to **0.002467bb surrogate NashConv** at
512 updates, but the real continuation preflop stability gates still fail.
Batch/stratified/reach-weighted sampling and a stale history cache did not fix
that. A fixed-policy 4-board × 4-turn diagnosis found substantial flop and turn
noise. The retained all-turn prediction control variate reduces measured turn
variance **94.7%**, without substituting predicted values for the native target.
Its paired training tests completed but still failed stability. Independent
preflop-only response evaluation now finds **0.71960bb summed held-out gain**
(two evaluation board clusters, SE **0.10876bb**, no qualifying confidence bound).
SB opening errors dominate; each old 32-round run used only 16 native samples per
seat. An opt-in simultaneous frozen-snapshot update reused BOTH already computed
CFV vectors at similar cost, but its stability results were not consistently better.
Its response-value comparison was worse on both development boards: **1.17835bb**
mean summed response gain versus the retained **0.71960bb**. Reject that schedule.
A targeted turn/river regression then confirmed zero/tiny private root priors
could erase learned average policies and trigger uniform fallback. An opt-in
own-realization averaging correction passes targeted correctness and playback
tests; its effect on actual candidate strength is not yet measured.
Its three-position preflight showed negligible candidate-value changes; no
training extension of that correction. After rejecting the simultaneous route
and screening further baseline variants on saved data, we extended only the
retained alternating control to the cost-audited **128 updates**, then measured
actual response gains. This completed pilot did not support further scaling.
Full-game strength, safety, EV-confidence,
coverage and serving qualification remain incomplete. No website promotion.

Native capture, bounded corpora, numerical fixes, paired fitting/parity and five
matched 32-update comparisons are complete. Those five failed native retention.
The diagnosed **128-update compute-allocation pair now improves on both native32
controls: 0.206310/0.203169bb versus 0.227128/0.229962bb**. All six pairs are
complete. Facing-bet transfer regressed (0.186343/0.179282bb versus native32
0.171905/0.089273bb); second-board transfer also regressed at
0.306303/0.259277bb versus 0.218398/0.212251bb. A fixed-budget,
training-only current-search belief refresh completed with exact held-out parity;
the paired fixed-budget fit and actual inference validation passed. The cheap
same-budget facing-bet comparison improved both policies to 0.150349/0.146662bb;
the second-board same-budget comparison regressed to 0.327539/0.377668bb.
A fixed-data learned-range-pooling pilot gave mixed facing results and worse
root-action rankings on saved second-board beliefs; it was not promoted.
Six new ace-high training families are captured (508 states, old targets and
heldouts exact). The non-pooled fit passed parity; fixed-belief ranking errors
fell sharply and the independent facing pair improved to 0.125979/0.120764bb.
The independent harder-board pair improved to 0.236544/0.252445bb. The original
low-card root improved to 0.188976/0.193007bb, below both original native32
controls. Retain this candidate for the next stages, not release. The explicit
16-state forced-action belief diagnostic also completed without changing policies.
No website candidate accepted or
deployed. Steps 1-2 are complete at the bounded-pilot level; Step 3 is active,
and Steps 4-5 remain pending. Failure of an acceptance gate blocks promotion, not
continued improvement work. The earlier pause was premature: there is no
external/resource issue preventing further bounded local diagnosis and pilots.

## Active bounded experiment: frozen importance allocation (September 8)

- Simple six-opening-stratum allocation is rejected: it raises BB-vs-limp
  conditional variance 2.53–2.56x, despite helping opening updates.
- New allocation fits squared, class-weighted regret-gradient contributions over
  **all 100 public states** from corrected32 policies, both seeds, boards 0/1.
  Per actor, use `q = 0.5/49 + 0.5*sqrt(meanEnergy)/sum(sqrt(meanEnergy))`.
  All endpoints retain positive support; largest inverse weight is at most98.
  Freeze q before new training; keep exact `mean + residual/q` correction.
- Development boards2/3 show conditional gradient variance ratios0.630–0.696
  (SB),0.323–0.418 (BB), relative to uniform-one. Expected native cost ratios
  1.20–1.26; variance-times-cost ratios0.773–0.864 (SB),0.388–0.525 (BB).
  Opening, BB-limp, BB-facing4 variance also improves in each of four comparisons.
  BB-limp benefit is small for seed1, so do not claim every decision is more
  compute-efficient. These are reused development captures, not qualification.
- Source screen: `preflop-solver/neural/runs/local-frozen-preflop-baseline-20260908-discountstamp32/importance-screen.json`,
  SHA256 `2e43e26a75c2e0f41833d8e34eac3725f69ef1b1234970a70dfe357819289cee`.
  `screen_endpoint_proposal.py` records source/capture/helper hashes and disjoint
  fit/evaluation indices. No new native queries in this screen.
- Implemented opt-in `fixed_importance` sampler, explicit pinned proposal,
  endpoint-order/full-support validation, exact-q selection tests, controller
  propagation and paired-proposal identity checks. Retained uniform RNG path is
  unchanged. New route is restricted to32 updates or fewer, played-profile
  targets, exact checkdown plus turn baseline, with no other experiment toggles.
- Execution order: build/test; run paired32 with seeds27001/27002, chance28001. Check first
  few updates, q parity, regret accounting and guards. Then measure frozen
  response gains against discountstamp32 **and** retained playedtargets32.
  Neither variance nor stability alone permits extension or promotion.
- Execution: Rust sampling tests4/4; library release regression343 passed,
  50 ignored,0 failed (144.543s), manifest SHA256
  `d8e34bab95d54a7e6f0edddf79f44c7bc22c9a4c0bcc6623bd07bd52d0b6f6db`.
  Nine focused Python tests pass. Binary SHA256
  `8c1d12d790c0a9b9952bff7e58524145b721239c2608dcdad517b636368e0d0d`.
- Paired32 complete in719.765s; peak worker footprints1.84GB/1.78GB.
  `local-compact-preflop-20260908-importance32-a/manifest.json` SHA256
  `0d2a6afe56d27d5e9a82d001dc201f6322543aea6a6d199f5640fec9dc210387`.
  Frozen policies27001 `b8b3f024ded0c9bff2828253f299678819bd7bbdec0387abf8dd72c9b074293a`,
  27002 `1bf3ca5ae76f9c7bf0690be838cf365950fbdbcc5889e58e9bc00ecbcf7be87c`.
  All64 traces pass full discount/mix/update recurrence and actor/history/q parity.
  Stability regresses: worst action MAE32.3823%, primary agreement15.9763%,
  max aggregate delta30.6493 percentage points. No strength improvement claimed.
- Both independent guarded response preflights passed in~103s, in
  `local-frozen-preflop-response-20260908-importance32-{27001,27002}-preflight`.
  Manifest hashes27001 `df73ddc2731a4bdde57762890951e88d8fddc76d59eb4856d1cd8cba62551de3`,
  27002 `61ef0f393c707454a4b13b7ad7c7c1c747b0ad91063d691effc27ba368f47695`.
- Completed: full four-board response capture for each seed, in corresponding
  `-complete` directories. Two controllers with two workers each, all2GB/3600s
  per-worker guards and20GB disk reserve. Both complete in38.9–39.3 minutes.
  Final manifest hashes27001 `1e652b61be8bda6366d01cce39540940f7eeddd7ac5602f6cd4f61e565c97248`,
  27002 `ec3d5a9a8ab5ba34406b9c6ff1bdf9454c13c3d46874a4df1e2b313d3e8f5f8e`.
- Mean summed restricted preflop response gains: **0.475653 /0.572239bb**,
  development-board SE0.121374 /0.179011bb. Versus corrected discountstamp32,
  both seed means improve from0.505904 /0.875800bb; paired mean decrease
  **0.166906bb (24.2%)**, board-cluster SE0.107789bb. Three of four individual
  seed/board comparisons improve; both seed-averaged board comparisons improve.
  Versus historical played32, seed1 regresses0.052980bb, seed2 improves0.311475bb.
  These are **not full-game exploitability** or qualifying confidence bounds.
- Comparison files in `local-compact-preflop-20260908-importance32-a`:
  `response-vs-discountstamp32.json` SHA256
  `6bf0ea964e2cc314a7018777c61065cedc4527b932908e8bff45c0bada831fac`;
  `response-vs-played32.json` SHA256
  `86825fc3e15782f9bf3bcef5174f433ccafbc3e49506d3b45790dc1185e68d7c`.
- Residual diagnosis: seed1 SB after limp/BB5bb contributes0.303677bb to the
  response gain (0.293098/0.314255bb on the two evaluation boards), largely
  over-raising to9bb. Its three sampled continuations under that history all
  followed a9bb raise; the direct-call native correction was never sampled.
  This is **not missing table coverage**: all nodes receive baseline updates.
  Under frozen q(call)=0.035978, probability of no such samples is55.6% over16
  SB updates, versus9.6% over64. This supports checking more updates, not proof
  that iterations alone fix the remaining strategic errors.
- Decision: retain the allocation as a promising research configuration, keep
  both previous references, and run an unchanged **128-update** paired pilot.
  Evidence is mixed and limited; this is NOT approval for512/1024 or promotion.
  Estimated48min from measured32 throughput, with unchanged3600s/2GB worker caps.
  Same seeds, chance stream, proposal, model, baseline, targets and averaging.
  Root tracing is disabled after its verified read-only32 diagnostic; compare
  the numerical first32-update prefix to prove the trajectory is unchanged.
  First attempt (subsequently stopped): `local-compact-preflop-20260908-importance128-a` used
  binary SHA256 `eaed2e32a35a5ca482f2a8ee7d67d996a6a240801cbce927f78cde0f679870f3`.
  Only the opt-in pilot-length admission changed; the sampling tests4/4 pass.
  Numerical progress matches the32 prefix so far. Exclude timing, root trace
  and `flopPolicySha256` when comparing: the latter hashes the whole native
  solution including `game.iterations`, which changes from32 to128. The native
  solve budgets/RNG rule and observed numerical results are unchanged.
- Consolidated validation:31 tests pass across12 CPU-only Python modules.
  Three legacy test modules (`test_train_public_value_network`,
  `test_native_value_dataset`, `test_native_forced_beliefs`) import MLX and cannot
  execute in this sandbox: `[metal::load_device] No Metal device available`.
  Do not report those as passing or change the CPU training to work around it.
- Small, byte-identical copies of the proposal and both response comparisons
  are retained outside ignored runs as `neural/20bb-20260908-fixed-importance*.json`
  for the source-control milestone. They are research evidence, not serving models.
- Milestone committed locally as `450d910` (86 accumulated research-source/report
  files; unrelated website/report files left untouched). Push to origin/main
  failed: DNS cannot resolve github.com in this environment. Do not claim pushed.
  Three JavaScript diagnostic tests also pass.
- The first128 attempt is **stopped, not completed**. Refined timing using the
  exact scheduled histories and observed native costs projected~62–74min versus
  its3600s cap; the compact loop had no checkpoints despite core Trainer support.
  Both workers stopped through the execution session's normal interrupt handler
  after~2040s. A direct cross-session signal was denied; session interruption
  succeeded. No completed128 policy exists, and no response evaluation was run.
  The32 policies/results remain intact. Do not resume from a frozen average—it
  lacks regrets, averaging accumulators and RNG state.
- Current work: local-only immutable MessagePack training checkpoint generations
  plus hash-pinned receipts carrying continuation identity and progress. Reuse
  Trainer's existing serializer/restorer; enforce unchanged model, proposal,
  baseline, chance seed and solver binary. Add deterministic split-run testing.
  Next restart same128 policy settings with checkpoints every8 updates and a
  **5400s (90min) maximum per worker**, retaining2GB memory/20GB disk safeguards.
  Checkpoint/resume artifacts are never serving exports. Do not restart until
  the new correctness test passes; compare the32 numerical prefix again.
- Recovery test first failed on exact bytes: generic Trainer restoration treats
  a zero sampled-deal count as legacy missing data and fills it from iterations.
  Compact exact-private integration legitimately records zero private deals.
  The adapter now preserves that counter; the uninterrupted/split checkpoint
  bytes match exactly, including regrets, averages, discount accumulators and RNG.
- Current: live8-update checkpoint preflight in
  `local-compact-preflop-20260908-importance8-recovery-preflight` plus full Rust
  library regression. New binary SHA256
  `345680d3e4736a478c2321032f3be0f7c1ee9d98932f1be143e3f07bab6ca967`.
  Resume these real eight-update checkpoints into a new128 output directory;
  do not recompute those initial updates. No claim of completed128 strength yet.
- Completed live8 checkpoint preflight in181.770s, manifest SHA256
  `73b8ef2fa4609bc2a2cfa6c0676304f984228f9144aab3fc10430d7da6d822c6`.
  Both first8 numerical progress prefixes match the32 reference. Receipts:
  seed27001 `fdc7a92be0f3c06821457b86ae3602f52a590a385adcf064ac49ca9d94012c60`,
  seed27002 `fd74622d6e15c9253823d500c16b80fa8d0874d7db34b131dedf7ea11b2d3572`.
  Each state file is~1.3MB. Full library regression **344 passed,50 ignored,0 failed**
  in137.109s; manifest SHA256
  `8e1e6321bdbaf47c75f85b37c7e44eb19e4dcb70d84d9e025db2025d106f58dd`.
  CPU Python31 tests pass again; the previously documented Metal exclusions remain.
- **Currently running** `local-compact-preflop-20260908-importance128-recovered-a`:
  both seeds restore their completed8-state, run through128, checkpoint every8,
  max5400s per worker,2GB memory cap,20GB disk reserve. Same345680 binary and
  frozen importance proposal/model/kernel. Do not confuse this with the stopped
  `importance128-a` attempt. Next verify the post-resume32 prefix, then evaluate
  final128 policies against importance32 using the same pinned response protocol.
- Strong live recovery verification is complete: loaded each recovered run's
  round32 checkpoint, requested target32, and exported without any new training
  updates. Both frozen policy files are **byte-identical to the original
  importance32 policies**, not merely close metrics. This also verifies the
  round8->128 resume across real native continuations. Export-only controller
  completed in1.191s (`importance32-resume-parity`), manifest SHA256
  `5333648b89fed9408e566c7eff263311e5c71cd4f83016ce87dc409a071fdf21`.
  Parity report SHA256 `ed6e6b8a9907cc5c2581c01ed95e1c0592a7e278754a4d9b2445a4683b87de1c`,
  with a tracked copy at `neural/20bb-20260908-checkpoint-replay.json`.
- Recovery milestone committed locally as `24cb701`; its push also failed DNS.
  Both450d910 and24cb701 are pending origin/main. The active128 run is unaffected.
- Exact345680 executable archived at
  `neural/runs/local-compact-binaries/preflop-solver-345680d3e4736a478c2321032f3be0f7c1ee9d98932f1be143e3f07bab6ca967`
  (10MB, hash verified). Use this archive for subsequent resume/evaluation jobs.
  The currently active controller still pins the original target/deps binary:
  **do not rebuild/overwrite it while that controller runs**. Checkpoint receipts
  pin binary content, not its location, so the archived copy supports later recovery.
- Checkpointed128 training is COMPLETE,2671.473s for the resumed pair;
  seed segments2249.345/2671.034s, peak1.902/1.893GB, all guards pass.
  Manifest SHA256 `e0c6ee671782698026a019908a0f020f6548298e1793978c8d8fe757a37463f5`.
  Frozen policies27001 `1cb331906c0dd7c576af1270460805baa662d0ac00af8bbf057d2aa0c5597499`,
  27002 `b5a9648a5094760f5bf30d77780d97fdd0e7f05f35d40a2e8ecf7fecf778dff1`.
  Root stability improves from importance32 but still FAILS: worst-action
  MAE17.5765%, minimum primary agreement24.2604%, aggregate delta12.1762pts.
  Probability sum error3.33e-16; no strength claim from stability alone.
  Seed27001 response preflight passed95.400s and complete capture is running;
  seed27002 preflight is running. Paths:
  `local-frozen-preflop-response-20260908-importance128-{27001,27002}-{preflight,complete}`.
  Both use the archived345680 executable and identical continuation protocol.
  The completed training controller no longer needs its target/deps path pinned.
- Paired128 response evaluations COMPLETE,2034.723/2020.022s; every worker
  succeeds within guards, peak1.199/1.036GB. Manifest hashes27001
  `a0b20b0cd3a1eaf05a90d854939ecc19153824e07d945e8926c5a5d7ae923c61`,27002
  `59a2f78f0cea260fe5e329b54376ed4c3b74b16692aa4fca1bcc79f9fcc5d7a0`.
  Response gains REGRESS from importance32: **0.475653->0.696300bb** and
  **0.572239->0.885924bb**, worse on BOTH evaluation boards for BOTH seeds.
  Paired mean increase0.267166bb, two-board cluster SE0.036870bb; this remains
  restricted preflop response, not full-game exploitability. **No512/1024**.
  Comparison SHA256 `de414a6a28675098013a7e164857f39a05111c10ffc2406eda7096177e5beca7`,
  tracked byte-identical copy `neural/20bb-20260908-fixed-importance128-response.json`.
- Fast saved-capture diagnosis reproduces the regression, conserves the exact
  response-gain decomposition and checks crossed frozen attackers. Seed1's old
  limp/BB5bb contribution falls0.303677->0.007910bb, but opening contribution
  rises-0.140430->0.298464bb. Both128 policies resist their OLD32 attackers better
  (0.184814/0.236002bb), while newly fitted attackers expose other weaknesses.
  Switching only the opening mix32->128 under frozen OLD continuation values
  loses0.181715/0.016501bb; under NEW values it loses0.114318/0.029046bb.
  All eight seed/board/frozen-value opening comparisons worsen. Weak offsuit
  over-opening is visible, not merely a changed evaluation opponent.
  This crossing is diagnostic only: ranges/downstream policies remain frozen,
  so it is not a newly re-solved full-game candidate or qualification metric.
- Next read the existing eight-update checkpoints to locate opening deterioration
  without replaying native training. Test late noisy updates versus sustained
  drift before proposing a new pilot. Read-only checkpoint-root export added;
  compiling targeted recovery regression, no training job active.
- Checkpoint-root inspection COMPLETE:32 checkpoints, ZERO native queries,
  original32/128 averages reproduced exactly. Manifest SHA256
  `581d16352eee1e9958c537e9ab6dd0d7b1467d9b1e2aaec3016ee62628370bc1`.
  It shows sustained mid-run opening deterioration, not just one final spike.
  Seed1 85o's2.5bb average rises0.27% at32 to60.21% at128; its current mix reaches
  77.4% at64 and92.6% at128. Seed2 differs. Equal weighting of eight-update
  blocks helps seed1 but hurts seed2: reject an averaging-only change.
  Tracked `20bb-20260908-opening-checkpoint-trajectory.json` documents the caveats.
- Research cross-check: Brown/Sandholm2019 distinguishes sampled LCFR from
  full-traversal DCFR settings (section Discounted Monte Carlo CFR):
  https://www.cs.cmu.edu/~sandholm/cs15-888F21/reweighting.aaai19.pdf .
  Saved32 action-value replay first reproduces the actual DCFR regrets/mixes/
  averages exactly. Plain CFR is mixed/worse. Linear weighting improves opening
  value on all eight seed/board/fixed-continuation comparisons (+0.00458 to
  +0.03701bb). This is an OPEN-LOOP screen, not new native strength: opponents
  and downstream policies do not react. Source `20bb-20260908-linear-weighting-screen.json`.
- Selected bounded next pilot: opt-in LCFR, keeping fixed importance proposal,
  chance seeds, native continuation budgets, targets and action abstraction.
  LCFR stores R_t=(sum_{s<=t} s*delta_s)/t, discounts BOTH signs by(t-1)/t
  before update, and uses exact linear own-realization average weights.
  It does not clip sample values or negative regret. This is a fresh <=32 pair,
  not an extension or reuse of incompatible DCFR regrets. Default DCFR unchanged.
  New actual-caller test failed before implementation on signed-regret weighting;
  mixed-schedule controller test also failed before the identity check. Controller
  tests now pass; native linear-regret/average regression compiling. No native
  LCFR pilot launched until correctness passes. Full-game release remains blocked.
- LCFR actual-caller regret/average test PASS (0.27s); first library regression
  345 passed/51 ignored/0 failed (126.67s). The initial `lcfr32-a` attempt was
  STOPPED after62.802s by an early parity check: even the unchanged first-round
  public state produced different continuation values (max root differences
  0.059559/0.323086bb). The preflop config was passed through to the native
  postflop solver, leaking the new regret parameters into the supposedly fixed
  continuation. No completed LCFR policy/result from that attempt.
- Fix: shared `solve_pinned_compact_continuation` boundary resets only optimizer
  schedule/parameters to the original fixed DCFR defaults while preserving game,
  cards, ranges and action abstraction. Compact training, frozen response captures
  and compact full-hand playback all use it. A boundary regression requires the
  resulting serialized configs to be identical for DCFR/LCFR preflop inputs.
  Rebuild/test, then restart fresh as `lcfr32-b`; verify first-update native values
  match importance32, plus every observed linear-regret recurrence and identical
  chance/endpoint draws. This is one isolated algorithm pilot, not a new oracle.
- Corrected `lcfr32-b` is RUNNING under archived binary
  `179750bf28bcda5a1a0efe34794566c790e2baf00ad3e6221f008f03a9684f2e`.
  Both first-update native policy hashes are BYTE-IDENTICAL to importance32,
  not merely similar EVs. All observed signed-regret recurrences and current
  mixtures pass, with identical boards/turns/endpoint selections/proposals.
  Checkpoints every8, target32, two workers,3600s/2GB worker guards.
  Boundary unit test passes; final library regression346 passed/51 ignored/
  0 failed in141.2s;25 CPU Python tests pass. No UI or serving model changes.
- Milestone committed as `e82beb3`; push again failed github.com DNS resolution.
  Final regression manifest SHA256
  `1dc2d247d5df6501d7100cc79c541f77811a19b7145cc7a4fd0680c4d26f8f7b`.
- Corrected LCFR32 pair COMPLETE in626.728s, seeds626.175/569.920s.
  Manifest SHA256 `1b129e4f7843766114833415064f9f36472b297d7667f620c370c632b649f876`.
  Frozen policies27001 `ca791b4a9fdc320ff903ee9e9ea1140270090f857d6ef07d4fad59ccb9323e87`,
  27002 `79373cc75185d75012323e9baf227ed80164c41341f7131bec23754cc3a7e7e8`.
  All64 traced linear recurrences match (max3.55e-15), sampled chance/proposals
  match the DCFR reference, and independently summed linear opening averages
  match exports (max4.44e-16). Root stability still FAILS: MAE23.1771%, primary
  agreement47.3373%, aggregate delta14.1896pts. Both response preflights running
  at `local-frozen-preflop-response-20260908-lcfr32-{27001,27002}-preflight`.
  Use archived179750 executable for completed captures. Compare against
  importance32 with identical native continuation metadata; no scale-up yet.
- LCFR32 response pair COMPLETE,1986.275/1987.246s. Seed means improve
  **0.475653->0.461800bb** and **0.572239->0.467628bb**. Paired decrease0.059232bb
  (11.3%), board-cluster SE0.054779bb. Three of four individual comparisons
  improve; both seed-averaged boards improve (-0.114012/-0.004453bb).
  Seed1's second board regresses0.036799bb, so evidence is modest, NOT
  conclusive, NOT a qualifying confidence bound and NOT full-game exploitability.
  Comparison SHA256 `d3a482a2410745cded5010f33020c4a62f8ff6739034e8de9f27bb34974db232`;
  tracked byte-identical `neural/20bb-20260908-lcfr32-response.json`.
- Decision: test whether the SAME linear weighting avoids the earlier128
  deterioration. Bounded128 pair only; no512/1024 or promotion. Native/runner
  admission expands to128, all learning/continuation settings remain unchanged.
  Fresh run required because old32 checkpoint receipts pin the old executable;
  do not silently relabel those receipts as produced by the new binary. Retain
  checkpoints every8,5400s/2GB worker guards and20GB disk reserve; verify the
  numerical32 prefix and export-only32 checkpoint parity again. No paid compute.
- **Initial attempt:** `local-compact-preflop-20260908-lcfr128-a`, two fresh
  seeds through128 with8-update checkpoints and5400s worker limits. Archived
  executable SHA256 `0b4d04bb281a0d69c5d8a0a52f6834a24856110804bef9eacbb2b51494af1f64`.
  The bounded-admission change passes the native linear-regret/average test and
  controller tests. Source/result milestone committed as `3d85bc7`; push still
  fails DNS. This attempt later stopped on its memory guard; see recovery below.
  Next compare its completed frozen policies to LCFR32 (not the failed DCFR128).
- Both first32 numerical prefixes match LCFR32. Export-only restoration of the
  new run's32 checkpoints also reproduces BOTH original frozen policies
  BYTE-FOR-BYTE, with zero additional training updates (1.178s).
  `local-compact-preflop-20260908-lcfr32-resume-parity` manifest SHA256
  `dae98571cce1ea9db3d45dd6c4caf46758f6fe62ae7d528e2780651c5ae78ab3`;
  policy parity report SHA256
  `c5f763921922aa4ea3f5c3154867b0912854e2b7a092bb7e0a64b19d7d78af25`.
  This verifies LCFR state recovery and the unchanged32->128 trajectory, not
  final128 policy strength.
- Initial128 pair stopped at940.399s when seed27001 reached2,160,642,328 bytes,
  slightly above its2GiB cap. Last complete checkpoints40/48 are intact and
  SHA-verified; only the incomplete tails must be repeated. No numerical failure.
  System memory free55% on16GiB before recovery. Added an explicit, bounded
  2560MiB-per-worker option restricted to checkpointed128; default remains2GiB.
  Controller regression was RED before this option, then10 focused controller/
  resource tests PASS. No Rust, policy, sampling, or continuation change.
  **Training complete:** `local-compact-preflop-20260908-lcfr128-recovered-a`,
  same archived0b4d04 executable, seeds resumed40/48, two workers with2.5GiB each,
  5400s worker limits, checkpoints every8 and20GiB disk reserve. Receipt SHAs
  `383f1deddafdd1a4cc152b9ef4c0cfccd607f72654e0e629f74480aa3a6a1b53` and
  `791b648375825165412bdd296580b30307284ad8852efd081fa90ac11295aa54`.
  Early resumed updates42/50 are healthy. Evaluate final recovered policies
  against LCFR32 only after training completes; no512/1024 yet.
- Recovered pair finished in1752.330s wall (seed segments1487.864/1752.115s),
  all128 progress entries and16900 nodes verified. Manifest SHA256
  `68363796fa6155c5922d66acc630f6d2bdb751b5fc006dbc516e48b2da809aec`.
  Frozen policies seed27001
  `7a964a869523619ae9c907e756ebc5e0db8d880f822b84b662a64a5cbd91124e`, seed27002
  `4d353737b60a1cdbf754886a58e6094341c4b37e844d8b4775cf2720fe58119d`.
  Stability remains failing/mixed: worst-action MAE21.5226%, minimum primary
  agreement34.3195%, max aggregate delta14.3338pts; probability sums valid
  to4.44e-16. Do not infer strength from these values. Seed1 response preflight
  passed93.000s; its full capture and seed2 preflight are now running at
  `local-frozen-preflop-response-20260908-lcfr128-{seed}-{preflight|complete}`.
- LCFR128 response pair COMPLETE: **0.571055197 / 0.515454001bb**, versus LCFR32
  **0.461800197 / 0.467628037bb**. ALL FOUR seed/board changes are worse.
  Pair mean increase0.078540482bb (16.9%), board-cluster SE0.028972737bb.
  Two reused development board clusters, not full-game exploitability or99% proof.
  No512/1024. Captures elapsed1971.277/1965.819s, peaks1.146/1.132GB, all8 guards
  pass. Manifest SHAs `48ad200d7bd116d8fccc446e0adac97f7624d20c0f6d11f78e6b88199319ceeb`
  and `8ae01e45e652beb123d5e386910a6e8378f1573e3c764e66e6d995c723267a1b`.
  Comparison SHA `5b740cd75678147ceb30c8f581775dacff7f51847b30ff62d79afe900a722933`.
- Saved cross-attacker replay: old32 attacker against128 gains0.163703/-0.191027bb,
  versus newly fitted128 attacker0.571055/0.515454bb. Old weaknesses improve,
  new ones appear. Switching only the opening mix32->128 loses value under BOTH
  old and new frozen continuations for both seeds and boards. This localizes
  drift but does not distinguish noisy updates from continuation bias.
  Diagnostic SHA `c2c596e901c69d86f51e5c4432b927b3432914138fc23179ba670e10bcb40b84`.
- Four exact Rust checkdown exports (ZERO native queries), plus saved captures,
  identify sampling-allocation drift. Old fixed proposal's BB gradient variance
  is0.95-1.01x uniform at128, versus0.35-0.62x at32. Refitting on128 boards0/1
  improves variance-times-cost on EVERY128 evaluation seed/board/actor:
  SB0.826-0.945x, BB0.650-0.724x. But the same refit hurts early32 policies,
  up to2.107x BB variance-times-cost. Do NOT replace the initial allocation.
  Report SHA `f5d45b491cb18f85e24f04647ddbb6f6e6f928933b8d2847ee90ed29868314db`;
  baseline manifest `948b1340a952e24f767072a5de5bd6efb444afa4e7b926b74c383a5bcc40a06f`.
  These are conditional endpoint-selection variances, not chance variance or
  policy-strength gains. No native refit pilot has run yet.
- Next bounded test: keep the original proposal through32, then use the frozen
  refitted proposal from33 onward. Both are pinned BEFORE training, keep50%
  uniform support and return the active exact q for the unbiased correction.
  Verify32 prefix parity and round32/33 selection/correction boundaries. Change
  no regrets, averaging, continuation, cards or action abstraction. At most128
  updates; compare response to BOTH retained LCFR32 and unchanged LCFR128.
  Research: [generalized unbiased sampling](https://ojs.aaai.org/index.php/AAAI/article/view/8241)
  ties estimator variance to convergence; [predictive baselines](https://proceedings.mlr.press/v119/davis20a.html)
  reduce sampled variance under their assumptions. Neither guarantees this
  finite-budget range-conditioned oracle is equilibrium-correct; [depth-limited
  solving](https://arxiv.org/abs/1805.08195) motivates a separate continuation-
  robustness comparison if this variance-only change fails.
- Late-refresh implementation COMPLETE. Optional `probabilitiesAfterRound32`
  is validated with the original distribution before the first draw, including
  all49 histories, normalization and50% uniform support. Actual selection
  receives the update number and returns the ACTIVE q. Old proposal files remain
  unchanged. Source/recovery identity pins the entire two-phase proposal;
  controller/result metadata reject unsupported or mismatched phase schedules.
  Round32/33 exact-expectation tests and two-seed first32 RNG-prefix checks pass.
  Final archived binary SHA256
  `210845ed3eccf857152cde22501571f26acdd11efe559e8f83f99d8e33b62a0b`.
  Full guarded Rust release suite **347 PASS,51 ignored,0 failed**,194.985s,
  peak1.491GB, manifest SHA256
  `08bc2c4e50eb96e3c7978e2b2ad4aaf462461b38884d456501c32fff742f6782`.
  Thirteen focused Python tests PASS, diff check PASS.
- **Training complete:** `local-compact-preflop-20260908-lcfr-refresh128-a`,
  fresh paired LCFR128,2 workers,2.5GiB/5400s caps,8-update checkpoints,20GiB
  disk reserve. Two-phase proposal SHA256
  `8786d933f44e70c17706a3ff8bd2838e6a2871e322d04dea9737c01ac99746fb`;
  tracked byte-identical `20bb-20260908-lcfr-late-refresh-proposal.json`.
  First32 must match retained LCFR32 numerically; verify actual sampled q switches
  starting at33. No new continuation/value model, action grid or regret schedule.
  After completion, run the SAME paired frozen response evaluation and compare
  against both LCFR32 and unchanged LCFR128. No512/1024.
- Early native verification PASS: both first32 numerical update prefixes match
  LCFR32 exactly (excluding timing/trace-only metadata); through updates35/38,
  every selected history reports the correct active proposal probability.
  The phase switch at33 is exercised in actual training, not only unit tests.
  Source/proposal milestone `18eafaf` committed; push still fails GitHub DNS.
- Refresh128 COMPLETE in3013.980s wall, seed training2837.623/3013.768s; all256
  actual sampled probabilities verified and both first32 prefixes exact.
  Manifest SHA256 `0e54976e666846b0cdd0835c32a00d8533c07a5ef090f7fce8bf44da418be5d4`.
  Frozen policies `20602d3c7bb319d8cbffe958e4e4fd0691a1bbcce1f7284cbde36fd0bf3b329a`
  and `6dff07df4ce764642080128d8332aa3ce3678613e42ae5489995d19865c967cd`.
  All guards pass, peak1.883/1.985GB. Cross-seed worst-action MAE19.0161%,
  primary agreement33.1361%, max aggregate delta8.6718pts: still FAIL/mixed,
  not evidence of lower exploitability. Probability sums within4.44e-16.
  **Currently running:** seed1 complete response capture and seed2 preflight at
  `local-frozen-preflop-response-20260908-lcfr-refresh128-{seed}-{preflight|complete}`.
  Seed1 preflight passed92.337s. No training remains active; do not restart it.

## Objective and scope

Make strong postflop search affordable, stabilize preflop against the continuation
actually played, then qualify the combined policy across complete hands. Pursue
the user's staged full-game targets of <0.50bb/hand and subsequently <0.05bb/hand.
Measured attacker gains are not a certificate of an upper bound on true
exploitability. Never promote a conditional-root result as a full-game result.

Work locally first, with bounded pilots and independent paired comparisons.
Reuse the native turn/river solver, public-value trainer, tabular preflop solver,
response evaluators, and website. Do not restart already completed averaging
work, introduce unrelated UI work, launch paid compute, or relax acceptance gates.

## Baseline and metric ownership

Baseline: commit `66bf1a38596e0f346b9e4aab66b30921942604fe` and
[the September 6 ledger](docs/solver/overnight-2026-09-05.md).

| Metric | Baseline | Target / interpretation | Steps |
| --- | --- | --- | --- |
| Cold strong flop query | 300s timeout; 11/128 rounds, 99 native turn queries | Complete below existing deadline; proposed usability objective cold p95 ~5s | 1, 2, 5 |
| Maximum individual-action preflop MAE | 8.29443% | <=5%; currently a root diagnostic, not whole-game reach-weighted MAE | 3 |
| Primary-action agreement | 64.49704% | >=85%; inspect EV ties without silently changing the gate | 3 |
| Maximum aggregate action delta | 5.25829 percentage points | <=3 points | 3 |
| Strict trained preflop coverage | 132,600/132,600, both seeds | Preserve 100% | 3, 5 |
| Full-hand held-out coverage | Unqualified | >=99.99%, authentic and forced-deviation trajectories | 5 |
| Action-EV precision | Unqualified for native candidate | SE <=0.02bb on >=95% of reach-weighted served decisions | 5 |
| Full-game response resistance | Strong combined candidate not measured | <0.50bb/hand, then <0.05; report attacker limitations and statistical uncertainty separately | 4 |
| Re-solving consistency/safety | No opponent-CFV safety guarantee | Correct pinned continuation and bound provenance; no unsupported global safety claim | 4 |
| Probabilities / serving integrity | Preflop max sum error 4.44e-16; native full-hand unqualified | Legal, finite normalized outputs, immutable model pinning, no fallback guessing | 5 |
| Local resources | Cold probe ~1.19GB; disk near 20GiB reserve | Preserve reserve; bounded 2-4 workers and aggregate memory guard | All |

Completed evidence to retain: exact preflop averaging eliminated missing averages
without changing regret training/RNG; stronger played turn continuation improved
two conditional-root results ~28-31%; 32 flop x64 training-turn beat 16x128 on both
seeds. None of these establishes full-game exploitability.

## Execution rules

- [x] Record this sequence, baseline, metric definitions, research, and stop rules.
- [ ] Preflight the cost and storage of each data-generation stage. Preserve at
  least 20GiB free disk; never delete user data or trained artifacts to meet it.
- [ ] Pin inputs, source/binary hashes, seeds, solver budgets, schemas, output
  digests, timings, and resource guards. Partial output is never an accepted corpus.
- [ ] Keep changes opt-in and the existing website model unchanged throughout pilots.
- [ ] Update the execution log below with actual results and unresolved work.

## Step 1 — Native-referenced fast continuation pilot

Research: [DeepStack](https://arxiv.org/html/1701.01724) replaces deep subtrees
with range-conditioned per-hand counterfactual values evaluated at the current
iteration's ranges. Its theoretical assumptions are not established merely by
training a neural network or using a restricted action tree.

1. [x] Introduce a small continuation-evaluation seam shared by the native
   reference and experimental learned adapter. Retain exact card removal,
   opponent-reach scaling, chance correction, action tree, and immutable updates.
2. [x] Implement native-label export from actual intermediate flop iterations without
   changing the policy, RNG, or ordered backups. Include both 1326-hand vectors,
   exact public state/ranges, frozen policy identity, finite-budget response
   residuals, and zero-own-reach completion semantics.
   Implemented with an ordered observer; runtime policy-parity and deterministic
   worker-order tests pass. Native and predicted diagnostics stay distinct.
3. [x] Run a bounded 16-state cost/provenance preflight. Label source is a
   finite-budget native policy, not exact equilibrium. Preserve null residuals
   for zero-joint-reach branches; do not convert them to passing zeroes.
4. [x] Only after preflight, generate a 256-512-state feasibility dataset and
   train two small models using the existing public-value trainer.
5. [x] Compare native and learned leaves with the same flop updates/chance seeds
   and unchanged played turn/river continuation. Score resulting policies using
   independent native response evaluation, not their own predicted values.
6. [x] Benchmark total feature construction plus inference plus search. A ~10x
   speedup is an initial engineering milestone, not a release or quality gate.
   Completed learned128 process timings include these operations: original
   root **10.190/11.442s**, second board **9.612/9.838s** for refreshed models.
   Historical native32 second-board processes took **3004.732/2995.285s**, but
   historical scheduling and worker settings differ; do not call their ratio
   a controlled speedup or serving p95. The retained coverage candidate improves
   all three learned references; historical native32 still wins some contexts.

Advance only when the replacement substantially reduces cost while retaining or
improving independently measured policy response resistance. Do not lengthen the
website timeout or serve a partially solved/unqualified policy.

## Step 2 — Correct target semantics and search-distribution coverage

Research: [ReBeL](https://papers.nips.cc/paper/2020/file/c61f571dbd2fb949d3fe5ae1608dd48b-Paper.pdf)
motivates training on public beliefs encountered during search, including
intermediate iterations and exploration. Borrowing this data strategy does not
implement ReBeL's complete algorithm or transfer its guarantees.

1. [x] Sample intermediate, final, low-reach, and forced-deviation public beliefs
   across boards, pot sizes, and betting histories. Keep a source-distribution tag.
   Explicit forced-root check/least-used non-all-in bet diagnostics cover 16
   additional states across two roots and both frozen proposers. This is not
   complete held-out trajectory coverage or an authentic population estimate.
2. [x] Implement the correct native target/data contract: positive-own-reach profile CFVs;
   zero-own-reach counterfactual BR completions; raw CFVs scale by original
   opponent reach, never own reach. Distinguish raw CFVs from conditional EVs.
3. [x] Give meaningful supervision to legal zero-own-reach holdings. Keep
   counterfactual training weights separate from true reach weights used for
   zero-sum projection, range pooling, and on-policy evaluation.
   Default native loss support is 90% own range + 10% board-legal uniform,
   multiplied by compatible opponent mass. It does not modify either range.
   This fraction is a pilot setting, not an empirically selected optimum.
4. [x] Implement splitting by public board/history family independent of suit augmentation; variants,
   intermediate iterations, and repeats from the same family stay together.
   Whole suit-canonical flop families, including all turns/histories, are held
   together. Fewer than three families is rejected before feature construction.
5. [x] Report value errors, action-ranking errors, and resulting response gains.
   RMSE/KL are diagnostic, not independent reasons to promote a policy.
6. [x] After two failed matched pilots, stop dataset/iteration scaling. Identify
   concrete harmful range/action predictions before another experiment.

## Step 3 — Preflop stability against the served continuation

Research: [VR-MCCFR](https://arxiv.org/abs/1809.03057) gives unbiased
baseline-corrected updates; a poor baseline need not reduce variance. Keep the
existing validated DCFR/exact-average implementation and prior failed-pilot evidence.

1. [x] Pin the continuation function/version and use the existing dedicated
   tabular preflop solver; do not grow the full-hand table again.
2. [x] Compare independent solver seeds against the same continuation function,
   then separately vary continuation sampling to distinguish regret convergence
   from continuation noise. Values must still respond to evolving input ranges.
3. [ ] If regret convergence dominates, increase only the small preflop solve at
   bounded checkpoints. If variance dominates, reuse exact private-hand
   integration and pilot correctly weighted control variates.
4. [ ] Check all three existing stability targets, complete trained coverage,
   and independent response gains. Report near-EV ties separately; never blend
   seeds just to manufacture agreement or silently redefine MAE.

## Step 4 — Combined policy consistency and complete-hand evaluation

Research: [Safe and Nested Subgame Solving](https://arxiv.org/abs/1705.02955)
explains why an independently improved subgame can worsen an overall policy.
[AIVAT](https://arxiv.org/abs/1612.06915) reduces evaluation variance under its
information/known-policy assumptions. [LBR](https://arxiv.org/abs/1612.07547)
is a lower-bound attack, not an upper-bound exploitability certificate.

1. [ ] Pin preflop, value model, played continuation, and deterministic search
   settings into one candidate. Validate that training and playback agree.
2. [ ] Check coherent opponent per-hand bound provenance using the existing
   safety machinery and small exact-game controls. Old safe flags alone are not
   a new solution; approximate bounds do not imply a global safety guarantee.
3. [ ] Train multiple responders, including LBR and learned response; check that
   they exploit positive-control weak policies. Freeze responders before test.
4. [ ] Evaluate both seats on matched full deals across diverse boards/history.
   Use disjoint training, tuning, and final holdout data. Existing development
   roots are not untouched validation roots.
5. [ ] Start with short screening batches; derive final sample counts from
   observed variance and a predeclared confidence requirement. Cluster uncertainty
   by independent hand/board, not by correlated private-hand rows. Use AIVAT only
   if its unbiased correction is verified; tighter CIs are not stronger attacks.
6. [ ] Pursue <0.50 then <0.05bb/hand measured response gains. Separate sampled
   uncertainty from unknown attacker weakness, abstraction, and re-solving error.
   Target subsequent changes at actual costly response lines/street failures.

## Step 5 — Action-EV, coverage, and website qualification

1. [ ] Evaluate each legal action against the same frozen subsequent policy,
   with shared random continuations and exact cheap terminals/runouts. Estimate
   chosen/best EV and EV loss; account for selecting the maximum noisy estimate.
2. [ ] Meet SE <=0.02bb for >=95% of reach-weighted served decisions. Check
   reference bias separately. Neural disagreement is not Monte Carlo SE.
   Preserve low-confidence values and warnings in feedback/history.
3. [ ] Measure >=99.99% full-hand lookup coverage on held-out authentic and
   forced-deviation trajectories, with zero illegal/nonfinite action outputs.
4. [ ] Measure cold/warm latency on every street; optimize only measured costs.
   Cache keys include model identity and complete relevant public belief state.
5. [ ] Verify model pinning, immutable artifacts, probability sums after export,
   missing-node pause/retry, and credential isolation. Reuse existing tests.
6. [ ] Integrate only the accepted candidate into the website, then run the
   applicable Rust/Python/TypeScript/build and browser checks. Do not borrow old
   v102 EV/coverage results to qualify this new candidate.

## Execution log

- 2026-09-06: Read baseline implementation and ledger; created this plan. Current
  work begins with native leaf capture and the data/weighting contract for the
  small reference pilot. No training job started and no website settings changed.
- Implemented `counterfactual_turn/flop_pilot/value_targets.rs`, with exact
  intermediate reach/CFV capture, per-leaf frozen-policy hashes, finite-budget
  residuals, deterministic worker-order observation, bounded gzip export, and
  explicit research-only provenance. The existing traversal is the continuation
  seam; the learned adapter remains pending.
- Added `neural/native_value_dataset.py` and connected it to the existing
  `train_public_value_network.py`. Native input is validated before conversion to
  f32; raw CFV/conditional-EV scaling, card removal, zero-own completions, and
  reference residuals are checked. Genuine reach weights remain separate from
  counterfactual loss weights. Legacy input behavior remains covered by tests.
- Added `neural/run_native_value_preflight.py`: pinned binary/input, 16 captured
  states from two native flop iterations, 64 turn iterations by default, 1/2/4
  workers, 900s timeout, 2GiB sampled worker-memory ceiling, 20GiB disk reserve plus
  32MiB startup headroom. This first-N capture is a cost/contract preflight only,
  not a representative training set or policy-quality result.
- Python regression command (from `preflop-solver/neural`):

  ```bash
  ../.venv-neural/bin/python -m unittest test_native_value_dataset test_train_public_value_network test_validate_public_value_parity test_diagnose_public_value_model test_select_public_value_config test_run_frozen_public_value -q
  ```

  Result: **65 tests passed**, including seven new native-corpus/runner tests.
  Existing constant-correlation fixtures emit NumPy divide warnings; no test
  failed and no model-training result is inferred from these checks.

- Rust test code successfully type-checked with `rustc --test --emit=metadata -o -`
  against the existing release dependency artifacts, sending metadata to stdout
  and discarding it. This is not a linked release build or runtime test pass.
  Targeted rustfmt and `git diff --check` pass. Repository-wide `cargo fmt --check`
  also reports existing formatting differences outside these changes; those
  unrelated files were not reformatted.
- Storage blocker: available space dipped below 20GiB. No build caches, training
  artifacts, or user files were removed. Asked the user to free approximately
  2-3GB. No native capture, neural fitting, or website activation has run.
- 2026-09-06 storage resolution (subsequent user-authorized cache cleanup):
  removed approximately 90.8GiB of regenerable feature caches and 5.6GiB of Rust
  debug build output. Feature-cache directories were checked for recognized
  metadata/array content, real paths, and no open handles before removal. This
  included the v81 roots2-9 cache, shared feature-cache, and 20 older experiment
  cache directories (v69/70/78/83/84/86/87/88/90/91/102/107). No source corpora,
  trained weights, checkpoints, validation reports, release binaries, active
  development output, or unrelated user files were deleted. Removed caches are
  not in Trash; regenerate them from source if those experiments are rerun.
  `neural/runs` fell from approximately 163.8GiB to 73.0GiB. Available disk space
  was 124.6GiB after cleanup; the storage blocker is resolved. The remaining
  historical experiment artifacts are not all required for serving, but their
  dependency/retention audit has not been completed; do not delete them blindly.
- Targeted linked Rust release tests did execute: 19 passed, 0 failed, 5 ignored
  (`counterfactual_turn::flop_pilot`). The cargo controller returned exit 130
  because it was interrupted when the user prioritized cleanup; the test log
  nevertheless contains a complete passing test result. No native preflight or
  training run started. Solver work remains paused for the cleanup explanation.
- User resumed the plan after cleanup. Targeted release tests completed with a
  clean exit (19 passed). The 16-state preflight completed in 46.78s with a
  1,177,486,680-byte sampled footprint and 1,841,834 compressed output bytes.
  Captured 18 observed native queries, retained 16 first-N states from rounds 1/2,
  and preserved the archived complete policy SHA
  `5d169bbd0ba1ff6c147732575e228a82a4a207b0d222627645471abf224047da`.
  Dataset SHA `019c869ae253fb2e5307560965df2a84fc65bcf2929b499dad8eb93e96a67caa`;
  output: `preflop-solver/neural/runs/local-native-value-20260906-preflight`.
- Added the opt-in native/learned continuation seam with unchanged exact chance
  backups, a separate `native-turn-cfv-full-stack-v1` prediction contract, and
  full-stack rather than current-investment output bounds. Native artifact
  serialization is preserved; predicted leaves report no measured native
  response residual. Zero-own-reach predictions remain available and convert
  back to raw CFVs using only raw compatible opponent mass. The adversarial
  prediction-contract release test passes. Python regression suite: 67 passed;
  native flop suite: 19 passed / 6 opt-in ignored. Fitted-artifact Rust/Python
  parity and independent policy-response comparison remain pending.
- Multi-family capture started at
  `preflop-solver/neural/runs/local-native-value-20260906-corpus-a`, using compact
  preflop A SHA `de81857c96d387c8089e7c48704b175d79ee2b023ddca26702a91d166504b32c`
  and test binary SHA `2da82b17da6e7fa63f91f2ce48eb8679ac13cc9726fb6fbf4e6ff5b7b8ab3279`.
  Eight authentic live flop roots were sampled in 44 full deals (root seed
  881902), with investments 1/3/7.5/11bb per seat and eight distinct flop
  families. Retain all intermediate leaves, aiming at 256-512 total states;
  four bounded leaf threads inside one sequential worker, 2GiB/900s per worker,
  20GiB free-disk floor. Collection provenance pins each source separately;
  its aggregate identity is explicitly not one frozen policy. No holdout or
  policy-quality claim is inferred from capture completion.
- Eight-root capture completed in 571.677s: **284 states**, eight flop families,
  86,504,058 decoded / 32,802,457 compressed bytes. Maximum sampled native worker
  footprint 1,483,753,248 bytes. Corpus SHA
  `78693b5e2571283ced2c8be2e44ae1b861e49185859ddf8da73d52c2e06f0d78`.
  No capture was truncated or substituted; each leaf retains native provenance.
- First compact/600-step student pair fitted but failed actual artifact parity
  with a maximum **25.1801bb** Python/Rust difference. Failure is preserved at
  `local-native-value-20260906-student-compact`; no policy comparison or activation
  used those weights. Diagnosis minimized the failure to state 45 (0.0093828bb),
  then varied ranges, masses, and investments independently. Only preserving
  f64 ranges eliminated the failure: cancellation while subtracting conflicting
  hands from f32 ranges corrupted small compatible-equity denominators. Keeping
  all native inputs in f64 reduced worst parity error over all 284 states to
  **4.24e-6bb**, with the same frozen weights.
- Regression `test_native_loader_features_preserve_sparse_blocker_cancellation_precision`
  failed before the loader fix and passes afterward. Native ranges/masses/
  investments now stay f64 through feature construction; stored network features
  and loss/projection tensors remain f32. Native feature-cache identity includes
  the precision contract, so old corrupted features cannot be reused. Legacy
  loading/serving behavior is unchanged. Python suite: **68 tests passed**.
  Same-budget corrected pair is running in
  `local-native-value-20260906-student-compact-f64`; it must be refitted because
  the first pair learned from affected features. This is a corrected replay,
  not evidence that additional iterations improved strategy.
- Corrected compact pair completed in 66.669s, with 143 training / 69 tuning /
  72 holdout states (4/2/2 whole flop families). Actual Rust/Python inference
  agrees over all 284 states and every holding to <=4.45e-6bb. Authentic holdout
  RMSE is 1.73800 / 1.65379bb; counterfactual-weighted RMSE is 1.84164 / 1.76391bb.
  These are approximation diagnostics, not policy acceptance. Manifest SHA
  `4809034ff956019f374f6c1b18f64019e19f9ba4192247b56b0d935874d1542d` at
  `preflop-solver/neural/runs/local-native-value-20260906-student-compact-f64`.
- Independent paired policy check is running at `local-native-value-20260906-response-compact`:
  32 flop updates, matched chance seeds, and unchanged 64-iteration native played
  turn/river continuation; all 49 public turns are integrated before flop
  response maximization. First seed cold learned search took 2.738s, but response
  gain worsened from 0.227128 to 0.394826bb. Flop-only restricted gains changed
  from [0.181960, 0.216866] to [0.400626, 0.341180]bb: the deterioration is already
  visible in flop decisions without giving the attacker additional turn/river
  freedom. Independent JS accounting agrees to 2.78e-17bb. Second seed is pending.
  This reused-root result is not full-game exploitability. Older serial/shared-host
  native timings are not a fair denominator for a claimed 1,000x speedup.
- A separate suit-augmentation regression reproduced a 0.014421bb projection
  error: Python parity inference used the unpermuted source ranges after building
  permuted features. Projecting with the actual loaded row's f64 ranges fixes the
  test without altering features. Current pilots use one suit variant, so their
  results are unaffected; the fix protects the existing 24-variant option.
- First matched policy pair completed in 2,062.510s. Conditional response gains
  were **0.394826 / 0.393384bb**, versus **0.227128 / 0.229962bb** for matched native
  controls. Both seeds regress, so the fast replacement is rejected. This is
  one failed matched pilot; the earlier f32 parity failure was not a policy test.
- Added opt-in action diagnostics to the independent JS audit, with analytic
  tests for raw-CFV / conditional-EV / joint-reach scaling. Default audited
  response arithmetic is unchanged. Diagnostics reuse saved packets, evaluating
  each action against the same frozen subsequent play; one-node contributions
  must not be added up and called exploitability. Both learned seeds lose about
  0.91-0.93bb locally when facing the 2.5bb flop bet, versus smaller native-control
  losses. Concrete seed-A examples include A-spades/T-spades continuing 58.4%
  although all continuations are worse than folding, and A-diamonds/A-clubs
  shoving 78.6% facing 1.5bb although calling is worth 6.689bb versus 4.705bb
  for shoving. These are values against this candidate's frozen later policy,
  not universal recommendations for those holdings.
- Started `local-native-value-20260906-student-exact-equity`: change only the
  existing input feature from immediate turn equity to exact river-runout equity.
  Retain the corpus, whole-family split, compact architecture, 600-step limit,
  seeds, loss weighting, and later matched search budgets. This also improves
  the existing checkdown residual baseline's equity input; it does not change
  labels or add more training. Run actual artifact parity and total inference
  cost checks before another native policy comparison. Python suite: 69 passed;
  independent-audit/action-diagnostic JS tests: 2 passed.
- Rare-range parity check reproduced a 0.001bb discrepancy for nearly
  incompatible ranges (joint reach 2^-60): the Python bounded projector used
  a 1e-12 denominator floor while native serving uses 1e-9. Matching the native
  cutoff fixes the exact-binary-weight Python regression; a corresponding Rust
  regression is added for the final release-test run. The current corpus has
  no positive-joint states below 1e-9, so this does not change either pilot's
  predictions. Existing Rust serving math and the running pinned binary stay
  unchanged; this is a parity-contract fix, not an exploitability improvement.
- Exact-equity policy seed A completed its independent evaluation at
  **0.374240bb**, versus 0.394826bb for the immediate-equity student and 0.227128bb
  for the native control. Cold search took 3.818s (guarded process time). The
  feature helps this seed modestly but still fails replacement acceptance.
  Its one-node loss facing the 2.5bb bet remains 0.89100bb, versus 0.47707bb
  for the native control. Seed B is still running. If the pair fails, use the
  already planned search-distribution remedy rather than more epochs: freeze a
  proposer model, sample its intermediate/final beliefs, and solve those exact
  beliefs natively for labels. Keep predicted values out of the target path,
  retain whole-flop-family splitting, and preflight cost before another pilot.
- Second pair completed in 2,100.045s: exact-equity gains **0.374240 / 0.378954bb**,
  improvements of 0.020586 / 0.014430bb over the first learned pair, but still
  0.147112 / 0.148992bb worse than native controls. Both replacements are rejected.
  The plan's two-failed-pilot diagnostic trigger is reached; no epoch, dataset-size,
  or long-run scaling is authorized by these results. Concrete harmful action
  predictions are recorded in the action diagnostics above.
- Implemented the planned distribution remedy at the existing continuation seam:
  a state-only observer cannot expose learned predictions as native labels.
  Stratified reservoirs cover early/middle/late search, with a separate final
  frozen-average belief sample. Sampling has independent RNG state; tests verify
  unchanged policies across observation and worker order. Each selected belief
  receives a fresh 64-iteration native solve, with its own policy hash and explicit
  distribution tag. Whole flop-family splits and the original 284-label budget
  are preserved. This is a targeted distribution change, not blind scaling or a
  claim to implement ReBeL's complete algorithm.
- New `run_search_distribution_pilot.py` requires the completed source corpus,
  fitted/parity-checked proposer pair, and completed policy comparison. It first
  labels 16 states from a low-pot nine-leaf root and verifies full proposal-policy
  parity with and without observation. Only then does it label the same eight
  authentic root families. Limits remain four native threads, 2GiB/900s per root,
  20GiB disk reserve, and a two-hour stage cap. Partial or missing source strata
  cannot be merged as a completed corpus.
- Rust release library suite passed: **310 tests, 0 failures, 39 opt-in ignored**
  in 179.85s. Includes the new observer/sampler and near-incompatible-range cutoff
  regressions. Native Python data tests: 13 passed. New release test executable SHA
  `92c806562a10a3744d4a864d1fb6ca8a1e1314bfd09d690a31c35f1390eca17f`.
  Search-distribution preflight/corpus job started at
  `preflop-solver/neural/runs/local-native-value-20260906-search-beliefs-a`.
- Search-distribution preflight passed in **50.120s**, sampled peak footprint
  **871,777,552 bytes**. Full 32-round proposal policies are identical with/without
  observation. The 16 labels comprise 3 early, 2 middle, 2 late, and 9 final-average
  states out of 297 observed/proposed queries. Maximum absolute conditional value
  is 19.242bb. Native finite-budget conditional reference gain is 0.02204bb median,
  0.07441bb maximum; this is not a full-game result. The same-root 284-label corpus
  is now being generated sequentially under the resource guard.
- Four-worker search-belief capture stopped safely at root 5: the labeling
  process reached **2,171,439,144 bytes**, just above its 2GiB ceiling, after
  30.825s. Five roots (0-4) and the preflight are complete and preserved; no
  truncated corpus was exported. The learned proposal had already finished.
  This is a labeling-memory issue, not a policy-quality result.
- Added strict hash-checked reuse of completed captures and a 1/2/4-worker
  scheduling option. A fresh two-worker retry is running in `search-beliefs-a-retry2`.
  It repeats the 16-state preflight and requires byte-identical output to the
  four-worker preflight, then reuses roots 0-4 and completes only missing roots.
  The model, RNG seeds, labels, and 64-iteration solve budget remain unchanged.
  The 2GiB cap is not raised. Previous failed manifest SHA:
  `1e314e5654fa7bc25e1cbe9143843da0ccee2032693a3e101d6dd67b1f404818`.
- Two-worker preflight passed in **64.084s**, with **666,289,256 bytes** peak
  footprint, and produced a byte-identical compressed dataset to the four-worker
  run. Both proposal and native-label worker-order parity therefore hold on this
  real preflight. Completed roots 0-4 passed integrity checks and were reused;
  only roots 5-7 remain to be labeled. No completed capture file was discarded;
  the interrupted root's unsaved partial work must be repeated.
- The previously failing root 5 completed with two workers in **148.549s**, peak
  **1,152,795,920 bytes**. Roots 6/7 then completed in 158.906/53.437s. Lowering
  concurrency resolved the observed memory failure without changing the corpus
  or numerical budget. The resumed stage completed in 434.186s; it retained all
  five prior completed roots rather than recomputing them.
- Search-distribution corpus completed: **284 states**, eight families,
  88,502,356 decoded / 36,768,780 compressed bytes. Corpus SHA
  `410643a85b39de6d909b4a0acb4a2c890a62b5c741fccf3a684746a8b704e487`.
  Maximum footprint among all retained successful captures is 2,048,641,136 bytes,
  including reused four-worker captures; all are below the 2GiB limit. The earlier
  interrupted capture is not included. Same-budget exact-equity student fitting
  started in `local-native-value-20260906-student-search-beliefs-a`.
- Verified identical train/tuning/holdout indices (143/69/72 states; 4/2/2 whole
  flop families). The same 284-label budget now covers **131 exact turn boards**,
  versus 36 originally, with 74 early / 74 middle / 74 late / 62 final-average
  states. Native conditional reference gains: median 0.01553bb, p95 0.06958bb,
  maximum 0.09082bb. These are finite-budget label diagnostics, not model acceptance.
  Python regression suite: **71 passed**; JS audit/diagnostic tests: **2 passed**.

- Search-distribution students completed in **525.409s**, including fitting and
  actual artifact checks. Rust/Python maximum disagreement is <=5.11e-6bb for
  both seeds over all 284 states. Authentic holdout RMSE is **1.79340 / 2.20532bb**
  on the new belief distribution; these errors are not directly comparable to
  errors on the old corpus. Completed student manifest SHA
  `d1462195ad188ac3d6df7f2c515c5ed7d75f568593a3c78752ea9d82139ad488`.
  Third matched policy comparison started at
  `preflop-solver/neural/runs/local-native-value-20260906-response-search-beliefs-a`.
  The same 32/64/64 budgets, native controls, seeds, and complete 49-turn
  integration remain fixed. No additional fitting or model selection is done
  using this response result while it runs.
- Added `diagnose_native_value_errors.py`, reusing saved Rust predictions and
  native labels without new solves, feature construction, fitting, or large
  caches. Reports separate train/tuning/holdout flop families, search strata,
  investment bands, per-seat errors, and zero-own-reach values. Missing reach
  produces null statistics. Public states are sampled, so these private-hand
  reach weights must not be presented as full-game reach. Combined signed bias
  near zero is enforced by projection and is not evidence of accurate values.
- Third-pair diagnostics: authentic within-state training RMSE **1.46016 /
  2.06341bb**, holdout **1.79340 / 2.20532bb**. Thus unseen-board generalization
  is not the sole remaining issue: the retained models have substantial errors
  on training families too. Holdout zero-own-reach RMSE **2.70458 / 2.95124bb**,
  signed bias **-1.10647 / -1.31193bb**. Two particularly costly holdout examples
  are 15bb turn pots after flop bet/raise/call, with state RMSE **3.60 / 4.02bb**
  and **3.11 / 4.16bb**. These errors diagnose approximation failure, not Monte
  Carlo SE, action rankings, or full-game exploitability. The matched policy
  comparison is still required. Diagnostic report:
  `local-native-value-20260906-student-search-beliefs-a/value-error-diagnostics-per-seat.json`.
- Python suite passed **75 tests** before adding the opposite-seat-bias test;
  the focused diagnostic suite subsequently passed all five tests. Independent
  JS audit/diagnostic tests: two passed. A new report serialization regression
  reproduced and fixed a NumPy index conversion error; the incomplete report
  was retained with `.failed-partial` suffix and never treated as completed data.
- Third policy pair seed A completed: **0.314342bb** conditional response gain,
  improved by **0.059898bb** from the previous exact-equity student (0.374240bb),
  but still **0.087214bb** worse than the native control (0.227128bb). Response SHA
  `ddc411ea2607c984a49393324ef3429850b4fc38db1f6e63f75119ada4af196b`.
  Cold learned search took 3.895s guarded process time. The native independent
  audit passed; seed B remains pending. This supports the search-distribution
  remedy on one seed, not replacement acceptance or full-game qualification.
- Checked whether zero-own-reach prediction errors simply reproduce the wrong
  frozen-profile targets instead of their required best-response completions.
  On training rows both students are closer to completed labels (RMSE
  2.584/2.813bb) than to frozen-profile values (3.745/3.624bb). Thus that simple
  explanation is not supported. Holdout differences are mixed; the check does
  not establish that support discontinuities are harmless. Native completion
  semantics remain unchanged; the next controlled variable remains capacity.
  The report preserves raw opponent-mass normalization and its analytic test.
- Third paired comparison completed in **2,001.913s**. Conditional response gains
  are **0.314342 / 0.597602bb**, versus native **0.227128 / 0.229962bb**. Relative
  to the previous learned pair, A improves 0.059898bb but B regresses 0.218647bb.
  The distribution remedy is therefore not a successful paired replacement.
  Completed comparison manifest SHA
  `8c03bae1c3e3ca3e8f9cdf5e6cf60a2baa797d6cee0b064cd724ec3c5fff934f`;
  seed-B response SHA
  `f4bc04bba790b70aaa7ae47022009a878d6df366ef75cdc3d77316d3739859f2`.
  Both independent audits passed. Seed B's action diagnostic is saved beside
  its response. No partially evaluated chance set was used for this verdict.
- Started `local-native-value-20260906-student-search-beliefs-wide` only after
  the third policy pair completed. Same corpus, feature schema, two seeds,
  600-step ceiling, early-stopping rule, split and loss; only architecture changes
  from `compact` to the existing `wide`. Actual prediction output hashes are now
  recorded alongside future parity results; older reports are not rewritten.
  No additional native labeling, paid compute, or website activation.
- Wider fitting/parity completed in **532.661s** (fitting 180.759s; sampled
  fitting footprint 2,219,886,392 bytes, below its 6GiB guard). Both students
  select step 250 and stop at step 550 under the unchanged 600-step ceiling.
  Actual artifact parity is <=5.35e-6bb. Same-corpus authentic holdout RMSE is
  **1.740977 / 1.673583bb**, improved from compact **1.793398 / 2.205324bb**.
  Rust feature/inference totals for all 284 states are 5.302/5.290s. Completed
  student manifest SHA
  `215a8412ef026c0a9687464d778216b2cec6571facc08fde3a772f2513ba0f8c`.
  Fourth matched policy comparison is running at
  `local-native-value-20260906-response-search-beliefs-wide`, retaining the same
  32/64/64 budgets and native controls. Better value fit alone is not acceptance.
- Same-state diagnostics support capacity as one contributor: authentic
  training-family RMSE falls **1.46016/2.06341 -> 1.20618/1.17358bb**. Holdout
  zero-own-reach RMSE falls **2.70458/2.95124 -> 2.57243/2.48933bb**, still large.
  No claim is made that wider networks remove the generalization or support
  limitations. Cold seed-A learned flop search is **3.898s**, essentially
  unchanged from compact **3.895s**; all-turn independent evaluation is pending.
- Fourth policy pair seed A completed at **0.380220bb**, worse than same-data
  compact **0.314342bb** and native **0.227128bb**, despite improved value RMSE.
  Its independently audited response SHA is
  `e219951dcf297a65035a260e8c5e6268c020293f1945b4f0658def9bb0fc0bfe`.
  The capacity change improves approximation diagnostics but does not establish
  better decision quality. Seed B remains running; do not promote this model.
- Fourth pair completed in **2,318.737s**: conditional response gains
  **0.380220 / 0.433073bb**, versus native **0.227128 / 0.229962bb**. Both still
  fail replacement acceptance. B improves from same-data compact 0.597602bb,
  while A worsens from 0.314342bb. Completed comparison manifest SHA
  `adb6b87faed3d59e7e578d25874f153f3371ea83a97c73745ebbea1554edc2d1`;
  seed-B response SHA
  `0a36168b648bebea3475fcf08720ed9848db0e8cc6aedf19327c936f2578f368`.
  Every native control replay and independent audit passed. No model accepted.
- Next controlled coverage hypothesis: the wider model fits the four training
  flop families better (~1.2bb RMSE) than unseen families (~1.7bb), yet its
  decisions on the development root remain exploitable. Add **128 native labels
  from eight new authentic flop families** to the existing 284 states, for
  **412 total**, still within the original 256-512-state feasibility budget.
  Preserve all existing targets and their exact tuning/holdout membership; new
  families are training-only. Preserve the original frozen search proposers,
  64-iteration native labels, wide architecture, seeds, loss and 600-step ceiling.
  This tests public-board/belief coverage, not the proposition that more fitting
  iterations alone solve the problem. Broader position/range coverage follows
  the already cited DeepStack/ReBeL motivation, without importing their guarantees.
  First verify authentic-root prefix parity and a 16-state, two-worker preflight.
  Reject any new family overlapping tuning/holdout or any altered old target.
- Implemented an explicit hash-pinned native split reference in the trainer
  and student controller. Appended families train only; the full old target
  prefix and game must remain identical. Tests reject changed targets, changed
  games and suit/turn variants of held-out flop families. The controller records
  exact split indices and prediction-file hashes; diagnostics use that partition.
- Implemented `run_native_family_extension.py`: checks the existing root prefix,
  uses the same frozen preflop policy and search proposers, skips duplicate flop
  families, and adds eight 16-state native captures with two workers each.
  The first new capture is the cost/parity preflight; no later capture starts
  until it passes. The final merge must contain exactly 412 states and preserve
  **271 training / 69 tuning / 72 holdout** indices. No corpus is accepted on a
  partial collection. Limits: 2GiB/900s per native worker, 20GiB disk reserve,
  two-hour collection cap. The old 284-state corpus is not rewritten.
- Rust release library suite passed again: **310 passed, 0 failed, 39 opt-in
  ignored**, 207.89s after compilation. New executable SHA
  `c7a0ff8674ef557c0cd438d0ff0cb0d854d5781c1d11b732c1b8fb1c245f6fe2`.
  The Rust change only parameterizes the opt-in authentic root sampler's count;
  default remains eight. Numerical solver code is unchanged in this extension.
- Full Python regression suite: **81 passed** in 9.497s; independent JS
  audit/diagnostic suite: **2 passed**. The one-step synthetic native CLI
  integration test verifies the actual exported report keeps reference tuning
  and holdout membership; it is not a real policy-training result.
  Bounded extension collection started at
  `preflop-solver/neural/runs/local-native-value-20260907-diverse-families-a`.
- Family extension completed in **284.316s**. Original eight public roots match
  exactly. The first 16-label preflight passed proposal-policy parity in 8.668s;
  all eight new captures completed under the guard (peak **964,412,808 bytes**).
  New corpus: **412 states**, 129,277,566 decoded / 53,749,401 compressed bytes;
  SHA `6bee05343756695d76d56fb0d407b8a048f3a18ba5accf4d23f4b126b209991d`.
  The entire original 284-target prefix is unchanged. The fixed split is
  **271 train / 69 tuning / 72 holdout**, now 12/2/2 flop families.
  New root families include paired, trips and monotone boards; preflop investment
  ranges from 1 to 11bb per seat. No gate result follows from corpus completion.
- Started `local-native-value-20260907-student-diverse-wide-a`: same wide
  architecture, seeds, 600-step ceiling and exact-equity features, with the old
  corpus hash explicitly pinning the tuning/holdout split. Next requires actual
  artifact parity and independent policy evaluation, not only improved RMSE.
- Extended-corpus fitting completed in **264.850s**, sampled peak footprint
  **2,927,266,120 bytes** under the 6GiB fitting guard. A selects step 250/stops
  550; B selects/completes step 600. Training-report holdout RMSE is mixed:
  **1.78274 / 1.56027bb**, versus previous wide **1.74114 / 1.67390bb** using the
  same report calculation and unchanged holdout. Both tuning scores improve
  (2.02784/1.98286 -> 1.81781/1.69865bb). Actual artifact parity and policy
  response comparisons remain pending. Do not infer acceptance from tuning.
- Extended-corpus student verification completed in **793.861s** total. Actual
  Rust/Python parity maxima are **5.83380e-6 / 5.69052e-6bb**; actual authentic
  holdout RMSE is **1.782792 / 1.560123bb**. Completed student manifest SHA
  `1953af49c4afc2390e9476505b2caa0843c41fd3f667e9bcfe56704a33fe010f`.
  Both seeds remain experimental. The fifth matched 32/64/64 policy comparison
  is running at `local-native-value-20260907-response-diverse-wide-a`.
  Saved-prediction diagnostics were produced without another solve; their
  private-hand weighting is not full-game public-state reach weighting.
- Fifth policy comparison seed A completed at **0.319894bb** conditional response
  gain: better than the preceding wide model's 0.380220bb, but **0.092766bb worse**
  than native 0.227128bb. Its independent audit passed; response SHA
  `3c0d6c31511a6118570313cb9730bc65bc0cd76d88b8d5d80603b6567ff617fe`.
  Cold learned search took **3.817s**. This coverage change helps this comparison,
  but does not pass replacement acceptance. Seed B remains pending.
- Fifth pair completed in **2,258.228s**, with all 49 public turns per seed and
  both independent audits passing. Gains are **0.319894 / 0.443959bb**, versus
  native **0.227128 / 0.229962bb**. B is 0.010886bb worse than the preceding wide
  student despite its improved holdout value RMSE (1.67358 -> 1.56012bb).
  Cold learned searches took **3.817 / 4.082s**, including feature construction
  and inference; these are two development-root observations, not a p95 or a
  same-host native speedup measurement. Completed comparison manifest SHA
  `0dfb6624d7823270bcba1e65cb4ac9ecf846e1fadc90eddb90b971e669511c3e`;
  seed-B response SHA
  `77b4df4fe3895fef7ba39d126f45fd576a8b43c746763ed25a52939556798166`.
- Saved action diagnostics isolate a substantial flop-policy weakness without
  additional native solves. Restricting the responder to flop deviations while
  leaving both players' turn/river play unchanged still produces half-summed
  gains **0.287034 / 0.414060bb** (89.7%/93.3% of this conditional attack's total).
  This ratio is descriptive, not a decomposition of true full-game exploitability.
  In B, after check/bet-to-2.5bb, 7c4h folds only 20.4% despite native frozen-play
  EVs fold -1, call -1.8505, raise -4.2989 and shove -5.1044bb; its resulting
  local deviation loss is 1.7406bb. Do not turn one frozen-policy example into
  a general poker recommendation. Audit differences are 0 / 5.55e-17bb.

## Quality findings and continued work

All comparisons below use the same reused development flop, matching 32 flop
updates and unchanged 64-iteration played native turn/river continuation. Lower
is better. These are conditional response gains, **not full-game exploitability**.

| Continuation used to train flop actions | Seed A (bb) | Seed B (bb) | Retains both native controls? |
| --- | ---: | ---: | --- |
| Native reference | 0.227128 | 0.229962 | Reference |
| Compact, immediate-equity features | 0.394826 | 0.393384 | No |
| Compact, exact-runout features | 0.374240 | 0.378954 | No |
| Compact, search-distribution labels | 0.314342 | 0.597602 | No |
| Wide, same search-distribution labels | 0.380220 | 0.433073 | No |
| Wide, eight additional training families | 0.319894 | 0.443959 | No |

The fast adapter works and numerical/export parity is verified. It does not yet
preserve the native solver's decision quality. Better aggregate value RMSE has
not reliably improved the policy. The bounded feasibility work therefore has
not met step 1's explicit condition for sequential advancement. This is not
evidence that neural continuation is impossible, nor justification to relax the
condition, fit preflop against an unaccepted replacement, or deploy it.

No further blind epoch or dataset expansion is scheduled. A revised experiment
must target the observed action-value/ranking errors, or test an alternative
native-cost reduction that does not replace native values with biased estimates.
Larger data coverage, action-sensitive fitting and unbiased learned baselines
remain hypotheses, not implemented successes. Larger runs require a measured
resource preflight, not a new user permission for ordinary in-scope local work.
The present 256-512-state pilot is not silently expanded into a large run.

Remaining dependent work: broader forced-deviation coverage; complete latency
benchmarking; preflop stability against the accepted continuation; pinned
complete-hand evaluation and safety provenance; action-EV precision and serving
coverage; accepted website integration. All remain explicitly unqualified.
The existing website model and unrelated user files are unchanged.

Verification at the earlier pause (before the resumed diagnostic changes):
Python regression suite **81 passed** (9.280s), independent
JS audit/diagnostic suite **2 passed**, and `git diff --check` clean. The last
Rust release library run on this binary was **310 passed, 39 opt-in ignored**;
no Rust code had changed after that run at that point. No app/worker/browser code changed, so npm
build/browser checks are deferred to actual integration, not claimed as passed.
At that pause, no pilot controller or packet worker remained running and free
disk was **125GiB**. No new deletion or website activation occurred.

## Immediate next actions

1. Reuse the saved fifth-pair native packets and frozen policy. Export the actual
   learned values at exactly those final public beliefs, without new native solves.
2. Compare native and learned action rankings with identical future policy,
   card removal and all-turn integration. Distinguish erroneous value rankings
   from unfinished policy optimization; do not equate aggregate RMSE with quality.
3. Implement and test the correction supported by this diagnosis, then run a
   bounded paired comparison. Failed pilots trigger revised experiments, not
   an automatic pause. Advance to step 3 only after the continuation qualifies.

## Continued execution after correcting the premature pause

- Reproduced the fifth-pair regression with the real independent audit and an
  explicit native-control assertion in **0.36s**, using saved packets. The failed
  policy is research evidence, not a reason to stop otherwise feasible work.
- Added a prediction-only Rust probe at the existing frozen-belief interface.
  It visits every turn/final-history belief without fitting or native solving;
  exports explicitly predicted raw CFVs, never native response estimates.
  Added action-ranking diagnostics that leave audited native scores untouched.
  Analytic tests detect wrong rankings versus harmless common offsets, reference
  identity mismatches, missing/duplicate turns, and zero-reach cases. Actual Rust
  reach-scaling and immutable-policy regression passes. Three JS tests pass.
- Actual paired prediction/action probes completed at
  `local-native-value-20260907-action-values-a`; Rust worker times **3.569/3.571s**.
  Manifest SHA `72b481eaa5e90b9de4dd82d00d3b1e8ebac54c960e56ae31975cc749ab3f61fe`.
  Native audit results are unchanged. In B after a check, native/predicted best
  action agreement is only **25.23%**, with **0.26381bb** native loss from the
  predicted best action. A facing 1.5bb with AsAc estimates call/raise EVs
  **3.0636/4.5885bb**, versus native **6.1333/6.5667bb**, while the exact shove EV
  remains **5.4785bb**: a concrete ranking reversal, not just a large RMSE.
- The same probe also isolates optimization shortfall independent of the value
  network: facing an all-in, values are exact, yet A/B frozen policies still lose
  **0.10377/0.32077bb locally**. Thus inaccurate leaf values are not the only
  remaining cause. These local losses are neither additive nor full-game gains.
- Next bounded compute-allocation test uses **128 flop updates with identical
  weights**, rather than retraining the network or adding corpus states. This
  targets the observed residual optimization error; it cannot eliminate value
  bias by itself. The distinction between finite search error and value-function
  error also follows the already cited DeepStack analysis; our restricted tree
  and current bounds do not inherit its global guarantee.
- Extended the existing response runner with explicit 32/128 update settings.
  Extra updates require a completed, hash-pinned, same-weight 32-update reference.
  Reports distinguish comparison against learned32 (same weights) from native32
  (**unequal update budgets**, a cost/quality comparison). Added a budget/reference
  regression test; it passes. No game or action-grid changes, no relaxed quality
  threshold, and no website changes.
- Started `local-native-value-20260907-response-diverse-wide-128-a` with the
  unchanged native played-turn budget of 64 and the same four guarded packet
  workers. Binary SHA `1f154f8891f4d8c489d5a982112e0e7dc37f9f349acd1e9bf61b877bef4bcafb`.
  Finish both all-turn audits, repeat the cheap final-belief diagnostic, and
  target whichever of optimization or value ranking remains limiting. Do not
  stop solely because another candidate fails promotion.
- The 128-update seed A completed at **0.206310bb** conditional response gain,
  improved from same-weight learned32 **0.319894bb** by **0.113585bb (35.5%)**.
  It also improves over native32 **0.227128bb** by 0.020819bb, with the explicitly
  different flop-update budgets. Independent audit passed; response SHA
  `af3ef06aeafdcbea948804b9c80c5fa161c146011471a1ab9b266eb4703e37bf`.
  Search took **10.190s** versus learned32 3.817s. Seed B is running. This supports
  the diagnosed optimization remedy on one seed, not paired acceptance or
  full-game qualification. No new weights were fitted.
- The full 128-update pair completed in **2,652.434s** with gains
  **0.206310 / 0.203169bb**, versus same-weight learned32 **0.319894 / 0.443959bb**
  and native32 **0.227128 / 0.229962bb**. Both independent native audits passed.
  Both seeds now improve on both references under the explicitly unequal
  native-update budget. This is successful development-root cost/quality
  evidence, not full-game or website qualification. Search times **10.190/11.442s**.
  Comparison manifest SHA
  `9e8696e182a0420d134dbf6cb2a13b3f65e11a7f87a48e9127ef71cea746d2b6`;
  B response SHA `b71ad8dad00f354b22c5181856d27df0f0ee71bea25e1ae71a57c5ab2474d90b`.
  Repeating the cheap final-belief diagnostic before expanding board/belief
  checks. The successful remedy increased actual policy search, not fitting
  epochs, changed weights, blended seeds, or relaxed gates.
- The final-belief probe at 128 updates completed in **9.343s** without native
  solves; manifest SHA
  `3717664cecba675ca4fdc30f9f08270505cbeb83524e82e89d604ea46108a900`.
  Predicted root policy-deviation loss falls **0.09025/0.12489 ->
  0.01064/0.00943bb**. After a check it falls **0.11036/0.13888 ->
  0.02167/0.03634bb**, while native loss there remains **0.16659/0.17342bb**.
  Thus further blind flop-update scaling is not the next remedy: value-ranking
  error remains significant in this high-reach line. Local diagnostics are not
  added together or used as full-game estimates.
- Located completed native controls and all-turn packets for the two previously
  studied broader contexts (different flop/prior and a three-bet facing-action
  root). Reuse them rather than regenerate controls. Original broader manifest
  SHA `eab6dc2137623417bad5bfdffb9cc40d18edb90e6491204816f0d7cb6fbbf842`.
  These are reused development contexts, not untouched final holdouts.
  Added explicit control-context selection and a successful fixed-128 transfer
  reference to the response runner. Tests reject changed weights, failed or
  incomplete source pairs, ambiguous contexts and old eight-update controls.
  Full Python suite now **84 passed**; JS suite **3 passed**. Full Rust release
  verification is running before the transfer experiments. No new model fit.
- Full release verification completed: **311 library tests and 9 CLI tests
  passed**, 40 explicit research tests ignored; library execution 183.24s, CLI
  0.57s, compilation 1m44s. Only the prediction probe was added to Rust in this
  resumed work; numerical solver updates are unchanged. After formatting,
  executable SHA is `b901e8abc0167381cd3fc54d49f8a5c536d3e01d8c7ec4ab50d5a5c33760f26a`.
  Normal non-test compilation reports the existing research-only
  `predict_native_turn` method as unused; it is exercised by the research tests.
- Confirmed the broader roots use identical game/action-abstraction parameters;
  source B differs only in source and evaluation seeds. Preserve those seed
  differences rather than editing the archived inputs. Started fixed-weight
  128-update transfer on the cheap facing-bet context first:
  `local-native-value-20260907-transfer-facing-a`. The second flop-start context
  follows sequentially. Both reuse complete native32 controls and packet sets.
- Facing-bet transfer completed in **36.141s**. A/B gains **0.186343/0.179282bb**
  versus native32 **0.171905/0.089273bb**: the original-root gain does not yet
  generalize. Manifest SHA
  `056178ed27e0b475a6d98aa2ca044dde7e0dfdece591e6308ce25baa9912b1f6`.
  Both native audits pass; no candidate promotion. The second reused flop-start
  transfer is running at `local-native-value-20260907-transfer-start-b`.
  Its first invocation rejected a mistyped input SHA before creating output;
  retried with the original on-disk/archive identity (no input was changed).
- The cheap facing-root value probe also completed. Native losses from predicted
  best actions are **0.389191/0.239247bb**, with best-action agreement
  **41.04%/58.86%**. Call is the only learned continuation there; exact fold/shove
  values expose optimistic calls for weak hands. Native frozen-policy local
  losses are **0.369697/0.353805bb**, versus predicted **0.123485/0.100078bb**.
  These are conditional diagnosis values, not additive full-game exploitability.
- Next bounded correction targets stale search beliefs: refresh **271 training
  targets across the same 12 training flop families**, using the current fixed
  wide proposer pair at 128 updates. Preserve the other **69 tuning + 72 holdout
  targets exactly**, total 412 labels, 64-update native labels, fitting budget,
  architecture, and seeds. This changes the training-belief distribution, not
  data volume or action abstraction. Re-read the cited ReBeL primary paper;
  it motivates search-generated beliefs, not any guarantee for this pilot.
- Implemented explicit training-refresh split handling (strict prefix remains
  the default), full-128-update stratified sampling, and the guarded refresh
  runner with a 16-label preflight and immutable provenance. Tests first failed
  on unsupported refresh, then passed; the real training CLI records the opt-in
  and preserves held-out indices. Full Python subset **86 passed**, compile and
  diff checks pass. Rust runtime checks await completion of the currently pinned
  binary audit; no rebuilding beneath that live job. Disk remains **125GiB free**.
- Second-board transfer finished in **2,007.609s**. A/B gains
  **0.306303/0.259277bb** versus native32 **0.218398/0.212251bb**. Both native
  audits passed. Manifest SHA
  `d6f058e5666a9fee12a5ab4fbfbc25654133308e5e10ae4b4d3f31b16cb1c927`.
  Fixed-belief probe SHA
  `df0bf49813495a13963ae65af6876715c3a164059e0acd08d4888df11f2e1d1c`.
  At the root, native best-action agreement is **96.89%/97.35%** and predicted
  local policy loss **0.00706/0.00827bb**. After a check, predicted local loss is
  **0.02007/0.02033bb**, versus native **0.45256/0.32926bb**; native loss from
  predicted best actions is **0.47668/0.45012bb**. This further supports targeting
  value rankings, not blind search scaling. Both seats' saved corpus errors are
  comparable overall; the after-check weakness is not evidence of a universally
  broken single output head. No generalization fix is claimed before the pilot.
- Confirmed all three reused development test flops are outside the 16 corpus
  families, including suit equivalences. They remain development evaluations,
  not untouched final validation. The refresh does not add those boards.
- Added an explicit updated-value comparison arm: new weights can be compared
  against a pinned old learned128 pair at the **same 128/64/64 budget**, without
  a redundant new learned32 run. Same-weight compute-allocation and transfer
  checks remain strict. Reports identify native32 as unequal-budget and record
  the learned-reference delta separately. Regression first failed, then passed.
  Full Python subset **87 passed**. Targeted release Rust tests **3 passed**,
  including full-128 sampling, deterministic quotas and observation parity;
  compilation 1m31s, tests 0.16s. Full release suite is running before capture.
- Full Rust release suite passed again: **312 library + 9 CLI tests**, 40
  explicit research tests ignored; 178.48s library and 0.58s CLI execution.
  New sampler-capable binary SHA
  `0d5ef1b982d838cc876ebb18ff25f574bdb358a531b94ba6335ad5007706ad04`.
  Started `local-native-value-20260907-current-search-refresh-a`; its 16-label
  preflight on training root 13 (11 turn leaves) passed in **51.099s**, peak
  **643.28MiB**, including unchanged-policy observation verification.
  Training-family collection is continuing on two native label workers.
- Current-search refresh completed in **810.556s**, maximum native worker
  footprint **1,558,366,032 bytes (1.451GiB)**. All 12 training families were
  refreshed; all **69 tuning + 72 holdout targets match exactly**. Total remains
  **412 states**, 130,536,903 decoded / 54,799,653 compressed bytes.
  Corpus SHA `f8bfde192fc0274764ddc94b6c2b6cf74cb7a01b0bac8d205706ca585602f894`;
  manifest SHA `5836e20ada0f52d93b27fa2fd8a8d05ff2b06cbd7e478f85aaa9bb42a9794ec3`.
  Started `local-native-value-20260907-student-current-search-refresh-a`: unchanged
  wide architecture, exact-runout v3 features, seeds 10601/10602, 600-step limit,
  learning-rate schedule and early-stopping rule. Explicit training-refresh
  split option enabled; independent native policy evaluation follows fitting
  and full-corpus Rust/Python inference parity, not RMSE-based promotion.
- Refreshed pair completed in **777.844s**, actual fitting **267.032s**, peak
  training footprint **2,917,976,392 bytes**. Both fits completed 600 steps;
  tuning selected steps **500/550**. Actual Rust/Python maximum discrepancies
  **5.80215e-6 / 5.46676e-6bb** across the complete corpus. Held-out authentic
  value RMSE **1.558650/1.546869bb**, versus prior **1.782792/1.560123bb**.
  Student manifest SHA
  `eaa433e23b5f46fe4a3813d887902e66a84f78739cd9cfa911b185e35fd95780`.
  Started `local-native-value-20260907-response-refresh-facing-a`, comparing new
  and old weights at the same **128/64/64** budget, with the same native32
  reference retained separately. This cheap screening pair comes before another
  costly flop-start audit. No policy-quality improvement is inferred from RMSE.
- The refreshed facing-bet pair completed: **0.150349/0.146662bb** versus the
  old same-budget learned policies **0.186343/0.179282bb**, improvements of
  **0.035995/0.032620bb (19.3%/18.2%)**. Both native audits passed.
  Relative to native32 **0.171905/0.089273bb**, A now improves and B still
  regresses; no blanket native-retention claim. Manifest SHA
  `dc399b255c54fe9dd52df5c90948fd960994a793e3851f61ff7d926bf0899df8`.
  Started the second-board same-budget test at
  `local-native-value-20260907-response-refresh-start-b`, with the completed
  original 128-update transfer as its learned reference. All weights remain
  frozen, with unchanged native played-turn budgets and no new training.
- Refreshed facing-root diagnosis agrees with the measured improvement: native
  loss from predicted best actions **0.279597/0.185790bb**, down from
  **0.389191/0.239247bb**; native action agreement **51.20%/64.60%**. Remaining
  value-ranking and finite-search errors are not claimed resolved.
- Read-only preparation for step 3 found reusable strict preflop replay and
  immutable native postflop playback in `response/native_policy.rs` and
  `flop_pilot/frozen_response/playback.rs`; learned leaves are not yet wired to
  that full-hand entry point. The older dedicated preflop solver uses a fixed
  continuation cache. The main PCS trainer's `ranges` are **estimator reaches**:
  the traverser's own action reach is not propagated there, opponent reaches
  include proposal corrections, and initial support excludes the entire sampled
  five-card board. These are valid for its existing estimator but **not public
  beliefs for a new learned continuation**. Step 3 must separate true policy
  beliefs from estimator weights and avoid leaking unrevealed cards; do not
  naively replace the terminal callback or redo validated exact averaging.
  No preflop training/website changes were made during this read-only review.
- Refreshed harder-board seed A regressed to **0.327539bb** from same-budget
  **0.306303bb**; seed B continues. Replayed the actual independent audit and
  same-budget assertion in **0.805s**, observing the expected failure. Do not
  infer cross-board action quality from the improved held-out value RMSE.
- Next hypotheses were ranked before further changes: (1) handcrafted range
  compression misses useful composition, test learned range pooling; (2) narrow
  board/belief coverage, test targeted coverage if needed; (3) action-critical
  errors underweighted by fitting loss, inspect before changing that loss.
  Exact-checkdown residual prediction already exists, so it is not a new remedy.
- Prepared **wide-pooled** using the existing v5 joint-reach-weighted own/opponent
  embedding pooling, unchanged wide context/query towers and head hidden width.
  This adds pooled context inputs, not the old much larger xwide towers or pot
  experts. Native contract validation now permits research v5 as well as v4;
  all other contract checks remain. Research motivation:
  [Deep Sets](https://papers.nips.cc/paper/2017/file/f22e4747da1aa27e363d86d40ff442fe-Paper.pdf)
  supports learned permutation-invariant aggregation; this finite weighted
  representation is not claimed lossless or an equilibrium guarantee.
  The intended pilot keeps the refreshed corpus, split, 600-step fitting budget
  and 128/64/64 policy evaluation fixed. No fit starts before this pair finishes.
- Pooling preparation exposed a real numerical mismatch: Python used a 1e-8
  pooling denominator floor versus Rust's 1e-9. An analytic forward regression
  produced **0.088795 versus expected 0.887954** at reach 2^-30, then passed after
  matching Python training/parity pooling to Rust. Existing training output
  projection is unchanged. This is a pooled-path fix, **not** the cause of the
  current non-pooled policy regression. Native future-payoff/zero-own tests now
  exercise both v4 and v5; Rust type-checking passes, runtime checks await the
  live pinned audit. The real native training CLI also exercises wide-pooled.
- The refreshed harder-board pair completed in **2012.349s**: **0.327539/0.377668bb**,
  versus same-budget **0.306303/0.259277bb**. Manifest SHA
  `f9490d5143bb7e5d0a6208227349cfcaff893d9e807400a69d75ec0fa8a9e451`.
  Both native audits passed; policy quality regressed. Completed the saved-packet
  action-value diagnosis without another native solve. Python subset **89 passed**
  in 9.448s, including the tiny-reach pooled regression. Started targeted Rust
  release validation after all pinned native workers finished.
- Native v4/v5 contract runtime regression passed (release compilation 94s,
  test 0.02s). Pooled-capable binary SHA
  `aa89c8607d92f1825ef0dfebfaed97b78f2b389eec1c1edcf30fac3c1d75bb4e`.
  Started `local-native-value-20260907-student-wide-pooled-a` on the same refreshed
  412-state corpus, frozen split, v3 features and 600-step/paired-seed budget.
  No new labels, search-budget increase, pot experts or larger encoder towers.
  Refreshed harder-board diagnostics after a check retain large native versus
  predicted local loss gaps: **0.40340/0.50992bb** versus **0.01624/0.01962bb**.
  Disk remains **125GiB free**; no resource blocker.
- Full compiled Rust regression suite passed again: **312 library tests** in
  117.73s with two test threads, **9 CLI tests** in 0.60s, 40 explicit research
  tests ignored. Independent JavaScript audit/diagnostic tests **3 passed**.
  Both pooled fits completed 600 steps; tuning chose steps 500/550. Actual
  native parity and the paired policy screening remain separate from fitting.
- Pooled pair completed in **788.969s**, actual Rust/Python maximum differences
  **5.15603e-6/5.37535e-6bb**, held-out authentic RMSE
  **1.531222/1.504287bb** versus refreshed non-pooled **1.558650/1.546869bb**.
  Manifest SHA `ccf1f5fbf7357da0297ffa0e952e4b51c7765d3bf1a1304201cd16538427ca36`.
  Started `local-native-value-20260907-response-pooled-facing-a`, same-budget
  comparison against the completed refreshed non-pooled facing pair. No policy
  improvement inferred from the small value-error reduction.
- Pooled facing pair completed with mixed results **0.141819/0.162784bb** versus
  refreshed non-pooled **0.150349/0.146662bb**. Manifest SHA
  `b8137b23463594fa98872b5fbe420033364ad2900cf9130d809c1e5c44b35e4c`.
  Before a costly second-board policy audit, added an explicit **alternative-value
  diagnostic on unchanged frozen source-policy beliefs**. Source and prediction
  model hashes are distinct; opt-in required; original identity checks retained.
  An analytic regression first failed, then passed, proving alternative predictions
  cannot alter the source policy's native response. Seed pairing is fixed, not
  selected by outcome. This is not an evaluation of the alternative policy.
  Rust prediction-probe test passed; Python selection/budget tests **5 passed**;
  JavaScript audit tests **3 passed**. Diagnostic-only binary SHA
  `a6314e3013b0a6c5b1e4a5e32397513908d2b8494a6738233031ee2c4d156eec`.
  Started `local-native-value-20260907-pooled-fixed-beliefs-start-b` with all
  native packets reused. Read-only coverage check found only one ace-high
  training flop family among 12; this is a coverage hypothesis, not proof of cause.
- Fixed-belief pooled probe completed (manifest SHA
  `bb812ad809550cab0ed645785459869416e28b79215db38a2d1f1cb6030692d1`).
  After-check native loss from predicted best actions barely changed:
  **0.409836/0.541420bb** versus **0.413618/0.557063bb**. At the root it
  worsened to **0.471307/0.275668bb** from **0.108887/0.059894bb**.
  Source policies/native responses remained identical. Did not spend another
  full paired native audit on this unpromising model; retain non-pooled control.
- Prepared a bounded targeted-coverage pilot: select the first six unused ace-high
  families from 128 authentic frozen-preflop roots; exclude all old corpus
  families and all three reused development flops including suit equivalents.
  Append **96 labels** (16/family), leaving the complete 412-target prefix exact,
  for **508 total / 367 train / 69 tuning / 72 holdout**. Same non-pooled proposer,
  128-update search, 64-update native labels, two workers. Most complex selected
  root runs first, with policy-observation parity and projected storage/time;
  never silently omit a root that exceeds the preflight quota. This is targeted
  training coverage, not an authentic evaluation distribution. Tests for prefix,
  capture provenance, selection/suit exclusions and response pairing **25 passed**.
  No labels or quality results for this new pilot yet.
- Started `local-native-value-20260907-ace-coverage-a`, binary SHA
  `dfee843815afb90cc530f200dd490e4a93111334c093e10acf589b29287ca77b`.
  Generated 128 authentic roots with exact old-prefix parity; selected source
  indices **19,38,48,18,25,55** (largest trees first). The first 16-state capture
  passed in **73.656s**, including unchanged-policy observation verification.
  Projected merged sizes **67,577,001 compressed / 163,705,203 decoded bytes**;
  conservative additional disk allocation **135,154,002 bytes**. All remain
  below existing bounds. Remaining five captures continue on two native workers.
- Targeted coverage completed in **311.808s**, maximum worker footprint
  **1,193,215,224 bytes (1.111GiB)**. All **412 old targets match exactly**;
  resulting split **367/69/72**, **508 states**. Corpus SHA
  `77f469394624ba93220d8c884d70563dc56a23fc0afb9915df96b95d07484fdd`,
  **67,552,067 compressed / 161,412,435 decoded bytes**. Manifest SHA
  `51360e803fc328228ae6aa7735f3fcaa23435abbff544e5731d3b7b43a2b887d`.
  Started `local-native-value-20260907-student-ace-coverage-a`, unchanged
  **wide non-pooled** architecture, 600 steps, seeds 10601/10602 and learning
  schedule. Coverage is the only experiment change; no quality claim yet.
  Expanded Python regression subset **91 passed** in 11.224s.
- Coverage pair completed in **991.244s**; both 600-step fits selected step 550.
  Actual Rust/Python maximum differences **5.93669e-6/5.60226e-6bb**;
  held-out authentic RMSE **1.472388/1.460843bb**, versus non-pooled reference
  **1.558650/1.546869bb**. Student manifest SHA
  `0313d5e15c5d3e14920a7182222bfbac4ae7bf58f25454ffd77df40a024d8e02`.
- Targeted coverage **does improve the diagnosed fixed-belief rankings**:
  after-check native loss from predicted best actions **0.062698/0.022894bb**
  versus **0.413618/0.557063bb**; root ranking losses **0.086123/0.035968bb**
  versus **0.108887/0.059894bb**. Diagnostic manifest SHA
  `f615d37b154f0ca2c57a5c814d9f65b40bed7008204b5d7a7b15b9fe4d27b7cf`.
  These are alternative predictions on unchanged old policies, **not new
  response gains**. Started the independent same-budget facing policy pair at
  `local-native-value-20260907-response-ace-facing-a`; the stronger diagnostic
  now justifies evaluating a newly solved harder-board policy as well.
- Coverage improvements survive the independent facing-policy comparison:
  **0.125979/0.120764bb**, down from same-budget **0.150349/0.146662bb**
  (reductions **0.024370/0.025899bb**, 16.2%/17.7%). Native32 comparisons remain
  separately **0.171905/0.089273bb**; no blanket superiority claim. Manifest SHA
  `81295322977cf14fa14e59cb43f8527de893dd1d722066cfc6c2072997ac7436`.
  Started `local-native-value-20260907-response-ace-start-b`, the complete
  same-budget harder-board native-response pair. No future-step training or
  website change starts while this current experiment is unresolved.
- Additional cheap fixed-belief screening on the original low-card development
  root improved after-check ranking loss **0.039942/0.071678bb**, versus the
  original diverse-wide128 model's **0.144619/0.224584bb**; root ranking loss
  **0.023724/0.023051bb** versus **0.058179/0.023611bb**. This comparison to
  the original model includes both belief refresh and ace coverage; it does
  not isolate coverage alone. Native policies/responses remain unchanged.
  Manifest SHA `f8039a3f27dffafffc5e22a4aa3165316bcfbe46e4ec3bb96ae1ea5b6112a9fd`.
- The coverage pair's harder-board independent audit completed in **1974.522s**:
  **0.236544/0.252445bb** versus same-budget refreshed non-pooled
  **0.327539/0.377668bb**, reductions **0.090995/0.125223bb (27.8%/33.2%)**.
  Both independent audits passed. Historical native32 remains slightly better
  at **0.218398/0.212251bb**; do not hide this cost/quality tradeoff or infer a
  full-game bound. Manifest SHA
  `0e331e0c7b61b3bf188e94c742762477d63e6a39d68f1d4b58a1e8b330fcdb53`.
  Started `local-native-value-20260907-response-ace-original-a`, same frozen
  weights and 128/64/64 budgets, against the original low-card 128-update
  learned reference and its historical native32 control. This finishes the
  current three-root development comparison; it is not untouched validation.
- Original-root coverage comparison completed in **2026.050s**: gains
  **0.188976/0.193007bb**, versus original learned128 **0.206310/0.203169bb**
  and native32 **0.227128/0.229962bb**. Both independent native audits passed.
  Manifest SHA `65e9eb15b92a00acdd38cffeac2da63cacad6548fbef2587d7494cd17dd9ba34`.
  Retain the ace-coverage pair for subsequent plan steps. All three learned
  comparisons improved; historical native32 wins some other contexts. This is
  neither full-game exploitability nor release acceptance.
- Explicit forced-root action diagnostic completed in **157.305s**, four frozen
  proposers (two contexts x two seeds), **16 native labels**. Forced check and
  least-used non-all-in bet operate on copies only; original policy hashes and
  all training/held-out corpora remain unchanged. Intervened-joint-reach RMSEs
  **0.941166/0.882677/0.780429/0.955441bb**. These are value diagnostics under
  newly solved intervened ranges, not attacks on the source policy. Manifest
  SHA `5051422e15061e5a4aad6a1137669738ed04bfc9512db3fd61e3e9e9fd42ea79`.
  Python provenance/shape regression and Rust copied-policy/reach regression
  passed. Steps 1-2's bounded experiment is complete; continue Step 3 without
  promoting this candidate or launching more architecture pilots.
- Step 3 continuation pinning now reuses the existing full-hand adapter with
  learned flop leaves and unchanged native played turn/river policies. Identity
  includes preflop/model hashes, budgets and the public-root seed rule; execution
  worker count is excluded. Pilot-versus-playback policy/native continuation
  parity and route-identity tests passed. No serving activation.
- One complete-hand integration/cost probe passed in **31.945s**, peak worker
  **779,600,832 bytes**. Preflop exhaustive missing rows **0**, maximum sum
  error **4.44e-16**; cold flop **12.674s**, cold native turn **18.858s**, remaining
  queries cached. This is one forced check/call trajectory, not a p95 latency,
  full-hand coverage, or strength result. Manifest SHA
  `430cb3405f189774c96d5baf4078768116182f6f94ffe2104d60b1f6e5af39e6`;
  pinned research route SHA
  `5a8e4992743d277ee013cdc4100d0df9f38eee1b46cd1ec6bf989c6984d3bb03`.
- Necessary preflop boundary correction: the old flop root legal mask excluded
  zero-prior holdings; that is insufficient for evolving counterfactual ranges.
  Added an opt-in complete-root-support path, preserving raw zeros and averaging
  actual trained own-realization strategies separately from the root prior.
  Old artifact support/serialization remains the default. The zero-own regression
  passes; this new path still requires policy-quality validation.
- Implemented a compact **16,900-node** preflop pilot using existing DCFR nodes,
  exact averaging, private-hand aggregation and exact all-in kernels. True public
  ranges are snapshotted without changing training/RNG or observing future cards.
  It samples one live flop endpoint per update, using current-range exact-checkdown
  baselines plus explicit endpoint/public-chance corrections; actual turn values
  come from the native continuation, not model self-scoring. Zero-own completions
  and finite-budget oracle error remain explicit. Snapshot and arbitrary-baseline
  unbiasedness tests passed; additional backup/chance checks and the two-round
  cost pilot are pending. This is not a claim that sampled dynamic continuations
  inherit exact CFR convergence guarantees.
- Compact preflop analytic tests **4 passed**: immutable true-range snapshot,
  arbitrary-baseline correction, exact 22,100/17,296 public-flop proposal factor,
  and existing exact-private-hand regret aggregation. Python comparison test
  passes and retains established thresholds; combined Python subset **96 passed**
  before that added test. Independent JavaScript audit tests **3 passed**.
- Two-round compact paired cost/integrity pilot completed in **36.629s**;
  seed processes **15.475/20.729s**, peaks **326,091,352/404,931,256 bytes**.
  All **16,900** information sets have trained averages, no postflop table growth,
  and zero-sum residuals are around 1e-14. Same value model and continuation chance
  seed for both independent preflop solver seeds. Root MAE **12.6842%**, aggregate
  delta **12.2952 points** fail; primary agreement **99.4083%** passes. Two rounds
  are a cost preflight, not a strength result. Manifest SHA
  `9abed75b00e0da16b7c0f85e4221c3b0479194c3e852fd558f2da7daba8953b0`.
  Started the bounded eight-round same-continuation pair; independent continuation
  chance comparison follows sequentially. No candidate activation.
- Eight-round compact pair with continuation seed 28001 completed in **265.472s**;
  root MAE **10.0136%**, aggregate delta **6.0929 points**, primary agreement
  **63.9053%**. All three stability gates fail. Manifest SHA
  `fb2a4efc05fa54569a90fae5922625a5c00c38c0bfd2dc83044351e9c2cfede9`.
  Independent continuation seed 28002 pair completed in **267.549s**; root MAE
  **12.7698%**, aggregate delta **12.3829 points**, primary agreement **57.3964%**.
  Manifest SHA `5464c62458e92cf1002bb9ad31b0a5b868bf429b91ce80f463efd23d06f89b13`.
- The 2x2 noise-source comparison finds **21.7100/25.5879%** root MAE when
  varying continuation chance with each solver seed fixed, versus **10.0136/
  12.7698%** with solver seeds varied and chance fixed. Two replicates do not
  identify a statistical variance component, but support targeting public-card
  payoff noise before lengthening training. Diagnostic SHA
  `cfb01897dd9e5e6b9eb9c5c427e1aa52b45060764401666f9cf54aaa705d6114`.
  Full release regression suite **320 library + 9 CLI passed**, 43 explicit
  research tests ignored (before the subsequently added exact-kernel tests).
- Implemented exact preflop class-payoff aggregation over all **1,755 suit flop
  orbits / 22,100 raw flops**, recovering integer showdown units from the existing
  exact all-in kernel. Class-constant preflop ranges allow exact aggregation;
  suit-dependent inputs are rejected. Complete counts must equal compatible
  private-pair counts x **17,296** flops, and opposite win units sum exactly to
  **1,980** per compatible pair/flop. Partial chunks are not usable kernels.
  This supplies an exact checkdown expectation for a control variate, not a new
  strategy or policy label. The accounting/class-range regression passed.
  Research connection: [Davis et al. (2020)](https://proceedings.mlr.press/v119/davis20a.html)
  motivates baseline-corrected sampling; reduction of this pilot's actual variance
  must still be measured, not inferred from their experimental results.
- The 16-orbit exact-kernel preflight passed in **18.188s**, peak **116,326,856
  bytes**, compressed output **157,391 bytes**. Projected two-worker total with
  50% headroom **1,496.3s** (~25 minutes), below a one-hour stage limit; generated
  aggregates fit comfortably above the unchanged 20GiB disk reserve. Preflight
  SHA `75b7e4e7c4b17c7f07b7145bf7ec50b82148b89cd621e35763598540b0aadada`.
  Started `local-exact-checkdown-20260907-complete`, two guarded workers and
  resumable immutable chunks. Binary SHA
  `5abe20a4f4659b562d6f64a387e8196e447f7e0885cc02374733f9f417a502ad`.
  Do not rebuild that pinned binary during the live exact-orbit pass.
- Exact kernel generation completed: **1,755/1,755 orbits, 22,100 raw flops**,
  **1,141.917s**, final compressed kernel **134,359 bytes**. All integer
  card-removal and zero-sum identities pass. Reversing the completed chunk order
  reconstructs byte-identical canonical JSON/gzip, without rerunning equity.
  Kernel SHA `6ac74c0183061b43a94c798570a8778b419993c45d6970be24ae9fe298a99275`;
  manifest SHA `ca1e644a685814bba6e582261ae4765b00c7a2066758ff7d087fcff6ff803a68`.
  This 134KB payoff baseline replaces sampled preflop checkdown expectations,
  not policies or action-EV validation. No training artifact was deleted.
- The opt-in update now uses exact checkdown means at every non-fold preflop
  endpoint, plus `(sampled native value - same-flop checkdown) / endpoint q`
  at the sampled live endpoint. The mean must not replace the sampled baseline
  inside that residual. Python kernel/paired-contract tests **3 passed**;
  Rust type-check passed. Runtime analytic checks and matched pilots follow.
- Found and corrected an experimental training/playback boundary mismatch:
  replay previously normalized after each action and rejected zero-reach lines,
  unlike compact training. The compact route now shares the exact same revealed-
  flop constructor, accumulates true reaches from ones, normalizes once, preserves
  zero reach, and pins this replay rule in its identity. Historical/website route
  defaults remain unchanged. Regression and runtime verification are pending.
- Runtime checks now pass: **six** compact update/kernel analytic tests, raw-
  reach replay parity, and route-identity pinning. Disabling the correction under
  the new binary reproduces all 16,900 rows and both frozen two-round artifacts
  **byte-for-byte**, isolating the exact-mean experiment from unrelated changes.
  The real complete kernel passed the independent Rust consumer checks when the
  corrected eight-round pair started. New binary SHA
  `ce1cde9fa96e02a9e12dd1ba2ff7483f925545c4ad517c3960b9145781dd7ee2`.
- Corrected eight-round 2x2 completed. Solver-seed root MAEs **9.84584/10.14601%**;
  primary agreement **94.67456% both** (passes), aggregate differences
  **9.84584/9.90041 points** (fails). Chance-seed sensitivity dropped from
  **21.70998/25.58785%** to **6.07232/10.53524%** with each solver fixed.
  The baseline reduces the diagnosed noise substantially but does not pass all
  stability targets or establish strength. Pair times **268.871/268.232s**.
  Pair manifest SHAs `b94bf3a9b5f1aa63e2da5c0f5f120521469615e0c61f2725473ed56345b3dc84`
  and `fff580dc443d36b8b4839170da54595430382b47c587235c51cb9a642bbbff08`;
  noise diagnostic SHA `e50cde50004add2b65644fce33e265ab640688c2c866696b95ca39b7b7658684`.
  This justifies a bounded 32-round extension, not another architecture change.
  Started both independent-chance comparisons with at most two concurrent
  2GiB-guarded workers (4GiB combined worker cap), unchanged model/search/seeds,
  and no paid compute or serving activation.
- Prepared, but not yet executed, the next complete-hand screening probe by
  reusing the existing exact-card Bayesian LBR (seed 90001, 16 early runouts per
  opponent combo). Both seats use one complete deal cluster; their self-play
  baselines cancel in the total, avoiding an extra expensive baseline rollout.
  Per-seat utilities are explicitly not called gains. Disjoint screening,
  calibration and holdout chance domains are pinned, with one-hand cost preflight
  required before a larger batch and a 300s/2GiB worker guard. The aggregation
  regression passes; Rust type-check passes. Positive-control runtime and all
  new evaluation execution wait until the pinned compact training jobs finish.
  No reported full-game improvement or gate pass exists yet.
- The 32-round exact-mean extension **regressed**. Solver-seed root MAEs
  **28.91548/31.94001%**, primary agreements **51.47929/40.82840%**, aggregate
  deltas **15.09754/15.65308 points**: all fail. Same-solver chance MAEs
  **18.14149/24.70978%**. All 16,900 rows remain trained and normalized.
  Pair wall times **855.628/865.935s** with two concurrent guarded workers.
  Manifest SHAs `d569262d68f5c7016b4c319646bebfe5e0204a3cdf279885dbd40c600d1f4168`
  and `207713d51ab37359f585d489bd49007bf0d69c3597c8905e318c913b52cc35fa`;
  diagnostic SHA `31152d80ecf0a78e0e1b184e2d147ce751a2103f6210c070c8ea8ac476ccc081`.
  **Do not scale iterations further.** Eight-round baseline-noise improvement
  did not translate into stable longer updates. Complete the required independent
  response screen to distinguish costly actions from harmless mix differences,
  then target update variance/continuation behavior with a matched bounded pilot.
  Neither the eight- nor 32-round policy is an accepted website candidate.
- Shared LBR positive control passes (fold-everywhere loses **1.5bb** paired
  total); all three existing Bayesian/card-isolation/action-value tests pass.
  Real candidate screens complete through postflop, **14.4–27.9s** per nontrivial
  paired deal; some deals terminate preflop in milliseconds. Five matched
  development deal clusters give LBR total means **-5.99980bb** (eight rounds),
  **-3.55536bb** (32), with SE **5.71325/3.52031bb**. Paired difference
  **2.44444 +/- 2.68153bb SE**. These are inconclusive, not passing negative
  exploitability estimates or justification to choose a release. Matched report
  SHA `116d7544fd6ecd59133824f3f7451303ea5e86168936d9b239dece84608badf5`.
  Prepared the existing full-continuation information-set learner's two-deal
  cost preflight; its native candidate rollout cost is not measured yet.
- The learner's new 64-train/64-test fold-everywhere positive control exposed
  **4.6875% lookup coverage** (3/64 test hands), despite all **36** learned rows
  being confident. Point gain **0.0703125bb**, SE **0.03994538bb**, one-sided
  99.5% lower bound **-0.03257998bb**. This is sparse class coverage, not a
  profitable-action tie rejection. All preflop backoff layers retain the 169
  private hand classes. The regression preserves this low-coverage failure to
  qualify, then verifies that **512 cheap training/test deals** detect the same
  weak defender with >50% coverage and positive gain lower bound. It passes.
  This fixes the positive-control sample-size assumption, not the poker policy.
  A two-deal native learner probe must never be interpreted as a strength test.
- Research cross-check: [Gibson et al., AAAI 2012](https://ojs.aaai.org/index.php/AAAI/article/view/8241/8100)
  relates bounded unbiased estimator variance to regret convergence;
  [Davis et al., ICML 2020, Sections 6–7](https://proceedings.mlr.press/v119/davis20a/davis20a.pdf)
  distinguishes private-history integration from public sampling variance and
  explicitly excludes inexact continuation evaluators from its zero-variance
  theorem. Our exact-private integration and checkdown mean do not eliminate
  sampled strategic residuals or approximate-oracle error. Do not assert such
  guarantees, or change both sampling and value targets in the next comparison.
- Full Rust release library regression after the new integration/evaluator work:
  **325 passed, 46 explicitly ignored research probes, zero failures**, 180.46s.
  New binary SHA `bf1a48c50504a28ba75609ecece209896b011b48eccab8b8d62bb332199a1f9e`.
  Started `local-compact-response-20260907-exact32-cost2`, two training deals for
  seat 0 against the pinned 32-round candidate, 300s/2GiB guarded. This is only
  the rollout-cost preflight; tiny critic support cannot qualify the defender.
- Native learned-response cost preflight completed in **177.875s for two deals**,
  retaining **zero preflop and zero postflop critic rows**. Manifest SHA
  `d8e772fded6dc154f2994013379370f912b85a720ea53f0d75a1a3e42684c38b`.
  Do not scale this sparse per-deal learner blindly or interpret no learned
  attacks as a passed gate. More efficient response coverage remains unresolved.
- Next bounded update experiment is implemented opt-in: **uniform batch versus
  root-stratified sampling**, both **six endpoints per update**, same exact mean,
  model, 128/64/64 searches and shared public chance. The six groups are limp and
  2/2.5/3/4/5bb opens; folds/all-ins remain exact terminals. Uniform inclusion
  is 6/49; stratified inclusion is reciprocal group size. Draws reset each update.
  Complete-tree group counts, without-replacement batch size, old RNG parity,
  and exact enumeration of correction expectations pass. **Eight** compact Rust
  tests and **eight** Python tests pass in the fast test profile. Release tests
  and frozen-artifact control replay precede the matched four-round pilots
  (24 native continuations per seed per arm). No value-target or abstraction change.
- All **eight** compact release tests pass. After sampler integration, the
  two-round default path still reproduces every original row and both frozen
  artifacts byte-for-byte. Started `local-compact-preflop-20260907-batch4-a`
  and `local-compact-preflop-20260907-stratified4-a`, four rounds, same continuation
  seed 28001, paired solver seeds 27001/27002, **24 selected continuations each**.
  At most two concurrent workers, 2GiB each; one-hour maximum per worker.
  Binary SHA `5e3904c900b8988867a3494d777a53ae0984dc2320a28a54f09e034fd00d63e1`.
- Cost diagnosis of the native learner trace: **12 flop solves, 12 distinct root
  seeds, no repeated root solves**, peak **1,093,764,320 bytes**. A larger root
  cache would not remove the observed work. Vector-valued response supervision
  is a possible next efficiency improvement, not implemented or validated yet.
  The six-endpoint experiment matches continuation counts and per-search budgets;
  different selected pot/tree shapes can still change actual wall time and leaf
  work. Report those costs rather than equating query counts with identical work.
- The four-round sampling comparison completed and **does not retain root
  stratification**. Uniform batch: MAE **16.78175%**, primary **52.07101%**,
  aggregate delta **12.59778 points**, **462.765s** per paired run. Stratified:
  **17.68561%**, **40.23669%**, **14.45426 points**, **521.396s**. Both fail;
  stratification is slower and worse on these diagnostics. Manifest SHAs
  `23a700746a4e2beb0e14c76ee2a8c131df15bc06e4eee4c8457de969815f8792` and
  `1bfce26c0f59de9ca385443c6ee798bd8346d5399277eab3d259c3f2b006081b`.
  No larger batch/stratified solve is queued.
- Next isolation control: forced-checkdown-after-preflop with the exact class
  payoff kernel, reusing the same DCFR discounting, regret backups and exact
  averages. Evaluate its average policies with an information-set-consistent
  preflop best response. This removes sampled/native/neural continuation error
  to test whether the core update can converge in a fixed game. The output is
  explicitly **surrogate-game NashConv, not full-hand exploitability** and exports
  no policy. Eight-round cost preflight precedes 128/512 checkpoints, with a
  ten-minute/2GiB guard. Implementation is awaiting release runtime verification.
- Exact fixed-game control completed: total checkdown-game NashConv at
  2/8/32/128/512 updates is **3.868643 / 1.577274 / 0.238859 / 0.034910 /
  0.002467bb**. It took **116.190s**, with no neural/native continuation or
  sampled chance and no policy export. Manifest SHA
  `19ef052ddb64668dabd2cd84921a9d50331edf3988a6d740a719bc8f3f5817f5`.
  This isolates convergence of the shared exact-average/regret code in the
  fixed surrogate; it is **not full-hand Hold'em exploitability** and does not
  validate the range-dependent approximate continuation oracle.
- The failed exact32 trace includes expensive endpoint queries with zero
  opponent reach for the current traverser. Next opt-in pilot changes only
  endpoint allocation: **80% opponent-reach-weighted + 20% uniform**, one
  endpoint per iteration, selected from the frozen current preflop snapshot
  before chance sampling. Own reach is deliberately excluded. Every endpoint
  retains q >= 0.2/49 and its actual q enters the existing correction. The
  mixture fraction is a heuristic, not a proven optimum; worst-case correction
  is larger than uniform, so retain only on paired evidence. All continuation
  budgets, exact kernel, value model, averaging and public chance stay fixed.
  Exact-enumeration expectation and zero-own-reach invariance tests precede an
  eight-round paired comparison. No policy promotion or longer run is implied.
- Opponent-reach eight-round pair completed and **is rejected for extension**:
  MAE **20.31285%**, primary agreement **14.79290%**, aggregate delta **20.31285
  points**, **377.770s**, versus uniform exact8-a **9.84584% / 94.67456% /
  9.84584 points / 268.871s**. Complete trained rows and numerical accounting
  still pass. Manifest SHA
  `02654f7b98bd7e9848f9f78f8dcf8e20dff47d0e393d4f1a985a4841afe9e09c`.
  Simple opponent mass is not a sufficient correction-variance proxy. Keep
  uniform selection as the control; next diagnosis targets the noisy strategic
  residual baseline, not another sampling allocation or longer failed pilot.
- Implemented an opt-in **learned history residual baseline** at the existing
  correction seam, motivated by Davis et al. Sections 4/7. Each live history
  caches two 169-class conditional residual vectors (EMA alpha .2); prediction
  uses current compatible opponent mass and a current-profile zero-sum
  projection. No own-reach factor scales individual CFVs. The update is
  `exactMean + cachedResidual + (observed - sampledCheckdown - cachedResidual)/q`.
  Cache observations are applied only after the complete regret backup; the
  cache is never exported or served. Arbitrarily stale-cache expectation,
  zero-own reach, immutable prediction and changed-range zero-sum tests pass.
  This is a variance heuristic for the same approximate oracle, not an exact
  continuation replacement or a transferred equilibrium theorem. Uniform
  sampling stays fixed. Fresh observed correction squared magnitudes are
  logged as diagnostics, explicitly not action-EV errors or variance bounds.
  **11 compact Rust debug-profile and eight Python tests pass**. Release
  runtime verification and the bounded paired pilot remain pending.
- History-baseline release verification passes **11 compact tests**, and the
  disabled two-round control reproduces both original frozen policy hashes
  exactly. Binary SHA
  `7244af4c36aafedca545d59773cf748ea8c2e2727489c0979c6dd1612d055522`.
  `history8-a` completed in **262.950s**: MAE **9.65566%**, primary **94.67456%**,
  aggregate **9.65566 points**; still unqualified. Manifest SHA
  `eb9737f9a4fce629e5b40dfcf3995dd5a84f956ce88345d87ab7905861d64943`.
  Each seed had only one nonzero cached prediction at its sampled endpoint.
  Sum of observed raw-CFV correction squared means changed **.467508 -> .514876**
  for 27001 and **.011206 -> .009748** for 27002: mixed, not a variance win.
  Started matched **32-update** `history32-a` to diagnose reuse once populated,
  against the already-completed exact32-a control. No change to EMA, model,
  oracle budgets, public chance, endpoint distribution, or acceptance gates.
  Each worker is capped at one hour/2GiB; no longer extension is queued.
- Full pinned Rust release library regression after the sampling/history-cache
  work: **330 passed, zero failures, 47 ignored research probes**, **192.664s**,
  sampled peak **1,970,194,448 bytes**, below its 2GiB guard. Manifest SHA
  `60e41790a38e3a12502cfa8b4df2389eb882c324cc1299246231034f1d58ce16`.
  This overlapped the first history32 worker, so their timings are not clean
  isolated performance comparisons. First seed completed in **478.432s**;
  second remains active. No release acceptance or website changes.
- The history32 pair completed in **807.531s** and **is rejected for extension**.
  MAE **29.02878%**, primary **40.82840%**, aggregate **15.53824 points**, versus
  exact32-a **28.91548% / 51.47929% / 15.09754 points**. Manifest SHA
  `a1508a12e125c261c5ee4d6787c9e46041c90adbe49c47928ebd3cdcb7862645`.
  With 9/11 nonzero reused predictions, correction squared-mean sums were
  **3.199078 -> 3.173517** and **6.120489 -> 6.673613**, respectively. No robust
  variance or stability benefit. Keep the cache off in the control and website.
  No further allocation, EMA, or iteration sweep is queued. Next diagnosis fixes
  a real frozen preflop policy and betting history while varying only public
  continuation chance, so value noise can be measured without policy drift.
- Fixed-input chance diagnostic implemented using the **existing strict frozen
  preflop reader and actual native continuation path**. It pins exact32-a seed
  27001, the retained value model, 2bb-open/call history and 128/64/64 budgets;
  freezes the flop policy once per board; samples public turns without replacement.
  Exported class CFVs and conditional strategic residuals retain exact card
  removal and current true-reach weights. No training, action maximization,
  or policy export. Analysis clusters by board, applies the turn finite-population
  correction and retains negative noise-component estimates when inconclusive.
  **11 compact release and nine Python tests pass**; binary SHA
  `91d827225b280ccc67fb25c4a103e73a6b6144184715bae39f71527d94fc43b7`.
- One-board/two-turn preflight completed in **113.016s**, with within-turn
  reach-weighted conditional-residual variance **0.980841bb²**. This is only two
  samples on one fixed line, not an action-EV gate or whole-game noise estimate.
  Manifest SHA `184baf83c769f00df1bdd14f54ff6598c7313949b82b885541ff3854f224ae5d`.
  Started **four boards × four turns**, two concurrent workers capped at 2GiB
  each, 600s per board and a preflighted 30-minute aggregate-work ceiling.
  No new training variant or larger training run is queued.
- Fixed policy/history **4×4 chance diagnostic completed in 472.080s**. Weighted
  within-turn variance is **1.715671bb²**; its contribution after averaging four
  turns is **0.393904bb²**. Observed board-mean variance is **2.538787bb²**, giving
  a noisy estimated flop component of **2.144883bb²**. These are conditional
  class-value diagnostic quantities at one frozen line, not full-game strength
  or action-EV precision gates. Both chance sources matter; adding turn samples
  alone leaves substantial between-flop variation.
- Prepared a paired, **prediction-only all-turn control-variate diagnostic**.
  Rebuild the identical frozen flop policy and verify its hash against the
  captured native source; predict all 49 public turns using the retained network.
  The candidate estimator is `mean(all predicted turns) + native(sampled turn)
  - predicted(same turn)`, with identical card-removal factors. Its expectation
  remains native even with a biased predictor, provided the complete prediction
  mean is used. Unit tests check this identity and reject incomplete turn sets.
  No new native labels or policy updates are needed. A one-board cost preflight
  precedes the four-board comparison; native serving/training remains unchanged.
- All-turn prediction preflight completed in **19.199s**, of which **4.510s**
  predicted all 49 turns. The four-board replay completed in **42.599s** using
  two workers; each reconstructed frozen policy hash exactly matched its native
  capture. Predictions cost **4.70–5.49s per board**, versus roughly 49–56s for
  one native sampled turn on these roots. The same saved native observations
  show within-turn variance **1.715671 -> 0.090240bb²**, a **94.7% reduction**;
  per-board reductions are **96.0%, 94.0%, 94.0%, 95.6%**. The remaining board
  component stays large (no claim that flop variance or exploitability fell).
  Native diagnostic manifest SHA
  `7f3bacfc5b3f5f1277fba9a9679987a37571e36045da7173acc663bb2b719066`;
  prediction replay SHA
  `a679fe5f2a1ad647525b6b7553786edef234590977d4005c65a10342f1bc595d`.
- Retain this **turn-chance estimator** for a bounded training comparison, not
  a release candidate. Implemented opt-in `--turn-baseline`: keep uniform
  endpoint sampling, exact checkdown mean and native played continuation fixed;
  disable the rejected history cache. Native plus exact mean prediction minus
  same-turn prediction preserves the expected native target. New Rust/Python
  complete-turn tests reject partial means and demonstrate cancellation of a
  biased predictor. Eight-update paired training follows release verification.
- Release verification passed **11 compact tests + the new complete-turn
  correction test** and ten Python tests. Started matched `turncv8-a/b` (eight
  updates, solver seeds 27001/27002, independent continuation seeds 28001/28002)
  against the existing exact8-a/b controls. Two concurrent guarded workers,
  2GiB each, 1200s each. Native target, model, preflop average, endpoint sampling
  and played continuation remain fixed; rejected history/reach variants are off.
  Binary SHA `8095f8d481646d34795b11e06e4e01386d56bdb376c698a616f4e13552dc20b0`.
- The turn-CV eight-update pairs completed in **324.740/328.257s** (concurrent).
  Endpoint-seed MAEs are **12.91128% / 9.36088%**, primary agreements
  **94.67456% / 99.40828%**, aggregate deltas **10.42356 / 8.74535 points**.
  Thus the policy-stability gates still fail and results are mixed versus
  exact8 controls. Same-solver/different-chance MAEs are **6.24040% / 4.45182%**,
  versus **6.07232% / 10.53524%** without turn correction: one is roughly
  unchanged and one improves substantially. No release claim.
  Pair SHAs `43d2bbb5d7534ae6d239d1b0a4d7e9b02dc49c6f7a4fffde72a495417c649a46`
  and `e9f3e56942f2bcf8d94a512289066aed461088a27533b5880ba8f858120e8b0d`;
  noise diagnostic SHA `a619bbb412f61452655acc9ac4f98cab547af2c5e0cfa42dc3e2d687f9e94b8b`.
  Started the **existing 32-update checkpoint comparison**, `turncv32-a/b`,
  unchanged estimator/model/budgets, two concurrent 2GiB workers, one-hour cap
  each. This tests the mechanism with more updates; no >32 extension queued.
- Read-only exploratory check while turncv32 runs: regress the fixed-policy
  turn-corrected values on the sampled checkdown values, centering by private
  class across boards and using the captured true-reach weights. Fit boards
  0/1 gives slope **1.998902**, reducing residual variance on boards 2/3 from
  **2.690769 to 1.459585bb²** (45.8%). Reverse split fits **3.012107** and makes
  its test variance slightly worse (**1.540964 -> 1.558598bb²**). The all-four
  in-sample optimum **2.436118** is NOT independent validation or an accepted
  coefficient. This suggests baseline scale as a possible next bounded test,
  not evidence to change the running experiment. A scaled baseline must scale
  BOTH its sampled values and exact expectation, and must never scale exact
  fold/all-in terminal payoffs that have no native sampling correction.
- Turncv32-a/b completed in **1017.128/1023.743s**. Endpoint-seed MAE remains
  **29.29641% / 28.32652%**, primary agreement **37.86982% / 43.19527%**,
  aggregate deltas **11.67951 / 15.26815 points**; all stability targets still
  fail. Same-solver/different-chance MAEs **13.50514% / 30.19156%**, mixed versus
  the original exact32 **18.14149% / 24.70978%**. No >32 extension is queued.
  Pair SHAs `29000181bec22a28ec9840576fa4935e72702f399a2880667d5221fa66fc5225`
  and `1e4896ec9960fc5b9a008ef4d19d9e6cd7d1d9b46deeb60428ce9251cda15a14`;
  diagnostic SHA `a21ea0887febbbcae3cef12431d89f168054c87a6e25a6632491b3a64a8b1b3a`.
  Reduced within-turn variance alone did not stabilize the evolving policy.
  Before any scale pilot, score the full endpoint estimator, including the
  1/49 sampling correction: a covariance-optimal flop-only scale can increase
  endpoint-sampling variance if it worsens the baseline mean.
- Full-estimator scale diagnostic at q=1/49 supports a bounded test: fitting
  boards 0/1 gives **2.302781** and lowers the opposite-split variance proxy
  **119.34275 -> 55.49481bb²**; reverse fit **2.547999** lowers its test proxy
  **124.40793 -> 66.03417bb²**. Fixed **2×** lowers the respective proxies to
  **62.05873 / 67.15893bb²** (roughly 46–48%). This is exploratory fixed-line
  evidence, not a whole-game variance measurement or untouched release test.
  Implemented conservative opt-in `--flop-baseline-scale 2` with the existing
  complete-turn correction, uniform endpoints, no stale cache. Both sampled
  and exact-mean live-flop checkdown values scale together; exact fold/all-in
  payoff endpoints stay untouched. Full-tree payoff-preservation and complete
  chance/endpoint expectation tests pass. A paired eight-update comparison
  follows release verification; no broader coefficient sweep is planned.
- **12 compact release and ten Python tests pass**. Started `scaledcv8-a/b`,
  same two solver/two continuation seeds and eight updates as turncv8-a/b,
  with only the live-flop checkdown control scaled 2×. Two concurrent 2GiB
  workers, 1200s each. Binary SHA
  `1d2f3923ad87a0a9f99687bb981b848ca7308a2921527e59c3a4e506e481dba4`.
- Scaledcv8-a/b completed in **327.782/329.410s** and are **not extended**:
  MAE **12.68861% / 19.34724%**, primary **60.94675% / 44.97041%**, aggregate
  **8.13534 / 13.99578 points**. The first matched continuation (identical policy
  hashes) has smaller correction magnitudes in both chance seeds, but that does
  not establish improved strategy. Pair SHAs
  `eea5d37214abebd0cd7bc94108716df5413fe5a9787a438d2360127c290f6f10` and
  `cd6e7e8caae71a2d1cdc4fe8b1304e4dfe11c4d6cc84655dab9b853229807d09`.
  No further scale/iteration sweep is queued.
- Next required Step 3 check is **independent frozen-policy preflop response
  gain**, not another stability-only pilot. Pin the canonical turncv32-a seed
  27001 (not asserted to be the strongest seed); enumerate all live preflop
  continuation endpoints on shared public boards, retaining exact folds/all-ins
  and the validated turn correction. Aggregate private holdings into their 169
  preflop information sets and average training chance BEFORE choosing response
  actions. Freeze those actions and evaluate on disjoint board draws. This is
  a restricted preflop attacker with unchanged postflop continuation, not a
  full-game upper-bound certificate. A three-endpoint resource preflight will
  precede complete four-board capture and held-out response evaluation.
- Frozen-preflop response capture and disjoint-chance response evaluator are
  implemented. Native targets explicitly use the **actual frozen profile**, not
  training's zero-own-reach BR completion; a sparse-range regression distinguishes
  those paths and matches frozen turn packets exactly. Private-class/chance-before-
  maximum and no-double-opponent-reach tests pass (13 compact release tests,
  three new Python tests; new profile-path test passed optimized debug).
  Three-endpoint cost preflight completed in **85.865s**: 2bb **67.661s**, 9bb
  **13.933s**, 18bb **2.729s**. Manifest SHA
  `79302e3636ffcd8ed655b45ad9324dbac8588175827e9e5b72456da792ad8c9c`;
  candidate preflop SHA
  `a9b8ba98e70c842d433b3190f0f998cd7b3d0ddbcbfe1cd8076ace06f934c94d`;
  capture binary SHA
  `46e94fed8a9257309f1e9680db72216c3da3429aa2f47e748c7bf83bc98f6587`.
  No strength result from the incomplete preflight. Cost projection uses the
  actual 49-root commitment distribution and smaller-pot representative costs,
  doubling the 2bb cost for the unmeasured 1bb root, plus 50% overall headroom.
  Full captures retain one-hour/2GiB per-worker guards and at most two workers.
- Started complete four-board capture, chance seed 32001, in
  `local-frozen-preflop-response-20260907-complete`. Predeclared fit indices 0/1
  and evaluation indices 2/3, all 49 live endpoints per board, no action selection
  inside chance draws. Guarded projected time **2031.949s per board** including
  headroom. Thirteen Python checks pass. Optimized-debug regressions additionally
  verify frozen reload preserves all 100 public rows, 16,900 information-set keys,
  average probabilities and true reaches, and profile evaluation matches frozen
  packets while differing from off-support training BR values. The pinned capture
  binary is not rebuilt while workers run; later test-only additions will be
  release-verified afterward. Full-game gates remain unqualified.
- Fitting boards 0/1 completed in **738.33/735.83s**. Response actions were saved
  before inspecting held-out captures, SHA
  `c8d7d37deec530c7059a09835ed3bca67077d26ac349dfbce91a633fcc587b5d`.
  Fitting-only gains are **0.718760684 / 0.592170140bb** by seat; not a validation
  result. An exact performance-difference decomposition uses responder own reach
  and reference continuation advantages, with opponent reach already in CFVs.
  Its contributions telescope to both fitted gains (regression test passes).
  Largest fitting contributions are SB opening **0.520878bb**, BB facing 4bb
  **0.344624bb**, facing 5bb **0.083260bb**, and facing 2bb **0.083110bb**.
  Fitting decomposition SHA
  `edf8475c41da6552b524cee7a2a143f425c22cb553565def4182fcfad0de3fcb`.
  Held-out boards 2/3 are running; do not tune or promote from fitting gains.
- All four response captures completed in **1489.264s** total (two workers).
  Manifest SHA `7e7312dc7e0bf007ebb91ec4bd0d087f7176d4ef826a1dcf2337c1bfbcae9b44`.
  Held-out summed gains are **0.828365084 / 0.610840839bb**, mean
  **0.719602961bb**, SE **0.108762122bb** across TWO independent board clusters.
  Seat means **0.538090064 / 0.181512897bb**. Saved fitting actions exactly match
  the final evaluator actions; no held-out refit. This is a restricted preflop
  attack using full-hand continuation values, NOT a full-game upper bound or a
  qualifying confidence result. Its point estimate exceeds the 0.50bb target.
  Held-out error attribution still puts SB opening first (**0.489144807bb**),
  then BB facing 4bb (**0.065393782bb**) and 2bb (**0.047366687bb**). Attribution
  SHA `a9404537aa675715973dca70b2ffcfd075c8f75e6e0f97ec0892511f9c4e27bd`.
- Targeted update-use audit: turncv32-a seeds 27001/27002 visited only **22/21**
  distinct native endpoints total. Alternation means **16** samples per seat,
  covering **15/11** and **12/15** endpoints respectively. In seed27001 the BB
  received **zero** native 4bb-open/call updates. Initialized/trained tabular
  coverage must not be confused with rich native continuation observation.
  The oracle already computes BOTH player CFV vectors. Added opt-in
  `--simultaneous-updates`: reuse the identical frozen snapshot and value arrays
  for the other seat, with no extra oracle, chance draw, averaging sweep, or
  clock advance. Default alternating behavior remains unchanged. This is a
  supported CFR update schedule ([OpenSpiel primary implementation](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/cfr.py)),
  but alternating updates can converge faster; improved cost-effectiveness is
  a hypothesis to test, not a claim. Compare the existing bounded 32-update
  matched controls after regression/default-byte verification; no larger run
  or website promotion is justified by this diagnostic alone.
- **336 Rust release tests pass, 49 opt-in tests ignored**, 184.491s guarded,
  peak **1,397,786,424 bytes**. Regression manifest SHA
  `4f4017cdd73747b0a1b8279bd7058b1118937cf2f49771e96030ef13a2f5c03c`.
  Fourteen Python tests pass. Default alternating two-round exports remain
  byte-identical to both original controls (`493c5258...` / `c68bb471...`),
  paired smoke manifest SHA
  `bf4325b21516916e3d46aa0e3a525950fd978d2cb6cb73063ca28b8b084d96cc`.
  Started `simultaneous32-a/b`, same 27001/27002 solver and 28001/28002 chance
  seeds, 32 updates, uniform endpoints, unscaled exact checkdown mean and retained
  complete-turn correction, native128/64/64/model10601. Only both-player update
  reuse differs from turncv32-a/b. Two concurrent 2GiB workers, one-hour cap each.
  Binary SHA `f6a1a42deb8123297245c473cb41e585f78e3e62306a6e6c5b90cb86f326785c`.
- Simultaneous32-a/b finished in **996.816/1010.736s**, no material extra native
  cost, but stability still fails: MAE **23.29778% / 29.96727%**, primary agreement
  **32.54438% / 17.75148%**, aggregate deltas **22.05674 / 25.31724 points**.
  Same-solver/different-chance MAEs **16.88549% / 21.44259%**. No adoption or
  longer simultaneous run. Pair SHAs
  `37db834ba5e916d00fcc815fe53672aca5eb412b43fe1041cb1e154192205182`,
  `41500239e4f40e3281860c390d1b01fef100fd06a1b69e1d1e91c60dd44ebc64`;
  diagnostic SHA `94a1777d9b364990de2c8d1cf2253f350a69ef9561b3a26aa37cdc480487de49`.
- Read-only uniform-endpoint RNG replay exactly reproduces all prior 32 draws.
  A 128-update alternating budget projects **3474/3587s per seed** including
  conservative headroom and visits **33/34** and **36/41** distinct endpoints by
  seat, but seed27001 BB still has **zero** direct 4bb-open/call observations.
  No 128 run is launched. Before choosing a larger budget, value-evaluate the
  completed canonical simultaneous32-a seed27001 with the existing frozen
  response pipeline: frequency disagreement alone is not a strength comparison.
  Reuse chance cohorts for a **matched development comparison**, not an untouched
  release test (the original evaluation cohort has now informed diagnosis).
- Simultaneous candidate response cost preflight completed in **81.047s**,
  manifest SHA `bf735114dbfe21dff5b087c07a36405a502a91717c47afa8574f5fafd3f88868`.
  Started `local-frozen-preflop-response-20260907-simultaneous-complete`, same four
  public chance cohorts and all 49 endpoints, two guarded workers. Candidate SHA
  `4b935953ecbb393c8b57033b6b16d79d0fb4f2b04421c0b9832150af97d31b06`.
  The action response will again be fitted only on indices 0/1 and frozen before
  inspecting indices 2/3. This is matched development evaluation, not untouched
  release qualification. The existing retained alternating candidate and website
  stay unchanged pending these actual response-value results.
- Simultaneous response evaluation completed in **1453.412s**, manifest SHA
  `83d1eef8acd246b4754290f1f68efb2297df9c44b0be65c57f6ea1afb44158fe`.
  The frozen fitting actions match the saved response exactly (SHA
  `99ffd8363eba2f3cff5e555a56347bcb31a8d12b9c869c0ed6f68cc2bcadcf50`).
  Development evaluation totals **1.372425135 / 0.984276626bb**, mean
  **1.178350880bb**, SE **0.194074255bb** across only two board clusters;
  seat means **0.623717655 / 0.554633226bb**. Both boards are worse than the
  retained alternating candidate. Reject adoption/extension; this does not
  establish a full-game exploitability upper bound.
- Diagnosed an independent, reproducible turn/river average-support defect:
  regrets update for legal zero-prior holdings, but averages were multiplied by
  the private root prior. Zero priors never accumulate averages; tiny priors can
  fall below the absolute 1e-9 normalization cutoff. A made-royal-flush fixture
  therefore folded **50%** facing a bet after 16 updates. The original behavior
  remains an explicit control in the regression. The opt-in correction starts
  own realization at one for each legal holding while leaving input ranges,
  payoff/regret calculations, chance sampling and discount clocks unchanged.
  This follows the own-action realization weighting in
  [CFR Equation 4](https://poker.cs.ualberta.ca/publications/NIPS07-cfr.pdf).
  The zero and 1e-16 prior cases now pass; on the positive-prior fixture all
  regrets are bit-identical and normalized average policies agree within 1e-12.
  A playback regression confirms distinct immutable policy identity, unchanged
  flop policy, and the same corrected turn policy in profile evaluation and play.
  Fourteen related Python checks pass. Correction stays opt-in and is not yet
  used by the website. Its contribution to the actual 0.71960bb response result
  is unknown: do not infer strength improvement from a corner-case regression.
- Turn-average cost/semantic preflight completed in **94.828s**, manifest SHA
  `88fb0e34bfd692a5d17b5f220bab2f4127bbd42e000ff04ba7d4eb5442188067`.
  On the retained candidate's same 2/9/18bb-committed endpoints, maximum absolute
  class-CFV changes are only **2.25489e-7 / 1.68012e-7 / 0bb**. Preflop rows and
  other endpoints are unchanged. These values include opponent/chance weighting,
  not conditional EVs. This does not explain the large response gain; retain the
  correctness fix as opt-in but do not launch a training pair or four-board
  strength recapture just for it. Delta report SHA
  `dec6463e076b8cbb05123a411723b2574b8862d4a68288e994ae98152f919b1f`.
  **339 Rust release tests pass, 49 opt-in ignored**, 186.772s guarded, peak
  **1,533,216,616 bytes**. Regression manifest SHA
  `f0c270b76970625ea05811ef20365b4968d952a6ddf6b98af3750063191d7e99`;
  binary SHA `ef1281e0a05c02a81cc806e85244aa53be7946f32d561d2e38278dc75da3e857`.
- Default two-round exports are byte-identical to the original sampled-checkdown
  controls (`493c5258...` / `c68bb471...`); manifest SHA
  `d7532a1964576ae746a532d47073b40a7e1d61ab36dbb1185b494a98ecc75d04`.
  The separate exact-mean/turn-control smoke also completed but is a different
  configuration and is not that byte-comparison reference.
- Saved-data baseline screen, motivated by
  [Davis et al.](https://proceedings.mlr.press/v119/davis20a.html): prefill all 49
  strategic residuals using fitting boards 0/1 rather than starting the sparse
  history cache empty. Exact enumeration of the uniform endpoint lottery at the
  fixed SB root shows development-board variances **22.98434 -> 17.47745bb²**
  and **5.19857 -> 8.50370bb²**: mixed, not adopted. A conservative coefficient
  **0.1927664**, derived only by cross-fitting boards 0/1, gives **20.93513 /
  4.84803bb²**, about 9%/7% improvement. This small fixed-range effect does not
  justify adding another training baseline or claiming improvement as ranges
  change. Report SHA
  `44f83ac5a77a78207eb2f2cbf43923fdfe1aaa42974dbc9bafb74196a7fbdf9c`.
  Four lottery/unbiasedness/cross-fit checks pass. No new native capture needed.
- Next bounded decision: extend ONLY retained alternating turncv32 to 128
  updates, same seeds, chance streams, targets and abstraction. The prior cost
  audit projects 3474/3587s per worker with headroom, under the unchanged one-hour
  and 2GiB caps. Two concurrent workers maximum; 20GiB disk reserve. No extension
  of rejected simultaneous/cache/sizing variants. This directly tests whether
  more observations of the unchanged estimator improve policy response, after
  cheap alternatives failed to deliver substantial robust variance reduction.
  Preserve canonical seed27001 selection and evaluate actual response gain after
  the paired results; do not select a seed by held-out strength or infer passing
  from frequency agreement. Subsequent user direction permits conditional
  512/1,024-update extensions after credible response improvement, not automatic
  extension on completion or website promotion.
- Started `local-compact-preflop-20260907-turncv128-a/b`: continuation cohorts
  28001/28002, each runs solver seeds 27001 then 27002. Two guarded workers total,
  each capped at 3600s/2GiB. The training-only round bound now permits 128 only
  for the retained alternating/uniform/exact-mean/complete-turn configuration.
  No algorithm/abstraction change in this extension. Pinned release binary SHA
  `c5387bc0b10df868baee8c183a9af6583373cf9c3376f87d15c362dcb26c90aa`.
  Eighteen related Python tests pass. Do not rebuild this release binary while
  either controller is active. The strength result remains pending.
- Predeclare the iteration-scaling screen before evaluating 128: compare
  turncv32-a versus turncv128-a for BOTH seeds 27001/27002, same pinned value
  model, 128/64/64 continuation and development chance boards. Fit each response
  on boards 0/1; evaluate frozen actions on 2/3. Reuse the completed 32/27001
  capture, and obtain the missing 32/27002 baseline as well as both 128 candidates.
  Advance to 512 only if mean response gain falls for both solver seeds and the
  seed-averaged improvement has the same sign on both evaluation board clusters.
  Report individual deltas and clustered uncertainty; two boards cannot qualify
  the release confidence gate. If directions conflict, do not automatically
  scale or select a favorable seed. This is a compute-allocation screen, not a
  new approximate-GTO acceptance gate. The second continuation cohort remains a
  separate stability/noise diagnostic, not extra independent evaluation boards.
- All four 128-update runs completed without resource failures. Cohort a/b total
  wall times **5281.428 / 5328.215s** (two sequential seeds per cohort; cohorts
  ran concurrently). Pair manifest SHAs
  `60ba2fc21b56f0d348ea47ac91834078aa4aa72592edf5923ea9e707f778404b` /
  `cd1e88de5fa17743cba50664582bc40df3fa1e18f87900c82da84235e58ea729`.
  Root MAE **20.48123% / 25.52932%** versus 32-update **29.29641% / 28.32652%**;
  primary agreement **22.48521% / 17.15976%**, aggregate deltas **13.35849 /
  16.30858 points**. Stability still fails and is mixed, not a strength verdict.
  Both cohort-a seeds reproduce the first 32 updates' boards, turns, selected
  histories, raw range totals and zero-sum residuals exactly. Evaluation candidates
  are the predeclared cohort-a seeds, SHA
  `81093566cb94d238c17b7d02fe1543384c815ef3fc01c4654f71b2d9d406d01d` /
  `f975b9f395590fc11b80553f83a9b6714e2fed2f6fd7a188c9065c128063ccd9`.
  The missing 32-update seed27002 baseline SHA is
  `4fe949a51c4f59acb548167bd9da10b612d04bdc3f0bad940ade234648bc2966`.
  Frozen response/full-hand probes now accept the explicitly planned checkpoint
  counts without altering continuation budgets or response semantics. Building
  the evaluation binary only after all training workers exited. No 512 run yet.
- All three response preflights passed: 128/27001 **96.380s**, 32/27002
  **96.416s**, 128/27002 **84.030s**, respective manifest SHAs
  `8e6043ee2e36c2222bbe7a442e181b00ef3b4f860ebf43704f364e815d9ff45f`,
  `3efc5cc721422926080e15e2133c62b7b331c9543eb3e874904899d23f074bfd`,
  `07cf7904cb6a580dcacf4c99f47729dfc30ac003009076cb2df3296afb185606`.
  Started their complete captures in that order, sequential controllers with
  at most two native workers total. Each captures all 49 live endpoints on four
  prescribed chance boards, fits on 0/1 and evaluates frozen response actions on
  2/3. Existing 32/27001 captures are reused. Any stage failure stops the queue.
  Evaluation binary SHA
  `2d4c8112b5a2e8766894c00b120b528d1bb6ae2dd0d533b643377e595834db99`;
  frozen-snapshot reload regression and four Python response checks pass.
- First matched strength result (seed27001): 128-update restricted response
  gain **0.860789925bb**, SE **0.087036625bb**, versus 32-update **0.719602961bb**.
  Same-board totals **0.947826549 / 0.773753300bb** versus **0.828365084 /
  0.610840839bb**: worse on BOTH evaluation boards. Seat means **0.663743674 /
  0.197046250bb**. Completed in **1797.268s**. This is not a full-game bound or
  proof of asymptotic stagnation, but does not support the predeclared scaling
  condition. Complete the second-seed matched comparison; no 512 run queued.
- Second-seed evaluation is complete. The missing 32/27002 baseline is
  **0.933973500bb**, SE **0.027508803bb**, board totals **0.906464697 /
  0.961482303bb**, 1847.092s, manifest SHA
  `8e361b24f19167e35f52aef823b11183d4cd75e6feffa6de68afe8ceaad8a921`.
  The 128/27002 candidate is **1.472834374bb**, SE **0.241769067bb**, board
  totals **1.714603441 / 1.231065306bb**, 1795.715s, manifest SHA
  `c599624ea04bb02c49425814bc517065617ae07c9f9380330319fd84d9c5841e`.
  Thus BOTH seeds worsened on BOTH evaluation boards. Seed-averaged gain changes
  are **+0.463800105 / +0.216247732bb**, mean **+0.340023918bb**, board-cluster
  SE **0.123776186bb**. There are only TWO independent chance clusters; do not
  count seed/board combinations as four independent boards or claim a qualifying
  full-game bound. Recomputed all four response summaries exactly from
  hash-verified captures; boards, classes, chance seed, model, kernel and
  continuation semantics match. Historical absent turn-average flag is explicitly
  normalized to its unchanged false default for identity comparison.
  Comparison report `local-compact-preflop-20260907-turncv128-a/response-scaling-comparison.json`,
  SHA `3783ba64d7d1fa17c6f316e4a142292b047dc576d541fd502a14826bcae0e402`.
  **Do not start 512/1,024 or paid compute on this unchanged configuration.**
  Retain the 32-update reference; the agreed conditional scaling screen is done,
  not the overall release plan. No active training/evaluation jobs remain.
  First-seed exact error attribution puts **0.591008924bb** of response gain at
  the opening decision (versus **0.489144807bb** at 32). A targeted next diagnosis
  should compare training-target advantages and frozen played-policy advantages
  for those decisions before proposing another algorithm or long run.
- User authorized continuing that targeted diagnosis. Ranked hypotheses:
  (1) zero-own-reach BR completion inflates opening targets relative to actual
  frozen play; (2) rare endpoint importance corrections destabilize otherwise
  consistent values; (3) averaging preflop then re-solving changes continuations
  relative to the training trajectory. Existing sparse-profile regression
  reproduces training/profile disagreement in **1.29s**. Extended it to verify
  a same-native-solve diagnostic pair matches both existing value paths exactly,
  with no gap at positive-own-root support (**3.82s** optimized debug). A separate
  pure diagnostic backup matches the actual preflop recursion exactly (**0.05s**).
  Added opt-in <=8-update `--diagnose-targets` replay: records exact selected
  public input and class-integrated opening action estimates under training vs
  played targets, same chance/control corrections; still updates from the old
  training targets. It must reproduce the old frozen policy bytes before any
  effect is interpreted. No target change or larger training run yet.
- Target replay completed in 372.738s. Both exported policies are byte-identical
  to the original turncv8-a pair. Manifest SHA
  `e0b362f5edf4836dbc63a17c527f30b119750c01740fd457b5083501d1e0ced3`;
  analysis SHA `2f0625af30d1de916953ec43ac1b9ed25fc43f9f3d7673c5199bfaedb4377a38`.
  At seed27001 update5, T8o limp advantage is +0.912147bb for training versus
  -0.383147bb for played continuation; T8s is +0.449316 versus -0.290111bb.
  These are importance-weighted sampled update estimates, not converged EVs.
  Best actions flip for 1.2066% of combinations in that update. Seed27002
  update5 has maximum advantage gap 0.114172bb but no best-action flips.
  Other observed opening updates have no gap. Current-profile value changes
  remain zero, so on-policy accounting cannot detect this off-support mismatch.
  This confirms a real inconsistent learning signal, NOT that it explains the
  entire 128-update regression or that fixing it guarantees convergence.
  Added isolated <=32-update `--played-profile-targets`: change only preflop
  continuation labels to actual frozen profile CFVs, retaining sampling,
  learned control variate, native budgets, alternating updates and old averages.
  No combination with turn-average correction in this first causal test.
  Caller regression first failed against legacy BR-completed labels; the
  profile-target branch must agree exactly with frozen playback off support.
- Implemented that isolated target branch and verified the caller regression
  green (5.18s optimized debug). Python target/identity/response checks: 5 pass.
  Full guarded release suite: **340 passed, 49 ignored, 0 failed**, 271.503s.
  Release binary SHA `cf1bdb5922b72a1bc10c7d4b7f8a89145189336281fd6832fba803edd3ce6814`.
  Paired 32-update run active at `local-compact-preflop-20260907-playedtargets32-a`.
  Only continuation labels differ from turncv32-a; turn-root averaging stays
  disabled. After training, run the same frozen restricted-response comparison
  against BOTH saved 32-update controls. Do not promote on stability alone.
- Played-target pair completed in **1120.568s** (seed27001 673.532s,
  seed27002 446.050s). Frozen policy SHAs:
  `e5ebcf6bc9ed6775f6c39cf23a49e048bba2e4aedc6d8925a218a9054ffb4bbf` and
  `b05afe1ee1d604891d841c31d1bc5d441450ac0b49d2951c1871a9360beae4eb`.
  Established root maximum-action MAE improves 29.2964% -> **22.9315%**;
  primary agreement 37.8698% -> **44.9704%**, but aggregate delta worsens
  11.6795 -> **21.2799 percentage points**. Probability sums pass; stability
  does NOT qualify. Both three-endpoint frozen evaluation preflights pass
  (116.211s/114.264s). Complete four-board captures active for both seeds at
  `local-frozen-preflop-response-20260907-played32-27001-complete` and
  `local-frozen-preflop-response-20260907-played32-27002-complete`.
  Four total guarded native workers, 10 physical cores, 16GiB RAM;
  representative preflight peak <0.5GB each, individual cap 2GiB/3600s.
  No candidate promotion until actual matched response comparison completes.
- Paired played-target strength evaluation COMPLETE. Seed27001:
  **0.422672821bb**, SE **0.007410911bb**, board totals **0.430083732 /
  0.415261909bb**, 2678.202s; manifest SHA
  `d477beb8fef24c984dc8ee271451eab9265c438133c739d95a0637b1af171d0e`.
  Seed27002: **0.883714613bb**, SE **0.012696573bb**, board totals
  **0.896411186 / 0.871018041bb**, 2704.518s; manifest SHA
  `6b2aa6c429233bbf22823808725ce639696cbde334b8bb19433d13c5b3e27dfa`.
  Both seeds improve on both evaluation boards versus their original 32-update
  controls. Seed mean reductions **0.296930141 / 0.050258887bb**; overall
  reduction **0.173594514bb (~21%)**, board-cluster SE **0.030572918bb**.
  Only TWO independent board clusters, already development data; not a release
  confidence bound, not full-game exploitability. First-seed opening contribution
  falls **0.489144807 -> 0.331969217bb**, still its dominant weakness.
  Recomputed summaries from hash-verified captures; matched chance draws,
  model/kernel, action tree and continuation semantics. Comparison SHA
  `95e6e46e27c6c0d25e2bf3d7e3bfc19cc0d95f44c34db31f1dae047c037428d1`
  at `local-compact-preflop-20260907-playedtargets32-a/response-comparison.json`.
  Retain the isolated target correction as a RESEARCH candidate and extend to
  128 updates with all other settings unchanged; compare actual response gains
  against these new 32-update controls before considering 512/1024. Expanded
  only the pilot allowlist to 128; resource caps and default legacy path unchanged.
- Started corrected 128-update pair at
  `local-compact-preflop-20260907-playedtargets128-a`, using two independent
  concurrent seed workers (`--workers 2`, default remains 1). Controller sorts
  results by seed and stops sibling work on failure; concurrency/identity/order
  regression passes. Binary SHA
  `d91ab7eee488340604229e08c694039190277f454ed20b8731da518dc4574899`;
  only Rust change after the 340-test release suite is the <=128 pilot allowlist.
  Re-ran actual playback/target regression on this release binary successfully.
  Check the first32 numerical update prefixes exactly against corrected32-a,
  then evaluate each resulting frozen policy against that corrected32 baseline.
  Do not rebuild this binary or modify the running controller/helpers mid-run.
- Corrected 128-update training pair COMPLETE in **3392.908s** wall time;
  seeds **3297.328 / 3392.657s**, within the original 3600s/2GiB guards.
  Frozen policies:
  `d6a188c4c6a03ccf342ae50a339b3dc13b066e07ac565dbdcb4b8a453b0e4893` and
  `0083aceaa88a4ebf4af178a637e4f6e42d395e8e6fb3cd6e9b39a8bd416db2dc`.
  All numerical fields in both first32 prefixes exactly match corrected32-a;
  only timings and config-dependent policy hashes excluded. Controller sibling
  stop-on-failure injection also passed without affecting the live jobs.
  Root worst-action MAE **21.0164%**, primary agreement **23.0769%**, aggregate
  delta **12.8119pts**, valid sums; stability still fails and is mixed versus
  corrected32. Started both response preflights at
  `local-frozen-preflop-response-20260907-played128-{27001,27002}-preflight`.
  Evaluate complete captures against CORRECTED32 controls (0.42267/0.88371bb),
  not only against the weaker original target configuration. No512 queued.
- Both corrected128 preflights passed (114.593s/114.316s). Complete paired
  response captures now RUNNING at
  `local-frozen-preflop-response-20260907-played128-27001-complete` and
  `local-frozen-preflop-response-20260907-played128-27002-complete`.
  Preflight manifest SHAs:
  `53126edc41ba2c581ad18499bfedcb2788bef3798a53edf84bdb28ed10c7d9bc`,
  `fe50f4803b42700d2f02294ad37a83ffd7887b4b394c8dc36608b896be81fb8a`.
  Training manifest SHA
  `4dfebe3d2488d24b18e0b2587fed8ae83c62893a979269d9f4375ff196092783`;
  saved numerical-prefix comparison SHA
  `52be8dc4e182d91366ff96929ef27afed994108fcea66ec5eb5857171569f273`.
  No live training remains. Four guarded evaluation workers; no new experiment
  queued until their actual response comparison finishes.
- Corrected128 response comparison COMPLETE and REJECTED for scaling. Seed27001
  **0.849507091bb**, SE **0.115742267bb**, boards **0.965249358 /
  0.733764824bb**, 2994.446s, manifest SHA
  `5dd0993ed20bcd15fd6bc0cd2174b0657066c88b8db4f68b38b7757d824bf5ab`.
  Seed27002 **1.050505165bb**, SE **0.218821510bb**, boards **1.269326675 /
  0.831683655bb**, 3002.808s, manifest SHA
  `d068cad416296502e66b282f5032f8223747e5cb182ca7b83c61006c5803ca73`.
  Versus corrected32, seed mean changes **+0.426834271 / +0.166790552bb**.
  Three of four seed/board comparisons regress. Seed-averaged board changes
  **+0.454040558 / +0.139584265bb**, overall **+0.296812411bb**, cluster
  SE **0.157228147bb**. Two independent development board clusters, not a
  qualifying confidence bound. Recomputed summaries from hash-verified captures;
  identical chance draws, model/kernel, action tree and continuation semantics.
  Report `local-compact-preflop-20260907-playedtargets128-a/response-comparison.json`,
  SHA `dc3fce1f2cd8a90bab97022b0562ebdebdb6991421744c9fab789e02c201d26c`.
  Retain corrected32; no512/1024 or paid compute. Next isolate existing
  `--turn-root-averages` on top of `--played-profile-targets`, <=32 rounds only.
  Rationale: known zero/tiny private-prior averaging defect erases trained
  off-support policies (royal-flush regression). CFR equation4 averages own
  action reach, not chance/private-root prior. Other unresolved hypotheses are
  endpoint/flop sampling variance and re-solving at averaged ranges. This next
  pilot tests strength; no claim yet that the averaging defect explains scaling.
- Enabled that combination only within the existing <=32-round average-support
  pilot guard. Expanded the corrected playback regression through the actual
  preflop-target caller. Initial assertion incorrectly compared a corrected
  control-variate estimate with a raw sample; fixed the TEST to include the same
  exact terminal-aware prediction correction on both sides. No solver math
  changed for that assertion. Corrected test passes (3.97s optimized debug),
  royal-flush zero/tiny-prior test passes (0.75s), unchanged-positive-prior
  regrets/averages test passes (0.10s), seven Python checks pass.
  Rebuilding release before paired32 average-support strength pilot.
- Average-support + played-target32 training COMPLETE at
  `local-compact-preflop-20260907-playedaverages32-a`, **789.763s** wall time;
  seeds **789.055 / 573.121s**. Manifest SHA
  `5f85986da1d6922ca0bfa03c16a2d1c02c5872edd8899c1c16b8cd7ed426911e`.
  Frozen policies:
  `af8a2b02ad493af7dc7056048c12403285e9b7ed44af6dec075ed07ba4de47f8`,
  `a0f24bed81d0975ec5c0b395e4a9369d277d172fd7ffe3420f912f6ce1426a94`.
  Root worst-action MAE **29.7088%**, primary agreement **37.2781%**, aggregate
  delta **11.7755pts**; valid sums, stability still mixed/failing. Full guarded
  release suite **340 pass, 49 ignored, 0 fail**, 298.037s, manifest SHA
  `6aae8fe3b94fb84a9a5cb3b73c4807ee9ee7808999a0c86944a4f6c96316dd8f`.
  Current binary SHA
  `ffd990ff1cfe40cbf2fe0f0930e7cc9d5485f00a63e92825bc99a5ea3f43a2be`.
  Seed27002 response preflight passed124.386s, manifest SHA
  `d6f5985892b8c44ea24799618ab8b00fbaf4c4bf3618cff5c3fb988214753d93`;
  complete capture running at
  `local-frozen-preflop-response-20260907-playedaverages32-27002-complete`.
  Seed27001 preflight running at matching `...-27001-preflight` path.
  IMPORTANT: use `--turn-root-averages` for this candidate's response captures;
  it pins actual corrected playback, not just training. Compare to retained
  playedtargets32 (0.42267/0.88371bb), allowing ONLY this intended continuation
  flag difference in otherwise matched model/kernel/chance/tree metadata.
- Both average-support preflights passed; seed27001 took123.333s, manifest SHA
  `e73f043b41c8a190c8430c25d6652e5147c48e887814125f6d640d7fbf186521`.
  A task interruption stopped controller bookkeeping but detached native board0/1
  workers finished successfully for BOTH seeds (1759.53/1758.80s and
  1656.88/1655.01s). Original manifests remain `running`; do not mistake that
  stale status for live workers or rerun those completed boards.
  Added explicit hash-pinned `--recover-from` / `--recover-sha256` support to
  the response controller, always into a NEW output directory. It validates
  native completion logs, complete numerical captures, primary input pins,
  chance/tree/continuation identity and any recorded output hashes; incomplete
  started boards fail closed rather than risking duplicate workers. Recovered
  resource telemetry is explicitly unavailable, not retroactively certified.
  Regression first failed on the unsupported recovery CLI, then passed through
  the actual controller: only boards2/3 dispatched, original files unchanged,
  mismatches and unfinished workers rejected. Six focused Python tests pass.
  A real guarded child completed under the current sandbox (2.029s, 6.72MB
  measured footprint, no resource stop). Four remaining native boards now run
  in `local-frozen-preflop-response-20260907-playedaverages32-{27001,27002}-recovered`.
  Original interrupted manifest SHAs:
  `779406fbbf3efd736c83b9c33c28cc254702120c23269c560a295e3952eddc20`,
  `d1c5623caba3e8e407fa58c0b3b7685b8c9af5be7afba0de1f35258674deda5c`.
  No training repeated; no further scaling authorized by the current evidence.
- Average-support response pair COMPLETE, REJECTED for retention/scaling.
  Recovered seed27001 manifest SHA
  `aeb55d49163c68c5892d0e88fe7e47edea34b271e61f0330f2c202622df8423d`,
  seed27002 SHA
  `0bf1b1c87e76978cc79d70046db2d6e923abeaf53c0818e0994e6e3fcf622e34`.
  Remaining-board wall times **1461.526 / 1462.840s**; all four fresh workers
  pass, peaks **0.804–0.916GB**, no resource stops. Recovered fitting-board
  guard coverage remains unavailable. Numerical outputs complete and verified.
  Summed restricted response gains **0.760010840 / 0.964696309bb**, SE
  **0.085016775 / 0.007989609bb**. Versus retained playedtargets32,
  mean changes **+0.337338019 / +0.080981695bb**; all four seed/board comparisons
  regress. Seed-averaged change **+0.209159857bb**, two-board cluster SE
  **0.028459841bb**. Not a qualifying bound or full-game exploitability.
  Comparison recomputed from hash-verified captures with matching chance,
  model/kernel/tree; only intended continuation flag differs. Report SHA
  `4b0557103fda6fd4ee5ac76cb8e957a05658f3a61cd79a8b49c69a1de63cf1c7` at
  `local-compact-preflop-20260907-playedaverages32-a/response-comparison.json`.
  Root-opening contributions rose from **0.331969/0.344578** to
  **0.521870/0.456598bb**. The correctness regression for zero-prior averages
  still stands; it does NOT imply a finite-budget full-candidate improvement.
  Keep the option/test, not the rejected policy. No website promotion.
  Saved actionable root decomposition (SHA
  `0a69bf4f24915d17c6f8b1631887348c16898a293bef7b4cc61b06d5956e4472`):
  weak 42o/52o/62o/etc. take costly 4–5bb openings; these are sampled
  development-board advantages, not precise action-EV grades. Retained seed2
  received NO native SB update on the 2bb or 2.5bb opening branches in32 rounds;
  seed1 received only1 and3 respectively. Exact checkdown baselines still cover
  these branches; this is missing expensive strategic correction, not a lookup
  coverage claim. Next trace the actual retained root updates before changing
  sampling/iterations again. Nine focused Python tests and diff check pass.
- Added opt-in `--trace-root-updates` only for unchanged <=32-round played-target
  replays. Records actual root action estimates, current mixes, and discounted
  regrets before/after the existing update; no additional native solve or RNG
  call. The new regression first failed to compile before the trace existed;
  after implementation it passes in0.05s and checks exact class/chance-weighted
  regret deltas plus unchanged root regrets on the other traverser's update.
  Controller tests cover traced flag propagation. Release build passed; the
  initial short-name command with `--exact` selected zero tests, so the fully
  qualified test was then explicitly run and passed (do not count the zero-test
  command as verification). Current binary SHA
  `de7e01aa558ad2f57ed448c4b98e7a249d2f4f2990a76da8c44497fefcedf152`.
  Full guarded release regression and paired unchanged32 trace RUNNING at
  `local-compact-regression-20260907-root-trace` and
  `local-compact-preflop-20260907-roottrace32-a`. Require frozen policy byte
  identity against playedtargets32 before interpreting diagnostic updates.
  This is diagnosis of the retained policy, not another candidate or scale-up.
- Read-only32 replay COMPLETE: **737.762s**, seeds737.574/561.674s, manifest
  SHA `fd913ca5851f5d412b1fa9c5df805f262e5e3c0b5eca25dadbad083b144bcc40`.
  BOTH frozen policies byte-identical to retained playedtargets32; all numerical
  progress fields exact after excluding trace/timings. Every traced regret delta
  reconciles with the actual update. Root diagnosis SHA
  `dc5f7e329e822c6ad1f619918ba09a03ce081c226ef25296a08ac1f1fbd702e9`.
  Seed1 round19 raises42o's current4bb probability0->93.32%, from a sampled
  estimate18.73bb; seed2 round15 raisesATo0->95.64%, estimate73.22bb, and
  62o0->82.60%, estimate16.94bb. These are importance-weighted update estimates,
  NOT physical action EVs. Do not clip them to stack bounds and claim unbiasedness.
  Full guarded release suite **341 passed/49 ignored/0 failed**,282.663s,
  manifest SHA `96ff4dd84198788b9ddb6f156160c6e52bf76ac5a107f92e688c122c5a5347a4`.
- A fixed-observed-sequence discount probe FAILED to replay the original at
  round2. This exposed a concrete initialization bug, not a new parameter result:
  first-round positive regrets received0.369398 instead of0.738796 on round2;
  negatives0.25 instead of0.5. `begin_update` eagerly discounts existing nodes,
  then `sweep_preflop_average` creates first-round nodes but discounts only COPY
  snapshots. Their original discount metadata remained at0 when round1 regrets
  were added, retroactively applying the round1 factor again on round2.
  New actual-caller regression failed with this exact symptom (0.11s).
  Fixed by stamping any newly created zero-regret nodes after the averaging sweep
  and before regret updates. Existing-node pass is idempotent; averages still
  observe properly discounted policies. Targeted release tests rebuilding now.
  Do not treat the extra first-update discount as proof of the sole strength
  bottleneck; sparse samples and large importance corrections remain observed.
  Research cross-check: Brown/Sandholm2019 section6 distinguishes sampled LCFR
  from its full-traversal DCFR experiments; VR-MCCFR addresses unbiased
  variance reduction. These support examining sampling/discount interaction,
  not promising any parameter choice will solve Holdem.
  Sources: https://www.cs.cmu.edu/~sandholm/cs15-888F21/reweighting.aaai19.pdf
  and https://arxiv.org/abs/1809.03057 . No discount-parameter pilot launched.
- Discount-stamp fix: all three targeted release tests PASS (0.13s), including
  the first-update regression that failed before the fix. Nine focused Python
  tests and diff check pass. Binary SHA
  `d5fa569ff79baf799169a651e328b0c4dc5989529e99aa19cee89dc4a53c4eb3`.
  Paired32 validation RUNNING at `local-compact-preflop-20260907-discountstamp32-a`
  with the retained played-profile/turn-control settings, original chance and
  endpoint seeds, root-turn-average option OFF, trace ON. Only algorithm change
  is the initialization stamp, not regret exponents, abstraction, or sampling.
  Full guarded release regression also running at
  `local-compact-regression-20260907-discount-stamp`. After training, evaluate
  actual response gains against retained playedtargets32; do not claim strength
  improvement from this correctness fix alone. No long extension queued.
- Discount-stamp32 training COMPLETE,706.810s wall, seeds706.623/517.701s;
  manifest SHA `3227f215e3c439cc18fb59d9d0f49099fad0c027562edba9f80926da7990693a`.
  Policies seed1 `443f89fc14ae9526fad09886f1acef95c106fc1eafd07595d49fd83b8d9384f5`,
  seed2 `60d8d12a024af983d09ee9fcd1a3049f0c76dd5f03c23144f3782bc439d4abe8`.
  All64 traced updates now satisfy the FULL inter-round discount/current-policy/
  regret-update recurrence; the original failed that recurrence on round2.
  Root stability remains mixed/failing: worst-action MAE24.4397%, primary
  agreement46.7456%, aggregate delta24.4397pts. Do not infer strength from these.
  Full release suite **342 passed,49 ignored,0 failed**,267.078s, manifest SHA
  `6837cbd97e38c44aeedb77cb3011d43ed8f627145c8c04ec4903f91bc883446e`.
  Response preflights passed103.520/111.073s, manifest SHAs
  `42226a710ff934903e18c4019ace152f60bfd398113261ea3d90e7708b56cbe4`,
  `56585c0fbe366d269946bbd2f25ad5d0b69c622507b8538bd99103983f76212f`.
  Complete paired response captures RUNNING at
  `local-frozen-preflop-response-20260907-discountstamp32-{27001,27002}-complete`.
  Compare to retained playedtargets32 with IDENTICAL continuation metadata.
- Cheap open-loop replay of the recorded corrected action-value sequence screened
  negative-regret exponents0/0.5/1.5 without launching any native training.
  Beta0 reproduces the actual corrected root. Higher exponents reduced some
  weak-hand4bb jumps in seed1 but not seed2; fixed-sequence regret also improved
  in seed1 and worsened in seed2. No paired support for a parameter pilot.
  This is NOT full-game performance or a new release metric. Report
  `local-compact-preflop-20260907-discountstamp32-a/discount-sequence-probe.json`,
  SHA `d05f62fda2eb12ac346b98a4d6b48a4c013fdd5b9e47d7660051d00e7c57abf5`.
- Discount-stamp32 response pair COMPLETE: **0.505903986 / 0.875799869bb**,
  SE **0.044827118 / 0.039978856bb**; mean changes versus retained playedtargets32
  **+0.083231165 / -0.007914744bb**. Three of four seed/board comparisons regress.
  Overall change **+0.037658210bb**, two-board cluster SE0.032349245bb. Root
  opening contributions improve slightly to0.307791/0.331411bb, but total
  resistance does not. Keep initialization correctness; no stronger-candidate
  or scale-up claim. All guards pass, peak1.143/1.036GB, elapsed2473.668/2429.287s.
  Capture manifest SHAs `d18991a0e29f37719e80913c93e447f6fff2ec47ceed05fafea7820f72842053`
  and `08d9eb0d5fb030fb6637a5feb50bd1e592e3fc28464aa32b6fb5a60f7969ae88`.
  Comparison SHA `776ce2e115a1aedf008d0c1e19d245e8ea85c152dc59eef6fafc634247c37ae9`.
- Added read-only exact checkdown baseline export using the EXISTING Rust
  kernel, no new poker calculations in Python and ZERO native postflop queries.
  Both frozen baselines captured in0.51s each at
  `local-frozen-preflop-baseline-20260908-discountstamp32`; manifest SHA
  `ecd18ff8416f8200ffe05f9e80529db3cf8c92f68071e82a0d73791abf2f1ad4`.
  Binary SHA `5296e93888c973c2b6ffc7ae16630f083bda9d54833f59b8e76c03a4bc22dd33`.
  Targeted Rust snapshot/reduction tests pass;11 Python tests pass. Baseline
  exact terminals match all saved captures, and isolated endpoint residuals
  sum exactly to full decision-advantage differences.
  Analytical sampling variances match exhaustive draws on tiny fixtures.
  Six-query root stratification lowers root conditional variance16–19% versus
  a six-query uniform batch, but INCREASES BB-vs-limp variance2.53–2.56x.
  Reject that simple allocation; do not rerun it as an alleged improvement.
  Decision variance report SHA
  `079d08523db0c51dd94b480770e0d584e1c522744f8034afc2f0e918aea33e59`.
  These are conditional variance diagnostics, not new release gates or strength.
  Next cheap screen: actor-specific importance proposals fitted to all preflop
  gradient components on boards0/1; inspect variance on boards2/3 before any
  native training. Preserve at least50% uniform support. No new model run yet.
