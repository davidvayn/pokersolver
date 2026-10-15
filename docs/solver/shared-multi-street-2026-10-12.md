# Shared multi-street engine and deeper-stack pilots

## Implementation sequence

1. Use the existing full-hand hand engine for both Practice and Solver. Solver
   selects public board cards; Practice retains authentic dealing and settlement.
2. Extend the existing pinned native resolver query with an optional complete
   public-range strategy. Export the same cached probabilities and action values
   used by Practice, not a separate one-street solve or made-up missing values.
3. Replace the Solver page's single-street WASM execution path with a study-line
   controller. Keep decision cards across streets and board edits, carry real
   stacks/pot/history forward, and condition ranges through the native trajectory.
4. Verify preflop, flop, turn, river, folds, all-ins, manual board entry, rewinding,
   loading/errors, terminal retention, and Solver/Practice policy parity. Preserve
   the existing action-card and combo layout and explicitly expose uncertainty.
5. Start bounded, independent 50bb and 100bb local training pilots using the
   depth-aware trainer and a pinned, spread-out rake-free action abstraction. Do not train
   from mislabeled 20bb targets, modify active weights, or activate pilot models.

## Acceptance constraints

- Only installed policy depths are selectable for live study. 50bb/100bb remain
  unavailable until independent training and validation support them.
- Existing 20bb policy is experimental; full-game exploitability has not passed.
- Manual board edits retain the chosen multi-street action trajectory. A missing
  required board card pauses the line rather than clearing or guessing it.
- Public range study cannot settle an unknown private showdown winner.
- No new rake work, paid compute, AWS resources, or scheduled automation.

## Pilot grid

At most three regular sizes per decision, with all-in separate. Opens:
2/3.5/5bb; limp raises: 3/5/7bb; three-bets: 8/12/16bb; four-bets: 20/30/45bb.
Deeper raises use 0.5/1/2 pot fractions. Existing distinct flop bets
(1/3, 3/4, 5/4 pot), turn/river bets (1/2, 1 pot), and one pot-sized postflop
raise are retained. Stack-clipped duplicate all-ins are deduplicated by the
hand engine. This reduces the five-open grid without claiming that abstraction
reduction itself decreases full-game exploitability.

The explicit grid is hashed into the Python run identity, passed to Rust
sampling on every round, checked against returned shard metadata, and exported
in the model. Existing Home resumes omit the new optional field to retain
their hashes. The live 20bb policy is unchanged: filtering its probabilities
would not be equivalent to training in this smaller game.

The supervisor runs two jobs concurrently, two independent seeds per depth,
up to three rounds with a 12-minute soft budget, 30-minute hard limit per seed,
and 3.5GiB process-family CPU RSS limit. GPU/unified-memory usage is not included
in RSS; the small reservoirs and two-job limit are intentional on this 16GB Mac.

## Verification and pilot checkpoint

- TypeScript: `npm test` — 286 passed, four optional native integration tests
  skipped. `npm run build` passed, including frozen serving-artifact hashes.
- Rust: library release suite — 457 passed, 59 guarded research tests ignored;
  CLI release suite — nine passed. Legacy WASM release suite — six passed;
  regenerated package/runtime WASM copies have matching SHA-256 hashes.
- Python: `python -m unittest test_depth_pilots test_train` — 26 passed.
- Real browser: complete eight-decision preflop/flop/turn/river line; pot/stacks
  maintained; missing river removal/restoration retained all cards/actions;
  terminal deselection retained the matrix/combos. Fold, all-in runout, blocked
  policy request with Retry recovery, keyboard focus, picker Escape, and
  Practice's full-hand all-in/showdown also checked. Responsive layouts at
  375/768/1024/1440px, light/dark and reduced motion; no page-wide horizontal
  overflow or application console exceptions observed. The decision rail
  scrolls the newly active card into view.

All four three-round pilots completed: 50bb seeds 50121/50122 and 100bb seeds
100121/100122. Each used 64 traversals per round; this is a pipeline smoke
pilot, not adequate strategy training. Final held-out action-value RMSE was
18.978/19.085bb (50bb) and 37.022/41.991bb (100bb). These sampled-target
diagnostics do not measure exploitability, EV confidence, or cross-seed
equilibrium convergence. No new depth or weights were activated.
The four final binary artifacts each have 660,868 finite parameters and are
about 2.65MB. Re-encoding their JSON sources produced identical binary bytes;
each exported grid has at most three regular sizes and the correct depth.

Artifacts/logs are under
`/Users/davidvayntrub/pokersolver-training/shared-depth-pilots-2026-10-12`.
One sampler launch for seed 100122 stalled before Rust execution: `sample`
showed only `_dyld_start`, near-zero footprint and no CPU/shard progress, while
a fresh binary `--help` invocation succeeded. Terminating that exact child and
retrying the unchanged checkpoint completed all three rounds in about ten
seconds. No poker algorithm/seed/grid change was made. A deterministic unit
test cannot reproduce this macOS loader incident; tests instead cover retry
history and preserving the other completed seeds. `--only-seed` supports that
bounded recovery and the result retains the failed attempt.

Next policy work is a matched, larger pilot and independent full-hand response
evaluation, including responses with omitted/off-grid sizes. Fewer actions can
improve throughput within the abstraction, but do not prove lower real-game
exploitability. The existing 20bb policy also needs matching retraining before
this spread-out grid can replace its five-open action set.

## Full run — 2026-10-13

User authorized the longer run after the smoke pilots. The controller is
`preflop-solver/neural/run_depth_full.py`; output is
`/Users/davidvayntrub/pokersolver-training/shared-depth-full-2026-10-13`.
Two independent seeds per depth (50bb: 50121/50122; 100bb: 100121/100122),
eight hours each, two jobs concurrently: roughly sixteen hours total plus
checkpoint/startup overhead. This run does not retrain or replace live 20bb.

Fresh run directories are required because the smoke configuration differs.
The full profile uses 400 traversals and 100 optimizer steps per round,
100,000-record reservoirs, 1,024-sized batches, 256/128 preflop/flop runout
samples and four independent action-value rollouts. Architecture and spread-out
action grid are unchanged. Later-street chance remains sampled. Snapshots are
exported every 100 rounds and on normal completion, not every round; consumed
shards and superseded checkpoints use the trainer's existing retention rules.

The Rust sampler is copied to the output directory and SHA-256-pinned in each
run configuration, independent of later website rebuilds. Python source and
exporter hashes are recorded; changed pinned inputs prevent queued seeds from
starting with different code. `--sampler-binary`/`--sampler-sha256` preserve the
legacy run identity when omitted. Resume budgets subtract the complete recorded
round durations rather than granting another eight hours.

Guards stop owned workers on non-finite metrics, truncated sample output,
3.5GiB CPU RSS, less than 25GiB disk free, thirty minutes without a checkpoint,
or the remaining time budget plus a thirty-minute hard grace. Worker cleanup
tracks descendant births so an orphaned sampler is not leaked and a reused PID
is not targeted. These guards catch execution/data-integrity problems; they
are not an exploitability or learning-quality evaluator. Normal time-budget
completion checkpoints the last round; forced termination can lose in-flight
work and requires a checkpoint/replay audit before resuming.

`caffeinate` prevents idle sleep only while the controller is alive. Closing
the lid or powering off can still suspend training. Keep the Mac plugged in
and lid open. No cloud billing or automatic model promotion is involved.

Preflight: native release build and 37 Python tests passed. Both first seeds
completed initial full-profile updates/checkpoints with finite metrics and
untruncated records. This verifies that training actually runs; early cold-start
errors are not evidence of equilibrium strength.

## Release audit — 2026-10-14: rejected, not activated

All four seeds completed their eight-hour training budgets normally. The 50bb
seeds finished 465/462 rounds (186,000/184,800 traversals); 100bb finished
366/370 rounds (146,400/148,000 traversals). Training completion did not satisfy
the existing release criteria. Neither depth is enabled, no winning seed was
selected, and the active experimental 20bb manifest/weights are unchanged.

The held-out authentic evaluation used an equal mixture of both frozen policies
over exact-card trajectories, with separate uniform-policy forced deviations.
Every evaluation uses the exact trained stack depth and spread-out action grid;
the validator now checks returned game metadata and can pin the sampler binary
by SHA-256. The native certificate entry point also accepts the explicit grid.
This fixes evaluation identity, not poker strategy or measured exploitability.

| Measured criterion | Existing gate | 50bb | 100bb |
| --- | --- | --- | --- |
| Cross-seed action-frequency MAE | at most 5 percentage points | 7.89 — fail | 7.42 — fail |
| Primary-action agreement | at least 85% | 73.49% — fail | 76.00% — fail |
| Maximum aggregate action delta | at most 3 percentage points | 2.28 — pass | 2.13 — pass |
| Authentic decisions evaluated | sample size, not a gate | 17,207 | 16,126 |
| Forced-deviation decisions | sample size, not a gate | 6,743 | 6,944 |

Observed inference coverage and normalized probabilities passed on both sampled
distributions. This is not exhaustive coverage or equilibrium proof. Preflop
primary-action agreement was especially weak (66.33%/69.37%). All four final
binary artifacts have finite parameters, correct grids/depths, and identical
bytes after source re-encoding; together they occupy 10,585,386 bytes. This is
not a total hosted-storage audit of a future sample/oracle serving corpus.

A separate independent EV reference pass used 128 traversals per reach dataset
and 64 rollouts per action. Only 5.06%/2.53% of evaluated decisions met the
0.02bb all-action reference-SE threshold, versus the existing 95% requirement.
The exported value heads' own policy-action-weighted predicted precision
coverage was also only 36.13%/30.07% (50bb seeds) and 34.73%/33.72% (100bb).
Those predictions describe Monte Carlo target noise, not calibrated neural
approximation error. More reference samples alone would not resolve the
cross-seed failures or validate learned action-EV accuracy.

Full-game exploitability and its 99% upper bound were **not measured** in this
audit: decisive normal-gate failures already reject both candidates. No numeric
exploitability improvement, regression, or pass is claimed. The existing
0.50bb/hand release thresholds were not changed or waived. Off-grid responses
and full-hand evaluation remain necessary for any future accepted candidate.

The machine-readable [rejection decision](depth-release-audit-2026-10-14.json)
records measured results, artifact/report hashes, evaluation budgets, and
unmeasured gates. The original reports and weights remain in the full-run
output directory, outside Git. Older reports' false exploitability booleans
must not be read as a measured failure: their certificates are absent/null.
The validator now distinguishes unmeasured gates from measured failures and
reports release eligibility without activating models.

Final verification: 47 Python tests passed (including a tiny native certificate
transport test, not a strength certificate); nine Rust CLI release tests
passed; 290 TypeScript tests passed with four unrelated optional native tests
skipped, including four real-artifact full-hand compatibility tests; production
build passed. No browser-facing implementation changed in this audit.
