# Bounded local public-chance training

The selected research configuration is fixed DCFR, zero averaging delay, full
default sizing, public-chance sampling and exact terminal-action integration.
Train seeds 26001/26002 **sequentially**. This is the tabular full-game trainer,
not the web app's older frozen neural model. Do not promote these pilots or
claim full-game exploitability from their root diagnostics.

**September 4 completion:** both 800-round seeds, exact checkpoint replays and
the four-policy full-game response cohort are complete. No training job remains
running. See the [final audit](../validation/2026-09-04-local-checkpoint-response-audit.md)
for measured improvements and the remaining late-street coverage problem.
The dated launch/failure sections below preserve the sequence's history.

## Safety and reproducibility

`cloud_blueprint_run.py` also runs locally. It now samples worker resources
every 250ms, including during checkpoint decode/write and policy export:

- `--max-worker-memory-gib`: macOS `ri_phys_footprint` includes compressed
  memory; Linux uses RSS plus swap. RSS alone can understate Mac pressure.
- `--max-worker-minutes`: wall time per seed, including checkpoint reload,
  evaluation and export. Two sequential seeds can consume twice this limit.
- `--minimum-free-disk-gb`: checked before launch and continuously by each
  worker monitor. Despite the historical flag name, the unit is GiB.
- A resource stop sends SIGTERM to the owned worker process group, escalates
  after three seconds, stops peers, and prevents queued seeds from launching.
  Three consecutive missing memory readings fail closed (normally within
  0.5–0.75s), allowing a dying child to be reaped after a transient empty read.
  Sampled peaks and stop reasons are
  recorded in `run-manifest.json` after each worker exits.

These are **sampled stops**, not kernel allocation caps or a guarantee against
OS memory pressure. Other applications still consume memory; use one worker
and leave several GiB for the OS. Mac preflight reports installed RAM, not
currently available RAM. Watch actual worker footprint, not the node-count
projection alone. Limits are operational controls and do not change the
policy fingerprint; the actual values are recorded in each attempt.

The Rust writer streams a temporary MessagePack/gzip checkpoint and renames it
only when complete. Terminating a write preserves the previous completed
checkpoint; it does not save work since that checkpoint. This is not a promise
of power-loss durability. Do not resume a `.tmp` file or automatically relaunch
a memory-stopped job unchanged. No watchdog restarts stopped training.

Large files stay under ignored `preflop-solver/neural/runs/`. Keep the run
manifest, compact summaries, binary digest, and original checkpoint together.
No cloud resources or external credentials are needed.

## First checkpoint stage

From the repository root, after a release build:

```bash
python3 preflop-solver/neural/cloud_blueprint_run.py \
  --binary "$PWD/preflop-solver/target/release/preflop-solver" \
  --output-dir "$PWD/preflop-solver/neural/runs/local-pcs-20260904-checkpoint400" \
  --depth 20 --iterations 400 --seeds 26001,26002 --max-concurrent 1 \
  --max-information-sets 12000000 --bytes-per-information-set 450 \
  --averaging-delay 0 --checkpoint-every 400 \
  --public-chance-sampling --integrate-terminal-actions \
  --no-export-postflop-strategies \
  --held-out-deals 2000 --root-deviation-samples 256 --action-value-deals 2000 \
  --max-worker-memory-gib 6 --max-worker-minutes 20 --minimum-free-disk-gb 20
```

The 400-round stage establishes real checkpoint files at a meaningful size;
it is not another claim of new strategy improvement. Exporting only preflop
keeps the diagnostic artifact small; the checkpoint retains **all-street**
learning state. It is not a deployable full-hand artifact.

Only after that stage completes, resume in a new directory to 800 rounds with
the same immutable training settings, a 22M information-set ceiling and a
7.5GiB worker stop. Then reload the resulting 800-round checkpoint in a separate
`--evaluation-only` stage and compare canonical artifact hashes against the
original. The larger reload must pass before any overnight extension. Do not
assume its peak equals training memory. Stop to investigate if it exceeds the
budget; do not silently raise the stop to consume the whole machine.

The runner verifies parent binary, immutable settings and checkpoint hashes.
A completed evaluation-only stage can change evaluation sample counts, but
an exact replay check must keep them unchanged. Retain both independent seeds;
no averaging of their regret tables.

## Verification

```bash
cd preflop-solver/neural
python3 -m unittest test_cloud_blueprint_run test_worker_resources -v
```

The tests exercise real process termination, SIGTERM-resistant escalation,
peer-stop propagation, failed memory measurements, disk reserve stops, intact
prior checkpoint files, and Linux swap accounting. A tiny-memory real-solver
smoke additionally checks that a resource-stopped stage does not start seed B.
Research progress still requires improved policy diagnostics and, separately,
a legal full-game response evaluation on this exact tabular policy.

## Active September 4 sequence (launch snapshot)

The initial checkpoint pair is running in
`preflop-solver/neural/runs/local-pcs-20260904-checkpoint400`.
Seed 26001 completed in 274.26s including checkpoint serialization/evaluation,
with 3,269,659,288 bytes sampled peak footprint and a 1,223,208,123-byte
checkpoint. Its root diagnostic is 0.463259bb, reproducing the earlier pilot;
this is checkpoint-readiness evidence, not a new policy gain. Seed B was still
running at this snapshot. Existing system swap was about 1.4GiB before launch;
do not claim this run established zero swap.

The one-off controller at
`preflop-solver/neural/runs/local-pcs-20260904-stage-control.py` waits for the
initial pair, then runs `local-pcs-20260904-checkpoint800` followed by
`local-pcs-20260904-reload800`. It stops on any failed stage and checks both
canonical artifact hashes for exact replay. It does **not** automatically
start a larger/overnight extension or publish a model. Consult the live
manifests rather than treating this launch snapshot as a completion report.

## Reload failure and lossless fix

Both 400-round seeds completed. The original reader then hit the 7.5GiB
watchdog after about 21 seconds: 8,093,947,904 bytes of process footprint,
before continuing training. The old decoder allocated private copies of all
descriptor/action strings and slices, then deduplicated only after reading the
entire table. Streaming compression was not the cause.

The reader now interns each node as it is decoded. A regression test asserts
sharing **before** trainer construction for both JSON/gzip and MessagePack/gzip,
checks serialized node equality, and verifies regret arrays remain independent.
The checkpoint schema, sampling sequence, and learning math are unchanged.

Both real 400-round checkpoints then reloaded under a stricter 6GiB stop:

| Seed | Sampled peak footprint | Reload + evaluation | Canonical artifact match |
| --- | ---: | ---: | --- |
| 26001 | 3,955,756,200 bytes | 43.196s | Exact |
| 26002 | 3,720,760,320 bytes | 38.185s | Exact |

The old binary and original checkpoints were preserved. The runner's new
`--verify-checkpoint-reader-upgrade` option is restricted to evaluation-only
at the same round count. It pins evaluation seeds/budgets, grid, export scope,
and checkpoint digests, then requires the exact parent canonical artifact
hash. It cannot train or accept a changed policy output. Revalidation manifests
keep the original immutable checkpoint path instead of duplicating gigabytes.
Normal resumptions still require the same solver binary.

The current continuation uses the frozen `local-pcs-20260904-solver-streaming-reader`
binary, the revalidated `local-pcs-20260904-reader-upgrade400` parent, and the
new output directory `local-pcs-20260904-streamed800`. The earlier
`local-pcs-20260904-checkpoint800` directory records the failed old-reader run;
do not overwrite it or confuse it with the corrected run.

## All-street response evaluation

`full-game-lbr --tabular-checkpoint <path>` loads this checkpoint's frozen
**average** policy directly, without distillation or a neural model. Use
`--maximum-response-granularity strategic` for the bounded broad-response pilot.
Training, calibration and final evaluation use separate deterministic deal
streams. The existing response changes at most one decision in each full hand;
this is a legal but restricted responder, not a full best response.

The output pins checkpoint SHA-256 and rounds. It reports policy unknown/
untrained lookups separately by street and phase. At unknown nodes, the
offline evaluator explicitly uses the trainer's uniform profile completion;
this does not create serving rows or make the incomplete policy publishable.
The confidence interval is on response gains, never an exploitability upper
bound. A zero deployed gain can mean the responder lacked power or coverage.

`total_response_gain_bb_per_hand` sums both seats (negative sample estimates
are retained). The legacy `approximate_exploitability_*` fields retain their
historical half-sum convention. Do not compare that half-sum to the user's
roughly 0.50bb **total** preference. Full-game rollouts settle the sampled exact
runout; root diagnostics use conditional all-in equity, so per-deal numerical
outcomes need not match even when policies do.

### Watchdog shutdown-race correction

Both corrected-reader 800-round workers completed and wrote valid outputs.
Seed B exited with code 0, but macOS briefly returned a zero footprint before
`waitpid` observed exit. The initial watchdog incorrectly set a resource stop,
then `killpg` hit an exit-time permission race. No training checkpoint was lost.

Two failing regression tests reproduce those races. The guard now confirms
persistent telemetry failure over three polls and falls back to Popen's
owned-child signalling if its process group is disappearing. Real memory/time/
disk stops and SIGKILL escalation still pass tests. The failed attempt manifest
is preserved as `run-manifest.attempt-1.json`. The same runner command reloads
the completed checkpoints without adding rounds or rewriting those checkpoints,
regenerated/validated outputs with identical canonical artifact and checkpoint
hashes for both seeds. This recovery itself completed the full-size replay;
a subsequently started duplicate replay was stopped as redundant.

## Completed sequence and next boundary

Both 800-round policies replayed below the 7.5GiB stop, peaking at approximately
7.00 / 6.94GiB, with no extra training rounds or checkpoint rewrites. Four
matched full-game response jobs then completed (both seeds at 400 and 800).
The total restricted-response gains changed from 0.920 / 0.875bb to
0.805 / 0.733bb per hand. These are paired point-estimate improvements, not a
statistically proven difference or an exploitability upper bound.

The new per-street coverage shows 89–92% unknown turn and over 99% unknown
river source-policy lookups on the independent response corpus. Known but
untrained nodes are additional. Do not treat the preflop-only export as a
full-hand serving policy, or use uniform completion in the website.

No larger extension or automatic promotion was launched. The bounded local
sequence is complete, but late-street coverage, root stability, action-EV
precision and full serving validation remain unresolved. Consult the final
audit before choosing another policy experiment; do not blindly double the
table or infer overnight/cloud readiness from 800-round feasibility.
