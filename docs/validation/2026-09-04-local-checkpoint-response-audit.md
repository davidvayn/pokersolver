# Local checkpoint and full-game response audit

September 4, 2026. Base commit `697a8db`; native solver 0.3.0, checkpoint
schema 5. This completes the bounded local continuation/replay/response
sequence, **not** the broader Approximate GTO release objective.

Machine-readable evidence: [audit JSON](2026-09-04-local-checkpoint-response-audit.json).
Reproduction and safety controls: [local runbook](../solver/local-blueprint-runbook.md).
This follows the [mathematical audit](2026-09-04-pcs-math-audit.md); its then-unmeasured
tabular full-game response now has a measured restricted-response result.

## Outcome

Both independent seeds completed 800 rounds locally. Frozen checkpoint replays
reproduced both canonical policy/evaluation artifacts exactly. A new adapter
evaluated these exact **tabular average policies**, rather than assigning the
older web neural model's scores to them. Both paired response-gain point
estimates improved, but both still exceed the user's roughly 0.50bb **total**
preference. No model was promoted, no website policy changed, and no paid
compute was provisioned. No solver job remains running.

| Full-game restricted response | 400 rounds | 800 rounds | Observed decrease | 800-round approximate 99% lower bound |
| --- | ---: | ---: | ---: | ---: |
| Seed 26001, total bb/hand | 0.920429 | 0.805145 | 12.5% | 0.613489 |
| Seed 26002, total bb/hand | 0.874716 | 0.733220 | 16.2% | 0.536147 |

This responder can change **at most one decision per hand**, on any street.
Its expected gain lower-bounds the gain of a stronger best response. It is not
an exploitability upper-bound certificate. The point-estimate decreases agree
in direction across both seeds, but are **not a statistically established
400-to-800 improvement**: intervals overlap and paired per-deal covariance is
not exported. Do not extrapolate a convergence date from four measurements.

## Implemented improvements

1. **Live local resource stops.** Track process physical footprint on macOS
   (including compressed pages), or RSS plus swap on Linux. Sample memory,
   elapsed time and free disk every 250ms. Stop owned workers and queued peers
   on a breach; escalate SIGTERM to SIGKILL. These are sampled stops, not kernel
   caps or whole-machine memory guarantees. A regression also confirms that
   persistently unavailable disk telemetry stops the worker even when memory
   telemetry remains valid.
2. **Lossless checkpoint decoding.** The old decoder allocated repeated strings
   and trajectory/action slices for the entire table before deduplicating.
   The corrected reader interns each node immediately during decode. Numeric
   arrays remain independent, and the checkpoint schema and learning math are
   unchanged. Regression tests cover both supported compressed codecs.
3. **Pinned reader migration and exact recovery.** A different binary is allowed
   only by an explicit evaluation-only reader migration with identical settings,
   checkpoint digests and canonical output. Normal training resume still pins
   the binary. Evaluation stages can refer to immutable original checkpoint
   files instead of duplicating gigabytes.
4. **Tabular full-game response adapter.** `full-game-lbr --tabular-checkpoint`
   uses the saved average policy, loads its game/abstraction, rejects ambiguous
   sources and game overrides, and pins checkpoint SHA-256/rounds in the result.
   Tests cover hidden-card invariance on all streets, legal normalized mixes,
   exact action-path/realized-settlement parity and deterministic evaluation.
5. **Visible missing-policy coverage and units.** Unknown and untrained source
   lookups are counted by street and evaluation phase. The new total field sums
   signed seat gains. Legacy fields retain the historical half-sum convention;
   comparing those directly to a 0.50bb total goal would understate this result.

The diagnostic skill's reproduce-first loop identified and regression-tested
both checkpoint metadata expansion and the watchdog shutdown race. Those fixes
improve reliability, **not strategy by themselves**; the measured policy change
here is continuation of the selected terminal-integration estimator to 800.

## Memory and exact replay

The original 400-round checkpoint reader exceeded the 7.5GiB stop at
8,093,947,904 bytes before additional training. The corrected reader replayed
the same checkpoints at 3,955,756,200 / 3,720,760,320 bytes, with identical
canonical artifacts. This is a same-checkpoint comparison, not a compression
ratio estimate from different policy sizes.

| Seed | 800-round table size | Training/write peak | Full checkpoint replay peak | Replay + evaluation |
| --- | ---: | ---: | ---: | ---: |
| 26001 | 19,064,302 infosets | 6,128,194,216 bytes | 7,518,049,920 bytes | 98.447s |
| 26002 | 18,871,516 infosets | 6,121,362,088 bytes | 7,453,496,912 bytes | 101.421s |

The full-game response jobs peaked at 7,528,437,352 / 7,463,540,304 bytes and
took 97.246 / 92.507s at 800 rounds, including checkpoint loading. All four
response jobs completed below their 7.5GiB sampled stops. This establishes
bounded 800-round local feasibility, not safe unrestricted growth on 16GiB.
The host already had about 1.4GiB system swap before these runs.

Both first-attempt 800-round solver processes actually exited successfully and
wrote valid checkpoints. A transient zero-footprint reading as seed B exited,
followed by a process-group permission race, falsely marked its runner status
as resource-stopped. The corrected guard requires three consecutive telemetry
failures and tolerates owned-child shutdown races. Both checkpoints were then
reloaded at **800, not trained beyond 800**; regenerated artifact and unchanged
checkpoint SHA-256 values matched the preserved first attempt exactly.

That successful recovery is the required full-size replay. A subsequently
started duplicate replay was stopped as redundant. Its interrupted manifest
is historical evidence, not an outstanding validation action or lost training.

## Late-street sparsity is now visible

Source-policy unknown lookup fractions during independent response evaluation:

| Street | 400 A | 400 B | 800 A | 800 B |
| --- | ---: | ---: | ---: | ---: |
| Preflop | 0% | 0% | 0% | 0% |
| Flop | 43.30% | 44.56% | 28.93% | 25.79% |
| Turn | 95.94% | 96.15% | 92.05% | 89.20% |
| River | 99.77% | 99.72% | 99.23% | 99.15% |

Known but untrained lookups are **additional** to those unknown fractions.
At 800 they account for 8.68% / 8.74% on flop, 2.98% / 3.42% on turn, and
0.41% / 0.27% on river. Complete preflop lookup coverage does not prove
preflop equilibrium; its separate root deviation and stability checks fail.

The offline evaluator completes missing rows uniformly, matching the trainer's
explicit evaluation convention. These are not trained recommendations and are
never synthesized into serving artifacts. The coverage corpus includes both
baseline and response trajectories, so it is not a standalone authentic-policy
distribution or a 99.99% serving audit. It is nevertheless direct evidence that
the current table is highly incomplete on later streets. The aggregate
11.5–11.7% held-out unknown fraction hid the much worse late-street fractions.

Trajectory recall differentiates earlier bucket sequences and action histories;
millions of total infosets therefore do not mean repeated updates at most
late-street states. The response probe also has limited own lookup coverage,
so its measured gains can miss further leaks. Do not assign causal streetwise
exploitability contributions from lookup fractions alone.

## Other measured checks

| Diagnostic | 800-round result | Existing target / status |
| --- | ---: | --- |
| Root deviation A / B | 0.40709 / 0.38435bb | <=0.10bb; fails; not full-game exploitability |
| Maximum combo-weighted root action MAE | 8.28% | <=5%; fails |
| Root primary agreement | 64.50% | >=85%; fails |
| Maximum aggregate root action delta | 5.27% | <=3%; fails |
| Root classes / probability sums | 169 each; max sum error 3.33e-16 | Passes root checks only |
| Action-EV SE <=0.02bb, reach-weighted A / B | 23.41% / 21.53% | >=95%; fails |
| Full serving coverage / quantized full-policy size | Not audited | Not implied by checkpoint or root export size |

No gate was weakened. No fixed-responder confidence interval certifies true
exploitability from above. The total lower bounds in the first table combine
two approximate one-sided 99.5% seat bounds (Bonferroni 99% joint coverage),
not a full-game best-response search or a finite-sample upper certificate.

## Reproduction and handoff

All rounds use seeds 26001/26002 sequentially, 20bb, full default sizing,
trajectory recall, fixed DCFR 1.5/0/2, zero averaging delay, public-chance
sampling and exact terminal-action integration. Root/held-out/action-EV budgets
remain 256/2,000/2,000. The artifact export is **preflop-only**; checkpoints
contain all-street learning state. See JSON for exact commands and digests.

For each of the four frozen checkpoints, the matched response command is:

```bash
preflop-solver/neural/runs/local-pcs-20260904-solver-tabular-response \
  full-game-lbr --tabular-checkpoint <checkpoint.msgpack.gz> \
  --training-deals 20000 --calibration-deals 5000 --evaluation-deals 20000 \
  --rollouts-per-action 4 --minimum-range-particles 4 \
  --maximum-response-granularity strategic --seed <526001-or-526002> \
  --output <new-result.json>
```

Run under the resource guard described in the runbook; the bare native command
does not enforce those limits. Training, calibration and independent evaluation
streams are disjoint within each response job. The root seed is matched across
400/800 for a training seed. Both responders passed calibration in all four jobs.

Local evidence directories under ignored `preflop-solver/neural/runs/`:

- `local-pcs-20260904-checkpoint400`: original successful checkpoint pair.
- `local-pcs-20260904-checkpoint800`: historical old-reader memory stop.
- `local-pcs-20260904-reader-upgrade400`: exact reader migration.
- `local-pcs-20260904-streamed800`: completed continuation and successful replay;
  `run-manifest.attempt-1.json` preserves the shutdown-race attempt.
- `local-pcs-20260904-streamed-reload800`: interrupted redundant replay.
- `local-pcs-20260904-full-game-response`: complete four-policy response cohort.

Keep the frozen binaries/checkpoints for lineage; do not replace their manifests'
binary hashes with the latest build's hash. Future runs using another binary
need an explicit validation path, not edited provenance.

The local sequence is complete. The earlier conditional CFR-BR escalation for
**non-improving** response gains is not established by this pair: both point
estimates fell. However table growth and late-street sparsity remain real
limits. The next policy experiment should address repeated learning or
range-conditioned turn/river continuation on reached and perturbed ranges,
then compare full-hand action outcomes against this frozen baseline. Reuse the
existing turn/river public-belief solver rather than adding unrelated release
infrastructure. Such a continuation model is **not implemented or validated
by this audit**, and neither more local rounds nor paid compute is guaranteed
to resolve the remaining gates.

## Final verification

- `cargo test --release`: 193 library and 4 CLI tests passed.
- `python3 -m unittest test_cloud_blueprint_run test_worker_resources -v`:
  25 tests passed, including actual process stops and regression race cases.
- `cargo fmt --check` and `git diff --check` passed.
- Both 400-round reader migrations and both 800-round replay artifact/checkpoint
  comparisons matched exactly; all four response outputs matched recorded SHA-256.

This change touches native offline training/evaluation, not the browser solver
or WASM interface. No new browser behavior is claimed by these checks.
