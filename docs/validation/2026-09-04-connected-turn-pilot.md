# Connected turn/river pilot and shared-table CPU workers

The full-hand pair is complete. Its result is **inconclusive for selecting a
full-hand policy winner**, not a release or Approximate GTO certificate. The
bounded two-seed protection probe also completed successfully.
The website model and original training checkpoints were not changed. No paid
compute was used.

## Full-hand result

Both controls and candidates use frozen 800-round 20bb tabular checkpoints.
The candidate preserves preflop/flop and jointly resolves turn and river with
four iterations. Each candidate gets newly trained responses: 2,000 training,
5,000 calibration, and 5,000 held-out hands per seat, four rollouts per action,
and at least four particles per learned information set. Training, calibration,
and held-out streams are disjoint; evaluation seed offset is 900000.

The BB response calibrated in both seeds. Values below are its held-out gains
in bb/hand; ± denotes **one standard error**, not a confidence interval.

| Seed | Control BB gain | Candidate BB gain | Candidate SB response |
| --- | ---: | ---: | --- |
| 26001 | 0.194983 ± 0.059900 | 0.176167 ± 0.058705 | Failed calibration |
| 26002 | 0.218866 ± 0.058162 | 0.159300 ± 0.047634 | Failed calibration |

The BB direction is encouraging, but the differences are small relative to the
marginal sampling errors. A paired difference confidence interval is not
available from these aggregate reports; do not assume independent errors for
evaluations using matched deal streams.

The SB calibration estimates were 0.0241664 ± 0.0291192 and
0.1152998 ± 0.0493699bb. Their one-sided 99.5% lower bounds were -0.0508398
and -0.0118687bb, so neither response was deployed for its held-out score.
The resulting zero SB gains are **not evidence of zero exploitability**.

Consequently, comparing control totals 0.441883 / 0.4805994 against candidate
deployed totals 0.1761668 / 0.1592998 would be misleading: the controls include
two calibrated seats and the candidates only one. This pair cannot qualify
the conditional neural-distillation or longer-policy-training stages.

These are restricted, one-step, legal imperfect-information response probes.
They do not provide a full-game exploitability upper bound. This 2,000-hand
training budget also cannot be compared directly with the older 20,000-hand
audit's 0.8051452 / 0.7332198 totals.

## Implemented and verified

- A connected exact-combo turn/river policy, including every compatible river
  chance outcome. Entry ranges replay each player's own observed actions and
  remove only revealed board cards. Opponent hole cards and future river cards
  are not inputs to the public solve.
- A bilateral opponent-CFV-protected variant. Each seat replaces its own
  complete continuation; the gadget adversary is never used as the served
  opponent policy. Exact-zero entry hands preserve original anchor rows so the
  second seat's all-opponent-hands protection has a complete boundary.
- An inference-only streaming checkpoint reader that retains exact average
  accumulators without allocating regrets/resumable training state. Both old
  complete response reports were reproduced byte for byte with this reader.
- Correct paired action-gap uncertainty for common-random rollouts. The
  previous independent-variance calculation could mark an exactly constant
  action advantage as uncertain. Regression checks cover common and
  anticorrelated noise; separate calibration requirements were not relaxed.
- Exact terminal-work reduction: alternating training and best-response passes
  calculate only the value vector they consume. Differential tests preserve
  regrets, averages, policies, response values, and protection calculations.
- `--response-workers 2`: CPU-parallel Rust workers share one immutable table
  and own separate solve caches. Hands and floating-point observations are
  reduced in original order. Training, calibration, and held-out evaluation all
  support this path. One through four workers have deterministic unit coverage;
  only two have full-size parallel memory/throughput validation here.

The protection design follows the distinction between range-only solving and
opponent-value-protected subgames in [Safe and Nested Subgame
Solving](https://arxiv.org/html/1705.02955v3). A future continuation model must
retain the strategy-dependent nature of poker values described in
[Depth-Limited Solving](https://arxiv.org/html/1805.08195v1), not just fit a
strategy-independent state scalar. Our finite-iteration 0.05bb local projection
tolerance is not an exact no-regression theorem or a full-game certificate.

## Measured cost and memory

| Measurement | Result |
| --- | --- |
| Matched connected probe, one worker | 110.333s; 6,131,143,360-byte peak |
| Identical probe, two workers | 95.814s; 6,306,599,736-byte peak |
| Total probe wall-time reduction | 13.2%, including serial checkpoint loading |
| Full candidate seed A, original serial build | 4,334.757s; 6,220,731,144-byte peak |
| Full candidate seed B, optimized two-worker build | 2,622.295s; 6,721,262,728-byte peak |

The short cost pair is a single matched measurement, not a universal speedup
claim. Different seed policies and deal streams prevent attributing the entire
A-versus-B time difference to parallelism. Peak memory is sampled macOS physical
footprint, including compressed memory; the 7.5GiB stop is not a hard kernel cap.
The runner also reserves 20GiB disk and starts no concurrent full-table process.

Separate order-reversed terminal benchmarks reduced root runtime by 23.9–26.6%
with byte-identical root-value/response outputs. Those speedups are distinct
from the shared-worker cost pair and must not be added together.

## Protected continuation probe

Both seeds completed a bounded two-iteration probe: four training, two
calibration, and eight evaluation hands per seat, two rollouts per action,
seed offset 700000, and one worker. These deliberately tiny budgets test the
connection, local protection, and resource use, **not policy quality**.

| Seed | Solved roots | Maximum opponent-CFV excess | Minimum proposal weight | Wall time | Peak footprint |
| --- | ---: | ---: | ---: | ---: | ---: |
| 26001 | 4 | 0.048828125bb | 0.015625 | 133.606s | 6,279,123,792 bytes |
| 26002 | 11 | 0.049509162bb | 0.015625 | 146.107s | 6,116,774,568 bytes |

All 15 roots completed within the configured 0.05bb local protection tolerance
and resource limits. At the most constrained replacement stages, only 1.5625%
of the proposal weight survived protection. This illustrates how conservative
the low-iteration protected replacement can be; it does not demonstrate a
full-hand improvement. Both tiny probes rejected both responders. Their zero
deployed gains are not meaningful quality measurements.

## Verification and provenance

- Rust release suite: 201 library and 5 CLI tests passed.
- Runner/resource Python suite: 30 tests passed.
- `cargo check --tests`, `cargo fmt --check`, and `git diff --check` passed.
- Both full-size baseline reports match the frozen controls after normalizing
  only worker count. Canonical JSON comparison retains signed-zero distinctions.
- The connected full-size serial/two-worker probe matches all policy, payoff,
  confidence, and source-coverage fields. Only worker count and cache-work/time
  diagnostics are excluded from comparison.
- Model source, compiled binary, and result digests are pinned in the cohort
  records. Seed A finished before the owned runner was stopped; its result was
  preserved. Seed B then used the verified performance-only build. Both main
  candidate results completed without a resource stop.

Optimized executable SHA-256:
`cdbabf150044c292325392b19e8e263c8bd8fbb61f2ff760bb9f1cc4742956f0`.

Candidate result SHA-256:

- Seed A: `e711ce7c89325c82a56809e6fd810b5adc6196a7632f296254cff1fc630a6a42`.
- Seed B: `3b58ec408806b8837eb9687eee02107db0647e2f34956d5933d02f5a04f10a29`.

No website/WASM interface changed, so this work did not require regenerating
browser assets or rerunning browser acceptance. Unrelated HTML work and
`blueprint-artifact.json` were left untouched. No commit or push was performed
in this sequence.

## Decision boundary

The connected resolver, memory fixes, and deterministic parallel path are
implemented. A full-hand policy-quality win is **not established**. No new
continuation neural weights, longer policy run, or model promotion was triggered.
The protected path still has no meaningful full-hand quality comparison.

Do not repeat this four-iteration pair unchanged or describe rejected-response
zeros as progress. A subsequent experiment needs better adversarial coverage
and evidence about earlier-street actions. The frozen flop policy is a relevant
next hypothesis because this pilot does not update it; the current reports do
not isolate how much gain originates on each street.

Detailed implementation and the earlier failed/cost-screened probes are in the
[runbook](../solver/tabular-turn-pilot.md). The
[machine-readable audit](2026-09-04-connected-turn-pilot.json) records source,
binary, and result digests, measured results, and the untriggered conditional
stages. All sequence-owned solver processes have completed.
