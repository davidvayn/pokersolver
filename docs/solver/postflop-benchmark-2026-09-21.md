# Fresh postflop benchmark — September 21, 2026

Status: stopped at the user's request on September 21; scheduled follow-up paused.

No benchmark controller, native worker or benchmark idle-sleep assertion remains
active. Preserve the existing artifacts and wait for the user to request a
restart; do not resume automatically. The preflight's `failed` status records
the intentional resource-guard cancellation, not a policy-quality failure.
One complete seed/spot and 57 of 1,176 total turn packets are saved. Their mean
packet time was 70.23 seconds: extrapolating to two workers gives 11.47 hours
of packet evaluation from scratch, plus solve/audit overhead, or about 12 hours
overall. Roughly 11 hours remain by the same extrapolation. This is uncertain:
only the limped paired-board case has been timed, not every pot/texture class.

When explicitly restarted, the local command can chain preflight into the
complete suite if the resource projection passes, with macOS idle-sleep
prevention for its lifetime.
The `Finish poker strength benchmark` thread follow-up is now paused; its former
30-minute checks must not be re-enabled without the user's request. The computer
must remain on with the desktop app running for follow-up; closing a laptop lid
or shutting down can interrupt work.

## Question and scope

Measure the retained research solver across fresh postflop spots, using the
half-summed conditional best-response gain divided by the **starting flop pot**.
This is not a benchmark of the older website model and not a head-to-head test
against PokerCortex/Deepsolver: their strategy outputs and exact benchmark cases
are unavailable here. Their historical 0.6%-of-pot average is only descriptive
context, not a comparable score or promotion gate.

## Frozen design

- 20bb equal stacks, 0.5/1 blinds, no rake or ante; existing sizing abstraction.
- Twelve spots: limped (2bb), single-raised (5bb), three-bet (15bb), crossed with
  paired, monotone, connected two-tone and high-card rainbow textures.
- Each spot gets a distinct suit-isomorphism flop family, chosen deterministically
  with seed20260921 before observing results. Exclude the pinned retained and
  rejected value corpora, named development roots and consumed response boards.
  This exclusion audit does not claim to cover every historical artifact.
- Ranges come from the canonical frozen LCFR32 seed27001 preflop policy, conditioned
  on the legal action sequence and exact flop card removal; no range floor.
- Both retained ace-coverage value models (10601/10602), paired with solver seeds
  100101/100102. Fixed128 flop updates and64 native turn updates. No weight fitting,
  selection of a winning seed, adaptive iteration budget, or model deployment.
- Freeze each flop policy; materialize its native continuation policies across all
  49 legal turn cards. Integrate chance before maximizing earlier-street actions.
- Existing native response evaluator and independent JavaScript **flop accounting**
  audit. The latter shares native terminal equities and turn/river response packets;
  it is not a separate independent poker engine or an external Pio replication.

## Reporting

Per seed/spot: conditional response gain in bb and percentage of starting flop
pot, cold flop-solve wall time, hashes, complete chance coverage and audit status.
Aggregate: seed means; equally weighted spot means (average seeds within spots),
median, p90, worst spot, worst individual seed/spot, fraction below1% of pot and
pot-type breakdown. Do not count two seeds as two independent boards. Do not
claim a population99% bound from this small stratified suite. Cold flop solve
time excludes continuation materialization and full-policy evaluation; do not
compare it directly with a vendor's end-to-end solve latency.

## Resources and recovery

At most two native processes, one internal leaf worker each,1536MiB sampled
physical-memory guard per worker,20GiB disk reserve (24GiB initial headroom),
900s per solve/turn packet,180s per aggregation/audit, twelve-hour stage ceiling.
The first, wider limped spot is a cost-only preflight. Twelve times its elapsed
time must fit the stage ceiling before the suite starts. No spot is removed
because its result is poor. If capacity prevents completion, report incomplete
coverage and the blocker instead of a headline average.

The user subsequently authorized higher parallelism for a future restart, but
did not request restarting now. The current runner/protocol still use two
workers. Observed packet peak memory averaged 0.85GiB, with a 1.10GiB maximum;
the Mac has 16GiB RAM, four performance cores and six efficiency cores. Test
six workers, and optionally eight, with system-memory-pressure monitoring before
choosing the faster configuration. The provisional six-worker estimate of
4–6 hours is not measured throughput. Change concurrency through an explicit
resource-only protocol revision that preserves completed work and scientific
settings; never silently invalidate the existing source hashes or results.

Each job has an immutable completed receipt pinning command, environment and
output hashes. On restart, completed jobs replay without recomputation; partial
attempts and temporary receipts are preserved as interrupted evidence. Missing
or corrupt data never scores as zero. Stop any surviving owned workers before
restarting a controller; do not run two controllers against the same directory.

Artifacts: `preflop-solver/neural/runs/local-postflop-benchmark-20260921-c`.
Runner: `preflop-solver/neural/run_postflop_benchmark.py` (`prepare`, `preflight`,
`run`). Pinned protocol and sources are saved before the first policy solve.
No paid cloud compute, external account access, frontend change, or release
promotion is part of this benchmark.

The first attempt (`local-postflop-benchmark-20260921-a`) stopped before scoring:
native range normalization changed floating-point weights by at most
1.3010426069826053e-16, triggering an overly strict identity assertion. The
regression-tested fix tolerates at most 1e-14 absolute weight roundoff, requires
normalized finite nonnegative ranges and identical zero support, and compares
all other public fields exactly. The new protocol preserves the same spots,
models and iteration budgets; the failed artifact remains unscored. Attempt B
confirmed working solves and continuation jobs, about 62 seconds and 0.7–0.8GB
per initial packet. Before observing any complete spot score, its owned workers
were stopped to revise only the wall-time ceiling from six to twelve hours.
Attempt C preserves identical scientific settings and remains within two workers;
neither interrupted attempt contributes results. Protocol SHA-256:
`093296987e627c94c0da8ffe0fd90f9959109b9f0cf3be76840a9641675277e6`.

## Checks

- Six Python benchmark-contract tests passed, including normalization identity
  and rejection of meaningful state/range changes. Saved original failure replay
  now passes without changing the solver or input policy.
- Native export test verifies legal root lines, exact pots and zero flop bets.
- Rust release test binary compiled successfully; unchanged existing dead-code
  warning for `with_compact_continuation_and_turn_averages`.
- `cargo test --release --lib -- --test-threads=1`: 352 passed, 53 explicit
  artifact-dependent experiments ignored, zero failed (293.71 seconds).
- `git diff --check` passed.

References:
- https://poker-cortex.com/methodology
- https://deepsolver.com/blog/speed-precision-benchmarks-and-testing
- https://piosolver.com/docs/viewer/numbers_in_piosolver/
