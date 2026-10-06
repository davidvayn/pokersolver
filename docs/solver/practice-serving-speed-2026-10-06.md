# Interactive practice solving speed

The user reprioritized solving speed for usability on October 6. The unfinished
native32/native64 strength cohort was stopped, not scored or promoted. This
change optimizes the website's existing experimental 20bb resolver: two
iterations per street, the same four artifacts and the same action abstraction.
No exploitability improvement or new equilibrium claim follows from it.

## Implemented

1. Reuse completed exact turn-equity matrices for flop all-in equity. Every
   compatible pair has 45 turns and 44 rivers. Summing the integer turn counts
   counts each unordered runout twice, so division by 3,960 reproduces the
   original division by 1,980 bit-for-bit. Keep every physical turn and its
   suit-to-combo mapping; do not collapse chance weights. If any matrix is
   missing/uninitialized, use the original exact enumeration. Sparse-only
   solves retain their original path and do not force dense cache population.
2. Cache the 24 immutable suit/combo permutations. Compact exact-turn range
   dot products by omitting only exactly zero weights, preserving the original
   summation order. No tiny positive weight is pruned or quantized. Avoid
   eagerly evaluating board/river-blocked hands during turn matrix creation.
3. Opt the Node process client into `streamResults: true`. The bounded native
   microbatch flushes each completed query immediately, identified by its own
   request ID. A ready decision no longer waits for its slow speculative
   sibling. Omitted/false keeps the existing aggregate CLI envelope and input
   order. Panics retain the failing request ID rather than failing every client
   with an unidentified response. Streaming does not eliminate cold solve cost;
   cross-batch scheduling is addressed by the next change below.
4. Add a cheap, guarded native serving benchmark. It reads the actual pinned
   manifest, preserves full probability/EV outputs, supports exact baseline
   comparison, and measures cold, cached and mixed-speed batch requests.
   The worker has a 120-second cap, 2GiB sampled-memory cap and 20GiB disk
   reserve. Failed/incomplete/resource-stopped receipts are not valid results.
5. Separate NDJSON transport scheduling from poker computation. A reader feeds
   a bounded dispatcher; two workers handle postflop computation with at most
   eight waiting jobs. Preflop uses the same frozen lookup/EV-table adapter
   immediately instead of queuing behind active postflop work. Responses retain
   their request IDs, default batches retain input ordering, accepted jobs drain
   on EOF, and saturation produces an explicit retry error, never fake policy.
   The poker engine, solve budget and per-solve card worker count are unchanged.

## Local evidence and limitations

Initial adjacent, unprofiled runs on this 16GiB Apple M5 at eight workers:

| Native request | Original | Optimized |
| --- | ---: | ---: |
| Cold limped flop | 6.057s | 5.472s |
| Cold raised flop | 7.471s | 6.654s |
| Cold third flop in mixed batch | 6.482s | 5.709s |
| Ready preflop sibling in that batch | 6.482s | 0.000367s |

All seven complete responses match exactly, including probabilities, EVs,
confidence and artifact hashes. Warm native flop lookups remain below 0.1ms;
HTTP/browser timings also include transport, rendering and dev compilation.
Observed worker peaks for this pair were approximately 303MiB/287MiB.

These are small local latency fixtures, not a p95 service SLA. Later rechecks
were noisy: the unchanged original binary itself rose to roughly 14s on cold
flops. Do not generalize the initial roughly 10–12% cold-speed gain to every
machine or silently exclude the slower observations. The ready-sibling benefit
remained present in every streaming run, including the noisy ones. A cold
decision can still take several seconds or longer under system contention.

The final static-worker build was replayed after removing the scheduling
prototype: 5.629s/6.554s/5.797s for the three cold flops and 0.000317s for the
ready batch sibling, again with all seven responses exactly equal to the
original. Receipt `serving-final/manifest.json` SHA-256:
`dd5196d87c684da0857a40f7b702cb5b6957362e3d14a2ce270b254f5f61adca`.

Receipts are ignored under
`preflop-solver/neural/runs/local-practice-speed-20261006-a/`:

- `serving-baseline-1/manifest.json` SHA-256:
  `a699307ea9adb02545f268e13723a66b31630cf603ece04bd21b3428338b1b67`
- `serving-optimized-1/manifest.json` SHA-256:
  `565e150c601caec7080d83e7c4c363a0c25e627f6a30f5ceaeeea0cf2198cbbd`
- Noisy unchanged control: `serving-baseline-2/manifest.json` SHA-256:
  `8b227bd74508eb9268292981f5d888f6ba66dbffa79d9c5d2d6c4789780824fa`

Four workers were slower than eight on the tested configuration. A dynamically
balanced card-leaf prototype preserved output parity but did not establish a
consistent incremental speed benefit; it was removed from production. An
isolated MIT-licensed [gemm](https://github.com/sarah-quinones/gemm) kernel
comparison did not justify replacing the existing math backend either. No new
Rust dependency, GPU service, paid compute or policy retraining was added.

Reproduce in fresh output directories with no competing training/build job:

```sh
preflop-solver/.venv-neural/bin/python \
  preflop-solver/neural/benchmark_practice_latency.py \
  --binary /absolute/path/to/frozen-original-binary \
  --output preflop-solver/neural/runs/practice-latency-original

preflop-solver/.venv-neural/bin/python \
  preflop-solver/neural/benchmark_practice_latency.py \
  --baseline preflop-solver/neural/runs/practice-latency-original \
  --output preflop-solver/neural/runs/practice-latency-current
```

The baseline binary must be preserved before rebuilding. Repeating a cached
request is not a cold benchmark. Keep power conditions/system load comparable
and retain all receipts rather than selecting the fastest run.

### Cross-batch head-of-line delay

The first streaming milestone still processed stdin batches serially. A second
matched test sent a cold flop in one batch, then a ready preflop in a separate
batch on the same loaded engine. The bounded asynchronous transport produced:

| Native request | Serial streaming transport | Asynchronous transport |
| --- | ---: | ---: |
| Earlier cold flop | 6.421s | 6.091s |
| Later ready preflop | 6.422s | 0.000306s |

All nine outputs in the expanded fixture matched exactly, including EVs and
confidence. Other cold requests took 5.523s/6.693s/5.841s in the asynchronous
run: this is evidence of eliminating avoidable queuing, not another established
cold-computation improvement. Two heavy solves may contend for CPU; neither
this pair nor the earlier fixture establishes a multi-user throughput SLA.
Cached postflop queries still use the bounded postflop lanes.

Append `--cross-batch` to both benchmark commands to reproduce. Receipts:

- `cross-batch-serial/manifest.json` SHA-256:
  `301a09155bf7513fcd4c5d00a5d8e220e39ca35309d0fe8c65cfd4f2d9685db8`
- `cross-batch-async/manifest.json` SHA-256:
  `4d936b072d37ea023a8492533ba6493f0db68cb1de9cb0752ecc52da3156d39b`

The transport regression first failed against the serial implementation, then
passed with two deliberately blocked postflop callbacks and a later preflop
request over a real NDJSON stream. Further gated tests cover completion order,
legacy batches, identified panics, malformed input, output failure, EOF draining,
and the two-active/eight-queued limit.

## Verification and next serving bottleneck

- `npm test`: 154 passed; the four opt-in native integration cases are skipped
  by default. Running them explicitly with `PRACTICE_RESOLVER_INTEGRATION=1`
  passes, including pinned flop policy and a trajectory through every street.
- Native `cargo test --release`: 367 library, seven transport and nine CLI tests
  passed; 59 explicit long research tests remain intentionally ignored.
- `npm run build` passes and verifies the unchanged model artifact identities.
- A real browser completed a full hand through an all-in runout and terminal
  review. Keyboard Tab/Enter showed visible focus on the action controls.
  Desktop/mobile layouts and 768/1024px overflow checks passed; reduced-motion
  emulation works. Captured practice requests returned 200, with no captured
  page error or unhandled rejection. Mobile next-hand initialization retained
  the table and alternated the hero to BB.
- After replacing serial batch scheduling, the rebuilt server also handled a
  fresh browser hand through a preflop raise/call line, a solved flop, and a
  mobile flop bet. All captured requests returned 200, with no captured errors
  or horizontal overflow at 375px. The accepted turn/river integration replay
  was rerun against this binary, not a stale server process.
- The observed hero response to one flop all-in still has unavailable EVs:
  the existing model shows the frequency grade and explicitly declines EV
  grading. This is not fixed or fabricated by the speed change; the integration
  trajectory is not proof that every possible decision has an EV estimate.

Browser tracing exposed the cross-batch bottleneck addressed above: one BB
hand's speculative flop branches took approximately 5.86s and 11.28s with the
serial streaming transport. Later cheap preflop lookups now bypass that queue.
The remaining speed target is cold postflop computation and the scheduling of
actual decisions versus speculative branches, not additional training rounds
or silently reducing solve quality.
