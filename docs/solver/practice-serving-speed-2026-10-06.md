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
6. Reuse complete continuation predictions only when the ordered board, actor,
   investments and both exact range vectors match bit-for-bit. Each session
   owns its immutable model; both independent-evaluation paths replace the
   session when switching models. Retain at most 512 predictions / 48MiB of
   estimated cache allocations per model per solve (including both FIFO/map
   keys and vector capacities). In-flight work and model storage are additional.
   Misses and eviction run the original predictor, with computation outside
   the cache lock. No feature, matrix-multiplication, projection, regret update,
   chance weighting or average-policy operation is changed.

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

### Exact continuation prediction reuse

A temporary input-duplication probe motivated two small pilots. The first held
board/range features and query embeddings while recomputing public-context
values. Its nine outputs matched, but cold requests took
6.072s/7.867s/6.780s/6.912s and sampled peak footprint reached 479MiB. Adjacent
unchanged controls remained faster. That prototype was removed; its source and
receipt remain ignored as `rejected-query-value-inference.rs` and
`memo-candidate-1/manifest.json` (SHA-256
`d13de4e8493042a12debd934e2ea4fdaa3b57771916e3e110cdef3533940bcd2`).
Temporary diagnostic logging was removed from production.

The smaller complete-prediction cache retains exact results, not large hidden
embeddings, and leaves the original predictor intact. Two matched comparisons,
including a reverse-order repeat, produced:

| Cold request | Control A | Cache A | Control B | Cache B |
| --- | ---: | ---: | ---: | ---: |
| Limped flop | 5.479s | 5.084s | 5.963s | 4.962s |
| Raised flop | 6.611s | 6.460s | 7.039s | 6.389s |
| Mixed-batch flop | 5.795s | 5.026s | 6.017s | 5.028s |
| Cross-batch flop | 6.140s | 5.364s | 6.355s | 5.414s |
| Total | 24.025s | 21.934s | 25.374s | 21.793s |

All nine full responses matched exactly in both comparisons. This is an
incremental 8.7%/14.1% cold-speed gain on these fixtures, not a generalized p95
claim. Sampled peak footprints were approximately 319MiB/336MiB for pair A and
334MiB/320MiB for pair B; retained-cache accounting is not a hard process-memory
limit. Ready preflop requests remained below 1ms natively.

Receipts in the same ignored run directory:

- Control A: `memo-control-after/manifest.json`.
- Cache A: `full-memo-candidate-1/manifest.json`, SHA-256
  `2b3f82518b9ae93b87fca9817ef7c8b336b7de4dad22ab9b7cf99da32bfcdd08`.
- Control B: `full-memo-control-2/manifest.json`, SHA-256
  `37196b683c0400addbf57e60718e8989c1cf244e2dd78ad459ae16be6259af53`.
- Cache B: `full-memo-candidate-2/manifest.json`, SHA-256
  `d70130ae93cfb71c9aeb6a6408f5b39cc6c6525bae10d807133100cd629912bc`.

The final release binary was replayed against Cache A and again reproduced all
nine responses exactly. This unpaired final replay took
6.623s/8.059s/5.559s/7.006s for cold requests, with a sampled 332MiB peak and
0.335ms/1.399ms ready preflop responses. Retain this noisier observation too;
it is a parity check, not a third matched speed comparison.
`serving-memo-final/manifest.json` SHA-256:
`714d8fefb61c6e4907bf2d9bfc13fb78c9db1b63a1e5a8abbdb709bf09bd7a81`.

Cache tests exercise actual predictor results, exact reuse counts, a one-bit
range change, actor/investment/board changes, isolated models, eviction,
oversized bypass and simultaneous callers. The full convergence tests also
exercise replacing the continuation model after training and at checkpoints.

## Verification and next serving bottleneck

- `npm test`: 154 passed; the four opt-in native integration cases are skipped
  by default. Running them explicitly with `PRACTICE_RESOLVER_INTEGRATION=1`
  passes, including pinned flop policy and a trajectory through every street.
- Native `cargo test --release`: 371 library, seven transport and nine CLI tests
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
- The final cache build passes the four pinned-model HTTP integration cases
  and initializes a fresh browser hand with successful preflop requests and
  no captured errors. Completing the final browser action-flow recheck was
  interrupted by Chrome losing its debugging connection and requesting
  renewed approval. Earlier desktop/mobile checks above remain valid for
  their tested builds; they are not claimed as a completed final-cache check.
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

## Follow-up: profiling, whole continuations and shared card workers

The next pass follows the user's three priorities: profile redundant work,
cache exact continuations, and improve independent parallel execution. The
serving path already used eight card workers, not four.

A macOS `sample` capture on the frozen control found neural GEMM most prominent
among active stacks (2,987 collapsed samples), followed by feature construction
(818), scalar activation `expf` (783), exact turn equity initialization (610),
and immediate-strength equity (119). These are sampled stacks, not wall-time
percentages; idle dispatcher/reader/join waits are not computation. The capture
is ignored under `local-practice-speed-20261006-b/profile-control/cpu-sample.txt`,
SHA-256 `5cfa59cfd96d33a4b87639445e81922f1767574d8c1e8ad5b8ea75e796894b92`.

Implementation:

- Schema v3 now computes exact runout equity without first calculating the
  unused immediate-strength equity it replaces. Older schemas retain their
  existing immediate-equity features.
- A 128-entry / estimated 16MiB per-solve cache holds complete continuation
  vectors and their residual, keyed by ordered board, stack depth, execution
  order, actor, exact investments/bets/raise state, history/trajectory and both
  raw reach vectors. No normalization, rounding or probabilistic hash is used
  for identity. Mean continuation ignores only its genuinely irrelevant
  traverser argument; robust hypothesis selection retains it. Replacing the
  frozen model resets both this cache and the prediction sessions. Diagnostics
  still count requested evaluations and preserve their maximum residual.
- Independent turn-card jobs use a reusable, shared, bounded
  [Rayon](https://github.com/rayon-rs/rayon) pool (MIT/Apache-2.0). Concurrent
  same-budget flop solves share these card workers instead of each repeatedly
  launching another full set of OS threads. Results retain the previous
  strided-worker order before the unchanged serial floating-point fold; regret
  updates remain sequential. See the primary
  [pool](https://docs.rs/rayon/latest/rayon/struct.ThreadPool.html) and
  [indexed iterator](https://docs.rs/rayon/latest/rayon/iter/trait.IndexedParallelIterator.html)
  interfaces.
- The exact turn matrix LRU has 128 entries (about 215MiB of matrix payload at
  capacity), sufficient for the two active flop populations' maximum 98
  canonical boards. A small real-cache regression reproduces the former
  64-entry eviction and checks both complete 49-turn populations at the new
  capacity. Other model/cache/scratch allocations are additional.
- The Node launcher keeps eight as its default, permits explicit whole-number
  budgets up to 16, and caps/divides available cores across configured model
  processes. Invalid/fractional settings use the safe default rather than
  producing an invalid Rust argument.

The extended benchmark adds `--cold-pair serial|parallel`, using identical new
boards/queries in both modes and retaining exact probability/EV comparisons,
resource limits and completeness checks. Four cheap Python tests cover these
fixtures, unchanged model/budget arguments and worker validation.

Initial controls were noisier/slower than the preceding session: four workers
took 16.529s/22.222s/16.149s on the three common cold fixtures versus eight's
11.654s/11.199s/10.985s. Ten static workers did not establish a consistent gain
(11.402s/13.685s/9.462s/11.488s across the four cold fixtures). The cache-only
pilot retained all eleven exact responses but did not establish a clear
single-request improvement: total four-cold time 45.206s versus 44.250s, while
the independent pair completed in 23.273s versus 24.640s. No general SLA or
single-variable speed claim follows from this noisy first pair.

### Shared-worker matched results

The combined implementation (whole-continuation cache, dead feature work
removed, shared workers and 128-matrix retention) reproduced all eleven full
responses exactly in two comparisons. Pair B reverses the execution order:
candidate first, then frozen control. No build or test ran alongside the
latency pilots.

| Cold request | Control A | Shared A | Control B | Shared B |
| --- | ---: | ---: | ---: | ---: |
| Limped flop | 10.115s | 7.149s | 16.278s | 8.737s |
| Raised flop | 13.204s | 13.254s | 15.280s | 13.081s |
| Mixed-batch flop | 10.574s | 10.263s | 11.204s | 9.199s |
| Cross-batch flop | 10.357s | 9.973s | 11.688s | 11.087s |
| Four-cold total | 44.250s | 40.639s | 54.450s | 42.103s |
| Two independent cold flops, concurrently | 24.640s | 19.561s | 55.871s | 17.475s |
| Sampled peak footprint | 450MiB | 567MiB | 468MiB | 596MiB |

The first comparison's concurrent pair improved 20.6%; its four-cold total
improved 8.2%. The reverse comparison also favors the implementation, but the
55.871s baseline pair is unusually slow. Keep that observation rather than
present its much larger percentage as a general throughput forecast. These
are local fixture measurements, not an SLA or a full-hand strength evaluation.
The larger exact-matrix cache trades additional memory for avoiding eviction;
the measured peak remains below the pilot's 2GiB safety stop. All receipts
completed without a resource stop. Ready native preflop replies stayed below
1ms in both eight-worker comparisons, and repeated solved-flop queries below
0.3ms.

An additional ten-worker replay retained all eleven responses exactly. Its
four-cold times were 8.176s/13.385s/11.344s/9.498s and its concurrent pair
finished in 16.188s. This unpaired sample does not establish a consistent
single-request benefit over eight, so the website default remains eight;
higher budgets remain explicitly configurable and core-bounded.

Receipts live in the ignored `local-practice-speed-20261006-b` directory:

- Control A: `pair-control-eight-1/manifest.json`, SHA-256
  `c83532fba12d88f1d66c1a12c368b4c685592f3c9126637b829970cac2705bff`.
- Shared A: `pair-pooled-eight-1/manifest.json`, SHA-256
  `4dc5d99ec09b850ec5ffb15d0912772f688e05e5bec4b4f8141f1abfea5ee2b4`.
- Control B: `pair-control-eight-2/manifest.json`, SHA-256
  `01a2dd874a5504fe2c9d820679a43102d3e494e194f45652abb6f69be07b3bd8`.
- Shared B: `pair-pooled-eight-2/manifest.json`, SHA-256
  `f0b74bff733dd48e2815936cb67752e173256088520e2516edb775efba01fc7e`.
- Ten-worker replay: `pair-pooled-ten-1/manifest.json`, SHA-256
  `32d882e5a405f912c0a8edcc55b175f5172fc214d8994b11fd09463af329ec81`.

The default worker count was not reduced, nor were iterations, legal actions,
sampling rules, model weights or precision changed. The pool preserves the
control's floating-point fold order. Cache tests reproduce exact reuse,
one-bit-range misses and invalidation after changing the frozen model; worker
tests exercise eight actual workers, shared concurrent budgets, ordered
results and recovery after a task panic.

The final release binary (`f80f4bfd5dcdb3344979063790ed7e849b1c0d4ee4be3c7881232ee1254cf06a`)
also reproduced all eleven responses exactly. Its unpaired replay was noisier:
8.699s/15.044s/11.423s/11.405s for the four cold requests and 32.055s for the
concurrent pair, with a sampled 520MiB peak and no resource stop. This is a
final-binary parity check, not another matched speed win. Receipt:
`shared-workers-final/manifest.json`, SHA-256
`6fdf6f88ed3fac66fdbf30a1bc270af2426dc5a404e07694e1ae7e7688310f8b`.

### Follow-up verification

- `npm test`: 155 passed; four opt-in native integration tests are skipped by
  default.
- `cargo test --release --manifest-path preflop-solver/Cargo.toml`: 377 library,
  seven transport and nine CLI tests passed. The 59 explicitly gated long
  research tests remain intentionally ignored.
- From `preflop-solver/neural`, `../.venv-neural/bin/python -m unittest
  test_practice_latency`: four passed.
- `npm run build` passed, including verification of the unchanged deployed
  artifact identities.
- The live development server returned 200 for preflop and limped-flop POSTs
  with full policy/EV equality to the frozen control, and `/practice` retained
  its COOP/COEP headers. The old idle native child was restarted so this was
  the final release binary, not a stale process.
- Running those live checks alongside the separate native integration process
  produced unusually slow cold requests (about 86.6s) despite passing all four
  integration cases. Both independently launched model instances requested
  eight card workers; this is outside the shared pool within one native
  process. A later isolated live raised-flop POST completed in 6.023s, with a
  5.942ms repeated reply, both exactly matching the control. Background CPU
  competition was also visible. This shows why isolated latency pilots must
  not be treated as a multi-process load guarantee.
- Repeating `PRACTICE_RESOLVER_INTEGRATION=1 npx vitest run
  lib/server/practice-solver-process.integration.test.ts` in isolation passed
  all four cases in 16.690s: 5.189s for the pinned solved flop and 11.375s for
  the trajectory across flop, turn and river. The unusually slow concurrent
  observations are retained above, not used as a policy-quality result.
- Chrome remains running but the browser harness has no active debugging
  connection. Renewed remote-debugging/Accessibility approval has been
  requested; final visual/keyboard/console rechecking is still pending. The
  preceding milestones' browser checks above are not claimed for this build.
