# Lower-pot flop studies — 2026-11-07

This batch targets a different gap from adding board textures: the six existing
flop studies start with roughly one stack behind per pot. Smaller starting pots
permit a non-all-in pot raise and more betting before stacks are exhausted.
The same action grid, shared engine, exact card removal, ordered arrivals and
single jointly solved flop-to-river policy are retained. No existing study,
universal model, numerical gate or serving limit is replaced or relaxed.

Starting ranges are deliberately narrow, disclosed study assumptions, not
certified frequencies from an earlier betting line. A conditional 20bb study
does not certify or replace the existing experimental full-hand 20bb model.

## Structural preflight

| Study | Board | Depth / pot / behind (bb) | BTN / BB combos | Compatible deals | Ordered deal–runouts |
| --- | --- | --- | --- | --- | --- |
| Deeper low rainbow | 8s 5d 2c | 50 / 12 / 44 | 13 / 10 | 109 | 215,820 |
| Low hearts | 6h 5h 3c | 20 / 6 / 17 | 13 / 10 | 112 | 221,760 |

The 50bb ranges are BTN AA/A8s/76s versus BB 55/87s/A4s. The 20bb ranges are
BTN AA/A6s/87s versus BB 55/64s/A4s. Each exact unblocked combination starts
equally weighted within its seat's declared support; private deals condition
the joint product on compatibility.

The native census checks the larger games before training:

| Study | Public decisions | All public nodes | Regret/average bytes | Maximum action depth |
| --- | ---: | ---: | ---: | ---: |
| 50bb deeper | 412,300 | 1,059,468 | 169,248,352 | 10 |
| 20bb low hearts | 374,668 | 946,572 | 150,153,568 | 9 |

Earlier flop games contain 104,872 decisions and allow at most eight actions
along a full line. These candidates therefore expand betting-history coverage,
not just board labels. Both remain below the unchanged 500,000-decision,
2,000,000-total-node and 512MiB accumulator bounds. No public branch or low-reach
combination is dropped to fit those bounds.

## Short pilots and training decision

| 128-update pilot | Summed best-response gain (bb/spot) | Maximum EV error (bb) | Available action values checked |
| --- | --- | --- | --- |
| [50bb deeper](reports/2026-11-07-flop-50bb-deeper-128.json) | 0.1429871463 | 0.000001907349 | 10,227,514 |
| [20bb low hearts](reports/2026-11-07-flop-20bb-low-hearts-128.json) | 0.004679385262 | 0.0000009536743 | 7,419,958 |

The unchanged acceptance gates are ≤0.05bb **summed** unilateral response gain
and ≤0.0001bb maximum available-action EV discrepancy. The 20bb pilot already
passes with a wide margin, so its candidate remains at 128 updates. Only the
50bb game needs the 512-update continuation of this experiment (a fresh full
solve, not an exported/resumable regret checkpoint). The short reports themselves
are research data, not serving policies. Zero-opponent-reach rows remain ungraded.

Concurrent pilot wall times were 81.75s / 70.93s, with about 540MB / 482MB maximum
RSS respectively. These are local contended observations, not universal solving
time estimates. No paid compute is used.

## Pot-building regression

The independent TypeScript suite now explicitly checks the half-pot bet →
non-all-in pot raise → call path where stacks allow it. If the starting pot is
P and initial depth D, that raise targets 2.5P; remaining chips are D − 0.5P,
so it is non-all-in exactly when D > 3P. After the call the pot is 6P and both
players have D − 3P behind. Thus the 50bb case reaches a 72bb turn pot with 14bb
behind, and the 20bb case a 36bb turn pot with 2bb behind. Check-down through
later streets must still settle correctly with exact chip conservation.

Every installed flop also retains complete independent EV/response checks,
all-public-branch traversal, exact joint-prior interval tests, pinned loading
and failed-request retry tests. Full reproduction remains `npm run flops:build`,
`npm test` and `npm run build` after artifact activation.

## Accepted exports

| Study | Updates | Summed response gain (bb/spot) | Maximum EV error (bb) | Binary bytes | Manifest bytes | Shards |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| 20bb low hearts | 128 | 0.004679385262 | 0.0000009536743 | 146,430,457 | 881,649 | 7,396 |
| 50bb deeper | 512 | 0.005879573188 | 0.000001907349 | 168,489,025 | 987,142 | 8,278 |

Both passed fresh native re-import of the actual served bytes. The 20bb frozen
policy and action-EV fingerprints exactly match its short pilot: it was not
trained longer after already passing. The 50bb response gain fell about 95.9%
from the same game's 128-update pilot. These are conditional subgame results,
not improvements to a universal policy or comparisons between different priors.

The largest shards are 22,698 / 22,797 bytes respectively; neither shard size
nor the 2MB manifest cap was raised. Concurrent exports took 94.46s / 324.86s
wall, including fresh re-import verification, with about 771MB / 1,251MB maximum
RSS. They stayed inside the existing ten-minute per-export build timeout. These
are local contended measurements, not clean hardware throughput benchmarks.

Immutable manifest identities:

- `flop-20bb-low-hearts`:
  `ddfaa7a7a0fabbe3f07506fd2c3351e9f165d57a6978793fd7b040af3ae6d8ea`.
- `flop-50bb-deeper`:
  `f9bd1122f58ce266835bd02bf66e50b987d0b075e94b62c83e532257e2ae881b`.

The library now has eight complete conditional flop games and the unchanged
eight river/turn games. Its 27,432 saved-strategy files total 598,865,922
uncompressed bytes. A hand still loads only its pinned manifest and required
shards, with the existing 16-shard cache. Larger public trees have a real storage
and manifest cost; this is not evidence that unrestricted Hold'em tables fit
the same budget.

### Deployment scope

The expanded source corpus exceeds Vercel's documented 15,000-file **CLI source
upload** limit. Its documentation distinguishes that from build outputs, which
have no stated file-count ceiling. Use the Git-connected build path for this
checkout, not a plain CLI source upload; this checkpoint does not certify a live
deployment. No packaging, storage service or paid infrastructure was provisioned.
See [Vercel's file and upload limits](https://vercel.com/docs/limits#files).

## Final checks

- `npm test`: 403 passed, 20 optional integration cases skipped. Every installed
  flop received full independent action-EV/response, all-public-branch,
  joint-prior and pot-raise checks. The larger cases fit the existing test time
  and memory settings; no test coverage or numerical tolerance was reduced.
- Saved-spot Rust release tests: all 17 passed. No Rust solver algorithm,
  generated WASM, runtime schema or universal model source changed.
- `npm run build`: passed with fresh native verification of all 16 published
  studies and unchanged full-hand artifact integrity checks.

Browser Use played both additions from both seats, including all three streets,
non-all-in raises, calls, all-in runouts and terminal settlement. The 20bb flow
included Bet 3bb → opponent Raise to 15bb → Call 12bb, reaching a 36bb turn pot
with 2bb behind each. A 50bb flow exposed the hero's Raise to 30bb button on the
turn; the opponent called and play continued onto the river. Opponent cards
stayed hidden until terminal review; feedback did not reveal mixes before acting.

A blocked 50bb shard at the flop-to-turn boundary immediately displayed the
fourth board card and both public checks, but no EV grade. Retry preserved that
board, hero cards and action history and produced exactly one feedback record.
The rest of the hand completed normally. Initial loading fetched just each
pinned manifest and root shard, not the thousands of descendants. No captured
unhandled JavaScript exception or practice-resolver API request occurred.

Layout checks passed at 375/768/1024/1440px in light/dark with no horizontal
overflow. Mobile controls remained 48px tall; keyboard actions, visible feedback
focus and reduced-motion playback worked. Native macOS selects used the existing
DOM change-event workaround, so native select-keyboard behavior is not newly
certified. Both manifests retained immutable caching and COOP/COEP headers.
These checks certify only these declared saved games, not arbitrary starting
ranges, a solved preflop line, or full-hand Hold'em exploitability.
