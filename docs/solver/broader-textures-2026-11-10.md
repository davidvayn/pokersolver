# Broader monotone and two-tone flops — 2026-11-10

This batch adds a 20bb king-high monotone game and a 50bb jack-high two-tone
game. It expands declared range support, including weighted offsuit broadways,
without raising the existing 32-combo-per-seat limit. These are illustrative
conditional starting priors, not certified preflop ranges or universal stack
policies. The earlier 19 studies and active full-hand artifacts stay unchanged.

## Games and acceptance

- `flop-20bb-monotone`: Kh Th 4h, 20bb initial depth, 12bb pot, 14bb behind.
  BTN AA/KQs/AJs/QJs/99/65s versus BB TT/44/KJs/AQs/Q9s/88/76s, with equal
  inclusion weight per unblocked exact combination. The game includes made
  flushes, sets, pairs, straight draws and single-heart blockers.
- `flop-50bb-two-tone`: Jd 9d 5s, 50bb initial depth, 32bb pot, 34bb behind.
  BTN JJ ×1, KJs ×1, QTs ×0.75, TT ×0.75, 87s ×0.5, AQo ×0.5 versus BB 99 ×1,
  55 ×1, JTs ×1, T8s ×0.75, A8s ×0.75, 88 ×0.5, QJs ×0.5, 76s ×0.5. Inclusion
  weights are not action frequencies. Public blockers are removed before
  normalization; private pairs use the exact compatible joint prior.

Each policy jointly solves all three streets using exact card removal, ordered
card arrivals and complete betting histories. Existing half-pot/pot/all-in bets,
one pot-sized raise plus all-in, and the one-subsequent-raise cap are unchanged.
There is no neural continuation, bucketing, zero-reach pruning or online solve.

An allocation-free census precedes 128-update pilots. Only a candidate passing
≤0.05bb summed unilateral best-response gain, ≤0.0001bb maximum available-action
EV discrepancy, valid frozen probabilities and fresh binary re-import checks
may enter the catalog. Missing EVs at zero compatible opponent reach remain
unavailable and ungraded. Rejected pilots produce reports, not serving policies.
Longer candidates retain the identical declared game and all existing limits.

Validation is scoped to each new study's six exhaustive TypeScript checks,
the production build with its normal complete artifact verification, and
browser-harness QA in isolated, muted local headless Chrome. The full CI suite
is not rerun merely to commit. CI remains PR/manual-only; deployment uses the
Git-connected build. No database, paid compute, credentials or serving-schema
change is needed. Task-scoped idle-sleep holds around active native runs leave
permanent Mac power settings unchanged.

## Structural preflight

| Study | BTN / BB combos | Compatible private deals | Ordered deal–runouts | Exact accumulator bytes |
| --- | --- | ---: | ---: | ---: |
| 20bb monotone | 27 / 27 | 674 | 1,334,520 | 90,975,744 |
| 50bb two-tone | 32 / 30 | 866 | 1,714,680 | 104,453,632 |

Each tree contains 104,872 public decisions and 250,374 total nodes including
chance/terminals. Counts stay within the unchanged structural and memory caps;
all supported combinations are retained, even after zero-reach actions. The
census's legacy monolithic-reader warning does not apply to the installed
bounded, hash-pinned flop-shard reader. Neither census measures strategy quality.

## Short pilots

| 128-update pilot | Summed response gain (bb/spot) | Maximum EV error (bb) | Available action values checked | Result |
| --- | --- | --- | ---: | --- |
| [20bb monotone](reports/2026-11-10-flop-20bb-monotone-128.json) | 0.02939061350 | 0.0000009536743 | 5,324,184 | Passed |
| [50bb two-tone](reports/2026-11-10-flop-50bb-two-tone-128.json) | 0.06375616016 | 0.0000019073485 | 5,475,444 | Rejected; no serving artifact |

The accepted 20bb candidate keeps 128 updates. The 50bb candidate doubles to
256 updates first, rather than spending four times the training on a game only
slightly above the unchanged gate. It must still pass independently before
publication. Reports embed their original inputs, so the shorter pilots remain
reproducible after the source's iteration count advances.

The pilots took 74.52s / 93.12s wall with about 327MB / 371MB maximum RSS while
running concurrently. Independent scalar/vector profile discrepancies were
≤6.67e-16bb. Respectively 180,900 / 526,008 rows had no compatible opponent reach
and remain unavailable and ungraded. These are contended local observations,
not clean hardware-throughput benchmarks or full-hand exploitability results.

## Accepted exports

The 20bb 128-update export passed fresh native re-import of all binary bytes.
Its policy and EV fingerprints exactly match the passing pilot; response gain
remains 0.0293906135011bb and maximum EV discrepancy 0.000000953674316bb.
Its 1,957 binary shards total 69,555,279 bytes, manifest 235,661 bytes, and
largest shard 40,802 bytes, within the unchanged 65,536-byte hard limit.
Broader range support exceeds the usual 32KB target for some shards; the fixed
128-node chunk layout and numerical precision were not changed to hide this.

20bb manifest SHA-256:
`608a4bd92fdbe1b7ca6f9fe5253834014e13e298cd82b69f6f315599d3e6ce18`.
Export plus fresh re-import took 86.69s wall with about 493MB maximum RSS while
the independent 50bb candidate ran. This is a contended local observation.

The 50bb 256-update export passed at **0.0172732761607bb** summed response gain,
about 72.9% below its own 128-update pilot, with maximum EV discrepancy
0.00000190734864bb. This improvement is within the identical conditional game,
not a comparison between different priors or a full-hand exploitability result.
The 1,957 binary shards total 74,485,407 bytes, the manifest is 236,141 bytes,
and the largest shard is 45,482 bytes, within the same hard limit.

50bb manifest SHA-256:
`fd6ae70fbc0c4567e77c2e9b2d3f24c391c1c738e860012ec4a8accc1b5eeccc`.
Export plus fresh re-import took 182.05s wall and about 540MB maximum RSS.
Only frozen probabilities and checked EVs are served, not regrets or resumable
training state. Neither solver code nor numerical/validation tolerances changed.

The complete library now has 21 studies: 13 flop-to-river games, four
turn-to-river games and four river games. Its 43,543 serving files total
1,037,270,975 uncompressed bytes. The earlier 19 studies and active full-hand
artifacts remain unchanged. A hand still lazily loads only its pinned manifest
and needed shards using the existing 16-shard cache; the complete corpus is
not loaded into the browser. Deployments retain the Git-connected build path,
not a plain CLI source upload.

## Focused verification

- `npx vitest run lib/solved-flop.test.ts -t flop-20bb-monotone`: six passed,
  66 unrelated cases skipped. Its independent exhaustive EV/response check
  took about 48.4 seconds.
- `npx vitest run lib/solved-flop.test.ts -t flop-50bb-two-tone`: six passed,
  72 unrelated cases skipped. Its exhaustive EV/response check took about
  50.3 seconds. Both studies also passed every-public-branch, exact joint-prior
  interval, bounded binary decoding, loading/retry and pot-raise legality tests.
- `npm run build`: passed, including fresh native verification of all 21 saved
  studies, integrity checks for the unchanged full-hand artifacts, TypeScript
  validation and Next.js production compilation. The full CI suite was not run.
- Both new manifest URLs retained immutable one-year caching and COOP/COEP.
  No serving contract, policy format or action grid was modified.

Browser-harness used a fresh local Chrome profile launched with
`--headless=new --mute-audio` and loopback-only CDP. No personal browser profile,
signed-in session or paid browser was used. Both games were exercised from both
seats. The 20bb BB flow called a 6bb turn bet and reached river showdown; the
BTN flow checked through all three streets. The 50bb BTN flow checked through
all streets to showdown, and the BB flow bet 16bb and settled an opponent fold.
Opponent cards stayed hidden until terminal review, with mixes/EV loss hidden
before the first user decision. The weighted details panel displayed every
declared multiplier, distinguished inclusion from action probability, and
showed the pinned manifest identity.

Blocking a real 50bb shard request at the flop-to-turn transition paused play
without grading. The dealt turn and exactly two checks remained in place.
Retry preserved the same board, private cards and action history and graded
the first decision exactly once before continuing to showdown. No guessed
policy, duplicate action or new random runout was substituted.

Desktop/mobile screenshots were visually inspected. Light/dark layouts at
375/768/1024/1440px had no horizontal overflow; visible mobile action controls
were 48px tall above navigation. Reduced-motion play and keyboard actions
worked, and “Review hand” visibly focused the feedback region. Native selects
used the existing DOM change-event workaround; native select-keyboard behavior
is not newly certified. No unhandled application exceptions or practice-inference
API requests were captured. The task-owned Chrome instance was closed after QA.

These checks certify the declared conditional games only, not unrestricted
ranges, a certified earlier preflop line, or full-hand Hold'em exploitability.
