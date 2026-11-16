# Broader trips-board and wheel-draw flops — 2026-11-15

This batch adds a 50bb trips-board game and a 100bb ace-low two-club game.
Both use broader weighted priors including offsuit ace-high hands. These are
illustrative conditional study assumptions, not certified earlier betting
lines or universal full-hand policies. Existing studies, active models,
solver algorithms, action abstraction and validation gates remain unchanged.

## Declared games

- `flop-50bb-trips-nines`: 9c 9d 9h, 50bb initial depth, 32bb pot, 34bb behind.
  BTN AA ×1, KK ×0.75, A9s ×1, AKo ×0.5, QJs ×0.5, J9s ×0.75 versus BB QQ ×1,
  JJ ×0.75, K9s ×1, AQo ×0.5, JTs ×0.5, 98s ×0.75. Quads, full houses and
  kicker battles remain supported. Private pairs cannot both hold the last nine.
- `flop-100bb-wheel-clubs`: Ac 5c 4d, 100bb initial depth, 60bb pot, 70bb behind.
  BTN AA ×1, AKo ×0.75, A5s ×1, 76s ×0.75, 33 ×0.5, KQs ×0.5, 65s ×0.5 versus
  BB 55 ×1, 44 ×1, A4s ×1, 32s ×1, A2s ×0.75, 87s ×0.5, TT ×0.5, QJs ×0.5,
  54s ×0.75. Made wheels, sets, two pair, pairs and straight/club draws remain
  in the same jointly solved game, including later paired-board and flush runouts.

Multipliers describe per-combination starting inclusion, not action frequency.
Public blockers are removed before normalization; private pairs use the exact
compatible joint prior. Each policy jointly solves the complete flop-to-river
game with exact card removal, ordered public-card arrivals and full action
history. Half-pot/pot/all-in bets, a pot-sized raise plus all-in and the
one-subsequent-raise cap are unchanged. No neural leaves, online resolves,
bucketing or zero-reach pruning are introduced.

## Preflight, pilots and acceptance

An allocation-free census precedes separate 128-update pilots. Publication
requires unchanged ≤0.05bb summed unilateral best-response gain, ≤0.0001bb
maximum available-action EV discrepancy, valid frozen probabilities and fresh
native re-import of every actual binary byte. Zero compatible opponent reach
remains explicitly unavailable and ungraded, not a zero-loss estimate. Rejected
pilots produce reports, never serving policies; longer candidates keep the
same board, ranges, betting tree and numerical tolerances.

The existing 32-combo-per-seat, 500,000-decision, 2,000,000-total-node, 512MiB
accumulator, 2MB manifest and 65,536-byte shard hard caps remain unchanged.
The earlier 27 studies and their bytes are preserved. Verification uses the six
exhaustive TypeScript checks per new study, normal production-build artifact
verification and browser-harness QA in isolated muted local headless Chrome.
The full CI suite is not rerun merely to commit; Actions remain PR/manual-only.
Serving stays static, hash-pinned and lazy with the existing bounded cache and
Git-connected deployment. No database, paid compute or new credentials are
needed. Task-scoped idle-sleep holds leave permanent Mac power settings unchanged.
Concurrent timings are local observations, not general throughput benchmarks
or full-hand exploitability measurements.

## Structural preflight

| Study | BTN / BB combos | Compatible private deals | Ordered deal–runouts | Exact accumulator bytes |
| --- | --- | ---: | ---: | ---: |
| 50bb trip nines | 30 / 30 | 771 | 1,526,580 | 101,084,160 |
| 100bb wheel/clubs | 32 / 31 | 926 | 1,833,480 | 106,138,368 |

Both complete games contain 104,872 public decisions and 250,374 total nodes
including chance/terminals. Every supported combination and ordered arrival
is retained within existing caps. The audit's legacy monolithic-reader warning
does not apply to the installed bounded flop-shard reader. The census measures
structure, not strategy quality.

## Short pilots

| 128-update pilot | Summed response gain (bb/spot) | Maximum EV error (bb) | Available action values checked | Result |
| --- | --- | --- | ---: | --- |
| [50bb trip nines](reports/2026-11-15-flop-50bb-trips-nines-128.json) | 0.1430124228 | 0.0000019073324 | 5,662,012 | Rejected |
| [100bb wheel/clubs](reports/2026-11-15-flop-100bb-wheel-clubs-128.json) | 0.08138686778 | 0.0000038146973 | 6,206,227 | Rejected |

Neither pilot produced a serving artifact. Both passed exact EV checks but
missed the unchanged 0.05bb summed response-gain gate. Candidates double to
256 updates first, retaining identical boards, ranges, actions and tolerances;
they must pass before activation. Longer work is reserved for a candidate
that still fails. Reports embed the original short-run inputs and remain
reproducible after a source's update count advances.

Concurrent pilots took 81.30s / 91.45s wall with about 359MB / 387MB maximum RSS.
Independent scalar/vector profile discrepancies were ≤1.78e-15bb. Respectively
326,695 / 213,464 combination rows had no compatible opponent reach and remain
explicitly unavailable and ungraded. These contended local observations do not
establish general throughput or full-hand exploitability.

## Accepted exports

Both candidates passed at 256 updates and passed fresh native re-import of
the actual binary bytes before catalog activation. No further training or
limit/tolerance change was needed.

| Study | Summed response gain (bb/spot) | Maximum EV error (bb) | Binary bytes | Manifest bytes | Largest shard |
| --- | --- | --- | ---: | ---: | ---: |
| 50bb trip nines | 0.0264486484763 | 0.00000190733787 | 73,946,239 | 236,110 | 44,314 |
| 100bb wheel/clubs | 0.0265158408456 | 0.00000381469681 | 78,071,663 | 236,163 | 45,482 |

Response gains fell about 81.5% / 67.4% from each game's own short pilot. These
are within-game iteration comparisons, not comparisons across different priors
or full-hand exploitability results. Each bundle contains 1,957 binary shards.
Some broader-range shards exceed the usual 32KB target but stay below the
unchanged 65,536-byte hard cap; chunk layout and precision were not changed to
hide the storage cost. Only frozen probabilities and checked EVs are served,
not regrets, resumable state or fabricated continuation values.

Manifest identities:

- 50bb trip nines:
  `a92c3cc7d39e7926471f130727cc16fb301f0bb159a4927200e0db3a49d2b40a`.
- 100bb wheel/clubs:
  `daddf929bba637b3d62b9189c7071435be620c9442f869ff06ee04b11e16e102`.

Concurrent export plus fresh re-import took 165.57s / 186.36s wall with about
635MB / 686MB maximum RSS. These are contended local measurements, not clean
hardware-throughput benchmarks.

The library now contains 29 studies: 21 flop-to-river games, four turn-to-river
games and four river games. Its 59,207 serving files total 1,634,054,623
uncompressed bytes. The earlier 27 studies and active full-hand artifacts stay
unchanged. A hand still fetches only its pinned manifest and needed shards
using the existing 16-shard cache, not the full corpus.

## Browser verification

Trip-nines mobile BTN play deliberately blocked a real binary request at the
flop-to-turn boundary. The dealt turn, hero cards and exactly two checks were
preserved without an EV grade. Retry kept the same state, graded the first
decision exactly once and resumed through all streets to showdown. Keyboard
“Review hand” visibly focused the feedback region. No guessed strategy,
duplicated action or replacement random runout was substituted.

Desktop BB play used normal motion, bet 16bb and faced a 16bb call, then
continued through both remaining streets to showdown. Hero actually held the
last nine, 9s, with 8s, exercising a legal quads deal without duplicate cards.
The range panel showed all declared inclusion weights separately from action
frequency and the pinned manifest identity.

Wheel/clubs desktop BB play bet 30bb and settled an opponent fold. Its mobile
BTN flow called a 30bb flop bet and the remaining 40bb against a turn all-in,
reaching showdown with a complete runout. Both details panels correctly showed
all inclusion weights and pinned versions. Opponent cards stayed hidden until
terminal review; mixes and EV loss stayed hidden before the first user action.

Browser-harness used a fresh isolated local profile with `--headless=new
--mute-audio` and loopback-only CDP; the task-owned instance was closed after
QA. Desktop games used normal motion and mobile games reduced motion. Screenshots
were visually inspected. Light/dark layouts at 375, 768, 1024 and 1440px had no
horizontal overflow; visible mobile controls remained 48px tall above navigation.
Keyboard actions and visible review focus worked. Native selects used the
existing DOM change-event workaround, so native select-keyboard behavior is not
newly certified. No unhandled application exception, console application error
or practice-inference API request was captured. The only unrelated console issue
was the existing `/favicon.ico` 404, left unchanged.

## Focused automated checks

`npx vitest run lib/solved-flop.test.ts -t 'flop-(50bb-trips-nines|100bb-wheel-clubs)'`
passed all 12 targeted checks; 114 unrelated cases were skipped. The independent
exhaustive EV/response checks took about 44.0s / 61.8s. Each new study received
every-available-EV, both information-set-consistent best-response, all-public-
branch, exact joint-prior interval, bounded decoding, loading/retry and pot-raise
legality checks. These are not checks of only the browser's sampled hands.
Existing cases, time limits and numerical tolerances were not changed.

Both new manifest URLs retained immutable one-year caching and COOP/COEP.

`npm run build` passed, including fresh native verification of all 29 saved
studies, integrity checks for unchanged full-hand artifacts, TypeScript checks
and Next.js production compilation. No full TypeScript/Rust CI suite was run;
the PR/manual-only workflow configuration is unchanged.

These are conditional solutions for declared priors, not unrestricted ranges,
certified preflop lines or full-hand exploitability guarantees. Root response
gain is prior-weighted, not a maximum error at every off-path decision. User
deviations can reach larger action-EV differences; those values still enumerate
the same frozen policy and are not capped by the small root response-gain bound.
