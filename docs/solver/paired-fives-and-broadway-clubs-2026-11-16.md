# Broader paired-fives and broadway-club flops — 2026-11-16

This batch adds a 20bb low-paired game and a 100bb two-club broadway game.
Both use broader weighted starting priors, including offsuit ace-high hands.
These are declared conditional study assumptions, not certified earlier
betting lines or universal full-hand policies. Existing studies, active models,
solver algorithms, action abstraction and validation gates remain unchanged.

## Declared games

- `flop-20bb-paired-fives`: 5c 5d 2h, 20bb initial depth, 12bb pot, 14bb behind.
  BTN AA ×1, KK ×0.75, A5s ×1, AKo ×0.5, QJs ×0.5, 65s ×0.75 versus BB QQ ×1,
  JJ ×0.75, K5s ×1, AQo ×0.5, 55 ×1, 22 ×1. Quads, a full house, trips,
  overpairs and unpaired holdings remain supported with exact card removal.
- `flop-100bb-broadway-clubs`: Kc Qc Td, 100bb initial depth, 60bb pot, 70bb behind.
  BTN AA ×1, KK ×1, AKo ×0.75, AJs ×1, J9s ×0.75, QTs ×0.75, 98s ×0.5 versus
  BB QQ ×1, TT ×1, KQs ×1, QJs ×0.75, ATs ×0.75, JJ ×0.5, AQo ×0.5.
  Straights, sets, two pair, pairs and club/straight draws share one jointly
  solved game, including later paired-board, flush and straight-flush runouts.

Multipliers describe per-combination starting inclusion, not action frequency.
Public blockers are removed before normalization; private pairs use the exact
compatible joint prior. Each policy jointly solves all three streets with
exact card removal, ordered public-card arrivals and complete action history.
Half-pot/pot/all-in bets, a pot-sized raise plus all-in and the one-subsequent-
raise cap are unchanged. No neural leaves, online resolves, buckets or
zero-reach pruning are introduced.

## Preflight, pilots and acceptance

An allocation-free census precedes separate 128-update pilots. Publication
requires unchanged ≤0.05bb summed unilateral best-response gain, ≤0.0001bb
maximum available-action EV discrepancy, valid frozen probabilities and fresh
native re-import of actual exported binary bytes. Zero compatible opponent
reach remains explicitly unavailable and ungraded. Rejected pilots produce
reports, never serving policies. Longer candidates retain the same board,
priors, actions and numerical tolerances.

The existing 32-combo-per-seat, 500,000-decision, 2,000,000-total-node, 512MiB
accumulator, 2MB manifest and 65,536-byte shard caps are unchanged. The earlier
29 studies and their bytes are preserved. Verification uses six exhaustive
TypeScript checks per new study, normal production-build artifact verification
and browser-harness QA in isolated local Chrome with `--headless=new --mute-audio`.
The full CI suite is not rerun merely to commit; Actions remain PR/manual-only.
Serving stays static, lazy and hash-pinned with the existing bounded cache and
Git-connected deployment. No database, paid compute or new credentials are
needed. Task-scoped idle-sleep holds leave permanent Mac power settings unchanged.
Concurrent timings are local observations, not general throughput benchmarks
or full-hand exploitability data.

## Structural preflight

| Study | BTN / BB combos | Compatible private deals | Ordered deal–runouts | Exact accumulator bytes |
| --- | --- | ---: | ---: | ---: |
| 20bb paired fives | 32 / 30 | 826 | 1,635,480 | 104,453,632 |
| 100bb broadway/clubs | 32 / 30 | 819 | 1,621,620 | 104,453,632 |

Each complete game contains 104,872 public decisions and 250,374 total nodes
including chance/terminals. All supported combinations and ordered arrivals
remain within existing limits. The audit's legacy monolithic-reader warning
does not apply to the installed bounded flop-shard reader. The census measures
structure, not strategy quality.

## Short pilots

| 128-update pilot | Summed response gain (bb/spot) | Maximum EV error (bb) | Available action values checked | Result |
| --- | --- | --- | ---: | --- |
| [20bb paired fives](reports/2026-11-16-flop-20bb-paired-fives-128.json) | 0.05476619610 | 0.0000009536743 | 5,713,868 | Rejected |
| [100bb broadway/clubs](reports/2026-11-16-flop-100bb-broadway-clubs-128.json) | 0.1224593238 | 0.0000038146973 | 6,263,818 | Rejected |

Both passed exact EV checks but missed the unchanged 0.05bb response-gain gate;
neither produced a serving artifact. Candidates double to 256 updates first,
keeping identical boards, priors, actions and tolerances. Longer work is reserved
for a candidate that still fails. Reports embed their original short-run inputs
and remain reproducible after a source's update count advances.

Concurrent pilots took 88.58s / 88.72s wall with about 364MB maximum RSS each.
Independent scalar/vector profile discrepancies were ≤1.78e-15bb. Respectively
407,135 / 131,280 combination rows had no compatible opponent reach and remain
explicitly unavailable and ungraded. These contended local measurements are
not general throughput benchmarks or full-hand exploitability results.

## Accepted exports

Both candidates passed at 256 updates and passed fresh native re-import of
the actual binary bytes before catalog activation. No extra training, game
change or tolerance relaxation was needed.

| Study | Summed response gain (bb/spot) | Maximum EV error (bb) | Binary bytes | Manifest bytes | Largest shard |
| --- | --- | --- | ---: | ---: | ---: |
| 20bb paired fives | 0.0173930026057 | 0.000000953674316 | 75,427,119 | 236,095 | 45,474 |
| 100bb broadway/clubs | 0.0384295649810 | 0.00000381469728 | 77,249,315 | 236,122 | 45,482 |

Response gains fell about 68.2% / 68.6% from each game's own short pilot. These
are within-game iteration comparisons, not comparisons between different priors
or full-hand exploitability results. Each bundle has 1,957 binary shards. Some
broader-range shards exceed the usual 32KB target but stay below the unchanged
65,536-byte hard cap; chunk layout and precision were not changed to conceal
the storage cost. Serving includes only frozen probabilities and checked EVs,
not regrets, resumable state or fabricated continuation values.

Manifest identities:

- 20bb paired fives:
  `b20cf7c5d37bfc0305f3547b896acf49e729653d43d4cc63e57ed0ed624a3ec9`.
- 100bb broadway/clubs:
  `52024e01da9d9d0c79f8da4ed59a0d9f255836fbe5b99f4b4627b1b36cd1009b`.

Concurrent export plus fresh re-import took 174.39s / 174.35s wall with about
594MB / 593MB maximum RSS. These are contended local observations, not clean
hardware-throughput benchmarks.

The library now contains 31 studies: 23 flop-to-river games, four turn-to-river
games and four river games. Its 63,123 serving files total 1,787,203,274
uncompressed bytes. The earlier 29 studies and active full-hand artifacts stay
unchanged. A hand fetches only its pinned manifest and needed shards with the
existing 16-shard cache, not the full corpus.

## Browser verification

Paired-fives mobile BTN play deliberately blocked a real binary request at the
flop-to-turn boundary. The dealt turn, hero cards and public action history
remained intact without an EV grade. Retry preserved the state, graded the
first decision exactly once and resumed through to terminal review. Keyboard
“Review hand” visibly focused the feedback region. No guessed strategy,
duplicated action or replacement runout was used.

Paired-fives desktop BB play used normal motion, bet 6bb and settled an opponent
fold. Its range panel showed every declared inclusion weight separately from
action frequency and the pinned manifest identity.

Broadway/clubs desktop BB play bet 30bb, faced a 30bb call and then called
the opponent's remaining 40bb on the turn, reaching showdown with a complete
runout. Its mobile BTN flow checked through flop/turn and called a 30bb river
bet to showdown. Both details panels displayed all declared inclusion weights
and pinned versions. Opponent cards remained hidden until terminal review;
mixes and EV loss stayed hidden before the first user action.

Browser-harness used a fresh isolated local profile with `--headless=new
--mute-audio` and loopback-only CDP; the task-owned instance was closed after
QA. Desktop games used normal motion and mobile games reduced motion.
Screenshots were visually inspected. Light/dark layouts at 375, 768, 1024 and
1440px had no horizontal overflow; visible mobile controls remained 48px tall
above navigation. Keyboard actions and visible review focus worked. Native
selects used the existing DOM change-event workaround, so native select-keyboard
behavior is not newly certified. No unhandled application exception, console
application error or practice-inference API request was captured. The only
unrelated console issue was the existing `/favicon.ico` 404, left unchanged.

## Focused automated checks

`npx vitest run lib/solved-flop.test.ts -t 'flop-(20bb-paired-fives|100bb-broadway-clubs)'`
passed all 12 targeted checks; 126 unrelated cases were skipped. Independent
exhaustive EV/response checks took about 48.9s / 55.7s. Each new study received
every-available-EV, both information-set-consistent best-response, all-public-
branch, exact joint-prior interval, bounded decoding, loading/retry and pot-raise
legality checks. These do not depend only on sampled browser decisions.
Existing cases, time limits and numerical tolerances were not changed.

`npm run build` passed: fresh artifact verification for all 31 studies,
full-hand artifact integrity checks, production compilation, TypeScript checks
and all 14 static pages. No full CI suite or manual CI dispatch was run; the
existing PR/manual-only workflow configuration is unchanged.

Both new manifest URLs retained immutable one-year caching and COOP/COEP.

These are conditional solutions for declared priors, not unrestricted ranges,
certified preflop lines or full-hand exploitability guarantees. Root response
gain is prior-weighted, not a maximum error at every off-path decision. User
deviations can reach larger action-EV differences; those values still enumerate
the same frozen policy and are not capped by the small root response-gain bound.
The existing exporter completes rows with zero or negligible accumulated average
weight uniformly. Such off-path frequencies are not evidence that actions have
equal EV; this batch does not change that behavior.
