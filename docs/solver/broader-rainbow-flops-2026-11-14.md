# Broader broadway and low-rainbow flops — 2026-11-14

This batch adds a 20bb broadway-rainbow game and a 50bb low-rainbow game, both
with broader weighted starting priors including offsuit hands. These are
illustrative conditional study assumptions, not certified preflop frequencies
or earlier betting lines. The earlier 25 studies, active full-hand models,
solver algorithm, action abstraction and validation gates remain unchanged.

## Declared games

- `flop-20bb-broadway`: Ac Kd Qh, 20bb initial depth, 12bb pot, 14bb behind.
  BTN AA ×1, AKo ×1, AQo ×0.75, JTs ×1, QQ ×1, 76s ×0.5, KQs ×0.75 versus
  BB KK ×1, AJs ×0.75, ATs ×0.75, AQo ×1, QJs ×0.75, KQo ×0.75, T9s ×0.5.
  The starting board includes broadway, sets, two pair, pairs and straight draws.
- `flop-50bb-low-rainbow`: 7c 4d 2h, 50bb initial depth, 32bb pot, 34bb behind.
  BTN AA ×1, 77 ×1, A7s ×1, KQo ×0.5, 65s ×0.75, 54s ×0.5 versus BB 44 ×1,
  22 ×1, 76s ×1, 88 ×0.75, A5s ×0.75, 53s ×0.5, KJs ×0.5, 98s ×0.5.
  Sets, overpairs, pairs, straight draws and unpaired holdings remain supported.
  This is a different board/pot/range game from the existing deeper 8s 5d 2c
  study, which stays unchanged.

Multipliers describe inclusion per exact combination, not action frequency.
Public blockers are removed before normalization; private pairs use the exact
compatible joint prior. Each policy jointly solves the complete flop-to-river
game with exact card removal, ordered card arrivals and full action history.
Half-pot/pot/all-in bets, a pot-sized raise plus all-in and the one-subsequent-
raise cap are unchanged. There are no neural leaves, online resolves, buckets
or zero-reach pruning.

## Preflight, pilots and acceptance

An allocation-free census precedes separate 128-update pilots. Publication
requires unchanged ≤0.05bb summed unilateral best-response gain, ≤0.0001bb
maximum available-action EV discrepancy, valid frozen probabilities and fresh
native re-import of the actual exported bytes. Zero compatible opponent reach
remains unavailable and ungraded, not a zero-loss estimate. A rejected pilot
produces a report, not a serving policy. Any longer candidate retains the same
board, priors, betting tree and numerical tolerances.

The existing 32-combo-per-seat, 500,000-decision, 2,000,000-total-node, 512MiB
accumulator, 2MB manifest and 65,536-byte shard hard caps stay unchanged.
Validation uses six exhaustive TypeScript checks per new study, normal
production-build artifact verification and browser-harness QA in isolated local
Chrome launched with `--headless=new --mute-audio`. The full CI suite is not
rerun merely to commit; Actions stay PR/manual-only. Serving remains static,
hash-pinned and lazy with the existing bounded cache and Git-connected builds.
No database, paid compute or new credentials are needed. Task-scoped idle-sleep
holds leave permanent Mac power settings unchanged. Concurrent timings are
local observations, not clean throughput or full-hand exploitability data.

## Structural preflight

| Study | BTN / BB combos | Compatible private deals | Ordered deal–runouts | Exact accumulator bytes |
| --- | --- | ---: | ---: | ---: |
| 20bb broadway | 30 / 30 | 684 | 1,354,320 | 101,084,160 |
| 50bb low rainbow | 31 / 31 | 902 | 1,785,960 | 104,453,632 |

Each complete game contains 104,872 public decisions and 250,374 total nodes
including chance/terminals. Every supported combination and ordered card
arrival is retained within the existing caps. The audit's legacy monolithic-
reader warning does not apply to the installed bounded flop-shard reader.
These counts establish structure, not strategy quality.

## Short pilots

| 128-update pilot | Summed response gain (bb/spot) | Maximum EV error (bb) | Available action values checked | Result |
| --- | --- | --- | ---: | --- |
| [20bb broadway](reports/2026-11-14-flop-20bb-broadway-128.json) | 0.007426211763 | 0.0000009536743 | 5,758,025 | Passed |
| [50bb low rainbow](reports/2026-11-14-flop-50bb-low-rainbow-128.json) | 0.09602673421 | 0.0000019073486 | 5,809,803 | Rejected; no serving artifact |

The 20bb candidate retains 128 updates. The 50bb candidate doubles to 256
updates first because its pilot is under twice the unchanged response-gain
limit; it must still pass exact verification before publication. Board, priors,
action tree and numerical tolerances stay fixed. Pilot reports embed their
original input, so they remain reproducible after the published update count
advances.

Concurrent pilots took 88.51s / 99.49s wall with about 350MB / 378MB maximum RSS.
Independent scalar/vector profile discrepancies were ≤8.89e-16bb. Respectively
278,646 / 359,261 rows had no compatible opponent reach and remain unavailable
and ungraded. These are contended local observations, not general throughput
benchmarks or full-hand exploitability results.

## Accepted exports

The 20bb broadway 128-update export passed fresh native re-import of all bytes.
Its frozen-policy and action-EV fingerprints exactly match the passing pilot;
summed response gain remains 0.00742621176333bb and maximum EV discrepancy
0.00000095367432bb. The 1,957 binary shards total 74,685,851 bytes, manifest
235,922 bytes and largest shard 44,306 bytes, within the unchanged 65,536-byte
hard cap. Broader-range shards can exceed the usual 32KB target; chunk layout,
precision and hard limits were not changed to hide this cost.

20bb manifest SHA-256:
`20df2f73987de041cb0fbe54eef88577affcc6ae1e531b31985f196ab3ec7337`.
Export plus fresh re-import took 95.83s wall with about 650MB maximum RSS while
the independent 50bb candidate ran. This is a contended local observation.

The 50bb low-rainbow 256-update export passed at **0.0258074878782bb** summed
response gain, about 73.1% below its own 128-update pilot. Maximum EV discrepancy
is 0.00000190734864bb. This compares iterations within the identical conditional
game, not different priors or full-hand exploitability. The 1,957 binary shards
total 76,024,615 bytes, manifest 236,139 bytes and largest shard 44,412 bytes,
within the unchanged hard caps.

50bb manifest SHA-256:
`b20c02ce5586891027842d942f8b3fcff1f2ba1bf3befea1bad285eeef0a9b0e`.
Export plus fresh re-import took 184.42s wall with about 596MB maximum RSS.
Serving contains only frozen probabilities and checked EVs, not regrets,
resumable training state or fabricated continuation values.

The library now contains 27 studies: 19 flop-to-river games, four turn-to-river
games and four river games. Its 55,291 serving files total 1,481,564,448
uncompressed bytes. The earlier 25 studies and active full-hand artifacts
remain unchanged. A hand still loads only its pinned manifest and required
shards with the existing 16-shard cache, not the whole corpus.

## Focused verification

`npx vitest run lib/solved-flop.test.ts -t flop-20bb-broadway` passed all six
checks; 102 unrelated cases were skipped. The independent exhaustive EV/response
check took about 46.4 seconds. Coverage includes every exported action EV,
both information-set-consistent best responses, every public branch, exact
joint-prior interval dealing, bounded decoding, loading/retry and pot-raise
legality. Existing cases and tolerances were not modified.

`npx vitest run lib/solved-flop.test.ts -t flop-50bb-low-rainbow` passed all six
checks; 108 unrelated cases were skipped. Its exhaustive EV/response check took
about 51.7 seconds. The same independent all-node, all-EV and both-response
coverage applies; verification does not depend on only sampled browser decisions.

`npm run build` passed, including fresh native verification of all 27 saved
studies, integrity checks for unchanged full-hand artifacts, TypeScript checks
and Next.js production compilation. No full TypeScript/Rust CI suite was run;
the PR/manual-only workflow configuration is unchanged.

Both new manifest URLs retained immutable one-year caching and COOP/COEP.
Serving format, action grid and numerical/validation tolerances are unchanged.

The broadway mobile flow deliberately blocked a real binary request at the
flop-to-turn boundary. The dealt turn, hero cards and exactly two public checks
remained in place without an EV grade. Retry preserved that state and graded
the first decision exactly once, then the hand checked through to showdown.
Keyboard “Review hand” visibly focused the feedback region. No guessed policy,
duplicate action or replacement runout was used.

Broadway desktop BB play bet 6bb, faced the opponent's 14bb all-in and called
the remaining 8bb to showdown. The declared ranges panel showed all inclusion
multipliers separately from action probabilities and displayed the pinned hash.

Low-rainbow desktop BB play bet 16bb and settled an opponent fold. Mobile BTN
called a 16bb flop bet and the remaining 18bb against a turn all-in, reaching
showdown with a full runout. Its range panel correctly displayed all declared
inclusion weights and the pinned version. Opponent cards stayed hidden until
terminal review, and mixes/EV loss stayed hidden before the first user action.

Browser-harness used a fresh local profile with `--headless=new --mute-audio`
and loopback-only CDP; the task-owned instance was closed after QA. Screenshots
were inspected at desktop/mobile sizes. Light/dark layouts at 375, 768, 1024
and 1440px had no horizontal overflow; visible mobile controls remained 48px
tall above navigation. Reduced-motion play, keyboard actions and visible
feedback focus worked. Native selects used the existing DOM change-event
workaround, so native select-keyboard behavior is not newly certified.

Console/network inspection found only `/favicon.ico` returning 404, unrelated
to the new studies and left unchanged. No unhandled application exception,
console application error or practice-inference API request was captured. The
deliberately interrupted shard request was handled by the paused-table UI.

These are conditional solutions for declared priors, not unrestricted ranges,
a certified preflop line or full-hand Hold'em exploitability. The root response-
gain bound is prior-weighted, not a maximum error at every off-path decision.
User deviations can reach rows with larger EV differences; those values still
enumerate the same frozen continuation and are not capped by root response gain.
