# Broader queen-high paired and monotone flops — 2026-11-11

This batch targets two missing stack/texture combinations: a weighted 20bb
queen-high paired game and a 100bb queen-high monotone game. Starting priors
are declared study assumptions, not certified preflop strategies. Existing
studies, active full-hand policies, solver algorithms and validation gates
remain unchanged. This is not a universal stack-depth model release.

## Declared games

- `flop-20bb-queen-paired`: Qc 6d 6s, 20bb initial depth, 12bb pot, 14bb behind.
  BTN QQ ×1, AQo ×0.75, KK ×1, A6s ×0.75, JTs ×0.5, 98s ×0.5, KQs ×1 versus
  BB 66 ×1, QJs ×1, 76s ×0.75, TT ×0.75, KQo ×0.5, 87s ×0.5, A5s ×0.5.
  The game includes quads, a full house, trips, pairs and unpaired holdings.
  Multipliers describe per-combination starting inclusion, not action frequency.
- `flop-100bb-monotone`: Qs 8s 3s, 100bb initial depth, 60bb pot, 70bb behind.
  BTN AA/QQ/KJs/AQs/JTs/99/65s versus BB 88/33/QJs/ATs/KTs/77/T9s, with equal
  inclusion per unblocked exact combination. Made flushes, sets, pairs and
  single-spade blockers create different turn/river decisions.

Public blockers are removed before normalization. Private pairs use the exact
compatible joint prior. Each policy jointly solves all three streets, retaining
every supported combination, complete action history and ordered card arrival.
Existing half-pot/pot/all-in bets, one pot-sized raise plus all-in, and the
one-subsequent-raise cap are unchanged. No bucketing, online solve, neural leaf
or zero-reach pruning is introduced.

## Preflight, pilots and publication

An allocation-free native census precedes separate 128-update pilots. The
unchanged acceptance gates are ≤0.05bb summed unilateral best-response gain,
≤0.0001bb maximum available-action EV discrepancy and valid frozen probability
sums. Every accepted export must pass fresh native re-import of the actual
binary bytes before catalog activation. EVs with no compatible opponent reach
remain unavailable and ungraded, not zero-loss estimates. Rejected pilots
produce research reports, never serving policies; longer candidates retain
the identical game and hard limits.

The existing 32-combo-per-seat, 500,000-decision, 2,000,000-total-node, 512MiB
accumulator, 2MB manifest and 65,536-byte shard caps are unchanged. The earlier
21 studies and their bytes are preserved. Verification uses each new study's
six exhaustive TypeScript tests, the production build's normal artifact checks,
and browser-harness QA in isolated local Chrome with `--headless=new --mute-audio`.
The full CI suite is not rerun merely to commit; Actions remain PR/manual-only.
Git-connected deployment, bounded browser caching and static version-pinned
shards remain in place. No database, paid compute or credentials are needed.
Task-scoped idle-sleep holds during active runs leave permanent Mac power
settings unchanged. Concurrent timings are local observations, not clean
throughput benchmarks or full-hand exploitability measurements.

## Structural preflight

| Study | BTN / BB combos | Compatible private deals | Ordered deal–runouts | Exact accumulator bytes |
| --- | --- | ---: | ---: | ---: |
| 20bb paired | 31 / 29 | 760 | 1,504,800 | 101,084,160 |
| 100bb monotone | 30 / 27 | 752 | 1,488,960 | 96,029,952 |

Both games contain 104,872 public decision nodes and 250,374 total nodes,
including chance and terminal nodes. They fit every existing structural limit
without removing any supported combination or action. The census's legacy
monolithic-reader warning does not apply to the installed bounded flop-shard
reader; the census does not establish strategy quality.

## Short pilot results

| 128-update pilot | Summed response gain (bb/spot) | Maximum EV error (bb) | Available action values checked | Result |
| --- | --- | --- | ---: | --- |
| [20bb paired](reports/2026-11-11-flop-20bb-queen-paired-128.json) | 0.03328529003 | 0.0000009536743 | 5,883,731 | Passed |
| [100bb monotone](reports/2026-11-11-flop-100bb-monotone-128.json) | 0.1603460281 | 0.0000038146973 | 5,725,209 | Rejected; no serving artifact |

The 20bb candidate retains 128 updates. The 100bb candidate advances to 512
updates because its pilot remains over three times the unchanged response-gain
limit; the board, priors, betting abstraction and numerical tolerances stay
fixed. Pilot reports embed the original inputs and remain reproducible after
the published source's update count changes.

Concurrent pilots took 82.25s / 77.05s wall with about 358MB / 349MB maximum RSS.
Independent scalar/vector profile discrepancies were ≤2.23e-15bb. Respectively
216,975 / 138,331 combination rows had no compatible opponent reach and remain
unavailable and ungraded. These local contended timings are not general
throughput benchmarks, nor do the conditional results certify full-hand play.

## Accepted exports

The 20bb paired export passed fresh byte re-import at 128 updates. Its frozen
policy and EV fingerprints exactly match the passing short pilot. Response
gain remains 0.0332852900290bb and maximum EV discrepancy 0.00000095367432bb.
The 1,957 binary shards total 75,188,675 bytes, manifest 236,089 bytes and largest
shard 44,306 bytes, within the existing 65,536-byte hard limit. Some shards are
above the usual 32KB target due to broader combination support; chunk layout,
precision and the hard cap were not changed to hide that cost.

20bb manifest SHA-256:
`ec9a61ab4436012e4544a6659d06bbd42e70c7608b6897881fb212e3884bf000`.
Export plus fresh re-import took 97.81s wall with about 580MB maximum RSS while
the independent 100bb candidate ran. This is a contended local observation.

The 100bb 512-update export passed at **0.0123857004169bb** summed response gain,
about 92.3% below its own 128-update pilot, with maximum EV discrepancy
0.00000381469727bb. This is an iteration comparison within the identical
conditional game, not between different priors or a full-hand exploitability
result. The 1,957 binary shards total 70,616,559 bytes, manifest 235,731 bytes,
and largest shard 41,978 bytes, all within the unchanged hard caps.

100bb manifest SHA-256:
`6240516fb508d193396e48268505008a0979566e65e8486060ba0edc7d519e0c`.
Export plus fresh re-import took 278.50s wall and about 617MB maximum RSS. No
regrets, resumable state, guessed continuation or neural leaf value is served.

The library now contains 23 studies: 15 flop-to-river games, four turn-to-river
games and four river games. Its 47,459 serving files total 1,183,548,029
uncompressed bytes. The earlier 21 studies and their artifacts remain unchanged.
A hand still loads only its pinned manifest and needed shards with the existing
16-shard cache, not the whole corpus. Deployment remains Git-connected rather
than a plain CLI source upload.

## Focused verification

`npx vitest run lib/solved-flop.test.ts -t flop-20bb-queen-paired` passed all six
checks; 78 unrelated cases were skipped. Its independent exhaustive EV/response
check took about 53.8 seconds. Coverage includes every exported action EV,
both information-set-consistent best responses, all public branches, exact
joint-prior interval dealing, bounded decoding, loading/retry and pot-raise
legality. Existing cases and tolerances were not modified.

`npx vitest run lib/solved-flop.test.ts -t flop-100bb-monotone` passed all six
checks; 84 unrelated cases were skipped. Its independent exhaustive EV/response
check took about 45.4 seconds. The same full-node, all-EV and both-response
coverage applies; tests do not rely on only the browser's sampled hands.

`npm run build` passed, including fresh native verification of all 23 saved
studies, integrity checks for unchanged full-hand artifacts, TypeScript checks
and Next.js production compilation. No full TypeScript/Rust CI suite was run;
the PR/manual-only workflow configuration is unchanged.

Both new manifest URLs retained immutable one-year caching and COOP/COEP
headers. The serving format, action grid and validation tolerances are unchanged.

Browser-harness exercised the 20bb paired game on mobile from BTN. Blocking a
real binary request at the flop-to-turn boundary preserved the dealt turn,
hero's private cards and exactly two checks, without an EV grade. Retry kept
the same state and graded the first decision exactly once, then the hand
checked through the remaining streets to showdown. Keyboard “Review hand”
visibly focused the feedback region. No guessed policy or duplicated action
was used.

The paired game's BB flow bet 6bb, faced the opponent's 14bb all-in and called
the remaining 8bb, reaching showdown with a complete five-card runout. Its
details panel correctly displayed both seats' declared weights, distinguished
inclusion from action probability, and showed the pinned version identity.
The 100bb game was played from both seats: BB bet 30bb and settled an opponent
fold; BTN called a 30bb flop bet and played through turn/river to showdown.
Opponent cards stayed hidden until terminal review and no EV loss appeared
before the first user action. No practice-inference API requests or unhandled
application exceptions were captured.

Browser QA used a fresh isolated local Chrome profile with `--headless=new
--mute-audio` and loopback-only CDP; the task-owned instance was closed after
QA. Desktop/mobile screenshots were inspected. Light/dark layouts at 375, 768,
1024 and 1440px had no horizontal overflow. Visible mobile action controls
remained 48px tall above navigation; reduced-motion play, keyboard actions and
visible feedback focus worked. Native selects used the existing DOM change-event
workaround, so native select-keyboard behavior is not newly certified.

These results certify only these declared conditional games, not unrestricted
starting ranges, a solved preflop line or full-hand Hold'em exploitability.
The root response-gain bound is prior-weighted, not a maximum error for every
off-path decision. Deliberate deviations can reach rows with large action-EV
differences; the displayed EV loss still enumerates the same frozen policy
through showdown and is not capped by the small root response-gain figure.
