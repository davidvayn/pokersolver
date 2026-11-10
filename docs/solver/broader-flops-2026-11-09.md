# Broader ace-high and low-paired flops — 2026-11-09

This expansion targets two missing declared-range situations: a 20bb ace-high
flop and a 100bb low-paired flop with weighted suited and offsuit support.
Existing studies, solver algorithms, action grids and validation gates stay
unchanged. These are illustrative conditional starting priors, not certified
preflop strategies or a universal full-hand model.

Both games jointly solve the entire flop-to-river tree. Exact private-card
removal, ordered public-card arrivals and complete betting history are retained.
The existing half-pot/pot/all-in bets, one pot-sized raise and all-in, and one
subsequent raise cap are used. No bucket, neural continuation, online resolve or
zero-reach pruning is introduced.

## Candidates and bounded pilots

- `flop-20bb-ace-high`: Ah 8d 3c, 20bb initial depth, 12bb pot, 14bb behind.
  BTN AA/AQs/KQs/QQ/JTs/76s versus BB 88/AJs/QTs/99/54s/33, with equal inclusion
  per unblocked exact combination.
- `flop-100bb-low-paired`: 7d 2h 2c, 100bb initial depth, 60bb pot, 70bb behind.
  BTN AA ×1, A7s ×1, A2s ×0.75, KQo ×0.5, 98s ×0.75 versus BB 77 ×1, 87s ×1,
  66 ×0.75, A5s ×0.5, QJs ×0.5, 22 ×1. Inclusion multipliers are not action
  frequencies; exact private pairs use the compatible joint prior.

An allocation-free native audit precedes separate 128-update pilots. A rejected
pilot produces only an immutable report, not a serving policy. Longer candidates
use the identical declared game; the source records the accepted update count.
Publication requires unchanged ≤0.05bb summed unilateral best-response gain,
≤0.0001bb maximum available-action EV discrepancy, valid probabilities, and fresh
native re-import verification of the actual exported binary bytes. Unavailable
EVs at zero compatible opponent reach remain null and ungraded.

The existing 32-combo-per-seat, 500,000-decision, 2,000,000-total-node, 512MiB
accumulator, 2MB manifest and 65,536-byte shard limits are not raised. Work is
local; task-scoped idle-sleep holds around active runs avoid Mac idle sleep
without changing permanent power settings. Independent pilots may run in
parallel processes; this is not an isolated throughput benchmark.

Verification will run the six existing exhaustive TypeScript checks for each
new study, the production build with its normal artifact verification, and
desktop/mobile browser flows. The full CI suite is not rerun merely to commit;
GitHub Actions remain PR/manual-only. Deployment continues through Git-connected
builds rather than a plain CLI source upload. No database, paid compute or new
cloud credentials are needed.

## Preflight and short-run results

| Study | BTN / BB combos | Compatible private deals | Ordered deal–runouts | Exact accumulator bytes |
| --- | --- | ---: | ---: | ---: |
| 20bb ace-high | 24 / 23 | 517 | 1,023,660 | 79,182,592 |
| 100bb low-paired | 27 / 21 | 524 | 1,037,520 | 80,867,328 |

Each complete tree has 104,872 decision nodes and 250,374 total nodes, including
chance and terminal nodes. They retain every supported combination even after
zero-reach actions. The audit's legacy monolithic-reader warning is expected;
the installed flop reader uses bounded version-pinned binary shards.

| 128-update pilot | Summed response gain (bb/spot) | Maximum EV error (bb) | Available action values checked | Result |
| --- | --- | --- | ---: | --- |
| [20bb ace-high](reports/2026-11-09-flop-20bb-ace-high-128.json) | 0.04909782296 | 0.0000009536743 | 4,935,741 | Passed unchanged gate |
| [100bb low-paired](reports/2026-11-09-flop-100bb-low-paired-128.json) | 0.2232663183 | 0.0000038146973 | 4,979,757 | Rejected; no artifact |

The 20bb margin is narrow but passes the existing exact conditional gate; its
candidate keeps 128 updates. The 100bb candidate advances to 512 updates without
changing its ranges, betting tree or tolerances. The pilot reports embed the
original inputs, so they remain reproducible after the published source changes.

Concurrent pilots took 58.27s / 59.04s wall and about 285MB / 290MB maximum RSS.
Independent scalar/vector profile discrepancies were ≤2.67e-15bb. Respectively
6,291 / 36,923 rows had no compatible opponent reach and remain ungraded.
These are local contended timings, not clean throughput or full-hand results.

## Accepted exports

The 20bb export passed fresh byte re-import verification at 128 updates. Its
frozen-policy and action-EV fingerprints exactly match the already-passing
pilot. The unchanged response gain is 0.0490978229593bb; maximum EV discrepancy
is 0.00000095367432bb. Its 1,957 binary shards total 64,040,431 bytes, the manifest
is 235,355 bytes, and the largest shard is 36,130 bytes. Broader combination
support can exceed the usual 32KB target, but remains within the existing
65,536-byte hard cap. No cap or game abstraction was changed.

20bb manifest SHA-256:
`f0c96617cb999f7fcc17dcf82beb92bcc4b789cde3e2f87bcae2ef10ba008781`.
Export plus fresh re-import took 75.37s wall and about 513MB maximum RSS while
the independent 100bb export was running. This is a contended local observation.

The 100bb 512-update export passed at **0.0222002273969bb** summed response gain,
about 90.06% below its own 128-update pilot. This is an iteration comparison
within the identical conditional game, not a full-hand exploitability result.
Maximum checked EV discrepancy is 0.00000381469727bb. The 1,957 binary shards
total 61,517,355 bytes, the manifest is 235,655 bytes and the largest shard is
37,306 bytes. Export plus fresh re-import took 218.96s wall with about 511MB
maximum RSS. No regrets or resumable training state are served.

100bb manifest SHA-256:
`20583af52c709afaeffc9e48d101ffb5a7b8faf6686f390f52f92e8dd7fe2cc3`.

The complete library now contains 19 studies: 11 flop-to-river games, four
turn-to-river games and four river games. The serving corpus has 39,627 files
totaling 892,758,487 uncompressed bytes. The earlier 17 studies and active
full-hand artifacts are unchanged. Initial loading still fetches only the
selected pinned manifest and root shard, using the existing 16-shard cache.

## Focused verification

- `npx vitest run lib/solved-flop.test.ts -t flop-20bb-ace-high`: six passed,
  54 unrelated cases skipped. The independent exhaustive EV/response test took
  about 44.8 seconds.
- `npx vitest run lib/solved-flop.test.ts -t flop-100bb-low-paired`: six passed,
  60 unrelated cases skipped. Its exhaustive EV/response test took about
  38.1 seconds. Each study also received all-public-branch, exact joint-prior
  interval, bounded decoding, loading/retry and pot-raise legality checks.
- `npm run build`: passed. The normal build re-verified all 19 saved studies
  from their actual bytes, checked unchanged active full-hand artifacts, and
  completed Next.js compilation and type validation. No full CI suite was run.
- Both new manifest URLs returned immutable one-year cache headers and retained
  COOP/COEP isolation. No serving schema or action grid changed.

Browser QA uses browser-harness and an isolated local headless Chrome profile,
not the user's signed-in session. Its table audio was initially audible despite
the hidden window; it was muted as soon as the user reported this. Per the
user's request, `AGENTS.md` now requires headless Chrome with `--mute-audio`, an
isolated profile and loopback-only CDP for subsequent browser QA, unless the
user explicitly requests a visible browser. This standing rule lives in the
existing browser-validation section.

Browser play-throughs completed both new games from both seats. The 20bb BB
flow checked the flop, called 6bb on the turn and the remaining 8bb on the river;
the BTN flow called 6bb on the flop and an 8bb turn all-in. The 100bb BTN flow
reached a 70bb river all-in/call, while the BB flow checked through all streets.
All settled at showdown. Opponent cards remained hidden until terminal review;
no mix or EV loss appeared before the first user decision. The 100bb details
panel displayed all declared weights, distinguished inclusion from action
frequency, and showed the pinned version hash.

A deliberately blocked 100bb binary shard at the flop-to-turn boundary paused
play without grading. The fourth board card and exactly two checks were
preserved. Retry kept the same board, hero cards and action history, produced
exactly one first-decision grade, and continued to showdown. No guessed policy
or duplicated action was substituted.

Desktop/mobile screenshots were inspected. Light/dark layouts at 375, 768, 1024
and 1440px had no horizontal overflow; visible mobile action controls remained
48px tall above navigation. Reduced-motion play and keyboard actions worked,
and “Review hand” focused the feedback region with visible focus. Native select
changes used the existing DOM change-event workaround, so this does not newly
certify native select-keyboard behavior. There were no captured unhandled
application exceptions or practice-inference API calls. Serving remained static
and lazy. The task-owned Chrome instance was closed after QA.

No full-hand exploitability claim, paid compute, database or universal 100bb
activation follows from these conditional-game results. CI stays PR/manual-only.
