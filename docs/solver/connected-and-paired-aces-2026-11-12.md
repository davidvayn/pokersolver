# Broader connected and paired-aces flops — 2026-11-12

This batch expands two stack/texture combinations: a 20bb connected two-heart
game and a 100bb paired-aces game. Both use broader weighted priors including
offsuit broadways. These are declared study assumptions, not certified earlier
preflop lines or universal full-hand policies. Existing studies, active models,
solver algorithms, betting abstraction and acceptance gates remain unchanged.

## Declared games

- `flop-20bb-connected`: Th 9h 8c, 20bb initial depth, 12bb pot, 14bb behind.
  BTN QJs ×1, JJ ×1, ATs ×1, T9s ×0.75, 76s ×0.75, KQo ×0.5 versus BB 88 ×1,
  JTs ×1, QTs ×0.75, J7s ×0.75, 97s ×0.5, AJs ×0.75, 99 ×1, KJs ×0.5,
  T8s ×0.75. Made straights, sets, two pair, pairs and heart/straight draws
  provide a more coordinated starting board at 20bb.
- `flop-100bb-paired-aces`: Ad As 7c, 100bb initial depth, 60bb pot, 70bb behind.
  BTN KK ×1, AQs ×1, KQo ×0.5, JTs ×0.5, 77 ×1, AA ×1, A7s ×0.75 versus
  BB AJs ×1, 76s ×0.75, QQ ×0.75, QJs ×0.5, T9s ×0.5, 88 ×0.5, 65s ×0.5.
  Quads, full houses, trips, pocket pairs and unpaired holdings remain in the
  same jointly solved conditional game.

Multipliers describe inclusion per exact combination, not action frequencies.
Public blockers are removed before normalization; private pairs use the exact
compatible joint prior. Each policy jointly solves all three streets with
complete betting history, exact card removal and ordered public card arrivals.
The half-pot/pot/all-in bet sizes, pot-sized raise plus all-in and one-subsequent-
raise cap are unchanged. No neural leaf, bucketing, online resolve, or zero-reach
pruning is introduced.

## Preflight, pilots and acceptance

An allocation-free census precedes separate 128-update pilots. Only candidates
passing ≤0.05bb summed unilateral best-response gain, ≤0.0001bb maximum available-
action EV discrepancy, valid frozen probabilities and fresh native re-import
of the actual binary bytes may enter the catalog. Zero compatible opponent
reach remains explicitly unavailable and ungraded. Rejected pilots produce
reports, never serving policies; longer candidates keep the identical game.

The existing 32-combo-per-seat, 500,000-decision, 2,000,000-total-node, 512MiB
accumulator, 2MB manifest and 65,536-byte shard hard limits stay unchanged.
The earlier 23 studies and full-hand artifacts are preserved. Verification
uses the six exhaustive TypeScript checks per new study, normal production-build
artifact verification and browser-harness QA in isolated muted headless Chrome.
The full CI suite is not rerun merely to commit; Actions stay PR/manual-only.
Serving remains static, lazy and hash-pinned with the existing bounded cache;
deployment uses Git-connected builds, not plain CLI source upload. No database,
paid compute or new credentials are needed. Idle-sleep holds around active runs
do not change permanent Mac power settings. Concurrent timings are local
observations, not clean throughput benchmarks or full-hand exploitability data.

## Structural preflight

| Study | BTN / BB combos | Compatible private deals | Ordered deal–runouts | Exact accumulator bytes |
| --- | --- | ---: | ---: | ---: |
| 20bb connected | 32 / 29 | 809 | 1,601,820 | 102,768,896 |
| 100bb paired aces | 29 / 29 | 763 | 1,510,740 | 97,714,688 |

Both complete games contain 104,872 public decision nodes and 250,374 total
nodes including chance/terminals. They fit the existing caps without removing
any supported combination, action or ordered public-card arrival. The audit's
legacy monolithic-reader warning does not apply to the installed bounded
flop-shard reader. The census measures structure, not strategy quality.

## Short pilots

| 128-update pilot | Summed response gain (bb/spot) | Maximum EV error (bb) | Available action values checked | Result |
| --- | --- | --- | ---: | --- |
| [20bb connected](reports/2026-11-12-flop-20bb-connected-128.json) | 0.02764165615 | 0.0000009536735 | 5,433,985 | Passed |
| [100bb paired aces](reports/2026-11-12-flop-100bb-paired-aces-128.json) | 0.1904333074 | 0.0000038146973 | 5,218,054 | Rejected; no serving artifact |

The 20bb candidate retains 128 updates. The 100bb candidate advances to 512
updates because its pilot is almost four times the unchanged response-gain
limit. Its board, priors, betting tree and numerical tolerances stay fixed;
the pilot report embeds the original input and remains reproducible after the
published source's update count advances.

Concurrent pilots took 84.03s / 78.34s wall with about 366MB / 354MB maximum RSS.
Independent scalar/vector profile discrepancies were ≤7.11e-15bb. Respectively
494,216 / 444,540 rows had no compatible opponent reach and remain unavailable
and ungraded. These are contended local measurements, not general solving-time
benchmarks or full-hand exploitability results.

## Accepted exports

The 20bb connected 128-update export passed fresh native re-import of all bytes.
Its frozen policy and EV fingerprints exactly match the passing pilot; summed
response gain remains 0.0276416561539bb and maximum EV discrepancy
0.000000953673482bb. Its 1,957 binary shards total 73,955,559 bytes, the manifest
is 236,145 bytes and the largest shard is 44,890 bytes, within the unchanged
65,536-byte hard cap. Some broader-range shards exceed the usual 32KB target;
chunk layout, precision and limits were not changed to hide that storage cost.

20bb manifest SHA-256:
`7e45d9da54dcba08ed9a0a3e433cde960f2a580115e0ccec4b5f8f360a504434`.
Export plus fresh re-import took 97.84s wall with about 656MB maximum RSS while
the independent 100bb candidate ran. This is a contended local observation.

The 100bb paired-aces 512-update export passed at **0.0195466260839bb** summed
response gain, about 89.7% below its own 128-update pilot, with maximum EV
discrepancy 0.00000381469727bb. This compares iterations of the identical
conditional game, not different priors or full-hand exploitability. The 1,957
binary shards total 72,406,207 bytes, the manifest is 235,981 bytes and the
largest shard is 43,146 bytes, within the unchanged hard caps.

100bb manifest SHA-256:
`dce6edd067c8ad89ed1eb54a6caa90190d720be1ce0b41460e188467ae3ab4e2`.
Export plus fresh re-import took 285.81s wall with about 646MB maximum RSS.
Serving includes only frozen probabilities and checked EVs, not regrets,
resumable state or fabricated continuation values.

The library now has 25 studies: 17 flop-to-river games, four turn-to-river
games and four river games. The 51,375 serving files total 1,330,381,921
uncompressed bytes. The earlier 23 studies and active full-hand artifacts
remain unchanged. A hand lazily loads only its pinned manifest and needed
shards with the existing 16-shard cache, not the full corpus.

## Focused verification

`npx vitest run lib/solved-flop.test.ts -t flop-20bb-connected` passed all six
checks; 90 unrelated cases were skipped. Its independent exhaustive EV/response
check took about 50.4 seconds. Coverage includes every exported action EV,
both information-set-consistent best responses, every public branch, exact
joint-prior interval dealing, bounded decoding, loading/retry and pot-raise
legality. Existing cases and tolerances were not modified.

`npx vitest run lib/solved-flop.test.ts -t flop-100bb-paired-aces` passed all six
checks; 96 unrelated cases were skipped. Its exhaustive EV/response check took
about 45.1 seconds. The same independent all-node, all-EV and both-response
coverage applies, not only the browser's sampled decisions. Both new manifest
URLs retained immutable one-year caching and COOP/COEP headers.

`npm run build` passed, including fresh native verification of all 25 saved
studies, integrity checks for unchanged full-hand artifacts, TypeScript checks
and Next.js production compilation. No full TypeScript/Rust CI suite was run;
the PR/manual-only workflow configuration is unchanged.

On mobile, a real binary request was blocked at the connected game's flop-to-
turn transition. The dealt turn, hero cards and exactly two public checks
remained visible without an EV grade. Retry preserved that state, graded the
first decision exactly once, and resumed through a 12bb river bet/call to
showdown. Keyboard “Review hand” visibly focused the feedback region. There
was no guessed policy, duplicated action or replacement random runout.

Desktop BB play of the 20bb game bet 6bb, faced an opponent 14bb all-in, and
called the remaining 8bb to showdown. The 100bb BB flow bet 30bb, faced a 70bb
all-in and called the remaining 40bb; its BTN flow checked through all three
streets to showdown. Opponent cards remained hidden until terminal review and
mixes/EV loss stayed hidden before the first user action. Both details panels
displayed every declared inclusion multiplier separately from action frequency,
with the pinned manifest identity.

Browser-harness used a fresh local Chrome profile launched with
`--headless=new --mute-audio` and loopback-only CDP. The task-owned instance was
closed after QA. Desktop/mobile screenshots were inspected. Light/dark layouts
at 375/768/1024/1440px had no horizontal overflow; visible mobile action controls
remained 48px tall above navigation. Reduced-motion play, keyboard actions and
visible feedback focus worked. Native selects used the existing DOM change-event
workaround, so native select-keyboard behavior is not newly certified. No
unhandled application exception or practice-inference API request was captured.

The response-gain bound is conditional and prior-weighted, not a maximum error
for every off-path decision or a full-hand exploitability certificate. User
deviations can reach decisions with large action-EV differences; those values
still enumerate the same frozen continuation policy and are not capped by the
small root response-gain figure. These results do not certify unrestricted
ranges, an earlier preflop line or universal 20bb/100bb play.
