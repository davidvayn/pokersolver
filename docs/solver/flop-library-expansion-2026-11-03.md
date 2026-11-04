# Flop library expansion — 2026-11-03

These are additional conditional saved games, not replacements for the earlier
studies or a release of a universal 50bb/100bb policy. Starting ranges are
illustrative and disclosed in each study; no solved preflop line is claimed.
Each study jointly solves all later streets and ordered public card arrivals.
No independent street policies are stitched together at playback.

The unchanged gates apply separately to each new artifact: summed unilateral
best-response gains ≤0.05bb per declared starting spot, every available action
EV checked within 0.0001bb, normalized probabilities, exact card removal and
chip conservation. Zero compatible opponent reach remains explicitly ungraded.
No bucketing, omitted chance edges, neural continuation or paid compute is used.

## Part 1 — Broader Qs 9h 6c ranges

`flop-50bb-broader` retains the original 50bb depth, 32bb pot, 34bb behind and
half-pot/pot/all-in sizing, with one subsequent raise. Only the declared range
support changes:

- BTN: QQ / QJs / T8s / AQs / TT — 19 unblocked combinations, previously 7.
- BB: 99 / JTs / QTs / 66 / A9s — 16 unblocked combinations, previously 7.

The original seven-combination supports are subsets of the new supports.
There are 259 compatible private deals and 512,820 deal–ordered-runout outcomes,
versus 45 and 89,100 in the original study. The complete public betting tree
still contains 104,872 decision nodes and 250,374 nodes including chance and
terminals; changing ranges does not remove any public action history.

The [128-update pilot](reports/2026-11-03-flop-broader-128.json) passed at
0.04739230849bb summed response gain, with maximum action-EV discrepancy
0.000001907349bb. It independently checked 3,088,149 action values and retained
298,051 zero-opponent-reach rows as unavailable. It took 31.43s wall time and
about 200MB maximum resident memory on the local 16GiB Apple M5. This is one
bounded pilot measurement, not a general training throughput claim.

The published candidate uses 512 updates to leave more margin than the short
pilot. Its serving format, hash-pinned lazy runtime and fixed resource bounds
are unchanged. The exact TypeScript validation suite now runs against **every
installed flop study**, rather than only the first catalog entry. It decodes one
study at a time so expanding the corpus does not retain all trees in memory.
Chip-conservation assertions use each study's actual depth, not a fixed 50bb.

The final 512-update artifact passed at **0.004005255458bb** summed response gain,
with maximum stored-EV error **0.000001907349bb**. Compared with the same
128-update game, more training reduced the response gain by about 91.5%; this
does not compare full-hand strength or different starting distributions.
It contains 1,957 shards totaling 48,831,411 bytes; the largest is 29,714 bytes,
below the existing 65,536-byte limit without changing the game or schema.
The manifest identity is
`97bd42530d7c1073e4ece106feaa2eb0a918f547be5130823387dc44a292c2c9`.
The original flop artifact was regenerated with its unchanged manifest hash;
the eight earlier river/turn catalogs and artifacts remain untouched.

Part 1 verification: 363 TypeScript tests passed (20 optional integration cases
skipped), and production build passed with fresh native re-import checks of all
10 studies. No Rust algorithm or full-hand model changed. Browser Use played
the broader game from both seats through bet/call, all-in runout and showdown.
Before acting, mixes/EVs and opponent cards were hidden; feedback and terminal
review revealed only their appropriate information. The initial fetch was one
234,635-byte manifest plus a root shard, not the full corpus. No resolver API or
captured unhandled JavaScript exception occurred. Layout checks at
375/768/1024/1440px in both themes found no horizontal overflow. Mobile keyboard
actions and Review hand worked with visible feedback focus and reduced motion.
Native macOS selects used the existing DOM change-event workaround; native
select-keyboard interaction is not newly certified.

## Part 2 — Ace-high dry and paired flops

Two new textures use the same action abstraction and complete-tree semantics:

| Study | Board | Initial depth / pot / behind (bb) | BTN / BB combinations | Compatible deals | Ordered deal–runouts |
| --- | --- | --- | --- | --- | --- |
| Ace-king-high | As Kd 7h | 100 / 60 / 70 | 14 / 13 | 155 | 306,900 |
| Paired king-high | Kh 7d 7c | 50 / 32 / 34 | 12 / 15 | 166 | 328,680 |

The ace-high study declares BTN AKs/AQs/QQ/76s versus BB 77/KQs/AJs/65s. The
paired study declares BTN KK/KQs/A7s/QJs versus BB 87s/KJs/88/98s. They add
different board/blocker and value/bluff situations; their ranges are still
illustrative, not certified public ranges from preflop. The 100bb study is not
a universal 100bb policy activation. Each public tree has 104,872 decision nodes
and retains all legal betting histories and ordered turn/river cards.

Both 128-update pilots failed the unchanged 0.05bb summed-response-gain gate:
[ace-high](reports/2026-11-03-flop-100bb-dry-128.json) at 0.1041256370bb and
[paired](reports/2026-11-03-flop-50bb-paired-128.json) at 0.0702701243bb. Every
available EV was independently checked, with maximum errors 0.000003814697bb
and 0.000001907343bb respectively. Neither short pilot exported a serving
artifact. The paired jobs ran concurrently, taking about 102s wall and 145–147MB
maximum RSS each; these contended observations are not clean serial throughput
or a comparison against the different broader-range game.

The final 512-update candidates are exported in separate native processes so
independent work can run in parallel without retraining the unchanged earlier
studies. These use the same exporter, acceptance gates, content-addressing and
fresh re-import check as `flops:build`. Catalog activation happens only after
successful export and re-verification. `flops:build` remains the full
deterministic reproduction command for all registered sources.

Both final candidates passed without changing any gate:

| Study | 512-update summed response gain (bb/spot) | Maximum EV error (bb) | Binary bytes | Largest shard |
| --- | --- | --- | --- | --- |
| 100bb ace-king-high | 0.02844397887 | 0.000003814697 | 43,337,403 | 25,042 bytes |
| 50bb paired king-high | 0.004514383693 | 0.000001907349 | 42,382,435 | 25,042 bytes |

Their independent native export/re-import processes completed in 99.32s and
97.35s wall respectively, with maximum RSS about 342MB / 346MB. This is measured
local execution under different contention, not a speed comparison with the
short pilots. Each candidate has 1,957 shards and the same fixed byte caps as
the earlier flop studies. No existing artifact was replaced or deleted.

Manifest identities:

- `flop-100bb-dry`:
  `1588f03f14ac267e83c9d36245ffb68a555330afbf92fbec4ab00bdcbe2437dc`.
- `flop-50bb-paired`:
  `e32f3cfd5c6f695de1116e730414782f2f6bcd56ed3b90d5618260a76e686eea`.

The library now contains four complete flop games, four complete turn-to-river
games and four river games. Their different starting ranges, boards and depths
make their response bounds distinct conditional measurements, not a full-hand
exploitability measurement or a ranking of universal playing strength.

The complete twelve-study library totals 178,412,666 uncompressed bytes,
including manifests. Only the selected study's manifest and required shards
are loaded. Static immutable hosting remains sufficient; no database, permanent
cloud credentials, paid compute or online inference was added.

Part 2 checks: 371 TypeScript tests passed (20 optional integrations skipped),
including complete independent EV/response and public-tree checks for all four
flop studies. The 17 saved-spot Rust release tests passed, including immutable
round-trip/corruption tests, exact card arrival memory, hidden-card-consistent
responses and zero-reach handling. Production build passed with fresh native
verification of all 12 studies and the unchanged full-hand artifacts.

Browser Use played both new studies from both seats. The flows included a
complete flop → turn → river line, turn betting/calling and flop/turn all-in
runouts, with terminal settlement and hidden cards until review. A blocked
paired-flop shard retained the hero cards and exactly one Bet 16.0bb in history,
displayed no EV grade, then Retry produced one feedback record without
reapplying the bet. Pending study selection preserved the 100bb hand until
New deal. Browser traffic fetched only pinned static manifests/shards, with no
practice-resolver API requests or captured unhandled JavaScript exceptions.

Both new studies passed 375/768/1024/1440px light/dark overflow checks. Mobile
action buttons were 48px tall and cleared bottom navigation; keyboard actions,
visible feedback focus and reduced-motion playback worked. Native select input
used the same previously documented DOM event fallback. Immutable caching and
COOP/COEP response headers were retained. The original nine studies, full-hand
model files, runtime schema and numerical acceptance gates remain unchanged.

## Reproduction

```sh
npm run flops:build
npm test
npm run build
```

The pilot report embeds its complete 128-update input. Save that embedded input
to a new JSON file to reproduce the pilot with `solved-flop-pilot`; the published
source intentionally records the final 512-update candidate instead.
`flops:build` regenerates registered flop studies and re-imports every exported
byte before updating the catalog. Production builds freshly verify all native
artifacts; the TypeScript tests separately reproduce every available EV and
both information-set-consistent best responses with the website evaluator.
