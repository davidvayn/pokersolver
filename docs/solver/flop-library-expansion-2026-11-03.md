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
