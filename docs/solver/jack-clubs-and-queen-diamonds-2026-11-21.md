# Broader jack-club and queen-diamond flops — 2026-11-21

Two new conditional studies use the existing rake-free flop-to-river game:
half-pot and pot bets, a single modeled pot raise, all-in, exact blockers
and all ordered runouts. Weighted range assumptions are declared per exact
combination. They are not action percentages or certified preflop ranges.

| Study | Flop | Pot / behind | BTN / BB combos | Compatible deals | Ordered deal–runouts |
| --- | --- | ---: | ---: | ---: | ---: |
| [20bb jack-high clubs](../../data/solved-spots/flop-20bb-jack-high-clubs.json) | Jc 7c 2d | 12 / 14bb | 31 / 32 | 877 | 1,736,460 |
| [50bb queen-high diamonds](../../data/solved-spots/flop-50bb-queen-diamonds.json) | Qd Td 7d | 32 / 34bb | 32 / 32 | 868 | 1,718,640 |

Jack-high clubs contrasts a comparatively static rank texture with club
draws, sets, top pair and two pair. Queen-high diamonds mixes flopped flushes
and diamond blockers with sets, two pair and straight draws.

The structural audits count 104,872 public decisions and 250,374 total nodes
per spot, with 106,138,368 and 107,823,104 bytes of sparse accumulators
respectively. Both fit the unchanged 32-combo, 500,000-decision and 512MiB
accumulator caps. Earlier studies and the live full-hand model stay unchanged.

## Accepted exports

Both passed at 256 updates. Native verification re-imported the actual
serving bytes. Existing policies, tolerances and the action abstraction
were unchanged.

| Study | Summed best-response gain (bb/spot) | Max available action-EV error (bb) | Binary bytes | Manifest bytes | Largest shard |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20bb jack-high clubs | 0.00638601577680 | 0.00000095367432 | 78,016,783 | 235,969 | 46,058 |
| 50bb queen-high diamonds | 0.01500210385048 | 0.00000190734035 | 78,556,795 | 236,013 | 46,650 |

Both pass the 0.05bb summed-response and 0.0001bb EV checks. Maximum
probability-sum error is 4.94e-8; the zero-sum residual is below 1e-15bb.
These are conditional NashConv scores, not full-game exploitability. The
existing exporter may complete negligible-average rows uniformly, which
does not imply every action at such a node has the same EV.

- 20bb manifest: `3a09846839c158f67da635d5b30ccc3fd7e38ad7c61ce73a5b9c4c2b29a3bef0`.
- 50bb manifest: `4154b7f87f25d28f0003adaa87c79001a130f7872788eb947f9298b58d308d02`.

Concurrent exports and native re-imports took 594.75s and 594.53s locally,
with sampled peak footprints of about 676MB and 662MB. These are not general
throughput estimates. Each new game has 1,957 binary shards plus a manifest;
all shards fit the existing 65,536-byte cap. The corpus now has **35 studies:
27 flops, four turns and four rivers**, in 70,955 serving files totaling
2,102,363,769 uncompressed bytes. The browser still fetches only a pinned
manifest and needed shards; no database or online solve is used for these spots.

## Browser and build checks

Isolated muted headless Chrome completed the 20bb game from BB on desktop:
check/call the 6bb flop bet, check the turn, and call the remaining 8bb on
the river. A deliberate blocked flop shard paused without a grade; Retry
preserved the board and first action and graded it once. The mobile BTN
line called the 16bb flop bet and checked through turn and river to review.
Keyboard review focus, exact cards, feedback timing and pinned range
details passed. Native select values were changed using DOM change events.

Light/dark at 375, 768, 1024 and 1440px stayed inside the viewport. The
mobile picker was 44px tall, with 48px action buttons above navigation.
Desktop used normal motion and mobile used reduced motion. Both manifests
served one-year immutable caching and COOP/COEP. No live resolver requests
or application exceptions were observed; the unrelated favicon 404 remains.

`next build` passed production compilation, type checks and generation of
all 14 static pages. The 12 targeted Vitest checks passed, including exact
action EV and best-response recomputation for both bundles, legal branches,
joint-prior dealing, bounded decoding and lazy retry. No full CI suite was run.
