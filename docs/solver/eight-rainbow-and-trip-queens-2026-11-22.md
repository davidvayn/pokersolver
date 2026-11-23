# Broader eight-high rainbow and trip-queens flops — 2026-11-22

Two new rake-free studies use 32 weighted, board-legal private combinations
per seat. Both retain the existing half-pot and pot bets, one modeled pot
raise, all-in, exact card removal, and every ordered turn and river.

| Study | Flop | Pot / behind | BTN / BB combos | Compatible deals | Ordered deal–runouts |
| --- | --- | ---: | ---: | ---: | ---: |
| [20bb eight-high rainbow](../../data/solved-spots/flop-20bb-eight-rainbow.json) | 8d 4s 2c | 12 / 14bb | 32 / 32 | 939 | 1,859,220 |
| [100bb trip queens](../../data/solved-spots/flop-100bb-trips-queens.json) | Qc Qd Qs | 60 / 70bb | 32 / 32 | 864 | 1,710,720 |

The low rainbow adds sets, two pair, overpairs, gutshots and unpaired ace-high
hands. The trips-board adds full houses, kicker battles, and quads in both
ranges, with exact card removal excluding deals in which both hold the
remaining queen. The priors are illustrative inclusion weights, not action
frequencies or solved preflop ranges.

The structural audits each counted 104,872 public decisions, 250,374 total
nodes, and 107,823,104 bytes of sparse training accumulators. No range, node
or memory limits were increased. These games are conditional on the declared
starting ranges and modeled sizes; they do not change the live Solver or
activate a full-hand strategy.

## Native acceptance

The 20bb candidate passed at 256 updates. The 100bb candidate was rejected by
the existing export gate at 256 updates after 167.90s and produced no serving
shards. The exporter does not return a per-gate measurement on rejection.
The 512-update retry changed only that budget and passed. Both accepted
versions passed fresh native re-import of the actual exported binary bytes.

| Study | Updates | Summed best-response gain (bb/spot) | Max available action-EV error (bb) | Binary bytes | Manifest bytes | Largest shard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20bb eight-high rainbow | 256 | 0.01183159618564 | 0.00000095367433 | 77,904,663 | 235,936 | 46,331 |
| 100bb trip queens | 512 | 0.02329656307140 | 0.00000381468162 | 79,261,119 | 235,967 | 46,650 |

Both pass the unchanged 0.05bb summed-response and 0.0001bb EV checks.
Maximum probability-sum error is 5.03e-8, and the zero-sum residual is
below 7.11e-15bb. These root scores are range-weighted within each modeled
game, not maximum errors or EV-loss bounds at every deviated line.

- 20bb manifest: `487ce1f2517ac6571b3dcd4df6140057e71c29acfe79097d23ba8ef3aba2f4ec`.
- 100bb manifest: `86d812eeaa8ab7b009209386ccad617c530c92e26542226dcf232628341a0bed`.

Accepted export and re-import took 178.26s and 329.67s wall, with peak
physical footprints of approximately 643MB and 682MB. These are local
observations under different concurrent load, not an iteration-scaling benchmark.

Each game has 1,957 shards plus a manifest; all are inside existing byte
limits. The library now has **37 studies: 29 flops, four turns and four
rivers**, in 74,871 serving files totaling 2,260,001,454 uncompressed bytes.
Prior entries and bundles are preserved. Clients still request only the
chosen manifest and needed immutable shards.

## Verification

Both new manifests served one-year immutable caching and COOP/COEP. Isolated
muted headless Chrome finished the 20bb desktop BB hand by calling 6bb on
the flop and checking down. A deliberate blocked shard paused without EV
feedback, and Retry preserved the board and first action and graded it
exactly once. The 100bb mobile BTN checked down to showdown through the
complete board. Both confirmed pinned manifest versions, keyboard review
focus, and no live resolver requests. Native select values were set via
DOM change events.

The mobile picker stayed within the layout and the 48px action controls
remained above navigation. Light/dark and 375/768/1024/1440px checks passed,
with normal motion on desktop and reduced motion on mobile. The existing
favicon 404 was the only observed browser error.

`next build` passed production compilation, type checks and all 14 static
pages. All 12 targeted Vitest tests passed (six per spot), including
independent exact action EVs and both best responses, all public betting
branches, exact joint deals, shard bounds and lazy retry. No full CI suite
was run.
