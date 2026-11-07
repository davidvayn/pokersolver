# Monotone and connected flop expansion — 2026-11-06

This batch adds two missing flop textures to the conditional saved-spot library.
It does not replace prior artifacts, train a universal stack-depth model, or
measure full-hand exploitability. Both starting ranges are declared study
assumptions, not outputs of a certified earlier betting line. Each candidate
jointly solves the entire flop-to-river game with exact private-card removal,
ordered public card arrivals, and the existing half-pot/pot/all-in sizing plus
one subsequent raise. No neural leaves, bucketing or independent street solves
are introduced.

## Declared games and short pilots

| Study | Board | Depth / pot / behind (bb) | BTN / BB combos | Compatible deals | Ordered deal–runouts |
| --- | --- | --- | --- | --- | --- |
| Monotone ace-high | As 9s 4s | 50 / 32 / 34 | 20 / 17 | 286 | 566,280 |
| Connected hearts | 9h 8h 7c | 100 / 60 / 70 | 18 / 19 | 312 | 617,760 |

Each game has 104,872 public decision nodes (250,374 nodes including chance and
terminals). The native census projects 62,335,232 bytes for its exact regret and
average accumulators, within the unchanged structural resource cap.

The monotone ranges are BTN AA/A9s/KQs/JTs/TT versus BB 99/AQs/QTs/T8s/44, with
equal unblocked-combination weights. This introduces made flushes and single
spade blockers alongside sets, pairs and unpaired holdings.

The connected ranges use explicit per-combination inclusion multipliers:

- BTN: JTs ×1, 99 ×1, 98s ×0.75, AJs ×0.5, 65s ×0.5.
- BB: 77 ×1, T6s ×0.75, 97s ×0.75, QTs ×0.5, TT ×0.5.

Public blockers are removed, each seat's prior is normalized, and private pairs
are dealt from the product conditioned on compatibility. Multipliers are not
action frequencies. All supported combinations remain in the solved game;
there is no low-reach pruning. The game includes made straights, sets, two pair,
an overpair, and heart/straight draws. The priors are illustrative, not claimed
preflop or flop population frequencies.

Both 128-update pilots failed the unchanged ≤0.05bb **summed** response-gain gate:

| Pilot | Summed response gain (bb/spot) | Maximum checked EV error (bb) | Available action values checked |
| --- | --- | --- | --- |
| [Monotone](reports/2026-11-06-flop-50bb-monotone-128.json) | 0.05640244552 | 0.000001907341 | 3,549,016 |
| [Connected](reports/2026-11-06-flop-100bb-connected-128.json) | 0.1365032215 | 0.000003814697 | 3,392,360 |

Neither pilot produced a serving artifact. Their EV checks passed the unchanged
0.0001bb maximum discrepancy gate. Zero-opponent-reach rows remained unavailable
(173,069 monotone and 251,305 connected), not zero-loss grades. Concurrent local
pilots took 35.74s / 34.00s wall with about 218MB / 225MB maximum RSS. These are
contended observations, not general throughput benchmarks.

## Publication and validation

Final candidates use 512 updates and the existing native exporter. Independent
exports can run in separate processes; the catalog is extended only after the
actual binary bytes pass a fresh native re-import and exact response/EV checks.
No existing acceptance gate or serving byte limit is relaxed.

The TypeScript suite already checks every installed flop's available EVs, both
information-set-consistent best responses and every public betting branch using
the independent website evaluator and hand engine. A new deterministic test also
visits the midpoint of **every compatible private pair's joint-prior interval**,
verifying the exact dealt cards and a unique 52-card deck. This covers weighted
and uniform studies without relying on noisy empirical frequency tolerances.

Full reproduction remains `npm run flops:build`, followed by `npm test` and
`npm run build`. Each pilot report embeds its complete short-run input; the
published sources intentionally record the final candidate's iteration count.
Static hash-pinned shards and the bounded browser cache remain the serving path.
No database, paid compute or permanent cloud credentials are needed.

## Accepted 512-update artifacts

| Study | Summed response gain (bb/spot) | Maximum EV error (bb) | Binary bytes | Largest shard |
| --- | --- | --- | --- | --- |
| Monotone ace-high | 0.004376828433 | 0.000001907349 | 51,460,319 | 30,882 bytes |
| Connected hearts | 0.008223727091 | 0.000003814697 | 51,735,187 | 30,882 bytes |

The same games' response gains fell about 92.2% and 94.0% from their short
pilots. These are within-game iteration comparisons, not improvements in the
universal full-hand model or comparisons between different starting priors.
Both artifacts passed without changing the game, precision, validation gates
or existing 65,536-byte shard cap. Each has 1,957 binary shards and a manifest
of 234,654 / 234,823 bytes respectively. The largest shards remain under 32KB.

Concurrent native exports, including fresh re-import verification, took
146.51s / 147.01s wall with about 433MB / 439MB maximum RSS. These measurements
include other local checks and are not isolated throughput benchmarks.
The exporter saves no regrets or resumable training state.

Manifest identities:

- `flop-50bb-monotone`:
  `482b1c96499acc38981eff347b9e21980261be06467b5903185c51fdafcf3cfb`.
- `flop-100bb-connected`:
  `8f93a68de4070b6c4a6fc4502754348a468b00902821e9089fd02d18b587fe22`.

The earlier twelve studies and active full-hand artifacts remain unchanged.
The library now has six conditional flop games, four turn-to-river games and
four river games. Total uncompressed serving storage, including manifests, is
282,077,649 bytes. The browser still fetches only the selected pinned manifest
and needed shards, with the unchanged 16-shard cache.

## Final checks

- `npm test`: 385 passed, 20 optional integration cases skipped. Every installed
  flop received full EV, best-response, public-tree and joint-prior interval
  checks, not just sampled decisions or the first catalog entry.
- `cargo test --release --manifest-path preflop-solver/Cargo.toml solved_spot`:
  all 17 saved-spot tests passed, including immutable re-import/corruption,
  hidden-hand response consistency and ordered card arrival memory. No native
  solver algorithm, generated WASM or full-hand model source changed.
- `npm run build`: passed, including fresh native verification of all 14
  published studies and unchanged full-hand artifact integrity checks.

Browser Use played both new games from both seats. Flows included all three
streets, turn betting/calling and all-in runouts to showdown. Initial loading
fetched only each selected manifest and root shard, and the model stayed pinned
through each hand. Opponent cards stayed hidden until terminal review; mixes
and EVs appeared only after the user's action. The weighted range panel showed
every declared multiplier and distinguished inclusion from action frequency.

The connected study's next binary request was intentionally blocked on mobile.
Cards and exactly one Check in history remained visible with no EV grade.
Retry resumed the same cards/history and produced exactly one feedback record.
The browser debug transport disconnected once before that extra test; reattaching
and repeating it completed successfully. No captured unhandled JavaScript
exception or practice-resolver API request occurred in the saved-hand flows.

Both new studies passed 375/768/1024/1440px light/dark overflow checks. Mobile
keyboard actions, visible feedback focus and reduced-motion playback worked.
Native macOS selects used the existing DOM change-event workaround, so native
select-keyboard behavior is not newly certified. Immutable caching and
COOP/COEP headers were verified on both manifests. These are conditional saved
game checks, not certification of arbitrary flops or full-hand GTO.
