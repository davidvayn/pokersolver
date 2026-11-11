# Exact flop saved-strategy shards

`flop-50bb-draws` is a conditional, rake-free 50bb flop-to-river study, not a
full-hand equilibrium certificate. It starts on Qs 9h 6c with a 32bb pot and
34bb behind. Its deliberately narrow seven-combination ranges are declared in
`data/solved-spots/flop-50bb-draws.json`; they are not certified preflop ranges.
The four existing river and four turn studies keep their original bytes and
catalog. Flop studies use a separate registry and catalog.

For additional published range/board situations and their individual gates,
see [the 2026-11-03 expansion report](flop-library-expansion-2026-11-03.md)
and [the monotone/connected expansion](flop-texture-expansion-2026-11-06.md).
The [lower-pot expansion](lower-pot-flops-2026-11-07.md) adds larger betting
trees and documents their storage and deployment implications.
The [deeper 100bb expansion](deeper-100bb-2026-11-08.md) completes the lower-pot
stack set with a validated broadway study and a non-all-in 60bb pot raise.
The [broader ace-high/low-paired expansion](broader-flops-2026-11-09.md) adds
24/23-combo 20bb and weighted 27/21-combo 100bb games, including offsuit hands.
The [broader monotone/two-tone expansion](broader-textures-2026-11-10.md) adds
27/27-combo 20bb and weighted 32/30-combo 50bb games without raising range limits.
The original study's figures below remain its own, not corpus-wide totals.

## Reproduction and acceptance

```sh
npm run flops:build
npm run spots:verify
```

The exporter trains alternating DCFR with the existing exact flop implementation,
freezes its average policy as f32, and computes conditional action values against
that frozen policy. No regrets or resumable training state are served. It refuses
an existing output directory or a candidate failing the existing 0.05bb summed
best-response-gain / 0.0001bb maximum action-EV-error gates.

The verifier reconstructs an untrained native game from the source parameters,
re-imports every shard, checks every public edge, and independently enumerates
private deals and ordered runouts. It recomputes both information-set-consistent
best responses and all action EVs from the exported probabilities. It does not
accept the exporter's reported result on trust. Build verification also checks
the source, manifest, catalog, and content-address identities.

The accepted 512-update artifact has 250,374 public tree nodes (104,872 decision
nodes), 45 compatible private deals and 89,100 deal–ordered-runout outcomes.
Summed response gain is 0.006275098945bb **for this conditional spot**. All
1,017,950 available action EVs have maximum f32 error 0.000001907349bb.
The 221,904 zero-opponent-reach combination rows have unavailable EVs, not zero
EV estimates. The maximum probability-sum residual is 4.47035e-8.

TypeScript tests separately reproduce the complete frozen-policy profile,
information-set responses, and every available action EV using the TypeScript
hand evaluator. A second exhaustive pass checks every public betting/chance
branch against the shared practice engine and chip conservation. Additional
tests check bounded decoding, corruption, lazy loading, deterministic hidden-card
independence, retry without double application, and complete hands from both seats.

## Format

The content-addressed `solved-flop-sharded-v1` JSON manifest contains the source
input and its hash, frozen-policy and action-value hashes, exact starting ranges,
root/node counts, verification summary, and contiguous shard descriptors
`{ start, count, sha256, bytes }`. Node IDs are postorder: every child ID is lower
than its parent, and the root is the final node. Separate ordered card arrivals
and betting histories remain separate; there is no lossy state merging.

Each `.bin` shard contains at most 128 nodes and 65,536 bytes. All numeric fields
are little-endian. The header is ASCII `PSF1`, u32 start, u16 count. Each record is
prefixed by u32 byte length and contains:

- u8 tag (fold=0, showdown=1, chance=2, decision=3), u8 actor (255 if absent).
- u8 board length and five u8 card slots (unused slots are 255).
- Two f64 total investments and two f64 current-street bets, in bb.
- u16 edge count and u16 combination-row count.
- Chance edges: u8 card, u32 child.
- Decision edges: u8 UTF-8 action-ID byte length, ID bytes, u8 action kind
  (fold/check/call/bet/raise/all-in = 0..5), f64 target, u32 child.
- Decision rows: canonical u16 combination key, u8 EV-availability flag, one f32
  probability per action, and (only when available) one f32 EV per action.

There are no trailing bytes. Canonical cards and combination keys use the shared
repository representation. Decoder bounds, action arity, probability sums, finite
values, boards, combinations and edge IDs are validated before a node is served.

The current corpus is 29,719,439 binary bytes in 1,957 shards plus a 234,147-byte
manifest. The largest shard is 17,450 bytes. There is no database dependency.

## Runtime contract

`loadSolvedSpot` dispatches this format to a lazy pinned runtime. Initial loading
fetches the manifest and root shard, not the whole corpus. Each shard is checked
against its exact length and SHA-256. A 16-shard LRU bounds retained policy data;
in-flight requests are coalesced and a failed shard is explicitly reloaded on retry.

`beginSpotAction` applies one legal action synchronously with the shared hand
engine, so dealt cards and action history can be displayed immediately.
`prepareSpotSession` resolves the resulting immutable child/chance nodes before
any subsequent decision or feedback is served. Only the already-visible board
selects a chance edge. It never looks ahead at the hidden deck or opponent cards.
The manifest remains pinned throughout a hand. Missing/corrupt shards pause the
hand; they never produce a guessed strategy, uniform fallback, or EV grade.

The UI must retain the pending session on failure, retry preparation rather than
reapply the action, and reveal opponent cards only in terminal review. Root reloads
after cache eviction use the same immutable version. The old synchronous
`advanceSpot` interface deliberately rejects this format to prevent accidental
unprepared lookups.
