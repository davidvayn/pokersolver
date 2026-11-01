# Complete flop library: feasibility and lossless storage pilots

## Part 1 — Exact census (2026-10-31)

The new offline command uses the existing native betting engine to count a
complete public game **without training, card evaluation or regret allocation**:

```sh
preflop-solver/target/release/preflop-solver solved-spot-audit \
  --input data/solved-spots/pilots/flop-50bb-footprint.json
```

An optional `--output <new-report.json>` creates an immutable research report,
never a serving policy. Three-card inputs remain rejected by `solved-spot`
solve/export. No library entry, live model, rake rule or validation gate changed.

The checked-in [report](reports/2026-10-31-flop-50bb-footprint.json) is for
Qs 9h 6c, initial 50bb depth, 12bb pot and 44bb behind. It uses the declared
30 / 27 combo ranges from the broader turn exercise, half-pot/pot/all-in bets,
one pot-sized raise plus all-in, and one subsequent raise cap. These are
illustrative ranges, not a certified earlier betting line.

| Count | Flop | Turn | River | Total |
| --- | ---: | ---: | ---: | ---: |
| Public decision nodes | 14 | 3,038 | 409,248 | 412,300 |
| Public action edges | 38 | 7,742 | 992,544 | 1,000,324 |
| Served combo rows | 399 | 83,049 | 10,721,358 | 10,804,806 |
| Per-action values | 1,083 | 211,641 | 26,002,374 | 26,215,098 |

There are 729 compatible starting private-card pairs, each with 45 × 44 = 1,980
ordered legal turn/river runouts: 1,443,420 pair–runout outcomes. The 619-state
action-history skeleton is traversed only once. For a node after `d` new public
cards, its public copies are `(52 − root board length) falling-factorial d`.
Each active private combo excludes two additional cards, so its row copies use
`50 − root board length` instead. Terminal all-ins stop the decision tree;
their later runouts belong to settlement, not extra decision nodes. Ordered
arrival and the full action trajectory are retained, not merged by final board.

The census was cross-checked against all eight published river/turn artifacts:
node counts, rows, action values/edges, public chance edges and compatible deals
match their actual exported trees. Debug/release reports are deterministic. This
validates counting, **not strategy quality or solving time**.

### First bottlenecks

- Dense reference `RangeNode` allocates two f64 arrays of `1326 × action count`
  at every public node. Their data alone would be **21,222,873,984 bytes**
  (21.2GB / 19.77GiB), before tree/history metadata, rank layouts, caches,
  snapshots or scratch space. It cannot fit a 16GB machine as represented.
- The same two arrays with only board-legal positive-support combos would need
  **419,441,568 bytes** (419MB / 400MiB). This is a target for an exact storage
  change, not a measured full-game peak-RAM result. No low-reach combos need
  be discarded and no information sets need to be merged to obtain it.
- A hypothetical fixed-layout policy with a u16 combo key and two f32 arrays
  per row is **231,330,396 bytes**, excluding metadata/indexes. This assumes
  action EVs are present; null EVs and compression change size. It is not a
  measured JSON artifact. The current reader's 4,096-node limit is exceeded
  regardless, so a full flop policy also needs bounded, pinned shards.

Next at this checkpoint: test positive-support accumulator storage against the dense turn solver,
requiring identical policies/EVs and unchanged artifacts before enabling it for
saved studies. Full flop training and chunked serving remain separate work;
the value-network-based live flop resolver is not an exact saved-game substitute.

Checks for this checkpoint: 349 TypeScript tests passed (20 optional integration
cases skipped); 477 library, 9 binary and 16 CLI release tests passed (60
resource/fixture cases ignored). `npm run build` passed, including fresh native
verification of all eight unchanged saved studies and active full-hand artifacts.
The new CLI test checks immutable reports and rejected flop policy export. This
is offline-only work; no browser UI or serving schema changed.

## Part 2 — Lossless saved-turn accumulators (2026-10-31)

Saved turn studies now allocate f64 regrets and average-policy accumulators only
for the declared root range's positive-support combinations, minus public-card
blockers. Combinations with zero reach after an action are **still retained**.
No betting action, chance outcome, information set or floating-point precision
is removed. Canonical combo IDs and dense terminal/traversal/export vectors are
unchanged. The only change is where accumulator rows live in memory.

A compact private module owns combo-to-row maps, shared by every betting history
with the same board and actor. The turn pilot uses 98 small maps, not one map per
node. Fresh saved-turn solves opt in; live resolving, safe gadgets, warm starts,
river-only solves and frozen-policy readers keep their existing dense path.

The `--dense-reference` solve-only CLI option retains the original allocation
for differential tests and repeatable measurements. Both paths apply the same
convergence and EV gates before exporting, with the same immutable artifact
schema and identity. Verification rejects this benchmark-only option.

```sh
preflop-solver/target/release/preflop-solver solved-spot \
  --dense-reference --input data/solved-spots/turn-50bb-weighted.json \
  --output /tmp/dense-new.json
preflop-solver/target/release/preflop-solver solved-spot \
  --input data/solved-spots/turn-50bb-weighted.json \
  --output /tmp/compact-new.json
cmp /tmp/dense-new.json /tmp/compact-new.json
```

Tests compare every active combo's regret and average accumulator bit-for-bit,
both with DCFR and regret-matching+, including delayed averages. They check
shared map identity and actual array lengths against the independent footprint
census: **65,854,464 → 1,357,056 bytes** for the 1,356-node weighted turn game.
That is a 97.94% reduction in these two arrays, not in total process memory.

### Measured 512-round comparison

The [recorded pilot](reports/2026-10-31-turn-compact-storage.json) runs the same
weighted study in four separate release processes on the 16GiB Apple M5:
dense, compact, compact, dense. No repository build/test/training jobs overlapped
the timed solves. Times include construction, training, action-value export and
independent verification, not just accumulator updates. macOS `/usr/bin/time -l`
reports maximum resident set size in bytes.

| Run | Wall time | Maximum resident memory |
| --- | ---: | ---: |
| Dense 1 | 29.87s | 224,559,104 bytes |
| Compact 1 | 29.22s | 149,045,248 bytes |
| Compact 2 | 25.95s | 148,979,712 bytes |
| Dense 2 | 28.62s | 224,346,112 bytes |

Mean maximum RSS falls **33.61%**, about 224MB → 149MB. Mean wall time falls
29.245s → 27.585s (5.68%), but two repeats are insufficient for a statistical
speed claim. This is primarily a demonstrated memory improvement.

All four outputs are byte-identical to the existing 3,546,345-byte artifact
`8f0773fcd61a486fadd5d86ec0bad3cead11b74c8aaa4f42966e62461ad462cf`.
Its summed response gain remains 0.005083669496069199bb/spot and maximum
action-EV discrepancy remains 0.0000025893798110132593bb. No strategy gate was
relaxed to obtain these results.

All eight registered studies were regenerated and independently verified;
catalog and artifact hashes remain unchanged (13,203,890 total bytes). Final
checks passed: `npm test` (349 tests, 20 optional skips),
`cargo test --release --manifest-path preflop-solver/Cargo.toml` (480 library +
9 binary + 16 CLI tests; 60 resource/fixture cases ignored), and `npm run build`,
including active-model integrity verification.
No browser UI or serving contract changed in this part.

This prepares a more economical representation; it does **not** implement or
release a complete flop study. The flop projection remains 419,441,568 bytes
for accumulators alone. Full flop rank/chance/traversal costs still need bounded
pilots, and the projected 412,300-node policy needs pinned, bounded shards rather
than raising the current monolithic reader limits. There is no new full-hand
exploitability result or 50bb/100bb full-hand policy activation.
