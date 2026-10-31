# Complete flop library: structural pilot

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

Next: test positive-support accumulator storage against the dense turn solver,
requiring identical policies/EVs and unchanged artifacts before enabling it for
saved studies. Full flop training and chunked serving remain separate work;
the value-network-based live flop resolver is not an exact saved-game substitute.

Checks for this checkpoint: 349 TypeScript tests passed (20 optional integration
cases skipped); 477 library, 9 binary and 16 CLI release tests passed (60
resource/fixture cases ignored). `npm run build` passed, including fresh native
verification of all eight unchanged saved studies and active full-hand artifacts.
The new CLI test checks immutable reports and rejected flop policy export. This
is offline-only work; no browser UI or serving schema changed.
