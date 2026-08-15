# 20bb v103 action-EV variance reduction

Date: 2026-08-14

## Result

The action-EV precision gate remains failed and the full-hand manifest remains
inactive. The existing ten-cycle estimator has 30.3754% policy-action-weighted
coverage at 0.02bb standard error, versus the required 95%. Doubling the same
chance design to twenty independent exact-combo cycles produced 30.3379%
coverage. Its worst standard error fell from 5.3670bb to 3.5279bb, but the
threshold coverage did not move because roughly 29.5% of the weight is on
deterministic terminal actions while sampled continuations remain far above
0.02bb.

This rejects additional unstructured exact-deal cycles as the next release
run. At the observed p95 error, square-root scaling would require an
impractical number of cycles.

## Retained exact-combo continuation vectors

The public-belief flop resolver already computes normalized continuation
values for both players and all 1,326 exact private combinations. The legacy
preflop bridge previously selected one value for each sampled player's cards
and discarded the other 2,650 values. V103 retains and validates the complete
vectors, allowing a Rao-Blackwellized evaluator to integrate the compatible
opponent range instead of sampling one opponent hand.

A two-flop end-to-end cache verified serialization and exact-card validation.
The range-integrated flop-leaf diagnostic still measured a 0.7757bb weighted
median SE and 2.9474bb p95 SE. Opponent-card sampling was therefore not the
dominant remaining variance source; public flop chance is.

## Canonical flop chance design

The new canonical enumerator proves that 1,755 suit-isomorphic flop orbits,
with orbit sizes 4, 12, or 24, cover all 22,100 raw Hold'em flops exactly.
Every orbit retains its multiplicity, and unit tests require the orbit weights
to sum to 22,100.

Exact card-removal-aware reach under the frozen preflop policy is highly
concentrated:

- total probability of a nonterminal flop leaf: 30.8042%;
- 9 of 49 leaves cover 96.1132% of conditional flop reach;
- 33 leaves cover 99.0689% of conditional flop reach.

This changes the exhaustive high-reach projection from 49 × 1,755 range
solves to 9 × 1,755. It does not alter the game or claim coverage for omitted
low-reach leaves.

An eight-orbit, top-nine-leaf pilot retained 3,042 player/hand-class/leaf
groups in a 705KiB compressed artifact. It measured 0% coverage at 0.02bb,
0.8994bb weighted median SE, and 1.5152bb p95 SE. Eight orbit clusters are far
too few for threshold precision; only complete canonical enumeration removes
flop sampling error in the frozen abstract model.

## Throughput and next implementation change

The eight-orbit pilot took about 30 minutes while saturating eight CPU cores
and using about 586MiB RSS. Dense exact flop all-in equity construction is the
dominant kernel. Linear scaling projects roughly 110 hours locally for all
1,755 orbits, despite the top-nine leaf restriction. The projected compressed
range-value payload is only about 155MiB, so compute—not hosted storage—is the
constraint.

Each canonical board can now be emitted as an atomic resumable shard using a
deterministic orbit offset. The provenance-checked merge rejects overlapping
boards and any policy, value-network, game, range, or resolver mismatch. A
two-board benchmark with two concurrent boards and four resolver threads per
board took 429.9 seconds, or 3.58 minutes per board. This was only about 4.5%
faster per board than the eight-board pilot and does not materially change the
exhaustive projection.

Two exact-semantics kernel experiments were rejected rather than retained:
bitmask/symmetric dense-equity updates took 453.1 seconds, and triangular
zero-sum updates took 446.0 seconds on the identical two boards. Replacing the
repeated scalar terminal matrix-range product with the existing optimized
matrix backend also regressed to 467.9 seconds. These controls indicate that
the dominant remaining work is repeated full postflop traversal/value
inference for each distinct preflop leaf, not the tested equity-loop surfaces.

Leaf-level profiling then found that concurrent boards were contending for the
same cores. On one canonical board, one high-reach leaf took 12.77 seconds,
two leaves took 26.71 seconds, and all nine took 102.26 seconds with eight
resolver threads. Giving a single board all ten local cores reduced the
nine-leaf time to 91.50 seconds. A complete 49-flop-leaf board took 171.31 to
196.62 seconds. The range-cache scheduler now defaults to one board worker,
records the actual ten resolver threads in provenance, and emits version-two
single-board shards that can be resumed independently. The observed complete
endpoint projection is approximately 90 hours for all 1,755 canonical boards
on this host, not the earlier multi-board extrapolation.

Version two also includes all 50 distinct preflop all-in showdown endpoints.
For each canonical flop it reuses the exact equity matrix and enumerates every
compatible turn/river; a fixed-combo reference test matches the direct 990
runouts. This closes the forced-all-in action-value gap without measurable
extra board time.

The new canonical action-value pass propagates those 99 endpoint vectors
(49 flop lines plus 50 all-in lines) through the frozen preflop policy. It
integrates exact compatible private ranges and board blockers at every
information set and reports the declared policy-reach/action-frequency
weighted SE. On a real two-orbit pilot it evaluated 8,450 information sets per
player with 100% lookup coverage. Precision remains intentionally failed at
31.8152% coverage and 12.6767bb maximum SE; two flops are only a plumbing
check, not validation evidence.

The resumable exhaustive shard run remains necessary before the action-EV gate
can pass. Complete 1,755-orbit enumeration makes finite-game chance-sampling
SE exactly zero; approximation error from the frozen resolver/value network
remains separate and must continue to carry a low-confidence warning.

No activation status, gate threshold, or served fallback was changed.
