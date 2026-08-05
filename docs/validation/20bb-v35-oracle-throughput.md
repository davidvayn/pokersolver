# 20bb v35 continuation-oracle throughput

Status: research infrastructure improved; no model was activated.

This change attacks the measured runtime and resumability blockers in the
range-conditioned continuation generator. It does not improve the frozen turn
value weights, and it does not turn the existing `0.540214bb` value uncertainty
into release-quality evidence.

## Batched shared-combo inference

The Rust runtime now evaluates every legal exact combo as a dense batch instead
of issuing thousands of individual dense-layer calls. It uses
`matrixmultiply` 0.3.11, an MIT/Apache-2.0 portable SIMD matrix kernel, while
retaining the existing deterministic CPU implementation and model schema.

The first accepted benchmark used the same two source deals, 49 preflop leaves
per deal, v31 seed 10902 turn network, v28 seed 8801 preflop ranges, two flop
resolver iterations, exact flop all-ins, and eight requested threads as the
v34 value-only benchmark.

| 98-leaf exhaustive smoke | v34 scalar | v35 batched | Change |
| --- | ---: | ---: | ---: |
| Wall time | 359.45s | 150.70s | 2.39x faster |
| User CPU time | 1,818.73s | 870.04s | 2.09x less |
| Peak resident memory | 285.0MB | 278.1MB | 2.4% less |

The complete validation summary was unchanged. Batched summation changed 34
persisted floating-point scalars, with a maximum absolute delta of
`9.5367431640625e-7bb`; the independent scalar-versus-batch layer test permits
at most `1e-6`. Repeated runs remain deterministic for a fixed binary and
machine.

Two follow-up implementations were rejected rather than retained. Reusing
per-combo scratch allocations took 419.77s despite producing a byte-identical
artifact. Splitting and caching the structural part of the first query layer
did not complete by 179.61s, and a dynamic-only query representation took
173.79s. The measured-fast full-query batch is the production path.

The board feature cache now holds all 49 turn boards for each of the 16 flop
boards supported by the dense exact-all-in cache. This prevents feature-cache
thrashing when multiple source deals are solved concurrently. A larger-deal
scheduler benchmark remains necessary before using a linear full-cycle runtime
projection.

## Resumable exact-deal chunks

`preflop-cache-resolver` now accepts `--deal-offset`. Resolver-derived caches
record the SHA-256 of the immutable source cache and the exact source-deal
indices included in the chunk. Validation rejects missing paired provenance,
invalid digests, length mismatches, and duplicate indices.

Merging provenance-aware chunks now:

- requires identical source cache, frozen value network, network list, game,
  public leaves, and range-policy mixture;
- sorts output deals back into source order;
- rejects overlapping source indices; and
- marks an exact-combo cycle balanced only when every unique position in that
  source cycle is present.

This permits bounded jobs and safe recovery without allowing duplicated or
incompatible fragments to masquerade as a complete exact-combo cycle.

The end-to-end check generated source offsets `0..2` and `2..4` independently.
The second board slice took 225.55s, demonstrating meaningful board-dependent
runtime variance. Their merged cache contained exactly four deals in source
order with indices `[0,1,2,3]`; merging the first chunk with itself exited with
`continuation cache chunks overlap source deal indices` and wrote no accepted
cache.

## Verification and remaining blockers

- Rust release tests: 70 library tests and 3 CLI tests passed.
- Dense batch inference matches scalar inference within `1e-6`.
- Suit equivariance, exact combo masking, exact all-in settlement, zero-sum
  projection, source-cycle coverage, provenance ordering, and overlap rejection
  all have passing regression tests.
- The accepted kernel dependency is free and open source (MIT/Apache-2.0).

The current v31 value model remains rejected. Its attached uncertainty makes
the continuation action-SE gate pass for 0% of generated leaves. The remaining
strategic blockers are a substantially more accurate range-conditioned value
model, paired independent continuation corpora, paired tabular preflop solves,
exact-all-in matched resolver selection, materially higher learned-response
coverage, and the independent one-sided 99% full-game exploitability upper
bound. Activation remains fail-closed until all of those gates pass.
