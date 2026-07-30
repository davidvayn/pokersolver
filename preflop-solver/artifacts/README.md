# Offline push/fold artifact set

Generated on 2026-07-24 with the native `preflop-solver` crate. These are
equal-stack, rake-free, heads-up push/fold abstractions. They are not ordinary
open/call/3-bet charts and are not full preflop Hold'em solutions.

## Accepted artifacts

| Stack | Iterations | Equity samples | Exploitability | SB shove | BB call |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2bb | 25,000,000 | 1,024 | 0.001013bb | 89.45% | 99.91% |
| 3bb | 25,000,000 | 1,024 | 0.002071bb | 77.81% | 91.98% |
| 5bb | 25,000,000 | 1,024 | 0.003920bb | 71.45% | 62.28% |
| 8bb | 25,000,000 | 1,024 | 0.005832bb | 62.17% | 45.21% |
| 10bb | 25,000,000 | 1,024 | 0.006931bb | 57.57% | 38.21% |
| 12bb | 25,000,000 | 1,024 | 0.007799bb | 52.38% | 33.74% |
| 15bb | 25,000,000 | 1,024 | 0.009085bb | 46.03% | 28.85% |
| 20bb | 50,000,000 | 1,024 | 0.007956bb | 39.83% | 22.34% |

Exploitability is half NashConv in big blinds per hand. The certificate uses
an independently sampled equity matrix. Every artifact is labeled
`approximate`; the Monte Carlo equity oracle prevents an exact-equilibrium
claim. The advisory acceptance limit is `0.01bb`, and the high-precision target
is `0.002bb`.

## Consistency checks

- Aggregate shove and call frequencies tighten monotonically as stacks deepen.
- AA is effectively 100% shove and call at every stored depth.
- Independent seeds at 5bb, 10bb, and 20bb produced exploitability differences
  below `0.000013bb`.
- Across those seed comparisons, aggregate range-frequency differences stayed
  below 0.09 percentage points. Individual boundary hands moved by as much as
  40 percentage points, so mixed frequencies near an indifference boundary
  should not be treated as high precision.
- The first 20bb run used 25 million iterations and failed at `0.010944bb`.
  Re-solving at 50 million iterations reduced the result to `0.007956bb`.

## Browser catalog

The full artifacts retain all 1,326 exact combos and total about 4.7MB. Run:

```bash
npm run preflop:catalog
```

The generator revalidates the artifact envelope, required solver checks,
combo keys, frequencies, canonical hand classes, and exact-to-class averages.
It writes the 68KB presentation catalog at
`data/preflop/solved-scenarios.json`. A database is not warranted for this
immutable data volume.

## Unsolved boundary

Calls after a non-all-in open require a range- and position-dependent
continuation value for future streets. A forced-checkdown or fixed
equity-realization shortcut can be solved mathematically but does not produce
credible deep-stack no-limit ranges. The app therefore keeps its existing
6-max and 9-max 100bb charts explicitly labeled as curated references instead
of presenting them as solver output.

## Experimental deep-stack blueprint

`preflop-solver blueprint` now trains a complete heads-up abstract betting tree
through the river rather than forcing non-all-in calls to check down. The
repository keeps compact cross-seed reports, while raw profiles and checkpoints
remain local because they range from tens of megabytes to several gigabytes.

The current 100bb model is deliberately quarantined from the browser catalog.
Independent 100,000-iteration seeds failed strategy-stability and deep-shove
sanity gates. Extending a resumable profile to 200,000 iterations improved
coverage but still left a 1.20bb aggregate first-action deviation estimate
(1.02bb one-sided 99% lower bound). That is evidence against treating the
profile as equilibrium guidance.

Run the deterministic comparison gate with:

```bash
npm run preflop:compare-blueprints -- \
  /path/to/seed-1.json \
  /path/to/seed-2.json \
  --require-pass
```

Passing this gate would establish only cross-seed root stability, poker-domain
sanity, coverage, and a small one-step local deviation. It would still not
certify full-game exploitability.
