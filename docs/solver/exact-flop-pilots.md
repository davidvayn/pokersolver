# Exact flop-to-river pilots

## Scope (2026-11-01)

`solved-flop-pilot` trains one joint conditional game from a declared flop.
It does not call a value network or solve separate turn/river policies after
fixing a flop strategy. It is an offline research command, not a new serving
schema, library activation, arbitrary-board web solver or full-hand GTO claim.
All eight existing saved studies and live resolver weights remain unchanged.

The first two inputs deliberately contain only seven starting combinations per
seat, with 45 compatible private deals. Every pair has 45 × 44 ordered public
runouts. They test implementation and cost, not broad range coverage:

| Input | Initial pot / behind | Public decisions | Regret + average arrays |
| --- | --- | ---: | ---: |
| `flop-50bb-exact-small` | 80bb / 10bb | 9,608 | 1,980,160 bytes |
| `flop-50bb-exact-sizing` | 32bb / 34bb | 104,872 | 23,586,304 bytes |

In the first case stack clipping collapses bets to all-in. The second retains
non-all-in half-pot/pot bets, raises, later-street betting and all-in alternatives.
All modeled legal actions come from the existing native hand rules. No action,
low-reach holding or chance outcome is pruned to clear a quality threshold.

## Implementation and checks

- Each public betting trajectory has separate decision nodes; ordered turn/river
  arrivals are retained even when their final five-card sets coincide.
- Compact accumulator rows retain every board-legal positive root-range combo.
  Zero action reach does not remove an information set. Regrets and average
  sums remain f64; only the frozen policy used for checking is quantized to f32.
- Board-dependent rank/equity data are cached separately from decision histories.
  A blocked private pair contributes zero. Public chance has 49/48 candidates,
  but values integrate the 45/44 outcomes legal for each exact private pair.
- Alternating DCFR updates use alpha=1.5, beta=0, gamma=2 and discard the first
  10% of rounds from the average. Averages use the player's own realization
  reach. The weighting basis follows [CFR, equation 4](https://papers.nips.cc/paper_files/paper/2007/file/08d98638c6fcd194a4b1e6992063e944-Paper.pdf);
  the discounts follow [Brown and Sandholm's DCFR](https://arxiv.org/abs/1809.04040).
  The recurrence matches the existing tested native turn solver.
- The frozen f32 policy is renormalized in f64, as serving would do. A scalar
  best-response traversal integrates hidden opponents before choosing an own
  action and branches on future cards only after their public arrival.
- This checker enumerates its own showdown runouts rather than reading the
  training equity cache. Its two profile values must agree with a separate
  cached vector traversal to 1e-8bb; profile payoffs must conserve to 1e-8bb.
  Both paths still share the native hand rules, card model and evaluator.
- The report records the **sum** of both unilateral gains; the existing
  conditional response-gain target remains ≤0.05bb/spot, not half that sum.
  Passing this diagnostic alone does not authorize serving. The initial v1
  reports lacked action-EV checking; v2 adds the checks below. Bounded, pinned
  policy shards and browser serving verification still remain outstanding.

Tests reproduce every decision's frozen frequencies against the existing
river/turn solver on matched inputs. Further tests cover repeatable full-flop
policy hashes, exact footprint/array counts, ordered-arrival memory, private
blockers, invalid probabilities, budget rejection and immutable CLI reports.
A synthetic fold/call case would yield 2bb to a responder that could see the
opponent's holding; the information-set-consistent checker correctly gets 0bb.
Corrupting the training equity cache changes the cached profile but not the
independent checker's result.

## Run safely

```sh
npm run build:solver
preflop-solver/target/release/preflop-solver solved-flop-pilot \
  --input data/solved-spots/pilots/flop-50bb-exact-small.json \
  --output /tmp/exact-flop-small-new.json \
  --max-decision-nodes 10000 --max-seconds 120
preflop-solver/target/release/preflop-solver solved-flop-pilot \
  --input data/solved-spots/pilots/flop-50bb-exact-sizing.json \
  --output /tmp/exact-flop-sizing-new.json \
  --max-decision-nodes 110000 --max-seconds 120
```

Both budget flags are mandatory. The structural census rejects excessive
decision counts before tree allocation. Additional fixed caps are 32 starting
combos per seat, 512MiB of regret/average elements, two million total tree nodes,
1,024 rounds and 600 seconds. A cooperative deadline is checked during board
construction, training, independent checking and profile evaluation. These
limits are not an OS-enforced RSS ceiling; metadata, frozen policy and checker
caches consume additional memory. Measure peak process memory separately.

Only an immutable research report is written. An unconverged completed pilot
may record `conditionalResponseGainGatePassed: false`; that is useful research,
not permission to export a failed policy. `servingArtifactProduced` and
`fullHandGtoClaim` remain false. Historical v1 reports also set
`actionEvValidationCompleted` false; v2 sets it true only after independently
checking all available frozen action values and their reach flags.
`solved-spot` still rejects flop inputs, and the normal serving catalog is not
changed by this command.

## Release-mode results

Measured sequentially on the local 16GiB Apple M5, after builds/tests had
finished. Wall time and maximum RSS come from macOS `/usr/bin/time -l`; these
are individual pilot measurements, not a multi-machine speed guarantee.

| Game / rounds | Summed response gain (bb/spot) | Wall time | Max RSS |
| --- | ---: | ---: | ---: |
| High-pot smoke / 32 | 0 | 0.26s | 14,516,224 bytes |
| Multi-size / 32 | 0.549626540 | 3.31s | 80,347,136 bytes |
| Multi-size / 128 | 0.077538165 | 11.62s | 80,478,208 bytes |
| Multi-size / 512 | 0.006275099 | 43.69s | 80,379,904 bytes |

The multi-size 512-round [report](reports/2026-11-01-flop-exact-sizing-512.json)
records 0.125s construction, 43.006s training and 0.546s checking. It passes the
unchanged conditional response-gain threshold; its independent/cached profile
difference and payoff residual are both 2.7e-15bb. This is genuine progress on
a complete **narrow conditional flop game**, not on full-hand exploitability.
The smoke game's zero gain reflects an easy high-pot toy game and must not be
extrapolated to a broader policy. Its repeat reproduces every frozen-policy hash
and value exactly; only timing fields differ.

All four reports retain the complete effective input and its SHA-256. The
32/128/512 multi-size inputs differ only in iteration budget, including the same
10%-of-budget averaging delay convention. Failed 32/128-round thresholds are
preserved in the reports rather than omitted or relabelled as accepted.

Checks: 349 TypeScript tests passed (20 optional integrations skipped); 512 native
release tests passed (60 resource/fixture cases ignored); `npm run build` passed,
including fresh integrity and native verification of all eight unchanged saved
studies. This part is offline-only; no browser-facing contract was changed.

Next useful part at v1: generate complete conditional action-EV rows from the frozen
policy, independently verify their quantization/error and zero-reach handling,
then design bounded, hash-pinned serving shards. No cloud compute is justified
for these measured narrow pilots.

## Frozen action-EV completion (v2)

The vector profile pass now records each action's EV under the **frozen policy's
continuation**, not under a best-response continuation. It divides counterfactual
values by the compatible opponent reach for that exact own combo at that node.
The stored rows are f32, with an explicit availability bit per combo. A zero
opponent denominator means unavailable, not a zero EV or permission to grade.
Zero own-policy reach does not prevent evaluating a user's deviation.

A separate scalar pass compares every available stored action against direct
private-pair/runout enumeration. It also requires every availability bit to
match independently observed opponent reach. All available rows must be visited;
a missing row or a spurious value on a zero-opponent-reach row fails checking.
The unchanged maximum action-EV discrepancy gate is 0.0001bb. Policy and EV
hashes are separate, with the EV hash bound to the frozen policy hash.

Tests explicitly reject altered EVs, missing reachable values, and falsely
available zero-reach values. Exact enumeration has no Monte Carlo sampling
error; f32 rounding and the finite policy's remaining strategy gap still exist.
This is not an assertion that every action is an equilibrium action or that
the full preflop-to-river policy is solved.

The 512-round [v2 report](reports/2026-11-01-flop-exact-sizing-512-values.json)
checks **1,017,950 action values across 453,288 combo rows**. Maximum stored-EV
error is **0.00000190734864bb**, below the unchanged 0.0001bb gate. The other
221,904 rows have zero compatible opponent reach and remain explicitly
unavailable. They are not treated as successful zero-loss decisions.

The policy hash, both profile/response values and summed response gain exactly
match v1's 512-round result. A repeated smoke run also reproduces its new EV
hash. This completes value generation/checking, not a new strategy improvement.
The full v2 run took 51.46s and 108,642,304 bytes maximum RSS on the same local
machine; checking itself took 1.136s. No paid compute was used.

Final v2 checks passed: 513 native release tests (60 resource/fixture cases
ignored), 349 TypeScript tests (20 optional integrations skipped), and production
build, including fresh verification of the existing eight saved studies. Their
catalog and immutable artifacts are unchanged. No browser-facing changes were
made in either offline part.

Remaining before this flop study can enter the library: export the frozen
policy/EV data as bounded immutable shards, implement pinned lazy lookup and
fail-closed loading in the library, reproduce values with the TypeScript
rules/evaluator, and exercise the complete browser practice flow. Broader range
and depth coverage remain separate from this narrow conditional pilot.

## Serving and practice completion (2026-11-02)

Those serving prerequisites are now implemented. The accepted exact 512-update
game is exported and re-imported as [bounded immutable shards](flop-shards.md),
with independent TypeScript checks of all public nodes, every available EV and
both information-set-consistent best responses. Repeating the complete export
produced the same manifest identity:
`dda53afb0ca6af19e6be7dc17b6aad8dc4d41ac60203856914c361f688617c50`.

`/practice/spots` includes **50bb · Flop to river · Sets and draws** alongside
the eight unchanged studies. It uses one frozen average policy through all
three streets, not independently solved street policies. All modeled actions
remain available. Private deals, betting, board arrivals and settlement use
the shared hand engine. The UI immediately publishes the action/dealt board,
then prepares the required saved node. It holds a pending cursor during loading
or failure. Retry does not apply an action twice or redraw the hand; a failed
request cannot create an EV grade. Seat/spot changes remain next-deal changes.

Browser Use completed flop → turn → river → showdown from both seats, including
turn betting/calling in the mobile flow. Opponent cards remained hidden until
terminal review; mixes and EVs appeared only after a user decision. Blocking
the next binary request retained the exact cards and single Check in history,
showed no grade, and Retry produced exactly one feedback record for that action.
Initial loading requested the manifest and one root shard, not all 1,957 shards.
No online resolver or neural-policy API was requested by these saved-hand flows.

Checks at 375/768/1024/1440px in light/dark found no horizontal overflow. Mobile
controls remain 48px tall above navigation; Review hand is keyboard-operable,
focuses feedback and shows a visible focus ring. Reduced-motion playback was
exercised. Native macOS select input again required the documented DOM
change-event fallback; native select-keyboard behavior is not certified.
No captured unhandled JavaScript exception occurred. A pre-existing missing
`favicon.ico` produced a 404, unrelated to strategy playback. Immutable caching
and COOP/COEP headers were checked on a binary response.

This publishes one narrow conditional study, not the rejected 50bb universal
neural model, arbitrary-board solving, or a full-hand exploitability improvement.
Broader starting ranges and additional flop/depth situations still require their
own measured pilots, complete solves and independent acceptance.

Final checks: 359 TypeScript tests passed (20 optional integrations skipped),
514 native release tests passed (60 resource/fixture cases ignored), and
`npm run build` passed with fresh verification of all nine studies and the
unchanged active full-hand artifacts. The complete native suite ran after the
exporter change; no Rust source changed during UI integration.
