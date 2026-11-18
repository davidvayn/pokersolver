# Live 20bb Solver: conditional Nash-distance benchmark

Date: November 17, 2026 (America/Los_Angeles).

Status: **all three exact turn cases completed; flop benchmark incomplete**.
The first flop continuation packet hit the explicitly bounded 600-second cost
pilot ceiling. No incomplete packet or missing branch was scored. The user
subsequently asked to skip the several-hour full flop sweep because the turn
sample already established a substantial gap on the tested spots. The full
sweep never started, and no benchmark worker remains running.

## What was measured

The live Solver route uses
`hu-20bb-v102-consensus-continual-resolver-experimental`, not the newer static
solved-spot library. It resolves BB actions with two updates on each street and
uses the pinned learned BTN policy. Merely measuring a bilateral internal
resolver solution would evaluate a different policy.

The test-only adapter calls the same `PracticePolicyEngine` / `FrozenPolicy`
implementation and reads the actual active manifest, action abstraction, DCFR
parameters and four artifact identities. The canonical artifacts were freshly
verified with `node scripts/practice/verify-resolver-artifacts.mjs`.

Each public policy row is frozen at its original policy-conditioned belief.
An attacking response does not give the opponent its changed private range.
Best responses are information-set consistent: they observe their own exact
cards, the action history and currently revealed public cards, not the other
player's cards or a future runout.

The response pass uses actual f64 served frequencies without another f32 export.
It preserves all card-legal private combinations, modeled actions, and future
river cards. Showdowns and betting accounting supply the evaluation payoffs;
the learned continuation EVs displayed in the app are **not** accepted as
terminal ground truth. The two-thread resource setting matched the normal
eight-thread serving profile with **zero probability and range difference**
at every frozen flop decision in all three cases.

## Definition and scope

Here “Nash distance” means NashConv, the sum of each player's unilateral
best-response gain:

`NashConv = sum_i (V_i(BR_i, policy_-i) - V_i(policy))`.

For heads-up zero-sum play, exploitability is `NashConv / 2`, following
[OpenSpiel's definition](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/exploitability.py).
Both quantities below are **bb per conditional starting spot**, not bb per
complete hand and not a distance between action-frequency vectors.

The three cases were fixed before observing scores: one limped paired board,
one single-raised rainbow board and one three-bet monotone board. This is
illustrative pot/texture coverage, not a random population sample or a certified
representative distribution. Ranges come from the complete live-policy action
trajectory with exact card removal, not illustrative hand-class priors.

## Completed turn results

All cases start with equal 20bb stacks, 0.5/1bb blinds, no rake or ante. All
turn cases condition on **BB check, BTN check on the flop**. The turn response
includes all remaining turn betting and the complete river continuation.

| Preflop line | Flop → turn | Starting pot | NashConv (bb/spot) | Exploitability (bb/spot) |
| --- | --- | ---: | ---: | ---: |
| BTN limp; BB check | As Ad 7h → 3c | 2bb | 5.970743490808 | 2.985371745404 |
| BTN raise to 2.5; BB call | Qs 9d 2c → 6h | 5bb | 6.427284513137 | 3.213642256569 |
| BTN raise to 2.5; BB raise to 7.5; BTN call | 9c Tc Jc → Ad | 15bb | 3.308723486927 | 1.654361743464 |
| Equal-weight descriptive mean | Three fixed cases | — | **5.235583830291** | **2.617791915145** |

| Case | BTN response gain | BB response gain | Frozen profile values (BTN / BB) | Public decision nodes |
| --- | ---: | ---: | --- | ---: |
| Limped paired | 2.683911269815 | 3.286832220993 | 3.318559144982 / -3.318559144982 | 5,584 |
| Single-raised rainbow | 2.676464053315 | 3.750820459822 | 3.722070697837 / -3.722070697837 | 2,990 |
| Three-bet monotone | 1.603765056798 | 1.704958430129 | -1.560757793445 / 1.560757793445 | 776 |

These cases are far from equilibrium in the modeled game. Nevertheless, the
mean is **not full-game exploitability**: the roots have equal illustrative
weight rather than authentic reach weights, and conditioning on check/check
can emphasize rare lines. There is no population confidence bound from three
fixed cases. A flop continuation score and a conditional turn score also cannot
be subtracted to obtain a pure “flop leak”; their ranges and conditioning differ.

The largest probability-sum error was 4.45e-16, and the largest absolute
zero-sum profile residual was 6.67e-15bb. Turn jobs took 190.845s / 109.111s /
43.329s with sampled peak footprints about 1.206GB / 0.692GB / 0.239GB. Some
jobs overlapped; these are observed costs, not a clean throughput benchmark.

## Why the flop result is still unavailable

A full flop result must cover every legal future turn, all relevant betting
histories and both exact best responses. Future-card integration must happen
**before** an earlier-street response action is maximized. Scoring one selected
turn as the full flop game, silently dropping hard branches, or using the
value-network cutoff as a true terminal payoff would answer a different question.

The three frozen flop trees contain 45 / 57 / 33 total betting nodes and
9 / 11 / 5 distinct nonterminal turn-entry histories, respectively. For each
flop, all 49 card-legal public turns are required (45 are compatible with any
particular private deal). Each turn entry then includes all 48 public river
cards (44 compatible with a private deal).

The limped flop's first turn-card packet completed five of its nine turn-entry
histories in 200.001s, 129.717s, 31.584s, 103.817s and 6.042s. The worker was
stopped at 600.383s by its wall-time guard, not by an invalid strategy or
failed Nash gate. Its sampled peak footprint was 1,855,129,472 bytes, below the
2GiB guard. The final portion overlapped other turn jobs, so this is not a
clean latency projection. It is sufficient to establish that full coverage
is a several-hour local computation rather than a cheap decision query.

Only the independent turn results are scored. There is no accepted aggregate
flop number, no populated overall `summary.json`, and no claim that the flop
distance is zero or below a release threshold. Continuing requires an explicit
resource-budget revision or independently checkpointed turn-entry jobs; the
existing coordinator correctly refuses to start its full sweep without a
completed preflight. Preserve these completed turn receipts instead of
retraining or repeating them for a different policy.

## Reproduction and evidence

Test-only implementation:

- `preflop-solver/src/blueprint/neural/live_nash_benchmark.rs`.
- `preflop-solver/src/blueprint/public_belief/live_policy_response.rs`.
- `preflop-solver/neural/run_live_nash_benchmark.py`.

The only edits to the production modules are `#[cfg(test)]` module declarations;
the serving code, frozen artifacts, model manifest, game tree and website UI
are unchanged. No full CI suite was run, and no commit or push was requested
for this measurement task.

Focused release-mode tests passed:

1. The new f64 adapter reproduces the existing frozen-turn evaluator's profile
   and both response gains within 1e-10. Rescaling either raw reach vector
   scales only the appropriate counterfactual values and preserves conditional
   response gains.
2. The three preflop lines, check/check turn selection, and flop profile / max /
   opponent-sum backups pass deterministic fixtures.

The first two setup attempts were unscored: A used `call` instead of the native
`limp` label; B selected check/check without the engine's street/actor history
prefixes and trailing `deal:Turn`. Both diagnostic mistakes were corrected
before C's completed measurements; no model was trained, selected or promoted.

Evidence is retained locally under ignored directory
`preflop-solver/neural/runs/live-solver-nash-20261117-c/`:

- `protocol.json`, SHA-256
  `a7769aef694a24f03c38fed2de26a8ee489612c7f59d6ac276c3c5120c3f9a2b`.
- Native test binary SHA-256
  `06164fc4045d5faa01eaa96212179cd633ecb102aaf90fda53c973d24c553e90`.
- Each case's `base.json`, `direct-turn-result.json`, and successful native
  job receipts with command, environment, output hash, time and memory evidence.
- Failed packet receipt and partial progress logs; no complete packet output.
- `diagnostic-inputs.tar.gz`, preserving the diagnostic source files, manifest,
  serving-command adapter and frozen native test binary before another build
  can replace the target-directory executable.

Direct-turn result hashes, in table order:

1. `d34448c4c8b883f94931726b25bd57cb2516d3be4420fb8d3b6a384154720637`.
2. `f38adf74ebd1991d4ece7cd1176960f75a684261de405e11238b8412a4fa6322`.
3. `2eeb9ed52310b0ff1feca158882a44c57896210cb549ff5701b62ac272df7ccc`.

No paid compute, database, external account, model activation, CI dispatch or
browser session was used. Task-scoped sleep-prevention assertions ended with
their controllers; no permanent power settings were changed.
