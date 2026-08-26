# 20bb multi-street policy-prior and cloud-run pilots

Date: 2026-08-25

## Scope and decision boundary

This research cycle tests policy changes only. It does not reactivate the
deferred exploitability gate, relabel the current model, or substitute a
two-deal matched estimate for a certificate. Every experiment preserves exact
card removal, legal actions, stack accounting, and the accepted action grid.

The retained exact-root research mechanism is an opt-in empirical policy prior
for the short flop, turn/river, and river public-belief solvers. Its routed
full-game candidates failed their matched promotion screens, so it is absent
from both serving and cloud defaults. The retained full-trainer change is
HS-DCFR(30) with an immutable 600,000-iteration horizon. A final 300k midpoint
pair on a larger diagnostic corpus improved both seeds and selected it for the
documented cloud command. Another
researched method, baseline-enhanced MCCFR, was implemented and tested at 500,
10,000, and 50,000 full-game blueprint iterations, then removed because both
50,000-iteration seeds made the policy-quality diagnostic worse.
The paper's less aggressive HS-DCFR(15) schedule was likewise screened and
removed after a higher-confidence paired comparison favored HS-DCFR(30).

Final disposition:

| Item | Decision | Reason |
| --- | --- | --- |
| Exact-root empirical prior 15 + two updates | Retain, research-only and opt-in | Preserves all 12 tested actor/street/texture roots and beats prior 8, but routed full-game profiles regress |
| HS-DCFR(30), 600k horizon | Retain as best scalar schedule | Improves the strongest fixed controls; terminal pair still fails release gates |
| HS-DCFR(15), VR-MCCFR, correlated chance sampling, 1M horizon | Remove/reject | Paired policy or coverage evidence is worse/mixed |
| Existing scalar cloud launch | Hold | A paid repeat would reproduce known policy-quality failures |
| Public-chance vector trainer | Next implementation candidate | Directly targets private-card variance and can reuse the exact range-vector subgame infrastructure |

## Research basis

[Strategy-Based Warm Starting](https://ojs.aaai.org/index.php/AAAI/article/view/10056)
shows why copying action probabilities without coherent regret/substitute-value
state is not a sound CFR warm start. ReBeL's
[supplement](https://proceedings.neurips.cc/paper/2020/file/c61f571dbd2fb949d3fe5ae1608dd48b-Supplemental.pdf)
reports an empirical implementation that computes an exact best response to a
warm policy and gives its regret and average strategy the mass of 15
iterations. The retained pilot adapts that empirical construction to the
repository's DCFR discount recurrence; it is not described as a theorem-backed
safe warm start.

[Regret-Based Pruning](https://proceedings.neurips.cc/paper/2015/hash/c54e7837e0cd0ced286cb5995327d1ab-Abstract.html)
was reviewed but not implemented. Its validity depends on a bounded skip
interval followed by a best response to the opponent's average strategy over
the skipped interval. A simple “do not visit negative-regret actions” shortcut
would omit that repair and is not an acceptable implementation.

[VR-MCCFR](https://ojs.aaai.org/index.php/AAAI/article/view/4048) gives an
unbiased bootstrapped state-action control variate for sampled MCCFR and reports
large Leduc convergence improvements. The later
[predictive-baseline analysis](https://proceedings.mlr.press/v119/davis20a.html)
shows that an appropriate baseline can approach zero-variance sampled value
estimates. This repository's external-sampling trainer samples the entire
private/public deal at the root, however, so an opponent-action baseline leaves
the dominant chance sample untouched. The paired results below did not justify
retaining the added algorithm or memory state.

[Public Chance Sampling CFR](https://poker.cs.ualberta.ca/publications/AAMAS12-pcs.pdf)
was the strongest next architectural match. It samples only chance outcomes
visible to both players and updates all compatible private hands together; the
paper reports faster convergence than chance-sampled CFR in poker and an
`O(n^2)` to `O(n)` terminal computation improvement. The repository's exact
flop and turn/river public-belief solvers already use that vectorized private
hand shape, while the full-game schema-v3 trainer samples one complete deal.
Moving the full trainer to a public-tree/vector representation is therefore a
promising separate implementation, not a safe one-line sampler change. It was
not mixed into the imminent cloud run or claimed by the scalar trainer.

DeepMind's official OpenSpiel
[external-sampling MCCFR implementation](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/algorithms/external_sampling_mccfr.cc)
was also checked as a framework/implementation reference. It enumerates the
traverser's actions, samples opponent and chance actions, and performs simple
average updates on the opponent path, matching the corrected trainer's core
sampling shape. Replacing the trainer with OpenSpiel would still require a new
hold'em game, action abstraction, trajectory-recall key, exact settlement, and
artifact pipeline; it does not remove the dominant poker-specific work or
justify changing algorithms before the paired long-run evidence.

[Correlated Chance Sampling MCCFR](https://arxiv.org/abs/2607.27035) was also
implemented as an exact finite randomized golden-ratio rotation over the
always-visited uniform root-deal space. The July 2026 paper reports strong
tabular-poker gains but explicitly reports no statistically detectable gain in
four HUNL turn/river endgames and leaves global convergence of the fully
adaptive persistent variant open. This repository's 10,000-iteration pair had
only a `0.0222bb` mean local-deviation improvement while held-out unknown
coverage worsened from roughly 23% to 28% and root-continuation unknown
coverage worsened from roughly 31% to 42%. The experimental sampler was
therefore removed rather than promoted to the cloud runner.

[Hyperparameter Schedules](https://ojs.aaai.org/index.php/AAAI/article/download/38784/42746)
is a materially different retained pilot. The AAAI 2026 paper reports that
HS-DCFR(30) outperforms fixed DCFR in its HUNL endgames and recommends it for
large poker. The implementation follows its published schedules
`alpha=1+3t/n`, `beta=-1-2t/n`, and `gamma=30-5t/n`. The horizon `n` is an
explicit immutable setting so a checkpoint can be extended without silently
changing earlier discount factors. Dynamic regret discounting is lazy and
exact across skipped information-set visits; average-policy contributions use
the exact product of the changing gamma discounts.

The final implementation audit rechecked Equation 3's transition indexing:
the existing average is multiplied by `(t/(t+1))^gamma(t)` before iteration
`t+1` is added, while Equation 4 evaluates each schedule at that current `t/n`.
A focused test now compares every precomputed terminal-relative weight against
an eager application of that published recurrence, preventing a dynamic-gamma
off-by-one from hiding behind the fixed-gamma normalization equivalence.

## Retained policy-prior implementation

For each exact public-belief subgame, the opt-in path:

1. loads a complete frozen strategy profile and validates every node/action;
2. replays it to obtain authentic reach-weighted average-policy mass;
3. computes exact best-response action counterfactual values for both players;
4. seeds action regrets from those value differences using the configured
   positive/negative DCFR discount recurrence;
5. seeds the average-policy accumulator with exact own reach and DCFR strategy
   mass; and
6. advances the solver's global discount clock before ordinary alternating
   updates begin.

Zero prior iterations preserves the former solver path. The prior is rejected
with safe resolving, delayed averaging, or detached river refinement, because
those combinations have not been derived or measured. The serving limit is 64
pseudo-iterations.

## Exact-root matched pilots

All roots use 20bb equal stacks, pot 4bb, uniform exact card-compatible ranges,
actor 1, DCFR `(alpha=1.5, beta=0, gamma=0)`, and zero averaging delay. The
same source solution is replayed for each two-update prior candidate.

| Root | Source | Two-update control | Prior 4 | Prior 8 | **Prior 15** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| River `2c,7d,Th,Js,Ac` | 0.098202 (64 updates) | 2.832452 | 0.349384 | 0.225154 | **0.159575** |
| Turn/river `2c,7d,Th,Js` | 0.486911 (32 updates) | 3.743374 | 0.792722 | 0.551043 | **0.480805** |
| Flop `2c,7d,Th` | 0.349060 (32 updates) | 2.207436 | 0.492083 | 0.380800 | **0.355716** |

Values are exact exploitability in each finite abstract root, in bb/hand. Prior
15 is the only tested weight that consistently preserves the converged source
while still allowing two fresh updates to matter. On the joint turn/river root
it slightly improves the 32-update source. Flop wall time rises from 20.76s for
the zero-regret control to 30.83s for prior 15; this is material but far below
the cost of recreating the 32-update source online.

A follow-up tested whether four fresh updates should replace two. The first
predeclared gate root, river, worsened from `0.159575bb` at two updates to
`0.285793bb` at four, moving farther from the `0.098202bb` source. Because the
promotion rule required improvement on every exact street root, the screen
stopped without spending turn/flop or full-game compute. The retained online
update count remains two.

### Out-of-sample board-texture check

A final holdout repeated source/control/prior 15 on boards not used to choose
the prior weight. It covers paired, monotone/connected, and ace-high dry
textures. All other game, range, DCFR, and two-update settings match the first
root suite. Lower exploitability is better.

| Holdout root | Source | Two-update control | Prior 15 + two updates | Prior minus source | Wall time source/control/prior |
| --- | ---: | ---: | ---: | ---: | ---: |
| River `As,Ah,7c,2s,Kd` | 0.109285 | 2.485099 | **0.185862** | +0.076577 | 0.47s / <0.01s / 0.01s |
| River `9s,Ts,Js,Qs,2d` | 0.109137 | 2.763554 | **0.157872** | +0.048735 | 0.07s / <0.01s / 0.01s |
| Turn/river `Ah,Ad,7c,2s` | 0.482998 | 3.795961 | **0.488734** | +0.005737 | 14.09s / 1.82s / 7.30s |
| Turn/river `9s,Ts,Js,2d` | 0.464320 | 3.524248 | **0.452312** | **-0.012008** | 15.13s / 2.10s / 7.97s |
| Flop `9s,Ts,Js` | 0.293665 | 1.900069 | **0.298143** | +0.004478 | 122.21s / 28.22s / 51.08s |
| Flop `Ac,7d,2h` | 0.438386 | 2.538189 | **0.429266** | **-0.009120** | 151.54s / 28.96s / 32.74s |

Across the six roots, the prior removes 96.8%–100.4% of the gap from the
immature control to the source. It slightly beats the source on the second
turn and flop while remaining close on the other four. This materially
strengthens the narrow claim that coherent pseudo-iteration state preserves a
nearby finite-root policy; it does not change the negative routed full-game
decision below. The flop runs also expose a real street/texture latency cost:
the 32-update holdout sources take 122–152 seconds even with eight workers,
versus 14–15 seconds on turn and less than half a second on river. That supports
a public-chance/vector implementation and board-stratified latency benchmark
before any online deployment.

Because the routed candidate applies the prior to actor 0, the paired texture
was also repeated with actor 0 at the root. This checks the actual seat-routing
path instead of inferring symmetry from actor 1:

| Actor-0 root | Source | Two-update control | Prior 15 + two updates | Prior minus source | Control-to-source gap removed |
| --- | ---: | ---: | ---: | ---: | ---: |
| River `As,Ah,7c,2s,Kd` | 0.109588 | 2.841100 | **0.165902** | +0.056314 | 97.9% |
| Turn/river `Ah,Ad,7c,2s` | 0.470882 | 3.706464 | **0.498129** | +0.027248 | 99.2% |
| Flop `9s,Ts,Js` | 0.285638 | 1.675324 | **0.294589** | +0.008951 | 99.4% |

All three actor-0 roots retain the same direction. The later routed full-game
failure is therefore not explained by a reversed seat index or a prior that
works only for actor 1; it arises only after the policy is placed on
action-conditioned ranges and interacts across streets.

The selected weight was then challenged against prior 8 on one holdout root
per street. Prior 15 wins every comparison: river `0.185862` versus `0.275795`,
turn/river `0.488734` versus `0.582836`, and flop `0.298143` versus `0.327315`.
This repeats the original board's ordering and keeps 15 as the sole retained
research weight; it does not create another runtime configuration.

Across all holdout source, control, and prior artifacts, the largest
probability-sum error is `4.45e-16` and the largest zero-sum residual is
`1.47e-14bb`; none of the gains comes from an invalid strategy row or payoff
imbalance.

The ignored holdout artifacts are stored under
`preflop-solver/neural/runs/v147-policy-prior-texture-holdout`; the six verbose
turn strategy profiles are gzip-compressed from roughly 440MB to 15MB each.
The corpus contains 34 files. SHA-256 over the lexicographically sorted lines
`<file-sha256><two spaces><repository-relative-path>\n` is
`c06d9dc0e68ff4174e4c4dd8e053a7b2d9e9e69a517e674f59ff38a43b87d5c5`.

### Hard-flop CPU scaling

The existing vectorized flop solver was benchmarked separately on the
monotone-connected holdout at eight updates. Only the worker count changes:

| Threads | Wall time | Speedup | Parallel efficiency | Exploitability bb/hand |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 153.13s | 1.00x | 100.0% | 0.631850266 |
| 2 | 92.25s | 1.66x | 83.0% | 0.631850267 |
| 4 | 70.88s | 2.16x | 54.0% | 0.631850267 |
| 8 | 49.37s | 3.10x | 38.8% | 0.631850266 |

The score spread is below `1.4e-9bb`; parallel summation order changes only
roundoff, not the conclusion. Eight workers materially reduce latency, but
scaling is already strongly sublinear after two. A public-chance full-trainer
prototype should therefore benchmark 4 and 8 workers before choosing a cloud
shape, while the current scalar schema-v3 trainer remains one worker per seed
and should continue to prioritize memory capacity and single-core speed. Even
49 seconds for an eight-update hard flop is unsuitable as an interactive
fallback, reinforcing the rule that practice must use frozen/cached policy
data rather than wait for this solve.

The first actor-0-only full-game causal pilot used the same two outer deals and
seed `19801` as the selected baseline. It deliberately removed the accepted
actor-1 resolver, so its total is not a candidate comparison; responder 1 is
the causal measurement of actor-0 policy quality.

| Profile seen by responder 1 | Responder 1 bb | Delta vs selected |
| --- | ---: | ---: |
| Selected zero-prior actor-1 resolver; frozen actor 0 | 2.151535 | — |
| Actor-0 prior 15 on flop, turn, and river | **2.033165** | **-0.118370** |
| Actor-0 prior 15 on turn and river | **2.041424** | **-0.110111** |
| Actor-0 prior 15 on river only | **2.063368** | **-0.088167** |

The preflop terminal attribution is unchanged (`0.506685bb`). Relative to the
selected profile, the actor-0 warm solve worsens flop attribution from
`0.308751` to `0.569345bb`, improves turn from `0.116736` to `0.090424bb`, and
improves river from `1.219364` to `0.866712bb`. The net win is substantial but
is dominated by river and partly cancelled by flop. River-only resolving keeps
the preflop attribution unchanged and still improves the measured response by
`0.088167bb`, but gives back `0.030203bb` relative to the all-street profile.
That residual can come from a turn/river interaction even though the direct
flop attribution is harmful. The turn+river-only run captures 93% of the
all-street improvement and beats river-only by `0.021944bb`. All-street is
only `0.008259bb` better on this corpus, while requiring a new asymmetric flop
serving path and worsening direct flop attribution to `0.569345bb` versus
`0.539309bb` for turn+river. The predeclared combined candidate therefore kept
the accepted actor-1 flop resolver, left actor 0 frozen on the flop, and
applied prior 15 to actor 0 only on turn and river. These isolated values are
research estimates with two outer deals, not a release certificate.

## Routed full-game integration decision

The final-binary integration screen compared the candidate with the selected
actor-1-only flop/turn/river control on the same two outer deals and seed
`19801`. Lower is better.

| Routed profile | Responder 0 bb | Responder 1 bb | Total bb/hand | Delta vs control |
| --- | ---: | ---: | ---: | ---: |
| Selected actor-1-only control | 0.214402 | **2.151535** | **1.182969** | — |
| Actor-0 prior 15 on turn+river | **0.207094** | 2.161760 | 1.184427 | +0.001458 |
| Actor-0 prior 15 on river only; turn pinned to actor 1 | 0.214402 | 2.157847 | 1.186125 | +0.003156 |

The combined candidate improves its direct river terminal attribution by
`0.027610bb`, but worsens turn attribution by `0.039475bb`; responder 1 is
therefore worse overall. The corrected river-only candidate holds responder 0
bit-for-bit at the control value but worsens responder 1 by `0.006312bb`.
Neither wrong-sign seed advances to an independent replicate.

An earlier river-only command omitted `--turn-resolver-actor 1`, so it silently
changed turn routing to bilateral and produced `1.250330bb/hand`. Its serialized
configuration exposed the mismatch; that score is excluded from algorithm
comparison. This is also why full certificate routing fields must be checked
before interpreting a point estimate.

The exact-root results remain evidence that coherent pseudo-iteration state
can preserve a nearby frozen solution in a short solve. They do not generalize
to the action-conditioned range distribution of the routed full game. The
policy prior remains opt-in for research, but no prior configuration is
selected for practice serving or the long cloud trainer.

## HS-DCFR schedule pilots

The first screen used identical seeds and evaluation corpora for fixed DCFR
and HS-DCFR(30). Lower root local-deviation gain is better; it remains a noisy
one-step diagnostic, not exploitability.

| Budget | Profile | Seed 27001 | Seed 27002 | Pair mean | Cross-seed spread |
| ---: | --- | ---: | ---: | ---: | ---: |
| 10k | Fixed DCFR, delay 0 | 2.255505 | 2.265443 | 2.260474 | 0.009937 |
| 10k | **HS-DCFR(30), delay 0** | **1.916300** | **2.222929** | **2.069614** | 0.306629 |
| 50k | Fixed DCFR, delay 0 | 1.834099 | 1.866301 | 1.850200 | 0.032202 |
| 50k | **HS-DCFR(30), delay 0** | **1.833501** | **1.811482** | **1.822491** | **0.022019** |
| 100k | Fixed DCFR, delay 0 | 1.828165 | 1.911625 | 1.869895 | 0.083460 |
| 100k | **HS-DCFR(30), delay 0** | **1.810375** | **1.861115** | **1.835745** | **0.050740** |
| 200k | Fixed DCFR, delay 0 | 1.759515 | **1.746553** | 1.753034 | **0.012962** |
| 200k | **HS-DCFR(30), delay 0** | **1.724778** | 1.755800 | **1.740289** | 0.031022 |
| 300k | Fixed DCFR, delay 0 | 1.862071 | 1.691402 | 1.776736 | 0.170669 |
| 300k | **HS-DCFR(30), horizon 300k, delay 0** | **1.775823** | **1.636124** | **1.705974** | **0.139699** |
| 300k | **HS-DCFR(30), horizon 600k, delay 0** | **1.610045** | 1.702900 | **1.656472** | **0.092856** |

At 10k both seeds improve and pair mean falls `0.190860bb` (8.44%), but the
spread is too large to select the schedule. At 50k both seeds again improve,
pair mean falls `0.027708bb` (1.50%), spread improves, and reach-weighted
action-EV precision rises from 16.96% to 19.20%. Root continuation unknown
coverage worsens by 0.87 percentage points. At 100k, both seeds improve for a
third time, pair mean falls `0.034150bb` (1.83%), spread improves, and action-EV
precision rises from 18.32% to 20.92%. Root continuation unknown coverage is
0.78 percentage points worse and held-out coverage is mixed. At 200k the pair
mean improves another `0.012745bb` (0.73%), but the result is no longer
unanimous: seed 27001 improves `0.034736bb`, while seed 27002 worsens
`0.009247bb`, and cross-seed spread increases. Action-EV precision is likewise
mixed. At 300k, terminal-horizon HS beats fixed on both seeds and improves pair
mean by `0.070763bb` (3.98%). Pinning the intended 600k cloud horizon from the
start improves pair mean by `0.120264bb` (6.77%) and narrows spread, although
seed 27002 is `0.011498bb` worse than fixed on the original 16-sample corpus.

That small reversal was evaluated once more with 128 root-deviation samples
per hand class. This is not numerically comparable to the 16-sample rows: the
larger within-class corpus reduces maximization bias as well as standard error.
The matched schedule comparison is the relevant result:

| 300k profile, 128/class | Seed 27001 | Seed 27002 | Pair mean | Spread |
| --- | ---: | ---: | ---: | ---: |
| Fixed DCFR, delay 0 | 0.569499 ± 0.037397 | 0.587275 ± 0.036077 | 0.578387 | 0.017776 |
| **HS-DCFR(30), horizon 600k, delay 0** | **0.458269 ± 0.038246** | **0.460165 ± 0.039040** | **0.459217** | **0.001897** |
| Fixed DCFR at 400k, delay 0 | 0.550309 ± 0.038118 | 0.545130 ± 0.035183 | 0.547720 | 0.005179 |
| Fixed DCFR at 400k, gamma 2, delay 40k | 0.475685 ± 0.039401 | 0.493567 ± 0.036395 | 0.484626 | 0.017882 |
| **HS-DCFR(30), horizon 600k at 400k** | **0.448295 ± 0.040146** | **0.453471 ± 0.038646** | **0.450883** | **0.005177** |

HS improves both matched seeds by `0.111230bb` and `0.127109bb`; pair mean is
20.6% lower and cross-seed spread is 89.3% lower. Action-EV precision also
improves on both seeds (`+1.32` and `+0.88` percentage points), while root
continuation unknown coverage worsens by `0.61` and `0.36` percentage points.
Policy action quality is the selection objective and the coverage tradeoff is
small, so the documented 20bb cloud pair now uses HS-DCFR(30), delay 0, and an
immutable 600k horizon. Fixed DCFR remains available as the generic runner
default and a control, not the selected 20bb command.

The final local 400k midpoint confirms both continued progress and schedule
selection. Against the delay-0 fixed research control, HS wins both seeds by
`0.102015bb` and `0.091659bb`. More importantly, it also beats the actual
former cloud profile—fixed gamma 2 with a 40k delay—by `0.027390bb` and
`0.040095bb`; pair mean is 6.96% lower and spread is 71.0% lower. Relative to
that actual control, action-EV precision changes by `+0.76` and `-0.47` points,
while root-continuation unknown coverage changes by only `+0.18` and `-0.32`
points. Within HS, both seeds improve from 300k by `0.009974bb` and
`0.006694bb`, and pair mean improves another 1.82%. The pair reaches
11.17M/11.08M information sets without hitting the 15M guard or swapping. This
is a useful first-stage checkpoint, not an exploitability certificate.

The paired 400k root-policy comparison adds a separate stability signal. The
maximum aggregate action-frequency delta is `1.1503` percentage points, below
the `3pp` gate. Primary-action agreement is `86.98%` by hand class and `91.25%`
by physical-combo weight, so the `85%` gate passes. The limiting stability
metric is combo-weighted per-action MAE: limp is `10.0481pp`, above the `5pp`
gate. Median hand-class total variation is `19.58%` and the 95th percentile is
`46.69%`. Thus the schedules agree on broad aggregate frequencies and most
primary actions, but the detailed mixes have not stabilized enough for
promotion. This is another concrete reason to run the planned 400k-to-600k
extension rather than treating the midpoint as releasable.

The paper's HS-DCFR(15) alternative was also implemented as a direct paired
screen. It uses the same alpha/beta schedule as HS30 but changes the average
policy schedule from `gamma=30-5t/n` to `gamma=15-5t/n`; consequently it tests
less aggressive recency weighting without changing the sampled regret
trajectory. At 50k and 100k its 16-sample local-deviation pair means were
`1.846522bb` and `1.876146bb`, worse than HS30's `1.822491bb` and
`1.835745bb`. It improved at 200k (`1.725377bb` versus `1.740289bb`) and was
effectively tied at the 300k/600k-horizon screen (`1.661698bb` versus
`1.656472bb`), so the noisy screen alone did not justify a decision.

The predeclared 128-sample rerun resolved the ambiguity:

| 300k profile, 128/class | Seed 27001 | Seed 27002 | Pair mean | Spread |
| --- | ---: | ---: | ---: | ---: |
| **HS-DCFR(30), horizon 600k** | **0.458269** | **0.460165** | **0.459217** | **0.001897** |
| HS-DCFR(15), horizon 600k | 0.470445 | 0.482424 | 0.476434 | 0.011979 |

HS15 is worse on both matched seeds and raises pair mean by `0.017217bb`
(`3.75%`). It does improve root mix stability at the same trained point:
maximum aggregate action delta falls from `1.8655pp` to `1.3202pp`, maximum
combo-weighted per-action MAE from `10.8455pp` to `10.0379pp`, median total
variation from `21.21%` to `20.20%`, p95 from `47.56%` to `45.56%`, and
primary-action agreement rises from `80.47%` to `81.66%`. Policy-quality EV is
the selection objective, however, and HS30 also has the much smaller
cross-seed diagnostic spread. HS15 was therefore removed from the solver CLI
and cloud runner rather than retained as an unselected production option. Its
ignored artifacts remain negative research evidence only.

The selected HS30 lineage was then run to its original 600k terminal horizon
with the required 256 root-deviation samples per class. Both seeds completed
without hitting the 22M guard:

| 600k terminal metric | Seed 27001 | Seed 27002 |
| --- | ---: | ---: |
| Information sets | 15,930,819 | 15,812,729 |
| Root local-deviation gain | 0.330815 ± 0.026824 | 0.319913 ± 0.024627 |
| One-sided 99% lower bound | 0.268413 | 0.262621 |
| Unknown root continuation | 17.878% | 17.825% |
| Untrained root continuation | 2.700% | 2.743% |
| Action-EV SE coverage | 17.875% | 17.856% |

The pair passes the 3pp aggregate-frequency gate (`1.0489pp`), 20% median-TV
gate (`19.11%`), 65% max-TV gate (`57.03%`), 85% primary-action gate
(`85.21%` by class, `87.78%` by combo), 256-sample floor, and 100-visit floor
(minimum `829`). It still fails the limiting 5pp per-action-MAE gate
(`9.2088pp`, driven by limp), 35% p95-TV gate (`45.91%`), 0.10bb local-gain
gate, 0.05bb lower-bound gate, 5% unknown-continuation gate, and narrowly the
2.5% untrained-continuation gate.

Relative to 400k, detailed mix MAE improves from `10.0481pp` to `9.2088pp`,
median TV from `19.58%` to `19.11%`, p95 from `46.69%` to `45.91%`, and maximum
aggregate delta from `1.1503pp` to `1.0489pp`. Primary-action agreement remains
above its gate but falls from `86.98%` to `85.21%`, so iteration count is not a
monotonic cure for every summary statistic. The local-gain values are not
strictly comparable to the 400k 128/class values because the terminal audit
uses twice as many forced continuations and therefore less maximization bias.
They nevertheless remain far above the promotion threshold. The 600k horizon
is therefore useful progress, not a viable terminal cloud lineage; a paired
1M-horizon midpoint screen follows before revising the long-run command.

That paired 1M-horizon midpoint screen was also negative. At 600k iterations
and 16 local-deviation samples per class, its two seeds score `1.699171bb` and
`1.596228bb` (mean `1.647700bb`, spread `0.102944bb`). The result is
effectively unchanged from the selected 300k/600k-horizon mean of
`1.656472bb`, despite twice the training, and seed 27001 moves in the wrong
direction from `1.610045bb` to `1.699171bb`.

The root distributions give a clearer same-budget comparison against the 600k
terminal schedule. Moving to the 1M horizon worsens maximum aggregate delta
from `1.0489pp` to `2.2762pp`, per-action MAE from `9.2088pp` to `10.2487pp`,
median TV from `19.11%` to `19.49%`, p95 TV from `45.91%` to `49.54%`, and max
TV from `57.03%` to `65.29%`. Primary agreement is unchanged at `85.21%` by
class and improves from `87.78%` to `89.14%` by combo, but that isolated gain
does not offset the broader regressions. The 1M horizon is rejected rather
than promoted to the cloud command.

HS30 with a 600k horizon remains the best tested full-trainer schedule, but its
terminal policy is not promotable. The cloud launch is therefore on hold: the
runner and reproducible commands are ready, but repeating the same seeds/game
only to reproduce known root-gate failures would waste paid compute. A future
cloud run needs a policy-changing implementation (the researched public-chance
vector trainer is the leading architectural candidate), or an explicit
diagnostic-only authorization to produce the full postflop export and larger
evaluation corpus despite the known root failure.

## Next policy-changing experiment: public-chance vector traversal

The next experiment is intentionally a new trainer version and checkpoint
schema, not a switch on schema v3. Its minimum viable vertical slice is one
public flop plus a bounded continuation tree. One sampled public board is held
fixed while the traversal carries both players' card-compatible private ranges
and evaluates every compatible private hand at the reached public nodes. This
matches the public-chance sampling paper's poker-specific variance reduction;
reusing one complete private deal many times would not.

The implementation must preserve these boundaries before any cloud scale-up:

1. the existing `GameState` legal-action and chip-settlement logic remains the
   single public-tree transition authority;
2. exact 1,326-combo masks enforce card removal at every public chance event;
3. trajectory-recall keys retain the full sequence of private/public buckets
   and public actions, so vectorization does not coarsen recall;
4. counterfactual reach is carried separately for each player and never
   normalized across incompatible hands during regret updates;
5. the two traversers update from the same frozen iteration policy, then apply
   their deltas deterministically, preventing worker order from changing a
   checkpoint; and
6. terminal vector values must agree with the scalar trainer on folds, river
   showdowns, and enumerated all-in runouts before performance is measured.

The first paired gate is deliberately cheap: on a finite flop root, compare
vector and scalar expected values/regret deltas with the same public sample,
then run two 10k and two 50k full-game seeds. Advance only if the vector pair
improves root local-deviation on both seeds without worsening held-out or root
continuation unknown coverage by more than one percentage point. Also record
wall time, terminal evaluations per second, information sets per iteration,
and bytes per information set. This distinguishes a variance/convergence gain
from a mere throughput gain.

Only after that gate should a 400k/600k cloud lineage be generated. The cloud
runner already supplies independent seed isolation, immutable binary/config
fingerprints, memory/disk refusal, checkpoint lineage, complete postflop export,
and evaluation-only retries. The vector trainer must plug into the same
summary contract so its result remains directly comparable to the retained
HS30 scalar evidence. OpenSpiel remains a recurrence and test oracle rather
than a drop-in hold'em implementation: its external-sampling code confirms the
current enumerate-traverser/sample-opponent update, but does not provide this
range-vector public-chance traversal or the repository's abstraction and
artifact contracts.

The four final research artifacts are ignored local evidence, not serving
assets. Their immutable identities and compressed SHA-256 hashes are:

| Profile / seed | Artifact ID | gzip SHA-256 |
| --- | --- | --- |
| HS30 h600k terminal / 27001 | `hu-blueprint-20bb-i600000-s27001-cebd7431282e8198` | `dd449fb23dde37d77d035255702ca6474af8781ec957f9f69442160bcb536220` |
| HS30 h600k terminal / 27002 | `hu-blueprint-20bb-i600000-s27002-2dd4bd4ce904ab2c` | `0291fbcbf675fb4c389b43228b4356925f3c52c7bfafbbfe1cf902bdbfa64051` |
| HS30 h1M at 600k / 27001 | `hu-blueprint-20bb-i600000-s27001-4c63022a3a3767c5` | `d625186d378deba26ec39be1cdc191c3eb9391a19d3f60b8ef3934067e54e6b2` |
| HS30 h1M at 600k / 27002 | `hu-blueprint-20bb-i600000-s27002-c3bdf4f904e3ca6b` | `f01b915b635b6f611b083b1342e8e44da8eb2b2bf57b163872d3fad780185321` |

Those local pilot artifacts intentionally exported preflop rows only and were
created before root visit diagnostics were added to schema-v3 rows, so their
visit/update counts remain unknown and cannot pass the complete cloud audit.
Future artifacts serialize those diagnostics only on the 169 root rows. The
required compact sidecar now also carries the root action distributions, and
the cloud runner computes the `3pp` aggregate-delta, `5pp` per-action-MAE, and
`85%` primary-agreement gates without loading the multi-gigabyte uncompressed
postflop policy. An end-to-end two-seed run and checkpoint retry validated this
path; deliberately tiny two-iteration policies correctly report stability as
unavailable rather than passing incomplete root coverage.

The runbook keeps 128 local-deviation samples per hand class at the 400k
midpoint, then uses 256 at the terminal 600k extension to meet the audit's
sample floor. This evaluation-only change is compatible with checkpoint
resume and does not change the policy training trajectory.

## Rejected variance-reduction pilot

The VR-MCCFR pilot used the same full-game schema-v3 trainer, exact seeds,
evaluation seeds, abstraction, DCFR `(1.5, 0, 2)`, and averaging delays as its
control. Lower root local-deviation gain is better.

| Iterations | Seed | Control gain bb | VR gain bb | VR minus control | Decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 10,000 | 27001 | 3.286395 | 3.098814 | -0.187580 | Mixed |
| 10,000 | 27002 | 3.027598 | 3.260817 | +0.233219 | Mixed |
| 50,000 | 27001 | 1.837376 | 1.940783 | +0.103407 | Reject |
| 50,000 | 27002 | 1.864376 | 1.904732 | +0.040356 | Reject |

At 50,000 iterations, the control used 1.50GB peak memory for 1,662,066
information sets; VR used 1.58GB for 1,694,400 sets on the measured seed. The
overhead was acceptable, but both policy-quality deltas had the wrong sign.
The code, checkpoint state, CLI flag, and cloud flag were removed. The pilot
artifacts remain ignored research evidence only.

## Corrected trainer iteration pair

The two retained 50,000-iteration control seeds were rerun independently at
100,000 iterations with the same game, DCFR tuple, evaluation seeds, 1,024
held-out/action-value deals, and 16 root-deviation samples per hand class. The
averaging delay remains 10% of each fresh run. Lower root local-deviation gain
is better; it is still only a one-step policy diagnostic.

| Seed | 50k gain bb | 100k gain bb | Delta | Held-out unknown 50k -> 100k | Root continuation unknown 50k -> 100k |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 27001 | 1.837376 | **1.816770** | -0.020606 | 20.96% -> 16.18% | 28.94% -> 26.67% |
| 27002 | **1.864376** | 1.916009 | +0.051633 | 18.09% -> 17.03% | 28.80% -> 26.14% |

The pair mean moves from `1.850876` to `1.866390bb` and the cross-seed spread
widens from `0.026999` to `0.099239bb`. Coverage improves on both seeds and
reach-weighted action-EV precision rises to `18.27%`/`18.59%`, but policy
quality is mixed and the pair mean is slightly worse. The deltas are small
relative to each root estimate's roughly `0.14bb` standard error, so this does
not prove a plateau; it does rule out claiming that 100k is already a clean
convergence win.

The fresh fixed-DCFR 200k pair then improved decisively:

| Seed | 100k gain bb | 200k gain bb | Delta | Root continuation unknown at 200k | Action-EV precision at 200k |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 27001 | 1.816770 | **1.768642** | -0.048128 | 23.10% | 18.73% |
| 27002 | 1.916009 | **1.748317** | -0.167693 | 23.21% | 20.52% |

Pair mean is `1.758479bb` and spread is `0.020326bb`. Training reached about
6.03 million information sets per seed at 4.99GB peak RSS with zero swaps.
The full trainer is therefore still making meaningful policy progress, and
the 400k cloud stage is justified independently of schedule selection.

The matched delay-0 fixed pair at 300k is mixed relative to its 200k point:
seed 27001 worsens by `0.102556bb`, seed 27002 improves by `0.055151bb`, and
pair mean worsens by `0.023703bb` on the 16-sample diagnostic. Both seeds do
improve held-out and root-continuation unknown coverage. They reach
8.68M/8.66M information sets at 7.19GB/7.00GB maximum RSS with no swaps. This
supports the 15M information-set first-stage guard but does not support a claim
that fixed delay-0 policy quality improves monotonically with iteration count.

## Certificate latency correction

Sampling the first full-game prior candidate showed most time inside a complete
resolver action-EV pass. Full-game response search consumes only action
probabilities; practice feedback consumes both probabilities and action EVs.
The retained runtime now separates those uses:

- practice keeps the full EV-bearing policy rows;
- exploitability certification exports probability-only resolver rows; and
- focused tests require flop, joint turn/river, and river probabilities to be
  bit-for-bit identical between the two paths.

This changes evaluation cost, not policy actions or the response algorithm.
The final-binary matched control reproduced the historical responder values
exactly (`0.214402112675bb` and `2.151535266400bb`, total
`1.182968689538bb`) while omitting resolver action values from certification.
It completed in 599.85 seconds with no swaps, providing an end-to-end parity
check beyond the focused root tests.

## Cloud-run preparation

The cloud runner launches two independent deterministic model-schema-v3
processes; it never merges their regrets. Internal resumable checkpoints use
schema 4, while frozen output policies remain model schema v3. Checkpoints and
final artifacts are streamed as gzip, and resume reproduces canonical
decompressed hashes. The selected 20bb command explicitly uses HS-DCFR(30),
zero averaging delay, and a 600k immutable horizon; fixed DCFR remains the
runner's generic default and gamma 0 remains confined to measured short-policy
comparisons.

Each seed also writes a compact summary sidecar. After a child exits, the
orchestrator streams the gzip to EOF, records its canonical SHA-256, validates
the summary schema/seed/target, and fails the run if `stoppedEarly` is true or
the completed iteration count differs from the request. A forced one-infoset
smoke test rejected both seeds and returned exit status 1; the retained
two-seed smoke completed with gamma 2.

The manifest now records the exact solver-binary SHA-256 and a stable
fingerprint over every artifact-affecting setting. An identical two-seed resume
advanced the manifest attempt counter and reproduced both artifact hashes; a
same-directory gamma change was rejected before launch with exit status 1.

The compact seed summary now includes held-out EV/error, root local-deviation
gain/error/99% lower bound, root training and continuation coverage, and
action-EV precision, plus the 169 compact root-policy rows and their
visit/update diagnostics, so an operator does not need to load a
multi-gigabyte policy merely to decide whether the next stage is justified.
The orchestrator requires those fields and their requested sample counts to be finite and
consistent before accepting a seed. It also rejects negative uncertainty,
out-of-range held-out, continuation, and action-EV coverage fractions, invalid
seed/iteration/table-size domains, non-finite solver parameters, invalid
evaluation counts, and impossible checkpoint cadence
before a cloud launch. A final aggregate is emitted only after every seed
passes integrity validation; it records pair mean/spread, worst continuation
and held-out coverage, minimum action-EV precision, and maximum table size. Fourteen isolated
tests cover seed identity and numeric
preflight, fingerprinting, local and parent-stage resume selection, checkpoint
lineage, missing-output rejection, canonical gzip hashing, and summary
validation/aggregation. Paired summary aggregation now also fails closed on
incomplete root coverage or incompatible action sets and evaluates the
cross-seed root-policy gates directly. This is deliberately named and scoped
as root stability; full postflop stability still requires the independent
matched-reach evaluation in the promotion procedure.

Summary acceptance also checks that preflop plus postflop information-set
counts equal the total, evaluated/exported counts stay within their domains,
both action-EV coverage fractions are in `[0,1]`, and a requested full
postflop export contains every trained average-policy row. A fresh two-seed
full-export smoke produced exact trained/exported equality on both seeds; this
prevents a successful process exit with a silently truncated serving policy
from being recorded as a complete cloud stage.

A 10,000-iteration end-to-end integration pair then reached all 169 root hand
classes with minimum seven average visits per class. The manifest reported
`1.7683pp` maximum aggregate delta, `15.5978pp` maximum per-action MAE, and
`21.30%` primary-action agreement, correctly passing only the aggregate-delta
gate at that deliberately immature checkpoint. An independent run of the
standalone artifact comparator reproduced those three values (floating-point
roundoff below `1e-15`), confirming the compact-summary calculation. These
numbers validate the reporting path, not the policy.

The standalone comparator now reads the cloud runner's canonical `.json.gz`
artifacts directly as well as plain JSON. A direct comparison of the two 600k
terminal gzip artifacts reproduced the documented `9.2088pp` limiting MAE,
`19.11%` median/`45.91%` p95 total variation, and `85.21%` primary-action
agreement without an intermediate decompressed copy.

Longer targets use a new lineage-pinned stage directory. The extension
preflight requires a complete parent run, identical binary and immutable
solver settings, the exact seed set, and matching checkpoint sizes and
SHA-256s. Completed validation-only retries do not rewrite the checkpoint, and
an interrupted extension prefers its own newer checkpoint over its parent.

The final retry audit also rejects an artifact or compact summary whose file
timestamp predates the current child attempt, closing the possibility that a
zero-exit bug could accept stale outputs. A two-seed end-to-end retry regenerated
both outputs while preserving both checkpoint hashes exactly and advanced the
manifest attempt from one to two.

Artifact identity is also checked across the serialization boundary without
loading the multi-gigabyte payload: the runner streams the artifact prefix and
requires its envelope-v1 schema, solver version, artifact ID, configuration
hashes, model, and approximate-bound flag to match the compact sidecar and
requested run. The first real-artifact smoke deliberately used the model-v3
number as the envelope expectation and failed; correcting that distinction to
the actual envelope-v1 contract made both retry artifacts pass at attempt two.

The 100,000-iteration memory scale point completed 3,242,828 information sets
at 2.68GB peak RSS; the 200,000-iteration pair reached about 6.03 million sets
per seed at 4.99GB peak RSS, the 300,000-iteration pairs reached about
8.6 million sets at no more than 7.22GB, and the 400,000-iteration pair reached
about 11.1 million sets at 7.13GB maximum RSS/9.31GB peak footprint. None
swapped. A separate 50,000-iteration
complete-postflop export wrote
961,899 served rows: 715.8MB canonical JSON and 110.9MB gzip, or about 115
compressed bytes per row. Export peak RSS was 2.42GB because table and output
rows coexist. Those measurements replaced the old one-million-iteration/
15-million-set sketch with a first 400,000-iteration stage and a 15-million-set
guard; later targets are chosen only after both cloud summaries are observed.

See [the cloud runbook](../solver/cloud-blueprint-runbook.md) for the exact
capacity, launch, resume, and promotion procedure.

## Public-chance vector vertical slice

The next policy-changing trainer prerequisite was implemented as a bounded
flop vertical slice before changing the cloud launcher. It reuses the existing
exact 1,326-combo masks, separate player counterfactual reaches, public action
history, legal action generator, fold settlement, exact all-in runouts, and
turn value boundary. The new update path materializes the public tree, applies
DCFR discounts, computes both players' regret and average-policy deltas against
one immutable iteration policy, and applies those deltas in deterministic
public-history order.

A focused release test establishes three implementation invariants:

- each player's regret delta matches an independent traversal of the same
  discounted snapshot;
- reversing the two frozen traversers produces identical regret and average
  buffers; and
- for the mean continuation oracle, one all-player vector traversal is exactly
  equal to the two frozen traversals, while retaining finite action EVs, exact
  all-in evaluation, and the zero-sum projection bound.

The one-pass form reduces repeated leaf work, but a three-texture controlled
pilot rejects the frozen simultaneous schedule on policy quality. All pilots
used 20bb, a 4bb root pot, actor 1, eight rounds, zero averaging delay, the
accepted v91 turn value network, eight threads, and the identical game/action
abstraction.

| Flop | Alternating exploitability bb/hand | Frozen vector exploitability bb/hand | Alternating wall time | Frozen vector wall time |
| --- | ---: | ---: | ---: | ---: |
| `2c,7d,Th` | **0.465431** | 1.255820 | 38.62s | **27.37s** |
| `As,Kd,7c` | **0.508019** | 1.529484 | 37.15s | **26.52s** |
| `9h,Th,Jh` | **0.414118** | 1.062009 | 45.11s | **30.41s** |

The two-round result initially favored the frozen schedule (`2.257748` versus
`2.401234bb/hand`), but the ordering reversed by round four. Giving the frozen
schedule 16 rounds still produced `0.792550bb/hand` in 42.84 seconds, worse
than the alternating schedule's eight-round `0.465431bb/hand` in 38.62
seconds. Regret-matching-plus did not repair the difference: at eight rounds
the frozen and alternating results were `1.272930` and `0.520801bb/hand`.
Peak RSS stayed below 264MB throughout, so memory was not the cause.

This experiment isolates the schedule: both arms already vectorize exact
private combos on a fixed public flop. It therefore rejects replacing the
current alternating resolver with simultaneous frozen updates; it does **not**
reject public-chance sampling as the variance-reduction mechanism for the
full-game trainer. The retained MVP is exposed only by the explicit
`--public-chance-vector-mvp` research flag and stamps artifacts
`research_only`. It is not wired into practice serving or the cloud runner.

The next justified trainer is narrower than the rejected combination: sample
one deterministic public board per iteration, enumerate all compatible
private combos for the active traverser, share the existing abstract
information sets across boards, and retain alternating traverser updates. A
small preflop-to-flop slice must first demonstrate scalar/vector regret parity,
lower sampling variance or better authentic coverage per unit time, bounded
memory, and deterministic replay. Only that evidence can lift the paid-compute
hold.

That proposal was then tested as a cheaper scalar stepping stone before
building the complete vector traversal. The optional trainer batch samples
without replacement from the 990 active-player hands compatible with one
sampled five-card board and opponent hand. Sub-sample regret and average-policy
weights sum to one; size one preserves the established RNG stream exactly;
actual deal counts survive checkpoint/resume; and focused tests cover unique
card-compatible sampling, determinism, normalized weighting, and increased
active-hand coverage.

At equal 1,000 outer rounds, batch four did improve coverage because it trained
4,000 deals: held-out unknown rate fell by 4.7--6.1 percentage points and root
continuation unknown rate by 11.4--12.4 points versus the 1,000-deal control.
It cost about 2.8 times the wall time and three times the memory. The relevant
matched-wall comparison therefore used scalar 3,000 rounds against batch-two
1,500 rounds; both process 3,000 deals in 0.56--0.59 seconds on this short
configuration.

| Seed | Mode | Held-out unknown | Root continuation unknown | Root trained combos | Root local-deviation gain bb | Action-EV precision |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 27101 | scalar 3,000 | **22.20%** | **37.29%** | **99.70%** | **3.4062** | **9.24%** |
| 27101 | batch-2 1,500 | 25.29% | 38.63% | 98.79% | 3.7022 | 9.03% |
| 27102 | scalar 3,000 | **21.75%** | **37.24%** | **100.00%** | 3.5524 | 8.67% |
| 27102 | batch-2 1,500 | 23.62% | 37.36% | 95.93% | **3.3491** | **8.94%** |

The second seed's root diagnostic favors batching, but its roughly 0.22bb
standard error makes that isolated delta inconclusive. Pair-mean root gain is
`3.4793bb` for scalar and `3.5257bb` for batch two, while scalar wins the
authentic coverage comparisons on both seeds. The optimization is rejected as
a cloud mode: sharing a board/opponent hand reduces the public/opponent chance
diversity that matters more at matched compute than the extra active private
hands. The implementation remains opt-in research infrastructure with a
default of one, not a selected training configuration.

Together the two pilots sharpen the remaining requirement. A viable
public-chance trainer cannot be a schedule-only change or a scalar loop over a
few hands. It must make the full compatible-hand traversal materially cheaper
through actual contiguous vector computation while still sampling enough
independent public boards, and it must beat scalar at matched wall time on both
policy quality and authentic coverage before paid compute.

## Policy-exact memory and checkpoint preparation

The production scalar path was then sampled during an authentic 20bb run.
Card evaluation and information-set construction dominated the active CPU
stacks, while the live table was dominated by duplicated variable-length
bucket and action-label containers. The retained changes do not alter the game
or training schedule:

- evaluator and visible-card rollout scratch space uses fixed stack buffers;
- the FNV identity hash consumes canonical components directly instead of
  allocating joined/formatted identity strings;
- money formatting retains the exact prior three-decimal byte representation;
- immutable trajectories and action vectors use boxed slices; and
- coarse hand/public buckets and legal-action labels share immutable strings,
  including after checkpoint loading.

The fixed 20,000-iteration seed produced the identical complete artifact
SHA-256 before and after all representation changes. Peak RSS fell from 586MB
to 377MB (36%). Retired instructions fell from 88.8 billion to 49.8 billion;
wall time was 4.96s versus 4.54s in the end-to-end artifact run, where
serialization and host scheduling remain material. A larger run stopped
cleanly at 2,000,050 information sets and 895MB RSS. These are capacity and
throughput improvements, not policy-quality improvements; exact artifact
identity proves that the same iteration count still produces the same policy.

Full export was separately retained in the measurement. At 20,000 iterations,
all 437,080 trained information sets were exported, peak RSS was 787MB, and
the gzip artifact was 48MB. The cloud runner's conservative memory preflight
was deliberately left unchanged because training nodes and export/evaluation
rows coexist.

The resume audit found that legacy JSON checkpointing could change normalized
strategy probabilities by roughly one ULP even without further training.
Named MessagePack-gzip checkpoints now preserve exact floating-point state.
On a roughly 690,000-set run, checkpoint size fell from 93MB to 78MB,
checkpointed wall time from 29.1s to 18.8s, and same-iteration resume/evaluation
from 5.78s to 2.53s. The resumed canonical artifact was byte identical. New
cloud stages use `.checkpoint.msgpack.gz`; old `.checkpoint.json.gz` parents
remain readable and discoverable.

This work removes substantial paid-run memory, I/O, and lineage risk. It does
not lift the paid-compute hold by itself: the last terminal pair's policy gates
remain the deciding evidence, and the rejected public-chance/batching pilots
remain rejected.

## Additional paired scalar policy pilots

Two more policy-changing ideas were screened after the representation work.
Every arm used the same 20,000 iterations, HS-DCFR(30) with a 60,000-round
horizon, identical evaluation seeds, 128 held-out deals, eight forced root
samples per class, 128 action-EV deals, and paired training seeds 27301/27302.
The control retained the 10 equity × 3 potential bucket grid.

| Arm | Root gain seed 27301 / 27302 (bb) | Held-out unknown seed 27301 / 27302 | Root continuation unknown seed 27301 / 27302 | Action-EV precision seed 27301 / 27302 |
| --- | ---: | ---: | ---: | ---: |
| Control 10×3 | 3.2275 / 3.1660 | 17.22% / 19.04% | 31.78% / 32.08% | 6.40% / 8.68% |
| Coarse 6×2 | **2.9825 / 3.0430** | 17.87% / 21.46% | 30.30% / 31.23% | 7.52% / 8.01% |
| Coarse 3×1 | 3.2441 / **2.9315** | 20.24% / 18.75% | **28.29% / 28.90%** | 5.41% / 5.35% |
| Distribution-only 10×3 | **3.0214 / 3.0718** | 20.41% / **15.40%** | 30.32% / 30.55% | 7.70% / 7.90% |
| 5% opponent exploration | **2.9345 / 3.1137** | 21.74% / 20.35% | 31.57% / 31.46% | 6.12% / 9.93% |
| 1% opponent exploration | **3.0802 / 3.1659** | 20.15% / 22.02% | 31.28% / 31.47% | 7.81% / 6.05% |

The 6×2 abstraction improves the noisy root diagnostic on both seeds, but one
seed worsens held-out unknown coverage by 2.43 percentage points, beyond the
predeclared one-point tolerance. The stronger 3×1 abstraction improves root
continuation coverage but worsens action-EV precision on both seeds and has
mixed root EV. Information-set growth falls only 1--3% even at 3×1, showing
that exact public action trajectories, not these final strength-bin counts,
dominate table cardinality. Neither coarse grid is selected.

A structural distribution-only variant also removed category, overcard,
kicker, and draw dimensions while retaining 10×3 rollout strength/potential
and future-mode labels. Root gain improved on both short seeds, but table growth
still fell only about 1%, held-out unknown coverage worsened by 3.19 points on
seed 27301, and action-EV precision was mixed. This stronger coarsening was
also removed rather than kept as a production flag.

Importance-corrected opponent exploration reused the dedicated preflop
solver's established recurrence. Five percent improves root gain on both
seeds, but worsens held-out unknown coverage by 4.52 and 1.31 points. Reducing
it to one percent still worsens held-out unknown coverage by roughly three
points on both seeds, with mixed action-EV precision. The full-game exploration
flag was removed after the pilot rather than retained as an unselected
production option.

These paired failures narrow the next useful policy implementation: scalar
bucket tuning and opponent sampling do not create enough authentic state
reuse. Further work should make the compatible-hand/public-tree traversal
genuinely contiguous while retaining alternating updates and independent
public boards; repeating scalar schedule or bin changes is not justified.

## Contiguous joint-hand traversal pilot

The existing opt-in hand batch was then changed from a scalar loop into a
single public-tree traversal. Traverser actions are enumerated for every lane;
opponent lanes are sampled independently and lanes choosing the same public
action share a recursive call. A second opt-in dimension can sample multiple
compatible opponent hands for each traverser hand. Batch-only card/action
randomness uses an iteration-derived stream, so it cannot perturb later public
board samples and requires no additional checkpoint state. Size one remains
byte-identical to the established scalar artifact.

Focused tests establish terminal regret parity with scalar traversal,
independent opponent sampling, exact card compatibility, deterministic joint
hand selection, and lossless checkpoint replay. The representation is a real
throughput improvement over the old scalar hand loop: at 1,000 outer rounds,
a two-hand traversal created 78,283 information sets in 0.23 CPU-seconds versus
42,953 in 0.18 seconds for scalar, or about 42% more table coverage per CPU
second. That implementation improvement does not by itself establish a better
policy.

The decisive paired screen used the same 20,000-round HS-DCFR(30) controls and
training/evaluation seeds as the preceding table. A 45,000-round scalar arm is
included as the approximate CPU-budget control for the two-lane arms.

| Arm | Root gain seed 27301 / 27302 (bb) | Held-out unknown seed 27301 / 27302 | Root continuation unknown seed 27301 / 27302 | Action-EV precision seed 27301 / 27302 |
| --- | ---: | ---: | ---: | ---: |
| Scalar 20k | 3.2275 / 3.1660 | **17.22%** / 19.04% | **31.78%** / **32.08%** | 6.40% / 8.68% |
| Contiguous traverser 2 × opponent 1, 20k | **2.8505** / 3.0434 | 21.66% / **17.93%** | 35.59% / 35.80% | **12.36%** / 9.52% |
| Contiguous traverser 1 × opponent 2, split RNG, 20k | 2.9470 / **2.8807** | 19.53% / 27.89% | 35.62% / 35.29% | 11.15% / 10.38% |
| Scalar 45k, approximate matched CPU | 2.9008 / 2.8866 | 25.20% / 23.68% | 35.78% / **35.05%** | 10.45% / **11.43%** |

The active-hand arm improves root gain on both seeds versus the 20k scalar and
improves action-EV precision, but misses authentic continuation coverage and
does not beat the longer scalar consistently at matched compute. Integrating
two opponent hands lowers pair-mean root gain versus the matched-compute scalar
only marginally (`2.9138bb` versus `2.8937bb`, so it is actually slightly
worse) and has a severe seed-27302 held-out failure. A systematic-lane variant
was also removed after worsening root gain and held-out coverage.

Neither batch dimension is selected for paid compute. The implementation
remains an explicit research bridge because it is deterministic, faster than
the former scalar loop, and can expand toward a real range traversal. The next
eligible policy candidate must carry both players' compatible reach vectors at
public nodes and integrate terminal values contiguously; adding more finite
deal lanes is not justified.

That next candidate now has a policy-independent vertical slice. A new exact
range kernel retains all 1,326 combos for both players, rejects board-blocked
reach, computes compatible-opponent mass, and returns per-combo counterfactual
values plus zero-sum profile values. Showdown uses one score distribution per
range and subtracts only card-conflicting hands, avoiding a dense joint-deal
table while matching a full pairwise reference. Action helpers split exact
actor reaches by policy and recombine child CFVs with the correct asymmetric
CFR semantics: strategy-weight actor values, but sum opponent values whose
child reaches already contain that strategy.

This slice is not a trainer and does not lift the cloud hold. It removes two
correctness/performance risks before shared abstract-node regret updates are
wired: terminal settlement no longer requires finite joint-deal lanes, and
range propagation has executable reach-conservation tests. The remaining
implementation is public-node strategy lookup/bucketing, alternating regret
updates, sampled street transitions, and a matched-wall paired pilot.

## Final verification

The retained implementation passed the complete repository checks after the
pilot selection and cloud-runner hardening:

- `npm test`: 114 passed, 4 skipped;
- `npm run build`: production build completed, including pinned resolver
  artifact verification;
- Rust release tests: 175 library tests and 3 CLI integration tests passed;
- full neural unittest discovery: 245 passed (the existing NumPy zero-variance
  warnings remain non-fatal);
- policy, practice, resolver-integration, artifact-comparator, and cloud-runner
  suites: 9, 11, 4, 9, and 14 tests passed respectively (the focused cloud
  suite is also part of full neural discovery); and
- formatting, lint, and whitespace checks passed after the final audit.

These checks establish implementation and artifact-lineage integrity. They do
not override the failed policy-quality gates or lift the cloud launch hold.
