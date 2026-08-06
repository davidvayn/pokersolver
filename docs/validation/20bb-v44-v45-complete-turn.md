# 20bb v44 rejection and v45 complete-turn correction

Status: v44 rejected; corrected v45 corpora accepted; first corrected value
students rejected; no active manifest was modified.

## Frozen v44 downstream result

The v44 value candidates pass the upstream prediction, Python/Rust parity,
provenance, and disjoint leaf gates. They fail the matched resolver-improvement,
cross-seed resolver-agreement, model-selection, and full-game exploitability
upper-bound gates. Re-evaluating each serialized frozen flop resolver reproduced
the original cross-evaluator within `4.7e-9bb/hand`, ruling out retraining or
serialization drift in that comparison.

Longer dry-low resolver audits did not reverse the result:

| Resolver iterations | v31 baseline | v44 seed 13502 | Candidate delta |
| ---: | ---: | ---: | ---: |
| 100 | 0.271713bb/hand | 0.304870bb/hand | +12.20% |
| 300 | 0.236926bb/hand | 0.264391bb/hand | +11.59% |

The fail-closed report is generated at
`neural/runs/v44-large-only/validation-v44-rejected.json`. Run artifacts remain
ignored and no rejected model is routed by the application.

## Label-semantics defect

The investigation found a more fundamental issue than v44 training length.
The legacy turn target generator averaged immediately over exact river cards
and solved only the resulting river games. The flop resolver, however, cuts at
the start of the turn. Legacy schema-v1 labels therefore omitted the complete
turn betting round.

V45 replaces that path with one exact-card public-belief game containing:

- every legal abstract turn action;
- 48 observed public river branches and exactly 44 compatible rivers per fixed
  pair of private hands;
- every legal abstract river action; and
- exact folds, all-in runouts, showdowns, chip accounting, and card removal.

Schema v2, its method identifier, and its checkpoint fingerprint explicitly
encode the corrected continuation semantics. Schema-v1 corpora remain readable
as legacy research inputs but are not valid activation evidence.

## Superseded pre-fix measurements

The measurements below are retained as an investigation audit trail, but they
are not valid exploitability estimates. The original best-response chance
aggregation masked each player's reach for a dealt river while still adding
that child's counterfactual value for private hands containing the river. It
therefore charged the profile for impossible chance branches. The training
loop also used one global discount/averaging clock per single-player traversal
instead of one logical P0-then-P1 alternating round.

## Structural verification

The corrected value-only and full-policy finishes return identical CFVs and
metrics. Tests pin the presence of both turn and river information sets, all 48
observed river branches, deterministic exact chance integration, and equality
with the old direct-river average only when turn betting is mechanically forced
to check. The resumable upgrader preserves belief and replay provenance while
rejecting mismatched checkpoints.

On the dense uniform 4bb turn state `2c,7d,Th,Js`, default DCFR measured:

| Iterations | Exact local exploitability |
| ---: | ---: |
| 20 | 0.963151bb/hand |
| 100 | 0.189254bb/hand |

On the first two authentic frozen-v26 states, the same solver measured:

| State | Pot | Iterations | Exact local exploitability | Wall time |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 7bb | 100 | 0.210020bb/hand | 271s |
| 0 | 7bb | 200 | 0.140692bb/hand | 525s |
| 1 | 5bb | 200 | 0.112178bb/hand | 1,008s |
| 2 | 2bb | 200 | 0.081833bb/hand | 2,375s |
| 0 | 7bb | 400 | 0.128838bb/hand | 1,653s |

The state-0 100-to-200 reduction was only 33.0%, and 200-to-400 then improved
only another 8.4%. This invalidates both the initial optimistic 400-iteration
projection and brute-force scaling of the default configuration. The in-flight
default 400-iteration states 1 and 2 were stopped without artifacts. A corrected
corpus will not be generated until paired authentic pilots identify a solver
configuration or iteration ceiling that passes the gate state by state.

A proper regret-matching-plus/linear-average ablation on the dense 4bb state
was worse at 100 iterations: `0.239932bb/hand` average-policy exploitability
versus default DCFR's `0.189254bb/hand`. Its current policy was worse again at
`0.363678bb/hand`. Regret clipping is therefore rejected as the default rather
than being scaled into the label corpus.

A bounded averaging/regret sweep at 100 iterations measured:

| Variant | Exact local exploitability |
| --- | ---: |
| Default DCFR, delay 20 | 0.187062bb/hand |
| Average exponent 3 | 0.174853bb/hand |
| Average exponent 4 | 0.169118bb/hand |
| Default DCFR, delay 50 | 0.172641bb/hand |
| Positive-regret exponent 1 | 0.233913bb/hand |
| Positive-regret exponent 2 | 0.202710bb/hand |

The average-exponent-4 improvement transferred only weakly to authentic state
0 at 200 iterations: `0.138729bb/hand` versus default `0.140692bb/hand` (1.4%).
The paired authentic state-1 run was stopped when the deeper default plateau
made the old sweep insufficiently promising.

The default DCFR implementation advances discount time on each alternating
single-player traversal. This matches the paper's description of alternating
which player updates on each iteration; an uncommitted attempt to reinterpret
two traversals as one discount round was therefore reverted without producing
artifacts. The paper also identifies LCFR as a useful candidate for HUNL
subgames with severe mistake actions, so a bounded comparison used exact LCFR
parameters `(alpha,beta,gamma)=(1,1,1)`. See [Brown and Sandholm,
2019](https://ojs.aaai.org/index.php/AAAI/article/view/4007).

LCFR failed the first authentic control at 100 iterations:
`0.249362bb/hand` versus default DCFR's `0.210020bb/hand` (18.7% worse). The
remaining dense and authentic LCFR jobs were stopped without artifacts.

## Superseded street attribution

The v4 evaluator restricts an otherwise exact best response to one street at a
time. On authentic state 0 after 100 default iterations it measured:

| Response scope | Gain over the frozen profile |
| --- | ---: |
| Turn only | 0.044062bb/hand |
| River only | 0.132468bb/hand |
| Unrestricted turn + river | 0.207871bb/hand |

The one-street gains are not additive because the unrestricted response can
coordinate deviations across streets. Nevertheless, the result localizes the
dominant standalone weakness: turn-only response is already below `0.05`,
while river-only response accounts for 63.7% of the full gap. The next solver
experiment should allocate additional updates to river information sets while
holding a frozen average turn profile, then use the unrestricted exact best
response to reject any unsafe cross-street result.

The dense uniform 4bb control corroborates the split: `0.040796bb/hand`
turn-only, `0.128512bb/hand` river-only, and `0.187062bb/hand` unrestricted.

## Corrected paired solver

The corrected solver now treats one configured iteration as a complete
alternating round. Player zero updates first, player one updates against that
new strategy, and the average profile is accumulated between the updates with
one DCFR discount clock per round. Both training and exact profile/best-response
evaluation discard a river child for any private hand containing that observed
river. This follows the paired update/profile indexing formalized in
[Burch, Moravčík, and Schmid (2019)](https://arxiv.org/abs/1810.11542). A
regression test constructs a fixed compatible private-hand pair and verifies
exactly 44 contributed river outcomes.

A profiler also found that terminal folds and showdowns repeatedly enumerated
every conflicting hand. Exact per-card range marginals and per-card strength
prefixes reproduce the enumerated values within `1e-10` while reducing the
authentic state-0 100-pass legacy control from roughly 271 seconds to 6.5
seconds. The corrected paired 100-round solve takes roughly 12 seconds because
each round intentionally performs both player updates.

Corrected authentic results are:

| State | Pot | Paired rounds | Averaging delay | Exact local exploitability | Current-policy exploitability |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 7bb | 100 | 20 | 0.025287bb/hand | 0.046524bb/hand |
| 0 | 7bb | 200 | 20 | 0.007241bb/hand | 0.010135bb/hand |
| 0 | 7bb | 400 | 80 | 0.002359bb/hand | 0.006435bb/hand |
| 1 | 5bb | 100 | 20 | 0.017113bb/hand | 0.031341bb/hand |
| 1 | 5bb | 200 | 20 | 0.005286bb/hand | 0.005932bb/hand |
| 2 | 2bb | 100 | 20 | 0.025820bb/hand | 0.086813bb/hand |
| 2 | 2bb | 200 | 20 | 0.007246bb/hand | 0.012202bb/hand |

All three authentic states pass the `0.05bb/hand` exact local gate at 100
paired rounds and improve substantially at 200. Probability sums remain within
`4.5e-16` and zero-sum residuals within `2.2e-15`. The dense uniform 4bb stress
control is harder: it measures `0.170260bb/hand` at 100 rounds but passes at
`0.013427bb/hand` after 400. It is not a served target, but confirms convergence
away from the authentic state distribution.

## Remaining boundary

Passing the exact local turn/river gate is necessary label-quality evidence,
not full-game equilibrium certification. All 256 authentic targets have now
been upgraded at 200 paired rounds. The composed schema-v2 corpus is accepted:

- maximum exact local exploitability: `0.011078829bb/hand`;
- maximum probability-sum error: `4.44e-16`;
- maximum absolute zero-sum residual: `1.95e-14`;
- exact public river branches per target: 48; and
- source policy SHA-256:
  `c78397af5900b3409d3dfc911fce56075cb54ce860c38fc2a1459fe5d56df948`.

The corrected labels differ materially from the legacy incomplete-turn labels.
Across the same 256 states, their mean per-state reach-weighted RMSE is
`0.670920bb` and MAE is `0.498743bb`; the global reach-weighted RMSE is
`0.807219bb`. Mean state RMSE is `0.521040bb` for small pots, `1.338724bb`
for medium pots, and `1.292437bb` for large pots. State 180 reaches
`2.040773bb` RMSE. This confirms that v44's low error against schema-v1 labels
did not measure the continuation function needed by the flop resolver.

The first schema-v2 paired students preserve exact Python/Rust inference parity
but do not pass value accuracy. The pure 256-state model measures
`0.511606bb` and `0.519112bb` holdout RMSE. Adding 64 corrected large-pot
training states improves those values to `0.396528bb` and `0.373431bb`, while
cross-seed prediction correlation remains `0.998612` and maximum runtime parity
error remains below `7.6e-6bb`. The remaining failure is therefore measured
function approximation/coverage, not seed instability or serialization drift.
Additional corrected coverage and resolver-leaf training supplements are being
evaluated without adding the disjoint resolver-leaf evaluation set to training.

## Tuning-only optimization controls

The original 64-state validation partition has now been viewed across several
research configurations. It remains useful as a regression diagnostic, but it
is no longer valid untouched evidence for choosing hyperparameters. Every
subsequent configuration comparison therefore uses only the identical
39-state, pot-stratified tuning partition. A separate selector rejects
mismatched data or split membership, requires two restored best checkpoints
and tuning cross-seed correlation of at least `0.95`, and ranks candidates by
worst-seed then mean-seed tuning RMSE. It never reads validation RMSE.

| Controlled configuration | Seed tuning RMSE | Worst seed | Mean |
| --- | ---: | ---: | ---: |
| Deep GELU, 7k, 50% primary, raw weight 0.25 | 0.282668 / 0.283618 | 0.283618 | 0.283143 |
| Deep GELU, raw weight 1.0 | 0.278360 / 0.292688 | 0.292688 | 0.285524 |
| Deep GELU, 25% primary replay | 0.314755 / 0.317268 | 0.317268 | 0.316012 |
| Xwide GELU, 7k | 0.282224 / 0.281680 | **0.282224** | 0.281952 |
| Deep GELU, constant-rate 14k | 0.302379 / 0.253680 | 0.302379 | **0.278030** |
| Deep GELU, cosine 14k | 0.303846 / 0.283181 | 0.303846 | 0.293514 |
| Deep GELU, cosine 14k, Adam bias correction | 0.316540 / 0.310415 | 0.316540 | 0.313477 |
| Xwide GELU, 7k, batch 24 | 0.256185 / 0.256307 | **0.256307** | **0.256246** |
| Xwide GELU, 10k, batch 24 | 0.251365 / 0.237825 | **0.251365** | **0.244595** |

The 14k constant-rate result proves that additional optimization can help one
seed, but its large cross-seed spread makes it unsafe. Cosine decay did not
stabilize the result. Standard Adam moment bias correction made the two seeds
more alike while making both worse, so it is not the retained optimizer.
Increasing xwide's pot-balanced batch from 12 to 24 reduced worst-seed tuning
RMSE by `0.025917bb` and produced nearly identical paired results. Its
two-output ensemble measured `0.249346bb` tuning RMSE, but this is a variance
diagnostic only; it does not replace the requirement that both independent
release seeds pass. Because both individual checkpoints selected the end of
the 7k budget, the exact winning configuration received one predeclared 10k
continuation check. Its `0.251365bb` worst-seed tuning RMSE beat the 7k result
by `0.004942bb`; tuning cross-seed correlation was `0.999538`, and its
diagnostic ensemble tuning RMSE was `0.239032bb`. The selector therefore froze
xwide GELU, 10k steps, batch 24, constant `0.0003` learning rate, no Adam bias
correction, and 50% authentic-primary replay. The old viewed holdout remained
rejected at a maximum `0.302288bb` and did not participate in that choice.

The frozen configuration is tracked in
`preflop-solver/neural/20bb-v46-value-config.json`. It pins the selector and
report hashes, every training-only supplement hash, the untouched shard
boundary, split parameters, and new release seeds `14721` and `14722`. Exact
tuning ties are resolved from configuration and training-seed data only;
changing holdout metric values cannot change the selected configuration.

Deterministic schema-v3 feature construction is now content-addressed by the
combined corpus digest, feature schema, implementation version, row count, and
group count. The final 926-state combined-corpus cache contains a 3.1MB context
tensor and a 1.218GB query tensor. Its context and query SHA-256 values are
`856bcf3cb49234091c924389a89aaeb2ae7be45b8c4bfce9c0c6ba4531e09251`
and `29f24caf61a1a66eadaf492d4bec4358d41f3c7d34886e29b1e9157b3a286061`.
Both arrays are verified before use and loaded as read-only memory maps. This
cut repeat-run preprocessing from minutes to seconds without changing any
feature, target, split, or strategy gate. These tensors are training-only and
are not part of the browser serving payload.

An independent 512-state authentic corpus was generated in eight resumable,
64-state shards with source seeds `14701` through `14708`. Every component and
the ordered merged corpus passed full schema-v2 revalidation. All 512 turn
boards are distinct; maximum local turn/river exploitability is
`0.016660bb/hand`, maximum probability-sum error is `4.44e-16`, maximum
absolute zero-sum residual is `1.42e-14bb`, minimum belief ESS is `802.5`, and
maximum replicate TV is `0.12810`. The merged bytes have SHA-256
`f5e87c5587801d395040b33a0bb2244c7bed79cb6b5fd77f23e52173ea950f2e`.

Shards 6 and 7 (merged indices 384--511) were reserved in advance as the
128-state final holdout. Candidate hyperparameters were frozen and committed
before that holdout was evaluated. No target CFV or model-error metric from
those shards participated in model or hyperparameter selection.

## V46 fresh-holdout result

The frozen pair completed both 10,000-step runs and was rejected. Seed 14721
measured `0.281479bb` holdout RMSE and seed 14722 measured `0.279646bb`, above
the per-seed `0.25bb` ceiling. Their output correlation was `0.999571`, so the
failure is stable rather than attributable to one unlucky initialization. The
diagnostic two-output ensemble measured `0.275097bb` and does not replace the
per-seed gate. The report SHA-256 is
`9e06723ab8a89a15f2db31ba3c71c24fe7d608cc7f717301103316bd7a741708`.

| Pot band | Seed 14721 RMSE | Seed 14722 RMSE |
| --- | ---: | ---: |
| Small | 0.138970bb | 0.142934bb |
| Medium | 0.448149bb | 0.416931bb |
| Large | 0.502267bb | 0.527444bb |

Both seeds selected step 9,700 and missed the same medium/large-pot and
paired/trips board regimes. More iterations of the unchanged network are
therefore not justified by this result. The holdout is now opened and cannot
be reused as untouched release evidence for a successor. `activationAllowed`
remains false and the active manifest remains empty.

## V47 exact-range pooling pilot

V46 supplied every exact private-combo query to the shared network, but its
head saw only the current combo and handcrafted public range summaries. That
representation could not learn an arbitrary range-conditioned continuation
function. V47 adds a permutation-invariant, joint-reach-weighted pooling pass
over all exact combo query embeddings for each player. Every combo head now
receives the public-state embedding, its own pooled range embedding, the
opponent's pooled range embedding, and its exact private-combo embedding. Card
removal remains exact, the same public state and combo features are retained,
and suit equivariance is unchanged.

The exported `hu-public-belief-combo-value-network-v5` artifact records this as
`joint-reach-weighted-own-and-opponent-query-pooling`. Python and Rust evaluate
the same pooling operation. A three-state smoke comparison spanning indices 0,
398, and 479 measured maximum absolute disagreement of
`0.0000026763bb`, below the frozen `0.0001bb` parity ceiling. The JSON payload
is approximately 9.74MB; the 1.218GB feature cache remains training-only.

The first independent 5,000-step pair improved tuning RMSE to `0.218158bb` and
`0.208671bb`, versus `0.222285bb` and `0.221602bb` for frozen V46 despite using
half as many steps. Its diagnostic opened-holdout results were `0.279023bb` and
`0.246452bb`. Those holdout values cannot select V47 and do not constitute new
release evidence. A second independent pair was predeclared to confirm the
tuning improvement before freezing the architecture. That pair reproduced the
gain at `0.208348bb` and `0.207010bb`; its tuning prediction correlation was
`0.999643`. Across all four pooled seeds, the worst tuning RMSE is
`0.218158bb` and the mean is `0.210547bb`, versus `0.222285bb` worst and
`0.221943bb` mean for V46.

The confirmation pair measured `0.254580bb` and `0.254240bb` on the already
opened diagnostic holdout. Its report is consequently rejected under the old
per-seed release gate, even though the diagnostic two-output ensemble is
`0.244941bb`. Neither result was available to the selector and none may serve
as fresh evidence. The confirmation report SHA-256 is
`9edaa8e5280b84fe13656501c1877482b4f729f761cf06b68c836a41684210ec`.

The tuning selector now groups identical configurations across every
independent replication. It ranks the worst per-seed tuning RMSE over the
entire group, then the all-seed mean, then the worst diagnostic pair-ensemble
tuning RMSE. Repeated pairs therefore increase the evidence burden and cannot
be cherry-picked as separate candidates. Opened-holdout metrics are not read
by the selector.

The resulting V48 freeze selects `xwide-gelu-pooled` at 5,000 steps using four
independent tuning seeds. Selector artifact SHA-256
`b240f573e664cb57193bdb724677f593309f54d250d1a527903a4c636059e34f`
binds configuration SHA-256
`842e3ca0340d700c31847b1e88c0f4f137144bd19cbd4612da948dfeb931b657`.
Release training seeds 14921 and 14922 are disjoint from every selection seed.
The future 128-state holdout is reserved for source seeds 14901 and 14902 and
will be generated only after this freeze is committed. Activation remains
false.

The remaining sequence is corrected value-network selection, disjoint leaf and
matched-resolver selection, continuation-cache and preflop DCFR regeneration,
action-EV uncertainty measurement, learned-response red teaming, and an
independent one-sided 99% full-game exploitability upper bound. The public
model stays unavailable until every declared activation gate passes.

## Tighter full-game certificate pilot

The earlier clairvoyant certificate revealed both private hands and the full
runout. Its implementation also settled pre-river all-ins through the ordinary
deterministic rollout-equity approximation rather than the sampled complete
board. The latter is unsuitable for a rigorous chance-sampling certificate, so
all earlier numeric certificate results remain diagnostic only. Certificate
showdowns now use the sampled five-card runout exactly.

A new conservative pilot removes the largest information leak: the responder
sees its own cards and the future public runout, but opponent cards remain
hidden behind one common conditional particle set. Every action at a public
history is selected jointly across all opponent particles reaching that
history. It therefore cannot choose an action separately for each hidden hand.
The empirical optimum has non-negative sample-optimization bias by convexity,
so its expectation is still above the corresponding relaxed best response,
which in turn is above the legal imperfect-information best response.

The outer confidence interval uses the one-sided empirical Bernstein bound
from [Maurer and Pontil (2009)](https://arxiv.org/abs/0907.3740), scaled from
`[0,1]` to `[0,20bb]`. The artifact also retains the looser Hoeffding margin for
audit comparison and pins the SHA-256 of the routed policy bytes it evaluates.
Two deliberately tiny v26 research checks measured point
values of `1.741143bb/hand` with 2 outer deals and 4 opponent particles and
`2.489149bb/hand` with 8 outer deals and 16 particles. Their 99% upper bounds
both correctly cap at `20bb`; neither is release evidence. The variation and
large finite-sample margins show that this pilot needs substantially larger
outer and hidden-hand samples and, more importantly, that future-board
revelation may remain too loose. Activation therefore remains blocked on the
declared full-game bound even though the evaluator itself is now materially
tighter and its chance settlement is exact.

## Causal sample-game certificate pilot

The next certificate removes future-board revelation as well. Each outer game
fixes only the responder's private cards, then constructs nested empirical
flop, turn, river, and hidden-opponent branches. At a responder node, scenarios
with the same currently visible public board and betting history share one
action. Later public cards therefore cannot influence an earlier decision.

For any fixed legal response, the nested terminal-deal average is unbiased.
The expectation of the empirical maximum is at least the maximum expected
value, so sample optimization remains a conservative relaxation. Independent
outer games retain the same one-sided empirical-Bernstein confidence bound.
The artifact pins the policy bytes, branch counts, total scenarios per outer
game, and exact tree-node count.

On the rejected routed v26 seed 5101, a two-outer-game smoke with two public
branches per street and four hidden hands per runout (32 terminal scenarios per
outer game) measured `0.962029bb/hand`. An eight-game pilot measured
`2.490222bb/hand` with `0.536798bb` standard error. Both 99% upper bounds
correctly cap at 20bb because the sample counts are tiny. These are diagnostic
only: the new evaluator closes the future-information defect, but v26 and the
pilot sample size remain far from the release gate.

## V48 fresh holdout and continual-resolver rejection

After the V48 freeze was committed, two independent 64-state authentic shards
from seeds 14901 and 14902 were generated and reserved exclusively as the
128-state holdout. The merged 512-state dataset has SHA-256
`0532dec5f96a79fc42c8beddc54c2a31985f7d383bfe675a9f160a7fcf8d7da6`.
All 512 input fingerprints are unique; every component is accepted, has valid
probability sums and zero-sum settlement, and remains below the target-label
exploitability ceiling.

The frozen release pair passed the value-only gates. Seeds 14921 and 14922
measured fresh holdout RMSE of `0.227714bb` and `0.245569bb`, respectively,
with prediction correlation `0.999481`. Their tuning RMSE values are
`0.203753bb` and `0.218561bb` with correlation `0.999473`. Report SHA-256 is
`2935b92570024c8515cd53438f09f6ec2ffa00eeb98119b9436dddaa4772d280`.
Six-state Python/Rust checks for each seed measured maximum absolute errors of
`0.0000055874bb` and `0.0000053585bb`, below the `0.0001bb` parity gate.

Passing the value-only gate did not pass the strategy gate. On the dry-low
`2c,7d,Th` root, 100-iteration cross-fit resolvers measured
`0.292119bb/hand` and `0.280040bb/hand` depth-limited exploitability. This is
an 87--88% reduction from the unresolved uniform strategy, but remains more
than five times the declared `0.05bb/hand` local ceiling. One failed root is
sufficient to reject the maximum-over-roots gate, so the other reserved roots
were not spent on the same candidate.

The disjoint nine-state legacy resolver-leaf evaluation explains the gap.
V48's resolver-reach-weighted RMSE is `0.459101bb` and `0.479159bb`; its
large-pot RMSE is `0.694966bb` and `0.661119bb`. The authentic holdout is
dominated by small pots and therefore cannot by itself certify the leaf
distribution induced by continual solving. V48 remains inactive.

## Resolver replay pilots and V49 corpus freeze

The trainer now accepts one explicit sampling weight per supplemental corpus.
This keeps authentic states, coverage states, large-pot states, and
resolver-leaf states in separate auditable replay strata instead of forcing a
single shared weight. A 10x resolver-leaf pilot improved legacy
resolver-reach-weighted RMSE to `0.441636bb` and `0.415193bb`, but degraded the
opened authentic holdout to `0.271980bb` and `0.291413bb`. A moderate 3x pilot
measured `0.433478bb` and `0.408794bb` on resolver reach while also failing the
opened authentic holdout at `0.270423bb` and `0.280363bb`. Its report SHA-256
is `7fb793b53de34cf1a93be0af7e531633c9d75cd3bd67f247681edec5d60333a5`.
Both experiments show that resolver replay has leverage, but oversampling the
same 18 leaves from six roots is not a valid substitute for broader coverage.

`neural/20bb-v49-resolver-reach-corpus.json` therefore freezes 24 disjoint
training roots across dry, connected, two-tone, monotone, paired, and trips
textures. The roots are split across both V48 source seeds and sample one
small-, medium-, and large-pot leaf per root, producing 72 exact complete
turn/river labels. Twelve additional roots are reserved for a 36-state
evaluation corpus whose labels may only be generated after the next candidate
configuration is frozen. The validator checks pinned source hashes, declared
texture counts, exact and suit-isomorphic separation from training,
evaluation, and legacy roots, completed shard provenance, and fail-closed
activation. Any failed successor requires new independent evaluation roots.

Configuration selection uses the two 36-label training shards as opposite
cross-validation folds before the reserved evaluation labels are generated.
Every diagnostic now records the exact model SHA-256, and the selector verifies
that the evaluated bytes are the corresponding report's restored checkpoint,
that neither held-out fold appears in its training components, and that the
candidate and V48 baseline use identical folds. A candidate must retain
authentic tuning RMSE at or below `0.25bb` and improve the worst cross-fit
resolver-reach RMSE by at least 20%. If none does, selection is rejected and no
fresh release holdout is spent.

The cross-fit experiment itself was frozen before either new training shard
completed. Four paired configurations compare uniform replay, increased
authentic protection with compensating resolver weight, stronger protection,
and a resolver-heavy control. Each configuration trains once on each
36-label fold and evaluates only on the opposite fold, using four unique seeds
per configuration and 16 seeds in total. The plan validator pins every input,
rejects duplicate configurations or seeds, enforces symmetric fold swapping,
and forbids treating the already opened V48 holdout as release evidence.
