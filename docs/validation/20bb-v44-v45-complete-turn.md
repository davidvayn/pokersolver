# 20bb v44 rejection and v45 complete-turn correction

Status: rejected; no active manifest was modified.

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

## Street attribution

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

## Remaining boundary

Passing the exact local turn/river gate is necessary label-quality evidence,
not full-game equilibrium certification. After an accepted v2 corpus exists,
the remaining sequence is paired value-network training, exact parity,
disjoint leaf and matched-resolver selection, continuation-cache and preflop
DCFR regeneration, action-EV uncertainty measurement, learned-response red
teaming, and an independent one-sided 99% full-game exploitability upper bound.
The public model stays unavailable until every declared activation gate passes.
