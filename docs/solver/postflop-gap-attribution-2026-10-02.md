# Postflop gap: frozen action-value diagnosis

Date: October 2, 2026 (PDT). Status: research diagnosis; no policy promotion.

This replays the completed September 21 benchmark without retraining or
re-solving its candidates. The original 12-root, two-seed benchmark is pinned by
completed-manifest SHA-256
`9a7f178b88fb595235c22e83f8fb173c9413d8692da34f6990a8a0540faf0ad8`.
Four deliberately selected development/control roots were probed with each
retained network against each frozen policy's exact beliefs and saved native
64-update continuation packets. All 16 probes completed; local receipts are in
`preflop-solver/neural/runs/local-postflop-gap-20261002-diagnostic-a/`.

The most costly *local* action-ranking comparisons are:

| Frozen root and highest-loss node per seed | Seed 100101 matched model: native-best agreement / root-weighted ranking loss | Seed 100102 matched model: native-best agreement / root-weighted ranking loss |
| --- | ---: | ---: |
| Single-raised high rainbow; BB check, BTN decision | 50.2% / 0.7881bb | 57.8% / 0.5314bb |
| Three-bet monotone; BB check, BTN decision | 17.0% / 0.5687bb | 15.6% / 0.5678bb |
| Limped paired; BTN after BB check for seed 100101, BB initial action for seed 100102 | 50.9% / 0.0973bb | 77.6% / 0.1698bb |
| Three-bet high rainbow control; BB check, BTN decision | 55.6% / 0.2423bb | 43.8% / 0.2416bb |

The single-raised node's native-best agreement is similarly low with the
*crossed* model (51.3% and 56.5%, respectively). Replacing the value model on
unchanged final-policy beliefs therefore does not make the mismatch disappear.
But these are **prediction comparisons**, not trained cross-seed policies.
Neither they nor local losses isolate chance-seed effects, establish that the
native64 reference is accurate, or measure full-game exploitability. Local
node losses overlap and must not be summed.

The evidence makes action-relevant value error a credible bottleneck: the
retained predictions often rank expensive flop actions differently from the
saved native reference. The next discriminating experiment is a matched
learned-leaf versus native-leaf flop solve at the same iteration count, chance
seed, game and evaluator. Its response gain—not regression error or action
agreement alone—will decide whether native leaves help. Separately increasing
flop updates will test search convergence. Until those comparisons finish,
this is an attribution hypothesis, not a solver fix.

Reproduce the read-only probe from `preflop-solver/neural/` with
`run_native_action_value_probe.py benchmark` and the pinned hashes recorded in
its manifest. The underlying native packets and candidate policies are not
modified by the probe.
