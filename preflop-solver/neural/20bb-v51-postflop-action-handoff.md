# 20bb v51 postflop action-policy handoff

## Boundary and release state

- Target-export implementation baseline: commit `c888a9c` (`Add range-conditioned postflop policy distillation`).
- The paired frequency and exact action-EV target jobs described below completed successfully.
- The mixed-replay v51 student remains the best routed policy. The complete v51 and rejected v55 evidence is pinned in `neural/20bb-v51-postflop-action-pilot.json` and `neural/20bb-v55-ev-aware-pilot.json`.
- `activationAllowed` remains `false`. Both full-game means improved, but the required exploitability gates still fail by a wide margin.
- Continue only the served action-policy correction. Do not add value-oracle, untouched-root, or unrelated diagnostic work.
- No training or evaluation job is running at this handoff.
- The preflop and postflop networks are already combined in `neural/runs/20bb-v51-routed-seed7601-mix25-s200.json` and `neural/runs/20bb-v51-routed-seed7602-mix25-s200.json`. No additional solver or attribution work is required merely to create a full-hand bundle.

## 2026-08-07 implementation checkpoint

The EV-aware policy objective is implemented and was evaluated in routed candidates:

- Flop and joint turn/river average strategies carry optional row-major `[combo][action]` counterfactual action EVs. Existing serialized strategy artifacts remain readable through the serde default, and the standalone river solver continues to omit this training-only field.
- The joint turn/river diagnostic walk preserves observed river chance keys and exact card removal. Every exported value is normalized by compatible opponent mass, finite-checked, and dimension-checked.
- Postflop action records now require and serialize one EV per legal action. Their metadata declares `evaluates_trajectory_action_values: true` and identifies the exact solver average-profile method. Missing values fail closed.
- The Python loader enforces declared EV label dimensions and finiteness. `distill_postflop_policy.py` adds `--ev-regret-scale` (default `0`) and `--ev-regret-cap-bb` (by default the full `2 × depth` utility span). A positive scale combines reach-weighted cross-entropy with bounded expected action regret; scale zero calls the original compiled frequency-only step unchanged.
- Reports now include `weightedExpectedEvLossBb` whenever EV labels are present. Tests cover masked actions, equal-EV support, dominated-action penalties, caps, both compiled objective paths, solved-policy finiteness, root EV reconstruction, river chance export, and missing labels.

The implementation was first checked with deliberately unconverged two-iteration teachers, then used to regenerate the converged v54 teachers and evaluate direct and conservative continual v55 updates. No v55 pair improved both routed full-hand means, so v51 remains the best measured candidate.

The current routed v50 candidate is frozen in `neural/20bb-v50-full-hand-candidate-freeze.json`. Its controlled certificate is recorded in `neural/20bb-v50-full-hand-certificate-pilot.json`: paired relaxed means are 2.2582742976 and 2.2056883553 bb/hand, and both one-sided 99% upper bounds are 20 bb/hand. The v26 postflop action student is therefore the measured policy gap this experiment addresses. The v27 preflop students stay frozen.

## Pinned inputs

| Role | Path | SHA-256 |
| --- | --- | --- |
| Routed source A | `neural/runs/20bb-v27-routed-seed7601-s5000.json` | `c5dba74744d817423c94803c2a1c92dc8a3acacd3b5f58094c700704c0fcbf92` |
| Routed source B | `neural/runs/20bb-v27-routed-seed7602-s5000.json` | `bf6a303459f2c853baaef8b62045119ba1a58332792c8e29f0fff6a8cfb0922f` |
| V49 teacher A | `neural/runs/v49-resolver-reach/release/uniform-expanded/turn-value-range-seed15301.json` | `4bd8eeac53849e3a69183cfa435ca32446ec8d9372cafe2053e3510e8eadec81` |
| V49 teacher B | `neural/runs/v49-resolver-reach/release/uniform-expanded/turn-value-range-seed15302.json` | `eb065e76fa91bb4d789a59962f7397ac8b4131f30648ac5955ca4ac05a9da9b1` |
| Frozen postflop weights A | `neural/runs/20bb-v26-probability-combined-distilled/seed-0.safetensors` | `cbc7135f6c4aa38232d511a71d05db1c0f6c5aa0fcd8b58e799c1fbc311eb427` |
| Frozen postflop weights B | `neural/runs/20bb-v26-probability-combined-distilled/seed-1.safetensors` | `fba9ae94b3af651a0057cd0e20c91de0c1be0985821b33db5470f2a95c97d07f` |
| Frozen preflop weights A | `neural/runs/20bb-v27-tabular-distilled-s5000/seed-0.safetensors` | `151fc1d90a5c03f105e543c54d02d097df51c66a422d01e696731b9c83b409dd` |
| Frozen preflop weights B | `neural/runs/20bb-v27-tabular-distilled-s5000/seed-1.safetensors` | `fcb3c62aad24a86bd04b4bdd1be8a1cbe0522a6f6552f830a4a3101965e00a9e` |

## Completed paired action targets

Both jobs used one authentic preflop-conditioned root, one turn leaf, 50 flop DCFR iterations with averaging delay 12, 50 turn/river iterations with averaging delay 5, exploration 0.05, four threads, and a deterministic street-stratified reservoir capped at 100,000 rows. Reservoir weights include inverse-inclusion correction.

| Field | Seed 16001 | Seed 16002 |
| --- | ---: | ---: |
| Corpus | `neural/runs/20bb-v51-postflop-action/targets-seed16001.jsonl.gz` | `neural/runs/20bb-v51-postflop-action/targets-seed16002.jsonl.gz` |
| Corpus SHA-256 | `45501e6619b9a6f9803af5eefcb98fb99c95a2ab8dab09a8f3c4d80a7425d6d8` | `be9d73a2248898086df5c5eb3bf171dde378ef24d74adcda148c8c6af7a85364` |
| Corpus bytes | 19,698,145 | 12,860,204 |
| Report | `neural/runs/20bb-v51-postflop-action/targets-seed16001-report.json` | `neural/runs/20bb-v51-postflop-action/targets-seed16002-report.json` |
| Report SHA-256 | `0415bddc8f502d424a1bdcd2452036232441764fbae552583efefa6191c9afd1` | `d9a032b6d725c987a1189e1762db553ceb51e5b42c4ac60c4ba4b767f555f68f` |
| Candidate rows | 2,176,748 | 370,892 |
| Retained rows | 77,235 | 57,739 |
| Flop / turn / river rows | 18,357 / 13,878 / 45,000 | 7,058 / 5,681 / 45,000 |
| Maximum flop local bound, bb/hand | 0.2561495761 | 0.0876163689 |
| Maximum turn/river local bound, bb/hand | 0.1088612312 | 0.0230528524 |
| Export status | `accepted_for_training` | `accepted_for_training` |

Independent streaming verification matched each header and report count, found finite positive weights and one feature hash per legal action, and found maximum target-probability sum errors of `7.6000000071e-08` and `7.9499999917e-08`. Both exports report `truncated: false`.

The local bounds above do not qualify either teacher for release. `accepted_for_training` means only that the complete-tree policies are valid supervised correction targets. The final routed student must pass the unchanged full-game release gates.

## Completed v51 mixed-replay pilot

Training each student only on its own one-root corpus caused a cross-seed regression. The retained configuration used 25% cross-seed replay for 200 steps at learning rate `0.00003`. It improved own-root and cross-root action-frequency MAE, primary agreement, and maximum aggregate action delta for both students. Distillation and routed export reproduce byte-for-byte.

The unchanged full-game certificate improved candidate 7601 from 2.2582742976 to 2.1763580639 bb/hand and candidate 7602 from 2.2056883553 to 2.0877645110 bb/hand on the paired chance design. Their one-sided 99% upper bounds remain 20 bb/hand, so this is a rejected directional pilot rather than a releasable model.

## Rejected v52 root-expansion pilot

The two-root corpora passed integrity and both distilled students improved every own/cross held-out metric from v26. Nevertheless, the unchanged full-game means regressed by 2.61% and 2.54% versus v51. A lower-rate continual update from v51 also failed its held-out checks and was not routed. Complete evidence is pinned in `neural/20bb-v52-postflop-roots2-pilot.json`; retain v51.

The additional roots had worse maximum local teacher bounds at the same 50 iterations.

## Rejected v53 100-iteration pilot

At 100 iterations, seed 16001's flop and turn/river bounds improved from 0.2561/0.1089 to 0.2185/0.0135 bb/hand. Seed 16002's turn/river bound improved from 0.0231 to 0.0080, but its flop bound slightly regressed from 0.0876 to 0.0895. The routed candidate 7601 improved 1.99% versus v51, while candidate 7602 regressed 1.50%. A street-hybrid seed 7602 also regressed. Complete evidence is pinned in `neural/20bb-v53-postflop-i100-pilot.json`; retain v51.

## Rejected v54 200-iteration/alignment pilot

At 200 iterations, all four local teacher bounds improved: seed 16001 reached 0.1900 flop / 0.00379 turn-river, and seed 16002 reached 0.08686 / 0.00185 bb/hand. A 200-step student improved 15/16 exact held-out comparisons versus v51, but only candidate 7601 improved in the routed test. A 1,000-step student improved all 16 frequency-fit comparisons, yet both routed means regressed to 2.1824 and 2.1898 bb/hand. Complete evidence is pinned in `neural/20bb-v54-postflop-i200-pilot.json`; retain v51.

This closes the root-count, solver-iteration, and student-step ladders. Frequency-only cross-entropy is the remaining measured policy blocker: better frequency fit is not producing lower exploitability.

## Completed v55 EV-aware smoke

The exact evidence is pinned in `neural/20bb-v55-ev-aware-smoke.json`. Both one-root, one-leaf, two-iteration action-EV exports reproduced byte-for-byte with four threads. Each retained 10,000 finite, dimension-matched action-value rows; the maximum target sum error was `6.4e-8`, and values remained within `[-20, 20]` bb. The zero-scale distillation also reproduced byte-identical weights and reports.

Because the observed action gaps span up to 40bb, the original 5bb cap could improve its saturated objective while worsening the uncapped reported EV metric. The distiller now defaults to the full `2 × depth` utility span. On identical 20-step batches, scale 1 with a 40bb cap was the first tested configuration to lower own- and cross-seed expected EV loss for both students while improving the principal frequency metrics. This accepts the objective for a converged-target pilot, not either smoke student for routing or release. Activation remains false.

## Rejected v55 converged EV-aware pilot

The complete evidence, hashes, and settings are pinned in `neural/20bb-v55-ev-aware-pilot.json`. The paired action-EV corpora used the exact v54 one-root, one-leaf, 200-iteration settings. They retained 71,794 and 56,190 finite, dimension-matched rows, reproduced all four v54 local bounds, kept action values within `[-20, 20]` bb, and had maximum target-probability sum errors below `8.5e-8`.

A direct 200-step update from v26 improved all own/cross expected-EV and frequency-fit checks, but its routed means were 2.1590190 and 2.1253586 bb/hand: candidate 7602 regressed by 0.0375941 versus v51. A conservative 20-step continual update from v51 improved expected EV loss for both students, but candidate 7602 again regressed by 0.0238347. Raising cross-seed replay to 50% still regressed candidate 7602 by 0.0213288, so candidate 7601 was skipped under the fail-fast paired rule.

This closes further local action-EV scale, step-count, and replay-mixture sweeps. Exact action values against a frozen local average policy improve that supervised objective but do not reliably reduce bilateral full-game exploitability after both policies change. Every evaluated one-sided 99% upper bound also remained 20 bb/hand. Retain v51 and keep activation false.

## Completed v56 causal-attribution implementation smoke

The implementation and four-corpus evidence are pinned in `neural/20bb-v56-causal-attribution-smoke.json`. The new Rust command freezes the exact information-set-consistent response selected by the existing causal certificate, replays that same response, and emits only reached postflop policy actions with negated responder utilities and exact chance/reach weights. A deterministic thread/street-stratified reservoir bounds memory and applies inverse-inclusion correction.

All four authentic v51 smoke corpora reconstructed their root response values within `1.8e-15` bb. Rust/Python feature hashes matched for 12,070 retained rows, action values stayed in `[-20, 20]` bb, and probability sums remained valid. The KL-capped mirror-descent distiller has independent-corpus fail-closed gates, but no student was trained or routed at this checkpoint.

This work is not needed to combine the frozen preflop and postflop networks; those routed v51 bundles already exist. It is only the next policy-action experiment if work continues toward lower full-game exploitability.

## Rejected v56 causal trust-region pair

The exact paired evidence is pinned in `neural/20bb-v56-causal-trust-pilot.json`. Both 20-step students failed before routing. Candidate 7601 improved the training and independent fixed-response objectives, but its maximum training-node KL was `0.00720`, above the declared `0.005` trust bound. Candidate 7602 reached `0.00604` and slightly regressed the independent fixed-response value. No routed bundle or full-game certificate was produced; retain v51.

The pilot also exposed a cross-runtime numerical boundary: all 503,297 postflop parameters match the attributed routed bundles exactly, while Rust scalar inference and MLX matrix inference differ by up to 0.00342 action probability. The distiller now verifies exact artifact identity, reports that bounded numerical difference, and leaves final acceptance to Rust. Do not loosen the trust gate or route a student using MLX-only evidence.

## Retained v57 exact-Rust projected pair

The exact Rust evaluator now reconstructs every causal-attribution decision,
verifies the canonical feature hash, and scores candidate action values with the
same dense inference implementation used by the full-game certificate. The
distiller evaluates every optimizer checkpoint and retains only a checkpoint
that improves both independently seeded fixed-response corpora while staying
inside the declared maximum-node and reach-weighted KL bounds.

On the unchanged 20-step trajectories, candidate 7601 selected step 8 and
candidate 7602 selected step 3. Their exact independent value gains were
`0.00635145bb` and `0.00461654bb`; maximum train/validation node KL remained
below `0.005`. Both then improved the unchanged paired full-game means: 7601
moved from `2.17635806` to `2.16690165bb/hand`, and 7602 moved from
`2.08776451` to `2.08427165bb/hand`. Complete hashes and gates are pinned in
`neural/20bb-v57-rust-projected-trust-pilot.json`.

V57 is therefore the best measured paired static policy, but activation remains
false: both means are far above `0.05bb/hand` and both one-sided 99% upper
bounds remain `20bb/hand`. The small improvement confirms that causal policy
updates point in the right direction; it does not support repeating static
frequency or fixed-response fitting as the primary route to release.

The next policy architecture should reuse the existing public-belief flop
resolver and exact-card turn/river solver, add a safe multi-valued continuation
boundary, and serve or distill the resulting range-conditioned policy. This is
the common mechanism in continual re-solving, safe nested subgame solving, and
public-belief search. Do not import ReBeL or PokerRL wholesale: their public
implementations do not provide this repository's HUNL game/runtime contract.

## Exact next sequence

Continue only with policy-action updates aligned to the full-game exploitability certificate:

1. Keep the v27 preflop and retained v57 postflop policies frozen as the baseline.
2. If the goal is only a combined full-hand artifact, use the two existing routed v51 bundles and stop here.
3. If the goal remains reducing exploitability, continue with the safe range-conditioned public-belief serving path described above. Do not sweep the rejected local objectives or loosen the KL bound.
4. Route every new paired candidate through the unchanged certificate and reject it immediately if either mean fails to improve. Activation remains false unless the existing exploitability and normal release gates all pass.

Do not resume more roots, local solver iterations, frequency-only training, or local fixed-policy EV scale/step/replay sweeps; v52–v55 already falsified those paths. Do not activate any candidate unless every release gate in `neural/20bb-v50-full-hand-candidate-freeze.json` passes.

## Verification at handoff

- `cargo test --release`: 99 library tests and 3 CLI tests passed after the action-EV implementation and root reconstruction test.
- Full neural unittest discovery is rerun after each EV-aware objective change (186 tests after the full-span default was added).
- A direct positive-scale compiled-step smoke returned a finite loss.
- `cargo fmt`, Python byte compilation, and `git diff --check` passed.
- The converged v55 corpora, students, and routed artifacts live under ignored `neural/runs/20bb-v55-*`; the tracked v55 evidence contains their hashes and rejection decision.
- No training or evaluation process is running. The causal-certificate attribution implementation is complete; no future agent should regenerate v52-v55 targets or repeat local-objective sweeps.
