# 20bb v51 postflop action-policy handoff

## Boundary and release state

- Target-export implementation baseline: commit `c888a9c` (`Add range-conditioned postflop policy distillation`).
- The paired action-target jobs described below completed successfully.
- The mixed-replay v51 student pilot has now been distilled and evaluated. Its complete evidence is pinned in `neural/20bb-v51-postflop-action-pilot.json`.
- `activationAllowed` remains `false`. Both full-game means improved, but the required exploitability gates still fail by a wide margin.
- Continue only the served action-policy correction. Do not add value-oracle, untouched-root, or unrelated diagnostic work.
- No training or evaluation job is running at this handoff.

## 2026-08-07 wrap-up checkpoint

The EV-aware policy objective is now implemented but has not yet produced a candidate:

- Flop and joint turn/river average strategies carry optional row-major `[combo][action]` counterfactual action EVs. Existing serialized strategy artifacts remain readable through the serde default, and the standalone river solver continues to omit this training-only field.
- The joint turn/river diagnostic walk preserves observed river chance keys and exact card removal. Every exported value is normalized by compatible opponent mass, finite-checked, and dimension-checked.
- Postflop action records now require and serialize one EV per legal action. Their metadata declares `evaluates_trajectory_action_values: true` and identifies the exact solver average-profile method. Missing values fail closed.
- The Python loader enforces declared EV label dimensions and finiteness. `distill_postflop_policy.py` adds `--ev-regret-scale` (default `0`) and `--ev-regret-cap-bb` (default `5`). A positive scale combines reach-weighted cross-entropy with bounded expected action regret; scale zero calls the original compiled frequency-only step unchanged.
- Reports now include `weightedExpectedEvLossBb` whenever EV labels are present. Tests cover masked actions, equal-EV support, dominated-action penalties, caps, both compiled objective paths, solved-policy finiteness, root EV reconstruction, river chance export, and missing labels.

This checkpoint deliberately did not regenerate corpora, train students, route a full-hand model, or change activation. V51 remains the best measured candidate.

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

## Exact next sequence

Continue only the EV-aware action-policy experiment; implementation is complete enough for a smoke, not for release:

1. Build release and generate two tiny deterministic target corpora (one root, one turn leaf, 2 flop and 2 turn/river iterations, averaging delays 0, at most 10,000 records) from the pinned seed A/B routed sources and value networks above.
2. Independently stream both gzip files. Require metadata `evaluates_trajectory_action_values: true`; every retained record must have equal nonzero lengths for actions, targets, feature hashes, and `action_values_bb`, with finite EVs and valid target sums. Repeat each export once and compare SHA-256 for determinism.
3. Run a zero-scale tiny distillation twice from identical inputs and require byte-identical weights/reports to establish the compatibility path. Then run a very small paired EV-aware smoke, initially `--ev-regret-scale 0.25 --ev-regret-cap-bb 5`, and require finite loss plus lower held-out `weightedExpectedEvLossBb` without nonsensical frequency metrics.
4. Only if both smokes pass, regenerate paired one-root targets using the proven v54 200-iteration settings and the same pinned sources. Do not add roots or increase solver iterations.
5. Run a short paired scale comparison using the same batches/seeds, select only a scale that improves held-out expected EV loss for both students while keeping action-frequency MAE, primary agreement, and aggregate action deltas sane.
6. Route that pair through the unchanged full-hand causal certificate. Retain it only if both routed means improve over v51. Activation remains false unless every existing release gate passes.

Do not resume more roots, higher DCFR iterations, or longer frequency-only distillation; v52–v54 already falsified those paths. Do not activate any candidate unless every release gate in `neural/20bb-v50-full-hand-candidate-freeze.json` passes.

## Verification at handoff

- `cargo test --release`: 99 library tests and 3 CLI tests passed after the action-EV implementation and root reconstruction test.
- Full neural unittest discovery: 185 tests passed after adding the EV-aware objective tests.
- A direct positive-scale compiled-step smoke returned a finite loss.
- `cargo fmt`, Python byte compilation, and `git diff --check` passed.
- No new corpus or weights have been generated; the next agent must start at the tiny paired export above.
