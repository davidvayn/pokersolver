# 20bb v51 postflop action-policy handoff

## Boundary and release state

- Target-export implementation baseline: commit `c888a9c` (`Add range-conditioned postflop policy distillation`).
- The paired action-target jobs described below completed successfully.
- The mixed-replay v51 student pilot has now been distilled and evaluated. Its complete evidence is pinned in `neural/20bb-v51-postflop-action-pilot.json`.
- `activationAllowed` remains `false`. Both full-game means improved, but the required exploitability gates still fail by a wide margin.
- Continue only the served action-policy correction. Do not add value-oracle, untouched-root, or unrelated diagnostic work.

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

## Exact next sequence

Increase authentic root coverage without increasing the 100,000-row memory ceiling. Reuse seeds 16001 and 16002 with two roots so each replacement corpus includes its prior deterministic root plus one new root:

```sh
target/release/preflop-solver postflop-action-targets \
  --effective-stack-bb 20 \
  --networks neural/runs/20bb-v27-routed-seed7601-s5000.json \
  --value-network neural/runs/v49-resolver-reach/release/uniform-expanded/turn-value-range-seed15301.json \
  --roots 2 --turn-leaves-per-root 1 \
  --flop-iterations 50 --flop-averaging-delay 12 \
  --turn-river-iterations 50 --turn-river-averaging-delay 5 \
  --seed 16001 --threads 4 --exploration 0.05 --max-records 100000 \
  --output neural/runs/20bb-v52-postflop-action/targets-seed16001.jsonl.gz \
  --report neural/runs/20bb-v52-postflop-action/targets-seed16001-report.json

target/release/preflop-solver postflop-action-targets \
  --effective-stack-bb 20 \
  --networks neural/runs/20bb-v27-routed-seed7602-s5000.json \
  --value-network neural/runs/v49-resolver-reach/release/uniform-expanded/turn-value-range-seed15302.json \
  --roots 2 --turn-leaves-per-root 1 \
  --flop-iterations 50 --flop-averaging-delay 12 \
  --turn-river-iterations 50 --turn-river-averaging-delay 5 \
  --seed 16002 --threads 4 --exploration 0.05 --max-records 100000 \
  --output neural/runs/20bb-v52-postflop-action/targets-seed16002.jsonl.gz \
  --report neural/runs/20bb-v52-postflop-action/targets-seed16002-report.json
```

Then repeat the validated mixed-replay update from the frozen v26 weights:

```sh
PYTHONPATH=neural .venv-neural/bin/python neural/distill_postflop_policy.py \
  --dataset-a neural/runs/20bb-v52-postflop-action/targets-seed16001.jsonl.gz \
  --dataset-b neural/runs/20bb-v52-postflop-action/targets-seed16002.jsonl.gz \
  --initial-weights-a neural/runs/20bb-v26-probability-combined-distilled/seed-0.safetensors \
  --initial-weights-b neural/runs/20bb-v26-probability-combined-distilled/seed-1.safetensors \
  --output-dir neural/runs/20bb-v52-postflop-distilled-roots2-mix25-s200 \
  --hidden-sizes 512,256 --steps 200 --batch-size 512 \
  --learning-rate 0.00003 --cross-seed-replay-probability 0.25 --seed 16301
```

Require all own-root and cross-root metrics to improve, route with the same frozen v27 preflop weights, and rerun the exact certificate controls pinned in `neural/20bb-v51-postflop-action-pilot.json`. Keep the successor only if both full-game means improve. Do not activate any candidate unless every release gate in `neural/20bb-v50-full-hand-candidate-freeze.json` passes; never infer a pass from teacher fit or cross-seed stability alone.

## Verification already completed

- `cargo test --release`: 98 library tests and 3 CLI tests passed after the implementation change.
- Relevant Python suite: 9 tests passed.
- One-step end-to-end smoke test loaded the exact features and slightly improved held-out teacher MAE for both seeds.
- `git diff --check` passed before the implementation commit.
