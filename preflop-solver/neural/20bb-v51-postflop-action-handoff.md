# 20bb v51 postflop action-policy handoff

## Boundary and release state

- Implementation baseline: commit `c888a9c` (`Add range-conditioned postflop policy distillation`).
- The paired action-target jobs described below completed successfully.
- No v51 student has been distilled or evaluated yet.
- `activationAllowed` remains `false`. These targets are training inputs, not a release candidate or an equilibrium certificate.
- Continue only the served action-policy correction. Do not add value-oracle, untouched-root, or unrelated diagnostic work before evaluating this correction.

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

## Exact next sequence

Run a short paired 200-step update first. This limits catastrophic forgetting from the intentionally small one-root pilot while testing whether the new targets causally improve the served policy.

```sh
PYTHONPATH=neural .venv-neural/bin/python neural/distill_postflop_policy.py \
  --dataset-a neural/runs/20bb-v51-postflop-action/targets-seed16001.jsonl.gz \
  --dataset-b neural/runs/20bb-v51-postflop-action/targets-seed16002.jsonl.gz \
  --initial-weights-a neural/runs/20bb-v26-probability-combined-distilled/seed-0.safetensors \
  --initial-weights-b neural/runs/20bb-v26-probability-combined-distilled/seed-1.safetensors \
  --output-dir neural/runs/20bb-v51-postflop-distilled-s200 \
  --hidden-sizes 512,256 \
  --steps 200 \
  --batch-size 512 \
  --learning-rate 0.00003 \
  --seed 16101
```

Require both students' held-out teacher fit and other-seed held-out teacher fit to improve. If they do, export paired routed bundles while preserving the frozen v27 preflop weights:

```sh
PYTHONPATH=neural .venv-neural/bin/python neural/export_routed_bundle.py \
  --preflop-run neural/runs/20bb-long-v1-narrow-seed5101 \
  --preflop-round 250 \
  --preflop-weights neural/runs/20bb-v27-tabular-distilled-s5000/seed-0.safetensors \
  --postflop-run neural/runs/20bb-long-v1-wide-seed5101 \
  --postflop-round 100 \
  --postflop-weights neural/runs/20bb-v51-postflop-distilled-s200/seed-0.safetensors \
  --output neural/runs/20bb-v51-routed-seed7601-s200.json

PYTHONPATH=neural .venv-neural/bin/python neural/export_routed_bundle.py \
  --preflop-run neural/runs/20bb-long-v1-narrow-seed5102 \
  --preflop-round 250 \
  --preflop-weights neural/runs/20bb-v27-tabular-distilled-s5000/seed-1.safetensors \
  --postflop-run neural/runs/20bb-long-v1-wide-seed5102 \
  --postflop-round 100 \
  --postflop-weights neural/runs/20bb-v51-postflop-distilled-s200/seed-1.safetensors \
  --output neural/runs/20bb-v51-routed-seed7602-s200.json
```

Then rerun the same paired controlled certificate, changing only the candidate bundle and output path:

```sh
target/release/preflop-solver neural-certificate \
  --effective-stack-bb 20 \
  --networks neural/runs/20bb-v51-routed-seed7601-s200.json \
  --deals 8 \
  --seed 14602 \
  --confidence 0.99 \
  --threads 8 \
  --public-branches-per-street 2 \
  --opponent-samples-per-runout 4 \
  --output neural/runs/20bb-v51-postflop-action/causal-certificate-seed7601-s200.json

target/release/preflop-solver neural-certificate \
  --effective-stack-bb 20 \
  --networks neural/runs/20bb-v51-routed-seed7602-s200.json \
  --deals 8 \
  --seed 14602 \
  --confidence 0.99 \
  --threads 8 \
  --public-branches-per-street 2 \
  --opponent-samples-per-runout 4 \
  --output neural/runs/20bb-v51-postflop-action/causal-certificate-seed7602-s200.json
```

Compare on the paired seed against the v50 means in `neural/20bb-v50-full-hand-certificate-pilot.json`. Keep the successor only if both routed candidates improve and normal action metrics remain sane. If it improves but still fails release, expand authentic roots using the same pipeline. If it regresses, retain v50 and first reduce the update steps or learning rate. Do not activate any candidate unless every release gate in `neural/20bb-v50-full-hand-candidate-freeze.json` passes; never infer a pass from teacher fit or cross-seed stability alone.

## Verification already completed

- `cargo test --release`: 98 library tests and 3 CLI tests passed after the implementation change.
- Relevant Python suite: 9 tests passed.
- One-step end-to-end smoke test loaded the exact features and slightly improved held-out teacher MAE for both seeds.
- `git diff --check` passed before the implementation commit.
