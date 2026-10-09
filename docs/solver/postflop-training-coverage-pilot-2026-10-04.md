# Scores-blind postflop TRAIN coverage pilot

Date: October 4, 2026. Status: **Complete; both arms rejected at the first
actual-policy control. No model accepted or activated.**

## Why this intervention

The previous full49 pair lowered frozen training ranking loss but failed the
actual-policy controls. C1's advantage over C0 was only 0.000078bb, with
opposing seed means. Its high-rainbow continuation errors persisted under
retained-policy beliefs, so pure own-policy range drift was insufficient.
See [the completed pilot](postflop-full-chance-pilot-2026-10-04.md).

This tests broader board/search-belief coverage, not unchanged iteration
scaling. Hypotheses remain distinct: missing board coverage, insufficient
current-search beliefs, or an inadequate feature/value contract. A successful
training probe alone cannot distinguish them or establish stronger play.

## Frozen input and label protocol

Generate 128 authentic roots with seed 2026100417 from the frozen preflop800
policy. Exclude all 34 suit-canonical families present in the frozen 615-state
corpus or twelve-root September benchmark. Select the first two unpaired
Q-or-higher rainbow and first two-tone five-leaf three-bet families in bank
order, without examining predictions or native action errors.

| TRAIN ID | Bank index | Board | Family |
| --- | --- | --- | --- |
| 100 | 22 | 3h Ad 4s | 4, 9, 50 |
| 101 | 76 | 9h Jh As | 28, 36, 49 |
| 102 | 115 | 9d 3h As | 4, 29, 50 |

The rejected C1 pair proposes search inputs only. Each root receives 16
native64 labels from early/middle/late/final-average beliefs at 128 flop
updates, plus native64 labels for all five live leaves on all 49 public turns.
Search observation/no-observation policy parity passed on the first root.
Each separate calibration shard contains 261 states, retaining the existing
640-state/256MiB decoded/128MiB compressed bounds. The original 615-state
corpus and 474/69/72 split remain unchanged. Sixteen extra zero columns in
each affine backup ensure search calibration never changes frozen action EVs.

## Native data results

All three families completed in 568.62 seconds, with 544.86 seconds labeling
and 23.77 seconds finalization. Two packet workers were used; no system
memory-pressure, disk, worker-memory or time stop occurred. The first two turns
per family projected 679.66 seconds of remaining labels with a safety margin.

Measured maximum native64/256 same-turn ranking loss: **0.000952bb**.
Replacing that one turn inside each all49 target changed the selected best
action's aggregate target loss by **0bb**. Profile-consistent support covers
effectively all measured authentic reach. The combined six-family data screen
also passes, retaining the old maximum sentinel loss **0.016998bb**.

This is permission for a bounded fitting pilot, **not** certification that
every continuation is converged, an action-EV precision gate, safe re-solving,
or full-game exploitability. Finite native targets remain approximate.

## Baseline diagnosis and pinned comparison

The previous C0/C1 probe completed in 320.94 seconds, peaking at 1.454GiB
with no resource stop. Across six groups and both seeds, mean frozen native
loss from the predicted-best action is **0.259933bb (C0)** and **0.297321bb
(C1)**. On two-tone TRAIN family 101, C1's BB and BTN errors span
0.5503–0.6603bb. Mean search-band RMSE is 1.0937–1.2874bb for C0 and
1.1548–1.3066bb for C1; these are continuation-value diagnostics, not policy
response scores. This supports testing targeted supervision before scaling
iterations. The read-only worker and receipt completed; its outer one-line
display command subsequently used a nonexistent telemetry key, producing a
display-only KeyError. No training or probe artifact failed or was rerun.

The matched comparison therefore runs two matched
600-step student seeds per arm, with all six separately bounded shards. Preserve
architecture, primary replay, optimizer, split and fixed checkpoints. Both arms
receive the same calibration; only C1 receives the action-contrast auxiliary.
TRAIN-only gradient conditioning chose its weight **1.883306** before fitting
(requested contrast/value gradient fraction 0.25). Conditioning took 36.18s
and projected 3,632.47s including fit and parity with margin. Equal-arm
resource cadence must also balance six families: cadence 4 gives 25 updates
each; cadence 8 is disallowed because it would be uneven.

After independent full-615-state NumPy/native parity, compare actual candidate
policies on the unchanged four cheap all49/native64 cases. A known-case
regression above 0.01bb rejects; a useful 0.02bb mean benefit with consistent
paired seeds is required for expansion. Better RMSE or frozen TRAIN action
rankings cannot promote a model. Older-pilot comparisons are diagnostic because
family scheduling and the conditioned auxiliary weight change.

## Completed students and policy decision

Conditioning, both 600-step student pairs and exhaustive export parity completed
in **596.58 seconds**. All four models used exactly 25 updates per family.
Maximum independent NumPy/native error across the original 615 states, both
players and every private holding was **0.000006146bb**. Fit peak was **3.966GiB**
under the 6GiB guard; no resource limit fired. No winner/checkpoint was chosen
from policy results.

The final read-only TRAIN probe took 54.04 seconds and peaked at 1.448GiB.
The numbers below are equal-group/seed means of locally reach-weighted ranking
loss, not response gains or full-game exploitability.

| Frozen TRAIN block | Previous C0 | New C0 | Previous C1 | New C1 |
| --- | --- | --- | --- | --- |
| Original three families | 0.089358bb | 0.094702bb | 0.051303bb | 0.074326bb |
| New three families | 0.259933bb | 0.183537bb | 0.297321bb | 0.126229bb |

C1's new-family ranking loss fell **57.5%**, but original-family loss worsened.
Reduced old-family scheduling and a changed conditioned auxiliary weight
confound a pure coverage interpretation. This is evidence of incomplete
generalization/retention, not proof that three additional families solve it.

Control-first actual-policy results (three-bet high-rainbow, seed 100101):

| Model | Conditional native response gain | Regression versus retained |
| --- | --- | --- |
| Retained | 0.278335bb | — |
| New C0 | 0.373479bb | +0.095143bb |
| New C1 | 0.350730bb | +0.072395bb |

New C1 improves this case versus the prior rejected C1's **0.418741bb** by
0.068011bb (**16.2%**). It is still worse than retained by much more than the
predeclared 0.01bb tolerance. Both arms therefore reject, before broad expansion.
The independent cached JS audit reproduced both complete case scores in under
a second. The first control is sufficient to reject; it is never sufficient
to accept.

To avoid unnecessary evaluations, both controllers were intentionally stopped
after the first audited failure. Their raw manifests report operator-stopped
`failed` stages, not completed four-case screens. Each completed first case is
preserved; partial second-seed packets are **not scored**. A separate hashed
control decision records rejection, missing cases and absence of paired means.
Full-game exploitability, fresh-family strength and release gates remain
unmeasured here. No model route or website serving weights changed.

A narrowly scoped optional `--stop-on-known-regression` now performs this
budget rule inside the policy controller after native aggregation and JS audit.
It writes an explicit `rejected` receipt, leaves paired means absent, and never
promotes partial success. It applies only to the declared control-first cases;
existing broader pilots keep their prior behavior unless opted in.
Use it with `--spots three-bet-high-rainbow,three-bet-monotone`; undeclared
broad-case use is rejected before any computation.

## Remaining blocker and next bounded question

The new C1's cached own-policy BTN-after-check diagnosis still has only **45.3%**
native best-action agreement. Native loss from its predicted-best action is
**0.29565bb**; native local policy-deviation loss is **0.55600bb** versus a
predicted **0.12161bb**, and contrast RMSE is **0.74174bb**. Exact all-in-only
branches remain internal controls. Thus a substantial continuation-ranking
error survives the training gains. These are local frozen-policy diagnostics,
not an exploitability estimate or EV confidence interval.

Do not scale the same rejected weights, start paid compute, or integrate them.
The next useful bounded question is whether old-data retention and stronger
board/range representation can fix that mismatch. First compare retained and
new representations/targets on hash-matched inputs and test a retained-weight
fine-tune with protected old-data replay against the same from-scratch control.
Keep teacher/game/action budgets fixed and require actual-policy control gains.
This is a proposed next intervention, not a run started or a proven solution.

New stage artifacts occupy about 0.56GiB excluding the shared cache; counting
the **entire** shared feature cache conservatively brings the total below 4GiB.
Free disk remains approximately 117GiB. All this pilot's processes are stopped.

## Verification and immutable identities

63 targeted Python tests passed across contrast backup/loss/training/probes,
new registry/join checks, full-chance qualification, native datasets, policy
screens, early-rejection semantics and resource guards. Two Node action-value
diagnostic tests also passed. Expected NumPy correlation warnings originate
from constant-vector fixtures. Rust and browser code were unchanged.
Commands from `preflop-solver/neural/`:

```sh
../.venv-neural/bin/python -m unittest \
  test_action_contrast_dataset test_action_contrast_loss \
  test_action_contrast_students test_probe_action_contrast_students \
  test_complete_action_bundle_chance test_join_full_chance_bundles \
  test_join_coverage_bundles test_training_coverage \
  test_decision_pilot_screen test_student_value_pilot \
  test_native_value_dataset test_worker_resources
node --test native_action_diagnostics.test.mjs
```

Also reran the cached independent JS response audit on both completed controls;
it reproduced their saved native scores and rejected them at the unchanged
limit. No `npm test`, build or Rust rebuild was required for these Python-only
research/controller changes; this is not website/release acceptance.

Paths below are relative to `preflop-solver/neural/`. Large local research
artifacts remain ignored; code and this report are committed.

| Artifact | SHA-256 |
| --- | --- |
| `runs/local-contrast-coverage-roots-20261004-a/roots.json` | `299b3eb8a0d2a3e8746d59ec3739e34825da5ca9c542615eca19d98c1537a6f8` |
| `runs/local-contrast-coverage-protocol-20261004-a/protocol.json` | `9c330d8747c1194187cbf2dd91938f4dfe55b1ca0a7a9c37f6423496fd81b896` |
| `runs/local-contrast-coverage-labels-20261004-a/manifest.json` | `87a422be5b14c628f806694c36b4876bcc1f3ef284bb22f9a1cd544d7928d180` |
| `runs/local-contrast-coverage-labels-20261004-a/decision.json` | `b72a8e5b52abf97196c5c55dcb7a08c7f97b2539404ebaa019f86426e8851db9` |
| `runs/local-contrast-coverage-bundles-20261004-a/manifest.json` | `bcd1c4a92469b8105139a21b832dabd7b56de142d9e2622a82d3b34999998450` |
| `runs/local-contrast-coverage-bundles-20261004-a/decision.json` | `3f6415ccc510cb9947efcafbd3e87e9941fb3bc88e5abbb8fbb53229af54a4c5` |
| `runs/local-contrast-coverage-baseline-probes-20261004-a/analysis/manifest.json` | `ac5e55017373b4e0f7089bb4a842fd2a8e1fcd468eb1db9334976943c70cd13a` |
| `runs/local-contrast-coverage-students-20261004-a/manifest.json` | `e4d85fab3194ff1412338f583efa6b6da560b26d66eb2d8e31cc7cbd781f2955` |
| `runs/local-contrast-coverage-students-20261004-a/C0/manifest.json` | `637af048081d61be4615d4bfdfec1088cf8b3f06a11c88ead64a33ee034d2fb4` |
| `runs/local-contrast-coverage-students-20261004-a/C1/manifest.json` | `00951cd6061cd244e63b734dfa9ef93e080cd39df422d046e7f4b132ad29ae11` |
| `runs/local-contrast-coverage-final-probes-20261004-a/analysis/manifest.json` | `861484ece6a2c395c90fcc0bdb68447fbfaea83326a1705c9a967fc5b85bc6e4` |
| `runs/local-contrast-coverage-response-20261004-C0-a/manifest.json` (intentionally stopped) | `fd22410936c513f1bd3ae4466b38a3ed72a318469c94fc8ebbdf5c8265cb89e8` |
| `runs/local-contrast-coverage-response-20261004-C1-a/manifest.json` (intentionally stopped) | `2feabf447b96f54f2fa5077be8fe34e7da401da0cf0b7dd89ee75c878f7f6f8f` |
| `runs/local-contrast-coverage-control-decision-20261004-a.json` | `b9ed6376fe22c77cb3969dbbe8c9bce9d4bfabe0b48ca578c10716406a529218` |
| `runs/local-contrast-coverage-own-policy-diagnosis-20261004-a/manifest.json` | `ed9fbee57f075ddb0a4a124c96e26fb14d09486ac60003fcffa192db9873b227` |
