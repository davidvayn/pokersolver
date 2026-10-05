# Scores-blind postflop TRAIN coverage pilot

Date: October 4, 2026. Status: **Native data qualified; baseline diagnosis and
matched students pending. No model accepted or activated.**

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

## Next pinned comparison

Diagnose the previous C0/C1 networks on the three new TRAIN families, including
search-band value errors. If targeted supervision is justified, run two matched
600-step student seeds per arm, with all six separately bounded shards. Preserve
architecture, primary replay, optimizer, split and fixed checkpoints. Both arms
receive the same calibration; only C1 receives the action-contrast auxiliary.
TRAIN-only gradient conditioning chooses its weight before fitting. Equal-arm
resource cadence must also balance six families: cadence 4 gives 25 updates
each; cadence 8 is disallowed because it would be uneven.

After independent full-615-state NumPy/native parity, compare actual candidate
policies on the unchanged four cheap all49/native64 cases. A known-case
regression above 0.01bb rejects; a useful 0.02bb mean benefit with consistent
paired seeds is required for expansion. Better RMSE or frozen TRAIN action
rankings cannot promote a model. Older-pilot comparisons are diagnostic because
family scheduling and the conditioned auxiliary weight change.

## Verification and immutable identities

60 targeted Python tests passed across contrast backup/loss/training/probes,
new registry/join checks, full-chance qualification, native datasets, policy
screens and resource guards. Expected NumPy correlation warnings originate
from constant-vector fixtures. Rust and browser code were unchanged.

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
