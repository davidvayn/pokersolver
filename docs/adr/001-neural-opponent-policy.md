# ADR 001: Frozen Deep DCFR+ baseline with a confidence-capped exploit response

Status: accepted for implementation; no full-hand model is active yet.

## Context

The tabular full-hand DCFR blueprint has exact abstract-game accounting but its
information-set corpus can exceed a local training process's memory ceiling and
requires a large hosted lookup store. A matched three-seed exact-Leduc
prototype compared compact Deep CFR and NFSP agents under the same architecture
and budget. Deep CFR had lower mean exploitability (0.741 vs 0.969), higher
value against the fixed leak (1.246 vs 1.113 after adaptation), lower opponent
best-response value (2.939 vs 3.294), and lower training time (5.39s vs 7.26s).

Those numbers choose an engineering direction only. They do not validate a
heads-up no-limit model or justify an Approximate GTO label.

## Decision

Use one neural architecture:

1. A frozen Deep DCFR+ average-policy network is the robust baseline. Its
   cumulative advantages are bootstrapped from the frozen prior round, clipped,
   discounted, and updated from grouped legal-action samples. The frozen
   average policy remains the only baseline serving output. A serving artifact
   may route one independently validated component preflop and another on flop,
   turn, and river; policy, response, and value heads always route together.
2. A separate one-sided Deep CFR response network is trained against a
   versioned opponent-profile schema.
3. A separately evaluated baseline action-value head supplies feedback and
   uncertainty. User decisions are always graded against this frozen head.
4. At opponent decisions, the browser runs both policy heads and serves
   `(1 - weight) * baseline + weight * response` after masking illegal actions.
5. The response weight is zero below the evidence threshold, zero for unstable
   evidence, and never exceeds the cap embedded in the pinned artifact.
6. NFSP is not part of the serving or training architecture. Its online DQN
   machinery can be reconsidered only for a future true-online-learning goal.

The exact model version and local opponent-profile snapshot are frozen for a
hand. Every opponent inference stores baseline, response, and served action
frequencies with its response weight, evidence count, and confidence.

## Storage and query path

No hosted database is required for this architecture.

- Versioned `.bin` weight artifacts live under `/public/models/practice/` (or
  an immutable HTTPS CDN path) and use one-year immutable caching.
- `data/practice/full-hand-manifests.json` is the checked-in activation
  registry. A manifest is ignored unless every existing two-seed validation
  gate passes.
- The first page load fetches the small manifest. The browser pins a version,
  fetches its artifact once without credentials, verifies the SHA-256 digest,
  decodes it, and performs deterministic CPU inference locally in a dedicated
  browser worker so the table UI remains responsive.
- Hand history and the latest 500 eligible user decisions remain in local
  IndexedDB. The fixed 16-value `local-opponent-profile-v1` feature vector is
  derived on device; raw hands are not uploaded.
- A missing artifact, bad hash, schema mismatch, invalid probability, or
  illegal action pauses the table. There is no uniform or guessed fallback.

A server database becomes useful only for opt-in cross-device history/profile
sync, centralized training-data ingestion, or remote per-user learning. Those
are separate privacy and product decisions. Existing DynamoDB policy-shard
support remains compatible with old tabular manifests but is not a dependency
of the neural path.

Schema-2 neural artifacts retain the same binary envelope and add an explicit
`street-v1` route, component model versions, and a second postflop network
group. The worker selects the group from the exact hand street and records the
route and component version in the opponent trace. Schema-1 artifacts remain
readable. A missing or partial routed group fails validation.

## Artifact contract

The compact binary envelope is:

- bytes 0–3: `PLNP`
- bytes 4–5: little-endian binary schema version (`1`)
- bytes 6–7: reserved
- bytes 8–11: little-endian JSON metadata byte length
- bytes 12–15: little-endian float parameter count
- metadata: UTF-8 JSON with layer offsets, feature schemas, action abstraction,
  adaptation gates, and value calibration
- parameters: contiguous little-endian float32 values

The browser envelope refuses artifacts above 32 million parameters (128MB of
float weights before metadata), preventing an accidentally oversized model
from exhausting a mobile tab. Expected artifacts are much smaller.

Policy networks score one legal state/action pair at a time. This avoids a
fixed global action vocabulary while keeping the action-sizing grid in the
pinned artifact. The state encoder retains exact private cards, exact board
cards, canonical suit identity, money/position state, up to 32 public actions,
and suit-invariant made-hand, draw, board-multiplicity, straight-window, and
wetness features. The latter are derived only from cards visible to the acting
player. Exceeding the pinned schema fails closed.

Use `npm run policy:export-neural -- --input <model.json> --output
public/models/practice/<version>/<depth>bb.bin --url
/models/practice/<version>/<depth>bb.bin` to produce the binary and its digest.
The command refuses overwrite through exclusive creation.

Use `npm run policy:compose-neural -- --preflop-input <model.json>
--postflop-input <model.json> --model-version <version> --output <path> --url
<immutable-url>` to compose compatible components. The exporter validates both
inputs before shifting postflop parameter offsets and refuses depth, schema,
abstraction, adaptation, or calibration mismatches.

Fresh v14 runs retain the two advantage networks only at browser artifact
rounds. Each offline `teacher-snapshot.json` pins hashes, completed traversals,
DCFR strategy weight, and the regret-matching transform. These bounded
snapshots enable a later SD-CFR-style direct-average comparison without placing
the ensemble in the browser or claiming that it is already superior.

## Promotion gates

The registry stays empty until two independent 8–12 hour full-game seeds for a
depth pass all existing activation checks: exploitability estimate and 99%
upper bound, cross-seed frequency MAE, primary-action agreement, aggregate
action delta, authentic and forced-deviation coverage, action-EV uncertainty,
probability integrity, and projected storage. Cross-seed agreement is only a
reproducibility gate, never equilibrium proof.

Independent trajectory evaluation can estimate every legal action with two or
more deterministically seeded external-sampling rollouts. It records the sample
mean and standard error, trains the uncertainty output instead of a fixed
placeholder, and passes the EV-confidence gate only when every action is at or
below 0.02bb for at least 95% of reached decisions. Approximate responses are
useful lower-bound evidence, not the required full-game exploitability upper
bound, so that release gate remains fail-closed.

Until then, full-hand depths remain hidden and the only active manifest is the
previously validated push/fold corpus.
