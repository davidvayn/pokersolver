# September 5 local policy-improvement sequence

User request: work overnight on remaining weaknesses; use short pilots and
research before long training; commit/push major milestones. First target is
all existing quality metrics plus full-game exploitability below 0.50bb/hand,
then continue toward 0.05bb/hand. Do not turn a restricted-response lower bound,
rejected-response zero, or fixed-opponent payoff gain into an upper certificate.
The expected user return is approximately 18:30 UTC. The local 16GiB machine
remains the only authorized compute; no deployment or paid resources.

## Preserved milestone

`e25ec44` was committed and pushed to `origin/main`. It contains the bounded
checkpoint/response tooling, paired-estimator corrections, connected turn/river
pilots, and the confirmed terminal-flop payoff improvement. See
[the preceding results](tabular-turn-pilot.md#continuing-policy-action-work-flop-corrections).
Unrelated `public/reports/`, `scripts/solver-review.test.mjs`, and the loose
`blueprint-artifact.json` were excluded and left untouched.

## Pilot 1: suit-consistent buckets

The card-feature estimator seeded its sampling from literal suit labels.
`cargo test --release rollout_buckets_do_not_split_suit_isomorphic_observations`
reproduced a split between equity buckets 3 and 2 after a pure suit relabeling.
The minimized observation has hero cards `[31,14]` and turn board
`[11,40,45,44]`. No opponent cards or unrevealed board are needed to reproduce
the classification issue. Suit symmetry is strategically irrelevant; see
[Waugh's primary hand-isomorphism paper](https://www.cs.cmu.edu/~waugh/publications/isomorphism13.pdf).
We use the existing visible-card suit-signature map, not a new implementation
of the paper's complete indexing algorithm.

Implemented opt-in `--canonical-suit-buckets`: canonicalize the known cards
and sampling deck, retaining exact card removal, betting abstraction, and
trajectory recall. Its identity is serialized and fingerprinted; a legacy
checkpoint cannot be resumed with the changed abstraction. The default stays
legacy-compatible. All 24 suit permutations are tested across all streets.

A second failing regression proved the per-deal bucket cache ignored
abstraction settings. Its key now includes those settings. The legacy-mode
eight-round seeded artifact is byte-identical across the old/new executable:
SHA-256 `93c85c4c86ca1b4a5b9203fba1d0bb3099888b785cacb00407ba2c9e900cead6`.
This verifies a small replay, not an unperformed full-size replay.

Both 400-round pairs completed: seeds 27001/27002, fixed DCFR, zero averaging
delay, public-chance sampling, terminal-action integration, full default action
grid. Held-out / root-deviation-per-class / action-value budgets: 4,000 / 64 /
1,000. Workers ran sequentially, with a 6GiB sampled footprint stop, 20-minute
time stop, and 20GiB disk reserve. No stop fired.

| Seed | Mode | Nodes | Held-out unknown | Held-out untrained | Root local gain (SE), bb |
| --- | --- | ---: | ---: | ---: | --- |
| 27001 | Legacy control | 9,064,421 | 11.410% | 1.508% | 0.74516 (0.05233) |
| 27001 | Suit canonical | 8,319,321 | 12.705% | 1.578% | 0.76742 (0.05213) |
| 27002 | Legacy control | 9,847,604 | 11.758% | 1.830% | 0.82207 (0.05378) |
| 27002 | Suit canonical | 8,533,620 | 12.861% | 1.806% | 0.74719 (0.05130) |

These held-out trajectories follow each policy; they are not a matched fixed
trajectory coverage experiment. Root local gains are not full-game
exploitability. Node counts decrease 8.2% / 13.3%, but the policy-quality
screen is mixed. **No longer run was triggered and no canonical policy was
selected as a quality winner.** Keep this as a tested optional abstraction,
not a reason to discard the preceding 800-round candidate.

Local records:
`preflop-solver/neural/runs/local-suit-20260905-control400/run-manifest.json`
and `local-suit-20260905-canonical400/run-manifest.json`.
Control runtime: 244.161 / 276.312 seconds; sampled footprint:
2,968,979,952 / 3,216,263,808 bytes. Canonical: 270.272 / 286.356 seconds;
2,725,939,480 / 2,793,441,608 bytes. Concurrent builds preclude isolated
runtime-speedup claims. Canonical binary:
`11a627aab735fcf2267cce725308c1747244fa181d938a727e00dbc6c65a6b2f`.

## Pilot 2: compare response actions to the actual baseline

The old learner required a confidently unique best action. That is a different
question from whether an action beats the profile being attacked. A regression
reproduced rejection when two actions tie for best but the baseline mixes in
losing actions. Conversely, a coarsened response can have a clearly best average
action while losing to a baseline that distinguishes the underlying situations.

The learner now records paired `Q(action) - sum_a policy(a) Q(a)` observations,
using the actual profile at each sampled state. Welford moments preserve the
common-random covariance. New response admission uses the selected action's
positive one-sided normal-approximation 99.5% advantage lower bound. The old
runner-up margins and `low_confidence` rank flags remain visible; legacy
reports without the new `response_advantage` data retain their old meaning.
The new method string distinguishes the estimator/admission version.

This changes neither the independent positive 99.5% calibration requirement
nor the independent holdout, and it does not edit the defender's policy.
Exploitability certification is still unresolved. A stronger attack may expose
larger leaks; that is useful diagnostic evidence, not a model regression.
The [primary LBR paper](https://arxiv.org/html/1612.07547v1) likewise treats
approximate responses as lower-bound evidence, not equilibrium certification.

The new regressions and full suite pass: 217 Rust library tests, 6 CLI tests,
and 32 Python runner/resource tests. The preserved milestone's remote CI also
passed (GitHub Actions run 33955548073).

`preflop-solver/neural/runs/local-advantage-20260905-pair1/cohort.json` is
**active**, using the retained 800-round seeds 26001/26002, joint turn/river
four iterations, and the 25% terminal-flop correction with 2,048 equity samples.
Response budgets per seat: 512 training / 2,000 calibration / 2,000 independent
holdout hands, four action rollouts, minimum four particles, offset 2000000.
Two shared-table workers; sequential seeds; 45-minute / 7.5GiB sampled-footprint
stop per seed and 20GiB disk reserve. The runner froze the executable and pins
source/output hashes. Do not launch a second full-size 800-round table
alongside it. Next policy changes must target the leaks actually found.
