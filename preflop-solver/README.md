# Native preflop solver

This standalone Rust crate is an offline validation and artifact-generation
tool. It deliberately does not extend the browser WASM solver: the existing
WASM game is a postflop, single-street all-in-equity model and has different
game semantics.

## Milestones implemented

1. Deterministic full-tree CFR+ for three-card Kuhn poker. Its tests check the
   known first-player game value of `-1/18` and compute exact exploitability by
   enumerating all pure best responses.
2. Exact two-round limit Leduc CFR+ with physical card removal and an exact
   information-set-consistent best-response evaluator. It provides the richer
   imperfect-information convergence benchmark used by the Rust tests.
3. Chance-sampled CFR+ for heads-up equal-stack push/fold Hold'em. Strategies
   are learned for all 1,326 exact two-card combinations. Chance sampling deals
   compatible hands, so card removal is retained.
4. A versioned JSON exporter with exact-combo and 169-class strategy views,
   configuration, payoff convention, best-response metrics, and validation
   status.
5. An experimental heads-up multi-street blueprint. It uses external-sampling
   Discounted CFR with alternating traversers, deterministic seeded chance,
   trajectory recall, exact sampled deals/runouts, a
   configurable no-limit action grid, and coarse rollout-derived postflop
   buckets. The implementation covers limp, open, call, 3-bet, 4-bet, deeper
   raises and all-in preflop, then check, bet, fold, call, limited raises and
   all-in through the river.

Card integers and combo keys match `lib/cards.ts`:

```text
card = rank * 4 + suit
rank: 0 = 2 ... 12 = A
suit: 0 = clubs, 1 = diamonds, 2 = hearts, 3 = spades
combo_key = high * (high - 1) / 2 + low
```

## Commands

```bash
cargo fmt --manifest-path preflop-solver/Cargo.toml
cargo test --release --manifest-path preflop-solver/Cargo.toml

cargo run --release --manifest-path preflop-solver/Cargo.toml -- \
  kuhn --iterations 20000

cargo run --release --manifest-path preflop-solver/Cargo.toml -- \
  solve \
  --effective-stack-bb 10 \
  --iterations 25000000 \
  --equity-samples 1024 \
  --seed 1 \
  --output /tmp/hu-push-fold-10bb.json

cargo run --release --manifest-path preflop-solver/Cargo.toml -- \
  blueprint \
  --effective-stack-bb 20 \
  --iterations 100000 \
  --max-information-sets 12000000 \
  --averaging-delay 20000 \
  --held-out-deals 30000 \
  --root-deviation-samples 512 \
  --action-value-deals 30000 \
  --export-postflop-strategies \
  --checkpoint /tmp/hu-blueprint.checkpoint.json \
  --checkpoint-every 100000 \
  --output /tmp/hu-blueprint.json

python3 -m venv preflop-solver/.venv-neural
preflop-solver/.venv-neural/bin/python -m pip install \
  -r preflop-solver/neural/requirements.txt
preflop-solver/.venv-neural/bin/python preflop-solver/neural/train.py \
  --run-dir preflop-solver/neural/runs/20bb-seed-1 \
  --depth-bb 20 \
  --seed 1 \
  --rounds 1000000 \
  --artifact-every 500 \
  --max-minutes 600

npm run preflop:compare-blueprints -- \
  /tmp/hu-blueprint-seed-1.json \
  /tmp/hu-blueprint-seed-2.json \
  --require-pass
```

### Neural full-hand training

The CLI defaults pin the current leading 20bb research profile: 400 traversals
per round, a 100,000-record reservoir, 256/128 hidden layers, batch size 1,024,
100 optimizer steps per round, AdamW learning rate 0.001, control-variate scale
0.5, authentic replay, four action-value samples, and exact turn/river runouts.
Override these only for an explicitly named experiment. This profile remains
research-only and is not an activated or release-validated policy.

The neural path keeps game semantics in this Rust crate and uses the free,
MIT-licensed MLX framework only for batched optimization. Each iteration block
freezes the two current advantage networks, alternates traversers, samples exact
cards, enumerates the traverser's actions, and writes complete legal-action
decisions with instantaneous advantage, average-strategy, and action-value
targets. The Python orchestrator expands the compact records into the pinned
suit-canonical browser feature schema and verifies a cross-language feature
digest for every action before accepting it.

The v4 state schema has 716 values. It keeps the exact-card encoding and adds
64 cheap, suit-invariant poker features for made-hand category, pair and
overpair relationships, board rank/suit multiplicity, straight windows,
flush/backdoor and straight draws, and board wetness. The new first-layer
columns are initialized to zero so a resumed comparison starts from the old
representation instead of an unrelated random projection.

Advantage learning follows the bootstrapped cumulative-advantage recurrence
from Deep DCFR+: each round clips the frozen prior network's negative outputs,
discounts the positive outputs, adds the current sampled advantages, and fits
only the current round's complete decisions. The average-policy reservoir
remains fixed-capacity and uses grouped masked softmax cross-entropy. Replay
arrays use bounded `float16` memory maps rather than retaining a growing game
tree. Network weights and AdamW optimizer state are checkpointed after every
round, along with deterministic Python/NumPy reservoir state. Re-running the
same command resumes the existing run; changing an immutable setting is
rejected. See the [Deep DCFR+ paper](https://arxiv.org/abs/2511.08174) for the
algorithmic basis; no third-party research implementation is vendored.

The scalar action-value target is trained in effective-stack fractions and
converted back to big blinds at the Rust traversal and browser-artifact
boundaries. At sampled opponent nodes it can be used as an action-dependent
control variate: `sum(policy * baseline) + sampled_value - sampled_baseline`.
The actor-perspective prediction is sign-corrected for the traverser first.
This estimator preserves the policy-weighted expectation; the configured
baseline scale controls variance only and does not change the target game.

Value-target calibration can average multiple independently seeded external
samples for each traverser action. The primary traversal sample is reused and
extra samples use a canonical state/action seed, so they cannot advance the CFR
RNG or change regret and average-policy records. This setting affects only the
separate action-value head and is pinned in the dataset, run, and artifact
metadata.

The original four-sample pilot predated the trainable uncertainty head and was
not a clean comparison for the final trainer. A matched v14 one-versus-four
rollout pair found that four rollouts improved four of six authentic/forced
stability measures, reduced the worst authentic aggregate delta from 6.08% to
3.72%, and supplied a genuine standard-error target at 2.43 times the measured
training cost. Four samples per action are therefore pinned for the 20bb long
run. A matched zero-control-variate pair regressed, supporting the current 0.5
baseline scale.

The frozen long-pair plan is `neural/long-run-20bb-v1.json`. It assigns two
fresh seeds four hours of narrow preflop training and four hours of wide
postflop training each, for eight training hours per composite seed and 16
seed-hours total. Its launcher accounts for completed atomic-round time across
resumes, runs each stage's two seeds concurrently, checkpoints on interruption,
and emits progress every ten minutes:

```bash
preflop-solver/.venv-neural/bin/python \
  preflop-solver/neural/long_run.py --preflight-only
preflop-solver/.venv-neural/bin/python \
  preflop-solver/neural/long_run.py
```

After both stages finish, the launcher prints the fully pinned validation
command for the four routed run directories. The plan reserves 125,000 exact
deals per candidate certificate; after splitting the 1% family error budget
across two observed seeds, its Hoeffding chance margin remains below the
0.10bb release threshold.

The completed narrow stage retained round 250 as its serving candidate after a
5,000-trajectory fixed-seed comparison against the final round 310. Round 250
won authentic MAE, both agreement checks, and both aggregate-delta checks;
round 310 only improved forced-deviation MAE by 0.05 percentage point. Routed
validation therefore pins narrow round 250 while selecting the latest immutable
wide artifact independently.

The full-game release validator includes a separate conservative certificate.
For each exact complete deal, a relaxed responder observes both private hands
and the full runout and solves the betting tree against the frozen network.
This response class contains every legal imperfect-information response, and a
one-sided Hoeffding bound covers independent chance sampling. The calculation
is deterministic across thread counts and parallelizes exact deals; because the
information relaxation is deliberately strong, it may reject a good strategy
and can never be replaced by a neural approximate best response.

The research CLI can opt into replay-street stratification without changing
Rust trajectory generation. It weights each example by its empirical reservoir
street probability divided by its realized batch probability, preserving the
authentic replay objective. The equal-four-street 20bb pilot did not improve
cross-seed stability, so authentic uniform replay remains the default. Any
proposal and correction method are pinned in run and artifact metadata.

Configured checkpoints and the final round export a framework-neutral `PLNP`
browser binary, but these files remain `training_not_activated`. The independent
validator samples pure exact-card trajectories (both players sampled once per
decision), preserves repeated visits, mixes equal hand-corpus mass from both
frozen seeds, and separately samples uniform-action forced-deviation hands.
Cross-seed stability and coverage still do not estimate exploitability or
action-EV uncertainty, so the validator fails those release gates closed. The
validator separately reports a research-only continuation bar (6% MAE, 80%
primary-action agreement, and 4% maximum aggregate delta), which can never
activate a model. A 512/256-width pilot improved the round-10 comparison but
regressed by round 25, so the 256/128 architecture remains the short-run
default. Lowering that wider model's constant learning rate from 0.001 to
0.0003 was also worse at the round-10 early-stop checkpoint. Experimental model
versions include a training-config fingerprint to
prevent differently configured artifacts from sharing a cache key. The
exploit-response head is initially a no-op copy of the baseline policy; it
cannot change play until a separately validated opponent-response phase
supplies genuine profile-conditioned weights.

The short 20bb paired-pilot evidence and rejected alternatives are recorded in
`docs/validation/20bb-neural-short-pairs.md`.

See `neural/OPEN_SOURCE_SOFTWARE.md` for the dependency/license inventory.

Stacks include posted blinds. Small-blind payoff is measured relative to the
start of the hand:

- small blind folds: `-small_blind_bb`
- big blind folds to the shove: `+big_blind_bb`
- shove is called: `(2 * equity - 1) * effective_stack_bb`

The game is zero-sum, has no ante or rake, and both players have the same
effective stack.

### Blueprint model

The button posts `0.5bb`, acts first preflop, and acts last after the flop.
Every amount in a betting action is a street-local `raise_to`. Total
contributions are tracked separately; fold and showdown utilities are exact net
stack changes. The board and both private hands are sampled without replacement,
but an information set sees only its own cards, visible board, public action
history, and stack/pot state.

The default preflop grid keeps five opens (`2/2.5/3/4/5bb`), multiple limp
raises, 3-bets, 4-bets, deeper pot-relative raises, and all-in. The default
postflop continuation grid is smaller so decisions recur often enough to learn:
flop `0.333/0.75/1.25 pot`, turn/river `0.5/1 pot`, one pot-sized raise, and
all-in. All grids are serialized in `config` and the main grids can be replaced
from the CLI.

Postflop buckets include made-hand/draw/board texture plus deterministic
rollout-derived equity, improvement, and future-category features. Rollouts use
only the acting player's private cards and currently visible board, never the
sampled opponent hand or unrevealed board. The default `trajectory` recall mode
retains every prior private/public bucket and the complete public action
history, providing perfect recall inside the abstract game.
`--current-street-recall` is an explicitly non-publishable, lower-memory
experiment because it drops prior street buckets.

DCFR uses alpha=1.5, beta=0, and gamma=2 by default. These parameters are
serialized and can be varied with the `--dcfr-alpha`, `--dcfr-beta`, and
`--dcfr-gamma` research flags. One seeded deal updates one traverser, and the
traverser alternates each iteration.

When an all-in reaches showdown before the river, the trainer integrates over
unseen runouts instead of using the single board sampled for that traversal.
Preflop and flop use deterministic configurable Monte Carlo; turn enumerates
every legal river by default; river evaluates the dealt board directly. This is
chance-variance reduction with both exact hole-card hands known at the terminal,
not a learned continuation heuristic.

By default, the artifact exports trained preflop information sets and aggregate
postflop coverage. `--export-postflop-strategies` includes the much larger
trained postflop profile. A checkpoint always retains the complete compact
profile and can be resumed. Checkpoints are streamed to an atomic temporary
file without cloning the in-memory node table. `--max-information-sets`
provides a clean memory guard; an early stop is marked
`validation.status=incomplete_advisory`.

The public artifact contains only the frozen average policy, not regrets or the
current training policy. Regrets remain solely in the resumable checkpoint. A
separate seeded action-value pass forces every action at each reached served
information set, follows the frozen average policy thereafter, and exports
mean EV, standard error, best-action EV loss, and a low-confidence flag.

Evaluation-only inputs live under `config.evaluation_controls`, separate from
training inputs. The artifact exposes both a full `config_hash` and a
`training_config_hash` that ignores held-out/local-deviation sample counts and
seeds. This allows two artifacts trained identically to be compared even when
their validation budgets differ.

The root local-deviation diagnostic evaluates every abstract button action for
all 169 starting-hand classes. It samples exact hero combinations uniformly
within each class, then samples compatible opponent cards and boards. Every
root action sees the same conditioned chance deals; play after that forced
action follows the frozen average profile. The artifact reports each action's
mean net bb, standard error, continuation coverage/fallback fractions, the
profile-weighted root EV, the best sampled action EV, and their gap. Class
results are weighted by 6 pair, 4 suited, and 12 offsuit combinations for the
aggregate. Common conditioned deals also produce a paired standard error for
the selected-action gap and a nonnegative one-sided 99% lower bound. Aggregate
uncertainty combines combo-weighted class variances. Because the best action is
selected on the same samples used to estimate its value, winner's-curse bias
remains and even the lower bound is advisory. This is a one-step local best
response only; it is not exploitability.

## Artifact contract

The root object contains:

- `schema_version`, `solver_version`, deterministic `artifact_id` /
  `config_hash`, and the explicit `heads-up-push-fold-monte-carlo-v1` model
  label. The hash covers solver/model versions and every numerical input but
  excludes the generation timestamp.
- `config`, including blinds, stack, CFR iterations, equity samples, and seed.
- `metrics`, including profile EV, both players' pure best-response values,
  NashConv, exploitability, independent evaluation seed, compatible-deal
  count, equity-cache sizes, and separate equity-sampling uncertainty
  advisories.
- `validation`, with versioned machine-readable checks. Runs above `0.01 bb`
  exploitability are rejected; `0.002 bb` is the high-precision target. Passing
  runs remain `approximate` because the equity oracle is sampled.
- `strategies.exact_combos`, with canonical card identity plus `fold/shove` and
  `fold/call` frequencies.
- `strategies.hand_classes`, the combo-weighted 169-class presentation view.

## Push/fold limitations

- This is a push/fold game, not a full first-betting-round or multi-street
  no-limit Hold'em solver. It cannot justify ordinary open, call, 3-bet, or
  postflop continuation frequencies.
- Equity is deterministic Monte Carlo, cached by unordered, suit-isomorphic
  exact matchups. Reversed seats return the exact complement. `equity_samples`
  controls showdown estimation error; CFR iterations control regret error.
  Both must be increased for publishable artifacts.
- Best responses are recomputed across all 1,624,350 ordered compatible deals
  with an equity seed independent from training. NashConv certifies that
  estimated evaluation game only. Exported equity intervals are conservative
  per-matchup standard-error advisories, not simultaneous confidence bounds.
- Chance-sampled CFR can leave rarely sampled exact combinations noisy at low
  iteration counts. Use the exported exploitability and cross-seed comparisons,
  not visual plausibility alone, as the acceptance gate.
- At the default 25 million iterations and 1,024 boards per canonical matchup,
  two 10bb seeds measured about `0.0069 bb` exploitability. Their 169-class
  mean absolute differences were about 1.27 percentage points for shoves and
  0.42 points for calls; individual boundary mixes differed substantially more.
  Consumers should preserve mixed frequencies and avoid thresholding them into
  categorical charts.
- The artifact contains a generation timestamp. Runs with otherwise identical
  inputs have identical numerical output but not byte-identical JSON.
- Multiplayer solving, antes, unequal stacks, rake, limping, non-all-in raise
  sizes, and future-street value functions are intentionally out of scope for
  this milestone.

## Full-hand release protocol

Train two independent 8–12 hour seeds for one depth at a time, starting at
20bb, then 50bb and 100bb. Never publish an unfinished checkpoint or an
`advisory_only` artifact. Use `npm run policy:validate-seeds` for the fail-closed
activation gates and `npm run policy:export` only after it passes. Cross-seed
stability is a reproducibility check, not equilibrium proof.

If projected storage is too large, first quantize and shard without changing
the game. Next test a compact open grid `2,2.5,3` plus all-in while preserving
the existing street sizes and one postflop raise. Only then consider coarsening
low-reach buckets. Any abstraction change requires two fresh seeds and every
gate again.

## Blueprint limitations

- The blueprint is much closer to a full heads-up 100bb game than push/fold,
  but it is still an action/card abstraction, not full-game GTO. It reports no
  exploitability or Nash-distance number. Self-play held-out EV is a smoke
  test, not a convergence certificate.
- Trajectory recall gives perfect recall within the chosen abstraction, but the
  card/action abstraction itself remains lossy.
- Evaluation reports unseen and pre-averaging information-set fractions. The
  runtime never uses their uniform evaluation fallback: a missing serving node
  pauses the table and is not scored.
- The root local-deviation gap can reveal a poor root mix, but it fixes every
  later decision and the opponent to the exported average strategy. It neither
  computes a recursive best response nor bounds whole-game exploitability.
- Two completed 100,000-iteration runs produced 9.4-9.5 million information
  sets each. Held-out unknown/untrained decisions fell below 4.8%/2.3%, but
  the pair still failed the automated stability gate: action MAE was 10.28%,
  median/p95 hand total variation was 22.40%/42.70%, primary-action agreement
  was 68.64%, and aggregate 100bb open-shoving remained about 1.8%.
- A resumable 200,000-iteration run produced 14.7 million information sets.
  Held-out unknown/untrained decisions improved to 2.94%/1.29%, while the
  root local-deviation estimate remained 1.20bb with a 1.02bb one-sided 99%
  lower bound. This is a decisive rejection, not a publishable deep-stack
  chart. The app therefore does not expose this blueprint as GTO guidance.
- The 200,000-iteration run peaked around 6.2GiB resident memory and its
  checkpoint is about 6.6GB. Raw blueprint artifacts/checkpoints are ignored;
  compact stability reports and the reproducible commands are versioned.
