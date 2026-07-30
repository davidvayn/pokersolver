# Native preflop solver

This standalone Rust crate is an offline validation and artifact-generation
tool. It deliberately does not extend the browser WASM solver: the existing
WASM game is a postflop, single-street all-in-equity model and has different
game semantics.

## Milestones implemented

1. Deterministic full-tree CFR+ for three-card Kuhn poker. Its tests check the
   known first-player game value of `-1/18` and compute exact exploitability by
   enumerating all pure best responses.
2. Chance-sampled CFR+ for heads-up equal-stack push/fold Hold'em. Strategies
   are learned for all 1,326 exact two-card combinations. Chance sampling deals
   compatible hands, so card removal is retained.
3. A versioned JSON exporter with exact-combo and 169-class strategy views,
   configuration, payoff convention, best-response metrics, and validation
   status.
4. An experimental heads-up multi-street blueprint. It uses external-sampling
   MCCFR, regret-matching+, post-delay linear averaging, exact sampled
   deals/runouts, a
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
  --effective-stack-bb 100 \
  --iterations 100000 \
  --max-information-sets 12000000 \
  --averaging-delay 20000 \
  --held-out-deals 30000 \
  --root-deviation-samples 512 \
  --checkpoint /tmp/hu-blueprint.checkpoint.json \
  --checkpoint-every 100000 \
  --output /tmp/hu-blueprint.json

npm run preflop:compare-blueprints -- \
  /tmp/hu-blueprint-seed-1.json \
  /tmp/hu-blueprint-seed-2.json \
  --require-pass
```

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
sampled opponent hand or unrevealed board. The default `current_street` recall
mode drops earlier private/public bucket labels after each street while keeping
the complete public action history. This is explicitly imperfect recall.
`--trajectory-recall` retains every prior bucket at a substantial memory cost.

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

## Blueprint limitations

- The blueprint is much closer to a full heads-up 100bb game than push/fold,
  but it is still an action/card abstraction, not full-game GTO. It reports no
  exploitability or Nash-distance number. Self-play held-out EV is a smoke
  test, not a convergence certificate.
- The default current-street postflop abstraction has imperfect recall.
  Perfect-recall CFR convergence guarantees therefore do not transfer.
- Unseen and pre-averaging held-out information sets fall back to uniform play,
  and their fractions are reported. Cross-seed stability and a local best
  response are still required before using output as solver-backed guidance.
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
