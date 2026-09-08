//! Native, range-conditioned flop-trunk pilot. One public turn proposal per
//! frozen iteration; exact all-in terminals, full joint turn/river leaves.
//! This does not implement the safe runtime reconstruction required by CFR-D.
use super::*;
use std::cell::RefCell;
mod chance_baseline;
mod chance_batch;
mod continuation;
mod root_support;
pub(in crate::blueprint) use root_support::exact_flop_kernel;
mod forced_beliefs;
mod frozen_response;
pub(in crate::blueprint) use frozen_response::playback::{NativeFlopOptions, NativePostflopPolicy};
mod parallel_leaves;
mod root_input;
mod search_beliefs;
mod value_targets;

// Observation is ordered, opt-in, and outside the regret/chance computation.
// A failed capture aborts the pilot; it never substitutes a continuation value.
type LeafObserver<'a> = dyn FnMut(u64, &TurnRiverSolveConfig, &Values) -> Result<(), String> + 'a;

// A proposer may expose its inputs, never its predictions as native labels.
enum LeafObservation<'a, 'b> {
    Native(&'a mut LeafObserver<'b>),
    Beliefs(&'a mut (dyn FnMut(u64, &TurnRiverSolveConfig) -> Result<(), String> + 'b)),
}

impl LeafObservation<'_, '_> {
    fn observe(
        &mut self,
        round: u64,
        config: &TurnRiverSolveConfig,
        leaf: &continuation::Leaf,
    ) -> Result<(), String> {
        match self {
            Self::Native(callback) => match leaf {
                continuation::Leaf::Native(values) => callback(round, config, values),
                continuation::Leaf::Predicted(_) => {
                    Err("predicted values cannot enter native target capture".into())
                }
            },
            Self::Beliefs(callback) => callback(round, config),
        }
    }
}

fn continuation_config(
    game: &BlueprintConfig,
    flop: &[u8],
    state: &GameState,
    reaches: &[Vec<f64>; 2],
    turn: u8,
    iterations: u64,
) -> TurnRiverSolveConfig {
    let mut board = flop.to_vec();
    board.push(turn);
    let mut masked = reaches.clone();
    for range in &mut masked {
        for combo in all_combos() {
            if combo.cards().contains(&turn) {
                range[combo.key()] = 0.0;
            }
        }
    }
    TurnRiverSolveConfig {
        game: game.clone(),
        state: PublicBeliefState::from_game_state(board, state, masked),
        iterations,
        averaging_delay: 0,
        river_refinement_iterations: 0,
        regret_matching_plus: false,
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct Solution {
    schema: String,
    game: BlueprintConfig,
    state: PublicBeliefState,
    seed: u64,
    iterations: u64,
    turn_iterations: u64,
    // Optional reconstruction budget; never rewrites the training budget.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    response_turn_iterations: Option<u64>,
    strategies: Vec<PublicBeliefStrategy>,
    turn_queries: u64,
    zero_own_reach_completions: [u64; 2],
    maximum_conditional_turn_response_gain_bb: Option<f64>,
    zero_joint_turn_queries: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    chance_baseline: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    turn_samples_per_iteration: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    learned_leaf_model_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    complete_root_support: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    root_realization_turn_averages: Option<bool>,
}

struct Trunk {
    game: BlueprintConfig,
    state: PublicBeliefState,
    legal: [Vec<bool>; 2],
    nodes: BTreeMap<Vec<String>, RangeNode>,
    equity: OnceLock<Arc<Vec<f32>>>,
    complete_root_support: bool,
}

impl Trunk {
    fn new(game: BlueprintConfig, state: PublicBeliefState) -> Result<Self, String> {
        game.validate()?;
        let state = state.validate_street_and_normalize(&game, Street::Flop, 3)?;
        let legal = std::array::from_fn(|p| state.ranges[p].iter().map(|v| *v > 0.0).collect());
        let mut trunk = Self {
            game,
            state,
            legal,
            nodes: BTreeMap::new(),
            equity: OnceLock::new(),
            complete_root_support: false,
        };
        trunk.prepare(trunk.state.game_state());
        Ok(trunk)
    }

    fn prepare(&mut self, state: GameState) {
        if state.terminal.is_some() || state.street == Street::Turn {
            return;
        }
        let actions = state.legal_actions(&self.game);
        self.nodes
            .entry(state.public_history.clone())
            .or_insert_with(|| RangeNode::new(state.actor, &actions));
        for action in actions {
            self.prepare(state.apply(&action, &self.game));
        }
    }

    fn terminal(&self, state: &GameState, reaches: &[Vec<f64>; 2]) -> [Vec<f64>; 2] {
        if let Some(Terminal::Fold { winner }) = state.terminal {
            let p0 = if winner == 0 {
                state.invested[1]
            } else {
                -state.invested[0]
            };
            let conflicts = combo_conflicts();
            return std::array::from_fn(|p| {
                (0..COMBO_COUNT)
                    .map(|combo| {
                        let utility = if p == 0 { p0 } else { -p0 };
                        utility * compatible_mass_from_conflicts(&reaches[1 - p], &conflicts, combo)
                    })
                    .collect()
            });
        }
        assert!(matches!(state.terminal, Some(Terminal::Showdown)));
        let equity = self.equity.get_or_init(|| {
            exact_flop_all_in_equities(self.state.board.clone().try_into().unwrap(), &self.legal, 1)
        });
        let mut values = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        for p0 in 0..COMBO_COUNT {
            if !self.legal[0][p0] {
                continue;
            }
            for p1 in 0..COMBO_COUNT {
                let e = equity[p0 * COMBO_COUNT + p1];
                if !e.is_finite() {
                    continue;
                }
                let v = e as f64 * state.invested[1] - (1.0 - e as f64) * state.invested[0];
                values[0][p0] += reaches[1][p1] * v;
                values[1][p1] -= reaches[0][p0] * v;
            }
        }
        values
    }

    fn iteration(
        &mut self,
        round: u64,
        leaf: &dyn Fn(&GameState, &[Vec<f64>; 2], Option<usize>) -> [Vec<f64>; 2],
    ) {
        for node in self.nodes.values_mut() {
            node.discount_regrets(round, &self.game.dcfr);
        }
        let mut deltas = BTreeMap::new();
        frozen_flop_walk::Walker {
            game: &self.game,
            legal: &self.legal,
            nodes: &self.nodes,
            terminal: &|state, ranges| self.terminal(state, ranges),
            turn: leaf,
        }
        .walk(
            self.state.game_state(),
            self.state.ranges.clone(),
            None,
            true,
            &mut deltas,
        );
        self.apply_deltas(round, deltas);
    }

    fn apply_deltas(&mut self, round: u64, mut deltas: BTreeMap<Vec<String>, FrozenRangeNodeDelta>) {
        if self.complete_root_support {
            self.replace_counterfactual_averages(&mut deltas);
        }
        for (key, delta) in deltas {
            let node = self.nodes.get_mut(&key).unwrap();
            for (r, d) in node.regrets.iter_mut().zip(delta.regrets) {
                *r += d;
            }
            for (s, d) in node.strategy_sum.iter_mut().zip(delta.strategy_sum) {
                *s += d;
            }
            node.finish_average_update(round, &self.game.dcfr);
        }
    }
}

fn train(
    game: BlueprintConfig,
    state: PublicBeliefState,
    seed: u64,
    iterations: u64,
    turn_iterations: u64,
) -> Result<Solution, String> {
    train_with_baseline(game, state, seed, iterations, turn_iterations, false)
}

fn train_with_baseline(
    game: BlueprintConfig,
    state: PublicBeliefState,
    seed: u64,
    iterations: u64,
    turn_iterations: u64,
    use_baseline: bool,
) -> Result<Solution, String> {
    train_with_sampling(
        game,
        state,
        seed,
        iterations,
        turn_iterations,
        use_baseline,
        1,
    )
}

fn train_with_sampling(
    game: BlueprintConfig,
    state: PublicBeliefState,
    seed: u64,
    iterations: u64,
    turn_iterations: u64,
    use_baseline: bool,
    turn_samples: usize,
) -> Result<Solution, String> {
    train_with_leaf_workers(
        game,
        state,
        seed,
        iterations,
        turn_iterations,
        use_baseline,
        turn_samples,
        1,
    )
}

fn train_with_leaf_workers(
    game: BlueprintConfig,
    state: PublicBeliefState,
    seed: u64,
    iterations: u64,
    turn_iterations: u64,
    use_baseline: bool,
    turn_samples: usize,
    leaf_workers: usize,
) -> Result<Solution, String> {
    train_with_observer(
        game,
        state,
        seed,
        iterations,
        turn_iterations,
        use_baseline,
        turn_samples,
        leaf_workers,
        None,
    )
}

fn train_with_observer(
    game: BlueprintConfig,
    state: PublicBeliefState,
    seed: u64,
    iterations: u64,
    turn_iterations: u64,
    use_baseline: bool,
    turn_samples: usize,
    leaf_workers: usize,
    observer: Option<&mut LeafObserver<'_>>,
) -> Result<Solution, String> {
    train_with_evaluator(game, state, seed, iterations, turn_iterations, use_baseline,
        turn_samples, leaf_workers, observer.map(LeafObservation::Native), continuation::Evaluator::Native)
}

fn train_with_evaluator(
    game: BlueprintConfig,
    state: PublicBeliefState,
    seed: u64,
    iterations: u64,
    turn_iterations: u64,
    use_baseline: bool,
    turn_samples: usize,
    leaf_workers: usize,
    observer: Option<LeafObservation<'_, '_>>,
    evaluator: continuation::Evaluator<'_>,
) -> Result<Solution, String> {
    train_with_root_support(game, state, seed, iterations, turn_iterations,
        use_baseline, turn_samples, leaf_workers, observer, evaluator, false)
}

fn train_with_root_support(
    game: BlueprintConfig,
    state: PublicBeliefState,
    seed: u64,
    iterations: u64,
    turn_iterations: u64,
    use_baseline: bool,
    turn_samples: usize,
    leaf_workers: usize,
    mut observer: Option<LeafObservation<'_, '_>>,
    evaluator: continuation::Evaluator<'_>,
    complete_root_support: bool,
) -> Result<Solution, String> {
    let learned_leaf_model_sha256 = evaluator.model_sha256()?;
    let captures_native_values = observer.as_ref().is_some_and(|o| matches!(o, LeafObservation::Native(_)));
    if learned_leaf_model_sha256.is_some() && (captures_native_values || use_baseline) {
        return Err("learned predictions cannot be exported as native labels or combined with a chance baseline".into());
    }
    if !(1..=4).contains(&leaf_workers) || (leaf_workers > 1 && use_baseline) {
        return Err(
            "native leaf workers require 1..4 threads and no shared learned baseline".into(),
        );
    }
    if iterations < 2 || turn_iterations < 2 {
        return Err("native flop pilot requires at least two iterations".into());
    }
    if !(1..=49).contains(&turn_samples) || (use_baseline && turn_samples != 1) {
        return Err("native flop turn batches require 1..49 cards and no combined baseline".into());
    }
    let mut trunk = if complete_root_support {
        Trunk::new_counterfactual(game.clone(), state)?
    } else { Trunk::new(game.clone(), state)? };
    let turns = (0..52u8)
        .filter(|c| !trunk.state.board.contains(c))
        .collect::<Vec<_>>();
    let mut chance = SplitMix64::new(seed);
    let queries = Cell::new(0u64);
    let zero_reach = Cell::new([0u64; 2]);
    let zero_joint = Cell::new(0u64);
    let residual = Cell::new(0.0f64);
    let flop: [u8; 3] = trunk.state.board.clone().try_into().unwrap();
    let references = RefCell::new(BTreeMap::new());
    for round in 1..=iterations {
        let sampled = chance_batch::draw(&turns, turn_samples, &mut chance);
        let game = &game;
        if leaf_workers > 1 || observer.is_some() || learned_leaf_model_sha256.is_some() {
            if use_baseline {
                return Err(
                    "native value capture cannot be combined with a learned chance baseline".into(),
                );
            }
            let diagnostics = parallel_leaves::iteration_evaluated(
                &mut trunk,
                round,
                &sampled,
                turn_iterations,
                leaf_workers,
                &mut observer,
                evaluator,
            )?;
            queries.set(queries.get() + diagnostics.queries);
            zero_reach.set(std::array::from_fn(|p| {
                zero_reach.get()[p] + diagnostics.zero_own[p]
            }));
            zero_joint.set(zero_joint.get() + diagnostics.zero_joint);
            residual.set(residual.get().max(diagnostics.maximum_gain));
        } else {
            // Neither player's update can affect the other's policy in this pass.
            // The chance draw precedes the immutable all-player vector traversal.
            trunk.iteration(round, &|state, reaches, traverser| {
                assert_eq!(traverser, None);
                let values_for_turn = |turn| {
                    let values = solve(continuation_config(
                        game,
                        &flop,
                        state,
                        reaches,
                        turn,
                        turn_iterations,
                    ))
                    .expect(
                        "native turn oracle failed; no substitute values or partial policy export",
                    );
                    queries.set(queries.get() + 1);
                    zero_reach.set(std::array::from_fn(|p| {
                        zero_reach.get()[p] + values.completed_zero_own_reach[p] as u64
                    }));
                    if let Some(gains) = values.conditional_response_gain_bb {
                        residual.set(residual.get().max(gains[0]).max(gains[1]));
                    } else {
                        zero_joint.set(zero_joint.get() + 1);
                    }
                    values.counterfactual_bb
                };
                // Proposal is uniform over 49 unseen public cards; a compatible
                // exact private pair has 45 possible turns. No river is sampled.
                if use_baseline {
                    let turn = sampled[0];
                    // Each perfect-recall public leaf is visited once per pass.
                    // Its reference contains only earlier rounds, never this draw.
                    references
                        .borrow_mut()
                        .entry(state.public_history.clone())
                        .or_insert_with(|| chance_baseline::Baseline::new(flop))
                        .correct_and_learn(turn, reaches, &values_for_turn(turn))
                } else {
                    chance_batch::estimate(&sampled, values_for_turn)
                }
            });
        }
        eprintln!("native-flop seed={seed} round={round}/{iterations} turns={sampled:?} turn_queries={} zero_own={:?}",
            queries.get(), zero_reach.get());
    }
    let strategies = trunk
        .nodes
        .iter()
        .map(|(history, node)| PublicBeliefStrategy {
            public_history: history.clone(),
            actor: node.actor,
            action_labels: node.action_labels.clone(),
            probabilities: node
                .average_strategy(&trunk.legal[node.actor])
                .into_iter()
                .map(|p| p as f32)
                .collect(),
            action_values_bb: None,
        })
        .collect();
    Ok(Solution {
        schema: "hu-native-counterfactual-turn-flop-pilot-v1".into(),
        game,
        state: trunk.state,
        seed,
        iterations,
        turn_iterations,
        response_turn_iterations: None,
        strategies,
        turn_queries: queries.get(),
        zero_own_reach_completions: zero_reach.get(),
        maximum_conditional_turn_response_gain_bb: learned_leaf_model_sha256.is_none().then(|| residual.get()),
        zero_joint_turn_queries: zero_joint.get(),
        chance_baseline: use_baseline.then(|| "learned_conditional_turn_v1".into()),
        turn_samples_per_iteration: (turn_samples > 1).then_some(turn_samples),
        learned_leaf_model_sha256,
        complete_root_support: complete_root_support.then_some(true),
        root_realization_turn_averages: None,
    })
}

#[test]
fn native_flop_batch_combines_four_turns_per_frozen_update() {
    let game = super::tests::config().game;
    let board = [0, 5, 10];
    let state = PublicBeliefState::flop_start(
        board,
        1,
        [1.0, 1.0],
        std::array::from_fn(|_| uniform_range(&board)),
    );
    let control = train(game.clone(), state.clone(), 100101, 2, 4).unwrap();
    let batch = train_with_sampling(game.clone(), state.clone(), 100101, 2, 4, false, 4).unwrap();
    assert_eq!(batch.turn_queries, 4 * control.turn_queries);
    assert_eq!(batch.iterations, control.iterations);
    assert_eq!(batch.strategies.len(), control.strategies.len());
    assert!(frozen_response::Frozen::new(&batch).is_ok());
    assert_eq!(batch.turn_samples_per_iteration, Some(4));
    assert_eq!(control.turn_samples_per_iteration, None);
    let repeat = train_with_sampling(game, state, 100101, 2, 4, false, 4).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(serde_json::to_vec(&batch).unwrap())),
        format!("{:x}", Sha256::digest(serde_json::to_vec(&repeat).unwrap()))
    );
    assert!(train_with_sampling(control.game, control.state, 100101, 2, 4, true, 4).is_err());
}

#[test]
fn native_flop_batch_does_not_multiply_terminal_regret_or_average_updates() {
    let game = super::tests::config().game;
    let board = [0, 5, 10];
    let public = PublicBeliefState::flop_start(
        board,
        1,
        [1.0, 1.0],
        std::array::from_fn(|_| uniform_range(&board)),
    );
    let root = public.game_state();
    let shove = root
        .legal_actions(&game)
        .into_iter()
        .find(|a| a.label.contains("all_in"))
        .unwrap();
    let facing = root.apply(&shove, &game);
    assert!(facing
        .legal_actions(&game)
        .iter()
        .all(|a| facing.apply(a, &game).terminal.is_some()));
    let state = PublicBeliefState::from_game_state(board.to_vec(), &facing, public.ranges);
    let control = train(game.clone(), state.clone(), 100101, 4, 4).unwrap();
    let batch = train_with_sampling(game, state, 100101, 4, 4, false, 4).unwrap();
    assert_eq!((control.turn_queries, batch.turn_queries), (0, 0));
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&control.strategies).unwrap())
        ),
        format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&batch.strategies).unwrap())
        )
    );
}

#[test]
fn shared_flop_traversal_matches_existing_solver_with_the_same_leaf_values() {
    let input = super::tests::config();
    let board = [0, 5, 10];
    let public = PublicBeliefState::flop_start(
        board,
        1,
        [1.0, 1.0],
        std::array::from_fn(|_| uniform_range(&board)),
    );
    let old_config = FlopResolveConfig {
        game: input.game.clone(),
        state: public.clone(),
        iterations: 2,
        averaging_delay: 0,
        regret_matching_plus: false,
        value_network: super::super::tests::zero_value_network(),
        auxiliary_value_networks: Vec::new(),
        continuation_selection: FlopContinuationSelection::Mean,
        threads: 1,
    };
    let mut old = FlopSolver::new(old_config).unwrap();
    old.prepare_public_tree(old.config.state.game_state());
    let mut native = Trunk::new(input.game, public).unwrap();
    assert_eq!(native.state, old.config.state);
    let root = native.state.game_state();
    let bet = root
        .legal_actions(&native.game)
        .into_iter()
        .find(|a| a.kind != ActionKind::Check)
        .unwrap();
    let facing = root.apply(&bet, &native.game);
    let fold = facing
        .legal_actions(&native.game)
        .into_iter()
        .find(|a| a.kind == ActionKind::Fold)
        .unwrap();
    let terminal = facing.apply(&fold, &native.game);
    assert_eq!(
        native.terminal(&terminal, &native.state.ranges),
        old.terminal_values(&terminal, &old.config.state.ranges)
    );
    for round in 1..=2 {
        native.iteration(round, &|state, ranges, traverser| {
            old.turn_leaf_values(state, ranges, traverser)
        });
        old.frozen_all_player_iteration(round);
        assert_eq!(native.nodes.len(), old.nodes.len());
        for (key, node) in &native.nodes {
            assert_eq!(node.regrets, old.nodes[key].regrets);
            assert_eq!(node.strategy_sum, old.nodes[key].strategy_sum);
        }
    }
}

#[test]
fn native_flop_updates_are_deterministic_legal_and_do_not_omit_public_nodes() {
    let game = super::tests::config().game;
    let board = [0, 5, 10];
    let state = PublicBeliefState::flop_start(
        board,
        1,
        [1.0, 1.0],
        std::array::from_fn(|_| uniform_range(&board)),
    );
    let first = train(game.clone(), state.clone(), 100101, 2, 4).unwrap();
    let second = train(game, state, 100101, 2, 4).unwrap();
    assert_eq!(
        serde_json::to_vec(&first).unwrap(),
        serde_json::to_vec(&second).unwrap()
    );
    assert!(first.turn_queries > 0 && first.zero_own_reach_completions.iter().sum::<u64>() > 0);
    for row in &first.strategies {
        let n = row.action_labels.len();
        for combo in all_combos() {
            if combo.cards().iter().any(|c| board.contains(c)) {
                continue;
            }
            let p = &row.probabilities[combo.key() * n..(combo.key() + 1) * n];
            assert!(p.iter().all(|p| p.is_finite() && *p >= 0.0));
            assert!((p.iter().map(|v| *v as f64).sum::<f64>() - 1.0).abs() < 1e-6);
        }
    }
}

#[test]
fn corrected_native_flop_pilot_is_deterministic_and_keeps_the_default_artifact_contract() {
    let game = super::tests::config().game;
    let board = [0, 5, 10];
    let state = PublicBeliefState::flop_start(
        board,
        1,
        [1.0, 1.0],
        std::array::from_fn(|_| uniform_range(&board)),
    );
    let first = train_with_baseline(game.clone(), state.clone(), 100101, 4, 4, true).unwrap();
    let repeat = train_with_baseline(game.clone(), state.clone(), 100101, 4, 4, true).unwrap();
    assert_eq!(
        serde_json::to_vec(&first).unwrap(),
        serde_json::to_vec(&repeat).unwrap()
    );
    assert_eq!(
        first.chance_baseline.as_deref(),
        Some("learned_conditional_turn_v1")
    );
    assert_eq!(first.turn_queries, 4); // one joint turn solve per iteration here
    assert!(frozen_response::Frozen::new(&first).is_ok());
    let old = train(game, state, 100101, 2, 4).unwrap();
    let bytes = serde_json::to_vec(&old).unwrap();
    assert!(!String::from_utf8_lossy(&bytes).contains("chance_baseline"));
    assert!(!String::from_utf8_lossy(&bytes).contains("turn_samples_per_iteration"));
    let parsed: Solution = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(serde_json::to_vec(&parsed).unwrap(), bytes);
}

#[test]
#[ignore = "explicit seed/output and external 2GiB/time/disk guard; native-turn flop development pilot, not full-game qualification"]
fn saved_20bb_native_flop_pilot() {
    let (input, input_sha256) = match std::env::var("POKER_NATIVE_FLOP_INPUT") {
        Ok(path) => {
            assert!(fs::metadata(&path).unwrap().len() <= 1024 * 1024);
            (
                fs::read(path).unwrap(),
                std::env::var("POKER_NATIVE_FLOP_INPUT_SHA").unwrap(),
            )
        }
        Err(std::env::VarError::NotPresent) => {
            assert!(std::env::var_os("POKER_NATIVE_FLOP_INPUT_SHA").is_none());
            let bytes =
                include_bytes!("../../../../tests/fixtures/flop-continuation-public-root-a.json")
                    .to_vec();
            let sha = format!("{:x}", Sha256::digest(&bytes));
            (bytes, sha)
        }
        other => panic!("invalid native flop input path: {other:?}"),
    };
    let (game, state) = root_input::decode(&input, &input_sha256).unwrap();
    let seed: u64 = std::env::var("POKER_NATIVE_FLOP_SEED")
        .unwrap()
        .parse()
        .unwrap();
    assert!([100101, 100102].contains(&seed));
    let iterations: u64 = std::env::var("POKER_NATIVE_FLOP_ITERATIONS")
        .unwrap_or_else(|_| "2".into())
        .parse()
        .unwrap();
    assert!([2, 8, 16, 32, 128].contains(&iterations));
    let turn_iterations: u64 = std::env::var("POKER_NATIVE_FLOP_TURN_ITERATIONS")
        .unwrap_or_else(|_| "64".into())
        .parse()
        .unwrap();
    assert!([4, 64, 128].contains(&turn_iterations));
    let started = std::time::Instant::now();
    let use_baseline = match std::env::var("POKER_NATIVE_FLOP_CHANCE_BASELINE").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("none") => false,
        Ok("learned_conditional_turn_v1") => true,
        other => panic!("unsupported native flop chance baseline: {other:?}"),
    };
    let turn_samples: usize = std::env::var("POKER_NATIVE_FLOP_TURN_SAMPLES")
        .unwrap_or_else(|_| "1".into())
        .parse()
        .unwrap();
    assert!([1, 4].contains(&turn_samples));
    let leaf_workers: usize = std::env::var("POKER_NATIVE_FLOP_LEAF_WORKERS")
        .unwrap_or_else(|_| "1".into())
        .parse()
        .unwrap();
    assert!((1..=4).contains(&leaf_workers));
    let learned = std::env::var("POKER_NATIVE_FLOP_VALUE_MODEL").ok().map(|path| {
        let network = PublicValueNetwork::read(Path::new(&path)).unwrap();
        assert_eq!(network.artifact_sha256().unwrap(), std::env::var("POKER_NATIVE_FLOP_VALUE_MODEL_SHA").unwrap());
        network
    });
    let result = train_with_evaluator(
        game,
        state,
        seed,
        iterations,
        turn_iterations,
        use_baseline,
        turn_samples,
        leaf_workers,
        None,
        learned.as_ref().map_or(continuation::Evaluator::Native, continuation::Evaluator::Learned),
    )
    .unwrap();
    let path = PathBuf::from(std::env::var("POKER_NATIVE_FLOP_OUTPUT").unwrap());
    let bytes = serde_json::to_vec(&result).unwrap();
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    file.write_all(&bytes).unwrap();
    file.sync_all().unwrap();
    println!(
        "{}",
        serde_json::json!({"stage":"native_flop_pilot","seed":seed,"iterations":iterations,"turnIterations":turn_iterations,
        "publicInputSha256":input_sha256,
        "turnSamplesPerIteration":turn_samples,
        "leafWorkers":leaf_workers,
        "seconds":started.elapsed().as_secs_f64(),"turnQueries":result.turn_queries,
        "zeroOwnReachCompletions":result.zero_own_reach_completions,"zeroJointTurnQueries":result.zero_joint_turn_queries,
        "maximumConditionalTurnResponseGainBb":result.maximum_conditional_turn_response_gain_bb,
        "policyRows":result.strategies.len(),"outputSha256":format!("{:x}",Sha256::digest(&bytes)),
        "interpretation":"development flop-policy update with native range-conditioned turn leaves; not full-game exploitability or safe runtime reconstruction"})
    );
}
