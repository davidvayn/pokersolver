//! Offline depth-limited cash pilot. The leaf network estimates a fresh turn
//! subgame, not a frozen full-hand continuation. No safety/equilibrium claim.
use super::*;
use crate::cash_game::{Outcome, TerminalReason};

pub mod trace;

const SCHEMA: &str = "hu-cash-depth-limited-flop-pilot-v1";

// Shared by inference and observation so captured ranges cannot drift from
// the actual model query. The returned masses preserve raw chance weighting.
fn turn_prediction_input(
    game: &BlueprintConfig,
    flop: &[u8],
    state: &GameState,
    reaches: &[Vec<f64>; 2],
    turn: u8,
) -> Option<(TurnRiverSolveConfig, [f64; 2], f64)> {
    let (ranges, totals, joint_mass) = normalized_turn_ranges(reaches, turn)?;
    let mut board = flop.to_vec();
    board.push(turn);
    Some((TurnRiverSolveConfig {
        game: game.clone(),
        state: PublicBeliefState::turn_start(board.try_into().expect("cash turn board"), state.actor, state.invested, ranges),
        iterations: 2, averaging_delay: 0, river_refinement_iterations: 0, regret_matching_plus: false,
    }, totals, joint_mass))
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CashFlopPilotInput {
    pub game: BlueprintConfig,
    pub state: PublicBeliefState,
    pub iterations: u64,
    #[serde(default)]
    pub averaging_delay: u64,
    #[serde(default = "default_threads")]
    pub threads: usize,
}

fn default_threads() -> usize {
    2
}

impl CashFlopPilotInput {
    pub fn validate(&self) -> Result<(), String> {
        if !(2..=32).contains(&self.iterations)
            || !(1..=4).contains(&self.threads)
            || self.averaging_delay >= self.iterations
        {
            return Err(
                "cash flop pilot permits 2..32 updates, 1..4 workers, and a valid averaging delay"
                    .into(),
            );
        }
        self.game.validate()?;
        let state = &self.state;
        let rules = self
            .game
            .cash_rules
            .as_ref()
            .ok_or("cash flop pilot requires explicit rules")?;
        if state.street != Street::Flop
            || state.street_invested_bb != [0.0; 2]
            || cash::cash_units(state.invested_bb[0], rules)?
                != cash::cash_units(state.invested_bb[1], rules)?
            || state.checks != 0
            || state.aggressions != 0
            || !state.raise_reopened
            || state.last_full_raise_bb != 1.0
            || !state.trajectory.is_empty()
            || state.public_history != ["public_belief:flop_start".to_owned()]
            || state
                .invested_bb
                .iter()
                .any(|v| *v >= self.game.effective_stack_bb)
        {
            return Err(
                "cash flop pilot currently requires a fresh live equal-investment flop root".into(),
            );
        }
        state.validate_street_and_normalize_impl(&self.game, Street::Flop, 3, true)?;
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CashFlopPilotSolution {
    pub schema: String,
    pub input_sha256: String,
    pub rules_sha256: String,
    pub value_network_sha256: String,
    pub iterations: u64,
    pub threads: usize,
    pub game: BlueprintConfig,
    pub state: PublicBeliefState,
    pub profile_net_bb: [f64; 2],
    pub predicted_profile_payoff_sum_bb: f64,
    pub root: PublicBeliefStrategy,
    pub strategies: Vec<PublicBeliefStrategy>,
    pub turn_leaf_evaluations: u64,
    pub exact_all_in_terminal_evaluations: u64,
    pub action_value_method: String,
    pub full_game_exploitability: String,
    pub expected_house_rake_bb: Option<f64>,
    pub continuation_model_confidence: String,
    pub validation: BlueprintValidation,
}

/// Compare continuation hypotheses on one frozen public action policy. The
/// inner values still use an unqualified learned continuation, not exact EVs.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CashFlopFrozenEvaluation {
    pub schema: String,
    pub frozen_solution_sha256: String,
    pub frozen_strategy_sha256: String,
    pub policy_value_network_sha256: String,
    pub evaluation_value_network_sha256: String,
    pub updates_performed: u64,
    pub scored: CashFlopPilotSolution,
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

pub fn evaluate_frozen(
    input: CashFlopPilotInput,
    frozen: &CashFlopPilotSolution,
    frozen_solution_sha256: String,
    evaluation_network: PublicValueNetwork,
) -> Result<CashFlopFrozenEvaluation, String> {
    if frozen.schema != SCHEMA
        || frozen.validation.status != "research_only"
        || frozen.full_game_exploitability != "unmeasured"
        || frozen.expected_house_rake_bb.is_some()
        || frozen.continuation_model_confidence != "unqualified-research-only"
        || !valid_digest(&frozen_solution_sha256)
        || !valid_digest(&frozen.value_network_sha256)
        || !frozen.strategies.contains(&frozen.root)
    {
        return Err(
            "cash cross-scoring requires an explicit immutable research-only source policy".into(),
        );
    }
    input.validate()?;
    // Reuse the exact original request, not normalized output ranges. Even an
    // extra normalization can change f64 bits and invalidate the source hash.
    let expected_input_sha = format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(&(SCHEMA, &input, Some(frozen.value_network_sha256.as_str())))
                .map_err(|e| e.to_string())?
        )
    );
    if expected_input_sha != frozen.input_sha256
        || input.game.cash_rules.as_ref().unwrap().sha256()? != frozen.rules_sha256
        || frozen.game != input.game
        || frozen.iterations != input.iterations
        || frozen.threads != input.threads
        || frozen.root.public_history != input.state.public_history
        || frozen.root.actor != input.state.actor
    {
        return Err("frozen cash input/rules identity changed".into());
    }
    evaluation_network.validate()?;
    evaluation_network.validate_cash_game(&input.game)?;
    let evaluation_sha = evaluation_network
        .artifact_sha256()
        .filter(|v| valid_digest(v))
        .ok_or("cash cross-scoring requires hashed frozen evaluation weights")?
        .to_owned();
    let policy_sha = flop_strategy_sha256(&frozen.strategies);
    let evaluation_input_sha = format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(&(
                "hu-cash-frozen-flop-evaluation-v1",
                &frozen_solution_sha256,
                &policy_sha,
                &evaluation_sha
            ))
            .map_err(|e| e.to_string())?
        )
    );
    let mut solver = FlopSolver::new_impl(
        FlopResolveConfig {
            game: input.game,
            state: input.state,
            iterations: input.iterations,
            averaging_delay: input.averaging_delay,
            regret_matching_plus: false,
            value_network: evaluation_network,
            auxiliary_value_networks: vec![],
            continuation_selection: FlopContinuationSelection::Mean,
            threads: input.threads,
        },
        true,
    )?;
    if frozen.state != solver.config.state {
        return Err("frozen cash state differs from its normalized original request".into());
    }
    solver.load_frozen_average_strategies(&frozen.strategies)?;
    for row in &frozen.strategies {
        for (combo, mix) in row
            .probabilities
            .chunks(row.action_labels.len())
            .enumerate()
        {
            if mix.iter().any(|v| *v > 1.)
                || (solver.legal[row.actor][combo]
                    && (mix.iter().map(|v| *v as f64).sum::<f64>() - 1.).abs() > 1e-6)
                || (!solver.legal[row.actor][combo] && mix.iter().any(|v| *v != 0.))
            {
                return Err(
                    "frozen cash probabilities violate legal support or normalization".into(),
                );
            }
        }
    }
    // No regret/average updates: only the same policy's profile/action walks.
    let mut scored = solver.finish_cash_flop(evaluation_input_sha)?;
    for row in &mut scored.strategies {
        let original = frozen
            .strategies
            .iter()
            .find(|old| old.public_history == row.public_history)
            .ok_or("cross-scoring added an unexpected policy node")?;
        // Loading f32 averages applies the same normalization used by the
        // original solver. Export original bits, not a newly rounded policy.
        row.probabilities.clone_from(&original.probabilities);
    }
    scored.root = scored
        .strategies
        .iter()
        .find(|row| row.public_history == frozen.root.public_history)
        .ok_or("cross-scoring lost its frozen root")?
        .clone();
    scored.action_value_method =
        "frozen-cash-flop-policy-scored-with-learned-own-payoff-turn-leaves-v1".into();
    scored.validation.reasons.push("Zero training updates; source iteration count is provenance, not new training. Different weights are not independent exploitability evidence.".into());
    Ok(CashFlopFrozenEvaluation {
        schema: "hu-cash-frozen-flop-evaluation-v1".into(),
        frozen_solution_sha256,
        frozen_strategy_sha256: policy_sha,
        policy_value_network_sha256: frozen.value_network_sha256.clone(),
        evaluation_value_network_sha256: evaluation_sha,
        updates_performed: 0,
        scored,
    })
}

impl FlopSolver {
    pub(super) fn cash_flop_terminal_values(
        &self,
        state: &GameState,
        reaches: &[Vec<f64>; 2],
    ) -> [Vec<f64>; 2] {
        let rules = self
            .config
            .game
            .cash_rules
            .as_ref()
            .expect("cash pilot rules");
        if let Some(Terminal::Fold { winner }) = state.terminal {
            let own = cash::cash_terminal_payoffs(
                rules,
                state.invested,
                TerminalReason::Fold,
                if winner == 0 {
                    Outcome::PlayerZero
                } else {
                    Outcome::PlayerOne
                },
                3,
            )
            .expect("legal cent-aligned flop fold");
            return std::array::from_fn(|p| {
                compatible_masses_from_card_marginals(&all_combos(), &reaches[1 - p])
                    .into_iter()
                    .map(|mass| mass * own[p])
                    .collect()
            });
        }
        self.exact_all_in_terminal_evaluations
            .set(self.exact_all_in_terminal_evaluations.get() + 1);
        let terminal = self.cash_all_in.get_or_init(|| {
            Arc::new(
                range_vector::ExactCashFlopTerminal::new(
                    self.config
                        .state
                        .board
                        .clone()
                        .try_into()
                        .expect("cash flop board"),
                    rules.clone(),
                )
                .expect("validated cash flop terminal"),
            )
        });
        terminal
            .counterfactual_values(state.invested, reaches)
            .expect("legal cash all-in ranges")
    }

    pub(super) fn compute_cash_turn_leaf_values(
        &self,
        state: &GameState,
        reaches: &[Vec<f64>; 2],
    ) -> [Vec<f64>; 2] {
        let turns = (0..52u8)
            .filter(|card| !self.config.state.board.contains(card))
            .collect::<Vec<_>>();
        let flop = &self.config.state.board;
        let game = &self.config.game;
        let inference = &self.value_inference[0];
        let chunks = self.card_workers.map(&turns, |turn| {
            let (input, totals, _) = turn_prediction_input(game, flop, state, reaches, *turn)?;
            let ranges = &input.state.ranges;
            let own = inference
                .predict_cash(&input)
                .expect("validated cash leaf; no fallback");
            let combos = all_combos();
            let contribution: [Vec<f64>; 2] = std::array::from_fn(|p| {
                let masses = compatible_masses_from_card_marginals(&combos, &ranges[1 - p]);
                own[p]
                    .iter()
                    .zip(masses)
                    .map(|(v, m)| v * m * totals[1 - p] / 45.0)
                    .collect()
            });
            Some(contribution)
        });
        // Enumerate all 49 public proposals, but each compatible private pair
        // has exactly 45 legal turns. Masked-out proposals contribute zero;
        // they are never redrawn, nor is the whole value zero-sum shifted.
        let mut result = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        for chunk in chunks.into_iter().flatten() {
            for p in 0..2 {
                for (v, child) in result[p].iter_mut().zip(&chunk[p]) {
                    *v += child;
                }
            }
        }
        result
    }

    fn finish_cash_flop(self, input_sha256: String) -> Result<CashFlopPilotSolution, String> {
        let reaches = self.config.state.ranges.clone();
        let joint = joint_compatibility_mass(&reaches);
        let values =
            self.profile_walk(self.config.state.game_state(), reaches.clone(), None, false);
        let profile_net_bb: [f64; 2] = std::array::from_fn(|p| {
            reaches[p]
                .iter()
                .zip(&values[p])
                .map(|(r, v)| r * v)
                .sum::<f64>()
                / joint
        });
        let mut node_reaches = BTreeMap::new();
        let mut action_values = BTreeMap::new();
        self.collect_average_profile_diagnostics(
            self.config.state.game_state(),
            reaches,
            None,
            &mut node_reaches,
            &mut action_values,
        );
        let mut strategies = self.average_strategies(None);
        for row in &mut strategies {
            row.action_values_bb = Some(normalized_action_values_bb(
                row.actor,
                row.action_labels.len(),
                node_reaches
                    .get(&row.public_history)
                    .ok_or("missing cash node reaches")?,
                action_values
                    .get(&row.public_history)
                    .ok_or("missing cash action values")?,
                &self.conflicts,
            )?);
            if row.probabilities.iter().any(|p| !p.is_finite() || *p < 0.0)
                || row
                    .action_values_bb
                    .as_ref()
                    .unwrap()
                    .iter()
                    .any(|v| !v.is_finite())
            {
                return Err("nonfinite cash flop policy or action estimates; no fallback".into());
            }
        }
        let root = strategies
            .iter()
            .find(|s| s.public_history == self.config.state.public_history)
            .ok_or("cash flop root is missing")?
            .clone();
        Ok(CashFlopPilotSolution {
            schema: SCHEMA.into(), input_sha256,
            rules_sha256: self.config.game.cash_rules.as_ref().unwrap().sha256()?,
            value_network_sha256: self.config.value_network.artifact_sha256().ok_or("cash pilot requires a hashed frozen value artifact")?.into(),
            iterations: self.config.iterations, threads: self.config.threads,
            game: self.config.game.clone(), state: self.config.state.clone(), profile_net_bb,
            predicted_profile_payoff_sum_bb: profile_net_bb[0] + profile_net_bb[1],
            root, strategies, turn_leaf_evaluations: self.turn_leaf_evaluations.get(),
            exact_all_in_terminal_evaluations: self.exact_all_in_terminal_evaluations.get(),
            action_value_method: "exact-public-chance-and-terminals-with-learned-own-payoff-turn-leaves-v1".into(),
            full_game_exploitability: "unmeasured".into(), expected_house_rake_bb: None,
            continuation_model_confidence: "unqualified-research-only".into(),
            validation: BlueprintValidation { status: "research_only".into(), reasons: vec![
                "Fresh turn-subgame value estimates are not the frozen served continuation; no full-hand or safe-resolving guarantee.".into(),
                "Action estimates retain continuation bias. An exact chance sum does not qualify action-EV confidence.".into(),
                "A separate routed continuation and house-ledger traversal are required; payoff sums are never repaired or reported as measured rake.".into(),
            ] },
        })
    }
}

pub fn solve(
    input: CashFlopPilotInput,
    network: PublicValueNetwork,
) -> Result<CashFlopPilotSolution, String> {
    let (mut solver, input_sha256) = prepare_solver(input, network)?;
    solver.train_frozen_public_chance_pairs()?;
    solver.finish_cash_flop(input_sha256)
}

fn prepare_solver(
    input: CashFlopPilotInput,
    network: PublicValueNetwork,
) -> Result<(FlopSolver, String), String> {
    input.validate()?;
    network.validate()?;
    network.validate_cash_game(&input.game)?;
    if network.artifact_sha256().is_none() {
        return Err("cash flop pilot requires a hashed frozen value artifact".into());
    }
    let input_sha256 = format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(&(SCHEMA, &input, network.artifact_sha256(),))
                .map_err(|e| e.to_string())?
        )
    );
    let solver = FlopSolver::new_impl(
        FlopResolveConfig {
            game: input.game,
            state: input.state,
            iterations: input.iterations,
            averaging_delay: input.averaging_delay,
            regret_matching_plus: false,
            value_network: network,
            auxiliary_value_networks: vec![],
            continuation_selection: FlopContinuationSelection::Mean,
            threads: input.threads,
        },
        true,
    )?;
    Ok((solver, input_sha256))
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn fixture() -> (CashFlopPilotInput, PublicValueNetwork) {
        let mut game = BlueprintConfig::default();
        game.small_blind_bb = 0.4;
        game.effective_stack_bb = 20.0;
        game.cash_rules = Some(crate::cash_game::study_rules("nl25").unwrap());
        let mut network = super::super::tests::zero_shared_value_network();
        network.schema = cash_value::NETWORK_SCHEMA.into();
        network.prediction_contract = Some("cash-turn-start-cfv-full-stack-v1".into());
        network.payoff_contract = Some("own-net-bb-after-refunds-and-house-rake-v1".into());
        network.cash_rules = game.cash_rules.clone();
        network.rules_sha256 = Some(game.cash_rules.as_ref().unwrap().sha256().unwrap());
        network.source_game = Some(game.clone());
        network.source_validation_status = Some("research_only".into());
        network.artifact_sha256 = Some("a".repeat(64));
        network.value_normalization = Some("pot".into());
        network.head[0].activation = "linear".into();
        let board = [8, 13, 22];
        let input = CashFlopPilotInput {
            game,
            state: PublicBeliefState::flop_start(
                board,
                1,
                [19.96; 2],
                std::array::from_fn(|_| uniform_range(&board)),
            ),
            iterations: 2,
            averaging_delay: 0,
            threads: 2,
        };
        (input, network)
    }

    #[test]
    fn frozen_cash_cross_scoring_changes_values_without_retraining_probabilities() {
        let (input, network) = fixture();
        let frozen = solve(input.clone(), network.clone()).unwrap();
        let before = serde_json::to_vec(&frozen).unwrap();
        let self_score =
            evaluate_frozen(input.clone(), &frozen, "b".repeat(64), network.clone()).unwrap();
        assert_eq!(self_score.updates_performed, 0);
        assert_eq!(
            self_score.scored.root.probabilities,
            frozen.root.probabilities
        );
        for (left, right) in self_score
            .scored
            .profile_net_bb
            .iter()
            .zip(frozen.profile_net_bb)
        {
            assert!((left - right).abs() < 1e-6);
        }
        let mut changed = network;
        changed.head[0].biases[0] = 0.01;
        changed.artifact_sha256 = Some("c".repeat(64));
        let cross = evaluate_frozen(input.clone(), &frozen, "b".repeat(64), changed).unwrap();
        assert_eq!(cross.policy_value_network_sha256, "a".repeat(64));
        assert_eq!(cross.evaluation_value_network_sha256, "c".repeat(64));
        assert_eq!(
            cross.frozen_strategy_sha256,
            flop_strategy_sha256(&frozen.strategies)
        );
        assert_eq!(cross.scored.root.probabilities, frozen.root.probabilities);
        for (left, right) in cross.scored.strategies.iter().zip(&frozen.strategies) {
            assert_eq!(left.probabilities, right.probabilities);
            assert_eq!(left.public_history, right.public_history);
            assert_eq!(left.action_labels, right.action_labels);
        }
        let mut check_reaches = input.state.ranges.clone();
        let mut check_state = input.state.game_state();
        for _ in 0..2 {
            let row = frozen
                .strategies
                .iter()
                .find(|row| row.public_history == check_state.public_history)
                .unwrap();
            let width = row.action_labels.len();
            let column = row
                .action_labels
                .iter()
                .position(|label| label == "check")
                .unwrap();
            for (combo, reach) in check_reaches[row.actor].iter_mut().enumerate() {
                *reach *= row.probabilities[combo * width + column] as f64;
            }
            let action = check_state
                .legal_actions(&input.game)
                .into_iter()
                .find(|a| a.kind == ActionKind::Check)
                .unwrap();
            check_state = check_state.apply(&action, &input.game);
        }
        // At this nearly all-in root only check/check reaches the learned
        // leaf. Its exact path probability determines the visible EV change.
        let branch_probability = joint_compatibility_mass(&check_reaches)
            / joint_compatibility_mass(&input.state.ranges);
        let expected_change =
            2. * 0.01 * input.state.invested_bb.iter().sum::<f64>() * branch_probability;
        let actual_change = cross.scored.predicted_profile_payoff_sum_bb
            - self_score.scored.predicted_profile_payoff_sum_bb;
        assert!(expected_change > 1e-5);
        assert!(
            (actual_change - expected_change).abs() < 1e-6,
            "{actual_change} vs {expected_change}"
        );
        assert!(cross.scored.expected_house_rake_bb.is_none());
        assert_eq!(cross.scored.full_game_exploitability, "unmeasured");
        assert_eq!(cross.scored.validation.status, "research_only");
        assert_eq!(serde_json::to_vec(&frozen).unwrap(), before);
    }

    #[test]
    fn frozen_cash_cross_scoring_rejects_illegal_combo_mass() {
        let (input, network) = fixture();
        let mut frozen = solve(input.clone(), network.clone()).unwrap();
        let root_index = frozen
            .strategies
            .iter()
            .position(|row| row.public_history == frozen.root.public_history)
            .unwrap();
        let blocked = input.state.ranges[frozen.root.actor]
            .iter()
            .position(|v| *v == 0.)
            .unwrap();
        let width = frozen.root.action_labels.len();
        frozen.strategies[root_index].probabilities[blocked * width] = 0.5;
        frozen.root = frozen.strategies[root_index].clone();
        assert!(evaluate_frozen(input, &frozen, "b".repeat(64), network).is_err());
    }

    #[test]
    fn frozen_cash_cross_scoring_rejects_tampering_and_incompatible_rules() {
        let (input, network) = fixture();
        let frozen = solve(input.clone(), network.clone()).unwrap();
        for hash in ["".into(), "g".repeat(64), "b".repeat(63)] {
            assert!(evaluate_frozen(input.clone(), &frozen, hash, network.clone()).is_err());
        }
        let mut bad = frozen.clone();
        bad.schema = "home".into();
        assert!(evaluate_frozen(input.clone(), &bad, "b".repeat(64), network.clone()).is_err());
        let mut bad = frozen.clone();
        bad.state.invested_bb = [19.92; 2];
        assert!(evaluate_frozen(input.clone(), &bad, "b".repeat(64), network.clone()).is_err());
        let mut bad = frozen.clone();
        bad.strategies[0].probabilities[0] = f32::NAN;
        assert!(evaluate_frozen(input.clone(), &bad, "b".repeat(64), network.clone()).is_err());
        let mut bad = frozen.clone();
        bad.root.probabilities[0] = 0.9;
        assert!(evaluate_frozen(input.clone(), &bad, "b".repeat(64), network.clone()).is_err());
        let mut wrong = network;
        wrong.cash_rules = Some(crate::cash_game::study_rules("nl25-rake-off-control").unwrap());
        assert!(evaluate_frozen(input, &frozen, "b".repeat(64), wrong).is_err());
    }

    #[test]
    fn cash_flop_pilot_conserves_exact_baseline_payoffs_without_zero_sum_projection() {
        let (input, network) = fixture();
        let result = solve(input.clone(), network.clone()).unwrap();
        let mut serial = input.clone();
        serial.threads = 1;
        let serial_result = solve(serial, network.clone()).unwrap();
        assert_eq!(result.root, serial_result.root);
        assert_eq!(result.profile_net_bb, serial_result.profile_net_bb);
        let rules = input.game.cash_rules.as_ref().unwrap();
        let expected_house = rules
            .rake_units(2 * cash::cash_units(19.96, rules).unwrap(), true)
            .unwrap() as f64
            / rules.units_per_bb as f64;
        assert!(
            (result.predicted_profile_payoff_sum_bb + expected_house).abs() < 1e-10,
            "exact checkdown + terminal ledger sums to {} instead of -{expected_house}bb",
            result.predicted_profile_payoff_sum_bb
        );
        assert!(
            result.expected_house_rake_bb.is_none(),
            "no fabricated independent house ledger"
        );
        assert_eq!(result.full_game_exploitability, "unmeasured");
        assert_eq!(result.validation.status, "research_only");
        assert!(result.exact_all_in_terminal_evaluations > 0);
        let width = result.root.action_labels.len();
        for combo in all_combos() {
            if input.state.ranges[result.root.actor][combo.key()] == 0.0 {
                continue;
            }
            let row = &result.root.probabilities[combo.key() * width..(combo.key() + 1) * width];
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
        // Probe a fixed check/check leaf. Re-solving a biased game could change
        // its policy to avoid that leaf and confound this projection test.
        let mut biased = network;
        biased.head[0].biases[0] = 0.01;
        let solver = FlopSolver::new_impl(
            FlopResolveConfig {
                game: input.game.clone(),
                state: input.state.clone(),
                iterations: 2,
                averaging_delay: 0,
                regret_matching_plus: false,
                value_network: biased,
                auxiliary_value_networks: vec![],
                continuation_selection: FlopContinuationSelection::Mean,
                threads: 2,
            },
            true,
        )
        .unwrap();
        let mut state = input.state.game_state();
        for _ in 0..2 {
            let check = state
                .legal_actions(&input.game)
                .into_iter()
                .find(|a| a.kind == ActionKind::Check)
                .unwrap();
            state = state.apply(&check, &input.game);
        }
        let biased_values = solver.compute_cash_turn_leaf_values(&state, &input.state.ranges);
        let joint = joint_compatibility_mass(&input.state.ranges);
        let biased_sum: f64 = (0..2)
            .map(|p| {
                input.state.ranges[p]
                    .iter()
                    .zip(&biased_values[p])
                    .map(|(r, v)| r * v)
                    .sum::<f64>()
                    / joint
            })
            .sum();
        assert!(
            biased_sum > -expected_house + 0.1,
            "network bias must remain visible, not shifted to pass accounting: {biased_sum}"
        );
        assert!(biased_sum.abs() > 0.1);
    }

    #[test]
    fn cash_flop_pilot_rejects_cross_game_weights_and_keeps_home_routes_closed() {
        let (input, network) = fixture();
        assert!(input
            .state
            .validate_street_and_normalize(&input.game, Street::Flop, 3)
            .is_err());
        assert!(solve(
            input.clone(),
            super::super::tests::zero_shared_value_network()
        )
        .is_err());
        let mut wrong = input.clone();
        wrong.game.effective_stack_bb = 40.0;
        assert!(solve(wrong, network.clone()).is_err());
        let mut wrong = input.clone();
        wrong.state.checks = 1;
        assert!(solve(wrong, network.clone()).is_err());
        let mut wrong = input.clone();
        wrong.iterations = 33;
        assert!(solve(wrong, network.clone()).is_err());
        let mut wrong = input.clone();
        wrong.threads = 5;
        assert!(solve(wrong, network.clone()).is_err());
        assert!(FlopSolver::new(FlopResolveConfig {
            game: input.game,
            state: input.state,
            iterations: 2,
            averaging_delay: 0,
            regret_matching_plus: false,
            value_network: network,
            auxiliary_value_networks: vec![],
            continuation_selection: FlopContinuationSelection::Mean,
            threads: 2,
        })
        .is_err());
    }

    #[test]
    fn cached_cash_turn_values_validate_rules_and_history_before_cache_hits() {
        let (input, network) = fixture();
        let session = value_inference::ValueInferenceSession::new(network.clone());
        let board = [8, 13, 22, 31];
        let query = TurnRiverSolveConfig {
            game: input.game,
            state: PublicBeliefState::turn_start(
                board,
                1,
                [2.0; 2],
                std::array::from_fn(|_| uniform_range(&board)),
            ),
            iterations: 2,
            averaging_delay: 0,
            river_refinement_iterations: 0,
            regret_matching_plus: false,
        };
        let values = session.predict_cash(&query).unwrap();
        assert_eq!(values, session.predict_cash(&query).unwrap());
        assert_eq!(values, network.predict_cash_turn(&query).unwrap());
        let mut counterfactual = query.clone();
        let key = Combo::new(48, 49).key();
        counterfactual.state.ranges[0][key] = 0.0;
        let absent = network.predict_cash_turn(&counterfactual).unwrap();
        assert!(
            (absent[0][key] - values[0][key]).abs() < 1e-10,
            "zero-correction checkdown query lost its counterfactual own hand: {} vs {}",
            absent[0][key],
            values[0][key]
        );
        let mut wrong = query.clone();
        wrong.state.checks = 1;
        assert!(session.predict_cash(&wrong).is_err());
        let mut wrong = query;
        wrong.game.action_abstraction.open_sizes_bb.push(6.0);
        assert!(session.predict_cash(&wrong).is_err());
    }
}
