//! Research fixed-flop subgame proposal, not a safe continual resolver.
//!
//! Reuse the full-hand PCS/DCFR betting traversal from a supplied public root.
//! Both future public cards are sampled; no learned leaf value is substituted.
//! Only the root's frozen average policy is returned. Full-game protection and
//! agreement with a separately served turn/river policy require external tests.
use super::*;
use crate::blueprint::neural::deal_for_policy_combo_on_board;

const FUTURE_CHANCE_CORRECTION: f64 = (49.0 * 48.0) / (45.0 * 44.0);

#[derive(Clone, Debug)]
pub struct SampledFlopConfig {
    pub game: BlueprintConfig,
    pub state: PublicBeliefState,
    pub iterations: u64,
    pub seed: u64,
    pub maximum_information_sets: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SampledFlopSolution {
    pub schema: String,
    pub input_sha256: String,
    pub iterations: u64,
    pub seed: u64,
    pub information_sets: usize,
    pub terminal_evaluations: u64,
    pub zero_joint_chance_samples: u64,
    pub trained_root_combos: usize,
    pub minimum_root_average_visits: u64,
    pub root: PublicBeliefStrategy,
    pub validation: BlueprintValidation,
}

fn sampled_board(flop: &[u8], rng: &mut SplitMix64) -> [u8; 5] {
    let deck = (0..52u8)
        .filter(|card| !flop.contains(card))
        .collect::<Vec<_>>();
    let first = rng.index(49);
    let second = rng.index(48);
    let second = if second >= first { second + 1 } else { second };
    [flop[0], flop[1], flop[2], deck[first], deck[second]]
}

fn masked_ranges(ranges: &[Vec<f64>; 2], board: &[u8; 5]) -> [Vec<f64>; 2] {
    std::array::from_fn(|player| {
        all_combos()
            .iter()
            .map(|combo| {
                if combo.cards().iter().any(|card| board.contains(card)) {
                    0.0
                } else {
                    ranges[player][combo.key()]
                }
            })
            .collect()
    })
}

pub fn solve(mut config: SampledFlopConfig) -> Result<SampledFlopSolution, String> {
    if config.iterations < 2 || config.maximum_information_sets == 0 {
        return Err("sampled flop requires >=2 iterations and a positive node limit".to_owned());
    }
    if config.game.recall_mode != RecallMode::Trajectory
        || config.game.dcfr_schedule != DcfrSchedule::Fixed
        || config.game.opponent_hand_batch_size != 1
        || config.game.traverser_hand_batch_size != 1
        || config.game.opponent_checkdown_baseline
        || config.game.streetwise_opponent_estimator
    {
        return Err("sampled flop requires fixed trajectory DCFR without alternative samplers or hand batching".to_owned());
    }
    // These are local-training controls, not changes to the supplied game.
    config.game.iterations = config.iterations;
    config.game.seed = config.seed;
    config.game.max_information_sets = config.maximum_information_sets;
    config.game.averaging_delay = 0;
    config.game.traversal = BlueprintTraversal::PublicChanceSampling;
    config.game.integrate_terminal_actions = true;
    config.game.validate()?;
    let state = config
        .state
        .validate_street_and_normalize(&config.game, Street::Flop, 3)?;
    let root = state.game_state();
    let actions = root.legal_actions(&config.game);
    if actions.is_empty() || root.remaining(root.actor, &config.game) <= 0.0 {
        return Err("sampled flop requires a live root decision".to_owned());
    }
    let input = serde_json::to_vec(&(&config.game, &state)).map_err(|e| e.to_string())?;
    let input_sha256 = format!("{:x}", Sha256::digest(&input));
    let mut trainer = Trainer::fresh(config.game.clone());
    let mut chance_rng = SplitMix64::new(config.seed ^ 0x666c_6f70_2d70_6373);
    let mut zero_joint_chance_samples = 0;
    for iteration in 0..config.iterations {
        trainer.discounts.advance(iteration + 1);
        let traverser = iteration as usize % 2;
        let board = sampled_board(&state.board, &mut chance_rng);
        let masked = masked_ranges(&state.ranges, &board);
        // Never resample or normalize after blocking future cards. A proposal
        // board with zero compatible joint range contributes zero, not another
        // conditional draw with a different distribution.
        if joint_compatibility_mass(&masked) == 0.0 {
            zero_joint_chance_samples += 1;
        } else {
            let private_chance = masked[traverser].clone();
            // q(turn,river)=1/(49*48); a fixed compatible private pair has
            // p(turn,river)=1/(45*44). Multiply CFVs once by p/q through the
            // opponent range. Own private chance remains unscaled.
            let corrected = masked.map(|range| {
                range
                    .into_iter()
                    .map(|weight| weight * FUTURE_CHANCE_CORRECTION)
                    .collect()
            });
            let mut action_rng = SplitMix64::new(batch_iteration_seed(
                config.seed ^ 0x666c_6f70_2d61_6374,
                iteration + 1,
            ));
            let mut cache = range_vector::PublicInformationSetCache::new(board)?;
            trainer.public_chance_external_sampling(
                root.clone(),
                board,
                &corrected,
                &private_chance,
                traverser,
                &mut action_rng,
                &mut cache,
            )?;
        }
        trainer.completed_iterations += 1;
    }
    let width = actions.len();
    let mut probabilities = vec![0.0; COMBO_COUNT * width];
    let mut trained_root_combos = 0;
    let mut minimum_root_average_visits = u64::MAX;
    for combo in all_combos() {
        if state.ranges[root.actor][combo.key()] == 0.0 {
            continue;
        }
        let deal = deal_for_policy_combo_on_board(combo, root.actor, &state.board)?;
        let (key, _, _) = information_set(&root, &deal, &config.game);
        let node = trainer
            .nodes
            .get(&key)
            .filter(|node| node.average_visits > 0 && node.regret_updates > 0)
            .ok_or_else(|| {
                format!(
                    "sampled flop root combo {} has no trained average policy",
                    combo.key()
                )
            })?;
        if !node
            .action_labels
            .iter()
            .map(AsRef::as_ref)
            .eq(actions.iter().map(|action| action.label.as_str()))
        {
            return Err("sampled flop root legal actions differ from its trained row".to_owned());
        }
        for (action, probability) in node.average_strategy().into_iter().enumerate() {
            probabilities[combo.key() * width + action] = probability as f32;
        }
        trained_root_combos += 1;
        minimum_root_average_visits = minimum_root_average_visits.min(node.average_visits);
    }
    Ok(SampledFlopSolution {
        schema: "hu-fixed-flop-sampled-subgame-root-v1".to_owned(),
        input_sha256,
        iterations: config.iterations,
        seed: config.seed,
        information_sets: trainer.nodes.len(),
        terminal_evaluations: trainer.terminal_evaluations,
        zero_joint_chance_samples,
        trained_root_combos,
        minimum_root_average_visits,
        root: PublicBeliefStrategy {
            public_history: root.public_history,
            actor: root.actor,
            action_labels: actions.into_iter().map(|a| a.label).collect(),
            probabilities,
            action_values_bb: None,
        },
        validation: BlueprintValidation {
            status: "research_only".to_owned(),
            reasons: vec!["Sampled range-conditioned subgame root only; no opponent-CFV protection, full-game exploitability bound or action-EV grade. Not a complete continuation policy.".to_owned()],
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> SampledFlopConfig {
        let board = [0, 5, 10];
        let mut game = BlueprintConfig::default();
        game.effective_stack_bb = 6.0;
        game.hand_abstraction.distribution_samples = 2;
        game.hand_abstraction.equity_bins = 4;
        game.hand_abstraction.potential_bins = 2;
        SampledFlopConfig {
            game,
            state: PublicBeliefState::flop_start(
                board,
                1,
                [5.0, 5.0],
                std::array::from_fn(|_| uniform_range(&board)),
            ),
            iterations: 32,
            seed: 82001,
            maximum_information_sets: 300_000,
        }
    }

    #[test]
    fn future_chance_correction_preserves_exact_joint_mass_without_reconditioning() {
        let flop = [0, 5, 10];
        let first = [(Combo::new(51, 50), 0.3), (Combo::new(47, 46), 0.7)];
        let second = [(Combo::new(51, 49), 0.6), (Combo::new(43, 42), 0.4)];
        let mut ranges = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        for (combo, weight) in first {
            ranges[0][combo.key()] = weight;
        }
        for (combo, weight) in second {
            ranges[1][combo.key()] = weight;
        }
        let exact_joint: f64 = first
            .iter()
            .flat_map(|(a, wa)| {
                second.iter().map(move |(b, wb)| {
                    if a.cards().iter().any(|card| b.cards().contains(card)) {
                        0.0
                    } else {
                        wa * wb
                    }
                })
            })
            .sum();
        let mut integrated_joint = 0.0;
        let mut samples = 0;
        for turn in (0..52u8).filter(|card| !flop.contains(card)) {
            for river in (0..52u8).filter(|card| !flop.contains(card) && *card != turn) {
                let masked = masked_ranges(&ranges, &[flop[0], flop[1], flop[2], turn, river]);
                let sampled_joint: f64 = first
                    .iter()
                    .flat_map(|(a, _)| {
                        second.iter().map(|(b, _)| {
                            if a.cards().iter().any(|card| b.cards().contains(card)) {
                                0.0
                            } else {
                                masked[0][a.key()] * masked[1][b.key()]
                            }
                        })
                    })
                    .sum();
                integrated_joint += sampled_joint * FUTURE_CHANCE_CORRECTION;
                samples += 1;
            }
        }
        assert_eq!(samples, 49 * 48);
        assert!((integrated_joint / samples as f64 - exact_joint).abs() < 1e-12);
        // A fold's constant utility multiplies the same joint mass; this also
        // detects a second p/q factor or normalizing each masked range.
        assert!((5.0 * integrated_joint / samples as f64 - 5.0 * exact_joint).abs() < 1e-11);
    }

    #[test]
    fn board_sampler_preserves_visible_cards_and_is_deterministic() {
        let flop = [0, 5, 10];
        let mut first = SplitMix64::new(9);
        let mut second = SplitMix64::new(9);
        for _ in 0..1024 {
            let board = sampled_board(&flop, &mut first);
            assert_eq!(board, sampled_board(&flop, &mut second));
            assert_eq!(&board[..3], &flop);
            assert_eq!(board.iter().collect::<BTreeSet<_>>().len(), 5);
            assert!(board.iter().all(|card| *card < 52));
        }
    }

    #[test]
    fn root_policy_is_deterministic_complete_and_never_grades_evs() {
        let config = fixture();
        let first = solve(config.clone()).unwrap();
        let second = solve(config.clone()).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.trained_root_combos, 1176);
        assert_eq!(first.validation.status, "research_only");
        assert!(first.root.action_values_bb.is_none());
        assert_eq!(first.root.public_history, config.state.public_history);
        let width = first.root.action_labels.len();
        for combo in all_combos() {
            let sum: f32 = first.root.probabilities[combo.key() * width..(combo.key() + 1) * width]
                .iter()
                .sum();
            let blocked = combo
                .cards()
                .iter()
                .any(|card| config.state.board.contains(card));
            assert!((sum - if blocked { 0.0 } else { 1.0 }).abs() < 1e-6);
        }
    }

    #[test]
    fn invalid_inputs_caps_and_untrained_root_combos_fail_closed() {
        let mut config = fixture();
        config.iterations = 1;
        assert!(solve(config.clone()).is_err());
        config = fixture();
        config.state.board[1] = config.state.board[0];
        assert!(solve(config.clone()).is_err());
        config = fixture();
        config.state.ranges[0][1000] = f64::NAN;
        assert!(solve(config.clone()).is_err());
        config = fixture();
        config.state.ranges[0].fill(0.0);
        assert!(solve(config.clone()).is_err());
        config = fixture();
        config.game.streetwise_opponent_estimator = true;
        assert!(solve(config.clone()).is_err());
        config = fixture();
        config.maximum_information_sets = 1;
        assert!(solve(config.clone())
            .unwrap_err()
            .contains("information-set guard"));
        config = fixture();
        config.iterations = 2;
        assert!(solve(config)
            .unwrap_err()
            .contains("no trained average policy"));
    }

    #[test]
    fn forced_all_in_root_learns_a_call_without_a_value_network() {
        let mut config = fixture();
        let mut state = GameState::initial(&config.game);
        while state.street != Street::Flop {
            let action = state
                .legal_actions(&config.game)
                .into_iter()
                .find(|action| matches!(action.kind, ActionKind::Check | ActionKind::Call))
                .unwrap();
            state = state.apply(&action, &config.game);
        }
        let shove = state
            .legal_actions(&config.game)
            .into_iter()
            .find(|action| action.label.contains("all_in"))
            .unwrap();
        state = state.apply(&shove, &config.game);
        let mut ranges = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        let aces = Combo::new(51, 50);
        ranges[0][aces.key()] = 1.0;
        ranges[1][Combo::new(47, 46).key()] = 1.0;
        config.state = PublicBeliefState::from_game_state(vec![0, 5, 10], &state, ranges);
        config.iterations = 128;
        let solution = solve(config).unwrap();
        let call = solution
            .root
            .action_labels
            .iter()
            .position(|label| label.starts_with("call"))
            .unwrap();
        let width = solution.root.action_labels.len();
        assert!(solution.root.probabilities[aces.key() * width + call] > 0.9);
        assert!(solution.zero_joint_chance_samples > 0);
    }

    #[test]
    #[ignore = "explicit bounded cost probe, not an equilibrium qualification"]
    fn cost_probe() {
        let mut failures = Vec::new();
        for (pot, seed) in [(4.0, 83001), (10.0, 83002), (20.0, 83003)] {
            let board = [48, 21, 2];
            let config = SampledFlopConfig {
                game: BlueprintConfig::default(),
                state: PublicBeliefState::flop_start(
                    board,
                    1,
                    [pot / 2.0; 2],
                    std::array::from_fn(|_| uniform_range(&board)),
                ),
                iterations: 32,
                seed,
                maximum_information_sets: 2_000_000,
            };
            let start = std::time::Instant::now();
            match solve(config) {
                Ok(solution) => println!(
                    "{}",
                    serde_json::json!({
                        "potBb": pot, "seed": seed, "iterations": solution.iterations,
                        "seconds": start.elapsed().as_secs_f64(),
                        "informationSets": solution.information_sets,
                        "trainedRootCombos": solution.trained_root_combos,
                        "inputSha256": solution.input_sha256,
                        "minimumRootAverageVisits": solution.minimum_root_average_visits,
                    })
                ),
                Err(error) => {
                    println!(
                        "{}",
                        serde_json::json!({
                            "potBb": pot, "seed": seed, "iterations": 32,
                            "seconds": start.elapsed().as_secs_f64(), "error": error,
                        })
                    );
                    failures.push((pot, error));
                }
            }
        }
        assert!(failures.is_empty(), "incomplete cost probes: {failures:?}");
    }
}
