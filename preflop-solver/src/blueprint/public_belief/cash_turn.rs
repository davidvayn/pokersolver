//! Complete turn/river own-payoff labels. Public river chance retains exactly
//! 44 outcomes per compatible private pair, including distinct tie payoffs.
use super::*;
use crate::cash_game::{Outcome, TerminalReason};

impl TurnRiverSolver {
    /// Training CFVs need all card-legal deviation queries, not just hands
    /// reached by the current action mix. Construction support is discarded
    /// before training: actual normalized ranges, including zeros, are kept.
    /// This is isolated from ordinary frozen cash/Home policy evaluation.
    pub(super) fn new_cash_training(mut config: TurnRiverSolveConfig) -> Result<Self, String> {
        if config.game.cash_rules.is_none() {
            return Err("cash counterfactual training requires explicit rules".into());
        }
        let actual = config
            .state
            .validate_street_and_normalize(&config.game, Street::Turn, 4)?;
        config.state = actual.clone();
        config.state.ranges = std::array::from_fn(|_| uniform_range(&actual.board));
        let mut solver = Self::new(config)?;
        solver.config.state = actual;
        Ok(solver)
    }

    pub(super) fn cash_terminal_values(
        &self,
        state: &GameState,
        reaches: &[Vec<f64>; 2],
        river: Option<u8>,
    ) -> [Vec<f64>; 2] {
        let rules = self
            .config
            .game
            .cash_rules
            .as_ref()
            .expect("cash turn rules");
        if let Some(Terminal::Fold { winner }) = state.terminal {
            let payoffs = cash::cash_terminal_payoffs(
                rules,
                state.invested,
                TerminalReason::Fold,
                if winner == 0 {
                    Outcome::PlayerZero
                } else {
                    Outcome::PlayerOne
                },
                if river.is_some() { 5 } else { 4 },
            )
            .expect("legal cash fold");
            return std::array::from_fn(|p| {
                self.constant_terminal_values(&reaches[1 - p], payoffs[p])
            });
        }
        self.cash_showdown_values(rules, state.invested, reaches, river)
    }

    /// Training and best-response walks consume one player's own payoff.
    /// Do not compute the other player's marginals only to discard them.
    /// Cash payoffs are not zero sum: settle this seat explicitly, including
    /// refunds, rake and its distinct odd-unit tie award.
    pub(super) fn cash_player_terminal_values(
        &self,
        state: &GameState,
        reaches: &[Vec<f64>; 2],
        river: Option<u8>,
        player: usize,
    ) -> Vec<f64> {
        let rules = self
            .config
            .game
            .cash_rules
            .as_ref()
            .expect("cash turn rules");
        match state.terminal.as_ref().expect("terminal cash state") {
            Terminal::Fold { winner } => {
                let payoffs = cash::cash_terminal_payoffs(
                    rules,
                    state.invested,
                    TerminalReason::Fold,
                    if *winner == 0 {
                        Outcome::PlayerZero
                    } else {
                        Outcome::PlayerOne
                    },
                    if river.is_some() { 5 } else { 4 },
                )
                .expect("legal cash fold");
                self.constant_terminal_values(&reaches[1 - player], payoffs[player])
            }
            Terminal::Showdown => {
                let zero = cash::cash_terminal_payoffs(
                    rules,
                    state.invested,
                    TerminalReason::Showdown,
                    Outcome::PlayerZero,
                    5,
                )
                .expect("cash showdown");
                let one = cash::cash_terminal_payoffs(
                    rules,
                    state.invested,
                    TerminalReason::Showdown,
                    Outcome::PlayerOne,
                    5,
                )
                .expect("cash showdown");
                let tie = cash::cash_terminal_payoffs(
                    rules,
                    state.invested,
                    TerminalReason::Showdown,
                    Outcome::Split,
                    5,
                )
                .expect("cash tie");
                let (win, loss) = if player == 0 {
                    (zero[player], one[player])
                } else {
                    (one[player], zero[player])
                };
                if let Some(card) = river {
                    return self.river_player_showdown_values(
                        &reaches[1 - player],
                        card,
                        win,
                        loss,
                        tie[player],
                    );
                }
                let mut values = vec![0.0; COMBO_COUNT];
                for card in &self.river_cards {
                    let mut masked = reaches[1 - player].clone();
                    for combo in &self.river_blocked_combos[*card as usize] {
                        masked[*combo] = 0.0;
                    }
                    let child =
                        self.river_player_showdown_values(&masked, *card, win, loss, tie[player]);
                    let legal = self.legal_for(Some(*card), player);
                    for combo in 0..COMBO_COUNT {
                        if legal[combo] {
                            values[combo] += child[combo] / 44.0;
                        }
                    }
                }
                values
            }
        }
    }

    /// Geometry is independent of rules and commitments. Always settle with
    /// the caller's current rules, including rake, refunds and odd-unit ties.
    pub(super) fn cash_showdown_values(
        &self,
        rules: &crate::cash_game::CashGameRules,
        invested: [f64; 2],
        reaches: &[Vec<f64>; 2],
        river: Option<u8>,
    ) -> [Vec<f64>; 2] {
        let zero = cash::cash_terminal_payoffs(
            rules,
            invested,
            TerminalReason::Showdown,
            Outcome::PlayerZero,
            5,
        )
        .expect("cash showdown");
        let one = cash::cash_terminal_payoffs(
            rules,
            invested,
            TerminalReason::Showdown,
            Outcome::PlayerOne,
            5,
        )
        .expect("cash showdown");
        let tie = cash::cash_terminal_payoffs(
            rules,
            invested,
            TerminalReason::Showdown,
            Outcome::Split,
            5,
        )
        .expect("cash tie");
        let on_river = |card: u8, masked: &[Vec<f64>; 2]| {
            [
                self.river_player_showdown_values(&masked[1], card, zero[0], one[0], tie[0]),
                self.river_player_showdown_values(&masked[0], card, one[1], zero[1], tie[1]),
            ]
        };
        if let Some(card) = river {
            return on_river(card, reaches);
        }
        let mut values = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        for card in &self.river_cards {
            let mut masked = reaches.clone();
            for player in 0..2 {
                for combo in &self.river_blocked_combos[*card as usize] {
                    masked[player][*combo] = 0.0;
                }
            }
            let child = on_river(*card, &masked);
            self.accumulate_compatible_river_child(&mut values, &child, *card);
        }
        values
    }

    pub(super) fn profile_house_rake(
        &self,
        state: GameState,
        reaches: [Vec<f64>; 2],
        river: Option<u8>,
    ) -> f64 {
        if state.terminal.is_some() {
            let rules = self
                .config
                .game
                .cash_rules
                .as_ref()
                .expect("cash turn rules");
            let gross = 2 * cash::cash_units(state.invested[0].min(state.invested[1]), rules)
                .expect("cash commitments");
            return rules.rake_units(gross, true).expect("cash rake") as f64
                / rules.units_per_bb as f64
                * joint_compatibility_mass(&reaches);
        }
        if state.street == Street::River && river.is_none() {
            return self
                .river_cards
                .iter()
                .map(|card| {
                    let mut masked = reaches.clone();
                    for player in 0..2 {
                        for combo in &self.river_blocked_combos[*card as usize] {
                            masked[player][*combo] = 0.0;
                        }
                    }
                    self.profile_house_rake(state.clone(), masked, Some(*card)) / 44.0
                })
                .sum();
        }
        let actions = state.legal_actions(&self.config.game);
        let node = self
            .nodes
            .get(&Self::node_key(&state, river))
            .expect("trained turn node");
        let strategy = node.average_strategy(self.legal_for(river, state.actor));
        actions
            .iter()
            .enumerate()
            .map(|(index, action)| {
                let mut child = reaches.clone();
                for combo in 0..COMBO_COUNT {
                    child[state.actor][combo] *= strategy[combo * actions.len() + index];
                }
                self.profile_house_rake(state.apply(action, &self.config.game), child, river)
            })
            .sum()
    }

    pub(super) fn cash_metrics(
        &self,
        profile: [f64; 2],
        best: [f64; 2],
        joint: f64,
    ) -> Option<CashRiverMetrics> {
        self.config.game.cash_rules.as_ref().map(|rules| {
            let rake = self.profile_house_rake(
                self.config.state.game_state(),
                self.config.state.ranges.clone(),
                None,
            ) / joint;
            let gains = std::array::from_fn::<_, 2, _>(|p| (best[p] - profile[p]).max(0.0));
            CashRiverMetrics {
                rules_sha256: rules.sha256().expect("cash identity"),
                unilateral_gain_bb: gains,
                nash_conv_bb_per_hand: gains[0] + gains[1],
                maximum_deviation_gain_bb_per_hand: gains[0].max(gains[1]),
                expected_house_rake_bb: rake,
                conservation_residual_bb: (profile[0] + profile[1] + rake).abs(),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> TurnRiverSolveConfig {
        let mut game = BlueprintConfig::default();
        game.small_blind_bb = 0.4;
        game.effective_stack_bb = 2.0;
        game.iterations = 2;
        game.averaging_delay = 0;
        game.cash_rules = Some(crate::cash_game::study_rules("nl25").unwrap());
        let board = [8, 13, 22, 31];
        TurnRiverSolveConfig {
            game,
            state: PublicBeliefState::turn_start(
                board,
                1,
                [1.0, 1.0],
                std::array::from_fn(|_| uniform_range(&board)),
            ),
            iterations: 2,
            averaging_delay: 0,
            river_refinement_iterations: 0,
            regret_matching_plus: false,
        }
    }

    #[test]
    fn cash_traverser_terminal_evaluates_only_requested_payoff() {
        let input = config();
        let solver = TurnRiverSolver::new(input.clone()).unwrap();
        let mut state = input.state.game_state();
        state.terminal = Some(Terminal::Fold { winner: 0 });
        for player in 0..2 {
            solver
                .terminal_value_kernel_evaluations
                .store(0, std::sync::atomic::Ordering::Relaxed);
            let values = solver.player_terminal_values(&state, &input.state.ranges, None, player);
            assert!(values[player].iter().any(|value| *value != 0.0));
            assert!(values[1 - player].iter().all(|value| *value == 0.0));
            assert_eq!(
                solver
                    .terminal_value_kernel_evaluations
                    .load(std::sync::atomic::Ordering::Relaxed),
                1,
                "training and best response need only the requested player's terminal payoff"
            );
        }
        state.terminal = Some(Terminal::Showdown);
        for river in [None, Some(0)] {
            for player in 0..2 {
                solver
                    .terminal_value_kernel_evaluations
                    .store(0, std::sync::atomic::Ordering::Relaxed);
                let values =
                    solver.player_terminal_values(&state, &input.state.ranges, river, player);
                assert!(values[1 - player].iter().all(|value| *value == 0.0));
                assert_eq!(
                    solver
                        .terminal_value_kernel_evaluations
                        .load(std::sync::atomic::Ordering::Relaxed),
                    if river.is_some() { 1 } else { 48 }
                );
            }
        }
    }

    #[test]
    fn cash_requested_terminal_payoff_is_bit_exact_for_both_seats() {
        for rake_off in [false, true] {
            for sparse in [false, true] {
                let mut input = config();
                if rake_off {
                    input.game.cash_rules =
                        Some(crate::cash_game::study_rules("nl25-rake-off-control").unwrap());
                }
                if sparse {
                    for (player, range) in input.state.ranges.iter_mut().enumerate() {
                        for (combo, weight) in range.iter_mut().enumerate() {
                            if combo % (player + 3) == 0 {
                                *weight = 0.0;
                            }
                        }
                    }
                }
                let solver = TurnRiverSolver::new_cash_training(input.clone()).unwrap();
                let mut state = input.state.game_state();
                // Different contributions test uncalled refunds; 1.40 per seat
                // yields an odd net pot after cent rounding and asymmetric ties.
                for invested in [[1.0, 1.0], [1.4, 1.4], [2.0, 1.0], [1.0, 2.0]] {
                    state.invested = invested;
                    for terminal in [
                        Terminal::Fold { winner: 0 },
                        Terminal::Fold { winner: 1 },
                        Terminal::Showdown,
                    ] {
                        if let Terminal::Fold { winner } = &terminal {
                            if invested[*winner] < invested[1 - *winner] {
                                continue; // A fold winner must own any unmatched wager.
                            }
                        }
                        state.terminal = Some(terminal);
                        for river in [None, Some(0), Some(49)] {
                            let mut reaches = solver.config.state.ranges.clone();
                            if let Some(card) = river {
                                for range in &mut reaches {
                                    for combo in &solver.river_blocked_combos[card as usize] {
                                        range[*combo] = 0.0;
                                    }
                                }
                            }
                            let reference = solver.cash_terminal_values(&state, &reaches, river);
                            for player in 0..2 {
                                let actual =
                                    solver.player_terminal_values(&state, &reaches, river, player);
                                assert_eq!(
                                    actual[player]
                                        .iter()
                                        .map(|v| v.to_bits())
                                        .collect::<Vec<_>>(),
                                    reference[player]
                                        .iter()
                                        .map(|v| v.to_bits())
                                        .collect::<Vec<_>>()
                                );
                                assert!(actual[1 - player].iter().all(|value| *value == 0.0));
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn cash_traverser_only_work_preserves_training_and_best_response_bits() {
        for plus in [false, true] {
            let mut input = config();
            input.regret_matching_plus = plus;
            for (player, range) in input.state.ranges.iter_mut().enumerate() {
                for (combo, weight) in range.iter_mut().enumerate() {
                    if combo % (player + 3) == 0 {
                        *weight = 0.0;
                    } else {
                        *weight *= (combo % 7 + 1) as f64;
                    }
                }
            }
            let mut fast = TurnRiverSolver::new_cash_training(input.clone()).unwrap();
            let mut reference = TurnRiverSolver::new_cash_training(input).unwrap();
            reference.reference_both_value_players = true;
            fast.train();
            reference.train();
            assert_eq!(fast.nodes.len(), reference.nodes.len());
            for (key, node) in &fast.nodes {
                let other = &reference.nodes[key];
                assert_eq!(node.regrets, other.regrets);
                assert_eq!(node.strategy_sum, other.strategy_sum);
                assert_eq!(
                    node.last_regret_discount_round,
                    other.last_regret_discount_round
                );
                assert_eq!(
                    node.last_strategy_discount_round,
                    other.last_strategy_discount_round
                );
            }
            assert_eq!(
                serde_json::to_vec(&fast.policy_strategies()).unwrap(),
                serde_json::to_vec(&reference.policy_strategies()).unwrap()
            );
            for player in 0..2 {
                assert_eq!(
                    fast.exact_best_response_conditional_values(player),
                    reference.exact_best_response_conditional_values(player)
                );
            }
            assert_eq!(
                serde_json::to_vec(&fast.finish()).unwrap(),
                serde_json::to_vec(&reference.finish()).unwrap()
            );
        }
    }

    #[test]
    fn turn_all_in_cfv_matches_independent_44_river_own_settlements() {
        let config = config();
        let solver = TurnRiverSolver::new(config.clone()).unwrap();
        let mut state = config.state.game_state();
        state.terminal = Some(Terminal::Showdown);
        state.invested = [2.0, 2.0];
        let holes = [[48, 49], [4, 5]];
        let mut reaches = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        for p in 0..2 {
            reaches[p][Combo::new(holes[p][0], holes[p][1]).key()] = 1.0;
        }
        let values = solver.cash_terminal_values(&state, &reaches, None);
        let mut expected = [0.0; 2];
        let mut count = 0;
        for card in 0..52u8 {
            if config.state.board.contains(&card)
                || holes.iter().flatten().any(|hole| *hole == card)
            {
                continue;
            }
            let mut board = config.state.board.clone();
            board.push(card);
            let outcome = match showdown_result(&holes, &board) {
                1.0 => Outcome::PlayerZero,
                0.5 => Outcome::Split,
                _ => Outcome::PlayerOne,
            };
            let payoffs = cash::cash_terminal_payoffs(
                config.game.cash_rules.as_ref().unwrap(),
                [2.0, 2.0],
                TerminalReason::Showdown,
                outcome,
                5,
            )
            .unwrap();
            for p in 0..2 {
                expected[p] += payoffs[p] / 44.0;
            }
            count += 1;
        }
        assert_eq!(count, 44);
        for p in 0..2 {
            assert!(
                (values[p][Combo::new(holes[p][0], holes[p][1]).key()] - expected[p]).abs() < 1e-10
            );
        }
    }

    #[test]
    fn complete_cash_turn_river_checks_independent_house_ledger() {
        let mut solver = TurnRiverSolver::new(config()).unwrap();
        solver.train();
        let solution = solver.finish();
        let cash = solution.metrics.cash.as_ref().unwrap();
        assert!(cash.expected_house_rake_bb > 0.0 && cash.conservation_residual_bb < 1e-8);
        assert_eq!(solution.metrics.exact_river_cards, 48);
        assert!(
            (solution.metrics.exact_abstract_exploitability_bb_per_hand
                - cash.nash_conv_bb_per_hand)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn cash_training_cfv_retains_zero_own_reach_royal_flush_deviation() {
        let mut input = config();
        // Tc Jc Qc Kc: Ac2c is already unbeatable, but its actual own reach is
        // zero. It still needs a deviation target for trunk regret updates.
        input.state.board = vec![32, 36, 40, 44];
        input.state.ranges = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        let nuts = Combo::new(48, 0).key();
        let own = Combo::new(49, 45).key();
        let opponent = Combo::new(41, 37).key();
        input.state.ranges[0][own] = 1.0;
        input.state.ranges[1][opponent] = 1.0;
        let actual = input.state.ranges.clone();
        let minimum = cash::cash_terminal_payoffs(
            input.game.cash_rules.as_ref().unwrap(),
            [1.0, 1.0],
            TerminalReason::Showdown,
            Outcome::PlayerZero,
            5,
        )
        .unwrap()[0];
        let result = solve_turn_river_continuation_values(input.clone()).unwrap();
        assert!(
            result.counterfactual_values_bb[0][nuts] as f64 >= minimum - 1e-6,
            "a zero-own-reach royal query must retain its guaranteed cash payoff; got {}",
            result.counterfactual_values_bb[0][nuts]
        );
        assert_eq!(input.state.ranges, actual);
        assert_eq!(input.state.ranges[0][nuts], 0.0);
        assert_eq!(result.schema, "hu-cash-turn-river-continuation-values-v2");
        assert_eq!(
            result.training_target_semantics.as_deref(),
            Some("profile-positive-own-reach-cbr-zero-own-reach-v1")
        );
        let profile = result.profile_counterfactual_values_bb.unwrap();
        let best = result.best_response_counterfactual_values_bb.unwrap();
        assert_eq!(result.counterfactual_values_bb[0][nuts], best[0][nuts]);
        assert_eq!(result.counterfactual_values_bb[0][own], profile[0][own]);
        assert_eq!(
            result.counterfactual_values_bb[1][opponent],
            profile[1][opponent]
        );
        assert_eq!(result.completed_zero_own_reach, Some([1127, 1127]));
        let blocked = Combo::new(32, 1).key();
        for p in 0..2 {
            assert_eq!(result.counterfactual_values_bb[p][blocked], 0.0);
        }
        assert!(result.metrics.cash.unwrap().conservation_residual_bb < 1e-8);
    }

    #[test]
    fn cash_training_support_does_not_change_actual_beliefs_or_reached_policy() {
        let mut input = config();
        let own = Combo::new(48, 49).key();
        let opponent = Combo::new(4, 5).key();
        input.state.ranges = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        input.state.ranges[0][own] = 1.0;
        input.state.ranges[1][opponent] = 1.0;
        let mut legacy = TurnRiverSolver::new(input.clone()).unwrap();
        let mut training = TurnRiverSolver::new_cash_training(input.clone()).unwrap();
        assert_eq!(training.config.state.ranges, input.state.ranges);
        assert_eq!(
            training.legal[0].iter().filter(|legal| **legal).count(),
            1128
        );
        legacy.train();
        training.train();
        let root = input.state.game_state();
        let reaches = input.state.ranges;
        let old = legacy.profile_walk(root.clone(), reaches.clone(), None, None, None, true);
        let new = training.profile_walk(root, reaches, None, None, None, true);
        assert!((old[0][own] - new[0][own]).abs() < 1e-10);
        assert!((old[1][opponent] - new[1][opponent]).abs() < 1e-10);
        for (history, node) in &legacy.nodes {
            let counterpart = training.nodes.get(history).unwrap();
            let river = TurnRiverSolver::river_from_key(history);
            let key = if node.actor == 0 { own } else { opponent };
            let old = node.average_strategy(legacy.legal_for(river, node.actor));
            let new = counterpart.average_strategy(training.legal_for(river, node.actor));
            let count = node.action_labels.len();
            assert_eq!(
                &old[key * count..(key + 1) * count],
                &new[key * count..(key + 1) * count]
            );
        }
    }
}
