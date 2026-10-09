//! Complete turn/river own-payoff labels. Public river chance retains exactly
//! 44 outcomes per compatible private pair, including distinct tie payoffs.
use super::*;
use crate::cash_game::{Outcome, TerminalReason};

impl TurnRiverSolver {
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
}
