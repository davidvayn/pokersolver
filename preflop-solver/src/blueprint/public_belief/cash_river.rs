//! Exact own-payoff river labels; no learned continuation or safety projection.
use super::*;
use crate::cash_game::{Outcome, TerminalReason};

impl RiverSolver {
    pub(super) fn cash_terminal_values(
        &self,
        state: &GameState,
        reaches: &[Vec<f64>; 2],
    ) -> [Vec<f64>; 2] {
        let rules = self
            .config
            .game
            .cash_rules
            .as_ref()
            .expect("cash river rules");
        match state.terminal.as_ref().expect("cash terminal") {
            Terminal::Fold { winner } => {
                let values = cash::cash_terminal_payoffs(
                    rules,
                    state.invested,
                    TerminalReason::Fold,
                    if *winner == 0 {
                        Outcome::PlayerZero
                    } else {
                        Outcome::PlayerOne
                    },
                    5,
                )
                .expect("legal cash fold");
                std::array::from_fn(|p| self.constant_terminal_values(&reaches[1 - p], values[p]))
            }
            Terminal::Showdown => {
                let zero_wins = cash::cash_terminal_payoffs(
                    rules,
                    state.invested,
                    TerminalReason::Showdown,
                    Outcome::PlayerZero,
                    5,
                )
                .expect("legal cash showdown");
                let one_wins = cash::cash_terminal_payoffs(
                    rules,
                    state.invested,
                    TerminalReason::Showdown,
                    Outcome::PlayerOne,
                    5,
                )
                .expect("legal cash showdown");
                let split = cash::cash_terminal_payoffs(
                    rules,
                    state.invested,
                    TerminalReason::Showdown,
                    Outcome::Split,
                    5,
                )
                .expect("legal cash split");
                [
                    self.showdown_terminal_values(
                        0,
                        &reaches[1],
                        zero_wins[0],
                        one_wins[0],
                        split[0],
                    ),
                    self.showdown_terminal_values(
                        1,
                        &reaches[0],
                        one_wins[1],
                        zero_wins[1],
                        split[1],
                    ),
                ]
            }
        }
    }

    /// Independent house ledger traversal under the SAME frozen joint profile.
    /// This is not computed by negating the two returned network/profile values.
    pub(super) fn profile_house_rake(&self, state: GameState, reaches: [Vec<f64>; 2]) -> f64 {
        if state.terminal.is_some() {
            let rules = self
                .config
                .game
                .cash_rules
                .as_ref()
                .expect("cash river rules");
            let gross = 2 * cash::cash_units(state.invested[0].min(state.invested[1]), rules)
                .expect("cent aligned terminal");
            let rake = rules.rake_units(gross, true).expect("pinned rake") as f64
                / rules.units_per_bb as f64;
            return rake * joint_compatibility_mass(&reaches);
        }
        let actions = state.legal_actions(&self.config.game);
        let node = self
            .nodes
            .get(&state.public_history)
            .expect("trained river node");
        let strategy = node.average_strategy(&self.legal[state.actor]);
        actions
            .iter()
            .enumerate()
            .map(|(action_index, action)| {
                let mut child_reaches = reaches.clone();
                for combo in 0..COMBO_COUNT {
                    child_reaches[state.actor][combo] *=
                        strategy[combo * actions.len() + action_index];
                }
                self.profile_house_rake(state.apply(action, &self.config.game), child_reaches)
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn river_config() -> RiverSolveConfig {
        let registry: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../data/practice/cash-game-rules.json"
        ))
        .unwrap();
        let mut game = BlueprintConfig::default();
        game.effective_stack_bb = 4.0;
        game.iterations = 10;
        game.averaging_delay = 0;
        game.action_abstraction.turn_river_bet_pot_fractions = vec![1.0];
        game.small_blind_bb = 0.4;
        game.cash_rules = Some(serde_json::from_value(registry["nl25"].clone()).unwrap());
        RiverSolveConfig {
            game,
            state: PublicBeliefState::uniform_river_start([0, 5, 10, 15, 20], 1, [1.0, 1.0]),
            iterations: 16,
            averaging_delay: 0,
        }
    }

    #[test]
    fn complete_raked_river_has_own_cfv_targets_and_exact_net_response_gains() {
        let config = river_config();
        let first = solve_river(config.clone()).unwrap();
        assert_eq!(first, solve_river(config.clone()).unwrap());
        let metrics = &first.metrics;
        let cash = metrics.cash.as_ref().unwrap();
        assert_eq!(first.schema, "hu-public-belief-cash-river-v2");
        assert!(cash.expected_house_rake_bb > 0.0);
        assert!(cash.conservation_residual_bb < 1e-8);
        assert!(
            (metrics.profile_value_p0_bb
                + metrics.profile_value_p1_bb
                + cash.expected_house_rake_bb)
                .abs()
                < 1e-8
        );
        assert!(
            (cash.unilateral_gain_bb.iter().sum::<f64>() - cash.nash_conv_bb_per_hand).abs()
                < 1e-10
        );
        assert_eq!(
            metrics.exact_abstract_exploitability_bb_per_hand,
            cash.nash_conv_bb_per_hand
        );
        assert!(first
            .counterfactual_values_bb
            .iter()
            .flatten()
            .all(|value| value.is_finite()));
        assert!(solve_river_safe_policy(config, &first.strategies, 1).is_err());
    }

    #[test]
    fn raked_river_rejects_subcent_states_and_unconverted_early_streets() {
        let mut config = river_config();
        config.state.invested_bb[0] = 1.01;
        assert!(solve_river(config).is_err());
        let config = river_config();
        let state = PublicBeliefState::flop_start(
            [0, 5, 10],
            1,
            [1.0, 1.0],
            std::array::from_fn(|_| uniform_range(&[0, 5, 10])),
        );
        assert!(state
            .validate_street_and_normalize(&config.game, Street::Flop, 3)
            .is_err());
    }
}
