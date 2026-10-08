//! Rule-aware own-payoff adapter. The legacy Home path retains its exact math.
use super::*;
use crate::cash_game::{settle_cash_hand, CashTerminal, Outcome, TerminalReason};

impl BlueprintConfig {
    pub(super) fn validate_cash_rules(&self) -> Result<(), String> {
        if let Some(rules) = &self.cash_rules {
            rules.validate()?;
            let scale = rules.units_per_bb as f64;
            if (self.small_blind_bb * scale - rules.blinds_units[0] as f64).abs() > 1e-8
                || (self.big_blind_bb - 1.0).abs() > 1e-8
            {
                return Err("blinds differ from the pinned cash rules".into());
            }
            cash_units(self.effective_stack_bb, rules)?;
            if self.effective_stack_bb * scale > 4_503_599_627_370_495.0 {
                return Err("two starting stacks exceed safe money units".into());
            }
        }
        Ok(())
    }

    /// Existing neural artifacts and resolver guarantees have no cash identity.
    /// Reject at entry rather than silently applying Home values to another game.
    pub(super) fn require_legacy_home(&self) -> Result<(), String> {
        if self.cash_rules.is_some() {
            return Err("this legacy continuation/policy path does not support explicit cash rules; rake-aware artifacts are required".into());
        }
        Ok(())
    }

    pub(super) fn quantize_bet_bb(&self, amount: f64) -> f64 {
        let Some(rules) = &self.cash_rules else {
            return quantize(amount, 0.001);
        };
        let scaled = amount * rules.units_per_bb as f64;
        let whole = scaled.floor();
        let fraction = scaled - whole;
        // Pot-fraction calculations start as f64. Treat machine noise around an
        // exact half-cent as the same rational tie in both language adapters.
        let upward = if (fraction - 0.5).abs() <= 1e-8 {
            rules.bet_rounding == crate::cash_game::MoneyRounding::HalfUp || whole as u64 % 2 == 1
        } else {
            fraction > 0.5
        };
        (whole + f64::from(upward)) / rules.units_per_bb as f64
    }
}

pub(super) fn cash_units(amount_bb: f64, rules: &CashGameRules) -> Result<u64, String> {
    let scaled = amount_bb * rules.units_per_bb as f64;
    if !scaled.is_finite()
        || scaled < 0.0
        || scaled > 9_007_199_254_740_991.0
        || (scaled - scaled.round()).abs() > 1e-8
    {
        return Err("cash amount must be finite, safe, and aligned to money units".into());
    }
    Ok(scaled.round() as u64)
}

pub(super) fn cash_terminal_payoffs(
    rules: &CashGameRules,
    invested: [f64; 2],
    reason: TerminalReason,
    outcome: Outcome,
    board_cards_dealt: u8,
) -> Result<[f64; 2], String> {
    let settlement = settle_cash_hand(
        rules,
        &CashTerminal {
            committed_units: [
                cash_units(invested[0], rules)?,
                cash_units(invested[1], rules)?,
            ],
            // Native player zero is always BTN/SB; runtime seat rotation is external.
            button: 0,
            reason,
            outcome,
            board_cards_dealt,
        },
    )?;
    Ok(settlement
        .net_payoff_units
        .map(|value| value as f64 / rules.units_per_bb as f64))
}

impl GameState {
    pub(super) fn own_utilities(&self, deal: &Deal, config: &BlueprintConfig) -> [f64; 2] {
        let Some(rules) = &config.cash_rules else {
            let value = self.utility_p0(deal, config);
            return [value, -value];
        };
        match self.terminal.as_ref().expect("terminal own utilities") {
            Terminal::Fold { winner } => cash_terminal_payoffs(
                rules,
                self.invested,
                TerminalReason::Fold,
                if *winner == 0 {
                    Outcome::PlayerZero
                } else {
                    Outcome::PlayerOne
                },
                self.street.board_len() as u8,
            )
            .expect("validated cash terminal"),
            Terminal::Showdown => {
                let weights = conditional_showdown_outcomes(deal, self.street, config);
                let mut values = [0.0; 2];
                for (weight, outcome) in weights.into_iter().zip([
                    Outcome::PlayerZero,
                    Outcome::Split,
                    Outcome::PlayerOne,
                ]) {
                    let payoffs = cash_terminal_payoffs(
                        rules,
                        self.invested,
                        TerminalReason::Showdown,
                        outcome,
                        5,
                    )
                    .expect("validated cash showdown");
                    for player in 0..2 {
                        values[player] += weight * payoffs[player];
                    }
                }
                values
            }
        }
    }

    pub(super) fn complete_runout_utilities(
        &self,
        deal: &Deal,
        config: &BlueprintConfig,
    ) -> [f64; 2] {
        if let Some(rules) = &config.cash_rules {
            let (reason, outcome, board_len) =
                match self.terminal.as_ref().expect("terminal utility") {
                    Terminal::Fold { winner } => (
                        TerminalReason::Fold,
                        if *winner == 0 {
                            Outcome::PlayerZero
                        } else {
                            Outcome::PlayerOne
                        },
                        self.street.board_len() as u8,
                    ),
                    Terminal::Showdown => (
                        TerminalReason::Showdown,
                        outcome_from_equity(showdown_result(&deal.holes, &deal.board)),
                        5,
                    ),
                };
            return cash_terminal_payoffs(rules, self.invested, reason, outcome, board_len)
                .expect("validated cash terminal");
        }
        let value = match self.terminal.as_ref().expect("terminal utility") {
            Terminal::Fold { winner: 0 } => self.invested[1],
            Terminal::Fold { .. } => -self.invested[0],
            Terminal::Showdown => {
                let equity = showdown_result(&deal.holes, &deal.board);
                equity * self.invested[1] - (1.0 - equity) * self.invested[0]
            }
        };
        [value, -value]
    }
}

fn outcome_from_equity(equity: f64) -> Outcome {
    if equity == 1.0 {
        Outcome::PlayerZero
    } else if equity == 0.5 {
        Outcome::Split
    } else {
        Outcome::PlayerOne
    }
}

fn conditional_showdown_outcomes(
    deal: &Deal,
    street: Street,
    config: &BlueprintConfig,
) -> [f64; 3] {
    let board_len = street.board_len();
    if let Some(weights) = deal.showdown_outcome_cache.borrow().get(&board_len) {
        return *weights;
    }
    let visible = &deal.board[..board_len];
    let mut counts = [0.0; 3];
    let mut observe = |board: &[u8]| {
        counts[match outcome_from_equity(showdown_result(&deal.holes, board)) {
            Outcome::PlayerZero => 0,
            Outcome::Split => 1,
            Outcome::PlayerOne => 2,
        }] += 1.0;
    };
    if street == Street::River {
        observe(visible);
    } else {
        let mut known = [false; 52];
        for card in deal.holes.iter().flatten().chain(visible) {
            known[*card as usize] = true;
        }
        let mut available = (0..52u8)
            .filter(|card| !known[*card as usize])
            .collect::<Vec<_>>();
        if street == Street::Turn && config.showdown_evaluation.exact_turn_rivers {
            for river in available {
                let mut board = visible.to_vec();
                board.push(river);
                observe(&board);
            }
        } else {
            let samples = if street == Street::Preflop {
                config.showdown_evaluation.preflop_runout_samples
            } else {
                config.showdown_evaluation.flop_runout_samples
            };
            let missing = 5 - board_len;
            let mut rng = SplitMix64::new(showdown_seed(&deal.holes, visible));
            for _ in 0..samples {
                for index in 0..missing {
                    let swap = index + rng.index(available.len() - index);
                    available.swap(index, swap);
                }
                let mut board = visible.to_vec();
                board.extend_from_slice(&available[..missing]);
                observe(&board);
            }
        }
    }
    let total = counts.iter().sum::<f64>();
    let weights = counts.map(|count| count / total);
    deal.showdown_outcome_cache
        .borrow_mut()
        .insert(board_len, weights);
    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    fn online_config() -> BlueprintConfig {
        let rules: serde_json::Value =
            serde_json::from_str(include_str!("../../../data/practice/cash-game-rules.json"))
                .unwrap();
        BlueprintConfig {
            cash_rules: Some(serde_json::from_value(rules["nl25"].clone()).unwrap()),
            small_blind_bb: 0.4,
            effective_stack_bb: 20.0,
            ..super::super::tests::tiny_config()
        }
    }

    #[test]
    fn explicit_rules_validate_blinds_stacks_and_legacy_isolation() {
        let mut config = online_config();
        config.validate().unwrap();
        assert!(config.require_legacy_home().is_err());
        config.small_blind_bb = 0.5;
        assert!(config.validate().is_err());
        config.small_blind_bb = 0.4;
        config.effective_stack_bb = 20.01;
        assert!(config.validate().is_err());
        config.cash_rules = None;
        config.require_legacy_home().unwrap();
        assert!(serde_json::to_value(&config)
            .unwrap()
            .get("cash_rules")
            .is_none());
    }

    #[test]
    fn exact_cash_utilities_match_shared_settlements_including_ties() {
        let mut config = online_config();
        let fixtures: serde_json::Value = serde_json::from_str(include_str!(
            "../../../data/practice/cash-settlement-fixtures.json"
        ))
        .unwrap();
        for fixture in fixtures["settlements"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|f| f["rules"] == "nl25")
        {
            let terminal: CashTerminal =
                serde_json::from_value(fixture["terminal"].clone()).unwrap();
            // Fixtures can use either button. Native zero is BTN/SB, so swap
            // commitments and expected values for a button-one fixture.
            let mut terminal = terminal;
            if terminal.button == 1 {
                terminal.committed_units.swap(0, 1);
                terminal.outcome = match terminal.outcome {
                    Outcome::PlayerZero => Outcome::PlayerOne,
                    Outcome::PlayerOne => Outcome::PlayerZero,
                    Outcome::Split => Outcome::Split,
                };
                terminal.button = 0;
            }
            let expected =
                settle_cash_hand(config.cash_rules.as_ref().unwrap(), &terminal).unwrap();
            let outcome_board = match terminal.outcome {
                Outcome::PlayerZero => ([[48, 49], [4, 5]], [8, 13, 22, 31, 40]),
                Outcome::PlayerOne => ([[4, 5], [48, 49]], [8, 13, 22, 31, 40]),
                Outcome::Split => ([[0, 5], [8, 13]], [32, 36, 40, 44, 48]),
            };
            let deal = Deal::from_cards(outcome_board.0, outcome_board.1);
            let mut state = GameState::initial(&config);
            state.invested = terminal.committed_units.map(|amount| amount as f64 / 25.0);
            state.street = match terminal.board_cards_dealt {
                0 => Street::Preflop,
                3 => Street::Flop,
                4 => Street::Turn,
                _ => Street::River,
            };
            state.terminal = Some(match terminal.reason {
                TerminalReason::Fold => Terminal::Fold {
                    winner: usize::from(terminal.outcome == Outcome::PlayerOne),
                },
                TerminalReason::Showdown => Terminal::Showdown,
            });
            let values = state.complete_runout_utilities(&deal, &config);
            for player in 0..2 {
                assert!(
                    (values[player] - expected.net_payoff_units[player] as f64 / 25.0).abs()
                        < 1e-10,
                    "{fixture:?}"
                );
            }
            if terminal.reason == TerminalReason::Showdown {
                state.street = Street::Preflop;
                let values = state.own_utilities(&deal, &config);
                assert!((values[0] + values[1] + expected.rake_units as f64 / 25.0).abs() < 1e-10);
            }
        }
        config.cash_rules = None;
        let state = GameState {
            invested: [20.0, 20.0],
            terminal: Some(Terminal::Showdown),
            ..GameState::initial(&config)
        };
        let deal = Deal::from_cards([[48, 49], [4, 5]], [8, 13, 22, 31, 40]);
        let values = state.own_utilities(&deal, &config);
        assert_eq!(values[1], -values[0]);
    }

    #[test]
    fn full_raked_tree_stays_unit_aligned_and_conserves_payoffs() {
        let config = online_config();
        let deal = Deal::from_cards([[48, 49], [4, 5]], [8, 13, 22, 31, 40]);
        fn walk(state: GameState, config: &BlueprintConfig, deal: &Deal, visited: &mut u64) {
            *visited += 1;
            for amount in state.invested.into_iter().chain(state.street_invested) {
                cash_units(amount, config.cash_rules.as_ref().unwrap()).unwrap();
            }
            if state.terminal.is_some() {
                let values = state.complete_runout_utilities(deal, config);
                let rules = config.cash_rules.as_ref().unwrap();
                let rake = rules
                    .rake_units(
                        2 * cash_units(state.invested[0].min(state.invested[1]), rules).unwrap(),
                        matches!(state.terminal, Some(Terminal::Showdown))
                            || state.street != Street::Preflop,
                    )
                    .unwrap() as f64
                    / 25.0;
                assert!((values[0] + values[1] + rake).abs() < 1e-10);
                return;
            }
            let actions = state.legal_actions(config);
            let targets = actions
                .iter()
                .filter_map(|a| {
                    if let ActionKind::RaiseTo(to) = a.kind {
                        Some(cash_units(to, config.cash_rules.as_ref().unwrap()).unwrap())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            assert_eq!(targets.len(), targets.iter().collect::<BTreeSet<_>>().len());
            for action in actions {
                walk(state.apply(&action, config), config, deal, visited);
            }
        }
        let mut visited = 0;
        walk(GameState::initial(&config), &config, &deal, &mut visited);
        assert!(visited > 100);
        assert_eq!(config.quantize_bet_bb(2.5), 2.48);
        assert_eq!(config.quantize_bet_bb(2.54), 2.56);
    }

    #[test]
    fn raked_checkpoint_resume_and_frozen_export_have_rules_identity() {
        let config = online_config();
        let mut trainer = Trainer::fresh(config.clone());
        trainer.train(&RunControl::default()).unwrap();
        let artifact = trainer.artifact();
        assert_eq!(artifact.schema_version, 2);
        assert_eq!(artifact.model, CASH_MODEL);
        assert_eq!(
            artifact.rules_sha256,
            Some(config.cash_rules.as_ref().unwrap().sha256().unwrap())
        );
        let metrics = artifact.metrics.held_out.cash_accounting.as_ref().unwrap();
        assert!(metrics.maximum_conservation_residual_bb < 1e-10);
        assert!(
            (artifact.metrics.held_out.button_mean_net_bb
                + metrics.big_blind_mean_net_bb
                + metrics.mean_house_rake_bb)
                .abs()
                < 1e-10
        );
        let checkpoint = serde_json::to_value(BlueprintCheckpointRef {
            schema_version: config.checkpoint_schema_version(),
            model: config.model_name(),
            approximate: true,
            config: &config,
            completed_iterations: trainer.completed_iterations,
            rng_state: trainer.rng.state(),
            regret_discount_cumulative_logs: trainer.discounts.cumulative_logs,
            sampled_deals: trainer.sampled_deals,
            terminal_evaluations: trainer.terminal_evaluations,
            public_histories: &trainer.public_histories,
            nodes: &trainer.nodes,
        })
        .unwrap();
        let mut target = config.clone();
        target.iterations += 2;
        let mut resumed =
            Trainer::from_checkpoint(serde_json::from_value(checkpoint.clone()).unwrap(), &target)
                .unwrap();
        resumed.train(&RunControl::default()).unwrap();
        let mut uninterrupted = Trainer::fresh(target.clone());
        uninterrupted.train(&RunControl::default()).unwrap();
        assert_eq!(resumed.artifact(), uninterrupted.artifact());
        target.cash_rules.as_mut().unwrap().rake.rate_basis_points = 400;
        assert!(Trainer::from_checkpoint(
            serde_json::from_value(checkpoint.clone()).unwrap(),
            &target
        )
        .is_err());
        target.cash_rules = None;
        assert!(
            Trainer::from_checkpoint(serde_json::from_value(checkpoint).unwrap(), &target).is_err()
        );
    }
}
