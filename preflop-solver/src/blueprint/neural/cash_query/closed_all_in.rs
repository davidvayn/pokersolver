//! Exact fixed-policy leaves after a postflop all-in. No future betting,
//! hidden-card oracle, learned continuation, or artificial zero-sum utility.
use super::*;

impl CashPolicyEngine {
    pub(super) fn exact_closed_all_in(
        &self,
        state: &GameState,
        deal: &Deal,
        visible: &[u8],
        outcomes: &mut u64,
    ) -> Result<Option<[f64; 3]>, String> {
        if !matches!(visible.len(), 3 | 4) {
            return Ok(None);
        }
        if let Some(terminal) = &state.terminal {
            return match terminal {
                Terminal::Fold { .. } => self.terminal_values(state, deal).map(Some),
                Terminal::Showdown => self
                    .exact_all_in_showdown(state, deal, visible, outcomes)
                    .map(Some),
            };
        }
        // Only closed trees qualify. A normal call advancing to another
        // street must still evaluate that street's actual frozen policy.
        let legal = state.legal_actions(&self.config.game);
        let children = legal
            .iter()
            .map(|a| state.apply(a, &self.config.game))
            .collect::<Vec<_>>();
        if children.is_empty() || children.iter().any(|s| s.terminal.is_none()) {
            return Ok(None);
        }
        let mix = cash_evaluation::checked_strategy(
            &self.policy,
            state,
            deal,
            &legal,
            &self.config.game,
        )?;
        let mut value = [0.; 3];
        for (child, probability) in children.iter().zip(mix) {
            if probability == 0. {
                continue;
            }
            let result = self
                .exact_closed_all_in(child, deal, visible, outcomes)?
                .ok_or("closed all-in continuation unexpectedly remained open")?;
            for i in 0..3 {
                value[i] += probability * result[i];
            }
        }
        Ok(Some(value))
    }

    fn exact_all_in_showdown(
        &self,
        state: &GameState,
        deal: &Deal,
        visible: &[u8],
        outcomes: &mut u64,
    ) -> Result<[f64; 3], String> {
        let available = (0..52u8)
            .filter(|c| !deal.holes.iter().flatten().any(|h| h == c) && !visible.contains(c))
            .collect::<Vec<_>>();
        let count = if visible.len() == 4 {
            available.len()
        } else {
            available.len() * (available.len() - 1) / 2
        };
        *outcomes += count as u64;
        if *outcomes > 2_000_000 {
            return Err("exact closed all-in chance budget exhausted; no partial score".into());
        }
        let mut board = [0u8; 5];
        board[..visible.len()].copy_from_slice(visible);
        let mut counts = [0u64; 3];
        let mut observe = |board: &[u8; 5]| {
            let win = showdown_result(&deal.holes, board);
            counts[if win == 1. {
                0
            } else if win == 0. {
                1
            } else {
                2
            }] += 1;
        };
        if visible.len() == 4 {
            for river in available {
                board[4] = river;
                observe(&board);
            }
        } else {
            // No later actions: unordered turn/river pairs are equiprobable
            // and have the same showdown/cent split as either arrival order.
            for i in 0..available.len() {
                for j in i + 1..available.len() {
                    board[3] = available[i];
                    board[4] = available[j];
                    observe(&board);
                }
            }
        }
        let rules = self.config.game.cash_rules.as_ref().expect("cash rules");
        let gross = 2 * cash::cash_units(state.invested[0].min(state.invested[1]), rules)?;
        let rake = rules.rake_units(gross, true)? as f64 / rules.units_per_bb as f64;
        let mut own = [0.; 2];
        for (weight, outcome) in counts.into_iter().zip([
            crate::cash_game::Outcome::PlayerZero,
            crate::cash_game::Outcome::PlayerOne,
            crate::cash_game::Outcome::Split,
        ]) {
            let payoff = cash::cash_terminal_payoffs(
                rules,
                state.invested,
                crate::cash_game::TerminalReason::Showdown,
                outcome,
                5,
            )?;
            for i in 0..2 {
                own[i] += weight as f64 / count as f64 * payoff[i];
            }
        }
        if (own[0] + own[1] + rake).abs() > 1e-8 {
            return Err("exact all-in independent house accounting failed".into());
        }
        Ok([own[0], own[1], rake])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_turn_all_in_matches_complete_terminal_enumeration_and_leaves_open_trees_sampled() {
        let engine = super::super::tests::engine();
        let mut state = GameState::initial(&engine.config.game);
        state.street = Street::Turn;
        state.invested = [20., 20.];
        state.terminal = Some(Terminal::Showdown);
        let deal = Deal::from_sampled_cards([[48, 49], [4, 5]], [8, 13, 22, 31, 40]);
        let visible = &deal.board[..4];
        let mut outcomes = 0;
        let actual = engine
            .exact_closed_all_in(&state, &deal, visible, &mut outcomes)
            .unwrap()
            .unwrap();
        assert_eq!(outcomes, 44);
        let mut expected = [0.; 3];
        for river in 0..52u8 {
            if visible.contains(&river) || deal.holes.iter().flatten().any(|c| *c == river) {
                continue;
            }
            let complete = Deal::from_sampled_cards(deal.holes, [8, 13, 22, 31, river]);
            let value = engine.terminal_values(&state, &complete).unwrap();
            for i in 0..3 {
                expected[i] += value[i] / 44.;
            }
        }
        for i in 0..3 {
            assert!((actual[i] - expected[i]).abs() < 1e-10);
        }
        assert!((actual[2] - 1.8).abs() < 1e-10);
        let open = GameState::initial(&engine.config.game);
        assert!(engine
            .exact_closed_all_in(&open, &deal, &[], &mut outcomes)
            .unwrap()
            .is_none());
        let mut open = open;
        open.street = Street::Turn;
        open.invested = [1., 1.];
        open.street_invested = [0., 0.];
        assert!(engine
            .exact_closed_all_in(&open, &deal, visible, &mut outcomes)
            .unwrap()
            .is_none());
        let mut exhausted = 2_000_000;
        assert!(engine
            .exact_closed_all_in(&state, &deal, visible, &mut exhausted)
            .is_err());
    }
}
