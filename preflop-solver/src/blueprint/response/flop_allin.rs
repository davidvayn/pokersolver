//! Terminal flop decisions can use exact private cards and an action-conditioned
//! opponent range instead of a coarse, sparsely observed action-value bucket.
//! Equity is sampled, not called exact; a Hoeffding margin avoids acting on a
//! small uncertain advantage. This is a response to the frozen opponent model,
//! not a minimax/safe-solving guarantee against arbitrary opponents.

use super::*;
use crate::blueprint::neural::{deal_for_policy_combo_on_board, trajectory_action_matches};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TerminalFlopOptions {
    pub equity_samples: u32,
    pub weight: f64,
}

impl TerminalFlopOptions {
    pub(super) fn validate(&self) -> Result<(), String> {
        if !(128..=16384).contains(&self.equity_samples)
            || !self.weight.is_finite()
            || !(0.0..=1.0).contains(&self.weight)
        {
            return Err(
                "terminal flop correction requires 128..16384 equity samples and weight 0..1"
                    .to_owned(),
            );
        }
        Ok(())
    }
}

pub(super) fn correction(
    base: &TabularResponsePolicy,
    state: &GameState,
    deal: &Deal,
    actions: &[LegalAction],
    game: &BlueprintConfig,
    samples: u32,
) -> Option<usize> {
    if state.street != Street::Flop
        || actions.len() != 2
        || state.to_call() <= EPSILON
        || state.remaining(1 - state.actor, game) > EPSILON
    {
        return None;
    }
    let fold = actions
        .iter()
        .position(|a| matches!(a.kind, ActionKind::Fold))?;
    let call = actions
        .iter()
        .position(|a| matches!(a.kind, ActionKind::Call))?;
    let after = state.apply(&actions[call], game);
    if !matches!(after.terminal, Some(Terminal::Showdown)) {
        return None;
    }
    let hero = deal.holes[state.actor];
    let board = &deal.board[..3];
    let ranges = opponent_range(base, state, hero, board, game)?;
    let mut total = 0.0;
    let cumulative: Vec<_> = ranges
        .iter()
        .map(|(_, w)| {
            total += w;
            total
        })
        .collect();
    if total <= 0.0 || !total.is_finite() {
        return None;
    }
    // No hidden opponent cards or unrevealed board cards enter the seed or EV.
    let mut identity = b"terminal-flop-range-response-v1".to_vec();
    let mut own = hero;
    own.sort_unstable();
    identity.extend_from_slice(&own);
    identity.extend_from_slice(board);
    identity.extend_from_slice(state.public_history.join("|").as_bytes());
    identity.push(state.actor as u8);
    let mut rng = SplitMix64::new(stable_hash(&identity));
    let mut equity = 0.0;
    for _ in 0..samples {
        let draw = rng.next_f64() * total;
        let index = cumulative
            .partition_point(|mass| *mass <= draw)
            .min(ranges.len() - 1);
        let opponent = ranges[index].0;
        let mut runout = [board[0], board[1], board[2], 0, 0];
        for next in 3..5 {
            loop {
                let card = rng.index(52) as u8;
                if !hero.contains(&card)
                    && !opponent.contains(&card)
                    && !runout[..next].contains(&card)
                {
                    runout[next] = card;
                    break;
                }
            }
        }
        equity += showdown_result(&[hero, opponent], &runout);
    }
    let call_cost = after.invested[state.actor] - state.invested[state.actor];
    let advantage = equity / samples as f64 * after.pot() - call_cost;
    // Two-sided 99.5% bound for independent bounded showdown outcomes in [0,1].
    // Unlike a sample-SE interval it remains nonzero for all-win/all-loss draws.
    let margin = (400.0f64.ln() / (2.0 * samples as f64)).sqrt() * after.pot();
    if advantage > margin {
        Some(call)
    } else if advantage < -margin {
        Some(fold)
    } else {
        None
    }
}

fn opponent_range(
    base: &TabularResponsePolicy,
    state: &GameState,
    hero: [u8; 2],
    board: &[u8],
    game: &BlueprintConfig,
) -> Option<Vec<([u8; 2], f64)>> {
    let opponent = 1 - state.actor;
    let mut ranges: Vec<_> = all_combos()
        .iter()
        .filter(|c| {
            !c.cards()
                .iter()
                .any(|card| board.contains(card) || hero.contains(card))
        })
        .map(|c| (c.cards(), 1.0))
        .collect();
    let mut cursor = GameState::initial(game);
    for observed in &state.trajectory {
        let actions = cursor.legal_actions(game);
        let selected = actions
            .iter()
            .position(|a| trajectory_action_matches(&cursor, a, observed, game))?;
        if cursor.actor == opponent {
            let visible = &board[..cursor.street.board_len()];
            for (cards, weight) in &mut ranges {
                if *weight <= 0.0 {
                    continue;
                }
                let combo = Combo::new(cards[0], cards[1]);
                let synthetic = deal_for_policy_combo_on_board(combo, opponent, visible).ok()?;
                // Only this opponent's own action likelihood updates its range.
                *weight *= base.frozen_strategy(&cursor, &synthetic, &actions, game)[selected];
            }
            let mass: f64 = ranges.iter().map(|(_, w)| w).sum();
            // An impossible forced line keeps the explicit baseline completion;
            // do not invent a uniform posterior for this response experiment.
            if mass <= 0.0 || !mass.is_finite() {
                return None;
            }
            for (_, weight) in &mut ranges {
                *weight /= mass;
            }
        }
        cursor = cursor.apply(&actions[selected], game);
    }
    if cursor.public_history != state.public_history {
        return None;
    }
    Some(ranges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_blend_allows_the_full_best_response_but_no_invalid_probability() {
        for weight in [0.0, 0.25, 0.5, 0.75, 1.0] {
            assert!(TerminalFlopOptions { equity_samples: 2048, weight }.validate().is_ok());
        }
        for weight in [-0.01, 1.0001, f64::NAN, f64::INFINITY] {
            assert!(TerminalFlopOptions { equity_samples: 2048, weight }.validate().is_err());
        }
    }

    #[test]
    fn terminal_response_uses_ranges_and_visible_cards_not_realized_runout() {
        let (policy, _) = super::super::tests::tabular_fixture();
        let game = &policy.table.config;
        let root = GameState::initial(game);
        let called = root.apply(&root.legal_actions(game)[1], game);
        let flop = called.apply(&called.legal_actions(game)[0], game);
        let facing = flop.apply(flop.legal_actions(game).last().unwrap(), game);
        assert_eq!(facing.actor, 0);
        let actions = facing.legal_actions(game);
        let nuts = Deal::from_cards([[51, 50], [36, 37]], [48, 49, 0, 5, 10]);
        let different_hidden = Deal::from_cards([[51, 50], [40, 41]], [48, 49, 0, 9, 13]);
        let first = correction(&policy, &facing, &nuts, &actions, game, 2048);
        let second = correction(&policy, &facing, &different_hidden, &actions, game, 2048);
        assert_eq!(first, second);
        assert_eq!(
            first,
            actions
                .iter()
                .position(|a| matches!(a.kind, ActionKind::Call))
        );
        let ranges =
            opponent_range(&policy, &facing, nuts.holes[0], &nuts.board[..3], game).unwrap();
        assert!((ranges.iter().map(|(_, w)| w).sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(ranges.iter().all(|(cards, _)| cards
            .iter()
            .all(|c| !nuts.holes[0].contains(c) && !nuts.board[..3].contains(c))));
        assert!(correction(&policy, &flop, &nuts, &flop.legal_actions(game), game, 2048).is_none());
        let weak = Deal::from_cards([[0, 5], [36, 37]], [48, 45, 42, 9, 13]);
        assert_eq!(
            correction(&policy, &facing, &weak, &actions, game, 2048),
            actions
                .iter()
                .position(|a| matches!(a.kind, ActionKind::Fold))
        );
    }

    #[test]
    fn posterior_removes_hands_that_never_take_the_observed_action() {
        let (mut policy, _) = super::super::tests::tabular_fixture();
        let game = policy.table.config.clone();
        let deal = Deal::from_cards([[51, 50], [36, 37]], [48, 49, 0, 5, 10]);
        let root = GameState::initial(&game);
        let called = root.apply(&root.legal_actions(&game)[1], &game);
        let actions = called.legal_actions(&game);
        let (key, descriptor, _) = information_set(&called, &deal, &game);
        let mut probabilities = vec![0.0; actions.len()];
        probabilities[1] = 1.0;
        Arc::get_mut(&mut policy.table).unwrap().nodes.insert(
            key,
            AverageNode {
                descriptor,
                action_labels: actions
                    .iter()
                    .map(|a| Arc::<str>::from(a.label.as_str()))
                    .collect::<Vec<_>>()
                    .into(),
                strategy_sum: probabilities.into_boxed_slice(),
                average_visits: 1,
            },
        );
        let flop = called.apply(&actions[0], &game);
        let facing = flop.apply(flop.legal_actions(&game).last().unwrap(), &game);
        let ranges =
            opponent_range(&policy, &facing, deal.holes[0], &deal.board[..3], &game).unwrap();
        assert_eq!(
            ranges
                .iter()
                .find(|(cards, _)| *cards == Combo::new(deal.holes[1][0], deal.holes[1][1]).cards())
                .unwrap()
                .1,
            0.0
        );
        assert!(ranges.iter().any(|(_, weight)| *weight > 0.0));
    }
}
