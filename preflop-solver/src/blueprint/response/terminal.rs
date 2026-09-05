//! Exact postflop terminal labels for response training and conditional-mean
//! payout assessment. Both private hands are used only by the offline evaluator,
//! never to select the defender's action. Preflop all-ins remain sampled.
use super::*;

/// Uniform ordered runout conditional on the two offline training holdings.
/// Does not accept the predealt board, so no unrevealed card can enter its seed.
pub(super) fn sample_preflop_runout(holes: [[u8; 2]; 2], rng: &mut SplitMix64) -> Deal {
    let mut deck: Vec<_> = (0..52u8)
        .filter(|card| !holes.iter().flatten().any(|hole| hole == card))
        .collect();
    for index in 0..5 {
        let other = index + rng.index(deck.len() - index);
        deck.swap(index, other);
    }
    Deal::from_sampled_cards(holes, [deck[0], deck[1], deck[2], deck[3], deck[4]])
}

pub(super) fn expectation(state: &GameState, deal: &Deal) -> Option<f64> {
    match state.terminal.as_ref()? {
        Terminal::Fold { .. } => Some(realized_utility_p0(state, deal)),
        Terminal::Showdown if state.street == Street::Preflop => None,
        Terminal::Showdown => {
            let visible = state.street.board_len();
            let mut board = deal.board;
            let remaining: Vec<_> = (0..52u8)
                .filter(|c| {
                    !deal.holes.iter().flatten().any(|h| h == c) && !board[..visible].contains(c)
                })
                .collect();
            let equity = match visible {
                3 => {
                    let mut total = 0.0;
                    let mut count = 0;
                    for i in 0..remaining.len() {
                        board[3] = remaining[i];
                        for card in &remaining[i + 1..] {
                            board[4] = *card;
                            total += showdown_result(&deal.holes, &board);
                            count += 1;
                        }
                    }
                    assert_eq!(count, 990);
                    total / count as f64
                }
                4 => {
                    assert_eq!(remaining.len(), 44);
                    remaining
                        .iter()
                        .map(|card| {
                            board[4] = *card;
                            showdown_result(&deal.holes, &board)
                        })
                        .sum::<f64>()
                        / 44.0
                }
                5 => showdown_result(&deal.holes, &board),
                _ => unreachable!("postflop visible-card count"),
            };
            Some(equity * state.invested[1] - (1.0 - equity) * state.invested[0])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preflop_action_labels_average_fresh_conditional_runouts() {
        let (policy, _) = super::super::tests::tabular_fixture();
        let game = &policy.table.config;
        let root = GameState::initial(game);
        let limped = root.apply(&root.legal_actions(game)[1], game);
        let facing = limped.apply(limped.legal_actions(game).last().unwrap(), game);
        assert_eq!(facing.street, Street::Preflop);
        assert_eq!(facing.actor, 0);
        let call = facing.legal_actions(game).iter()
            .position(|a| a.kind == ActionKind::Call).unwrap();
        let observe = |board, conditional| {
            let deal = Deal::from_cards([[51, 50], [44, 45]], board);
            let mut records = Vec::new();
            collect_trajectory_decisions(
                &policy, facing.clone(), &deal, game, 0, 64, true, conditional, false,
                17, 0, &mut SplitMix64::new(3), &mut records,
            );
            assert_eq!(records.len(), 1);
            records.pop().unwrap().values
        };
        let losing = Deal::from_cards([[51, 50], [44, 45]], [0, 5, 10, 46, 47]);
        let winning = Deal::from_cards([[51, 50], [44, 45]], [0, 5, 10, 46, 49]);
        assert_eq!(information_set(&facing, &losing, game).0, information_set(&facing, &winning, game).0);
        let called = facing.apply(&facing.legal_actions(game)[call], game);
        assert_eq!(realized_utility_p0(&called, &losing), -20.0);
        assert_eq!(realized_utility_p0(&called, &winning), 20.0);
        assert_eq!(observe(losing.board, false)[call], -20.0);
        assert_eq!(observe(winning.board, false)[call], 20.0);
        let unlucky = observe(losing.board, true);
        let lucky = observe(winning.board, true);
        assert_eq!(unlucky, lucky, "preflop Q sampling must not repeat the one predealt future board");
        assert!(unlucky[call] > -20.0 && unlucky[call] < 20.0);
    }

    #[test]
    fn conditional_preflop_boards_are_legal_and_deterministic() {
        let holes = [[51, 50], [44, 45]];
        let mut seen = BTreeSet::new();
        for seed in 0..128 {
            let a = sample_preflop_runout(holes, &mut SplitMix64::new(seed));
            let b = sample_preflop_runout(holes, &mut SplitMix64::new(seed));
            assert_eq!(a.holes, holes);
            assert_eq!(a.board, b.board);
            assert_eq!(a.holes.iter().flatten().chain(a.board.iter()).copied()
                .collect::<BTreeSet<_>>().len(), 9);
            seen.insert(a.board);
        }
        assert!(seen.len() > 120);
    }

    #[test]
    fn postflop_only_budget_preserves_the_authentic_line_and_postflop_labels() {
        let (policy, deal) = super::super::tests::tabular_fixture();
        let game = &policy.table.config;
        let mut compared = 0;
        for seed in 0..32 {
            let observe = |only, conditional| {
                let mut records = Vec::new();
                collect_trajectory_decisions(
                    &policy,
                    GameState::initial(game),
                    &deal,
                    game,
                    0,
                    2,
                    true,
                    conditional,
                    only,
                    17,
                    0,
                    &mut SplitMix64::new(seed),
                    &mut records,
                );
                records
                    .into_iter()
                    .filter(|r| r.keys[0].1.street != Street::Preflop)
                    .map(|r| (r.keys, r.strategic_labels, r.values, r.baseline_strategy))
                    .collect::<Vec<_>>()
            };
            let full = observe(false, false);
            compared += full.len();
            assert_eq!(full, observe(true, false));
            assert_eq!(full, observe(false, true), "preflop resampling must not alter the authentic postflop line or labels");
        }
        assert!(compared > 0, "fixture must reach actual postflop decisions");
    }

    #[test]
    fn terminal_action_training_no_longer_repeats_one_lucky_or_unlucky_runout() {
        let (policy, _) = super::super::tests::tabular_fixture();
        let game = &policy.table.config;
        let root = GameState::initial(game);
        let called = root.apply(&root.legal_actions(game)[1], game);
        let flop = called.apply(&called.legal_actions(game)[0], game);
        let facing = flop.apply(flop.legal_actions(game).last().unwrap(), game);
        let actions = facing.legal_actions(game);
        let call = actions
            .iter()
            .position(|a| a.kind == ActionKind::Call)
            .unwrap();
        let observe = |future: [u8; 2], integrated| {
            let deal = Deal::from_cards([[51, 50], [36, 37]], [0, 5, 10, future[0], future[1]]);
            let mut records = Vec::new();
            collect_trajectory_decisions(
                &policy,
                facing.clone(),
                &deal,
                game,
                facing.actor,
                4,
                integrated,
                false,
                false,
                17,
                0,
                &mut SplitMix64::new(3),
                &mut records,
            );
            assert_eq!(records.len(), 1);
            records.pop().unwrap().values
        };
        let unlucky = observe([38, 39], false);
        let lucky = observe([12, 13], false);
        assert_eq!(unlucky[call], -20.0);
        assert_eq!(lucky[call], 20.0);
        let exact = observe([38, 39], true);
        assert_eq!(exact, observe([12, 13], true));
        assert!(exact[call] > -20.0 && exact[call] < 20.0);
    }

    #[test]
    fn exact_terminal_labels_integrate_visible_prefix_and_preserve_payoff_units() {
        let (policy, _) = super::super::tests::tabular_fixture();
        let game = &policy.table.config;
        let mut terminal = GameState::initial(game);
        terminal.terminal = Some(Terminal::Showdown);
        terminal.invested = [20.0, 20.0];
        let deal = Deal::from_cards([[51, 50], [36, 37]], [48, 5, 10, 0, 1]);
        assert_eq!(expectation(&terminal, &deal), None);
        terminal.street = Street::Flop;
        let first = expectation(&terminal, &deal).unwrap();
        let changed = Deal::from_cards(deal.holes, [48, 5, 10, 12, 13]);
        assert_eq!(first, expectation(&terminal, &changed).unwrap());
        assert!((-20.0..=20.0).contains(&first));
        let reversed = Deal::from_cards([deal.holes[1], deal.holes[0]], deal.board);
        assert!((first + expectation(&terminal, &reversed).unwrap()).abs() < 1e-12);
        terminal.street = Street::Turn;
        let value = expectation(&terminal, &deal).unwrap();
        let mut total = 0.0;
        let mut count = 0;
        for card in 0..52u8 {
            if deal.holes.iter().flatten().any(|h| *h == card) || deal.board[..4].contains(&card) {
                continue;
            }
            let mut board = deal.board;
            board[4] = card;
            total += realized_utility_p0(&terminal, &Deal::from_cards(deal.holes, board));
            count += 1;
        }
        assert_eq!(count, 44);
        assert!((value - total / 44.0).abs() < 1e-12);
        terminal.street = Street::River;
        assert_eq!(
            expectation(&terminal, &deal).unwrap(),
            realized_utility_p0(&terminal, &deal)
        );
        terminal.terminal = Some(Terminal::Fold { winner: 1 });
        assert_eq!(expectation(&terminal, &deal), Some(-20.0));
        terminal.terminal = None;
        assert_eq!(expectation(&terminal, &deal), None);
    }
}
