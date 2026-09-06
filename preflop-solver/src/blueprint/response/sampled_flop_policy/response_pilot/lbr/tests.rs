use super::*;

pub(super) enum Pattern {
    Call,
    Fold,
    RankLimp,
    Invalid,
}
impl ResponsePolicy for Pattern {
    fn strategy(
        &self,
        state: &GameState,
        deal: &Deal,
        actions: &[LegalAction],
        _: &BlueprintConfig,
    ) -> Vec<f64> {
        let mut mix = vec![0.0; actions.len()];
        if matches!(self, Self::Invalid) {
            mix[0] = f64::NAN;
            return mix;
        }
        if matches!(self, Self::RankLimp) {
            let call = actions
                .iter()
                .position(|a| a.kind == ActionKind::Call)
                .unwrap();
            let fold = actions
                .iter()
                .position(|a| a.kind == ActionKind::Fold)
                .unwrap();
            let cards = deal.holes[state.actor];
            mix[call] = if cards.iter().all(|c| *c >= 48) {
                0.8
            } else {
                0.2
            };
            mix[fold] = 1.0 - mix[call];
            return mix;
        }
        let selected = if matches!(self, Self::Fold) {
            actions.iter().position(|a| a.kind == ActionKind::Fold)
        } else {
            None
        }
        .or_else(|| {
            actions
                .iter()
                .position(|a| matches!(a.kind, ActionKind::Call | ActionKind::Check))
        })
        .unwrap();
        mix[selected] = 1.0;
        mix
    }
}

pub(super) fn fixture(street: Street) -> (BlueprintConfig, GameState) {
    let mut game = BlueprintConfig::default();
    game.effective_stack_bb = 4.0;
    let mut state = GameState::initial(&game);
    while state.street != street {
        let action = state
            .legal_actions(&game)
            .into_iter()
            .find(|a| matches!(a.kind, ActionKind::Call | ActionKind::Check))
            .unwrap();
        state = state.apply(&action, &game);
    }
    (game, state)
}

#[test]
fn lbr_bayes_weights_only_the_observed_opponent_action_and_removes_cards() {
    let (game, state) = fixture(Street::Preflop);
    let mut belief = Belief::new(1, [0, 1]).unwrap();
    let actions = state.legal_actions(&game);
    let call = actions
        .iter()
        .position(|a| a.kind == ActionKind::Call)
        .unwrap();
    belief
        .observe(&Pattern::RankLimp, &game, &state, &[], &actions, call)
        .unwrap();
    assert!((belief.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    let aces = Combo::new(48, 49).key();
    let kings = Combo::new(44, 45).key();
    assert!((belief.weights[aces] / belief.weights[kings] - 4.0).abs() < 1e-12);
    belief.reveal(&[4, 5, 6]).unwrap();
    for combo in all_combos() {
        if combo.cards().iter().any(|c| [0, 1, 4, 5, 6].contains(c)) {
            assert_eq!(belief.weights[combo.key()], 0.0);
        }
    }
    let next = state.apply(&actions[call], &game);
    assert!(belief
        .observe(
            &Pattern::Call,
            &game,
            &next,
            &[],
            &next.legal_actions(&game),
            0
        )
        .is_err());
    let mut impossible = Belief::new(1, [0, 1]).unwrap();
    assert!(impossible
        .observe(&Pattern::Fold, &game, &state, &[], &actions, call)
        .is_err());
}

#[test]
fn lbr_exact_river_values_choose_value_shoves_and_reject_losing_calls() {
    let lbr = Lbr {
        seed: 90001,
        early_runouts_per_combo: 16,
    };
    let (game, state) = fixture(Street::River);
    let board = [50, 51, 0, 5, 10];
    let mut nuts = Belief::new(state.actor, [48, 49]).unwrap();
    nuts.reveal(&board).unwrap();
    let actions = state.legal_actions(&game);
    let q = lbr
        .values(&nuts, &Pattern::Call, &game, &state, &board, &actions)
        .unwrap();
    let selected = best_index(&q);
    assert_eq!(selected, actions.len() - 1);
    assert!((q[selected] - game.effective_stack_bb).abs() < 1e-10);

    let board = [50, 51, 0, 13, 18];
    let mut weak = Belief::new(1 - state.actor, [4, 9]).unwrap();
    weak.weights.fill(0.0);
    weak.weights[Combo::new(48, 49).key()] = 1.0;
    weak.reveal(&board).unwrap();
    let facing = state.apply(actions.last().unwrap(), &game);
    let replies = facing.legal_actions(&game);
    let q = lbr
        .values(&weak, &Pattern::Call, &game, &facing, &board, &replies)
        .unwrap();
    assert_eq!(replies[best_index(&q)].kind, ActionKind::Fold);
    assert_eq!(q[best_index(&q)], -1.0);
    let call = replies
        .iter()
        .position(|a| a.kind == ActionKind::Call)
        .unwrap();
    assert_eq!(q[call], -4.0);
}

#[test]
fn lbr_samples_are_public_and_shared_across_actions_not_taken_from_the_deal() {
    let (game, state) = fixture(Street::Flop);
    let a = Deal::from_sampled_cards([[44, 45], [48, 49]], [0, 5, 10, 15, 20]);
    let b = Deal::from_sampled_cards([[40, 41], [48, 49]], [0, 5, 10, 19, 24]);
    let lbr = Lbr {
        seed: 90001,
        early_runouts_per_combo: 8,
    };
    let actions = state.legal_actions(&game);
    let values = |deal: &Deal| {
        let mut belief = Belief::new(state.actor, deal.holes[state.actor]).unwrap();
        belief.reveal(&deal.board[..3]).unwrap();
        lbr.values(
            &belief,
            &Pattern::Call,
            &game,
            &state,
            &deal.board[..3],
            &actions,
        )
        .unwrap()
    };
    assert_eq!(values(&a), values(&b));
    let mut belief = Belief::new(state.actor, a.holes[state.actor]).unwrap();
    belief.reveal(&a.board[..3]).unwrap();
    let q = lbr
        .values(
            &belief,
            &Pattern::Fold,
            &game,
            &state,
            &a.board[..3],
            &actions,
        )
        .unwrap();
    for (action, value) in actions.iter().zip(q) {
        if matches!(action.kind, ActionKind::RaiseTo(_)) {
            assert!((value - state.invested[1 - state.actor]).abs() < 1e-10);
        }
    }
    assert!(lbr
        .values(
            &belief,
            &Pattern::Invalid,
            &game,
            &state,
            &a.board[..3],
            &actions
        )
        .is_err());
}
