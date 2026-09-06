use super::*;
use crate::blueprint::range_vector::{
    ExactFlopTerminal, PublicInformationSetCache, RangeTerminalKind,
};

fn exact_matrix() -> Arc<Vec<f32>> {
    let flop = [0, 5, 10];
    let legal = std::array::from_fn(|_| {
        all_combos()
            .iter()
            .map(|c| !c.cards().iter().any(|card| flop.contains(card)))
            .collect::<Vec<_>>()
    });
    exact_flop_all_in_equities(flop, &legal, 1)
}

#[test]
fn exact_flop_chance_preserves_mean_with_future_masks_and_reduces_terminal_variance() {
    let flop = [0, 5, 10];
    let hero = Combo::new(51, 50);
    let opponent = Combo::new(47, 46);
    let matrix = exact_matrix();
    let exact = Arc::new(ExactFlopTerminal::from_equities(flop, &matrix).unwrap());
    let mut original = Vec::new();
    let mut integrated = Vec::new();
    let mut legal_runouts = 0;
    for turn in (0..52u8).filter(|c| !flop.contains(c)) {
        for river in (0..52u8).filter(|c| !flop.contains(c) && *c != turn) {
            let board = [flop[0], flop[1], flop[2], turn, river];
            let cache = PublicInformationSetCache::new(board)
                .unwrap()
                .with_exact_flop_terminal(exact.clone())
                .unwrap();
            let mut reach = vec![0.0; COMBO_COUNT];
            if !opponent.cards().iter().any(|c| board.contains(c)) {
                reach[opponent.key()] = FUTURE_CHANCE_CORRECTION;
            }
            let actual = cache
                .terminal_values_at_street(
                    Street::Flop,
                    [20.0, 20.0],
                    &reach,
                    0,
                    RangeTerminalKind::Showdown,
                )
                .unwrap()[hero.key()];
            let valid = !hero.cards().iter().any(|c| board.contains(c))
                && !opponent.cards().iter().any(|c| board.contains(c));
            let observed = if valid {
                legal_runouts += 1;
                (40.0 * showdown_result(&[hero.cards(), opponent.cards()], &board) - 20.0)
                    * FUTURE_CHANCE_CORRECTION
            } else {
                assert_eq!(actual, 0.0);
                0.0
            };
            original.push(observed);
            integrated.push(actual);
        }
    }
    assert_eq!(legal_runouts, 45 * 44);
    assert_eq!(original.len(), 49 * 48);
    let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
    let variance = |xs: &[f64]| {
        let mu = mean(xs);
        xs.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / xs.len() as f64
    };
    assert!((mean(&original) - mean(&integrated)).abs() < 1e-11);
    assert!(variance(&integrated) < variance(&original));
    // This is variance reduction for the checked terminal estimator, not a
    // claim about regret variance throughout the game or policy convergence.
}

#[test]
fn exact_terminal_keeps_raw_reach_scale_and_leaves_other_streets_and_folds_unchanged() {
    let matrix = exact_matrix();
    let exact = Arc::new(ExactFlopTerminal::from_equities([0, 5, 10], &matrix).unwrap());
    let board = [0, 5, 10, 1, 2];
    let old = PublicInformationSetCache::new(board).unwrap();
    let new = PublicInformationSetCache::new(board)
        .unwrap()
        .with_exact_flop_terminal(exact.clone())
        .unwrap();
    let hands = [Combo::new(51, 50), Combo::new(47, 46)];
    let weights = [0.3, 0.7];
    let mut total = 0.0;
    for player in 0..2 {
        let mut reach = vec![0.0; COMBO_COUNT];
        reach[hands[1 - player].key()] = weights[1 - player];
        let result = new
            .terminal_values_at_street(
                Street::Flop,
                [20.0, 20.0],
                &reach,
                player,
                RangeTerminalKind::Showdown,
            )
            .unwrap();
        total += weights[player] * result[hands[player].key()];
        let doubled = reach.iter().map(|v| 2.0 * v).collect::<Vec<_>>();
        let twice = new
            .terminal_values_at_street(
                Street::Flop,
                [20.0, 20.0],
                &doubled,
                player,
                RangeTerminalKind::Showdown,
            )
            .unwrap();
        assert_eq!(
            twice[hands[player].key()],
            2.0 * result[hands[player].key()]
        );
        for street in [Street::Preflop, Street::Turn, Street::River] {
            assert_eq!(
                new.terminal_values_at_street(
                    street,
                    [20.0, 20.0],
                    &reach,
                    player,
                    RangeTerminalKind::Showdown,
                )
                .unwrap(),
                old.terminal_values([20.0, 20.0], &reach, player, RangeTerminalKind::Showdown)
                    .unwrap()
            );
        }
        for winner in 0..2 {
            let kind = RangeTerminalKind::Fold { winner };
            assert_eq!(
                new.terminal_values_at_street(Street::Flop, [3.0, 5.0], &reach, player, kind)
                    .unwrap(),
                old.terminal_values([3.0, 5.0], &reach, player, kind)
                    .unwrap()
            );
        }
        reach[Combo::new(1, 51).key()] = 1.0;
        assert!(new
            .terminal_values_at_street(
                Street::Flop,
                [20.0, 20.0],
                &reach,
                player,
                RangeTerminalKind::Showdown,
            )
            .is_err());
    }
    assert!(total.abs() < 1e-12);
    assert!(PublicInformationSetCache::new([0, 5, 11, 1, 2])
        .unwrap()
        .with_exact_flop_terminal(exact)
        .is_err());
}

#[test]
fn exact_equity_import_rejects_missing_blocked_off_lattice_and_asymmetric_entries() {
    let original = exact_matrix();
    let key = Combo::new(51, 50).key() * COMBO_COUNT + Combo::new(47, 46).key();
    assert!(ExactFlopTerminal::from_equities([0, 0, 10], &original).is_err());
    assert!(ExactFlopTerminal::from_equities([0, 5, 10], &original[..100]).is_err());
    for replacement in [f32::NAN, f32::INFINITY, -0.1, 0.1234567] {
        let mut changed = original.as_ref().clone();
        changed[key] = replacement;
        assert!(ExactFlopTerminal::from_equities([0, 5, 10], &changed).is_err());
    }
    let mut changed = original.as_ref().clone();
    changed[key] = ((original[key] as f64 * 1980.0).round() as f32 - 1.0) / 1980.0;
    assert!(ExactFlopTerminal::from_equities([0, 5, 10], &changed).is_err());
    changed = original.as_ref().clone();
    changed[0] = 0.5;
    assert!(ExactFlopTerminal::from_equities([0, 5, 10], &changed).is_err());
}

#[test]
fn exact_terminal_training_is_deterministic_separately_identified_and_does_not_mutate_control() {
    let config = fixture();
    let control = solve(config.clone()).unwrap();
    let candidate = solve_with_exact_terminals(config.clone()).unwrap();
    assert_eq!(
        candidate,
        solve_with_exact_terminals(config.clone()).unwrap()
    );
    assert_eq!(control, solve(config).unwrap());
    assert_ne!(candidate.input_sha256, control.input_sha256);
    assert_ne!(candidate.schema, control.schema);
    assert_eq!(candidate.root.action_labels, control.root.action_labels);
    assert_eq!(candidate.trained_root_combos, 1176);
    assert_eq!(candidate.validation.status, "research_only");
    assert!(candidate.root.action_values_bb.is_none());
    let width = candidate.root.action_labels.len();
    for c in all_combos() {
        let mass: f32 = candidate.root.probabilities[c.key() * width..(c.key() + 1) * width]
            .iter()
            .sum();
        let blocked = c.cards().iter().any(|card| [0, 5, 10].contains(card));
        assert!((mass - if blocked { 0.0 } else { 1.0 }).abs() < 1e-6);
    }
}
