//! Research-only control variate for the uniformly sampled public turn.
//! The reference is fixed before the draw, but compatible opponent masses use
//! CURRENT reaches. This is not a cached substitute for a native leaf solve.
//! See Davis et al., ICML 2020, https://proceedings.mlr.press/v119/davis20a.html.
use super::*;

#[derive(Clone)]
pub(super) struct Baseline {
    board: [u8; 3],
    conditional_bb: [Vec<f64>; 2],
}

impl Baseline {
    pub(super) fn new(board: [u8; 3]) -> Self {
        assert!(board.iter().all(|c| *c < 52));
        assert_eq!(board.iter().collect::<BTreeSet<_>>().len(), 3);
        Self {
            board,
            conditional_bb: std::array::from_fn(|_| vec![0.0; COMBO_COUNT]),
        }
    }

    pub(super) fn correct_and_learn(
        &mut self,
        turn: u8,
        reaches: &[Vec<f64>; 2],
        sampled: &[Vec<f64>; 2],
    ) -> [Vec<f64>; 2] {
        assert!(turn < 52 && !self.board.contains(&turn));
        assert!(reaches
            .iter()
            .all(|r| r.len() == COMBO_COUNT && r.iter().all(|v| v.is_finite() && *v >= 0.0)));
        assert!(sampled
            .iter()
            .all(|v| v.len() == COMBO_COUNT && v.iter().all(|x| x.is_finite())));
        let combos = all_combos();
        let mut result = std::array::from_fn(|_| vec![0.0; COMBO_COUNT]);
        for p in 0..2 {
            let before = compatible_masses_from_card_marginals(&combos, &reaches[1 - p]);
            let masked: Vec<_> = combos
                .iter()
                .map(|c| {
                    if c.cards().contains(&turn) {
                        0.0
                    } else {
                        reaches[1 - p][c.key()]
                    }
                })
                .collect();
            let after = compatible_masses_from_card_marginals(&combos, &masked);
            for combo in &combos {
                let k = combo.key();
                if combo.cards().iter().any(|c| self.board.contains(c)) {
                    assert_eq!(sampled[p][k], 0.0);
                    continue;
                }
                let available = !combo.cards().contains(&turn);
                if !available {
                    assert_eq!(sampled[p][k], 0.0);
                }
                let reference = self.conditional_bb[p][k];
                // For each compatible private pair, exactly 45 of the 49
                // public proposals are possible. E[49/45 * b * M_turn] = b*M.
                let sampled_reference = if available { reference * after[k] } else { 0.0 };
                result[p][k] =
                    reference * before[k] + (sampled[p][k] - sampled_reference) * 49.0 / 45.0;
                assert!(result[p][k].is_finite());
                // Update only AFTER forming this iteration's correction. A
                // small mass may safely skip learning: ANY fixed reference
                // keeps the estimator unbiased. Never divide by own reach.
                if available && after[k] > 1e-12 {
                    let observed = sampled[p][k] / after[k];
                    assert!(observed.is_finite());
                    self.conditional_bb[p][k] = 0.5 * reference + 0.5 * observed;
                }
            }
        }
        // Do not clip corrected CFVs or erase a combo containing the sampled
        // turn: these are FLOP update estimates, not values conditioned on turn.
        result
    }
}

#[test]
fn correction_is_unbiased_with_current_ranges_zero_own_reach_and_blockers() {
    let board = [0, 5, 10];
    let combos = all_combos();
    let mut reaches = std::array::from_fn(|_| vec![0.0; COMBO_COUNT]);
    // Non-normalized, asymmetric reaches. Most own combos have zero reach.
    reaches[0][Combo::new(51, 47).key()] = 0.3;
    reaches[0][Combo::new(46, 42).key()] = 0.1;
    reaches[1][Combo::new(49, 45).key()] = 0.2;
    reaches[1][Combo::new(48, 44).key()] = 0.4;
    let mut baseline = Baseline::new(board);
    for p in 0..2 {
        for c in 0..COMBO_COUNT {
            baseline.conditional_bb[p][c] = ((c * 7 + p * 3) % 17) as f64 - 8.0;
        }
    }
    let mut exact = std::array::from_fn::<_, 2, _>(|_| vec![0.0; COMBO_COUNT]);
    let mut mean = exact.clone();
    for turn in (0..52u8).filter(|c| !board.contains(c)) {
        let mut values = std::array::from_fn(|_| vec![0.0; COMBO_COUNT]);
        for p in 0..2 {
            for own in &combos {
                if own.cards().iter().any(|c| board.contains(c) || *c == turn) {
                    continue;
                }
                for other in &combos {
                    if reaches[1 - p][other.key()] == 0.0
                        || own.overlaps(*other)
                        || other.cards().contains(&turn)
                    {
                        continue;
                    }
                    // Arbitrary chance-varying leaf, independent of the baseline.
                    let utility =
                        ((own.key() + other.key() * 3 + turn as usize * 5) % 23) as f64 - 11.0;
                    values[p][own.key()] += reaches[1 - p][other.key()] * utility;
                }
            }
        }
        // Each possible draw starts with the SAME pre-draw baseline.
        let corrected = baseline.clone().correct_and_learn(turn, &reaches, &values);
        for p in 0..2 {
            for c in 0..COMBO_COUNT {
                exact[p][c] += values[p][c] / 45.0;
                mean[p][c] += corrected[p][c] / 49.0;
            }
        }
    }
    for p in 0..2 {
        for c in 0..COMBO_COUNT {
            assert!(
                (mean[p][c] - exact[p][c]).abs() < 1e-12,
                "seat {p} combo {c}"
            );
        }
    }
    // With no opponent reach, the reference cannot inject a value.
    let empty = std::array::from_fn(|_| vec![0.0; COMBO_COUNT]);
    assert_eq!(baseline.correct_and_learn(1, &empty, &empty), empty);
}

#[test]
fn exact_reference_removes_blocker_variance_and_zero_reference_preserves_old_update() {
    let board = [0, 5, 10];
    let own = Combo::new(51, 47);
    let opponent = Combo::new(49, 45);
    let mut reaches = std::array::from_fn(|_| vec![0.0; COMBO_COUNT]);
    reaches[1][opponent.key()] = 0.7;
    let mut baseline = Baseline::new(board);
    baseline.conditional_bb[0][own.key()] = 3.0;
    for turn in (0..52u8).filter(|c| !board.contains(c)) {
        let mut sampled = std::array::from_fn(|_| vec![0.0; COMBO_COUNT]);
        if !own.cards().contains(&turn) && !opponent.cards().contains(&turn) {
            sampled[0][own.key()] = 3.0 * 0.7;
        }
        let corrected = baseline.clone().correct_and_learn(turn, &reaches, &sampled);
        assert!((corrected[0][own.key()] - 2.1).abs() < 1e-12);
        let old = Baseline::new(board).correct_and_learn(turn, &reaches, &sampled);
        assert_eq!(old[0][own.key()], sampled[0][own.key()] * 49.0 / 45.0);
    }
}
