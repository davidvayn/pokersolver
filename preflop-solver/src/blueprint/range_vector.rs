//! Exact compatible-range primitives for the next full-game public-tree
//! traversal. These routines operate on every exact two-card combination and
//! return counterfactual values, rather than expanding a finite list of joint
//! private deals.

use super::*;
use std::sync::OnceLock;

pub const EXACT_COMBO_COUNT: usize = 1_326;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RangeTerminalKind {
    Fold { winner: usize },
    Showdown,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RangeTerminalEvaluation {
    /// Counterfactual values indexed `[player][exact combo key]`. Each entry
    /// already integrates the compatible opponent reach and excludes the
    /// player's own reach, matching the quantity used by CFR regret updates.
    pub counterfactual_values_bb: [Vec<f64>; 2],
    pub compatible_opponent_mass: [Vec<f64>; 2],
    pub joint_reach_mass: f64,
    pub profile_value_p0_bb: f64,
    pub profile_value_p1_bb: f64,
    pub zero_sum_residual_bb: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RangeActionEvaluation {
    /// Values for every actor action, indexed `[action][actor combo]`.
    pub actor_action_values_bb: Vec<Vec<f64>>,
    /// Expected counterfactual values after applying the actor's strategy.
    pub expected_counterfactual_values_bb: [Vec<f64>; 2],
}

/// Apply one exact-combo policy row at a public node. The actor reach is split
/// among actions; the opponent reach is copied because it excludes the actor's
/// private choice at this point in the tree.
pub fn propagate_action_ranges(
    ranges: &[Vec<f64>; 2],
    actor: usize,
    action_probabilities: &[Vec<f64>],
) -> Result<Vec<[Vec<f64>; 2]>, String> {
    validate_action_probabilities(ranges, actor, action_probabilities)?;
    Ok(action_probabilities
        .iter()
        .map(|probabilities| {
            let mut child = ranges.clone();
            for (reach, probability) in child[actor].iter_mut().zip(probabilities) {
                *reach *= probability;
            }
            child
        })
        .collect())
}

/// Recombine child counterfactual values at a public action node. The acting
/// player's values are strategy-weighted because their own reach is excluded
/// from those values. Opponent values are summed without another strategy
/// factor because each child already contains the actor's split reach.
pub fn combine_action_counterfactual_values(
    actor: usize,
    action_probabilities: &[Vec<f64>],
    child_values_bb: &[[Vec<f64>; 2]],
) -> Result<RangeActionEvaluation, String> {
    if actor > 1
        || action_probabilities.is_empty()
        || child_values_bb.len() != action_probabilities.len()
        || action_probabilities
            .iter()
            .any(|row| row.len() != EXACT_COMBO_COUNT)
        || action_probabilities
            .iter()
            .flatten()
            .any(|probability| !probability.is_finite() || *probability < 0.0)
        || child_values_bb.iter().any(|values| {
            values
                .iter()
                .any(|player| player.len() != EXACT_COMBO_COUNT)
        })
    {
        return Err("range action values have incompatible dimensions".to_owned());
    }
    if (0..EXACT_COMBO_COUNT).any(|combo| {
        (action_probabilities
            .iter()
            .map(|action| action[combo])
            .sum::<f64>()
            - 1.0)
            .abs()
            > 1e-9
    }) {
        return Err("exact-combo action probabilities must sum to one".to_owned());
    }
    let opponent = 1 - actor;
    let mut expected = std::array::from_fn(|_| vec![0.0; EXACT_COMBO_COUNT]);
    for combo in 0..EXACT_COMBO_COUNT {
        for action in 0..action_probabilities.len() {
            expected[actor][combo] +=
                action_probabilities[action][combo] * child_values_bb[action][actor][combo];
            expected[opponent][combo] += child_values_bb[action][opponent][combo];
        }
    }
    Ok(RangeActionEvaluation {
        actor_action_values_bb: child_values_bb
            .iter()
            .map(|values| values[actor].clone())
            .collect(),
        expected_counterfactual_values_bb: expected,
    })
}

fn validate_action_probabilities(
    ranges: &[Vec<f64>; 2],
    actor: usize,
    action_probabilities: &[Vec<f64>],
) -> Result<(), String> {
    if actor > 1
        || ranges.iter().any(|range| range.len() != EXACT_COMBO_COUNT)
        || action_probabilities.is_empty()
        || action_probabilities
            .iter()
            .any(|row| row.len() != EXACT_COMBO_COUNT)
    {
        return Err("range action probabilities have incompatible dimensions".to_owned());
    }
    for combo in 0..EXACT_COMBO_COUNT {
        let mut total = 0.0;
        for action in action_probabilities {
            let probability = action[combo];
            if !probability.is_finite() || probability < 0.0 {
                return Err("range action probabilities must be finite and non-negative".to_owned());
            }
            total += probability;
        }
        if ranges[actor][combo] > EPSILON && (total - 1.0).abs() > 1e-9 {
            return Err("reached exact-combo action probabilities must sum to one".to_owned());
        }
    }
    Ok(())
}

/// Evaluate a terminal state on one fully sampled public board while retaining
/// both exact private ranges. Ranges may be unnormalized realization reaches,
/// but board-blocked entries must already be zero.
pub fn evaluate_terminal_ranges(
    board: [u8; 5],
    invested_bb: [f64; 2],
    ranges: [Vec<f64>; 2],
    terminal: RangeTerminalKind,
) -> Result<RangeTerminalEvaluation, String> {
    validate_inputs(&board, invested_bb, &ranges, terminal)?;
    let combos = exact_combos();
    debug_assert_eq!(combos.len(), EXACT_COMBO_COUNT);
    let conflicts = public_belief::combo_conflicts();
    let compatible_opponent_mass =
        std::array::from_fn(|player| compatible_masses(&ranges[1 - player], &conflicts));
    let joint_reach_mass = ranges[0]
        .iter()
        .zip(&compatible_opponent_mass[0])
        .map(|(reach, mass)| reach * mass)
        .sum::<f64>();
    if joint_reach_mass <= EPSILON {
        return Err("exact ranges contain no compatible private deals".to_owned());
    }

    let counterfactual_values_bb = match terminal {
        RangeTerminalKind::Fold { winner } => {
            let utility_p0 = if winner == 0 {
                invested_bb[1]
            } else {
                -invested_bb[0]
            };
            std::array::from_fn(|player| {
                let utility = if player == 0 { utility_p0 } else { -utility_p0 };
                compatible_opponent_mass[player]
                    .iter()
                    .map(|mass| utility * mass)
                    .collect()
            })
        }
        RangeTerminalKind::Showdown => {
            showdown_counterfactual_values(&board, invested_bb, &ranges, combos)
        }
    };
    let profile_value_p0_bb = ranges[0]
        .iter()
        .zip(&counterfactual_values_bb[0])
        .map(|(reach, value)| reach * value)
        .sum::<f64>()
        / joint_reach_mass;
    let profile_value_p1_bb = ranges[1]
        .iter()
        .zip(&counterfactual_values_bb[1])
        .map(|(reach, value)| reach * value)
        .sum::<f64>()
        / joint_reach_mass;
    Ok(RangeTerminalEvaluation {
        counterfactual_values_bb,
        compatible_opponent_mass,
        joint_reach_mass,
        profile_value_p0_bb,
        profile_value_p1_bb,
        zero_sum_residual_bb: (profile_value_p0_bb + profile_value_p1_bb).abs(),
    })
}

fn validate_inputs(
    board: &[u8; 5],
    invested_bb: [f64; 2],
    ranges: &[Vec<f64>; 2],
    terminal: RangeTerminalKind,
) -> Result<(), String> {
    let unique = board.iter().copied().collect::<BTreeSet<_>>();
    if unique.len() != board.len() || board.iter().any(|card| *card >= 52) {
        return Err("sampled public board must contain five unique cards".to_owned());
    }
    if invested_bb
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err("terminal investments must be finite and non-negative".to_owned());
    }
    if matches!(terminal, RangeTerminalKind::Fold { winner } if winner > 1) {
        return Err("fold winner must be player zero or one".to_owned());
    }
    let combos = exact_combos();
    for (player, range) in ranges.iter().enumerate() {
        if range.len() != EXACT_COMBO_COUNT
            || range
                .iter()
                .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err(format!(
                "player {player} range must contain finite non-negative exact-combo reaches"
            ));
        }
        if range.iter().zip(combos).any(|(weight, combo)| {
            *weight > EPSILON && combo.cards().iter().any(|card| unique.contains(card))
        }) {
            return Err(format!(
                "player {player} range assigns reach to a board-blocked combination"
            ));
        }
    }
    Ok(())
}

fn exact_combos() -> &'static [Combo] {
    static COMBOS: OnceLock<Vec<Combo>> = OnceLock::new();
    COMBOS.get_or_init(all_combos)
}

fn compatible_masses(range: &[f64], conflicts: &[Vec<usize>]) -> Vec<f64> {
    let total = range.iter().sum::<f64>();
    conflicts
        .iter()
        .map(|blocked| (total - blocked.iter().map(|combo| range[*combo]).sum::<f64>()).max(0.0))
        .collect()
}

fn showdown_counterfactual_values(
    board: &[u8; 5],
    invested_bb: [f64; 2],
    ranges: &[Vec<f64>; 2],
    combos: &[Combo],
) -> [Vec<f64>; 2] {
    let scores = combos
        .iter()
        .map(|combo| {
            if combo.cards().iter().any(|card| board.contains(card)) {
                None
            } else {
                let cards = [
                    combo.cards()[0],
                    combo.cards()[1],
                    board[0],
                    board[1],
                    board[2],
                    board[3],
                    board[4],
                ];
                Some(evaluate(&cards))
            }
        })
        .collect::<Vec<_>>();
    let conflicts = public_belief::combo_conflicts();
    std::array::from_fn(|player| {
        let opponent = 1 - player;
        let mut masses_by_score = BTreeMap::<u32, f64>::new();
        for (score, reach) in scores.iter().zip(&ranges[opponent]) {
            if let Some(score) = score {
                *masses_by_score.entry(*score).or_default() += reach;
            }
        }
        let total_mass = masses_by_score.values().sum::<f64>();
        let mut lower_and_equal = BTreeMap::<u32, (f64, f64)>::new();
        let mut lower = 0.0;
        for (score, equal) in masses_by_score {
            lower_and_equal.insert(score, (lower, equal));
            lower += equal;
        }
        let win_utility = invested_bb[opponent];
        let lose_utility = -invested_bb[player];
        let tie_utility = 0.5 * (invested_bb[opponent] - invested_bb[player]);
        scores
            .iter()
            .enumerate()
            .map(|(own, score)| {
                let Some(score) = score else {
                    return 0.0;
                };
                let (mut weaker, mut equal) = lower_and_equal[score];
                let mut stronger = total_mass - weaker - equal;
                for blocked in &conflicts[own] {
                    let reach = ranges[opponent][*blocked];
                    if reach <= 0.0 {
                        continue;
                    }
                    match scores[*blocked].map(|blocked_score| blocked_score.cmp(score)) {
                        Some(std::cmp::Ordering::Less) => weaker -= reach,
                        Some(std::cmp::Ordering::Equal) => equal -= reach,
                        Some(std::cmp::Ordering::Greater) => stronger -= reach,
                        None => {}
                    }
                }
                weaker.max(0.0) * win_utility
                    + equal.max(0.0) * tie_utility
                    + stronger.max(0.0) * lose_utility
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_pair_ranges(first: Combo, second: Combo) -> [Vec<f64>; 2] {
        let mut ranges = std::array::from_fn(|_| vec![0.0; EXACT_COMBO_COUNT]);
        ranges[0][first.key()] = 1.0;
        ranges[1][second.key()] = 1.0;
        ranges
    }

    #[test]
    fn exact_range_showdown_matches_scalar_terminal_utility() {
        let deal = Deal::from_cards([[51, 50], [45, 44]], [0, 5, 10, 27, 28]);
        let ranges = single_pair_ranges(
            Combo::new(deal.holes[0][0], deal.holes[0][1]),
            Combo::new(deal.holes[1][0], deal.holes[1][1]),
        );
        let evaluation = evaluate_terminal_ranges(
            deal.board,
            [20.0, 20.0],
            ranges,
            RangeTerminalKind::Showdown,
        )
        .expect("compatible exact ranges");
        let scalar = showdown_result(&deal.holes, &deal.board) * 20.0
            - (1.0 - showdown_result(&deal.holes, &deal.board)) * 20.0;
        assert_eq!(evaluation.joint_reach_mass, 1.0);
        assert_eq!(evaluation.profile_value_p0_bb, scalar);
        assert_eq!(evaluation.profile_value_p1_bb, -scalar);
        assert_eq!(evaluation.zero_sum_residual_bb, 0.0);
    }

    #[test]
    fn exact_range_fold_values_integrate_compatible_opponent_mass() {
        let board = [0, 5, 10, 27, 28];
        let combos = exact_combos();
        let ranges = std::array::from_fn(|_| {
            combos
                .iter()
                .map(|combo| {
                    if combo.cards().iter().any(|card| board.contains(card)) {
                        0.0
                    } else {
                        1.0
                    }
                })
                .collect::<Vec<_>>()
        });
        let evaluation = evaluate_terminal_ranges(
            board,
            [3.0, 5.0],
            ranges,
            RangeTerminalKind::Fold { winner: 0 },
        )
        .expect("uniform compatible ranges");
        let hero = Combo::new(51, 50).key();
        assert_eq!(evaluation.compatible_opponent_mass[0][hero], 990.0);
        assert_eq!(evaluation.counterfactual_values_bb[0][hero], 4_950.0);
        assert_eq!(evaluation.counterfactual_values_bb[1][hero], -4_950.0);
        assert_eq!(evaluation.profile_value_p0_bb, 5.0);
        assert_eq!(evaluation.profile_value_p1_bb, -5.0);
        assert!(evaluation.zero_sum_residual_bb <= 1e-12);
    }

    #[test]
    fn exact_range_terminal_rejects_board_blocked_reach() {
        let board = [0, 5, 10, 27, 28];
        let mut ranges = std::array::from_fn(|_| vec![0.0; EXACT_COMBO_COUNT]);
        ranges[0][Combo::new(0, 1).key()] = 1.0;
        ranges[1][Combo::new(51, 50).key()] = 1.0;
        assert!(
            evaluate_terminal_ranges(board, [1.0, 1.0], ranges, RangeTerminalKind::Showdown,)
                .unwrap_err()
                .contains("board-blocked")
        );
    }

    #[test]
    fn score_prefix_terminal_kernel_matches_pairwise_enumeration() {
        let board = [0, 5, 10, 27, 28];
        let combos = exact_combos();
        let mut ranges = std::array::from_fn(|_| vec![0.0; EXACT_COMBO_COUNT]);
        for (key, combo) in combos.iter().enumerate() {
            if combo.cards().iter().any(|card| board.contains(card)) {
                continue;
            }
            if key.is_multiple_of(17) {
                ranges[0][key] = (key % 5 + 1) as f64;
            }
            if key.is_multiple_of(19) {
                ranges[1][key] = (key % 7 + 1) as f64;
            }
        }
        let measured = evaluate_terminal_ranges(
            board,
            [7.0, 11.0],
            ranges.clone(),
            RangeTerminalKind::Showdown,
        )
        .expect("sparse compatible ranges");
        let scores = combos
            .iter()
            .map(|combo| {
                let cards = [
                    combo.cards()[0],
                    combo.cards()[1],
                    board[0],
                    board[1],
                    board[2],
                    board[3],
                    board[4],
                ];
                (!combo.cards().iter().any(|card| board.contains(card))).then(|| evaluate(&cards))
            })
            .collect::<Vec<_>>();
        let mut expected: [Vec<f64>; 2] = std::array::from_fn(|_| vec![0.0; EXACT_COMBO_COUNT]);
        for (first, first_combo) in combos.iter().enumerate() {
            let Some(first_score) = scores[first] else {
                continue;
            };
            for (second, second_combo) in combos.iter().enumerate() {
                if first_combo.overlaps(*second_combo) {
                    continue;
                }
                let Some(second_score) = scores[second] else {
                    continue;
                };
                let equity = match first_score.cmp(&second_score) {
                    std::cmp::Ordering::Greater => 1.0,
                    std::cmp::Ordering::Equal => 0.5,
                    std::cmp::Ordering::Less => 0.0,
                };
                let utility = equity * 11.0 - (1.0 - equity) * 7.0;
                expected[0][first] += ranges[1][second] * utility;
                expected[1][second] -= ranges[0][first] * utility;
            }
        }
        for player in 0..2 {
            for (actual, expected) in measured.counterfactual_values_bb[player]
                .iter()
                .zip(&expected[player])
            {
                assert!((actual - expected).abs() <= 1e-9);
            }
        }
        assert!(measured.zero_sum_residual_bb <= 1e-12);
    }

    #[test]
    fn action_range_propagation_conserves_actor_reach() {
        let mut ranges = std::array::from_fn(|_| vec![0.0; EXACT_COMBO_COUNT]);
        ranges[0][10] = 0.25;
        ranges[0][20] = 0.75;
        ranges[1][30] = 1.0;
        let mut probabilities = vec![vec![0.5; EXACT_COMBO_COUNT]; 2];
        probabilities[0][10] = 0.2;
        probabilities[1][10] = 0.8;
        probabilities[0][20] = 0.6;
        probabilities[1][20] = 0.4;
        let children =
            propagate_action_ranges(&ranges, 0, &probabilities).expect("normalized exact policy");
        for combo in 0..EXACT_COMBO_COUNT {
            assert!(
                (children[0][0][combo] + children[1][0][combo] - ranges[0][combo]).abs() < 1e-12
            );
            assert_eq!(children[0][1][combo], ranges[1][combo]);
            assert_eq!(children[1][1][combo], ranges[1][combo]);
        }
    }

    #[test]
    fn action_value_recombination_uses_counterfactual_reach_semantics() {
        let mut probabilities = vec![vec![0.5; EXACT_COMBO_COUNT]; 2];
        probabilities[0][10] = 0.25;
        probabilities[1][10] = 0.75;
        let mut children = vec![
            std::array::from_fn(|_| vec![0.0; EXACT_COMBO_COUNT]),
            std::array::from_fn(|_| vec![0.0; EXACT_COMBO_COUNT]),
        ];
        children[0][0][10] = 4.0;
        children[1][0][10] = 8.0;
        children[0][1][20] = -1.5;
        children[1][1][20] = 0.5;
        let combined = combine_action_counterfactual_values(0, &probabilities, &children)
            .expect("compatible child values");
        assert_eq!(combined.actor_action_values_bb[0][10], 4.0);
        assert_eq!(combined.actor_action_values_bb[1][10], 8.0);
        assert_eq!(combined.expected_counterfactual_values_bb[0][10], 7.0);
        assert_eq!(combined.expected_counterfactual_values_bb[1][20], -1.0);
    }
}
