//! Research CFR-D turn leaf. Unlike on-policy value records, trunk regret
//! updates need values for zero-own-reach deviations too. Preserve the actual
//! reaches (including zero), retain card-legal support, and use counterfactual
//! best responses only for zero-own-reach holdings. No policy is deployed here.
use super::*;
mod flop_pilot;
pub(in crate::blueprint) use flop_pilot::{NativeFlopOptions, NativePostflopPolicy};

#[derive(Debug)]
struct Values {
    /// Raw CFVs: scaled by the original opponent reach, not own reach. Board-
    /// blocked combos are mathematical zeros, not missing-node replacements.
    counterfactual_bb: [Vec<f64>; 2],
    /// Frozen-policy values for evaluation, without training's zero-own-reach
    /// completion. A response must attack this fixed policy, not re-solve it.
    profile_counterfactual_bb: [Vec<f64>; 2],
    best_response_counterfactual_bb: [Vec<f64>; 2],
    policy_sha256: String,
    completed_zero_own_reach: [usize; 2],
    /// Undefined for a public branch with zero joint reach; never a passing zero.
    conditional_response_gain_bb: Option<[f64; 2]>,
    policy_rows: usize,
}

fn solve(config: TurnRiverSolveConfig) -> Result<Values, String> {
    solve_impl(config, false).map(|(values, _)| values)
}

// Serving must retain the actual average rows evaluated below, not the
// best-response completion used only for zero-own-reach trunk updates.
fn frozen_policy(config: TurnRiverSolveConfig) -> Result<Vec<PublicBeliefStrategy>, String> {
    solve_impl(config, true).map(|(_, rows)| rows)
}

fn solve_impl(
    mut config: TurnRiverSolveConfig,
    retain_policy: bool,
) -> Result<(Values, Vec<PublicBeliefStrategy>), String> {
    let raw = config.state.ranges.clone();
    if raw
        .iter()
        .any(|r| r.len() != COMBO_COUNT || r.iter().any(|v| !v.is_finite() || *v < 0.0))
    {
        return Err("counterfactual turn needs finite nonnegative exact-combo reaches".into());
    }
    // Allocate card/strength support using board legality, independently of
    // current action reach. These construction weights never enter training.
    config.state.ranges = std::array::from_fn(|_| uniform_range(&config.state.board));
    let mut solver = TurnRiverSolver::new(config)?;
    let mut normalized = raw.clone();
    let mut totals = [0.0; 2];
    for seat in 0..2 {
        for combo in &solver.combos {
            if !solver.legal[seat][combo.key()] {
                if normalized[seat][combo.key()] != 0.0 {
                    return Err("counterfactual turn has board-blocked reach".into());
                }
            }
            totals[seat] += normalized[seat][combo.key()];
        }
        if !totals[seat].is_finite() {
            return Err("counterfactual turn reach overflow".into());
        }
        if totals[seat] > 0.0 {
            for value in &mut normalized[seat] {
                *value /= totals[seat];
            }
        }
    }
    // A player with zero total reach stays zero. Do not inject an exploration
    // range to make the subgame appear reachable or to change the opponent mix.
    solver.config.state.ranges = normalized.clone();
    solver.train();
    let rows = solver.policy_strategies();
    let policy_rows = rows.len();
    let policy_sha256 = format!("{:x}", Sha256::digest(serde_json::to_vec(&rows).unwrap()));
    solver.nodes.clear();
    solver.load_frozen_average_strategies(&rows)?;
    let retained_rows = if retain_policy {
        rows
    } else {
        drop(rows);
        Vec::new()
    };
    let root = solver.config.state.game_state();
    let profile = solver.profile_walk(root.clone(), normalized.clone(), None, None, None, true);
    let best: [Vec<f64>; 2] = std::array::from_fn(|seat| {
        solver.profile_walk(
            root.clone(),
            normalized.clone(),
            None,
            Some(seat),
            None,
            true,
        )[seat]
            .clone()
    });
    let mut completed_zero_own_reach = [0; 2];
    let counterfactual_bb = std::array::from_fn(|seat| {
        (0..COMBO_COUNT)
            .map(|combo| {
                if !solver.legal[seat][combo] {
                    return 0.0;
                }
                let value = if normalized[seat][combo] == 0.0 {
                    completed_zero_own_reach[seat] += 1;
                    best[seat][combo]
                } else {
                    profile[seat][combo]
                };
                value * totals[1 - seat]
            })
            .collect::<Vec<_>>()
    });
    let joint = joint_compatibility_mass(&normalized);
    let conditional_response_gain_bb = (joint > 0.0).then(|| {
        std::array::from_fn(|seat| {
            (0..COMBO_COUNT)
                .map(|c| normalized[seat][c] * (best[seat][c] - profile[seat][c]))
                .sum::<f64>()
                / joint
        })
    });
    if counterfactual_bb.iter().flatten().any(|v| !v.is_finite())
        || conditional_response_gain_bb
            .is_some_and(|g| g.iter().any(|v| !v.is_finite() || *v < -1e-8))
    {
        return Err("invalid counterfactual turn values or response residual".into());
    }
    Ok((
        Values {
            counterfactual_bb,
            profile_counterfactual_bb: std::array::from_fn(|seat| {
                profile[seat]
                    .iter()
                    .enumerate()
                    .map(|(c, v)| {
                        if solver.legal[seat][c] {
                            v * totals[1 - seat]
                        } else {
                            0.0
                        }
                    })
                    .collect()
            }),
            best_response_counterfactual_bb: std::array::from_fn(|seat| {
                best[seat]
                    .iter()
                    .enumerate()
                    .map(|(c, v)| {
                        if solver.legal[seat][c] {
                            v * totals[1 - seat]
                        } else {
                            0.0
                        }
                    })
                    .collect()
            }),
            policy_sha256,
            completed_zero_own_reach,
            conditional_response_gain_bb,
            policy_rows,
        },
        retained_rows,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn config() -> TurnRiverSolveConfig {
        let mut game = BlueprintConfig::default();
        game.effective_stack_bb = 2.0;
        TurnRiverSolveConfig {
            game,
            state: PublicBeliefState::turn_start(
                [0, 5, 10, 15],
                1,
                [1.0, 1.0],
                std::array::from_fn(|_| uniform_range(&[0, 5, 10, 15])),
            ),
            iterations: 4,
            averaging_delay: 0,
            river_refinement_iterations: 0,
            regret_matching_plus: false,
        }
    }

    #[test]
    fn retained_policy_is_exactly_the_average_scored_by_counterfactual_values() {
        let mut input = config();
        // Keep a legal combo outside own reach: the policy must be the frozen
        // average, not a unilateral best-response substitution for that combo.
        let combo = Combo::new(51, 50).key();
        input.state.ranges[0][combo] = 0.0;
        let control = solve(input.clone()).unwrap();
        let rows = frozen_policy(input).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(serde_json::to_vec(&rows).unwrap())),
            control.policy_sha256
        );
        assert_eq!(rows.len(), control.policy_rows);
        assert!(control.completed_zero_own_reach[0] > 0);
        for row in rows.iter().filter(|row| row.actor == 0) {
            let n = row.action_labels.len();
            // River-blocked combos legitimately have all-zero rows.
            let sum: f64 = row.probabilities[combo * n..(combo + 1) * n]
                .iter()
                .map(|v| *v as f64)
                .sum();
            assert!(sum == 0.0 || (sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn full_support_matches_frozen_served_values_and_preserves_counterfactual_scaling() {
        let input = config();
        let rows = solve_turn_river_policy_probabilities(input.clone()).unwrap();
        let served = frozen_turn_response::continuation_values(input.clone(), &rows).unwrap();
        let raw = solve(input.clone()).unwrap();
        assert_eq!(raw.completed_zero_own_reach, [0, 0]);
        assert_eq!(raw.policy_rows, rows.len());
        for seat in 0..2 {
            for combo in 0..COMBO_COUNT {
                let expected = served.conditional_values_bb[seat][combo].unwrap_or(0.0)
                    * served.opponent_compatible_mass[seat][combo];
                assert!((raw.counterfactual_bb[seat][combo] - expected).abs() < 1e-12);
            }
        }
        let mut scaled = input;
        for v in &mut scaled.state.ranges[0] {
            *v *= 0.125;
        }
        for v in &mut scaled.state.ranges[1] {
            *v *= 0.25;
        }
        let scaled = solve(scaled).unwrap();
        for seat in 0..2 {
            for combo in 0..COMBO_COUNT {
                let factor = if seat == 0 { 0.25 } else { 0.125 };
                assert!(
                    (scaled.counterfactual_bb[seat][combo]
                        - factor * raw.counterfactual_bb[seat][combo])
                        .abs()
                        < 1e-12
                );
            }
        }
    }

    #[test]
    fn zero_own_reach_gets_a_counterfactual_response_not_a_zero_value() {
        let mut input = config();
        // TcJcQcKc on board; Ac2c has a royal flush on every legal river.
        // The trunk assigned Ac2c zero probability at this public branch.
        input.state.board = vec![32, 36, 40, 44];
        let nuts = Combo::new(48, 0).key();
        let other = Combo::new(49, 45).key();
        input.state.ranges = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        input.state.ranges[0][other] = 1.0;
        input.state.ranges[1][Combo::new(41, 37).key()] = 1.0;
        let result = solve(input.clone()).unwrap();
        assert!(result.counterfactual_bb[0][nuts] >= 1.0 - 1e-10);
        assert!(result.completed_zero_own_reach[0] > 0);
        assert!(result.conditional_response_gain_bb.is_some());
        input.state.ranges[0].fill(0.0);
        let zero_branch = solve(input).unwrap();
        assert!(zero_branch.counterfactual_bb[0][nuts] >= 1.0 - 1e-10);
        assert!(zero_branch.counterfactual_bb[1].iter().all(|v| *v == 0.0));
        assert_eq!(zero_branch.conditional_response_gain_bb, None);
    }

    #[test]
    fn rejects_illegal_reaches_without_changing_the_input() {
        let mut input = config();
        input.state.ranges[0][Combo::new(0, 1).key()] = 0.01;
        assert!(solve(input).unwrap_err().contains("board-blocked"));
        let mut input = config();
        input.state.ranges[0][10] = f64::NAN;
        assert!(solve(input).is_err());
    }
}
