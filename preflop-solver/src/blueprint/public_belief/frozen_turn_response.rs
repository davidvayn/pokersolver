//! Exact information-set best responses to the actual f32 endgame policy.
//! A conditional turn-subgame result, not a full-game exploitability bound.
use super::*;

#[derive(Debug, Serialize)]
pub(in crate::blueprint) struct FrozenTurnResponse {
    pub profile_bb: [f64; 2],
    pub best_response_bb: [f64; 2],
    pub gain_bb: [f64; 2],
    pub turn_only_gain_bb: [f64; 2],
    pub river_only_gain_bb: [f64; 2],
}

pub(in crate::blueprint) fn evaluate(
    config: TurnRiverSolveConfig,
    rows: &[PublicBeliefStrategy],
) -> Result<FrozenTurnResponse, String> {
    let mut solver = TurnRiverSolver::new(config)?;
    // This validates every descendant, loads the exported f32 probabilities,
    // and does NOT train, install a prior, or consult regrets.
    solver.load_frozen_average_strategies(rows)?;
    let reaches = solver.config.state.ranges.clone();
    let joint = joint_compatibility_mass(&reaches);
    let root = solver.config.state.game_state();
    let aggregate = |values: &[f64], player: usize| {
        values
            .iter()
            .zip(&reaches[player])
            .map(|(v, p)| v * p)
            .sum::<f64>()
            / joint
    };
    let profile = solver.profile_walk(root.clone(), reaches.clone(), None, None, None, true);
    let profile_bb = std::array::from_fn(|p| aggregate(&profile[p], p));
    let response = |street| {
        std::array::from_fn(|p| {
            let values =
                solver.profile_walk(root.clone(), reaches.clone(), None, Some(p), street, true);
            aggregate(&values[p], p)
        })
    };
    let best_response_bb: [f64; 2] = response(None);
    let turn: [f64; 2] = response(Some(Street::Turn));
    let river: [f64; 2] = response(Some(Street::River));
    let result = FrozenTurnResponse {
        profile_bb,
        best_response_bb,
        gain_bb: std::array::from_fn(|p| best_response_bb[p] - profile_bb[p]),
        turn_only_gain_bb: std::array::from_fn(|p| turn[p] - profile_bb[p]),
        river_only_gain_bb: std::array::from_fn(|p| river[p] - profile_bb[p]),
    };
    if result
        .profile_bb
        .iter()
        .chain(&result.best_response_bb)
        .any(|v| !v.is_finite())
        || result.profile_bb.iter().sum::<f64>().abs() > 1e-8
        || result
            .gain_bb
            .iter()
            .chain(&result.turn_only_gain_bb)
            .chain(&result.river_only_gain_bb)
            .any(|v| !v.is_finite() || *v < -1e-8)
    {
        return Err(
            "frozen turn response violates finite/zero-sum/best-response invariants".into(),
        );
    }
    Ok(result)
}

#[test]
fn frozen_turn_response_matches_solver_and_rejects_incomplete_policy() {
    let board = [0, 5, 10, 15];
    let mut game = BlueprintConfig::default();
    game.effective_stack_bb = 4.0;
    game.averaging_delay = 0;
    game.action_abstraction.turn_river_bet_pot_fractions = vec![1.0];
    game.action_abstraction.postflop_raise_pot_fractions = vec![1.0];
    let config = TurnRiverSolveConfig {
        game,
        state: PublicBeliefState::turn_start(
            board,
            1,
            [1.0, 1.0],
            std::array::from_fn(|_| uniform_range(&board)),
        ),
        iterations: 4,
        averaging_delay: 0,
        river_refinement_iterations: 0,
        regret_matching_plus: false,
    };
    let solution = solve_turn_river(config.clone()).unwrap();
    let before = serde_json::to_vec(&solution.strategies).unwrap();
    let actual = evaluate(config.clone(), &solution.strategies).unwrap();
    assert_eq!(before, serde_json::to_vec(&solution.strategies).unwrap());
    let expected = &solution.metrics;
    assert!((actual.profile_bb[0] - expected.profile_value_p0_bb).abs() < 1e-6);
    assert!((actual.best_response_bb[0] - expected.best_response_value_p0_bb).abs() < 1e-6);
    assert!((actual.best_response_bb[1] - expected.best_response_value_p1_bb).abs() < 1e-6);
    assert!(
        (actual.gain_bb.iter().sum::<f64>() / 2.0
            - expected.exact_abstract_exploitability_bb_per_hand)
            .abs()
            < 1e-6
    );
    assert!(
        (actual.turn_only_gain_bb.iter().sum::<f64>() / 2.0
            - expected.turn_only_best_response_gain_bb_per_hand)
            .abs()
            < 1e-6
    );
    assert!(
        (actual.river_only_gain_bb.iter().sum::<f64>() / 2.0
            - expected.river_only_best_response_gain_bb_per_hand)
            .abs()
            < 1e-6
    );
    assert!(actual.gain_bb.iter().all(|v| *v > 0.0));
    assert!(evaluate(config.clone(), &solution.strategies[1..]).is_err());
    let mut malformed = solution.strategies;
    malformed[0].probabilities[0] = f32::NAN;
    assert!(evaluate(config, &malformed).is_err());
}
