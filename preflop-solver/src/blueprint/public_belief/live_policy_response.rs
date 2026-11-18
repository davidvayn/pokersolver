//! Test-only read-only evaluation adapters. Never train or replace a served row.
use super::*;

pub(in crate::blueprint) fn game_state(state: &PublicBeliefState) -> GameState {
    state.game_state()
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(in crate::blueprint) struct Values {
    pub profile: [Vec<f64>; 2],
    pub best_response: [Vec<f64>; 2],
}

pub(in crate::blueprint) struct FlopTerminals(FlopSolver);
impl FlopTerminals {
    pub fn new(config: FlopResolveConfig) -> Result<Self, String> {
        FlopSolver::new(config).map(Self)
    }
    pub fn values(&self, state: &GameState, reaches: &[Vec<f64>; 2]) -> Values {
        let profile = self.0.terminal_values(state, reaches);
        Values {
            best_response: profile.clone(),
            profile,
        }
    }
}

/// Freeze every actual f64 policy row before evaluating. The callback receives
/// original public histories, never the responder's changed private range.
pub(in crate::blueprint) fn turn(
    mut config: TurnRiverSolveConfig,
    reaches: [Vec<f64>; 2],
    mut policy: impl FnMut(&GameState, &[u8]) -> Result<Vec<f64>, String>,
) -> Result<(Values, usize, f64), String> {
    // Construction support is card legality, not the candidate's own reach.
    // The actual (possibly zero) reaches are used unchanged in evaluation.
    config.state.ranges = std::array::from_fn(|_| uniform_range(&config.state.board));
    let mut solver = TurnRiverSolver::new(config)?;
    fn collect(
        solver: &mut TurnRiverSolver,
        state: GameState,
        river: Option<u8>,
        policy: &mut impl FnMut(&GameState, &[u8]) -> Result<Vec<f64>, String>,
        maximum_error: &mut f64,
    ) -> Result<(), String> {
        if state.terminal.is_some() {
            return Ok(());
        }
        if state.street == Street::River && river.is_none() {
            for card in solver.river_cards.clone() {
                collect(solver, state.clone(), Some(card), policy, maximum_error)?;
            }
            return Ok(());
        }
        let mut board = solver.config.state.board.clone();
        if let Some(card) = river {
            board.push(card);
        }
        let actions = state.legal_actions(&solver.config.game);
        let probabilities = policy(&state, &board)?;
        let n = actions.len();
        if probabilities.len() != COMBO_COUNT * n
            || probabilities.iter().any(|p| !p.is_finite() || *p < 0.0)
        {
            return Err("invalid served probability dimensions/values".into());
        }
        for combo in all_combos() {
            if combo.cards().iter().any(|card| board.contains(card)) {
                continue;
            }
            let error = (probabilities[combo.key() * n..(combo.key() + 1) * n]
                .iter()
                .sum::<f64>()
                - 1.0)
                .abs();
            *maximum_error = (*maximum_error).max(error);
            if error > 1e-6 {
                return Err("invalid served probability sum".into());
            }
        }
        let count = probabilities.len();
        let node = RangeNode {
            actor: state.actor,
            action_labels: actions.iter().map(|a| a.label.clone()).collect(),
            combo_layout: None,
            regrets: vec![0.0; count],
            strategy_sum: probabilities,
            last_regret_discount_round: 0,
            last_strategy_discount_round: 0,
        };
        if solver
            .nodes
            .insert(TurnRiverSolver::node_key(&state, river), node)
            .is_some()
        {
            return Err("duplicate served public node".into());
        }
        for action in actions {
            collect(
                solver,
                state.apply(&action, &solver.config.game),
                river,
                policy,
                maximum_error,
            )?;
        }
        Ok(())
    }
    let root = solver.config.state.game_state();
    let mut maximum_error = 0.0;
    collect(
        &mut solver,
        root.clone(),
        None,
        &mut policy,
        &mut maximum_error,
    )?;
    solver.validate_frozen_strategy_tree(root.clone(), None)?;
    let profile = solver.profile_walk(root.clone(), reaches.clone(), None, None, None, true);
    let best_response = std::array::from_fn(|seat| {
        solver.profile_walk(root.clone(), reaches.clone(), None, Some(seat), None, true)[seat]
            .clone()
    });
    Ok((
        Values {
            profile,
            best_response,
        },
        solver.nodes.len(),
        maximum_error,
    ))
}

pub(in crate::blueprint) fn metrics(
    values: &Values,
    ranges: &[Vec<f64>; 2],
) -> Result<serde_json::Value, String> {
    let joint = joint_compatibility_mass(ranges);
    if !joint.is_finite() || joint <= 0.0 {
        return Err("undefined zero-joint-reach result".into());
    }
    let aggregate = |rows: &[Vec<f64>; 2]| -> [f64; 2] {
        std::array::from_fn(|seat| {
            rows[seat]
                .iter()
                .zip(&ranges[seat])
                .map(|(v, w)| v * w)
                .sum::<f64>()
                / joint
        })
    };
    let profile = aggregate(&values.profile);
    let br = aggregate(&values.best_response);
    let gains: [f64; 2] = std::array::from_fn(|seat| br[seat] - profile[seat]);
    if profile.iter().chain(&br).any(|v| !v.is_finite())
        || profile.iter().sum::<f64>().abs() > 1e-7
        || gains.iter().any(|g| *g < -1e-7)
    {
        return Err("response violates finite/zero-sum/nonnegative-gain invariants".into());
    }
    let nash_conv = gains.iter().sum::<f64>();
    Ok(
        serde_json::json!({"profileBb":profile, "bestResponseBb":br, "gainBb":gains,
        "nashConvBb":nash_conv, "exploitabilityBb":nash_conv/2.0, "jointMass":joint}),
    )
}

#[test]
fn read_only_f64_adapter_matches_existing_frozen_turn_evaluator() {
    let game = BlueprintConfig {
        effective_stack_bb: 5.0,
        ..BlueprintConfig::default()
    };
    let state = PublicBeliefState::turn_start(
        [48, 21, 2, 31],
        1,
        [2.5, 2.5],
        [
            uniform_range(&[48, 21, 2, 31]),
            uniform_range(&[48, 21, 2, 31]),
        ],
    );
    let config = TurnRiverSolveConfig {
        game,
        state,
        iterations: 2,
        averaging_delay: 0,
        river_refinement_iterations: 0,
        regret_matching_plus: false,
    };
    let rows = solve_turn_river_policy(config.clone()).unwrap();
    let existing = frozen_turn_response::evaluate(config.clone(), &rows).unwrap();
    let map: BTreeMap<_, _> = rows.iter().map(|r| (r.public_history.clone(), r)).collect();
    let (values, nodes, error) = turn(
        config.clone(),
        config.state.ranges.clone(),
        |state, board| {
            let key = TurnRiverSolver::node_key(state, (board.len() == 5).then(|| board[4]));
            Ok(map[&key]
                .probabilities
                .iter()
                .map(|p| f64::from(*p))
                .collect())
        },
    )
    .unwrap();
    let measured = metrics(&values, &config.state.ranges).unwrap();
    assert_eq!(nodes, rows.len());
    assert!(error < 1e-6);
    for seat in 0..2 {
        assert!(
            (measured["profileBb"][seat].as_f64().unwrap() - existing.profile_bb[seat]).abs()
                < 1e-10
        );
        assert!(
            (measured["gainBb"][seat].as_f64().unwrap() - existing.gain_bb[seat]).abs() < 1e-10
        );
    }
    let mut raw = config.state.ranges.clone();
    for value in &mut raw[0] {
        *value *= 0.3;
    }
    for value in &mut raw[1] {
        *value *= 0.7;
    }
    let (scaled, _, _) = turn(config.clone(), raw.clone(), |state, board| {
        let key = TurnRiverSolver::node_key(state, (board.len() == 5).then(|| board[4]));
        Ok(map[&key]
            .probabilities
            .iter()
            .map(|p| f64::from(*p))
            .collect())
    })
    .unwrap();
    let scaled_metrics = metrics(&scaled, &raw).unwrap();
    for seat in 0..2 {
        let opponent_scale = if seat == 0 { 0.7 } else { 0.3 };
        for combo in 0..COMBO_COUNT {
            assert!(
                (scaled.profile[seat][combo] - values.profile[seat][combo] * opponent_scale).abs()
                    < 1e-10
            );
            assert!(
                (scaled.best_response[seat][combo]
                    - values.best_response[seat][combo] * opponent_scale)
                    .abs()
                    < 1e-10
            );
        }
        assert!(
            (scaled_metrics["gainBb"][seat].as_f64().unwrap() - existing.gain_bb[seat]).abs()
                < 1e-10
        );
    }
}
