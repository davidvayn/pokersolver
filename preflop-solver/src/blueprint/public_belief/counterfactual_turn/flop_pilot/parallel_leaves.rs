//! Evaluate immutable continuation leaves concurrently; back up in original
//! action/chance order. Thread scheduling is not part of the policy identity.
use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

type Reaches = [Vec<f64>; 2];

#[derive(Default)]
pub(super) struct Diagnostics {
    pub queries: u64,
    pub zero_own: [u64; 2],
    pub zero_joint: u64,
    pub maximum_gain: f64,
}

fn collect(
    trunk: &Trunk,
    state: GameState,
    reaches: Reaches,
    leaves: &mut Vec<(GameState, Reaches)>,
) {
    if state.street == Street::Turn && state.terminal.is_none() {
        leaves.push((state, reaches));
        return;
    }
    if state.terminal.is_some() {
        return;
    }
    let actions = state.legal_actions(&trunk.game);
    let actor = state.actor;
    let strategy = trunk.nodes[&state.public_history].strategy(&trunk.legal[actor]);
    for (index, action) in actions.iter().enumerate() {
        let mut child = reaches.clone();
        for combo in 0..COMBO_COUNT {
            child[actor][combo] *= strategy[combo * actions.len() + index];
        }
        collect(trunk, state.apply(action, &trunk.game), child, leaves);
    }
}

pub(super) fn iteration(
    trunk: &mut Trunk,
    round: u64,
    sampled: &[u8],
    turn_iterations: u64,
    workers: usize,
) -> Result<Diagnostics, String> {
    // Same single discount and immutable-policy snapshot as serial iteration.
    for node in trunk.nodes.values_mut() {
        node.discount_regrets(round, &trunk.game.dcfr);
    }
    let mut leaves = Vec::new();
    collect(
        trunk,
        trunk.state.game_state(),
        trunk.state.ranges.clone(),
        &mut leaves,
    );
    let task_count = leaves.len() * sampled.len();
    let next = AtomicUsize::new(0);
    let trunk_ref = &*trunk;
    let results = std::thread::scope(|scope| {
        let handles = (0..workers.min(task_count))
            .map(|_| {
                let leaves = &leaves;
                let next = &next;
                scope.spawn(move || -> Result<Vec<(usize, Values)>, String> {
                    let mut output = Vec::new();
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        if index >= task_count {
                            break;
                        }
                        let (state, reaches) = &leaves[index / sampled.len()];
                        let turn = sampled[index % sampled.len()];
                        let mut board = trunk_ref.state.board.clone();
                        board.push(turn);
                        let mut masked = reaches.clone();
                        for range in &mut masked {
                            for combo in all_combos() {
                                if combo.cards().contains(&turn) {
                                    range[combo.key()] = 0.0;
                                }
                            }
                        }
                        let values = solve(TurnRiverSolveConfig {
                            game: trunk_ref.game.clone(),
                            state: PublicBeliefState::from_game_state(board, state, masked),
                            iterations: turn_iterations,
                            averaging_delay: 0,
                            river_refinement_iterations: 0,
                            regret_matching_plus: false,
                        })?;
                        output.push((index, values));
                    }
                    Ok(output)
                })
            })
            .collect::<Vec<_>>();
        let mut results = Vec::with_capacity(task_count);
        for handle in handles {
            results.extend(
                handle
                    .join()
                    .map_err(|_| "native continuation worker panicked")??,
            );
        }
        Ok::<_, String>(results)
    })?;
    let mut ordered = BTreeMap::from_iter(results);
    if ordered.len() != task_count {
        return Err("incomplete parallel continuation batch; no fallback".into());
    }
    let mut diagnostics = Diagnostics::default();
    let mut by_history = BTreeMap::new();
    for (leaf_index, (state, reaches)) in leaves.into_iter().enumerate() {
        let mut position = 0;
        let value = chance_batch::estimate(sampled, |_| {
            let value = ordered
                .remove(&(leaf_index * sampled.len() + position))
                .unwrap();
            position += 1;
            diagnostics.queries += 1;
            for p in 0..2 {
                diagnostics.zero_own[p] += value.completed_zero_own_reach[p] as u64;
            }
            if let Some(gain) = value.conditional_response_gain_bb {
                diagnostics.maximum_gain = diagnostics.maximum_gain.max(gain[0]).max(gain[1]);
            } else {
                diagnostics.zero_joint += 1;
            }
            value.counterfactual_bb
        });
        if by_history
            .insert(state.public_history, (reaches, value))
            .is_some()
        {
            return Err("duplicate perfect-recall continuation leaf".into());
        }
    }
    let mut deltas = BTreeMap::new();
    frozen_flop_walk::Walker {
        game: &trunk.game,
        legal: &trunk.legal,
        nodes: &trunk.nodes,
        terminal: &|state, reaches| trunk.terminal(state, reaches),
        turn: &|state, reaches, traverser| {
            assert_eq!(traverser, None);
            let (expected, values) = &by_history[&state.public_history];
            assert_eq!(
                reaches, expected,
                "parallel leaf must use the backup's exact own reaches"
            );
            values.clone()
        },
    }
    .walk(
        trunk.state.game_state(),
        trunk.state.ranges.clone(),
        None,
        true,
        &mut deltas,
    );
    trunk.apply_deltas(round, deltas);
    Ok(diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_leaf_backup_preserves_f64_accumulators_each_round() {
        let mut game = super::super::super::tests::config().game;
        game.effective_stack_bb = 4.0;
        let board = [0, 5, 10];
        let state = PublicBeliefState::flop_start(
            board,
            1,
            [1.0, 1.0],
            std::array::from_fn(|_| uniform_range(&board)),
        );
        let mut reference = Trunk::new(game.clone(), state.clone()).unwrap();
        let mut candidate = Trunk::new(game.clone(), state).unwrap();
        for (index, turn) in [33, 29, 33].into_iter().enumerate() {
            let round = index as u64 + 1;
            reference.iteration(round, &|state, reaches, traverser| {
                assert_eq!(traverser, None);
                chance_batch::estimate(&[turn], |turn| {
                    let mut masked = reaches.clone();
                    for range in &mut masked {
                        for combo in all_combos() {
                            if combo.cards().contains(&turn) {
                                range[combo.key()] = 0.0;
                            }
                        }
                    }
                    let mut public = board.to_vec();
                    public.push(turn);
                    solve(TurnRiverSolveConfig {
                        game: game.clone(),
                        state: PublicBeliefState::from_game_state(public, state, masked),
                        iterations: 4,
                        averaging_delay: 0,
                        river_refinement_iterations: 0,
                        regret_matching_plus: false,
                    })
                    .unwrap()
                    .counterfactual_bb
                })
            });
            iteration(&mut candidate, round, &[turn], 4, 4).unwrap();
            assert_eq!(candidate.nodes.len(), reference.nodes.len());
            for (key, a) in &reference.nodes {
                let b = &candidate.nodes[key];
                assert_eq!(
                    a.regrets.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    b.regrets.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
                assert_eq!(
                    a.strategy_sum
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    b.strategy_sum
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
                assert_eq!(a.last_regret_discount_round, b.last_regret_discount_round);
                assert_eq!(
                    a.last_strategy_discount_round,
                    b.last_strategy_discount_round
                );
            }
        }
    }

    #[test]
    fn parallel_native_leaves_preserve_complete_policy_and_diagnostics() {
        let mut game = super::super::super::tests::config().game;
        // Multiple continuation leaves, not just a single-call thread test.
        game.effective_stack_bb = 4.0;
        let board = [0, 5, 10];
        let state = PublicBeliefState::flop_start(
            board,
            1,
            [1.0, 1.0],
            std::array::from_fn(|_| uniform_range(&board)),
        );
        for batch in [1, 4] {
            let reference =
                train_with_sampling(game.clone(), state.clone(), 100101, 3, 4, false, batch)
                    .unwrap();
            assert!(reference.turn_queries > 3 * batch as u64);
            let expected = serde_json::to_vec(&reference).unwrap();
            for workers in [2, 4] {
                let actual = train_with_leaf_workers(
                    game.clone(),
                    state.clone(),
                    100101,
                    3,
                    4,
                    false,
                    batch,
                    workers,
                )
                .unwrap();
                assert_eq!(
                    serde_json::to_vec(&actual).unwrap(),
                    expected,
                    "workers={workers}, batch={batch}"
                );
            }
        }
        assert!(train_with_leaf_workers(game, state, 100101, 2, 4, true, 1, 2).is_err());
    }
}
