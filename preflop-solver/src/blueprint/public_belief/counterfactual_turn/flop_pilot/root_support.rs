//! Opt-in counterfactual root support for evolving preflop ranges. Historical
//! frozen policies retain their original support/bytes. No range floor is used.
use super::*;

pub(in crate::blueprint) fn exact_flop_kernel(
    board: [u8; 3],
) -> Result<range_vector::ExactFlopTerminal, String> {
    if board.iter().any(|c| *c >= 52) || board.iter().collect::<BTreeSet<_>>().len() != 3 {
        return Err("invalid preflop continuation flop".into());
    }
    let legal = std::array::from_fn(|_| {
        all_combos()
            .iter()
            .map(|c| !c.cards().iter().any(|v| board.contains(v)))
            .collect()
    });
    let matrix = exact_flop_all_in_equities(board, &legal, 1);
    range_vector::ExactFlopTerminal::from_equities(board, &matrix)
}

impl Trunk {
    pub(super) fn new_counterfactual(
        game: BlueprintConfig,
        state: PublicBeliefState,
    ) -> Result<Self, String> {
        let mut raw = state.ranges.clone();
        if raw
            .iter()
            .any(|r| r.len() != COMBO_COUNT || r.iter().any(|v| !v.is_finite() || *v < 0.0))
        {
            return Err("counterfactual flop requires finite nonnegative exact reaches".into());
        }
        for range in &mut raw {
            for combo in all_combos() {
                if combo.cards().iter().any(|c| state.board.contains(c))
                    && range[combo.key()] != 0.0
                {
                    return Err("counterfactual flop contains board-blocked reach".into());
                }
            }
            let total: f64 = range.iter().sum();
            if !total.is_finite() {
                return Err("counterfactual flop reach overflow".into());
            }
            if total > 0.0 {
                for v in range {
                    *v /= total;
                }
            }
        }
        // Uniform weights allocate legal cards/nodes only. Replace them before
        // any regret update, feature construction, sampling or policy backup.
        let mut allocation = state;
        allocation.ranges = std::array::from_fn(|_| uniform_range(&allocation.board));
        let mut trunk = Self::new(game, allocation)?;
        trunk.state.ranges = raw;
        trunk.complete_root_support = true;
        Ok(trunk)
    }

    pub(super) fn replace_counterfactual_averages(
        &self,
        deltas: &mut BTreeMap<Vec<String>, FrozenRangeNodeDelta>,
    ) {
        fn visit(
            trunk: &Trunk,
            state: GameState,
            own: [Vec<f64>; 2],
            deltas: &mut BTreeMap<Vec<String>, FrozenRangeNodeDelta>,
        ) {
            if state.terminal.is_some() || state.street == Street::Turn {
                return;
            }
            let actions = state.legal_actions(&trunk.game);
            let node = &trunk.nodes[&state.public_history];
            let strategy = node.strategy(&trunk.legal[state.actor]);
            let delta = deltas
                .get_mut(&state.public_history)
                .expect("complete frozen update tree");
            for c in 0..COMBO_COUNT {
                for a in 0..actions.len() {
                    delta.strategy_sum[c * actions.len() + a] =
                        own[state.actor][c] * strategy[c * actions.len() + a];
                }
            }
            for (a, action) in actions.iter().enumerate() {
                let mut child = own.clone();
                for c in 0..COMBO_COUNT {
                    child[state.actor][c] *= strategy[c * actions.len() + a];
                }
                visit(trunk, state.apply(action, &trunk.game), child, deltas);
            }
        }
        // Removing each combo's constant root prior leaves its normalized
        // average unchanged where that prior is positive. At zero root prior,
        // this accumulates the actually trained counterfactual strategies;
        // it does not substitute an untrained uniform average at export time.
        let own = std::array::from_fn(|p| {
            self.legal[p]
                .iter()
                .map(|v| if *v { 1.0 } else { 0.0 })
                .collect()
        });
        visit(self, self.state.game_state(), own, deltas);
    }
}

#[test]
fn counterfactual_root_retains_zero_ranges_and_averages_actual_zero_own_strategies() {
    let game = BlueprintConfig {
        effective_stack_bb: 20.0,
        ..BlueprintConfig::default()
    };
    let mut ranges = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
    ranges[1][Combo::new(47, 46).key()] = 2.0;
    let state = PublicBeliefState::flop_start([0, 5, 10], 0, [1.0, 1.0], ranges);
    assert!(Trunk::new(game.clone(), state.clone()).is_err());
    let mut trunk = Trunk::new_counterfactual(game, state).unwrap();
    assert_eq!(trunk.state.ranges[0].iter().sum::<f64>(), 0.0);
    assert_eq!(trunk.state.ranges[1][Combo::new(47, 46).key()], 1.0);
    let c = Combo::new(51, 50).key();
    assert!(trunk.legal[0][c]);
    let root = trunk.state.public_history.clone();
    let node = trunk.nodes.get_mut(&root).unwrap();
    let n = node.action_labels.len();
    node.regrets[c * n] = 5.0;
    let mut deltas = trunk
        .nodes
        .iter()
        .map(|(h, node)| (h.clone(), FrozenRangeNodeDelta::new(node.regrets.len())))
        .collect();
    trunk.replace_counterfactual_averages(&mut deltas);
    assert_eq!(deltas[&root].strategy_sum[c * n], 1.0);
    assert!(deltas[&root].strategy_sum[c * n + 1..(c + 1) * n]
        .iter()
        .all(|v| *v == 0.0));
    assert_eq!(trunk.state.ranges[0].iter().sum::<f64>(), 0.0);
}
