//! Average own realization independently of the constant private root prior.
//! The prior still enters all payoffs/regrets/Bayesian conditioning unchanged.
use super::*;

impl TurnRiverSolver {
    pub(in crate::blueprint::public_belief) fn sweep_root_realization_averages(
        &mut self,
        round: u64,
        mode: TurnRiverTrainingMode,
    ) {
        assert!(
            self.safe_root.is_none(),
            "counterfactual averages need explicit safe-gadget support"
        );
        if round <= self.config.averaging_delay {
            return;
        }
        // The ordinary second traversal discounts actor 1 before observing its
        // strategy. Do that same once-per-round discount before this sweep;
        // its later traversal sees the same bits and does not discount twice.
        for (history, node) in &mut self.nodes {
            if node.actor == 1
                && (mode == TurnRiverTrainingMode::Joint || Self::river_from_key(history).is_some())
            {
                node.discount_regrets(round, &self.config.game.dcfr);
            }
        }
        let own = std::array::from_fn(|p| {
            self.legal[p]
                .iter()
                .map(|legal| if *legal { 1.0 } else { 0.0 })
                .collect()
        });
        self.root_average_walk(self.config.state.game_state(), own, None, round, mode);
    }

    fn root_average_walk(
        &mut self,
        state: GameState,
        own: [Vec<f64>; 2],
        river: Option<u8>,
        round: u64,
        mode: TurnRiverTrainingMode,
    ) {
        if state.terminal.is_some() {
            return;
        }
        if state.street == Street::River && river.is_none() {
            for card in self.river_cards.clone() {
                let mut masked = own.clone();
                for p in 0..2 {
                    for c in &self.river_blocked_combos[card as usize] {
                        masked[p][*c] = 0.0;
                    }
                }
                self.root_average_walk(state.clone(), masked, Some(card), round, mode);
            }
            return;
        }
        let actions = state.legal_actions(&self.config.game);
        let actor = state.actor;
        let key = Self::node_key(&state, river);
        let legal = self.legal_for(river, actor).to_vec();
        let node = &self.nodes[&key];
        let frozen_turn = mode == TurnRiverTrainingMode::FrozenAverageTurnRiverRefinement
            && state.street == Street::Turn;
        let strategy = if frozen_turn {
            node.average_strategy(&legal)
        } else {
            node.strategy(&legal)
        };
        if !frozen_turn {
            let node = self.nodes.get_mut(&key).unwrap();
            for c in 0..COMBO_COUNT {
                if legal[c] {
                    for a in 0..actions.len() {
                        let offset = c * actions.len() + a;
                        node.strategy_sum[offset] += own[actor][c] * strategy[offset];
                    }
                }
            }
            node.finish_average_update(round, &self.config.game.dcfr);
        }
        for (a, action) in actions.iter().enumerate() {
            let mut child = own.clone();
            for c in 0..COMBO_COUNT {
                child[actor][c] *= strategy[c * actions.len() + a];
            }
            self.root_average_walk(
                state.apply(action, &self.config.game),
                child,
                river,
                round,
                mode,
            );
        }
    }
}

#[test]
fn root_realization_average_preserves_regrets_and_positive_prior_policies() {
    let input = super::tests::config();
    let mut original = TurnRiverSolver::new(input.clone()).unwrap();
    let mut corrected = TurnRiverSolver::new(input).unwrap();
    corrected.root_realization_averages = true;
    original.train();
    corrected.train();
    assert_eq!(original.config.state.ranges, corrected.config.state.ranges);
    assert_eq!(original.nodes.len(), corrected.nodes.len());
    for (key, old) in &original.nodes {
        let new = &corrected.nodes[key];
        assert_eq!(
            old.regrets, new.regrets,
            "regret trajectory changed at {key:?}"
        );
        assert_eq!(
            old.last_regret_discount_round,
            new.last_regret_discount_round
        );
        let legal = original.legal_for(TurnRiverSolver::river_from_key(key), old.actor);
        let a = old.average_strategy(legal);
        let b = new.average_strategy(legal);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-12);
        }
        for c in 0..COMBO_COUNT {
            for action in 0..old.action_labels.len() {
                let k = c * old.action_labels.len() + action;
                let expected = new.strategy_sum[k] * original.config.state.ranges[old.actor][c];
                assert!((old.strategy_sum[k] - expected).abs() < 1e-12);
            }
        }
    }
}
