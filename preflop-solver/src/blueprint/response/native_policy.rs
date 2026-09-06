//! Full-hand research adapter: strict frozen preflop + a pinned native flop
//! tree and its evaluated turn/river continuations. No serving activation or
//! exploitability claim. Re-solving remains unconstrained by opponent CFVs.
use super::frozen_preflop::FrozenPreflopPolicy;
use super::*;
use crate::blueprint::neural::{normalize_ranges_for_board, trajectory_action_matches};
use crate::blueprint::public_belief::counterfactual_turn::{
    NativeFlopOptions, NativePostflopPolicy,
};
use crate::blueprint::public_belief::PublicBeliefState;
use std::sync::Mutex;
mod pilot;

pub(super) struct NativeFullHandPolicy {
    preflop: Arc<FrozenPreflopPolicy>,
    options: NativeFlopOptions,
    postflop: Mutex<Option<NativePostflopPolicy>>,
}

impl NativeFullHandPolicy {
    fn new(preflop: Arc<FrozenPreflopPolicy>, options: NativeFlopOptions) -> Self {
        Self {
            preflop,
            options,
            postflop: Mutex::new(None),
        }
    }

    // Replay only preflop. The complete frozen flop tree already contains the
    // exact ranges produced by its own actions; later queries must not re-solve
    // at a different belief or use the old sampled-flop policy for replay.
    fn flop_root(&self, state: &GameState) -> Result<GameState, String> {
        let game = &self.preflop.game;
        let mut cursor = GameState::initial(game);
        for observed in &state.trajectory {
            if cursor.street == Street::Flop {
                break;
            }
            if cursor.terminal.is_some() {
                return Err("native full-hand query follows a terminal preflop line".into());
            }
            let action = cursor
                .legal_actions(game)
                .into_iter()
                .find(|a| trajectory_action_matches(&cursor, a, observed, game))
                .ok_or("native full-hand query contains an illegal preflop line")?;
            cursor = cursor.apply(&action, game);
        }
        if cursor.street != Street::Flop
            || cursor.terminal.is_some()
            || !state.public_history.starts_with(&cursor.public_history)
        {
            return Err("native full-hand query has no live flop ancestor".into());
        }
        Ok(cursor)
    }

    fn root_input(&self, root: &GameState, board: &[u8]) -> Result<PublicBeliefState, String> {
        let game = &self.preflop.game;
        let mut cursor = GameState::initial(game);
        let mut ranges = [vec![1.0 / 1326.0; 1326], vec![1.0 / 1326.0; 1326]];
        for observed in &root.trajectory {
            let actions = cursor.legal_actions(game);
            let selected = actions
                .iter()
                .position(|a| trajectory_action_matches(&cursor, a, observed, game))
                .ok_or("native full-hand range replay contains an illegal action")?;
            for combo in all_combos() {
                if ranges[cursor.actor][combo.key()] > 0.0 {
                    ranges[cursor.actor][combo.key()] *=
                        self.preflop.strategy(&cursor, combo)?[selected];
                }
            }
            normalize_ranges_for_board(&mut ranges, &[])?;
            cursor = cursor.apply(&actions[selected], game);
        }
        if cursor.public_history != root.public_history || cursor.street != Street::Flop {
            return Err("native full-hand preflop replay differs from root".into());
        }
        normalize_ranges_for_board(&mut ranges, board)?;
        Ok(PublicBeliefState::from_game_state(
            board.to_vec(),
            root,
            ranges,
        ))
    }

    fn query(&self, state: &GameState, visible: &[u8], combo: Combo) -> Result<Vec<f64>, String> {
        if state.terminal.is_some()
            || state.actor > 1
            || visible.len() != state.street.board_len()
            || visible.iter().any(|c| *c >= 52)
            || visible.iter().collect::<BTreeSet<_>>().len() != visible.len()
            || combo.cards()[0] == combo.cards()[1]
            || combo
                .cards()
                .iter()
                .any(|c| *c >= 52 || visible.contains(c))
        {
            return Err("invalid native full-hand observable query".into());
        }
        if state.street == Street::Preflop {
            return self.preflop.strategy(state, combo);
        }
        let root = self.flop_root(state)?;
        let board = &visible[..3];
        let mut cache = self
            .postflop
            .lock()
            .map_err(|_| "native policy cache poisoned")?;
        let hit = cache.as_ref().is_some_and(|p| {
            p.root().board == board && p.root().public_history == root.public_history
        });
        if !hit {
            // Bound memory before allocating the replacement; cache order
            // changes runtime only, not public-root seeds or policy identity.
            cache.take();
            let input = self.root_input(&root, board)?;
            let digest = Sha256::digest(serde_json::to_vec(&input).map_err(|e| e.to_string())?);
            let mut options = self.options.clone();
            options.seed ^= u64::from_le_bytes(digest[..8].try_into().unwrap());
            *cache = Some(NativePostflopPolicy::solve(
                self.preflop.game.clone(),
                input,
                &options,
            )?);
        }
        cache.as_mut().unwrap().strategy(state, visible, combo)
    }
}

impl ResponsePolicy for NativeFullHandPolicy {
    fn strategy(
        &self,
        state: &GameState,
        deal: &Deal,
        actions: &[LegalAction],
        game: &BlueprintConfig,
    ) -> Vec<f64> {
        assert_eq!(game, &self.preflop.game, "native full-hand game mismatch");
        assert_eq!(
            actions,
            state.legal_actions(game),
            "native full-hand action mismatch"
        );
        // The old evaluator trait is infallible. Abort the offline pilot on a
        // missing/invalid policy rather than fabricate or score an action.
        self.query(
            state,
            &deal.board[..state.street.board_len()],
            Combo::new(deal.holes[state.actor][0], deal.holes[state.actor][1]),
        )
        .expect("native full-hand policy unavailable; evaluation must stop")
    }

    fn take_resolution_diagnostics(&self) -> Option<serde_json::Value> {
        let cache = self.postflop.lock().unwrap();
        Some(serde_json::json!({
            "route": "research-strict-preflop-native-flop-turn-river-v1",
            "preflopRounds": self.preflop.rounds,
            "preflopNodes": self.preflop.node_count(),
            "flopIterations": self.options.iterations,
            "trainingTurnIterations": self.options.training_turn_iterations,
            "responseTurnIterations": self.options.response_turn_iterations,
            "cachedPolicySha256": cache.as_ref().map(|p| p.identity()),
            "cachedContinuationSha256": cache.as_ref().and_then(|p| p.continuation_identity()),
            "cachedTurnSolves": cache.as_ref().map(|p| p.solved_turn_roots()),
            "safeResolving": false, "releaseQualified": false,
        }))
    }

    fn parallel_copy(&self) -> Option<Box<dyn ResponsePolicy + Send>> {
        Some(Box::new(Self::new(
            self.preflop.clone(),
            self.options.clone(),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Synthetic positive averages solely for interface tests, never exported
    // as a model or used by a quality evaluation.
    fn fixture() -> NativeFullHandPolicy {
        let game = BlueprintConfig {
            effective_stack_bb: 2.0,
            iterations: 4,
            averaging_delay: 0,
            ..BlueprintConfig::default()
        };
        let mut trainer = Trainer::fresh(game.clone());
        let mut pending = vec![GameState::initial(&game)];
        while let Some(state) = pending.pop() {
            if state.terminal.is_some() || state.street != Street::Preflop {
                continue;
            }
            let actions = state.legal_actions(&game);
            for combo in all_combos() {
                let deal = neural::deal_for_policy_combo_on_board(combo, state.actor, &[]).unwrap();
                let (key, descriptor, history) = information_set(&state, &deal, &game);
                trainer
                    .public_histories
                    .insert(descriptor.public_history_id, history);
                let mut node = Node::new(descriptor, &actions, &mut trainer.string_interner);
                node.strategy_sum.fill(1.0);
                node.average_visits = 1;
                node.regret_updates = 1;
                trainer.nodes.insert(key, node);
            }
            pending.extend(actions.iter().map(|a| state.apply(a, &game)));
        }
        let path = std::env::temp_dir().join(format!(
            "native-full-hand-fixture-{}.msgpack",
            std::process::id()
        ));
        trainer.write_checkpoint(&path).unwrap();
        let preflop = Arc::new(FrozenPreflopPolicy::read(&path).unwrap());
        fs::remove_file(path).unwrap();
        NativeFullHandPolicy::new(
            preflop,
            NativeFlopOptions {
                seed: 100101,
                iterations: 2,
                training_turn_iterations: 4,
                response_turn_iterations: 4,
            },
        )
    }

    #[test]
    fn native_full_hand_uses_one_observable_policy_through_showdown_and_parallel_copy() {
        let policy = fixture();
        let game = &policy.preflop.game;
        let deal = Deal::from_sampled_cards([[51, 50], [47, 46]], [0, 5, 10, 15, 20]);
        let mut state = GameState::initial(game);
        let mut streets = BTreeSet::new();
        while state.terminal.is_none() {
            streets.insert(format!("{:?}", state.street));
            let actions = state.legal_actions(game);
            let mix = policy.strategy(&state, &deal, &actions, game);
            assert!((mix.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            let mut holes = deal.holes;
            holes[1 - state.actor] = [43, 42];
            let changed = Deal::from_sampled_cards(holes, deal.board);
            assert_eq!(mix, policy.strategy(&state, &changed, &actions, game));
            let mut board = deal.board;
            for (i, card) in board.iter_mut().enumerate().skip(state.street.board_len()) {
                *card = 24 + i as u8;
            }
            let changed = Deal::from_sampled_cards(deal.holes, board);
            assert_eq!(mix, policy.strategy(&state, &changed, &actions, game));
            if state.street == Street::Flop {
                let copy = policy.parallel_copy().unwrap();
                assert_eq!(mix, copy.strategy(&state, &deal, &actions, game));
            }
            let action = actions
                .iter()
                .find(|a| matches!(a.kind, ActionKind::Call | ActionKind::Check))
                .unwrap();
            state = state.apply(action, game);
        }
        assert_eq!(streets.len(), 4);
        assert_eq!(
            policy.take_resolution_diagnostics().unwrap()["cachedTurnSolves"],
            1
        );
        let mut rng = SplitMix64::new(77001);
        let value = baseline_rollout(&policy, GameState::initial(game), &deal, game, &mut rng);
        assert!(value.is_finite() && value.abs() <= game.effective_stack_bb);
        let state = GameState::initial(game);
        assert!(policy
            .query(&state, &[0, 5, 10], Combo::new(51, 50))
            .is_err());
    }
}
