//! Research playback of the same fixed profile as the native response audit.
//! Public state and the acting combo are the entire query interface. Neither
//! an opponent holding nor an unrevealed board can enter reconstruction.
use super::*;

#[derive(Clone, Debug)]
pub(in crate::blueprint) struct NativeFlopOptions {
    pub seed: u64,
    pub iterations: u64,
    pub training_turn_iterations: u64,
    pub response_turn_iterations: u64,
}

struct Generation {
    turn: u8,
    history: History,
    rows: BTreeMap<History, PublicBeliefStrategy>,
    sha256: String,
}

pub(in crate::blueprint) struct NativePostflopPolicy {
    frozen: Frozen,
    generation: Option<Generation>,
    solved_turn_roots: u64,
}

impl NativePostflopPolicy {
    pub fn solve(
        game: BlueprintConfig,
        state: PublicBeliefState,
        options: &NativeFlopOptions,
    ) -> Result<Self, String> {
        Self::solve_with_leaf_workers(game, state, options, 1)
    }

    // Execution setting only. Canonical policy bytes and root seeds remain
    // independent of worker count; the ordinary entry stays serial.
    pub fn solve_with_leaf_workers(
        game: BlueprintConfig,
        state: PublicBeliefState,
        options: &NativeFlopOptions,
        leaf_workers: usize,
    ) -> Result<Self, String> {
        if options.response_turn_iterations < 2 {
            return Err("native playback requires at least two continuation iterations".into());
        }
        let mut candidate = super::super::train_with_leaf_workers(
            game,
            state,
            options.seed,
            options.iterations,
            options.training_turn_iterations,
            false,
            1,
            leaf_workers,
        )?;
        candidate.response_turn_iterations = (options.response_turn_iterations
            != candidate.turn_iterations)
            .then_some(options.response_turn_iterations);
        Ok(Self::from_frozen(Frozen::new(&candidate)?))
    }

    pub fn from_bytes(bytes: &[u8], expected_sha256: &str) -> Result<Self, String> {
        let digest = format!("{:x}", Sha256::digest(bytes));
        if digest != expected_sha256 {
            return Err("native playback candidate hash mismatch".into());
        }
        let candidate: Solution = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        let frozen = Frozen::new(&candidate)?;
        if frozen.candidate_sha256 != digest {
            return Err("native playback candidate is not canonically serialized".into());
        }
        Ok(Self::from_frozen(frozen))
    }

    fn from_frozen(frozen: Frozen) -> Self {
        Self {
            frozen,
            generation: None,
            solved_turn_roots: 0,
        }
    }

    pub fn root(&self) -> &PublicBeliefState {
        &self.frozen.trunk.state
    }

    pub fn identity(&self) -> &str {
        &self.frozen.candidate_sha256
    }

    pub fn continuation_identity(&self) -> Option<&str> {
        self.generation
            .as_ref()
            .map(|generation| generation.sha256.as_str())
    }

    pub fn solved_turn_roots(&self) -> u64 {
        self.solved_turn_roots
    }

    // Reconstruct accounting from the captured root. Matching a history string
    // alone must not permit querying that policy with different money or actor.
    fn turn_ancestor(&self, state: &GameState) -> Result<Option<History>, String> {
        let mut cursor = self.root().game_state();
        let mut turn = None;
        loop {
            if !state.public_history.starts_with(&cursor.public_history)
                || cursor.terminal.is_some()
            {
                return Err("native playback state is outside its captured subtree".into());
            }
            if cursor.street == Street::Turn && turn.is_none() {
                turn = Some(cursor.public_history.clone());
            }
            if cursor.public_history == state.public_history {
                let public = |value: &GameState| {
                    PublicBeliefState::from_game_state(
                        self.root().board.clone(),
                        value,
                        [Vec::new(), Vec::new()],
                    )
                };
                if public(&cursor) != public(state) {
                    return Err("native playback public accounting mismatch".into());
                }
                return Ok(turn);
            }
            cursor = cursor
                .legal_actions(&self.frozen.trunk.game)
                .iter()
                .map(|action| cursor.apply(action, &self.frozen.trunk.game))
                .find(|next| state.public_history.starts_with(&next.public_history))
                .ok_or("native playback history contains an illegal action")?;
        }
    }

    fn ensure_turn(&mut self, history: &History, turn: u8) -> Result<(), String> {
        if self
            .generation
            .as_ref()
            .is_some_and(|g| g.turn == turn && &g.history == history)
        {
            return Ok(());
        }
        // Evict before solving, retaining at most one complete turn/river
        // policy. Cache order changes cost only, never the public ranges.
        self.generation.take();
        let (state, prior) = self
            .frozen
            .turns
            .get(history)
            .ok_or("missing native turn entry")?;
        let mut board = self.root().board.clone();
        board.push(turn);
        let mut ranges = prior.clone();
        for combo in all_combos() {
            if combo.cards().contains(&turn) {
                ranges[0][combo.key()] = 0.0;
                ranges[1][combo.key()] = 0.0;
            }
        }
        let rows = frozen_policy(TurnRiverSolveConfig {
            game: self.frozen.trunk.game.clone(),
            state: PublicBeliefState::from_game_state(board, state, ranges),
            iterations: self.frozen.turn_iterations,
            averaging_delay: 0,
            river_refinement_iterations: 0,
            regret_matching_plus: false,
        })?;
        let sha256 = format!("{:x}", Sha256::digest(serde_json::to_vec(&rows).unwrap()));
        let rows = rows
            .into_iter()
            .map(|row| (row.public_history.clone(), row))
            .collect();
        self.generation = Some(Generation {
            turn,
            history: history.clone(),
            rows,
            sha256,
        });
        self.solved_turn_roots += 1;
        Ok(())
    }

    pub fn strategy(
        &mut self,
        state: &GameState,
        visible_board: &[u8],
        combo: Combo,
    ) -> Result<Vec<f64>, String> {
        if state.terminal.is_some()
            || state.street == Street::Preflop
            || visible_board.len() != state.street.board_len()
            || !visible_board.starts_with(&self.root().board)
            || visible_board.iter().any(|card| *card >= 52)
            || visible_board.iter().collect::<BTreeSet<_>>().len() != visible_board.len()
            || combo.cards()[0] == combo.cards()[1]
            || combo
                .cards()
                .iter()
                .any(|card| *card >= 52 || visible_board.contains(card))
        {
            return Err(
                "native playback needs a live visible public state and legal acting combo".into(),
            );
        }
        let turn = self.turn_ancestor(state)?;
        let actions = state.legal_actions(&self.frozen.trunk.game);
        let count = actions.len();
        let mut values = if let Some(history) = turn {
            self.ensure_turn(&history, visible_board[3])?;
            let key = TurnRiverSolver::node_key(
                state,
                (state.street == Street::River).then(|| visible_board[4]),
            );
            let row = self
                .generation
                .as_ref()
                .unwrap()
                .rows
                .get(&key)
                .ok_or("native playback is missing a frozen continuation row")?;
            if row.actor != state.actor
                || row
                    .action_labels
                    .iter()
                    .map(String::as_str)
                    .ne(actions.iter().map(|a| a.label.as_str()))
            {
                return Err("native playback continuation action mismatch".into());
            }
            row.probabilities[combo.key() * count..(combo.key() + 1) * count]
                .iter()
                .map(|v| *v as f64)
                .collect::<Vec<_>>()
        } else {
            let row = self
                .frozen
                .strategies
                .get(&state.public_history)
                .ok_or("native playback is missing a frozen flop row")?;
            row[combo.key() * count..(combo.key() + 1) * count].to_vec()
        };
        let total: f64 = values.iter().sum();
        if values.iter().any(|v| !v.is_finite() || *v < 0.0) || (total - 1.0).abs() > 1e-6 {
            return Err("native playback combo has no valid frozen policy; no fallback".into());
        }
        // Flop rows were already normalized by Frozen::new. Continuation rows
        // use the exact f32-to-f64 normalization in the response evaluator.
        if state.street != Street::Flop {
            for value in &mut values {
                *value /= total;
            }
        }
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate() -> Solution {
        let board = [0, 5, 10];
        let mut game = BlueprintConfig::default();
        game.effective_stack_bb = 2.0;
        let mut state = GameState::initial(&game);
        while state.street == Street::Preflop {
            let action = state
                .legal_actions(&game)
                .into_iter()
                .find(|a| matches!(a.kind, ActionKind::Call | ActionKind::Check))
                .unwrap();
            state = state.apply(&action, &game);
        }
        super::super::super::train(
            game,
            PublicBeliefState::from_game_state(
                board.to_vec(),
                &state,
                std::array::from_fn(|_| uniform_range(&board)),
            ),
            100101,
            2,
            4,
        )
        .unwrap()
    }

    fn check(state: &GameState, game: &BlueprintConfig) -> GameState {
        let action = state
            .legal_actions(game)
            .into_iter()
            .find(|a| a.kind == ActionKind::Check)
            .unwrap();
        state.apply(&action, game)
    }

    #[test]
    fn playback_keeps_the_evaluated_flop_turn_and_river_policy_and_cache_identity() {
        let candidate = candidate();
        let bytes = serde_json::to_vec(&candidate).unwrap();
        let digest = format!("{:x}", Sha256::digest(&bytes));
        let mut policy = NativePostflopPolicy::from_bytes(&bytes, &digest).unwrap();
        assert_eq!(policy.identity(), digest);
        let combo = Combo::new(51, 50);
        let root = candidate.state.game_state();
        let expected = Frozen::new(&candidate).unwrap();
        let width = root.legal_actions(&candidate.game).len();
        assert_eq!(
            policy.strategy(&root, &[0, 5, 10], combo).unwrap(),
            expected.strategies[&root.public_history]
                [combo.key() * width..(combo.key() + 1) * width]
        );
        let turn = check(&check(&root, &candidate.game), &candidate.game);
        let mix = policy.strategy(&turn, &[0, 5, 10, 15], combo).unwrap();
        let packet = expected.turn_packet(15).unwrap();
        let leaf = packet
            .leaves
            .iter()
            .find(|leaf| leaf.history == turn.public_history)
            .unwrap();
        assert_eq!(
            policy.continuation_identity(),
            Some(leaf.policy_sha256.as_str())
        );
        let river = check(&check(&turn, &candidate.game), &candidate.game);
        let river_mix = policy.strategy(&river, &[0, 5, 10, 15, 20], combo).unwrap();
        assert!((river_mix.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(policy.solved_turn_roots(), 1);
        policy.strategy(&turn, &[0, 5, 10, 16], combo).unwrap();
        assert_eq!(policy.strategy(&turn, &[0, 5, 10, 15], combo).unwrap(), mix);
        assert_eq!(
            policy.continuation_identity(),
            Some(leaf.policy_sha256.as_str())
        );
        assert_eq!(policy.solved_turn_roots(), 3);
        assert!(policy.strategy(&root, &[0, 5, 10, 15], combo).is_err());
        assert!(policy.strategy(&turn, &[0, 5, 10, 10], combo).is_err());
        assert!(policy
            .strategy(&turn, &[0, 5, 10, 15], Combo::new(15, 50))
            .is_err());
        let mut wrong = turn.clone();
        wrong.invested[0] += 0.5;
        assert!(policy.strategy(&wrong, &[0, 5, 10, 15], combo).is_err());
        assert!(NativePostflopPolicy::from_bytes(&bytes, &"0".repeat(64)).is_err());
    }
}
