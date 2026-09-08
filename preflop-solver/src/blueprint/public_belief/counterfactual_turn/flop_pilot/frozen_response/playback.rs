//! Research playback of the same fixed profile as the native response audit.
//! Public state and the acting combo are the entire query interface. Neither
//! an opponent holding nor an unrevealed board can enter reconstruction.
use super::*;

fn compact_continuation_game(mut game: BlueprintConfig) -> BlueprintConfig {
    // A preflop optimizer's settings are not part of the pinned postflop
    // oracle. Preserve cards, stacks, action abstraction and all other inputs.
    game.dcfr = crate::blueprint::DcfrParameters::default();
    game.dcfr_schedule = crate::blueprint::DcfrSchedule::Fixed;
    game.dcfr_schedule_horizon = 0;
    game
}

#[test]
fn compact_continuation_does_not_inherit_preflop_update_schedule() {
    let game = BlueprintConfig { seed:27001, iterations:32, effective_stack_bb:20.0,
        ..BlueprintConfig::default() };
    let mut linear = game.clone();
    linear.dcfr_schedule = crate::blueprint::DcfrSchedule::Lcfr;
    linear.dcfr = crate::blueprint::DcfrParameters { positive_regret_exponent:1.0,
        negative_regret_exponent:1.0, strategy_exponent:1.0 };
    assert_ne!(serde_json::to_vec(&game).unwrap(),serde_json::to_vec(&linear).unwrap());
    assert_eq!(serde_json::to_vec(&game).unwrap(),
        serde_json::to_vec(&compact_continuation_game(linear)).unwrap());
    assert_eq!(serde_json::to_vec(&game).unwrap(),
        serde_json::to_vec(&compact_continuation_game(game.clone())).unwrap());
}

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
    /// Shared compact-preflop training, evaluation and full-hand playback
    /// boundary. Its postflop optimizer stays fixed when preflop training varies.
    pub fn solve_pinned_compact_continuation(
        game: BlueprintConfig, state: PublicBeliefState, options: &NativeFlopOptions,
        model: &PublicValueNetwork, root_realization_turn_averages: bool,
    ) -> Result<Self,String> {
        Self::solve_counterfactual_learned_with_turn_averages(
            compact_continuation_game(game),state,options,model,root_realization_turn_averages)
    }

    pub fn validate_learned_model(
        game: &BlueprintConfig,
        model: &PublicValueNetwork,
    ) -> Result<(), String> {
        super::super::continuation::Evaluator::Learned(model).model_sha256()?;
        if game.effective_stack_bb != model.target_scale_bb {
            return Err("learned continuation stack differs from pinned game".into());
        }
        Ok(())
    }

    /// Same learned trunk as the measured pilots; actual played turn/river
    /// policies remain native and use the independently pinned response budget.
    pub fn solve_with_learned_leaves(
        game: BlueprintConfig,
        state: PublicBeliefState,
        options: &NativeFlopOptions,
        model: &PublicValueNetwork,
    ) -> Result<Self, String> {
        Self::validate_learned_model(&game, model)?;
        if options.response_turn_iterations < 2 {
            return Err("native playback requires at least two continuation iterations".into());
        }
        let mut candidate = super::super::train_with_evaluator(
            game, state, options.seed, options.iterations,
            options.training_turn_iterations, false, 1, 1, None,
            super::super::continuation::Evaluator::Learned(model),
        )?;
        candidate.response_turn_iterations = (options.response_turn_iterations
            != candidate.turn_iterations).then_some(options.response_turn_iterations);
        Ok(Self::from_frozen(Frozen::new(&candidate)?))
    }

    /// Training-only root completion. Zero prior holdings retain trained
    /// counterfactual averages; zero ranges remain zero, never floored.
    pub fn solve_counterfactual_learned(
        game: BlueprintConfig,
        state: PublicBeliefState,
        options: &NativeFlopOptions,
        model: &PublicValueNetwork,
    ) -> Result<Self, String> {
        Self::solve_counterfactual_learned_with_turn_averages(game,state,options,model,false)
    }

    pub fn solve_counterfactual_learned_with_turn_averages(
        game: BlueprintConfig,
        state: PublicBeliefState,
        options: &NativeFlopOptions,
        model: &PublicValueNetwork,
        root_realization_turn_averages: bool,
    ) -> Result<Self, String> {
        Self::validate_learned_model(&game, model)?;
        if options.response_turn_iterations < 2 {
            return Err("invalid native counterfactual continuation budget".into());
        }
        let mut candidate = super::super::train_with_root_support(
            game, state, options.seed, options.iterations,
            options.training_turn_iterations, false, 1, 1, None,
            super::super::continuation::Evaluator::Learned(model), true,
        )?;
        candidate.root_realization_turn_averages = root_realization_turn_averages.then_some(true);
        candidate.response_turn_iterations = (options.response_turn_iterations
            != candidate.turn_iterations).then_some(options.response_turn_iterations);
        Ok(Self::from_frozen(Frozen::new(&candidate)?))
    }

    /// One uniformly proposed public turn. This is a training estimator, not
    /// a partial response report: no action maximum is taken after this draw.
    /// The turn solver completes zero-own-reach targets by counterfactual BR;
    /// positive-reach profile values use its actual native average policy.
    pub fn sampled_training_values(&self, turn: u8) -> Result<[Vec<f64>; 2], String> {
        self.sampled_native_values(turn, true)
    }

    fn sampled_native_values(&self, turn: u8, training: bool) -> Result<[Vec<f64>; 2], String> {
        Ok(self.sampled_native_targets(turn, &[training])?.remove(0))
    }

    fn sampled_native_targets(&self, turn: u8, targets: &[bool]) -> Result<Vec<Ranges>, String> {
        if !self.frozen.trunk.complete_root_support {
            return Err("preflop training requires complete counterfactual flop support".into());
        }
        let mut leaves = vec![BTreeMap::new(); targets.len()];
        for config in self.frozen.belief_queries(turn)? {
            let history = config.state.public_history.clone();
            let solved = self.frozen.solve_turn(config)?;
            for (map, training) in leaves.iter_mut().zip(targets) {
                let values = if *training { &solved.counterfactual_bb } else { &solved.profile_counterfactual_bb };
                let corrected: Ranges = std::array::from_fn(|p|
                    values[p].iter().map(|v| v * 49.0 / 45.0).collect());
                map.insert(history.clone(), (corrected.clone(), corrected));
            }
        }
        let mut results = Vec::new();
        for map in leaves {
            let mut result = self.frozen.back_up(self.root().game_state(), self.root().ranges.clone(), None, &map);
            for combo in all_combos() {
                if combo.cards().iter().any(|c| self.root().board.contains(c)) {
                    result[0][combo.key()] = 0.0;
                    result[1][combo.key()] = 0.0;
                }
            }
            results.push(result);
        }
        Ok(results)
    }

    /// Diagnostic only: both targets from the SAME native solves and predictor
    /// lottery. Returning the pair does not change training or playing policy.
    pub fn sampled_training_profile_comparison(&self, turn: u8, model: &PublicValueNetwork)
        -> Result<(Ranges, Ranges), String> {
        let native = self.sampled_native_targets(turn, &[true, false])?;
        let mut predictions = BTreeMap::new();
        for card in (0..52).filter(|c| !self.root().board.contains(c)) {
            predictions.insert(card, self.sampled_predicted_training_values(card, model)?);
        }
        Ok((corrected_turn_sample(&self.root().board, turn, &native[0], &predictions)?,
            corrected_turn_sample(&self.root().board, turn, &native[1], &predictions)?))
    }

    /// Cheap baseline observation under THIS same frozen flop policy. This is
    /// a prediction, never a native value label. Enumerating all 49 public
    /// turns supplies its exact chance mean for an unbiased native correction.
    pub fn sampled_predicted_training_values(&self, turn: u8, model: &PublicValueNetwork)
        -> Result<[Vec<f64>;2],String> {
        if !self.frozen.trunk.complete_root_support {
            return Err("predicted preflop baseline requires complete flop support".into());
        }
        Self::validate_learned_model(&self.frozen.trunk.game,model)?;
        let mut leaves=BTreeMap::new();
        for config in self.frozen.belief_queries(turn)? {
            let history=config.state.public_history.clone();
            let values=match super::super::continuation::Evaluator::Learned(model).evaluate(config)? {
                super::super::continuation::Leaf::Predicted(values)=>values,
                _=>return Err("prediction baseline unexpectedly called native solver".into()),
            };
            let corrected=std::array::from_fn(|p|values[p].iter().map(|v|v*49.0/45.0).collect::<Vec<_>>());
            leaves.insert(history,(corrected.clone(),corrected));
        }
        let mut result=self.frozen.back_up(self.root().game_state(),self.root().ranges.clone(),None,&leaves);
        for c in all_combos() { if c.cards().iter().any(|v|self.root().board.contains(v)) {
            result[0][c.key()]=0.0; result[1][c.key()]=0.0;
        }}
        Ok(result)
    }

    /// Same native target as sampled_training_values; the complete prediction
    /// mean cancels the predictor's bias. This option changes training variance,
    /// never the frozen/served flop policy or the played native continuation.
    pub fn sampled_training_values_with_turn_baseline(&self, turn: u8, model: &PublicValueNetwork)
        -> Result<[Vec<f64>;2],String> {
        let native=self.sampled_training_values(turn)?;
        let mut predictions=BTreeMap::new();
        for card in (0..52).filter(|c|!self.root().board.contains(c)) {
            predictions.insert(card,self.sampled_predicted_training_values(card,model)?);
        }
        corrected_turn_sample(&self.root().board,turn,&native,&predictions)
    }

    /// Evaluation of the actual frozen continuation, including zero-own-reach
    /// holdings. Training's best-response completions must not enter this path.
    pub fn sampled_profile_values_with_turn_baseline(&self, turn: u8, model: &PublicValueNetwork)
        -> Result<[Vec<f64>;2],String> {
        let native=self.sampled_native_values(turn,false)?;
        let mut predictions=BTreeMap::new();
        for card in (0..52).filter(|c|!self.root().board.contains(c)) {
            predictions.insert(card,self.sampled_predicted_training_values(card,model)?);
        }
        corrected_turn_sample(&self.root().board,turn,&native,&predictions)
    }

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
        let rows = self.frozen.freeze_turn(TurnRiverSolveConfig {
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

fn corrected_turn_sample(board: &[u8], turn: u8, native: &[Vec<f64>;2],
    predictions: &BTreeMap<u8,[Vec<f64>;2]>) -> Result<[Vec<f64>;2],String> {
    if board.len()!=3 || board.iter().any(|c|*c>=52)
        || board.iter().collect::<BTreeSet<_>>().len()!=3 || turn>=52 || board.contains(&turn)
        || predictions.keys().copied().collect::<BTreeSet<_>>()
            != (0..52).filter(|c|!board.contains(c)).collect::<BTreeSet<_>>()
        || native.iter().any(|v|v.len()!=COMBO_COUNT || v.iter().any(|v|!v.is_finite()))
        || predictions.values().any(|pair|pair.iter().any(|v|v.len()!=COMBO_COUNT || v.iter().any(|v|!v.is_finite()))) {
        return Err("turn correction requires native values and all 49 finite prediction vectors".into());
    }
    let mut mean=[vec![0.0;COMBO_COUNT],vec![0.0;COMBO_COUNT]];
    for values in predictions.values() { for p in 0..2 { for c in 0..COMBO_COUNT {
        mean[p][c] += values[p][c]/49.0;
    }}}
    for p in 0..2 { for c in 0..COMBO_COUNT {
        mean[p][c] += native[p][c]-predictions[&turn][p][c];
    }}
    if mean.iter().flatten().any(|v|!v.is_finite()) { return Err("turn correction overflow".into()); }
    Ok(mean)
}

#[test]
fn complete_turn_prediction_mean_cancels_predictor_bias_and_preserves_native_expectation() {
    let board=[0,1,2];
    let predictions: BTreeMap<_,_>=(3..52).map(|t|
        (t,[vec![t as f64-7.0;COMBO_COUNT],vec![7.0-t as f64;COMBO_COUNT]])).collect();
    for t in 3..52 {
        let native=[vec![t as f64;COMBO_COUNT],vec![-(t as f64);COMBO_COUNT]];
        let corrected=corrected_turn_sample(&board,t,&native,&predictions).unwrap();
        assert!((corrected[0][0]-27.0).abs()<1e-12);
        assert!((corrected[1][0]+27.0).abs()<1e-12);
    }
    let mut partial=predictions.clone(); partial.remove(&3);
    assert!(corrected_turn_sample(&board,4,&[vec![0.0;COMBO_COUNT],vec![0.0;COMBO_COUNT]],&partial).is_err());
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
    fn corrected_turn_averages_are_pinned_and_played_without_changing_flop_policy() {
        let mut model = crate::blueprint::public_belief::tests::zero_shared_value_network();
        model.schema = "hu-public-belief-combo-value-network-v4".into();
        model.value_normalization = Some("payoff-exposure".into());
        model.prediction_contract = Some("native-turn-cfv-full-stack-v1".into());
        model.artifact_sha256 = Some("e".repeat(64));
        let game = BlueprintConfig { effective_stack_bb:20.0,..BlueprintConfig::default() };
        let mut ranges = [vec![0.0;COMBO_COUNT],vec![0.0;COMBO_COUNT]];
        ranges[0][Combo::new(51,50).key()] = 1.0;
        ranges[1][Combo::new(47,46).key()] = 1.0;
        let input = PublicBeliefState::flop_start([0,5,10],1,[18.0,18.0],ranges);
        let options = NativeFlopOptions { seed:101,iterations:2,
            training_turn_iterations:2,response_turn_iterations:2 };
        let old = NativePostflopPolicy::solve_counterfactual_learned(
            game.clone(),input.clone(),&options,&model).unwrap();
        let mut fixed = NativePostflopPolicy::solve_counterfactual_learned_with_turn_averages(
            game,input,&options,&model,true).unwrap();
        assert_ne!(old.identity(),fixed.identity());
        assert_eq!(old.frozen.strategies,fixed.frozen.strategies);
        let old_packet = old.frozen.turn_packet(15).unwrap();
        let packet = fixed.frozen.turn_packet(15).unwrap();
        let changed = packet.leaves.iter().find(|leaf| old_packet.leaves.iter()
            .find(|other|other.history==leaf.history).unwrap().policy_sha256 != leaf.policy_sha256)
            .expect("sparse fixture must expose corrected turn policies");
        fixed.ensure_turn(&changed.history,15).unwrap();
        assert_eq!(fixed.continuation_identity(),Some(changed.policy_sha256.as_str()));
        let leaves = packet.leaves.iter().map(|leaf| {
            let values: Ranges = std::array::from_fn(|p|
                leaf.profile_bb[p].iter().map(|v|v*49.0/45.0).collect());
            (leaf.history.clone(),(values.clone(),values))
        }).collect();
        let mut expected = fixed.frozen.back_up(fixed.root().game_state(),fixed.root().ranges.clone(),None,&leaves);
        for c in all_combos() { if c.cards().iter().any(|v|fixed.root().board.contains(v)) {
            expected[0][c.key()]=0.0; expected[1][c.key()]=0.0;
        }}
        assert_eq!(fixed.sampled_native_values(15,false).unwrap(),expected);
        // Even a zero neural network leaves exact terminal values in the
        // prediction backup. Compare corrected estimates, not raw samples.
        let predictions = (0..52).filter(|c|!fixed.root().board.contains(c))
            .map(|c|(c,fixed.sampled_predicted_training_values(c,&model).unwrap()))
            .collect();
        let expected = corrected_turn_sample(&fixed.root().board,15,&expected,&predictions).unwrap();
        assert_eq!(crate::blueprint::preflop_continuation::continuation_target(
            &fixed,15,&model,true).unwrap(),expected,
            "corrected preflop targets must use the corrected frozen playback policy");
        let old_values = old.sampled_profile_values_with_turn_baseline(15,&model).unwrap();
        assert!(expected.iter().flatten().zip(old_values.iter().flatten())
            .any(|(a,b)|(a-b).abs()>1e-8),
            "sparse fixture must distinguish corrected from erased off-support averages");
    }

    #[test]
    fn sampled_profile_evaluation_matches_frozen_packets_not_training_completions() {
        let mut model = crate::blueprint::public_belief::tests::zero_shared_value_network();
        model.schema = "hu-public-belief-combo-value-network-v4".into();
        model.value_normalization = Some("payoff-exposure".into());
        model.prediction_contract = Some("native-turn-cfv-full-stack-v1".into());
        model.artifact_sha256 = Some("e".repeat(64));
        let game = BlueprintConfig { effective_stack_bb: 20.0, ..BlueprintConfig::default() };
        let mut ranges = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        ranges[0][Combo::new(51, 50).key()] = 1.0;
        ranges[1][Combo::new(47, 46).key()] = 1.0;
        let input = PublicBeliefState::flop_start([0, 5, 10], 1, [18.0, 18.0], ranges);
        let options = NativeFlopOptions { seed: 101, iterations: 2,
            training_turn_iterations: 2, response_turn_iterations: 2 };
        let policy = NativePostflopPolicy::solve_counterfactual_learned(game,input,&options,&model).unwrap();
        let packet = policy.frozen.turn_packet(15).unwrap();
        let leaves = packet.leaves.iter().map(|leaf| {
            let values: Ranges = std::array::from_fn(|p|
                leaf.profile_bb[p].iter().map(|v|v*49.0/45.0).collect());
            (leaf.history.clone(),(values.clone(),values))
        }).collect();
        let mut expected = policy.frozen.back_up(policy.root().game_state(),policy.root().ranges.clone(),None,&leaves);
        for c in all_combos() { if c.cards().iter().any(|v|policy.root().board.contains(v)) {
            expected[0][c.key()]=0.0; expected[1][c.key()]=0.0;
        }}
        let actual = policy.sampled_native_values(15,false).unwrap();
        assert_eq!(actual,expected);
        let training = policy.sampled_training_values(15).unwrap();
        assert!(training.iter().flatten().zip(actual.iter().flatten()).any(|(a,b)|(a-b).abs()>1e-8),
            "fixture must distinguish the actual profile from off-support training completions");
        let pair = policy.sampled_training_profile_comparison(15,&model).unwrap();
        assert_eq!(pair.0,policy.sampled_training_values_with_turn_baseline(15,&model).unwrap());
        assert_eq!(pair.1,policy.sampled_profile_values_with_turn_baseline(15,&model).unwrap());
        assert_eq!(pair.0,crate::blueprint::preflop_continuation::continuation_target(
            &policy,15,&model,false).unwrap());
        assert_eq!(pair.1,crate::blueprint::preflop_continuation::continuation_target(
            &policy,15,&model,true).unwrap(), "preflop profile targets must match frozen playback, including zero-reach hands");
        for p in 0..2 {
            for c in all_combos() {
                if policy.root().ranges[p][c.key()]>0.0 {
                    assert!((pair.0[p][c.key()]-pair.1[p][c.key()]).abs()<1e-10);
                }
            }
        }
    }

    #[test]
    fn learned_playback_matches_pilot_bytes_and_native_played_continuation() {
        let mut model = crate::blueprint::public_belief::tests::zero_shared_value_network();
        model.schema = "hu-public-belief-combo-value-network-v4".into();
        model.value_normalization = Some("payoff-exposure".into());
        model.prediction_contract = Some("native-turn-cfv-full-stack-v1".into());
        model.artifact_sha256 = Some("e".repeat(64));
        let game = BlueprintConfig { effective_stack_bb: 20.0, ..BlueprintConfig::default() };
        let mut ranges = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        let combo = Combo::new(51, 50);
        ranges[0][combo.key()] = 1.0;
        ranges[1][Combo::new(47, 46).key()] = 1.0;
        let input = PublicBeliefState::flop_start([0, 5, 10], 1, [18.0, 18.0], ranges);
        let options = NativeFlopOptions { seed: 101, iterations: 2,
            training_turn_iterations: 4, response_turn_iterations: 8 };
        let mut candidate = super::super::super::train_with_evaluator(
            game.clone(), input.clone(), options.seed, 2, 4, false, 1, 1, None,
            super::super::super::continuation::Evaluator::Learned(&model),
        ).unwrap();
        candidate.response_turn_iterations = Some(8);
        let mut playback = NativePostflopPolicy::solve_with_learned_leaves(
            game.clone(), input.clone(), &options, &model,
        ).unwrap();
        let expected = Frozen::new(&candidate).unwrap();
        assert_eq!(playback.identity(), expected.candidate_sha256);
        let turn = check(&check(&input.game_state(), &game), &game);
        playback.strategy(&turn, &[0, 5, 10, 15], combo).unwrap();
        let packet = expected.turn_packet(15).unwrap();
        let leaf = packet.leaves.iter().find(|l| l.history == turn.public_history).unwrap();
        assert_eq!(playback.continuation_identity(), Some(leaf.policy_sha256.as_str()));
        let mut wrong_game = game.clone();
        wrong_game.effective_stack_bb = 50.0;
        assert!(NativePostflopPolicy::validate_learned_model(&wrong_game, &model).is_err());
        model.artifact_sha256 = None;
        assert!(NativePostflopPolicy::validate_learned_model(&game, &model).is_err());
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
