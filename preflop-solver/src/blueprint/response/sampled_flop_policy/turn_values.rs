//! Research continuation targets from a pinned public flop posterior. Replay
//! the nonterminal policies actually served, then evaluate the complete frozen
//! f32 turn/river policy. No private holding, hidden river, sparse local average,
//! old cached action payoff, or source checkpoint is needed by the generator.
use super::*;
use crate::blueprint::neural::{normalize_ranges_for_board, trajectory_action_matches};
use crate::blueprint::public_belief::{self as belief, frozen_turn_response, TurnRiverSolveConfig};

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Request {
    source_checkpoint_sha256: String,
    game: BlueprintConfig,
    flop: PublicBeliefState,
    policy_seed: u64,
    flop_iterations: u64,
    maximum_flop_information_sets: usize,
    /// Exact action labels after `flop`; must end on first live turn entry.
    flop_line: Vec<String>,
    turn_card: u8,
    turn_river_iterations: u64,
}

#[derive(Debug, Serialize)]
struct Target {
    schema: &'static str,
    request_sha256: String,
    request: Request,
    turn_state: PublicBeliefState,
    policy_rows: usize,
    policy_sha256: String,
    values: frozen_turn_response::FrozenTurnValues,
}

impl Request {
    fn root(&self) -> Result<GameState, String> {
        self.game.validate()?;
        let public = &self.flop;
        if self.source_checkpoint_sha256.len() != 64
            || !self
                .source_checkpoint_sha256
                .bytes()
                .all(|b| b.is_ascii_hexdigit())
            || self.flop_iterations < 2
            || self.turn_river_iterations < 2
            || self.maximum_flop_information_sets == 0
            || public.street != Street::Flop
            || public.actor > 1
            || public.board.len() != 3
            || public.board.iter().any(|c| *c >= 52)
            || public.board.iter().collect::<BTreeSet<_>>().len() != 3
            || self.turn_card >= 52
            || public.board.contains(&self.turn_card)
        {
            return Err("invalid rooted turn request".into());
        }
        for range in &public.ranges {
            if range.len() != 1326
                || range.iter().any(|v| !v.is_finite() || *v < 0.0)
                || (range.iter().sum::<f64>() - 1.0).abs() > 1e-10
                || all_combos().iter().any(|combo| {
                    combo.cards().iter().any(|c| public.board.contains(c))
                        && range[combo.key()] != 0.0
                })
            {
                return Err("invalid rooted flop posterior".into());
            }
        }
        let mut root = GameState::initial(&self.game);
        for observed in &public.trajectory {
            if root.terminal.is_some() {
                return Err("rooted flop trajectory continues after terminal".into());
            }
            let action = root
                .legal_actions(&self.game)
                .into_iter()
                .find(|a| trajectory_action_matches(&root, a, observed, &self.game))
                .ok_or("rooted flop trajectory is illegal")?;
            root = root.apply(&action, &self.game);
        }
        if root.terminal.is_some()
            || PublicBeliefState::from_game_state(
                public.board.clone(),
                &root,
                public.ranges.clone(),
            ) != *public
        {
            return Err("rooted flop trajectory/accounting mismatch".into());
        }
        Ok(root)
    }

    fn turn_input(&self) -> Result<TurnRiverSolveConfig, String> {
        let mut cursor = self.root()?;
        // Validate the entire requested line before any expensive solve.
        let mut steps = Vec::new();
        for label in &self.flop_line {
            if cursor.terminal.is_some() || cursor.street != Street::Flop {
                return Err("rooted line continues beyond live flop play".into());
            }
            let actions = cursor.legal_actions(&self.game);
            let selected = actions
                .iter()
                .position(|a| &a.label == label)
                .ok_or("rooted flop line contains an illegal action")?;
            let next = cursor.apply(&actions[selected], &self.game);
            steps.push((cursor, actions, selected));
            cursor = next;
        }
        if cursor.street != Street::Turn || cursor.terminal.is_some() {
            return Err("rooted flop line must end at first live turn entry".into());
        }
        let resolver = FlopResolve::new(
            self.flop_iterations,
            self.policy_seed,
            self.maximum_flop_information_sets,
        );
        let mut ranges = self.flop.ranges.clone();
        for (index, (state, actions, selected)) in steps.iter().enumerate() {
            // The saved root is already normalized at flop entry. Do not add
            // an extra normalization absent from full-history range replay.
            if index > 0 {
                normalize_ranges_for_board(&mut ranges, &self.flop.board)?;
            }
            let solution = resolver.solve_public(
                &self.game,
                PublicBeliefState::from_game_state(self.flop.board.clone(), state, ranges.clone()),
            )?;
            for combo in all_combos() {
                if ranges[state.actor][combo.key()] > 0.0 {
                    ranges[state.actor][combo.key()] *=
                        row_mix(&solution.root, combo, state, actions)[*selected];
                }
            }
        }
        let mut board = self.flop.board.clone();
        board.push(self.turn_card);
        normalize_ranges_for_board(&mut ranges, &board)?;
        Ok(TurnRiverSolveConfig {
            game: self.game.clone(),
            state: PublicBeliefState::from_game_state(board, &cursor, ranges),
            iterations: self.turn_river_iterations,
            averaging_delay: 0,
            river_refinement_iterations: 0,
            regret_matching_plus: false,
        })
    }

    fn generate(&self) -> Result<Target, String> {
        let input = self.turn_input()?;
        let rows = belief::solve_turn_river_policy_probabilities(input.clone())?;
        // Loading the frozen rows validates every descendant. This evaluates
        // the same exported probabilities as research play, with no retraining.
        let values = frozen_turn_response::continuation_values(input.clone(), &rows)?;
        Ok(Target {
            schema: "hu-rooted-flop-frozen-turn-values-v1",
            request_sha256: format!(
                "{:x}",
                Sha256::digest(serde_json::to_vec(self).map_err(|e| e.to_string())?)
            ),
            request: self.clone(),
            turn_state: input.state,
            policy_rows: rows.len(),
            policy_sha256: format!(
                "{:x}",
                Sha256::digest(serde_json::to_vec(&rows).map_err(|e| e.to_string())?)
            ),
            values,
        })
    }
}

mod tests;
