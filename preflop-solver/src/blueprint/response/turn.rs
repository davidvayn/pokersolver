//! Bounded, deterministic complete turn/river replacement for tabular pilots.
//! Cache lifetime changes cost only: every query reconstructs the same turn
//! root from public history. Neither private cards nor the unrevealed river
//! enter the solve or its cache key. Missing descendants are errors, not a
//! switch back to the sparse blueprint.

use super::*;
use crate::blueprint::neural::{
    deal_for_policy_combo_on_board, normalize_ranges_for_board, trajectory_action_matches,
};
use crate::blueprint::public_belief::{
    self as belief, PublicBeliefState, PublicBeliefStrategy, TurnRiverSolveConfig,
};
use std::time::Instant;

const COMBOS: usize = 1326;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TurnResolveOptions {
    pub iterations: u64,
    pub safe_bilateral: bool,
    pub maximum_policy_rows: usize,
}

impl TurnResolveOptions {
    pub(super) fn validate(&self) -> Result<(), String> {
        if self.iterations < 2 || self.maximum_policy_rows == 0 {
            return Err(
                "turn resolver requires >=2 iterations and a positive complete-policy row limit"
                    .to_owned(),
            );
        }
        Ok(())
    }
}

#[derive(Default, Deserialize, Serialize)]
struct Diagnostics {
    solved_roots: u64,
    resolved_turn_decisions: u64,
    resolved_river_decisions: u64,
    zero_entry_blueprint_decisions: u64,
    solve_seconds: f64,
    maximum_policy_rows: usize,
    maximum_policy_probability_bytes: usize,
    maximum_opponent_cfv_excess_bb: f64,
    minimum_deployed_weight: Option<f64>,
}

struct Generation {
    board: Vec<u8>,
    root_history: Vec<String>,
    rows: BTreeMap<Vec<String>, PublicBeliefStrategy>,
    root_ranges: [Vec<f64>; 2],
}

struct BoardDeals {
    board: Vec<u8>,
    deals: Vec<(usize, [Deal; 2])>,
}

pub(super) struct TabularTurnPolicy {
    base: TabularResponsePolicy,
    options: TurnResolveOptions,
    generation: RefCell<Option<Generation>>,
    diagnostics: RefCell<Diagnostics>,
}

impl TabularTurnPolicy {
    pub(super) fn new(base: TabularResponsePolicy, options: TurnResolveOptions) -> Self {
        Self {
            base,
            options,
            generation: RefCell::new(None),
            diagnostics: RefCell::default(),
        }
    }

    fn turn_root(&self, state: &GameState) -> Result<GameState, String> {
        let game = &self.base.table.config;
        let mut root = GameState::initial(game);
        for observed in &state.trajectory {
            if root.street == Street::Turn {
                break;
            }
            let actions = root.legal_actions(game);
            let action = actions
                .iter()
                .find(|a| trajectory_action_matches(&root, a, observed, game))
                .ok_or("turn root could not replay the observed public action")?;
            root = root.apply(action, game);
        }
        if root.street != Street::Turn
            || root.terminal.is_some()
            || !state.public_history.starts_with(&root.public_history)
        {
            return Err("turn root is not an ancestor of this decision".to_owned());
        }
        Ok(root)
    }

    fn ranges_at_root(&self, root: &GameState, board: &[u8]) -> Result<[Vec<f64>; 2], String> {
        let game = &self.base.table.config;
        let mut cursor = GameState::initial(game);
        let mut ranges = [
            vec![1.0 / COMBOS as f64; COMBOS],
            vec![1.0 / COMBOS as f64; COMBOS],
        ];
        for observed in &root.trajectory {
            let visible = &board[..cursor.street.board_len()];
            normalize_ranges_for_board(&mut ranges, visible)?;
            let actions = cursor.legal_actions(game);
            let selected = actions
                .iter()
                .position(|a| trajectory_action_matches(&cursor, a, observed, game))
                .ok_or("turn ranges could not replay the observed public action")?;
            for combo in all_combos() {
                if ranges[cursor.actor][combo.key()] == 0.0 {
                    continue;
                }
                let synthetic = deal_for_policy_combo_on_board(combo, cursor.actor, visible)?;
                ranges[cursor.actor][combo.key()] *= self
                    .base
                    .frozen_strategy(&cursor, &synthetic, &actions, game)[selected];
            }
            cursor = cursor.apply(&actions[selected], game);
        }
        if cursor.public_history != root.public_history
            || cursor.actor != root.actor
            || cursor.street != root.street
        {
            return Err("turn range replay differs from requested root".to_owned());
        }
        normalize_ranges_for_board(&mut ranges, board)?;
        Ok(ranges)
    }

    fn anchor_rows(
        &self,
        state: GameState,
        board: &[u8],
        river: Option<u8>,
        rows: &mut Vec<PublicBeliefStrategy>,
        cached_deals: &mut Option<BoardDeals>,
    ) -> Result<(), String> {
        if state.terminal.is_some() {
            return Ok(());
        }
        if state.street == Street::River && river.is_none() {
            for card in 0..52u8 {
                if board.contains(&card) {
                    continue;
                }
                let mut next_board = board.to_vec();
                next_board.push(card);
                self.anchor_rows(state.clone(), &next_board, Some(card), rows, cached_deals)?;
            }
            return Ok(());
        }
        if rows.len() >= self.options.maximum_policy_rows {
            return Err("tabular anchor exceeds complete-policy row limit".to_owned());
        }
        let game = &self.base.table.config;
        let actions = state.legal_actions(game);
        let mut probabilities = vec![0.0; COMBOS * actions.len()];
        if cached_deals
            .as_ref()
            .is_none_or(|cached| cached.board != board)
        {
            let mut deals = Vec::new();
            for combo in all_combos() {
                if combo.cards().iter().any(|card| board.contains(card)) {
                    continue;
                }
                deals.push((
                    combo.key(),
                    [
                        deal_for_policy_combo_on_board(combo, 0, board)?,
                        deal_for_policy_combo_on_board(combo, 1, board)?,
                    ],
                ));
            }
            *cached_deals = Some(BoardDeals {
                board: board.to_vec(),
                deals,
            });
        }
        for (combo, deals) in &cached_deals
            .as_ref()
            .expect("board cache initialized")
            .deals
        {
            let mix = self
                .base
                .frozen_strategy(&state, &deals[state.actor], &actions, game);
            for (a, p) in mix.into_iter().enumerate() {
                probabilities[*combo * actions.len() + a] = p as f32;
            }
        }
        let mut key = state.public_history.clone();
        if let Some(card) = river {
            key.push(format!("chance:river:{card}"));
        }
        rows.push(PublicBeliefStrategy {
            public_history: key,
            actor: state.actor,
            action_labels: actions.iter().map(|a| a.label.clone()).collect(),
            probabilities,
            action_values_bb: None,
        });
        for action in &actions {
            self.anchor_rows(state.apply(action, game), board, river, rows, cached_deals)?;
        }
        Ok(())
    }

    fn ensure_generation(&self, root: &GameState, board: &[u8]) -> Result<(), String> {
        if self
            .generation
            .borrow()
            .as_ref()
            .is_some_and(|g| g.board == board && g.root_history == root.public_history)
        {
            return Ok(());
        }
        // Never hold an old complete subtree while constructing another.
        self.generation.borrow_mut().take();
        let started = Instant::now();
        eprintln!(
            "turn-resolve starting board={board:?} history={:?}",
            root.public_history
        );
        let ranges = self.ranges_at_root(root, board)?;
        let config = TurnRiverSolveConfig {
            game: self.base.table.config.clone(),
            state: PublicBeliefState::from_game_state(board.to_vec(), root, ranges),
            iterations: self.options.iterations,
            averaging_delay: 0,
            river_refinement_iterations: 0,
            regret_matching_plus: false,
        };
        let root_ranges = config.state.ranges.clone();
        let mut max_excess: f64 = 0.0;
        let mut min_weight: f64 = 1.0;
        let rows = if self.options.safe_bilateral {
            let mut anchor = Vec::new();
            self.anchor_rows(root.clone(), board, None, &mut anchor, &mut None)?;
            // Each step replaces exactly one complete player's policy. Do not
            // resolve separately at later rows or replace the opponent with
            // the gadget's adversarial training strategy.
            for player in [root.actor, 1 - root.actor] {
                let mut solved =
                    belief::solve_turn_river_safe_policy_for_seat(config.clone(), &anchor, player)?;
                if solved.opponent_maximum_cfv_excess_bb
                    > belief::SAFE_RESOLVE_MAXIMUM_CFV_EXCESS_BB
                {
                    return Err("tabular safe turn solve exceeds opponent-CFV limit".to_owned());
                }
                max_excess = max_excess.max(solved.opponent_maximum_cfv_excess_bb);
                min_weight = min_weight.min(solved.deployed_resolved_policy_weight);
                // Zero-entry hands are masked by the resolving player's
                // solver. Preserve their original policy for the next seat's
                // all-opponent-hands gadget. Their own reach is exactly zero,
                // so this does not change any protected CFV. Never invent a
                // new completion or replace a positive-reach hand this way.
                let original: BTreeMap<_, _> = anchor
                    .iter()
                    .filter(|r| r.actor == player)
                    .map(|r| (&r.public_history, r))
                    .collect();
                for row in &mut solved.strategies {
                    let source = original
                        .get(&row.public_history)
                        .ok_or("safe solve lost its anchor row")?;
                    if source.action_labels != row.action_labels {
                        return Err("safe zero-reach completion grid mismatch".to_owned());
                    }
                    let width = row.action_labels.len();
                    for combo in 0..COMBOS {
                        if config.state.ranges[player][combo] == 0.0 {
                            row.probabilities[combo * width..(combo + 1) * width].copy_from_slice(
                                &source.probabilities[combo * width..(combo + 1) * width],
                            );
                        }
                    }
                }
                drop(original);
                anchor.retain(|row| row.actor != player);
                anchor.extend(solved.strategies);
            }
            anchor
        } else {
            belief::solve_turn_river_policy_probabilities(config)?
        };
        if rows.is_empty() || rows.len() > self.options.maximum_policy_rows {
            return Err("turn solution exceeds complete-policy row limit or is empty".to_owned());
        }
        let mut generation = Generation {
            board: board.to_vec(),
            root_history: root.public_history.clone(),
            rows: BTreeMap::new(),
            root_ranges,
        };
        let bytes = rows
            .iter()
            .map(|r| r.probabilities.len() * std::mem::size_of::<f32>())
            .sum();
        for row in rows {
            if generation
                .rows
                .insert(row.public_history.clone(), row)
                .is_some()
            {
                return Err("duplicate complete-policy row".to_owned());
            }
        }
        let mut stats = self.diagnostics.borrow_mut();
        stats.solved_roots += 1;
        stats.solve_seconds += started.elapsed().as_secs_f64();
        stats.maximum_policy_rows = stats.maximum_policy_rows.max(generation.rows.len());
        stats.maximum_policy_probability_bytes = stats.maximum_policy_probability_bytes.max(bytes);
        stats.maximum_opponent_cfv_excess_bb = stats.maximum_opponent_cfv_excess_bb.max(max_excess);
        if self.options.safe_bilateral {
            stats.minimum_deployed_weight =
                Some(stats.minimum_deployed_weight.unwrap_or(1.0).min(min_weight));
        }
        eprintln!(
            "turn-resolve roots={} seconds={:.3} rows={} safe={} min_weight={:.5}",
            stats.solved_roots,
            started.elapsed().as_secs_f64(),
            generation.rows.len(),
            self.options.safe_bilateral,
            min_weight
        );
        *self.generation.borrow_mut() = Some(generation);
        Ok(())
    }

    fn resolved_strategy(
        &self,
        state: &GameState,
        deal: &Deal,
        actions: &[LegalAction],
    ) -> Result<Vec<f64>, String> {
        let root = self.turn_root(state)?;
        self.ensure_generation(&root, &deal.board[..4])?;
        let mut key = state.public_history.clone();
        if state.street == Street::River {
            key.push(format!("chance:river:{}", deal.board[4]));
        }
        let cache = self.generation.borrow();
        let row = cache
            .as_ref()
            .and_then(|g| g.rows.get(&key))
            .ok_or("resolved complete subtree omitted a descendant")?;
        if row.actor != state.actor
            || !row
                .action_labels
                .iter()
                .map(String::as_str)
                .eq(actions.iter().map(|a| a.label.as_str()))
            || row.probabilities.len() != COMBOS * actions.len()
        {
            return Err("resolved turn/river action grid mismatch".to_owned());
        }
        let combo = Combo::new(deal.holes[state.actor][0], deal.holes[state.actor][1]).key();
        let values = row.probabilities[combo * actions.len()..(combo + 1) * actions.len()]
            .iter()
            .map(|p| f64::from(*p))
            .collect::<Vec<_>>();
        let zero_entry =
            cache.as_ref().expect("generation present").root_ranges[state.actor][combo] == 0.0;
        if zero_entry {
            // A red-team player may force its own zero-entry private hand
            // down this line. Complete that unreachable-own-hand strategy
            // with the original policy, explicitly counted; never treat it
            // as a resolved recommendation. Positive-entry zero rows still
            // fail closed. The protected arm preserves these rows eagerly.
            self.diagnostics.borrow_mut().zero_entry_blueprint_decisions += 1;
            if values.iter().all(|p| *p == 0.0) {
                return Ok(self
                    .base
                    .strategy(state, deal, actions, &self.base.table.config));
            }
        }
        if values.iter().any(|p| !p.is_finite() || *p < 0.0) || values.iter().sum::<f64>() <= 0.0 {
            return Err("resolved legal hand has no finite strategy".to_owned());
        }
        let mut stats = self.diagnostics.borrow_mut();
        if zero_entry {
            // A protected generation eagerly kept this original anchor row.
            // Count it as completion too, not as newly solved coverage.
        } else if state.street == Street::Turn {
            stats.resolved_turn_decisions += 1;
        } else {
            stats.resolved_river_decisions += 1;
        }
        Ok(normalize_or_uniform(values))
    }
}

impl ResponsePolicy for TabularTurnPolicy {
    fn parallel_copy(&self) -> Option<Box<dyn ResponsePolicy + Send>> {
        Some(Box::new(Self::new(
            self.base.isolated_copy(),
            self.options.clone(),
        )))
    }
    fn take_raw_coverage(&self) -> [CoverageCounter; 4] {
        self.base.take_raw_coverage()
    }
    fn take_completion_coverage(&self) -> backoff::CompletionCoverage {
        self.base.take_completion_coverage()
    }
    fn absorb_worker(&self, worker: &dyn ResponsePolicy) {
        self.base.absorb_worker(worker);
        let incoming: Diagnostics = serde_json::from_value(
            worker
                .take_resolution_diagnostics()
                .expect("turn worker diagnostics"),
        )
        .expect("typed turn worker diagnostics");
        let mut own = self.diagnostics.borrow_mut();
        own.solved_roots += incoming.solved_roots;
        own.resolved_turn_decisions += incoming.resolved_turn_decisions;
        own.resolved_river_decisions += incoming.resolved_river_decisions;
        own.zero_entry_blueprint_decisions += incoming.zero_entry_blueprint_decisions;
        own.solve_seconds += incoming.solve_seconds;
        own.maximum_policy_rows = own.maximum_policy_rows.max(incoming.maximum_policy_rows);
        own.maximum_policy_probability_bytes = own
            .maximum_policy_probability_bytes
            .max(incoming.maximum_policy_probability_bytes);
        own.maximum_opponent_cfv_excess_bb = own
            .maximum_opponent_cfv_excess_bb
            .max(incoming.maximum_opponent_cfv_excess_bb);
        if let Some(weight) = incoming.minimum_deployed_weight {
            own.minimum_deployed_weight =
                Some(own.minimum_deployed_weight.unwrap_or(1.0).min(weight));
        }
    }
    fn strategy(
        &self,
        state: &GameState,
        deal: &Deal,
        actions: &[LegalAction],
        game: &BlueprintConfig,
    ) -> Vec<f64> {
        if matches!(state.street, Street::Turn | Street::River) {
            self.resolved_strategy(state, deal, actions)
                .unwrap_or_else(|error| panic!("tabular turn resolver failed closed: {error}"))
        } else {
            self.base.strategy(state, deal, actions, game)
        }
    }
    fn take_coverage(&self) -> Vec<StreetPolicyCoverage> {
        self.base.take_coverage()
    }
    fn take_resolution_diagnostics(&self) -> Option<serde_json::Value> {
        Some(
            serde_json::to_value(std::mem::take(&mut *self.diagnostics.borrow_mut()))
                .expect("finite resolution diagnostics"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (TabularTurnPolicy, GameState, Deal) {
        let (base, deal) = super::super::tests::tabular_fixture();
        let mut root = GameState::initial(&base.table.config);
        while root.street != Street::Turn {
            let actions = root.legal_actions(&base.table.config);
            let action = actions
                .iter()
                .find(|a| matches!(a.kind, ActionKind::Call | ActionKind::Check))
                .unwrap();
            root = root.apply(action, &base.table.config);
        }
        let options = TurnResolveOptions {
            iterations: 2,
            safe_bilateral: false,
            maximum_policy_rows: 20000,
        };
        (TabularTurnPolicy::new(base, options), root, deal)
    }

    #[test]
    fn root_ranges_follow_observed_actions_not_private_cards_or_future_board() {
        let (policy, root, deal) = fixture();
        let ranges = policy.ranges_at_root(&root, &deal.board[..4]).unwrap();
        for range in &ranges {
            assert!((range.iter().sum::<f64>() - 1.0).abs() < 1e-10);
            for c in all_combos() {
                if c.cards().iter().any(|card| deal.board[..4].contains(card)) {
                    assert_eq!(range[c.key()], 0.0);
                }
            }
        }
        // Only fixture AA had nonuniform preflop probabilities; all other
        // combos used the same explicit uniform completion.
        let aa = Combo::new(51, 50).key();
        let other = Combo::new(47, 46).key();
        assert!(ranges[0][aa] > ranges[0][other]);
        assert_eq!(
            policy.turn_root(&root).unwrap().public_history,
            root.public_history
        );
    }

    #[test]
    fn complete_joint_cache_is_public_and_keeps_river_descendants() {
        let (policy, root, deal) = fixture();
        let game = &policy.base.table.config;
        let actions = root.legal_actions(game);
        let mix = policy.resolved_strategy(&root, &deal, &actions).unwrap();
        let other = deal_for_policy_combo_on_board(
            Combo::new(deal.holes[root.actor][0], deal.holes[root.actor][1]),
            root.actor,
            &deal.board[..4],
        )
        .unwrap();
        assert_eq!(
            mix,
            policy.resolved_strategy(&root, &other, &actions).unwrap()
        );
        assert_eq!(policy.diagnostics.borrow().solved_roots, 1);
        let mut state = root;
        while state.terminal.is_none() {
            let actions = state.legal_actions(game);
            let mix = policy.resolved_strategy(&state, &deal, &actions).unwrap();
            assert!((mix.iter().sum::<f64>() - 1.0).abs() < 1e-10);
            let action = actions
                .iter()
                .find(|a| matches!(a.kind, ActionKind::Call | ActionKind::Check))
                .unwrap();
            state = state.apply(action, game);
        }
        assert!(policy.diagnostics.borrow().resolved_river_decisions > 0);
        assert_eq!(policy.diagnostics.borrow().solved_roots, 1);
        policy.generation.borrow_mut().take();
        assert_eq!(
            mix,
            policy
                .resolved_strategy(
                    &policy.turn_root(&state).unwrap(),
                    &deal,
                    &policy.turn_root(&state).unwrap().legal_actions(game)
                )
                .unwrap()
        );
    }

    #[test]
    fn parallel_resolvers_preserve_policies_coverage_and_protection() {
        // Force every hand through a complete turn/river continuation. 33
        // deals crosses the two-worker wave boundary and leaves a partial wave.
        // Three public roots, one per 16-hand block, exercise reuse and eviction
        // without repeating 198 expensive safety solves in the release suite.
        let mut cards = SplitMix64::new(251);
        let test_cards: [_; 3] = std::array::from_fn(|_| {
            let deal = Deal::sample(&mut cards);
            (deal.holes, deal.board)
        });
        for safe_bilateral in [false, true] {
            let mut reference = None;
            for workers in [1, 2, 4] {
                let (mut policy, _, _) = fixture();
                let table = Arc::get_mut(&mut policy.base.table).unwrap();
                table.config.effective_stack_bb = 2.0;
                table.nodes.clear();
                policy.options.safe_bilateral = safe_bilateral;
                let game = policy.base.table.config.clone();
                let mut rows = Vec::new();
                parallel::for_each_deal(
                    &policy,
                    workers,
                    &mut SplitMix64::new(251),
                    33,
                    |local, _, index| {
                        let (holes, board) = test_cards[index as usize / 16];
                        let deal = Deal::from_sampled_cards(holes, board);
                        let mut state = GameState::initial(&game);
                        let mut output = Vec::new();
                        while state.terminal.is_none() {
                            let actions = state.legal_actions(&game);
                            output.push(local.strategy(&state, &deal, &actions, &game));
                            state = state.apply(
                                actions
                                    .iter()
                                    .find(|a| {
                                        matches!(a.kind, ActionKind::Call | ActionKind::Check)
                                    })
                                    .unwrap(),
                                &game,
                            );
                        }
                        output
                    },
                    |row| rows.push(row),
                );
                let mut diagnostics = policy.take_resolution_diagnostics().unwrap();
                assert_eq!(diagnostics["solved_roots"], 3);
                assert!(diagnostics["resolved_turn_decisions"].as_u64().unwrap() > 0);
                assert!(diagnostics["resolved_river_decisions"].as_u64().unwrap() > 0);
                if safe_bilateral {
                    assert!(
                        diagnostics["maximum_opponent_cfv_excess_bb"]
                            .as_f64()
                            .unwrap()
                            <= belief::SAFE_RESOLVE_MAXIMUM_CFV_EXCESS_BB
                    );
                }
                // Wall time is the only nondeterministic field for these
                // unique roots. Exact integer counts and policy bytes agree.
                diagnostics.as_object_mut().unwrap().remove("solve_seconds");
                let actual =
                    serde_json::to_vec(&(rows, policy.take_coverage(), diagnostics)).unwrap();
                if let Some(expected) = &reference {
                    assert_eq!(expected, &actual);
                } else {
                    reference = Some(actual);
                }
            }
        }
    }

    #[test]
    fn bilateral_safe_replacement_protects_both_seats_and_rejects_partial_cache() {
        let (mut policy, _, deal) = fixture();
        Arc::get_mut(&mut policy.base.table)
            .unwrap()
            .config
            .effective_stack_bb = 2.0;
        // This is a public-range/solver integration test, not training data:
        // clear the 20bb fixture row before changing the test game.
        Arc::get_mut(&mut policy.base.table).unwrap().nodes.clear();
        let game = policy.base.table.config.clone();
        let mut root = GameState::initial(&game);
        while root.street != Street::Turn {
            let actions = root.legal_actions(&game);
            root = root.apply(
                actions
                    .iter()
                    .find(|a| matches!(a.kind, ActionKind::Call | ActionKind::Check))
                    .unwrap(),
                &game,
            );
        }
        // A real frozen trunk can assign an exact hand zero probability of
        // checking its preflop BB option. It still needs a complete anchor
        // when the second safe solve considers all opponent private hands.
        let mut bb = GameState::initial(&game);
        let actions = bb.legal_actions(&game);
        bb = bb.apply(
            actions
                .iter()
                .find(|a| matches!(a.kind, ActionKind::Call))
                .unwrap(),
            &game,
        );
        let actions = bb.legal_actions(&game);
        let (key, descriptor, _) = information_set(&bb, &deal, &game);
        let mut node = Node::new(descriptor, &actions, &mut NodeStorageInterner::default());
        node.strategy_sum[actions.len() - 1] = 1.0;
        node.average_visits = 1;
        Arc::get_mut(&mut policy.base.table)
            .unwrap()
            .nodes
            .insert(key, node.into());
        policy.options.safe_bilateral = true;
        policy.ensure_generation(&root, &deal.board[..4]).unwrap();
        let stats = policy.diagnostics.borrow();
        assert!(stats.maximum_opponent_cfv_excess_bb <= belief::SAFE_RESOLVE_MAXIMUM_CFV_EXCESS_BB);
        assert!(stats.minimum_deployed_weight.is_some());
        drop(stats);
        let actions = root.legal_actions(&game);
        let mix = policy.resolved_strategy(&root, &deal, &actions).unwrap();
        let original = policy.base.frozen_strategy(&root, &deal, &actions, &game);
        assert!(mix.iter().zip(original).all(|(a, b)| (a - b).abs() < 1e-7));
        assert_eq!(
            policy.diagnostics.borrow().zero_entry_blueprint_decisions,
            1
        );
        assert_eq!(policy.diagnostics.borrow().resolved_turn_decisions, 0);
        assert!(policy
            .generation
            .borrow()
            .as_ref()
            .unwrap()
            .rows
            .values()
            .any(|r| r.actor == 0));
        assert!(policy
            .generation
            .borrow()
            .as_ref()
            .unwrap()
            .rows
            .values()
            .any(|r| r.actor == 1));
        policy
            .generation
            .borrow_mut()
            .as_mut()
            .unwrap()
            .rows
            .remove(&root.public_history);
        assert!(policy
            .resolved_strategy(&root, &deal, &root.legal_actions(&game))
            .unwrap_err()
            .contains("omitted"));
    }
}
