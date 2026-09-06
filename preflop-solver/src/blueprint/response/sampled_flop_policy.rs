//! Development-only complete flop routing through the existing frozen-policy
//! seam. The same probabilities drive play and public-range replay. This is
//! unconstrained re-solving, not an opponent-CFV-protected strategy.
use super::sampled_flop_pilot::{public_ranges, row_mix};
use super::*;
use crate::blueprint::public_belief::{sampled_flop, PublicBeliefState, PublicBeliefStrategy};
use std::sync::Mutex;
use std::time::Instant;
mod response_pilot;
mod turn_values;

#[derive(Default)]
struct Cache {
    board: Vec<u8>,
    rows: BTreeMap<Vec<String>, PublicBeliefStrategy>,
    solves: u64,
    seconds: f64,
    maximum_information_sets: usize,
}

pub(super) struct FlopResolve {
    iterations: u64,
    seed: u64,
    maximum_information_sets: usize,
    exact_terminal_chance: bool,
    cache: Mutex<Cache>,
}

impl FlopResolve {
    // Shared by live research routing and checkpoint-free public-range replay.
    // Only the visible board/history determine chance; neither holding is input.
    fn solve_public(
        &self,
        game: &BlueprintConfig,
        state: PublicBeliefState,
    ) -> Result<sampled_flop::SampledFlopSolution, String> {
        let identity = serde_json::to_vec(&(&state.board, &state.public_history))
            .map_err(|error| error.to_string())?;
        let digest = Sha256::digest(&identity);
        let config = sampled_flop::SampledFlopConfig {
            game: game.clone(),
            state,
            iterations: self.iterations,
            seed: self.seed ^ u64::from_le_bytes(digest[..8].try_into().unwrap()),
            maximum_information_sets: self.maximum_information_sets,
        };
        if self.exact_terminal_chance {
            sampled_flop::solve_with_exact_terminals(config)
        } else {
            sampled_flop::solve(config)
        }
    }

    pub(super) fn clear_cached_rows(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.rows.clear();
        cache.board.clear();
    }

    pub(super) fn new(iterations: u64, seed: u64, maximum_information_sets: usize) -> Self {
        assert!(iterations >= 2 && maximum_information_sets > 0);
        Self {
            iterations,
            seed,
            maximum_information_sets,
            exact_terminal_chance: false,
            cache: Mutex::default(),
        }
    }

    pub(super) fn new_with_exact_terminal_chance(
        iterations: u64,
        seed: u64,
        maximum_information_sets: usize,
    ) -> Self {
        let mut resolver = Self::new(iterations, seed, maximum_information_sets);
        resolver.exact_terminal_chance = true;
        resolver
    }

    pub(super) fn diagnostics(&self) -> serde_json::Value {
        let cache = self.cache.lock().unwrap();
        serde_json::json!({ "solves": cache.solves, "seconds": cache.seconds,
            "maximumInformationSets": cache.maximum_information_sets,
            "cachedRows": cache.rows.len(), "iterations": self.iterations, "seed": self.seed,
            "exactTerminalChance": self.exact_terminal_chance })
    }

    pub(super) fn strategy(
        &self,
        base: &TabularResponsePolicy,
        state: &GameState,
        deal: &Deal,
        actions: &[LegalAction],
    ) -> Result<Vec<f64>, String> {
        if state.street != Street::Flop || state.terminal.is_some() {
            return Err("sampled flop routing requires a live flop decision".to_owned());
        }
        let board = &deal.board[..3];
        let combo = Combo::new(deal.holes[state.actor][0], deal.holes[state.actor][1]);
        {
            let cache = self.cache.lock().unwrap();
            if cache.board == board {
                if let Some(row) = cache.rows.get(&state.public_history) {
                    return Ok(row_mix(row, combo, state, actions));
                }
            }
        }
        // No cache lock survives this call. Replay may recursively resolve a
        // strictly earlier flop prefix, including the other player's actions.
        // It never consults either actual holding or the future turn/river.
        let ranges = public_ranges(base, state, board)?;
        let start = Instant::now();
        let solution = self.solve_public(
            &base.table.config,
            PublicBeliefState::from_game_state(board.to_vec(), state, ranges),
        )?;
        let mix = row_mix(&solution.root, combo, state, actions);
        let mut cache = self.cache.lock().unwrap();
        cache.solves += 1;
        cache.seconds += start.elapsed().as_secs_f64();
        cache.maximum_information_sets = cache
            .maximum_information_sets
            .max(solution.information_sets);
        // Eviction affects cost only, never the public-state seed or policy.
        // Limit both board lifetime and rows under forced-deviation queries.
        if cache.board != board || cache.rows.len() >= 64 {
            cache.board = board.to_vec();
            cache.rows.clear();
        }
        cache
            .rows
            .insert(state.public_history.clone(), solution.root);
        Ok(mix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routed_flop_replays_its_own_ranges_and_ignores_hidden_cards_and_cache_order() {
        check_routed_flop(false);
    }

    #[test]
    fn exact_terminal_routed_flop_replays_ranges_without_hidden_cards_or_cache_order() {
        check_routed_flop(true);
    }

    fn check_routed_flop(exact: bool) {
        let (mut base, _) = super::super::tests::tabular_fixture();
        let table = Arc::get_mut(&mut base.table).unwrap();
        table.config.effective_stack_bb = 2.0;
        table.nodes.clear();
        let game = base.table.config.clone();
        let mut patch = flop::FlopPatch::terminal(&TerminalFlopOptions {
            equity_samples: 128,
            weight: 0.5,
        });
        patch.sampled = Some(if exact {
            FlopResolve::new_with_exact_terminal_chance(32, 87001, 300_000)
        } else {
            FlopResolve::new(32, 87001, 300_000)
        });
        let patch = Arc::new(patch);
        base.flop_patch = Some(patch.clone());
        let mut root = GameState::initial(&game);
        while root.street != Street::Flop {
            let action = root
                .legal_actions(&game)
                .into_iter()
                .find(|a| matches!(a.kind, ActionKind::Call | ActionKind::Check))
                .unwrap();
            root = root.apply(&action, &game);
        }
        let deal = Deal::from_sampled_cards([[48, 49], [44, 45]], [0, 5, 10, 15, 20]);
        let actions = root.legal_actions(&game);
        let expected = base.strategy(&root, &deal, &actions, &game);
        assert_eq!(
            expected,
            base.frozen_strategy(&root, &deal, &actions, &game)
        );
        assert!((expected.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let mut hidden = deal.holes;
        hidden[1 - root.actor] = [40, 41];
        let changed = Deal::from_sampled_cards(hidden, [0, 5, 10, 19, 24]);
        assert_eq!(
            expected,
            base.frozen_strategy(&root, &changed, &actions, &game)
        );
        let initial_ranges = public_ranges(&base, &root, &deal.board[..3]).unwrap();
        let check = actions
            .iter()
            .position(|a| a.kind == ActionKind::Check)
            .unwrap();
        let next = root.apply(&actions[check], &game);
        let replayed = public_ranges(&base, &next, &deal.board[..3]).unwrap();
        let mut exact = initial_ranges[root.actor].clone();
        for c in all_combos() {
            if exact[c.key()] > 0.0 {
                let synthetic = crate::blueprint::neural::deal_for_policy_combo_on_board(
                    c,
                    root.actor,
                    &deal.board[..3],
                )
                .unwrap();
                exact[c.key()] *= base.frozen_strategy(&root, &synthetic, &actions, &game)[check];
            }
        }
        let mass: f64 = exact.iter().sum();
        for (actual, expected) in replayed[root.actor].iter().zip(exact) {
            assert!((actual - expected / mass).abs() < 1e-12);
        }
        let next_actions = next.legal_actions(&game);
        base.strategy(&next, &deal, &next_actions, &game);
        assert_eq!(patch.sampled.as_ref().unwrap().diagnostics()["solves"], 2);
        let other_board = Deal::from_sampled_cards(deal.holes, [1, 6, 11, 16, 21]);
        base.frozen_strategy(&root, &other_board, &actions, &game);
        assert_eq!(
            expected,
            base.frozen_strategy(&root, &deal, &actions, &game)
        );
        assert_eq!(patch.sampled.as_ref().unwrap().diagnostics()["solves"], 4);
    }
}
