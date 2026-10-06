//! Exact whole-continuation reuse within one frozen flop solve. The owner
//! replaces this cache whenever it replaces any continuation model.
use super::{FlopContinuationSelection, FlopResolveConfig, GameState, Terminal};
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

type Values = ([Vec<f64>; 2], f64);
const MAX_ENTRIES: usize = 128;
const MAX_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    board: Vec<u8>,
    depth: u64,
    threads: usize,
    selection: u8,
    traverser: Option<usize>,
    street: u8,
    actor: usize,
    invested: [u64; 2],
    street_invested: [u64; 2],
    last_full_raise: u64,
    aggressions: u8,
    checks: u8,
    raise_reopened: bool,
    history: Vec<String>,
    trajectory: Vec<(usize, u8, u8, u64, Option<u64>, u64)>,
    terminal: Option<(u8, usize)>,
    reaches: [Vec<u64>; 2],
}

impl Key {
    fn new(
        config: &FlopResolveConfig,
        state: &GameState,
        reaches: &[Vec<f64>; 2],
        traverser: Option<usize>,
    ) -> Self {
        Self {
            board: config.state.board.clone(),
            depth: config.game.effective_stack_bb.to_bits(),
            threads: config.threads,
            selection: config.continuation_selection as u8,
            // Mean prediction/projection does not depend on the traverser.
            traverser: if config.continuation_selection == FlopContinuationSelection::Mean {
                None
            } else {
                traverser
            },
            street: state.street as u8,
            actor: state.actor,
            invested: state.invested.map(f64::to_bits),
            street_invested: state.street_invested.map(f64::to_bits),
            last_full_raise: state.last_full_raise.to_bits(),
            aggressions: state.aggressions,
            checks: state.checks,
            raise_reopened: state.raise_reopened,
            history: state.public_history.clone(),
            trajectory: state
                .trajectory
                .iter()
                .map(|action| {
                    (
                        action.actor,
                        action.street as u8,
                        action.kind as u8,
                        action.amount_bb.to_bits(),
                        action.amount_to_bb.map(f64::to_bits),
                        action.pot_after_bb.to_bits(),
                    )
                })
                .collect(),
            terminal: match state.terminal {
                None => None,
                Some(Terminal::Fold { winner }) => Some((0, winner)),
                Some(Terminal::Showdown) => Some((1, 0)),
            },
            reaches: std::array::from_fn(|p| reaches[p].iter().map(|v| v.to_bits()).collect()),
        }
    }

    fn bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.board.capacity()
            + self.history.capacity() * std::mem::size_of::<String>()
            + self.history.iter().map(String::capacity).sum::<usize>()
            + self.trajectory.capacity()
                * std::mem::size_of::<(usize, u8, u8, u64, Option<u64>, u64)>()
            + self.reaches.iter().map(|v| v.capacity() * 8).sum::<usize>()
    }
}

#[derive(Clone)]
struct Entry {
    values: Arc<Values>,
    bytes: usize,
}

#[derive(Clone)]
pub(super) struct ContinuationCache {
    entries: BTreeMap<Key, Entry>,
    order: VecDeque<Key>,
    bytes: usize,
    max_entries: usize,
    max_bytes: usize,
}

impl Default for ContinuationCache {
    fn default() -> Self {
        Self {
            entries: BTreeMap::new(),
            order: VecDeque::new(),
            bytes: 0,
            max_entries: MAX_ENTRIES,
            max_bytes: MAX_BYTES,
        }
    }
}

impl ContinuationCache {
    pub(super) fn get_or_compute(
        &mut self,
        config: &FlopResolveConfig,
        state: &GameState,
        reaches: &[Vec<f64>; 2],
        traverser: Option<usize>,
        compute: impl FnOnce() -> Values,
    ) -> Values {
        let key = Key::new(config, state, reaches, traverser);
        if let Some(entry) = self.entries.get(&key) {
            return entry.values.as_ref().clone();
        }
        let values = Arc::new(compute());
        // Estimated retained allocations include both map/FIFO keys. Scratch,
        // models and the immutable prediction cache have separate lifetimes.
        let bytes = key.bytes() * 2
            + std::mem::size_of::<Entry>()
            + std::mem::size_of::<Values>()
            + values.0.iter().map(|v| v.capacity() * 8).sum::<usize>()
            + 128;
        if self.max_entries > 0
            && bytes <= self.max_bytes
            && values.1.is_finite()
            && values.0.iter().flatten().all(|v| v.is_finite())
        {
            while self.entries.len() >= self.max_entries || self.bytes + bytes > self.max_bytes {
                let oldest = self
                    .order
                    .pop_front()
                    .expect("continuation eviction candidate");
                self.bytes -= self
                    .entries
                    .remove(&oldest)
                    .expect("cached continuation")
                    .bytes;
            }
            self.bytes += bytes;
            self.order.push_back(key.clone());
            self.entries.insert(
                key,
                Entry {
                    values: values.clone(),
                    bytes,
                },
            );
        }
        values.as_ref().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        tests::zero_shared_value_network, uniform_range, BlueprintConfig, PublicBeliefState,
    };
    use super::*;

    fn fixture() -> FlopResolveConfig {
        let board = [0, 5, 10];
        FlopResolveConfig {
            game: BlueprintConfig::default(),
            state: PublicBeliefState::flop_start(
                board,
                1,
                [1.0, 1.0],
                std::array::from_fn(|_| uniform_range(&board)),
            ),
            iterations: 2,
            averaging_delay: 0,
            regret_matching_plus: false,
            value_network: zero_shared_value_network(),
            auxiliary_value_networks: vec![],
            continuation_selection: FlopContinuationSelection::Mean,
            threads: 8,
        }
    }

    #[test]
    fn key_preserves_board_stack_betting_recall_and_every_range_bit() {
        let config = fixture();
        let state = config.state.game_state();
        let key = Key::new(&config, &state, &config.state.ranges, None);
        assert!(key == Key::new(&config, &state, &config.state.ranges, Some(0)));
        for changed in 0..8 {
            let mut other = config.clone();
            let mut state = state.clone();
            let mut reaches = config.state.ranges.clone();
            match changed {
                0 => other.state.board.swap(0, 1),
                1 => other.game.effective_stack_bb = 50.0,
                2 => state.actor = 1 - state.actor,
                3 => state.invested[0] = f64::from_bits(state.invested[0].to_bits() + 1),
                4 => state.street_invested[0] = 0.5,
                5 => state.raise_reopened = !state.raise_reopened,
                6 => state.public_history.push("check".into()),
                _ => reaches[1][0] = f64::from_bits(reaches[1][0].to_bits() + 1),
            }
            assert!(
                key != Key::new(&other, &state, &reaches, None),
                "changed {changed}"
            );
        }
        let mut robust = config.clone();
        robust.continuation_selection = FlopContinuationSelection::OpponentPublicChoice;
        assert!(
            Key::new(&robust, &state, &config.state.ranges, Some(0))
                != Key::new(&robust, &state, &config.state.ranges, Some(1))
        );
    }

    #[test]
    fn eviction_byte_limit_and_nonfinite_bypass_do_not_change_results() {
        let config = fixture();
        let state = config.state.game_state();
        let reaches = &config.state.ranges;
        let mut cache = ContinuationCache::default();
        cache.max_entries = 1;
        let calls = std::cell::Cell::new(0);
        let compute = || {
            calls.set(calls.get() + 1);
            ([vec![0.5; 1326], vec![-0.5; 1326]], 0.125)
        };
        let expected = compute();
        calls.set(0);
        for actor in [0, 0, 1, 0] {
            let mut state = state.clone();
            state.actor = actor;
            assert_eq!(
                cache.get_or_compute(&config, &state, reaches, None, compute),
                expected
            );
            assert_eq!(cache.entries.len(), 1);
            assert!(cache.bytes <= cache.max_bytes);
        }
        assert_eq!(calls.get(), 3);
        cache = ContinuationCache::default();
        cache.max_bytes = 1;
        assert_eq!(
            cache.get_or_compute(&config, &state, reaches, None, compute),
            expected
        );
        assert!(cache.entries.is_empty());
        cache = ContinuationCache::default();
        let invalid = cache.get_or_compute(&config, &state, reaches, None, || {
            ([vec![f64::NAN], vec![0.0]], 0.0)
        });
        assert!(invalid.0[0][0].is_nan());
        assert!(cache.entries.is_empty());
    }
}
