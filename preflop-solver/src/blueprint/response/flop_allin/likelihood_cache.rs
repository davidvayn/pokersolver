//! Bounded memoization of public action likelihoods, not hero-conditioned ranges.
//! Weak source identities prevent cross-model reuse without retaining checkpoints.
use super::*;
use std::sync::{Mutex, OnceLock, Weak};

type Key = (Vec<u8>, Vec<String>, usize);
type Row = Arc<Vec<OnceLock<f64>>>;

#[derive(Default)]
struct Cache {
    table: Weak<InferenceTable>,
    backoff: Option<Weak<backoff::FlopBackoff>>,
    rows: BTreeMap<Key, Row>,
}

#[derive(Default)]
pub(in crate::blueprint::response) struct LikelihoodCache(Mutex<Cache>);

impl LikelihoodCache {
    #[cfg(test)]
    pub(super) fn clear(&self) {
        *self.0.lock().unwrap() = Cache::default();
    }

    pub(super) fn row(
        &self,
        base: &TabularResponsePolicy,
        state: &GameState,
        board: &[u8],
        selected: usize,
        game: &BlueprintConfig,
    ) -> Option<Row> {
        // A differently configured caller retains the original uncached path.
        if game != &base.table.config {
            return None;
        }
        let mut cache = self.0.lock().unwrap();
        let same_table = cache
            .table
            .upgrade()
            .is_some_and(|t| Arc::ptr_eq(&t, &base.table));
        let same_backoff = match (&cache.backoff, &base.flop_backoff) {
            (None, None) => true,
            (Some(old), Some(current)) => old.upgrade().is_some_and(|b| Arc::ptr_eq(&b, current)),
            _ => false,
        };
        if !same_table || !same_backoff {
            cache.rows.clear();
            cache.table = Arc::downgrade(&base.table);
            cache.backoff = base.flop_backoff.as_ref().map(Arc::downgrade);
        }
        let key = (board.to_vec(), state.public_history.clone(), selected);
        if let Some(row) = cache.rows.get(&key) {
            return Some(row.clone());
        }
        // At most ~0.7MiB of cells plus small public-history keys. A live query
        // owns its row, so eviction cannot invalidate an in-flight initializer.
        if cache.rows.len() >= 32 {
            cache.rows.pop_first();
        }
        let row = Arc::new((0..1326).map(|_| OnceLock::new()).collect::<Vec<_>>());
        cache.rows.insert(key, row.clone());
        Some(row)
        // No mutex is held during policy queries: sampled flop replay may
        // recurse into strictly earlier public prefixes.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn likelihood_cache_is_bounded_source_pinned_and_does_not_own_the_checkpoint() {
        let (base, _) = super::super::super::tests::tabular_fixture();
        let cache = LikelihoodCache::default();
        let game = &base.table.config;
        let mut state = GameState::initial(game);
        let owners = Arc::strong_count(&base.table);
        let first = cache.row(&base, &state, &[], 0, game).unwrap();
        first[0].set(0.25).unwrap();
        assert_eq!(
            cache.row(&base, &state, &[], 0, game).unwrap()[0].get(),
            Some(&0.25)
        );
        assert!(cache.row(&base, &state, &[], 1, game).unwrap()[0]
            .get()
            .is_none());
        assert!(cache.row(&base, &state, &[0, 1, 2], 0, game).unwrap()[0]
            .get()
            .is_none());
        for index in 0..40 {
            state.public_history = vec![format!("test-public-key-{index}")];
            cache.row(&base, &state, &[], 0, game).unwrap();
        }
        assert_eq!(cache.0.lock().unwrap().rows.len(), 32);
        assert_eq!(
            first[0].get(),
            Some(&0.25),
            "eviction must preserve an in-flight row"
        );
        assert_eq!(Arc::strong_count(&base.table), owners);
        let (other, _) = super::super::super::tests::tabular_fixture();
        let state = GameState::initial(&other.table.config);
        let replacement = cache
            .row(&other, &state, &[], 0, &other.table.config)
            .unwrap();
        assert!(replacement[0].get().is_none());
        assert_eq!(cache.0.lock().unwrap().rows.len(), 1);
        let mut wrong_game = other.table.config.clone();
        wrong_game.effective_stack_bb += 1.0;
        assert!(cache.row(&other, &state, &[], 0, &wrong_game).is_none());
    }
}
