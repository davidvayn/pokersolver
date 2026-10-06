//! Bounded exact prediction reuse, scoped to one immutable continuation model.
use super::PublicValueNetwork;
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};

const MAX_ENTRIES: usize = 512;
const MAX_BYTES: usize = 48 * 1024 * 1024;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    board: Vec<u8>,
    actor: usize,
    invested: [u64; 2],
    ranges: [Vec<u64>; 2],
}

struct Entry {
    values: Arc<[Vec<f64>; 2]>,
    bytes: usize,
}

#[derive(Default)]
struct Cache {
    entries: BTreeMap<Key, Entry>,
    insertion_order: VecDeque<Key>,
    bytes: usize,
}

pub(super) struct ValueInferenceSession {
    // Owning the frozen model prevents stale entries after a model switch.
    network: PublicValueNetwork,
    cache: Mutex<Cache>,
    max_entries: usize,
    max_bytes: usize,
    #[cfg(test)]
    preparations: std::sync::atomic::AtomicUsize,
}

impl ValueInferenceSession {
    pub(super) fn new(network: PublicValueNetwork) -> Self {
        Self {
            network,
            cache: Mutex::new(Cache::default()),
            max_entries: MAX_ENTRIES,
            max_bytes: MAX_BYTES,
            #[cfg(test)]
            preparations: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub(super) fn predict(
        &self,
        board: &[u8],
        actor: usize,
        invested: [f64; 2],
        ranges: &[Vec<f64>; 2],
    ) -> [Vec<f64>; 2] {
        // Preserve every input bit; no probabilistic hash, rounded ranges,
        // epsilon pruning, private-card shortcut, or global model-independent key.
        let key = Key {
            board: board.to_vec(),
            actor,
            invested: invested.map(f64::to_bits),
            ranges: std::array::from_fn(|p| ranges[p].iter().map(|v| v.to_bits()).collect()),
        };
        let cached = self
            .cache
            .lock()
            .expect("value prediction cache")
            .entries
            .get(&key)
            .map(|entry| entry.values.clone());
        if let Some(values) = cached {
            return values.as_ref().clone();
        }

        // Card workers remain parallel: computation happens outside the lock.
        #[cfg(test)]
        self.preparations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let values = Arc::new(self.network.predict(board, actor, invested, ranges));
        // Include both retained FIFO/map keys and vector capacities, not only
        // their lengths. In-flight work/model storage are outside this budget.
        let bytes = 2
            * (std::mem::size_of::<Key>()
                + key.board.capacity()
                + key.ranges.iter().map(|r| r.capacity() * 8).sum::<usize>())
            + std::mem::size_of::<Entry>()
            + std::mem::size_of::<[Vec<f64>; 2]>()
            + values.iter().map(|v| v.capacity() * 8).sum::<usize>()
            + 128;
        if self.max_entries > 0
            && bytes <= self.max_bytes
            && values.iter().flatten().all(|v| v.is_finite())
        {
            let mut cache = self.cache.lock().expect("value prediction cache");
            if let Some(entry) = cache.entries.get(&key) {
                return entry.values.as_ref().clone();
            }
            while cache.entries.len() >= self.max_entries || cache.bytes + bytes > self.max_bytes {
                let oldest = cache
                    .insertion_order
                    .pop_front()
                    .expect("cache eviction candidate");
                cache.bytes -= cache
                    .entries
                    .remove(&oldest)
                    .expect("cached prediction")
                    .bytes;
            }
            cache.bytes += bytes;
            cache.insertion_order.push_back(key.clone());
            cache.entries.insert(
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
    use super::super::tests::zero_shared_value_network;
    use super::super::{uniform_range, Combo};
    use super::*;
    use std::sync::atomic::Ordering;

    fn fixture() -> PublicValueNetwork {
        let mut network = zero_shared_value_network();
        network.context_tower[0].weights[17] = 0.125;
        network.context_tower[0].weights[19] = 0.25;
        network.query_tower[0].weights[94] = 0.5;
        network.head[0].weights[0] = 0.125;
        network.head[0].weights[1] = 0.25;
        network.validate().unwrap();
        network
    }

    #[test]
    fn duplicate_predictions_are_computed_once_without_rounding_ranges() {
        let network = fixture();
        let session = ValueInferenceSession::new(network.clone());
        let board = [0, 5, 10, 15];
        let ranges = std::array::from_fn(|_| uniform_range(&board));
        for _ in 0..3 {
            let actual = session.predict(&board, 0, [1.0, 1.0], &ranges);
            let expected = network.predict(&board, 0, [1.0, 1.0], &ranges);
            assert!(actual
                .iter()
                .flatten()
                .zip(expected.iter().flatten())
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
        assert_eq!(session.preparations.load(Ordering::Relaxed), 1);
        let mut changed = ranges.clone();
        let combo = Combo::new(51, 50).key();
        changed[1][combo] = f64::from_bits(changed[1][combo].to_bits() + 1);
        assert_eq!(
            session.predict(&board, 0, [1.0, 1.0], &changed),
            network.predict(&board, 0, [1.0, 1.0], &changed)
        );
        assert_eq!(session.preparations.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn actor_investment_board_and_model_are_not_interchangeable() {
        let network = fixture();
        let session = ValueInferenceSession::new(network.clone());
        for (board, actor, invested) in [
            ([0, 5, 10, 15], 0, [1.0, 1.0]),
            ([0, 5, 10, 15], 1, [1.0, 1.0]),
            ([0, 5, 10, 15], 1, [2.5, 4.0]),
            ([1, 6, 11, 16], 1, [2.5, 4.0]),
        ] {
            let ranges = std::array::from_fn(|_| uniform_range(&board));
            assert_eq!(
                session.predict(&board, actor, invested, &ranges),
                network.predict(&board, actor, invested, &ranges)
            );
        }
        assert_eq!(session.cache.lock().unwrap().entries.len(), 4);
        let mut other = network;
        other.head[0].biases[0] += 0.5;
        let independent = ValueInferenceSession::new(other.clone());
        let board = [0, 5, 10, 15];
        let ranges = std::array::from_fn(|_| uniform_range(&board));
        assert_eq!(
            independent.predict(&board, 0, [1.0, 1.0], &ranges),
            other.predict(&board, 0, [1.0, 1.0], &ranges)
        );
        assert_eq!(session.cache.lock().unwrap().entries.len(), 4);
    }

    #[test]
    fn eviction_and_oversized_bypass_preserve_predictions() {
        let network = fixture();
        let mut session = ValueInferenceSession::new(network.clone());
        session.max_entries = 1;
        let board = [0, 5, 10, 15];
        let ranges = std::array::from_fn(|_| uniform_range(&board));
        for actor in [0, 1, 0] {
            assert_eq!(
                session.predict(&board, actor, [1.0, 1.0], &ranges),
                network.predict(&board, actor, [1.0, 1.0], &ranges)
            );
            assert_eq!(session.cache.lock().unwrap().entries.len(), 1);
            assert!(session.cache.lock().unwrap().bytes <= session.max_bytes);
        }
        assert_eq!(session.preparations.load(Ordering::Relaxed), 3);
        session.max_bytes = 1;
        session.cache = Mutex::new(Cache::default());
        session.predict(&board, 0, [1.0, 1.0], &ranges);
        assert!(session.cache.lock().unwrap().entries.is_empty());
    }

    #[test]
    fn simultaneous_identical_queries_preserve_every_value_and_one_cache_entry() {
        let network = fixture();
        let session = ValueInferenceSession::new(network.clone());
        let board = [0, 5, 10, 15];
        let ranges = std::array::from_fn(|_| uniform_range(&board));
        std::thread::scope(|scope| {
            let workers = (0..2)
                .map(|_| scope.spawn(|| session.predict(&board, 0, [1.0, 1.0], &ranges)))
                .collect::<Vec<_>>();
            for worker in workers {
                assert_eq!(
                    worker.join().unwrap(),
                    network.predict(&board, 0, [1.0, 1.0], &ranges)
                );
            }
        });
        assert_eq!(session.cache.lock().unwrap().entries.len(), 1);
    }
}
