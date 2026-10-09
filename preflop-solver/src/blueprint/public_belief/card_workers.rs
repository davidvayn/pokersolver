//! Reusable bounded workers for independent card work. Concurrent flop solves
//! with the same worker budget share the pool, not another N OS threads each.
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::sync::{Arc, LazyLock, Mutex, Weak};

pub(super) struct CardWorkers {
    pool: rayon::ThreadPool,
}

impl CardWorkers {
    pub(super) fn shared(requested: usize) -> Result<Arc<Self>, String> {
        if requested == 0 {
            return Err("card worker count must be positive".into());
        }
        let count = requested.min(49);
        static POOLS: LazyLock<Mutex<BTreeMap<usize, Weak<CardWorkers>>>> =
            LazyLock::new(|| Mutex::new(BTreeMap::new()));
        let mut pools = POOLS.lock().expect("card worker registry");
        if let Some(pool) = pools.get(&count).and_then(Weak::upgrade) {
            return Ok(pool);
        }
        pools.retain(|_, pool| pool.strong_count() > 0);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(count)
            .thread_name(move |index| format!("flop-card-{count}-{index}"))
            .build()
            .map_err(|cause| format!("could not start card workers: {cause}"))?;
        let workers = Arc::new(Self { pool });
        pools.insert(count, Arc::downgrade(&workers));
        Ok(workers)
    }

    /// Tasks may finish in any order, but returned results retain input order.
    /// Callers keep their existing deterministic floating-point fold order.
    pub(super) fn map<T: Sync, R: Send>(
        &self,
        inputs: &[T],
        compute: impl Fn(&T) -> R + Sync + Send,
    ) -> Vec<R> {
        self.pool
            .install(|| inputs.par_iter().with_max_len(1).map(compute).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Barrier,
    };

    #[test]
    fn eight_workers_execute_independent_tasks_and_preserve_order() {
        let workers = CardWorkers::shared(8).unwrap();
        assert!(Arc::ptr_eq(&workers, &CardWorkers::shared(8).unwrap()));
        let barrier = Barrier::new(8);
        let ids = Mutex::new(HashSet::new());
        let values = workers.map(&(0..32).collect::<Vec<_>>(), |value| {
            ids.lock().unwrap().insert(std::thread::current().id());
            barrier.wait();
            barrier.wait();
            value * value
        });
        assert_eq!(values, (0..32).map(|v| v * v).collect::<Vec<_>>());
        assert_eq!(ids.lock().unwrap().len(), 8);
    }

    #[test]
    fn concurrent_callers_share_one_worker_budget_and_pool_survives_panics() {
        let workers = CardWorkers::shared(4).unwrap();
        let active = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let ids = Mutex::new(HashSet::new());
        let barrier = Barrier::new(4);
        let compute = |value: &usize| {
            ids.lock().unwrap().insert(std::thread::current().id());
            let current = active.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(current, Ordering::SeqCst);
            barrier.wait();
            active.fetch_sub(1, Ordering::SeqCst);
            barrier.wait();
            value * value
        };
        let inputs = (0..8).collect::<Vec<_>>();
        std::thread::scope(|scope| {
            let first = scope.spawn(|| workers.map(&inputs, compute));
            let second = scope.spawn(|| workers.map(&inputs, compute));
            let expected = inputs.iter().map(|v| v * v).collect::<Vec<_>>();
            assert_eq!(first.join().unwrap(), expected);
            assert_eq!(second.join().unwrap(), expected);
        });
        assert_eq!(peak.load(Ordering::SeqCst), 4);
        assert_eq!(ids.lock().unwrap().len(), 4);
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            workers.map(&[0], |_| panic!("gated worker failure"))
        }))
        .is_err());
        assert_eq!(workers.map(&[2, 3], |v| v * v), vec![4, 9]);
        assert!(CardWorkers::shared(0).is_err());
    }
}
