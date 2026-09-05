//! Bounded CPU workers sharing an immutable checkpoint. A wave contains only
//! at most 16 * workers hands; idle workers take the next available hand.
//! Results are consumed in the original sampled order.
//! Each worker owns its solver cache and coverage counters, never the table.
use super::*;

pub(super) fn for_each_deal<T: Send>(
    policy: &dyn ResponsePolicy,
    workers: usize,
    chance: &mut SplitMix64,
    deals: u64,
    compute: impl Fn(&dyn ResponsePolicy, &Deal, u64) -> T + Sync,
    mut consume: impl FnMut(T),
) {
    if workers == 1 {
        for index in 0..deals {
            consume(compute(policy, &Deal::sample(chance), index));
        }
        return;
    }
    assert!((2..=4).contains(&workers));
    let mut policies: Vec<_> = (0..workers)
        .map(|_| policy.parallel_copy().expect("validated parallel policy"))
        .collect();
    let wave_size = workers * 16;
    for start in (0..deals).step_by(wave_size) {
        let cards: Vec<_> = (start..deals.min(start + wave_size as u64))
            .map(|index| {
                let deal = Deal::sample(chance);
                (index, deal.holes, deal.board)
            })
            .collect();
        // Public deals and rollout seeds are already fixed. Assignment to
        // a worker must not strand cheap hands behind an expensive solve.
        let next = std::sync::atomic::AtomicUsize::new(0);
        let results = std::thread::scope(|scope| {
            let handles: Vec<_> = policies
                .drain(..)
                .map(|local| {
                    let cards = &cards;
                    let next = &next;
                    let compute = &compute;
                    scope.spawn(move || {
                        let mut output = Vec::new();
                        loop {
                            let slot = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let Some((index, holes, board)) = cards.get(slot) else {
                                break;
                            };
                            let value = compute(
                                local.as_ref(),
                                &Deal::from_sampled_cards(*holes, *board),
                                *index,
                            );
                            output.push((slot, value));
                        }
                        (local, output)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .unwrap_or_else(|error| std::panic::resume_unwind(error))
                })
                .collect::<Vec<_>>()
        });
        let mut ordered = Vec::with_capacity(cards.len());
        for (local, output) in results {
            ordered.extend(output);
            policies.push(local);
        }
        // Worker completion order must not change floating-point accumulation.
        ordered.sort_unstable_by_key(|(slot, _)| *slot);
        for (_, value) in ordered {
            consume(value);
        }
    }
    for local in policies {
        policy.absorb_worker(local.as_ref());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_workers_can_take_a_hand_behind_a_slow_hand() {
        use std::sync::{Condvar, Mutex};
        use std::time::Duration;

        let (policy, _) = super::super::tests::tabular_fixture();
        // A full wave reproduces the production straggler pattern. The short
        // tail is the same scheduling problem with all but one chunk empty.
        for deals in [2, 32] {
            let progress = (Mutex::new(false), Condvar::new());
            let mut values = Vec::new();
            for_each_deal(
                &policy,
                2,
                &mut SplitMix64::new(37),
                deals,
                |_, _, index| {
                    if index == 0 {
                        let (lock, ready) = &progress;
                        let (started, _) = ready
                            .wait_timeout_while(
                                lock.lock().unwrap(),
                                Duration::from_secs(1),
                                |started| !*started,
                            )
                            .unwrap();
                        *started
                    } else {
                        if index == 1 {
                            *progress.0.lock().unwrap() = true;
                            progress.1.notify_all();
                        }
                        true
                    }
                },
                |value| values.push(value),
            );
            assert_eq!(values.len(), deals as usize);
            assert!(
                values[0],
                "an idle worker must take hand 1 while hand 0 is occupied ({deals} hands)"
            );
        }
    }

    #[test]
    fn workers_keep_deals_and_accumulation_order_including_partial_waves() {
        let (policy, _) = super::super::tests::tabular_fixture();
        let mut reference = Vec::new();
        for workers in [1, 2, 3, 4] {
            let mut output = Vec::new();
            for_each_deal(
                &policy,
                workers,
                &mut SplitMix64::new(37),
                71,
                |_, deal, index| (index, deal.holes, deal.board),
                |value| output.push(value),
            );
            if workers == 1 {
                reference = output;
            } else {
                assert_eq!(reference, output);
            }
        }
        let duplicate = policy.isolated_copy();
        assert!(Arc::ptr_eq(&policy.table, &duplicate.table));
    }
}
