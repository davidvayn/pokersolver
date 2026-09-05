//! Bounded CPU workers sharing an immutable checkpoint. A wave contains only
//! 16 hands per worker; results are consumed in the original sampled order.
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
        let results = std::thread::scope(|scope| {
            let handles: Vec<_> = policies
                .drain(..)
                .enumerate()
                .map(|(worker, local)| {
                    let first = (worker * 16).min(cards.len());
                    let last = (first + 16).min(cards.len());
                    let chunk = &cards[first..last];
                    let compute = &compute;
                    scope.spawn(move || {
                        let output: Vec<_> = chunk
                            .iter()
                            .map(|(index, holes, board)| {
                                compute(
                                    local.as_ref(),
                                    &Deal::from_sampled_cards(*holes, *board),
                                    *index,
                                )
                            })
                            .collect();
                        (local, output)
                    })
                })
                .collect();
            // Join/replay in worker and deal order, never completion order.
            handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .unwrap_or_else(|error| std::panic::resume_unwind(error))
                })
                .collect::<Vec<_>>()
        });
        for (local, output) in results {
            for value in output {
                consume(value);
            }
            policies.push(local);
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
    fn workers_keep_deals_and_accumulation_order_including_partial_waves() {
        let (policy, _) = super::super::tests::tabular_fixture();
        let mut reference = Vec::new();
        for workers in [1, 2, 4] {
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
