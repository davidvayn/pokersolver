//! Fresh public-turn subsets inside a single immutable trunk update. Nothing
//! persists across updates: counterfactual values are averaged before regrets
//! or strategy sums change. This does not remove continuation-oracle error.
use super::*;

pub(super) fn draw(turns: &[u8], count: usize, chance: &mut SplitMix64) -> Vec<u8> {
    draw_with(turns, count, |upper| chance.index(upper))
}

fn draw_with(turns: &[u8], count: usize, mut index: impl FnMut(usize) -> usize) -> Vec<u8> {
    assert!((1..=turns.len()).contains(&count));
    let mut remaining = turns.to_vec();
    // A partial Fisher-Yates shuffle samples distinct cards. For count=1 this
    // consumes exactly the old RNG draw and returns exactly the old turn.
    for selected in 0..count {
        let offset = index(remaining.len() - selected);
        assert!(offset < remaining.len() - selected);
        remaining.swap(selected, selected + offset);
    }
    remaining.truncate(count);
    remaining
}

pub(super) fn estimate(sampled: &[u8], mut leaf: impl FnMut(u8) -> [Vec<f64>; 2]) -> [Vec<f64>; 2] {
    assert!(!sampled.is_empty());
    // Start with the first vector, not +0, to preserve signed zeros and the
    // original multiplication/division order when the batch contains one turn.
    let mut sum = leaf(sampled[0]);
    for &turn in &sampled[1..] {
        let next = leaf(turn);
        for seat in 0..2 {
            assert_eq!(sum[seat].len(), next[seat].len());
            for (total, value) in sum[seat].iter_mut().zip(&next[seat]) {
                *total += value;
            }
        }
    }
    let denominator = 45.0 * sampled.len() as f64;
    let result = sum.map(|v| {
        v.into_iter()
            .map(|x| x * 49.0 / denominator)
            .collect::<Vec<_>>()
    });
    assert!(result.iter().flatten().all(|x| x.is_finite()));
    result
}

#[test]
fn subset_draw_preserves_single_turn_rng_and_is_uniform_without_replacement() {
    let turns = (0..52u8)
        .filter(|c| ![0, 5, 10].contains(c))
        .collect::<Vec<_>>();
    let mut old = SplitMix64::new(100101);
    let mut new = old.clone();
    for _ in 0..128 {
        assert_eq!(
            draw(&turns, 1, &mut new),
            vec![turns[old.index(turns.len())]]
        );
        assert_eq!(old.state(), new.state());
    }
    // Enumerate every equally likely bounded-index choice for four of five.
    // Each subset must have precisely 4! ordered realisations.
    let mut counts = BTreeMap::new();
    for a in 0..5 {
        for b in 0..4 {
            for c in 0..3 {
                for d in 0..2 {
                    let mut choices = [a, b, c, d].into_iter();
                    let mut chosen = draw_with(&[1, 2, 3, 4, 6], 4, |_| choices.next().unwrap());
                    assert_eq!(chosen.iter().collect::<BTreeSet<_>>().len(), 4);
                    chosen.sort_unstable();
                    *counts.entry(chosen).or_insert(0) += 1;
                }
            }
        }
    }
    assert_eq!(counts.len(), 5);
    assert!(counts.values().all(|n| *n == 24));
}

#[test]
fn subset_estimator_preserves_blockers_mean_and_finite_population_variance() {
    // Flop 49/50/51, private pair 0/1 versus 2/3: four of the 49 public
    // proposals contribute zero. Non-normalized opponent reach stays in CFVs.
    let value = |turn: u8| {
        if turn < 4 {
            0.0
        } else {
            0.7 * ((turn as f64 % 11.0) - 5.0)
        }
    };
    let leaf = |turn| [vec![value(turn)], vec![0.0]];
    let turns = (0..49).collect::<Vec<_>>();
    let exact = estimate(&turns, leaf)[0][0];
    let mut estimates = Vec::new();
    for a in 0..49 {
        for b in a + 1..49 {
            let result = estimate(&[a, b], leaf);
            assert_eq!(result[1][0], 0.0); // zero opponent total remains zero
            estimates.push(result[0][0]);
        }
    }
    let mean = estimates.iter().sum::<f64>() / estimates.len() as f64;
    let variance =
        estimates.iter().map(|x| (x - exact).powi(2)).sum::<f64>() / estimates.len() as f64;
    let single_variance = turns
        .iter()
        .map(|t| (estimate(&[*t], leaf)[0][0] - exact).powi(2))
        .sum::<f64>()
        / 49.0;
    assert!((mean - exact).abs() < 1e-12);
    assert!((variance - single_variance * 47.0 / 96.0).abs() < 1e-12);
    let raw = [vec![0.0, -0.0, 3.2], vec![-4.6, -0.0, 0.0]];
    let single = estimate(&[7], |_| raw.clone());
    for seat in 0..2 {
        for (actual, before) in single[seat].iter().zip(&raw[seat]) {
            assert_eq!(actual.to_bits(), (before * 49.0 / 45.0).to_bits());
        }
    }
}
