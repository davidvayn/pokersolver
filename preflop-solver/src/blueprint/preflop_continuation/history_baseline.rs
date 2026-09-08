//! Training-only, past-sample strategic residual baseline. It is not a policy,
//! continuation oracle, EV label, or resumable/exported training artifact.
use super::*;

pub(super) struct HistoryBaseline {
    classes: Vec<usize>,
    rows: BTreeMap<History, [Vec<f64>; 2]>,
}

impl HistoryBaseline {
    pub(super) fn new() -> Self {
        let labels: Vec<_> = all_combos()
            .iter()
            .map(|c| c.label())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        Self {
            classes: all_combos()
                .iter()
                .map(|c| labels.binary_search(&c.label()).unwrap())
                .collect(),
            rows: BTreeMap::new(),
        }
    }

    pub(super) fn len(&self) -> usize {
        self.rows.len()
    }

    pub(super) fn predict(&self, history: &History, prior: &[Vec<f64>; 2]) -> [Vec<f64>; 2] {
        let Some(row) = self.rows.get(history) else {
            return [vec![0.0; 1326], vec![0.0; 1326]];
        };
        let scales = scales(prior);
        let mut predicted: [Vec<f64>; 2] = std::array::from_fn(|p| {
            (0..1326)
                .map(|c| row[p][self.classes[c]] * scales[p][c])
                .collect()
        });
        // Reweight stale conditional values at CURRENT opponent reaches, then
        // project their current-profile expectation to zero sum. No own reach
        // enters the per-hand CFV scale. This keeps the accounting guard useful
        // without requiring stale baselines to remain exact game values.
        let mut total = 0.0;
        let mut mass = 0.0;
        for p in 0..2 {
            for c in 0..1326 {
                total += prior[p][c] * predicted[p][c];
                mass += prior[p][c] * scales[p][c];
            }
        }
        if mass > 0.0 {
            let offset = total / mass;
            for p in 0..2 {
                for c in 0..1326 {
                    predicted[p][c] -= offset * scales[p][c];
                }
            }
        }
        predicted
    }

    /// Called only AFTER the iteration's complete regret backup. Fresh sample
    /// observations never leak into the baseline used to correct that sample.
    /// EMA alpha=.2 is fixed for this pilot, not a tuned success criterion.
    pub(super) fn observe(
        &mut self,
        history: History,
        prior: &[Vec<f64>; 2],
        residual: &[Vec<f64>; 2],
    ) -> Result<(), String> {
        if residual
            .iter()
            .any(|v| v.len() != 1326 || v.iter().any(|x| !x.is_finite()))
        {
            return Err("invalid history-baseline observation".into());
        }
        let scales = scales(prior);
        let row = self
            .rows
            .entry(history)
            .or_insert_with(|| [vec![0.0; 169], vec![0.0; 169]]);
        for p in 0..2 {
            let mut sums = vec![0.0; 169];
            let mut masses = vec![0.0; 169];
            for c in 0..1326 {
                sums[self.classes[c]] += residual[p][c];
                masses[self.classes[c]] += scales[p][c];
            }
            for k in 0..169 {
                if masses[k] > 0.0 {
                    let observation = sums[k] / masses[k];
                    if !observation.is_finite() {
                        return Err("history baseline overflow".into());
                    }
                    row[p][k] = 0.8 * row[p][k] + 0.2 * observation;
                } else if sums[k].abs() > 1e-10 {
                    return Err("nonzero CFV without compatible opponent reach".into());
                }
            }
        }
        Ok(())
    }
}

fn scales(prior: &[Vec<f64>; 2]) -> [Vec<f64>; 2] {
    let conflicts = public_belief::combo_conflicts();
    std::array::from_fn(|p| {
        (0..1326)
            .map(|c| {
                public_belief::compatible_mass_from_conflicts(&prior[1 - p], &conflicts, c)
                    / OPPONENT_HANDS
            })
            .collect()
    })
}

#[test]
fn stale_history_baseline_reweights_current_reach_and_preserves_unbiased_update() {
    let history = vec!["live-flop".into()];
    let mut baseline = HistoryBaseline::new();
    let prior = [vec![1.0; 1326], vec![1.0; 1326]];
    assert!(baseline
        .predict(&history, &prior)
        .iter()
        .flatten()
        .all(|v| *v == 0.0));
    baseline
        .observe(
            history.clone(),
            &prior,
            &[vec![2.0; 1326], vec![-2.0; 1326]],
        )
        .unwrap();
    let prediction = baseline.predict(&history, &prior);
    assert!((prediction[0][0] - 0.4).abs() < 1e-12);
    assert!((prediction[1][0] + 0.4).abs() < 1e-12);
    let current = [vec![0.0; 1326], vec![0.5; 1326]];
    let next = baseline.predict(&history, &current);
    assert!((next[0][0] - 0.2).abs() < 1e-12); // no traverser own-reach multiplication
    assert!(next[1].iter().all(|v| *v == 0.0));
    // Arbitrarily stale cache and chance-dependent correction: enumerate the
    // complete endpoint/chance lottery, not an empirical frequency check.
    for stale in [0.4, -7.0, 11.0] {
        for q in [0.02, 0.5, 1.0] {
            let exact_mean = 3.0;
            let mut expectation = (1.0 - q) * (exact_mean + stale);
            for residual in [-4.0, 6.0] {
                expectation += q
                    * 0.5
                    * corrected_known_expectation(&[exact_mean + stale], &[stale], &[residual], q)
                        .unwrap()[0];
            }
            assert!((expectation - 4.0).abs() < 1e-10);
        }
    }
    assert_eq!(baseline.len(), 1);
    assert_eq!(prediction, baseline.predict(&history, &prior)); // immutable prediction
}

#[test]
fn historical_projection_keeps_zero_sum_for_changed_asymmetric_ranges() {
    let h = vec!["flop".into()];
    let prior = [vec![1.0; 1326], vec![1.0; 1326]];
    let mut b = HistoryBaseline::new();
    let values: [Vec<f64>; 2] = std::array::from_fn(|p| {
        (0..1326)
            .map(|c| ((c % 17) as f64 - 8.0) * (if p == 0 { 1.0 } else { -0.7 }))
            .collect()
    });
    b.observe(h.clone(), &prior, &values).unwrap();
    let current: [Vec<f64>; 2] = std::array::from_fn(|p| {
        (0..1326)
            .map(|c| {
                if c % (3 + p) == 0 {
                    0.0
                } else {
                    (c % 11) as f64 / 11.0
                }
            })
            .collect()
    });
    let predicted = b.predict(&h, &current);
    let weighted: f64 = (0..2)
        .map(|p| {
            (0..1326)
                .map(|c| current[p][c] * predicted[p][c])
                .sum::<f64>()
        })
        .sum();
    assert!(weighted.abs() < 1e-10);
    assert!(predicted.iter().flatten().all(|v| v.is_finite()));
}
