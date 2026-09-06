//! Exact sparse card/strength marginals. Each card belongs to
//! only 51 exact holdings, regardless of the board's number of strength ranks.
//! Remove structurally empty columns while preserving every nonzero addition
//! and its original order. No chance, combo, action or value is approximated.
use super::*;

pub(super) struct CardStrengthLayout {
    offsets: Vec<[usize; 2]>,
    starts: [usize; 53],
}

impl CardStrengthLayout {
    pub(super) fn new(combos: &[Combo], strength_ranks: &[usize]) -> Self {
        let mut ranks: [Vec<usize>; 52] = std::array::from_fn(|_| Vec::with_capacity(51));
        for (combo, rank) in combos.iter().zip(strength_ranks) {
            for card in combo.cards() {
                ranks[card as usize].push(*rank);
            }
        }
        let mut starts = [0; 53];
        for card in 0..52 {
            ranks[card].sort_unstable();
            ranks[card].dedup();
            starts[card + 1] = starts[card] + ranks[card].len();
        }
        let offsets = combos
            .iter()
            .zip(strength_ranks)
            .map(|(combo, rank)| {
                combo.cards().map(|card| {
                    starts[card as usize] + ranks[card as usize].binary_search(rank).unwrap()
                })
            })
            .collect();
        Self { offsets, starts }
    }

    pub(super) fn entries(&self) -> usize {
        self.starts[52]
    }

    pub(super) fn values(
        &self,
        combos: &[Combo],
        strength_ranks: &[usize],
        strength_group_count: usize,
        opponent_reach: &[f64],
        win: f64,
        loss: f64,
        tie: f64,
    ) -> Vec<f64> {
        let mut by_strength = vec![0.0; strength_group_count];
        let mut by_card_strength = vec![0.0; self.starts[52]];
        let mut by_card = [0.0f64; 52];
        for (index, (combo, weight)) in combos.iter().zip(opponent_reach).enumerate() {
            if *weight == 0.0 {
                continue;
            }
            let rank = strength_ranks[index];
            let [first, second] = combo.cards();
            by_strength[rank] += *weight;
            by_card[first as usize] += *weight;
            by_card[second as usize] += *weight;
            by_card_strength[self.offsets[index][0]] += *weight;
            by_card_strength[self.offsets[index][1]] += *weight;
        }
        let mut lower_by_strength = vec![0.0; strength_group_count];
        let mut total = 0.0;
        for (rank, weight) in by_strength.iter().enumerate() {
            lower_by_strength[rank] = total;
            total += *weight;
        }
        let mut lower_by_card_strength = vec![0.0; self.starts[52]];
        for card in 0..52 {
            let mut running = 0.0;
            for offset in self.starts[card]..self.starts[card + 1] {
                lower_by_card_strength[offset] = running;
                running += by_card_strength[offset];
            }
        }
        combos
            .iter()
            .enumerate()
            .map(|(own, combo)| {
                let rank = strength_ranks[own];
                let [first, second] = combo.cards();
                let [first_offset, second_offset] = self.offsets[own];
                let lower = (lower_by_strength[rank]
                    - lower_by_card_strength[first_offset]
                    - lower_by_card_strength[second_offset])
                    .max(0.0);
                let equal = (by_strength[rank]
                    - by_card_strength[first_offset]
                    - by_card_strength[second_offset]
                    + opponent_reach[own])
                    .max(0.0);
                let compatible = (total - by_card[first as usize] - by_card[second as usize]
                    + opponent_reach[own])
                    .max(0.0);
                let higher = (compatible - lower - equal).max(0.0);
                lower * win + equal * tie + higher * loss
            })
            .collect()
    }
}

#[test]
fn compact_card_strength_marginals_preserve_every_value_bit() {
    let combos = all_combos();
    let mut chance = SplitMix64::new(872001);
    for groups in [1, 7, 197, 1326] {
        let ranks = (0..COMBO_COUNT)
            .map(|i| (i * 37) % groups)
            .collect::<Vec<_>>();
        let layout = CardStrengthLayout::new(&combos, &ranks);
        assert!(layout.starts[52] <= 2 * COMBO_COUNT);
        for sparse in [false, true] {
            let range = (0..COMBO_COUNT)
                .map(|_| {
                    if sparse && chance.index(5) != 0 {
                        0.0
                    } else {
                        (chance.index(1000) as f64 + 1.0) / 1326000.0
                    }
                })
                .collect::<Vec<_>>();
            for (win, loss, tie) in [(1.0, -1.0, 0.0), (20.0, -20.0, 0.0), (7.5, -9.0, -0.75)] {
                let reference = showdown_values_from_card_strength_marginals(
                    &combos, &ranks, groups, &range, win, loss, tie,
                );
                let actual = layout.values(&combos, &ranks, groups, &range, win, loss, tie);
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }
}

#[test]
#[ignore = "explicit kernel timing, not a policy-quality evaluation"]
fn compact_card_strength_marginal_cost_probe() {
    use std::hint::black_box;
    use std::time::Instant;
    let board = [0, 5, 10, 15];
    let config = TurnRiverSolveConfig {
        game: BlueprintConfig::default(),
        state: PublicBeliefState::turn_start(
            board,
            1,
            [1.0, 1.0],
            std::array::from_fn(|_| uniform_range(&board)),
        ),
        iterations: 4,
        averaging_delay: 0,
        river_refinement_iterations: 0,
        regret_matching_plus: false,
    };
    let solver = TurnRiverSolver::new(config).unwrap();
    for river in [20, 21, 50] {
        let data = solver.river_data[river as usize].as_ref().unwrap();
        let layout = CardStrengthLayout::new(&solver.combos, &data.strength_ranks);
        let range = all_combos()
            .iter()
            .map(|combo| {
                if data.legal[1][combo.key()] {
                    (combo.key() % 19 + 1) as f64 / 20000.0
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        let reference = showdown_values_from_card_strength_marginals(
            &solver.combos,
            &data.strength_ranks,
            data.strength_group_count,
            &range,
            20.0,
            -20.0,
            0.0,
        );
        assert_eq!(
            reference,
            layout.values(
                &solver.combos,
                &data.strength_ranks,
                data.strength_group_count,
                &range,
                20.0,
                -20.0,
                0.0
            )
        );
        let mut timings = Vec::new();
        for compact in [false, true, true, false] {
            let start = Instant::now();
            for _ in 0..1000 {
                let result = if compact {
                    layout.values(
                        &solver.combos,
                        &data.strength_ranks,
                        data.strength_group_count,
                        black_box(&range),
                        20.0,
                        -20.0,
                        0.0,
                    )
                } else {
                    showdown_values_from_card_strength_marginals(
                        &solver.combos,
                        &data.strength_ranks,
                        data.strength_group_count,
                        black_box(&range),
                        20.0,
                        -20.0,
                        0.0,
                    )
                };
                black_box(result);
            }
            timings.push(
                serde_json::json!({"compact":compact,"seconds":start.elapsed().as_secs_f64()}),
            );
        }
        println!(
            "{}",
            serde_json::json!({"river":river,"denseCardStrengthEntries":52*data.strength_group_count,
            "compactCardStrengthEntries":layout.starts[52],"timings":timings})
        );
    }
}
