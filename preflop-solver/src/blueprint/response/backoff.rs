//! Experimental completion of missing flop rows from frozen average-policy
//! mass. No retraining, guessed action EVs, or claim of perfect-recall safety.
//! Exact trained rows retain precedence. Pooling forgets the private preflop
//! bucket but preserves the current hand/board buckets and public action line.

use super::*;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FlopBackoffOptions {
    pub minimum_average_visits: u64,
    pub weight: f64,
}

impl FlopBackoffOptions {
    pub(super) fn validate(&self) -> Result<(), String> {
        if self.minimum_average_visits == 0
            || !self.weight.is_finite()
            || !(0.0..=1.0).contains(&self.weight)
        {
            return Err("flop pooling requires positive support and weight 0..1".to_owned());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub(super) struct CompletionCoverage {
    pub eligible_missing_or_untrained_queries: u64,
    pub matched_queries: u64,
}

impl CompletionCoverage {
    pub fn add(&mut self, other: Self) {
        self.eligible_missing_or_untrained_queries += other.eligible_missing_or_untrained_queries;
        self.matched_queries += other.matched_queries;
    }
}

#[derive(Eq, Ord, PartialEq, PartialOrd)]
struct Key {
    history: u64,
    actor: usize,
    hand: Arc<str>,
    board: Arc<[Arc<str>]>,
}

impl Key {
    fn from_descriptor(d: &NodeDescriptor) -> Option<Self> {
        (d.street == Street::Flop).then(|| Self {
            history: d.public_history_id,
            actor: match d.actor {
                Position::ButtonSmallBlind => 0,
                Position::BigBlind => 1,
            },
            hand: d
                .hand_bucket_trajectory
                .last()
                .expect("flop hand bucket")
                .clone(),
            board: Arc::clone(&d.public_bucket_trajectory),
        })
    }
}

struct Row {
    descriptor: NodeDescriptor,
    labels: Arc<[Arc<str>]>,
    mass: Vec<f64>,
    visits: u64,
}

pub(super) struct FlopBackoff {
    options: FlopBackoffOptions,
    rows: BTreeMap<Key, Row>,
}

impl FlopBackoff {
    pub fn build(table: &InferenceTable, options: FlopBackoffOptions) -> Result<Self, String> {
        options.validate()?;
        let mut rows = BTreeMap::<Key, Row>::new();
        // The checkpoint iterator is sorted; floating-point reduction order is
        // independent of worker scheduling. Raw mass retains DCFR iteration and
        // own-reach weights; this is not a uniform average of row probabilities.
        for node in table.nodes.values() {
            if node.average_visits == 0 || !node.strategy_sum.iter().any(|p| *p > 0.0) {
                continue;
            }
            let Some(key) = Key::from_descriptor(&node.descriptor) else {
                continue;
            };
            let row = rows.entry(key).or_insert_with(|| Row {
                descriptor: node.descriptor.clone(),
                labels: Arc::clone(&node.action_labels),
                mass: vec![0.0; node.strategy_sum.len()],
                visits: 0,
            });
            if !compatible(&row.descriptor, &node.descriptor) || row.labels != node.action_labels {
                return Err("pooled flop history or legal-grid collision".to_owned());
            }
            row.visits = row
                .visits
                .checked_add(node.average_visits)
                .ok_or("flop support overflow")?;
            for (sum, value) in row.mass.iter_mut().zip(node.strategy_sum.iter()) {
                *sum += value;
                if !sum.is_finite() {
                    return Err("pooled flop average mass overflow".to_owned());
                }
            }
        }
        rows.retain(|_, row| row.visits >= options.minimum_average_visits);
        for row in rows.values_mut() {
            row.mass = normalize_or_uniform(std::mem::take(&mut row.mass));
        }
        Ok(Self { options, rows })
    }

    pub fn complete(
        &self,
        descriptor: &NodeDescriptor,
        actions: &[LegalAction],
        baseline: &mut [f64],
    ) -> bool {
        if self.options.weight == 0.0 {
            return false;
        }
        let Some(key) = Key::from_descriptor(descriptor) else {
            return false;
        };
        let Some(row) = self.rows.get(&key) else {
            return false;
        };
        assert!(
            compatible(&row.descriptor, descriptor),
            "pooled flop descriptor collision"
        );
        assert!(
            row.labels
                .iter()
                .map(|a| a.as_ref())
                .eq(actions.iter().map(|a| a.label.as_str())),
            "pooled flop action grid collision"
        );
        assert_eq!(baseline.len(), row.mass.len());
        for (p, learned) in baseline.iter_mut().zip(&row.mass) {
            *p = (1.0 - self.options.weight) * *p + self.options.weight * learned;
        }
        true
    }

    pub fn summary(&self) -> serde_json::Value {
        serde_json::json!({"options":self.options, "pooled_rows":self.rows.len(),
            "interpretation":"Frozen strategy-mass completion, not newly trained exact rows or equilibrium certification; support counts are averaging contributions, not independent effective samples."})
    }
}

fn compatible(a: &NodeDescriptor, b: &NodeDescriptor) -> bool {
    a.actor == b.actor
        && a.street == b.street
        && a.public_history_id == b.public_history_id
        && a.hand_bucket_trajectory.last() == b.hand_bucket_trajectory.last()
        && a.public_bucket_trajectory == b.public_bucket_trajectory
        && a.pot_bb == b.pot_bb
        && a.to_call_bb == b.to_call_bb
        && a.effective_stack_remaining_bb == b.effective_stack_remaining_bb
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (TabularResponsePolicy, Deal, GameState) {
        let (mut policy, deal) = super::super::tests::tabular_fixture();
        let game = &policy.table.config;
        let root = GameState::initial(game);
        let called = root.apply(&root.legal_actions(game)[1], game);
        let flop = called.apply(&called.legal_actions(game)[0], game);
        let actions = flop.legal_actions(game);
        let (_, mut d, _) = information_set(&flop, &deal, game);
        for (index, mass, visits) in [(1, 2.0, 2), (2, 6.0, 6)] {
            d.hand_bucket_trajectory = vec![
                format!("other-preflop-{index}").into(),
                d.hand_bucket_trajectory.last().unwrap().clone(),
            ]
            .into();
            let mut strategy_sum = vec![0.0; actions.len()];
            strategy_sum[index - 1] = mass;
            Arc::get_mut(&mut policy.table).unwrap().nodes.insert(
                index as u64,
                AverageNode {
                    descriptor: d.clone(),
                    action_labels: actions
                        .iter()
                        .map(|a| Arc::<str>::from(a.label.as_str()))
                        .collect::<Vec<_>>()
                        .into(),
                    strategy_sum: strategy_sum.into_boxed_slice(),
                    average_visits: visits,
                },
            );
        }
        (policy, deal, flop)
    }

    #[test]
    fn pooling_retains_frozen_mass_weights_and_requires_support() {
        let (policy, deal, flop) = fixture();
        let game = &policy.table.config;
        let (_, d, _) = information_set(&flop, &deal, game);
        let actions = flop.legal_actions(game);
        let mut mix = vec![1.0 / actions.len() as f64; actions.len()];
        let supported = FlopBackoff::build(
            &policy.table,
            FlopBackoffOptions {
                minimum_average_visits: 8,
                weight: 1.0,
            },
        )
        .unwrap();
        assert!(supported.complete(&d, &actions, &mut mix));
        assert_eq!(&mix[..2], &[0.25, 0.75]);
        assert!((mix.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let unsupported = FlopBackoff::build(
            &policy.table,
            FlopBackoffOptions {
                minimum_average_visits: 9,
                weight: 1.0,
            },
        )
        .unwrap();
        assert!(!unsupported.complete(&d, &actions, &mut mix));
        let mut other = d.clone();
        other.public_history_id += 1;
        assert!(!supported.complete(&other, &actions, &mut mix));
    }

    #[test]
    fn completion_changes_only_untrained_flop_and_does_not_hide_source_coverage() {
        let (mut policy, deal, flop) = fixture();
        let game = policy.table.config.clone();
        policy.flop_backoff = Some(Arc::new(
            FlopBackoff::build(
                &policy.table,
                FlopBackoffOptions {
                    minimum_average_visits: 8,
                    weight: 1.0,
                },
            )
            .unwrap(),
        ));
        let actions = flop.legal_actions(&game);
        let first = policy.strategy(&flop, &deal, &actions, &game);
        assert_eq!(&first[..2], &[0.25, 0.75]);
        assert_eq!(policy.take_completion_coverage().matched_queries, 1);
        assert_eq!(policy.take_raw_coverage()[1].unknown, 1);
        let mut different_board = deal.board;
        different_board[3] = 32;
        different_board[4] = 33;
        // Fixture flop actor is BB: preserve its cards, vary the other hand.
        let hidden = Deal::from_cards([[36, 37], deal.holes[1]], different_board);
        assert_eq!(
            first,
            policy.frozen_strategy(&flop, &hidden, &actions, &game)
        );
        let (key, descriptor, _) = information_set(&flop, &deal, &game);
        let mut mass = vec![0.0; actions.len()];
        *mass.last_mut().unwrap() = 10.0;
        Arc::get_mut(&mut policy.table).unwrap().nodes.insert(
            key,
            AverageNode {
                descriptor,
                action_labels: actions
                    .iter()
                    .map(|a| Arc::<str>::from(a.label.as_str()))
                    .collect::<Vec<_>>()
                    .into(),
                strategy_sum: mass.clone().into_boxed_slice(),
                average_visits: 1,
            },
        );
        assert_eq!(
            policy.strategy(&flop, &deal, &actions, &game),
            normalize_or_uniform(mass)
        );
        assert_eq!(policy.take_completion_coverage().matched_queries, 0);
        assert_eq!(
            policy.frozen_strategy(&flop, &deal, &actions, &game),
            policy.strategy(&flop, &deal, &actions, &game)
        );
        Arc::get_mut(&mut policy.table)
            .unwrap()
            .nodes
            .get_mut(&key)
            .unwrap()
            .average_visits = 0;
        assert_eq!(policy.strategy(&flop, &deal, &actions, &game), first);
    }

    #[test]
    fn shared_pooling_workers_replay_exactly_and_merge_completion_counts() {
        let (mut policy, deal, flop) = fixture();
        let game = policy.table.config.clone();
        let holes = deal.holes;
        let board = deal.board;
        let actions = flop.legal_actions(&game);
        policy.flop_backoff = Some(Arc::new(
            FlopBackoff::build(
                &policy.table,
                FlopBackoffOptions {
                    minimum_average_visits: 8,
                    weight: 0.5,
                },
            )
            .unwrap(),
        ));
        let mut reference = None;
        for workers in [1, 2, 4] {
            let mut rows = Vec::new();
            parallel::for_each_deal(
                &policy,
                workers,
                &mut SplitMix64::new(17),
                71,
                |p, _, _| {
                    p.strategy(
                        &flop,
                        &Deal::from_sampled_cards(holes, board),
                        &actions,
                        &game,
                    )
                },
                |row| rows.push(row),
            );
            let counts = policy.take_completion_coverage();
            assert_eq!(counts.matched_queries, 71);
            assert_eq!(counts.eligible_missing_or_untrained_queries, 71);
            if let Some(previous) = &reference {
                assert_eq!(previous, &rows);
            } else {
                reference = Some(rows);
            }
        }
    }
}
