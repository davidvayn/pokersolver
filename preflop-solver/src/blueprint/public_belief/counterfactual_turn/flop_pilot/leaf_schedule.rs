//! Research-only cheap warmup / accurate-tail allocation. The matching learned
//! control omits the same early averages. Regrets and chance sampling continue.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct Schedule {
    pub native_tail_iterations: u64,
    pub averaging_start_iteration: u64,
}

impl Schedule {
    pub fn validate(self, iterations: u64) -> Result<(), String> {
        if iterations < 2 || self.native_tail_iterations > iterations
            || !(1..=iterations).contains(&self.averaging_start_iteration)
            || (self.native_tail_iterations > 0
                && self.averaging_start_iteration <= iterations - self.native_tail_iterations)
        {
            return Err("invalid late-native schedule or averaging of approximate warmup".into());
        }
        Ok(())
    }

    pub fn native_round(self, round: u64, iterations: u64) -> bool {
        round > iterations - self.native_tail_iterations
    }

    pub fn prepare_average(self, round: u64, trunk: &mut Trunk) {
        if round == self.averaging_start_iteration {
            for node in trunk.nodes.values_mut() {
                // The DCFR clocks keep their original global round numbers.
                // This changes only which realization-weighted iterates export.
                node.strategy_sum.fill(0.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_tail_boundaries_and_learned_control() {
        let schedule = Schedule { native_tail_iterations: 8, averaging_start_iteration: 25 };
        schedule.validate(32).unwrap();
        assert_eq!((1..=32).filter(|&round| schedule.native_round(round, 32)).collect::<Vec<_>>(),
                   (25..=32).collect::<Vec<_>>());
        let control = Schedule { native_tail_iterations: 0, ..schedule };
        control.validate(32).unwrap();
        assert!((1..=32).all(|round| !control.native_round(round, 32)));
    }

    #[test]
    fn reject_averaging_warmup_and_unbounded_or_empty_schedules() {
        for (total, tail, start) in [(1, 0, 1), (32, 33, 25), (32, 8, 0),
                                    (32, 8, 33), (32, 8, 24)] {
            assert!(Schedule { native_tail_iterations: tail, averaging_start_iteration: start }
                .validate(total).is_err());
        }
    }

    #[test]
    fn average_reset_preserves_regrets_strategy_and_both_discount_clocks() {
        let fixture = frozen_response::tests::royal_fixture();
        let mut trunk = Trunk::new(fixture.game, fixture.state).unwrap();
        for node in trunk.nodes.values_mut() {
            for (index, regret) in node.regrets.iter_mut().enumerate() {
                *regret = (index % 7) as f64 - 2.5;
            }
            node.strategy_sum.fill(13.5);
            node.last_regret_discount_round = 24;
            node.last_strategy_discount_round = 24;
        }
        let before = trunk.nodes.iter().map(|(key, node)| (key.clone(),
            (node.regrets.clone(), node.strategy(&trunk.legal[node.actor])))).collect::<BTreeMap<_, _>>();
        let schedule = Schedule { native_tail_iterations: 8, averaging_start_iteration: 25 };
        schedule.prepare_average(24, &mut trunk);
        assert!(trunk.nodes.values().all(|node| node.strategy_sum.iter().all(|v| *v == 13.5)));
        schedule.prepare_average(25, &mut trunk);
        for (key, node) in &trunk.nodes {
            assert_eq!(node.regrets, before[key].0);
            assert_eq!(node.strategy(&trunk.legal[node.actor]), before[key].1);
            assert!(node.strategy_sum.iter().all(|v| *v == 0.0));
            assert_eq!(node.last_regret_discount_round, 24);
            assert_eq!(node.last_strategy_discount_round, 24);
        }
    }

    #[test]
    fn schedule_identity_is_explicit_and_omitted_by_default() {
        let mut fixture = frozen_response::tests::royal_fixture();
        let original = serde_json::to_vec(&fixture).unwrap();
        assert!(!String::from_utf8_lossy(&original).contains("leaf_schedule"));
        fixture.leaf_schedule = Some(Schedule { native_tail_iterations: 0, averaging_start_iteration: 2 });
        let scheduled = serde_json::to_vec(&fixture).unwrap();
        assert_ne!(original, scheduled);
        let mut restored: Solution = serde_json::from_slice(&scheduled).unwrap();
        assert_eq!(restored.leaf_schedule, fixture.leaf_schedule);
        restored.leaf_schedule = None;
        assert_eq!(serde_json::to_vec(&restored).unwrap(), original);
    }

    #[test]
    fn frozen_schedule_requires_valid_bounds_and_a_learned_proposer() {
        let mut fixture = frozen_response::tests::royal_fixture();
        fixture.leaf_schedule = Some(Schedule { native_tail_iterations: 0, averaging_start_iteration: 2 });
        assert!(frozen_response::Frozen::new(&fixture).is_err());
        fixture.learned_leaf_model_sha256 = Some("b".repeat(64));
        assert!(frozen_response::Frozen::new(&fixture).is_ok());
        fixture.leaf_schedule.as_mut().unwrap().averaging_start_iteration = 3;
        assert!(frozen_response::Frozen::new(&fixture).is_err());
    }

    #[test]
    fn actual_trunk_regret_trajectory_is_identical_when_only_averages_are_reset() {
        let fixture = frozen_response::tests::royal_fixture();
        let mut full = Trunk::new(fixture.game.clone(), fixture.state.clone()).unwrap();
        let mut tail = Trunk::new(fixture.game, fixture.state).unwrap();
        let schedule = Schedule { native_tail_iterations: 0, averaging_start_iteration: 3 };
        for round in 1..=4 {
            let leaf = |state: &GameState, ranges: &[Vec<f64>; 2], _: Option<usize>| {
                let utility = 0.17 * state.invested[0] - 0.13 * state.invested[1];
                let conflicts = combo_conflicts();
                std::array::from_fn(|seat| (0..COMBO_COUNT).map(|combo|
                    (if seat == 0 { utility } else { -utility })
                    * compatible_mass_from_conflicts(&ranges[1-seat], &conflicts, combo)).collect())
            };
            schedule.prepare_average(round, &mut tail);
            full.iteration(round, &leaf);
            tail.iteration(round, &leaf);
            for (key, node) in &full.nodes {
                let changed = &tail.nodes[key];
                assert_eq!(node.regrets, changed.regrets);
                assert_eq!(node.strategy(&full.legal[node.actor]), changed.strategy(&tail.legal[changed.actor]));
                assert_eq!(node.last_regret_discount_round, changed.last_regret_discount_round);
                assert_eq!(node.last_strategy_discount_round, changed.last_strategy_discount_round);
            }
        }
    }
}
