//! Exact own-realization-weighted preflop averaging, independent of regret sampling.
use super::*;

#[cfg(test)]
mod pilot;

impl Trainer {
    /// Immutable, preflop-only serving input for the strict existing reader.
    /// The legacy `strategy_sum` field contains normalized frozen averages;
    /// no regrets, RNG, discount state or resumable checkpoint is exported.
    #[cfg(test)]
    pub(super) fn write_frozen_preflop_average(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        if path.exists() {
            return Err("refusing to overwrite frozen preflop policy".into());
        }
        if self.completed_iterations != self.config.iterations {
            return Err("refusing to freeze incomplete preflop training".into());
        }
        let classes = all_combos()
            .into_iter()
            .map(|c| format!("preflop:{}", c.label()))
            .collect::<BTreeSet<_>>();
        let mut rows = BTreeMap::new();
        let mut histories = BTreeMap::new();
        let mut pending = vec![GameState::initial(&self.config)];
        while let Some(state) = pending.pop() {
            if state.terminal.is_some() || state.street != Street::Preflop {
                continue;
            }
            let actions = state.legal_actions(&self.config);
            for class in &classes {
                let (key, descriptor, _) = information_set_from_bucket_trajectories(
                    &state,
                    &self.config,
                    vec![Arc::from(class.as_str())],
                    Vec::new(),
                );
                let node = self
                    .nodes
                    .get(&key)
                    .ok_or("missing preflop row; no frozen fallback")?;
                let history = self
                    .public_histories
                    .get(&descriptor.public_history_id)
                    .ok_or("missing preflop history")?;
                if history != &state.public_history {
                    return Err("preflop history mismatch".into());
                }
                histories.insert(descriptor.public_history_id, history);
                if node.descriptor != descriptor
                    || node.regret_updates == 0
                    || node.average_visits == 0
                    || !node.strategy_sum.iter().any(|p| *p > 0.0)
                    || node.strategy_sum.iter().any(|p| !p.is_finite() || *p < 0.0)
                    || node
                        .action_labels
                        .iter()
                        .map(AsRef::as_ref)
                        .ne(actions.iter().map(|a| a.label.as_str()))
                {
                    return Err(format!(
                        "untrained or invalid preflop row {key}; cannot freeze complete policy"
                    )
                    .into());
                }
                let probabilities = node.average_strategy();
                if probabilities
                    .iter()
                    .any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
                    || (probabilities.iter().sum::<f64>() - 1.0).abs() > 1e-12
                {
                    return Err("invalid normalized frozen preflop policy".into());
                }
                rows.insert(
                    key,
                    serde_json::json!({"descriptor":node.descriptor,
                    "action_labels":node.action_labels,"strategy_sum":probabilities,
                    "average_visits":node.average_visits,"regret_updates":node.regret_updates}),
                );
            }
            pending.extend(actions.iter().map(|a| state.apply(a, &self.config)));
        }
        write_json_atomic(
            path,
            &serde_json::json!({
                "artifact_kind":"immutable-preflop-average-v1", "schema_version":self.config.checkpoint_schema_version(),
                "model":MODEL,"approximate":true,"config":self.config,"completed_iterations":self.completed_iterations,
                "public_histories":histories,"nodes":rows,
                "interpretation":"Frozen trained preflop average only. Normalized strategy_sum compatibility field; not resumable and not release-qualified."
            }),
        )
    }

    pub(super) fn sweep_preflop_average(&mut self) -> Result<(), String> {
        if self.completed_iterations < self.config.averaging_delay {
            return Ok(());
        }
        // One class is sufficient: every exact holding in a preflop class has
        // the same information-set key. No board/private chance is sampled.
        // Class multiplicity is constant over time and cancels in each row.
        let classes = all_combos()
            .into_iter()
            .map(|combo| format!("preflop:{}", combo.label()))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(Arc::<str>::from)
            .collect::<Vec<_>>();
        let weight = self.strategy_averaging_weight(self.completed_iterations + 1);
        self.sweep_preflop_state(
            GameState::initial(&self.config),
            &classes,
            vec![[1.0; 2]; classes.len()],
            weight,
        )
    }

    fn sweep_preflop_state(
        &mut self,
        state: GameState,
        classes: &[Arc<str>],
        own_reaches: Vec<[f64; 2]>,
        weight: f64,
    ) -> Result<(), String> {
        if state.terminal.is_some() || state.street != Street::Preflop {
            return Ok(());
        }
        let actions = state.legal_actions(&self.config);
        let mut strategies = Vec::with_capacity(classes.len());
        for (class, own_reach) in classes.iter().zip(&own_reaches) {
            let (key, mut descriptor, history) = information_set_from_bucket_trajectories(
                &state,
                &self.config,
                vec![class.clone()],
                Vec::new(),
            );
            if !self.nodes.contains_key(&key)
                && self.nodes.len() >= self.config.max_information_sets
            {
                return Err("exact preflop averaging reached the information-set guard".into());
            }
            self.intern_descriptor_buckets(&mut descriptor);
            match self.public_histories.get(&descriptor.public_history_id) {
                Some(existing) if existing != &history => {
                    return Err("preflop history hash collision".into())
                }
                Some(_) => {}
                None => {
                    self.public_histories
                        .insert(descriptor.public_history_id, history);
                }
            }
            let node = self.nodes.entry(key).or_insert_with(|| {
                Node::new(descriptor.clone(), &actions, &mut self.string_interner)
            });
            if node.descriptor != descriptor
                || node
                    .action_labels
                    .iter()
                    .map(AsRef::as_ref)
                    .ne(actions.iter().map(|action| action.label.as_str()))
            {
                return Err("exact preflop key/descriptor/action collision".into());
            }
            // Observe one iteration-wide current policy, before any regret
            // update. Discount a COPY: touching lazy discounts in the trainer
            // would change floating-point recurrence and break matched pilots.
            let mut snapshot = node.clone();
            snapshot.apply_dcfr_regret_discount(self.completed_iterations + 1, &self.discounts);
            let strategy = snapshot.current_strategy();
            let mass = weight * own_reach[state.actor];
            if !mass.is_finite() || mass < 0.0 {
                return Err("invalid exact preflop averaging mass".into());
            }
            if mass > 0.0 {
                for (sum, probability) in node.strategy_sum.iter_mut().zip(&strategy) {
                    *sum += mass * probability;
                }
                node.average_visits += 1;
            }
            strategies.push(strategy);
        }
        // Enumerate ALL public actions, including zero opponent-reach branches.
        // Each column is that player's own realization for the indexed class;
        // these columns are not a joint distribution of compatible holdings.
        for (action_index, action) in actions.iter().enumerate() {
            let mut child_reaches = own_reaches.clone();
            for (reach, strategy) in child_reaches.iter_mut().zip(&strategies) {
                reach[state.actor] *= strategy[action_index];
            }
            self.sweep_preflop_state(
                state.apply(action, &self.config),
                classes,
                child_reaches,
                weight,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flow_error(
        trainer: &Trainer,
        state: GameState,
        combo: Combo,
        own: [f64; 2],
        total_weight: f64,
    ) -> f64 {
        if state.terminal.is_some() || state.street != Street::Preflop {
            return 0.0;
        }
        let deal = neural::deal_for_policy_combo_on_board(combo, state.actor, &[]).unwrap();
        let (key, _, _) = information_set(&state, &deal, &trainer.config);
        let node = &trainer.nodes[&key];
        let mut error =
            (node.strategy_sum.iter().sum::<f64>() / total_weight - own[state.actor]).abs();
        let strategy = node.average_strategy();
        for (action, probability) in state.legal_actions(&trainer.config).iter().zip(strategy) {
            let mut child_own = own;
            child_own[state.actor] *= probability;
            error = error.max(flow_error(
                trainer,
                state.apply(action, &trainer.config),
                combo,
                child_own,
                total_weight,
            ));
        }
        error
    }

    #[test]
    fn exact_preflop_average_obeys_temporal_realization_flow_not_opponent_reach() {
        let mut trainer = Trainer::fresh(BlueprintConfig {
            effective_stack_bb: 20.0,
            averaging_delay: 0,
            ..BlueprintConfig::default()
        });
        let mut total_weight = 0.0;
        for iteration in 1..=3 {
            trainer.completed_iterations = iteration - 1;
            trainer.discounts.advance(iteration);
            // A changing, non-uniform policy at EVERY public state. Squaring
            // and reversing the action pattern exercises correlations between
            // each player's earlier and later decisions, including own zeros.
            for node in trainer.nodes.values_mut() {
                for (action, regret) in node.regrets.iter_mut().enumerate() {
                    *regret = if (action + iteration as usize) % 3 == 0 {
                        0.0
                    } else {
                        ((action + iteration as usize) as f64).powi(2)
                    };
                }
            }
            let before = trainer
                .nodes
                .iter()
                .map(|(key, n)| {
                    (
                        *key,
                        (
                            n.regrets.clone(),
                            n.last_discount_iteration,
                            n.last_regret_discount_cumulative_logs,
                        ),
                    )
                })
                .collect::<BTreeMap<_, _>>();
            trainer.sweep_preflop_average().unwrap();
            for (key, (regrets, last, logs)) in before {
                let node = &trainer.nodes[&key];
                assert_eq!(node.regrets, regrets);
                assert_eq!(node.last_discount_iteration, last);
                assert_eq!(node.last_regret_discount_cumulative_logs, logs);
            }
            total_weight += trainer.strategy_averaging_weight(iteration);
            for combo in [Combo::new(51, 50), Combo::new(48, 44), Combo::new(4, 1)] {
                assert!(
                    flow_error(
                        &trainer,
                        GameState::initial(&trainer.config),
                        combo,
                        [1.0; 2],
                        total_weight
                    ) < 1e-12
                );
            }
        }
    }

    #[test]
    fn exact_preflop_average_preserves_matched_regrets_rng_and_postflop_averages() {
        let mut config = super::super::tests::tiny_config();
        config.traversal = BlueprintTraversal::PublicChanceSampling;
        config.integrate_terminal_actions = true;
        config.max_information_sets = 500_000;
        let mut control = Trainer::fresh(config.clone());
        config.exact_preflop_averaging = true;
        let mut candidate = Trainer::fresh(config);
        for iteration in 1..=3 {
            control.config.iterations = iteration;
            candidate.config.iterations = iteration;
            control.train(&RunControl::default()).unwrap();
            candidate.train(&RunControl::default()).unwrap();
            assert_eq!(control.completed_iterations, candidate.completed_iterations);
            assert_eq!(control.rng.state(), candidate.rng.state());
            assert_eq!(control.sampled_deals, candidate.sampled_deals);
            assert_eq!(control.terminal_evaluations, candidate.terminal_evaluations);
            for (key, old) in &control.nodes {
                let new = &candidate.nodes[key];
                assert_eq!(new.descriptor, old.descriptor);
                assert_eq!(new.regrets, old.regrets, "regrets at {key}");
                assert_eq!(new.regret_updates, old.regret_updates);
                assert_eq!(new.last_discount_iteration, old.last_discount_iteration);
                assert_eq!(
                    new.last_regret_discount_cumulative_logs,
                    old.last_regret_discount_cumulative_logs
                );
                if old.descriptor.street != Street::Preflop {
                    assert_eq!(new.strategy_sum, old.strategy_sum);
                    assert_eq!(new.average_visits, old.average_visits);
                }
            }
            for (key, node) in &candidate.nodes {
                if !control.nodes.contains_key(key) {
                    assert_eq!(node.descriptor.street, Street::Preflop);
                    assert_eq!(node.regret_updates, 0);
                    assert_eq!(node.last_discount_iteration, 0);
                    assert!(node.regrets.iter().all(|r| *r == 0.0));
                }
            }
        }
    }

    #[test]
    fn exact_preflop_average_pins_resume_and_preserves_default_serialization() {
        let default = BlueprintConfig::default();
        assert!(serde_json::to_value(&default)
            .unwrap()
            .get("exact_preflop_averaging")
            .is_none());
        let mut target = super::super::tests::tiny_config();
        target.exact_preflop_averaging = true;
        assert!(target.validate().is_err());
        target.traversal = BlueprintTraversal::PublicChanceSampling;
        target.integrate_terminal_actions = true;
        target.max_information_sets = 500_000;
        assert!(target.validate().is_ok());
        assert_eq!(target.checkpoint_schema_version(), 7);
        let mut partial = Trainer::fresh(target.clone());
        partial.config.iterations = 1;
        partial.train(&RunControl::default()).unwrap();
        let path = std::env::temp_dir().join(format!(
            "exact-preflop-average-{}.msgpack.gz",
            std::process::id()
        ));
        partial.write_checkpoint(&path).unwrap();
        let mut changed = target.clone();
        changed.exact_preflop_averaging = false;
        assert!(Trainer::from_checkpoint(read_checkpoint(&path).unwrap(), &changed).is_err());
        let mut resumed =
            Trainer::from_checkpoint(read_checkpoint(&path).unwrap(), &target).unwrap();
        resumed.train(&RunControl::default()).unwrap();
        let mut fresh = Trainer::fresh(target);
        fresh.train(&RunControl::default()).unwrap();
        assert_eq!(resumed.rng.state(), fresh.rng.state());
        assert_eq!(
            rmp_serde::to_vec_named(&resumed.nodes).unwrap(),
            rmp_serde::to_vec_named(&fresh.nodes).unwrap()
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn exact_preflop_average_covers_every_initially_reachable_information_set() {
        let mut trainer = Trainer::fresh(BlueprintConfig {
            effective_stack_bb: 20.0,
            averaging_delay: 0,
            ..BlueprintConfig::default()
        });
        trainer.discounts.advance(1);
        let rng = trainer.rng.state();
        trainer.sweep_preflop_average().unwrap();
        assert_eq!(trainer.rng.state(), rng);
        assert_eq!(
            trainer.nodes.len(),
            16_900,
            "100 public states times 169 classes"
        );
        for node in trainer.nodes.values() {
            assert!(node.strategy_sum.iter().sum::<f64>() > 0.0);
            assert_eq!(node.average_visits, 1);
            assert_eq!(node.regret_updates, 0);
            assert_eq!(node.last_discount_iteration, 0);
            assert!(node.regrets.iter().all(|r| *r == 0.0));
        }
    }
}
