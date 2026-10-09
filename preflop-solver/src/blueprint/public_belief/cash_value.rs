//! Isolated research adapter: own values are bounded, never zero-sum shifted.
use super::*;

pub(super) const NETWORK_SCHEMA: &str = "hu-cash-public-belief-combo-value-network-v2";
pub(super) const POOLED_NETWORK_SCHEMA: &str = "hu-cash-public-belief-combo-value-network-v3";
const CONTRACT: &str = "cash-turn-start-cfv-full-stack-v1";
const PAYOFF: &str = "own-net-bb-after-refunds-and-house-rake-v1";

fn is_cash_schema(schema: &str) -> bool {
    matches!(schema, NETWORK_SCHEMA | POOLED_NETWORK_SCHEMA)
}

impl PublicValueNetwork {
    pub(super) fn validate_cash_contract(&self) -> Result<(), String> {
        if !is_cash_schema(&self.schema) {
            if self.cash_rules.is_some()
                || self.rules_sha256.is_some()
                || self.payoff_contract.is_some()
                || self.source_game.is_some()
            {
                return Err("legacy public values cannot be relabelled as cash".into());
            }
            return Ok(());
        }
        let rules = self
            .cash_rules
            .as_ref()
            .ok_or("cash value weights lack rules")?;
        rules.validate()?;
        let source = self
            .source_game
            .as_ref()
            .ok_or("cash weights lack betting abstraction identity")?;
        source.validate_cash_rules()?;
        if self.rules_sha256.as_deref() != Some(rules.sha256()?.as_str())
            || source.cash_rules != self.cash_rules
            || source.effective_stack_bb != self.target_scale_bb
            || self.payoff_contract.as_deref() != Some(PAYOFF)
            || self.prediction_contract.as_deref() != Some(CONTRACT)
            || !self.uses_exact_ranges
            || self.source_validation_status.as_deref() != Some("research_only")
            || self.source_dataset_sha256.as_ref().map_or(true, |s| {
                s.len() != 64 || !s.bytes().all(|c| c.is_ascii_hexdigit())
            })
        {
            return Err("cash value network has incompatible identity or payoff provenance".into());
        }
        Ok(())
    }

    pub fn read_cash(path: &Path, game: &BlueprintConfig) -> Result<Self, Box<dyn Error>> {
        let (mut network, sha): (Self, String) = super::super::read_model_artifact(path)?;
        network.artifact_sha256 = Some(sha);
        network.validate()?;
        network.validate_cash_game(game)?;
        Ok(network)
    }

    pub(super) fn validate_cash_game(&self, game: &BlueprintConfig) -> Result<(), String> {
        game.validate_cash_rules()?;
        if !is_cash_schema(&self.schema)
            || game.cash_rules.is_none()
            || self.cash_rules != game.cash_rules
            || self.target_scale_bb != game.effective_stack_bb
            || self.source_game.as_ref().map_or(true, |source| {
                source.action_abstraction != game.action_abstraction
            })
        {
            return Err("cash value network does not match requested rules and depth".into());
        }
        Ok(())
    }

    pub(super) fn cash_checkdown_baseline(
        &self,
        board: &[u8],
        actor: usize,
        invested: [f64; 2],
        ranges: &[Vec<f64>; 2],
    ) -> [Vec<f64>; 2] {
        let game = self
            .source_game
            .as_ref()
            .expect("validated cash source game")
            .clone();
        let state = PublicBeliefState::turn_start(
            board.try_into().expect("cash turn board"),
            actor,
            invested,
            // Prepare ranks/legality for every board-legal query hand, not
            // just positive own reach. Regret matching can assign zero own
            // behavioral mass to a hand whose counterfactual EV is needed.
            std::array::from_fn(|_| uniform_range(board)),
        );
        let config = TurnRiverSolveConfig {
            game,
            state,
            iterations: 2,
            averaging_delay: 0,
            river_refinement_iterations: 0,
            regret_matching_plus: false,
        };
        let solver = TurnRiverSolver::new(config).expect("validated cash root");
        let mut terminal = solver.config.state.game_state();
        terminal.terminal = Some(Terminal::Showdown);
        let cfvs = solver.cash_terminal_values(&terminal, ranges, None);
        std::array::from_fn(|player| {
            let mass = compatible_masses_from_card_marginals(
                &solver.combos,
                &ranges[1 - player],
            );
            cfvs[player]
                .iter()
                .zip(mass)
                .map(|(value, m)| if m > 0.0 { value / m } else { 0.0 })
                .collect()
        })
    }

    pub fn predict_cash_turn(
        &self,
        config: &TurnRiverSolveConfig,
    ) -> Result<[Vec<f64>; 2], String> {
        let normalized = self.validated_cash_turn(config)?;
        let result = self.predict_shared_combo_with_bounds(
            &normalized.board,
            normalized.actor,
            normalized.invested_bb,
            &normalized.ranges,
            [self.target_scale_bb; 2],
        );
        if result.iter().flatten().any(|v| !v.is_finite()) {
            return Err("cash prediction is nonfinite; no fallback".into());
        }
        Ok(result)
    }

    pub(super) fn validated_cash_turn(
        &self,
        config: &TurnRiverSolveConfig,
    ) -> Result<PublicBeliefState, String> {
        self.validate_cash_game(&config.game)?;
        let state = &config.state;
        let rules = config
            .game
            .cash_rules
            .as_ref()
            .expect("validated cash game");
        let invested_units = [
            super::super::cash::cash_units(state.invested_bb[0], rules)?,
            super::super::cash::cash_units(state.invested_bb[1], rules)?,
        ];
        if state.street != Street::Turn
            || state.street_invested_bb != [0.0; 2]
            || invested_units[0] != invested_units[1]
            || state.checks != 0
            || state.aggressions != 0
            || !state.raise_reopened
            || state.last_full_raise_bb != 1.0
            || !state.trajectory.is_empty()
            || state.public_history != ["public_belief:turn_start".to_owned()]
        {
            return Err(
                "cash value features currently describe only fresh, equal-investment turn roots"
                    .into(),
            );
        }
        state.validate_street_and_normalize(&config.game, Street::Turn, 4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (PublicValueNetwork, TurnRiverSolveConfig) {
        let mut network = super::super::tests::zero_shared_value_network();
        let mut game = BlueprintConfig::default();
        game.small_blind_bb = 0.4;
        game.effective_stack_bb = 20.0;
        game.cash_rules = Some(crate::cash_game::study_rules("nl25").unwrap());
        assert!(network.validate_cash_game(&game).is_err());
        network.schema = NETWORK_SCHEMA.into();
        network.prediction_contract = Some(CONTRACT.into());
        network.cash_rules = game.cash_rules.clone();
        network.source_game = Some(game.clone());
        network.rules_sha256 = Some(game.cash_rules.as_ref().unwrap().sha256().unwrap());
        network.payoff_contract = Some(PAYOFF.into());
        network.source_validation_status = Some("research_only".into());
        network.value_normalization = Some("pot".into());
        network.head[0].activation = "linear".into();
        network.head[0].biases[0] = -0.1;
        network.validate().unwrap();
        let board = [8, 13, 22, 31];
        let state = PublicBeliefState::turn_start(
            board,
            1,
            [2.0; 2],
            std::array::from_fn(|_| uniform_range(&board)),
        );
        let config = TurnRiverSolveConfig {
            game,
            state,
            iterations: 2,
            averaging_delay: 0,
            river_refinement_iterations: 0,
            regret_matching_plus: false,
        };
        (network, config)
    }

    #[test]
    fn cash_contract_rejects_legacy_weights_and_preserves_negative_house_share() {
        let (network, config) = fixture();
        let values = network.predict_cash_turn(&config).unwrap();
        let mut roundtrip = config.clone();
        roundtrip.state.invested_bb[1] = f64::from_bits(2.0f64.to_bits() - 1);
        let roundtrip_values = network.predict_cash_turn(&roundtrip).unwrap();
        assert!(values
            .iter()
            .flatten()
            .zip(roundtrip_values.iter().flatten())
            .all(|(a, b)| (a - b).abs() < 1e-6));
        roundtrip.state.invested_bb[1] = 2.04;
        assert!(network.predict_cash_turn(&roundtrip).is_err());
        roundtrip.state.invested_bb = [2.01; 2];
        assert!(network.predict_cash_turn(&roundtrip).is_err());
        let masses: [Vec<f64>; 2] = std::array::from_fn(|p| {
            compatible_masses_from_card_marginals(&all_combos(), &config.state.ranges[1 - p])
        });
        let total: f64 = (0..2)
            .map(|p| {
                values[p]
                    .iter()
                    .zip(&masses[p])
                    .zip(&config.state.ranges[p])
                    .map(|((v, m), r)| v * m * r)
                    .sum::<f64>()
            })
            .sum();
        assert!(total < -0.01, "rake loss must not be shifted to zero sum");
        let mut wrong = config.clone();
        wrong
            .game
            .cash_rules
            .as_mut()
            .unwrap()
            .rake
            .rate_basis_points = 0;
        assert!(network.predict_cash_turn(&wrong).is_err());
    }

    #[test]
    fn cash_pooled_schema_requires_its_distinct_head_layout_and_rules_reader() {
        let (mut network, config) = fixture();
        network.schema = POOLED_NETWORK_SCHEMA.into();
        assert!(network.validate().is_err(), "a pooled schema cannot relabel a compact head");
        network.head[0].input_size = 4;
        network.head[0].weights = vec![0.0; 4];
        network.validate().unwrap();
        network.validate_cash_game(&config.game).unwrap();
        network.schema = NETWORK_SCHEMA.into();
        assert!(network.validate().is_err(), "old compact schema cannot discard pooling");
        network.schema = "hu-public-belief-combo-value-network-v5".into();
        assert!(network.validate().is_err(), "cash pooling cannot enter the Home reader");
    }

    #[test]
    fn cash_pooled_predictions_match_explicit_joint_reach_embeddings_for_zero_own_reach_hands() {
        let (mut network, mut config) = fixture();
        network.schema = POOLED_NETWORK_SCHEMA.into();
        network.query_tower[0].weights[94] = 0.5;
        network.head[0].input_size = 4;
        network.head[0].weights = vec![0.0, 0.025, 0.075, 0.05];
        network.validate().unwrap();
        let absent = Combo::new(51, 50).key();
        config.state.ranges[0][absent] = 0.0;
        for combo in all_combos() {
            if combo.cards()[0] / 4 >= 10 {
                config.state.ranges[1][combo.key()] *= 3.0;
            }
        }
        for range in &mut config.state.ranges {
            let total = range.iter().sum::<f64>();
            for v in range { *v /= total; }
        }
        let state = &config.state;
        let combos = all_combos();
        let (_, queries) = shared_combo_features(&state.board, state.actor, state.invested_bb,
            &state.ranges, &combo_conflicts(), 20.0, network.feature_schema.as_deref().unwrap());
        let masses: [Vec<f64>; 2] = std::array::from_fn(|p|
            compatible_masses_from_card_marginals(&combos, &state.ranges[1-p]));
        let pools: [f64; 2] = std::array::from_fn(|p| {
            let weights = state.ranges[p].iter().zip(&masses[p]).map(|(r,m)| r*m).collect::<Vec<_>>();
            weights.iter().zip(&queries[p]).map(|(w,q)| w * (q[94] * 0.5) as f64).sum::<f64>()
                / weights.iter().sum::<f64>()
        });
        let baseline = network.cash_checkdown_baseline(&state.board, state.actor, state.invested_bb, &state.ranges);
        let actual = network.predict_cash_turn(&config).unwrap();
        let mut aggregate = 0.0;
        for p in 0..2 {
            for combo in &combos {
                let key = combo.key();
                if combo.cards().iter().any(|c| state.board.contains(c)) {
                    assert_eq!(actual[p][key], 0.0);
                    continue;
                }
                let residual = -0.1 + 0.025*pools[p] + 0.075*pools[1-p]
                    + 0.05*(queries[p][key][94]*0.5) as f64;
                assert!((actual[p][key] - baseline[p][key] - residual*4.0).abs() < 1e-6);
                aggregate += state.ranges[p][key]*masses[p][key]*actual[p][key];
            }
        }
        assert!(actual[0][absent].abs() > 0.01, "zero own reach must not censor a query");
        assert!(aggregate < -0.1, "pooled cash values must not be projected to zero sum");
    }
}
