//! Isolated research adapter: own values are bounded, never zero-sum shifted.
use super::*;

pub(super) const NETWORK_SCHEMA: &str = "hu-cash-public-belief-combo-value-network-v2";
pub(super) const POOLED_NETWORK_SCHEMA: &str = "hu-cash-public-belief-combo-value-network-v3";
pub(super) const BLOCKER_POOLED_NETWORK_SCHEMA: &str = "hu-cash-public-belief-combo-value-network-v4";
pub(super) const BASELINE_CONDITIONED_NETWORK_SCHEMA: &str = "hu-cash-public-belief-combo-value-network-v5";
const CONTRACT: &str = "cash-turn-start-cfv-full-stack-v1";
const BLOCKER_POOLED_CONTRACT: &str = "cash-turn-start-cfv-full-stack-blocker-pooled-v1";
const BASELINE_CONDITIONED_CONTRACT: &str = "cash-turn-start-cfv-full-stack-baseline-conditioned-v1";
const PAYOFF: &str = "own-net-bb-after-refunds-and-house-rake-v1";

#[cfg(test)]
thread_local! {
    static GEOMETRY_PREPARATIONS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
pub(super) fn note_geometry_preparation() {
    GEOMETRY_PREPARATIONS.with(|count| count.set(count.get() + 1));
}

fn is_cash_schema(schema: &str) -> bool {
    matches!(schema, NETWORK_SCHEMA | POOLED_NETWORK_SCHEMA | BLOCKER_POOLED_NETWORK_SCHEMA | BASELINE_CONDITIONED_NETWORK_SCHEMA)
}

/// Each query conditions on its own two cards, using raw opponent reaches.
/// Jointly weighted summaries can censor an opponent hand that conflicts with
/// all currently reached own hands but is compatible with a zero-own-reach query.
pub(super) fn card_removed_opponent_embeddings(
    embeddings: &[Vec<f32>; 2],
    keys: &[usize],
    ranges: &[Vec<f64>; 2],
    masses: &[Vec<f64>; 2],
    width: usize,
) -> [Vec<f32>; 2] {
    let combos = all_combos();
    let mut totals = [vec![0.0f64; width], vec![0.0f64; width]];
    let mut cards: [Vec<Vec<f64>>; 2] = std::array::from_fn(|_| vec![vec![0.0; width]; 52]);
    for player in 0..2 {
        for (row, key) in keys.iter().copied().enumerate() {
            let weight = ranges[player][key];
            let [a, b] = combos[key].cards();
            for col in 0..width {
                let value = weight * embeddings[player][row * width + col] as f64;
                totals[player][col] += value;
                cards[player][a as usize][col] += value;
                cards[player][b as usize][col] += value;
            }
        }
    }
    std::array::from_fn(|player| {
        let other = 1 - player;
        let mut result = vec![0.0; keys.len() * width];
        for (row, key) in keys.iter().copied().enumerate() {
            if masses[player][key] <= EPSILON { continue; }
            let [a, b] = combos[key].cards();
            for col in 0..width {
                let numerator = totals[other][col] - cards[other][a as usize][col]
                    - cards[other][b as usize][col]
                    + ranges[other][key] * embeddings[other][row * width + col] as f64;
                result[row * width + col] = (numerator / masses[player][key]) as f32;
            }
        }
        result
    })
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
            || self.prediction_contract.as_deref() != Some(if self.schema == BASELINE_CONDITIONED_NETWORK_SCHEMA {
                BASELINE_CONDITIONED_CONTRACT
            } else if self.schema == BLOCKER_POOLED_NETWORK_SCHEMA {
                BLOCKER_POOLED_CONTRACT
            } else { CONTRACT })
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
        _actor: usize,
        invested: [f64; 2],
        ranges: &[Vec<f64>; 2],
    ) -> [Vec<f64>; 2] {
        let rules = self
            .cash_rules
            .as_ref()
            .expect("validated cash rules");
        cash_checkdown::kernel(board.try_into().expect("cash turn board"))
            .values(rules, invested, ranges)
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
    fn cash_value_queries_reuse_geometry_when_only_beliefs_and_commitments_change() {
        let (network, config) = fixture();
        let before = GEOMETRY_PREPARATIONS.with(Cell::get);
        let mut ranges = config.state.ranges.clone();
        for query in 0..3 {
            ranges[0][Combo::new(50, 51).key()] = query as f64 * 0.01;
            let values = network.cash_checkdown_baseline(
                &config.state.board,
                query % 2,
                [2.0 + query as f64; 2],
                &ranges,
            );
            assert!(values.iter().flatten().all(|value| value.is_finite()));
        }
        let prepared = GEOMETRY_PREPARATIONS.with(Cell::get) - before;
        assert!(prepared <= 1, "one board rebuilt {prepared} times across value queries");
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
        network.schema = BLOCKER_POOLED_NETWORK_SCHEMA.into();
        assert!(network.validate().is_err(), "new pooling cannot silently reinterpret old weights");
        network.prediction_contract = Some(BLOCKER_POOLED_CONTRACT.into());
        network.validate().unwrap();
        network.schema = POOLED_NETWORK_SCHEMA.into();
        assert!(network.validate().is_err(), "blocker weights cannot downgrade to global pooling");
        network.prediction_contract = Some(CONTRACT.into());
        network.schema = NETWORK_SCHEMA.into();
        assert!(network.validate().is_err(), "old compact schema cannot discard pooling");
        network.schema = "hu-public-belief-combo-value-network-v5".into();
        assert!(network.validate().is_err(), "cash pooling cannot enter the Home reader");
    }

    #[test]
    fn cash_baseline_conditioned_head_uses_own_exact_payoff_without_removing_current_features() {
        let (mut network, mut config) = fixture();
        network.schema = BASELINE_CONDITIONED_NETWORK_SCHEMA.into();
        network.prediction_contract = Some(BASELINE_CONDITIONED_CONTRACT.into());
        assert!(network.validate().is_err(), "new cash input cannot relabel an old head");
        network.head[0].input_size = 5;
        network.head[0].weights = vec![0.0, 0.0, 0.0, 0.0, 0.5];
        network.validate().unwrap();
        let absent = Combo::new(51,50).key();
        config.state.ranges[0][absent] = 0.0;
        let normalized = network.validated_cash_turn(&config).unwrap();
        let baseline = network.cash_checkdown_baseline(&normalized.board,normalized.actor,normalized.invested_bb,&normalized.ranges);
        let actual = network.predict_cash_turn(&config).unwrap();
        for p in 0..2 {
            for combo in all_combos() {
                let key = combo.key();
                if combo.cards().iter().any(|c| normalized.board.contains(c)) {
                    assert_eq!(actual[p][key],0.0);
                } else {
                    let expected = baseline[p][key] + 4.0 * (-0.1 + 0.5 * baseline[p][key] / 20.0);
                    assert!((actual[p][key]-expected).abs() < 1e-6);
                }
            }
        }
        assert!(actual[0][absent].abs() > 0.01);
        network.schema = POOLED_NETWORK_SCHEMA.into();
        network.prediction_contract = Some(CONTRACT.into());
        assert!(network.validate().is_err(),"cash input must not be dropped on a silent downgrade");
        network.schema = "hu-public-belief-combo-value-network-v5".into();
        network.prediction_contract = Some("native-turn-cfv-full-stack-v1".into());
        assert!(network.validate().is_err(),"cash own-payoff input cannot enter Home inference");
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
        for schema in [POOLED_NETWORK_SCHEMA, BLOCKER_POOLED_NETWORK_SCHEMA] {
        network.schema = schema.into();
        network.prediction_contract = Some(if schema == BLOCKER_POOLED_NETWORK_SCHEMA { BLOCKER_POOLED_CONTRACT } else { CONTRACT }.into());
        network.validate().unwrap();
        let actual = network.predict_cash_turn(&config).unwrap();
        let mut aggregate = 0.0;
        for p in 0..2 {
            for combo in &combos {
                let key = combo.key();
                if combo.cards().iter().any(|c| state.board.contains(c)) {
                    assert_eq!(actual[p][key], 0.0);
                    continue;
                }
                let opponent_pool = if schema == BLOCKER_POOLED_NETWORK_SCHEMA {
                    combos.iter().filter(|other| !other.cards().iter().any(|c| combo.cards().contains(c)))
                        .map(|other| state.ranges[1-p][other.key()] * (queries[1-p][other.key()][94] * 0.5) as f64).sum::<f64>()
                        / masses[p][key]
                } else { pools[1-p] };
                let residual = -0.1 + 0.025*pools[p] + 0.075*opponent_pool
                    + 0.05*(queries[p][key][94]*0.5) as f64;
                assert!((actual[p][key] - baseline[p][key] - residual*4.0).abs() < 1e-6);
                aggregate += state.ranges[p][key]*masses[p][key]*actual[p][key];
            }
        }
        assert!(actual[0][absent].abs() > 0.01, "zero own reach must not censor a query");
        assert!(aggregate < -0.1, "pooled cash values must not be projected to zero sum");
        }
    }

    #[test]
    fn card_removed_pool_uses_raw_opponent_reaches_for_counterfactual_queries() {
        let combos = all_combos();
        let board = [8, 13, 22, 31];
        let keys = combos.iter().filter(|c| !c.cards().iter().any(|v| board.contains(v)))
            .map(|c| c.key()).collect::<Vec<_>>();
        let embeddings: [Vec<f32>; 2] = std::array::from_fn(|_| keys.iter()
            .map(|key| combos[*key].cards().iter().map(|c| *c as f32).sum()).collect());
        let mut ranges = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        ranges[0][Combo::new(50, 51).key()] = 1.0;
        for (cards, weight) in [([50, 40], 0.25), ([51, 44], 0.25), ([32, 33], 0.5)] {
            ranges[1][Combo::new(cards[0], cards[1]).key()] = weight;
        }
        let masses: [Vec<f64>; 2] = std::array::from_fn(|p|
            compatible_masses_from_card_marginals(&combos, &ranges[1-p]));
        let actual = card_removed_opponent_embeddings(&embeddings, &keys, &ranges, &masses, 1);
        for player in 0..2 {
            for (row, key) in keys.iter().copied().enumerate() {
                let expected = if masses[player][key] > EPSILON {
                    combos.iter().filter(|other| !other.cards().iter().any(|c| combos[key].cards().contains(c)))
                        .map(|other| ranges[1-player][other.key()] * other.cards().iter().map(|c| *c as f64).sum::<f64>())
                        .sum::<f64>() / masses[player][key]
                } else { 0.0 };
                assert!((actual[player][row] as f64 - expected).abs() < 1e-5);
            }
        }
        let row = keys.iter().position(|key| *key == Combo::new(32, 33).key()).unwrap();
        assert_eq!(ranges[0][keys[row]], 0.0);
        assert_eq!(actual[0][row], 92.5, "joint pooling would censor both compatible opponent hands");
    }
}
