use super::*;

#[test]
fn cash_card_arrival_features_preserve_recall_without_changing_home_or_leaking_future_cards() {
    let cash = config().game;
    let mut home = cash.clone();
    home.cash_rules = None;
    let mut state = GameState::initial(&cash);
    state.street = Street::River;
    let deal = Deal::from_sampled_cards([[48, 49], [4, 5]], [8, 13, 22, 31, 40]);
    let swapped = Deal::from_sampled_cards(deal.holes, [8, 13, 22, 40, 31]);
    let permuted = Deal::from_sampled_cards(deal.holes, [22, 8, 13, 31, 40]);
    let features = encode_state_features(&state, &deal, &cash);
    assert_eq!(features.len(), CASH_STATE_FEATURE_COUNT);
    assert_ne!(features, encode_state_features(&state, &swapped, &cash));
    assert_eq!(features, encode_state_features(&state, &permuted, &cash));
    assert_eq!(
        encode_state_features(&state, &deal, &home),
        encode_state_features(&state, &swapped, &home)
    );
    state.street = Street::Preflop;
    assert_eq!(
        encode_state_features(&state, &deal, &cash),
        encode_state_features(&state, &swapped, &cash)
    );
    assert!(
        encode_state_features(&state, &deal, &cash)[STATE_FEATURE_COUNT..]
            .iter()
            .all(|v| *v == 0.)
    );
}

pub(super) fn config() -> SampleGenerationConfig {
    let mut game = super::super::tests::tiny_config();
    game.small_blind_bb = 0.4;
    game.effective_stack_bb = 20.0;
    game.cash_rules = Some(crate::cash_game::study_rules("nl25").unwrap());
    SampleGenerationConfig {
        game,
        traversals: 2,
        start_iteration: 0,
        seed: 41,
        max_records: 1000,
        output: PathBuf::from("unused-cash-test.jsonl.gz"),
        network_path: None,
        trajectory_sampling: false,
        evaluate_trajectory_values: false,
        value_rollouts_per_action: 2,
        enumerate_turn_river_chance: false,
    }
}

#[test]
fn cash_neural_terminals_train_each_seats_own_net_payoff() {
    let mut generator = SampleGenerator::new(config()).unwrap();
    let deal = Deal::from_sampled_cards([[48, 49], [4, 5]], [8, 13, 22, 31, 40]);
    let state = GameState {
        street: Street::River,
        invested: [5.0, 5.0],
        terminal: Some(Terminal::Showdown),
        ..GameState::initial(&generator.config.game)
    };
    let zero = generator.external_sampling(state.clone(), &deal, 0, 1, 1.0);
    let one = generator.external_sampling(state.clone(), &deal, 1, 1, 1.0);
    assert!((zero - 4.56).abs() < 1e-12);
    assert_eq!(one, -5.0);
    assert!((zero + one + 0.44).abs() < 1e-12);
    let mut rng = SplitMix64::new(3);
    assert_eq!(
        generator.value_only_external_sampling(state, &deal, 1, &mut rng),
        one
    );
}

#[test]
fn cash_closed_opponent_responses_integrate_own_payoffs_and_keep_average_visits() {
    let mut generator = SampleGenerator::new(config()).unwrap();
    let deal = Deal::from_sampled_cards([[48, 49], [4, 5]], [8, 13, 22, 31, 40]);
    for street in [Street::Preflop, Street::River] {
        let mut root = GameState::initial(&generator.config.game);
        if street == Street::River {
            root.street = street;
            root.actor = 1;
            root.invested = [5.0, 5.0];
            root.street_invested = [0.0, 0.0];
        }
        let shove = root
            .legal_actions(&generator.config.game)
            .into_iter()
            .find(|action| action.label.contains("all_in"))
            .unwrap();
        let response = root.apply(&shove, &generator.config.game);
        let traverser = 1 - response.actor;
        let actions = response.legal_actions(&generator.config.game);
        assert_eq!(actions.len(), 2);
        let strategy = generator.current_strategy(&response, &deal, &actions);
        let mut expected = [0.0; 2];
        for (action, probability) in actions.iter().zip(&strategy) {
            let child = response.apply(action, &generator.config.game);
            assert!(child.terminal.is_some());
            for (player, value) in child
                .own_utilities(&deal, &generator.config.game)
                .iter()
                .enumerate()
            {
                expected[player] += probability * value;
            }
        }
        let expected_house = if street == Street::Preflop { 0.9 } else { 1.12 };
        assert!((expected.iter().sum::<f64>() + expected_house).abs() < 1e-12);
        let before_records = generator.records.len();
        for seed in 0..12 {
            generator.rng = SplitMix64::new(seed);
            let before_rng = generator.rng.clone();
            assert_eq!(
                generator.external_sampling(response.clone(), &deal, traverser, seed, 0.25),
                expected[traverser]
            );
            assert_eq!(generator.rng.next_u64(), before_rng.clone().next_u64());
            let record = generator.records.last().unwrap();
            assert!(matches!(record.kind, SampleKind::AverageStrategy));
            assert_eq!(record.reach_probability, 0.25);
            assert_eq!(
                record.targets,
                strategy
                    .iter()
                    .map(|value| *value as f32)
                    .collect::<Vec<_>>()
            );
            let mut rng = SplitMix64::new(seed);
            let mut before_rng = rng.clone();
            assert_eq!(
                generator.value_only_external_sampling(
                    response.clone(),
                    &deal,
                    traverser,
                    &mut rng
                ),
                expected[traverser]
            );
            assert_eq!(rng.next_u64(), before_rng.next_u64());
        }
        assert_eq!(generator.records.len() - before_records, 12);
    }
}

#[test]
fn terminal_integration_does_not_close_open_cash_trees_or_change_home_sampling() {
    let mut generator = SampleGenerator::new(config()).unwrap();
    let deal = Deal::from_sampled_cards([[48, 49], [4, 5]], [8, 13, 22, 31, 40]);
    let root = GameState::initial(&generator.config.game);
    let actions = root.legal_actions(&generator.config.game);
    let strategy = generator.current_strategy(&root, &deal, &actions);
    assert!(generator
        .cash_closed_terminal_expectation(&root, &deal, &actions, &strategy, 1)
        .is_none());
    generator.config.game.cash_rules = None;
    let root = GameState::initial(&generator.config.game);
    let shove = root
        .legal_actions(&generator.config.game)
        .into_iter()
        .find(|action| action.label.contains("all_in"))
        .unwrap();
    let response = root.apply(&shove, &generator.config.game);
    let actions = response.legal_actions(&generator.config.game);
    let strategy = generator.current_strategy(&response, &deal, &actions);
    assert!(generator
        .cash_closed_terminal_expectation(&response, &deal, &actions, &strategy, 0)
        .is_none());
    let mut rng = SplitMix64::new(0);
    let mut before_rng = rng.clone();
    generator.value_only_external_sampling(response, &deal, 0, &mut rng);
    assert_ne!(rng.next_u64(), before_rng.next_u64());
}

#[test]
fn native_weights_cannot_cross_cash_rules_depth_or_baseline_contract() {
    let generator = SampleGenerator::new(config()).unwrap();
    let scorer = DenseScorer {
        layers: vec![DenseLayer {
            input_size: CASH_MODEL_INPUT_COUNT,
            output_size: 1,
            activation: DenseActivation::Linear,
            weights: vec![0.0; CASH_MODEL_INPUT_COUNT],
            biases: vec![0.0],
        }],
    };
    let mut bundle = TrainingNetworkBundle {
        schema: CASH_TRAINING_NETWORK_SCHEMA.into(),
        input_size: CASH_MODEL_INPUT_COUNT,
        strategy_transform: StrategyTransform::RegretMatching,
        networks: vec![scorer.clone(), scorer.clone()],
        postflop_networks: None,
        sampling_baseline: None,
        sampling_baseline_scale: None,
        cash_rules: generator.config.game.cash_rules.clone(),
        cash_depth_bb: Some(20.0),
        cash_action_abstraction: Some(generator.config.game.action_abstraction.clone()),
    };
    validate_training_bundle(&bundle).unwrap();
    let mut obsolete = bundle.clone();
    obsolete.schema = "hu-neural-own-payoff-training-networks-v2".into();
    assert!(validate_training_bundle(&obsolete).is_err());
    obsolete = bundle.clone();
    obsolete.input_size = MODEL_INPUT_COUNT;
    assert!(validate_training_bundle(&obsolete).is_err());
    bundle.validate_game(&generator.config.game).unwrap();
    let mut wrong = generator.config.game.clone();
    wrong.cash_rules = None;
    assert!(bundle.validate_game(&wrong).is_err());
    wrong = generator.config.game.clone();
    wrong.action_abstraction.open_sizes_bb.push(4.0);
    assert!(bundle.validate_game(&wrong).is_err());
    wrong = generator.config.game.clone();
    wrong.effective_stack_bb = 40.0;
    assert!(bundle.validate_game(&wrong).is_err());
    wrong = generator.config.game.clone();
    wrong.cash_rules = Some(crate::cash_game::study_rules("nl25-rake-off-control").unwrap());
    assert!(bundle.validate_game(&wrong).is_err());
    bundle.sampling_baseline = Some(scorer);
    assert!(validate_training_bundle(&bundle).is_err());
    bundle.sampling_baseline = None;
    bundle.schema = TRAINING_NETWORK_SCHEMA.into();
    assert!(validate_training_bundle(&bundle).is_err());
}
