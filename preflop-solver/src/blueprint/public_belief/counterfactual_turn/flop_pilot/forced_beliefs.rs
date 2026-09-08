//! Explicit public-action interventions for bounded value diagnostics only.
//! A forced proposer is not a trained average and is never exported as one.
use super::*;

fn force_root_action(source: &Solution, label: &str) -> Result<Solution, String> {
    frozen_response::Frozen::new(source)?;
    if label != "check" && !label.starts_with("bet_to_") {
        return Err("forced diagnostic requires check or a non-all-in opening bet".into());
    }
    let mut forced = source.clone();
    let row = forced.strategies.iter_mut()
        .find(|r| r.public_history == source.state.public_history)
        .ok_or("missing root row")?;
    let action = row.action_labels.iter().position(|a| a == label)
        .ok_or("forced action is not legal at this root")?;
    let n = row.action_labels.len();
    row.probabilities.fill(0.0);
    for c in 0..COMBO_COUNT {
        if source.state.ranges[row.actor][c] > 0.0 {
            row.probabilities[c*n+action] = 1.0;
        }
    }
    frozen_response::Frozen::new(&forced)?;
    Ok(forced)
}

#[test]
#[ignore = "four native labels on explicit forced public-action beliefs; pinned files, exclusive output and external resource guard"]
fn saved_forced_action_native_capture() {
    let input = fs::read(std::env::var("POKER_FORCED_CANDIDATE").unwrap()).unwrap();
    assert!(input.len() <= 8*1024*1024);
    let source_sha = format!("{:x}", Sha256::digest(&input));
    assert_eq!(source_sha, std::env::var("POKER_FORCED_CANDIDATE_SHA").unwrap());
    let source: Solution = serde_json::from_slice(&input).unwrap();
    let before = serde_json::to_vec(&source).unwrap();
    let model = PublicValueNetwork::read(Path::new(&std::env::var("POKER_FORCED_MODEL").unwrap())).unwrap();
    assert_eq!(model.artifact_sha256().unwrap(), std::env::var("POKER_FORCED_MODEL_SHA").unwrap());
    assert_eq!(source.learned_leaf_model_sha256.as_deref(), model.artifact_sha256());
    assert_eq!(source.iterations, 128);
    assert_eq!(source.turn_iterations, 64);
    assert_eq!(source.response_turn_iterations.unwrap_or(64), 64);
    frozen_response::Frozen::new(&source).unwrap();
    let root = source.strategies.iter().find(|r| r.public_history == source.state.public_history).unwrap();
    assert!(root.action_labels.iter().any(|a| a == "check"));
    let mass = compatible_masses_from_card_marginals(&all_combos(), &source.state.ranges[1-root.actor]);
    let joint = joint_compatibility_mass(&source.state.ranges);
    assert!(joint > 1e-12);
    let frequency = |a: usize| (0..COMBO_COUNT).map(|c|
        source.state.ranges[root.actor][c]*mass[c]*root.probabilities[c*root.action_labels.len()+a] as f64
    ).sum::<f64>()/joint;
    // Explicitly probe checking and the least-used legal non-all-in opening
    // bet. Selection uses only the frozen mix, never the resulting value error.
    let bet = (0..root.action_labels.len()).filter(|a| root.action_labels[*a].starts_with("bet_to_"))
        .min_by(|a,b| frequency(*a).total_cmp(&frequency(*b))).unwrap();
    let check = root.action_labels.iter().position(|a| a == "check").unwrap();
    let mut turns = (0..52u8).filter(|c| !source.state.board.contains(c)).collect::<Vec<_>>();
    let sample_seed: u64 = std::env::var("POKER_FORCED_SAMPLE_SEED").unwrap().parse().unwrap();
    let mut rng = SplitMix64::new(sample_seed);
    let sampled = [turns.remove(rng.index(turns.len())), turns.remove(rng.index(turns.len()))];
    let started = std::time::Instant::now();
    let mut targets = Vec::new();
    let mut predictions = Vec::new();
    let mut interventions = Vec::new();
    for action in [check, bet] {
        let label = &root.action_labels[action];
        let forced = force_root_action(&source, label).unwrap();
        let proposer_sha = format!("{:x}", Sha256::digest(serde_json::to_vec(&forced).unwrap()));
        let frozen = frozen_response::Frozen::new(&forced).unwrap();
        let mut prefix = source.state.public_history.clone();
        prefix.push(format!("Flop:p{}:{}", root.actor, label));
        for turn in sampled {
            let config = frozen.belief_queries(turn).unwrap().into_iter()
                .filter(|q| q.state.public_history.starts_with(&prefix)
                    && joint_compatibility_mass(&q.state.ranges) > 1e-12)
                .max_by(|a,b| joint_compatibility_mass(&a.state.ranges)
                    .total_cmp(&joint_compatibility_mass(&b.state.ranges))).unwrap();
            let values = solve(config.clone()).unwrap();
            targets.push(value_targets::Target::new(source.iterations, &config, &values).unwrap()
                .with_distribution("explicit_forced_root_action_public_belief_native_label"));
            predictions.push(model.predict_native_turn(&config).unwrap());
            interventions.push(serde_json::json!({"action":label,"actor":root.actor,
                "source_action_frequency":frequency(action),"forced_action_probability":1.0,
                "intervened_proposer_sha256":proposer_sha,"turn":turn,
                "public_belief_sha256":format!("{:x}",Sha256::digest(serde_json::to_vec(&config.state).unwrap()))}));
        }
    }
    assert_eq!(before, serde_json::to_vec(&source).unwrap());
    let root_bytes = serde_json::to_vec(&serde_json::json!({"game":source.game,"public":source.state})).unwrap();
    let payload = serde_json::json!({"schema":"hu-native-turn-cfv-dataset-v1","game":source.game,
        "source_public_input_sha256":format!("{:x}",Sha256::digest(root_bytes)),
        "source_policy_sha256":source_sha,"source_policy_unchanged":true,
        "proposal_model_sha256":model.artifact_sha256(),"seed":source.seed,"sampling_seed":sample_seed,
        "flop_iterations":128,"turn_iterations":64,"maximum_states":4,"observed_queries":4,
        "native_label_queries":4,"capture_selection":"explicit_forced_root_actions_diagnostic_only",
        "validation":{"status":"research_only","reasons":["forced proposer is not a trained average; finite native reference, not full-game response or EV sampling-error qualification"]},
        "targets":targets,"predictions":predictions,"interventions":interventions,
        "seconds":started.elapsed().as_secs_f64(),"releaseAccepted":false});
    let encoded = serde_json::to_vec(&payload).unwrap();
    assert!(encoded.len() <= 16*1024*1024);
    let file = fs::OpenOptions::new().write(true).create_new(true)
        .open(std::env::var("POKER_FORCED_OUTPUT").unwrap()).unwrap();
    let mut gzip = GzEncoder::new(file, Compression::default());
    gzip.write_all(&encoded).unwrap();
    gzip.finish().unwrap().sync_all().unwrap();
    println!("{}",serde_json::json!({"states":4,"seconds":started.elapsed().as_secs_f64(),"sourcePolicyUnchanged":true}));
}

#[test]
fn forced_root_action_preserves_source_and_propagates_coherent_private_reaches() {
    let mut model = super::super::super::tests::zero_shared_value_network();
    model.schema = "hu-public-belief-combo-value-network-v4".into();
    model.value_normalization = Some("payoff-exposure".into());
    model.prediction_contract = Some("native-turn-cfv-full-stack-v1".into());
    model.artifact_sha256 = Some("d".repeat(64));
    model.validate().unwrap();
    let mut game = BlueprintConfig::default();
    game.effective_stack_bb = 20.0;
    let mut ranges = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
    let hands = [Combo::new(51, 50).key(), Combo::new(47, 46).key()];
    for p in 0..2 { ranges[p][hands[p]] = 1.0; }
    let state = PublicBeliefState::flop_start([0, 5, 10], 1, [1.0, 1.0], ranges);
    let source = train_with_evaluator(game, state, 100101, 2, 4, false, 1, 1,
        None, continuation::Evaluator::Learned(&model)).unwrap();
    let before = serde_json::to_vec(&source).unwrap();
    let forced = force_root_action(&source, "check").unwrap();
    assert_eq!(before, serde_json::to_vec(&source).unwrap());
    assert_ne!(before, serde_json::to_vec(&forced).unwrap());
    let root = forced.strategies.iter().find(|r| r.public_history == source.state.public_history).unwrap();
    let check = root.action_labels.iter().position(|s| s == "check").unwrap();
    for c in 0..COMBO_COUNT {
        for a in 0..root.action_labels.len() {
            assert_eq!(root.probabilities[c*root.action_labels.len()+a],
                if c == hands[1] && a == check { 1.0 } else { 0.0 });
        }
    }
    let mut history = source.state.public_history.clone();
    history.extend(["Flop:p1:check".into(), "Flop:p0:check".into(), "deal:Turn".into()]);
    let queries = frozen_response::Frozen::new(&forced).unwrap().belief_queries(2).unwrap();
    let query = queries.iter().find(|q| q.state.public_history == history).unwrap();
    assert_eq!(query.state.ranges[1], source.state.ranges[1]);
    assert!(query.state.ranges[0][hands[0]] > 0.0);
    assert!(force_root_action(&source, "fold").is_err());
    assert!(force_root_action(&source, "bet_all_in_to_19.000bb").is_err());
}
