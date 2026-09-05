//! Count support on the exact authentic training trajectories without computing
//! action values. This selects a useful critic budget, not profitable actions.
use super::*;

fn trajectory_keys(
    policy: &dyn ResponsePolicy,
    config: &ResponseEvaluationConfig,
    responder: usize,
    deal: &Deal,
    index: u64,
) -> Vec<(Street, [u64; 4])> {
    let mut rng = SplitMix64::new(derived_seed(config.seed, responder as u64, index + 1));
    let mut state = GameState::initial(&config.game);
    let mut output = Vec::new();
    while state.terminal.is_none() {
        let actions = state.legal_actions(&config.game);
        let mix = policy.strategy(&state, deal, &actions, &config.game);
        if state.actor == responder
            && (!config.postflop_only_response || state.street != Street::Preflop)
        {
            output.push((
                state.street,
                [
                    information_set(&state, deal, &config.game).0,
                    observable_backoff_information_set(&state, deal, &config.game, &actions).0,
                    coarse_observable_backoff_information_set(&state, deal, &config.game, &actions)
                        .0,
                    strategic_observable_backoff_information_set(
                        &state,
                        deal,
                        &config.game,
                        &actions,
                    )
                    .0,
                ],
            ));
        }
        state = state.apply(&actions[sample_index(&mix, &mut rng)], &config.game);
    }
    output
}

#[test]
fn cheap_support_replay_matches_actual_training_observations() {
    let (mut policy, _) = super::super::super::tests::tabular_fixture();
    let table = Arc::get_mut(&mut policy.table).unwrap();
    table.config.effective_stack_bb = 2.0;
    table.nodes.clear();
    let mut config = response_config(table.config.clone(), "unused".into(), 88004, 16, 32);
    for postflop_only in [false, true] {
        config.postflop_only_response = postflop_only;
        for seat in 0..2 {
            let mut chance = SplitMix64::new(derived_seed(config.seed, seat as u64, 0));
            for index in 0..16 {
                let deal = Deal::sample(&mut chance);
                let actual = trajectory_keys(&policy, &config, seat, &deal, index);
                let mut rng = SplitMix64::new(derived_seed(config.seed, seat as u64, index + 1));
                let mut observations = Vec::new();
                collect_trajectory_decisions(
                    &policy,
                    GameState::initial(&config.game),
                    &deal,
                    &config.game,
                    seat,
                    config.rollouts_per_action,
                    config.exact_terminal_training_values,
                    config.conditional_preflop_runouts,
                    config.postflop_only_response,
                    config.seed,
                    index,
                    &mut rng,
                    &mut observations,
                );
                let expected: Vec<_> = observations
                    .into_iter()
                    .map(|row| (row.keys[0].1.street, row.keys.map(|entry| entry.0)))
                    .collect();
                assert_eq!(actual, expected, "seat={seat} index={index}");
            }
        }
    }
}

#[test]
#[ignore = "authentic full-size support census; explicit inputs and external resource guard required"]
fn sampled_profile_support_census() {
    let source = PathBuf::from(std::env::var("POKER_FLOP_PILOT_CHECKPOINT").unwrap());
    let policy_seed: u64 = std::env::var("POKER_FLOP_RESPONSE_POLICY_SEED")
        .unwrap()
        .parse()
        .unwrap();
    assert!([87001, 87002].contains(&policy_seed));
    let checkpoint_sha = sha256_file(&source).unwrap();
    let table = Arc::new(InferenceTable::read(&source).unwrap());
    assert_eq!(table.rounds, 800);
    assert_eq!(table.config.effective_stack_bb, 20.0);
    let config = response_config(table.config.clone(), source, 88004, 1024, 32);
    let (policy, patch) = profile(table, policy_seed);
    for seat in 0..2 {
        let started = Instant::now();
        let mut chance = SplitMix64::new(derived_seed(config.seed, seat as u64, 0));
        let mut banks: [BTreeMap<u64, (Street, u64)>; 4] = std::array::from_fn(|_| BTreeMap::new());
        for index in 0..config.training_deals {
            let deal = Deal::sample(&mut chance);
            for (street, keys) in trajectory_keys(&policy, &config, seat, &deal, index) {
                for (bank, key) in banks.iter_mut().zip(keys) {
                    bank.entry(key).or_insert((street, 0)).1 += 1;
                }
            }
            if [16, 128, 512, 1024].contains(&(index + 1)) {
                let layers: Vec<_> = banks.iter().map(|bank| {
                    let counts: Vec<_> = [Street::Preflop, Street::Flop, Street::Turn, Street::River]
                        .into_iter().map(|street| {
                            let rows: Vec<_> = bank.values().filter(|row| row.0 == street).collect();
                            serde_json::json!({ "street": street, "groups": rows.len(),
                                "observations": rows.iter().map(|row| row.1).sum::<u64>(),
                                "maximumSupport": rows.iter().map(|row| row.1).max().unwrap_or(0),
                                "supportedGroups": rows.iter().filter(|row| row.1 >= config.minimum_range_particles).count() })
                        }).collect();
                    counts
                }).collect();
                println!(
                    "{}",
                    serde_json::json!({ "stage": "sampled_response_support", "seat": seat,
                    "deals": index + 1, "checkpointSha256": checkpoint_sha, "policySeed": policy_seed,
                    "responseSeed": config.seed, "minimumRangeParticles": config.minimum_range_particles,
                    "layersExactFineCoarseStrategic": layers, "seconds": started.elapsed().as_secs_f64(),
                    "flopDiagnostics": patch.sampled.as_ref().unwrap().diagnostics() })
                );
            }
        }
    }
}
