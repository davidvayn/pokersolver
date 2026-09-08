//! One guarded, immutable full-deal cluster using the existing frozen LBR.
use super::*;
use std::io::Write;
use std::time::Instant;

fn pinned_policy() -> NativeFullHandPolicy {
    let source = PathBuf::from(std::env::var("POKER_NATIVE_CHECKPOINT").unwrap());
    let source_sha = std::env::var("POKER_NATIVE_CHECKPOINT_SHA").unwrap();
    assert_eq!(sha256_file(&source).unwrap(), source_sha);
    let model_path = PathBuf::from(std::env::var("POKER_NATIVE_VALUE_MODEL").unwrap());
    let model_sha = std::env::var("POKER_NATIVE_VALUE_MODEL_SHA").unwrap();
    let model = Arc::new(PublicValueNetwork::read(&model_path).unwrap());
    assert_eq!(model.artifact_sha256().unwrap(), model_sha);
    let preflop = Arc::new(FrozenPreflopPolicy::read(&source).unwrap());
    assert!([8, 16, 32, 128, 512, 1024].contains(&preflop.rounds));
    let game = preflop.game.clone();
    assert_eq!(game.effective_stack_bb, 20.0);
    NativeFullHandPolicy::with_compact_continuation(
        preflop,
        NativeFlopOptions {
            seed: 100101,
            iterations: 128,
            training_turn_iterations: 64,
            response_turn_iterations: 64,
        },
        model,
    )
    .unwrap()
}

#[test]
#[ignore = "one pinned full-hand LBR cluster; explicit inputs/output and external resource guard"]
fn compact_native_full_hand_lbr_probe() {
    let policy = pinned_policy();
    let game = policy.preflop.game.clone();
    let cohort = std::env::var("POKER_NATIVE_LBR_COHORT").unwrap();
    let domain = match cohort.as_str() {
        "screening" => 0,
        "calibration" => 1,
        "holdout" => 2,
        _ => panic!("explicit disjoint LBR cohort required"),
    };
    let index: u64 = std::env::var("POKER_NATIVE_LBR_INDEX")
        .unwrap()
        .parse()
        .unwrap();
    assert!(index < 4096);
    let chance_seed = derived_seed(29001, domain, index * 2);
    let action_seed = derived_seed(29001, domain, index * 2 + 1);
    let deal = Deal::sample(&mut SplitMix64::new(chance_seed));
    let output = PathBuf::from(std::env::var("POKER_NATIVE_OUTPUT").unwrap());
    assert!(!output.exists());
    eprintln!(
        "{}",
        serde_json::json!({"stage":"native_full_hand_lbr_started",
        "cohort":cohort,"index":index,"chanceSeed":chance_seed,"actionSeed":action_seed,
        "routeSha256":policy.route_identity()})
    );
    let started = Instant::now();
    let report = sampled_flop_policy::paired_lbr_hand(&policy, &game, &deal, action_seed).unwrap();
    let value = serde_json::json!({"schema":"compact-native-full-hand-lbr-cluster-v1",
        "cohort":cohort,"index":index,"chanceSeed":chance_seed,
        "deal":{"holes":deal.holes,"board":deal.board},"report":report,
        "seconds":started.elapsed().as_secs_f64(),
        "diagnostics":policy.take_resolution_diagnostics(),"releaseAccepted":false});
    let bytes = serde_json::to_vec(&value).unwrap();
    assert!(bytes.len() < 1024 * 1024);
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output)
        .unwrap();
    file.write_all(&bytes).unwrap();
    file.sync_all().unwrap();
    eprintln!(
        "{}",
        serde_json::json!({"stage":"native_full_hand_lbr_complete",
        "seconds":started.elapsed().as_secs_f64(),"cohort":cohort,"index":index,
        "pairedTotalResponseGainBb":report["pairedTotalResponseGainBb"]})
    );
}

#[test]
#[ignore = "bounded existing full-continuation response learner; pinned source and external resource guard"]
fn compact_native_learned_response_train() {
    let policy = pinned_policy();
    let seat: usize = std::env::var("POKER_NATIVE_RESPONSE_SEAT")
        .unwrap()
        .parse()
        .unwrap();
    let deals: u64 = std::env::var("POKER_NATIVE_RESPONSE_DEALS")
        .unwrap()
        .parse()
        .unwrap();
    assert!(seat < 2 && [2, 8, 16, 32].contains(&deals));
    let output = PathBuf::from(std::env::var("POKER_NATIVE_OUTPUT").unwrap());
    assert!(!output.exists());
    let config = ResponseEvaluationConfig {
        game: policy.preflop.game.clone(),
        source: ResponsePolicySource::TabularCheckpoint(PathBuf::from(
            std::env::var("POKER_NATIVE_CHECKPOINT").unwrap(),
        )),
        training_deals: deals,
        calibration_deals: 2,
        evaluation_deals: 2,
        rollouts_per_action: 2,
        minimum_range_particles: 2,
        maximum_granularity: ResolverGranularity::StrategicObservableBackoff,
        seed: 30001,
        turn_resolver: None,
        terminal_flop: None,
        flop_backoff: None,
        exact_terminal_training_values: true,
        conditional_preflop_runouts: false,
        postflop_only_response: false,
        response_workers: 1,
    };
    let started = Instant::now();
    let (preflop, resolver) = train_learned_response(&policy, &config, seat);
    let value = serde_json::json!({"schema":"compact-native-learned-response-v1",
        "responder":seat,"trainingDeals":deals,"responseSeed":config.seed,
        "rolloutsPerAction":config.rollouts_per_action,
        "minimumRangeParticles":config.minimum_range_particles,
        "preflop":preflop,"resolver":resolver,"seconds":started.elapsed().as_secs_f64(),
        "diagnostics":policy.take_resolution_diagnostics(),"releaseAccepted":false,
        "interpretation":"Frozen information-set response learned from actual full-policy rollouts. Training/cost only, not a gain or strength result; disjoint calibration and holdout are still required."});
    let bytes = serde_json::to_vec(&value).unwrap();
    assert!(bytes.len() < 8 * 1024 * 1024);
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output)
        .unwrap();
    file.write_all(&bytes).unwrap();
    file.sync_all().unwrap();
    eprintln!(
        "{}",
        serde_json::json!({"stage":"native_response_training_complete",
        "seat":seat,"trainingDeals":deals,"seconds":started.elapsed().as_secs_f64(),
        "preflopRows":preflop.len(),"postflopRows":resolver.decisions.len()})
    );
}
