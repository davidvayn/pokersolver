//! Uses the existing response learner, calibration rule and independent
//! evaluation against the experimental policy. Rejected critics retain raw
//! diagnostic results, never a fabricated zero exploitability certificate.
use super::*;
use std::io::Write;
mod support;
mod tail;
mod lbr;

fn profile(
    table: Arc<InferenceTable>,
    seed: u64,
) -> (turn::TabularTurnPolicy, Arc<flop::FlopPatch>) {
    profile_with_turn_iterations(table, seed, 4)
}

// Keep the archived four-iteration control explicit. Higher-iteration full-hand
// candidates use the same preflop/flop policy and complete river descendants.
fn profile_with_turn_iterations(
    table: Arc<InferenceTable>,
    seed: u64,
    iterations: u64,
) -> (turn::TabularTurnPolicy, Arc<flop::FlopPatch>) {
    assert!([4, 16, 64].contains(&iterations));
    let mut patch = flop::FlopPatch::terminal(&TerminalFlopOptions {
        equity_samples: 2048,
        weight: 0.5,
    });
    patch.sampled = Some(FlopResolve::new(32, seed, 2_000_000));
    let patch = Arc::new(patch);
    let base = TabularResponsePolicy {
        table,
        coverage: RefCell::default(),
        flop_patch: Some(patch.clone()),
        flop_backoff: None,
        completion_coverage: RefCell::default(),
    };
    (
        turn::TabularTurnPolicy::new(
            base,
            TurnResolveOptions {
                iterations,
                safe_bilateral: false,
                maximum_policy_rows: 20000,
            },
        ),
        patch,
    )
}

fn response_config(
    game: BlueprintConfig,
    source: PathBuf,
    seed: u64,
    training: u64,
    evaluation: u64,
) -> ResponseEvaluationConfig {
    ResponseEvaluationConfig {
        game,
        source: ResponsePolicySource::TabularCheckpoint(source),
        seed,
        training_deals: training,
        calibration_deals: evaluation,
        evaluation_deals: evaluation,
        rollouts_per_action: 2,
        minimum_range_particles: 4,
        maximum_granularity: ResolverGranularity::StrategicObservableBackoff,
        turn_resolver: Some(TurnResolveOptions {
            iterations: 4,
            safe_bilateral: false,
            maximum_policy_rows: 20000,
        }),
        terminal_flop: Some(TerminalFlopOptions {
            equity_samples: 2048,
            weight: 0.5,
        }),
        flop_backoff: None,
        exact_terminal_training_values: true,
        conditional_preflop_runouts: false,
        postflop_only_response: false,
        response_workers: 1,
    }
}

#[test]
#[ignore = "full-size response cost/interface probe; explicit inputs, new output directory and external guard required"]
fn sampled_profile_response_probe() {
    let source =
        PathBuf::from(std::env::var("POKER_FLOP_PILOT_CHECKPOINT").expect("explicit checkpoint"));
    let output = PathBuf::from(
        std::env::var("POKER_FLOP_PILOT_OUTPUT_DIR").expect("explicit new output directory"),
    );
    assert!(output.is_dir());
    let policy_seed: u64 = std::env::var("POKER_FLOP_RESPONSE_POLICY_SEED")
        .expect("explicit frozen policy seed")
        .parse()
        .unwrap();
    assert!([87001, 87002].contains(&policy_seed));
    let checkpoint_sha = sha256_file(&source).unwrap();
    let table = Arc::new(InferenceTable::read(&source).unwrap());
    assert_eq!(table.rounds, 800);
    assert_eq!(table.config.effective_stack_bb, 20.0);
    let config = response_config(table.config.clone(), source, 88004, 16, 32);
    let (policy, patch) = profile(table, policy_seed);
    for seat in 0..2 {
        let started = Instant::now();
        let (preflop, resolver) = train_learned_response(&policy, &config, seat);
        let name = format!("response-seat{seat}.msgpack");
        let encoded = rmp_serde::to_vec_named(&serde_json::json!({
            "schema": "experimental-sampled-flop-response-training-v1",
            "checkpointSha256": checkpoint_sha, "policySeed": policy_seed,
            "responseSeed": config.seed, "responder": seat,
            "preflop": preflop, "resolver": resolver,
        }))
        .unwrap();
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output.join(&name))
            .unwrap();
        file.write_all(&encoded).unwrap();
        file.sync_all().unwrap();
        println!(
            "{}",
            serde_json::json!({ "stage": "sampled_response_trained", "seat": seat,
            "file": name, "sha256": format!("{:x}", Sha256::digest(&encoded)),
            "seconds": started.elapsed().as_secs_f64(), "flopDiagnostics": patch.sampled.as_ref().unwrap().diagnostics() })
        );
        let calibration = evaluate_resolver(
            &policy,
            &preflop,
            &resolver,
            &config,
            seat,
            config.calibration_deals,
            u64::MAX - 1,
            true,
        );
        let qualified = response_lower_bound_passes_calibration(
            calibration.approximate_one_sided_99_5_percent_gain_lower_bound_bb,
        );
        // Evaluate the frozen raw critic on independent deals even if rejected,
        // and label it explicitly. Never replace this diagnostic with zero.
        let raw_holdout = evaluate_resolver(
            &policy,
            &preflop,
            &resolver,
            &config,
            seat,
            config.evaluation_deals,
            u64::MAX,
            true,
        );
        println!(
            "{}",
            serde_json::json!({ "stage": "sampled_response_probe", "seat": seat,
            "checkpointSha256": checkpoint_sha, "policySeed": policy_seed, "responseSeed": config.seed,
            "trainingDeals": config.training_deals, "calibrationDeals": config.calibration_deals,
            "evaluationDeals": config.evaluation_deals, "rolloutsPerAction": config.rollouts_per_action,
            "minimumRangeParticles": config.minimum_range_particles, "method": response_method(&config),
            "calibration": calibration, "qualifiedByCalibration": qualified, "rawIndependentHoldout": raw_holdout,
            "qualifiedHoldout": if qualified { Some(&raw_holdout) } else { None },
            "seconds": started.elapsed().as_secs_f64(), "flopDiagnostics": patch.sampled.as_ref().unwrap().diagnostics(),
            "turnDiagnostics": policy.take_resolution_diagnostics(),
            "validation": "bounded_all_street_one_step_response_probe_not_exploitability_upper_bound" })
        );
    }
}

#[test]
fn existing_response_learner_and_rollouts_use_the_sampled_full_hand_policy() {
    let (mut base, _) = super::super::tests::tabular_fixture();
    let table = Arc::get_mut(&mut base.table).unwrap();
    table.config.effective_stack_bb = 2.0;
    table.nodes.clear();
    let config = response_config(
        table.config.clone(),
        PathBuf::from("unused-test-only-source"),
        88004,
        8,
        8,
    );
    let (policy, patch) = profile(base.table, 87001);
    let (preflop, resolver) = train_learned_response(&policy, &config, 0);
    assert!(
        patch.sampled.as_ref().unwrap().diagnostics()["solves"]
            .as_u64()
            .unwrap()
            > 0
    );
    let unchanged = evaluate_resolver(&policy, &preflop, &resolver, &config, 0, 8, u64::MAX, false);
    assert_eq!(unchanged.estimated_gain_bb, 0.0);
    assert_eq!(unchanged.gain_standard_error_bb, 0.0);
    let raw = evaluate_resolver(&policy, &preflop, &resolver, &config, 0, 8, u64::MAX, true);
    assert!(raw.estimated_gain_bb.is_finite() && raw.gain_standard_error_bb.is_finite());
    assert_eq!(raw.responder, 0);
}
