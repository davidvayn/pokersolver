//! Fresh calibration/holdout of immutable response rows, without retraining.
//! This saves label-generation work; it does not strengthen the response class
//! or turn repeated adaptive diagnostics into an exploitability certificate.
use super::*;

#[derive(Clone, Debug)]
pub struct ResponseRecheckConfig {
    pub checkpoint: PathBuf,
    pub retained_response: PathBuf,
    pub calibration_deals: u64,
    pub evaluation_deals: u64,
    pub seed: u64,
    pub workers: usize,
}

pub fn recheck_full_game_response(
    config: ResponseRecheckConfig,
) -> Result<FullGameResponseEvaluation, Box<dyn Error>> {
    if !(2..=1_000_000).contains(&config.calibration_deals)
        || !(2..=1_000_000).contains(&config.evaluation_deals)
        || !(1..=4).contains(&config.workers)
    {
        return Err("response recheck requires 2..1000000 hands per phase and 1..4 workers".into());
    }
    let bytes = fs::read(&config.retained_response)?;
    let digest = format!("{:x}", Sha256::digest(&bytes));
    let report: FullGameResponseEvaluation = serde_json::from_slice(&bytes)?;
    if report.schema != "hu-tabular-checkpoint-information-set-response-v1"
        || report.retained_training.is_some()
        || report.uses_seed(config.seed)
    {
        return Err("recheck requires an original tabular training report and a fresh seed".into());
    }
    let pinned = ResponseEvaluationConfig {
        game: BlueprintConfig::default(), // Replaced by the verified checkpoint.
        source: ResponsePolicySource::TabularCheckpoint(config.checkpoint),
        training_deals: report.training_deals,
        calibration_deals: config.calibration_deals,
        evaluation_deals: config.evaluation_deals,
        rollouts_per_action: report.rollouts_per_action,
        minimum_range_particles: report.minimum_range_particles,
        maximum_granularity: report.maximum_granularity,
        seed: config.seed,
        response_workers: config.workers,
        turn_resolver: report.turn_resolver.clone(),
        terminal_flop: report.terminal_flop.clone(),
        flop_backoff: report.flop_backoff.clone(),
        exact_terminal_training_values: report.exact_terminal_training_values,
        conditional_preflop_runouts: report.conditional_preflop_runouts,
        postflop_only_response: report.postflop_only_response,
    };
    // Check the method and source identity before loading the large table.
    if report.method != response_method(&pinned) {
        return Err("recheck cannot silently upgrade an older response method".into());
    }
    flop::validate_report(&report, &sha256_file(match &pinned.source {
        ResponsePolicySource::TabularCheckpoint(path) => path,
        _ => unreachable!(),
    })?, report.depth_bb)?;
    let mut result = evaluate_full_game_response_inner(pinned, Some((report, digest)))?;
    result.interpretation.push_str(
        "; response training rows were reused unchanged from the hashed retained report, with no new training-phase coverage measurement; old calibration data are not reused; fresh calibration and independent holdout seeds are separate from the original report; this is an adaptively chosen diagnostic recheck, not a multiple-testing-adjusted qualification certificate",
    );
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recheck_preserves_rows_and_profile_and_is_parallel_deterministic() {
        let (trainer, _) = super::super::tests::fixture_trainer();
        let directory = std::env::temp_dir().join(format!("response-recheck-{}", std::process::id()));
        fs::create_dir(&directory).unwrap();
        let checkpoint = directory.join("checkpoint.msgpack.gz");
        let path = directory.join("trained.json");
        trainer.write_checkpoint(&checkpoint).unwrap();
        let original = evaluate_full_game_response(ResponseEvaluationConfig {
            game: trainer.config.clone(),
            source: ResponsePolicySource::TabularCheckpoint(checkpoint.clone()),
            training_deals: 32, calibration_deals: 16, evaluation_deals: 32,
            rollouts_per_action: 2, minimum_range_particles: 2,
            maximum_granularity: ResolverGranularity::StrategicObservableBackoff,
            seed: 715, response_workers: 1, turn_resolver: None,
            terminal_flop: Some(TerminalFlopOptions { equity_samples: 128, weight: 0.25 }),
            flop_backoff: None, exact_terminal_training_values: true, postflop_only_response: false,
            conditional_preflop_runouts: true,
        }).unwrap();
        fs::write(&path, serde_json::to_vec(&original).unwrap()).unwrap();
        let mut config = ResponseRecheckConfig {
            checkpoint, retained_response: path.clone(), calibration_deals: 32,
            evaluation_deals: 71, seed: 716, workers: 1,
        };
        let serial = recheck_full_game_response(config.clone()).unwrap();
        assert!(serial.conditional_preflop_runouts);
        assert_eq!(serial.method, original.method);
        assert_eq!(serde_json::to_vec(&original.resolvers).unwrap(), serde_json::to_vec(&serial.resolvers).unwrap());
        assert_eq!(serde_json::to_vec(&original.preflop_responses).unwrap(), serde_json::to_vec(&serial.preflop_responses).unwrap());
        assert_eq!(original.terminal_flop, serial.terminal_flop);
        assert_eq!(original.policy_source_kind, serial.policy_source_kind);
        assert!(!serial.source_policy_coverage.contains_key("response_training"));
        assert_eq!(serial.retained_training.as_ref().unwrap().report_sha256, sha256_file(&path).unwrap());
        assert!(serial.uses_seed(715) && serial.uses_seed(716) && !serial.uses_seed(717));
        config.workers = 2;
        let mut parallel = recheck_full_game_response(config.clone()).unwrap();
        parallel.response_workers = 1;
        assert_eq!(serde_json::to_vec(&serial).unwrap(), serde_json::to_vec(&parallel).unwrap());
        config.seed = original.seed;
        assert!(recheck_full_game_response(config.clone()).unwrap_err().to_string().contains("fresh"));
        config.seed = 717;
        fs::write(&path, serde_json::to_vec(&serial).unwrap()).unwrap();
        assert!(recheck_full_game_response(config.clone()).unwrap_err().to_string().contains("original"));
        let mut corrupt = original.clone();
        corrupt.method = "old-or-unknown-method".to_owned();
        fs::write(&path, serde_json::to_vec(&corrupt).unwrap()).unwrap();
        assert!(recheck_full_game_response(config.clone()).unwrap_err().to_string().contains("method"));
        corrupt = original;
        corrupt.policy_sha256 = "0".repeat(64);
        fs::write(&path, serde_json::to_vec(&corrupt).unwrap()).unwrap();
        assert!(recheck_full_game_response(config).unwrap_err().to_string().contains("checkpoint"));
        fs::remove_dir_all(directory).unwrap();
    }
}
