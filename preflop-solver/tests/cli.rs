use serde_json::Value;
use std::fs;
use std::process::Command;

#[test]
fn kuhn_command_prints_machine_readable_validation() {
    let output = Command::new(env!("CARGO_BIN_EXE_preflop-solver"))
        .args(["kuhn", "--iterations", "5000"])
        .output()
        .expect("run kuhn command");
    assert!(output.status.success());
    let payload: Value = serde_json::from_slice(&output.stdout).expect("valid JSON");
    assert_eq!(payload["iterations"], 5000);
    assert!(payload["exploitability"].as_f64().unwrap().is_finite());
}

#[test]
fn solve_command_writes_versioned_artifact() {
    let output_path = std::env::temp_dir().join(format!(
        "preflop-solver-cli-{}-{}.json",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    let output = Command::new(env!("CARGO_BIN_EXE_preflop-solver"))
        .args([
            "solve",
            "--effective-stack-bb",
            "5",
            "--iterations",
            "10000",
            "--equity-samples",
            "4",
            "--seed",
            "31",
            "--output",
            output_path.to_str().expect("UTF-8 temp path"),
        ])
        .output()
        .expect("run solve command");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: Value = serde_json::from_slice(&fs::read(&output_path).expect("artifact file"))
        .expect("valid JSON");
    assert_eq!(payload["schema_version"], 1);
    assert_eq!(payload["model"], "heads-up-push-fold-monte-carlo-v1");
    assert_eq!(
        payload["strategies"]["exact_combos"]
            .as_array()
            .unwrap()
            .len(),
        1326
    );
    assert!(payload["artifact_id"]
        .as_str()
        .unwrap()
        .starts_with("hu-push-fold-"));
    fs::remove_file(output_path).expect("remove test artifact");
}

#[test]
fn blueprint_command_writes_explicit_approximate_artifact() {
    let output_path = std::env::temp_dir().join(format!(
        "preflop-blueprint-cli-{}-{}.json",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    let summary_path = output_path.with_extension("summary.json");
    let output = Command::new(env!("CARGO_BIN_EXE_preflop-solver"))
        .args([
            "blueprint",
            "--iterations",
            "2",
            "--averaging-delay",
            "0",
            "--hs-dcfr-30-horizon",
            "2",
            "--held-out-deals",
            "2",
            "--distribution-samples",
            "2",
            "--root-deviation-samples",
            "1",
            "--action-value-deals",
            "2",
            "--opponent-hand-batch-size",
            "2",
            "--max-information-sets",
            "100000",
            "--output",
            output_path.to_str().expect("UTF-8 temp path"),
            "--summary",
            summary_path.to_str().expect("UTF-8 temp path"),
        ])
        .output()
        .expect("run blueprint command");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: Value = serde_json::from_slice(&fs::read(&output_path).expect("artifact file"))
        .expect("valid JSON");
    assert_eq!(payload["schema_version"], 1);
    assert_eq!(payload["approximate"], true);
    assert_eq!(
        payload["model"],
        "hu-abstracted-external-sampling-dcfr-trajectory-v3"
    );
    assert_eq!(payload["metrics"]["training_iterations"], 2);
    assert_eq!(payload["metrics"]["sampled_deals"], 4);
    assert_eq!(payload["config"]["opponent_hand_batch_size"], 2);
    assert_eq!(payload["config"]["dcfr_schedule"], "hs30");
    assert_eq!(payload["config"]["dcfr_schedule_horizon"], 2);
    assert_eq!(payload["validation"]["status"], "advisory_only");
    let summary: Value = serde_json::from_slice(&fs::read(&summary_path).expect("summary file"))
        .expect("valid summary JSON");
    assert_eq!(summary["averagingDelay"], 0);
    assert_eq!(summary["dcfrSchedule"], "hs30");
    assert_eq!(summary["dcfrScheduleHorizon"], 2);
    assert_eq!(summary["dcfr"]["strategy_exponent"], 2.0);
    assert_eq!(summary["opponentHandBatchSize"], 2);
    let root_strategies = summary["rootStrategies"]
        .as_array()
        .expect("compact root strategies");
    assert!(!root_strategies.is_empty());
    assert!(root_strategies.iter().all(|strategy| {
        strategy["hand"].is_string()
            && strategy["averageVisits"].as_u64().is_some()
            && strategy["regretUpdates"].as_u64().is_some()
            && strategy["trainedAverage"] == true
            && strategy["actions"]
                .as_array()
                .is_some_and(|actions| !actions.is_empty())
    }));
    fs::remove_file(output_path).expect("remove test artifact");
    fs::remove_file(summary_path).expect("remove summary");
}
