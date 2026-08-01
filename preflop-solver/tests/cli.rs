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
    let output = Command::new(env!("CARGO_BIN_EXE_preflop-solver"))
        .args([
            "blueprint",
            "--iterations",
            "2",
            "--averaging-delay",
            "0",
            "--held-out-deals",
            "2",
            "--distribution-samples",
            "2",
            "--root-deviation-samples",
            "1",
            "--action-value-deals",
            "2",
            "--max-information-sets",
            "100000",
            "--output",
            output_path.to_str().expect("UTF-8 temp path"),
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
    assert_eq!(payload["validation"]["status"], "advisory_only");
    fs::remove_file(output_path).expect("remove test artifact");
}
