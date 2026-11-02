use std::{
    fs,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn exact_flop_cli_requires_explicit_valid_budgets_before_input_io() {
    for options in [
        vec![],
        vec!["--max-decision-nodes", "10000"],
        vec!["--max-decision-nodes", "3", "--max-seconds", "60"],
        vec!["--max-decision-nodes", "10000", "--max-seconds", "601"],
    ] {
        let result = Command::new(env!("CARGO_BIN_EXE_preflop-solver"))
            .args([
                "solved-flop-pilot",
                "--input",
                "must-not-read",
                "--output",
                "must-not-write",
            ])
            .args(options)
            .output()
            .unwrap();
        assert!(!result.status.success());
        let message = String::from_utf8_lossy(&result.stderr);
        assert!(
            !message.contains("No such file"),
            "budget must be checked before input: {message}"
        );
    }
}

#[test]
fn exact_flop_cli_writes_immutable_research_not_serving_artifacts() {
    let input = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../data/solved-spots/pilots/flop-50bb-exact-small.json");
    let directory = std::env::temp_dir().join(format!(
        "poker-exact-flop-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&directory).unwrap();
    let output = directory.join("report.json");
    let run = || {
        Command::new(env!("CARGO_BIN_EXE_preflop-solver"))
            .args(["solved-flop-pilot", "--input"])
            .arg(&input)
            .arg("--output")
            .arg(&output)
            .args(["--max-decision-nodes", "10000", "--max-seconds", "60"])
            .output()
            .unwrap()
    };
    let result = run();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let bytes = fs::read(&output).unwrap();
    let report: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(report["schema"], "exact-flop-pilot-v2");
    assert_eq!(report["publicDecisionNodes"], 9608);
    assert_eq!(report["servingArtifactProduced"], false);
    assert_eq!(report["actionEvValidationCompleted"], true);
    assert_eq!(report["fullHandGtoClaim"], false);
    assert!(report.get("accepted").is_none());
    assert!(!run().status.success());
    assert_eq!(fs::read(&output).unwrap(), bytes);
    let policy = directory.join("policy.json");
    let result = Command::new(env!("CARGO_BIN_EXE_preflop-solver"))
        .args(["solved-spot", "--input"])
        .arg(&input)
        .arg("--output")
        .arg(&policy)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(!policy.exists());
    let rejected = directory.join("rejected.json");
    let result = Command::new(env!("CARGO_BIN_EXE_preflop-solver"))
        .args(["solved-flop-pilot", "--input"])
        .arg(&input)
        .arg("--output")
        .arg(&rejected)
        .args(["--max-decision-nodes", "4", "--max-seconds", "60"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(!rejected.exists());
    fs::remove_file(output).unwrap();
    fs::remove_dir(directory).unwrap();
}
