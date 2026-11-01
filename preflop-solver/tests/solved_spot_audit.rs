use std::{
    fs,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn footprint_cli_writes_only_immutable_research_reports_and_keeps_flop_serving_closed() {
    let input = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../data/solved-spots/pilots/flop-50bb-footprint.json");
    let directory = std::env::temp_dir().join(format!(
        "poker-footprint-cli-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&directory).unwrap();
    let report = directory.join("report.json");
    let run = || {
        Command::new(env!("CARGO_BIN_EXE_preflop-solver"))
            .arg("solved-spot-audit")
            .arg("--input")
            .arg(&input)
            .arg("--output")
            .arg(&report)
            .output()
            .unwrap()
    };
    let result = run();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let bytes = fs::read(&report).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["schema"], "solved-spot-footprint-v1");
    assert_eq!(value["strategyStrengthEvaluated"], false);
    assert_eq!(value["publicDecisionNodes"], 412300);
    assert!(value.get("accepted").is_none());
    assert!(!run().status.success());
    assert_eq!(fs::read(&report).unwrap(), bytes);
    let policy = directory.join("policy.json");
    let solve = Command::new(env!("CARGO_BIN_EXE_preflop-solver"))
        .arg("solved-spot")
        .arg("--input")
        .arg(&input)
        .arg("--output")
        .arg(&policy)
        .output()
        .unwrap();
    assert!(!solve.status.success());
    assert!(!policy.exists());
    let invalid = Command::new(env!("CARGO_BIN_EXE_preflop-solver"))
        .args([
            "solved-spot",
            "--verify",
            "--dense-reference",
            "--input",
            "must-not-read",
        ])
        .output()
        .unwrap();
    assert!(!invalid.status.success());
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("solve-only"));
    fs::remove_file(report).unwrap();
    fs::remove_dir(directory).unwrap();
}
