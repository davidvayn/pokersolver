//! Fixed short full-profile screen after the exact-terminal training pilot.
//! Only flop training chance changes; both serving terminal weights are 0.5.
use super::*;

fn paired_training(
    table: Arc<InferenceTable>,
    budget: Budget,
    parallel: bool,
    output: bool,
) -> Result<Vec<Vec<serde_json::Value>>, String> {
    let stopped = AtomicBool::new(false);
    if !parallel {
        return [false, true]
            .into_iter()
            .map(|exact| variant_with_training(table.clone(), 0.5, budget, &stopped, output, exact))
            .collect();
    }
    std::thread::scope(|scope| {
        let mut workers = Vec::new();
        for exact in [false, true] {
            let table = table.clone();
            let stopped = &stopped;
            workers.push(scope.spawn(move || {
                let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    variant_with_training(table, 0.5, budget, stopped, output, exact)
                }));
                let result = attempt.unwrap_or_else(|_| {
                    Err("exact-terminal training pair worker panicked; no replacement".into())
                });
                if result.is_err() {
                    stopped.store(true, Ordering::Relaxed);
                }
                result
            }));
        }
        workers
            .into_iter()
            .map(|worker| {
                worker
                    .join()
                    .map_err(|_| "training worker join failed".to_owned())?
            })
            .collect()
    })
}

#[test]
fn exact_training_pair_matches_serial_and_keeps_control_and_terminal_weight() {
    let (mut base, _) = crate::blueprint::response::tests::tabular_fixture();
    let table = Arc::get_mut(&mut base.table).unwrap();
    table.config.effective_stack_bb = 2.0;
    table.nodes.clear();
    let budget = Budget {
        calibration: 2,
        holdout: 4,
        seed: 98005,
    };
    let serial = paired_training(base.table.clone(), budget, false, false).unwrap();
    assert_eq!(
        serial,
        paired_training(base.table.clone(), budget, true, false).unwrap()
    );
    assert_eq!(
        serial[0],
        variant(
            base.table.clone(),
            0.5,
            budget,
            &AtomicBool::new(false),
            false
        )
        .unwrap()
    );
    for (index, records) in serial.iter().enumerate() {
        for record in records {
            assert_eq!(record["weight"], 0.5);
            assert_eq!(
                record["exactTerminalTraining"].as_bool().unwrap_or(false),
                index == 1
            );
        }
    }
    assert!(
        variant_with_training(base.table, 0.5, budget, &AtomicBool::new(true), false, true)
            .is_err()
    );
}

#[test]
#[ignore = "explicit source and external 7.5GiB/time/disk guard; short paired full-profile screen, not certification"]
fn fixed_full_profile_exact_training_pair() {
    let source = PathBuf::from(std::env::var("POKER_FLOP_PILOT_CHECKPOINT").unwrap());
    let source_sha = sha256_file(&source).unwrap();
    assert!([
        "8fc95d56696af9fc8a858fceb8efdad4782a049b5af7b0a8f1cdcc5d694f95aa",
        "7a71a1b13975173af32aed3c610ae62a7c0fb25680ddf4397efef0bde220893d",
    ]
    .contains(&source_sha.as_str()));
    let table = Arc::new(InferenceTable::read(&source).unwrap());
    assert_eq!(table.rounds, 800);
    assert_eq!(table.config.effective_stack_bb, 20.0);
    let budget = Budget {
        calibration: 16,
        holdout: 64,
        seed: 98004,
    };
    emit(serde_json::json!({"stage":"exact_training_configuration",
        "sourceSha256":source_sha,"sharedSourceTables":1,"workers":2,
        "policySeed":87001,"flopIterations":32,"turnRiverIterations":64,
        "terminalWeight":0.5,"terminalEquitySamples":2048,
        "lbrSeed":90001,"earlyRunoutsPerCombo":16,"firstAttackStreet":"flop",
        "evaluationSeed":budget.seed,"calibrationHands":budget.calibration,"holdoutHands":budget.holdout,
        "interpretation":"absent exactTerminalTraining tag means control, true means candidate; only flop training chance changes, not terminal serving mix; shared deals across sources; no exploitability upper bound or automatic activation"}));
    let results = paired_training(table, budget, true, true).unwrap();
    for phase in ["calibration", "raw_holdout"] {
        let mut values = Vec::new();
        for records in &results {
            values.push(
                records
                    .iter()
                    .filter(|e| e["stage"] == "terminal_confirm_hand" && e["phase"] == phase)
                    .map(|e| e["marginalSeatSumGainBb"].as_f64().unwrap())
                    .collect::<Vec<_>>(),
            );
        }
        assert_eq!(values[0].len(), values[1].len());
        let deltas = values[0]
            .iter()
            .zip(&values[1])
            .map(|(a, b)| b - a)
            .collect::<Vec<_>>();
        emit(
            serde_json::json!({"stage":"exact_training_paired","phase":phase,
            "candidateMinusControlSeatSum":estimate(&deltas),
            "candidateMinusControlHalfSeatSum":estimate(&deltas.iter().map(|v| v/2.0).collect::<Vec<_>>())}),
        );
    }
}
