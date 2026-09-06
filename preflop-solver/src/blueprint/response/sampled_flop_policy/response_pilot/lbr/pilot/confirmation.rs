//! Fixed-budget paired confirmation. Two isolated policy/cache instances share
//! one immutable source table. Exact final-action scoring is fixed in advance;
//! original sampled payouts and raw response-calibration flags remain visible.
use super::*;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy)]
struct Budget {
    calibration: u64,
    holdout: u64,
    seed: u64,
}

mod training_pair;

fn variant(
    table: Arc<InferenceTable>,
    weight: f64,
    budget: Budget,
    stopped: &AtomicBool,
    output: bool,
) -> Result<Vec<serde_json::Value>, String> {
    variant_with_training(table, weight, budget, stopped, output, false)
}

fn variant_with_training(
    table: Arc<InferenceTable>,
    weight: f64,
    budget: Budget,
    stopped: &AtomicBool,
    output: bool,
    exact_terminal_chance: bool,
) -> Result<Vec<serde_json::Value>, String> {
    let game = table.config.clone();
    let (policy, _) = profile_with_training_options(
        table,
        87001,
        64,
        &TerminalFlopOptions {
            equity_samples: 2048,
            weight,
        },
        exact_terminal_chance,
    );
    // Keep archived default records byte-compatible. The new pilot explicitly
    // defines an absent tag as sampled training and a true tag as exact chance.
    let tag = |mut record: serde_json::Value| {
        if exact_terminal_chance {
            record["exactTerminalTraining"] = true.into();
        }
        record
    };
    let lbr = Lbr {
        seed: 90001,
        early_runouts_per_combo: 16,
    };
    let mut records = Vec::new();
    let mut qualified = [false; 2];
    for (phase, count, domain) in [
        ("calibration", budget.calibration, 0),
        ("raw_holdout", budget.holdout, 1),
    ] {
        let chance_seed = derived_seed(budget.seed, domain, 0);
        let mut chance = SplitMix64::new(chance_seed);
        let mut raw_gains: [Vec<f64>; 2] = Default::default();
        let mut raw_sums = Vec::new();
        let mut marginal_sums = Vec::new();
        for index in 0..count {
            if stopped.load(Ordering::Relaxed) {
                return Err("paired confirmation stopped after its other worker failed".into());
            }
            let deal = Deal::sample(&mut chance);
            let action_seed = derived_seed(budget.seed, domain, index + 1);
            if output {
                emit(
                    tag(serde_json::json!({"stage":"terminal_confirm_start", "weight":weight,
                    "phase":phase, "index":index, "chanceSeed":chance_seed, "actionSeed":action_seed,
                    "holes":deal.holes, "board":deal.board})),
                );
            }
            policy.clear_experiment_hand_caches();
            let baseline_p0 = baseline(&policy, &game, &deal, action_seed)?;
            let mut attacks = Vec::new();
            let mut assessments = Vec::new();
            for seat in 0..2 {
                policy.clear_experiment_hand_caches();
                let attack = play_from(
                    &policy,
                    &game,
                    &deal,
                    seat,
                    action_seed,
                    &lbr,
                    Some(Street::Flop),
                )?;
                let assessment = terminal_marginal::assess(
                    &policy,
                    &game,
                    &deal,
                    seat,
                    &attack.history,
                    attack.utility,
                )?;
                raw_gains[seat]
                    .push(attack.utility - baseline_p0 * if seat == 0 { 1.0 } else { -1.0 });
                attacks.push(attack);
                assessments.push(assessment);
            }
            let raw_sum = attacks.iter().map(|a| a.utility).sum::<f64>();
            let marginal_sum = assessments
                .iter()
                .map(|a| a.conditional_utility)
                .sum::<f64>();
            assert!(
                (raw_sum - raw_gains.iter().map(|g| g.last().unwrap()).sum::<f64>()).abs() < 1e-10
            );
            raw_sums.push(raw_sum);
            marginal_sums.push(marginal_sum);
            let record = tag(serde_json::json!({"stage":"terminal_confirm_hand", "weight":weight,
                "phase":phase, "index":index, "chanceSeed":chance_seed, "actionSeed":action_seed,
                "holes":deal.holes, "board":deal.board, "baselineP0Bb":baseline_p0,
                "attacks":attacks, "assessments":assessments,
                "rawSeatSumGainBb":raw_sum, "marginalSeatSumGainBb":marginal_sum}));
            if output {
                emit(record.clone());
            }
            records.push(record);
        }
        let per_seat = raw_gains
            .iter()
            .map(|values| estimate(values))
            .collect::<Vec<_>>();
        if phase == "calibration" {
            qualified = std::array::from_fn(|seat| {
                response_lower_bound_passes_calibration(
                    per_seat[seat]["individualNormal99PercentInterval"][0]
                        .as_f64()
                        .unwrap(),
                )
            });
        }
        let summary = tag(serde_json::json!({"stage":"terminal_confirm_summary", "weight":weight,
            "phase":phase, "rawSeatGains":per_seat, "rawQualifiedByCalibration":qualified,
            "rawSeatSumGainBb":estimate(&raw_sums), "marginalSeatSumGainBb":estimate(&marginal_sums),
            "interpretation":"fixed delayed legal attack; raw calibration unchanged; marginalized sums are not an exploitability upper bound"}));
        if output {
            emit(summary.clone());
        }
        records.push(summary);
    }
    policy.clear_experiment_hand_caches();
    Ok(records)
}

fn panel(
    table: Arc<InferenceTable>,
    budget: Budget,
    parallel: bool,
    output: bool,
) -> Result<Vec<Vec<serde_json::Value>>, String> {
    let stopped = AtomicBool::new(false);
    if !parallel {
        return [0.5, 1.0]
            .into_iter()
            .map(|weight| variant(table.clone(), weight, budget, &stopped, output))
            .collect();
    }
    std::thread::scope(|scope| {
        let mut workers = Vec::new();
        for weight in [0.5, 1.0] {
            let table = table.clone();
            let stopped = &stopped;
            workers.push(scope.spawn(move || {
                let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    variant(table, weight, budget, stopped, output)
                }));
                let result = attempt.unwrap_or_else(|_| {
                    Err("paired policy worker panicked; no replacement result".into())
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
                    .map_err(|_| "paired worker join failed".to_owned())?
            })
            .collect()
    })
}

#[test]
fn shared_confirmation_matches_serial_play_and_exact_terminal_scoring() {
    let (mut base, _) = crate::blueprint::response::tests::tabular_fixture();
    let table = Arc::get_mut(&mut base.table).unwrap();
    table.config.effective_stack_bb = 2.0;
    table.nodes.clear();
    let budget = Budget {
        calibration: 4,
        holdout: 8,
        seed: 95004,
    };
    let serial = panel(base.table.clone(), budget, false, false).unwrap();
    assert_eq!(
        serial,
        panel(base.table.clone(), budget, true, false).unwrap()
    );
    assert!(variant(base.table, 0.5, budget, &AtomicBool::new(true), false).is_err());
    let hands = serial
        .iter()
        .flatten()
        .filter(|e| e["stage"] == "terminal_confirm_hand")
        .collect::<Vec<_>>();
    assert_eq!(hands.len(), 24);
    assert!(hands.iter().any(|e| e["assessments"]
        .as_array()
        .unwrap()
        .iter()
        .any(|a| a["integrated"] == true)));
    for hand in hands {
        for assessment in hand["assessments"].as_array().unwrap() {
            assert!(assessment["conditional_utility"].as_f64().unwrap().abs() <= 2.0);
        }
    }
}

#[test]
#[ignore = "guarded two-worker serial-parity pilot or preregistered independent confirmation; one full checkpoint only"]
fn shared_terminal_weight_confirmation() {
    let mode = std::env::var("POKER_TERMINAL_CONFIRM_MODE").unwrap();
    let budget = match mode.as_str() {
        "parity" => Budget {
            calibration: 8,
            holdout: 24,
            seed: 93004,
        },
        "fresh" => Budget {
            calibration: 64,
            holdout: 256,
            // Separate from both archived deals and the small-game unit test.
            seed: 96004,
        },
        _ => panic!("explicit parity/fresh confirmation mode required"),
    };
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
    emit(
        serde_json::json!({"stage":"terminal_confirm_configuration", "mode":mode,
        "sourceSha256":source_sha, "sharedSourceTables":1, "workers":2,
        "policySeed":87001, "flopIterations":32, "turnRiverIterations":64,
        "terminalWeights":[0.5,1.0], "terminalEquitySamples":2048,
        "lbrSeed":90001, "earlyRunoutsPerCombo":16, "firstAttackStreet":"flop",
        "evaluationSeed":budget.seed, "calibrationHands":budget.calibration, "holdoutHands":budget.holdout,
        "interpretation":"same deals across variants and sources; exact terminal estimator frozen before new data; no exploitability certificate or automatic activation"}),
    );
    let results = panel(table, budget, true, true).unwrap();
    for phase in ["calibration", "raw_holdout"] {
        let values = results
            .iter()
            .map(|records| {
                records
                    .iter()
                    .filter(|e| e["stage"] == "terminal_confirm_hand" && e["phase"] == phase)
                    .map(|e| e["marginalSeatSumGainBb"].as_f64().unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(values[0].len(), values[1].len());
        let deltas = values[0]
            .iter()
            .zip(&values[1])
            .map(|(a, b)| b - a)
            .collect::<Vec<_>>();
        emit(
            serde_json::json!({"stage":"terminal_confirm_paired", "phase":phase,
            "candidateMinusControlSeatSum":estimate(&deltas),
            "candidateMinusControlHalfSeatSum":estimate(&deltas.iter().map(|x| x/2.0).collect::<Vec<_>>())}),
        );
    }
}
