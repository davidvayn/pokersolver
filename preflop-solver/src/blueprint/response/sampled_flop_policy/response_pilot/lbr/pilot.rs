//! Actual full-hand play. LBR's own input excludes the hidden deal; only the
//! dealer/defender action sampling and final payout see the complete deal.
use super::*;
mod delayed;
mod confirmation;
mod terminal_capture;
mod terminal_marginal;

#[derive(Default, Serialize)]
struct HandAttack {
    utility: f64,
    decisions: [u64; 4],
    actions: Vec<serde_json::Value>,
    history: Vec<String>,
}

fn street_index(street: Street) -> usize {
    match street {
        Street::Preflop => 0,
        Street::Flop => 1,
        Street::Turn => 2,
        Street::River => 3,
    }
}

fn payoff(state: &GameState, deal: &Deal) -> f64 {
    terminal::expectation(state, deal).unwrap_or_else(|| realized_utility_p0(state, deal))
}

fn baseline(
    policy: &dyn ResponsePolicy,
    game: &BlueprintConfig,
    deal: &Deal,
    seed: u64,
) -> Result<f64, String> {
    let mut state = GameState::initial(game);
    let mut rng = SplitMix64::new(seed);
    while state.terminal.is_none() {
        let actions = state.legal_actions(game);
        let own = deal.holes[state.actor];
        let mix = query(
            policy,
            game,
            &state,
            &deal.board[..state.street.board_len()],
            &actions,
            Combo::new(own[0], own[1]),
        )?;
        state = state.apply(&actions[sample_index(&mix, &mut rng)], game);
    }
    Ok(payoff(&state, deal))
}

fn play(
    policy: &dyn ResponsePolicy,
    game: &BlueprintConfig,
    deal: &Deal,
    seat: usize,
    seed: u64,
    lbr: &Lbr,
) -> Result<HandAttack, String> {
    play_from(policy, game, deal, seat, seed, lbr, Some(Street::Preflop))
}

fn play_from(
    policy: &dyn ResponsePolicy,
    game: &BlueprintConfig,
    deal: &Deal,
    seat: usize,
    seed: u64,
    lbr: &Lbr,
    first_street: Option<Street>,
) -> Result<HandAttack, String> {
    let mut belief = Belief::new(seat, deal.holes[seat])?;
    let mut state = GameState::initial(game);
    let mut rng = SplitMix64::new(seed);
    let mut record = HandAttack::default();
    while state.terminal.is_none() {
        let board = &deal.board[..state.street.board_len()];
        belief.reveal(board)?;
        let actions = state.legal_actions(game);
        let selected = if state.actor == seat
            && first_street.is_some_and(|first| street_index(state.street) >= street_index(first))
        {
            let values = lbr.values(&belief, policy, game, &state, board, &actions)?;
            let selected = best_index(&values);
            rng.next_f64(); // replaces the baseline's random action draw
            record.decisions[street_index(state.street)] += 1;
            record
                .actions
                .push(serde_json::json!({ "street":state.street, "board":board,
                "history":state.public_history, "action":actions[selected].label,
                "legalActions":actions.iter().map(|a| &a.label).collect::<Vec<_>>(),
                "heuristicCheckdownValuesBb":values }));
            selected
        } else {
            let own = deal.holes[state.actor];
            let mix = query(
                policy,
                game,
                &state,
                board,
                &actions,
                Combo::new(own[0], own[1]),
            )?;
            let selected = sample_index(&mix, &mut rng);
            if state.actor != seat && state.apply(&actions[selected], game).terminal.is_none() {
                belief.observe(policy, game, &state, board, &actions, selected)?;
            }
            selected
        };
        state = state.apply(&actions[selected], game);
    }
    let utility_p0 = payoff(&state, deal);
    record.utility = if seat == 0 { utility_p0 } else { -utility_p0 };
    record.history = state.public_history;
    Ok(record)
}

fn estimate(values: &[f64]) -> serde_json::Value {
    assert!(values.len() >= 2 && values.iter().all(|v| v.is_finite()));
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let se = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n * (n - 1.0))).sqrt();
    serde_json::json!({ "n":values.len(), "mean":mean, "standardError":se,
        "individualNormal99PercentInterval":[mean - 2.575_829_303_548_900_4 * se,
            mean + 2.575_829_303_548_900_4 * se] })
}

fn emit(event: serde_json::Value) {
    println!("{event}");
    std::io::stdout().flush().unwrap();
}

#[test]
#[ignore = "full-size all-street LBR cost/interface pilot; frozen inputs and external resource guard required"]
fn sampled_profile_lbr_cost_probe() {
    run_challenge(8, 24, 90004);
}

#[test]
#[ignore = "paired independent LBR challenge; frozen source and external resource guard required"]
fn sampled_profile_lbr_paired_challenge() {
    run_challenge(64, 128, 91004);
}

fn run_challenge(calibration_hands: u64, holdout_hands: u64, evaluation_seed: u64) {
    run_challenge_from(
        calibration_hands,
        holdout_hands,
        evaluation_seed,
        Street::Preflop,
    );
}

fn run_challenge_from(
    calibration_hands: u64,
    holdout_hands: u64,
    evaluation_seed: u64,
    first_street: Street,
) {
    run_challenge_with_terminal_options(
        calibration_hands,
        holdout_hands,
        evaluation_seed,
        first_street,
        TerminalFlopOptions {
            equity_samples: 2048,
            weight: 0.5,
        },
    );
}

fn run_challenge_with_terminal_options(
    calibration_hands: u64,
    holdout_hands: u64,
    evaluation_seed: u64,
    first_street: Street,
    terminal: TerminalFlopOptions,
) {
    let source = PathBuf::from(std::env::var("POKER_FLOP_PILOT_CHECKPOINT").unwrap());
    let checkpoint_sha = sha256_file(&source).unwrap();
    assert!([
        "8fc95d56696af9fc8a858fceb8efdad4782a049b5af7b0a8f1cdcc5d694f95aa",
        "7a71a1b13975173af32aed3c610ae62a7c0fb25680ddf4397efef0bde220893d",
    ]
    .contains(&checkpoint_sha.as_str()));
    let table = Arc::new(InferenceTable::read(&source).unwrap());
    assert_eq!(table.rounds, 800);
    assert_eq!(table.config.effective_stack_bb, 20.0);
    let game = table.config.clone();
    let (policy, patch) = profile_with_terminal_options(table, 87001, 64, &terminal);
    let lbr = Lbr {
        seed: 90001,
        early_runouts_per_combo: 16,
    };
    let started = Instant::now();
    if first_street != Street::Preflop {
        emit(
            serde_json::json!({ "stage":"lbr_attack_scope", "firstAttackStreet":first_street,
            "earlierActions":"sample the unchanged baseline policy; do not condition opponent belief on hero actions",
            "interpretation":"restricted delayed attack, not a full-game exploitability upper bound" }),
        );
    }
    emit(
        serde_json::json!({ "stage":"lbr_configuration", "checkpointSha256":checkpoint_sha,
        "policySeed":87001, "flopIterations":32, "turnRiverIterations":64,
        "terminalFlopWeight":terminal.weight, "terminalFlopEquitySamples":terminal.equity_samples,
        "lbrSeed":lbr.seed, "earlyRunoutsPerCombo":lbr.early_runouts_per_combo,
        "evaluationSeed":evaluation_seed, "calibrationHands":calibration_hands, "rawHoldoutHands":holdout_hands,
        "interpretation":if first_street == Street::Preflop {
            "restricted legal all-street attack; approximate checkdown action values; not an exploitability upper bound"
        } else {
            "restricted legal delayed attack; approximate checkdown action values; not an exploitability upper bound"
        } }),
    );
    let mut qualified = [false; 2];
    for (phase, count, domain) in [
        ("calibration", calibration_hands, 0),
        ("raw_holdout", holdout_hands, 1),
    ] {
        // No overlap between calibration and untouched holdout chance streams.
        // No policy selection/critic tuning occurs within this cost probe.
        let chance_seed = derived_seed(evaluation_seed, domain, 0);
        let mut chance = SplitMix64::new(chance_seed);
        let mut gains: [Vec<f64>; 2] = Default::default();
        let mut paired_sums = Vec::new();
        let mut counts = [[0_u64; 4]; 2];
        for index in 0..count {
            let hand_started = Instant::now();
            let deal = Deal::sample(&mut chance);
            let action_seed = derived_seed(evaluation_seed, domain, index + 1);
            emit(serde_json::json!({ "stage":"lbr_hand_start", "phase":phase,
                "index":index, "chanceSeed":chance_seed, "actionSeed":action_seed,
                "holes":deal.holes, "board":deal.board }));
            policy.clear_experiment_hand_caches();
            let baseline_p0 = baseline(&policy, &game, &deal, action_seed).unwrap();
            let mut attacks = Vec::new();
            let mut sum = 0.0;
            for seat in 0..2 {
                policy.clear_experiment_hand_caches();
                emit(serde_json::json!({ "stage":"lbr_seat_start", "phase":phase,
                    "index":index, "seat":seat, "baselineP0Bb":baseline_p0 }));
                let attack = play_from(
                    &policy,
                    &game,
                    &deal,
                    seat,
                    action_seed,
                    &lbr,
                    Some(first_street),
                )
                .unwrap();
                let gain = attack.utility - if seat == 0 { baseline_p0 } else { -baseline_p0 };
                gains[seat].push(gain);
                sum += gain;
                for (total, n) in counts[seat].iter_mut().zip(attack.decisions) {
                    *total += n;
                }
                emit(
                    serde_json::json!({ "stage":"lbr_seat", "phase":phase, "index":index,
                    "seat":seat, "gainBb":gain, "attack":attack }),
                );
                attacks.push(attack);
            }
            // Baseline cancels exactly for the two seat-summed unilateral gains.
            assert!((sum - attacks.iter().map(|a| a.utility).sum::<f64>()).abs() < 1e-10);
            paired_sums.push(sum);
            emit(
                serde_json::json!({ "stage":"lbr_hand", "phase":phase, "index":index,
                "seatSumGainBb":sum, "seconds":hand_started.elapsed().as_secs_f64(),
                "elapsedSeconds":started.elapsed().as_secs_f64(),
                "flopDiagnostics":patch.sampled.as_ref().unwrap().diagnostics(),
                "turnDiagnostics":policy.take_resolution_diagnostics() }),
            );
        }
        let estimates = [estimate(&gains[0]), estimate(&gains[1])];
        if phase == "calibration" {
            qualified = std::array::from_fn(|seat| {
                response_lower_bound_passes_calibration(
                    estimates[seat]["individualNormal99PercentInterval"][0]
                        .as_f64()
                        .unwrap(),
                )
            });
        }
        emit(serde_json::json!({ "stage":"lbr_summary", "phase":phase,
            "seatGainsBbPerFullHand":estimates, "seatSumGainBbPerFullHand":estimate(&paired_sums),
            "halfSeatSumGainBbPerFullHand":estimate(&paired_sums.iter().map(|v| v / 2.0).collect::<Vec<_>>()),
            "qualifiedByCalibration":qualified, "decisionsBySeatAndStreet":counts,
            "elapsedSeconds":started.elapsed().as_secs_f64(),
            "confidenceWarning":"small cost/interface pilot; normal intervals are diagnostic, not certification; raw holdout retained even for rejected critics" }));
    }
    policy.clear_experiment_hand_caches();
}

#[test]
fn lbr_paired_gain_estimator_preserves_shared_deal_covariance() {
    let a = [1.0, 3.0, 5.0];
    let b = [5.0, 3.0, 1.0];
    let sums = a.iter().zip(b).map(|(x, y)| x + y).collect::<Vec<_>>();
    assert_eq!(estimate(&sums)["mean"], 6.0);
    assert_eq!(estimate(&sums)["standardError"], 0.0);
    assert!(estimate(&a)["standardError"].as_f64().unwrap() > 0.0);
}

#[test]
fn lbr_full_hand_play_is_deterministic_bounded_and_covers_all_streets() {
    let (game, _) = super::tests::fixture(Street::Preflop);
    let lbr = Lbr {
        seed: 90001,
        early_runouts_per_combo: 16,
    };
    let mut chance = SplitMix64::new(90004);
    let mut totals = [0; 4];
    for _ in 0..16 {
        let deal = Deal::sample(&mut chance);
        for seat in 0..2 {
            let record = play(&super::tests::Pattern::Call, &game, &deal, seat, 12, &lbr).unwrap();
            assert!(record.utility.abs() <= game.effective_stack_bb);
            for (count, value) in totals.iter_mut().zip(record.decisions) {
                *count += value;
            }
            let repeat = play(&super::tests::Pattern::Call, &game, &deal, seat, 12, &lbr).unwrap();
            assert_eq!(
                serde_json::to_vec(&record).unwrap(),
                serde_json::to_vec(&repeat).unwrap()
            );
        }
    }
    assert!(totals.iter().all(|n| *n > 0), "visited {totals:?}");
}
