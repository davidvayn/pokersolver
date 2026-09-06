//! Exact final-action marginalization of archived LBR trajectories. This only
//! changes a payoff estimator, never a policy, response or admission decision.
//! It is terminal Rao-Blackwellization, not full AIVAT or an exploitability bound.
use super::*;

#[derive(Debug, PartialEq, Serialize)]
pub(super) struct TerminalAssessment {
    integrated: bool,
    observed_utility: f64,
    pub(super) conditional_utility: f64,
    actor: usize,
    selected: usize,
    probabilities: Vec<f64>,
    action_values_p0_bb: Vec<f64>,
}

fn final_decision(
    game: &BlueprintConfig,
    history: &[String],
) -> Result<(GameState, usize), String> {
    let mut state = GameState::initial(game);
    if !history.starts_with(&state.public_history) {
        return Err("terminal replay has different initial blinds".into());
    }
    loop {
        let successors = state
            .legal_actions(game)
            .iter()
            .enumerate()
            .map(|(i, action)| (i, state.apply(action, game)))
            .filter(|(_, child)| history.starts_with(&child.public_history))
            .collect::<Vec<_>>();
        if successors.len() != 1 {
            return Err("terminal replay history is illegal, incomplete or ambiguous".into());
        }
        let (selected, child) = successors.into_iter().next().unwrap();
        if child.terminal.is_some() {
            if child.public_history != history {
                return Err("terminal replay has actions after settlement".into());
            }
            return Ok((state, selected));
        }
        state = child;
    }
}

pub(super) fn assess(
    policy: &dyn ResponsePolicy,
    game: &BlueprintConfig,
    deal: &Deal,
    responder: usize,
    history: &[String],
    observed_utility: f64,
) -> Result<TerminalAssessment, String> {
    if responder > 1 || !observed_utility.is_finite() {
        return Err("terminal replay has an invalid seat or payoff".into());
    }
    let (state, selected) = final_decision(game, history)?;
    let actions = state.legal_actions(game);
    let sign = if responder == 0 { 1.0 } else { -1.0 };
    let expected_observed = sign * payoff(&state.apply(&actions[selected], game), deal);
    if (expected_observed - observed_utility).abs() > 1e-10 {
        return Err("terminal replay does not reproduce the archived payout".into());
    }
    let mut result = TerminalAssessment {
        integrated: false,
        observed_utility,
        conditional_utility: observed_utility,
        actor: state.actor,
        selected,
        probabilities: Vec::new(),
        action_values_p0_bb: Vec::new(),
    };
    // Eligibility depends on the state BEFORE the action, not whether the
    // sampled action happened to fold. No terminal/nonterminal reweighting.
    if state.actor == responder
        || state.street != Street::Flop
        || actions.len() != 2
        || state.remaining(responder, game) > EPSILON
        || !actions
            .iter()
            .all(|a| state.apply(a, game).terminal.is_some())
    {
        return Ok(result);
    }
    let own = deal.holes[state.actor];
    // The strategy seam receives own cards and the visible board only. Both
    // private hands and exact runouts are permitted solely in payout scoring.
    result.probabilities = query(
        policy,
        game,
        &state,
        &deal.board[..3],
        &actions,
        Combo::new(own[0], own[1]),
    )?;
    if result.probabilities[selected] <= 0.0 {
        return Err("archived action has zero probability under its claimed policy".into());
    }
    result.action_values_p0_bb = actions
        .iter()
        .map(|action| terminal::expectation(&state.apply(action, game), deal).unwrap())
        .collect();
    result.conditional_utility = sign
        * result
            .probabilities
            .iter()
            .zip(&result.action_values_p0_bb)
            .map(|(p, value)| p * value)
            .sum::<f64>();
    result.integrated = true;
    Ok(result)
}

#[test]
fn terminal_marginal_replays_both_actions_and_preserves_their_exact_mean() {
    let (game, flop) = super::super::tests::fixture(Street::Flop);
    let facing = flop.apply(flop.legal_actions(&game).last().unwrap(), &game);
    let deal = Deal::from_sampled_cards([[48, 49], [36, 37]], [0, 5, 10, 15, 20]);
    let changed_future = Deal::from_sampled_cards(deal.holes, [0, 5, 10, 19, 24]);
    let policy = super::super::tests::Pattern::RankLimp;
    let responder = 1 - facing.actor;
    let sign = if responder == 0 { 1.0 } else { -1.0 };
    let actions = facing.legal_actions(&game);
    let mix = policy.strategy(&facing, &deal, &actions, &game);
    let raw = actions
        .iter()
        .map(|a| sign * payoff(&facing.apply(a, &game), &deal))
        .collect::<Vec<_>>();
    let expected: f64 = mix.iter().zip(&raw).map(|(p, v)| p * v).sum();
    assert!(
        (raw[0] - raw[1]).abs() > 0.1,
        "fixture must have action-sampling noise"
    );
    let mut residual_mean = 0.0;
    for (i, action) in actions.iter().enumerate() {
        let final_state = facing.apply(action, &game);
        let value = assess(
            &policy,
            &game,
            &deal,
            responder,
            &final_state.public_history,
            raw[i],
        )
        .unwrap();
        assert!(value.integrated);
        assert_eq!(value.probabilities, mix);
        assert!((value.conditional_utility - expected).abs() < 1e-12);
        assert_eq!(
            value,
            assess(
                &policy,
                &game,
                &changed_future,
                responder,
                &final_state.public_history,
                raw[i]
            )
            .unwrap()
        );
        residual_mean += mix[i] * (value.conditional_utility - raw[i]);
        let attacker = assess(
            &policy,
            &game,
            &deal,
            facing.actor,
            &final_state.public_history,
            -raw[i],
        )
        .unwrap();
        assert!(
            !attacker.integrated,
            "do not replace the attacker's actual strategy"
        );
        assert_eq!(attacker.observed_utility, attacker.conditional_utility);
    }
    assert!(
        residual_mean.abs() < 1e-12,
        "correction has zero exact expectation"
    );
    let folded = facing.apply(&actions[0], &game);
    for invalid in [
        super::super::tests::Pattern::Invalid,
        super::super::tests::Pattern::Call,
    ] {
        assert!(
            assess(
                &invalid,
                &game,
                &deal,
                responder,
                &folded.public_history,
                raw[0]
            )
            .is_err(),
            "reject invalid mixes and an observed action with zero policy probability"
        );
    }
    // Symmetric accounting when the other seat is the final defender.
    let checked = flop.apply(&flop.legal_actions(&game)[0], &game);
    let facing = checked.apply(checked.legal_actions(&game).last().unwrap(), &game);
    let state = facing.apply(&facing.legal_actions(&game)[0], &game);
    let responder = 1 - facing.actor;
    let sign = if responder == 0 { 1.0 } else { -1.0 };
    assert!(
        assess(
            &policy,
            &game,
            &deal,
            responder,
            &state.public_history,
            sign * payoff(&state, &deal)
        )
        .unwrap()
        .integrated
    );
}

#[test]
fn terminal_marginal_rejects_bad_records_and_leaves_noneligible_folds_alone() {
    let (game, flop) = super::super::tests::fixture(Street::Flop);
    let bet = flop
        .legal_actions(&game)
        .into_iter()
        .find(|a| {
            let child = flop.apply(a, &game);
            child.to_call() > 0.0 && child.remaining(flop.actor, &game) > EPSILON
        })
        .unwrap();
    let facing = flop.apply(&bet, &game);
    let folded = facing.apply(&facing.legal_actions(&game)[0], &game);
    let deal = Deal::from_sampled_cards([[48, 49], [36, 37]], [0, 5, 10, 15, 20]);
    let policy = super::super::tests::Pattern::Invalid;
    let responder = 1 - facing.actor;
    let value = payoff(&folded, &deal) * if responder == 0 { 1.0 } else { -1.0 };
    let record = assess(
        &policy,
        &game,
        &deal,
        responder,
        &folded.public_history,
        value,
    )
    .unwrap();
    assert!(!record.integrated && record.conditional_utility == value);
    assert!(assess(
        &policy,
        &game,
        &deal,
        responder,
        &folded.public_history,
        value + 1.0
    )
    .is_err());
    assert!(final_decision(&game, &flop.public_history).is_err());
    let mut invalid = folded.public_history.clone();
    invalid.push("Flop:p0:check".into());
    assert!(final_decision(&game, &invalid).is_err());
    invalid[0] = "blinds:1.000/2.000".into();
    assert!(final_decision(&game, &invalid).is_err());
}

#[test]
#[ignore = "read-only marginal replay of a complete hash-pinned paired LBR run; external memory guard required"]
fn archived_terminal_marginal_replay() {
    let path = PathBuf::from(std::env::var("POKER_TERMINAL_PAIR_MANIFEST").unwrap());
    let manifest_sha = sha256_file(&path).unwrap();
    assert_eq!(
        manifest_sha,
        std::env::var("POKER_TERMINAL_PAIR_SHA256").unwrap()
    );
    let manifest: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    assert_eq!(manifest["status"], "complete");
    assert_eq!(
        manifest["schema"],
        "terminal-flop-weight-fresh-paired-lbr-v1"
    );
    let source = PathBuf::from(std::env::var("POKER_FLOP_PILOT_CHECKPOINT").unwrap());
    let source_sha = sha256_file(&source).unwrap();
    let jobs = manifest["jobs"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|job| job["sourceCheckpointSha256"] == source_sha && job["weight"].is_number())
        .collect::<Vec<_>>();
    assert_eq!(jobs.len(), 2);
    let table = Arc::new(InferenceTable::read(&source).unwrap());
    assert_eq!(table.rounds, 800);
    let game = table.config.clone();
    emit(
        serde_json::json!({"stage":"terminal_marginal_configuration", "manifestSha256":manifest_sha,
        "sourceSha256":source_sha, "policySeed":87001, "flopIterations":32, "turnRiverIterations":64,
        "interpretation":"exact final-defender-action integration on existing deals; no new independent samples, fitted values, response qualification or exploitability upper bound"}),
    );
    for job in jobs {
        let name = job["name"].as_str().unwrap();
        assert!([
            "26001-control",
            "26001-candidate",
            "26002-control",
            "26002-candidate"
        ]
        .contains(&name));
        assert_eq!(job["status"], "complete");
        let log = path.parent().unwrap().join(name).join("worker.log");
        assert_eq!(
            sha256_file(&log).unwrap(),
            job["logSha256"].as_str().unwrap()
        );
        let weight = job["weight"].as_f64().unwrap();
        assert!([0.5, 1.0].contains(&weight));
        let events = job["events"].as_array().unwrap();
        let config = events
            .iter()
            .find(|e| e["stage"] == "lbr_configuration")
            .unwrap();
        assert_eq!(config["checkpointSha256"], source_sha);
        assert_eq!(config["terminalFlopWeight"], weight);
        assert_eq!(config["terminalFlopEquitySamples"], 2048);
        assert_eq!(config["policySeed"], 87001);
        assert_eq!(config["flopIterations"], 32);
        assert_eq!(config["turnRiverIterations"], 64);
        let (policy, _) = profile_with_terminal_options(
            table.clone(),
            87001,
            64,
            &TerminalFlopOptions {
                equity_samples: 2048,
                weight,
            },
        );
        for (phase, count) in [("calibration", 32), ("raw_holdout", 96)] {
            let mut raw_sums = Vec::new();
            let mut marginal_sums = Vec::new();
            let mut integrated = 0;
            for index in 0..count {
                let start = events
                    .iter()
                    .find(|e| {
                        e["stage"] == "lbr_hand_start" && e["phase"] == phase && e["index"] == index
                    })
                    .unwrap();
                let holes = serde_json::from_value(start["holes"].clone()).unwrap();
                let board = serde_json::from_value(start["board"].clone()).unwrap();
                let deal = Deal::from_sampled_cards(holes, board);
                let mut raw_sum = 0.0;
                let mut marginal_sum = 0.0;
                for seat in 0..2 {
                    let row = events
                        .iter()
                        .find(|e| {
                            e["stage"] == "lbr_seat"
                                && e["phase"] == phase
                                && e["index"] == index
                                && e["seat"] == seat
                        })
                        .unwrap();
                    let history: Vec<String> =
                        serde_json::from_value(row["attack"]["history"].clone()).unwrap();
                    let observed = row["attack"]["utility"].as_f64().unwrap();
                    policy.clear_experiment_hand_caches();
                    let result = assess(&policy, &game, &deal, seat, &history, observed).unwrap();
                    raw_sum += observed;
                    marginal_sum += result.conditional_utility;
                    integrated += usize::from(result.integrated);
                    emit(
                        serde_json::json!({"stage":"terminal_marginal_seat", "name":name,
                        "phase":phase, "index":index, "seat":seat, "assessment":result}),
                    );
                }
                raw_sums.push(raw_sum);
                marginal_sums.push(marginal_sum);
                emit(
                    serde_json::json!({"stage":"terminal_marginal_hand", "name":name,
                    "phase":phase, "index":index, "rawSeatSum":raw_sum, "marginalSeatSum":marginal_sum}),
                );
            }
            emit(
                serde_json::json!({"stage":"terminal_marginal_summary", "name":name, "phase":phase,
                "integratedDecisions":integrated, "rawSeatSum":estimate(&raw_sums),
                "marginalSeatSum":estimate(&marginal_sums)}),
            );
        }
        policy.clear_experiment_hand_caches();
    }
}
