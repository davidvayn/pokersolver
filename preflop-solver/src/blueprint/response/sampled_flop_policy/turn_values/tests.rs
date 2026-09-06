use super::*;

fn pass(state: &GameState, game: &BlueprintConfig) -> GameState {
    let action = state
        .legal_actions(game)
        .into_iter()
        .find(|a| matches!(a.kind, ActionKind::Call | ActionKind::Check))
        .unwrap();
    state.apply(&action, game)
}

fn fixture() -> (Request, TabularResponsePolicy) {
    let (mut base, _) = crate::blueprint::response::tests::tabular_fixture();
    let table = Arc::get_mut(&mut base.table).unwrap();
    table.config.effective_stack_bb = 2.0;
    table.nodes.clear(); // Explicit synthetic fixture, never a deployed fallback.
    let mut patch = flop::FlopPatch::terminal(&TerminalFlopOptions {
        equity_samples: 128,
        weight: 0.5,
    });
    patch.sampled = Some(FlopResolve::new(32, 87001, 2_000_000));
    base.flop_patch = Some(Arc::new(patch));
    let game = base.table.config.clone();
    let mut root = GameState::initial(&game);
    while root.street != Street::Flop {
        root = pass(&root, &game);
    }
    let board = vec![0, 5, 10];
    let ranges = public_ranges(&base, &root, &board).unwrap();
    (
        Request {
            source_checkpoint_sha256: "0".repeat(64),
            game,
            flop: PublicBeliefState::from_game_state(board, &root, ranges),
            policy_seed: 87001,
            flop_iterations: 32,
            maximum_flop_information_sets: 2_000_000,
            flop_line: vec!["check".into(), "check".into()],
            turn_card: 15,
            turn_river_iterations: 4,
        },
        base,
    )
}

fn full_replay(request: &Request, base: TabularResponsePolicy) -> TurnRiverSolveConfig {
    let mut state = request.root().unwrap();
    for label in &request.flop_line {
        let action = state
            .legal_actions(&request.game)
            .into_iter()
            .find(|a| a.label == *label)
            .unwrap();
        state = state.apply(&action, &request.game);
    }
    let mut board = request.flop.board.clone();
    board.push(request.turn_card);
    turn::TabularTurnPolicy::new(
        base,
        TurnResolveOptions {
            iterations: request.turn_river_iterations,
            safe_bilateral: false,
            maximum_policy_rows: 20000,
        },
    )
    .turn_root_input(&state, &board)
    .unwrap()
}

#[test]
fn rooted_replay_matches_full_history_and_does_not_observe_turn_early() {
    let (request, base) = fixture();
    let compact = request.turn_input().unwrap();
    let full = full_replay(&request, base);
    assert_eq!(compact.state, full.state);
    assert_eq!(compact.game, full.game);
    assert_eq!(compact.iterations, full.iterations);
    assert_eq!(compact.averaging_delay, full.averaging_delay);
    assert_eq!(
        compact.river_refinement_iterations,
        full.river_refinement_iterations
    );
    assert_eq!(compact.regret_matching_plus, full.regret_matching_plus);
    let mut other = request.clone();
    other.turn_card = 19;
    let changed = other.turn_input().unwrap();
    // A different revealed turn only blocks/renormalizes the common flop
    // posterior. Relative odds of unaffected combos cannot change.
    for seat in 0..2 {
        let pairs = all_combos()
            .into_iter()
            .filter_map(|c| {
                let a = compact.state.ranges[seat][c.key()];
                let b = changed.state.ranges[seat][c.key()];
                (a > 0.0 && b > 0.0).then_some(a / b)
            })
            .collect::<Vec<_>>();
        assert!(!pairs.is_empty());
        assert!(pairs.iter().all(|r| (r - pairs[0]).abs() < 1e-12));
    }
}

#[test]
fn rooted_requests_reject_illegal_lines_accounting_ranges_and_board() {
    let (request, _) = fixture();
    for case in 0..8 {
        let mut bad = request.clone();
        match case {
            0 => bad.turn_card = bad.flop.board[0],
            1 => bad.flop.ranges[0].pop().map(|_| ()).unwrap(),
            2 => bad.flop.ranges[0][0] = f64::NAN,
            3 => bad.flop.invested_bb[0] += 0.5,
            4 => bad.flop_line.pop().map(|_| ()).unwrap(),
            5 => bad.flop_line.push("check".into()),
            6 => bad.flop_line[0] = "bet_to_999bb".into(),
            7 => bad.flop_line = vec!["all_in_to_1.000bb".into(), "call_all_in".into()],
            _ => unreachable!(),
        }
        assert!(bad.turn_input().is_err(), "case {case}");
    }
}

#[test]
fn rooted_targets_are_deterministic_and_value_the_exported_policy() {
    let (request, _) = fixture();
    let target = request.generate().unwrap();
    assert_eq!(
        serde_json::to_vec(&target).unwrap(),
        serde_json::to_vec(&request.generate().unwrap()).unwrap()
    );
    let config = request.turn_input().unwrap();
    let solution = belief::solve_turn_river_continuation_values(config).unwrap();
    let values = &target.values;
    for seat in 0..2 {
        let mut weighted = 0.0;
        let mut joint = 0.0;
        for combo in 0..1326 {
            let weight = target.turn_state.ranges[seat][combo];
            let mass = values.opponent_compatible_mass[seat][combo];
            if let Some(value) = values.conditional_values_bb[seat][combo] {
                assert!(
                    weight > 0.0 && mass > 0.0 && value.abs() <= request.game.effective_stack_bb
                );
                // Native f64 versus served f32 is roundoff, not a policy change.
                assert!(
                    (value - solution.counterfactual_values_bb[seat][combo] as f64).abs() < 1e-5
                );
                weighted += weight * mass * value;
                joint += weight * mass;
            } else {
                assert!(weight == 0.0 || mass <= 0.0);
            }
        }
        assert!((weighted / joint - values.response.profile_bb[seat]).abs() < 1e-12);
    }
    assert!(target.policy_rows > 0);
    assert!(values.response.profile_bb.iter().sum::<f64>().abs() < 1e-10);
    assert!(values.response.gain_bb.iter().all(|g| *g >= -1e-10));
}

fn saved_request() -> Request {
    let bytes =
        include_bytes!("../../../../../tests/fixtures/flop-continuation-public-root-a.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        "6d4c437507c5083d4930c9fea471660477a73f166cec680490c488f9c2cce260"
    );
    let data: serde_json::Value = serde_json::from_slice(bytes).unwrap();
    Request {
        source_checkpoint_sha256: data["checkpoint_sha256"].as_str().unwrap().into(),
        game: serde_json::from_value(data["game"].clone()).unwrap(),
        flop: serde_json::from_value(data["public"].clone()).unwrap(),
        policy_seed: 87001,
        flop_iterations: 32,
        maximum_flop_information_sets: 2_000_000,
        flop_line: vec!["check".into(), "bet_to_2.500bb".into(), "call".into()],
        turn_card: 50,
        turn_river_iterations: 64,
    }
}

#[test]
fn saved_root_json_preserves_source_float_bits() {
    // Minimized from the real checkpoint's combo 70. Default serde_json
    // decoding previously shifted this token down by one f64 ULP.
    let token = "0.00010786472252877025";
    let expected: f64 = token.parse().unwrap();
    let decoded: f64 = serde_json::from_str(token).unwrap();
    assert_eq!(decoded.to_bits(), expected.to_bits());
    let request = saved_request();
    assert_eq!(request.flop.ranges[0][70].to_bits(), expected.to_bits());
    let again: Request = serde_json::from_slice(&serde_json::to_vec(&request).unwrap()).unwrap();
    assert_eq!(request.flop, again.flop);
}

#[test]
#[ignore = "explicit source checkpoint plus external 7.5GiB/time/disk guard; public replay parity, not quality certification"]
fn saved_root_matches_full_800_round_reconstruction() {
    let mut request = saved_request();
    let source = PathBuf::from(std::env::var("POKER_FLOP_PILOT_CHECKPOINT").unwrap());
    assert_eq!(
        sha256_file(&source).unwrap(),
        request.source_checkpoint_sha256
    );
    let table = Arc::new(InferenceTable::read(&source).unwrap());
    assert_eq!(table.config, request.game);
    assert_eq!(table.rounds, 800);
    let mut base = TabularResponsePolicy {
        table,
        coverage: RefCell::default(),
        flop_patch: None,
        flop_backoff: None,
        completion_coverage: RefCell::default(),
    };
    let mut patch = flop::FlopPatch::terminal(&TerminalFlopOptions {
        equity_samples: 2048,
        weight: 0.5,
    });
    patch.sampled = Some(FlopResolve::new(32, 87001, 2_000_000));
    base.flop_patch = Some(Arc::new(patch));
    assert_eq!(
        public_ranges(&base, &request.root().unwrap(), &request.flop.board).unwrap(),
        request.flop.ranges
    );
    for line in [
        vec!["check", "check"],
        vec!["check", "bet_to_2.500bb", "call"],
        vec!["check", "bet_to_2.500bb", "raise_to_9.500bb", "call"],
    ] {
        request.flop_line = line.into_iter().map(String::from).collect();
        let compact = request.turn_input().unwrap();
        let full = full_replay(&request, base.isolated_copy());
        assert_eq!(compact.state, full.state);
        println!(
            "{}",
            serde_json::json!({"stage":"rooted_replay_parity", "sourceSha256":request.source_checkpoint_sha256,
            "flopLine":request.flop_line, "turnCard":request.turn_card,
            "inputSha256":format!("{:x}",Sha256::digest(serde_json::to_vec(&compact.state).unwrap())),
            "exactPublicStateParity":true})
        );
    }
}

#[test]
#[ignore = "external 2GiB/time/disk guard; one reused public-root value pilot, not full-game improvement"]
fn saved_root_generates_complete_frozen_turn_values() {
    let target = saved_request().generate().unwrap();
    let path = PathBuf::from(std::env::var("POKER_ROOTED_TURN_OUTPUT").unwrap());
    use std::io::Write;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    let bytes = serde_json::to_vec(&target).unwrap();
    file.write_all(&bytes).unwrap();
    file.sync_all().unwrap();
    println!(
        "{}",
        serde_json::json!({"stage":"rooted_turn_values", "targetSha256":format!("{:x}",Sha256::digest(&bytes)),
        "requestSha256":target.request_sha256, "policySha256":target.policy_sha256, "policyRows":target.policy_rows,
        "supportedCombos":target.values.conditional_values_bb.each_ref().map(|row| row.iter().flatten().count()),
        "response":target.values.response, "interpretation":"one development turn root; complete frozen f32 conditional values, not a full-game exploitability certificate or user EV precision result"})
    );
}
