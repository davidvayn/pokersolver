//! Opt-in matched policy and latency tests. No website routing changes.
use super::*;
use std::time::Instant;

#[derive(Serialize, Deserialize)]
struct Prefix {
    protocol_sha256: String,
    state: PublicBeliefState,
    rows: Vec<FrozenRow>,
}

fn check(state: &GameState, game: &BlueprintConfig) -> GameState {
    let action = state
        .legal_actions(game)
        .into_iter()
        .find(|a| a.label == "check")
        .unwrap();
    state.apply(&action, game)
}

fn fresh_prefix(
    engine: &PracticePolicyEngine,
    case: &serde_json::Value,
    protocol_sha256: String,
) -> Prefix {
    let board: Vec<u8> = serde_json::from_value(case["board"].clone()).unwrap();
    let mut state = preflop(&engine.game, case["preflop"].as_array().unwrap());
    assert_eq!(
        state.invested.iter().sum::<f64>(),
        case["potBb"].as_f64().unwrap()
    );
    let mut raw = served(engine, &state, &board).ranges.clone();
    let mut rows = vec![];
    for _ in 0..2 {
        let row = served(engine, &state, &board);
        let n = row.action_labels.len();
        let a = row.action_labels.iter().position(|s| s == "check").unwrap();
        for (c, w) in raw[state.actor].iter_mut().enumerate() {
            *w *= row.probabilities[c * n + a];
        }
        rows.push(FrozenRow {
            state: PublicBeliefState::from_game_state(board.clone(), &state, row.ranges.clone()),
            probabilities: row.probabilities.clone(),
        });
        state = check(&state, &engine.game);
    }
    assert_eq!(state.street, Street::Turn);
    Prefix {
        protocol_sha256,
        state: PublicBeliefState::from_game_state(board, &state, raw),
        rows,
    }
}

type JointRows = BTreeMap<Vec<String>, (usize, Vec<String>, Vec<f64>)>;

fn joint_key(state: &GameState, board: &[u8]) -> Vec<String> {
    let mut key = state.public_history.clone();
    if state.street == Street::River {
        key.push(format!("chance:river:{}", board[4]));
    }
    key
}

fn joint_row(
    rows: &JointRows,
    state: &GameState,
    board: &[u8],
    game: &BlueprintConfig,
) -> Result<Vec<f64>, String> {
    let (actor, labels, probabilities) = rows
        .get(&joint_key(state, board))
        .ok_or("missing joint continuation row")?;
    if *actor != state.actor
        || labels
            .iter()
            .ne(state.legal_actions(game).iter().map(|a| &a.label))
    {
        return Err("joint continuation actor or action order mismatch".into());
    }
    Ok(probabilities.clone())
}

#[test]
#[ignore = "requires pinned matched-turn protocol and artifacts"]
fn run_matched_turn_task() {
    let root = PathBuf::from(std::env::var("POKER_LIVE_NASH_ROOT").unwrap());
    let protocol_path = PathBuf::from(std::env::var("POKER_MATCHED_PROTOCOL").unwrap());
    let protocol: serde_json::Value = read(&protocol_path);
    let case_id = std::env::var("POKER_LIVE_NASH_CASE").unwrap();
    let case = protocol["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["id"] == case_id)
        .unwrap();
    let output = PathBuf::from(std::env::var("POKER_MATCHED_RESULT").unwrap());
    let task = std::env::var("POKER_MATCHED_TASK").unwrap();
    let mut engine = serving_engine(
        &root,
        &protocol["manifest"],
        protocol["threadsPerWorker"].as_u64().unwrap() as usize,
    );
    let protocol_sha256 = digest(&protocol_path);
    if task == "prefix" {
        assert_eq!(case["cohort"], "fresh");
        write(&output, &fresh_prefix(&engine, case, protocol_sha256));
        return;
    }
    let source = PathBuf::from(std::env::var("POKER_MATCHED_PREFIX").unwrap());
    let mut public = if case["cohort"] == "matched" {
        let base: Base = read(&source);
        assert_eq!(base.case, case_id);
        assert_eq!(
            base.protocol_sha256,
            protocol["originalProtocolSha256"].as_str().unwrap()
        );
        assert_eq!(digest(&source), case["baseSha256"].as_str().unwrap());
        install(&engine, &base);
        base.nodes
            .iter()
            .find(|node| checkdown_turn(node))
            .unwrap()
            .state
            .clone()
    } else {
        let prefix: Prefix = read(&source);
        assert_eq!(prefix.protocol_sha256, protocol_sha256);
        install_flop_rows(&engine, &prefix.rows);
        prefix.state
    };
    let board: Vec<u8> = serde_json::from_value(case["board"].clone()).unwrap();
    assert_eq!(public.board, board);
    let turn = case["turn"].as_u64().unwrap() as u8;
    assert!(turn < 52 && !public.board.contains(&turn));
    public.board.push(turn);
    for range in &mut public.ranges {
        for combo in all_combos() {
            if combo.cards().contains(&turn) {
                range[combo.key()] = 0.0;
            }
        }
    }
    let reaches = public.ranges.clone();
    let root = response::game_state(&public);
    let board = public.board.clone();
    // Start timing after local model initialization and entry-belief assembly;
    // exclude evaluator construction, best responses, I/O and browser latency.
    let belief = engine
        .policy
        .ranges_for_state(&root, &deal(&board, root.actor), &engine.game)
        .unwrap();
    let mut normalized = reaches.clone();
    normalize_ranges_for_board(&mut normalized, &board).unwrap();
    let reach_delta = belief
        .iter()
        .flatten()
        .zip(normalized.iter().flatten())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(
        reach_delta < 1e-12,
        "incompatible turn start prior: {reach_delta}"
    );
    let mut config = pbs::TurnRiverSolveConfig {
        game: engine.game.clone(),
        state: public,
        iterations: 2,
        averaging_delay: 0,
        river_refinement_iterations: 0,
        regret_matching_plus: false,
    };
    let variant = std::env::var("POKER_MATCHED_VARIANT").unwrap();
    let control = engine.policy.turn_resolver.unwrap();
    assert_eq!(control.iterations, 2);
    assert!(!control.safe_resolving && control.resolved_actor == Some(1));
    let joint = variant == "joint-2";
    match variant.as_str() {
        "live-2" | "joint-2" => (),
        "routed-32" => engine
            .policy
            .enable_turn_resolver(TurnResolverConfig {
                iterations: 32,
                ..control
            })
            .unwrap(),
        _ => panic!("unexpected turn variant"),
    }
    let began = Instant::now();
    let mut rows: JointRows = BTreeMap::new();
    if joint {
        let mut solve_config = config.clone();
        solve_config.state.ranges = belief;
        for row in pbs::solve_turn_river_policy(solve_config).unwrap() {
            let probabilities = stabilize_resolved_policy(
                row.probabilities.into_iter().map(f64::from).collect(),
                row.action_labels.len(),
            )
            .unwrap();
            assert!(rows
                .insert(
                    row.public_history,
                    (row.actor, row.action_labels, probabilities)
                )
                .is_none());
        }
        joint_row(&rows, &root, &board, &engine.game).unwrap();
    } else {
        served(&engine, &root, &board);
    }
    let cold = began.elapsed().as_secs_f64();
    let mut result = serde_json::json!({"variant":variant,"case":case_id,
        "protocolSha256":protocol_sha256,"prefixSha256":digest(&source),"rangeDelta":reach_delta,
        "turn":turn,"coldTurnSeconds":cold});
    if task == "timing" {
        let checked = check(&root, &engine.game);
        let river = check(&checked, &engine.game);
        assert_eq!(river.street, Street::River);
        if !joint {
            served(&engine, &checked, &board);
        }
        let mut runout = board.clone();
        runout.push((0..52).find(|c| !board.contains(c)).unwrap());
        let river_started = Instant::now();
        if joint {
            joint_row(&rows, &river, &runout, &engine.game).unwrap();
        } else {
            served(&engine, &river, &runout);
        }
        result["coldRiverSeconds"] = serde_json::json!(river_started.elapsed().as_secs_f64());
        result["river"] = serde_json::json!(runout[4]);
    } else {
        assert_eq!(task, "evaluate");
        let evaluation_started = Instant::now();
        // Iterations on the evaluation config validate dimensions only; the
        // evaluator imports frozen rows and never calls train.
        config.state.ranges = reaches.clone();
        let (values, nodes, error) = response::turn(config, reaches.clone(), |state, board| {
            if joint {
                joint_row(&rows, state, board, &engine.game)
            } else {
                Ok(served(&engine, state, board).probabilities.clone())
            }
        })
        .unwrap();
        if joint {
            assert_eq!(nodes, rows.len());
        }
        result["metrics"] = response::metrics(&values, &reaches).unwrap();
        result["publicNodes"] = serde_json::json!(nodes);
        result["maximumProbabilitySumError"] = serde_json::json!(error);
        result["offlineEvaluationSeconds"] =
            serde_json::json!(evaluation_started.elapsed().as_secs_f64());
        eprintln!(
            "{} {} NashConv: {}",
            case_id, variant, result["metrics"]["nashConvBb"]
        );
    }
    write(&output, &result);
}
