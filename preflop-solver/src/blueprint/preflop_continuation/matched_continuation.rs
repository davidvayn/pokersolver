//! Fixed-policy, matched-chance allocation experiment. No trainer mutation.
//! Only redistribute the existing fold/call mass at two disjoint public roots;
//! this is not a best response, new preflop policy, or exploitability bound.
use super::*;
use super::frozen_response::Classes;

fn roots(game: &BlueprintConfig) -> Vec<GameState> {
    [vec!["raise_to_4.000bb"], vec!["limp", "raise_to_5.000bb"]]
        .into_iter().map(|line| {
            let mut state = GameState::initial(game);
            for label in line {
                let action = state.legal_actions(game).into_iter().find(|a| a.label == label).unwrap();
                state = state.apply(&action, game);
            }
            state
        }).collect()
}

#[test]
fn matched_roots_are_disjoint_and_call_reaches_flop() {
    let game = BlueprintConfig { effective_stack_bb: 20.0, ..BlueprintConfig::default() };
    let states = roots(&game);
    assert!(!states[0].public_history.starts_with(&states[1].public_history));
    assert!(!states[1].public_history.starts_with(&states[0].public_history));
    for state in states {
        for label in ["fold", "call"] {
            let action = state.legal_actions(&game).into_iter().find(|a| a.label == label).unwrap();
            let next = state.apply(&action, &game);
            if label == "call" { assert_eq!(next.street, Street::Flop); assert!(next.terminal.is_none()); }
            else { assert!(next.terminal.is_some()); }
        }
    }
}

#[test]
#[ignore = "bounded matched continuation allocation probe; external guard required"]
fn matched_call_fold_capture() {
    let preflop = response::FrozenPreflopPolicy::read(Path::new(&std::env::var("POKER_NOISE_PREFLOP").unwrap())).unwrap();
    assert_eq!(preflop.artifact_sha256, std::env::var("POKER_NOISE_PREFLOP_SHA").unwrap());
    assert_eq!(preflop.node_count(), 16900);
    assert_eq!(preflop.game.effective_stack_bb, 20.0);
    let model = PublicValueNetwork::read(Path::new(&std::env::var("POKER_COMPACT_MODEL").unwrap())).unwrap();
    assert_eq!(model.artifact_sha256().unwrap(), std::env::var("POKER_COMPACT_MODEL_SHA").unwrap());
    let kernel_sha = std::env::var("POKER_COMPACT_CHECKDOWN_SHA").unwrap();
    let exact = exact_checkdown::ExactCheckdown::read(Path::new(&std::env::var("POKER_COMPACT_CHECKDOWN").unwrap()), &kernel_sha).unwrap();
    let index: u64 = std::env::var("POKER_MATCH_BOARD").unwrap().parse().unwrap();
    let strength: u64 = std::env::var("POKER_MATCH_STRENGTH").unwrap().parse().unwrap();
    assert!(index < 8 && [1, 2].contains(&strength));
    let output = PathBuf::from(std::env::var("POKER_COMPACT_OUTPUT").unwrap());
    assert!(!output.exists());
    let started = Instant::now();
    let snapshot = Snapshot::capture_frozen(&preflop).unwrap();
    let classes = Classes::new();
    let mut rng = SplitMix64::new(batch_iteration_seed(71201, index + 1));
    let deal = Deal::sample(&mut rng);
    let board: [u8; 3] = deal.board[..3].try_into().unwrap();
    let turns: Vec<_> = (0..52).filter(|c| !board.contains(c)).collect();
    let turn = turns[rng.index(turns.len())];
    let kernel = exact_flop_kernel(board).unwrap();
    let baseline_values = fixed_control::values(&snapshot, &exact).unwrap();
    let mut records = Vec::new();
    for state in roots(&preflop.game) {
        let before = Instant::now();
        let row = &snapshot.rows[&state.public_history];
        let actions = state.legal_actions(&preflop.game);
        let fold = actions.iter().position(|a| a.label == "fold").unwrap();
        let call = actions.iter().position(|a| a.label == "call").unwrap();
        let folded = state.apply(&actions[fold], &preflop.game);
        let called = state.apply(&actions[call], &preflop.game);
        let (end, prior) = &snapshot.endpoints[&called.public_history];
        let input = snapshot.flop_input(&called.public_history, board).unwrap();
        let root_json = serde_json::to_value(&input).unwrap();
        let digest = Sha256::digest(serde_json::to_vec(&input).unwrap());
        let options = NativeFlopOptions {
            seed: 100101 ^ u64::from_le_bytes(digest[..8].try_into().unwrap()),
            iterations: 128 * strength, training_turn_iterations: 64 * strength,
            response_turn_iterations: 64 * strength,
        };
        let policy = NativePostflopPolicy::solve_pinned_compact_continuation(
            preflop.game.clone(), input, &options, &model, false).unwrap();
        let native = policy.sampled_profile_values_with_turn_baseline(turn, &model).unwrap();
        let mut masked = prior.clone();
        mask_ranges(&mut masked, board);
        let totals: [f64; 2] = std::array::from_fn(|p| masked[p].iter().sum());
        let p = state.actor;
        let baseline = kernel.values_on_flop(end.invested, &masked[1-p], p).unwrap();
        let values: Vec<_> = (0..1326).map(|c| baseline_values[&called.public_history][p][c]
            + (native[p][c] * totals[1-p] - baseline[c]) * FLOP_CHANCE_CORRECTION / OPPONENT_HANDS).collect();
        let record = serde_json::json!({"history":state.public_history,"seat":p,
            "foldCfvBb":classes.reduce(&baseline_values[&folded.public_history][p]),
            "callCfvBb":classes.reduce(&values),"ownPrefixReach":classes.reduce(&row.reaches[p]),
            "foldProbability":classes.reduce(&row.probabilities[fold]),
            "callProbability":classes.reduce(&row.probabilities[call]),
            "publicInput":root_json,"publicInputSha256":format!("{:x}",digest),
            "continuationPolicySha256":policy.identity(),"seconds":before.elapsed().as_secs_f64()});
        eprintln!("{}",serde_json::json!({"stage":"matched_call_fold","board":index,"strength":strength,
            "completed":records.len()+1,"seconds":before.elapsed().as_secs_f64()}));
        records.push(record);
    }
    let payload = serde_json::json!({"schema":"matched-call-fold-capture-v1",
        "preflopSha256":preflop.artifact_sha256,"modelSha256":model.artifact_sha256(),"kernelSha256":kernel_sha,
        "chanceSeed":71201,"boardIndex":index,"board":board,"turn":turn,"strength":strength,
        "flopIterations":128*strength,"turnIterations":64*strength,
        "classes":classes.labels,"multiplicities":classes.counts,"records":records,
        "seconds":started.elapsed().as_secs_f64(),"releaseAccepted":false,"fullGameGateEvaluated":false});
    let mut file = fs::OpenOptions::new().write(true).create_new(true).open(output).unwrap();
    file.write_all(&serde_json::to_vec(&payload).unwrap()).unwrap();
    file.sync_all().unwrap();
}
