//! Memory-separated continuation of the interrupted pair: authentic public
//! inputs first, then independent per-root solvers without a full checkpoint.
use super::*;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RootInput {
    game: BlueprintConfig,
    state: PublicBeliefState,
    checkpoint_sha256: String,
    policy_seed: u64,
    hand_index: u64,
    evaluation_seed: u64,
}

#[test]
#[ignore = "complete only the missing source-B prefix; explicit source/output and external guard required"]
fn collect_missing_tail_prefixes() {
    let source = PathBuf::from(std::env::var("POKER_FLOP_PILOT_CHECKPOINT").unwrap());
    let output = PathBuf::from(std::env::var("POKER_FLOP_PILOT_OUTPUT_DIR").unwrap());
    let checkpoint_sha = sha256_file(&source).unwrap();
    assert_eq!(
        checkpoint_sha,
        "7a71a1b13975173af32aed3c610ae62a7c0fb25680ddf4397efef0bde220893d"
    );
    let table = Arc::new(InferenceTable::read(&source).unwrap());
    assert_eq!(table.rounds, 800);
    let game = table.config.clone();
    let policy_seed = 87002;
    let (policy, _) = profile(table, policy_seed);
    let mut chance = SplitMix64::new(89004);
    for index in 0..128 {
        let deal = Deal::sample(&mut chance);
        if index < 29 {
            continue;
        }
        let mut rng = SplitMix64::new(derived_seed(89004, index, 1));
        let mut state = GameState::initial(&game);
        while state.terminal.is_none() && state.street != Street::Turn {
            let actions = state.legal_actions(&game);
            let mix = policy.strategy(&state, &deal, &actions, &game);
            state = state.apply(&actions[sample_index(&mix, &mut rng)], &game);
        }
        let mut input = None;
        if state.terminal.is_none() {
            let config = policy.turn_root_input(&state, &deal.board[..4]).unwrap();
            let encoded = rmp_serde::to_vec_named(&serde_json::json!({
                "game": game, "state": config.state, "checkpointSha256": checkpoint_sha,
                "policySeed": policy_seed, "handIndex": index, "evaluationSeed": 89004,
            }))
            .unwrap();
            let name = format!("root-{policy_seed}-{index}.msgpack");
            let mut file = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(output.join(&name))
                .unwrap();
            file.write_all(&encoded).unwrap();
            file.sync_all().unwrap();
            input = Some(
                serde_json::json!({ "file": name, "sha256": format!("{:x}", Sha256::digest(encoded)) }),
            );
        }
        println!(
            "{}",
            serde_json::json!({ "stage": "split_tail_prefix", "policySeed": policy_seed,
            "index": index, "reachedTurn": state.terminal.is_none(), "input": input })
        );
    }
}

#[test]
#[ignore = "one saved public root without checkpoint; explicit input and external resource guard required"]
fn evaluate_saved_tail_root() {
    let path = PathBuf::from(std::env::var("POKER_TAIL_ROOT_INPUT").unwrap());
    let encoded = fs::read(&path).unwrap();
    let input: RootInput = rmp_serde::from_slice(&encoded).unwrap();
    assert_eq!(input.game.effective_stack_bb, 20.0);
    assert_eq!(input.evaluation_seed, 89004);
    assert!([87001, 87002].contains(&input.policy_seed));
    assert!(input.hand_index < 128);
    let mut results = Vec::new();
    for iterations in [4, 16, 64] {
        let config = belief::TurnRiverSolveConfig {
            game: input.game.clone(),
            state: input.state.clone(),
            iterations,
            averaging_delay: 0,
            river_refinement_iterations: 0,
            regret_matching_plus: false,
        };
        let started = Instant::now();
        let rows = belief::solve_turn_river_policy_probabilities(config.clone()).unwrap();
        let solve_seconds = started.elapsed().as_secs_f64();
        let policy_sha = format!(
            "{:x}",
            Sha256::digest(rmp_serde::to_vec_named(&rows).unwrap())
        );
        let started = Instant::now();
        let response = frozen_turn_response::evaluate(config, &rows).unwrap();
        results.push(
            serde_json::json!({ "iterations": iterations, "response": response,
            "solveSeconds": solve_seconds, "auditSeconds": started.elapsed().as_secs_f64(),
            "policyRows": rows.len(), "policySha256": policy_sha }),
        );
    }
    println!(
        "{}",
        serde_json::json!({ "stage": "split_tail_result", "policySeed": input.policy_seed,
        "index": input.hand_index, "checkpointSha256": input.checkpoint_sha256,
        "inputSha256": format!("{:x}", Sha256::digest(encoded)), "results": results })
    );
}
