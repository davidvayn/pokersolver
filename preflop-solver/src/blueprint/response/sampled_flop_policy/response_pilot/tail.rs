//! Paired exact turn/river attacks, averaged over authentic full-hand prefixes.
//! Earlier-street deviations are excluded: these are lower bounds, not a
//! full-game exploitability certificate. All non-turn-reaching hands count zero.
use super::*;
use crate::blueprint::public_belief::{self as belief, frozen_turn_response};

#[test]
fn frozen_tail_snapshot_is_the_policy_used_by_actual_play() {
    let (mut base, _) = super::super::super::tests::tabular_fixture();
    let table = Arc::get_mut(&mut base.table).unwrap();
    table.config.effective_stack_bb = 2.0;
    table.nodes.clear();
    let game = table.config.clone();
    let (policy, _) = profile(base.table, 87001);
    let deal = Deal::from_sampled_cards([[48, 49], [44, 45]], [0, 5, 10, 15, 20]);
    let mut state = GameState::initial(&game);
    while state.street != Street::Turn {
        let action = state
            .legal_actions(&game)
            .into_iter()
            .find(|a| matches!(a.kind, ActionKind::Call | ActionKind::Check))
            .unwrap();
        state = state.apply(&action, &game);
    }
    let (config, rows) = policy
        .take_frozen_turn_profile(&state, &deal.board[..4])
        .unwrap();
    let repeated = belief::solve_turn_river_policy_probabilities(config.clone()).unwrap();
    // Snapshot order is sorted, whereas the solve's traversal export may differ.
    let sorted = |rows: Vec<PublicBeliefStrategy>| {
        rows.into_iter()
            .map(|r| (r.public_history.clone(), r))
            .collect::<BTreeMap<_, _>>()
    };
    assert_eq!(
        rmp_serde::to_vec_named(&sorted(rows.clone())).unwrap(),
        rmp_serde::to_vec_named(&sorted(repeated)).unwrap()
    );
    let row = rows
        .iter()
        .find(|r| r.public_history == state.public_history)
        .unwrap();
    let actions = state.legal_actions(&game);
    let expected = row_mix(
        row,
        Combo::new(deal.holes[state.actor][0], deal.holes[state.actor][1]),
        &state,
        &actions,
    );
    assert_eq!(expected, policy.strategy(&state, &deal, &actions, &game));
    let result = frozen_turn_response::evaluate(config, &rows).unwrap();
    assert!(result.gain_bb.iter().all(|v| *v >= -1e-8));
}

#[test]
#[ignore = "bounded exact tail-response policy pilot; explicit source/output and external guard required"]
fn sampled_profile_exact_tail_pair() {
    let source = PathBuf::from(std::env::var("POKER_FLOP_PILOT_CHECKPOINT").unwrap());
    let output = PathBuf::from(std::env::var("POKER_FLOP_PILOT_OUTPUT_DIR").unwrap());
    assert!(output.is_dir());
    let checkpoint_sha = sha256_file(&source).unwrap();
    let table = Arc::new(InferenceTable::read(&source).unwrap());
    assert_eq!(table.rounds, 800);
    assert_eq!(table.config.effective_stack_bb, 20.0);
    let game = table.config.clone();
    for policy_seed in [87001, 87002] {
        let (policy, _) = profile(table.clone(), policy_seed);
        let mut chance = SplitMix64::new(89004);
        let started = Instant::now();
        let mut reached = 0;
        for index in 0..128 {
            let deal = Deal::sample(&mut chance);
            let mut rng = SplitMix64::new(derived_seed(89004, index, 1));
            let mut state = GameState::initial(&game);
            while state.terminal.is_none() && state.street != Street::Turn {
                let actions = state.legal_actions(&game);
                let mix = policy.strategy(&state, &deal, &actions, &game);
                state = state.apply(&actions[sample_index(&mix, &mut rng)], &game);
            }
            let mut results = Vec::new();
            if state.terminal.is_none() {
                reached += 1;
                let (config, baseline) = policy
                    .take_frozen_turn_profile(&state, &deal.board[..4])
                    .unwrap();
                let name = format!("root-{policy_seed}-{index}.msgpack");
                // Store public state/ranges so follow-up improvements need not
                // rerun the preflop/flop solver. Never store an actual holding.
                let encoded = rmp_serde::to_vec_named(&serde_json::json!({
                    "game": game, "state": config.state, "checkpointSha256": checkpoint_sha,
                    "policySeed": policy_seed, "handIndex": index, "evaluationSeed": 89004,
                }))
                .unwrap();
                let mut file = fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(output.join(&name))
                    .unwrap();
                file.write_all(&encoded).unwrap();
                file.sync_all().unwrap();
                let mut baseline = Some(baseline);
                for iterations in [4, 16, 64] {
                    let mut candidate = config.clone();
                    candidate.iterations = iterations;
                    let solve_start = Instant::now();
                    let rows = if iterations == 4 {
                        baseline.take().unwrap()
                    } else {
                        belief::solve_turn_river_policy_probabilities(candidate.clone()).unwrap()
                    };
                    let solve_seconds = solve_start.elapsed().as_secs_f64();
                    let policy_sha = format!(
                        "{:x}",
                        Sha256::digest(rmp_serde::to_vec_named(&rows).unwrap())
                    );
                    let audit_start = Instant::now();
                    let response = frozen_turn_response::evaluate(candidate, &rows).unwrap();
                    results.push(serde_json::json!({ "iterations": iterations, "response": response,
                        "solveSeconds": solve_seconds, "auditSeconds": audit_start.elapsed().as_secs_f64(),
                        "policyRows": rows.len(), "policySha256": policy_sha }));
                }
                println!(
                    "{}",
                    serde_json::json!({ "stage": "exact_tail_root", "policySeed": policy_seed,
                    "index": index, "file": name, "sha256": format!("{:x}", Sha256::digest(encoded)),
                    "results": results })
                );
            }
            println!(
                "{}",
                serde_json::json!({ "stage": "exact_tail_hand", "policySeed": policy_seed,
                "index": index, "reachedTurn": state.terminal.is_none(), "results": results })
            );
        }
        println!(
            "{}",
            serde_json::json!({ "stage": "exact_tail_complete", "policySeed": policy_seed,
            "hands": 128, "turnRoots": reached, "seconds": started.elapsed().as_secs_f64(),
            "validation": "authentic_prefix_exact_turn_river_response_lower_bound_not_full_game_upper_bound" })
        );
    }
}
