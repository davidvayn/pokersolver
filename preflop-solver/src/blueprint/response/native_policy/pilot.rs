//! Explicit bounded integration probe, not a policy-strength measurement.
use super::*;
use std::io::Write;
use std::time::Instant;

fn own_prefix_reach(
    policy: &FrozenPreflopPolicy,
    state: &GameState,
    combo: Combo,
) -> Result<f64, String> {
    let mut cursor = GameState::initial(&policy.game);
    let mut own = 1.0;
    for observed in &state.trajectory {
        let actions = cursor.legal_actions(&policy.game);
        let selected = actions
            .iter()
            .position(|a| trajectory_action_matches(&cursor, a, observed, &policy.game))
            .ok_or("invalid preflop diagnostic history")?;
        if cursor.actor == state.actor && own > 0.0 {
            own *= policy.strategy(&cursor, combo)?[selected];
        }
        cursor = cursor.apply(&actions[selected], &policy.game);
    }
    if cursor.public_history != state.public_history
        || cursor.actor != state.actor
        || cursor.street != state.street
        || cursor.invested != state.invested
    {
        return Err("preflop diagnostic replay mismatch".into());
    }
    Ok(own)
}

#[test]
#[ignore = "retained checkpoint full-hand integration; requires explicit inputs and external resource guard"]
fn native_full_hand_retained_checkpoint_probe() {
    let source =
        PathBuf::from(std::env::var("POKER_NATIVE_CHECKPOINT").expect("explicit checkpoint"));
    let source_sha =
        std::env::var("POKER_NATIVE_CHECKPOINT_SHA").expect("explicit checkpoint hash");
    let output = PathBuf::from(std::env::var("POKER_NATIVE_OUTPUT").expect("explicit new output"));
    assert!(!output.exists());
    assert_eq!(sha256_file(&source).unwrap(), source_sha);
    let started = Instant::now();
    let preflop = Arc::new(FrozenPreflopPolicy::read(&source).unwrap());
    let loading_seconds = started.elapsed().as_secs_f64();
    assert_eq!(preflop.rounds, 800);
    assert_eq!(preflop.game.effective_stack_bb, 20.0);
    let game = preflop.game.clone();
    let policy = NativeFullHandPolicy::new(
        preflop,
        NativeFlopOptions {
            seed: 100101,
            iterations: 2,
            training_turn_iterations: 4,
            response_turn_iterations: 4,
        },
    );
    println!(
        "{}",
        serde_json::json!({"stage":"preflop_loaded", "seconds":loading_seconds,
        "nodes":policy.preflop.node_count()})
    );
    let mut pending = vec![GameState::initial(&game)];
    let mut queried = 0;
    let mut public_nodes = 0;
    let mut missing = 0;
    let mut first_error = None;
    let mut missing_examples = Vec::new();
    let mut missing_zero_own_reach = 0;
    let mut missing_terminal_fold_call = 0;
    let mut maximum_missing_own_reach = 0.0f64;
    let mut maximum_sum_error = 0.0f64;
    while let Some(state) = pending.pop() {
        if state.terminal.is_some() || state.street != Street::Preflop {
            continue;
        }
        public_nodes += 1;
        for combo in all_combos() {
            queried += 1;
            match policy.preflop.strategy(&state, combo) {
                Ok(mix) => {
                    maximum_sum_error = maximum_sum_error.max((mix.iter().sum::<f64>() - 1.0).abs())
                }
                Err(error) => {
                    missing += 1;
                    let own_reach = own_prefix_reach(&policy.preflop, &state, combo).unwrap();
                    maximum_missing_own_reach = maximum_missing_own_reach.max(own_reach);
                    missing_zero_own_reach += usize::from(own_reach == 0.0);
                    let actions = state.legal_actions(&game);
                    missing_terminal_fold_call += usize::from(
                        actions.len() == 2
                            && actions.iter().any(|a| a.kind == ActionKind::Fold)
                            && actions.iter().any(|a| a.kind == ActionKind::Call)
                            && actions
                                .iter()
                                .all(|a| state.apply(a, &game).terminal.is_some()),
                    );
                    if missing_examples.len() < 8 {
                        missing_examples.push(serde_json::json!({"combo":combo.cards(),
                            "history":state.public_history,"error":error,"ownPrefixReach":own_reach}));
                    }
                    first_error.get_or_insert(error);
                }
            }
        }
        pending.extend(
            state
                .legal_actions(&game)
                .iter()
                .map(|a| state.apply(a, &game)),
        );
    }
    println!(
        "{}",
        serde_json::json!({"stage":"preflop_exhaustive", "publicNodes":public_nodes,
        "queries":queried,"missing":missing,"maximumSumError":maximum_sum_error,"firstError":first_error,
        "missingExamples":missing_examples,"missingZeroOwnReach":missing_zero_own_reach,
        "maximumMissingOwnReach":maximum_missing_own_reach,"missingTerminalFoldCall":missing_terminal_fold_call})
    );
    // Preserve failures in the report. A missing off-path row need not prevent
    // diagnosing supported trajectories; query itself still fails closed if
    // a missing row is required. Completion is not coverage qualification.

    // Cross-check the old public-root fixture without loading its full source
    // table. Both source seeds are tested by the external paired controller;
    // only seed A owns this archived fixture.
    let mut fixture_range_error = None;
    if source_sha == "8fc95d56696af9fc8a858fceb8efdad4782a049b5af7b0a8f1cdcc5d694f95aa" {
        let saved: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/flop-continuation-public-root-a.json"
        ))
        .unwrap();
        let expected: PublicBeliefState = serde_json::from_value(saved["public"].clone()).unwrap();
        let mut root = GameState::initial(&game);
        while root.street == Street::Preflop {
            let action = root
                .legal_actions(&game)
                .into_iter()
                .find(|a| matches!(a.kind, ActionKind::Call | ActionKind::Check))
                .unwrap();
            root = root.apply(&action, &game);
        }
        assert_eq!(root.public_history, expected.public_history);
        let replay = policy.root_input(&root, &expected.board).unwrap();
        let error = expected
            .ranges
            .iter()
            .flatten()
            .zip(replay.ranges.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            error < 1e-14,
            "strict reader changed archived preflop range: {error}"
        );
        fixture_range_error = Some(error);
    }

    // Fresh deterministic cards, explicitly forced check/call trajectory to
    // exercise all streets. This is not a random-policy strength sample.
    let deal = Deal::sample(&mut SplitMix64::new(881901));
    let mut state = GameState::initial(&game);
    let mut decisions = Vec::new();
    while state.terminal.is_none() {
        let actions = state.legal_actions(&game);
        let before = Instant::now();
        let mix = policy.strategy(&state, &deal, &actions, &game);
        let seconds = before.elapsed().as_secs_f64();
        let action = actions
            .iter()
            .find(|a| matches!(a.kind, ActionKind::Call | ActionKind::Check))
            .unwrap();
        decisions.push(serde_json::json!({"street":state.street,"actor":state.actor,
            "history":state.public_history,"actions":actions.iter().map(|a| &a.label).collect::<Vec<_>>(),
            "probabilities":mix,"seconds":seconds,"forcedAction":action.label}));
        state = state.apply(action, &game);
    }
    assert_eq!(decisions.len(), 8);
    let result = serde_json::json!({"schema":"native-full-hand-integration-probe-v1",
        "checkpointSha256":source_sha,"loadingSeconds":loading_seconds,
        "preflopPublicNodes":public_nodes,"preflopQueries":queried,"preflopMissing":missing,
        "preflopExhaustiveCoveragePass":(missing as f64 / queried as f64) <= 0.0001,
        "preflopMissingExamples":missing_examples,
        "preflopMissingZeroOwnReach":missing_zero_own_reach,
        "preflopMaximumMissingOwnReach":maximum_missing_own_reach,
        "preflopMissingTerminalFoldCall":missing_terminal_fold_call,
        "maximumProbabilitySumError":maximum_sum_error,"fixtureRangeMaximumError":fixture_range_error,
        "cardsSeed":881901,"decisions":decisions,"diagnostics":policy.take_resolution_diagnostics(),
        "seconds":started.elapsed().as_secs_f64(),
        "interpretation":"Strict checkpoint reader and full-hand research integration only. Forced trajectory, tiny 2/4/4 solve budget; no exploitability, strength or release qualification."});
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap();
    file.write_all(&serde_json::to_vec_pretty(&result).unwrap())
        .unwrap();
    file.sync_all().unwrap();
    println!("{}", result);
}
