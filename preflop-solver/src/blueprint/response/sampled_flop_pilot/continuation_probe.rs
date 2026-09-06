//! Bounded red-capable diagnostic, not a policy-quality or release gate.
use super::*;

#[test]
#[ignore = "explicit frozen public-root cache and 2GiB/time guard; intentionally fails if a frozen training continuation cannot be evaluated"]
fn training_continuation_must_be_complete_before_comparing_action_values() {
    let file = PathBuf::from(std::env::var("POKER_FLOP_PILOT_CACHE").unwrap());
    let bytes = fs::read(&file).unwrap();
    let data = cache::RootActionCache::decode(&bytes).unwrap();
    let game = &data.context.game;
    assert_eq!(game.effective_stack_bb, 20.0);
    let root = data.context.root().unwrap();
    assert_eq!(root.actor, 1);
    assert_eq!(root.to_call(), 0.0);
    assert!(root.trajectory.iter().all(|a| a.street == Street::Preflop));
    let identity = serde_json::to_vec(&(&data.context.public.board, &root.public_history)).unwrap();
    let digest = Sha256::digest(identity);
    let seed = 87001 ^ u64::from_le_bytes(digest[..8].try_into().unwrap());
    let mut total_missing = 0;
    for exact in [false, true] {
        let (solution, frozen) = sampled_flop::continuation::capture(
            sampled_flop::SampledFlopConfig {
                game: game.clone(),
                state: data.context.public.clone(),
                iterations: 32,
                seed,
                maximum_information_sets: 2_000_000,
            },
            exact,
        )
        .unwrap();
        let actions = root.legal_actions(game);
        let mut chance = SplitMix64::new(99004);
        let mut finished = 0;
        let mut missing = BTreeMap::<String, usize>::new();
        let mut first_failure = None;
        for sample in 0..64 {
            let deal = conditional_deal(
                &data.context.public.board,
                &data.context.public.ranges,
                &mut chance,
            );
            // Enumerate the root action, matching the planned action-value test.
            for action in &actions {
                let mut state = root.apply(action, game);
                let mut rng = SplitMix64::new(derived_seed(99004, 0x636f6e74696e, sample));
                while state.terminal.is_none() {
                    let legal = state.legal_actions(game);
                    match frozen.query(&state, &deal, &legal) {
                        Ok(mix) => state = state.apply(&legal[sample_index(mix, &mut rng)], game),
                        Err(reason) => {
                            *missing
                                .entry(format!("{:?}:{reason}", state.street))
                                .or_default() += 1;
                            if first_failure.is_none() {
                                first_failure = Some(serde_json::json!({"sample":sample,
                                    "rootAction":action.label,"street":state.street,"actor":state.actor,
                                    "ownCards":deal.holes[state.actor],"visibleBoard":&deal.board[..state.street.board_len()],
                                    "history":state.public_history,"reason":reason}));
                            }
                            break;
                        }
                    }
                }
                if state.terminal.is_some() {
                    finished += 1;
                }
            }
        }
        total_missing += missing.values().sum::<usize>();
        assert_eq!(
            finished + missing.values().sum::<usize>(),
            64 * actions.len()
        );
        println!(
            "{}",
            serde_json::json!({"stage":"training_continuation_support",
            "cacheSha256":format!("{:x}",Sha256::digest(&bytes)),"sourceSha256":data.context.checkpoint_sha256,
            "inputSha256":solution.input_sha256,"exactTerminalTraining":exact,"iterations":32,
            "trainingSeed":seed,"evaluationSeed":99004,"samples":64,"rootActions":actions.len(),
            "board":data.context.public.board,"history":root.public_history,
            "completedActionRollouts":finished,"missing":missing,"firstFailure":first_failure,
            "interpretation":"continuation diagnostic on a reused development root; interrupted decisions unscored, no fallback; not full-hand coverage or exploitability"})
        );
    }
    assert_eq!(total_missing, 0, "cannot compare trained-versus-served action EVs without inventing missing continuation actions");
}
