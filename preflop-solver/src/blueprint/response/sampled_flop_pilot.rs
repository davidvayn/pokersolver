//! Explicit development pilots: conditional one-decision comparisons and an
//! opt-in full-hand flop-routing experiment. Neither certifies exploitability.
use super::*;
use crate::blueprint::neural::{
    deal_for_policy_combo_on_board, normalize_ranges_for_board, trajectory_action_matches,
};
use crate::blueprint::public_belief::{sampled_flop, PublicBeliefState, PublicBeliefStrategy};
use std::time::Instant;
mod cache;

pub(super) fn public_ranges(
    base: &TabularResponsePolicy,
    root: &GameState,
    board: &[u8],
) -> Result<[Vec<f64>; 2], String> {
    let game = &base.table.config;
    let mut cursor = GameState::initial(game);
    let mut ranges = [vec![1.0 / 1326.0; 1326], vec![1.0 / 1326.0; 1326]];
    for observed in &root.trajectory {
        let visible = &board[..cursor.street.board_len()];
        normalize_ranges_for_board(&mut ranges, visible)?;
        let actions = cursor.legal_actions(game);
        let selected = actions
            .iter()
            .position(|action| trajectory_action_matches(&cursor, action, observed, game))
            .ok_or("flop pilot could not replay the public line")?;
        for combo in all_combos() {
            if ranges[cursor.actor][combo.key()] == 0.0 {
                continue;
            }
            let synthetic = deal_for_policy_combo_on_board(combo, cursor.actor, visible)?;
            ranges[cursor.actor][combo.key()] *=
                base.frozen_strategy(&cursor, &synthetic, &actions, game)[selected];
        }
        cursor = cursor.apply(&actions[selected], game);
    }
    if cursor.public_history != root.public_history
        || cursor.actor != root.actor
        || cursor.street != Street::Flop
    {
        return Err("flop pilot range replay differs from root".to_owned());
    }
    normalize_ranges_for_board(&mut ranges, board)?;
    Ok(ranges)
}

fn conditional_deal(board: &[u8], ranges: &[Vec<f64>; 2], rng: &mut SplitMix64) -> Deal {
    let combos = all_combos();
    for _ in 0..100_000 {
        let holes = std::array::from_fn(|seat| combos[sample_index(&ranges[seat], rng)].cards());
        if holes[0].iter().any(|card| holes[1].contains(card)) {
            continue;
        }
        let mut deck = (0..52u8)
            .filter(|card| {
                !board.contains(card) && !holes.iter().flatten().any(|hole| hole == card)
            })
            .collect::<Vec<_>>();
        assert_eq!(deck.len(), 45);
        let turn = deck.swap_remove(rng.index(deck.len()));
        let river = deck.swap_remove(rng.index(deck.len()));
        return Deal::from_sampled_cards(holes, [board[0], board[1], board[2], turn, river]);
    }
    panic!("flop pilot cannot sample a compatible pair; no fallback deal");
}

pub(super) fn row_mix(
    row: &PublicBeliefStrategy,
    combo: Combo,
    state: &GameState,
    actions: &[LegalAction],
) -> Vec<f64> {
    assert_eq!(row.actor, state.actor);
    assert_eq!(row.public_history, state.public_history);
    assert!(row
        .action_labels
        .iter()
        .map(String::as_str)
        .eq(actions.iter().map(|a| a.label.as_str())));
    let n = actions.len();
    let mut mix = row.probabilities[combo.key() * n..(combo.key() + 1) * n]
        .iter()
        .map(|p| *p as f64)
        .collect::<Vec<_>>();
    let sum: f64 = mix.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6 && mix.iter().all(|p| p.is_finite() && *p >= 0.0));
    // Only remove f32 export roundoff; missing/invalid rows were rejected above.
    for p in &mut mix {
        *p /= sum;
    }
    mix
}

// Evaluate both proposals on precisely the same independent hidden-card draws
// and per-action continuation streams. Enumerate the first action (no sampled
// action-frequency noise); all subsequent actions use the retained profile.
#[allow(clippy::too_many_arguments)]
fn compare(
    policy: &dyn ResponsePolicy,
    game: &BlueprintConfig,
    state: &GameState,
    board: &[u8],
    ranges: &[Vec<f64>; 2],
    rows: &[PublicBeliefStrategy],
    seed: u64,
    samples: u64,
) -> Vec<ValueAccumulator> {
    let actions = state.legal_actions(game);
    let mut rng = SplitMix64::new(seed);
    let mut values = vec![ValueAccumulator::default(); rows.len()];
    for sample in 0..samples {
        let deal = conditional_deal(board, ranges, &mut rng);
        let combo = Combo::new(deal.holes[state.actor][0], deal.holes[state.actor][1]);
        let baseline = policy.strategy(state, &deal, &actions, game);
        let q: Vec<_> = actions
            .iter()
            .map(|action| {
                let mut continuation_rng = SplitMix64::new(derived_seed(seed, 0x666c6f70, sample));
                let p0 = baseline_rollout(
                    policy,
                    state.apply(action, game),
                    &deal,
                    game,
                    &mut continuation_rng,
                );
                if state.actor == 0 {
                    p0
                } else {
                    -p0
                }
            })
            .collect();
        for (row, accumulator) in rows.iter().zip(&mut values) {
            let proposed = row_mix(row, combo, state, &actions);
            let gain = proposed
                .iter()
                .zip(&baseline)
                .zip(&q)
                .map(|((candidate, control), q)| (candidate - control) * q)
                .sum();
            accumulator.observe(gain, &CoverageCounter::default());
        }
    }
    values
}

#[test]
fn conditional_root_samples_preserve_public_cards_ranges_and_determinism() {
    let board = [0, 5, 10];
    let mut ranges = [vec![0.0; 1326], vec![0.0; 1326]];
    ranges[0][Combo::new(51, 50).key()] = 0.3;
    ranges[0][Combo::new(47, 46).key()] = 0.7;
    ranges[1][Combo::new(51, 49).key()] = 0.6;
    ranges[1][Combo::new(43, 42).key()] = 0.4;
    let mut a = SplitMix64::new(91);
    let mut b = SplitMix64::new(91);
    for _ in 0..1024 {
        let first = conditional_deal(&board, &ranges, &mut a);
        let second = conditional_deal(&board, &ranges, &mut b);
        assert_eq!(first.board, second.board);
        assert_eq!(first.holes, second.holes);
        assert_eq!(&first.board[..3], &board);
        assert_eq!(
            first
                .holes
                .iter()
                .flatten()
                .chain(first.board.iter())
                .collect::<BTreeSet<_>>()
                .len(),
            9
        );
        for seat in 0..2 {
            assert!(
                ranges[seat][Combo::new(first.holes[seat][0], first.holes[seat][1]).key()] > 0.0
            );
        }
    }
}

#[test]
fn identical_root_proposals_have_zero_paired_gain_and_public_range_replay() {
    let (base, deal) = super::tests::tabular_fixture();
    let game = &base.table.config;
    let mut root = GameState::initial(game);
    while root.street != Street::Flop {
        let action = root
            .legal_actions(game)
            .into_iter()
            .find(|a| matches!(a.kind, ActionKind::Call | ActionKind::Check))
            .unwrap();
        root = root.apply(&action, game);
    }
    let board = &deal.board[..3];
    let ranges = public_ranges(&base, &root, board).unwrap();
    assert_eq!(ranges, public_ranges(&base, &root, board).unwrap());
    let actions = root.legal_actions(game);
    let mut row = PublicBeliefStrategy {
        public_history: root.public_history.clone(),
        actor: root.actor,
        action_labels: actions.iter().map(|a| a.label.clone()).collect(),
        probabilities: vec![0.0; 1326 * actions.len()],
        action_values_bb: None,
    };
    for combo in all_combos() {
        if ranges[root.actor][combo.key()] == 0.0 {
            continue;
        }
        let synthetic = deal_for_policy_combo_on_board(combo, root.actor, board).unwrap();
        for (a, p) in base
            .frozen_strategy(&root, &synthetic, &actions, game)
            .iter()
            .enumerate()
        {
            row.probabilities[combo.key() * actions.len() + a] = *p as f32;
        }
    }
    let values = compare(
        &base,
        game,
        &root,
        board,
        &ranges,
        &[row.clone(), row],
        92,
        32,
    );
    assert!(values
        .iter()
        .all(|v| v.mean().abs() < 1e-7 && v.standard_error() < 1e-7));
    assert_eq!(values[0].sum, values[1].sum);
}

#[test]
#[ignore = "requires an explicitly supplied frozen 800-round checkpoint and an external resource guard"]
fn paired_root_screen() {
    let path = PathBuf::from(
        std::env::var("POKER_FLOP_PILOT_CHECKPOINT").expect("explicit checkpoint path"),
    );
    let table = Arc::new(InferenceTable::read(&path).unwrap());
    assert_eq!(table.rounds, 800);
    assert_eq!(table.config.effective_stack_bb, 20.0);
    let game = table.config.clone();
    let base = TabularResponsePolicy {
        table,
        coverage: RefCell::default(),
        flop_patch: Some(Arc::new(flop::FlopPatch::terminal(&TerminalFlopOptions {
            equity_samples: 2048,
            weight: 0.5,
        }))),
        flop_backoff: None,
        completion_coverage: RefCell::default(),
    };
    let policy = turn::TabularTurnPolicy::new(
        base.isolated_copy(),
        TurnResolveOptions {
            iterations: 4,
            safe_bilateral: false,
            maximum_policy_rows: 20000,
        },
    );
    let mut rng = SplitMix64::new(84003);
    let mut roots: [Option<(GameState, Vec<u8>)>; 2] = [None, None];
    for _ in 0..10000 {
        let deal = Deal::sample(&mut rng);
        let mut state = GameState::initial(&game);
        while state.terminal.is_none() && matches!(state.street, Street::Preflop | Street::Flop) {
            let actions = state.legal_actions(&game);
            if state.street == Street::Flop
                && roots[state.actor].is_none()
                && actions
                    .iter()
                    .any(|a| state.apply(a, &game).terminal.is_none())
            {
                roots[state.actor] = Some((state.clone(), deal.board[..3].to_vec()));
            }
            let mix = base.frozen_strategy(&state, &deal, &actions, &game);
            state = state.apply(&actions[sample_index(&mix, &mut rng)], &game);
        }
        if roots.iter().all(Option::is_some) {
            break;
        }
    }
    assert!(
        roots.iter().all(Option::is_some),
        "authentic trajectory root collection failed"
    );
    for (seat, root) in roots.into_iter().enumerate() {
        let (state, board) = root.unwrap();
        let ranges = public_ranges(&base, &state, &board).unwrap();
        let public = PublicBeliefState::from_game_state(board.clone(), &state, ranges.clone());
        let mut proposals = Vec::new();
        for seed in [84001, 84002] {
            let start = Instant::now();
            let solution = sampled_flop::solve(sampled_flop::SampledFlopConfig {
                game: game.clone(),
                state: public.clone(),
                iterations: 32,
                seed,
                maximum_information_sets: 2_000_000,
            })
            .unwrap();
            println!(
                "{}",
                serde_json::json!({ "stage": "proposal", "seat": seat,
                    "seed": seed, "seconds": start.elapsed().as_secs_f64(),
                    "informationSets": solution.information_sets, "inputSha256": solution.input_sha256,
                    "trainedRootCombos": solution.trained_root_combos,
                    "policySha256": format!("{:x}", Sha256::digest(serde_json::to_vec(&solution.root).unwrap())),
                })
            );
            proposals.push(solution.root);
        }
        let start = Instant::now();
        let values = compare(
            &policy,
            &game,
            &state,
            &board,
            &ranges,
            &proposals,
            84004 + seat as u64,
            128,
        );
        println!(
            "{}",
            serde_json::json!({ "stage": "paired_conditional_root_payoff", "seat": seat,
                "board": board, "history": state.public_history, "potBb": state.pot(),
                "evaluationSamples": 128, "evaluationSeed": 84004 + seat as u64,
                "seconds": start.elapsed().as_secs_f64(), "validation": "research_only_not_full_game_exploitability",
                "proposals": values.iter().enumerate().map(|(i,v)| serde_json::json!({
                    "trainingSeed": 84001 + i, "gainBb": v.mean(), "standardErrorBb": v.standard_error(),
                    "individualNormalApprox99LowerBb": v.mean() - 2.5758293035489004 * v.standard_error(),
                    "individualNormalApprox99UpperBb": v.mean() + 2.5758293035489004 * v.standard_error(),
                })).collect::<Vec<_>>(),
            })
        );
    }
}

// Build fresh development samples once. All paths are mandatory and files use
// create_new: an interrupted or completed artifact can never be overwritten.
#[test]
#[ignore = "explicit frozen checkpoint, new output directory and external resource guard required"]
fn build_root_action_caches() {
    use std::io::Write;
    let path =
        PathBuf::from(std::env::var("POKER_FLOP_PILOT_CHECKPOINT").expect("explicit checkpoint"));
    let output = PathBuf::from(
        std::env::var("POKER_FLOP_PILOT_OUTPUT_DIR").expect("explicit output directory"),
    );
    assert!(output.is_dir());
    let digest = sha256_file(&path).unwrap();
    let table = Arc::new(InferenceTable::read(&path).unwrap());
    assert_eq!(table.rounds, 800);
    assert_eq!(table.config.effective_stack_bb, 20.0);
    let game = table.config.clone();
    let terminal = TerminalFlopOptions {
        equity_samples: 2048,
        weight: 0.5,
    };
    let turns = TurnResolveOptions {
        iterations: 4,
        safe_bilateral: false,
        maximum_policy_rows: 20000,
    };
    let base = TabularResponsePolicy {
        table,
        coverage: RefCell::default(),
        flop_patch: Some(Arc::new(flop::FlopPatch::terminal(&terminal))),
        flop_backoff: None,
        completion_coverage: RefCell::default(),
    };
    let policy = turn::TabularTurnPolicy::new(base.isolated_copy(), turns.clone());
    let mut rng = SplitMix64::new(85003);
    let mut roots: [Vec<(GameState, Vec<u8>)>; 4] = std::array::from_fn(|_| Vec::new());
    let mut boards: [BTreeSet<Vec<u8>>; 4] = std::array::from_fn(|_| BTreeSet::new());
    for _ in 0..10000 {
        let deal = Deal::sample(&mut rng);
        let mut state = GameState::initial(&game);
        while state.terminal.is_none() && matches!(state.street, Street::Preflop | Street::Flop) {
            let actions = state.legal_actions(&game);
            let group = state.actor * 2 + usize::from(state.to_call() > 0.0);
            let mut board_key = deal.board[..3].to_vec();
            board_key.sort_unstable();
            if state.street == Street::Flop
                && roots[group].len() < 2
                && !boards[group].contains(&board_key)
                && actions
                    .iter()
                    .any(|a| state.apply(a, &game).terminal.is_none())
            {
                boards[group].insert(board_key);
                roots[group].push((state.clone(), deal.board[..3].to_vec()));
            }
            let mix = base.frozen_strategy(&state, &deal, &actions, &game);
            state = state.apply(&actions[sample_index(&mix, &mut rng)], &game);
        }
        if roots.iter().all(|group| group.len() == 2) {
            break;
        }
    }
    assert!(
        roots.iter().all(|group| group.len() == 2),
        "incomplete authentic root strata"
    );
    for (group, roots) in roots.into_iter().enumerate() {
        for (index, (root, board)) in roots.into_iter().enumerate() {
            let ranges = public_ranges(&base, &root, &board).unwrap();
            let context = cache::Context {
                checkpoint_sha256: digest.clone(),
                game: game.clone(),
                public: PublicBeliefState::from_game_state(board.clone(), &root, ranges),
                turn_resolver: turns.clone(),
                terminal_flop: terminal.clone(),
                evaluation_seed: 85004 + group as u64 * 100 + index as u64,
            };
            let start = Instant::now();
            let (data, _) = cache::RootActionCache::collect_grouped(&policy, context, 128).unwrap();
            let bytes = data.encode().unwrap();
            assert_eq!(data, cache::RootActionCache::decode(&bytes).unwrap());
            let name = format!("root-{group}-{index}.msgpack");
            let mut file = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(output.join(&name))
                .unwrap();
            file.write_all(&bytes).unwrap();
            file.sync_all().unwrap();
            println!(
                "{}",
                serde_json::json!({ "stage": "root_cache", "file": name,
                    "sha256": format!("{:x}", Sha256::digest(&bytes)), "bytes": bytes.len(),
                    "seat": root.actor, "facingBet": root.to_call() > 0.0, "group": group, "index": index,
                    "board": board, "history": root.public_history, "potBb": root.pot(),
                    "samples": 128, "seconds": start.elapsed().as_secs_f64(),
                    "rootCollectionSeed": 85003, "evaluationSeed": data.context.evaluation_seed,
                    "checkpointSha256": digest, "validation": "reusable_development_samples_not_fresh_validation",
                })
            );
        }
    }
}

fn score_summary(value: &ValueAccumulator) -> serde_json::Value {
    serde_json::json!({ "meanBb": value.mean(), "standardErrorBb": value.standard_error(),
        "normalApproxIndividual99LowerBb": value.mean() - 2.5758293035489004 * value.standard_error(),
        "normalApproxIndividual99UpperBb": value.mean() + 2.5758293035489004 * value.standard_error() })
}

#[test]
#[ignore = "explicit frozen root cache and new output directory; no checkpoint loading or new evaluation"]
fn compare_cached_short_pair() {
    use std::io::Write;
    let path = PathBuf::from(std::env::var("POKER_FLOP_PILOT_CACHE").expect("explicit cache"));
    let output = PathBuf::from(
        std::env::var("POKER_FLOP_PILOT_OUTPUT_DIR").expect("explicit output directory"),
    );
    assert!(output.is_dir());
    let bytes = fs::read(&path).unwrap();
    let data = cache::RootActionCache::decode(&bytes).unwrap();
    let cache_sha = format!("{:x}", Sha256::digest(&bytes));
    for seed in [85001, 85002] {
        let mut rows = Vec::new();
        let mut descriptions = Vec::new();
        for iterations in [32, 128] {
            let start = Instant::now();
            let result = sampled_flop::solve(sampled_flop::SampledFlopConfig {
                game: data.context.game.clone(),
                state: data.context.public.clone(),
                iterations,
                seed,
                maximum_information_sets: 2_000_000,
            });
            match result {
                Ok(solution) => {
                    let encoded = rmp_serde::to_vec_named(&solution).unwrap();
                    let name = format!("seed{seed}-round{iterations}.msgpack");
                    let mut file = fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(output.join(&name))
                        .unwrap();
                    file.write_all(&encoded).unwrap();
                    file.sync_all().unwrap();
                    descriptions.push(serde_json::json!({ "iterations": iterations,
                        "seconds": start.elapsed().as_secs_f64(), "informationSets": solution.information_sets,
                        "trainedRootCombos": solution.trained_root_combos,
                        "file": name, "sha256": format!("{:x}", Sha256::digest(&encoded)) }));
                    rows.push(solution.root);
                }
                Err(error) => {
                    println!(
                        "{}",
                        serde_json::json!({ "stage": "proposal_failed", "seed": seed,
                        "iterations": iterations, "error": error, "cacheSha256": cache_sha })
                    );
                    break;
                }
            }
        }
        if rows.len() != 2 {
            continue;
        }
        let (gains, differences) = data.score(&rows).unwrap();
        println!(
            "{}",
            serde_json::json!({ "stage": "cached_short_pair", "seed": seed,
            "cacheSha256": cache_sha, "proposals": descriptions,
            "gainOverFrozenContinuation": gains.iter().map(score_summary).collect::<Vec<_>>(),
            "paired128Minus32": score_summary(&differences[1]),
            "validation": "conditional_development_payoff_not_full_game_exploitability" })
        );
    }
}

#[test]
#[ignore = "full-size frozen-cache parity/cost check; explicit inputs and external guard required"]
fn verify_grouped_frozen_cache() {
    let path =
        PathBuf::from(std::env::var("POKER_FLOP_PILOT_CHECKPOINT").expect("explicit checkpoint"));
    let cache_path =
        PathBuf::from(std::env::var("POKER_FLOP_PILOT_CACHE").expect("explicit cache"));
    let bytes = fs::read(&cache_path).unwrap();
    let expected = cache::RootActionCache::decode(&bytes).unwrap();
    assert_eq!(
        sha256_file(&path).unwrap(),
        expected.context.checkpoint_sha256
    );
    let table = Arc::new(InferenceTable::read(&path).unwrap());
    assert_eq!(table.config, expected.context.game);
    let base = TabularResponsePolicy {
        table,
        coverage: RefCell::default(),
        flop_patch: Some(Arc::new(flop::FlopPatch::terminal(
            &expected.context.terminal_flop,
        ))),
        flop_backoff: None,
        completion_coverage: RefCell::default(),
    };
    let policy = turn::TabularTurnPolicy::new(base, expected.context.turn_resolver.clone());
    let start = Instant::now();
    let (grouped, unique) = cache::RootActionCache::collect_grouped(
        &policy,
        expected.context.clone(),
        expected.sample_count(),
    )
    .unwrap();
    let seconds = start.elapsed().as_secs_f64();
    assert_eq!(expected, grouped);
    assert_eq!(bytes, grouped.encode().unwrap());
    let diagnostics = policy.take_resolution_diagnostics().unwrap();
    assert_eq!(diagnostics["solved_roots"].as_u64().unwrap(), unique as u64);
    println!(
        "{}",
        serde_json::json!({ "stage": "grouped_cache_parity", "byteIdentical": true,
        "cacheSha256": format!("{:x}", Sha256::digest(&bytes)), "seconds": seconds,
        "uniqueTurnRoots": unique, "resolutionDiagnostics": diagnostics })
    );
}

#[test]
#[ignore = "fresh conditional recheck of frozen proposals; explicit inputs and external guard required"]
fn recheck_frozen_proposals() {
    use std::io::Write;
    let path =
        PathBuf::from(std::env::var("POKER_FLOP_PILOT_CHECKPOINT").expect("explicit checkpoint"));
    let cache_path =
        PathBuf::from(std::env::var("POKER_FLOP_PILOT_CACHE").expect("explicit cache"));
    let proposals = PathBuf::from(
        std::env::var("POKER_FLOP_PILOT_POLICY_DIR").expect("explicit frozen policy directory"),
    );
    let output = PathBuf::from(
        std::env::var("POKER_FLOP_PILOT_OUTPUT_DIR").expect("explicit new output directory"),
    );
    assert!(output.is_dir());
    let original = cache::RootActionCache::decode(&fs::read(&cache_path).unwrap()).unwrap();
    assert_eq!(
        sha256_file(&path).unwrap(),
        original.context.checkpoint_sha256
    );
    let mut context = original.context.clone();
    context.evaluation_seed = std::env::var("POKER_FLOP_PILOT_EVAL_SEED")
        .expect("explicit fresh evaluation seed")
        .parse()
        .unwrap();
    assert_ne!(context.evaluation_seed, original.context.evaluation_seed);
    let samples: u64 = std::env::var("POKER_FLOP_PILOT_EVAL_SAMPLES")
        .expect("explicit evaluation sample count")
        .parse()
        .unwrap();
    assert!((2..=4096).contains(&samples));
    let mut rows = Vec::new();
    let mut identities = Vec::new();
    for seed in [85001, 85002] {
        for iterations in [32, 128] {
            let name = format!("seed{seed}-round{iterations}.msgpack");
            let encoded = fs::read(proposals.join(&name)).unwrap();
            let solution: sampled_flop::SampledFlopSolution =
                rmp_serde::from_slice(&encoded).unwrap();
            assert_eq!(solution.seed, seed);
            assert_eq!(solution.iterations, iterations);
            identities.push(serde_json::json!({ "file": name,
                "sha256": format!("{:x}", Sha256::digest(&encoded)) }));
            rows.push(solution.root);
        }
    }
    // Validate all supplied root rows before loading the large checkpoint.
    original.score(&rows).unwrap();
    let table = Arc::new(InferenceTable::read(&path).unwrap());
    assert_eq!(table.config, context.game);
    let base = TabularResponsePolicy {
        table,
        coverage: RefCell::default(),
        flop_patch: Some(Arc::new(flop::FlopPatch::terminal(&context.terminal_flop))),
        flop_backoff: None,
        completion_coverage: RefCell::default(),
    };
    let policy = turn::TabularTurnPolicy::new(base, context.turn_resolver.clone());
    let start = Instant::now();
    let (fresh, unique) =
        cache::RootActionCache::collect_grouped(&policy, context, samples).unwrap();
    let encoded = fresh.encode().unwrap();
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output.join("fresh-cache.msgpack"))
        .unwrap();
    file.write_all(&encoded).unwrap();
    file.sync_all().unwrap();
    let comparisons: Vec<_> = rows
        .chunks_exact(2)
        .zip([85001, 85002])
        .map(|(pair, seed)| {
            let (gains, differences) = fresh.score(pair).unwrap();
            serde_json::json!({ "seed": seed,
                "gainOverFrozenContinuation": gains.iter().map(score_summary).collect::<Vec<_>>(),
                "paired128Minus32": score_summary(&differences[1]) })
        })
        .collect();
    println!(
        "{}",
        serde_json::json!({ "stage": "frozen_proposal_recheck", "samples": samples,
            "evaluationSeed": fresh.context.evaluation_seed, "seconds": start.elapsed().as_secs_f64(),
            "freshCacheSha256": format!("{:x}", Sha256::digest(&encoded)),
            "originalCacheSha256": sha256_file(&cache_path).unwrap(),
            "checkpointSha256": fresh.context.checkpoint_sha256,
            "proposals": identities, "comparisons": comparisons, "uniqueTurnRoots": unique,
            "validation": "fresh_samples_at_selected_development_root_not_full_game_validation" })
    );
}

#[test]
#[ignore = "full-hand routing pilot; explicit checkpoint, hand budget and external resource guard required"]
fn full_hand_routed_pair() {
    let path =
        PathBuf::from(std::env::var("POKER_FLOP_PILOT_CHECKPOINT").expect("explicit checkpoint"));
    let hands: u64 = std::env::var("POKER_FLOP_PILOT_FULL_HANDS")
        .expect("explicit full-hand budget per seat")
        .parse()
        .unwrap();
    assert!((16..=256).contains(&hands));
    let digest = sha256_file(&path).unwrap();
    let table = Arc::new(InferenceTable::read(&path).unwrap());
    assert_eq!(table.rounds, 800);
    assert_eq!(table.config.effective_stack_bb, 20.0);
    let game = table.config.clone();
    let terminal = TerminalFlopOptions {
        equity_samples: 2048,
        weight: 0.5,
    };
    let turns = TurnResolveOptions {
        iterations: 4,
        safe_bilateral: false,
        maximum_policy_rows: 20000,
    };
    let control_base = TabularResponsePolicy {
        table,
        coverage: RefCell::default(),
        flop_patch: Some(Arc::new(flop::FlopPatch::terminal(&terminal))),
        flop_backoff: None,
        completion_coverage: RefCell::default(),
    };
    let control = turn::TabularTurnPolicy::new(control_base.isolated_copy(), turns.clone());
    fn play(
        profiles: [&dyn ResponsePolicy; 2],
        deal: &Deal,
        game: &BlueprintConfig,
        mut rng: SplitMix64,
        visits: &mut [u64; 4],
    ) -> f64 {
        let mut state = GameState::initial(game);
        while state.terminal.is_none() {
            visits[match state.street {
                Street::Preflop => 0,
                Street::Flop => 1,
                Street::Turn => 2,
                Street::River => 3,
            }] += 1;
            let actions = state.legal_actions(game);
            let mix = profiles[state.actor].strategy(&state, deal, &actions, game);
            state = state.apply(&actions[sample_index(&mix, &mut rng)], game);
        }
        realized_utility_p0(&state, deal)
    }
    for seed in [87001, 87002] {
        let mut patch = flop::FlopPatch::terminal(&terminal);
        patch.sampled = Some(super::sampled_flop_policy::FlopResolve::new(
            32, seed, 2_000_000,
        ));
        let patch = Arc::new(patch);
        let mut candidate_base = control_base.isolated_copy();
        candidate_base.flop_patch = Some(patch.clone());
        let candidate = turn::TabularTurnPolicy::new(candidate_base, turns.clone());
        for seat in 0..2 {
            let start = Instant::now();
            let mut chance = SplitMix64::new(derived_seed(87004, 0x6465616c, seat as u64));
            let mut gains = ValueAccumulator::default();
            let mut visits = [0; 4];
            for index in 0..hands {
                let deal = Deal::sample(&mut chance);
                let action_seed = derived_seed(87004, seat as u64, index);
                let old = play(
                    [&control, &control],
                    &deal,
                    &game,
                    SplitMix64::new(action_seed),
                    &mut [0; 4],
                );
                let profiles: [&dyn ResponsePolicy; 2] = if seat == 0 {
                    [&candidate, &control]
                } else {
                    [&control, &candidate]
                };
                let new = play(
                    profiles,
                    &deal,
                    &game,
                    SplitMix64::new(action_seed),
                    &mut visits,
                );
                gains.observe(
                    if seat == 0 { new - old } else { old - new },
                    &CoverageCounter::default(),
                );
                if (index + 1) % 16 == 0 {
                    println!(
                        "{}",
                        serde_json::json!({ "stage": "full_hand_progress", "seed": seed,
                        "seat": seat, "hands": index + 1, "seconds": start.elapsed().as_secs_f64(),
                        "flopDiagnostics": patch.sampled.as_ref().unwrap().diagnostics() })
                    );
                }
            }
            println!(
                "{}",
                serde_json::json!({ "stage": "full_hand_routed_pair", "checkpointSha256": digest,
                "seed": seed, "seat": seat, "hands": hands, "evaluationSeed": 87004,
                "pairedPayoffGainBbPerHand": score_summary(&gains), "candidateStreetDecisionVisits": visits,
                "seconds": start.elapsed().as_secs_f64(), "flopDiagnostics": patch.sampled.as_ref().unwrap().diagnostics(),
                "turnDiagnostics": candidate.take_resolution_diagnostics(),
                "validation": "full_hand_fixed_opponent_development_payoff_not_exploitability" })
            );
        }
    }
}
