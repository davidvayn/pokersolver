//! Explicit development pilot: change one flop decision, freeze everything else.
//! This measures conditional root payoff, not a complete policy or exploitability.
use super::*;
use crate::blueprint::neural::{
    deal_for_policy_combo_on_board, normalize_ranges_for_board, trajectory_action_matches,
};
use crate::blueprint::public_belief::{sampled_flop, PublicBeliefState, PublicBeliefStrategy};
use std::time::Instant;

fn public_ranges(
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

fn row_mix(
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
