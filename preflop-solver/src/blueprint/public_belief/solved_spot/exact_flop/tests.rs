use super::*;

fn input() -> SpotInput {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../data/solved-spots/pilots/flop-50bb-exact-small.json"
    )))
    .unwrap()
}

fn guard() -> Guard {
    Guard {
        deadline: Instant::now() + Duration::from_secs(120),
        visits: Cell::new(0),
    }
}

#[test]
fn complete_flop_is_deterministic_and_agrees_with_independent_pair_values() {
    let limits = PilotLimits {
        max_decision_nodes: 10_000,
        max_seconds: 120,
    };
    let a = run(input(), limits.clone()).unwrap();
    let b = run(input(), limits).unwrap();
    assert_eq!(a.public_decision_nodes, 9608);
    assert_eq!(a.regret_average_array_bytes, 1_980_160);
    assert_eq!(a.board_contexts, 2402);
    assert_eq!(a.frozen_policy_sha256, b.frozen_policy_sha256);
    assert_eq!(a.profile_value_bb, b.profile_value_bb);
    assert_eq!(a.total_best_response_gain_bb, b.total_best_response_gain_bb);
    assert!(a.maximum_independent_profile_difference_bb < 1e-10);
    assert!(a.total_best_response_gain_bb >= -1e-10);
    assert!(
        !a.serving_artifact_produced && !a.action_ev_validation_completed && !a.full_hand_gto_claim
    );
}

#[test]
fn board_cache_removes_private_cards_without_merging_arrival_memory() {
    let config = config_for_audit(&input(), true).unwrap();
    let game = Game::new(&config, &guard()).unwrap();
    let a = game
        .boards
        .iter()
        .position(|b| b.cards == [43, 30, 16, 0, 1])
        .unwrap();
    let b = game
        .boards
        .iter()
        .position(|b| b.cards == [43, 30, 16, 1, 0])
        .unwrap();
    assert_ne!(a, b);
    assert_eq!(game.boards[a].equity, game.boards[b].equity);
    let node = |board| {
        game.nodes
            .iter()
            .position(|n| n.board == board && matches!(n.kind, Kind::Decision { .. }))
            .unwrap()
    };
    assert_ne!(node(a), node(b));
    let card = game.combos[0][0].cards()[0];
    let turn = game
        .boards
        .iter()
        .find(|b| b.cards == [43, 30, 16, card])
        .unwrap();
    assert!(turn.rows[0][0].is_none());
    assert!(!turn.legal[0].contains(&0));
}

#[test]
fn exact_pilot_rejects_structural_time_and_invalid_probability_inputs() {
    assert!(run(
        input(),
        PilotLimits {
            max_decision_nodes: 4,
            max_seconds: 120
        }
    )
    .unwrap_err()
    .contains("before tree allocation"));
    assert!(run(
        input(),
        PilotLimits {
            max_decision_nodes: 10_000,
            max_seconds: 0
        }
    )
    .is_err());
    let expired = Guard {
        deadline: Instant::now() - Duration::from_secs(1),
        visits: Cell::new(4095),
    };
    assert!(expired.tick().is_err());
    let mut data = input();
    data.board.push(1);
    assert!(run(
        data,
        PilotLimits {
            max_decision_nodes: 10_000,
            max_seconds: 120
        }
    )
    .is_err());
    let config = config_for_audit(&input(), true).unwrap();
    let game = Game::new(&config, &guard()).unwrap();
    let mut frozen = game.freeze().unwrap();
    frozen[game.root][0] = f32::NAN;
    assert!(check::verify(&game, &frozen, &guard()).is_err());
    assert!(game.check_probabilities(&[]).is_err());
}

#[test]
fn information_set_response_cannot_choose_by_hidden_opponent_holding() {
    use crate::cards::Combo;
    use tree::{Board, Node};
    let mut game = Game {
        combos: [
            vec![Combo::new(44, 45)],
            vec![Combo::new(48, 49), Combo::new(40, 41)],
        ],
        prior: [vec![1.0], vec![0.5, 0.5]],
        boards: vec![Board {
            cards: vec![0, 5, 10, 15, 24],
            legal: [vec![0], vec![0, 1]],
            rows: [vec![Some(0)], vec![Some(0), Some(1)]],
            equity: vec![-1.0, 1.0],
        }],
        compatible: vec![true, true],
        root: 2,
        nodes: vec![
            Node {
                board: 0,
                kind: Kind::Terminal {
                    terminal: Terminal::Fold { winner: 1 },
                    invested: [4.0, 8.0],
                },
                regrets: vec![],
                sums: vec![],
            },
            Node {
                board: 0,
                kind: Kind::Terminal {
                    terminal: Terminal::Showdown,
                    invested: [8.0, 8.0],
                },
                regrets: vec![],
                sums: vec![],
            },
            Node {
                board: 0,
                kind: Kind::Decision {
                    actor: 0,
                    children: vec![0, 1],
                },
                regrets: vec![0.0; 2],
                sums: vec![0.0, 1.0],
            },
        ],
    };
    let frozen = game.freeze().unwrap();
    let (profile, best) = check::verify(&game, &frozen, &guard()).unwrap();
    assert_eq!(profile, [0.0, 0.0]);
    assert_eq!(best[0], 0.0); // Clairvoyant fold-vs-AA/call-vs-QQ would get 2bb.
    game.boards[0].equity[0] += 0.5;
    assert!((game.profile_value(&frozen, 0, &guard()).unwrap() - profile[0]).abs() > 1.0);
    assert_eq!(check::verify(&game, &frozen, &guard()).unwrap().0, profile);
}

#[test]
fn compact_tree_matches_existing_river_and_turn_training() {
    fn compare(
        game: &Game,
        frozen: &[Vec<f32>],
        id: usize,
        state: GameState,
        spot: &SolvedSpot,
        turn: bool,
    ) {
        let node = &game.nodes[id];
        match &node.kind {
            Kind::Terminal { .. } => assert!(state.terminal.is_some()),
            Kind::Chance { children, .. } => {
                for (_, child) in children {
                    compare(game, frozen, *child, state.clone(), spot, turn);
                }
            }
            Kind::Decision { actor, children } => {
                let history = node_id(&key(&state, &game.boards[node.board].cards, turn));
                let reference = spot.nodes.iter().find(|n| n.id == history).unwrap();
                assert_eq!(*actor, reference.actor);
                let config = config(&spot.input).unwrap();
                let actions = state.legal_actions(&config.game);
                assert_eq!(children.len(), actions.len());
                for combo in &game.boards[node.board].legal[*actor] {
                    let row = reference
                        .rows
                        .iter()
                        .find(|r| r.combo == game.combos[*actor][*combo].key())
                        .unwrap();
                    let sum: f64 = row.probabilities.iter().map(|p| f64::from(*p)).sum();
                    for action in 0..children.len() {
                        assert!(
                            (game.probability(frozen, id, *combo, action)
                                - f64::from(row.probabilities[action]) / sum)
                                .abs()
                                < 1e-5
                        );
                    }
                }
                for (action, child) in actions.iter().zip(children) {
                    compare(
                        game,
                        frozen,
                        *child,
                        state.apply(action, &config.game),
                        spot,
                        turn,
                    );
                }
            }
        }
    }
    for board in [vec![43, 30, 16, 1], vec![43, 30, 16, 1, 3]] {
        let mut data = input();
        data.board = board;
        data.iterations = 20;
        let config = config(&data).unwrap();
        let guard = guard();
        let mut game = Game::new(&config, &guard).unwrap();
        game.train(data.iterations, &config.game.dcfr, &guard)
            .unwrap();
        let frozen = game.freeze().unwrap();
        let spot = solve_dense_reference(data).unwrap();
        compare(
            &game,
            &frozen,
            game.root,
            config.state.game_state(),
            &spot,
            spot.input.board.len() == 4,
        );
        let (profile, _) = check::verify(&game, &frozen, &guard).unwrap();
        for p in 0..2 {
            assert!((profile[p] - spot.verification.profile_value_bb[p]).abs() < 1e-5);
        }
    }
}
