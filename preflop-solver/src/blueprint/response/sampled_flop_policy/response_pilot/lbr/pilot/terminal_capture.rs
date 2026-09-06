//! Development trace capture/replay. No policy or release-rule change.
//! Only this saved terminal decision is assessed, not full-game exploitability.
use super::*;
use crate::blueprint::response::table::AverageNode;

const SOURCE_B: &str = "7a71a1b13975173af32aed3c610ae62a7c0fb25680ddf4397efef0bde220893d";
const OWN: [u8; 2] = [34, 21];
const BOARD: [u8; 3] = [27, 2, 9];
const LINE: [&str; 3] = ["limp", "check", "bet_all_in_to_19.000bb"];
const FIXTURE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/terminal-flop-defender-b.msgpack"
));

#[derive(Deserialize, Serialize)]
struct SavedNode {
    key: u64,
    descriptor: NodeDescriptor,
    action_labels: Vec<String>,
    strategy_sum: Vec<f64>,
    average_visits: u64,
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
struct Measurement {
    mix: Vec<f64>,
    opponent_posterior: Vec<f64>,
    exact_equity: f64,
    exact_fold_ev_bb: f64,
    exact_call_ev_bb: f64,
    exact_policy_ev_bb: f64,
    exact_conditional_loss_bb: f64,
    showdown_evaluations: u64,
    sampled_lbr_values_bb: Vec<f64>,
}

#[derive(Deserialize, Serialize)]
struct Capture {
    schema: String,
    checkpoint_sha256: String,
    game: BlueprintConfig,
    rounds: u64,
    nodes: Vec<SavedNode>,
    expected: Measurement,
}

fn root(game: &BlueprintConfig) -> GameState {
    let mut state = GameState::initial(game);
    for label in LINE {
        let action = state
            .legal_actions(game)
            .into_iter()
            .find(|action| action.label == label)
            .expect("captured action must be legal");
        state = state.apply(&action, game);
    }
    assert_eq!(state.actor, 0);
    assert_eq!(state.street, Street::Flop);
    state
}

fn measure(table: Arc<InferenceTable>) -> Measurement {
    measure_with_options(table, None)
}

fn measure_with_options(
    table: Arc<InferenceTable>,
    terminal: Option<&TerminalFlopOptions>,
) -> Measurement {
    let game = table.config.clone();
    let (policy, _) = match terminal {
        Some(options) => profile_with_terminal_options(table, 87001, 64, options),
        None => profile_with_turn_iterations(table, 87001, 64),
    };
    let mut state = GameState::initial(&game);
    let mut belief = Belief::new(0, OWN).unwrap();
    for label in LINE {
        let board = &BOARD[..state.street.board_len()];
        belief.reveal(board).unwrap();
        let actions = state.legal_actions(&game);
        let index = actions
            .iter()
            .position(|action| action.label == label)
            .unwrap();
        if state.actor != belief.seat {
            belief
                .observe(&policy, &game, &state, board, &actions, index)
                .unwrap();
        }
        state = state.apply(&actions[index], &game);
    }
    belief.reveal(&BOARD).unwrap();
    assert_eq!(state.public_history, root(&game).public_history);
    let actions = state.legal_actions(&game);
    assert_eq!(actions.len(), 2);
    let fold = actions
        .iter()
        .position(|a| a.kind == ActionKind::Fold)
        .unwrap();
    let call = actions
        .iter()
        .position(|a| a.kind == ActionKind::Call)
        .unwrap();
    assert!(actions
        .iter()
        .all(|a| state.apply(a, &game).terminal.is_some()));
    let mix = query(
        &policy,
        &game,
        &state,
        &BOARD,
        &actions,
        Combo::new(OWN[0], OWN[1]),
    )
    .unwrap();
    let sampled_lbr_values_bb = Lbr {
        seed: 90001,
        early_runouts_per_combo: 16,
    }
    .values(&belief, &policy, &game, &state, &BOARD, &actions)
    .unwrap();
    let mut exact_equity = 0.0;
    let mut showdown_evaluations = 0;
    for combo in all_combos() {
        let weight = belief.weights[combo.key()];
        if weight == 0.0 {
            continue;
        }
        let opponent = combo.cards();
        let deck = (0..52u8)
            .filter(|card| !OWN.contains(card) && !opponent.contains(card) && !BOARD.contains(card))
            .collect::<Vec<_>>();
        assert_eq!(deck.len(), 45);
        let mut sum = 0.0;
        for (i, turn) in deck.iter().enumerate() {
            for river in &deck[i + 1..] {
                sum += showdown_result(
                    &[OWN, opponent],
                    &[BOARD[0], BOARD[1], BOARD[2], *turn, *river],
                );
                showdown_evaluations += 1;
            }
        }
        // Once the call ends betting, unordered remaining cards have identical
        // showdown payoffs; enumerate all C(45,2)=990, with exact card removal.
        exact_equity += weight * (sum / 990.0);
    }
    let called = state.apply(&actions[call], &game);
    let exact_fold_ev_bb = -state.invested[0];
    let exact_call_ev_bb = exact_equity * called.pot() - called.invested[0];
    let exact_policy_ev_bb = mix[fold] * exact_fold_ev_bb + mix[call] * exact_call_ev_bb;
    Measurement {
        mix,
        opponent_posterior: belief.weights,
        exact_equity,
        exact_fold_ev_bb,
        exact_call_ev_bb,
        exact_policy_ev_bb,
        exact_conditional_loss_bb: exact_fold_ev_bb.max(exact_call_ev_bb) - exact_policy_ev_bb,
        showdown_evaluations,
        sampled_lbr_values_bb,
    }
}

fn compact_table(capture: &Capture) -> Arc<InferenceTable> {
    assert_eq!(capture.schema, "development-terminal-flop-replay-v1");
    assert_eq!(capture.checkpoint_sha256, SOURCE_B);
    let nodes = capture
        .nodes
        .iter()
        .map(|node| {
            (
                node.key,
                AverageNode {
                    descriptor: node.descriptor.clone(),
                    action_labels: node
                        .action_labels
                        .iter()
                        .map(|s| Arc::<str>::from(s.as_str()))
                        .collect::<Vec<_>>()
                        .into(),
                    strategy_sum: node.strategy_sum.clone().into_boxed_slice(),
                    average_visits: node.average_visits,
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(nodes.len(), capture.nodes.len());
    Arc::new(InferenceTable {
        config: capture.game.clone(),
        rounds: capture.rounds,
        nodes,
    })
}

fn output_path() -> PathBuf {
    PathBuf::from(
        std::env::var("POKER_TERMINAL_CAPTURE_DIR").expect("explicit development directory"),
    )
    .join("terminal-case.msgpack")
}

#[test]
#[ignore = "one full-size trace extraction; explicit frozen source/output and external guard required"]
fn capture_delayed_terminal_flop_case() {
    let source = PathBuf::from(std::env::var("POKER_FLOP_PILOT_CHECKPOINT").unwrap());
    assert_eq!(sha256_file(&source).unwrap(), SOURCE_B);
    let path = output_path();
    assert!(!path.exists(), "never overwrite captured evidence");
    let table = Arc::new(InferenceTable::read(&source).unwrap());
    assert_eq!(table.rounds, 800);
    assert_eq!(table.config.effective_stack_bb, 20.0);
    let expected = measure(table.clone());
    // Bind this replay to the actual saved delayed-pilot decision, not a
    // merely similar board. These were recorded before this diagnostic existed.
    assert_eq!(
        expected.sampled_lbr_values_bb,
        vec![-1.0, -9.78626961397315]
    );
    let mut needed = BTreeSet::new();
    let mut state = GameState::initial(&table.config);
    for label in LINE {
        if state.street == Street::Preflop {
            for combo in all_combos() {
                let deal = deal_for_policy_combo_on_board(combo, state.actor, &[]).unwrap();
                needed.insert(information_set(&state, &deal, &table.config).0);
            }
        }
        let action = state
            .legal_actions(&table.config)
            .into_iter()
            .find(|a| a.label == label)
            .unwrap();
        state = state.apply(&action, &table.config);
    }
    let deal = deal_for_policy_combo_on_board(Combo::new(OWN[0], OWN[1]), 0, &BOARD).unwrap();
    needed.insert(information_set(&state, &deal, &table.config).0);
    let capture = Capture {
        schema: "development-terminal-flop-replay-v1".into(),
        checkpoint_sha256: SOURCE_B.into(),
        game: table.config.clone(),
        rounds: table.rounds,
        nodes: needed
            .into_iter()
            .filter_map(|key| {
                table.nodes.get(&key).map(|node| SavedNode {
                    key,
                    descriptor: node.descriptor.clone(),
                    action_labels: node.action_labels.iter().map(|s| s.to_string()).collect(),
                    strategy_sum: node.strategy_sum.to_vec(),
                    average_visits: node.average_visits,
                })
            })
            .collect(),
        expected,
    };
    drop(table); // one full checkpoint at a time; compact replay owns only its subset
    let encoded = rmp_serde::to_vec_named(&capture).unwrap();
    let decoded: Capture = rmp_serde::from_slice(&encoded).unwrap();
    let actual = measure(compact_table(&decoded));
    assert_eq!(
        actual, capture.expected,
        "minimization changed the real policy path"
    );
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    file.write_all(&encoded).unwrap();
    file.sync_all().unwrap();
    println!(
        "{}",
        serde_json::json!({"stage":"terminal_capture", "path":path,
        "sha256":sha256_file(&path).unwrap(), "bytes":encoded.len(), "retainedNodes":capture.nodes.len(),
        "mix":actual.mix, "exactCallEvBb":actual.exact_call_ev_bb,
        "exactFoldEvBb":actual.exact_fold_ev_bb, "exactConditionalLossBb":actual.exact_conditional_loss_bb,
        "showdownEvaluations":actual.showdown_evaluations,
        "interpretation":"conditional terminal decision against the frozen public posterior; not full-game exploitability"})
    );
}

#[test]
#[ignore = "development red/green policy-action diagnostic; requires the explicit captured fixture"]
fn replay_terminal_flop_conditional_loss() {
    let capture: Capture = rmp_serde::from_slice(&fs::read(output_path()).unwrap()).unwrap();
    let started = Instant::now();
    let terminal = std::env::var("POKER_TERMINAL_WEIGHT")
        .ok()
        .map(|weight| TerminalFlopOptions {
            equity_samples: 2048,
            weight: weight.parse().expect("numeric development weight"),
        });
    let actual = measure_with_options(compact_table(&capture), terminal.as_ref());
    assert_eq!(
        actual.opponent_posterior,
        capture.expected.opponent_posterior
    );
    assert_eq!(actual.exact_call_ev_bb, capture.expected.exact_call_ev_bb);
    assert_eq!(actual.exact_fold_ev_bb, capture.expected.exact_fold_ev_bb);
    println!(
        "{}",
        serde_json::json!({"stage":"terminal_replay", "seconds":started.elapsed().as_secs_f64(),
        "terminalOptions":terminal, "mix":actual.mix, "exactCallEvBb":actual.exact_call_ev_bb,
        "exactFoldEvBb":actual.exact_fold_ev_bb, "exactConditionalLossBb":actual.exact_conditional_loss_bb})
    );
    // Existing decision-quality tolerance, not a new full-game release gate.
    assert!(
        actual.exact_conditional_loss_bb <= 0.05,
        "captured terminal policy loses {}bb in conditional expectation",
        actual.exact_conditional_loss_bb
    );
}

#[test]
#[ignore = "separate controlled probes on the captured development fixture, not a full-game policy comparison"]
fn diagnose_terminal_flop_capture() {
    let capture: Capture = rmp_serde::from_slice(&fs::read(output_path()).unwrap()).unwrap();
    for (samples, weight) in [(128, 0.5), (2048, 0.5), (16384, 0.5), (2048, 1.0)] {
        let table = compact_table(&capture);
        let options = TerminalFlopOptions {
            equity_samples: samples,
            weight,
        };
        let actual = measure_with_options(table.clone(), Some(&options));
        let (_, patch) = profile_with_terminal_options(table.clone(), 87001, 64, &options);
        let base = TabularResponsePolicy {
            table,
            coverage: RefCell::default(),
            flop_patch: Some(patch),
            flop_backoff: None,
            completion_coverage: RefCell::default(),
        };
        let game = &base.table.config;
        let state = root(game);
        let actions = state.legal_actions(game);
        let deal = deal_for_policy_combo_on_board(Combo::new(OWN[0], OWN[1]), 0, &BOARD).unwrap();
        let selected = crate::blueprint::response::flop_allin::correction(
            &base, &state, &deal, &actions, game, samples,
        );
        let direct = crate::blueprint::response::flop_allin::opponent_range_for_audit(
            &base, &state, OWN, &BOARD, game,
        )
        .unwrap();
        let mut direct_posterior = vec![0.0; 1326];
        for (cards, weight) in direct {
            direct_posterior[Combo::new(cards[0], cards[1]).key()] = weight;
        }
        let maximum_range_error = direct_posterior
            .iter()
            .zip(&actual.opponent_posterior)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(maximum_range_error < 1e-12);
        assert_eq!(
            actual.opponent_posterior,
            capture.expected.opponent_posterior
        );
        assert_eq!(actual.exact_call_ev_bb, capture.expected.exact_call_ev_bb);
        println!(
            "{}",
            serde_json::json!({"stage":"terminal_hypothesis_probe",
            "options":options, "correctionSelected":selected.map(|index| &actions[index].label),
            "maximumPosteriorDifference":maximum_range_error, "mix":actual.mix,
            "exactConditionalLossBb":actual.exact_conditional_loss_bb,
            "interpretation":"one captured development decision; no equilibrium or full-game gain claim"})
        );
    }
}

#[test]
fn full_weight_terminal_candidate_removes_the_captured_bad_call_without_changing_its_range() {
    let capture: Capture = rmp_serde::from_slice(FIXTURE).unwrap();
    let table = compact_table(&capture);
    let state = root(&table.config);
    let deal = deal_for_policy_combo_on_board(Combo::new(OWN[0], OWN[1]), 0, &BOARD).unwrap();
    let key = information_set(&state, &deal, &table.config).0;
    let node = table.nodes.get(&key).unwrap();
    println!(
        "captured terminal row averageVisits={} strategySum={:?}",
        node.average_visits, node.strategy_sum
    );
    let control = measure(table.clone());
    assert_eq!(control, capture.expected);
    assert_eq!(control.mix, [0.75, 0.25]);
    assert!(control.exact_conditional_loss_bb > 2.0);
    let candidate = measure_with_options(
        table,
        Some(&TerminalFlopOptions {
            equity_samples: 2048,
            weight: 1.0,
        }),
    );
    assert_eq!(candidate.opponent_posterior, control.opponent_posterior);
    assert_eq!(candidate.exact_call_ev_bb, control.exact_call_ev_bb);
    assert_eq!(candidate.exact_fold_ev_bb, control.exact_fold_ev_bb);
    assert_eq!(candidate.mix, [1.0, 0.0]);
    assert_eq!(candidate.exact_conditional_loss_bb, 0.0);
}

#[test]
#[ignore = "one guarded full-checkpoint replay of the minimized candidate; not a full-game quality result"]
fn verify_terminal_candidate_on_full_checkpoint() {
    let source = PathBuf::from(std::env::var("POKER_FLOP_PILOT_CHECKPOINT").unwrap());
    assert_eq!(sha256_file(&source).unwrap(), SOURCE_B);
    let capture: Capture = rmp_serde::from_slice(FIXTURE).unwrap();
    let table = Arc::new(InferenceTable::read(&source).unwrap());
    let control = measure(table.clone());
    assert_eq!(control, capture.expected);
    let candidate = measure_with_options(
        table,
        Some(&TerminalFlopOptions {
            equity_samples: 2048,
            weight: 1.0,
        }),
    );
    assert_eq!(candidate.opponent_posterior, control.opponent_posterior);
    assert_eq!(candidate.exact_call_ev_bb, control.exact_call_ev_bb);
    assert_eq!(candidate.exact_conditional_loss_bb, 0.0);
    println!(
        "{}",
        serde_json::json!({"stage":"terminal_candidate_full_replay",
        "sourceSha256":SOURCE_B, "controlMix":control.mix, "candidateMix":candidate.mix,
        "controlConditionalLossBb":control.exact_conditional_loss_bb,
        "candidateConditionalLossBb":candidate.exact_conditional_loss_bb,
        "interpretation":"exact conditional action-value replay, not full-game exploitability"})
    );
}

#[test]
#[ignore = "fresh controlled full-hand terminal-weight pilot; explicit weight/source and resource guard required"]
fn terminal_weight_lbr_pilot() {
    let weight: f64 = std::env::var("POKER_TERMINAL_WEIGHT")
        .unwrap()
        .parse()
        .unwrap();
    assert!([0.5, 1.0].contains(&weight));
    run_challenge_with_terminal_options(
        32,
        96,
        93004,
        Street::Flop,
        TerminalFlopOptions {
            equity_samples: 2048,
            weight,
        },
    );
}
