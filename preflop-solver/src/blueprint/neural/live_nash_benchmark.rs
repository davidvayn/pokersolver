//! Explicit, opt-in diagnostic of the live website, not the solved-spot library.
//! No policy fitting, changed search budget, training-state import or promotion.
use super::*;
use crate::blueprint::public_belief::{self as pbs, live_policy_response as response};
use serde::de::DeserializeOwned;
use std::path::Path;

#[path = "matched_turn.rs"]
mod matched_turn;

#[derive(Clone, Deserialize, Serialize)]
struct FrozenRow {
    state: PublicBeliefState,
    probabilities: Vec<f64>,
}
#[derive(Clone, Deserialize, Serialize)]
struct Node {
    state: PublicBeliefState,
    probabilities: Vec<f64>,
    children: Vec<usize>,
    terminal: Option<response::Values>,
    turn_leaf: bool,
}
#[derive(Deserialize, Serialize)]
struct Base {
    schema: String,
    protocol_sha256: String,
    case: String,
    game: BlueprintConfig,
    root: usize,
    nodes: Vec<Node>,
    frozen_flop_rows: Vec<FrozenRow>,
    maximum_serving_thread_probability_delta: f64,
    maximum_serving_thread_range_delta: f64,
}
#[derive(Deserialize, Serialize)]
struct Packet {
    schema: String,
    base_sha256: String,
    turn: u8,
    leaves: BTreeMap<usize, response::Values>,
    public_nodes: usize,
    maximum_probability_sum_error: f64,
}

fn read<T: DeserializeOwned>(path: &Path) -> T {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}
fn digest(path: &Path) -> String {
    format!("{:x}", Sha256::digest(fs::read(path).unwrap()))
}
fn write(path: &Path, value: &impl Serialize) {
    assert!(
        !path.exists(),
        "refuse to overwrite diagnostic evidence: {}",
        path.display()
    );
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, serde_json::to_vec(value).unwrap()).unwrap();
    fs::rename(temporary, path).unwrap();
}
fn deal(board: &[u8], actor: usize) -> Deal {
    let combo = all_combos()
        .into_iter()
        .find(|c| !c.cards().iter().any(|card| board.contains(card)))
        .unwrap();
    deal_for_policy_combo_on_board(combo, actor, board).unwrap()
}
fn serving_engine(
    root: &Path,
    manifest: &serde_json::Value,
    threads: usize,
) -> PracticePolicyEngine {
    let runtime = &manifest["runtime"];
    let profile = &runtime["resolver"];
    assert_eq!(runtime["kind"], "rust-continual-resolver-v1");
    assert_eq!(manifest["depthsBb"], serde_json::json!([20]));
    assert_eq!(profile["deterministic"], true);
    let mut game = BlueprintConfig::default();
    game.effective_stack_bb = 20.0;
    let mut grid = serde_json::Map::new();
    for (snake, camel) in [
        ("open_sizes_bb", "openSizesBb"),
        ("limp_raise_sizes_bb", "limpRaiseSizesBb"),
        ("three_bet_sizes_bb", "threeBetSizesBb"),
        ("four_bet_sizes_bb", "fourBetSizesBb"),
        ("deeper_raise_pot_fractions", "deeperRaisePotFractions"),
        ("preflop_raise_cap", "preflopRaiseCap"),
        ("flop_bet_pot_fractions", "flopBetPotFractions"),
        ("turn_river_bet_pot_fractions", "turnRiverBetPotFractions"),
        ("postflop_raise_pot_fractions", "postflopRaisePotFractions"),
        ("postflop_raise_cap", "postflopRaiseCap"),
        ("include_all_in", "includeAllIn"),
    ] {
        grid.insert(snake.into(), runtime["actionAbstraction"][camel].clone());
    }
    game.action_abstraction = serde_json::from_value(serde_json::Value::Object(grid)).unwrap();
    game.dcfr = DcfrParameters {
        positive_regret_exponent: runtime["dcfr"]["positiveRegretExponent"].as_f64().unwrap(),
        negative_regret_exponent: runtime["dcfr"]["negativeRegretExponent"].as_f64().unwrap(),
        strategy_exponent: runtime["dcfr"]["strategyExponent"].as_f64().unwrap(),
    };
    let count = |street: &str| profile[format!("{street}Iterations")].as_u64().unwrap();
    let actor = |street: &str| {
        profile[format!("{street}ResolvedActor")]
            .as_u64()
            .map(|n| n as usize)
    };
    let models = root.join("preflop-solver/models/practice");
    let artifact = |key: &str| models.join(runtime["artifactFiles"][key].as_str().unwrap());
    let engine = PracticePolicyEngine::load(PracticePolicyEngineConfig {
        game,
        model_version: manifest["version"].as_str().unwrap().into(),
        network_path: artifact("networks"),
        range_policy_path: artifact("rangePolicy"),
        preflop_action_values_path: Some(artifact("preflopActionValues")),
        flop_resolver: Some(FlopResolverConfig {
            iterations: count("flop"),
            averaging_delay: 0,
            policy_prior_iterations: 0,
            regret_matching_plus: false,
            deploy_solved_policy: profile["flopDeploySolvedPolicy"].as_bool().unwrap(),
            threads,
            value_network_path: artifact("flopValueNetwork"),
            auxiliary_value_network_paths: vec![],
            continuation_selection: pbs::FlopContinuationSelection::Mean,
            safe_resolving: false,
            safe_anchor_iterations: 0,
            safe_bilateral: false,
            resolved_policy_weight: 1.0,
            resolved_actor: actor("flop"),
        }),
        turn_resolver: Some(TurnResolverConfig {
            iterations: count("turn"),
            averaging_delay: 0,
            policy_prior_iterations: 0,
            policy_prior_actor: None,
            river_refinement_iterations: 0,
            regret_matching_plus: false,
            threads,
            safe_resolving: false,
            safe_anchor_iterations: 0,
            resolved_policy_weight: 1.0,
            resolved_actor: actor("turn"),
        }),
        river_resolver: Some(RiverResolverConfig {
            iterations: count("river"),
            averaging_delay: 0,
            policy_prior_iterations: 0,
            policy_prior_actor: None,
            safe_resolving: profile["riverSafeResolving"].as_bool().unwrap(),
            safe_maxmargin: profile["riverSafeMaxmargin"].as_bool().unwrap(),
            safe_iterations: profile["riverSafeIterations"].as_u64(),
            safe_resolved_actor: profile["riverSafeResolvedActor"]
                .as_u64()
                .map(|n| n as usize),
            resolved_actor: actor("river"),
        }),
    })
    .unwrap();
    assert_eq!(
        engine.policy.bundle_sha256(),
        runtime["networkSha256"].as_str().unwrap()
    );
    assert_eq!(
        engine.range_policy_sha256,
        runtime["rangePolicySha256"].as_str().unwrap()
    );
    assert_eq!(
        engine.value_network_sha256.as_deref(),
        runtime["valueNetworkSha256"].as_str()
    );
    assert_eq!(
        engine.preflop_action_values.as_ref().unwrap().sha256,
        runtime["preflopActionValuesSha256"].as_str().unwrap()
    );
    engine
}

fn preflop(game: &BlueprintConfig, labels: &[serde_json::Value]) -> GameState {
    let mut state = GameState::initial(game);
    for label in labels {
        let actions = state.legal_actions(game);
        let action = actions
            .iter()
            .find(|a| a.label == label.as_str().unwrap())
            .unwrap_or_else(|| panic!("missing {label}; legal={actions:?}"));
        state = state.apply(action, game);
    }
    assert_eq!(state.street, Street::Flop);
    assert!(state.terminal.is_none());
    state
}
fn served(
    engine: &PracticePolicyEngine,
    state: &GameState,
    board: &[u8],
) -> Arc<RangePolicyCachedNode> {
    engine
        .policy
        .policy_node(
            state,
            &deal(board, state.actor),
            &state.legal_actions(&engine.game),
            &engine.game,
        )
        .unwrap()
}
fn checkdown_turn(node: &Node) -> bool {
    node.turn_leaf
        && node.state.public_history.ends_with(&[
            "Flop:p1:check".into(),
            "Flop:p0:check".into(),
            "deal:Turn".into(),
        ])
}
fn base(
    engine: &PracticePolicyEngine,
    protocol: &serde_json::Value,
    case: &serde_json::Value,
    protocol_sha256: String,
) -> Base {
    let board: Vec<u8> = serde_json::from_value(case["board"].clone()).unwrap();
    let root = preflop(&engine.game, case["preflop"].as_array().unwrap());
    assert!((root.invested.iter().sum::<f64>() - case["potBb"].as_f64().unwrap()).abs() < 1e-9);
    let ranges = served(engine, &root, &board).ranges.clone();
    let terminals = response::FlopTerminals::new(pbs::FlopResolveConfig {
        game: engine.game.clone(),
        state: PublicBeliefState::from_game_state(board.clone(), &root, ranges.clone()),
        iterations: 2,
        averaging_delay: 0,
        regret_matching_plus: false,
        value_network: engine
            .policy
            .flop_resolver
            .as_ref()
            .unwrap()
            .value_network
            .clone(),
        auxiliary_value_networks: vec![],
        continuation_selection: pbs::FlopContinuationSelection::Mean,
        threads: protocol["threadsPerWorker"].as_u64().unwrap() as usize,
    })
    .unwrap();
    fn capture(
        engine: &PracticePolicyEngine,
        terminals: &response::FlopTerminals,
        state: GameState,
        board: &[u8],
        ranges: [Vec<f64>; 2],
        nodes: &mut Vec<Node>,
        rows: &mut Vec<FrozenRow>,
    ) -> usize {
        let public = PublicBeliefState::from_game_state(board.to_vec(), &state, ranges.clone());
        let terminal = state
            .terminal
            .is_some()
            .then(|| terminals.values(&state, &ranges));
        let turn_leaf = state.terminal.is_none() && state.street == Street::Turn;
        let mut probabilities = vec![];
        let mut children = vec![];
        if terminal.is_none() && !turn_leaf {
            let node = served(engine, &state, board);
            probabilities = node.probabilities.clone();
            rows.push(FrozenRow {
                state: PublicBeliefState::from_game_state(
                    board.to_vec(),
                    &state,
                    node.ranges.clone(),
                ),
                probabilities: probabilities.clone(),
            });
            let actions = state.legal_actions(&engine.game);
            for (a, action) in actions.iter().enumerate() {
                let mut reaches = ranges.clone();
                for combo in 0..COMBO_COUNT {
                    reaches[state.actor][combo] *= probabilities[combo * actions.len() + a];
                }
                children.push(capture(
                    engine,
                    terminals,
                    state.apply(action, &engine.game),
                    board,
                    reaches,
                    nodes,
                    rows,
                ));
            }
        }
        let id = nodes.len();
        nodes.push(Node {
            state: public,
            probabilities,
            children,
            terminal,
            turn_leaf,
        });
        id
    }
    let mut nodes = vec![];
    let mut rows = vec![];
    let root = capture(
        engine, &terminals, root, &board, ranges, &mut nodes, &mut rows,
    );
    Base {
        schema: "live-nash-frozen-flop-v1".into(),
        protocol_sha256,
        case: case["id"].as_str().unwrap().into(),
        game: engine.game.clone(),
        root,
        nodes,
        frozen_flop_rows: rows,
        maximum_serving_thread_probability_delta: 0.0,
        maximum_serving_thread_range_delta: 0.0,
    }
}
fn install(engine: &PracticePolicyEngine, base: &Base) {
    assert_eq!(engine.game, base.game);
    install_flop_rows(engine, &base.frozen_flop_rows);
}

fn install_flop_rows(engine: &PracticePolicyEngine, rows: &[FrozenRow]) {
    for row in rows {
        let state = response::game_state(&row.state);
        let deal = deal(&row.state.board, state.actor);
        let key = range_policy_cache_key(&state, &deal);
        let labels: Vec<String> = state
            .legal_actions(&engine.game)
            .iter()
            .map(|a| a.label.clone())
            .collect();
        // Preserve the root solve's descendant rows even when the bounded
        // public-node cache evicts an ancestor during exhaustive enumeration.
        // This reproduces visiting the flop root before the turn, as the UI does.
        if resolver_serves_actor(
            engine
                .policy
                .flop_resolver
                .as_ref()
                .unwrap()
                .config
                .resolved_actor,
            state.actor,
        ) {
            engine
                .policy
                .flop_strategy_cache
                .lock()
                .unwrap()
                .rows
                .insert(
                    key,
                    Arc::new(ResolvedPolicyCachedRow {
                        actor: state.actor,
                        action_labels: labels.clone(),
                        probabilities: row.probabilities.clone(),
                        action_values_bb: None,
                    }),
                );
        }
        engine.policy.range_cache.lock().unwrap().insert(
            key,
            Arc::new(RangePolicyCachedNode {
                action_labels: labels,
                probabilities: row.probabilities.clone(),
                action_values_bb: None,
                ranges: row.state.ranges.clone(),
            }),
        );
    }
}
fn leaf(engine: &PracticePolicyEngine, node: &Node, turn: u8) -> (response::Values, usize, f64) {
    let mut state = node.state.clone();
    assert!(!state.board.contains(&turn));
    state.board.push(turn);
    for range in &mut state.ranges {
        for combo in all_combos() {
            if combo.cards().contains(&turn) {
                range[combo.key()] = 0.0;
            }
        }
    }
    let reaches = state.ranges.clone();
    response::turn(
        pbs::TurnRiverSolveConfig {
            game: engine.game.clone(),
            state,
            iterations: 2,
            averaging_delay: 0,
            river_refinement_iterations: 0,
            regret_matching_plus: false,
        },
        reaches,
        |state, board| Ok(served(engine, state, board).probabilities.clone()),
    )
    .unwrap()
}
fn backup(base: &Base, leaves: &BTreeMap<usize, response::Values>) -> response::Values {
    let mut values: Vec<response::Values> = vec![];
    for (id, node) in base.nodes.iter().enumerate() {
        if let Some(terminal) = &node.terminal {
            values.push(terminal.clone());
            continue;
        }
        if node.turn_leaf {
            values.push(leaves[&id].clone());
            continue;
        }
        let actor = node.state.actor;
        let opponent = 1 - actor;
        let n = node.children.len();
        let mut result = response::Values {
            profile: [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]],
            best_response: [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]],
        };
        for combo in all_combos() {
            let c = combo.key();
            if combo
                .cards()
                .iter()
                .any(|card| node.state.board.contains(card))
            {
                continue;
            }
            result.best_response[actor][c] = f64::NEG_INFINITY;
            for (a, child) in node.children.iter().enumerate() {
                assert!(*child < id);
                let probability = node.probabilities[c * n + a];
                let child = &values[*child];
                result.profile[actor][c] += probability * child.profile[actor][c];
                result.profile[opponent][c] += child.profile[opponent][c];
                result.best_response[actor][c] =
                    result.best_response[actor][c].max(child.best_response[actor][c]);
                result.best_response[opponent][c] += child.best_response[opponent][c];
            }
        }
        values.push(result);
    }
    values[base.root].clone()
}

#[test]
#[ignore = "explicit live policy diagnostic, with protocol/output environment"]
fn run_live_nash_task() {
    let root = PathBuf::from(std::env::var("POKER_LIVE_NASH_ROOT").unwrap());
    let output = PathBuf::from(std::env::var("POKER_LIVE_NASH_OUTPUT").unwrap());
    let protocol_path = output.join("protocol.json");
    let protocol: serde_json::Value = read(&protocol_path);
    let case_id = std::env::var("POKER_LIVE_NASH_CASE").unwrap();
    let case = protocol["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["id"] == case_id)
        .unwrap();
    let directory = output.join(&case_id);
    fs::create_dir_all(&directory).unwrap();
    let base_path = directory.join("base.json");
    let task = std::env::var("POKER_LIVE_NASH_TASK").unwrap();
    if task == "aggregate" {
        let base: Base = read(&base_path);
        let hash = digest(&base_path);
        assert_eq!(base.protocol_sha256, digest(&protocol_path));
        let mut integrated: BTreeMap<usize, response::Values> = base
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.turn_leaf)
            .map(|(id, _)| {
                (
                    id,
                    response::Values {
                        profile: [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]],
                        best_response: [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]],
                    },
                )
            })
            .collect();
        let mut count = 0;
        let mut public_nodes = 0;
        let mut max_error: f64 = 0.0;
        for turn in 0..52u8 {
            if base.nodes[base.root].state.board.contains(&turn) {
                continue;
            }
            let packet: Packet = read(&directory.join(format!("turn-{turn:02}.json")));
            assert_eq!(packet.base_sha256, hash);
            assert_eq!(packet.turn, turn);
            assert_eq!(
                packet.leaves.keys().collect::<Vec<_>>(),
                integrated.keys().collect::<Vec<_>>()
            );
            for (id, values) in packet.leaves {
                let target = integrated.get_mut(&id).unwrap();
                for combo in all_combos() {
                    if combo.cards().contains(&turn) {
                        continue;
                    }
                    for seat in 0..2 {
                        target.profile[seat][combo.key()] +=
                            values.profile[seat][combo.key()] / 45.0;
                        target.best_response[seat][combo.key()] +=
                            values.best_response[seat][combo.key()] / 45.0;
                    }
                }
            }
            count += 1;
            public_nodes += packet.public_nodes;
            max_error = max_error.max(packet.maximum_probability_sum_error);
        }
        assert_eq!(count, 49);
        let values = backup(&base, &integrated);
        let metrics = response::metrics(&values, &base.nodes[base.root].state.ranges).unwrap();
        write(
            &directory.join("flop-result.json"),
            &serde_json::json!({"baseSha256":hash,"metrics":metrics,
            "publicTurns":count,"continuationPublicNodes":public_nodes,"maximumProbabilitySumError":max_error}),
        );
        println!("{}", metrics);
        return;
    }
    let manifest = &protocol["manifest"];
    let engine = serving_engine(
        &root,
        manifest,
        protocol["threadsPerWorker"].as_u64().unwrap() as usize,
    );
    if task == "base" {
        let mut candidate = base(&engine, &protocol, case, digest(&protocol_path));
        assert_eq!(
            candidate
                .nodes
                .iter()
                .filter(|node| checkdown_turn(node))
                .count(),
            1
        );
        // The resource-only two-thread setting must reproduce the normal
        // eight-thread website profile, including both actors and all flop nodes.
        let normal = serving_engine(&root, manifest, 8);
        for row in &candidate.frozen_flop_rows {
            let state = response::game_state(&row.state);
            let actual = served(&normal, &state, &row.state.board);
            for (left, right) in row.probabilities.iter().zip(&actual.probabilities) {
                candidate.maximum_serving_thread_probability_delta = candidate
                    .maximum_serving_thread_probability_delta
                    .max((left - right).abs());
            }
            for (left, right) in row
                .state
                .ranges
                .iter()
                .flatten()
                .zip(actual.ranges.iter().flatten())
            {
                candidate.maximum_serving_thread_range_delta = candidate
                    .maximum_serving_thread_range_delta
                    .max((left - right).abs());
            }
        }
        assert!(candidate.maximum_serving_thread_probability_delta < 1e-8);
        assert!(candidate.maximum_serving_thread_range_delta < 1e-8);
        let count = candidate.nodes.iter().filter(|n| n.turn_leaf).count();
        println!(
            "{}",
            serde_json::json!({"case":case_id,"flopNodes":candidate.nodes.len(),"turnLeaves":count})
        );
        write(&base_path, &candidate);
        return;
    }
    let candidate: Base = read(&base_path);
    assert_eq!(candidate.protocol_sha256, digest(&protocol_path));
    install(&engine, &candidate);
    let turn = std::env::var("POKER_LIVE_NASH_TURN")
        .unwrap()
        .parse::<u8>()
        .unwrap();
    if task == "packet" {
        let mut leaves = BTreeMap::new();
        let mut nodes = 0;
        let mut error: f64 = 0.0;
        for (id, node) in candidate
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.turn_leaf)
        {
            let started = std::time::Instant::now();
            let (value, count, residual) = leaf(&engine, node, turn);
            println!(
                "{}",
                serde_json::json!({"turn":turn,"leaf":id,"publicNodes":count,"seconds":started.elapsed().as_secs_f64()})
            );
            leaves.insert(id, value);
            nodes += count;
            error = error.max(residual);
        }
        write(
            &directory.join(format!("turn-{turn:02}.json")),
            &Packet {
                schema: "live-nash-turn-packet-v1".into(),
                base_sha256: digest(&base_path),
                turn,
                leaves,
                public_nodes: nodes,
                maximum_probability_sum_error: error,
            },
        );
        return;
    }
    assert_eq!(task, "turn");
    let node = candidate
        .nodes
        .iter()
        .find(|node| checkdown_turn(node))
        .unwrap();
    let (values, nodes, error) = leaf(&engine, node, turn);
    let mut ranges = node.state.ranges.clone();
    for range in &mut ranges {
        for combo in all_combos() {
            if combo.cards().contains(&turn) {
                range[combo.key()] = 0.0;
            }
        }
    }
    let metrics = response::metrics(&values, &ranges).unwrap();
    write(
        &directory.join("direct-turn-result.json"),
        &serde_json::json!({"baseSha256":digest(&base_path),
        "turn":turn,"history":node.state.public_history,"metrics":metrics,"publicNodes":nodes,"maximumProbabilitySumError":error}),
    );
    println!("{}", metrics);
}

#[test]
fn benchmark_lines_and_flop_backup_are_consistent() {
    let game = BlueprintConfig::default();
    for labels in [
        vec!["limp", "check"],
        vec!["raise_to_2.500bb", "call"],
        vec!["raise_to_2.500bb", "raise_to_7.500bb", "call"],
    ] {
        let labels = labels
            .into_iter()
            .map(|s| serde_json::json!(s))
            .collect::<Vec<_>>();
        let state = preflop(&game, &labels);
        let mut checked = state.clone();
        for _ in 0..2 {
            let action = checked
                .legal_actions(&game)
                .into_iter()
                .find(|a| a.label == "check")
                .unwrap();
            checked = checked.apply(&action, &game);
        }
        let public = PublicBeliefState::from_game_state(
            vec![51, 49, 22],
            &checked,
            [
                pbs::uniform_range(&[51, 49, 22]),
                pbs::uniform_range(&[51, 49, 22]),
            ],
        );
        let leaf = Node {
            state: public.clone(),
            probabilities: vec![],
            children: vec![],
            terminal: None,
            turn_leaf: true,
        };
        assert!(checkdown_turn(&leaf));
        let mut root_public = public;
        root_public.actor = 0;
        let root = Node {
            state: root_public,
            probabilities: vec![0.5; COMBO_COUNT * 2],
            children: vec![0, 1],
            terminal: None,
            turn_leaf: false,
        };
        let base = Base {
            schema: "test".into(),
            protocol_sha256: "test".into(),
            case: "test".into(),
            game: game.clone(),
            root: 2,
            nodes: vec![leaf.clone(), leaf, root],
            frozen_flop_rows: vec![],
            maximum_serving_thread_probability_delta: 0.0,
            maximum_serving_thread_range_delta: 0.0,
        };
        let values = |value: f64| response::Values {
            profile: [vec![value; COMBO_COUNT], vec![-value / 2.0; COMBO_COUNT]],
            best_response: [vec![value; COMBO_COUNT], vec![-value / 2.0; COMBO_COUNT]],
        };
        let result = backup(&base, &BTreeMap::from([(0, values(1.0)), (1, values(3.0))]));
        let combo = all_combos()
            .into_iter()
            .find(|c| !c.cards().iter().any(|card| [51, 49, 22].contains(card)))
            .unwrap()
            .key();
        assert_eq!(result.profile[0][combo], 2.0);
        assert_eq!(result.best_response[0][combo], 3.0);
        assert_eq!(result.profile[1][combo], -2.0);
        assert_eq!(result.best_response[1][combo], -2.0);
    }
}
