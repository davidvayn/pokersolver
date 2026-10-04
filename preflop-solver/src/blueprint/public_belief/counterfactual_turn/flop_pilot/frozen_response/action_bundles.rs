//! Research-only, complete frozen-prefix export. No solver/playing changes.
use super::*;

fn prefix(frozen: &Frozen) -> serde_json::Value {
    fn visit(f: &Frozen, state: GameState, reaches: Ranges,
        nodes: &mut Vec<serde_json::Value>, terminals: &mut Vec<serde_json::Value>,
        leaves: &mut Vec<History>) {
        if state.terminal.is_some() {
            terminals.push(serde_json::json!({"history":state.public_history,
                "raw_cfvs":f.trunk.terminal(&state,&reaches)}));
            return;
        }
        if state.street == Street::Turn {
            leaves.push(state.public_history.clone());
            return;
        }
        let actions=state.legal_actions(&f.trunk.game);
        let sigma=&f.strategies[&state.public_history];
        let children=actions.iter().map(|action| {
            let child=state.apply(action,&f.trunk.game);
            serde_json::json!({"history":child.public_history,
                "kind":if child.terminal.is_some(){"terminal"}
                    else if child.street==Street::Turn{"leaf"}else{"node"}})
        }).collect::<Vec<_>>();
        nodes.push(serde_json::json!({"history":state.public_history,"actor":state.actor,
            "actions":actions.iter().map(|a|a.label.clone()).collect::<Vec<_>>(),
            "probabilities":sigma,"reaches":reaches,"children":children}));
        for (a,action) in actions.iter().enumerate() {
            let mut child=reaches.clone();
            for c in 0..COMBO_COUNT { child[state.actor][c]*=sigma[c*actions.len()+a]; }
            visit(f,state.apply(action,&f.trunk.game),child,nodes,terminals,leaves);
        }
    }
    let (mut nodes,mut terminals,mut leaves)=(vec![],vec![],vec![]);
    visit(frozen,frozen.trunk.state.game_state(),frozen.trunk.state.ranges.clone(),
        &mut nodes,&mut terminals,&mut leaves);
    serde_json::json!({"schema":"hu-frozen-action-prefix-v1",
        "candidate_sha256":frozen.candidate_sha256,"game":frozen.trunk.game,
        "root":frozen.trunk.state,"legal":frozen.trunk.legal,
        "nodes":nodes,"terminals":terminals,"leaves":leaves,
        "leaf_states":frozen.turns.values().map(|(state,reaches)|
            PublicBeliefState::from_game_state(frozen.trunk.state.board.clone(),state,reaches.clone()))
            .collect::<Vec<_>>(),
        "releaseAccepted":false})
}

fn labels(frozen:&Frozen,turn:u8,iterations:u64)->Result<serde_json::Value,String> {
    if ![64,256,1024].contains(&iterations) { return Err("unbounded bundle reference".into()); }
    let mut targets=vec![];
    for mut config in frozen.belief_queries(turn)? {
        config.iterations=iterations;
        let values=frozen.solve_turn(config.clone())?;
        targets.push(value_targets::Target::new(1,&config,&values)?
            .with_distribution("complete_frozen_prefix_action_bundle"));
    }
    Ok(serde_json::json!({"schema":"hu-frozen-action-turn-labels-v1",
        "candidate_sha256":frozen.candidate_sha256,"turn":turn,"turn_iterations":iterations,
        "targets":targets,"releaseAccepted":false}))
}

#[test]
fn bundle_prefix_is_complete_and_matches_frozen_terminal_backup() {
    let fixture=super::tests::royal_fixture();
    let before=serde_json::to_vec(&fixture).unwrap();
    let frozen=Frozen::new(&fixture).unwrap();
    let exported=prefix(&frozen);
    assert_eq!(exported["nodes"].as_array().unwrap().len(),fixture.strategies.len());
    assert_eq!(exported["leaves"].as_array().unwrap().len(),frozen.turns.len());
    assert_eq!(exported["candidate_sha256"],frozen.candidate_sha256);
    assert_eq!(before,serde_json::to_vec(&fixture).unwrap());
    assert!(labels(&frozen,0,65).is_err());
    assert!(labels(&frozen,32,64).is_err());
}

#[test]
#[ignore="immutable candidate, complete-prefix export; invoke under external resource guard"]
fn saved_action_bundle_prefix() {
    let frozen=super::tests::read_candidate();
    let digest=super::tests::exclusive_output(&prefix(&frozen));
    println!("{}",serde_json::json!({"prefixSha256":digest,"leaves":frozen.turns.len()}));
}

#[test]
#[ignore="immutable candidate/turn/reference budget; invoke under external resource guard"]
fn saved_action_bundle_turn_labels() {
    let frozen=super::tests::read_candidate();
    let turn=std::env::var("POKER_NATIVE_FLOP_TURN").unwrap().parse().unwrap();
    let iterations=std::env::var("POKER_ACTION_LABEL_ITERATIONS").unwrap().parse().unwrap();
    let result=labels(&frozen,turn,iterations).unwrap();
    let digest=super::tests::exclusive_output(&result);
    println!("{}",serde_json::json!({"labelsSha256":digest,"turn":turn,"leaves":frozen.turns.len()}));
}
