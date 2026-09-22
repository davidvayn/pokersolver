//! Export public flop roots from a pinned preflop policy. No training, private
//! holding, future board card, or synthetic range floor enters the benchmark.
use super::*;

fn root(game: &BlueprintConfig, kind: &str) -> Result<GameState, String> {
    let line: &[&str] = match kind {
        "limped" => &["limp", "check"],
        "single-raised" => &["raise_to_2.500bb", "call"],
        "three-bet" => &["raise_to_2.000bb", "raise_to_7.500bb", "call"],
        _ => return Err("unknown benchmark pot type".into()),
    };
    let mut state = GameState::initial(game);
    for label in line {
        let action = state.legal_actions(game).into_iter().find(|a| &a.label == label)
            .ok_or("benchmark line outside pinned abstraction")?;
        state = state.apply(&action, game);
    }
    if state.street != Street::Flop || state.terminal.is_some() {
        return Err("benchmark root must start a live flop".into());
    }
    Ok(state)
}

#[test]
fn benchmark_lines_have_exact_starting_pots_and_no_flop_actions() {
    let game = BlueprintConfig { effective_stack_bb: 20.0, ..BlueprintConfig::default() };
    for (kind, pot) in [("limped", 2.0), ("single-raised", 5.0), ("three-bet", 15.0)] {
        let state = root(&game, kind).unwrap();
        assert_eq!(state.invested.iter().sum::<f64>(), pot);
        assert_eq!(state.street_invested, [0.0, 0.0]);
        assert_eq!(state.actor, 1);
    }
    assert!(root(&game, "invented").is_err());
}

#[test]
#[ignore = "hash-pinned benchmark protocol and preflop artifact; public-root export only"]
fn export_postflop_benchmark_roots() {
    let policy = response::FrozenPreflopPolicy::read(Path::new(
        &std::env::var("POKER_NOISE_PREFLOP").unwrap())).unwrap();
    assert_eq!(policy.artifact_sha256, std::env::var("POKER_NOISE_PREFLOP_SHA").unwrap());
    assert_eq!(policy.node_count(), 16900);
    assert_eq!(policy.game.effective_stack_bb, 20.0);
    let bytes = fs::read(std::env::var("POKER_BENCHMARK_PROTOCOL").unwrap()).unwrap();
    assert_eq!(format!("{:x}", Sha256::digest(&bytes)),
        std::env::var("POKER_BENCHMARK_PROTOCOL_SHA").unwrap());
    let protocol: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(protocol["schema"], "postflop-benchmark-protocol-v1");
    let spots = protocol["spots"].as_array().unwrap();
    assert_eq!(spots.len(), 12);
    let snapshot = Snapshot::capture_frozen(&policy).unwrap();
    let mut roots = Vec::new();
    for spot in spots {
        let state = root(&policy.game, spot["potType"].as_str().unwrap()).unwrap();
        let board: [u8; 3] = serde_json::from_value(spot["board"].clone()).unwrap();
        assert!(board.iter().all(|c| *c < 52));
        assert_eq!(board.iter().collect::<std::collections::BTreeSet<_>>().len(), 3);
        let public = snapshot.flop_input(&state.public_history, board).unwrap();
        assert_eq!(public.invested_bb.iter().sum::<f64>(), spot["startingPotBb"].as_f64().unwrap());
        roots.push(serde_json::json!({"id":spot["id"], "input":{
            "schema":"native-flop-public-root-v1", "game":policy.game, "public":public}}));
    }
    let path = PathBuf::from(std::env::var("POKER_COMPACT_OUTPUT").unwrap());
    assert!(!path.exists());
    write_json_atomic(&path, &serde_json::json!({"schema":"postflop-benchmark-roots-v1",
        "preflopSha256":policy.artifact_sha256,"roots":roots})).unwrap();
}
