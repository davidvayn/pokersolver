//! Public-only, hash-pinned input for the ignored development pilot.
//! No rollout samples or hidden holdings from a cache enter the solve.
use super::*;

#[derive(Debug, Deserialize, Serialize)]
struct Input {
    schema: String,
    game: BlueprintConfig,
    public: PublicBeliefState,
}

pub(super) fn decode(
    bytes: &[u8],
    expected_sha256: &str,
) -> Result<(BlueprintConfig, PublicBeliefState), String> {
    if expected_sha256 != format!("{:x}", Sha256::digest(bytes)) {
        return Err("native flop public input hash mismatch".into());
    }
    let input: Input = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    if !matches!(
        input.schema.as_str(),
        "sampled-flop-public-root-diagnostic-v1" | "native-flop-public-root-v1"
    ) {
        return Err("unsupported native flop public input schema".into());
    }
    input.game.validate()?;
    if input.game.effective_stack_bb != 20.0 {
        return Err("native flop development pilot is limited to 20bb".into());
    }
    input
        .public
        .validate_street_and_normalize(&input.game, Street::Flop, 3)?;
    for range in &input.public.ranges {
        if (range.iter().sum::<f64>() - 1.0).abs() > 1e-10 {
            return Err("native flop input must preserve normalized public ranges".into());
        }
        for combo in all_combos() {
            if combo.cards().iter().any(|c| input.public.board.contains(c))
                && range[combo.key()] != 0.0
            {
                return Err("native flop input has board-blocked reach".into());
            }
        }
    }
    let mut cursor = GameState::initial(&input.game);
    for observed in &input.public.trajectory {
        if cursor.terminal.is_some() {
            return Err("native flop input continues after a terminal action".into());
        }
        cursor = cursor
            .legal_actions(&input.game)
            .iter()
            .map(|action| cursor.apply(action, &input.game))
            .find(|next| next.trajectory.last() == Some(observed))
            .ok_or("native flop input has an illegal public action")?;
    }
    if cursor.terminal.is_some()
        || PublicBeliefState::from_game_state(
            input.public.board.clone(),
            &cursor,
            input.public.ranges.clone(),
        ) != input.public
    {
        return Err("native flop input public line/accounting mismatch".into());
    }
    Ok((input.game, input.public))
}

#[test]
fn default_input_preserves_the_existing_game_and_exact_public_ranges() {
    let bytes =
        include_bytes!("../../../../../tests/fixtures/flop-continuation-public-root-a.json");
    let (game, state) = decode(bytes, &format!("{:x}", Sha256::digest(bytes))).unwrap();
    let previous: serde_json::Value = serde_json::from_slice(bytes).unwrap();
    assert_eq!(
        game,
        serde_json::from_value(previous["game"].clone()).unwrap()
    );
    assert_eq!(
        state,
        serde_json::from_value(previous["public"].clone()).unwrap()
    );
}

#[test]
fn public_input_rejects_hash_schema_card_and_accounting_corruption() {
    let bytes =
        include_bytes!("../../../../../tests/fixtures/flop-continuation-public-root-a.json");
    assert!(decode(bytes, &"0".repeat(64)).unwrap_err().contains("hash"));
    let base: Input = serde_json::from_slice(bytes).unwrap();
    for corruption in 0..5 {
        let mut bad: Input = serde_json::from_slice(bytes).unwrap();
        match corruption {
            0 => bad.schema = "unknown".into(),
            1 => {
                bad.public.ranges[0][Combo::new(bad.public.board[0], 0).key()] = 0.1;
                let total = bad.public.ranges[0].iter().sum::<f64>();
                for value in &mut bad.public.ranges[0] {
                    *value /= total;
                }
            }
            2 => bad.public.invested_bb[0] += 0.25,
            3 => bad.public.trajectory.pop().map(|_| ()).unwrap(),
            4 => bad.public.public_history.push("invented".into()),
            _ => unreachable!(),
        }
        let encoded = serde_json::to_vec(&bad).unwrap();
        assert!(decode(&encoded, &format!("{:x}", Sha256::digest(&encoded))).is_err());
    }
    let mut changed = base;
    changed.public.board = vec![51, 17, 10];
    changed.public.ranges = std::array::from_fn(|_| uniform_range(&changed.public.board));
    let encoded = serde_json::to_vec(&changed).unwrap();
    assert!(decode(&encoded, &format!("{:x}", Sha256::digest(&encoded))).is_ok());
}
