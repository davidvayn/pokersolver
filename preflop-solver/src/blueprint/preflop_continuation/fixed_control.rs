//! Exact preflop/checkdown surrogate: isolates the shared update/average code.
//! There is no postflop betting in this control, and NO policy is exported.
use super::*;

pub(super) fn values(
    snapshot: &Snapshot,
    kernel: &exact_checkdown::ExactCheckdown,
) -> Result<BTreeMap<History, [Vec<f64>; 2]>, String> {
    let conflicts = public_belief::combo_conflicts();
    snapshot
        .endpoints
        .iter()
        .map(|(history, (state, prior))| {
            let value = if let Some(Terminal::Fold { winner }) = state.terminal {
                let p0 = if winner == 0 {
                    state.invested[1]
                } else {
                    -state.invested[0]
                };
                std::array::from_fn(|p| {
                    (0..1326)
                        .map(|c| {
                            (if p == 0 { p0 } else { -p0 })
                                * public_belief::compatible_mass_from_conflicts(
                                    &prior[1 - p],
                                    &conflicts,
                                    c,
                                )
                                / OPPONENT_HANDS
                        })
                        .collect()
                })
            } else {
                kernel.values(state.invested, prior)?
            };
            Ok((history.clone(), value))
        })
        .collect()
}

/// One response action per observable private class/history. Opponent reaches
/// are already in the terminal CFVs; never maximize separately by opponent hand.
fn walk(
    snapshot: &Snapshot,
    game: &BlueprintConfig,
    state: &GameState,
    endpoints: &BTreeMap<History, [Vec<f64>; 2]>,
    seat: usize,
    best: bool,
) -> Vec<f64> {
    if let Some(value) = endpoints.get(&state.public_history) {
        return value[seat].clone();
    }
    let row = &snapshot.rows[&state.public_history];
    let children: Vec<_> = state
        .legal_actions(game)
        .iter()
        .map(|a| walk(snapshot, game, &state.apply(a, game), endpoints, seat, best))
        .collect();
    (0..1326)
        .map(|c| {
            if state.actor == seat && best {
                children
                    .iter()
                    .map(|v| v[c])
                    .fold(f64::NEG_INFINITY, f64::max)
            } else if state.actor == seat {
                children
                    .iter()
                    .enumerate()
                    .map(|(a, v)| row.probabilities[a][c] * v[c])
                    .sum()
            } else {
                children.iter().map(|v| v[c]).sum()
            }
        })
        .collect()
}

fn evaluate(
    trainer: &Trainer,
    kernel: &exact_checkdown::ExactCheckdown,
) -> Result<serde_json::Value, String> {
    let snapshot = Snapshot::capture_policy(trainer, true)?;
    let endpoints = values(&snapshot, kernel)?;
    let root = GameState::initial(&trainer.config);
    let mut profile = [0.0; 2];
    let mut response = [0.0; 2];
    for p in 0..2 {
        profile[p] = walk(&snapshot, &trainer.config, &root, &endpoints, p, false)
            .iter()
            .sum::<f64>()
            / 1326.0;
        response[p] = walk(&snapshot, &trainer.config, &root, &endpoints, p, true)
            .iter()
            .sum::<f64>()
            / 1326.0;
        if !response[p].is_finite() || response[p] < profile[p] - 1e-9 {
            return Err("invalid information-set-consistent checkdown response".into());
        }
    }
    if (profile[0] + profile[1]).abs() > 1e-9 {
        return Err("checkdown profile lost zero sum".into());
    }
    Ok(serde_json::json!({"rounds":trainer.completed_iterations,
        "profileBb":profile,"bestResponseBb":response,
        "exactCheckdownGameNashConvBb":response[0]+response[1],
        "fullHandExploitability":null,"releaseAccepted":false}))
}

#[test]
#[ignore = "bounded exact fixed-payoff preflop control; real kernel hash and resource guard required"]
fn exact_checkdown_preflop_convergence_control() {
    let path = PathBuf::from(std::env::var("POKER_COMPACT_CHECKDOWN").unwrap());
    let digest = std::env::var("POKER_COMPACT_CHECKDOWN_SHA").unwrap();
    let kernel = exact_checkdown::ExactCheckdown::read(&path, &digest).unwrap();
    let rounds: u64 = std::env::var("POKER_FIXED_CONTROL_ROUNDS")
        .unwrap()
        .parse()
        .unwrap();
    assert!([8, 32, 128, 512].contains(&rounds));
    let output = PathBuf::from(std::env::var("POKER_COMPACT_OUTPUT").unwrap());
    assert!(!output.exists());
    let config = BlueprintConfig {
        seed: 27001,
        effective_stack_bb: 20.0,
        iterations: rounds,
        averaging_delay: 0,
        exact_preflop_averaging: true,
        max_information_sets: 20_000,
        traversal: BlueprintTraversal::PublicChanceSampling,
        ..BlueprintConfig::default()
    };
    let mut trainer = Trainer::fresh(config.clone());
    let mut checkpoints = Vec::new();
    let started = Instant::now();
    for round in 1..=rounds {
        let snapshot = begin_update(&mut trainer).unwrap();
        let endpoints = values(&snapshot, &kernel).unwrap();
        let root = GameState::initial(&config);
        back_up_preflop(
            &mut trainer,
            &snapshot,
            &root,
            &endpoints,
            (round as usize - 1) % 2,
        )
        .unwrap();
        trainer.completed_iterations += 1;
        assert_eq!(trainer.nodes.len(), 16900);
        if [2, 8, 32, 128, 512].contains(&round) {
            let result = evaluate(&trainer, &kernel).unwrap();
            eprintln!(
                "{}",
                serde_json::json!({"stage":"fixed_checkdown_control",
                "result":result,"seconds":started.elapsed().as_secs_f64()})
            );
            checkpoints.push(result);
        }
    }
    let payload = serde_json::json!({"schema":"fixed-checkdown-preflop-control-v1",
        "kernelSha256":digest,"rounds":rounds,"checkpoints":checkpoints,
        "seconds":started.elapsed().as_secs_f64(),"releaseAccepted":false,
        "interpretation":"Exact fixed-payoff surrogate, with forced checkdown after preflop. Same DCFR update and exact-average code; no neural values or sampled chance. These NashConv values are NOT full-hand Holdem exploitability. No policy exported."});
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap();
    file.write_all(&serde_json::to_vec(&payload).unwrap())
        .unwrap();
    file.sync_all().unwrap();
}
