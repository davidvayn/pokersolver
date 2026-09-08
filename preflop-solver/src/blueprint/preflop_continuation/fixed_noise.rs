//! Fixed policy/history: distinguish public-flop and sampled-turn variability.
//! No regret updates, no policy selection, no full-game strength claim.
use super::*;

#[test]
#[ignore = "pinned frozen policy continuation-noise probe; external resource guard required"]
fn frozen_open_call_continuation_noise() {
    let source = PathBuf::from(std::env::var("POKER_NOISE_PREFLOP").unwrap());
    let preflop = response::FrozenPreflopPolicy::read(&source).unwrap();
    assert_eq!(
        preflop.artifact_sha256,
        std::env::var("POKER_NOISE_PREFLOP_SHA").unwrap()
    );
    assert_eq!(preflop.game.effective_stack_bb, 20.0);
    assert_eq!(preflop.node_count(), 16900);
    let model = PublicValueNetwork::read(Path::new(&std::env::var("POKER_COMPACT_MODEL").unwrap()))
        .unwrap();
    assert_eq!(
        model.artifact_sha256().unwrap(),
        std::env::var("POKER_COMPACT_MODEL_SHA").unwrap()
    );
    let kernel_sha = std::env::var("POKER_COMPACT_CHECKDOWN_SHA").unwrap();
    let exact = exact_checkdown::ExactCheckdown::read(
        Path::new(&std::env::var("POKER_COMPACT_CHECKDOWN").unwrap()),
        &kernel_sha,
    )
    .unwrap();
    let index: u64 = std::env::var("POKER_NOISE_BOARD_INDEX")
        .unwrap()
        .parse()
        .unwrap();
    let count: usize = std::env::var("POKER_NOISE_TURNS").unwrap().parse().unwrap();
    assert!(index < 4 && [2, 4].contains(&count));
    let output = PathBuf::from(std::env::var("POKER_COMPACT_OUTPUT").unwrap());
    assert!(!output.exists());
    let started = Instant::now();
    let mut state = GameState::initial(&preflop.game);
    let mut prior = [vec![1.0; 1326], vec![1.0; 1326]];
    for label in ["raise_to_2.000bb", "call"] {
        let actions = state.legal_actions(&preflop.game);
        let a = actions.iter().position(|a| a.label == label).unwrap();
        for c in all_combos() {
            prior[state.actor][c.key()] *= preflop.strategy(&state, c).unwrap()[a];
        }
        state = state.apply(&actions[a], &preflop.game);
    }
    assert!(state.street == Street::Flop && state.terminal.is_none());
    let mut rng = SplitMix64::new(batch_iteration_seed(31001, index + 1));
    let deal = Deal::sample(&mut rng);
    let board: [u8; 3] = deal.board[..3].try_into().unwrap();
    let input = PublicBeliefState::from_preflop_reaches(&state, board, prior.clone()).unwrap();
    let digest = Sha256::digest(serde_json::to_vec(&input).unwrap());
    let options = NativeFlopOptions {
        seed: 100101 ^ u64::from_le_bytes(digest[..8].try_into().unwrap()),
        iterations: 128,
        training_turn_iterations: 64,
        response_turn_iterations: 64,
    };
    let mut masked = prior.clone();
    mask_ranges(&mut masked, board);
    let totals: [f64; 2] = std::array::from_fn(|p| masked[p].iter().sum());
    let flop_kernel = exact_flop_kernel(board).unwrap();
    let mean = exact.values(state.invested, &prior).unwrap();
    let baseline: [Vec<f64>; 2] = std::array::from_fn(|p| {
        flop_kernel
            .values_on_flop(state.invested, &masked[1 - p], p)
            .unwrap()
            .iter()
            .map(|v| v * FLOP_CHANCE_CORRECTION / OPPONENT_HANDS)
            .collect()
    });
    let labels: Vec<_> = all_combos()
        .iter()
        .map(|c| c.label())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let class_index: Vec<_> = all_combos()
        .iter()
        .map(|c| labels.binary_search(&c.label()).unwrap())
        .collect();
    let mut multiplicities = vec![0; 169];
    for k in &class_index {
        multiplicities[*k] += 1;
    }
    let class_values = |values: &[Vec<f64>; 2]| -> [Vec<f64>; 2] {
        std::array::from_fn(|p| {
            let mut row = vec![0.0; 169];
            for c in 0..1326 {
                row[class_index[c]] += values[p][c] / multiplicities[class_index[c]] as f64;
            }
            row
        })
    };
    let conflicts = public_belief::combo_conflicts();
    let mass: [Vec<f64>; 2] = std::array::from_fn(|p| {
        (0..1326)
            .map(|c| {
                public_belief::compatible_mass_from_conflicts(&prior[1 - p], &conflicts, c)
                    / OPPONENT_HANDS
            })
            .collect()
    });
    let class_mass = class_values(&mass);
    let own = class_values(&prior);
    let weights: [Vec<f64>; 2] = std::array::from_fn(|p| {
        (0..169)
            .map(|k| own[p][k] * class_mass[p][k] * multiplicities[k] as f64)
            .collect()
    });
    let policy = NativePostflopPolicy::solve_counterfactual_learned(
        preflop.game.clone(),
        input,
        &options,
        &model,
    )
    .unwrap();
    if let Ok(source) = std::env::var("POKER_NOISE_NATIVE_SOURCE") {
        let bytes = fs::read(source).unwrap();
        assert!(bytes.len() < 4 * 1024 * 1024);
        let native_sha = std::env::var("POKER_NOISE_NATIVE_SOURCE_SHA").unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), native_sha);
        let native: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(native["schema"], "frozen-preflop-continuation-noise-v1");
        assert_eq!(native["preflopSha256"], preflop.artifact_sha256);
        assert_eq!(native["modelSha256"], model.artifact_sha256().unwrap());
        assert_eq!(native["kernelSha256"], kernel_sha);
        assert_eq!(native["boardIndex"], index);
        assert_eq!(native["board"], serde_json::json!(board));
        assert_eq!(native["policySha256"], policy.identity());
        assert_eq!(native["classes"], serde_json::json!(labels));
        assert_eq!(native["classReachWeights"], serde_json::json!(weights));
        let before = Instant::now();
        let mut predictions = Vec::new();
        for turn in (0..52).filter(|c| !board.contains(c)) {
            let mut values = policy
                .sampled_predicted_training_values(turn, &model)
                .unwrap();
            for p in 0..2 {
                for v in &mut values[p] {
                    *v *= totals[1 - p] * FLOP_CHANCE_CORRECTION / OPPONENT_HANDS;
                }
            }
            let raw = class_values(&values);
            assert!(raw.iter().flatten().all(|v| v.is_finite()));
            predictions.push(serde_json::json!({"turn":turn,"predictedRawCfvBb":raw}));
        }
        let payload = serde_json::json!({"schema":"fixed-continuation-turn-predictions-v1",
            "nativeSourceSha256":native_sha,"preflopSha256":preflop.artifact_sha256,
            "modelSha256":model.artifact_sha256(),"kernelSha256":kernel_sha,
            "boardIndex":index,"board":board,"policySha256":policy.identity(),
            "classes":labels,"classOpponentMass":class_mass,"classReachWeights":weights,
            "predictions":predictions,"predictionSeconds":before.elapsed().as_secs_f64(),
            "seconds":started.elapsed().as_secs_f64(),"releaseAccepted":false,
            "interpretation":"Predicted CFVs only, all 49 public turns at the identical frozen flop policy. Complete prediction chance mean permits native-minus-predicted correction without treating predictions as ground truth."});
        write_output(&output, &payload);
        return;
    }
    let mut turns: Vec<_> = (0..52).filter(|c| !board.contains(c)).collect();
    let mut records = Vec::new();
    for _ in 0..count {
        let turn = turns.swap_remove(rng.index(turns.len()));
        let before = Instant::now();
        let mut values = policy.sampled_training_values(turn).unwrap();
        for p in 0..2 {
            for v in &mut values[p] {
                *v *= totals[1 - p] * FLOP_CHANCE_CORRECTION / OPPONENT_HANDS;
            }
        }
        let raw = class_values(&values);
        let sampled_checkdown = class_values(&baseline);
        let exact_mean = class_values(&mean);
        let conditional_residual: [Vec<f64>; 2] = std::array::from_fn(|p| {
            (0..169)
                .map(|k| {
                    if class_mass[p][k] > 0.0 {
                        (raw[p][k] - sampled_checkdown[p][k]) / class_mass[p][k]
                    } else {
                        0.0
                    }
                })
                .collect()
        });
        assert!(conditional_residual.iter().flatten().all(|v| v.is_finite()));
        let weighted_sum: f64 = (0..2)
            .map(|p| {
                (0..169)
                    .map(|k| weights[p][k] * conditional_residual[p][k])
                    .sum::<f64>()
            })
            .sum();
        assert!(weighted_sum.abs() < 1e-8);
        records.push(serde_json::json!({"turn":turn,"conditionalStrategicResidualBb":conditional_residual,
            "rawCfvBb":raw,"sampledCheckdownCfvBb":sampled_checkdown,"exactMeanCheckdownCfvBb":exact_mean,
            "seconds":before.elapsed().as_secs_f64()}));
        eprintln!(
            "{}",
            serde_json::json!({"stage":"fixed_noise_turn_complete","boardIndex":index,
            "turn":turn,"completed":records.len(),"seconds":before.elapsed().as_secs_f64()})
        );
    }
    let payload = serde_json::json!({"schema":"frozen-preflop-continuation-noise-v1",
        "preflopSha256":preflop.artifact_sha256,"modelSha256":model.artifact_sha256(),"kernelSha256":kernel_sha,
        "boardIndex":index,"board":board,"history":state.public_history,"policySha256":policy.identity(),
        "turnSampleCount":count,"classes":labels,"classReachWeights":weights,"records":records,
        "seconds":started.elapsed().as_secs_f64(),"releaseAccepted":false,
        "interpretation":"Fixed frozen 2bb-open/call preflop ranges; uniform public flops and turns sampled without replacement. Conditional class residuals include exact card-removal chance correction. Diagnostic continuation variability, not action-EV confidence or full-game exploitability."});
    write_output(&output, &payload);
}

fn write_output(output: &Path, payload: &serde_json::Value) {
    let mut f = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap();
    f.write_all(&serde_json::to_vec(&payload).unwrap()).unwrap();
    f.sync_all().unwrap();
}
