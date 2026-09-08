//! Frozen-reference endpoint values. Chance and private classes are integrated
//! before preflop response selection; held-out captures never select actions.
use super::*;

struct Classes {
    labels: Vec<String>,
    index: Vec<usize>,
    counts: Vec<usize>,
}

impl Classes {
    fn new() -> Self {
        let labels: Vec<_> = all_combos()
            .iter()
            .map(|c| c.label())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let index: Vec<_> = all_combos()
            .iter()
            .map(|c| labels.binary_search(&c.label()).unwrap())
            .collect();
        let mut counts = vec![0; labels.len()];
        for k in &index {
            counts[*k] += 1;
        }
        Self {
            labels,
            index,
            counts,
        }
    }

    fn reduce(&self, values: &[f64]) -> Vec<f64> {
        assert_eq!(values.len(), 1326);
        let mut result = vec![0.0; 169];
        for (c, v) in values.iter().enumerate() {
            assert!(v.is_finite());
            let k = self.index[c];
            result[k] += v / self.counts[k] as f64;
        }
        result
    }
}

#[test]
#[ignore = "read-only exact baseline export; pinned preflop/kernel and external resource guard required"]
fn frozen_preflop_checkdown_baseline_capture() {
    let preflop = response::FrozenPreflopPolicy::read(Path::new(
        &std::env::var("POKER_NOISE_PREFLOP").unwrap())).unwrap();
    assert_eq!(preflop.artifact_sha256, std::env::var("POKER_NOISE_PREFLOP_SHA").unwrap());
    assert_eq!(preflop.game.effective_stack_bb, 20.0);
    assert_eq!(preflop.node_count(), 16900);
    let kernel_sha = std::env::var("POKER_COMPACT_CHECKDOWN_SHA").unwrap();
    let exact = exact_checkdown::ExactCheckdown::read(Path::new(
        &std::env::var("POKER_COMPACT_CHECKDOWN").unwrap()), &kernel_sha).unwrap();
    let output = PathBuf::from(std::env::var("POKER_COMPACT_OUTPUT").unwrap());
    assert!(!output.exists());
    let snapshot = Snapshot::capture_frozen(&preflop).unwrap();
    let classes = Classes::new();
    let values = fixed_control::values(&snapshot, &exact).unwrap();
    let live: Vec<_> = snapshot.endpoints.iter()
        .filter(|(_, (s,_))| s.terminal.is_none() && s.street==Street::Flop)
        .map(|(h,_)| h).collect();
    assert_eq!(live.len(),49);
    assert_eq!(snapshot.rows.len(),100);
    let endpoints: Vec<_> = values.iter().map(|(h,v)| serde_json::json!({
        "history":h,"cfvBb":[classes.reduce(&v[0]),classes.reduce(&v[1])]})).collect();
    let payload = serde_json::json!({"schema":"frozen-preflop-checkdown-baseline-v1",
        "preflopSha256":preflop.artifact_sha256,"kernelSha256":kernel_sha,
        "classes":classes.labels,"multiplicities":classes.counts,"liveHistories":live,
        "rootHistory":GameState::initial(&preflop.game).public_history,
        "endpoints":endpoints,"nativeQueries":0,"releaseAccepted":false,
        "interpretation":"Exact baseline from the existing Rust checkdown kernel at frozen preflop reaches. No native postflop queries, new policy, or exploitability result."});
    let mut file = fs::OpenOptions::new().write(true).create_new(true).open(output).unwrap();
    file.write_all(&serde_json::to_vec(&payload).unwrap()).unwrap();
    file.sync_all().unwrap();
}

#[test]
fn frozen_snapshot_matches_average_rows_keys_and_true_reaches() {
    let config = BlueprintConfig {
        effective_stack_bb: 20.0,
        iterations: 1,
        averaging_delay: 0,
        exact_preflop_averaging: true,
        traversal: BlueprintTraversal::PublicChanceSampling,
        ..BlueprintConfig::default()
    };
    let mut trainer = Trainer::fresh(config);
    trainer.discounts.advance(1);
    trainer.sweep_preflop_average().unwrap();
    trainer.completed_iterations = 1;
    // Nonuniform codec fixture, not a trained or accepted poker strategy.
    for node in trainer.nodes.values_mut() {
        node.regret_updates = 1;
        for (a, mass) in node.strategy_sum.iter_mut().enumerate() {
            *mass = (a + 1) as f64;
        }
    }
    let path = std::env::temp_dir().join(format!(
        "snapshot-average-parity-{}.json.gz",
        std::process::id()
    ));
    trainer.write_frozen_preflop_average(&path).unwrap();
    let frozen = response::FrozenPreflopPolicy::read(&path).unwrap();
    let actual = Snapshot::capture_frozen(&frozen).unwrap();
    let expected = Snapshot::capture_policy(&trainer, true).unwrap();
    assert_eq!(actual.rows.len(), expected.rows.len());
    for (h, row) in &actual.rows {
        let other = &expected.rows[h];
        assert_eq!(row.keys, other.keys);
        for (a, b) in row
            .probabilities
            .iter()
            .flatten()
            .zip(other.probabilities.iter().flatten())
        {
            assert!((a - b).abs() < 1e-14);
        }
        for (a, b) in row
            .reaches
            .iter()
            .flatten()
            .zip(other.reaches.iter().flatten())
        {
            assert!((a - b).abs() < 1e-14);
        }
    }
    assert_eq!(actual.endpoints.len(), expected.endpoints.len());
    for (h, (state, range)) in &actual.endpoints {
        let (other, other_range) = &expected.endpoints[h];
        assert_eq!(state.invested, other.invested);
        for (a, b) in range.iter().flatten().zip(other_range.iter().flatten()) {
            assert!((a - b).abs() < 1e-14);
        }
    }
    fs::remove_file(path).unwrap();
}

#[test]
fn private_class_reduction_precedes_response_maximum() {
    let classes = Classes::new();
    assert_eq!(classes.labels.len(), 169);
    assert_eq!(classes.counts.iter().sum::<usize>(), 1326);
    let k = classes.labels.binary_search(&"AA".to_string()).unwrap();
    let mut a = vec![0.0; 1326];
    let mut b = a.clone();
    let hands: Vec<_> = (0..1326).filter(|c| classes.index[*c] == k).collect();
    for (i, c) in hands.iter().enumerate() {
        a[*c] = if i < 3 { 1.0 } else { -1.0 };
        b[*c] = -a[*c];
    }
    assert!(classes.reduce(&a)[k].abs() < 1e-12);
    assert!(classes.reduce(&b)[k].abs() < 1e-12);
    let clairvoyant: Vec<_> = a.iter().zip(&b).map(|(a, b)| a.max(*b)).collect();
    assert!((classes.reduce(&clairvoyant)[k] - 1.0).abs() < 1e-12);
}

#[test]
#[ignore = "complete frozen preflop response capture; pinned inputs and external resource guard required"]
fn frozen_preflop_response_capture() {
    let preflop = response::FrozenPreflopPolicy::read(Path::new(
        &std::env::var("POKER_NOISE_PREFLOP").unwrap(),
    ))
    .unwrap();
    assert_eq!(
        preflop.artifact_sha256,
        std::env::var("POKER_NOISE_PREFLOP_SHA").unwrap()
    );
    assert_eq!(preflop.game.effective_stack_bb, 20.0);
    assert!([32, 128, 512, 1024].contains(&preflop.rounds));
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
    assert!(index < 4);
    let preflight = match std::env::var("POKER_RESPONSE_PREFLIGHT").unwrap().as_str() {
        "1" => true,
        "0" => false,
        _ => panic!("invalid preflight flag"),
    };
    assert!(!preflight || index == 0);
    let root_turn_averages = std::env::var("POKER_COMPACT_TURN_ROOT_AVERAGES")
        .map(|v| { assert!(v == "0" || v == "1"); v == "1" }).unwrap_or(false);
    let output = PathBuf::from(std::env::var("POKER_COMPACT_OUTPUT").unwrap());
    assert!(!output.exists());
    let started = Instant::now();
    let snapshot = Snapshot::capture_frozen(&preflop).unwrap();
    assert_eq!(snapshot.rows.len(), 100);
    let classes = Classes::new();
    let mut values = fixed_control::values(&snapshot, &exact).unwrap();
    let live: Vec<_> = snapshot
        .endpoints
        .iter()
        .filter(|(_, (s, _))| s.terminal.is_none() && s.street == Street::Flop)
        .map(|(h, _)| h.clone())
        .collect();
    assert_eq!(live.len(), 49);
    let selected = if preflight {
        [2.0, 9.0, 18.0]
            .iter()
            .map(|amount| {
                live.iter()
                    .find(|h| {
                        let state = &snapshot.endpoints[*h].0;
                        state.invested == [*amount, *amount]
                    })
                    .expect("representative preflight endpoint absent")
                    .clone()
            })
            .collect::<Vec<_>>()
    } else {
        live.clone()
    };
    let mut rng = SplitMix64::new(batch_iteration_seed(32001, index + 1));
    let deal = Deal::sample(&mut rng);
    let board: [u8; 3] = deal.board[..3].try_into().unwrap();
    let turns: Vec<u8> = (0..52).filter(|c| !board.contains(c)).collect();
    let turn = turns[rng.index(turns.len())];
    let flop_kernel = exact_flop_kernel(board).unwrap();
    let mut captures = Vec::new();
    for history in &selected {
        let before = Instant::now();
        let (state, prior) = &snapshot.endpoints[history];
        let input = snapshot.flop_input(history, board).unwrap();
        let digest = Sha256::digest(serde_json::to_vec(&input).unwrap());
        let options = NativeFlopOptions {
            seed: 100101 ^ u64::from_le_bytes(digest[..8].try_into().unwrap()),
            iterations: 128,
            training_turn_iterations: 64,
            response_turn_iterations: 64,
        };
        let policy = NativePostflopPolicy::solve_counterfactual_learned_with_turn_averages(
            preflop.game.clone(),
            input,
            &options,
            &model,
            root_turn_averages,
        )
        .unwrap();
        let native = policy
            .sampled_profile_values_with_turn_baseline(turn, &model)
            .unwrap();
        let mut masked = prior.clone();
        mask_ranges(&mut masked, board);
        let totals: [f64; 2] = std::array::from_fn(|p| masked[p].iter().sum());
        let mean = values.get_mut(history).unwrap();
        for p in 0..2 {
            let baseline = flop_kernel
                .values_on_flop(state.invested, &masked[1 - p], p)
                .unwrap();
            for c in 0..1326 {
                mean[p][c] += (native[p][c] * totals[1 - p] - baseline[c]) * FLOP_CHANCE_CORRECTION
                    / OPPONENT_HANDS;
                assert!(mean[p][c].is_finite());
            }
        }
        let record = serde_json::json!({"history":history, "investedBb":state.invested,
            "policySha256":policy.identity(), "seconds":before.elapsed().as_secs_f64()});
        eprintln!(
            "{}",
            serde_json::json!({"stage":"frozen_response_endpoint_complete",
            "boardIndex":index,"completed":captures.len()+1,"total":selected.len(),"record":record})
        );
        captures.push(record);
    }
    let rows: Vec<_> = snapshot.rows.iter().map(|(history,row)| {
        let actions = row.state.legal_actions(&preflop.game);
        serde_json::json!({"history":history,"actor":row.state.actor,
            "actions":actions.iter().map(|a|a.label.clone()).collect::<Vec<_>>(),
            "children":actions.iter().map(|a|row.state.apply(a,&preflop.game).public_history).collect::<Vec<_>>(),
            "probabilities":row.probabilities.iter().map(|v|classes.reduce(v)).collect::<Vec<_>>()})
    }).collect();
    let endpoints: Vec<_> = values
        .iter()
        .filter(|(h, _)| !preflight || !live.contains(h) || selected.contains(h))
        .map(|(history, v)| {
            serde_json::json!({"history":history,
            "cfvBb":[classes.reduce(&v[0]),classes.reduce(&v[1])]})
        })
        .collect();
    let payload = serde_json::json!({"schema":"frozen-preflop-response-capture-v1",
        "preflopSha256":preflop.artifact_sha256,"modelSha256":model.artifact_sha256(),"kernelSha256":kernel_sha,
        "chanceSeed":32001,"boardIndex":index,"board":board,"turn":turn,
        "complete":!preflight,"classes":classes.labels,"multiplicities":classes.counts,
        "rootHistory":GameState::initial(&preflop.game).public_history,
        "rows":rows,"endpoints":endpoints,"captures":captures,
        "continuationFunction":{"flopIterations":128,"trainingTurnIterations":64,"playedTurnIterations":64,
            "rootRealizationTurnAverages":root_turn_averages,
            "completeRootSupport":true,"nativeValueKind":"frozen_profile_not_training_br_completion",
            "turnControl":"all_49_predictions_plus_native_minus_same_turn_prediction",
            "flopControl":"exact_checkdown_mean_plus_native_minus_sampled_checkdown_scale_1"},
        "seconds":started.elapsed().as_secs_f64(),"releaseAccepted":false,
        "interpretation":"All live preflop endpoints on shared uniform public flop/turn draws. Reference reaches and actual frozen postflop profile are unchanged. CFVs integrated over private classes before response selection. Not a full-game exploitability bound."});
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap();
    file.write_all(&serde_json::to_vec(&payload).unwrap())
        .unwrap();
    file.sync_all().unwrap();
}
