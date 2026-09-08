//! Native and learned adapters at the same immutable turn-leaf seam.
//! Predictions are never represented as measured native response diagnostics.
use super::*;

#[derive(Clone, Copy)]
pub(super) enum Evaluator<'a> {
    Native,
    Learned(&'a PublicValueNetwork),
}

#[test]
fn frozen_prediction_probe_preserves_beliefs_and_raw_reach_scaling() {
    let mut model = super::super::super::tests::zero_shared_value_network();
    model.schema = "hu-public-belief-combo-value-network-v4".into();
    model.value_normalization = Some("payoff-exposure".into());
    model.prediction_contract = Some("native-turn-cfv-full-stack-v1".into());
    model.artifact_sha256 = Some("b".repeat(64));
    model.query_tower[0].weights[94] = 1.0;
    model.head[0].activation = "linear".into();
    model.head[0].weights[1] = 0.5;
    let mut game = BlueprintConfig::default();
    game.effective_stack_bb = 20.0;
    // A frozen test-only uniform trunk from the existing pure constructor;
    // prediction observation must not train either street.
    let board = [8, 9, 16];
    let state = PublicBeliefState::flop_start(
        board,
        1,
        [1.0, 1.0],
        std::array::from_fn(|_| uniform_range(&board)),
    );
    let trunk = Trunk::new(game.clone(), state.clone()).unwrap();
    let strategies = trunk
        .nodes
        .iter()
        .map(|(history, node)| PublicBeliefStrategy {
            public_history: history.clone(),
            actor: node.actor,
            action_labels: node.action_labels.clone(),
            probabilities: node
                .average_strategy(&trunk.legal[node.actor])
                .iter()
                .map(|v| *v as f32)
                .collect(),
            action_values_bb: None,
        })
        .collect();
    let candidate = Solution {
        schema: "hu-native-counterfactual-turn-flop-pilot-v1".into(),
        game,
        state,
        seed: 0,
        iterations: 2,
        turn_iterations: 64,
        response_turn_iterations: None,
        strategies,
        turn_queries: 0,
        zero_own_reach_completions: [0, 0],
        maximum_conditional_turn_response_gain_bb: None,
        zero_joint_turn_queries: 0,
        chance_baseline: None,
        turn_samples_per_iteration: None,
        learned_leaf_model_sha256: model.artifact_sha256.clone(),
        complete_root_support: None,
        root_realization_turn_averages: None,
    };
    let before = serde_json::to_vec(&candidate).unwrap();
    let frozen = frozen_response::Frozen::new(&candidate).unwrap();
    let queries = frozen.belief_queries(20).unwrap();
    let packet = frozen_prediction_packet(&frozen, &model, 20).unwrap();
    assert!(!queries.is_empty());
    assert_eq!(packet.leaves.len(), queries.len());
    for (leaf, config) in packet.leaves.iter().zip(queries) {
        assert_eq!(leaf.history, config.state.public_history);
        let conditional = model.predict_native_turn(&config).unwrap();
        for p in 0..2 {
            let masses =
                compatible_masses_from_card_marginals(&all_combos(), &config.state.ranges[1 - p]);
            for c in 0..COMBO_COUNT {
                assert_eq!(
                    leaf.predicted_counterfactual_bb[p][c],
                    conditional[p][c] * masses[c]
                );
            }
        }
    }
    assert_eq!(before, serde_json::to_vec(&candidate).unwrap());
    assert!(frozen_prediction_packet(&frozen, &model, board[0]).is_err());
}

#[derive(Serialize)]
struct PredictedFrozenLeaf {
    history: Vec<String>,
    predicted_counterfactual_bb: [Vec<f64>; 2],
}

#[derive(Serialize)]
struct PredictedFrozenPacket {
    turn: u8,
    leaves: Vec<PredictedFrozenLeaf>,
}

fn frozen_prediction_packet(
    frozen: &frozen_response::Frozen,
    model: &PublicValueNetwork,
    turn: u8,
) -> Result<PredictedFrozenPacket, String> {
    let evaluator = Evaluator::Learned(model);
    evaluator.model_sha256()?;
    let leaves = frozen
        .belief_queries(turn)?
        .into_iter()
        .map(|config| {
            let history = config.state.public_history.clone();
            match evaluator.evaluate(config)? {
                Leaf::Predicted(predicted_counterfactual_bb) => Ok(PredictedFrozenLeaf {
                    history,
                    predicted_counterfactual_bb,
                }),
                Leaf::Native(_) => Err("prediction diagnostic cannot call native solver".into()),
            }
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(PredictedFrozenPacket { turn, leaves })
}

#[test]
#[ignore = "hash-pinned frozen candidate/model and resource guard; prediction-only diagnostic, no native solve"]
fn saved_frozen_prediction_probe() {
    let path = PathBuf::from(std::env::var("POKER_FROZEN_PREDICT_CANDIDATE").unwrap());
    assert!(fs::metadata(&path).unwrap().len() <= 8 * 1024 * 1024);
    let bytes = fs::read(path).unwrap();
    let digest = format!("{:x}", Sha256::digest(&bytes));
    assert_eq!(
        digest,
        std::env::var("POKER_FROZEN_PREDICT_CANDIDATE_SHA").unwrap()
    );
    let candidate: Solution = serde_json::from_slice(&bytes).unwrap();
    let model = PublicValueNetwork::read(Path::new(
        &std::env::var("POKER_FROZEN_PREDICT_MODEL").unwrap(),
    ))
    .unwrap();
    let model_sha = model.artifact_sha256().unwrap();
    assert_eq!(
        model_sha,
        std::env::var("POKER_FROZEN_PREDICT_MODEL_SHA").unwrap()
    );
    let alternative = std::env::var("POKER_FROZEN_PREDICT_ALTERNATIVE").unwrap_or_default();
    assert!(alternative.is_empty() || alternative == "1");
    if alternative.is_empty() {
        assert_eq!(candidate.learned_leaf_model_sha256.as_deref(), Some(model_sha));
    }
    let frozen = frozen_response::Frozen::new(&candidate).unwrap();
    let started = std::time::Instant::now();
    let packets = (0..52u8)
        .filter(|c| !candidate.state.board.contains(c))
        .map(|turn| frozen_prediction_packet(&frozen, &model, turn).unwrap())
        .collect::<Vec<_>>();
    let output = PathBuf::from(std::env::var("POKER_FROZEN_PREDICT_OUTPUT").unwrap());
    let payload = serde_json::to_vec(
        &serde_json::json!({"schema":"hu-native-flop-frozen-leaf-predictions-v1",
        "candidate_sha256":digest,"model_sha256":model_sha,"packets":packets,
        "source_policy_model_sha256":candidate.learned_leaf_model_sha256,
        "prediction_role":if alternative.is_empty() { "same_policy_model" } else { "alternative_value_model" },
        "seconds":started.elapsed().as_secs_f64(),"releaseAccepted":false}),
    )
    .unwrap();
    assert!(payload.len() <= 64 * 1024 * 1024);
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap();
    file.write_all(&payload).unwrap();
    file.sync_all().unwrap();
}

pub(super) enum Leaf {
    Native(Values),
    Predicted([Vec<f64>; 2]),
}

impl Evaluator<'_> {
    pub fn model_sha256(&self) -> Result<Option<String>, String> {
        match self {
            Self::Native => Ok(None),
            Self::Learned(model) => {
                model.validate()?;
                if model.prediction_contract.as_deref() != Some("native-turn-cfv-full-stack-v1") {
                    return Err("learned native leaf lacks its explicit prediction contract".into());
                }
                model
                    .artifact_sha256()
                    .map(|s| Some(s.to_owned()))
                    .ok_or_else(|| "learned continuation needs a frozen artifact hash".into())
            }
        }
    }

    pub fn evaluate(&self, config: TurnRiverSolveConfig) -> Result<Leaf, String> {
        match self {
            Self::Native => solve(config).map(Leaf::Native),
            Self::Learned(model) => {
                let conditional = model.predict_native_turn(&config)?;
                let combos = all_combos();
                // The network consumes normalized beliefs. CFR consumes raw
                // opponent-reach-scaled values. Own reach is never a multiplier.
                let raw = std::array::from_fn(|p| {
                    let masses =
                        compatible_masses_from_card_marginals(&combos, &config.state.ranges[1 - p]);
                    conditional[p]
                        .iter()
                        .zip(masses)
                        .map(|(ev, mass)| ev * mass)
                        .collect()
                });
                Ok(Leaf::Predicted(raw))
            }
        }
    }
}

#[test]
#[ignore = "hash-pinned native model/corpus and external resource guard; exports actual Rust predictions for parity"]
fn saved_native_prediction_probe() {
    let corpus_path = PathBuf::from(std::env::var("POKER_NATIVE_PREDICT_DATASET").unwrap());
    let source = fs::read(corpus_path).unwrap();
    assert!(source.len() < 128 * 1024 * 1024);
    assert_eq!(
        format!("{:x}", Sha256::digest(&source)),
        std::env::var("POKER_NATIVE_PREDICT_DATASET_SHA").unwrap()
    );
    let mut decoded = Vec::new();
    flate2::read::GzDecoder::new(source.as_slice())
        .take(256 * 1024 * 1024 + 1)
        .read_to_end(&mut decoded)
        .unwrap();
    assert!(decoded.len() <= 256 * 1024 * 1024);
    let corpus: serde_json::Value = serde_json::from_slice(&decoded).unwrap();
    assert_eq!(corpus["schema"], "hu-native-turn-cfv-dataset-v1");
    let game: BlueprintConfig = serde_json::from_value(corpus["game"].clone()).unwrap();
    let model = PublicValueNetwork::read(Path::new(
        &std::env::var("POKER_NATIVE_PREDICT_MODEL").unwrap(),
    ))
    .unwrap();
    assert_eq!(
        model.artifact_sha256().unwrap(),
        std::env::var("POKER_NATIVE_PREDICT_MODEL_SHA").unwrap()
    );
    let started = std::time::Instant::now();
    let predictions = corpus["targets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|target| {
            let config = TurnRiverSolveConfig {
                game: game.clone(),
                state: serde_json::from_value(target["public_state"].clone()).unwrap(),
                iterations: 64,
                averaging_delay: 0,
                river_refinement_iterations: 0,
                regret_matching_plus: false,
            };
            model.predict_native_turn(&config).unwrap()
        })
        .collect::<Vec<_>>();
    let output = PathBuf::from(std::env::var("POKER_NATIVE_PREDICT_OUTPUT").unwrap());
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap();
    serde_json::to_writer(&mut file, &serde_json::json!({"modelSha256":model.artifact_sha256(),
        "seconds":started.elapsed().as_secs_f64(),"predictions":predictions,"releaseAccepted":false})).unwrap();
    file.sync_all().unwrap();
}
