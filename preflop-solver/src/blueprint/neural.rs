//! Bounded Deep-CFR trajectory generation for the neural training pipeline.
//!
//! The exact game remains in Rust. This module freezes the neural policy for a
//! traversal batch, performs deterministic external sampling, and writes
//! compact gzip JSONL. The ML trainer expands records into the browser's pinned
//! suit-canonical 716-state + 9-action feature schema only when a minibatch is
//! needed.

use super::*;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;

pub const STATE_FEATURE_COUNT: usize = 716;
pub const ACTION_FEATURE_COUNT: usize = 9;
pub const MODEL_INPUT_COUNT: usize = STATE_FEATURE_COUNT + ACTION_FEATURE_COUNT;
pub const MAX_TRAJECTORY_ACTIONS: usize = 32;

const DATASET_SCHEMA: &str = "hu-neural-traversal-jsonl-v6";
const TRAINING_NETWORK_SCHEMA: &str = "hu-neural-training-networks-v4";
const POKER_FEATURE_OFFSET: usize = 604;
const TEXTURE_FEATURE_OFFSET: usize = 652;
const TEXTURE_FEATURE_COUNT: usize = 64;

#[derive(Clone, Debug)]
pub struct SampleGenerationConfig {
    pub game: BlueprintConfig,
    pub traversals: u64,
    pub start_iteration: u64,
    pub seed: u64,
    pub max_records: usize,
    pub output: PathBuf,
    pub network_path: Option<PathBuf>,
    pub trajectory_sampling: bool,
    pub value_rollouts_per_action: u32,
}

#[derive(Clone, Debug, Deserialize)]
struct TrainingNetworkBundle {
    schema: String,
    input_size: usize,
    strategy_transform: StrategyTransform,
    networks: Vec<DenseScorer>,
    #[serde(default)]
    postflop_networks: Option<Vec<DenseScorer>>,
    #[serde(default)]
    sampling_baseline: Option<DenseScorer>,
    #[serde(default)]
    sampling_baseline_scale: Option<f64>,
}

impl TrainingNetworkBundle {
    fn policy_network(&self, street: Street, actor: usize) -> &DenseScorer {
        let networks = if street == Street::Preflop {
            &self.networks
        } else {
            self.postflop_networks.as_ref().unwrap_or(&self.networks)
        };
        &networks[actor]
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum StrategyTransform {
    RegretMatching,
    Softmax,
}

#[derive(Clone, Debug, Deserialize)]
struct DenseScorer {
    layers: Vec<DenseLayer>,
}

#[derive(Clone, Debug, Deserialize)]
struct DenseLayer {
    input_size: usize,
    output_size: usize,
    activation: DenseActivation,
    weights: Vec<f32>,
    biases: Vec<f32>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum DenseActivation {
    Linear,
    Relu,
    Tanh,
}

impl DenseScorer {
    fn validate(&self, expected_input: usize, expected_output: usize) -> Result<(), String> {
        if self.layers.is_empty() {
            return Err("scoring network has no layers".to_owned());
        }
        let mut input_size = expected_input;
        for (index, layer) in self.layers.iter().enumerate() {
            if layer.input_size != input_size || layer.output_size == 0 {
                return Err(format!(
                    "scoring network layer {index} has an invalid shape"
                ));
            }
            if layer.weights.len() != layer.input_size * layer.output_size
                || layer.biases.len() != layer.output_size
            {
                return Err(format!(
                    "scoring network layer {index} has an invalid parameter count"
                ));
            }
            if layer
                .weights
                .iter()
                .chain(&layer.biases)
                .any(|value| !value.is_finite())
            {
                return Err(format!(
                    "scoring network layer {index} contains non-finite parameters"
                ));
            }
            input_size = layer.output_size;
        }
        if input_size != expected_output {
            return Err(format!(
                "scoring network must produce {expected_output} values"
            ));
        }
        Ok(())
    }

    fn score_state_actions(&self, state: &[f32], actions: &[Vec<f32>]) -> Vec<f64> {
        debug_assert_eq!(state.len(), STATE_FEATURE_COUNT);
        debug_assert!(actions
            .iter()
            .all(|action| action.len() == ACTION_FEATURE_COUNT));
        let first = &self.layers[0];
        debug_assert_eq!(first.input_size, MODEL_INPUT_COUNT);
        let nonzero_state = state
            .iter()
            .enumerate()
            .filter(|(_, value)| **value != 0.0)
            .collect::<Vec<_>>();
        let mut batch = vec![vec![0.0f32; first.output_size]; actions.len()];
        for row in 0..first.output_size {
            let offset = row * first.input_size;
            let mut shared = first.biases[row];
            for (column, value) in &nonzero_state {
                shared += first.weights[offset + *column] * **value;
            }
            for (action_index, action) in actions.iter().enumerate() {
                let mut sum = shared;
                for (column, value) in action.iter().enumerate() {
                    sum += first.weights[offset + STATE_FEATURE_COUNT + column] * value;
                }
                batch[action_index][row] = activate_dense(sum, first.activation);
            }
        }
        for layer in self.layers.iter().skip(1) {
            let mut next = vec![vec![0.0f32; layer.output_size]; actions.len()];
            for (values, output) in batch.iter().zip(&mut next) {
                for (row, value) in output.iter_mut().enumerate() {
                    let offset = row * layer.input_size;
                    let mut sum = layer.biases[row];
                    for column in 0..layer.input_size {
                        sum += layer.weights[offset + column] * values[column];
                    }
                    *value = activate_dense(sum, layer.activation);
                }
            }
            batch = next;
        }
        batch.into_iter().map(|values| values[0] as f64).collect()
    }
}

fn activate_dense(value: f32, activation: DenseActivation) -> f32 {
    match activation {
        DenseActivation::Linear => value,
        DenseActivation::Relu => value.max(0.0),
        DenseActivation::Tanh => value.tanh(),
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum SampleKind {
    AdvantageP0,
    AdvantageP1,
    AverageStrategy,
}

#[derive(Clone, Debug, Serialize)]
struct CompactState {
    private_cards: [u8; 2],
    board: Vec<u8>,
    street: Street,
    actor: usize,
    button: usize,
    pot_bb: f32,
    stacks_bb: [f32; 2],
    street_bets_bb: [f32; 2],
    total_committed_bb: [f32; 2],
    to_call_bb: f32,
    last_full_raise_bb: f32,
    raise_reopened: bool,
    trajectory: Vec<CompactTrajectoryAction>,
}

#[derive(Clone, Debug, Serialize)]
struct CompactTrajectoryAction {
    actor: usize,
    street: Street,
    kind: TrajectoryActionKind,
    amount_bb: f32,
    amount_to_bb: Option<f32>,
    pot_after_bb: f32,
}

#[derive(Clone, Debug, Serialize)]
struct CompactLegalAction {
    kind: TrajectoryActionKind,
    amount_to_bb: Option<f32>,
}

#[derive(Clone, Debug, Serialize)]
struct TrainingSample {
    kind: SampleKind,
    iteration: u64,
    weight: f32,
    reach_probability: f32,
    state: CompactState,
    actions: Vec<CompactLegalAction>,
    feature_sha256: Vec<String>,
    targets: Vec<f32>,
    action_values_bb: Option<Vec<f32>>,
}

#[derive(Clone, Debug, Serialize)]
struct DatasetMetadata<'a> {
    record_type: &'static str,
    schema: &'static str,
    state_feature_schema: &'static str,
    state_feature_count: usize,
    action_feature_schema: &'static str,
    action_feature_count: usize,
    depth_bb: f64,
    seed: u64,
    start_iteration: u64,
    traversals: u64,
    records: usize,
    truncated: bool,
    sampling_mode: &'static str,
    value_rollouts_per_action: u32,
    action_abstraction: &'a ActionAbstraction,
}

struct SampleGenerator {
    config: SampleGenerationConfig,
    networks: Option<TrainingNetworkBundle>,
    rng: SplitMix64,
    records: Vec<TrainingSample>,
    attempted_records: usize,
}

impl SampleGenerator {
    fn new(config: SampleGenerationConfig) -> Result<Self, Box<dyn Error>> {
        config
            .game
            .validate()
            .map_err(|error| format!("invalid neural game config: {error}"))?;
        if config.traversals == 0 {
            return Err("neural traversals must be positive".into());
        }
        if config.max_records == 0 {
            return Err("neural record limit must be positive".into());
        }
        if config.value_rollouts_per_action == 0 {
            return Err("value rollouts per action must be positive".into());
        }
        let networks = match &config.network_path {
            Some(path) => {
                let file = fs::File::open(path)?;
                let bundle: TrainingNetworkBundle = serde_json::from_reader(BufReader::new(file))?;
                if bundle.schema != TRAINING_NETWORK_SCHEMA
                    || bundle.input_size != MODEL_INPUT_COUNT
                    || bundle.networks.len() != 2
                {
                    return Err("training network bundle is incompatible".into());
                }
                for network in &bundle.networks {
                    network.validate(MODEL_INPUT_COUNT, 1)?;
                }
                if let Some(networks) = &bundle.postflop_networks {
                    if networks.len() != 2 {
                        return Err("postflop training network bundle is incompatible".into());
                    }
                    for network in networks {
                        network.validate(MODEL_INPUT_COUNT, 1)?;
                    }
                }
                if let Some(baseline) = &bundle.sampling_baseline {
                    baseline.validate(MODEL_INPUT_COUNT, 1)?;
                    let scale = bundle.sampling_baseline_scale.unwrap_or(0.0);
                    if !scale.is_finite() || !(0.0..=1.0).contains(&scale) {
                        return Err("sampling baseline scale must be between zero and one".into());
                    }
                }
                Some(bundle)
            }
            None => None,
        };
        Ok(Self {
            rng: SplitMix64::new(config.seed),
            records: Vec::with_capacity(config.max_records.min(65_536)),
            attempted_records: 0,
            config,
            networks,
        })
    }

    fn run(
        mut self,
    ) -> Result<(SampleGenerationConfig, Vec<TrainingSample>, usize), Box<dyn Error>> {
        for offset in 0..self.config.traversals {
            let iteration = self.config.start_iteration + offset;
            let deal = Deal::sample(&mut self.rng);
            if self.config.trajectory_sampling {
                self.sample_trajectory(
                    GameState::initial(&self.config.game),
                    &deal,
                    iteration,
                    1.0,
                );
            } else {
                let traverser = iteration as usize % 2;
                self.external_sampling(
                    GameState::initial(&self.config.game),
                    &deal,
                    traverser,
                    iteration,
                    1.0,
                );
            }
        }
        Ok((self.config, self.records, self.attempted_records))
    }

    fn current_strategy(
        &self,
        state: &GameState,
        deal: &Deal,
        actions: &[LegalAction],
    ) -> Vec<f64> {
        let Some(bundle) = &self.networks else {
            return vec![1.0 / actions.len() as f64; actions.len()];
        };
        let network = bundle.policy_network(state.street, state.actor);
        let state_features = encode_state_features(state, deal, &self.config.game);
        let action_features = actions
            .iter()
            .map(|action| encode_action_features(state, action, &self.config.game))
            .collect::<Vec<_>>();
        let scores = network.score_state_actions(&state_features, &action_features);
        match bundle.strategy_transform {
            StrategyTransform::RegretMatching => {
                normalize_or_uniform(scores.into_iter().map(|value| value.max(0.0)).collect())
            }
            StrategyTransform::Softmax => stable_softmax(&scores),
        }
    }

    fn sampled_value_baseline(
        &self,
        state: &GameState,
        deal: &Deal,
        actions: &[LegalAction],
        traverser: usize,
    ) -> Option<Vec<f64>> {
        let bundle = self.networks.as_ref()?;
        let baseline = bundle.sampling_baseline.as_ref()?;
        let scale = bundle.sampling_baseline_scale.unwrap_or(0.0);
        if scale <= 0.0 {
            return None;
        }
        let state_features = encode_state_features(state, deal, &self.config.game);
        let action_features = actions
            .iter()
            .map(|action| encode_action_features(state, action, &self.config.game))
            .collect::<Vec<_>>();
        // The action-value network is fitted at traverser decisions, so its
        // target is always the acting player's utility. Convert it to the
        // current traverser's perspective at sampled opponent decisions.
        let perspective = if state.actor == traverser { 1.0 } else { -1.0 };
        let limit = self.config.game.effective_stack_bb;
        Some(
            baseline
                .score_state_actions(&state_features, &action_features)
                .into_iter()
                .map(|value| value.clamp(-limit, limit) * perspective * scale)
                .collect(),
        )
    }

    fn push_record(&mut self, sample: TrainingSample) {
        self.attempted_records += 1;
        if self.records.len() < self.config.max_records {
            self.records.push(sample);
        }
    }

    fn value_only_external_sampling(
        &self,
        state: GameState,
        deal: &Deal,
        traverser: usize,
        rng: &mut SplitMix64,
    ) -> f64 {
        if state.terminal.is_some() {
            let utility = state.utility_p0(deal, &self.config.game);
            return if traverser == 0 { utility } else { -utility };
        }
        let actions = state.legal_actions(&self.config.game);
        debug_assert!(!actions.is_empty());
        let strategy = self.current_strategy(&state, deal, &actions);
        if state.actor == traverser {
            actions
                .iter()
                .enumerate()
                .map(|(index, action)| {
                    strategy[index]
                        * self.value_only_external_sampling(
                            state.apply(action, &self.config.game),
                            deal,
                            traverser,
                            rng,
                        )
                })
                .sum()
        } else {
            let selected = sample_index(&strategy, rng);
            let baselines = self.sampled_value_baseline(&state, deal, &actions, traverser);
            let sampled_value = self.value_only_external_sampling(
                state.apply(&actions[selected], &self.config.game),
                deal,
                traverser,
                rng,
            );
            match baselines {
                Some(values) => {
                    baseline_corrected_sample(&strategy, &values, selected, sampled_value)
                }
                None => sampled_value,
            }
        }
    }

    fn action_value_targets(
        &self,
        state: &GameState,
        deal: &Deal,
        actions: &[LegalAction],
        traverser: usize,
        iteration: u64,
        primary_values: &[f64],
    ) -> Vec<f64> {
        let samples = self.config.value_rollouts_per_action;
        if samples == 1 {
            return primary_values.to_vec();
        }
        actions
            .iter()
            .enumerate()
            .map(|(action_index, action)| {
                let mut total = primary_values[action_index];
                for sample_index in 1..samples {
                    let mut rng = SplitMix64::new(value_rollout_seed(
                        &self.config,
                        state,
                        deal,
                        action,
                        iteration,
                        sample_index,
                    ));
                    total += self.value_only_external_sampling(
                        state.apply(action, &self.config.game),
                        deal,
                        traverser,
                        &mut rng,
                    );
                }
                total / samples as f64
            })
            .collect()
    }

    fn sample_trajectory(
        &mut self,
        state: GameState,
        deal: &Deal,
        iteration: u64,
        reach_probability: f64,
    ) {
        if state.terminal.is_some() {
            return;
        }
        let actions = state.legal_actions(&self.config.game);
        debug_assert!(!actions.is_empty());
        let strategy = self.current_strategy(&state, deal, &actions);
        self.push_record(training_sample(
            SampleKind::AverageStrategy,
            iteration,
            1.0,
            reach_probability,
            &state,
            deal,
            &actions,
            strategy.clone(),
            None,
            &self.config.game,
        ));
        let selected = sample_index(&strategy, &mut self.rng);
        self.sample_trajectory(
            state.apply(&actions[selected], &self.config.game),
            deal,
            iteration,
            reach_probability * strategy[selected],
        );
    }

    fn external_sampling(
        &mut self,
        state: GameState,
        deal: &Deal,
        traverser: usize,
        iteration: u64,
        reach_probability: f64,
    ) -> f64 {
        if state.terminal.is_some() {
            let utility = state.utility_p0(deal, &self.config.game);
            return if traverser == 0 { utility } else { -utility };
        }

        let actions = state.legal_actions(&self.config.game);
        debug_assert!(!actions.is_empty());
        let strategy = self.current_strategy(&state, deal, &actions);
        if state.actor == traverser {
            let mut values = Vec::with_capacity(actions.len());
            for action in &actions {
                values.push(self.external_sampling(
                    state.apply(action, &self.config.game),
                    deal,
                    traverser,
                    iteration,
                    reach_probability * strategy[values.len()],
                ));
            }
            let node_value = strategy
                .iter()
                .zip(&values)
                .map(|(probability, value)| probability * value)
                .sum::<f64>();
            let action_value_targets =
                self.action_value_targets(&state, deal, &actions, traverser, iteration, &values);
            self.push_record(training_sample(
                if traverser == 0 {
                    SampleKind::AdvantageP0
                } else {
                    SampleKind::AdvantageP1
                },
                iteration,
                1.0,
                reach_probability,
                &state,
                deal,
                &actions,
                values.iter().map(|value| value - node_value).collect(),
                Some(action_value_targets),
                &self.config.game,
            ));
            node_value
        } else {
            let iteration_weight = ((iteration + 1) as f64)
                .powf(self.config.game.dcfr.strategy_exponent)
                .min(f32::MAX as f64) as f32;
            self.push_record(training_sample(
                SampleKind::AverageStrategy,
                iteration,
                iteration_weight,
                reach_probability,
                &state,
                deal,
                &actions,
                strategy.clone(),
                None,
                &self.config.game,
            ));
            let selected = sample_index(&strategy, &mut self.rng);
            let baselines = self.sampled_value_baseline(&state, deal, &actions, traverser);
            let sampled_value = self.external_sampling(
                state.apply(&actions[selected], &self.config.game),
                deal,
                traverser,
                iteration,
                reach_probability * strategy[selected],
            );
            match baselines {
                Some(values) => {
                    baseline_corrected_sample(&strategy, &values, selected, sampled_value)
                }
                None => sampled_value,
            }
        }
    }
}

fn baseline_corrected_sample(
    strategy: &[f64],
    baseline_values: &[f64],
    selected: usize,
    sampled_value: f64,
) -> f64 {
    debug_assert_eq!(strategy.len(), baseline_values.len());
    debug_assert!(selected < strategy.len());
    let baseline_node_value = strategy
        .iter()
        .zip(baseline_values)
        .map(|(probability, baseline)| probability * baseline)
        .sum::<f64>();
    baseline_node_value + sampled_value - baseline_values[selected]
}

fn value_rollout_seed(
    config: &SampleGenerationConfig,
    state: &GameState,
    deal: &Deal,
    action: &LegalAction,
    iteration: u64,
    sample_index: u32,
) -> u64 {
    let mut digest = Sha256::new();
    digest.update(b"hu-action-value-rollout-v1");
    digest.update(config.seed.to_le_bytes());
    digest.update(iteration.to_le_bytes());
    digest.update(sample_index.to_le_bytes());
    for card in deal.holes.iter().flatten().chain(&deal.board) {
        digest.update([*card]);
    }
    for feature in encode_state_action(state, deal, action, &config.game) {
        let canonical_micro_units = (feature as f64 * 1_000_000.0).round() as i32;
        digest.update(canonical_micro_units.to_le_bytes());
    }
    let bytes = digest.finalize();
    u64::from_le_bytes(
        bytes[..8]
            .try_into()
            .expect("SHA-256 prefix has eight bytes"),
    )
}

fn training_sample(
    kind: SampleKind,
    iteration: u64,
    weight: f32,
    reach_probability: f64,
    state: &GameState,
    deal: &Deal,
    actions: &[LegalAction],
    targets: Vec<f64>,
    action_values: Option<Vec<f64>>,
    config: &BlueprintConfig,
) -> TrainingSample {
    let feature_sha256 = actions
        .iter()
        .map(|action| feature_sha256(&encode_state_action(state, deal, action, config)))
        .collect();
    TrainingSample {
        kind,
        iteration,
        weight,
        reach_probability: reach_probability as f32,
        state: compact_state(state, deal, config),
        actions: actions
            .iter()
            .map(|action| compact_action(state, action, config))
            .collect(),
        feature_sha256,
        targets: targets.into_iter().map(|value| value as f32).collect(),
        action_values_bb: action_values
            .map(|values| values.into_iter().map(|value| value as f32).collect()),
    }
}

fn feature_sha256(features: &[f32]) -> String {
    let mut digest = Sha256::new();
    for feature in features {
        let canonical_micro_units = (*feature as f64 * 1_000_000.0).round() as i32;
        digest.update(canonical_micro_units.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn stable_softmax(values: &[f64]) -> Vec<f64> {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exponentials = values
        .iter()
        .map(|value| (value - maximum).exp())
        .collect::<Vec<_>>();
    normalize_or_uniform(exponentials)
}

fn compact_state(state: &GameState, deal: &Deal, config: &BlueprintConfig) -> CompactState {
    let settled_pot = state.pot() - state.street_invested[0] - state.street_invested[1];
    CompactState {
        private_cards: deal.holes[state.actor],
        board: deal.board[..state.street.board_len()].to_vec(),
        street: state.street,
        actor: state.actor,
        button: 0,
        pot_bb: settled_pot as f32,
        stacks_bb: [
            state.remaining(0, config) as f32,
            state.remaining(1, config) as f32,
        ],
        street_bets_bb: [
            state.street_invested[0] as f32,
            state.street_invested[1] as f32,
        ],
        total_committed_bb: [state.invested[0] as f32, state.invested[1] as f32],
        to_call_bb: state.to_call() as f32,
        last_full_raise_bb: state.last_full_raise as f32,
        raise_reopened: state.raise_reopened,
        trajectory: state
            .trajectory
            .iter()
            .map(|action| CompactTrajectoryAction {
                actor: action.actor,
                street: action.street,
                kind: action.kind,
                amount_bb: action.amount_bb as f32,
                amount_to_bb: action.amount_to_bb.map(|value| value as f32),
                pot_after_bb: action.pot_after_bb as f32,
            })
            .collect(),
    }
}

fn action_kind(
    state: &GameState,
    action: &LegalAction,
    config: &BlueprintConfig,
) -> TrajectoryActionKind {
    match action.kind {
        ActionKind::Fold => TrajectoryActionKind::Fold,
        ActionKind::Check => TrajectoryActionKind::Check,
        ActionKind::Call => TrajectoryActionKind::Call,
        ActionKind::RaiseTo(target)
            if (target
                - (state.street_invested[state.actor] + state.remaining(state.actor, config)))
            .abs()
                <= EPSILON =>
        {
            TrajectoryActionKind::AllIn
        }
        ActionKind::RaiseTo(_)
            if state.street_invested[0].max(state.street_invested[1]) <= EPSILON =>
        {
            TrajectoryActionKind::Bet
        }
        ActionKind::RaiseTo(_) => TrajectoryActionKind::Raise,
    }
}

fn compact_action(
    state: &GameState,
    action: &LegalAction,
    config: &BlueprintConfig,
) -> CompactLegalAction {
    CompactLegalAction {
        kind: action_kind(state, action, config),
        amount_to_bb: match action.kind {
            ActionKind::RaiseTo(target) => Some(target as f32),
            _ => None,
        },
    }
}

fn encode_state_action(
    state: &GameState,
    deal: &Deal,
    action: &LegalAction,
    config: &BlueprintConfig,
) -> Vec<f32> {
    let mut features = encode_state_features(state, deal, config);
    features.extend(encode_action_features(state, action, config));
    debug_assert_eq!(features.len(), MODEL_INPUT_COUNT);
    features
}

fn encode_state_features(state: &GameState, deal: &Deal, config: &BlueprintConfig) -> Vec<f32> {
    assert!(state.trajectory.len() <= MAX_TRAJECTORY_ACTIONS);
    let depth = config.effective_stack_bb as f32;
    let mut features = vec![0.0f32; STATE_FEATURE_COUNT];
    let visible_board = &deal.board[..state.street.board_len()];
    let suit_map = canonical_suit_map(deal.holes[state.actor], visible_board);
    for card in deal.holes[state.actor] {
        features[canonical_card(card, &suit_map)] = 1.0;
    }
    for card in visible_board {
        features[52 + canonical_card(*card, &suit_map)] = 1.0;
    }
    features[104 + street_index(state.street)] = 1.0;
    features[108 + state.actor] = 1.0;
    features[110] = 1.0; // Player zero is always BTN/SB in this game indexing.
    let opponent = 1 - state.actor;
    let settled_pot = state.pot() - state.street_invested[0] - state.street_invested[1];
    let scalars = [
        settled_pot as f32 / depth,
        state.remaining(state.actor, config) as f32 / depth,
        state.remaining(opponent, config) as f32 / depth,
        state.street_invested[state.actor] as f32 / depth,
        state.street_invested[opponent] as f32 / depth,
        state.invested[state.actor] as f32 / depth,
        state.invested[opponent] as f32 / depth,
        state.to_call() as f32 / depth,
        state.last_full_raise as f32 / depth,
        f32::from(state.raise_reopened),
        state.street.board_len() as f32 / 5.0,
        state.trajectory.len() as f32 / MAX_TRAJECTORY_ACTIONS as f32,
    ];
    features[112..124].copy_from_slice(&scalars);
    for (index, history) in state.trajectory.iter().enumerate() {
        let offset = 124 + index * 15;
        features[offset + history.actor] = 1.0;
        features[offset + 2 + street_index(history.street)] = 1.0;
        features[offset + 6 + trajectory_kind_index(history.kind)] = 1.0;
        features[offset + 12] = history.amount_bb as f32 / depth;
        features[offset + 13] = history.amount_to_bb.unwrap_or(0.0) as f32 / depth;
        features[offset + 14] = history.pot_after_bb as f32 / depth;
    }

    for card in deal.holes[state.actor] {
        let rank = (card / 4) as usize;
        let suit = suit_map[(card % 4) as usize];
        features[POKER_FEATURE_OFFSET + rank] += 0.5;
        features[POKER_FEATURE_OFFSET + 26 + rank] += 0.25;
        features[POKER_FEATURE_OFFSET + 43 + suit] += 1.0 / 7.0;
    }
    for card in visible_board {
        let rank = (*card / 4) as usize;
        let suit = suit_map[(*card % 4) as usize];
        features[POKER_FEATURE_OFFSET + 13 + rank] += 0.25;
        features[POKER_FEATURE_OFFSET + 26 + rank] += 0.25;
        features[POKER_FEATURE_OFFSET + 39 + suit] += 0.2;
        features[POKER_FEATURE_OFFSET + 43 + suit] += 1.0 / 7.0;
    }
    features[POKER_FEATURE_OFFSET + 47] =
        f32::from(deal.holes[state.actor][0] % 4 == deal.holes[state.actor][1] % 4);
    encode_texture_features(
        deal.holes[state.actor],
        visible_board,
        state.street,
        &mut features[TEXTURE_FEATURE_OFFSET..TEXTURE_FEATURE_OFFSET + TEXTURE_FEATURE_COUNT],
    );

    features
}

/// Cheap, exact, suit-invariant poker concepts. These deliberately supplement
/// rather than replace the canonical exact-card encoding above. Every offset
/// is mirrored by the Python trainer and TypeScript runtime.
fn encode_texture_features(
    private_cards: [u8; 2],
    board: &[u8],
    street: Street,
    output: &mut [f32],
) {
    debug_assert_eq!(output.len(), TEXTURE_FEATURE_COUNT);
    if board.is_empty() {
        output[30] = f32::from(private_cards[0] / 4 == private_cards[1] / 4);
        return;
    }

    output[0] = 1.0;
    let mut cards = Vec::with_capacity(board.len() + 2);
    cards.extend_from_slice(&private_cards);
    cards.extend_from_slice(board);
    let category = (evaluate(&cards) >> 24) as usize;
    output[1 + category] = 1.0;

    let mut board_rank_counts = [0u8; 13];
    let mut board_suit_counts = [0u8; 4];
    let mut board_rank_mask = 0u16;
    for card in board {
        let rank = (card / 4) as usize;
        board_rank_counts[rank] += 1;
        board_suit_counts[(card % 4) as usize] += 1;
        board_rank_mask |= 1 << rank;
    }
    let board_max_rank = board_rank_counts.iter().copied().max().unwrap_or(0) as usize;
    let board_max_suit = board_suit_counts.iter().copied().max().unwrap_or(0) as usize;
    let board_density = straight_window_density(board_rank_mask) as usize;
    output[10 + board_max_rank.saturating_sub(1).min(3)] = 1.0;
    output[14 + board_max_suit.saturating_sub(1).min(4)] = 1.0;
    output[19 + board_density.saturating_sub(1).min(4)] = 1.0;

    let board_high = board
        .iter()
        .map(|card| card / 4)
        .max()
        .expect("postflop board has a high card");
    let board_low = board
        .iter()
        .map(|card| card / 4)
        .min()
        .expect("postflop board has a low card");
    let high_band = match board_high {
        10..=12 => 2,
        7..=9 => 1,
        _ => 0,
    };
    output[24 + high_band] = 1.0;

    let hole_ranks = [private_cards[0] / 4, private_cards[1] / 4];
    let overcards = hole_ranks.iter().filter(|rank| **rank > board_high).count();
    output[27 + overcards.min(2)] = 1.0;
    let pocket_pair = hole_ranks[0] == hole_ranks[1];
    output[30] = f32::from(pocket_pair);
    output[31] = f32::from(pocket_pair && hole_ranks[0] > board_high);
    let matches = hole_ranks.map(|rank| board_rank_counts[rank as usize] > 0);
    output[32] = f32::from(hole_ranks.iter().any(|rank| *rank == board_high));
    output[33] = f32::from(
        hole_ranks
            .iter()
            .any(|rank| *rank != board_high && *rank != board_low && matches_rank(board, *rank)),
    );
    output[34] =
        f32::from(board_low != board_high && hole_ranks.iter().any(|rank| *rank == board_low));
    output[35] = f32::from(matches[0] && matches[1]);
    output[36] = f32::from(matches[0] ^ matches[1]);

    let board_pairs = board_rank_counts
        .iter()
        .filter(|count| **count == 2)
        .count();
    output[37] = f32::from(board_pairs >= 1);
    output[38] = f32::from(board_pairs >= 2);
    output[39] = f32::from(board_rank_counts.contains(&3));
    output[40] = f32::from(board_rank_counts.contains(&4));

    let mut full_rank_counts = board_rank_counts;
    let mut full_suit_counts = board_suit_counts;
    let mut full_rank_mask = board_rank_mask;
    for card in private_cards {
        let rank = (card / 4) as usize;
        full_rank_counts[rank] += 1;
        full_suit_counts[(card % 4) as usize] += 1;
        full_rank_mask |= 1 << rank;
    }
    let full_max_rank = full_rank_counts.iter().copied().max().unwrap_or(0) as usize;
    let full_max_suit = full_suit_counts.iter().copied().max().unwrap_or(0) as usize;
    output[41 + full_max_rank.saturating_sub(1).min(3)] = 1.0;
    output[45 + full_max_suit.saturating_sub(1).min(4)] = 1.0;

    let made_straight = rank_mask_has_straight(full_rank_mask);
    output[50] = f32::from(made_straight);
    output[51] = f32::from(street != Street::River && full_max_suit == 4);
    output[52] = f32::from(street == Street::Flop && full_max_suit == 3);
    let straight_outs = if street == Street::River || made_straight {
        0
    } else {
        (0..13)
            .filter(|rank| {
                full_rank_mask & (1 << rank) == 0
                    && rank_mask_has_straight(full_rank_mask | (1 << rank))
            })
            .count()
    };
    output[53 + straight_outs.min(2)] = 1.0;

    output[56] = f32::from(board_max_suit == 1);
    output[57] = f32::from(board_max_suit == 2);
    output[58] = f32::from(board_max_suit >= 3);
    output[59] = f32::from(board_density >= 3);
    output[60] = f32::from(board_density >= 4);
    output[61] = board.iter().filter(|card| **card / 4 >= 10).count() as f32 / 5.0;
    output[62] = board_rank_counts.iter().filter(|count| **count > 0).count() as f32 / 5.0;
    output[63] = (f32::from(board_max_rank >= 2)
        + f32::from(board_max_suit >= 2)
        + f32::from(board_density >= 3))
        / 3.0;
}

fn matches_rank(board: &[u8], rank: u8) -> bool {
    board.iter().any(|card| card / 4 == rank)
}

fn rank_mask_has_straight(mask: u16) -> bool {
    (0..=8).any(|low| ((mask >> low) & 0b1_1111) == 0b1_1111)
        || mask & ((1 << 12) | 0b1111) == ((1 << 12) | 0b1111)
}

fn straight_window_density(mask: u16) -> u32 {
    let regular = (0..=8)
        .map(|low| ((mask >> low) & 0b1_1111).count_ones())
        .max()
        .unwrap_or(0);
    regular.max((mask & ((1 << 12) | 0b1111)).count_ones())
}

fn canonical_suit_map(private_cards: [u8; 2], board: &[u8]) -> [usize; 4] {
    let mut private_masks = [0u16; 4];
    let mut board_masks = [0u16; 4];
    for card in private_cards {
        private_masks[(card % 4) as usize] |= 1u16 << (card / 4);
    }
    for card in board {
        board_masks[(card % 4) as usize] |= 1u16 << (card / 4);
    }
    let mut signatures = (0usize..4)
        .map(|suit| (suit, private_masks[suit], board_masks[suit]))
        .collect::<Vec<_>>();
    signatures.sort_by(|left, right| {
        right
            .1
            .cmp(&left.1)
            .then_with(|| right.2.cmp(&left.2))
            .then_with(|| left.0.cmp(&right.0))
    });
    let mut mapping = [0usize; 4];
    for (canonical, (original, _, _)) in signatures.into_iter().enumerate() {
        mapping[original] = canonical;
    }
    mapping
}

fn canonical_card(card: u8, suit_map: &[usize; 4]) -> usize {
    (card / 4) as usize * 4 + suit_map[(card % 4) as usize]
}

fn encode_action_features(
    state: &GameState,
    action: &LegalAction,
    config: &BlueprintConfig,
) -> Vec<f32> {
    let depth = config.effective_stack_bb as f32;
    let mut features = vec![0.0f32; ACTION_FEATURE_COUNT];
    let kind = action_kind(state, action, config);
    features[trajectory_kind_index(kind)] = 1.0;
    let current = state.street_invested[state.actor];
    let highest = state.street_invested[0].max(state.street_invested[1]);
    let target = match action.kind {
        ActionKind::Call => highest,
        ActionKind::RaiseTo(target) => target,
        ActionKind::Fold | ActionKind::Check => current,
    };
    let paid = (target - current).max(0.0);
    features[6] = target as f32 / depth;
    features[7] = paid as f32 / depth;
    features[8] = (paid / state.pot().max(1.0)) as f32;
    features
}

fn street_index(street: Street) -> usize {
    match street {
        Street::Preflop => 0,
        Street::Flop => 1,
        Street::Turn => 2,
        Street::River => 3,
    }
}

fn trajectory_kind_index(kind: TrajectoryActionKind) -> usize {
    match kind {
        TrajectoryActionKind::Fold => 0,
        TrajectoryActionKind::Check => 1,
        TrajectoryActionKind::Call => 2,
        TrajectoryActionKind::Bet => 3,
        TrajectoryActionKind::Raise => 4,
        TrajectoryActionKind::AllIn => 5,
    }
}

pub fn generate_samples(config: SampleGenerationConfig) -> Result<(), Box<dyn Error>> {
    let (config, records, attempted_records) = SampleGenerator::new(config)?.run()?;
    if let Some(parent) = config.output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let temporary = config.output.with_extension("tmp");
    let file = fs::File::create(&temporary)?;
    let buffered = BufWriter::new(file);
    let mut writer = GzEncoder::new(buffered, Compression::fast());
    let metadata = DatasetMetadata {
        record_type: "metadata",
        schema: DATASET_SCHEMA,
        state_feature_schema: "hu-cash-trajectory-poker-aware-v4",
        state_feature_count: STATE_FEATURE_COUNT,
        action_feature_schema: "hu-cash-legal-action-v1",
        action_feature_count: ACTION_FEATURE_COUNT,
        depth_bb: config.game.effective_stack_bb,
        seed: config.seed,
        start_iteration: config.start_iteration,
        traversals: config.traversals,
        records: records.len(),
        truncated: attempted_records > records.len(),
        sampling_mode: if config.trajectory_sampling {
            "trajectory"
        } else {
            "external_sampling"
        },
        value_rollouts_per_action: config.value_rollouts_per_action,
        action_abstraction: &config.game.action_abstraction,
    };
    serde_json::to_writer(&mut writer, &metadata)?;
    writer.write_all(b"\n")?;
    for record in &records {
        serde_json::to_writer(&mut writer, record)?;
        writer.write_all(b"\n")?;
    }
    writer.finish()?.flush()?;
    fs::rename(temporary, config.output)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_feature_encoder_matches_the_pinned_shape_and_initial_scalars() {
        let mut config = BlueprintConfig::default();
        config.effective_stack_bb = 20.0;
        let deal = Deal::from_cards([[48, 49], [44, 45]], [0, 1, 2, 3, 4]);
        let state = GameState::initial(&config);
        let action = state
            .legal_actions(&config)
            .into_iter()
            .find(|action| action.label == "fold")
            .expect("fold available");
        let features = encode_state_action(&state, &deal, &action, &config);
        assert_eq!(features.len(), MODEL_INPUT_COUNT);
        assert_eq!(features[48], 1.0);
        assert_eq!(features[49], 1.0);
        assert_eq!(features[104], 1.0);
        assert_eq!(features[108], 1.0);
        assert_eq!(features[110], 1.0);
        assert_eq!(features[112], 0.0);
        assert!((features[113] - 19.5 / 20.0).abs() < 1e-6);
        assert_eq!(features[STATE_FEATURE_COUNT], 1.0);
    }

    #[test]
    fn suit_permutations_have_identical_state_action_features() {
        let mut config = BlueprintConfig::default();
        config.effective_stack_bb = 20.0;
        let original = Deal::from_cards([[48, 45], [42, 39]], [0, 5, 10, 15, 20]);
        let permute = |card: u8| (card / 4) * 4 + ((card % 4 + 1) % 4);
        let permuted = Deal::from_cards(
            [
                original.holes[0].map(permute),
                original.holes[1].map(permute),
            ],
            original.board.map(permute),
        );
        let state = GameState::initial(&config);
        let action = state.legal_actions(&config)[0].clone();
        assert_eq!(
            encode_state_action(&state, &original, &action, &config),
            encode_state_action(&state, &permuted, &action, &config)
        );
    }

    #[test]
    fn postflop_texture_features_capture_made_hand_draw_and_board() {
        let mut config = BlueprintConfig::default();
        config.effective_stack_bb = 20.0;
        // Ac Kd on Kc Qc Jc: one pair, four clubs, and one rank that
        // completes a straight. The board is monotone and connected.
        let deal = Deal::from_cards([[48, 45], [1, 2]], [44, 40, 36, 0, 4]);
        let mut state = GameState::initial(&config);
        state.street = Street::Flop;
        let features = encode_state_features(&state, &deal, &config);
        let texture = &features[TEXTURE_FEATURE_OFFSET..STATE_FEATURE_COUNT];
        assert_eq!(texture.len(), TEXTURE_FEATURE_COUNT);
        assert_eq!(texture[0], 1.0);
        assert_eq!(texture[2], 1.0); // one pair
        assert_eq!(texture[16], 1.0); // three board cards share a suit
        assert_eq!(texture[21], 1.0); // three ranks in a straight window
        assert_eq!(texture[28], 1.0); // one overcard
        assert_eq!(texture[32], 1.0); // hole card pairs top board rank
        assert_eq!(texture[36], 1.0); // exactly one hole rank matches
        assert_eq!(texture[48], 1.0); // four visible cards share a suit
        assert_eq!(texture[51], 1.0); // flush draw
        assert_eq!(texture[54], 1.0); // one straight-completing rank
        assert_eq!(texture[58], 1.0); // monotone board
        assert_eq!(texture[59], 1.0); // connected board
    }

    #[test]
    fn action_dependent_baseline_correction_preserves_expectation() {
        let strategy = [0.25, 0.75];
        let baselines = [0.5, 2.0];
        let sampled_values = [1.0, 4.0];
        let estimators = [
            baseline_corrected_sample(&strategy, &baselines, 0, sampled_values[0]),
            baseline_corrected_sample(&strategy, &baselines, 1, sampled_values[1]),
        ];
        let corrected_expectation = strategy
            .iter()
            .zip(estimators)
            .map(|(probability, estimate)| probability * estimate)
            .sum::<f64>();
        let raw_expectation = strategy
            .iter()
            .zip(sampled_values)
            .map(|(probability, value)| probability * value)
            .sum::<f64>();
        assert!((corrected_expectation - raw_expectation).abs() < 1e-12);
    }

    #[test]
    fn shared_state_batch_scoring_matches_individual_dense_inference() {
        let first = DenseLayer {
            input_size: MODEL_INPUT_COUNT,
            output_size: 4,
            activation: DenseActivation::Relu,
            weights: (0..MODEL_INPUT_COUNT * 4)
                .map(|index| ((index % 17) as f32 - 8.0) * 0.0001)
                .collect(),
            biases: vec![-0.03, 0.02, 0.01, -0.01],
        };
        let second = DenseLayer {
            input_size: 4,
            output_size: 1,
            activation: DenseActivation::Linear,
            weights: vec![0.2, -0.3, 0.4, -0.5],
            biases: vec![0.07],
        };
        let scorer = DenseScorer {
            layers: vec![first, second],
        };
        let mut state = vec![0.0; STATE_FEATURE_COUNT];
        for (index, value) in [(3, 1.0), (51, 1.0), (104, 1.0), (112, 0.075)] {
            state[index] = value;
        }
        let actions = vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.1, 0.025, 0.5],
        ];
        let batched = scorer.score_state_actions(&state, &actions);
        for (index, action) in actions.iter().enumerate() {
            let mut values = state.clone();
            values.extend(action);
            for layer in &scorer.layers {
                let mut output = vec![0.0; layer.output_size];
                for (row, value) in output.iter_mut().enumerate() {
                    let offset = row * layer.input_size;
                    let mut sum = layer.biases[row];
                    for column in 0..layer.input_size {
                        sum += layer.weights[offset + column] * values[column];
                    }
                    *value = activate_dense(sum, layer.activation);
                }
                values = output;
            }
            assert_eq!(batched[index].to_bits(), (values[0] as f64).to_bits());
        }
    }

    #[test]
    fn street_routed_bundle_selects_preflop_and_postflop_networks() {
        let scorer = |marker| DenseScorer {
            layers: vec![DenseLayer {
                input_size: MODEL_INPUT_COUNT,
                output_size: 1,
                activation: DenseActivation::Linear,
                weights: vec![0.0; MODEL_INPUT_COUNT],
                biases: vec![marker],
            }],
        };
        let bundle = TrainingNetworkBundle {
            schema: TRAINING_NETWORK_SCHEMA.to_owned(),
            input_size: MODEL_INPUT_COUNT,
            strategy_transform: StrategyTransform::Softmax,
            networks: vec![scorer(1.0), scorer(2.0)],
            postflop_networks: Some(vec![scorer(3.0), scorer(4.0)]),
            sampling_baseline: None,
            sampling_baseline_scale: None,
        };
        assert_eq!(
            bundle.policy_network(Street::Preflop, 0).layers[0].biases,
            [1.0]
        );
        assert_eq!(
            bundle.policy_network(Street::Preflop, 1).layers[0].biases,
            [2.0]
        );
        assert_eq!(
            bundle.policy_network(Street::Flop, 0).layers[0].biases,
            [3.0]
        );
        assert_eq!(
            bundle.policy_network(Street::River, 1).layers[0].biases,
            [4.0]
        );
    }

    #[test]
    fn deterministic_batches_are_bounded_and_contain_both_sample_families() {
        let mut game = BlueprintConfig::default();
        game.effective_stack_bb = 20.0;
        game.iterations = 10;
        game.averaging_delay = 0;
        game.showdown_evaluation.preflop_runout_samples = 4;
        game.showdown_evaluation.flop_runout_samples = 4;
        game.showdown_evaluation.exact_turn_rivers = false;
        let make = || SampleGenerationConfig {
            game: game.clone(),
            traversals: 2,
            start_iteration: 0,
            seed: 17,
            max_records: 32,
            output: PathBuf::from("unused.jsonl.gz"),
            network_path: None,
            trajectory_sampling: false,
            value_rollouts_per_action: 1,
        };
        let (_, first, attempted) = SampleGenerator::new(make()).unwrap().run().unwrap();
        let (_, second, _) = SampleGenerator::new(make()).unwrap().run().unwrap();
        assert!(attempted >= first.len());
        assert!(first.len() <= 32);
        assert_eq!(
            serde_json::to_vec(&first).unwrap(),
            serde_json::to_vec(&second).unwrap()
        );
        assert!(first
            .iter()
            .any(|sample| matches!(sample.kind, SampleKind::AdvantageP0)));
        assert!(first
            .iter()
            .any(|sample| matches!(sample.kind, SampleKind::AverageStrategy)));
        for sample in first {
            assert_eq!(sample.actions.len(), sample.targets.len());
            assert_eq!(sample.actions.len(), sample.feature_sha256.len());
            assert!(sample.targets.iter().all(|value| value.is_finite()));
            assert!(sample.state.trajectory.len() <= MAX_TRAJECTORY_ACTIONS);
        }
    }

    #[test]
    fn extra_value_rollouts_do_not_change_primary_cfr_samples() {
        let mut game = BlueprintConfig::default();
        game.effective_stack_bb = 20.0;
        game.iterations = 10;
        game.averaging_delay = 0;
        game.showdown_evaluation.preflop_runout_samples = 4;
        game.showdown_evaluation.flop_runout_samples = 4;
        game.showdown_evaluation.exact_turn_rivers = false;
        let make = |value_rollouts_per_action| SampleGenerationConfig {
            game: game.clone(),
            traversals: 4,
            start_iteration: 0,
            seed: 41,
            max_records: 256,
            output: PathBuf::from("unused.jsonl.gz"),
            network_path: None,
            trajectory_sampling: false,
            value_rollouts_per_action,
        };
        let (_, primary, _) = SampleGenerator::new(make(1)).unwrap().run().unwrap();
        let (_, averaged, _) = SampleGenerator::new(make(4)).unwrap().run().unwrap();
        assert_eq!(primary.len(), averaged.len());
        let mut value_target_changed = false;
        for (first, second) in primary.iter().zip(&averaged) {
            if first.action_values_bb != second.action_values_bb {
                value_target_changed = true;
            }
            let mut first_without_values = serde_json::to_value(first).unwrap();
            let mut second_without_values = serde_json::to_value(second).unwrap();
            first_without_values
                .as_object_mut()
                .unwrap()
                .remove("action_values_bb");
            second_without_values
                .as_object_mut()
                .unwrap()
                .remove("action_values_bb");
            assert_eq!(first_without_values, second_without_values);
        }
        assert!(value_target_changed);
    }

    #[test]
    fn trajectory_mode_samples_one_path_and_records_every_visited_decision() {
        let mut game = BlueprintConfig::default();
        game.effective_stack_bb = 20.0;
        game.showdown_evaluation.preflop_runout_samples = 4;
        game.showdown_evaluation.flop_runout_samples = 4;
        game.showdown_evaluation.exact_turn_rivers = false;
        let config = SampleGenerationConfig {
            game,
            traversals: 4,
            start_iteration: 0,
            seed: 29,
            max_records: 256,
            output: PathBuf::from("unused.jsonl.gz"),
            network_path: None,
            trajectory_sampling: true,
            value_rollouts_per_action: 1,
        };
        let (_, records, attempted) = SampleGenerator::new(config).unwrap().run().unwrap();
        assert_eq!(attempted, records.len());
        assert!(records.len() >= 4);
        assert!(records
            .iter()
            .all(|sample| matches!(sample.kind, SampleKind::AverageStrategy)));
        assert!(records
            .iter()
            .all(|sample| sample.weight == 1.0 && sample.reach_probability > 0.0));
    }
}
