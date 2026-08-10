//! Exact-card-removal public-belief solving for late-street subgames.
//!
//! The full HUNL game is too large to solve tabularly. River public subgames,
//! however, have no remaining chance events. This module keeps one probability
//! for every exact two-card combination, solves the configured river betting
//! abstraction with alternating CFR, and exports per-combination
//! counterfactual values. No private cards are sampled by the solver.

use super::neural::{FrozenPolicy, MAX_TRAJECTORY_ACTIONS};
use super::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::path::PathBuf;
use std::sync::{Arc, LazyLock, Mutex, OnceLock};

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;

pub const COMBO_COUNT: usize = 1_326;
const RIVER_SCHEMA: &str = "hu-river-public-belief-solution-v1";
const SHARED_CONTEXT_PUBLIC_COUNT: usize = 21;
const SHARED_CONTEXT_COUNT: usize = 359;
const SHARED_CONTEXT_BOARD_RELATIVE_COUNT: usize = 417;
const SHARED_QUERY_STRUCTURAL_COUNT: usize = 76;
const SHARED_QUERY_COUNT: usize = 95;
const SHARED_QUERY_BOARD_RELATIVE_COUNT: usize = 124;
const SHARED_FEATURE_SCHEMA_V1: &str = "rank-suit-invariant-combo-query-v1";
const SHARED_FEATURE_SCHEMA_V2: &str = "rank-suit-invariant-combo-query-v2";
const SHARED_FEATURE_SCHEMA_V3: &str = "rank-suit-invariant-combo-query-v3";
const RANGE_POLICY_FEATURE_SCHEMA_V1: &str = "rank-suit-invariant-combo-policy-query-v1";
const RANGE_POLICY_FEATURE_SCHEMA_V2: &str = "rank-suit-invariant-combo-policy-query-v2";
const RANGE_POLICY_TRAJECTORY_FEATURE_COUNT: usize = 15;
const RANGE_POLICY_PUBLIC_STATE_COUNT: usize =
    20 + MAX_TRAJECTORY_ACTIONS * RANGE_POLICY_TRAJECTORY_FEATURE_COUNT;
const RANGE_POLICY_CONTEXT_V2_COUNT: usize =
    SHARED_CONTEXT_BOARD_RELATIVE_COUNT + RANGE_POLICY_PUBLIC_STATE_COUNT;
const RANGE_POLICY_SCHEMA_V1: &str = "hu-public-belief-combo-policy-network-v1";
const RANGE_POLICY_REPLACE: &str = "replace";
const RANGE_POLICY_SOURCE_LOGIT_RESIDUAL: &str = "source_bundle_logit_residual";
const ACTION_FEATURE_SCHEMA_V1: &str = "hu-cash-legal-action-v1";
const ACTION_FEATURE_COUNT: usize = 9;
const HAND_CLASS_COUNT: usize = 169;
const RESOLVER_REACH_CANONICAL_SCALE: f64 = 1e10;
const RESOLVER_ROOT_CHECKPOINT_SCHEMA: &str = "hu-resolver-root-leaf-checkpoint-v1";
const DENSE_ALL_IN_EQUITY_CACHE_BOARDS: usize = 16;
const DENSE_TURN_EQUITY_CACHE_BOARDS: usize = 64;
const BOARD_QUERY_FEATURE_CACHE_ENTRIES: usize = DENSE_ALL_IN_EQUITY_CACHE_BOARDS * 49;

type DenseAllInEquityCell = Arc<OnceLock<Arc<Vec<f32>>>>;
type DenseTurnEquityCell = Arc<OnceLock<Arc<Vec<u8>>>>;

#[derive(Default)]
struct DenseTurnEquityCache {
    clock: u64,
    entries: BTreeMap<[u8; 4], (u64, DenseTurnEquityCell)>,
}

impl DenseTurnEquityCache {
    fn cell(&mut self, key: [u8; 4]) -> DenseTurnEquityCell {
        self.clock = self.clock.saturating_add(1);
        if let Some((last_used, cell)) = self.entries.get_mut(&key) {
            *last_used = self.clock;
            return cell.clone();
        }
        if self.entries.len() >= DENSE_TURN_EQUITY_CACHE_BOARDS {
            let least_recent = self
                .entries
                .iter()
                .min_by_key(|(_, (last_used, _))| *last_used)
                .map(|(board, _)| *board)
                .expect("non-empty turn equity cache");
            self.entries.remove(&least_recent);
        }
        let cell = Arc::new(OnceLock::new());
        self.entries.insert(key, (self.clock, cell.clone()));
        cell
    }
}

static DENSE_ALL_IN_EQUITY_CACHE: LazyLock<Mutex<BTreeMap<[u8; 3], DenseAllInEquityCell>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));
static DENSE_TURN_EQUITY_CACHE: LazyLock<Mutex<DenseTurnEquityCache>> =
    LazyLock::new(|| Mutex::new(DenseTurnEquityCache::default()));

fn shared_feature_sizes(schema: &str) -> Option<(usize, usize)> {
    match schema {
        SHARED_FEATURE_SCHEMA_V1 => Some((SHARED_CONTEXT_COUNT, SHARED_QUERY_COUNT)),
        SHARED_FEATURE_SCHEMA_V2 | SHARED_FEATURE_SCHEMA_V3 | RANGE_POLICY_FEATURE_SCHEMA_V1 => {
            Some((
                SHARED_CONTEXT_BOARD_RELATIVE_COUNT,
                SHARED_QUERY_BOARD_RELATIVE_COUNT,
            ))
        }
        _ => None,
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PublicBeliefState {
    pub street: Street,
    pub board: Vec<u8>,
    pub actor: usize,
    pub invested_bb: [f64; 2],
    pub street_invested_bb: [f64; 2],
    pub last_full_raise_bb: f64,
    pub aggressions: u8,
    pub checks: u8,
    pub raise_reopened: bool,
    pub public_history: Vec<String>,
    pub ranges: [Vec<f64>; 2],
    /// Exact action trajectory consumed by the served action-policy encoder.
    /// Older value-only artifacts omit it and deserialize to an empty vector.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) trajectory: Vec<TrajectoryAction>,
}

impl PublicBeliefState {
    pub fn flop_start(
        board: [u8; 3],
        actor: usize,
        invested_bb: [f64; 2],
        ranges: [Vec<f64>; 2],
    ) -> Self {
        Self {
            street: Street::Flop,
            board: board.to_vec(),
            actor,
            invested_bb,
            street_invested_bb: [0.0, 0.0],
            last_full_raise_bb: 1.0,
            aggressions: 0,
            checks: 0,
            raise_reopened: true,
            public_history: vec!["public_belief:flop_start".to_owned()],
            ranges,
            trajectory: Vec::new(),
        }
    }

    pub fn turn_start(
        board: [u8; 4],
        actor: usize,
        invested_bb: [f64; 2],
        ranges: [Vec<f64>; 2],
    ) -> Self {
        Self {
            street: Street::Turn,
            board: board.to_vec(),
            actor,
            invested_bb,
            street_invested_bb: [0.0, 0.0],
            last_full_raise_bb: 1.0,
            aggressions: 0,
            checks: 0,
            raise_reopened: true,
            public_history: vec!["public_belief:turn_start".to_owned()],
            ranges,
            trajectory: Vec::new(),
        }
    }

    pub fn river_start(
        board: [u8; 5],
        actor: usize,
        invested_bb: [f64; 2],
        ranges: [Vec<f64>; 2],
    ) -> Self {
        Self {
            street: Street::River,
            board: board.to_vec(),
            actor,
            invested_bb,
            street_invested_bb: [0.0, 0.0],
            last_full_raise_bb: 1.0,
            aggressions: 0,
            checks: 0,
            raise_reopened: true,
            public_history: vec!["public_belief:river_start".to_owned()],
            ranges,
            trajectory: Vec::new(),
        }
    }

    pub fn uniform_river_start(board: [u8; 5], actor: usize, invested_bb: [f64; 2]) -> Self {
        let ranges = std::array::from_fn(|_| uniform_range(&board));
        Self::river_start(board, actor, invested_bb, ranges)
    }

    fn validate_and_normalize(&self, game: &BlueprintConfig) -> Result<Self, String> {
        self.validate_street_and_normalize(game, Street::River, 5)
    }

    fn validate_street_and_normalize(
        &self,
        game: &BlueprintConfig,
        street: Street,
        board_len: usize,
    ) -> Result<Self, String> {
        if self.street != street || self.board.len() != board_len {
            return Err(format!(
                "public-belief state must be {street:?} with {board_len} board cards"
            ));
        }
        if self.actor > 1 {
            return Err("public-belief actor must be player zero or one".to_owned());
        }
        let unique = self.board.iter().copied().collect::<BTreeSet<_>>();
        if unique.len() != self.board.len() || self.board.iter().any(|card| *card >= 52) {
            return Err("public board must contain five unique cards".to_owned());
        }
        if self
            .invested_bb
            .iter()
            .chain(&self.street_invested_bb)
            .any(|value| !value.is_finite() || *value < 0.0 || *value > game.effective_stack_bb)
        {
            return Err("public-belief commitments must be finite and within the stack".to_owned());
        }
        if !self.last_full_raise_bb.is_finite() || self.last_full_raise_bb <= 0.0 {
            return Err("last full raise must be positive".to_owned());
        }
        let mut normalized = self.clone();
        for player in 0..2 {
            if normalized.ranges[player].len() != COMBO_COUNT {
                return Err(format!(
                    "player {player} range must contain {COMBO_COUNT} exact combinations"
                ));
            }
            let mut total = 0.0;
            for combo in all_combos() {
                let weight = &mut normalized.ranges[player][combo.key()];
                if !weight.is_finite() || *weight < 0.0 {
                    return Err(format!(
                        "player {player} range contains a non-finite or negative weight"
                    ));
                }
                if combo.cards().iter().any(|card| unique.contains(card)) {
                    *weight = 0.0;
                }
                total += *weight;
            }
            if total <= EPSILON {
                return Err(format!(
                    "player {player} range has no legal river combinations"
                ));
            }
            for weight in &mut normalized.ranges[player] {
                *weight /= total;
            }
        }
        let joint_mass = joint_compatibility_mass(&normalized.ranges);
        if joint_mass <= EPSILON {
            return Err("the two ranges have no mutually compatible deals".to_owned());
        }
        Ok(normalized)
    }

    fn game_state(&self) -> GameState {
        GameState {
            street: self.street,
            actor: self.actor,
            invested: self.invested_bb,
            street_invested: self.street_invested_bb,
            last_full_raise: self.last_full_raise_bb,
            aggressions: self.aggressions,
            checks: self.checks,
            raise_reopened: self.raise_reopened,
            public_history: self.public_history.clone(),
            trajectory: self.trajectory.clone(),
            terminal: None,
        }
    }

    pub(super) fn from_game_state(
        board: Vec<u8>,
        state: &GameState,
        ranges: [Vec<f64>; 2],
    ) -> Self {
        Self {
            street: state.street,
            board,
            actor: state.actor,
            invested_bb: state.invested,
            street_invested_bb: state.street_invested,
            last_full_raise_bb: state.last_full_raise,
            aggressions: state.aggressions,
            checks: state.checks,
            raise_reopened: state.raise_reopened,
            public_history: state.public_history.clone(),
            ranges,
            trajectory: state.trajectory.clone(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct ValueNetworkLayer {
    input_size: usize,
    output_size: usize,
    activation: String,
    weights: Vec<f32>,
    biases: Vec<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    normalization: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    normalization_weights: Vec<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    normalization_biases: Vec<f32>,
    #[serde(default)]
    normalization_epsilon: f32,
    #[serde(default)]
    residual: bool,
}

impl ValueNetworkLayer {
    fn validate(&self, expected_input: usize) -> Result<usize, String> {
        let normalization_is_valid = match self.normalization.as_deref() {
            None => self.normalization_weights.is_empty() && self.normalization_biases.is_empty(),
            Some("layernorm") => {
                self.normalization_weights.len() == self.output_size
                    && self.normalization_biases.len() == self.output_size
                    && self.normalization_epsilon.is_finite()
                    && self.normalization_epsilon > 0.0
                    && self
                        .normalization_weights
                        .iter()
                        .chain(&self.normalization_biases)
                        .all(|value| value.is_finite())
            }
            _ => false,
        };
        if self.input_size != expected_input
            || self.output_size == 0
            || self.weights.len() != self.input_size * self.output_size
            || self.biases.len() != self.output_size
            || self
                .weights
                .iter()
                .chain(&self.biases)
                .any(|value| !value.is_finite())
            || !matches!(
                self.activation.as_str(),
                "relu" | "linear" | "tanh" | "gelu-fast"
            )
            || !normalization_is_valid
            || (self.residual && self.input_size != self.output_size)
        {
            return Err("public value network contains an invalid dense layer".to_owned());
        }
        Ok(self.output_size)
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = Vec::new();
        self.forward_into(input, &mut output);
        output
    }

    fn forward_into(&self, input: &[f32], output: &mut Vec<f32>) {
        debug_assert_eq!(input.len(), self.input_size);
        output.clear();
        output.extend_from_slice(&self.biases);
        for (out, row) in output
            .iter_mut()
            .zip(self.weights.chunks_exact(self.input_size))
        {
            *out += row
                .iter()
                .zip(input)
                .map(|(weight, value)| weight * value)
                .sum::<f32>();
        }
        if self.residual {
            for (value, skip) in output.iter_mut().zip(input) {
                *value += skip;
            }
        }
        self.normalize(output);
        for value in output {
            *value = self.activate(*value);
        }
    }

    fn normalize(&self, values: &mut [f32]) {
        if self.normalization.as_deref() != Some("layernorm") {
            return;
        }
        let denominator = values.len() as f32;
        let mean = values.iter().sum::<f32>() / denominator;
        let variance = values
            .iter()
            .map(|value| {
                let centered = *value - mean;
                centered * centered
            })
            .sum::<f32>()
            / denominator;
        let inverse_stddev = 1.0 / (variance + self.normalization_epsilon).sqrt();
        for ((value, scale), bias) in values
            .iter_mut()
            .zip(&self.normalization_weights)
            .zip(&self.normalization_biases)
        {
            *value = (*value - mean) * inverse_stddev * scale + bias;
        }
    }

    fn activate(&self, value: f32) -> f32 {
        match self.activation.as_str() {
            "relu" => value.max(0.0),
            "tanh" => value.tanh(),
            "gelu-fast" => value / (1.0 + (-1.702 * value).exp()),
            _ => value,
        }
    }
}

fn forward_batch_layer_into(
    layer: &ValueNetworkLayer,
    input: &[f32],
    samples: usize,
    output: &mut Vec<f32>,
) {
    debug_assert_eq!(input.len(), samples * layer.input_size);
    output.clear();
    output.resize(samples * layer.output_size, 0.0);
    if samples == 0 {
        return;
    }
    // SAFETY: the validated dimensions and assertions above guarantee that
    // all row-major matrix strides remain within the input, weight, and output
    // allocations for the full multiplication.
    unsafe {
        matrixmultiply::sgemm(
            samples,
            layer.input_size,
            layer.output_size,
            1.0,
            input.as_ptr(),
            layer.input_size as isize,
            1,
            layer.weights.as_ptr(),
            1,
            layer.input_size as isize,
            0.0,
            output.as_mut_ptr(),
            layer.output_size as isize,
            1,
        );
    }
    for (row, input_row) in output
        .chunks_exact_mut(layer.output_size)
        .zip(input.chunks_exact(layer.input_size))
    {
        for (value, bias) in row.iter_mut().zip(&layer.biases) {
            *value += bias;
        }
        if layer.residual {
            for (value, skip) in row.iter_mut().zip(input_row) {
                *value += skip;
            }
        }
        layer.normalize(row);
        for value in row {
            *value = layer.activate(*value);
        }
    }
}

fn forward_batch_tower(layers: &[ValueNetworkLayer], input: &[f32], samples: usize) -> Vec<f32> {
    debug_assert!(!layers.is_empty());
    let mut scratch = [Vec::new(), Vec::new()];
    forward_batch_layer_into(&layers[0], input, samples, &mut scratch[0]);
    for (index, layer) in layers.iter().enumerate().skip(1) {
        if index % 2 == 1 {
            let (first, second) = scratch.split_at_mut(1);
            forward_batch_layer_into(layer, &first[0], samples, &mut second[0]);
        } else {
            let (first, second) = scratch.split_at_mut(1);
            forward_batch_layer_into(layer, &second[0], samples, &mut first[0]);
        }
    }
    std::mem::take(&mut scratch[(layers.len() - 1) % 2])
}

fn forward_batch_head(
    layers: &[ValueNetworkLayer],
    context: &[f32],
    query: &[f32],
    samples: usize,
) -> Vec<f32> {
    debug_assert!(!layers.is_empty());
    let first = &layers[0];
    let query_size = first.input_size - context.len();
    debug_assert_eq!(query.len(), samples * query_size);
    let mut scratch = [vec![0.0; samples * first.output_size], Vec::new()];
    if samples > 0 {
        // SAFETY: the query portion of every validated head weight row starts
        // at context.len(); the matrix dimensions and strides stay inside all
        // three allocations.
        unsafe {
            matrixmultiply::sgemm(
                samples,
                query_size,
                first.output_size,
                1.0,
                query.as_ptr(),
                query_size as isize,
                1,
                first.weights.as_ptr().add(context.len()),
                1,
                first.input_size as isize,
                0.0,
                scratch[0].as_mut_ptr(),
                first.output_size as isize,
                1,
            );
        }
        let context_offsets = first
            .weights
            .chunks_exact(first.input_size)
            .zip(&first.biases)
            .map(|(row, bias)| {
                bias + row[..context.len()]
                    .iter()
                    .zip(context)
                    .map(|(weight, value)| weight * value)
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        for (sample, row) in scratch[0].chunks_exact_mut(first.output_size).enumerate() {
            for (value, offset) in row.iter_mut().zip(&context_offsets) {
                *value += offset;
            }
            if first.residual {
                for (value, skip) in row.iter_mut().take(context.len()).zip(context) {
                    *value += skip;
                }
                for (value, skip) in row
                    .iter_mut()
                    .skip(context.len())
                    .zip(&query[sample * query_size..(sample + 1) * query_size])
                {
                    *value += skip;
                }
            }
            first.normalize(row);
            for value in row {
                *value = first.activate(*value);
            }
        }
    }
    for (index, layer) in layers.iter().enumerate().skip(1) {
        if index % 2 == 1 {
            let (first, second) = scratch.split_at_mut(1);
            forward_batch_layer_into(layer, &first[0], samples, &mut second[0]);
        } else {
            let (first, second) = scratch.split_at_mut(1);
            forward_batch_layer_into(layer, &second[0], samples, &mut first[0]);
        }
    }
    std::mem::take(&mut scratch[(layers.len() - 1) % 2])
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PublicValueNetwork {
    #[serde(skip)]
    artifact_sha256: Option<String>,
    schema: String,
    seed: u64,
    uses_exact_ranges: bool,
    target_scale_bb: f64,
    range_scale: f64,
    #[serde(default)]
    value_normalization: Option<String>,
    #[serde(default)]
    residual_scale_bb: f64,
    #[serde(default)]
    source_dataset_sha256: Option<String>,
    #[serde(default)]
    source_policy_sha256: Option<String>,
    #[serde(default)]
    source_validation_status: Option<String>,
    #[serde(default)]
    feature_schema: Option<String>,
    #[serde(default)]
    context_public_count: usize,
    #[serde(default)]
    context_size: usize,
    #[serde(default)]
    query_structural_count: usize,
    #[serde(default)]
    query_size: usize,
    #[serde(default)]
    public_tower: Vec<ValueNetworkLayer>,
    #[serde(default)]
    range_tower: Vec<ValueNetworkLayer>,
    #[serde(default)]
    context_tower: Vec<ValueNetworkLayer>,
    #[serde(default)]
    query_tower: Vec<ValueNetworkLayer>,
    #[serde(default)]
    head: Vec<ValueNetworkLayer>,
}

impl PublicValueNetwork {
    pub fn read(path: &Path) -> Result<Self, Box<dyn Error>> {
        let bytes = fs::read(path)?;
        let mut network: Self = serde_json::from_slice(&bytes)?;
        network.artifact_sha256 = Some(format!("{:x}", Sha256::digest(&bytes)));
        network.validate()?;
        Ok(network)
    }

    pub(crate) fn has_distinct_training_identity(&self, other: &Self) -> bool {
        self.seed != other.seed
            && self.artifact_sha256.is_some()
            && other.artifact_sha256.is_some()
            && self.artifact_sha256 != other.artifact_sha256
    }

    fn validate(&self) -> Result<(), String> {
        if !self.target_scale_bb.is_finite()
            || self.target_scale_bb <= 0.0
            || !self.range_scale.is_finite()
            || self.range_scale <= 0.0
            || self.head.is_empty()
        {
            return Err("public value network header is incompatible".to_owned());
        }
        match self.schema.as_str() {
            "hu-public-belief-value-network-v2" => {
                if self.public_tower.is_empty() || self.range_tower.is_empty() {
                    return Err("legacy public value network towers are missing".to_owned());
                }
                let mut public_size = 56;
                for layer in &self.public_tower {
                    public_size = layer.validate(public_size)?;
                }
                let mut range_size = COMBO_COUNT * 2;
                for layer in &self.range_tower {
                    range_size = layer.validate(range_size)?;
                }
                let mut head_size = public_size + range_size;
                for layer in &self.head {
                    head_size = layer.validate(head_size)?;
                }
                if head_size != COMBO_COUNT * 2 {
                    return Err(
                        "legacy public value network must output two exact-combo CFV vectors"
                            .to_owned(),
                    );
                }
            }
            "hu-public-belief-combo-value-network-v3"
            | "hu-public-belief-combo-value-network-v4"
            | "hu-public-belief-combo-value-network-v5" => {
                let Some((expected_context_size, expected_query_size)) = self
                    .feature_schema
                    .as_deref()
                    .and_then(shared_feature_sizes)
                else {
                    return Err(
                        "shared-combo public value network feature schema is incompatible"
                            .to_owned(),
                    );
                };
                if self.schema == "hu-public-belief-combo-value-network-v3"
                    && self.feature_schema.as_deref() != Some(SHARED_FEATURE_SCHEMA_V1)
                {
                    return Err("legacy shared-combo v3 requires feature schema v1".to_owned());
                }
                if self.context_public_count != SHARED_CONTEXT_PUBLIC_COUNT
                    || self.context_size != expected_context_size
                    || self.query_structural_count != SHARED_QUERY_STRUCTURAL_COUNT
                    || self.query_size != expected_query_size
                    || self.context_tower.is_empty()
                    || self.query_tower.is_empty()
                {
                    return Err(
                        "shared-combo public value network schema is incompatible".to_owned()
                    );
                }
                if self.schema == "hu-public-belief-combo-value-network-v3"
                    && (!self.residual_scale_bb.is_finite() || self.residual_scale_bb <= 0.0)
                {
                    return Err("shared-combo v3 residual scale is invalid".to_owned());
                }
                if matches!(
                    self.schema.as_str(),
                    "hu-public-belief-combo-value-network-v4"
                        | "hu-public-belief-combo-value-network-v5"
                ) && !matches!(
                    self.value_normalization.as_deref(),
                    Some("pot" | "payoff-exposure")
                ) {
                    return Err("shared-combo value normalization is invalid".to_owned());
                }
                let mut context_size = expected_context_size;
                for layer in &self.context_tower {
                    context_size = layer.validate(context_size)?;
                }
                let mut query_size = expected_query_size;
                for layer in &self.query_tower {
                    query_size = layer.validate(query_size)?;
                }
                let mut head_size = context_size
                    + if self.schema == "hu-public-belief-combo-value-network-v5" {
                        query_size * 3
                    } else {
                        query_size
                    };
                for layer in &self.head {
                    head_size = layer.validate(head_size)?;
                }
                if head_size != 1 {
                    return Err("shared-combo public value network must output one CFV".to_owned());
                }
            }
            _ => return Err("public value network schema is incompatible".to_owned()),
        }
        Ok(())
    }

    pub fn predict(
        &self,
        board: &[u8],
        actor: usize,
        invested: [f64; 2],
        ranges: &[Vec<f64>; 2],
    ) -> [Vec<f64>; 2] {
        if matches!(
            self.schema.as_str(),
            "hu-public-belief-combo-value-network-v3"
                | "hu-public-belief-combo-value-network-v4"
                | "hu-public-belief-combo-value-network-v5"
        ) {
            return self.predict_shared_combo(board, actor, invested, ranges);
        }
        let mut range_features = ranges
            .iter()
            .flatten()
            .map(|weight| (*weight * self.range_scale) as f32)
            .collect::<Vec<_>>();
        if !self.uses_exact_ranges {
            range_features.fill(0.0);
        }
        let range_embedding = self
            .range_tower
            .iter()
            .fold(range_features, |values, layer| layer.forward(&values));
        let mut public_features = vec![0.0f32; 52];
        for card in board {
            public_features[*card as usize] = 1.0;
        }
        public_features.extend([
            if actor == 0 { 1.0 } else { 0.0 },
            if actor == 1 { 1.0 } else { 0.0 },
        ]);
        public_features.extend([
            (invested[0] / self.target_scale_bb) as f32,
            (invested[1] / self.target_scale_bb) as f32,
        ]);
        let public_embedding = self
            .public_tower
            .iter()
            .fold(public_features, |values, layer| layer.forward(&values));
        let mut head = public_embedding;
        head.extend(range_embedding);
        let output = self
            .head
            .iter()
            .fold(head, |values, layer| layer.forward(&values));
        let board_legal = all_combos()
            .iter()
            .map(|private| !private.cards().iter().any(|card| board.contains(card)))
            .collect::<Vec<_>>();
        std::array::from_fn(|player| {
            output[player * COMBO_COUNT..(player + 1) * COMBO_COUNT]
                .iter()
                .enumerate()
                .map(|(combo, value)| {
                    if board_legal[combo] {
                        *value as f64 * self.target_scale_bb
                    } else {
                        0.0
                    }
                })
                .collect()
        })
    }

    fn predict_shared_combo(
        &self,
        board: &[u8],
        actor: usize,
        invested: [f64; 2],
        ranges: &[Vec<f64>; 2],
    ) -> [Vec<f64>; 2] {
        let conflicts = combo_conflicts();
        let (mut contexts, mut queries) = shared_combo_features(
            board,
            actor,
            invested,
            ranges,
            &conflicts,
            self.target_scale_bb,
            self.feature_schema
                .as_deref()
                .expect("validated shared feature schema"),
        );
        if !self.uses_exact_ranges {
            for context in &mut contexts {
                context[SHARED_CONTEXT_PUBLIC_COUNT..].fill(0.0);
            }
            for player_queries in &mut queries {
                for query in player_queries {
                    query[SHARED_QUERY_STRUCTURAL_COUNT..].fill(0.0);
                }
            }
        }
        let context_embeddings: [Vec<f32>; 2] = std::array::from_fn(|player| {
            self.context_tower
                .iter()
                .fold(contexts[player].clone(), |values, layer| {
                    layer.forward(&values)
                })
        });
        // Counterfactual regret updates need a value for every private hand
        // that could deviate into this public branch, including hands whose
        // current strategy assigns the branch zero reach. Masking queries by
        // current reach makes a zero-probability action permanently invisible
        // to regret matching and can freeze an exploitable policy.
        let board_cards = board.iter().copied().collect::<BTreeSet<_>>();
        let board_legal_combos = all_combos()
            .iter()
            .filter(|combo| !combo.cards().iter().any(|card| board_cards.contains(card)))
            .map(|combo| combo.key())
            .collect::<Vec<_>>();
        let legal_combos = [board_legal_combos.clone(), board_legal_combos];
        let query_embeddings: [Vec<f32>; 2] = std::array::from_fn(|player| {
            let mut query_batch = Vec::with_capacity(legal_combos[player].len() * self.query_size);
            for combo in &legal_combos[player] {
                query_batch.extend_from_slice(&queries[player][*combo]);
            }
            forward_batch_tower(&self.query_tower, &query_batch, legal_combos[player].len())
        });
        let masses: [Vec<f64>; 2] = std::array::from_fn(|player| {
            (0..COMBO_COUNT)
                .map(|combo| compatible_mass_from_conflicts(&ranges[1 - player], &conflicts, combo))
                .collect()
        });
        let query_embedding_size = self
            .query_tower
            .last()
            .expect("validated shared query tower")
            .output_size;
        let pooled_queries: Option<[Vec<f32>; 2]> =
            (self.schema == "hu-public-belief-combo-value-network-v5").then(|| {
                std::array::from_fn(|player| {
                    let denominator = legal_combos[player]
                        .iter()
                        .map(|combo| ranges[player][*combo] * masses[player][*combo])
                        .sum::<f64>()
                        .max(EPSILON);
                    let mut pooled = vec![0.0f32; query_embedding_size];
                    for (row, combo) in legal_combos[player].iter().enumerate() {
                        let weight =
                            (ranges[player][*combo] * masses[player][*combo] / denominator) as f32;
                        for (value, embedding) in pooled.iter_mut().zip(
                            &query_embeddings[player]
                                [row * query_embedding_size..(row + 1) * query_embedding_size],
                        ) {
                            *value += weight * embedding;
                        }
                    }
                    pooled
                })
            });
        let mut result: [Vec<f64>; 2] = std::array::from_fn(|player| {
            let mut head_context = context_embeddings[player].clone();
            if let Some(pooled) = &pooled_queries {
                head_context.extend_from_slice(&pooled[player]);
                head_context.extend_from_slice(&pooled[1 - player]);
            }
            let output = forward_batch_head(
                &self.head,
                &head_context,
                &query_embeddings[player],
                legal_combos[player].len(),
            );
            let output_size = self.head.last().expect("validated shared head").output_size;
            let mut values = vec![0.0; COMBO_COUNT];
            for (row, combo) in legal_combos[player].iter().copied().enumerate() {
                let query = &queries[player][combo];
                let equity = if self.uses_exact_ranges {
                    query[94] as f64
                } else {
                    query[65] as f64
                };
                let opponent = 1 - player;
                let baseline = equity * invested[opponent] - (1.0 - equity) * invested[player];
                let residual = output[row * output_size] as f64;
                values[combo] = if matches!(
                    self.schema.as_str(),
                    "hu-public-belief-combo-value-network-v4"
                        | "hu-public-belief-combo-value-network-v5"
                ) {
                    baseline + residual * self.state_value_scale_bb(invested)
                } else {
                    baseline + residual * self.residual_scale_bb
                };
            }
            values
        });
        let joint_mass = ranges[0]
            .iter()
            .zip(&masses[0])
            .map(|(reach, mass)| reach * mass)
            .sum::<f64>()
            .max(EPSILON);
        let aggregate = |player: usize| {
            ranges[player]
                .iter()
                .zip(&result[player])
                .zip(&masses[player])
                .map(|((reach, value), mass)| reach * value * mass)
                .sum::<f64>()
                / joint_mass
        };
        let residual = aggregate(0) + aggregate(1);
        for player in 0..2 {
            for combo in &legal_combos[player] {
                result[player][*combo] -= residual / 2.0;
            }
        }
        result
    }

    fn state_value_scale_bb(&self, invested: [f64; 2]) -> f64 {
        match self.value_normalization.as_deref() {
            Some("pot") => (invested[0] + invested[1]).max(1.0),
            Some("payoff-exposure") => {
                let remaining = [
                    (self.target_scale_bb - invested[0]).max(0.0),
                    (self.target_scale_bb - invested[1]).max(0.0),
                ];
                (invested[0].max(invested[1]) + remaining[0].min(remaining[1])).max(1.0)
            }
            _ => self.target_scale_bb,
        }
    }
}

/// A served postflop policy conditioned on the complete public belief state.
///
/// Unlike the legacy action scorer, this network observes both exact private
/// ranges and scores every board-legal private combo. Scoring zero-reach actor
/// combos is intentional: a user can force an off-policy action without
/// making the next decision unavailable.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RangeConditionedPolicyNetwork {
    #[serde(skip)]
    artifact_sha256: Option<String>,
    schema: String,
    seed: u64,
    depth_bb: f64,
    uses_exact_ranges: bool,
    feature_schema: String,
    context_size: usize,
    query_size: usize,
    action_feature_schema: String,
    action_feature_size: usize,
    context_tower: Vec<ValueNetworkLayer>,
    query_tower: Vec<ValueNetworkLayer>,
    action_tower: Vec<ValueNetworkLayer>,
    head: Vec<ValueNetworkLayer>,
    source_dataset_sha256: String,
    source_dataset_schema: String,
    source_validation_status: String,
    #[serde(default = "default_range_policy_composition")]
    policy_composition: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source_policy_sha256: Option<String>,
}

fn default_range_policy_composition() -> String {
    RANGE_POLICY_REPLACE.to_owned()
}

fn range_policy_state_features(
    state: &PublicBeliefState,
    depth_bb: f64,
) -> Result<Vec<f32>, String> {
    if !depth_bb.is_finite() || depth_bb <= 0.0 || state.trajectory.len() > MAX_TRAJECTORY_ACTIONS {
        return Err("range-policy public action state is incompatible".to_owned());
    }
    let mut features = vec![0.0f32; RANGE_POLICY_PUBLIC_STATE_COUNT];
    let street_index = |street| match street {
        Street::Preflop => 0,
        Street::Flop => 1,
        Street::Turn => 2,
        Street::River => 3,
    };
    let kind_index = |kind| match kind {
        TrajectoryActionKind::Fold => 0,
        TrajectoryActionKind::Check => 1,
        TrajectoryActionKind::Call => 2,
        TrajectoryActionKind::Bet => 3,
        TrajectoryActionKind::Raise => 4,
        TrajectoryActionKind::AllIn => 5,
    };
    let actor = state.actor;
    let opponent = 1 - actor;
    features[street_index(state.street)] = 1.0;
    features[4 + actor] = 1.0;
    let settled_pot =
        state.invested_bb.iter().sum::<f64>() - state.street_invested_bb.iter().sum::<f64>();
    let to_call = (state.street_invested_bb[opponent] - state.street_invested_bb[actor]).max(0.0);
    let scalars = [
        settled_pot / depth_bb,
        (depth_bb - state.invested_bb[actor]) / depth_bb,
        (depth_bb - state.invested_bb[opponent]) / depth_bb,
        state.street_invested_bb[actor] / depth_bb,
        state.street_invested_bb[opponent] / depth_bb,
        state.invested_bb[actor] / depth_bb,
        state.invested_bb[opponent] / depth_bb,
        to_call / depth_bb,
        state.last_full_raise_bb / depth_bb,
        f64::from(state.raise_reopened),
        state.board.len() as f64 / 5.0,
        state.trajectory.len() as f64 / MAX_TRAJECTORY_ACTIONS as f64,
    ];
    if scalars.iter().any(|value| !value.is_finite()) {
        return Err("range-policy public action state is non-finite".to_owned());
    }
    for (target, value) in features[6..18].iter_mut().zip(scalars) {
        *target = value as f32;
    }
    features[18] = f32::from(state.aggressions) / 2.0;
    features[19] = f32::from(state.checks);
    for (index, action) in state.trajectory.iter().enumerate() {
        if action.actor > 1
            || !action.amount_bb.is_finite()
            || action.amount_to_bb.is_some_and(|value| !value.is_finite())
            || !action.pot_after_bb.is_finite()
        {
            return Err("range-policy trajectory action is invalid".to_owned());
        }
        let offset = 20 + index * RANGE_POLICY_TRAJECTORY_FEATURE_COUNT;
        features[offset + action.actor] = 1.0;
        features[offset + 2 + street_index(action.street)] = 1.0;
        features[offset + 6 + kind_index(action.kind)] = 1.0;
        features[offset + 12] = (action.amount_bb / depth_bb) as f32;
        features[offset + 13] = (action.amount_to_bb.unwrap_or(0.0) / depth_bb) as f32;
        features[offset + 14] = (action.pot_after_bb / depth_bb) as f32;
    }
    Ok(features)
}

impl RangeConditionedPolicyNetwork {
    pub fn read(path: &Path) -> Result<Self, Box<dyn Error>> {
        let bytes = fs::read(path)?;
        let mut network: Self = serde_json::from_slice(&bytes)?;
        network.artifact_sha256 = Some(format!("{:x}", Sha256::digest(&bytes)));
        network.validate()?;
        Ok(network)
    }

    fn validate(&self) -> Result<(), String> {
        let expected_context_size = match self.feature_schema.as_str() {
            RANGE_POLICY_FEATURE_SCHEMA_V1 => SHARED_CONTEXT_BOARD_RELATIVE_COUNT,
            RANGE_POLICY_FEATURE_SCHEMA_V2 => RANGE_POLICY_CONTEXT_V2_COUNT,
            _ => return Err("range-conditioned policy feature schema is incompatible".to_owned()),
        };
        if self.schema != RANGE_POLICY_SCHEMA_V1
            || !self.uses_exact_ranges
            || self.context_size != expected_context_size
            || self.query_size != SHARED_QUERY_BOARD_RELATIVE_COUNT
            || self.action_feature_schema != ACTION_FEATURE_SCHEMA_V1
            || self.action_feature_size != ACTION_FEATURE_COUNT
            || !self.depth_bb.is_finite()
            || self.depth_bb <= 0.0
            || self.context_tower.is_empty()
            || self.query_tower.is_empty()
            || self.action_tower.is_empty()
            || self.head.is_empty()
            || self.source_dataset_sha256.len() != 64
            || self.source_dataset_schema != "hu-range-conditioned-postflop-policy-dataset-v1"
            || self.source_validation_status != "accepted_for_training"
            || !matches!(
                self.policy_composition.as_str(),
                RANGE_POLICY_REPLACE | RANGE_POLICY_SOURCE_LOGIT_RESIDUAL
            )
            || (self.policy_composition == RANGE_POLICY_SOURCE_LOGIT_RESIDUAL
                && self
                    .source_policy_sha256
                    .as_deref()
                    .is_none_or(|hash| hash.len() != 64))
        {
            return Err("range-conditioned policy network header is incompatible".to_owned());
        }
        let mut context_embedding = self.context_size;
        for layer in &self.context_tower {
            context_embedding = layer.validate(context_embedding)?;
        }
        let mut query_embedding = self.query_size;
        for layer in &self.query_tower {
            query_embedding = layer.validate(query_embedding)?;
        }
        let mut action_embedding = self.action_feature_size;
        for layer in &self.action_tower {
            action_embedding = layer.validate(action_embedding)?;
        }
        let mut head_size = context_embedding + query_embedding * 3 + action_embedding;
        for layer in &self.head {
            head_size = layer.validate(head_size)?;
        }
        if head_size != 1 {
            return Err("range-conditioned policy head must output one logit".to_owned());
        }
        Ok(())
    }

    pub(crate) fn requires_source_policy(&self) -> bool {
        self.policy_composition == RANGE_POLICY_SOURCE_LOGIT_RESIDUAL
    }

    pub(crate) fn validate_source_policy_sha256(&self, sha256: &str) -> Result<(), String> {
        if self.requires_source_policy() && self.source_policy_sha256.as_deref() != Some(sha256) {
            return Err(
                "range-conditioned residual policy source bundle hash does not match".to_owned(),
            );
        }
        Ok(())
    }

    pub(crate) fn strategy(
        &self,
        state: &PublicBeliefState,
        game: &BlueprintConfig,
        source_policy: Option<&[f64]>,
    ) -> Result<Vec<f64>, String> {
        self.validate()?;
        game.validate()?;
        if (game.effective_stack_bb - self.depth_bb).abs() > EPSILON {
            return Err("range-conditioned policy depth does not match the game".to_owned());
        }
        let expected_board = state.street.board_len();
        let normalized = state.validate_street_and_normalize(game, state.street, expected_board)?;
        let game_state = normalized.game_state();
        let actions = game_state.legal_actions(game);
        if actions.is_empty() {
            return Err("range-conditioned policy cannot score a terminal state".to_owned());
        }
        let source_policy = if self.requires_source_policy() {
            let source = source_policy.ok_or_else(|| {
                "range-conditioned residual policy requires source bundle probabilities".to_owned()
            })?;
            if source.len() != COMBO_COUNT * actions.len()
                || source
                    .iter()
                    .any(|probability| !probability.is_finite() || *probability < 0.0)
            {
                return Err(
                    "range-conditioned residual policy received invalid source probabilities"
                        .to_owned(),
                );
            }
            Some(source)
        } else {
            None
        };
        let conflicts = combo_conflicts();
        let (mut contexts, queries) = shared_combo_features(
            &normalized.board,
            normalized.actor,
            normalized.invested_bb,
            &normalized.ranges,
            &conflicts,
            self.depth_bb,
            RANGE_POLICY_FEATURE_SCHEMA_V1,
        );
        if self.feature_schema == RANGE_POLICY_FEATURE_SCHEMA_V2 {
            let public_state = range_policy_state_features(&normalized, self.depth_bb)?;
            for context in &mut contexts {
                context.extend_from_slice(&public_state);
            }
        }
        let context_embedding = self
            .context_tower
            .iter()
            .fold(contexts[normalized.actor].clone(), |values, layer| {
                layer.forward(&values)
            });
        let board = normalized.board.iter().copied().collect::<BTreeSet<_>>();
        let legal_combos = all_combos()
            .iter()
            .filter(|combo| !combo.cards().iter().any(|card| board.contains(card)))
            .map(|combo| combo.key())
            .collect::<Vec<_>>();
        let query_embeddings: [Vec<f32>; 2] = std::array::from_fn(|player| {
            let mut batch = Vec::with_capacity(legal_combos.len() * self.query_size);
            for combo in &legal_combos {
                batch.extend_from_slice(&queries[player][*combo]);
            }
            forward_batch_tower(&self.query_tower, &batch, legal_combos.len())
        });
        let query_embedding_size = self
            .query_tower
            .last()
            .expect("validated query tower")
            .output_size;
        let masses: [Vec<f64>; 2] = std::array::from_fn(|player| {
            (0..COMBO_COUNT)
                .map(|combo| {
                    compatible_mass_from_conflicts(
                        &normalized.ranges[1 - player],
                        &conflicts,
                        combo,
                    )
                })
                .collect()
        });
        let pooled: [Vec<f32>; 2] = std::array::from_fn(|player| {
            let denominator = legal_combos
                .iter()
                .map(|combo| normalized.ranges[player][*combo] * masses[player][*combo])
                .sum::<f64>()
                .max(EPSILON);
            let mut result = vec![0.0f32; query_embedding_size];
            for (row, combo) in legal_combos.iter().enumerate() {
                let weight = (normalized.ranges[player][*combo] * masses[player][*combo]
                    / denominator) as f32;
                for (output, embedded) in result.iter_mut().zip(
                    &query_embeddings[player]
                        [row * query_embedding_size..(row + 1) * query_embedding_size],
                ) {
                    *output += weight * embedded;
                }
            }
            result
        });
        let action_features = actions
            .iter()
            .map(|action| super::neural::encode_action_features(&game_state, action, game))
            .flatten()
            .collect::<Vec<_>>();
        let action_embeddings =
            forward_batch_tower(&self.action_tower, &action_features, actions.len());
        let action_embedding_size = self
            .action_tower
            .last()
            .expect("validated action tower")
            .output_size;
        let actor = normalized.actor;
        let mut head_context = context_embedding;
        head_context.extend_from_slice(&pooled[actor]);
        head_context.extend_from_slice(&pooled[1 - actor]);
        let mut query_action_batch = Vec::with_capacity(
            legal_combos.len() * actions.len() * (query_embedding_size + action_embedding_size),
        );
        for row in 0..legal_combos.len() {
            let query = &query_embeddings[actor]
                [row * query_embedding_size..(row + 1) * query_embedding_size];
            for action in 0..actions.len() {
                query_action_batch.extend_from_slice(query);
                query_action_batch.extend_from_slice(
                    &action_embeddings
                        [action * action_embedding_size..(action + 1) * action_embedding_size],
                );
            }
        }
        let logits = forward_batch_head(
            &self.head,
            &head_context,
            &query_action_batch,
            legal_combos.len() * actions.len(),
        );
        let mut probabilities = vec![0.0; COMBO_COUNT * actions.len()];
        for (row, combo) in legal_combos.iter().enumerate() {
            let offset = row * actions.len();
            let source_offset = *combo * actions.len();
            let composed_logits = logits[offset..offset + actions.len()]
                .iter()
                .enumerate()
                .map(|(action, value)| {
                    let residual = f64::from(*value);
                    source_policy.map_or(residual, |source| {
                        source[source_offset + action].max(1e-12).ln() + residual
                    })
                })
                .collect::<Vec<_>>();
            let strategy = super::neural::stable_softmax(&composed_logits);
            probabilities[*combo * actions.len()..(*combo + 1) * actions.len()]
                .copy_from_slice(&strategy);
        }
        if probabilities
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err("range-conditioned policy produced invalid probabilities".to_owned());
        }
        Ok(probabilities)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RangePolicyEvaluationReport {
    pub schema: String,
    pub network_sha256: String,
    pub dataset_sha256: String,
    pub source_dataset_match: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_policy_sha256: Option<String>,
    pub records: usize,
    pub weighted_public_reach: f64,
    pub weighted_probability_mae: f64,
    pub weighted_total_variation: f64,
    pub weighted_teacher_cross_entropy: f64,
    pub weighted_teacher_kl: f64,
    pub maximum_combo_teacher_kl: f64,
    pub reach_weighted_primary_action_agreement: f64,
    pub teacher_ev_minus_candidate_ev_bb: f64,
    pub maximum_probability_sum_error: f64,
    pub minimum_scored_combo_coverage: f64,
    pub maximum_stored_source_probability_difference: f64,
    pub validation: BlueprintValidation,
}

/// Evaluate a frozen range-conditioned policy using Rust serving inference.
/// The dataset hash and solver provenance are pinned before any metric is
/// accepted, so Python/MLX diagnostics cannot silently substitute for runtime
/// behavior.
pub fn evaluate_range_conditioned_policy_dataset(
    network_path: &Path,
    dataset_path: &Path,
    allow_independent_dataset: bool,
    source_policy_path: Option<&Path>,
) -> Result<RangePolicyEvaluationReport, Box<dyn Error>> {
    let dataset_bytes = fs::read(dataset_path)?;
    let dataset_sha256 = format!("{:x}", Sha256::digest(&dataset_bytes));
    let network = RangeConditionedPolicyNetwork::read(network_path)?;
    let source_policy = if network.requires_source_policy() {
        let path = source_policy_path
            .ok_or("range-conditioned residual evaluation requires a source network")?;
        let policy = FrozenPolicy::load(path)?;
        network.validate_source_policy_sha256(policy.bundle_sha256())?;
        Some(policy)
    } else {
        None
    };
    let source_dataset_match = network.source_dataset_sha256 == dataset_sha256;
    if !source_dataset_match && !allow_independent_dataset {
        return Err("range policy source dataset hash does not match evaluation corpus".into());
    }
    let network_sha256 = network
        .artifact_sha256
        .clone()
        .expect("read range policy records an artifact hash");
    let decoder = GzDecoder::new(dataset_bytes.as_slice());
    let mut lines = BufReader::new(decoder).lines();
    let metadata: serde_json::Value =
        serde_json::from_str(&lines.next().ok_or("range policy dataset is empty")??)?;
    if metadata["record_type"] != "metadata"
        || metadata["schema"] != "hu-range-conditioned-postflop-policy-dataset-v1"
        || metadata["feature_schema"] != RANGE_POLICY_FEATURE_SCHEMA_V1
        || metadata["context_size"] != SHARED_CONTEXT_BOARD_RELATIVE_COUNT
        || metadata["query_size"] != SHARED_QUERY_BOARD_RELATIVE_COUNT
        || metadata["action_feature_schema"] != ACTION_FEATURE_SCHEMA_V1
        || metadata["action_feature_count"] != ACTION_FEATURE_COUNT
        || metadata["teacher"]["validation"]["status"] != "accepted_for_training"
        || metadata["records"].as_u64().unwrap_or(0) == 0
    {
        return Err("range policy dataset metadata is incompatible or unvalidated".into());
    }
    let depth = metadata["depth_bb"]
        .as_f64()
        .ok_or("range policy dataset omits its depth")?;
    if (depth - network.depth_bb).abs() > EPSILON {
        return Err("range policy dataset and network use different depths".into());
    }
    let mut game = BlueprintConfig::default();
    game.effective_stack_bb = depth;
    if let Ok(action_abstraction) =
        serde_json::from_value(metadata["teacher"]["actionAbstraction"].clone())
    {
        game.action_abstraction = action_abstraction;
    }
    game.validate()?;
    let conflicts = combo_conflicts();
    let mut records = 0usize;
    let mut public_weight = 0.0f64;
    let mut combo_weight = 0.0f64;
    let mut absolute_probability_error = 0.0f64;
    let mut total_variation = 0.0f64;
    let mut cross_entropy = 0.0f64;
    let mut teacher_entropy = 0.0f64;
    let mut maximum_kl = 0.0f64;
    let mut primary_agreement = 0.0f64;
    let mut ev_loss = 0.0f64;
    let mut maximum_sum_error = 0.0f64;
    let mut minimum_coverage = 1.0f64;
    let mut maximum_source_difference = 0.0f64;
    for line in lines {
        let record: RangeConditionedPolicyRecord = serde_json::from_str(&line?)?;
        if record.record_type != "range_conditioned_average_strategy"
            || !record.weight.is_finite()
            || record.weight <= 0.0
            || record.ranges.iter().any(|range| range.len() != COMBO_COUNT)
        {
            return Err("range policy dataset contains an invalid record".into());
        }
        let state = PublicBeliefState {
            street: record.state.street,
            board: record.state.board.clone(),
            actor: record.state.actor,
            invested_bb: record.state.invested_bb,
            street_invested_bb: record.state.street_invested_bb,
            last_full_raise_bb: record.state.last_full_raise_bb,
            aggressions: record.state.aggressions,
            checks: record.state.checks,
            raise_reopened: record.state.raise_reopened,
            public_history: record.state.public_history.clone(),
            ranges: std::array::from_fn(|player| {
                record.ranges[player]
                    .iter()
                    .map(|value| f64::from(*value))
                    .collect()
            }),
            trajectory: record.state.trajectory.clone(),
        };
        let normalized =
            state.validate_street_and_normalize(&game, state.street, state.street.board_len())?;
        let game_state = normalized.game_state();
        let actions = game_state.legal_actions(&game);
        let action_count = actions.len();
        if action_count == 0
            || record.action_labels
                != actions
                    .iter()
                    .map(|action| action.label.clone())
                    .collect::<Vec<_>>()
            || record.action_features.len() != action_count
            || record.action_features.iter().any(|features| {
                features.len() != ACTION_FEATURE_COUNT
                    || features.iter().any(|value| !value.is_finite())
            })
            || record.probabilities.len() != COMBO_COUNT * action_count
            || record.action_values_bb.len() != COMBO_COUNT * action_count
        {
            return Err("range policy record does not match its legal action state".into());
        }
        for (stored, action) in record.action_features.iter().zip(&actions) {
            let exact = super::neural::encode_action_features(&game_state, action, &game);
            if stored
                .iter()
                .zip(exact)
                .any(|(left, right)| (left - right).abs() > 1e-6)
            {
                return Err("range policy record action features fail Rust parity".into());
            }
        }
        let source_probabilities = source_policy
            .as_ref()
            .map(|policy| {
                policy.bundle_strategy_matrix(&game_state, &normalized.board, &actions, &game)
            })
            .transpose()?;
        if let Some(source) = &source_probabilities {
            if record.source_policy_probabilities.len() != source.len() {
                return Err(
                    "residual range policy record omits its pinned source probabilities".into(),
                );
            }
            for (stored, exact) in record.source_policy_probabilities.iter().zip(source) {
                maximum_source_difference =
                    maximum_source_difference.max((f64::from(*stored) - exact).abs());
            }
            if maximum_source_difference > 1e-6 {
                return Err(
                    "stored residual source probabilities fail exact Rust bundle parity".into(),
                );
            }
        }
        let candidate = network.strategy(&normalized, &game, source_probabilities.as_deref())?;
        let actor = normalized.actor;
        let masses = (0..COMBO_COUNT)
            .map(|combo| {
                compatible_mass_from_conflicts(&normalized.ranges[1 - actor], &conflicts, combo)
            })
            .collect::<Vec<_>>();
        let joint_mass = normalized.ranges[actor]
            .iter()
            .zip(&masses)
            .map(|(reach, mass)| reach * mass)
            .sum::<f64>();
        if joint_mass <= EPSILON {
            return Err("range policy record has no compatible private deals".into());
        }
        let mut scored = 0usize;
        let mut reachable = 0usize;
        for combo in 0..COMBO_COUNT {
            let private_weight = normalized.ranges[actor][combo] * masses[combo] / joint_mass;
            if private_weight <= EPSILON {
                continue;
            }
            reachable += 1;
            let offset = combo * action_count;
            let teacher = &record.probabilities[offset..offset + action_count];
            let predicted = &candidate[offset..offset + action_count];
            let teacher_sum = teacher.iter().map(|value| f64::from(*value)).sum::<f64>();
            let predicted_sum = predicted.iter().sum::<f64>();
            maximum_sum_error = maximum_sum_error
                .max((teacher_sum - 1.0).abs())
                .max((predicted_sum - 1.0).abs());
            if (teacher_sum - 1.0).abs() > 1e-5
                || (predicted_sum - 1.0).abs() > 1e-6
                || teacher
                    .iter()
                    .any(|value| !value.is_finite() || *value < 0.0)
                || predicted
                    .iter()
                    .any(|value| !value.is_finite() || *value < 0.0)
            {
                return Err("range policy record contains invalid probability mass".into());
            }
            scored += 1;
            let weight = f64::from(record.weight) * private_weight;
            let mut local_l1 = 0.0;
            let mut local_cross_entropy = 0.0;
            let mut local_entropy = 0.0;
            let mut teacher_ev = 0.0;
            let mut predicted_ev = 0.0;
            for action in 0..action_count {
                let target = f64::from(teacher[action]);
                let policy = predicted[action].max(1e-12);
                local_l1 += (target - policy).abs();
                if target > 0.0 {
                    local_cross_entropy -= target * policy.ln();
                    local_entropy -= target * target.ln();
                }
                let value = f64::from(record.action_values_bb[offset + action]);
                if !value.is_finite() {
                    return Err("range policy record contains a non-finite action EV".into());
                }
                teacher_ev += target * value;
                predicted_ev += policy * value;
            }
            let local_kl = (local_cross_entropy - local_entropy).max(0.0);
            maximum_kl = maximum_kl.max(local_kl);
            combo_weight += weight;
            absolute_probability_error += weight * local_l1 / action_count as f64;
            total_variation += weight * local_l1 / 2.0;
            cross_entropy += weight * local_cross_entropy;
            teacher_entropy += weight * local_entropy;
            ev_loss += weight * (teacher_ev - predicted_ev);
            let teacher_primary = teacher
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .expect("non-empty teacher actions")
                .0;
            let candidate_primary = predicted
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .expect("non-empty candidate actions")
                .0;
            if teacher_primary == candidate_primary {
                primary_agreement += weight;
            }
        }
        minimum_coverage = minimum_coverage.min(scored as f64 / reachable.max(1) as f64);
        public_weight += f64::from(record.weight);
        records += 1;
    }
    if records != metadata["records"].as_u64().unwrap_or(0) as usize || combo_weight <= EPSILON {
        return Err("range policy dataset record count or reach weight is invalid".into());
    }
    let validation = BlueprintValidation {
        status: "accepted_for_comparison".to_owned(),
        reasons: vec![
            "exact Rust serving inference covered every reachable teacher combo with valid probability mass; release activation still requires paired full-game gates"
                .to_owned(),
        ],
    };
    Ok(RangePolicyEvaluationReport {
        schema: "hu-range-conditioned-policy-rust-evaluation-v1".to_owned(),
        network_sha256,
        dataset_sha256,
        source_dataset_match,
        source_policy_sha256: source_policy
            .as_ref()
            .map(|policy| policy.bundle_sha256().to_owned()),
        records,
        weighted_public_reach: public_weight,
        weighted_probability_mae: absolute_probability_error / combo_weight,
        weighted_total_variation: total_variation / combo_weight,
        weighted_teacher_cross_entropy: cross_entropy / combo_weight,
        weighted_teacher_kl: (cross_entropy - teacher_entropy).max(0.0) / combo_weight,
        maximum_combo_teacher_kl: maximum_kl,
        reach_weighted_primary_action_agreement: primary_agreement / combo_weight,
        teacher_ev_minus_candidate_ev_bb: ev_loss / combo_weight,
        maximum_probability_sum_error: maximum_sum_error,
        minimum_scored_combo_coverage: minimum_coverage,
        maximum_stored_source_probability_difference: maximum_source_difference,
        validation,
    })
}

type SharedContexts = [Vec<f32>; 2];
type SharedQueries = [Vec<Vec<f32>>; 2];

#[derive(Debug)]
struct BoardQueryFeatures {
    context: [f32; 17],
    strengths: Vec<u32>,
    queries: Vec<Vec<f32>>,
}

fn board_query_features(board: &[u8]) -> Arc<BoardQueryFeatures> {
    static CACHE: OnceLock<Mutex<BTreeMap<Vec<u8>, Arc<BoardQueryFeatures>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    if let Some(features) = cache
        .lock()
        .expect("board feature cache")
        .get(board)
        .cloned()
    {
        return features;
    }
    let combos = all_combos();
    let mut context = [0.0f32; 17];
    let mut suit_counts = [0.0f32; 4];
    for card in board {
        context[(card >> 2) as usize] += 0.25;
        suit_counts[(card & 3) as usize] += 0.25;
    }
    suit_counts.sort_by(|left, right| right.total_cmp(left));
    context[13..].copy_from_slice(&suit_counts);
    let mut strengths = vec![0u32; COMBO_COUNT];
    let mut queries = vec![vec![0.0f32; SHARED_QUERY_STRUCTURAL_COUNT]; COMBO_COUNT];
    let mut river_category_counts = vec![[0u8; 9]; COMBO_COUNT];
    let mut improvements = vec![0u8; COMBO_COUNT];
    let blocked = board.iter().copied().collect::<BTreeSet<_>>();
    for combo in &combos {
        let [first, second] = combo.cards();
        if blocked.contains(&first) || blocked.contains(&second) {
            continue;
        }
        let first_rank = first >> 2;
        let second_rank = second >> 2;
        let pair = first_rank == second_rank;
        let (high, low) = if !pair && second_rank > first_rank {
            (second, first)
        } else {
            (first, second)
        };
        let high_rank = (high >> 2) as usize;
        let low_rank = (low >> 2) as usize;
        let mut first_suit_board = [0.0f32; 13];
        let mut second_suit_board = [0.0f32; 13];
        for card in board {
            let rank = (card >> 2) as usize;
            first_suit_board[rank] += f32::from((card & 3) == (high & 3));
            second_suit_board[rank] += f32::from((card & 3) == (low & 3));
        }
        let (high_suit_board, low_suit_board) = if pair {
            (
                std::array::from_fn(|rank| first_suit_board[rank] + second_suit_board[rank]),
                std::array::from_fn(|rank| {
                    (first_suit_board[rank] - second_suit_board[rank]).abs()
                }),
            )
        } else {
            (first_suit_board, second_suit_board)
        };
        let query = &mut queries[combo.key()];
        query[high_rank] = 1.0;
        query[13 + low_rank] = 1.0;
        query[26] = f32::from(pair);
        query[27] = f32::from((high & 3) == (low & 3));
        query[28..41].copy_from_slice(&high_suit_board);
        query[41..54].copy_from_slice(&low_suit_board);
        query[54] = high_suit_board.iter().sum::<f32>() / 4.0;
        query[55] = low_suit_board.iter().sum::<f32>() / 4.0;
        let mut cards = board.to_vec();
        cards.extend([first, second]);
        let current = evaluate(&cards);
        strengths[combo.key()] = current;
        let current_category = (current >> 24) as usize;
        query[56 + current_category] = 1.0;
        if board.len() < 5 {
            let mut future_cards = 0u8;
            for next_card in 0..52u8 {
                if blocked.contains(&next_card) || next_card == first || next_card == second {
                    continue;
                }
                cards.insert(board.len(), next_card);
                let final_category = (evaluate(&cards) >> 24) as usize;
                cards.remove(board.len());
                river_category_counts[combo.key()][final_category] += 1;
                improvements[combo.key()] += u8::from(final_category > current_category);
                future_cards += 1;
            }
            debug_assert_eq!(future_cards as usize, 50 - board.len());
        } else {
            river_category_counts[combo.key()][current_category] = 1;
        }
    }
    let mut strength_counts = BTreeMap::<u32, usize>::new();
    for strength in strengths.iter().filter(|strength| **strength > 0) {
        *strength_counts.entry(*strength).or_default() += 1;
    }
    let legal_count = strength_counts.values().sum::<usize>() as f32;
    let mut lower = 0usize;
    let mut percentile = BTreeMap::new();
    for (strength, count) in strength_counts {
        percentile.insert(strength, (lower as f32 + count as f32 / 2.0) / legal_count);
        lower += count;
    }
    for combo in 0..COMBO_COUNT {
        if strengths[combo] == 0 {
            continue;
        }
        queries[combo][65] = percentile[&strengths[combo]];
        let future_cards = if board.len() < 5 { 50 - board.len() } else { 1 } as f32;
        for category in 0..9 {
            queries[combo][66 + category] =
                river_category_counts[combo][category] as f32 / future_cards;
        }
        queries[combo][75] = improvements[combo] as f32 / future_cards;
    }
    let result = Arc::new(BoardQueryFeatures {
        context,
        strengths,
        queries,
    });
    let mut guard = cache.lock().expect("board feature cache");
    if guard.len() >= BOARD_QUERY_FEATURE_CACHE_ENTRIES {
        if let Some(oldest) = guard.keys().next().cloned() {
            guard.remove(&oldest);
        }
    }
    guard.insert(board.to_vec(), result.clone());
    result
}

#[allow(clippy::type_complexity)]
fn shared_combo_features(
    board: &[u8],
    actor: usize,
    invested: [f64; 2],
    ranges: &[Vec<f64>; 2],
    conflicts: &[Vec<usize>],
    depth_bb: f64,
    feature_schema: &str,
) -> (SharedContexts, SharedQueries) {
    let combos = all_combos();
    let board_features = board_query_features(board);
    let mut card_mass = [[0.0f64; 52]; 2];
    let mut rank_mass = [[0.0f64; 13]; 2];
    let mut suit_mass = [[0.0f64; 4]; 2];
    let mut class_mass: [Vec<f64>; 2] = std::array::from_fn(|_| vec![0.0; HAND_CLASS_COUNT]);
    for player in 0..2 {
        for combo in &combos {
            let weight = ranges[player][combo.key()];
            if weight <= 0.0 {
                continue;
            }
            let [first, second] = combo.cards();
            card_mass[player][first as usize] += weight;
            card_mass[player][second as usize] += weight;
            let first_rank = (first >> 2) as usize;
            let second_rank = (second >> 2) as usize;
            rank_mass[player][first_rank] += weight;
            if second_rank != first_rank {
                rank_mass[player][second_rank] += weight;
            }
            let first_suit = (first & 3) as usize;
            let second_suit = (second & 3) as usize;
            suit_mass[player][first_suit] += weight;
            if second_suit != first_suit {
                suit_mass[player][second_suit] += weight;
            }
            class_mass[player][hand_class_index(first, second)] += weight;
        }
    }
    let uses_board_relative = matches!(
        feature_schema,
        SHARED_FEATURE_SCHEMA_V2 | SHARED_FEATURE_SCHEMA_V3 | RANGE_POLICY_FEATURE_SCHEMA_V1
    );
    let mut board_relative_features = vec![[0.0f64; 29]; COMBO_COUNT];
    let mut board_relative_totals = [[0.0f64; 29]; 2];
    if uses_board_relative {
        for combo in &combos {
            let key = combo.key();
            let source = &board_features.queries[key];
            for feature in 0..9 {
                board_relative_features[key][feature] = source[56 + feature] as f64;
            }
            for feature in 0..10 {
                board_relative_features[key][9 + feature] = source[66 + feature] as f64;
            }
            let strength_bin = ((source[65] * 10.0) as usize).min(9);
            board_relative_features[key][19 + strength_bin] = 1.0;
            for player in 0..2 {
                for feature in 0..29 {
                    board_relative_totals[player][feature] +=
                        ranges[player][key] * board_relative_features[key][feature];
                }
            }
        }
    }
    let immediate_equity = [
        current_range_equity(&board_features.strengths, &ranges[1], conflicts),
        current_range_equity(&board_features.strengths, &ranges[0], conflicts),
    ];
    let exact_runout_equity = (feature_schema == SHARED_FEATURE_SCHEMA_V3)
        .then(|| exact_turn_range_equities(board, ranges, conflicts));
    let (context_count, query_count) =
        shared_feature_sizes(feature_schema).expect("validated shared feature schema");
    let mut contexts: SharedContexts = std::array::from_fn(|_| Vec::new());
    let mut queries: SharedQueries = std::array::from_fn(|_| Vec::with_capacity(COMBO_COUNT));
    let range_totals = [ranges[0].iter().sum::<f64>(), ranges[1].iter().sum::<f64>()];
    for player in 0..2 {
        let opponent = 1 - player;
        let context = &mut contexts[player];
        context.extend(board_features.context);
        context.extend([
            f32::from(actor == player),
            f32::from(actor == opponent),
            (invested[player] / depth_bb) as f32,
            (invested[opponent] / depth_bb) as f32,
        ]);
        context.extend(
            class_mass[player]
                .iter()
                .map(|value| scaled_log_feature(*value, HAND_CLASS_COUNT as f64)),
        );
        context.extend(
            class_mass[opponent]
                .iter()
                .map(|value| scaled_log_feature(*value, HAND_CLASS_COUNT as f64)),
        );
        if uses_board_relative {
            let own_total = range_totals[player].max(EPSILON);
            let opponent_total = range_totals[opponent].max(EPSILON);
            context.extend(
                board_relative_totals[player][..9]
                    .iter()
                    .map(|value| (value / own_total) as f32),
            );
            context.extend(
                board_relative_totals[opponent][..9]
                    .iter()
                    .map(|value| (value / opponent_total) as f32),
            );
            context.extend(
                board_relative_totals[player][19..]
                    .iter()
                    .map(|value| (value / own_total) as f32),
            );
            context.extend(
                board_relative_totals[opponent][19..]
                    .iter()
                    .map(|value| (value / opponent_total) as f32),
            );
            context.extend(
                board_relative_totals[player][9..19]
                    .iter()
                    .map(|value| (value / own_total) as f32),
            );
            context.extend(
                board_relative_totals[opponent][9..19]
                    .iter()
                    .map(|value| (value / opponent_total) as f32),
            );
        }
        debug_assert_eq!(context.len(), context_count);
        for combo in &combos {
            let [first, second] = combo.cards();
            let first_rank = first >> 2;
            let second_rank = second >> 2;
            let pair = first_rank == second_rank;
            let (high, low) = if !pair && second_rank > first_rank {
                (second, first)
            } else {
                (first, second)
            };
            let high_rank = (high >> 2) as usize;
            let low_rank = (low >> 2) as usize;
            let mut query = board_features.queries[combo.key()].clone();
            query.resize(query_count, 0.0);
            let key = combo.key();
            query[76] = scaled_log_feature(ranges[player][key], COMBO_COUNT as f64);
            query[77] = scaled_log_feature(ranges[opponent][key], COMBO_COUNT as f64);
            query[78] = compatible_mass_from_conflicts(&ranges[opponent], conflicts, key) as f32;
            query[79] = compatible_mass_from_conflicts(&ranges[player], conflicts, key) as f32;
            let (own_cards, opponent_cards, own_suits, opponent_suits) = if pair {
                (
                    [
                        card_mass[player][high as usize] + card_mass[player][low as usize],
                        (card_mass[player][high as usize] - card_mass[player][low as usize]).abs(),
                    ],
                    [
                        card_mass[opponent][high as usize] + card_mass[opponent][low as usize],
                        (card_mass[opponent][high as usize] - card_mass[opponent][low as usize])
                            .abs(),
                    ],
                    [
                        suit_mass[player][(high & 3) as usize]
                            + suit_mass[player][(low & 3) as usize],
                        (suit_mass[player][(high & 3) as usize]
                            - suit_mass[player][(low & 3) as usize])
                            .abs(),
                    ],
                    [
                        suit_mass[opponent][(high & 3) as usize]
                            + suit_mass[opponent][(low & 3) as usize],
                        (suit_mass[opponent][(high & 3) as usize]
                            - suit_mass[opponent][(low & 3) as usize])
                            .abs(),
                    ],
                )
            } else {
                (
                    [
                        card_mass[player][high as usize],
                        card_mass[player][low as usize],
                    ],
                    [
                        card_mass[opponent][high as usize],
                        card_mass[opponent][low as usize],
                    ],
                    [
                        suit_mass[player][(high & 3) as usize],
                        suit_mass[player][(low & 3) as usize],
                    ],
                    [
                        suit_mass[opponent][(high & 3) as usize],
                        suit_mass[opponent][(low & 3) as usize],
                    ],
                )
            };
            for (offset, value) in own_cards.into_iter().chain(opponent_cards).enumerate() {
                query[80 + offset] = scaled_log_feature(value, 26.0);
            }
            for (offset, value) in [
                rank_mass[player][high_rank],
                rank_mass[player][low_rank],
                rank_mass[opponent][high_rank],
                rank_mass[opponent][low_rank],
            ]
            .into_iter()
            .enumerate()
            {
                query[84 + offset] = scaled_log_feature(value, 6.5);
            }
            for (offset, value) in own_suits.into_iter().chain(opponent_suits).enumerate() {
                query[88 + offset] = scaled_log_feature(value, 2.0);
            }
            query[92] = range_totals[player] as f32;
            query[93] = range_totals[opponent] as f32;
            query[94] = exact_runout_equity
                .as_ref()
                .map_or(immediate_equity[player][key], |values| values[player][key]);
            if uses_board_relative {
                let compatible = compatible_mass_from_conflicts(&ranges[opponent], conflicts, key);
                for feature in 0..29 {
                    let blocked = conflicts[key]
                        .iter()
                        .map(|other| {
                            ranges[opponent][*other] * board_relative_features[*other][feature]
                        })
                        .sum::<f64>();
                    query[SHARED_QUERY_COUNT + feature] = if compatible > EPSILON {
                        ((board_relative_totals[opponent][feature] - blocked) / compatible)
                            .clamp(0.0, 1.0) as f32
                    } else {
                        0.0
                    };
                }
            }
            queries[player].push(query);
        }
    }
    (contexts, queries)
}

fn current_range_equity(
    strengths: &[u32],
    opponent_range: &[f64],
    conflicts: &[Vec<usize>],
) -> Vec<f32> {
    let unique = strengths
        .iter()
        .copied()
        .filter(|strength| *strength > 0)
        .collect::<BTreeSet<_>>();
    let ranks = unique
        .iter()
        .copied()
        .enumerate()
        .map(|(rank, strength)| (strength, rank))
        .collect::<BTreeMap<_, _>>();
    let mut group_mass = vec![0.0f64; unique.len()];
    for (strength, weight) in strengths.iter().zip(opponent_range) {
        if let Some(rank) = ranks.get(strength) {
            group_mass[*rank] += *weight;
        }
    }
    let mut lower_by_rank = vec![0.0f64; unique.len()];
    let mut running = 0.0;
    for (rank, mass) in group_mass.iter().enumerate() {
        lower_by_rank[rank] = running;
        running += *mass;
    }
    (0..COMBO_COUNT)
        .map(|own| {
            let Some(rank) = ranks.get(&strengths[own]).copied() else {
                return 0.0;
            };
            let mut lower = lower_by_rank[rank];
            let mut equal = group_mass[rank];
            for opponent in &conflicts[own] {
                let weight = opponent_range[*opponent];
                match strengths[*opponent].cmp(&strengths[own]) {
                    std::cmp::Ordering::Less => lower -= weight,
                    std::cmp::Ordering::Equal => equal -= weight,
                    std::cmp::Ordering::Greater => {}
                }
            }
            let compatible = compatible_mass_from_conflicts(opponent_range, conflicts, own);
            if compatible > EPSILON {
                ((lower.max(0.0) + equal.max(0.0) / 2.0) / compatible) as f32
            } else {
                0.0
            }
        })
        .collect()
}

fn exact_turn_range_equities(
    board: &[u8],
    ranges: &[Vec<f64>; 2],
    conflicts: &[Vec<usize>],
) -> [Vec<f32>; 2] {
    assert_eq!(
        board.len(),
        4,
        "exact turn equity requires four board cards"
    );
    let original: [u8; 4] = board.try_into().expect("validated turn board");
    let (key, suit_permutation) = canonical_turn_board_suits(original);
    let canonical_combo_keys = all_combos()
        .into_iter()
        .map(|combo| {
            let [first, second] = combo.cards();
            Combo::new(
                permute_card_suit(first, suit_permutation),
                permute_card_suit(second, suit_permutation),
            )
            .key()
        })
        .collect::<Vec<_>>();
    let cell = {
        let mut cache = DENSE_TURN_EQUITY_CACHE
            .lock()
            .expect("dense turn equity cache poisoned");
        cache.cell(key)
    };
    let matrix = cell
        .get_or_init(|| compute_exact_turn_equity_units(key))
        .clone();
    std::array::from_fn(|player| {
        (0..COMBO_COUNT)
            .map(|own| {
                let compatible =
                    compatible_mass_from_conflicts(&ranges[1 - player], conflicts, own);
                if compatible > EPSILON {
                    let row = canonical_combo_keys[own] * COMBO_COUNT;
                    let numerator = ranges[1 - player]
                        .iter()
                        .enumerate()
                        .map(|(opponent, weight)| {
                            weight * f64::from(matrix[row + canonical_combo_keys[opponent]]) / 88.0
                        })
                        .sum::<f64>();
                    (numerator / compatible).clamp(0.0, 1.0) as f32
                } else {
                    0.0
                }
            })
            .collect()
    })
}

fn permute_card_suit(card: u8, permutation: [u8; 4]) -> u8 {
    (card & !3) | permutation[(card & 3) as usize]
}

/// Returns the lexicographically smallest suit-isomorphic turn board and the
/// original-to-canonical suit permutation. Exact equity is invariant under a
/// global suit relabeling, so canonical matrices can be shared without
/// changing any card-removal or showdown result.
fn canonical_turn_board_suits(board: [u8; 4]) -> ([u8; 4], [u8; 4]) {
    let mut best_board = [u8::MAX; 4];
    let mut best_permutation = [0, 1, 2, 3];
    for first in 0..4u8 {
        for second in 0..4u8 {
            if second == first {
                continue;
            }
            for third in 0..4u8 {
                if third == first || third == second {
                    continue;
                }
                let fourth = 6 - first - second - third;
                let permutation = [first, second, third, fourth];
                let mut candidate = board.map(|card| permute_card_suit(card, permutation));
                candidate.sort_unstable();
                if candidate < best_board {
                    best_board = candidate;
                    best_permutation = permutation;
                }
            }
        }
    }
    (best_board, best_permutation)
}

fn compute_exact_turn_equity_units(board: [u8; 4]) -> Arc<Vec<u8>> {
    let combos = all_combos();
    let legal = combos
        .iter()
        .map(|combo| !combo.cards().iter().any(|card| board.contains(card)))
        .collect::<Vec<_>>();
    let mut counts = vec![0u8; COMBO_COUNT * COMBO_COUNT];
    for river in 0..52u8 {
        if board.contains(&river) {
            continue;
        }
        let ranked = combos
            .iter()
            .enumerate()
            .filter_map(|(key, combo)| {
                let cards = combo.cards();
                (legal[key] && !cards.contains(&river)).then_some((
                    key,
                    *combo,
                    evaluate(&[
                        board[0], board[1], board[2], board[3], river, cards[0], cards[1],
                    ]),
                ))
            })
            .collect::<Vec<_>>();
        for left_index in 0..ranked.len() {
            let (left_key, left_combo, left_score) = ranked[left_index];
            for &(right_key, right_combo, right_score) in &ranked[left_index + 1..] {
                if left_combo.overlaps(right_combo) {
                    continue;
                }
                let units = equity_units(left_score, right_score) as u8;
                counts[left_key * COMBO_COUNT + right_key] += units;
                counts[right_key * COMBO_COUNT + left_key] += 2 - units;
            }
        }
    }
    Arc::new(counts)
}

fn hand_class_index(first: u8, second: u8) -> usize {
    let first_rank = (first >> 2) as usize;
    let second_rank = (second >> 2) as usize;
    if first_rank == second_rank {
        return first_rank;
    }
    let high = first_rank.max(second_rank);
    let low = first_rank.min(second_rank);
    let unordered_index = high * (high - 1) / 2 + low;
    13 + unordered_index * 2 + usize::from((first & 3) == (second & 3))
}

fn scaled_log_feature(value: f64, scale: f64) -> f32 {
    (value.max(0.0).mul_add(scale, 1.0).ln() / scale.ln_1p()) as f32
}

#[derive(Clone, Debug)]
pub struct FlopResolveConfig {
    pub game: BlueprintConfig,
    pub state: PublicBeliefState,
    pub iterations: u64,
    pub averaging_delay: u64,
    pub regret_matching_plus: bool,
    pub value_network: PublicValueNetwork,
    /// Additional independently trained turn value estimates used only while
    /// optimizing the frozen flop strategy. Their counterfactual values are
    /// averaged before the single zero-sum projection, so the resolver no
    /// longer overfits one continuation model's errors.
    pub auxiliary_value_networks: Vec<PublicValueNetwork>,
    pub threads: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FlopResolveMetrics {
    pub information_sets: usize,
    pub turn_leaf_evaluations: u64,
    pub exact_all_in_terminal_evaluations: u64,
    pub profile_value_p0_bb: f64,
    pub profile_value_p1_bb: f64,
    pub depth_limited_best_response_value_p0_bb: f64,
    pub depth_limited_best_response_value_p1_bb: f64,
    pub depth_limited_exploitability_bb_per_hand: f64,
    pub unresolved_uniform_exploitability_bb_per_hand: f64,
    pub resolver_relative_exploitability_improvement: f64,
    pub maximum_leaf_zero_sum_residual_before_projection_bb: f64,
    pub zero_sum_residual_after_projection_bb: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FlopSolution {
    pub schema: String,
    pub method: String,
    pub approximate: bool,
    #[serde(default)]
    pub effective_stack_bb: f64,
    pub value_network_seed: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value_network_sha256: Option<String>,
    pub uses_exact_ranges: bool,
    pub value_network_source_dataset_sha256: Option<String>,
    #[serde(default)]
    pub value_network_source_policy_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub auxiliary_value_network_seeds: Vec<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub auxiliary_value_network_sha256s: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluation_value_network_seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluation_value_network_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluation_value_network_source_dataset_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluation_value_network_source_policy_sha256: Option<String>,
    #[serde(default)]
    pub evaluation_has_distinct_training_identity: bool,
    pub state: PublicBeliefState,
    pub iterations: u64,
    #[serde(default)]
    pub averaging_delay: u64,
    #[serde(default)]
    pub regret_matching_plus: bool,
    #[serde(default)]
    pub dcfr: DcfrParameters,
    #[serde(default)]
    pub threads: usize,
    pub strategies: Vec<PublicBeliefStrategy>,
    pub counterfactual_values_bb: [Vec<f32>; 2],
    pub opponent_compatible_mass: [Vec<f32>; 2],
    pub metrics: FlopResolveMetrics,
    pub validation: BlueprintValidation,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FlopConvergenceCheckpoint {
    pub iterations: u64,
    pub metrics: FlopResolveMetrics,
    pub validation: BlueprintValidation,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FlopConvergenceReport {
    pub schema: String,
    pub method: String,
    pub approximate: bool,
    pub value_network_seed: u64,
    pub value_network_sha256: Option<String>,
    pub value_network_source_dataset_sha256: Option<String>,
    pub value_network_source_policy_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub auxiliary_value_network_seeds: Vec<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub auxiliary_value_network_sha256s: Vec<String>,
    pub evaluation_value_network_seed: u64,
    pub evaluation_value_network_sha256: Option<String>,
    pub evaluation_value_network_source_dataset_sha256: Option<String>,
    pub evaluation_value_network_source_policy_sha256: Option<String>,
    #[serde(default)]
    pub evaluation_has_distinct_training_identity: bool,
    pub state: PublicBeliefState,
    pub averaging_delay: u64,
    #[serde(default)]
    pub regret_matching_plus: bool,
    #[serde(default)]
    pub dcfr: DcfrParameters,
    pub threads: usize,
    pub checkpoints: Vec<FlopConvergenceCheckpoint>,
    #[serde(default)]
    pub checkpoint_solutions: Vec<FlopSolution>,
    pub final_strategy_sha256: String,
    pub final_solution: FlopSolution,
}

impl FlopConvergenceReport {
    pub fn solution_at_iterations(&self, iterations: Option<u64>) -> Result<FlopSolution, String> {
        let Some(iterations) = iterations else {
            return Ok(self.final_solution.clone());
        };
        if self.final_solution.iterations == iterations {
            return Ok(self.final_solution.clone());
        }
        self.checkpoint_solutions
            .iter()
            .find(|solution| solution.iterations == iterations)
            .cloned()
            .ok_or_else(|| {
                format!("flop convergence report has no frozen strategy at iteration {iterations}")
            })
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FlopRangeResponseCheckpoint {
    pub iterations: u64,
    pub response_value_p0_bb: f64,
    pub response_value_p1_bb: f64,
    pub response_gain_p0_bb: f64,
    pub response_gain_p1_bb: f64,
    pub range_consistent_response_gain_bb_per_hand: f64,
    pub maximum_zero_sum_residual_bb: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FlopResponseComboDeviation {
    pub combo_key: usize,
    pub cards: [u8; 2],
    pub card_names: [String; 2],
    pub hand_class: String,
    pub reach_probability: f64,
    pub total_variation: f64,
    pub frozen_primary_action: String,
    pub response_primary_action: String,
    pub frozen_probabilities: Vec<f32>,
    pub response_probabilities: Vec<f32>,
    pub action_ev_bb: Vec<f64>,
    pub frozen_strategy_ev_bb: f64,
    pub response_strategy_ev_bb: f64,
    pub best_action_ev_bb: f64,
    pub frozen_ev_loss_bb: f64,
    pub response_ev_loss_bb: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FlopResponseInformationSetAttribution {
    pub public_history: Vec<String>,
    pub actor: usize,
    pub action_labels: Vec<String>,
    pub node_reach_probability: f64,
    pub frozen_action_frequencies: Vec<f64>,
    pub response_action_frequencies: Vec<f64>,
    pub action_frequency_deltas: Vec<f64>,
    pub conditional_action_ev_bb: Vec<f64>,
    pub conditional_frozen_strategy_ev_bb: f64,
    pub conditional_response_strategy_ev_bb: f64,
    pub conditional_best_action_ev_bb: f64,
    pub conditional_frozen_strategy_ev_loss_bb: f64,
    pub conditional_response_strategy_ev_loss_bb: f64,
    pub reach_weighted_combo_policy_total_variation: f64,
    pub reach_weighted_primary_action_agreement: f64,
    pub maximum_combo_total_variation: f64,
    pub top_combo_deviations: Vec<FlopResponseComboDeviation>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FlopRangeResponseReport {
    pub schema: String,
    pub method: String,
    pub approximate: bool,
    pub interpretation: String,
    pub frozen_strategy_sha256: String,
    pub frozen_strategy_iterations: u64,
    pub strategy_value_network_seed: u64,
    pub strategy_value_network_sha256: Option<String>,
    pub strategy_value_network_source_dataset_sha256: Option<String>,
    pub strategy_value_network_source_policy_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub strategy_auxiliary_value_network_seeds: Vec<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub strategy_auxiliary_value_network_sha256s: Vec<String>,
    pub evaluation_value_network_seed: u64,
    pub evaluation_value_network_sha256: Option<String>,
    pub evaluation_value_network_source_dataset_sha256: Option<String>,
    pub evaluation_value_network_source_policy_sha256: Option<String>,
    #[serde(default)]
    pub evaluation_has_distinct_training_identity: bool,
    pub state: PublicBeliefState,
    pub baseline_profile_value_p0_bb: f64,
    pub baseline_profile_value_p1_bb: f64,
    pub response_averaging_delay: u64,
    #[serde(default)]
    pub response_regret_matching_plus: bool,
    #[serde(default)]
    pub response_dcfr: DcfrParameters,
    pub threads: usize,
    pub checkpoints: Vec<FlopRangeResponseCheckpoint>,
    #[serde(default)]
    pub final_response_strategies: [Vec<PublicBeliefStrategy>; 2],
    #[serde(default)]
    pub information_set_attribution: [Vec<FlopResponseInformationSetAttribution>; 2],
    pub validation: BlueprintValidation,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FlopContinuationValues {
    pub schema: String,
    #[serde(default)]
    pub regret_matching_plus: bool,
    #[serde(default)]
    pub dcfr: DcfrParameters,
    pub counterfactual_values_bb: [Vec<f32>; 2],
    pub profile_value_p0_bb: f64,
    pub profile_value_p1_bb: f64,
    pub exact_all_in_terminal_evaluations: u64,
    pub maximum_leaf_zero_sum_residual_before_projection_bb: f64,
    pub zero_sum_residual_after_projection_bb: f64,
}

#[derive(Clone)]
struct FlopSolver {
    config: FlopResolveConfig,
    legal: [Vec<bool>; 2],
    conflicts: Arc<Vec<Vec<usize>>>,
    nodes: BTreeMap<Vec<String>, RangeNode>,
    turn_leaf_evaluations: Cell<u64>,
    exact_all_in_terminal_evaluations: Cell<u64>,
    all_in_equities: OnceLock<Arc<Vec<f32>>>,
    maximum_leaf_zero_sum_residual: Cell<f64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct ResolverTurnLeaf {
    root_board: [u8; 3],
    public_history: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    trajectory: Vec<TrajectoryAction>,
    board: [u8; 4],
    actor: usize,
    invested: [f64; 2],
    ranges: [Vec<f64>; 2],
    reach_probability: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct ResolverRootLeafCheckpoint {
    schema: String,
    input_sha256: String,
    source_value_network_sha256: String,
    root_index: usize,
    root_board: [u8; 3],
    leaf_population: usize,
    leaf_probability_mass: f64,
    selected_leaves: Vec<ResolverTurnLeaf>,
}

impl FlopSolver {
    fn new(mut config: FlopResolveConfig) -> Result<Self, String> {
        config.game.validate()?;
        config.value_network.validate()?;
        for network in &config.auxiliary_value_networks {
            network.validate()?;
            if !config.value_network.has_distinct_training_identity(network) {
                return Err(
                    "auxiliary turn value networks must have distinct training identities"
                        .to_owned(),
                );
            }
        }
        for (index, network) in config.auxiliary_value_networks.iter().enumerate() {
            if config.auxiliary_value_networks[..index]
                .iter()
                .any(|other| !other.has_distinct_training_identity(network))
            {
                return Err("auxiliary turn value networks must be pairwise distinct".to_owned());
            }
        }
        if config.iterations < 2
            || config.averaging_delay >= config.iterations
            || config.threads == 0
        {
            return Err(
                "flop resolving requires alternating iterations and a valid averaging delay"
                    .to_owned(),
            );
        }
        config.state = config
            .state
            .validate_street_and_normalize(&config.game, Street::Flop, 3)?;
        let legal = std::array::from_fn(|player| {
            config.state.ranges[player]
                .iter()
                .map(|weight| *weight > 0.0)
                .collect()
        });
        Ok(Self {
            config,
            legal,
            conflicts: combo_conflicts(),
            nodes: BTreeMap::new(),
            turn_leaf_evaluations: Cell::new(0),
            exact_all_in_terminal_evaluations: Cell::new(0),
            all_in_equities: OnceLock::new(),
            maximum_leaf_zero_sum_residual: Cell::new(0.0),
        })
    }

    fn train(&mut self) {
        let root = self.config.state.game_state();
        let reaches = self.config.state.ranges.clone();
        for offset in 0..self.config.iterations {
            let round = offset + 1;
            self.walk(root.clone(), reaches.clone(), 0, round, false);
            self.walk(root.clone(), reaches.clone(), 1, round, true);
        }
    }

    fn load_frozen_average_strategies(
        &mut self,
        strategies: &[PublicBeliefStrategy],
    ) -> Result<(), String> {
        if strategies.is_empty() {
            return Err("frozen flop solution contains no strategies".to_owned());
        }
        for strategy in strategies {
            if strategy.actor > 1 || strategy.action_labels.is_empty() {
                return Err("frozen flop strategy has an invalid actor or action set".to_owned());
            }
            let expected = COMBO_COUNT * strategy.action_labels.len();
            if strategy.probabilities.len() != expected
                || strategy
                    .probabilities
                    .iter()
                    .any(|value| !value.is_finite() || *value < 0.0)
            {
                return Err("frozen flop strategy has invalid probabilities".to_owned());
            }
            let node = RangeNode {
                actor: strategy.actor,
                action_labels: strategy.action_labels.clone(),
                regrets: vec![0.0; expected],
                strategy_sum: strategy
                    .probabilities
                    .iter()
                    .map(|value| *value as f64)
                    .collect(),
                last_regret_discount_round: 0,
                last_strategy_discount_round: 0,
            };
            if self
                .nodes
                .insert(strategy.public_history.clone(), node)
                .is_some()
            {
                return Err("frozen flop solution contains duplicate public histories".to_owned());
            }
        }
        self.validate_frozen_strategy_tree(self.config.state.game_state())
    }

    fn validate_frozen_strategy_tree(&self, state: GameState) -> Result<(), String> {
        if state.terminal.is_some() || state.street == Street::Turn {
            return Ok(());
        }
        let actions = state.legal_actions(&self.config.game);
        let node = self.nodes.get(&state.public_history).ok_or_else(|| {
            "frozen flop solution is missing a reachable public history".to_owned()
        })?;
        let labels = actions
            .iter()
            .map(|action| action.label.clone())
            .collect::<Vec<_>>();
        if node.actor != state.actor || node.action_labels != labels {
            return Err("frozen flop strategy does not match the configured game tree".to_owned());
        }
        let action_count = actions.len();
        for combo in 0..COMBO_COUNT {
            let offset = combo * action_count;
            let sum = node.strategy_sum[offset..offset + action_count]
                .iter()
                .sum::<f64>();
            if self.legal[state.actor][combo] && (sum - 1.0).abs() > 1e-4 {
                return Err("frozen flop strategy probabilities do not sum to one".to_owned());
            }
        }
        for action in actions {
            self.validate_frozen_strategy_tree(state.apply(&action, &self.config.game))?;
        }
        Ok(())
    }

    fn walk(
        &mut self,
        state: GameState,
        reaches: [Vec<f64>; 2],
        traverser: usize,
        round: u64,
        accumulate_average: bool,
    ) -> [Vec<f64>; 2] {
        if state.street == Street::Turn && state.terminal.is_none() {
            return self.turn_leaf_values(&state, &reaches);
        }
        if state.terminal.is_some() {
            return self.terminal_values(&state, &reaches);
        }
        let actions = state.legal_actions(&self.config.game);
        let key = state.public_history.clone();
        let actor = state.actor;
        let strategy = {
            let node = self
                .nodes
                .entry(key.clone())
                .or_insert_with(|| RangeNode::new(actor, &actions));
            if actor == traverser {
                node.discount_regrets(round, &self.config.game.dcfr);
            }
            if accumulate_average {
                node.discount_strategy_sum(round, &self.config.game.dcfr);
            }
            node.strategy(&self.legal[actor])
        };
        let action_count = actions.len();
        let mut children = Vec::with_capacity(action_count);
        for (action_index, action) in actions.iter().enumerate() {
            let mut child_reaches = reaches.clone();
            for combo in 0..COMBO_COUNT {
                child_reaches[actor][combo] *= strategy[combo * action_count + action_index];
            }
            children.push(self.walk(
                state.apply(action, &self.config.game),
                child_reaches,
                traverser,
                round,
                accumulate_average,
            ));
        }
        let opponent = 1 - actor;
        let mut values = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        for combo in 0..COMBO_COUNT {
            for action in 0..action_count {
                values[actor][combo] +=
                    strategy[combo * action_count + action] * children[action][actor][combo];
                values[opponent][combo] += children[action][opponent][combo];
            }
        }
        let node = self.nodes.get_mut(&key).expect("flop range node inserted");
        if actor == traverser {
            for combo in 0..COMBO_COUNT {
                if !self.legal[actor][combo] {
                    continue;
                }
                let offset = combo * action_count;
                for action in 0..action_count {
                    node.regrets[offset + action] +=
                        children[action][actor][combo] - values[actor][combo];
                    if self.config.regret_matching_plus {
                        node.regrets[offset + action] = node.regrets[offset + action].max(0.0);
                    }
                }
            }
        }
        if accumulate_average && round > self.config.averaging_delay {
            for combo in 0..COMBO_COUNT {
                let offset = combo * action_count;
                for action in 0..action_count {
                    node.strategy_sum[offset + action] +=
                        reaches[actor][combo] * strategy[offset + action];
                }
            }
        }
        values
    }

    fn reset_player_for_range_response(&mut self, responder: usize) {
        for node in self.nodes.values_mut() {
            if node.actor == responder {
                node.regrets.fill(0.0);
                node.strategy_sum.fill(0.0);
                node.last_regret_discount_round = 0;
                node.last_strategy_discount_round = 0;
            }
        }
    }

    fn range_response_walk(
        &mut self,
        state: GameState,
        reaches: [Vec<f64>; 2],
        responder: usize,
        round: u64,
        regret_matching_plus: bool,
    ) -> [Vec<f64>; 2] {
        if state.street == Street::Turn && state.terminal.is_none() {
            return self.turn_leaf_values(&state, &reaches);
        }
        if state.terminal.is_some() {
            return self.terminal_values(&state, &reaches);
        }
        let actions = state.legal_actions(&self.config.game);
        let key = state.public_history.clone();
        let actor = state.actor;
        let strategy = {
            let node = self.nodes.get_mut(&key).expect("frozen flop response node");
            if actor == responder {
                node.discount_regrets(round, &self.config.game.dcfr);
                node.discount_strategy_sum(round, &self.config.game.dcfr);
                node.strategy(&self.legal[actor])
            } else {
                node.average_strategy(&self.legal[actor])
            }
        };
        let action_count = actions.len();
        let mut children = Vec::with_capacity(action_count);
        for (action_index, action) in actions.iter().enumerate() {
            let mut child_reaches = reaches.clone();
            for combo in 0..COMBO_COUNT {
                child_reaches[actor][combo] *= strategy[combo * action_count + action_index];
            }
            children.push(self.range_response_walk(
                state.apply(action, &self.config.game),
                child_reaches,
                responder,
                round,
                regret_matching_plus,
            ));
        }
        let opponent = 1 - actor;
        let mut values = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        for combo in 0..COMBO_COUNT {
            for action in 0..action_count {
                values[actor][combo] +=
                    strategy[combo * action_count + action] * children[action][actor][combo];
                values[opponent][combo] += children[action][opponent][combo];
            }
        }
        if actor == responder {
            let node = self.nodes.get_mut(&key).expect("flop response node");
            for combo in 0..COMBO_COUNT {
                if !self.legal[actor][combo] {
                    continue;
                }
                let offset = combo * action_count;
                for action in 0..action_count {
                    node.regrets[offset + action] +=
                        children[action][actor][combo] - values[actor][combo];
                    if regret_matching_plus {
                        node.regrets[offset + action] = node.regrets[offset + action].max(0.0);
                    }
                }
                if round > self.config.averaging_delay {
                    for action in 0..action_count {
                        node.strategy_sum[offset + action] +=
                            reaches[actor][combo] * strategy[offset + action];
                    }
                }
            }
        }
        values
    }

    fn average_strategies(&self, actor_filter: Option<usize>) -> Vec<PublicBeliefStrategy> {
        self.nodes
            .iter()
            .filter(|(_, node)| actor_filter.is_none_or(|actor| node.actor == actor))
            .map(|(history, node)| PublicBeliefStrategy {
                public_history: history.clone(),
                actor: node.actor,
                action_labels: node.action_labels.clone(),
                probabilities: node
                    .average_strategy(&self.legal[node.actor])
                    .into_iter()
                    .map(|value| value as f32)
                    .collect(),
                action_values_bb: None,
            })
            .collect()
    }

    fn collect_average_profile_diagnostics(
        &self,
        state: GameState,
        reaches: [Vec<f64>; 2],
        reach_output: &mut BTreeMap<Vec<String>, [Vec<f64>; 2]>,
        action_value_output: &mut BTreeMap<Vec<String>, Vec<Vec<f64>>>,
    ) -> [Vec<f64>; 2] {
        if state.street == Street::Turn && state.terminal.is_none() {
            return self.turn_leaf_values(&state, &reaches);
        }
        if state.terminal.is_some() {
            return self.terminal_values(&state, &reaches);
        }
        let actions = state.legal_actions(&self.config.game);
        let history = state.public_history.clone();
        let actor = state.actor;
        let node = self
            .nodes
            .get(&history)
            .expect("response profile contains every public node");
        let strategy = node.average_strategy(&self.legal[actor]);
        reach_output.insert(history.clone(), reaches.clone());
        let mut children = Vec::with_capacity(actions.len());
        for (action_index, action) in actions.iter().enumerate() {
            let mut child_reaches = reaches.clone();
            for combo in 0..COMBO_COUNT {
                child_reaches[actor][combo] *= strategy[combo * actions.len() + action_index];
            }
            children.push(self.collect_average_profile_diagnostics(
                state.apply(action, &self.config.game),
                child_reaches,
                reach_output,
                action_value_output,
            ));
        }
        action_value_output.insert(
            history,
            children.iter().map(|child| child[actor].clone()).collect(),
        );
        let opponent = 1 - actor;
        let mut values = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        for combo in 0..COMBO_COUNT {
            for action in 0..actions.len() {
                values[actor][combo] +=
                    strategy[combo * actions.len() + action] * children[action][actor][combo];
                values[opponent][combo] += children[action][opponent][combo];
            }
        }
        values
    }

    fn response_information_set_attribution(
        &self,
        responder: usize,
        frozen_strategies: &[PublicBeliefStrategy],
    ) -> Result<Vec<FlopResponseInformationSetAttribution>, String> {
        let frozen = frozen_strategies
            .iter()
            .map(|strategy| (strategy.public_history.clone(), strategy))
            .collect::<BTreeMap<_, _>>();
        let root_reaches = self.config.state.ranges.clone();
        let root_joint = joint_compatibility_mass(&root_reaches);
        if root_joint <= EPSILON {
            return Err("response attribution root ranges have no compatible mass".to_owned());
        }
        let mut reaches = BTreeMap::new();
        let mut action_values = BTreeMap::new();
        self.collect_average_profile_diagnostics(
            self.config.state.game_state(),
            root_reaches,
            &mut reaches,
            &mut action_values,
        );
        let combos = all_combos();
        let mut attribution = Vec::new();
        for (history, node) in &self.nodes {
            if node.actor != responder {
                continue;
            }
            let frozen_strategy = frozen
                .get(history)
                .ok_or_else(|| "response attribution is missing the frozen strategy".to_owned())?;
            if frozen_strategy.actor != responder
                || frozen_strategy.action_labels != node.action_labels
                || frozen_strategy.probabilities.len() != COMBO_COUNT * node.action_labels.len()
            {
                return Err("response attribution frozen strategy is incompatible".to_owned());
            }
            let node_reaches = reaches
                .get(history)
                .ok_or_else(|| "response attribution is missing public reach".to_owned())?;
            let response_probabilities = node.average_strategy(&self.legal[responder]);
            let node_action_values = action_values
                .get(history)
                .ok_or_else(|| "response attribution is missing action values".to_owned())?;
            let action_count = node.action_labels.len();
            if node_action_values.len() != action_count
                || node_action_values
                    .iter()
                    .any(|values| values.len() != COMBO_COUNT)
            {
                return Err("response attribution action values are incompatible".to_owned());
            }
            let mut combo_joint_mass = vec![0.0; COMBO_COUNT];
            let mut node_joint_mass = 0.0;
            for combo in 0..COMBO_COUNT {
                let mass = node_reaches[responder][combo]
                    * compatible_mass_from_conflicts(
                        &node_reaches[1 - responder],
                        &self.conflicts,
                        combo,
                    );
                combo_joint_mass[combo] = mass;
                node_joint_mass += mass;
            }
            if node_joint_mass <= EPSILON {
                continue;
            }
            let mut frozen_action_frequencies = vec![0.0; action_count];
            let mut response_action_frequencies = vec![0.0; action_count];
            let mut conditional_action_ev_bb = vec![0.0; action_count];
            let mut conditional_frozen_strategy_ev_bb = 0.0;
            let mut conditional_response_strategy_ev_bb = 0.0;
            let mut conditional_best_action_ev_bb = 0.0;
            let mut reach_weighted_tv = 0.0;
            let mut primary_agreement_mass = 0.0;
            let mut maximum_combo_total_variation: f64 = 0.0;
            let mut top_combo_deviations = Vec::new();
            for combo in 0..COMBO_COUNT {
                let mass = combo_joint_mass[combo];
                // Node reach can be extremely small while many exact combos
                // still carry valid positive conditional mass. Dropping each
                // such combo against a global epsilon can discard most of a
                // rare node and leave its attributed action frequencies
                // unnormalised. Attribution is an exact post-training pass, so
                // retain every representable positive mass.
                if mass <= 0.0 {
                    continue;
                }
                let offset = combo * action_count;
                let frozen_row = &frozen_strategy.probabilities[offset..offset + action_count];
                let response_row = &response_probabilities[offset..offset + action_count];
                let weight = mass / node_joint_mass;
                let opponent_mass = compatible_mass_from_conflicts(
                    &node_reaches[1 - responder],
                    &self.conflicts,
                    combo,
                );
                if opponent_mass <= 0.0 {
                    continue;
                }
                let combo_action_ev_bb = node_action_values
                    .iter()
                    .map(|values| values[combo] / opponent_mass)
                    .collect::<Vec<_>>();
                if combo_action_ev_bb.iter().any(|value| !value.is_finite()) {
                    return Err("response attribution produced a non-finite action EV".to_owned());
                }
                let frozen_strategy_ev_bb = frozen_row
                    .iter()
                    .zip(&combo_action_ev_bb)
                    .map(|(probability, value)| *probability as f64 * value)
                    .sum::<f64>();
                let response_strategy_ev_bb = response_row
                    .iter()
                    .zip(&combo_action_ev_bb)
                    .map(|(probability, value)| probability * value)
                    .sum::<f64>();
                let best_action_ev_bb = combo_action_ev_bb
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                let mut total_variation = 0.0;
                for action in 0..action_count {
                    frozen_action_frequencies[action] += weight * frozen_row[action] as f64;
                    response_action_frequencies[action] += weight * response_row[action];
                    conditional_action_ev_bb[action] += weight * combo_action_ev_bb[action];
                    total_variation +=
                        (frozen_row[action] as f64 - response_row[action]).abs() / 2.0;
                }
                conditional_frozen_strategy_ev_bb += weight * frozen_strategy_ev_bb;
                conditional_response_strategy_ev_bb += weight * response_strategy_ev_bb;
                conditional_best_action_ev_bb += weight * best_action_ev_bb;
                total_variation = total_variation.clamp(0.0, 1.0);
                reach_weighted_tv += weight * total_variation;
                maximum_combo_total_variation = maximum_combo_total_variation.max(total_variation);
                let frozen_primary = frozen_row
                    .iter()
                    .enumerate()
                    .max_by(|left, right| left.1.total_cmp(right.1))
                    .map(|(index, _)| index)
                    .expect("response node has actions");
                let response_primary = response_row
                    .iter()
                    .enumerate()
                    .max_by(|left, right| left.1.total_cmp(right.1))
                    .map(|(index, _)| index)
                    .expect("response node has actions");
                if frozen_primary == response_primary {
                    primary_agreement_mass += weight;
                }
                if total_variation > EPSILON {
                    let identity = crate::cards::ComboIdentity::from(combos[combo]);
                    top_combo_deviations.push(FlopResponseComboDeviation {
                        combo_key: combo,
                        cards: identity.cards,
                        card_names: identity.card_names,
                        hand_class: identity.label,
                        reach_probability: mass / root_joint,
                        total_variation,
                        frozen_primary_action: node.action_labels[frozen_primary].clone(),
                        response_primary_action: node.action_labels[response_primary].clone(),
                        frozen_probabilities: frozen_row.to_vec(),
                        response_probabilities: response_row
                            .iter()
                            .map(|value| *value as f32)
                            .collect(),
                        action_ev_bb: combo_action_ev_bb,
                        frozen_strategy_ev_bb,
                        response_strategy_ev_bb,
                        best_action_ev_bb,
                        frozen_ev_loss_bb: (best_action_ev_bb - frozen_strategy_ev_bb).max(0.0),
                        response_ev_loss_bb: (best_action_ev_bb - response_strategy_ev_bb).max(0.0),
                    });
                }
            }
            top_combo_deviations.sort_by(|left, right| {
                (right.reach_probability * right.total_variation)
                    .total_cmp(&(left.reach_probability * left.total_variation))
                    .then_with(|| left.combo_key.cmp(&right.combo_key))
            });
            top_combo_deviations.truncate(20);
            let action_frequency_deltas = response_action_frequencies
                .iter()
                .zip(&frozen_action_frequencies)
                .map(|(response, frozen)| response - frozen)
                .collect();
            attribution.push(FlopResponseInformationSetAttribution {
                public_history: history.clone(),
                actor: responder,
                action_labels: node.action_labels.clone(),
                node_reach_probability: node_joint_mass / root_joint,
                frozen_action_frequencies,
                response_action_frequencies,
                action_frequency_deltas,
                conditional_action_ev_bb,
                conditional_frozen_strategy_ev_bb,
                conditional_response_strategy_ev_bb,
                conditional_best_action_ev_bb,
                conditional_frozen_strategy_ev_loss_bb: (conditional_best_action_ev_bb
                    - conditional_frozen_strategy_ev_bb)
                    .max(0.0),
                conditional_response_strategy_ev_loss_bb: (conditional_best_action_ev_bb
                    - conditional_response_strategy_ev_bb)
                    .max(0.0),
                reach_weighted_combo_policy_total_variation: reach_weighted_tv.clamp(0.0, 1.0),
                reach_weighted_primary_action_agreement: primary_agreement_mass.clamp(0.0, 1.0),
                maximum_combo_total_variation: maximum_combo_total_variation.clamp(0.0, 1.0),
                top_combo_deviations,
            });
        }
        attribution.sort_by(|left, right| {
            (right.node_reach_probability * right.reach_weighted_combo_policy_total_variation)
                .total_cmp(
                    &(left.node_reach_probability
                        * left.reach_weighted_combo_policy_total_variation),
                )
                .then_with(|| left.public_history.cmp(&right.public_history))
        });
        Ok(attribution)
    }

    fn projected_profile_values(&self) -> ([f64; 2], f64) {
        let reaches = self.config.state.ranges.clone();
        let joint = joint_compatibility_mass(&reaches);
        let profile =
            self.profile_walk(self.config.state.game_state(), reaches.clone(), None, false);
        let aggregate = |values: &[f64], player: usize| {
            reaches[player]
                .iter()
                .zip(values)
                .map(|(reach, value)| reach * value)
                .sum::<f64>()
                / joint
        };
        let raw = [aggregate(&profile[0], 0), aggregate(&profile[1], 1)];
        let residual = (raw[0] + raw[1]).abs();
        let midpoint = (raw[0] - raw[1]) / 2.0;
        ([midpoint, -midpoint], residual)
    }

    fn terminal_values(&self, state: &GameState, reaches: &[Vec<f64>; 2]) -> [Vec<f64>; 2] {
        match state.terminal.as_ref().expect("terminal") {
            Terminal::Fold { winner } => {
                let utility_p0 = if *winner == 0 {
                    state.invested[1]
                } else {
                    -state.invested[0]
                };
                std::array::from_fn(|player| {
                    let utility = if player == 0 { utility_p0 } else { -utility_p0 };
                    (0..COMBO_COUNT)
                        .map(|combo| {
                            utility
                                * compatible_mass_from_conflicts(
                                    &reaches[1 - player],
                                    &self.conflicts,
                                    combo,
                                )
                        })
                        .collect()
                })
            }
            Terminal::Showdown => self.exact_all_in_terminal_values(state, reaches),
        }
    }

    fn exact_all_in_terminal_values(
        &self,
        state: &GameState,
        reaches: &[Vec<f64>; 2],
    ) -> [Vec<f64>; 2] {
        self.exact_all_in_terminal_evaluations
            .set(self.exact_all_in_terminal_evaluations.get() + 1);
        let board: [u8; 3] = self
            .config
            .state
            .board
            .clone()
            .try_into()
            .expect("validated flop board");
        let equities = self
            .all_in_equities
            .get_or_init(|| exact_flop_all_in_equities(board, &self.legal, self.config.threads));
        let mut values = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        for player_zero in 0..COMBO_COUNT {
            if !self.legal[0][player_zero] {
                continue;
            }
            let row = player_zero * COMBO_COUNT;
            for player_one in 0..COMBO_COUNT {
                let equity_p0 = equities[row + player_one];
                if !equity_p0.is_finite() {
                    continue;
                }
                let utility_p0 = equity_p0 as f64 * state.invested[1]
                    - (1.0 - equity_p0 as f64) * state.invested[0];
                values[0][player_zero] += reaches[1][player_one] * utility_p0;
                values[1][player_one] -= reaches[0][player_zero] * utility_p0;
            }
        }
        values
    }

    fn turn_leaf_values(&self, state: &GameState, reaches: &[Vec<f64>; 2]) -> [Vec<f64>; 2] {
        self.turn_leaf_evaluations
            .set(self.turn_leaf_evaluations.get() + 1);
        let mut result = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        let turns = (0..52u8)
            .filter(|turn| !self.config.state.board.contains(turn))
            .collect::<Vec<_>>();
        let worker_count = self.config.threads.min(turns.len()).max(1);
        let solved = std::thread::scope(|scope| {
            let mut workers = Vec::with_capacity(worker_count);
            for worker in 0..worker_count {
                let assigned = turns
                    .iter()
                    .copied()
                    .skip(worker)
                    .step_by(worker_count)
                    .collect::<Vec<_>>();
                let network = &self.config.value_network;
                let auxiliary_networks = &self.config.auxiliary_value_networks;
                let conflicts = self.conflicts.clone();
                let board = self.config.state.board.clone();
                workers.push(scope.spawn(move || {
                    assigned
                        .into_iter()
                        .filter_map(|turn| {
                            turn_leaf_card_values(
                                network,
                                auxiliary_networks,
                                &conflicts,
                                &board,
                                state.actor,
                                state.invested,
                                reaches,
                                turn,
                            )
                        })
                        .collect::<Vec<_>>()
                }));
            }
            let mut values = Vec::with_capacity(turns.len());
            for worker in workers {
                values.extend(worker.join().expect("turn value worker panicked"));
            }
            values
        });
        let mut maximum_residual = self.maximum_leaf_zero_sum_residual.get();
        for (contribution, residual) in solved {
            maximum_residual = maximum_residual.max(residual);
            for player in 0..2 {
                for combo in 0..COMBO_COUNT {
                    result[player][combo] += contribution[player][combo];
                }
            }
        }
        self.maximum_leaf_zero_sum_residual.set(maximum_residual);
        result
    }

    fn capture_average_turn_leaves(&self) -> Vec<ResolverTurnLeaf> {
        let state = self.config.state.game_state();
        let reaches = self.config.state.ranges.clone();
        let root_joint_mass = joint_compatibility_mass(&reaches);
        let mut leaves = Vec::new();
        self.capture_average_turn_leaves_walk(state, reaches, root_joint_mass, &mut leaves);
        leaves
    }

    fn capture_average_turn_leaves_walk(
        &self,
        state: GameState,
        reaches: [Vec<f64>; 2],
        root_joint_mass: f64,
        leaves: &mut Vec<ResolverTurnLeaf>,
    ) {
        if state.street == Street::Turn && state.terminal.is_none() {
            let root_board: [u8; 3] = self
                .config
                .state
                .board
                .clone()
                .try_into()
                .expect("validated flop board");
            for turn in 0..52u8 {
                if root_board.contains(&turn) {
                    continue;
                }
                let Some((ranges, _, unnormalized_joint_mass)) =
                    normalized_turn_ranges(&reaches, turn)
                else {
                    continue;
                };
                let reach_probability =
                    unnormalized_joint_mass / root_joint_mass.max(EPSILON) / 45.0;
                if !reach_probability.is_finite() || reach_probability <= EPSILON {
                    continue;
                }
                leaves.push(ResolverTurnLeaf {
                    root_board,
                    public_history: state.public_history.clone(),
                    trajectory: state.trajectory.clone(),
                    board: [root_board[0], root_board[1], root_board[2], turn],
                    actor: state.actor,
                    invested: state.invested,
                    ranges,
                    reach_probability,
                });
            }
            return;
        }
        if state.terminal.is_some() {
            return;
        }
        let actions = state.legal_actions(&self.config.game);
        let actor = state.actor;
        let action_count = actions.len();
        let strategy = self
            .nodes
            .get(&state.public_history)
            .expect("trained flop node")
            .average_strategy(&self.legal[actor]);
        for (action_index, action) in actions.iter().enumerate() {
            let mut child_reaches = reaches.clone();
            for combo in 0..COMBO_COUNT {
                child_reaches[actor][combo] *= strategy[combo * action_count + action_index];
            }
            self.capture_average_turn_leaves_walk(
                state.apply(action, &self.config.game),
                child_reaches,
                root_joint_mass,
                leaves,
            );
        }
    }

    fn profile_walk(
        &self,
        state: GameState,
        reaches: [Vec<f64>; 2],
        best_responder: Option<usize>,
        unresolved_uniform: bool,
    ) -> [Vec<f64>; 2] {
        if state.street == Street::Turn && state.terminal.is_none() {
            return self.turn_leaf_values(&state, &reaches);
        }
        if state.terminal.is_some() {
            return self.terminal_values(&state, &reaches);
        }
        let actions = state.legal_actions(&self.config.game);
        let actor = state.actor;
        let action_count = actions.len();
        let strategy = if unresolved_uniform {
            let mut values = vec![0.0; COMBO_COUNT * action_count];
            for combo in 0..COMBO_COUNT {
                if self.legal[actor][combo] {
                    values[combo * action_count..(combo + 1) * action_count]
                        .fill(1.0 / action_count as f64);
                }
            }
            values
        } else {
            self.nodes
                .get(&state.public_history)
                .expect("trained flop node")
                .average_strategy(&self.legal[actor])
        };
        let mut children = Vec::with_capacity(action_count);
        for (action_index, action) in actions.iter().enumerate() {
            let mut child_reaches = reaches.clone();
            if best_responder != Some(actor) {
                for combo in 0..COMBO_COUNT {
                    child_reaches[actor][combo] *= strategy[combo * action_count + action_index];
                }
            }
            children.push(self.profile_walk(
                state.apply(action, &self.config.game),
                child_reaches,
                best_responder,
                unresolved_uniform,
            ));
        }
        let opponent = 1 - actor;
        let mut values = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        for combo in 0..COMBO_COUNT {
            if best_responder == Some(actor) {
                values[actor][combo] = children
                    .iter()
                    .map(|child| child[actor][combo])
                    .fold(f64::NEG_INFINITY, f64::max);
                for child in &children {
                    values[opponent][combo] += child[opponent][combo];
                }
            } else {
                for action in 0..action_count {
                    values[actor][combo] +=
                        strategy[combo * action_count + action] * children[action][actor][combo];
                    values[opponent][combo] += children[action][opponent][combo];
                }
            }
        }
        values
    }

    fn finish_continuation_values(self) -> FlopContinuationValues {
        let root = self.config.state.game_state();
        let reaches = self.config.state.ranges.clone();
        let joint = joint_compatibility_mass(&reaches);
        let mut profile = self.profile_walk(root, reaches.clone(), None, false);
        let aggregate = |values: &[f64], player: usize| {
            reaches[player]
                .iter()
                .zip(values)
                .map(|(reach, value)| reach * value)
                .sum::<f64>()
                / joint
        };
        let root_residual = aggregate(&profile[0], 0) + aggregate(&profile[1], 1);
        for player in 0..2 {
            for (combo, value) in profile[player].iter_mut().enumerate() {
                let mass =
                    compatible_mass_from_conflicts(&reaches[1 - player], &self.conflicts, combo);
                *value -= root_residual / 2.0 * mass;
            }
        }
        let profile_value_p0_bb = aggregate(&profile[0], 0);
        let profile_value_p1_bb = aggregate(&profile[1], 1);
        let opponent_compatible_mass: [Vec<f32>; 2] = std::array::from_fn(|player| {
            (0..COMBO_COUNT)
                .map(|combo| {
                    compatible_mass_from_conflicts(&reaches[1 - player], &self.conflicts, combo)
                        as f32
                })
                .collect()
        });
        let counterfactual_values_bb = std::array::from_fn(|player| {
            profile[player]
                .iter()
                .zip(&opponent_compatible_mass[player])
                .map(|(value, mass)| {
                    if *mass > 0.0 {
                        (*value / *mass as f64) as f32
                    } else {
                        0.0
                    }
                })
                .collect()
        });
        FlopContinuationValues {
            schema: "hu-depth-limited-flop-continuation-values-v1".to_owned(),
            regret_matching_plus: self.config.regret_matching_plus,
            dcfr: self.config.game.dcfr.clone(),
            counterfactual_values_bb,
            profile_value_p0_bb,
            profile_value_p1_bb,
            exact_all_in_terminal_evaluations: self.exact_all_in_terminal_evaluations.get(),
            maximum_leaf_zero_sum_residual_before_projection_bb: self
                .maximum_leaf_zero_sum_residual
                .get(),
            zero_sum_residual_after_projection_bb: (profile_value_p0_bb + profile_value_p1_bb)
                .abs(),
        }
    }

    fn finish(self) -> FlopSolution {
        let root = self.config.state.game_state();
        let reaches = self.config.state.ranges.clone();
        let joint = joint_compatibility_mass(&reaches);
        let mut profile = self.profile_walk(root.clone(), reaches.clone(), None, false);
        let br0 = self.profile_walk(root.clone(), reaches.clone(), Some(0), false);
        let br1 = self.profile_walk(root.clone(), reaches.clone(), Some(1), false);
        let unresolved_profile = self.profile_walk(root.clone(), reaches.clone(), None, true);
        let unresolved_br0 = self.profile_walk(root.clone(), reaches.clone(), Some(0), true);
        let unresolved_br1 = self.profile_walk(root, reaches.clone(), Some(1), true);
        let aggregate = |values: &[f64], player: usize| {
            reaches[player]
                .iter()
                .zip(values)
                .map(|(reach, value)| reach * value)
                .sum::<f64>()
                / joint
        };
        let unprojected_profile_p0 = aggregate(&profile[0], 0);
        let unprojected_profile_p1 = aggregate(&profile[1], 1);
        let root_residual = unprojected_profile_p0 + unprojected_profile_p1;
        for player in 0..2 {
            for combo in 0..COMBO_COUNT {
                let mass =
                    compatible_mass_from_conflicts(&reaches[1 - player], &self.conflicts, combo);
                profile[player][combo] -= root_residual / 2.0 * mass;
            }
        }
        let profile_p0 = aggregate(&profile[0], 0);
        let profile_p1 = aggregate(&profile[1], 1);
        let best_p0 = aggregate(&br0[0], 0);
        let best_p1 = aggregate(&br1[1], 1);
        let exploitability =
            ((best_p0 - unprojected_profile_p0) + (best_p1 - unprojected_profile_p1)) / 2.0;
        let unresolved_p0 = aggregate(&unresolved_profile[0], 0);
        let unresolved_p1 = aggregate(&unresolved_profile[1], 1);
        let unresolved_best_p0 = aggregate(&unresolved_br0[0], 0);
        let unresolved_best_p1 = aggregate(&unresolved_br1[1], 1);
        let unresolved_exploitability =
            ((unresolved_best_p0 - unresolved_p0) + (unresolved_best_p1 - unresolved_p1)) / 2.0;
        let mut diagnostic_reaches = BTreeMap::new();
        let mut diagnostic_action_values = BTreeMap::new();
        self.collect_average_profile_diagnostics(
            self.config.state.game_state(),
            reaches.clone(),
            &mut diagnostic_reaches,
            &mut diagnostic_action_values,
        );
        let mut strategies = self.average_strategies(None);
        for strategy in &mut strategies {
            let node_reaches = diagnostic_reaches
                .get(&strategy.public_history)
                .expect("flop action-value pass contains every strategy node");
            let action_values = diagnostic_action_values
                .get(&strategy.public_history)
                .expect("flop action-value pass contains every strategy action");
            strategy.action_values_bb = Some(
                normalized_action_values_bb(
                    strategy.actor,
                    strategy.action_labels.len(),
                    node_reaches,
                    action_values,
                    &self.conflicts,
                )
                .expect("trained flop action values are finite and compatible"),
            );
        }
        let opponent_compatible_mass: [Vec<f32>; 2] = std::array::from_fn(|player| {
            (0..COMBO_COUNT)
                .map(|combo| {
                    compatible_mass_from_conflicts(&reaches[1 - player], &self.conflicts, combo)
                        as f32
                })
                .collect()
        });
        let counterfactual_values_bb = std::array::from_fn(|player| {
            profile[player]
                .iter()
                .zip(&opponent_compatible_mass[player])
                .map(|(value, mass)| {
                    if *mass > 0.0 {
                        (*value / *mass as f64) as f32
                    } else {
                        0.0
                    }
                })
                .collect()
        });
        let metrics = FlopResolveMetrics {
            information_sets: strategies.len(),
            turn_leaf_evaluations: self.turn_leaf_evaluations.get(),
            exact_all_in_terminal_evaluations: self.exact_all_in_terminal_evaluations.get(),
            profile_value_p0_bb: profile_p0,
            profile_value_p1_bb: profile_p1,
            depth_limited_best_response_value_p0_bb: best_p0,
            depth_limited_best_response_value_p1_bb: best_p1,
            depth_limited_exploitability_bb_per_hand: exploitability.max(0.0),
            unresolved_uniform_exploitability_bb_per_hand: unresolved_exploitability.max(0.0),
            resolver_relative_exploitability_improvement: (unresolved_exploitability
                - exploitability)
                / unresolved_exploitability.max(EPSILON),
            maximum_leaf_zero_sum_residual_before_projection_bb: self
                .maximum_leaf_zero_sum_residual
                .get(),
            zero_sum_residual_after_projection_bb: (profile_p0 + profile_p1).abs(),
        };
        let mut reasons = Vec::new();
        if std::iter::once(&self.config.value_network)
            .chain(&self.config.auxiliary_value_networks)
            .any(|network| network.source_validation_status.as_deref() != Some("accepted"))
        {
            reasons
                .push("a turn value network was trained from a rejected source corpus".to_owned());
        }
        if std::iter::once(&self.config.value_network)
            .chain(&self.config.auxiliary_value_networks)
            .any(|network| !network.uses_exact_ranges)
        {
            reasons
                .push("range-blind value network is an ablation and cannot be promoted".to_owned());
        }
        if metrics.depth_limited_exploitability_bb_per_hand > 0.05 {
            reasons.push(format!(
                "depth-limited exploitability {:.6}bb/hand exceeds 0.05bb/hand",
                metrics.depth_limited_exploitability_bb_per_hand
            ));
        }
        if metrics.resolver_relative_exploitability_improvement <= 0.0 {
            reasons.push("resolver did not improve over the unresolved uniform control".to_owned());
        }
        if metrics.zero_sum_residual_after_projection_bb > 1e-6 {
            reasons.push(format!(
                "root zero-sum residual {:.3e} exceeds 1e-6",
                metrics.zero_sum_residual_after_projection_bb
            ));
        }
        let mut method = "exact_turn_chance_enumeration_with_exact_flop_all_in_runouts_full_vector_turn_cfv_network_and_paired_alternating_dcfr".to_owned();
        if self.config.regret_matching_plus {
            method.push_str("_regret_matching_plus");
        }
        if !self.config.auxiliary_value_networks.is_empty() {
            method.push_str("_mean_independent_turn_cfv_ensemble");
        }
        let auxiliary_value_network_seeds = self
            .config
            .auxiliary_value_networks
            .iter()
            .map(|network| network.seed)
            .collect();
        let auxiliary_value_network_sha256s = self
            .config
            .auxiliary_value_networks
            .iter()
            .filter_map(|network| network.artifact_sha256.clone())
            .collect();
        FlopSolution {
            schema: "hu-depth-limited-flop-public-belief-solution-v2".to_owned(),
            method,
            approximate: true,
            effective_stack_bb: self.config.game.effective_stack_bb,
            value_network_seed: self.config.value_network.seed,
            value_network_sha256: self.config.value_network.artifact_sha256,
            uses_exact_ranges: self.config.value_network.uses_exact_ranges,
            value_network_source_dataset_sha256: self.config.value_network.source_dataset_sha256,
            value_network_source_policy_sha256: self.config.value_network.source_policy_sha256,
            auxiliary_value_network_seeds,
            auxiliary_value_network_sha256s,
            evaluation_value_network_seed: None,
            evaluation_value_network_sha256: None,
            evaluation_value_network_source_dataset_sha256: None,
            evaluation_value_network_source_policy_sha256: None,
            evaluation_has_distinct_training_identity: false,
            state: self.config.state,
            iterations: self.config.iterations,
            averaging_delay: self.config.averaging_delay,
            regret_matching_plus: self.config.regret_matching_plus,
            dcfr: self.config.game.dcfr.clone(),
            threads: self.config.threads,
            strategies,
            counterfactual_values_bb,
            opponent_compatible_mass,
            metrics,
            validation: BlueprintValidation {
                status: "rejected".to_owned(),
                reasons,
            },
        }
    }
}

fn exact_flop_all_in_equities(
    flop: [u8; 3],
    legal: &[Vec<bool>; 2],
    threads: usize,
) -> Arc<Vec<f32>> {
    let relevant_count = (0..COMBO_COUNT)
        .filter(|key| legal[0][*key] || legal[1][*key])
        .count();
    if relevant_count < 100 {
        return compute_exact_flop_all_in_equities(flop, legal, threads);
    }

    let mut key = flop;
    key.sort_unstable();
    let cell = {
        let mut cache = DENSE_ALL_IN_EQUITY_CACHE
            .lock()
            .expect("dense all-in equity cache poisoned");
        if !cache.contains_key(&key) && cache.len() >= DENSE_ALL_IN_EQUITY_CACHE_BOARDS {
            let oldest = *cache.keys().next().expect("non-empty equity cache");
            cache.remove(&oldest);
        }
        cache
            .entry(key)
            .or_insert_with(|| Arc::new(OnceLock::new()))
            .clone()
    };
    cell.get_or_init(|| {
        let combos = all_combos();
        let dense_legal = std::array::from_fn(|_| {
            combos
                .iter()
                .map(|combo| !combo.cards().iter().any(|card| flop.contains(card)))
                .collect::<Vec<_>>()
        });
        compute_exact_flop_all_in_equities(flop, &dense_legal, threads)
    })
    .clone()
}

fn compute_exact_flop_all_in_equities(
    flop: [u8; 3],
    legal: &[Vec<bool>; 2],
    threads: usize,
) -> Arc<Vec<f32>> {
    const EQUITY_UNITS: f32 = 1_980.0;
    let combos = Arc::new(all_combos());
    let relevant = combos
        .iter()
        .enumerate()
        .filter_map(|(key, combo)| (legal[0][key] || legal[1][key]).then_some((key, *combo)))
        .collect::<Vec<_>>();
    let runouts = (0..52u8)
        .filter(|card| !flop.contains(card))
        .flat_map(|turn| {
            ((turn + 1)..52u8)
                .filter(move |river| !flop.contains(river))
                .map(move |river| (turn, river))
        })
        .collect::<Vec<_>>();
    let worker_count = threads.min(runouts.len()).max(1);
    let partials = std::thread::scope(|scope| {
        let mut workers = Vec::with_capacity(worker_count);
        for worker in 0..worker_count {
            let assigned = runouts
                .iter()
                .copied()
                .skip(worker)
                .step_by(worker_count)
                .collect::<Vec<_>>();
            let relevant = &relevant;
            workers.push(scope.spawn(move || {
                let mut counts = vec![0u16; COMBO_COUNT * COMBO_COUNT];
                for (turn, river) in assigned {
                    let mut ranked = Vec::with_capacity(relevant.len());
                    for (key, combo) in relevant {
                        let cards = combo.cards();
                        if cards.contains(&turn) || cards.contains(&river) {
                            continue;
                        }
                        ranked.push((
                            *key,
                            *combo,
                            evaluate(&[cards[0], cards[1], flop[0], flop[1], flop[2], turn, river]),
                        ));
                    }
                    for left_index in 0..ranked.len() {
                        let (left_key, left_combo, left_score) = ranked[left_index];
                        for &(right_key, right_combo, right_score) in &ranked[left_index + 1..] {
                            if left_combo.overlaps(right_combo) {
                                continue;
                            }
                            if legal[0][left_key] && legal[1][right_key] {
                                counts[left_key * COMBO_COUNT + right_key] +=
                                    equity_units(left_score, right_score);
                            }
                            if legal[0][right_key] && legal[1][left_key] {
                                counts[right_key * COMBO_COUNT + left_key] +=
                                    equity_units(right_score, left_score);
                            }
                        }
                    }
                }
                counts
            }));
        }
        workers
            .into_iter()
            .map(|worker| worker.join().expect("all-in equity worker panicked"))
            .collect::<Vec<_>>()
    });
    let mut counts = vec![0u16; COMBO_COUNT * COMBO_COUNT];
    for partial in partials {
        for (total, value) in counts.iter_mut().zip(partial) {
            *total = total
                .checked_add(value)
                .expect("exact all-in equity count overflow");
        }
    }
    let mut equities = vec![f32::NAN; COMBO_COUNT * COMBO_COUNT];
    for (player_zero, first) in combos.iter().enumerate() {
        if !legal[0][player_zero] {
            continue;
        }
        for (player_one, second) in combos.iter().enumerate() {
            if legal[1][player_one] && !first.overlaps(*second) {
                equities[player_zero * COMBO_COUNT + player_one] =
                    counts[player_zero * COMBO_COUNT + player_one] as f32 / EQUITY_UNITS;
            }
        }
    }
    Arc::new(equities)
}

fn equity_units(first: u32, second: u32) -> u16 {
    match first.cmp(&second) {
        std::cmp::Ordering::Greater => 2,
        std::cmp::Ordering::Equal => 1,
        std::cmp::Ordering::Less => 0,
    }
}

fn turn_leaf_card_values(
    network: &PublicValueNetwork,
    auxiliary_networks: &[PublicValueNetwork],
    conflicts: &[Vec<usize>],
    flop_board: &[u8],
    actor: usize,
    invested: [f64; 2],
    reaches: &[Vec<f64>; 2],
    turn: u8,
) -> Option<([Vec<f64>; 2], f64)> {
    let (masked, totals, _) = normalized_turn_ranges(reaches, turn)?;
    let mut board = flop_board.to_vec();
    board.push(turn);
    let mut predicted = network.predict(&board, actor, invested, &masked);
    for auxiliary in auxiliary_networks {
        let auxiliary_prediction = auxiliary.predict(&board, actor, invested, &masked);
        for player in 0..2 {
            for combo in 0..COMBO_COUNT {
                predicted[player][combo] += auxiliary_prediction[player][combo];
            }
        }
    }
    let inverse_network_count = 1.0 / (auxiliary_networks.len() + 1) as f64;
    if !auxiliary_networks.is_empty() {
        for player in &mut predicted {
            for value in player {
                *value *= inverse_network_count;
            }
        }
    }
    let masses: [Vec<f64>; 2] = std::array::from_fn(|player| {
        (0..COMBO_COUNT)
            .map(|combo| compatible_mass_from_conflicts(&masked[1 - player], conflicts, combo))
            .collect()
    });
    let joint = joint_compatibility_mass(&masked);
    let aggregate = |player: usize| {
        masked[player]
            .iter()
            .zip(&predicted[player])
            .zip(&masses[player])
            .map(|((reach, value), mass)| reach * value * mass)
            .sum::<f64>()
            / joint.max(EPSILON)
    };
    let residual = aggregate(0) + aggregate(1);
    for values in &mut predicted {
        for value in values {
            *value -= residual / 2.0;
        }
    }
    let contribution = std::array::from_fn(|player| {
        (0..COMBO_COUNT)
            .map(|combo| {
                predicted[player][combo] * masses[player][combo] * totals[1 - player] / 45.0
            })
            .collect()
    });
    Some((contribution, residual.abs()))
}

fn normalized_turn_ranges(
    reaches: &[Vec<f64>; 2],
    turn: u8,
) -> Option<([Vec<f64>; 2], [f64; 2], f64)> {
    let mut masked = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
    let mut totals = [0.0; 2];
    for player in 0..2 {
        for combo in all_combos() {
            let weight = if combo.cards().contains(&turn) {
                0.0
            } else {
                reaches[player][combo.key()]
            };
            masked[player][combo.key()] = weight;
            totals[player] += weight;
        }
        if totals[player] <= EPSILON {
            return None;
        }
    }
    let unnormalized_joint_mass = joint_compatibility_mass(&masked);
    if unnormalized_joint_mass <= EPSILON {
        return None;
    }
    for player in 0..2 {
        for weight in &mut masked[player] {
            *weight /= totals[player];
        }
    }
    Some((masked, totals, unnormalized_joint_mass))
}

pub fn solve_flop(config: FlopResolveConfig) -> Result<FlopSolution, String> {
    let mut solver = FlopSolver::new(config)?;
    solver.train();
    Ok(solver.finish())
}

fn evaluation_is_distinct_from_optimization_networks(
    primary: &PublicValueNetwork,
    auxiliary: &[PublicValueNetwork],
    evaluation: &PublicValueNetwork,
) -> bool {
    std::iter::once(primary)
        .chain(auxiliary)
        .all(|network| network.has_distinct_training_identity(evaluation))
}

fn evaluation_is_distinct_from_frozen_solution(
    frozen: &FlopSolution,
    evaluation: &PublicValueNetwork,
) -> bool {
    frozen.value_network_seed != evaluation.seed
        && frozen.value_network_sha256.is_some()
        && evaluation.artifact_sha256.is_some()
        && frozen.value_network_sha256.as_ref() != evaluation.artifact_sha256.as_ref()
        && frozen.auxiliary_value_network_seeds.len()
            == frozen.auxiliary_value_network_sha256s.len()
        && frozen
            .auxiliary_value_network_seeds
            .iter()
            .zip(&frozen.auxiliary_value_network_sha256s)
            .all(|(seed, digest)| {
                *seed != evaluation.seed && evaluation.artifact_sha256.as_ref() != Some(digest)
            })
}

/// Train the resolver with one or more frozen leaf models and score its frozen
/// average strategy with another. The artifact records whether the evaluation
/// model is distinct from every optimization model; release callers must
/// require that flag.
pub fn solve_flop_cross_evaluated(
    config: FlopResolveConfig,
    evaluation_value_network: PublicValueNetwork,
) -> Result<FlopSolution, String> {
    evaluation_value_network.validate()?;
    let evaluation_has_distinct_training_identity =
        evaluation_is_distinct_from_optimization_networks(
            &config.value_network,
            &config.auxiliary_value_networks,
            &evaluation_value_network,
        );
    let resolver_seed = config.value_network.seed;
    let resolver_sha256 = config.value_network.artifact_sha256.clone();
    let resolver_uses_exact_ranges = config.value_network.uses_exact_ranges;
    let resolver_source_dataset = config.value_network.source_dataset_sha256.clone();
    let resolver_source_policy = config.value_network.source_policy_sha256.clone();
    let auxiliary_seeds = config
        .auxiliary_value_networks
        .iter()
        .map(|network| network.seed)
        .collect::<Vec<_>>();
    let auxiliary_sha256s = config
        .auxiliary_value_networks
        .iter()
        .filter_map(|network| network.artifact_sha256.clone())
        .collect::<Vec<_>>();
    let evaluation_seed = evaluation_value_network.seed;
    let evaluation_sha256 = evaluation_value_network.artifact_sha256.clone();
    let evaluation_source_dataset = evaluation_value_network.source_dataset_sha256.clone();
    let evaluation_source_policy = evaluation_value_network.source_policy_sha256.clone();
    let regret_matching_plus = config.regret_matching_plus;
    let mut solver = FlopSolver::new(config)?;
    solver.train();
    solver.config.value_network = evaluation_value_network;
    solver.config.auxiliary_value_networks.clear();
    solver.turn_leaf_evaluations.set(0);
    solver.exact_all_in_terminal_evaluations.set(0);
    solver.maximum_leaf_zero_sum_residual.set(0.0);
    let mut solution = solver.finish();
    solution.method = if evaluation_has_distinct_training_identity {
        "frozen_average_resolver_strategy_scored_by_independent_turn_cfv_network_with_exact_turn_chance_and_exact_flop_all_in_runouts"
    } else {
        "frozen_average_resolver_strategy_self_scored_by_same_training_identity_turn_cfv_network_with_exact_turn_chance_and_exact_flop_all_in_runouts"
    }
    .to_owned();
    if regret_matching_plus {
        solution.method.push_str("_regret_matching_plus");
    }
    solution.value_network_seed = resolver_seed;
    solution.value_network_sha256 = resolver_sha256;
    solution.uses_exact_ranges = resolver_uses_exact_ranges;
    solution.value_network_source_dataset_sha256 = resolver_source_dataset;
    solution.value_network_source_policy_sha256 = resolver_source_policy;
    solution.auxiliary_value_network_seeds = auxiliary_seeds;
    solution.auxiliary_value_network_sha256s = auxiliary_sha256s;
    solution.evaluation_value_network_seed = Some(evaluation_seed);
    solution.evaluation_value_network_sha256 = evaluation_sha256;
    solution.evaluation_value_network_source_dataset_sha256 = evaluation_source_dataset;
    solution.evaluation_value_network_source_policy_sha256 = evaluation_source_policy;
    solution.evaluation_has_distinct_training_identity = evaluation_has_distinct_training_identity;
    Ok(solution)
}

/// Measure a single resolver trajectory at multiple iteration counts without
/// restarting its regrets. The diagnostic board is research evidence only;
/// release roots must remain independently precommitted and unopened.
pub fn diagnose_flop_cross_evaluated_convergence(
    config: FlopResolveConfig,
    evaluation_value_network: PublicValueNetwork,
    checkpoints: &[u64],
) -> Result<FlopConvergenceReport, String> {
    evaluation_value_network.validate()?;
    let evaluation_has_distinct_training_identity =
        evaluation_is_distinct_from_optimization_networks(
            &config.value_network,
            &config.auxiliary_value_networks,
            &evaluation_value_network,
        );
    if checkpoints.is_empty()
        || checkpoints.iter().any(|checkpoint| *checkpoint < 2)
        || checkpoints.windows(2).any(|pair| pair[0] >= pair[1])
        || checkpoints.last().copied() != Some(config.iterations)
    {
        return Err(
            "flop convergence checkpoints must be strictly increasing from at least two through the configured final iteration"
                .to_owned(),
        );
    }
    let regret_matching_plus = config.regret_matching_plus;
    let value_network_seed = config.value_network.seed;
    let value_network_sha256 = config.value_network.artifact_sha256.clone();
    let value_network_uses_exact_ranges = config.value_network.uses_exact_ranges;
    let value_network_source_dataset_sha256 = config.value_network.source_dataset_sha256.clone();
    let value_network_source_policy_sha256 = config.value_network.source_policy_sha256.clone();
    let auxiliary_value_network_seeds = config
        .auxiliary_value_networks
        .iter()
        .map(|network| network.seed)
        .collect::<Vec<_>>();
    let auxiliary_value_network_sha256s = config
        .auxiliary_value_networks
        .iter()
        .filter_map(|network| network.artifact_sha256.clone())
        .collect::<Vec<_>>();
    let evaluation_value_network_seed = evaluation_value_network.seed;
    let evaluation_value_network_sha256 = evaluation_value_network.artifact_sha256.clone();
    let evaluation_value_network_source_dataset_sha256 =
        evaluation_value_network.source_dataset_sha256.clone();
    let evaluation_value_network_source_policy_sha256 =
        evaluation_value_network.source_policy_sha256.clone();
    let mut solver = FlopSolver::new(config)?;
    let root = solver.config.state.game_state();
    let reaches = solver.config.state.ranges.clone();
    let mut completed = 0;
    let mut evidence = Vec::with_capacity(checkpoints.len());
    let mut checkpoint_solutions = Vec::with_capacity(checkpoints.len());
    let mut final_solution = None;
    for checkpoint in checkpoints {
        for round in (completed + 1)..=*checkpoint {
            solver.walk(root.clone(), reaches.clone(), 0, round, false);
            solver.walk(root.clone(), reaches.clone(), 1, round, true);
        }
        completed = *checkpoint;
        let mut evaluator = solver.clone();
        evaluator.config.iterations = *checkpoint;
        evaluator.config.value_network = evaluation_value_network.clone();
        evaluator.config.auxiliary_value_networks.clear();
        evaluator.turn_leaf_evaluations.set(0);
        evaluator.exact_all_in_terminal_evaluations.set(0);
        evaluator.maximum_leaf_zero_sum_residual.set(0.0);
        let mut solution = evaluator.finish();
        solution.method = if evaluation_has_distinct_training_identity {
            "frozen_average_resolver_strategy_scored_by_independent_turn_cfv_network_with_exact_turn_chance_and_exact_flop_all_in_runouts"
        } else {
            "frozen_average_resolver_strategy_self_scored_by_same_training_identity_turn_cfv_network_with_exact_turn_chance_and_exact_flop_all_in_runouts"
        }
        .to_owned();
        if regret_matching_plus {
            solution.method.push_str("_regret_matching_plus");
        }
        solution.value_network_seed = value_network_seed;
        solution.value_network_sha256 = value_network_sha256.clone();
        solution.uses_exact_ranges = value_network_uses_exact_ranges;
        solution.value_network_source_dataset_sha256 = value_network_source_dataset_sha256.clone();
        solution.value_network_source_policy_sha256 = value_network_source_policy_sha256.clone();
        solution.auxiliary_value_network_seeds = auxiliary_value_network_seeds.clone();
        solution.auxiliary_value_network_sha256s = auxiliary_value_network_sha256s.clone();
        solution.evaluation_value_network_seed = Some(evaluation_value_network_seed);
        solution.evaluation_value_network_sha256 = evaluation_value_network_sha256.clone();
        solution.evaluation_value_network_source_dataset_sha256 =
            evaluation_value_network_source_dataset_sha256.clone();
        solution.evaluation_value_network_source_policy_sha256 =
            evaluation_value_network_source_policy_sha256.clone();
        solution.evaluation_has_distinct_training_identity =
            evaluation_has_distinct_training_identity;
        evidence.push(FlopConvergenceCheckpoint {
            iterations: *checkpoint,
            metrics: solution.metrics.clone(),
            validation: solution.validation.clone(),
        });
        checkpoint_solutions.push(solution.clone());
        final_solution = Some(solution);
    }
    let final_solution = final_solution.expect("nonempty convergence checkpoints");
    let final_strategy_sha256 = flop_strategy_sha256(&final_solution.strategies);
    let mut method = if evaluation_has_distinct_training_identity {
        "single_paired_alternating_dcfr_trajectory_with_frozen_average_checkpoints_cross_scored_by_independent_turn_cfv_network"
    } else {
        "single_paired_alternating_dcfr_trajectory_with_frozen_average_checkpoints_self_scored_by_same_training_identity_turn_cfv_network"
    }
    .to_owned();
    if regret_matching_plus {
        method.push_str("_regret_matching_plus");
    }
    Ok(FlopConvergenceReport {
        schema: "hu-flop-resolver-convergence-diagnostic-v3".to_owned(),
        method,
        approximate: true,
        value_network_seed,
        value_network_sha256,
        value_network_source_dataset_sha256,
        value_network_source_policy_sha256,
        auxiliary_value_network_seeds,
        auxiliary_value_network_sha256s,
        evaluation_value_network_seed,
        evaluation_value_network_sha256,
        evaluation_value_network_source_dataset_sha256,
        evaluation_value_network_source_policy_sha256,
        evaluation_has_distinct_training_identity,
        state: solver.config.state,
        averaging_delay: solver.config.averaging_delay,
        regret_matching_plus,
        dcfr: solver.config.game.dcfr.clone(),
        threads: solver.config.threads,
        checkpoints: evidence,
        checkpoint_solutions,
        final_strategy_sha256,
        final_solution,
    })
}

fn flop_strategy_sha256(strategies: &[PublicBeliefStrategy]) -> String {
    fn update_u64(digest: &mut Sha256, value: usize) {
        digest.update((value as u64).to_le_bytes());
    }
    fn update_string(digest: &mut Sha256, value: &str) {
        update_u64(digest, value.len());
        digest.update(value.as_bytes());
    }

    let mut digest = Sha256::new();
    digest.update(b"hu-flop-strategy-v1\0");
    update_u64(&mut digest, strategies.len());
    for strategy in strategies {
        update_u64(&mut digest, strategy.public_history.len());
        for history in &strategy.public_history {
            update_string(&mut digest, history);
        }
        update_u64(&mut digest, strategy.actor);
        update_u64(&mut digest, strategy.action_labels.len());
        for label in &strategy.action_labels {
            update_string(&mut digest, label);
        }
        update_u64(&mut digest, strategy.probabilities.len());
        for probability in &strategy.probabilities {
            digest.update(probability.to_bits().to_le_bytes());
        }
    }
    format!("{:x}", digest.finalize())
}

/// Approximate an information-set-consistent response while preserving the
/// responder-induced range passed to the value network. The opponent's frozen
/// average strategy is never updated. This is red-team evidence and not an
/// exploitability upper bound for either the depth-limited or full game.
pub fn evaluate_frozen_flop_range_response_convergence(
    game: BlueprintConfig,
    frozen: &FlopSolution,
    evaluation_value_network: PublicValueNetwork,
    checkpoints: &[u64],
    averaging_delay: u64,
    regret_matching_plus: bool,
    threads: usize,
) -> Result<FlopRangeResponseReport, String> {
    evaluation_value_network.validate()?;
    let evaluation_has_distinct_training_identity =
        evaluation_is_distinct_from_frozen_solution(frozen, &evaluation_value_network);
    if frozen.effective_stack_bb > 0.0
        && (frozen.effective_stack_bb - game.effective_stack_bb).abs() > EPSILON
    {
        return Err("frozen resolver effective stack does not match response game".to_owned());
    }
    if checkpoints.is_empty()
        || checkpoints.iter().any(|checkpoint| *checkpoint < 2)
        || checkpoints.windows(2).any(|pair| pair[0] >= pair[1])
        || averaging_delay >= checkpoints[0]
        || threads == 0
    {
        return Err(
            "range-response checkpoints must be strictly increasing from at least two and begin after the averaging delay"
                .to_owned(),
        );
    }
    let final_iterations = *checkpoints.last().expect("nonempty response checkpoints");
    let frozen_strategy_sha256 = flop_strategy_sha256(&frozen.strategies);
    let mut base = FlopSolver::new(FlopResolveConfig {
        game,
        state: frozen.state.clone(),
        iterations: final_iterations,
        averaging_delay,
        regret_matching_plus: false,
        value_network: evaluation_value_network.clone(),
        auxiliary_value_networks: Vec::new(),
        threads,
    })?;
    base.load_frozen_average_strategies(&frozen.strategies)?;
    let (baseline, baseline_residual) = base.projected_profile_values();
    let mut response_values: [Vec<f64>; 2] =
        std::array::from_fn(|_| Vec::with_capacity(checkpoints.len()));
    let mut response_residuals: [Vec<f64>; 2] =
        std::array::from_fn(|_| Vec::with_capacity(checkpoints.len()));
    let mut final_response_strategies: [Vec<PublicBeliefStrategy>; 2] =
        std::array::from_fn(|_| Vec::new());
    let mut information_set_attribution: [Vec<FlopResponseInformationSetAttribution>; 2] =
        std::array::from_fn(|_| Vec::new());
    for responder in 0..2 {
        let mut solver = base.clone();
        solver.reset_player_for_range_response(responder);
        let root = solver.config.state.game_state();
        let reaches = solver.config.state.ranges.clone();
        let mut completed = 0;
        for checkpoint in checkpoints {
            for round in (completed + 1)..=*checkpoint {
                solver.range_response_walk(
                    root.clone(),
                    reaches.clone(),
                    responder,
                    round,
                    regret_matching_plus,
                );
            }
            completed = *checkpoint;
            let (profile, residual) = solver.projected_profile_values();
            response_values[responder].push(profile[responder].max(baseline[responder]));
            response_residuals[responder].push(residual);
        }
        final_response_strategies[responder] = solver.average_strategies(Some(responder));
        information_set_attribution[responder] =
            solver.response_information_set_attribution(responder, &frozen.strategies)?;
    }
    let evidence = checkpoints
        .iter()
        .enumerate()
        .map(|(index, iterations)| {
            let gain_p0 = (response_values[0][index] - baseline[0]).max(0.0);
            let gain_p1 = (response_values[1][index] - baseline[1]).max(0.0);
            FlopRangeResponseCheckpoint {
                iterations: *iterations,
                response_value_p0_bb: response_values[0][index],
                response_value_p1_bb: response_values[1][index],
                response_gain_p0_bb: gain_p0,
                response_gain_p1_bb: gain_p1,
                range_consistent_response_gain_bb_per_hand: (gain_p0 + gain_p1) / 2.0,
                maximum_zero_sum_residual_bb: baseline_residual
                    .max(response_residuals[0][index])
                    .max(response_residuals[1][index]),
            }
        })
        .collect();
    let mut method = if evaluation_has_distinct_training_identity {
        "one_player_depth_limited_dcfr_with_frozen_opponent_and_response_conditioned_public_ranges_cross_scored_by_independent_turn_cfv_network"
    } else {
        "one_player_depth_limited_dcfr_with_frozen_opponent_and_response_conditioned_public_ranges_self_scored_by_same_training_identity_turn_cfv_network"
    }
    .to_owned();
    if regret_matching_plus {
        method.push_str("_regret_matching_plus");
    }
    Ok(FlopRangeResponseReport {
        schema: "hu-flop-range-response-diagnostic-v1".to_owned(),
        method,
        approximate: true,
        interpretation: "information-set-consistent learned-response rejection evidence; finite response iterations and an approximate leaf network make this a lower-bound search signal, not an exploitability upper bound".to_owned(),
        frozen_strategy_sha256,
        frozen_strategy_iterations: frozen.iterations,
        strategy_value_network_seed: frozen.value_network_seed,
        strategy_value_network_sha256: frozen.value_network_sha256.clone(),
        strategy_value_network_source_dataset_sha256: frozen
            .value_network_source_dataset_sha256
            .clone(),
        strategy_value_network_source_policy_sha256: frozen
            .value_network_source_policy_sha256
            .clone(),
        strategy_auxiliary_value_network_seeds: frozen
            .auxiliary_value_network_seeds
            .clone(),
        strategy_auxiliary_value_network_sha256s: frozen
            .auxiliary_value_network_sha256s
            .clone(),
        evaluation_value_network_seed: evaluation_value_network.seed,
        evaluation_value_network_sha256: evaluation_value_network.artifact_sha256,
        evaluation_value_network_source_dataset_sha256: evaluation_value_network
            .source_dataset_sha256,
        evaluation_value_network_source_policy_sha256: evaluation_value_network
            .source_policy_sha256,
        evaluation_has_distinct_training_identity,
        state: frozen.state.clone(),
        baseline_profile_value_p0_bb: baseline[0],
        baseline_profile_value_p1_bb: baseline[1],
        response_averaging_delay: averaging_delay,
        response_regret_matching_plus: regret_matching_plus,
        response_dcfr: base.config.game.dcfr.clone(),
        threads,
        checkpoints: evidence,
        final_response_strategies,
        information_set_attribution,
        validation: BlueprintValidation {
            status: "diagnostic_only".to_owned(),
            reasons: vec![
                "range-consistent finite learned response cannot establish an exploitability upper bound"
                    .to_owned(),
            ],
        },
    })
}

/// Re-score a serialized frozen resolver strategy without retraining it.
pub fn evaluate_frozen_flop_solution(
    game: BlueprintConfig,
    frozen: &FlopSolution,
    evaluation_value_network: PublicValueNetwork,
    threads: usize,
) -> Result<FlopSolution, String> {
    evaluation_value_network.validate()?;
    let evaluation_has_distinct_training_identity =
        evaluation_is_distinct_from_frozen_solution(frozen, &evaluation_value_network);
    if frozen.effective_stack_bb > 0.0
        && (frozen.effective_stack_bb - game.effective_stack_bb).abs() > EPSILON
    {
        return Err("frozen resolver effective stack does not match evaluation game".to_owned());
    }
    let mut solver = FlopSolver::new(FlopResolveConfig {
        game,
        state: frozen.state.clone(),
        iterations: frozen.iterations.max(2),
        averaging_delay: 0,
        regret_matching_plus: false,
        value_network: evaluation_value_network.clone(),
        auxiliary_value_networks: Vec::new(),
        threads,
    })?;
    solver.load_frozen_average_strategies(&frozen.strategies)?;
    let mut solution = solver.finish();
    solution.method = "serialized_frozen_average_resolver_strategy_scored_by_independent_turn_cfv_network_with_exact_turn_chance_and_exact_flop_all_in_runouts".to_owned();
    if frozen.regret_matching_plus {
        solution.method.push_str("_regret_matching_plus");
    }
    solution.value_network_seed = frozen.value_network_seed;
    solution.averaging_delay = frozen.averaging_delay;
    solution.regret_matching_plus = frozen.regret_matching_plus;
    solution.value_network_sha256 = frozen.value_network_sha256.clone();
    solution.uses_exact_ranges = frozen.uses_exact_ranges;
    solution.value_network_source_dataset_sha256 =
        frozen.value_network_source_dataset_sha256.clone();
    solution.value_network_source_policy_sha256 = frozen.value_network_source_policy_sha256.clone();
    solution.auxiliary_value_network_seeds = frozen.auxiliary_value_network_seeds.clone();
    solution.auxiliary_value_network_sha256s = frozen.auxiliary_value_network_sha256s.clone();
    solution.evaluation_value_network_seed = Some(evaluation_value_network.seed);
    solution.evaluation_value_network_sha256 = evaluation_value_network.artifact_sha256.clone();
    solution.evaluation_value_network_source_dataset_sha256 =
        evaluation_value_network.source_dataset_sha256;
    solution.evaluation_value_network_source_policy_sha256 =
        evaluation_value_network.source_policy_sha256;
    solution.evaluation_has_distinct_training_identity = evaluation_has_distinct_training_identity;
    solution.strategies = frozen.strategies.clone();
    Ok(solution)
}

pub fn solve_flop_continuation_values(
    config: FlopResolveConfig,
) -> Result<FlopContinuationValues, String> {
    let mut solver = FlopSolver::new(config)?;
    solver.train();
    Ok(solver.finish_continuation_values())
}

#[derive(Clone, Debug)]
pub struct RiverSolveConfig {
    pub game: BlueprintConfig,
    pub state: PublicBeliefState,
    pub iterations: u64,
    pub averaging_delay: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PublicBeliefStrategy {
    pub public_history: Vec<String>,
    pub actor: usize,
    pub action_labels: Vec<String>,
    /// Row-major `[combo][action]` frozen average policy.
    pub probabilities: Vec<f32>,
    /// Row-major `[combo][action]` counterfactual action EVs in big blinds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action_values_bb: Option<Vec<f32>>,
}

fn normalized_action_values_bb(
    actor: usize,
    action_count: usize,
    reaches: &[Vec<f64>; 2],
    action_values: &[Vec<f64>],
    conflicts: &[Vec<usize>],
) -> Result<Vec<f32>, String> {
    if actor > 1
        || action_values.len() != action_count
        || action_values
            .iter()
            .any(|values| values.len() != COMBO_COUNT)
    {
        return Err("counterfactual action values have incompatible dimensions".to_owned());
    }
    let mut normalized = vec![0.0f32; COMBO_COUNT * action_count];
    for combo in 0..COMBO_COUNT {
        let opponent_mass = compatible_mass_from_conflicts(&reaches[1 - actor], conflicts, combo);
        if opponent_mass <= 0.0 {
            continue;
        }
        for action in 0..action_count {
            let value = action_values[action][combo] / opponent_mass;
            if !value.is_finite() || value.abs() > 1e6 {
                return Err("counterfactual action values are non-finite or unbounded".to_owned());
            }
            normalized[combo * action_count + action] = value as f32;
        }
    }
    Ok(normalized)
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RiverSolveMetrics {
    pub information_sets: usize,
    pub legal_combinations: [usize; 2],
    pub joint_compatibility_mass: f64,
    pub profile_value_p0_bb: f64,
    pub profile_value_p1_bb: f64,
    pub best_response_value_p0_bb: f64,
    pub best_response_value_p1_bb: f64,
    pub exact_abstract_exploitability_bb_per_hand: f64,
    pub zero_sum_residual_bb: f64,
    pub maximum_probability_sum_error: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RiverSolution {
    pub schema: String,
    pub method: String,
    pub approximate: bool,
    pub game: BlueprintConfig,
    pub state: PublicBeliefState,
    pub iterations: u64,
    pub averaging_delay: u64,
    pub counterfactual_values_bb: [Vec<f32>; 2],
    pub opponent_compatible_mass: [Vec<f32>; 2],
    pub strategies: Vec<PublicBeliefStrategy>,
    pub metrics: RiverSolveMetrics,
    pub validation: BlueprintValidation,
}

#[derive(Clone)]
struct RangeNode {
    actor: usize,
    action_labels: Vec<String>,
    regrets: Vec<f64>,
    strategy_sum: Vec<f64>,
    last_regret_discount_round: u64,
    last_strategy_discount_round: u64,
}

impl RangeNode {
    fn new(actor: usize, actions: &[LegalAction]) -> Self {
        let action_count = actions.len();
        Self {
            actor,
            action_labels: actions.iter().map(|action| action.label.clone()).collect(),
            regrets: vec![0.0; COMBO_COUNT * action_count],
            strategy_sum: vec![0.0; COMBO_COUNT * action_count],
            last_regret_discount_round: 0,
            last_strategy_discount_round: 0,
        }
    }

    fn strategy(&self, legal: &[bool]) -> Vec<f64> {
        let action_count = self.action_labels.len();
        let mut result = vec![0.0; COMBO_COUNT * action_count];
        for combo in 0..COMBO_COUNT {
            if !legal[combo] {
                continue;
            }
            let offset = combo * action_count;
            let positive_sum = self.regrets[offset..offset + action_count]
                .iter()
                .map(|regret| regret.max(0.0))
                .sum::<f64>();
            if positive_sum > EPSILON {
                for action in 0..action_count {
                    result[offset + action] = self.regrets[offset + action].max(0.0) / positive_sum;
                }
            } else {
                result[offset..offset + action_count].fill(1.0 / action_count as f64);
            }
        }
        result
    }

    fn average_strategy(&self, legal: &[bool]) -> Vec<f64> {
        let action_count = self.action_labels.len();
        let mut result = vec![0.0; COMBO_COUNT * action_count];
        for combo in 0..COMBO_COUNT {
            if !legal[combo] {
                continue;
            }
            let offset = combo * action_count;
            let total = self.strategy_sum[offset..offset + action_count]
                .iter()
                .sum::<f64>();
            if total > EPSILON {
                for action in 0..action_count {
                    result[offset + action] = self.strategy_sum[offset + action] / total;
                }
            } else {
                result[offset..offset + action_count].fill(1.0 / action_count as f64);
            }
        }
        result
    }

    fn discount_regrets(&mut self, round: u64, parameters: &DcfrParameters) {
        if round == 0 || self.last_regret_discount_round == round {
            return;
        }
        let time = round as f64;
        let positive_power = time.powf(parameters.positive_regret_exponent);
        let negative_power = time.powf(parameters.negative_regret_exponent);
        let positive_factor = positive_power / (positive_power + 1.0);
        let negative_factor = negative_power / (negative_power + 1.0);
        for regret in &mut self.regrets {
            *regret *= if *regret >= 0.0 {
                positive_factor
            } else {
                negative_factor
            };
        }
        self.last_regret_discount_round = round;
    }

    fn discount_strategy_sum(&mut self, round: u64, parameters: &DcfrParameters) {
        if round == 0 || self.last_strategy_discount_round == round {
            return;
        }
        let time = round as f64;
        let strategy_factor = (time / (time + 1.0)).powf(parameters.strategy_exponent);
        for weight in &mut self.strategy_sum {
            *weight *= strategy_factor;
        }
        self.last_strategy_discount_round = round;
    }
}

fn compatible_masses_from_card_marginals(combos: &[Combo], range: &[f64]) -> Vec<f64> {
    debug_assert_eq!(combos.len(), COMBO_COUNT);
    debug_assert_eq!(range.len(), COMBO_COUNT);
    let total = range.iter().sum::<f64>();
    let mut by_card = [0.0f64; 52];
    for (combo, weight) in combos.iter().zip(range) {
        let [first, second] = combo.cards();
        by_card[first as usize] += *weight;
        by_card[second as usize] += *weight;
    }
    combos
        .iter()
        .enumerate()
        .map(|(index, combo)| {
            let [first, second] = combo.cards();
            // The identical two-card holding occurs in both card marginals, so
            // add it back once after subtracting both blocked-card totals.
            (total - by_card[first as usize] - by_card[second as usize] + range[index]).max(0.0)
        })
        .collect()
}

fn showdown_values_from_card_strength_marginals(
    combos: &[Combo],
    strength_ranks: &[usize],
    strength_group_count: usize,
    opponent_reach: &[f64],
    win: f64,
    loss: f64,
    tie: f64,
) -> Vec<f64> {
    debug_assert_eq!(combos.len(), COMBO_COUNT);
    debug_assert_eq!(strength_ranks.len(), COMBO_COUNT);
    debug_assert_eq!(opponent_reach.len(), COMBO_COUNT);
    let mut by_strength = vec![0.0; strength_group_count];
    let mut by_card_strength = vec![0.0; 52 * strength_group_count];
    let mut by_card = [0.0f64; 52];
    for (index, (combo, weight)) in combos.iter().zip(opponent_reach).enumerate() {
        if *weight == 0.0 {
            continue;
        }
        let rank = strength_ranks[index];
        let [first, second] = combo.cards();
        by_strength[rank] += *weight;
        by_card[first as usize] += *weight;
        by_card[second as usize] += *weight;
        by_card_strength[first as usize * strength_group_count + rank] += *weight;
        by_card_strength[second as usize * strength_group_count + rank] += *weight;
    }

    let mut lower_by_strength = vec![0.0; strength_group_count];
    let mut total = 0.0;
    for (rank, weight) in by_strength.iter().enumerate() {
        lower_by_strength[rank] = total;
        total += *weight;
    }
    let mut lower_by_card_strength = vec![0.0; 52 * strength_group_count];
    for card in 0..52 {
        let mut running = 0.0;
        for rank in 0..strength_group_count {
            let offset = card * strength_group_count + rank;
            lower_by_card_strength[offset] = running;
            running += by_card_strength[offset];
        }
    }

    combos
        .iter()
        .enumerate()
        .map(|(own, combo)| {
            let rank = strength_ranks[own];
            let [first, second] = combo.cards();
            let first_offset = first as usize * strength_group_count + rank;
            let second_offset = second as usize * strength_group_count + rank;
            let lower = (lower_by_strength[rank]
                - lower_by_card_strength[first_offset]
                - lower_by_card_strength[second_offset])
                .max(0.0);
            let equal = (by_strength[rank]
                - by_card_strength[first_offset]
                - by_card_strength[second_offset]
                + opponent_reach[own])
                .max(0.0);
            let compatible = (total - by_card[first as usize] - by_card[second as usize]
                + opponent_reach[own])
                .max(0.0);
            let higher = (compatible - lower - equal).max(0.0);
            lower * win + equal * tie + higher * loss
        })
        .collect()
}

struct RiverSolver {
    config: RiverSolveConfig,
    combos: Vec<Combo>,
    legal: [Vec<bool>; 2],
    strength_ranks: Vec<usize>,
    strength_group_count: usize,
    nodes: BTreeMap<Vec<String>, RangeNode>,
}

impl RiverSolver {
    fn new(config: RiverSolveConfig) -> Result<Self, String> {
        config.game.validate()?;
        if config.iterations < 2 {
            return Err("river solving requires at least two alternating iterations".to_owned());
        }
        if config.averaging_delay >= config.iterations {
            return Err("river averaging delay must be smaller than iterations".to_owned());
        }
        let state = config.state.validate_and_normalize(&config.game)?;
        let combos = all_combos();
        let legal = std::array::from_fn(|player| {
            state.ranges[player]
                .iter()
                .map(|weight| *weight > 0.0)
                .collect()
        });
        let strengths: Vec<u32> = combos
            .iter()
            .map(|combo| {
                let mut cards = state.board.clone();
                cards.extend(combo.cards());
                evaluate(&cards)
            })
            .collect();
        let strength_groups = strengths.iter().copied().collect::<BTreeSet<_>>();
        let strength_to_rank = strength_groups
            .iter()
            .copied()
            .enumerate()
            .map(|(rank, strength)| (strength, rank))
            .collect::<BTreeMap<_, _>>();
        let strength_ranks = strengths
            .iter()
            .map(|strength| strength_to_rank[strength])
            .collect();
        let strength_group_count = strength_groups.len();
        Ok(Self {
            config: RiverSolveConfig { state, ..config },
            combos,
            legal,
            strength_ranks,
            strength_group_count,
            nodes: BTreeMap::new(),
        })
    }

    fn train(&mut self) {
        let root = self.config.state.game_state();
        let reaches = self.config.state.ranges.clone();
        for offset in 0..self.config.iterations {
            let round = offset + 1;
            self.walk(root.clone(), reaches.clone(), 0, round, false);
            self.walk(root.clone(), reaches.clone(), 1, round, true);
        }
    }

    fn walk(
        &mut self,
        state: GameState,
        reaches: [Vec<f64>; 2],
        traverser: usize,
        round: u64,
        accumulate_average: bool,
    ) -> [Vec<f64>; 2] {
        if state.terminal.is_some() {
            return self.terminal_values(&state, &reaches);
        }
        let actions = state.legal_actions(&self.config.game);
        let key = state.public_history.clone();
        let actor = state.actor;
        let strategy = {
            let node = self
                .nodes
                .entry(key.clone())
                .or_insert_with(|| RangeNode::new(actor, &actions));
            assert_eq!(node.actor, actor);
            assert_eq!(
                node.action_labels,
                actions
                    .iter()
                    .map(|action| action.label.clone())
                    .collect::<Vec<_>>()
            );
            if actor == traverser {
                node.discount_regrets(round, &self.config.game.dcfr);
            }
            if accumulate_average {
                node.discount_strategy_sum(round, &self.config.game.dcfr);
            }
            node.strategy(&self.legal[actor])
        };
        let action_count = actions.len();
        let mut children = Vec::with_capacity(action_count);
        for (action_index, action) in actions.iter().enumerate() {
            let mut child_reaches = reaches.clone();
            for combo in 0..COMBO_COUNT {
                child_reaches[actor][combo] *= strategy[combo * action_count + action_index];
            }
            children.push(self.walk(
                state.apply(action, &self.config.game),
                child_reaches,
                traverser,
                round,
                accumulate_average,
            ));
        }
        let opponent = 1 - actor;
        let mut values = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        for combo in 0..COMBO_COUNT {
            for action in 0..action_count {
                values[actor][combo] +=
                    strategy[combo * action_count + action] * children[action][actor][combo];
                values[opponent][combo] += children[action][opponent][combo];
            }
        }
        let node = self.nodes.get_mut(&key).expect("range node inserted");
        if actor == traverser {
            for combo in 0..COMBO_COUNT {
                if !self.legal[actor][combo] {
                    continue;
                }
                let offset = combo * action_count;
                for action in 0..action_count {
                    node.regrets[offset + action] +=
                        children[action][actor][combo] - values[actor][combo];
                }
            }
        }
        if accumulate_average && round > self.config.averaging_delay {
            for combo in 0..COMBO_COUNT {
                if !self.legal[actor][combo] {
                    continue;
                }
                let offset = combo * action_count;
                for action in 0..action_count {
                    node.strategy_sum[offset + action] +=
                        reaches[actor][combo] * strategy[offset + action];
                }
            }
        }
        values
    }

    fn terminal_values(&self, state: &GameState, reaches: &[Vec<f64>; 2]) -> [Vec<f64>; 2] {
        match state.terminal.as_ref().expect("terminal public state") {
            Terminal::Fold { winner } => {
                let utility_p0 = if *winner == 0 {
                    state.invested[1]
                } else {
                    -state.invested[0]
                };
                [
                    self.constant_terminal_values(&reaches[1], utility_p0),
                    self.constant_terminal_values(&reaches[0], -utility_p0),
                ]
            }
            Terminal::Showdown => [
                self.showdown_terminal_values(
                    0,
                    &reaches[1],
                    state.invested[1],
                    -state.invested[0],
                    (state.invested[1] - state.invested[0]) / 2.0,
                ),
                self.showdown_terminal_values(
                    1,
                    &reaches[0],
                    state.invested[0],
                    -state.invested[1],
                    (state.invested[0] - state.invested[1]) / 2.0,
                ),
            ],
        }
    }

    fn constant_terminal_values(&self, opponent_reach: &[f64], utility: f64) -> Vec<f64> {
        compatible_masses_from_card_marginals(&self.combos, opponent_reach)
            .into_iter()
            .map(|mass| utility * mass)
            .collect()
    }

    fn showdown_terminal_values(
        &self,
        _player: usize,
        opponent_reach: &[f64],
        win: f64,
        loss: f64,
        tie: f64,
    ) -> Vec<f64> {
        showdown_values_from_card_strength_marginals(
            &self.combos,
            &self.strength_ranks,
            self.strength_group_count,
            opponent_reach,
            win,
            loss,
            tie,
        )
    }

    fn profile_walk(
        &self,
        state: GameState,
        reaches: [Vec<f64>; 2],
        best_responder: Option<usize>,
    ) -> [Vec<f64>; 2] {
        if state.terminal.is_some() {
            return self.terminal_values(&state, &reaches);
        }
        let actions = state.legal_actions(&self.config.game);
        let key = state.public_history.clone();
        let actor = state.actor;
        let node = self.nodes.get(&key).expect("trained range node");
        let strategy = node.average_strategy(&self.legal[actor]);
        let action_count = actions.len();
        let mut children = Vec::with_capacity(action_count);
        for (action_index, action) in actions.iter().enumerate() {
            let mut child_reaches = reaches.clone();
            if best_responder != Some(actor) {
                for combo in 0..COMBO_COUNT {
                    child_reaches[actor][combo] *= strategy[combo * action_count + action_index];
                }
            }
            children.push(self.profile_walk(
                state.apply(action, &self.config.game),
                child_reaches,
                best_responder,
            ));
        }
        let opponent = 1 - actor;
        let mut values = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        for combo in 0..COMBO_COUNT {
            if best_responder == Some(actor) {
                let best = children
                    .iter()
                    .map(|child| child[actor][combo])
                    .fold(f64::NEG_INFINITY, f64::max);
                values[actor][combo] = best;
                for child in &children {
                    values[opponent][combo] += child[opponent][combo];
                }
            } else {
                for action in 0..action_count {
                    values[actor][combo] +=
                        strategy[combo * action_count + action] * children[action][actor][combo];
                    values[opponent][combo] += children[action][opponent][combo];
                }
            }
        }
        values
    }

    fn finish(self) -> RiverSolution {
        let root = self.config.state.game_state();
        let reaches = self.config.state.ranges.clone();
        let joint_mass = joint_compatibility_mass(&reaches);
        let profile = self.profile_walk(root.clone(), reaches.clone(), None);
        let br0 = self.profile_walk(root.clone(), reaches.clone(), Some(0));
        let br1 = self.profile_walk(root, reaches.clone(), Some(1));
        let aggregate = |values: &[f64], player: usize| {
            reaches[player]
                .iter()
                .zip(values)
                .map(|(reach, value)| reach * value)
                .sum::<f64>()
                / joint_mass
        };
        let profile_p0 = aggregate(&profile[0], 0);
        let profile_p1 = aggregate(&profile[1], 1);
        let best_p0 = aggregate(&br0[0], 0);
        let best_p1 = aggregate(&br1[1], 1);
        let exploitability = ((best_p0 - profile_p0) + (best_p1 - profile_p1)) / 2.0;
        let compatible_mass: [Vec<f32>; 2] = std::array::from_fn(|player| {
            compatible_masses_from_card_marginals(&self.combos, &reaches[1 - player])
                .into_iter()
                .map(|mass| mass as f32)
                .collect()
        });
        let counterfactual_values_bb = std::array::from_fn(|player| {
            profile[player]
                .iter()
                .zip(&compatible_mass[player])
                .map(|(value, mass)| {
                    if *mass > 0.0 {
                        (*value / *mass as f64) as f32
                    } else {
                        0.0
                    }
                })
                .collect()
        });
        let mut maximum_probability_sum_error = 0.0f64;
        let strategies = self
            .nodes
            .iter()
            .map(|(history, node)| {
                let probabilities = node.average_strategy(&self.legal[node.actor]);
                let action_count = node.action_labels.len();
                for combo in 0..COMBO_COUNT {
                    if self.legal[node.actor][combo] {
                        let sum = probabilities[combo * action_count..(combo + 1) * action_count]
                            .iter()
                            .sum::<f64>();
                        maximum_probability_sum_error =
                            maximum_probability_sum_error.max((sum - 1.0).abs());
                    }
                }
                PublicBeliefStrategy {
                    public_history: history.clone(),
                    actor: node.actor,
                    action_labels: node.action_labels.clone(),
                    probabilities: probabilities
                        .into_iter()
                        .map(|value| value as f32)
                        .collect(),
                    action_values_bb: None,
                }
            })
            .collect::<Vec<_>>();
        let zero_sum_residual = (profile_p0 + profile_p1).abs();
        let metrics = RiverSolveMetrics {
            information_sets: strategies.len(),
            legal_combinations: std::array::from_fn(|player| {
                self.legal[player].iter().filter(|legal| **legal).count()
            }),
            joint_compatibility_mass: joint_mass,
            profile_value_p0_bb: profile_p0,
            profile_value_p1_bb: profile_p1,
            best_response_value_p0_bb: best_p0,
            best_response_value_p1_bb: best_p1,
            exact_abstract_exploitability_bb_per_hand: exploitability.max(0.0),
            zero_sum_residual_bb: zero_sum_residual,
            maximum_probability_sum_error,
        };
        let mut reasons = Vec::new();
        if metrics.zero_sum_residual_bb > 1e-8 {
            reasons.push(format!(
                "zero-sum residual {:.3e} exceeds 1e-8",
                metrics.zero_sum_residual_bb
            ));
        }
        if metrics.maximum_probability_sum_error > 1e-6 {
            reasons.push(format!(
                "probability sum error {:.3e} exceeds 1e-6",
                metrics.maximum_probability_sum_error
            ));
        }
        if metrics.exact_abstract_exploitability_bb_per_hand > 0.05 {
            reasons.push(format!(
                "river abstraction exploitability {:.6}bb/hand exceeds 0.05bb/hand",
                metrics.exact_abstract_exploitability_bb_per_hand
            ));
        }
        RiverSolution {
            schema: RIVER_SCHEMA.to_owned(),
            method:
                "paired_alternating_vectorized_dcfr_exact_private-card_and_river_chance_enumeration"
                    .to_owned(),
            approximate: true,
            game: self.config.game,
            state: self.config.state,
            iterations: self.config.iterations,
            averaging_delay: self.config.averaging_delay,
            counterfactual_values_bb,
            opponent_compatible_mass: compatible_mass,
            strategies,
            validation: BlueprintValidation {
                status: if reasons.is_empty() {
                    "accepted"
                } else {
                    "rejected"
                }
                .to_owned(),
                reasons,
            },
            metrics,
        }
    }
}

pub fn solve_river(config: RiverSolveConfig) -> Result<RiverSolution, String> {
    let mut solver = RiverSolver::new(config)?;
    solver.train();
    Ok(solver.finish())
}

#[derive(Clone, Debug)]
pub struct TurnRiverSolveConfig {
    pub game: BlueprintConfig,
    pub state: PublicBeliefState,
    pub iterations: u64,
    pub averaging_delay: u64,
    pub river_refinement_iterations: u64,
    pub regret_matching_plus: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TurnRiverSolveMetrics {
    pub information_sets: usize,
    pub turn_information_sets: usize,
    pub river_information_sets: usize,
    pub exact_river_cards: usize,
    pub profile_value_p0_bb: f64,
    pub profile_value_p1_bb: f64,
    pub best_response_value_p0_bb: f64,
    pub best_response_value_p1_bb: f64,
    pub exact_abstract_exploitability_bb_per_hand: f64,
    pub turn_only_best_response_gain_bb_per_hand: f64,
    pub river_only_best_response_gain_bb_per_hand: f64,
    pub current_strategy_exploitability_bb_per_hand: f64,
    pub zero_sum_residual_bb: f64,
    pub maximum_probability_sum_error: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TurnRiverSolution {
    pub schema: String,
    pub method: String,
    pub approximate: bool,
    pub game: BlueprintConfig,
    pub state: PublicBeliefState,
    pub iterations: u64,
    pub averaging_delay: u64,
    pub river_refinement_iterations: u64,
    pub counterfactual_values_bb: [Vec<f32>; 2],
    pub opponent_compatible_mass: [Vec<f32>; 2],
    pub strategies: Vec<PublicBeliefStrategy>,
    pub metrics: TurnRiverSolveMetrics,
    pub validation: BlueprintValidation,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TurnRiverContinuationValues {
    pub schema: String,
    pub method: String,
    pub joint_iterations: u64,
    pub river_refinement_iterations: u64,
    pub counterfactual_values_bb: [Vec<f32>; 2],
    pub opponent_compatible_mass: [Vec<f32>; 2],
    pub metrics: TurnRiverSolveMetrics,
}

struct RiverBoardData {
    strength_ranks: Vec<usize>,
    strength_group_count: usize,
    legal: [Vec<bool>; 2],
}

/// Solves the complete abstract turn and river continuation in one CFR game.
/// The river card is public chance, but each exact private-card pair still has
/// exactly 44 compatible river outcomes. Keeping chance inside the same game
/// prevents a turn value target from accidentally skipping turn betting.
struct TurnRiverSolver {
    config: TurnRiverSolveConfig,
    combos: Vec<Combo>,
    legal: [Vec<bool>; 2],
    river_cards: Vec<u8>,
    river_blocked_combos: Vec<Vec<usize>>,
    river_data: Vec<Option<RiverBoardData>>,
    nodes: BTreeMap<Vec<String>, RangeNode>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TurnRiverTrainingMode {
    Joint,
    FrozenAverageTurnRiverRefinement,
}

impl TurnRiverSolver {
    fn new(mut config: TurnRiverSolveConfig) -> Result<Self, String> {
        config.game.validate()?;
        if config.iterations < 2 || config.averaging_delay >= config.iterations {
            return Err(
                "turn-river solving requires alternating iterations and a valid averaging delay"
                    .to_owned(),
            );
        }
        config.state = config
            .state
            .validate_street_and_normalize(&config.game, Street::Turn, 4)?;
        let legal: [Vec<bool>; 2] = std::array::from_fn(|player| {
            config.state.ranges[player]
                .iter()
                .map(|weight| *weight > 0.0)
                .collect()
        });
        let combos = all_combos();
        let river_cards = (0..52u8)
            .filter(|card| !config.state.board.contains(card))
            .collect::<Vec<_>>();
        let river_blocked_combos = (0..52u8)
            .map(|card| {
                combos
                    .iter()
                    .filter_map(|combo| combo.cards().contains(&card).then_some(combo.key()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut river_data = (0..52).map(|_| None).collect::<Vec<_>>();
        for river in &river_cards {
            let strengths = combos
                .iter()
                .map(|combo| {
                    if combo.cards().contains(river) {
                        return 0;
                    }
                    let mut cards = config.state.board.clone();
                    cards.push(*river);
                    cards.extend(combo.cards());
                    evaluate(&cards)
                })
                .collect::<Vec<_>>();
            let groups = strengths
                .iter()
                .copied()
                .filter(|strength| *strength > 0)
                .collect::<BTreeSet<_>>();
            let ranks = groups
                .iter()
                .copied()
                .enumerate()
                .map(|(rank, strength)| (strength, rank))
                .collect::<BTreeMap<_, _>>();
            let strength_ranks = strengths
                .iter()
                .map(|strength| ranks.get(strength).copied().unwrap_or(0))
                .collect();
            let river_legal = std::array::from_fn(|player| {
                combos
                    .iter()
                    .map(|combo| legal[player][combo.key()] && !combo.cards().contains(river))
                    .collect()
            });
            river_data[*river as usize] = Some(RiverBoardData {
                strength_ranks,
                strength_group_count: groups.len(),
                legal: river_legal,
            });
        }
        Ok(Self {
            config,
            combos,
            legal,
            river_cards,
            river_blocked_combos,
            river_data,
            nodes: BTreeMap::new(),
        })
    }

    fn train(&mut self) {
        let root = self.config.state.game_state();
        let reaches = self.config.state.ranges.clone();
        for offset in 0..self.config.iterations {
            let round = offset + 1;
            self.walk(
                root.clone(),
                reaches.clone(),
                None,
                0,
                round,
                false,
                TurnRiverTrainingMode::Joint,
            );
            self.walk(
                root.clone(),
                reaches.clone(),
                None,
                1,
                round,
                true,
                TurnRiverTrainingMode::Joint,
            );
        }
        for offset in 0..self.config.river_refinement_iterations {
            let round = self.config.iterations + offset + 1;
            self.walk(
                root.clone(),
                reaches.clone(),
                None,
                0,
                round,
                false,
                TurnRiverTrainingMode::FrozenAverageTurnRiverRefinement,
            );
            self.walk(
                root.clone(),
                reaches.clone(),
                None,
                1,
                round,
                true,
                TurnRiverTrainingMode::FrozenAverageTurnRiverRefinement,
            );
        }
    }

    fn node_key(state: &GameState, river: Option<u8>) -> Vec<String> {
        let mut key = state.public_history.clone();
        if let Some(card) = river {
            key.push(format!("chance:river:{card}"));
        }
        key
    }

    fn river_from_key(key: &[String]) -> Option<u8> {
        key.last()
            .and_then(|part| part.strip_prefix("chance:river:"))
            .and_then(|value| value.parse().ok())
    }

    fn legal_for(&self, river: Option<u8>, player: usize) -> &[bool] {
        match river {
            Some(card) => {
                &self.river_data[card as usize]
                    .as_ref()
                    .expect("known river card")
                    .legal[player]
            }
            None => &self.legal[player],
        }
    }

    fn accumulate_compatible_river_child(
        &self,
        values: &mut [Vec<f64>; 2],
        child: &[Vec<f64>; 2],
        river: u8,
    ) {
        let data = self.river_data[river as usize]
            .as_ref()
            .expect("known river card");
        for player in 0..2 {
            for combo in 0..COMBO_COUNT {
                if data.legal[player][combo] {
                    values[player][combo] += child[player][combo] / 44.0;
                }
            }
        }
    }

    fn walk(
        &mut self,
        state: GameState,
        reaches: [Vec<f64>; 2],
        river: Option<u8>,
        traverser: usize,
        round: u64,
        accumulate_average: bool,
        mode: TurnRiverTrainingMode,
    ) -> [Vec<f64>; 2] {
        if state.terminal.is_some() {
            return self.terminal_values(&state, &reaches, river);
        }
        if state.street == Street::River && river.is_none() {
            return self.chance_walk(state, reaches, traverser, round, accumulate_average, mode);
        }
        let actions = state.legal_actions(&self.config.game);
        let key = Self::node_key(&state, river);
        let actor = state.actor;
        let legal = self.legal_for(river, actor).to_vec();
        let strategy = {
            let node = self
                .nodes
                .entry(key.clone())
                .or_insert_with(|| RangeNode::new(actor, &actions));
            if mode == TurnRiverTrainingMode::FrozenAverageTurnRiverRefinement
                && state.street == Street::Turn
            {
                node.average_strategy(&legal)
            } else {
                if actor == traverser {
                    node.discount_regrets(round, &self.config.game.dcfr);
                }
                if accumulate_average {
                    node.discount_strategy_sum(round, &self.config.game.dcfr);
                }
                node.strategy(&legal)
            }
        };
        let action_count = actions.len();
        let mut children = Vec::with_capacity(action_count);
        for (action_index, action) in actions.iter().enumerate() {
            let mut child_reaches = reaches.clone();
            for combo in 0..COMBO_COUNT {
                child_reaches[actor][combo] *= strategy[combo * action_count + action_index];
            }
            children.push(self.walk(
                state.apply(action, &self.config.game),
                child_reaches,
                river,
                traverser,
                round,
                accumulate_average,
                mode,
            ));
        }
        let opponent = 1 - actor;
        let mut values = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        for combo in 0..COMBO_COUNT {
            for action in 0..action_count {
                values[actor][combo] +=
                    strategy[combo * action_count + action] * children[action][actor][combo];
                values[opponent][combo] += children[action][opponent][combo];
            }
        }
        let node = self.nodes.get_mut(&key).expect("turn-river node inserted");
        let update_node = mode == TurnRiverTrainingMode::Joint || state.street == Street::River;
        if update_node && actor == traverser {
            for combo in 0..COMBO_COUNT {
                if !legal[combo] {
                    continue;
                }
                let offset = combo * action_count;
                for action in 0..action_count {
                    node.regrets[offset + action] +=
                        children[action][actor][combo] - values[actor][combo];
                    if self.config.regret_matching_plus {
                        node.regrets[offset + action] = node.regrets[offset + action].max(0.0);
                    }
                }
            }
        }
        if update_node && accumulate_average && round > self.config.averaging_delay {
            for combo in 0..COMBO_COUNT {
                if !legal[combo] {
                    continue;
                }
                let offset = combo * action_count;
                for action in 0..action_count {
                    node.strategy_sum[offset + action] +=
                        reaches[actor][combo] * strategy[offset + action];
                }
            }
        }
        values
    }

    fn chance_walk(
        &mut self,
        state: GameState,
        reaches: [Vec<f64>; 2],
        traverser: usize,
        round: u64,
        accumulate_average: bool,
        mode: TurnRiverTrainingMode,
    ) -> [Vec<f64>; 2] {
        let mut values = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        let cards = self.river_cards.clone();
        for river in cards {
            let mut masked = reaches.clone();
            for player in 0..2 {
                for combo in &self.river_blocked_combos[river as usize] {
                    masked[player][*combo] = 0.0;
                }
            }
            let child = self.walk(
                state.clone(),
                masked,
                Some(river),
                traverser,
                round,
                accumulate_average,
                mode,
            );
            self.accumulate_compatible_river_child(&mut values, &child, river);
        }
        values
    }

    fn terminal_values(
        &self,
        state: &GameState,
        reaches: &[Vec<f64>; 2],
        river: Option<u8>,
    ) -> [Vec<f64>; 2] {
        match state.terminal.as_ref().expect("terminal turn-river state") {
            Terminal::Fold { winner } => {
                let utility_p0 = if *winner == 0 {
                    state.invested[1]
                } else {
                    -state.invested[0]
                };
                [
                    self.constant_terminal_values(&reaches[1], utility_p0),
                    self.constant_terminal_values(&reaches[0], -utility_p0),
                ]
            }
            Terminal::Showdown => match river {
                Some(card) => self.river_showdown_values(state, reaches, card),
                None => self.turn_all_in_values(state, reaches),
            },
        }
    }

    fn constant_terminal_values(&self, opponent_reach: &[f64], utility: f64) -> Vec<f64> {
        compatible_masses_from_card_marginals(&self.combos, opponent_reach)
            .into_iter()
            .map(|mass| utility * mass)
            .collect()
    }

    fn river_showdown_values(
        &self,
        state: &GameState,
        reaches: &[Vec<f64>; 2],
        river: u8,
    ) -> [Vec<f64>; 2] {
        [
            self.river_player_showdown_values(
                &reaches[1],
                river,
                state.invested[1],
                -state.invested[0],
                (state.invested[1] - state.invested[0]) / 2.0,
            ),
            self.river_player_showdown_values(
                &reaches[0],
                river,
                state.invested[0],
                -state.invested[1],
                (state.invested[0] - state.invested[1]) / 2.0,
            ),
        ]
    }

    fn river_player_showdown_values(
        &self,
        opponent_reach: &[f64],
        river: u8,
        win: f64,
        loss: f64,
        tie: f64,
    ) -> Vec<f64> {
        let data = self.river_data[river as usize]
            .as_ref()
            .expect("known river card");
        let mut values = showdown_values_from_card_strength_marginals(
            &self.combos,
            &data.strength_ranks,
            data.strength_group_count,
            opponent_reach,
            win,
            loss,
            tie,
        );
        for (own, value) in values.iter_mut().enumerate() {
            if !(data.legal[0][own] || data.legal[1][own]) {
                *value = 0.0;
            }
        }
        values
    }

    fn turn_all_in_values(&self, state: &GameState, reaches: &[Vec<f64>; 2]) -> [Vec<f64>; 2] {
        let mut values = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        for river in &self.river_cards {
            let mut masked = reaches.clone();
            for player in 0..2 {
                for combo in &self.river_blocked_combos[*river as usize] {
                    masked[player][*combo] = 0.0;
                }
            }
            let child = self.river_showdown_values(state, &masked, *river);
            for player in 0..2 {
                for combo in 0..COMBO_COUNT {
                    values[player][combo] += child[player][combo] / 44.0;
                }
            }
        }
        values
    }

    fn profile_walk(
        &self,
        state: GameState,
        reaches: [Vec<f64>; 2],
        river: Option<u8>,
        best_responder: Option<usize>,
        best_response_street: Option<Street>,
        average_strategy: bool,
    ) -> [Vec<f64>; 2] {
        if state.terminal.is_some() {
            return self.terminal_values(&state, &reaches, river);
        }
        if state.street == Street::River && river.is_none() {
            let mut values = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
            for card in &self.river_cards {
                let mut masked = reaches.clone();
                for player in 0..2 {
                    for combo in &self.river_blocked_combos[*card as usize] {
                        masked[player][*combo] = 0.0;
                    }
                }
                let child = self.profile_walk(
                    state.clone(),
                    masked,
                    Some(*card),
                    best_responder,
                    best_response_street,
                    average_strategy,
                );
                self.accumulate_compatible_river_child(&mut values, &child, *card);
            }
            return values;
        }
        let actions = state.legal_actions(&self.config.game);
        let actor = state.actor;
        let key = Self::node_key(&state, river);
        let legal = self.legal_for(river, actor);
        let node = self.nodes.get(&key).expect("trained turn-river node");
        let strategy = if average_strategy {
            node.average_strategy(legal)
        } else {
            node.strategy(legal)
        };
        let action_count = actions.len();
        let mut children = Vec::with_capacity(action_count);
        for (action_index, action) in actions.iter().enumerate() {
            let mut child_reaches = reaches.clone();
            let actor_takes_best_response = best_responder == Some(actor)
                && best_response_street.map_or(true, |street| state.street == street);
            if !actor_takes_best_response {
                for combo in 0..COMBO_COUNT {
                    child_reaches[actor][combo] *= strategy[combo * action_count + action_index];
                }
            }
            children.push(self.profile_walk(
                state.apply(action, &self.config.game),
                child_reaches,
                river,
                best_responder,
                best_response_street,
                average_strategy,
            ));
        }
        let opponent = 1 - actor;
        let mut values = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        for combo in 0..COMBO_COUNT {
            let actor_takes_best_response = best_responder == Some(actor)
                && best_response_street.map_or(true, |street| state.street == street);
            if actor_takes_best_response {
                values[actor][combo] = children
                    .iter()
                    .map(|child| child[actor][combo])
                    .fold(f64::NEG_INFINITY, f64::max);
                for child in &children {
                    values[opponent][combo] += child[opponent][combo];
                }
            } else {
                for action in 0..action_count {
                    values[actor][combo] +=
                        strategy[combo * action_count + action] * children[action][actor][combo];
                    values[opponent][combo] += children[action][opponent][combo];
                }
            }
        }
        values
    }

    fn collect_average_profile_diagnostics(
        &self,
        state: GameState,
        reaches: [Vec<f64>; 2],
        river: Option<u8>,
        reach_output: &mut BTreeMap<Vec<String>, [Vec<f64>; 2]>,
        action_value_output: &mut BTreeMap<Vec<String>, Vec<Vec<f64>>>,
    ) -> [Vec<f64>; 2] {
        if state.terminal.is_some() {
            return self.terminal_values(&state, &reaches, river);
        }
        if state.street == Street::River && river.is_none() {
            let mut values = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
            for card in &self.river_cards {
                let mut masked = reaches.clone();
                for player in 0..2 {
                    for combo in &self.river_blocked_combos[*card as usize] {
                        masked[player][*combo] = 0.0;
                    }
                }
                let child = self.collect_average_profile_diagnostics(
                    state.clone(),
                    masked,
                    Some(*card),
                    reach_output,
                    action_value_output,
                );
                self.accumulate_compatible_river_child(&mut values, &child, *card);
            }
            return values;
        }
        let actions = state.legal_actions(&self.config.game);
        let actor = state.actor;
        let key = Self::node_key(&state, river);
        let legal = self.legal_for(river, actor);
        let node = self.nodes.get(&key).expect("trained turn-river node");
        let strategy = node.average_strategy(legal);
        reach_output.insert(key.clone(), reaches.clone());
        let mut children = Vec::with_capacity(actions.len());
        for (action_index, action) in actions.iter().enumerate() {
            let mut child_reaches = reaches.clone();
            for combo in 0..COMBO_COUNT {
                child_reaches[actor][combo] *= strategy[combo * actions.len() + action_index];
            }
            children.push(self.collect_average_profile_diagnostics(
                state.apply(action, &self.config.game),
                child_reaches,
                river,
                reach_output,
                action_value_output,
            ));
        }
        action_value_output.insert(
            key,
            children.iter().map(|child| child[actor].clone()).collect(),
        );
        let opponent = 1 - actor;
        let mut values = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        for combo in 0..COMBO_COUNT {
            for action in 0..actions.len() {
                values[actor][combo] +=
                    strategy[combo * actions.len() + action] * children[action][actor][combo];
                values[opponent][combo] += children[action][opponent][combo];
            }
        }
        values
    }

    fn finish_continuation_values(self) -> TurnRiverContinuationValues {
        let root = self.config.state.game_state();
        let reaches = self.config.state.ranges.clone();
        let joint_mass = joint_compatibility_mass(&reaches);
        let profile = self.profile_walk(root.clone(), reaches.clone(), None, None, None, true);
        let br0 = self.profile_walk(root.clone(), reaches.clone(), None, Some(0), None, true);
        let br1 = self.profile_walk(root.clone(), reaches.clone(), None, Some(1), None, true);
        let turn_br0 = self.profile_walk(
            root.clone(),
            reaches.clone(),
            None,
            Some(0),
            Some(Street::Turn),
            true,
        );
        let turn_br1 = self.profile_walk(
            root.clone(),
            reaches.clone(),
            None,
            Some(1),
            Some(Street::Turn),
            true,
        );
        let river_br0 = self.profile_walk(
            root.clone(),
            reaches.clone(),
            None,
            Some(0),
            Some(Street::River),
            true,
        );
        let river_br1 = self.profile_walk(
            root.clone(),
            reaches.clone(),
            None,
            Some(1),
            Some(Street::River),
            true,
        );
        let current_profile =
            self.profile_walk(root.clone(), reaches.clone(), None, None, None, false);
        let current_br0 =
            self.profile_walk(root.clone(), reaches.clone(), None, Some(0), None, false);
        let current_br1 = self.profile_walk(root, reaches.clone(), None, Some(1), None, false);
        let aggregate = |values: &[f64], player: usize| {
            reaches[player]
                .iter()
                .zip(values)
                .map(|(reach, value)| reach * value)
                .sum::<f64>()
                / joint_mass
        };
        let profile_p0 = aggregate(&profile[0], 0);
        let profile_p1 = aggregate(&profile[1], 1);
        let best_p0 = aggregate(&br0[0], 0);
        let best_p1 = aggregate(&br1[1], 1);
        let turn_best_p0 = aggregate(&turn_br0[0], 0);
        let turn_best_p1 = aggregate(&turn_br1[1], 1);
        let river_best_p0 = aggregate(&river_br0[0], 0);
        let river_best_p1 = aggregate(&river_br1[1], 1);
        let current_profile_p0 = aggregate(&current_profile[0], 0);
        let current_profile_p1 = aggregate(&current_profile[1], 1);
        let current_best_p0 = aggregate(&current_br0[0], 0);
        let current_best_p1 = aggregate(&current_br1[1], 1);
        let compatible_mass: [Vec<f32>; 2] = std::array::from_fn(|player| {
            compatible_masses_from_card_marginals(&self.combos, &reaches[1 - player])
                .into_iter()
                .map(|mass| mass as f32)
                .collect()
        });
        let counterfactual_values_bb = std::array::from_fn(|player| {
            profile[player]
                .iter()
                .zip(&compatible_mass[player])
                .map(|(value, mass)| {
                    if *mass > 0.0 {
                        (*value / *mass as f64) as f32
                    } else {
                        0.0
                    }
                })
                .collect()
        });
        let mut turn_information_sets = 0usize;
        let mut river_information_sets = 0usize;
        let mut maximum_probability_sum_error = 0.0f64;
        for (history, node) in &self.nodes {
            let river = Self::river_from_key(history);
            if river.is_some() {
                river_information_sets += 1;
            } else {
                turn_information_sets += 1;
            }
            let legal = self.legal_for(river, node.actor);
            let probabilities = node.average_strategy(legal);
            let action_count = node.action_labels.len();
            for combo in 0..COMBO_COUNT {
                if legal[combo] {
                    let sum = probabilities[combo * action_count..(combo + 1) * action_count]
                        .iter()
                        .sum::<f64>();
                    maximum_probability_sum_error =
                        maximum_probability_sum_error.max((sum - 1.0).abs());
                }
            }
        }
        let mut method = "value_only_paired_alternating_vectorized_dcfr_exact_private_cards_observed_river_chance_and_complete_turn_river_betting".to_owned();
        if self.config.regret_matching_plus {
            method.push_str("_regret_matching_plus");
        }
        if self.config.river_refinement_iterations > 0 {
            method.push_str("_frozen_average_turn_river_refinement");
        }
        TurnRiverContinuationValues {
            schema: "hu-turn-river-public-belief-continuation-values-v2".to_owned(),
            method,
            joint_iterations: self.config.iterations,
            river_refinement_iterations: self.config.river_refinement_iterations,
            counterfactual_values_bb,
            opponent_compatible_mass: compatible_mass,
            metrics: TurnRiverSolveMetrics {
                information_sets: self.nodes.len(),
                turn_information_sets,
                river_information_sets,
                exact_river_cards: self.river_cards.len(),
                profile_value_p0_bb: profile_p0,
                profile_value_p1_bb: profile_p1,
                best_response_value_p0_bb: best_p0,
                best_response_value_p1_bb: best_p1,
                exact_abstract_exploitability_bb_per_hand: (((best_p0 - profile_p0)
                    + (best_p1 - profile_p1))
                    / 2.0)
                    .max(0.0),
                turn_only_best_response_gain_bb_per_hand: (((turn_best_p0 - profile_p0)
                    + (turn_best_p1 - profile_p1))
                    / 2.0)
                    .max(0.0),
                river_only_best_response_gain_bb_per_hand: (((river_best_p0 - profile_p0)
                    + (river_best_p1 - profile_p1))
                    / 2.0)
                    .max(0.0),
                current_strategy_exploitability_bb_per_hand: (((current_best_p0
                    - current_profile_p0)
                    + (current_best_p1 - current_profile_p1))
                    / 2.0)
                    .max(0.0),
                zero_sum_residual_bb: (profile_p0 + profile_p1).abs(),
                maximum_probability_sum_error,
            },
        }
    }

    fn finish(self) -> TurnRiverSolution {
        let root = self.config.state.game_state();
        let reaches = self.config.state.ranges.clone();
        let joint_mass = joint_compatibility_mass(&reaches);
        let profile = self.profile_walk(root.clone(), reaches.clone(), None, None, None, true);
        let br0 = self.profile_walk(root.clone(), reaches.clone(), None, Some(0), None, true);
        let br1 = self.profile_walk(root.clone(), reaches.clone(), None, Some(1), None, true);
        let turn_br0 = self.profile_walk(
            root.clone(),
            reaches.clone(),
            None,
            Some(0),
            Some(Street::Turn),
            true,
        );
        let turn_br1 = self.profile_walk(
            root.clone(),
            reaches.clone(),
            None,
            Some(1),
            Some(Street::Turn),
            true,
        );
        let river_br0 = self.profile_walk(
            root.clone(),
            reaches.clone(),
            None,
            Some(0),
            Some(Street::River),
            true,
        );
        let river_br1 = self.profile_walk(
            root.clone(),
            reaches.clone(),
            None,
            Some(1),
            Some(Street::River),
            true,
        );
        let current_profile =
            self.profile_walk(root.clone(), reaches.clone(), None, None, None, false);
        let current_br0 =
            self.profile_walk(root.clone(), reaches.clone(), None, Some(0), None, false);
        let current_br1 = self.profile_walk(root, reaches.clone(), None, Some(1), None, false);
        let aggregate = |values: &[f64], player: usize| {
            reaches[player]
                .iter()
                .zip(values)
                .map(|(reach, value)| reach * value)
                .sum::<f64>()
                / joint_mass
        };
        let profile_p0 = aggregate(&profile[0], 0);
        let profile_p1 = aggregate(&profile[1], 1);
        let best_p0 = aggregate(&br0[0], 0);
        let best_p1 = aggregate(&br1[1], 1);
        let turn_best_p0 = aggregate(&turn_br0[0], 0);
        let turn_best_p1 = aggregate(&turn_br1[1], 1);
        let river_best_p0 = aggregate(&river_br0[0], 0);
        let river_best_p1 = aggregate(&river_br1[1], 1);
        let current_profile_p0 = aggregate(&current_profile[0], 0);
        let current_profile_p1 = aggregate(&current_profile[1], 1);
        let current_best_p0 = aggregate(&current_br0[0], 0);
        let current_best_p1 = aggregate(&current_br1[1], 1);
        let exploitability = ((best_p0 - profile_p0) + (best_p1 - profile_p1)) / 2.0;
        let compatible_mass: [Vec<f32>; 2] = std::array::from_fn(|player| {
            compatible_masses_from_card_marginals(&self.combos, &reaches[1 - player])
                .into_iter()
                .map(|mass| mass as f32)
                .collect()
        });
        let counterfactual_values_bb = std::array::from_fn(|player| {
            profile[player]
                .iter()
                .zip(&compatible_mass[player])
                .map(|(value, mass)| {
                    if *mass > 0.0 {
                        (*value / *mass as f64) as f32
                    } else {
                        0.0
                    }
                })
                .collect()
        });
        let mut diagnostic_reaches = BTreeMap::new();
        let mut diagnostic_action_values = BTreeMap::new();
        self.collect_average_profile_diagnostics(
            self.config.state.game_state(),
            reaches.clone(),
            None,
            &mut diagnostic_reaches,
            &mut diagnostic_action_values,
        );
        let conflicts = combo_conflicts();
        let mut maximum_probability_sum_error = 0.0f64;
        let mut turn_information_sets = 0usize;
        let mut river_information_sets = 0usize;
        let strategies = self
            .nodes
            .iter()
            .map(|(history, node)| {
                let river = Self::river_from_key(history);
                if river.is_some() {
                    river_information_sets += 1;
                } else {
                    turn_information_sets += 1;
                }
                let legal = self.legal_for(river, node.actor);
                let probabilities = node.average_strategy(legal);
                let action_count = node.action_labels.len();
                for combo in 0..COMBO_COUNT {
                    if legal[combo] {
                        let sum = probabilities[combo * action_count..(combo + 1) * action_count]
                            .iter()
                            .sum::<f64>();
                        maximum_probability_sum_error =
                            maximum_probability_sum_error.max((sum - 1.0).abs());
                    }
                }
                PublicBeliefStrategy {
                    public_history: history.clone(),
                    actor: node.actor,
                    action_labels: node.action_labels.clone(),
                    probabilities: probabilities
                        .into_iter()
                        .map(|value| value as f32)
                        .collect(),
                    action_values_bb: Some(
                        normalized_action_values_bb(
                            node.actor,
                            action_count,
                            diagnostic_reaches
                                .get(history)
                                .expect("turn-river action-value pass contains every node"),
                            diagnostic_action_values
                                .get(history)
                                .expect("turn-river action-value pass contains every action"),
                            &conflicts,
                        )
                        .expect("trained turn-river action values are finite and compatible"),
                    ),
                }
            })
            .collect::<Vec<_>>();
        let zero_sum_residual = (profile_p0 + profile_p1).abs();
        let metrics = TurnRiverSolveMetrics {
            information_sets: strategies.len(),
            turn_information_sets,
            river_information_sets,
            exact_river_cards: self.river_cards.len(),
            profile_value_p0_bb: profile_p0,
            profile_value_p1_bb: profile_p1,
            best_response_value_p0_bb: best_p0,
            best_response_value_p1_bb: best_p1,
            exact_abstract_exploitability_bb_per_hand: exploitability.max(0.0),
            turn_only_best_response_gain_bb_per_hand: (((turn_best_p0 - profile_p0)
                + (turn_best_p1 - profile_p1))
                / 2.0)
                .max(0.0),
            river_only_best_response_gain_bb_per_hand: (((river_best_p0 - profile_p0)
                + (river_best_p1 - profile_p1))
                / 2.0)
                .max(0.0),
            current_strategy_exploitability_bb_per_hand: (((current_best_p0 - current_profile_p0)
                + (current_best_p1 - current_profile_p1))
                / 2.0)
                .max(0.0),
            zero_sum_residual_bb: zero_sum_residual,
            maximum_probability_sum_error,
        };
        let mut reasons = Vec::new();
        if metrics.zero_sum_residual_bb > 1e-8 {
            reasons.push(format!(
                "zero-sum residual {:.3e} exceeds 1e-8",
                metrics.zero_sum_residual_bb
            ));
        }
        if metrics.maximum_probability_sum_error > 1e-6 {
            reasons.push(format!(
                "probability sum error {:.3e} exceeds 1e-6",
                metrics.maximum_probability_sum_error
            ));
        }
        if metrics.exact_abstract_exploitability_bb_per_hand > 0.05 {
            reasons.push(format!(
                "turn-river abstraction exploitability {:.6}bb/hand exceeds 0.05bb/hand",
                metrics.exact_abstract_exploitability_bb_per_hand
            ));
        }
        let mut method = "paired_alternating_vectorized_dcfr_exact_private_cards_observed_river_chance_and_complete_turn_river_betting".to_owned();
        if self.config.regret_matching_plus {
            method.push_str("_regret_matching_plus");
        }
        if self.config.river_refinement_iterations > 0 {
            method.push_str("_frozen_average_turn_river_refinement");
        }
        TurnRiverSolution {
            schema: "hu-turn-river-public-belief-solution-v2".to_owned(),
            method,
            approximate: true,
            game: self.config.game,
            state: self.config.state,
            iterations: self.config.iterations,
            averaging_delay: self.config.averaging_delay,
            river_refinement_iterations: self.config.river_refinement_iterations,
            counterfactual_values_bb,
            opponent_compatible_mass: compatible_mass,
            strategies,
            validation: BlueprintValidation {
                status: if reasons.is_empty() {
                    "accepted"
                } else {
                    "rejected"
                }
                .to_owned(),
                reasons,
            },
            metrics,
        }
    }
}

pub fn solve_turn_river(config: TurnRiverSolveConfig) -> Result<TurnRiverSolution, String> {
    let mut solver = TurnRiverSolver::new(config)?;
    solver.train();
    Ok(solver.finish())
}

fn deal_with_visible_board_and_private_combo(board: &[u8], player: usize, combo: Combo) -> Deal {
    debug_assert!(board.len() <= 5);
    let private = combo.cards();
    let available = (0..52u8)
        .filter(|card| !board.contains(card) && !private.contains(card))
        .take(7 - board.len())
        .collect::<Vec<_>>();
    debug_assert_eq!(available.len(), 7 - board.len());
    let mut holes = [[0u8; 2]; 2];
    holes[player] = private;
    holes[1 - player] = [available[0], available[1]];
    let mut full_board = [0u8; 5];
    full_board[..board.len()].copy_from_slice(board);
    full_board[board.len()..].copy_from_slice(&available[2..]);
    Deal::from_sampled_cards(holes, full_board)
}

struct BoundedActionRecordCollector {
    capacities: [usize; 3],
    seed: u64,
    seen: usize,
    seen_by_street: [usize; 3],
    records: [BTreeMap<(u64, u64), Vec<u8>>; 3],
}

impl BoundedActionRecordCollector {
    fn new(capacity: usize, seed: u64) -> Self {
        let flop = (capacity / 4).max(1);
        let turn = (capacity * 3 / 10).max(1);
        Self {
            capacities: [flop, turn, capacity - flop - turn],
            seed,
            seen: 0,
            seen_by_street: [0; 3],
            records: std::array::from_fn(|_| BTreeMap::new()),
        }
    }

    fn street_index(street: Street) -> usize {
        match street {
            Street::Flop => 0,
            Street::Turn => 1,
            Street::River => 2,
            Street::Preflop => unreachable!("postflop action collector"),
        }
    }

    fn consider(&mut self, street: Street, record: Vec<u8>) {
        let street_index = Self::street_index(street);
        let mut digest = Sha256::new();
        digest.update(b"hu-postflop-action-record-reservoir-v1");
        digest.update(self.seed.to_le_bytes());
        digest.update(self.seen.to_le_bytes());
        digest.update([street_index as u8]);
        digest.update(&record);
        let bytes = digest.finalize();
        let key = (
            u64::from_le_bytes(bytes[..8].try_into().expect("SHA-256 prefix")),
            u64::from_le_bytes(bytes[8..16].try_into().expect("SHA-256 suffix")),
        );
        self.seen += 1;
        self.seen_by_street[street_index] += 1;
        let records = &mut self.records[street_index];
        if records.len() < self.capacities[street_index] {
            records.insert(key, record);
            return;
        }
        let largest = *records
            .last_key_value()
            .expect("positive action-record capacity")
            .0;
        if key < largest {
            records.pop_last();
            records.insert(key, record);
        }
    }

    fn len(&self) -> usize {
        self.records.iter().map(BTreeMap::len).sum()
    }

    fn into_records(self) -> Vec<Vec<u8>> {
        let mut output = Vec::with_capacity(self.len());
        for (street, records) in self.records.into_iter().enumerate() {
            let inclusion_correction =
                self.seen_by_street[street] as f64 / records.len().max(1) as f64;
            for record in records.into_values() {
                let mut value: serde_json::Value = serde_json::from_slice(&record)
                    .expect("generated action record remains valid JSON");
                let weight = value["weight"]
                    .as_f64()
                    .expect("generated action record has a numeric weight");
                value["weight"] = serde_json::json!(weight * inclusion_correction);
                output.push(
                    serde_json::to_vec(&value)
                        .expect("importance-corrected action record remains serializable"),
                );
            }
        }
        output
    }
}

fn append_public_belief_policy_records(
    game: &BlueprintConfig,
    source_policy: Option<&FrozenPolicy>,
    root_state: GameState,
    root_board: &[u8],
    root_ranges: [Vec<f64>; 2],
    strategies: &[PublicBeliefStrategy],
    root_weight: f64,
    records: &mut BoundedActionRecordCollector,
    mut range_records: Option<&mut BoundedActionRecordCollector>,
    emit_legacy_records: bool,
) -> Result<usize, String> {
    let strategy_map = strategies
        .iter()
        .map(|strategy| (strategy.public_history.clone(), strategy))
        .collect::<BTreeMap<_, _>>();
    let conflicts = combo_conflicts();
    let before = records.seen;
    let root_joint_mass = joint_compatibility_mass(&root_ranges);
    if root_joint_mass <= EPSILON {
        return Err("solver policy export root ranges have no compatible mass".to_owned());
    }

    struct RecordWalk<'a> {
        game: &'a BlueprintConfig,
        source_policy: Option<&'a FrozenPolicy>,
        strategies: &'a BTreeMap<Vec<String>, &'a PublicBeliefStrategy>,
        conflicts: &'a [Vec<usize>],
        root_weight: f64,
        root_joint_mass: f64,
        records: &'a mut BoundedActionRecordCollector,
        range_records: Option<&'a mut BoundedActionRecordCollector>,
        emit_legacy_records: bool,
    }

    impl RecordWalk<'_> {
        fn walk(
            &mut self,
            state: GameState,
            board: Vec<u8>,
            reaches: [Vec<f64>; 2],
            chance_weight: f64,
            river: Option<u8>,
        ) -> Result<(), String> {
            if state.terminal.is_some() {
                return Ok(());
            }
            let mut key = state.public_history.clone();
            if let Some(card) = river {
                key.push(format!("chance:river:{card}"));
            }
            if let Some(strategy) = self.strategies.get(&key).copied() {
                return self.emit_and_descend(
                    state,
                    board,
                    reaches,
                    chance_weight,
                    river,
                    strategy,
                );
            }
            // Complete turn-river teachers append the observed river card to
            // the information-set key instead of mutating the betting state.
            if state.street == Street::River && board.len() == 4 {
                for river in 0..52u8 {
                    if board.contains(&river) {
                        continue;
                    }
                    let mut key = state.public_history.clone();
                    key.push(format!("chance:river:{river}"));
                    let Some(strategy) = self.strategies.get(&key).copied() else {
                        continue;
                    };
                    let mut masked = reaches.clone();
                    for player in 0..2 {
                        for combo in all_combos() {
                            if combo.cards().contains(&river) {
                                masked[player][combo.key()] = 0.0;
                            }
                        }
                    }
                    let mut river_board = board.clone();
                    river_board.push(river);
                    self.emit_and_descend(
                        state.clone(),
                        river_board,
                        masked,
                        chance_weight / 44.0,
                        Some(river),
                        strategy,
                    )?;
                }
            }
            Ok(())
        }

        fn emit_and_descend(
            &mut self,
            state: GameState,
            board: Vec<u8>,
            reaches: [Vec<f64>; 2],
            chance_weight: f64,
            river: Option<u8>,
            strategy: &PublicBeliefStrategy,
        ) -> Result<(), String> {
            let actions = state.legal_actions(self.game);
            if strategy.actor != state.actor
                || strategy.action_labels
                    != actions
                        .iter()
                        .map(|action| action.label.clone())
                        .collect::<Vec<_>>()
                || strategy.probabilities.len() != COMBO_COUNT * actions.len()
            {
                return Err(
                    "solver strategy does not match the exact legal action state".to_owned(),
                );
            }
            let actor = state.actor;
            let action_count = actions.len();
            let action_values = strategy.action_values_bb.as_ref().ok_or_else(|| {
                "solver strategy is missing counterfactual action values".to_owned()
            })?;
            if action_values.len() != COMBO_COUNT * action_count {
                return Err(
                    "solver counterfactual action values have incompatible dimensions".to_owned(),
                );
            }
            if let Some(range_records) = self.range_records.as_deref_mut() {
                let node_joint_mass = joint_compatibility_mass(&reaches);
                let node_weight = self.root_weight * chance_weight * node_joint_mass
                    / self.root_joint_mass.max(EPSILON);
                if node_weight.is_finite() && node_weight > EPSILON {
                    let normalized_ranges = normalize_policy_ranges(&reaches, &board)?;
                    let source_policy_probabilities = self
                        .source_policy
                        .map(|policy| {
                            policy.bundle_strategy_matrix(&state, &board, &actions, self.game)
                        })
                        .transpose()?
                        .unwrap_or_default()
                        .into_iter()
                        .map(|probability| probability as f32)
                        .collect();
                    let record = RangeConditionedPolicyRecord {
                        record_type: "range_conditioned_average_strategy".to_owned(),
                        weight: node_weight as f32,
                        state: RangeConditionedPolicyState {
                            board: board.clone(),
                            street: state.street,
                            actor: state.actor,
                            invested_bb: state.invested,
                            street_invested_bb: state.street_invested,
                            last_full_raise_bb: state.last_full_raise,
                            aggressions: state.aggressions,
                            checks: state.checks,
                            raise_reopened: state.raise_reopened,
                            public_history: state.public_history.clone(),
                            trajectory: state.trajectory.clone(),
                        },
                        ranges: normalized_ranges,
                        action_labels: strategy.action_labels.clone(),
                        action_features: actions
                            .iter()
                            .map(|action| {
                                super::neural::encode_action_features(&state, action, self.game)
                            })
                            .collect(),
                        probabilities: strategy.probabilities.clone(),
                        source_policy_probabilities,
                        action_values_bb: action_values.clone(),
                    };
                    range_records.consider(
                        state.street,
                        serde_json::to_vec(&record)
                            .expect("range-conditioned policy record is serializable"),
                    );
                }
            }
            for combo in all_combos() {
                let key = combo.key();
                if combo.cards().iter().any(|card| board.contains(card)) {
                    continue;
                }
                let offset = key * action_count;
                let targets = strategy.probabilities[offset..offset + action_count]
                    .iter()
                    .map(|value| f64::from(*value))
                    .collect::<Vec<_>>();
                let probability_sum = targets.iter().sum::<f64>();
                if probability_sum <= EPSILON {
                    continue;
                }
                if (probability_sum - 1.0).abs() > 1e-5 {
                    return Err("solver action probabilities do not sum to one".to_owned());
                }
                let action_values_bb = action_values[offset..offset + action_count]
                    .iter()
                    .map(|value| f64::from(*value))
                    .collect::<Vec<_>>();
                if action_values_bb.iter().any(|value| !value.is_finite()) {
                    return Err("solver counterfactual action values are non-finite".to_owned());
                }
                if !self.emit_legacy_records {
                    continue;
                }
                let opponent_mass =
                    compatible_mass_from_conflicts(&reaches[1 - actor], self.conflicts, key);
                let weight = self.root_weight * chance_weight * reaches[actor][key] * opponent_mass;
                if !weight.is_finite() || weight <= EPSILON {
                    continue;
                }
                let deal = deal_with_visible_board_and_private_combo(&board, actor, combo);
                self.records.consider(
                    state.street,
                    super::neural::average_strategy_record_bytes(
                        &state,
                        &deal,
                        &actions,
                        targets,
                        action_values_bb,
                        weight as f32,
                        self.game,
                    ),
                );
            }
            for (action_index, action) in actions.iter().enumerate() {
                let mut child_reaches = reaches.clone();
                for combo in 0..COMBO_COUNT {
                    child_reaches[actor][combo] *=
                        f64::from(strategy.probabilities[combo * action_count + action_index]);
                }
                self.walk(
                    state.apply(action, self.game),
                    board.clone(),
                    child_reaches,
                    chance_weight,
                    river,
                )?;
            }
            Ok(())
        }
    }

    RecordWalk {
        game,
        source_policy,
        strategies: &strategy_map,
        conflicts: &conflicts,
        root_weight,
        root_joint_mass,
        records,
        range_records: range_records.take(),
        emit_legacy_records,
    }
    .walk(root_state, root_board.to_vec(), root_ranges, 1.0, None)?;
    Ok(records.seen - before)
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct RangeConditionedPolicyState {
    board: Vec<u8>,
    street: Street,
    actor: usize,
    invested_bb: [f64; 2],
    street_invested_bb: [f64; 2],
    last_full_raise_bb: f64,
    aggressions: u8,
    checks: u8,
    raise_reopened: bool,
    public_history: Vec<String>,
    trajectory: Vec<TrajectoryAction>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct RangeConditionedPolicyRecord {
    record_type: String,
    weight: f32,
    state: RangeConditionedPolicyState,
    ranges: [Vec<f32>; 2],
    action_labels: Vec<String>,
    action_features: Vec<Vec<f32>>,
    /// Row-major `[combo][action]` frozen average policy.
    probabilities: Vec<f32>,
    /// Row-major `[combo][action]` probabilities from the pinned source bundle.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    source_policy_probabilities: Vec<f32>,
    /// Row-major `[combo][action]` counterfactual action EVs in big blinds.
    action_values_bb: Vec<f32>,
}

fn normalize_policy_ranges(reaches: &[Vec<f64>; 2], board: &[u8]) -> Result<[Vec<f32>; 2], String> {
    let board = board.iter().copied().collect::<BTreeSet<_>>();
    let mut normalized: [Vec<f32>; 2] = std::array::from_fn(|_| vec![0.0; COMBO_COUNT]);
    for player in 0..2 {
        if reaches[player].len() != COMBO_COUNT {
            return Err("policy export range has an incompatible size".to_owned());
        }
        let total = all_combos()
            .iter()
            .filter(|combo| !combo.cards().iter().any(|card| board.contains(card)))
            .map(|combo| reaches[player][combo.key()])
            .sum::<f64>();
        if !total.is_finite() || total <= EPSILON {
            return Err("policy export range has no legal conditional mass".to_owned());
        }
        for combo in all_combos() {
            if !combo.cards().iter().any(|card| board.contains(card)) {
                let value = reaches[player][combo.key()] / total;
                if !value.is_finite() || value < 0.0 {
                    return Err("policy export range is non-finite or negative".to_owned());
                }
                normalized[player][combo.key()] = value as f32;
            }
        }
    }
    Ok(normalized)
}

fn write_range_conditioned_policy_dataset(
    game: &BlueprintConfig,
    seed: u64,
    teacher: &serde_json::Value,
    records: &[Vec<u8>],
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    if records.is_empty() {
        return Err("range-conditioned policy dataset has no records".into());
    }
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let temporary = output.with_extension("tmp");
    let file = fs::File::create(&temporary)?;
    let mut writer = GzEncoder::new(BufWriter::new(file), Compression::fast());
    serde_json::to_writer(
        &mut writer,
        &serde_json::json!({
            "record_type": "metadata",
            "schema": "hu-range-conditioned-postflop-policy-dataset-v1",
            "feature_schema": RANGE_POLICY_FEATURE_SCHEMA_V1,
            "context_size": SHARED_CONTEXT_BOARD_RELATIVE_COUNT,
            "query_size": SHARED_QUERY_BOARD_RELATIVE_COUNT,
            "action_feature_schema": "hu-cash-legal-action-v1",
            "action_feature_count": 9,
            "depth_bb": game.effective_stack_bb,
            "seed": seed,
            "records": records.len(),
            "sampling_mode": "authentic_public_reach_weighted_solver_nodes",
            "source_policy_baseline": {
                "composition": RANGE_POLICY_SOURCE_LOGIT_RESIDUAL,
                "sha256": teacher["sourcePolicySha256"],
            },
            "teacher": teacher,
        }),
    )?;
    writer.write_all(b"\n")?;
    for record in records {
        writer.write_all(record)?;
        writer.write_all(b"\n")?;
    }
    writer.finish()?.flush()?;
    fs::rename(temporary, output)?;
    Ok(())
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RangePolicyBaselineReport {
    pub schema: String,
    pub input_sha256: String,
    pub output_sha256: String,
    pub source_policy_sha256: String,
    pub records: usize,
    pub maximum_probability_sum_error: f64,
    pub output: PathBuf,
}

/// Attach exact serving-inference probabilities from a pinned source bundle to
/// every solver target row. A residual student can then start at the already
/// validated policy and learn only bounded postflop corrections.
pub fn attach_source_policy_baseline(
    source_policy_path: &Path,
    input: &Path,
    output: &Path,
) -> Result<RangePolicyBaselineReport, Box<dyn Error>> {
    if input == output {
        return Err("range-policy baseline augmentation requires a distinct output path".into());
    }
    let input_bytes = fs::read(input)?;
    let input_sha256 = format!("{:x}", Sha256::digest(&input_bytes));
    let policy = FrozenPolicy::load(source_policy_path)?;
    let source_policy_sha256 = policy.bundle_sha256().to_owned();
    let decoder = GzDecoder::new(input_bytes.as_slice());
    let mut lines = BufReader::new(decoder).lines();
    let mut metadata: serde_json::Value =
        serde_json::from_str(&lines.next().ok_or("range policy dataset is empty")??)?;
    if metadata["record_type"] != "metadata"
        || metadata["schema"] != "hu-range-conditioned-postflop-policy-dataset-v1"
        || metadata["teacher"]["validation"]["status"] != "accepted_for_training"
    {
        return Err("range policy dataset metadata is incompatible or unvalidated".into());
    }
    let depth = metadata["depth_bb"]
        .as_f64()
        .ok_or("range policy dataset omits its depth")?;
    let mut game = BlueprintConfig::default();
    game.effective_stack_bb = depth;
    if let Ok(action_abstraction) =
        serde_json::from_value(metadata["teacher"]["actionAbstraction"].clone())
    {
        game.action_abstraction = action_abstraction;
    }
    game.validate()?;
    let mut encoded_records = Vec::new();
    let mut maximum_sum_error = 0.0f64;
    for line in lines {
        let mut record: RangeConditionedPolicyRecord = serde_json::from_str(&line?)?;
        let state = PublicBeliefState {
            street: record.state.street,
            board: record.state.board.clone(),
            actor: record.state.actor,
            invested_bb: record.state.invested_bb,
            street_invested_bb: record.state.street_invested_bb,
            last_full_raise_bb: record.state.last_full_raise_bb,
            aggressions: record.state.aggressions,
            checks: record.state.checks,
            raise_reopened: record.state.raise_reopened,
            public_history: record.state.public_history.clone(),
            ranges: std::array::from_fn(|player| {
                record.ranges[player]
                    .iter()
                    .map(|value| f64::from(*value))
                    .collect()
            }),
            trajectory: record.state.trajectory.clone(),
        }
        .validate_street_and_normalize(
            &game,
            record.state.street,
            record.state.board.len(),
        )?;
        let game_state = state.game_state();
        let actions = game_state.legal_actions(&game);
        if record.record_type != "range_conditioned_average_strategy"
            || record.ranges.iter().any(|range| range.len() != COMBO_COUNT)
            || record.probabilities.len() != COMBO_COUNT * actions.len()
            || record.action_values_bb.len() != COMBO_COUNT * actions.len()
            || record.action_features.len() != actions.len()
            || record.action_labels
                != actions
                    .iter()
                    .map(|action| action.label.clone())
                    .collect::<Vec<_>>()
        {
            return Err("range policy record action labels do not match the source game".into());
        }
        let baseline =
            policy.bundle_strategy_matrix(&game_state, &record.state.board, &actions, &game)?;
        let board = record.state.board.iter().copied().collect::<BTreeSet<_>>();
        for combo in all_combos() {
            if combo.cards().iter().any(|card| board.contains(card)) {
                continue;
            }
            let offset = combo.key() * actions.len();
            let sum = baseline[offset..offset + actions.len()].iter().sum::<f64>();
            maximum_sum_error = maximum_sum_error.max((sum - 1.0).abs());
        }
        record.source_policy_probabilities = baseline
            .into_iter()
            .map(|probability| probability as f32)
            .collect();
        encoded_records.push(serde_json::to_vec(&record)?);
    }
    if encoded_records.len() != metadata["records"].as_u64().unwrap_or(0) as usize {
        return Err("range policy baseline augmentation record count changed".into());
    }
    metadata["source_policy_baseline"] = serde_json::json!({
        "composition": RANGE_POLICY_SOURCE_LOGIT_RESIDUAL,
        "sha256": source_policy_sha256,
        "inference": "exact_rust_frozen_source_bundle",
    });
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let temporary = output.with_extension("tmp");
    let file = fs::File::create(&temporary)?;
    let mut writer = GzEncoder::new(BufWriter::new(file), Compression::fast());
    serde_json::to_writer(&mut writer, &metadata)?;
    writer.write_all(b"\n")?;
    for record in &encoded_records {
        writer.write_all(record)?;
        writer.write_all(b"\n")?;
    }
    writer.finish()?.flush()?;
    fs::rename(&temporary, output)?;
    let output_sha256 = format!("{:x}", Sha256::digest(fs::read(output)?));
    Ok(RangePolicyBaselineReport {
        schema: "hu-range-policy-source-baseline-augmentation-v1".to_owned(),
        input_sha256,
        output_sha256,
        source_policy_sha256,
        records: encoded_records.len(),
        maximum_probability_sum_error: maximum_sum_error,
        output: output.to_owned(),
    })
}

#[derive(Clone, Debug)]
pub struct PostflopActionTargetConfig {
    pub game: BlueprintConfig,
    pub roots: usize,
    pub root_offset: usize,
    pub turn_leaves_per_root: usize,
    pub flop_iterations: u64,
    pub flop_iteration_checkpoints: Vec<u64>,
    pub flop_averaging_delay: u64,
    pub flop_regret_matching_plus: bool,
    pub require_accepted_flop_teachers: bool,
    pub require_range_consistent_flop_teachers: bool,
    pub flop_response_checkpoints: Vec<u64>,
    pub flop_response_averaging_delay: u64,
    pub flop_response_regret_matching_plus: bool,
    pub maximum_flop_range_response_gain_bb_per_hand: f64,
    pub require_accepted_turn_river_teachers: bool,
    pub turn_river_iterations: u64,
    pub turn_river_averaging_delay: u64,
    pub seed: u64,
    pub threads: usize,
    pub exploration_probability: f64,
    pub max_records: usize,
    pub source_policy_path: PathBuf,
    pub value_network_path: PathBuf,
    pub auxiliary_value_network_paths: Vec<PathBuf>,
    pub evaluation_value_network_path: Option<PathBuf>,
    pub output: PathBuf,
    pub range_output: Option<PathBuf>,
    pub range_only: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PostflopActionTeacherCheckpoint {
    pub root: usize,
    pub iterations: u64,
    pub depth_limited_exploitability_bb_per_hand: f64,
    pub validation_status: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PostflopActionRangeResponseCheckpoint {
    pub root: usize,
    pub response_iterations: u64,
    pub range_consistent_response_gain_bb_per_hand: f64,
    pub evaluation_has_distinct_training_identity: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PostflopActionTargetReport {
    pub schema: String,
    pub method: String,
    pub seed: u64,
    pub source_policy_sha256: String,
    pub value_network_sha256: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub auxiliary_value_network_sha256s: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluation_value_network_sha256: Option<String>,
    #[serde(default)]
    pub root_offset: usize,
    pub roots: usize,
    pub turn_leaves: usize,
    pub flop_regret_matching_plus: bool,
    pub required_accepted_flop_teachers: bool,
    pub required_range_consistent_flop_teachers: bool,
    pub required_accepted_turn_river_teachers: bool,
    pub flop_convergence: Vec<PostflopActionTeacherCheckpoint>,
    pub flop_range_response: Vec<PostflopActionRangeResponseCheckpoint>,
    pub flop_records: usize,
    pub turn_river_records: usize,
    pub total_records: usize,
    pub candidate_records: usize,
    pub truncated: bool,
    pub maximum_flop_exploitability_bb_per_hand: f64,
    pub maximum_turn_river_exploitability_bb_per_hand: f64,
    pub output: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range_output: Option<PathBuf>,
    #[serde(default)]
    pub range_conditioned_records: usize,
    pub validation: BlueprintValidation,
}

fn sample_flop_root(
    policy: &FrozenPolicy,
    game: &BlueprintConfig,
    deal: &Deal,
    rng: &mut SplitMix64,
    explorer: Option<usize>,
    exploration_probability: f64,
) -> Option<(GameState, Vec<String>)> {
    let mut state = GameState::initial(game);
    let mut line = Vec::new();
    while state.terminal.is_none() && state.street == Street::Preflop {
        let actions = state.legal_actions(game);
        let strategy = policy.strategy(&state, deal, &actions, game);
        let sampling = if explorer == Some(state.actor) {
            epsilon_sampling_strategy(&strategy, exploration_probability)
        } else {
            strategy
        };
        let selected = sample_index(&sampling, rng);
        line.push(actions[selected].label.clone());
        state = state.apply(&actions[selected], game);
    }
    (state.street == Street::Flop && state.terminal.is_none()).then_some((state, line))
}

fn selected_root_ordinal(
    root_offset: usize,
    roots: usize,
    absolute_root: usize,
) -> Result<Option<usize>, String> {
    let end = root_offset
        .checked_add(roots)
        .ok_or_else(|| "postflop action target root window overflows".to_owned())?;
    Ok(if (root_offset..end).contains(&absolute_root) {
        Some(absolute_root - root_offset)
    } else {
        None
    })
}

fn sample_reach_weighted_turn_leaves(
    leaves: &[ResolverTurnLeaf],
    count: usize,
    seed: u64,
) -> Result<Vec<ResolverTurnLeaf>, String> {
    let mut rng = SplitMix64::new(seed);
    let mut candidates = (0..leaves.len()).collect::<Vec<_>>();
    let mut selected = Vec::with_capacity(count);
    while selected.len() < count {
        let Some(index) = weighted_resolver_leaf_index(leaves, &candidates, &mut rng) else {
            return Err("resolver has too few positive-reach turn leaves".to_owned());
        };
        selected.push(leaves[index].clone());
        candidates.retain(|candidate| *candidate != index);
    }
    Ok(selected)
}

pub fn generate_postflop_action_targets(
    config: PostflopActionTargetConfig,
) -> Result<PostflopActionTargetReport, Box<dyn Error>> {
    config.game.validate()?;
    let root_window_end = config
        .root_offset
        .checked_add(config.roots)
        .ok_or("postflop action target root window overflows")?;
    if config.roots == 0
        || config.turn_leaves_per_root == 0
        || config.flop_iterations < 2
        || config.flop_iteration_checkpoints.is_empty()
        || config
            .flop_iteration_checkpoints
            .iter()
            .any(|checkpoint| *checkpoint < 2)
        || config
            .flop_iteration_checkpoints
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        || config.flop_iteration_checkpoints.last().copied() != Some(config.flop_iterations)
        || config.flop_averaging_delay >= config.flop_iteration_checkpoints[0]
        || (config.require_range_consistent_flop_teachers
            && (config.evaluation_value_network_path.is_none()
                || config.flop_response_checkpoints.is_empty()
                || config
                    .flop_response_checkpoints
                    .iter()
                    .any(|checkpoint| *checkpoint < 2)
                || config
                    .flop_response_checkpoints
                    .windows(2)
                    .any(|pair| pair[0] >= pair[1])
                || config.flop_response_averaging_delay >= config.flop_response_checkpoints[0]
                || !config
                    .maximum_flop_range_response_gain_bb_per_hand
                    .is_finite()
                || config.maximum_flop_range_response_gain_bb_per_hand < 0.0))
        || config.turn_river_iterations < 2
        || config.turn_river_averaging_delay >= config.turn_river_iterations
        || config.threads == 0
        || config.max_records < 3
        || !config.exploration_probability.is_finite()
        || !(0.0..1.0).contains(&config.exploration_probability)
        || (config.range_only && config.range_output.is_none())
    {
        return Err(
            "postflop action targets require valid roots, solve controls, and record capacity"
                .into(),
        );
    }
    let policy_bytes = fs::read(&config.source_policy_path)?;
    let source_policy_sha256 = format!("{:x}", Sha256::digest(&policy_bytes));
    let policy = FrozenPolicy::load(&config.source_policy_path)?;
    let value_bytes = fs::read(&config.value_network_path)?;
    let value_network_sha256 = format!("{:x}", Sha256::digest(&value_bytes));
    let value_network = PublicValueNetwork::read(&config.value_network_path)?;
    let auxiliary_value_networks = config
        .auxiliary_value_network_paths
        .iter()
        .map(|path| PublicValueNetwork::read(path))
        .collect::<Result<Vec<_>, _>>()?;
    let auxiliary_value_network_sha256s = config
        .auxiliary_value_network_paths
        .iter()
        .map(fs::read)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(|bytes| format!("{:x}", Sha256::digest(bytes)))
        .collect::<Vec<_>>();
    let evaluation_value_network = config
        .evaluation_value_network_path
        .as_ref()
        .map(|path| PublicValueNetwork::read(path))
        .transpose()?;
    let evaluation_value_network_sha256 = config
        .evaluation_value_network_path
        .as_ref()
        .map(fs::read)
        .transpose()?
        .map(|bytes| format!("{:x}", Sha256::digest(bytes)));
    let mut chance = SplitMix64::new(config.seed);
    let mut records = BoundedActionRecordCollector::new(config.max_records, config.seed);
    let mut range_records = config
        .range_output
        .as_ref()
        .map(|_| BoundedActionRecordCollector::new(config.max_records, config.seed ^ 0xA11C_E5));
    let mut roots_solved = 0usize;
    let mut leaves_solved = 0usize;
    let mut maximum_flop_exploitability = 0.0f64;
    let mut maximum_turn_river_exploitability = 0.0f64;
    let mut flop_convergence = Vec::new();
    let mut flop_range_response = Vec::new();
    let mut attempts = 0usize;
    let mut absolute_root = 0usize;

    while roots_solved < config.roots {
        attempts += 1;
        if attempts > root_window_end.saturating_mul(1_000) {
            return Err("could not sample enough authentic nonterminal flop roots".into());
        }
        let true_deal = Deal::sample(&mut chance);
        let explorer = (config.exploration_probability > 0.0).then_some(absolute_root % 2);
        let Some((flop_state, line)) = sample_flop_root(
            &policy,
            &config.game,
            &true_deal,
            &mut chance,
            explorer,
            config.exploration_probability,
        ) else {
            continue;
        };
        let board = [true_deal.board[0], true_deal.board[1], true_deal.board[2]];
        let ranges = match exact_reach_factors_for_visible_line(
            &policy,
            &config.game,
            &board,
            &line,
            config.threads,
        ) {
            Ok(ranges) => ranges,
            Err(error) if error.contains("zero exact reach") => continue,
            Err(error) => return Err(error.into()),
        };
        let Some(root_ordinal) =
            selected_root_ordinal(config.root_offset, config.roots, absolute_root)?
        else {
            absolute_root += 1;
            continue;
        };
        debug_assert_eq!(root_ordinal, roots_solved);
        let root_state =
            PublicBeliefState::from_game_state(board.to_vec(), &flop_state, ranges.clone());
        let mut flop_solver = FlopSolver::new(FlopResolveConfig {
            game: config.game.clone(),
            state: root_state,
            iterations: config.flop_iterations,
            averaging_delay: config.flop_averaging_delay,
            regret_matching_plus: config.flop_regret_matching_plus,
            value_network: value_network.clone(),
            auxiliary_value_networks: auxiliary_value_networks.clone(),
            threads: config.threads,
        })?;
        let solver_root = flop_solver.config.state.game_state();
        let solver_reaches = flop_solver.config.state.ranges.clone();
        let mut completed_iterations = 0u64;
        let mut final_flop_solution = None;
        for checkpoint in &config.flop_iteration_checkpoints {
            for round in (completed_iterations + 1)..=*checkpoint {
                flop_solver.walk(solver_root.clone(), solver_reaches.clone(), 0, round, false);
                flop_solver.walk(solver_root.clone(), solver_reaches.clone(), 1, round, true);
            }
            completed_iterations = *checkpoint;
            let mut evaluator = flop_solver.clone();
            evaluator.config.iterations = *checkpoint;
            let solution = evaluator.finish();
            let accepted = solution.validation.status == "accepted";
            flop_convergence.push(PostflopActionTeacherCheckpoint {
                root: absolute_root,
                iterations: *checkpoint,
                depth_limited_exploitability_bb_per_hand: solution
                    .metrics
                    .depth_limited_exploitability_bb_per_hand,
                validation_status: solution.validation.status.clone(),
            });
            eprintln!(
                "{}",
                serde_json::to_string(&serde_json::json!({
                    "event": "postflop-action-flop-checkpoint",
                    "root": absolute_root,
                    "iterations": checkpoint,
                    "depthLimitedExploitabilityBbPerHand": solution
                        .metrics
                        .depth_limited_exploitability_bb_per_hand,
                    "validationStatus": solution.validation.status.clone(),
                }))
                .expect("flop checkpoint progress event remains serializable")
            );
            final_flop_solution = Some(solution);
            if config.require_accepted_flop_teachers && accepted {
                break;
            }
        }
        let leaves = flop_solver
            .capture_average_turn_leaves()
            .into_iter()
            .filter_map(canonicalize_resolver_turn_leaf)
            .collect::<Vec<_>>();
        let selected_leaves = sample_reach_weighted_turn_leaves(
            &leaves,
            config.turn_leaves_per_root,
            config.seed ^ (absolute_root as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
        )?;
        let flop_solution = final_flop_solution.expect("nonempty flop iteration checkpoints");
        let mut range_response_usable = !config.require_range_consistent_flop_teachers;
        if let Some(evaluation) = evaluation_value_network.as_ref() {
            if !config.flop_response_checkpoints.is_empty() {
                let response = evaluate_frozen_flop_range_response_convergence(
                    config.game.clone(),
                    &flop_solution,
                    evaluation.clone(),
                    &config.flop_response_checkpoints,
                    config.flop_response_averaging_delay,
                    config.flop_response_regret_matching_plus,
                    config.threads,
                )?;
                let final_response = response
                    .checkpoints
                    .last()
                    .expect("configured range response has a checkpoint");
                range_response_usable = response.evaluation_has_distinct_training_identity
                    && final_response.maximum_zero_sum_residual_bb <= 1e-6
                    && final_response.range_consistent_response_gain_bb_per_hand
                        <= config.maximum_flop_range_response_gain_bb_per_hand;
                flop_range_response.push(PostflopActionRangeResponseCheckpoint {
                    root: absolute_root,
                    response_iterations: final_response.iterations,
                    range_consistent_response_gain_bb_per_hand: final_response
                        .range_consistent_response_gain_bb_per_hand,
                    evaluation_has_distinct_training_identity: response
                        .evaluation_has_distinct_training_identity,
                });
                eprintln!(
                    "{}",
                    serde_json::to_string(&serde_json::json!({
                        "event": "postflop-action-flop-range-response",
                        "root": absolute_root,
                        "iterations": final_response.iterations,
                        "rangeConsistentResponseGainBbPerHand": final_response
                            .range_consistent_response_gain_bb_per_hand,
                        "evaluationHasDistinctTrainingIdentity": response
                            .evaluation_has_distinct_training_identity,
                        "acceptedForTraining": range_response_usable,
                    }))
                    .expect("flop range-response progress event remains serializable")
                );
            }
        }
        let structurally_usable_flop_teacher = flop_solution
            .validation
            .reasons
            .iter()
            .all(|reason| reason.starts_with("depth-limited exploitability "))
            && flop_solution
                .metrics
                .depth_limited_exploitability_bb_per_hand
                .is_finite()
            && flop_solution
                .metrics
                .resolver_relative_exploitability_improvement
                > 0.0
            && flop_solution.metrics.zero_sum_residual_after_projection_bb <= 1e-6;
        let flop_training_usable = structurally_usable_flop_teacher
            && (!config.require_accepted_flop_teachers
                || flop_solution.validation.status == "accepted")
            && range_response_usable;
        if !flop_training_usable {
            return Err(format!(
                "flop action teacher failed the configured quality gate at root {absolute_root}: {}",
                format!(
                    "{}; convergence {:?}",
                    flop_solution.validation.reasons.join("; "),
                    flop_convergence
                        .iter()
                        .filter(|checkpoint| checkpoint.root == absolute_root)
                        .map(|checkpoint| (
                            checkpoint.iterations,
                            checkpoint.depth_limited_exploitability_bb_per_hand
                        ))
                        .collect::<Vec<_>>()
                )
            )
            .into());
        }
        maximum_flop_exploitability = maximum_flop_exploitability.max(
            flop_solution
                .metrics
                .depth_limited_exploitability_bb_per_hand,
        );
        append_public_belief_policy_records(
            &config.game,
            Some(&policy),
            flop_solution.state.game_state(),
            &flop_solution.state.board,
            flop_solution.state.ranges.clone(),
            &flop_solution.strategies,
            1.0 / config.roots as f64,
            &mut records,
            range_records.as_mut(),
            !config.range_only,
        )?;

        for leaf in selected_leaves {
            let turn_game_state = GameState {
                street: Street::Turn,
                actor: leaf.actor,
                invested: leaf.invested,
                street_invested: [0.0, 0.0],
                last_full_raise: config.game.big_blind_bb,
                aggressions: 0,
                checks: 0,
                raise_reopened: true,
                public_history: leaf.public_history.clone(),
                trajectory: leaf.trajectory.clone(),
                terminal: None,
            };
            let turn_state = PublicBeliefState::from_game_state(
                leaf.board.to_vec(),
                &turn_game_state,
                leaf.ranges.clone(),
            );
            let turn_solution = solve_turn_river(TurnRiverSolveConfig {
                game: config.game.clone(),
                state: turn_state,
                iterations: config.turn_river_iterations,
                averaging_delay: config.turn_river_averaging_delay,
                river_refinement_iterations: 0,
                regret_matching_plus: false,
            })?;
            let structurally_usable_turn_teacher = turn_solution
                .validation
                .reasons
                .iter()
                .all(|reason| reason.starts_with("turn-river abstraction exploitability "))
                && turn_solution
                    .metrics
                    .exact_abstract_exploitability_bb_per_hand
                    .is_finite()
                && turn_solution.metrics.maximum_probability_sum_error <= 1e-6
                && turn_solution.metrics.zero_sum_residual_bb <= 1e-8;
            let turn_training_usable = structurally_usable_turn_teacher
                && (!config.require_accepted_turn_river_teachers
                    || turn_solution.validation.status == "accepted");
            if !turn_training_usable {
                return Err(format!(
                    "turn-river action teacher failed the configured quality gate at root {absolute_root} leaf {leaves_solved}: {}",
                    turn_solution.validation.reasons.join("; ")
                )
                .into());
            }
            maximum_turn_river_exploitability = maximum_turn_river_exploitability.max(
                turn_solution
                    .metrics
                    .exact_abstract_exploitability_bb_per_hand,
            );
            append_public_belief_policy_records(
                &config.game,
                Some(&policy),
                turn_solution.state.game_state(),
                &turn_solution.state.board,
                turn_solution.state.ranges.clone(),
                &turn_solution.strategies,
                1.0 / (config.roots * config.turn_leaves_per_root) as f64,
                &mut records,
                range_records.as_mut(),
                !config.range_only,
            )?;
            leaves_solved += 1;
        }
        roots_solved += 1;
        absolute_root += 1;
    }

    let candidate_records = records.seen;
    let sampled = candidate_records > records.len();
    let records = records.into_records();
    let flop_records = records
        .iter()
        .filter(|record| {
            serde_json::from_slice::<serde_json::Value>(record)
                .expect("generated action record remains valid JSON")["state"]["street"]
                == "flop"
        })
        .count();
    let turn_river_records = records.len() - flop_records;
    let truncated = false;
    let validation = BlueprintValidation {
        status: if roots_solved == config.roots
            && leaves_solved == config.roots * config.turn_leaves_per_root
        {
            "accepted_for_training"
        } else {
            "rejected"
        }
        .to_owned(),
        reasons: vec![
            "local postflop teachers are successor-training targets, not release-qualified policies; activation still requires the unchanged full-game exploitability gate"
                .to_owned(),
            if sampled {
                format!(
                    "deterministic street-stratified reservoir with inverse-inclusion weight correction retained {} of {candidate_records} complete-tree action rows",
                    records.len()
                )
            } else {
                "every complete-tree action row was retained".to_owned()
            },
        ],
    };
    let teacher = serde_json::json!({
        "schema": "hu-range-conditioned-postflop-action-teacher-v1",
        "sourcePolicySha256": source_policy_sha256,
        "valueNetworkSha256": value_network_sha256,
        "auxiliaryValueNetworkSha256s": auxiliary_value_network_sha256s,
        "evaluationValueNetworkSha256": evaluation_value_network_sha256,
        "flopIterations": config.flop_iterations,
        "flopIterationCheckpoints": config.flop_iteration_checkpoints,
        "flopConvergence": flop_convergence,
        "flopAveragingDelay": config.flop_averaging_delay,
        "flopRegretMatchingPlus": config.flop_regret_matching_plus,
        "requiresAcceptedFlopTeachers": config.require_accepted_flop_teachers,
        "requiresRangeConsistentFlopTeachers": config.require_range_consistent_flop_teachers,
        "flopRangeResponseCheckpoints": config.flop_response_checkpoints,
        "flopRangeResponse": flop_range_response,
        "maximumFlopRangeResponseGainBbPerHand": config.maximum_flop_range_response_gain_bb_per_hand,
        "requiresAcceptedTurnRiverTeachers": config.require_accepted_turn_river_teachers,
        "turnRiverIterations": config.turn_river_iterations,
        "turnRiverAveragingDelay": config.turn_river_averaging_delay,
        "rootOffset": config.root_offset,
        "roots": roots_solved,
        "turnLeaves": leaves_solved,
        "explorationProbability": config.exploration_probability,
        "actionAbstraction": config.game.action_abstraction,
        "validation": validation,
    });
    if !config.range_only {
        super::neural::write_average_strategy_dataset(
            &config.game,
            config.seed,
            teacher.clone(),
            &records,
            &config.output,
        )?;
    }
    let range_conditioned_records = range_records.as_ref().map_or(0, |records| records.len());
    if let (Some(output), Some(records)) = (&config.range_output, range_records) {
        write_range_conditioned_policy_dataset(
            &config.game,
            config.seed,
            &teacher,
            &records.into_records(),
            output,
        )?;
    }
    Ok(PostflopActionTargetReport {
        schema: "hu-range-conditioned-postflop-action-target-report-v1".to_owned(),
        method: "authentic_preflop_reach_conditioning_v49_depth_limited_flop_resolver_and_exact_complete_turn_river_dcfr_action_export".to_owned(),
        seed: config.seed,
        source_policy_sha256,
        value_network_sha256,
        auxiliary_value_network_sha256s,
        evaluation_value_network_sha256,
        root_offset: config.root_offset,
        roots: roots_solved,
        turn_leaves: leaves_solved,
        flop_regret_matching_plus: config.flop_regret_matching_plus,
        required_accepted_flop_teachers: config.require_accepted_flop_teachers,
        required_range_consistent_flop_teachers: config
            .require_range_consistent_flop_teachers,
        required_accepted_turn_river_teachers: config.require_accepted_turn_river_teachers,
        flop_convergence,
        flop_range_response,
        flop_records,
        turn_river_records,
        total_records: records.len(),
        candidate_records,
        truncated,
        maximum_flop_exploitability_bb_per_hand: maximum_flop_exploitability,
        maximum_turn_river_exploitability_bb_per_hand: maximum_turn_river_exploitability,
        output: config.output,
        range_output: config.range_output,
        range_conditioned_records,
        validation,
    })
}

pub fn solve_turn_river_continuation_values(
    config: TurnRiverSolveConfig,
) -> Result<TurnRiverContinuationValues, String> {
    let mut solver = TurnRiverSolver::new(config)?;
    solver.train();
    Ok(solver.finish_continuation_values())
}

#[derive(Clone, Debug)]
pub struct TurnTargetGenerationConfig {
    pub game: BlueprintConfig,
    pub states: usize,
    pub river_iterations: u64,
    pub river_averaging_delay: u64,
    pub seed: u64,
    pub threads: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TurnValueTarget {
    pub state_id: String,
    pub board: [u8; 4],
    pub actor: usize,
    pub invested_bb: [f64; 2],
    pub ranges: [Vec<f32>; 2],
    pub counterfactual_values_bb: [Vec<f32>; 2],
    pub opponent_compatible_mass: [Vec<f32>; 2],
    pub exact_river_cards: usize,
    pub maximum_river_exploitability_bb_per_hand: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_river_exploitability_bb_per_hand: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_turn_river_exploitability_bb_per_hand: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_river_maximum_probability_sum_error: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_only_best_response_gain_bb_per_hand: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub river_only_best_response_gain_bb_per_hand: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_river_solver_method: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_river_information_sets: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_information_sets: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub river_information_sets: Option<usize>,
    pub zero_sum_residual_bb: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range_particles: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range_replicates: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range_effective_sample_size: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub belief_method: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range_maximum_total_variation: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub off_policy_explorer: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_exploration_probability: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub public_action_line: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolver_root_board: Option<[u8; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolver_public_history: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolver_leaf_reach_probability: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TurnTargetDataset {
    pub schema: String,
    pub method: String,
    pub approximate: bool,
    pub game: BlueprintConfig,
    pub seed: u64,
    pub river_iterations: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_river_iterations: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_river_averaging_delay: Option<u64>,
    pub state_distribution: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_policy_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_exploration_probability: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exploration_method: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimum_sampled_pot_bb: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolver_source_value_network_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolver_iterations: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolver_leaf_population: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolver_leaf_probability_mass: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub component_dataset_sha256: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub component_seeds: Option<Vec<u64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub component_target_counts: Option<Vec<usize>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merge_method: Option<String>,
    pub targets: Vec<TurnValueTarget>,
    pub validation: BlueprintValidation,
}

/// Merge independently generated deterministic shards without weakening any
/// target or provenance gate. State identifiers are reassigned only after all
/// component bytes and source seeds have been recorded.
pub fn merge_turn_target_datasets(
    components: Vec<(TurnTargetDataset, String)>,
) -> Result<TurnTargetDataset, String> {
    if components.len() < 2 {
        return Err("turn-target merging requires at least two components".to_owned());
    }
    let mut iterator = components.into_iter();
    let (mut merged, first_hash) = iterator.next().expect("checked component count");
    if merged.component_dataset_sha256.is_some() {
        return Err("nested turn-target merges are not supported".to_owned());
    }
    let mut hashes = vec![first_hash];
    let mut seeds = vec![merged.seed];
    let mut counts = vec![merged.targets.len()];
    let mut targets = std::mem::take(&mut merged.targets);
    if merged.validation.status != "accepted" {
        return Err("every turn-target component must be accepted".to_owned());
    }
    for (dataset, hash) in iterator {
        if dataset.component_dataset_sha256.is_some() {
            return Err("nested turn-target merges are not supported".to_owned());
        }
        if dataset.validation.status != "accepted" {
            return Err("every turn-target component must be accepted".to_owned());
        }
        if dataset.schema != merged.schema
            || dataset.method != merged.method
            || dataset.approximate != merged.approximate
            || dataset.game != merged.game
            || dataset.river_iterations != merged.river_iterations
            || dataset.turn_river_iterations != merged.turn_river_iterations
            || dataset.turn_river_averaging_delay != merged.turn_river_averaging_delay
            || dataset.state_distribution != merged.state_distribution
            || dataset.source_policy_sha256 != merged.source_policy_sha256
            || dataset.sampling_exploration_probability != merged.sampling_exploration_probability
            || dataset.exploration_method != merged.exploration_method
            || dataset.minimum_sampled_pot_bb != merged.minimum_sampled_pot_bb
            || dataset.resolver_source_value_network_sha256
                != merged.resolver_source_value_network_sha256
            || dataset.resolver_iterations != merged.resolver_iterations
            || dataset.resolver_leaf_population != merged.resolver_leaf_population
            || dataset.resolver_leaf_probability_mass != merged.resolver_leaf_probability_mass
        {
            return Err("turn-target components have incompatible provenance".to_owned());
        }
        hashes.push(hash);
        seeds.push(dataset.seed);
        counts.push(dataset.targets.len());
        targets.extend(dataset.targets);
    }
    if hashes
        .iter()
        .any(|hash| hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()))
    {
        return Err("turn-target component hashes must be SHA-256 hex".to_owned());
    }
    if merged.schema != "hu-turn-public-belief-cfv-dataset-v2" {
        return Err("only complete-turn schema-v2 targets may be merged".to_owned());
    }
    if targets.len() < 64 {
        return Err("merged turn-target corpus requires at least 64 states".to_owned());
    }
    let distinct_boards = targets
        .iter()
        .map(|target| target.board)
        .collect::<BTreeSet<_>>()
        .len();
    if distinct_boards * 100 < targets.len() * 95 {
        return Err("merged turn-target corpus has fewer than 95% distinct boards".to_owned());
    }
    let mut fingerprints = BTreeSet::new();
    for (index, target) in targets.iter_mut().enumerate() {
        let exploitability = target
            .turn_river_exploitability_bb_per_hand
            .ok_or_else(|| format!("target {index} lacks complete-turn exploitability"))?;
        let current_exploitability = target
            .current_turn_river_exploitability_bb_per_hand
            .ok_or_else(|| format!("target {index} lacks current-policy diagnostics"))?;
        let probability_error = target
            .turn_river_maximum_probability_sum_error
            .ok_or_else(|| format!("target {index} lacks probability diagnostics"))?;
        let turn_gain = target
            .turn_only_best_response_gain_bb_per_hand
            .ok_or_else(|| format!("target {index} lacks turn attribution"))?;
        let river_gain = target
            .river_only_best_response_gain_bb_per_hand
            .ok_or_else(|| format!("target {index} lacks river attribution"))?;
        let method = target
            .turn_river_solver_method
            .as_deref()
            .ok_or_else(|| format!("target {index} lacks solver provenance"))?;
        let turn_information_sets = target
            .turn_information_sets
            .ok_or_else(|| format!("target {index} lacks turn information sets"))?;
        let river_information_sets = target
            .river_information_sets
            .ok_or_else(|| format!("target {index} lacks river information sets"))?;
        let information_sets = target
            .turn_river_information_sets
            .ok_or_else(|| format!("target {index} lacks total information sets"))?;
        let particles = target
            .range_particles
            .ok_or_else(|| format!("target {index} lacks belief particles"))?;
        let replicates = target
            .range_replicates
            .ok_or_else(|| format!("target {index} lacks belief replicates"))?;
        let effective_sample_size = target
            .range_effective_sample_size
            .ok_or_else(|| format!("target {index} lacks belief ESS"))?;
        let total_variation = target
            .range_maximum_total_variation
            .ok_or_else(|| format!("target {index} lacks belief variation"))?;
        let fingerprint = target
            .input_sha256
            .as_deref()
            .ok_or_else(|| format!("target {index} lacks an input fingerprint"))?;
        if !fingerprints.insert(fingerprint.to_owned()) {
            return Err(format!("target {index} duplicates an input fingerprint"));
        }
        if target.actor > 1
            || target.board.iter().collect::<BTreeSet<_>>().len() != 4
            || !exploitability.is_finite()
            || exploitability > 0.05
            || !current_exploitability.is_finite()
            || current_exploitability < 0.0
            || !probability_error.is_finite()
            || probability_error > 1e-6
            || !turn_gain.is_finite()
            || turn_gain < 0.0
            || turn_gain > exploitability + 1e-8
            || !river_gain.is_finite()
            || river_gain < 0.0
            || river_gain > exploitability + 1e-8
            || !method.contains("complete_turn_river_betting")
            || !method.contains("paired_alternating")
            || target.exact_river_cards != 48
            || turn_information_sets == 0
            || river_information_sets == 0
            || turn_information_sets.checked_add(river_information_sets) != Some(information_sets)
            || target.zero_sum_residual_bb.abs() > 1e-7
            || particles < 4_096
            || replicates < 2
            || effective_sample_size < particles as f64 * 0.1
            || total_variation > 0.15
            || !target
                .belief_method
                .as_deref()
                .is_some_and(|value| value.starts_with("exact_per-player_reach_factors"))
        {
            return Err(format!("target {index} fails a complete-turn merge gate"));
        }
        for player in 0..2 {
            if target.ranges[player].len() != COMBO_COUNT
                || target.counterfactual_values_bb[player].len() != COMBO_COUNT
                || target.opponent_compatible_mass[player].len() != COMBO_COUNT
                || target.ranges[player]
                    .iter()
                    .any(|value| !value.is_finite() || *value < 0.0)
                || target.counterfactual_values_bb[player]
                    .iter()
                    .any(|value| !value.is_finite())
                || target.opponent_compatible_mass[player]
                    .iter()
                    .any(|value| !value.is_finite() || *value < 0.0)
                || (target.ranges[player]
                    .iter()
                    .map(|value| *value as f64)
                    .sum::<f64>()
                    - 1.0)
                    .abs()
                    > 1e-5
            {
                return Err(format!("target {index} has invalid exact-combo vectors"));
            }
        }
        target.state_id = format!("turn-pbs-{index:06}");
    }
    merged.targets = targets;
    merged.component_dataset_sha256 = Some(hashes);
    merged.component_seeds = Some(seeds);
    merged.component_target_counts = Some(counts);
    merged.merge_method =
        Some("ordered_accepted_component_concatenation_with_full_target_revalidation".to_owned());
    merged.validation = BlueprintValidation {
        status: "accepted".to_owned(),
        reasons: Vec::new(),
    };
    Ok(merged)
}

/// Generates a small, deterministic pilot corpus. Range shapes deliberately
/// vary by state and player so a range-aware network can be compared against a
/// range-blind ablation. These are synthetic reachable-like public beliefs;
/// promotion remains blocked until the same generator is fed beliefs sampled
/// from frozen-policy self play.
pub fn generate_turn_targets(
    config: TurnTargetGenerationConfig,
) -> Result<TurnTargetDataset, String> {
    config.game.validate()?;
    if config.states == 0 {
        return Err("turn target generation requires at least one state".to_owned());
    }
    if config.river_iterations < 2 {
        return Err("turn target generation requires at least two river iterations".to_owned());
    }
    if config.threads == 0 {
        return Err("turn target generation requires at least one worker thread".to_owned());
    }
    let mut rng = SplitMix64::new(config.seed);
    let mut targets = Vec::with_capacity(config.states);
    for state_index in 0..config.states {
        let board = sample_unique_board4(&mut rng);
        let pot_options: [f64; 5] = [2.0, 4.0, 6.0, 10.0, 16.0];
        let pot =
            pot_options[state_index % pot_options.len()].min(config.game.effective_stack_bb * 1.5);
        let invested = [pot / 2.0, pot / 2.0];
        let ranges = [
            shaped_range(&board, state_index, 0),
            shaped_range(&board, state_index, 1),
        ];
        targets.push(turn_target_from_complete_continuation(
            &config.game,
            board,
            1,
            invested,
            ranges,
            config.river_iterations,
            config.river_averaging_delay,
            config.threads,
            state_index,
        )?);
    }
    let mut reasons = vec![
        "pilot public beliefs are deterministic synthetic reachable-like ranges, not frozen-policy self-play samples"
            .to_owned(),
    ];
    let maximum_zero_sum_residual = targets
        .iter()
        .map(|target| target.zero_sum_residual_bb)
        .fold(0.0f64, f64::max);
    if maximum_zero_sum_residual > 1e-7 {
        reasons.push(format!(
            "turn target zero-sum residual {maximum_zero_sum_residual:.3e} exceeds 1e-7"
        ));
    }
    let maximum_river_exploitability = targets
        .iter()
        .map(|target| target.maximum_river_exploitability_bb_per_hand)
        .fold(0.0f64, f64::max);
    if maximum_river_exploitability > 0.05 {
        reasons.push(format!(
            "at least one complete turn-river solve has {:.6}bb/hand abstract exploitability",
            maximum_river_exploitability
        ));
    }
    Ok(TurnTargetDataset {
        schema: "hu-turn-public-belief-cfv-dataset-v2".to_owned(),
        method: "complete_turn_river_public_belief_paired_alternating_dcfr_with_exact_private_cards_observed_river_chance_and_exact_showdown"
            .to_owned(),
        approximate: true,
        game: config.game,
        seed: config.seed,
        river_iterations: config.river_iterations,
        turn_river_iterations: Some(config.river_iterations),
        turn_river_averaging_delay: Some(config.river_averaging_delay),
        state_distribution: "synthetic_reachable_like_pilot".to_owned(),
        source_policy_sha256: None,
        sampling_exploration_probability: None,
        exploration_method: None,
        minimum_sampled_pot_bb: None,
        resolver_source_value_network_sha256: None,
        resolver_iterations: None,
        resolver_leaf_population: None,
        resolver_leaf_probability_mass: None,
        component_dataset_sha256: None,
        component_seeds: None,
        component_target_counts: None,
        merge_method: None,
        targets,
        validation: BlueprintValidation {
            status: "rejected".to_owned(),
            reasons,
        },
    })
}

#[allow(clippy::too_many_arguments)]
fn turn_target_from_complete_continuation(
    game: &BlueprintConfig,
    board: [u8; 4],
    actor: usize,
    invested: [f64; 2],
    ranges: [Vec<f64>; 2],
    river_iterations: u64,
    river_averaging_delay: u64,
    _threads: usize,
    state_index: usize,
) -> Result<TurnValueTarget, String> {
    let state = PublicBeliefState::turn_start(board, actor, invested, ranges)
        .validate_street_and_normalize(game, Street::Turn, 4)?;
    let solution = solve_turn_river_continuation_values(TurnRiverSolveConfig {
        game: game.clone(),
        state: state.clone(),
        iterations: river_iterations,
        averaging_delay: river_averaging_delay,
        river_refinement_iterations: 0,
        regret_matching_plus: false,
    })?;
    Ok(TurnValueTarget {
        state_id: format!("turn-pbs-{state_index:06}"),
        board,
        actor,
        invested_bb: invested,
        ranges: std::array::from_fn(|player| {
            state.ranges[player]
                .iter()
                .map(|value| *value as f32)
                .collect()
        }),
        counterfactual_values_bb: solution.counterfactual_values_bb,
        opponent_compatible_mass: solution.opponent_compatible_mass,
        exact_river_cards: solution.metrics.exact_river_cards,
        maximum_river_exploitability_bb_per_hand: solution
            .metrics
            .exact_abstract_exploitability_bb_per_hand,
        turn_river_exploitability_bb_per_hand: Some(
            solution.metrics.exact_abstract_exploitability_bb_per_hand,
        ),
        current_turn_river_exploitability_bb_per_hand: Some(
            solution.metrics.current_strategy_exploitability_bb_per_hand,
        ),
        turn_river_maximum_probability_sum_error: Some(
            solution.metrics.maximum_probability_sum_error,
        ),
        turn_only_best_response_gain_bb_per_hand: Some(
            solution.metrics.turn_only_best_response_gain_bb_per_hand,
        ),
        river_only_best_response_gain_bb_per_hand: Some(
            solution.metrics.river_only_best_response_gain_bb_per_hand,
        ),
        turn_river_solver_method: Some(solution.method),
        turn_river_information_sets: Some(solution.metrics.information_sets),
        turn_information_sets: Some(solution.metrics.turn_information_sets),
        river_information_sets: Some(solution.metrics.river_information_sets),
        zero_sum_residual_bb: solution.metrics.zero_sum_residual_bb,
        range_particles: None,
        range_replicates: None,
        range_effective_sample_size: None,
        belief_method: None,
        range_maximum_total_variation: None,
        input_sha256: None,
        off_policy_explorer: None,
        sampling_exploration_probability: None,
        public_action_line: None,
        resolver_root_board: None,
        resolver_public_history: None,
        resolver_leaf_reach_probability: None,
    })
}

pub fn upgrade_turn_value_target(
    game: &BlueprintConfig,
    source: &TurnValueTarget,
    iterations: u64,
    averaging_delay: u64,
) -> Result<TurnValueTarget, String> {
    let ranges = std::array::from_fn(|player| {
        source.ranges[player]
            .iter()
            .map(|value| *value as f64)
            .collect::<Vec<_>>()
    });
    let fingerprint = turn_target_input_sha256(
        game,
        source.board,
        source.actor,
        source.invested_bb,
        &ranges,
        iterations,
        averaging_delay,
    )
    .map_err(|error| error.to_string())?;
    let mut upgraded = turn_target_from_complete_continuation(
        game,
        source.board,
        source.actor,
        source.invested_bb,
        ranges,
        iterations,
        averaging_delay,
        1,
        0,
    )?;
    upgraded.state_id = source.state_id.clone();
    upgraded.range_particles = source.range_particles;
    upgraded.range_replicates = source.range_replicates;
    upgraded.range_effective_sample_size = source.range_effective_sample_size;
    upgraded.belief_method = source.belief_method.clone();
    upgraded.range_maximum_total_variation = source.range_maximum_total_variation;
    upgraded.input_sha256 = Some(fingerprint);
    upgraded.off_policy_explorer = source.off_policy_explorer;
    upgraded.sampling_exploration_probability = source.sampling_exploration_probability;
    upgraded.public_action_line = source.public_action_line.clone();
    upgraded.resolver_root_board = source.resolver_root_board;
    upgraded.resolver_public_history = source.resolver_public_history.clone();
    upgraded.resolver_leaf_reach_probability = source.resolver_leaf_reach_probability;
    Ok(upgraded)
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn legacy_turn_target_from_exact_rivers(
    game: &BlueprintConfig,
    board: [u8; 4],
    actor: usize,
    invested: [f64; 2],
    ranges: [Vec<f64>; 2],
    river_iterations: u64,
    river_averaging_delay: u64,
    threads: usize,
    state_index: usize,
) -> Result<TurnValueTarget, String> {
    let original_ranges = std::array::from_fn(|player| normalize_masked(&ranges[player], &board));
    let conflicts = combo_conflicts();
    let compatible_mass: [Vec<f64>; 2] = std::array::from_fn(|player| {
        (0..COMBO_COUNT)
            .map(|own| {
                compatible_mass_from_conflicts(&original_ranges[1 - player], &conflicts, own)
            })
            .collect()
    });
    let mut raw_values = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
    let mut exact_river_cards = 0usize;
    let mut maximum_river_exploitability = 0.0f64;
    let river_cards = (0..52u8)
        .filter(|river| !board.contains(river))
        .collect::<Vec<_>>();
    let worker_count = threads.min(river_cards.len()).max(1);
    let solved = std::thread::scope(|scope| {
        let mut workers = Vec::with_capacity(worker_count);
        for worker in 0..worker_count {
            let assigned = river_cards
                .iter()
                .copied()
                .skip(worker)
                .step_by(worker_count)
                .collect::<Vec<_>>();
            let game = game.clone();
            let original_ranges = original_ranges.clone();
            workers.push(scope.spawn(move || {
                assigned
                    .into_iter()
                    .map(|river| {
                        let river_board = [board[0], board[1], board[2], board[3], river];
                        let mut masked = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
                        let mut totals = [0.0; 2];
                        for player in 0..2 {
                            for combo in all_combos() {
                                let weight = if combo.cards().contains(&river) {
                                    0.0
                                } else {
                                    original_ranges[player][combo.key()]
                                };
                                masked[player][combo.key()] = weight;
                                totals[player] += weight;
                            }
                            if totals[player] <= EPSILON {
                                return Err("river removal exhausted a turn range".to_owned());
                            }
                            for weight in &mut masked[player] {
                                *weight /= totals[player];
                            }
                        }
                        let solution = solve_river(RiverSolveConfig {
                            game: game.clone(),
                            state: PublicBeliefState::river_start(
                                river_board,
                                actor,
                                invested,
                                masked,
                            ),
                            iterations: river_iterations,
                            averaging_delay: river_averaging_delay,
                        })?;
                        Ok((river, totals, solution))
                    })
                    .collect::<Result<Vec<_>, String>>()
            }));
        }
        let mut leaves = Vec::with_capacity(river_cards.len());
        for worker in workers {
            leaves.extend(
                worker
                    .join()
                    .map_err(|_| "river solve worker panicked".to_owned())??,
            );
        }
        leaves.sort_by_key(|(river, _, _)| *river);
        Ok::<_, String>(leaves)
    })?;
    for (_, totals, solution) in solved {
        exact_river_cards += 1;
        maximum_river_exploitability = maximum_river_exploitability
            .max(solution.metrics.exact_abstract_exploitability_bb_per_hand);
        for player in 0..2 {
            let opponent_total = totals[1 - player];
            for combo in 0..COMBO_COUNT {
                raw_values[player][combo] += solution.counterfactual_values_bb[player][combo]
                    as f64
                    * solution.opponent_compatible_mass[player][combo] as f64
                    * opponent_total;
            }
        }
    }
    // Every compatible pair of hole cards has 44 remaining river cards.
    let chance_denominator = 44.0;
    let counterfactual_values_bb: [Vec<f32>; 2] = std::array::from_fn(|player| {
        raw_values[player]
            .iter()
            .zip(&compatible_mass[player])
            .map(|(raw, mass)| {
                if *mass > EPSILON {
                    (raw / (chance_denominator * mass)) as f32
                } else {
                    0.0
                }
            })
            .collect()
    });
    let joint_mass = joint_compatibility_mass(&original_ranges);
    let aggregate = |player: usize| {
        original_ranges[player]
            .iter()
            .zip(&counterfactual_values_bb[player])
            .zip(&compatible_mass[player])
            .map(|((reach, value), opponent_mass)| reach * *value as f64 * opponent_mass)
            .sum::<f64>()
            / joint_mass
    };
    let zero_sum_residual = (aggregate(0) + aggregate(1)).abs();
    Ok(TurnValueTarget {
        state_id: format!("turn-pbs-{state_index:06}"),
        board,
        actor,
        invested_bb: invested,
        ranges: std::array::from_fn(|player| {
            original_ranges[player]
                .iter()
                .map(|value| *value as f32)
                .collect()
        }),
        counterfactual_values_bb,
        opponent_compatible_mass: std::array::from_fn(|player| {
            compatible_mass[player]
                .iter()
                .map(|value| *value as f32)
                .collect()
        }),
        exact_river_cards,
        maximum_river_exploitability_bb_per_hand: maximum_river_exploitability,
        turn_river_exploitability_bb_per_hand: None,
        current_turn_river_exploitability_bb_per_hand: None,
        turn_river_maximum_probability_sum_error: None,
        turn_only_best_response_gain_bb_per_hand: None,
        river_only_best_response_gain_bb_per_hand: None,
        turn_river_solver_method: None,
        turn_river_information_sets: None,
        turn_information_sets: None,
        river_information_sets: None,
        zero_sum_residual_bb: zero_sum_residual,
        range_particles: None,
        range_replicates: None,
        range_effective_sample_size: None,
        belief_method: None,
        range_maximum_total_variation: None,
        input_sha256: None,
        off_policy_explorer: None,
        sampling_exploration_probability: None,
        public_action_line: None,
        resolver_root_board: None,
        resolver_public_history: None,
        resolver_leaf_reach_probability: None,
    })
}

#[derive(Clone, Debug)]
pub struct SelfPlayTurnTargetConfig {
    pub game: BlueprintConfig,
    pub states: usize,
    pub range_particles: u64,
    pub river_iterations: u64,
    pub river_averaging_delay: u64,
    pub seed: u64,
    pub threads: usize,
    pub network_path: PathBuf,
    pub belief_replicates: u32,
    pub exploration_probability: f64,
    pub minimum_pot_bb: f64,
    pub checkpoint_dir: Option<PathBuf>,
}

struct PreparedSelfPlayTurnTarget {
    state_index: usize,
    board: [u8; 4],
    actor: usize,
    invested: [f64; 2],
    ranges: [Vec<f64>; 2],
    minimum_ess: f64,
    maximum_total_variation: f64,
    fingerprint: String,
    checkpoint_path: Option<PathBuf>,
    explorer: Option<usize>,
    exploration_action_line: Option<Vec<String>>,
}

fn solve_prepared_self_play_turn_target(
    config: &SelfPlayTurnTargetConfig,
    prepared: &PreparedSelfPlayTurnTarget,
) -> Result<TurnValueTarget, String> {
    let mut target = if let Some(path) = &prepared.checkpoint_path {
        if path.exists() {
            let cached: TurnValueTarget =
                serde_json::from_slice(&fs::read(path).map_err(|error| error.to_string())?)
                    .map_err(|error| error.to_string())?;
            if cached.input_sha256.as_deref() != Some(prepared.fingerprint.as_str()) {
                return Err(format!(
                    "checkpoint {} has the wrong input fingerprint",
                    path.display()
                ));
            }
            cached
        } else {
            let mut solved = turn_target_from_complete_continuation(
                &config.game,
                prepared.board,
                prepared.actor,
                prepared.invested,
                prepared.ranges.clone(),
                config.river_iterations,
                config.river_averaging_delay,
                1,
                prepared.state_index,
            )?;
            solved.input_sha256 = Some(prepared.fingerprint.clone());
            solved
        }
    } else {
        turn_target_from_complete_continuation(
            &config.game,
            prepared.board,
            prepared.actor,
            prepared.invested,
            prepared.ranges.clone(),
            config.river_iterations,
            config.river_averaging_delay,
            1,
            prepared.state_index,
        )?
    };
    let belief_method =
        "exact_per-player_reach_factors_with_independent_stratified_resampling_replicates";
    if let Some(path) = &prepared.checkpoint_path {
        let diagnostics_match = target.range_particles == Some(config.range_particles)
            && target.range_replicates == Some(config.belief_replicates)
            && target.belief_method.as_deref() == Some(belief_method)
            && target
                .range_effective_sample_size
                .is_some_and(|stored| (stored - prepared.minimum_ess).abs() <= 1e-9)
            && target
                .range_maximum_total_variation
                .is_some_and(|stored| (stored - prepared.maximum_total_variation).abs() <= 1e-12)
            && target.off_policy_explorer == prepared.explorer
            && target.sampling_exploration_probability
                == (config.exploration_probability > 0.0).then_some(config.exploration_probability)
            && target.public_action_line == prepared.exploration_action_line;
        if !diagnostics_match {
            target.range_particles = Some(config.range_particles);
            target.range_replicates = Some(config.belief_replicates);
            target.range_effective_sample_size = Some(prepared.minimum_ess);
            target.belief_method = Some(belief_method.to_owned());
            target.range_maximum_total_variation = Some(prepared.maximum_total_variation);
            target.input_sha256 = Some(prepared.fingerprint.clone());
            target.off_policy_explorer = prepared.explorer;
            target.sampling_exploration_probability =
                prepared.explorer.map(|_| config.exploration_probability);
            target.public_action_line = prepared.exploration_action_line.clone();
            write_target_checkpoint(path, &target).map_err(|error| error.to_string())?;
            // Normalize a freshly solved or upgraded checkpoint once at the
            // persistence boundary. Stable cached checkpoints are deliberately
            // not rewritten: repeated parse/serialize cycles can move diagnostic
            // f64 values by one ULP.
            target = serde_json::from_slice(&fs::read(path).map_err(|error| error.to_string())?)
                .map_err(|error| error.to_string())?;
        }
    } else {
        target.range_particles = Some(config.range_particles);
        target.range_replicates = Some(config.belief_replicates);
        target.range_effective_sample_size = Some(prepared.minimum_ess);
        target.belief_method = Some(belief_method.to_owned());
        target.range_maximum_total_variation = Some(prepared.maximum_total_variation);
        target.input_sha256 = Some(prepared.fingerprint.clone());
        target.off_policy_explorer = prepared.explorer;
        target.sampling_exploration_probability =
            prepared.explorer.map(|_| config.exploration_probability);
        target.public_action_line = prepared.exploration_action_line.clone();
    }
    Ok(target)
}

fn solve_prepared_self_play_turn_targets(
    config: &SelfPlayTurnTargetConfig,
    prepared: &[PreparedSelfPlayTurnTarget],
) -> Result<Vec<TurnValueTarget>, String> {
    let worker_count = config.threads.min(prepared.len()).max(1);
    let next_state = std::sync::atomic::AtomicUsize::new(0);
    let worker_results = std::thread::scope(|scope| {
        let workers = (0..worker_count)
            .map(|_| {
                let next_state = &next_state;
                scope.spawn(move || {
                    let mut solved = Vec::new();
                    loop {
                        let index = next_state.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(state) = prepared.get(index) else {
                            break;
                        };
                        solved.push((
                            state.state_index,
                            solve_prepared_self_play_turn_target(config, state),
                        ));
                    }
                    solved
                })
            })
            .collect::<Vec<_>>();
        workers
            .into_iter()
            .map(|worker| {
                worker
                    .join()
                    .map_err(|_| "self-play turn-target worker panicked".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()
    })?;
    let mut targets = vec![None; prepared.len()];
    for (state_index, result) in worker_results.into_iter().flatten() {
        let target = result?;
        if state_index >= targets.len() || targets[state_index].replace(target).is_some() {
            return Err("self-play turn-target workers produced invalid indices".to_owned());
        }
    }
    targets
        .into_iter()
        .enumerate()
        .map(|(index, target)| {
            target.ok_or_else(|| format!("self-play turn-target worker omitted state {index}"))
        })
        .collect()
}

pub fn generate_self_play_turn_targets(
    config: SelfPlayTurnTargetConfig,
) -> Result<TurnTargetDataset, Box<dyn Error>> {
    config.game.validate()?;
    if config.states == 0
        || config.range_particles < 2
        || config.belief_replicates < 2
        || config.threads == 0
        || !config.exploration_probability.is_finite()
        || !(0.0..1.0).contains(&config.exploration_probability)
        || !config.minimum_pot_bb.is_finite()
        || config.minimum_pot_bb < 0.0
        || config.minimum_pot_bb > config.game.effective_stack_bb * 2.0
    {
        return Err("self-play targets require states, range particles, and threads".into());
    }
    if let Some(directory) = &config.checkpoint_dir {
        fs::create_dir_all(directory)?;
    }
    let policy_bytes = fs::read(&config.network_path)?;
    let source_policy_sha256 = format!("{:x}", Sha256::digest(&policy_bytes));
    let policy = FrozenPolicy::load(&config.network_path)?;
    let mut chance = SplitMix64::new(config.seed);
    let mut prepared = Vec::with_capacity(config.states);
    let mut attempts = 0usize;
    while prepared.len() < config.states {
        attempts += 1;
        if attempts > config.states * 1_000 {
            return Err("could not sample enough nonterminal self-play turn states".into());
        }
        let true_deal = Deal::sample(&mut chance);
        let explorer = (config.exploration_probability > 0.0).then_some(prepared.len() % 2);
        let Some((turn_state, action_line)) = sample_turn_line(
            &policy,
            &config.game,
            &true_deal,
            &mut chance,
            explorer,
            config.exploration_probability,
        ) else {
            continue;
        };
        if turn_state.invested.iter().sum::<f64>() + EPSILON < config.minimum_pot_bb {
            continue;
        }
        let board = [
            true_deal.board[0],
            true_deal.board[1],
            true_deal.board[2],
            true_deal.board[3],
        ];
        let ranges = match exact_reach_factors_for_line(
            &policy,
            &config.game,
            board,
            &action_line,
            config.threads,
        ) {
            Ok(ranges) => ranges,
            Err(error)
                if config.exploration_probability > 0.0 && error.contains("zero exact reach") =>
            {
                continue
            }
            Err(error) => return Err(error.into()),
        };
        let mut minimum_ess = f64::INFINITY;
        let mut maximum_total_variation = 0.0f64;
        let mut particle_replicates = Vec::with_capacity(config.belief_replicates as usize);
        for replicate in 0..config.belief_replicates {
            let mut particle_rng = SplitMix64::new(
                config.seed
                    ^ 0xB311_EF00_0000_0000
                    ^ (prepared.len() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ replicate as u64,
            );
            let (estimate, effective_sample_size) = particle_reach_factors_from_exact(
                &ranges,
                board,
                config.range_particles,
                &mut particle_rng,
            )?;
            minimum_ess = minimum_ess.min(effective_sample_size);
            maximum_total_variation =
                maximum_total_variation.max(maximum_range_total_variation(&ranges, &estimate));
            for previous in &particle_replicates {
                maximum_total_variation =
                    maximum_total_variation.max(maximum_range_total_variation(previous, &estimate));
            }
            particle_replicates.push(estimate);
        }
        let state_index = prepared.len();
        let exploration_action_line = explorer.map(|_| action_line.clone());
        let fingerprint = turn_target_input_sha256(
            &config.game,
            board,
            turn_state.actor,
            turn_state.invested,
            &ranges,
            config.river_iterations,
            config.river_averaging_delay,
        )?;
        let checkpoint_path = config
            .checkpoint_dir
            .as_ref()
            .map(|directory| directory.join(format!("turn-{state_index:06}-{fingerprint}.json")));
        prepared.push(PreparedSelfPlayTurnTarget {
            state_index,
            board,
            actor: turn_state.actor,
            invested: turn_state.invested,
            ranges,
            minimum_ess,
            maximum_total_variation,
            fingerprint,
            checkpoint_path,
            explorer,
            exploration_action_line,
        });
    }
    let targets = solve_prepared_self_play_turn_targets(&config, &prepared)
        .map_err(|error| -> Box<dyn Error> { error.into() })?;
    let maximum_river_exploitability = targets
        .iter()
        .map(|target| target.maximum_river_exploitability_bb_per_hand)
        .fold(0.0f64, f64::max);
    let minimum_ess = targets
        .iter()
        .filter_map(|target| target.range_effective_sample_size)
        .fold(f64::INFINITY, f64::min);
    let mut reasons = Vec::new();
    let maximum_range_total_variation = targets
        .iter()
        .filter_map(|target| target.range_maximum_total_variation)
        .fold(0.0f64, f64::max);
    let distinct_turn_boards = targets
        .iter()
        .map(|target| target.board)
        .collect::<BTreeSet<_>>()
        .len();
    if config.states < 64 {
        reasons.push(format!(
            "authentic public-belief pilot has {} states; release requires at least 64 before model gating",
            config.states
        ));
    }
    if config.range_particles < 4_096 {
        reasons.push(format!(
            "belief validation uses {} particles per replicate; the pilot gate requires 4096",
            config.range_particles
        ));
    }
    if distinct_turn_boards * 100 < config.states * 95 {
        reasons.push(format!(
            "only {distinct_turn_boards} of {} target states have distinct turn boards; the pilot requires at least 95%",
            config.states
        ));
    }
    if maximum_range_total_variation > 0.15 {
        reasons.push(format!(
            "maximum exact-vs-particle or cross-replicate exact-combo range total variation {maximum_range_total_variation:.6} exceeds the stratified 4096-particle pilot bound 0.15"
        ));
    }
    if maximum_river_exploitability > 0.05 {
        reasons.push(format!(
            "at least one complete turn-river solve has {maximum_river_exploitability:.6}bb/hand abstract exploitability"
        ));
    }
    if minimum_ess < config.range_particles as f64 * 0.1 {
        reasons.push(format!(
            "minimum range effective sample size {minimum_ess:.1} is below 10% of particles"
        ));
    }
    Ok(TurnTargetDataset {
        schema: "hu-turn-public-belief-cfv-dataset-v2".to_owned(),
        method: "frozen_policy_self_play_public_states_with_exact_per_player_reach_factors_particle_replicate_validation_and_complete_turn_river_public_belief_solving"
            .to_owned(),
        approximate: true,
        game: config.game,
        seed: config.seed,
        river_iterations: config.river_iterations,
        turn_river_iterations: Some(config.river_iterations),
        turn_river_averaging_delay: Some(config.river_averaging_delay),
        state_distribution: if config.exploration_probability > 0.0 {
            "frozen_v26_one_player_epsilon_exploration_exact_reach_factor_public_beliefs"
        } else {
            "frozen_v26_self_play_exact_reach_factor_public_beliefs"
        }
        .to_owned(),
        source_policy_sha256: Some(source_policy_sha256),
        sampling_exploration_probability: (config.exploration_probability > 0.0)
            .then_some(config.exploration_probability),
        exploration_method: (config.exploration_probability > 0.0).then(|| {
            "one_player_per_trajectory_epsilon_uniform_action_sampling_with_frozen_policy_belief_conditioning"
                .to_owned()
        }),
        minimum_sampled_pot_bb: (config.minimum_pot_bb > 0.0)
            .then_some(config.minimum_pot_bb),
        resolver_source_value_network_sha256: None,
        resolver_iterations: None,
        resolver_leaf_population: None,
        resolver_leaf_probability_mass: None,
        component_dataset_sha256: None,
        component_seeds: None,
        component_target_counts: None,
        merge_method: None,
        targets,
        validation: BlueprintValidation {
            status: if reasons.is_empty() { "accepted" } else { "rejected" }.to_owned(),
            reasons,
        },
    })
}

#[derive(Clone, Debug)]
pub struct ResolverLeafTurnTargetConfig {
    pub game: BlueprintConfig,
    pub root_boards: Vec<[u8; 3]>,
    pub states_per_board: usize,
    pub root_pot_bb: f64,
    pub root_actor: usize,
    pub resolver_iterations: u64,
    pub resolver_averaging_delay: u64,
    pub river_iterations: u64,
    pub river_averaging_delay: u64,
    pub seed: u64,
    pub threads: usize,
    pub value_network_path: PathBuf,
    pub checkpoint_dir: Option<PathBuf>,
}

pub fn generate_resolver_leaf_turn_targets(
    config: ResolverLeafTurnTargetConfig,
) -> Result<TurnTargetDataset, Box<dyn Error>> {
    config.game.validate()?;
    let distinct_roots = config.root_boards.iter().copied().collect::<BTreeSet<_>>();
    if config.root_boards.len() < 3
        || distinct_roots.len() != config.root_boards.len()
        || config
            .root_boards
            .iter()
            .any(|board| board.iter().copied().collect::<BTreeSet<_>>().len() != 3)
        || config.states_per_board < 3
        || !config.root_pot_bb.is_finite()
        || config.root_pot_bb < 2.0
        || config.root_pot_bb > config.game.effective_stack_bb * 2.0
        || config.root_actor > 1
        || config.resolver_iterations < 2
        || config.resolver_averaging_delay >= config.resolver_iterations
        || config.river_iterations < 2
        || config.river_averaging_delay >= config.river_iterations
        || config.threads == 0
    {
        return Err("resolver-leaf targets require three distinct roots, three states per root, and valid solve controls".into());
    }
    if let Some(directory) = &config.checkpoint_dir {
        fs::create_dir_all(directory)?;
    }
    let network_bytes = fs::read(&config.value_network_path)?;
    let source_value_network_sha256 = format!("{:x}", Sha256::digest(&network_bytes));
    let network = PublicValueNetwork::read(&config.value_network_path)?;
    let source_policy_sha256 = network.source_policy_sha256.clone();
    let source_validation_status = network.source_validation_status.clone();
    let mut selected_leaves =
        Vec::with_capacity(config.root_boards.len() * config.states_per_board);
    let mut leaf_population = 0usize;
    let mut leaf_probability_mass = 0.0f64;
    for (root_index, board) in config.root_boards.iter().copied().enumerate() {
        let fingerprint = resolver_root_checkpoint_input_sha256(
            &config,
            root_index,
            board,
            &source_value_network_sha256,
        )?;
        let checkpoint_path = config.checkpoint_dir.as_ref().map(|directory| {
            directory.join(format!("resolver-root-{root_index:06}-{fingerprint}.json"))
        });
        let (checkpoint, reused) = if let Some(path) = &checkpoint_path {
            if path.exists() {
                (
                    serde_json::from_slice::<ResolverRootLeafCheckpoint>(&fs::read(path)?)?,
                    true,
                )
            } else {
                (
                    solve_resolver_root_leaf_checkpoint(
                        &config,
                        &network,
                        root_index,
                        board,
                        &source_value_network_sha256,
                        fingerprint.clone(),
                    )?,
                    false,
                )
            }
        } else {
            (
                solve_resolver_root_leaf_checkpoint(
                    &config,
                    &network,
                    root_index,
                    board,
                    &source_value_network_sha256,
                    fingerprint.clone(),
                )?,
                false,
            )
        };
        validate_resolver_root_checkpoint(
            &checkpoint,
            &fingerprint,
            &source_value_network_sha256,
            root_index,
            board,
            config.states_per_board,
            config.game.effective_stack_bb,
        )?;
        if !reused {
            if let Some(path) = &checkpoint_path {
                write_resolver_root_checkpoint(path, &checkpoint)?;
            }
        }
        eprintln!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "event": "resolver-root-leaves-ready",
                "rootIndex": root_index,
                "roots": config.root_boards.len(),
                "selectedLeaves": checkpoint.selected_leaves.len(),
                "reusedCheckpoint": reused,
            }))?
        );
        leaf_population += checkpoint.leaf_population;
        leaf_probability_mass += checkpoint.leaf_probability_mass;
        selected_leaves.extend(checkpoint.selected_leaves);
    }

    let mut targets = Vec::with_capacity(selected_leaves.len());
    for (state_index, leaf) in selected_leaves.into_iter().enumerate() {
        let fingerprint = turn_target_input_sha256(
            &config.game,
            leaf.board,
            leaf.actor,
            leaf.invested,
            &leaf.ranges,
            config.river_iterations,
            config.river_averaging_delay,
        )?;
        let checkpoint_path = config.checkpoint_dir.as_ref().map(|directory| {
            directory.join(format!("resolver-turn-{state_index:06}-{fingerprint}.json"))
        });
        let mut target = if let Some(path) = &checkpoint_path {
            if path.exists() {
                let cached: TurnValueTarget = serde_json::from_slice(&fs::read(path)?)?;
                if cached.input_sha256.as_deref() != Some(fingerprint.as_str()) {
                    return Err(format!(
                        "checkpoint {} has the wrong input fingerprint",
                        path.display()
                    )
                    .into());
                }
                cached
            } else {
                turn_target_from_complete_continuation(
                    &config.game,
                    leaf.board,
                    leaf.actor,
                    leaf.invested,
                    leaf.ranges.clone(),
                    config.river_iterations,
                    config.river_averaging_delay,
                    config.threads,
                    state_index,
                )?
            }
        } else {
            turn_target_from_complete_continuation(
                &config.game,
                leaf.board,
                leaf.actor,
                leaf.invested,
                leaf.ranges.clone(),
                config.river_iterations,
                config.river_averaging_delay,
                config.threads,
                state_index,
            )?
        };
        let diagnostics_match = target.input_sha256.as_deref() == Some(fingerprint.as_str())
            && target.belief_method.as_deref()
                == Some("exact_resolver_average_strategy_counterfactual_reach")
            && target.resolver_root_board == Some(leaf.root_board)
            && target.resolver_public_history.as_ref() == Some(&leaf.public_history)
            && target
                .resolver_leaf_reach_probability
                .is_some_and(|stored| (stored - leaf.reach_probability).abs() <= 1e-15);
        if !diagnostics_match {
            target.input_sha256 = Some(fingerprint);
            target.belief_method =
                Some("exact_resolver_average_strategy_counterfactual_reach".to_owned());
            target.resolver_root_board = Some(leaf.root_board);
            target.resolver_public_history = Some(leaf.public_history);
            target.resolver_leaf_reach_probability = Some(leaf.reach_probability);
            if let Some(path) = &checkpoint_path {
                write_target_checkpoint(path, &target)?;
                target = serde_json::from_slice(&fs::read(path)?)?;
            }
        }
        targets.push(target);
    }

    let maximum_river_exploitability = targets
        .iter()
        .map(|target| target.maximum_river_exploitability_bb_per_hand)
        .fold(0.0f64, f64::max);
    let maximum_zero_sum_residual = targets
        .iter()
        .map(|target| target.zero_sum_residual_bb.abs())
        .fold(0.0f64, f64::max);
    let distinct_turn_boards = targets
        .iter()
        .map(|target| target.board)
        .collect::<BTreeSet<_>>()
        .len();
    let mut reasons = Vec::new();
    if maximum_river_exploitability > 0.05 {
        reasons.push(format!(
            "at least one resolver-leaf complete turn-river solve has {maximum_river_exploitability:.6}bb/hand abstract exploitability"
        ));
    }
    if maximum_zero_sum_residual > 1e-7 {
        reasons.push(format!(
            "maximum resolver-leaf zero-sum residual {maximum_zero_sum_residual:.3e} exceeds 1e-7"
        ));
    }
    if distinct_turn_boards * 100 < targets.len() * 95 {
        reasons.push(format!(
            "only {distinct_turn_boards} of {} resolver targets have distinct turn boards",
            targets.len()
        ));
    }
    if source_validation_status.as_deref() != Some("accepted") {
        reasons
            .push("resolver source value network was not trained from accepted targets".to_owned());
    }
    Ok(TurnTargetDataset {
        schema: "hu-turn-public-belief-cfv-dataset-v2".to_owned(),
        method: "flop_resolver_average_strategy_counterfactual_turn_leaf_capture_with_complete_turn_river_public_belief_solving".to_owned(),
        approximate: true,
        game: config.game,
        seed: config.seed,
        river_iterations: config.river_iterations,
        turn_river_iterations: Some(config.river_iterations),
        turn_river_averaging_delay: Some(config.river_averaging_delay),
        state_distribution:
            "flop_resolver_average_strategy_counterfactual_turn_leaves".to_owned(),
        source_policy_sha256,
        sampling_exploration_probability: None,
        exploration_method: None,
        minimum_sampled_pot_bb: None,
        resolver_source_value_network_sha256: Some(source_value_network_sha256),
        resolver_iterations: Some(config.resolver_iterations),
        resolver_leaf_population: Some(leaf_population),
        resolver_leaf_probability_mass: Some(
            leaf_probability_mass / config.root_boards.len() as f64,
        ),
        component_dataset_sha256: None,
        component_seeds: None,
        component_target_counts: None,
        merge_method: None,
        targets,
        validation: BlueprintValidation {
            status: if reasons.is_empty() { "accepted" } else { "rejected" }.to_owned(),
            reasons,
        },
    })
}

fn sample_resolver_turn_leaves(
    leaves: &[ResolverTurnLeaf],
    count: usize,
    seed: u64,
) -> Result<Vec<ResolverTurnLeaf>, String> {
    let mut rng = SplitMix64::new(seed);
    let mut selected = Vec::with_capacity(count);
    let mut used = BTreeSet::new();
    for band in 0..3 {
        if selected.len() == count {
            break;
        }
        let candidates = leaves
            .iter()
            .enumerate()
            .filter(|(_, leaf)| {
                resolver_leaf_pot_band(leaf.invested) == band && !used.contains(&leaf.board)
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if let Some(index) = weighted_resolver_leaf_index(leaves, &candidates, &mut rng) {
            used.insert(leaves[index].board);
            selected.push(leaves[index].clone());
        }
    }
    while selected.len() < count {
        let candidates = leaves
            .iter()
            .enumerate()
            .filter(|(_, leaf)| !used.contains(&leaf.board))
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let Some(index) = weighted_resolver_leaf_index(leaves, &candidates, &mut rng) else {
            return Err(
                "resolver leaves do not contain enough distinct positive-reach boards".to_owned(),
            );
        };
        used.insert(leaves[index].board);
        selected.push(leaves[index].clone());
    }
    Ok(selected)
}

fn canonicalize_resolver_turn_leaf(mut leaf: ResolverTurnLeaf) -> Option<ResolverTurnLeaf> {
    for range in &mut leaf.ranges {
        for weight in range {
            *weight = (*weight as f32) as f64;
        }
    }
    leaf.reach_probability = (leaf.reach_probability * RESOLVER_REACH_CANONICAL_SCALE).round()
        / RESOLVER_REACH_CANONICAL_SCALE;
    (leaf.reach_probability > EPSILON).then_some(leaf)
}

fn weighted_resolver_leaf_index(
    leaves: &[ResolverTurnLeaf],
    candidates: &[usize],
    rng: &mut SplitMix64,
) -> Option<usize> {
    let total = candidates
        .iter()
        .map(|index| leaves[*index].reach_probability)
        .sum::<f64>();
    if total <= EPSILON {
        return None;
    }
    let roll = rng.next_f64() * total;
    let mut cumulative = 0.0;
    for index in candidates {
        cumulative += leaves[*index].reach_probability;
        if roll < cumulative {
            return Some(*index);
        }
    }
    candidates.last().copied()
}

fn resolver_leaf_pot_band(invested: [f64; 2]) -> usize {
    let maximum = invested[0].max(invested[1]);
    if maximum <= 3.5 {
        0
    } else if maximum <= 7.5 {
        1
    } else {
        2
    }
}

fn solve_resolver_root_leaf_checkpoint(
    config: &ResolverLeafTurnTargetConfig,
    network: &PublicValueNetwork,
    root_index: usize,
    board: [u8; 3],
    source_value_network_sha256: &str,
    input_sha256: String,
) -> Result<ResolverRootLeafCheckpoint, Box<dyn Error>> {
    let ranges = std::array::from_fn(|_| uniform_range(&board));
    let mut solver = FlopSolver::new(FlopResolveConfig {
        game: config.game.clone(),
        state: PublicBeliefState::flop_start(
            board,
            config.root_actor,
            [config.root_pot_bb / 2.0, config.root_pot_bb / 2.0],
            ranges,
        ),
        iterations: config.resolver_iterations,
        averaging_delay: config.resolver_averaging_delay,
        regret_matching_plus: false,
        value_network: network.clone(),
        auxiliary_value_networks: Vec::new(),
        threads: config.threads,
    })?;
    solver.train();
    // Resolver inference runs across worker threads. Platform math can differ
    // below persisted f32 precision, so canonicalize before checkpointing,
    // hashing, sampling, or solving labels.
    let leaves = solver
        .capture_average_turn_leaves()
        .into_iter()
        .filter_map(canonicalize_resolver_turn_leaf)
        .collect::<Vec<_>>();
    if leaves.len() < config.states_per_board {
        return Err(format!(
            "resolver root {root_index} produced only {} positive-reach leaves",
            leaves.len()
        )
        .into());
    }
    let leaf_probability_mass = leaves
        .iter()
        .map(|leaf| leaf.reach_probability)
        .sum::<f64>();
    let sampled = sample_resolver_turn_leaves(
        &leaves,
        config.states_per_board,
        config.seed ^ (root_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
    )?;
    let sampled_bands = sampled
        .iter()
        .map(|leaf| resolver_leaf_pot_band(leaf.invested))
        .collect::<BTreeSet<_>>();
    if ![0usize, 1, 2]
        .into_iter()
        .all(|band| sampled_bands.contains(&band))
    {
        return Err(format!(
            "resolver root {root_index} does not produce one sampled leaf in every pot band"
        )
        .into());
    }
    Ok(ResolverRootLeafCheckpoint {
        schema: RESOLVER_ROOT_CHECKPOINT_SCHEMA.to_owned(),
        input_sha256,
        source_value_network_sha256: source_value_network_sha256.to_owned(),
        root_index,
        root_board: board,
        leaf_population: leaves.len(),
        leaf_probability_mass,
        selected_leaves: sampled,
    })
}

fn resolver_root_checkpoint_input_sha256(
    config: &ResolverLeafTurnTargetConfig,
    root_index: usize,
    root_board: [u8; 3],
    source_value_network_sha256: &str,
) -> Result<String, serde_json::Error> {
    let bytes = serde_json::to_vec(&serde_json::json!({
        "schema": RESOLVER_ROOT_CHECKPOINT_SCHEMA,
        "game": config.game,
        "rootIndex": root_index,
        "rootBoard": root_board,
        "statesPerBoard": config.states_per_board,
        "rootPotBb": config.root_pot_bb,
        "rootActor": config.root_actor,
        "resolverIterations": config.resolver_iterations,
        "resolverAveragingDelay": config.resolver_averaging_delay,
        "seed": config.seed,
        "sourceValueNetworkSha256": source_value_network_sha256,
    }))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn validate_resolver_root_checkpoint(
    checkpoint: &ResolverRootLeafCheckpoint,
    expected_fingerprint: &str,
    source_value_network_sha256: &str,
    root_index: usize,
    root_board: [u8; 3],
    states_per_board: usize,
    effective_stack_bb: f64,
) -> Result<(), String> {
    if checkpoint.schema != RESOLVER_ROOT_CHECKPOINT_SCHEMA
        || checkpoint.input_sha256 != expected_fingerprint
        || checkpoint.source_value_network_sha256 != source_value_network_sha256
        || checkpoint.root_index != root_index
        || checkpoint.root_board != root_board
        || checkpoint.leaf_population < states_per_board
        || !checkpoint.leaf_probability_mass.is_finite()
        || checkpoint.leaf_probability_mass <= 0.0
        || checkpoint.leaf_probability_mass > 1.0 + 1e-6
        || checkpoint.selected_leaves.len() != states_per_board
    {
        return Err("resolver root checkpoint provenance does not match".to_owned());
    }
    let root_cards = root_board.into_iter().collect::<BTreeSet<_>>();
    let mut boards = BTreeSet::new();
    let mut bands = BTreeSet::new();
    for leaf in &checkpoint.selected_leaves {
        if leaf.root_board != root_board
            || leaf.board[..3] != root_board
            || leaf.board.iter().copied().collect::<BTreeSet<_>>().len() != 4
            || !root_cards.iter().all(|card| leaf.board.contains(card))
            || leaf.actor > 1
            || leaf.public_history.is_empty()
            || !leaf.reach_probability.is_finite()
            || leaf.reach_probability <= 0.0
            || leaf.reach_probability
                != (leaf.reach_probability * RESOLVER_REACH_CANONICAL_SCALE).round()
                    / RESOLVER_REACH_CANONICAL_SCALE
            || leaf
                .invested
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0 || *value > effective_stack_bb)
        {
            return Err("resolver root checkpoint contains an invalid leaf".to_owned());
        }
        if !boards.insert(leaf.board) {
            return Err("resolver root checkpoint repeats a turn board".to_owned());
        }
        bands.insert(resolver_leaf_pot_band(leaf.invested));
        for range in &leaf.ranges {
            if range.len() != COMBO_COUNT
                || range.iter().any(|weight| {
                    !weight.is_finite() || *weight < 0.0 || *weight != (*weight as f32) as f64
                })
                || (range.iter().sum::<f64>() - 1.0).abs() > 1e-5
            {
                return Err("resolver root checkpoint contains an invalid range".to_owned());
            }
            for combo in all_combos() {
                if combo.cards().iter().any(|card| leaf.board.contains(card))
                    && range[combo.key()] != 0.0
                {
                    return Err(
                        "resolver root checkpoint range violates exact card removal".to_owned()
                    );
                }
            }
        }
    }
    if ![0usize, 1, 2].into_iter().all(|band| bands.contains(&band)) {
        return Err("resolver root checkpoint does not cover every pot band".to_owned());
    }
    Ok(())
}

fn write_resolver_root_checkpoint(
    path: &Path,
    checkpoint: &ResolverRootLeafCheckpoint,
) -> Result<(), Box<dyn Error>> {
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, serde_json::to_vec(checkpoint)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn sample_turn_line(
    policy: &FrozenPolicy,
    game: &BlueprintConfig,
    deal: &Deal,
    rng: &mut SplitMix64,
    explorer: Option<usize>,
    exploration_probability: f64,
) -> Option<(GameState, Vec<String>)> {
    let mut state = GameState::initial(game);
    let mut line = Vec::new();
    while state.terminal.is_none() && state.street != Street::Turn {
        let actions = state.legal_actions(game);
        let strategy = policy.strategy(&state, deal, &actions, game);
        let sampling = if explorer == Some(state.actor) {
            epsilon_sampling_strategy(&strategy, exploration_probability)
        } else {
            strategy
        };
        let selected = sample_index(&sampling, rng);
        line.push(actions[selected].label.clone());
        state = state.apply(&actions[selected], game);
    }
    (state.street == Street::Turn && state.terminal.is_none()).then_some((state, line))
}

fn epsilon_sampling_strategy(strategy: &[f64], exploration_probability: f64) -> Vec<f64> {
    if strategy.is_empty() {
        return Vec::new();
    }
    let uniform = exploration_probability / strategy.len() as f64;
    strategy
        .iter()
        .map(|probability| (1.0 - exploration_probability) * probability + uniform)
        .collect()
}

fn exact_reach_factors_for_line(
    policy: &FrozenPolicy,
    game: &BlueprintConfig,
    board: [u8; 4],
    line: &[String],
    threads: usize,
) -> Result<[Vec<f64>; 2], String> {
    exact_reach_factors_for_visible_line(policy, game, &board, line, threads)
}

fn exact_reach_factors_for_visible_line(
    policy: &FrozenPolicy,
    game: &BlueprintConfig,
    board: &[u8],
    line: &[String],
    threads: usize,
) -> Result<[Vec<f64>; 2], String> {
    let mut ranges = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
    let legal = all_combos()
        .into_iter()
        .filter(|combo| !combo.cards().iter().any(|card| board.contains(card)))
        .collect::<Vec<_>>();
    let worker_count = threads.min(legal.len()).max(1);
    let solved = std::thread::scope(|scope| {
        let mut workers = Vec::with_capacity(worker_count);
        for worker in 0..worker_count {
            let assigned = legal
                .iter()
                .copied()
                .skip(worker)
                .step_by(worker_count)
                .collect::<Vec<_>>();
            workers.push(scope.spawn(move || {
                assigned
                    .into_iter()
                    .map(|combo| {
                        Ok((
                            combo.key(),
                            [
                                reach_likelihood_for_combo(policy, game, board, line, 0, combo)?,
                                reach_likelihood_for_combo(policy, game, board, line, 1, combo)?,
                            ],
                        ))
                    })
                    .collect::<Result<Vec<_>, String>>()
            }));
        }
        let mut result = Vec::with_capacity(legal.len());
        for worker in workers {
            result.extend(
                worker
                    .join()
                    .map_err(|_| "exact reach-factor worker panicked".to_owned())??,
            );
        }
        Ok::<_, String>(result)
    })?;
    for (combo, likelihoods) in solved {
        for player in 0..2 {
            ranges[player][combo] = likelihoods[player];
        }
    }
    for player in 0..2 {
        let total = ranges[player].iter().sum::<f64>();
        if total <= EPSILON {
            return Err(format!(
                "self-play line has zero exact reach for player {player}"
            ));
        }
        for weight in &mut ranges[player] {
            *weight /= total;
        }
    }
    Ok(ranges)
}

fn particle_reach_factors_from_exact(
    exact: &[Vec<f64>; 2],
    board: [u8; 4],
    particles: u64,
    rng: &mut SplitMix64,
) -> Result<([Vec<f64>; 2], f64), String> {
    let legal = all_combos()
        .into_iter()
        .filter(|combo| !combo.cards().iter().any(|card| board.contains(card)))
        .collect::<Vec<_>>();
    let mut ranges = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
    let mut minimum_effective_sample_size = f64::INFINITY;
    for player in 0..2 {
        let mut sum = 0.0;
        let mut squared_sum = 0.0;
        let offset = rng.index(legal.len());
        for sample in 0..particles {
            // Randomly rotated stratification covers every legal exact combo
            // before repeating one. This preserves the uniform proposal while
            // avoiding multinomial coverage noise in the replicate diagnostic.
            let combo = legal[(offset + sample as usize) % legal.len()];
            let likelihood = exact[player][combo.key()];
            if likelihood <= 0.0 || !likelihood.is_finite() {
                continue;
            }
            ranges[player][combo.key()] += likelihood;
            sum += likelihood;
            squared_sum += likelihood * likelihood;
        }
        if sum <= EPSILON {
            return Err(format!(
                "particle replicate has zero reach for player {player}"
            ));
        }
        for weight in &mut ranges[player] {
            *weight /= sum;
        }
        minimum_effective_sample_size =
            minimum_effective_sample_size.min(sum * sum / squared_sum.max(EPSILON));
    }
    Ok((ranges, minimum_effective_sample_size))
}

fn reach_likelihood_for_combo(
    policy: &FrozenPolicy,
    game: &BlueprintConfig,
    board: &[u8],
    line: &[String],
    player: usize,
    combo: Combo,
) -> Result<f64, String> {
    let deal = deal_with_visible_board_and_private_combo(board, player, combo);
    let mut state = GameState::initial(game);
    let mut likelihood = 1.0;
    for selected_label in line {
        let actions = state.legal_actions(game);
        let selected = actions
            .iter()
            .position(|action| &action.label == selected_label)
            .ok_or_else(|| format!("self-play line contains illegal action {selected_label}"))?;
        if state.actor == player {
            let strategy = policy.strategy(&state, &deal, &actions, game);
            likelihood *= strategy[selected];
        }
        state = state.apply(&actions[selected], game);
    }
    Ok(likelihood)
}

fn maximum_range_total_variation(first: &[Vec<f64>; 2], second: &[Vec<f64>; 2]) -> f64 {
    (0..2)
        .map(|player| {
            first[player]
                .iter()
                .zip(&second[player])
                .map(|(left, right)| (left - right).abs())
                .sum::<f64>()
                / 2.0
        })
        .fold(0.0f64, f64::max)
}

#[allow(clippy::too_many_arguments)]
pub fn turn_target_input_sha256(
    game: &BlueprintConfig,
    board: [u8; 4],
    actor: usize,
    invested: [f64; 2],
    ranges: &[Vec<f64>; 2],
    river_iterations: u64,
    river_averaging_delay: u64,
) -> Result<String, serde_json::Error> {
    let bytes = serde_json::to_vec(&serde_json::json!({
        "continuationSemantics": "complete_turn_river_public_belief_v5_with_paired_alternating_updates_street_attribution_solver_provenance_current_policy_and_probability_diagnostics",
        "game": game,
        "board": board,
        "actor": actor,
        "investedBb": invested,
        "ranges": ranges,
        "riverIterations": river_iterations,
        "riverAveragingDelay": river_averaging_delay,
    }))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn write_target_checkpoint(path: &Path, target: &TurnValueTarget) -> Result<(), Box<dyn Error>> {
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, serde_json::to_vec(target)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn sample_unique_board4(rng: &mut SplitMix64) -> [u8; 4] {
    let mut deck = [0u8; 52];
    for (index, card) in deck.iter_mut().enumerate() {
        *card = index as u8;
    }
    for index in 0..4 {
        let swap = index + rng.index(52 - index);
        deck.swap(index, swap);
    }
    [deck[0], deck[1], deck[2], deck[3]]
}

fn shaped_range(board: &[u8], state_index: usize, player: usize) -> Vec<f64> {
    let blocked = board.iter().copied().collect::<BTreeSet<_>>();
    let direction = if (state_index + player).is_multiple_of(2) {
        1.0
    } else {
        -0.7
    };
    let mut weights = all_combos()
        .into_iter()
        .map(|combo| {
            if combo.cards().iter().any(|card| blocked.contains(card)) {
                return 0.0;
            }
            let cards = combo.cards();
            let high = (cards[0] >> 2).max(cards[1] >> 2) as f64 / 12.0;
            let low = (cards[0] >> 2).min(cards[1] >> 2) as f64 / 12.0;
            let pair = f64::from((cards[0] >> 2) == (cards[1] >> 2));
            let suited = f64::from((cards[0] & 3) == (cards[1] & 3));
            let board_match = cards
                .iter()
                .filter(|card| board.iter().any(|public| public >> 2 == **card >> 2))
                .count() as f64;
            let phase =
                ((combo.key() * 17 + state_index * 31 + player * 13) % 97) as f64 / 96.0 - 0.5;
            (direction * (1.4 * high + 0.6 * low + 1.0 * pair + 0.2 * suited)
                + 0.35 * board_match
                + 0.25 * phase)
                .exp()
        })
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f64>();
    for weight in &mut weights {
        *weight /= total;
    }
    weights
}

#[cfg(test)]
fn normalize_masked(range: &[f64], board: &[u8]) -> Vec<f64> {
    let blocked = board.iter().copied().collect::<BTreeSet<_>>();
    let mut normalized = all_combos()
        .into_iter()
        .map(|combo| {
            if combo.cards().iter().any(|card| blocked.contains(card)) {
                0.0
            } else {
                range[combo.key()].max(0.0)
            }
        })
        .collect::<Vec<_>>();
    let total = normalized.iter().sum::<f64>();
    for weight in &mut normalized {
        *weight /= total;
    }
    normalized
}

fn combo_conflicts() -> Arc<Vec<Vec<usize>>> {
    static CONFLICTS: OnceLock<Arc<Vec<Vec<usize>>>> = OnceLock::new();
    CONFLICTS
        .get_or_init(|| {
            let combos = all_combos();
            Arc::new(
                combos
                    .iter()
                    .map(|own| {
                        combos
                            .iter()
                            .enumerate()
                            .filter_map(|(index, other)| own.overlaps(*other).then_some(index))
                            .collect()
                    })
                    .collect(),
            )
        })
        .clone()
}

fn compatible_mass_from_conflicts(range: &[f64], conflicts: &[Vec<usize>], own: usize) -> f64 {
    let total = range.iter().sum::<f64>();
    (total
        - conflicts[own]
            .iter()
            .map(|index| range[*index])
            .sum::<f64>())
    .max(0.0)
}

pub fn uniform_range(board: &[u8]) -> Vec<f64> {
    let blocked = board.iter().copied().collect::<BTreeSet<_>>();
    let mut range = all_combos()
        .into_iter()
        .map(|combo| {
            if combo.cards().iter().any(|card| blocked.contains(card)) {
                0.0
            } else {
                1.0
            }
        })
        .collect::<Vec<_>>();
    let total = range.iter().sum::<f64>();
    for weight in &mut range {
        *weight /= total;
    }
    range
}

fn joint_compatibility_mass(ranges: &[Vec<f64>; 2]) -> f64 {
    let conflicts = combo_conflicts();
    let player_one_total = ranges[1].iter().sum::<f64>();
    ranges[0]
        .iter()
        .enumerate()
        .map(|(first, first_weight)| {
            first_weight
                * (player_one_total
                    - conflicts[first]
                        .iter()
                        .map(|second| ranges[1][*second])
                        .sum::<f64>())
                .max(0.0)
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_game() -> BlueprintConfig {
        let mut game = BlueprintConfig::default();
        game.effective_stack_bb = 4.0;
        game.iterations = 10;
        game.averaging_delay = 0;
        game.action_abstraction.turn_river_bet_pot_fractions = vec![1.0];
        game.action_abstraction.postflop_raise_pot_fractions = vec![1.0];
        game
    }

    fn zero_value_network() -> PublicValueNetwork {
        let layer = |input_size, output_size, activation: &str| ValueNetworkLayer {
            input_size,
            output_size,
            activation: activation.to_owned(),
            weights: vec![0.0; input_size * output_size],
            biases: vec![0.0; output_size],
            normalization: None,
            normalization_weights: Vec::new(),
            normalization_biases: Vec::new(),
            normalization_epsilon: 0.0,
            residual: false,
        };
        PublicValueNetwork {
            artifact_sha256: None,
            schema: "hu-public-belief-value-network-v2".to_owned(),
            seed: 1,
            uses_exact_ranges: true,
            target_scale_bb: 20.0,
            range_scale: COMBO_COUNT as f64,
            value_normalization: None,
            residual_scale_bb: 0.0,
            source_dataset_sha256: Some("0".repeat(64)),
            source_policy_sha256: None,
            source_validation_status: Some("accepted".to_owned()),
            feature_schema: None,
            context_public_count: 0,
            context_size: 0,
            query_structural_count: 0,
            query_size: 0,
            public_tower: vec![layer(56, 1, "linear")],
            range_tower: vec![layer(COMBO_COUNT * 2, 1, "linear")],
            context_tower: Vec::new(),
            query_tower: Vec::new(),
            head: vec![layer(2, COMBO_COUNT * 2, "tanh")],
        }
    }

    fn zero_shared_value_network() -> PublicValueNetwork {
        let layer = |input_size, output_size, activation: &str| ValueNetworkLayer {
            input_size,
            output_size,
            activation: activation.to_owned(),
            weights: vec![0.0; input_size * output_size],
            biases: vec![0.0; output_size],
            normalization: None,
            normalization_weights: Vec::new(),
            normalization_biases: Vec::new(),
            normalization_epsilon: 0.0,
            residual: false,
        };
        PublicValueNetwork {
            artifact_sha256: None,
            schema: "hu-public-belief-combo-value-network-v3".to_owned(),
            seed: 2,
            uses_exact_ranges: true,
            target_scale_bb: 20.0,
            range_scale: COMBO_COUNT as f64,
            value_normalization: None,
            residual_scale_bb: 5.0,
            source_dataset_sha256: Some("1".repeat(64)),
            source_policy_sha256: None,
            source_validation_status: Some("rejected".to_owned()),
            feature_schema: Some("rank-suit-invariant-combo-query-v1".to_owned()),
            context_public_count: SHARED_CONTEXT_PUBLIC_COUNT,
            context_size: SHARED_CONTEXT_COUNT,
            query_structural_count: SHARED_QUERY_STRUCTURAL_COUNT,
            query_size: SHARED_QUERY_COUNT,
            public_tower: Vec::new(),
            range_tower: Vec::new(),
            context_tower: vec![layer(SHARED_CONTEXT_COUNT, 1, "linear")],
            query_tower: vec![layer(SHARED_QUERY_COUNT, 1, "linear")],
            head: vec![layer(2, 1, "tanh")],
        }
    }

    fn check_preferring_range_policy() -> RangeConditionedPolicyNetwork {
        let zero_layer = |input_size, output_size| ValueNetworkLayer {
            input_size,
            output_size,
            activation: "linear".to_owned(),
            weights: vec![0.0; input_size * output_size],
            biases: vec![0.0; output_size],
            normalization: None,
            normalization_weights: Vec::new(),
            normalization_biases: Vec::new(),
            normalization_epsilon: 0.0,
            residual: false,
        };
        let mut action = zero_layer(ACTION_FEATURE_COUNT, 1);
        action.weights[1] = 1.0;
        let mut head = zero_layer(5, 1);
        head.weights[4] = 1.0;
        RangeConditionedPolicyNetwork {
            artifact_sha256: None,
            schema: RANGE_POLICY_SCHEMA_V1.to_owned(),
            seed: 71,
            depth_bb: 20.0,
            uses_exact_ranges: true,
            feature_schema: RANGE_POLICY_FEATURE_SCHEMA_V1.to_owned(),
            context_size: SHARED_CONTEXT_BOARD_RELATIVE_COUNT,
            query_size: SHARED_QUERY_BOARD_RELATIVE_COUNT,
            action_feature_schema: ACTION_FEATURE_SCHEMA_V1.to_owned(),
            action_feature_size: ACTION_FEATURE_COUNT,
            context_tower: vec![zero_layer(SHARED_CONTEXT_BOARD_RELATIVE_COUNT, 1)],
            query_tower: vec![zero_layer(SHARED_QUERY_BOARD_RELATIVE_COUNT, 1)],
            action_tower: vec![action],
            head: vec![head],
            source_dataset_sha256: "1".repeat(64),
            source_dataset_schema: "hu-range-conditioned-postflop-policy-dataset-v1".to_owned(),
            source_validation_status: "accepted_for_training".to_owned(),
            policy_composition: RANGE_POLICY_REPLACE.to_owned(),
            source_policy_sha256: None,
        }
    }

    #[test]
    fn range_policy_v2_encodes_markov_state_and_perfect_recall_trajectory() {
        let board = [0, 5, 10];
        let ranges = std::array::from_fn(|_| uniform_range(&board));
        let mut state = PublicBeliefState::flop_start(board, 0, [4.0, 4.0], ranges);
        state.checks = 1;
        state.trajectory = vec![TrajectoryAction {
            actor: 1,
            street: Street::Flop,
            kind: TrajectoryActionKind::Check,
            amount_bb: 0.0,
            amount_to_bb: None,
            pot_after_bb: 8.0,
        }];
        let features = range_policy_state_features(&state, 20.0).unwrap();
        assert_eq!(features.len(), RANGE_POLICY_PUBLIC_STATE_COUNT);
        assert_eq!(features[1], 1.0);
        assert_eq!(features[4], 1.0);
        assert!((features[6] - 0.4).abs() < 1e-7);
        assert_eq!(features[19], 1.0);
        assert_eq!(features[21], 1.0);
        assert_eq!(features[23], 1.0);
        assert_eq!(features[27], 1.0);
        assert!((features[34] - 0.4).abs() < 1e-7);

        let mut root = state;
        root.checks = 0;
        root.trajectory.clear();
        assert_ne!(features, range_policy_state_features(&root, 20.0).unwrap());
    }

    #[test]
    fn range_policy_scores_zero_reach_forced_deviation_combos_and_normalizes() {
        let mut game = BlueprintConfig::default();
        game.effective_stack_bb = 20.0;
        let board = [0, 5, 10];
        let mut ranges = std::array::from_fn(|_| uniform_range(&board));
        let forced_combo = all_combos()
            .into_iter()
            .find(|combo| !combo.cards().iter().any(|card| board.contains(card)))
            .unwrap();
        ranges[1][forced_combo.key()] = 0.0;
        let state = PublicBeliefState::flop_start(board, 1, [1.0, 1.0], ranges);
        let network = check_preferring_range_policy();
        network.validate().unwrap();
        let actions = state.game_state().legal_actions(&game);
        let probabilities = network.strategy(&state, &game, None).unwrap();
        let row = &probabilities
            [forced_combo.key() * actions.len()..(forced_combo.key() + 1) * actions.len()];
        assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let expected_check =
            std::f64::consts::E / (std::f64::consts::E + actions.len() as f64 - 1.0);
        assert!((row[0] - expected_check).abs() < 1e-6);
        for combo in all_combos() {
            let row =
                &probabilities[combo.key() * actions.len()..(combo.key() + 1) * actions.len()];
            if combo.cards().iter().any(|card| board.contains(card)) {
                assert_eq!(row.iter().sum::<f64>(), 0.0);
            } else {
                assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn zero_residual_range_policy_preserves_the_pinned_source_policy() {
        let mut game = BlueprintConfig::default();
        game.effective_stack_bb = 20.0;
        let board = [0, 5, 10];
        let ranges = std::array::from_fn(|_| uniform_range(&board));
        let state = PublicBeliefState::flop_start(board, 1, [1.0, 1.0], ranges);
        let actions = state.game_state().legal_actions(&game);
        let mut network = check_preferring_range_policy();
        network.policy_composition = RANGE_POLICY_SOURCE_LOGIT_RESIDUAL.to_owned();
        network.source_policy_sha256 = Some("a".repeat(64));
        network.head[0].weights.fill(0.0);
        network.head[0].biases.fill(0.0);
        network.validate().unwrap();
        network
            .validate_source_policy_sha256(&"a".repeat(64))
            .unwrap();
        assert!(network
            .validate_source_policy_sha256(&"b".repeat(64))
            .is_err());
        assert!(network.strategy(&state, &game, None).is_err());

        let mut source = vec![0.0; COMBO_COUNT * actions.len()];
        for combo in all_combos() {
            if combo.cards().iter().any(|card| board.contains(card)) {
                continue;
            }
            let row = &mut source[combo.key() * actions.len()..(combo.key() + 1) * actions.len()];
            row.fill(0.3 / (actions.len() - 1) as f64);
            row[0] = 0.7;
        }
        let probabilities = network.strategy(&state, &game, Some(&source)).unwrap();
        for combo in all_combos() {
            if combo.cards().iter().any(|card| board.contains(card)) {
                continue;
            }
            let offset = combo.key() * actions.len();
            for action in 0..actions.len() {
                assert!((probabilities[offset + action] - source[offset + action]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn value_network_read_records_the_exact_artifact_hash() {
        let network = zero_value_network();
        let bytes = serde_json::to_vec(&network).unwrap();
        let path = std::env::temp_dir().join(format!(
            "public-value-network-hash-{}-{}.json",
            std::process::id(),
            Sha256::digest(&bytes)
                .iter()
                .take(4)
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        ));
        fs::write(&path, &bytes).unwrap();
        let loaded = PublicValueNetwork::read(&path).unwrap();
        fs::remove_file(path).unwrap();
        assert_eq!(
            loaded.artifact_sha256,
            Some(format!("{:x}", Sha256::digest(&bytes)))
        );
    }

    #[test]
    fn independent_value_network_identity_requires_distinct_seed_and_artifact() {
        let mut first = zero_value_network();
        first.artifact_sha256 = Some("a".repeat(64));
        let mut second = first.clone();
        second.seed = 2;
        second.artifact_sha256 = Some("b".repeat(64));
        assert!(first.has_distinct_training_identity(&second));

        second.seed = first.seed;
        assert!(!first.has_distinct_training_identity(&second));
        second.seed = 2;
        second.artifact_sha256 = first.artifact_sha256.clone();
        assert!(!first.has_distinct_training_identity(&second));
    }

    #[test]
    fn v4_value_normalization_matches_training_scales() {
        let mut network = zero_shared_value_network();
        network.schema = "hu-public-belief-combo-value-network-v4".to_owned();
        network.value_normalization = Some("pot".to_owned());
        network.residual_scale_bb = 0.0;
        network.head[0].activation = "linear".to_owned();
        network.validate().unwrap();
        assert_eq!(network.state_value_scale_bb([2.0, 2.0]), 4.0);
        assert_eq!(network.state_value_scale_bb([18.0, 18.0]), 36.0);
        network.value_normalization = Some("payoff-exposure".to_owned());
        assert_eq!(network.state_value_scale_bb([2.0, 2.0]), 20.0);
        assert_eq!(network.state_value_scale_bb([18.0, 18.0]), 20.0);
    }

    #[test]
    fn v5_range_pooling_requires_three_query_embeddings_in_head_context() {
        let layer = |input_size, output_size, activation: &str| ValueNetworkLayer {
            input_size,
            output_size,
            activation: activation.to_owned(),
            weights: vec![0.0; input_size * output_size],
            biases: vec![0.0; output_size],
            normalization: None,
            normalization_weights: Vec::new(),
            normalization_biases: Vec::new(),
            normalization_epsilon: 0.0,
            residual: false,
        };
        let mut network = zero_shared_value_network();
        network.schema = "hu-public-belief-combo-value-network-v5".to_owned();
        network.value_normalization = Some("pot".to_owned());
        network.residual_scale_bb = 0.0;
        assert!(network.validate().is_err());
        network.head = vec![layer(4, 1, "linear")];
        network.validate().unwrap();
    }

    #[test]
    fn observed_turn_leaf_values_exclude_holdings_containing_the_turn() {
        let board = [0, 5, 10];
        let ranges = std::array::from_fn(|_| uniform_range(&board));
        let mut network = zero_value_network();
        network.head[0].biases[..COMBO_COUNT].fill(0.1);
        network.head[0].biases[COMBO_COUNT..].fill(-0.1);
        let turn = 15;
        let conflicts = combo_conflicts();
        let (values, residual) = turn_leaf_card_values(
            &network,
            &[],
            &conflicts,
            &board,
            0,
            [1.0, 1.0],
            &ranges,
            turn,
        )
        .expect("turn is reachable");

        for player_values in &values {
            for combo in all_combos()
                .into_iter()
                .filter(|combo| combo.cards().contains(&turn))
            {
                assert_eq!(player_values[combo.key()], 0.0);
            }
        }
        let compatible = Combo::new(1, 2).key();
        assert!(values[0][compatible] > 0.0);
        assert!(values[1][compatible] < 0.0);
        assert!(residual < 1e-10, "zero-sum residual was {residual}");
    }

    #[test]
    fn turn_leaf_ensemble_averages_models_before_zero_sum_projection() {
        let board = [0, 5, 10];
        let ranges = std::array::from_fn(|_| uniform_range(&board));
        let mut first = zero_value_network();
        first.seed = 41;
        first.artifact_sha256 = Some("a".repeat(64));
        first.head[0].biases[..COMBO_COUNT].fill(0.2);
        first.head[0].biases[COMBO_COUNT..].fill(-0.1);
        let mut second = zero_value_network();
        second.seed = 42;
        second.artifact_sha256 = Some("b".repeat(64));
        second.head[0].biases[..COMBO_COUNT].fill(-0.05);
        second.head[0].biases[COMBO_COUNT..].fill(0.15);
        let conflicts = combo_conflicts();
        let turn = 15;
        let values = |primary: &PublicValueNetwork, auxiliary: &[PublicValueNetwork]| {
            turn_leaf_card_values(
                primary,
                auxiliary,
                &conflicts,
                &board,
                0,
                [1.0, 1.0],
                &ranges,
                turn,
            )
            .unwrap()
            .0
        };
        let first_values = values(&first, &[]);
        let second_values = values(&second, &[]);
        let ensemble_values = values(&first, &[second]);
        for player in 0..2 {
            for combo in 0..COMBO_COUNT {
                let expected = (first_values[player][combo] + second_values[player][combo]) / 2.0;
                assert!((ensemble_values[player][combo] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn epsilon_sampling_strategy_explores_without_changing_probability_mass() {
        let mixed = epsilon_sampling_strategy(&[1.0, 0.0, 0.0, 0.0], 0.2);
        for (measured, expected) in mixed.iter().zip([0.85, 0.05, 0.05, 0.05]) {
            assert!((measured - expected).abs() <= 1e-12);
        }
        assert!((mixed.iter().sum::<f64>() - 1.0).abs() <= f64::EPSILON);
        assert_eq!(
            epsilon_sampling_strategy(&[0.25, 0.75], 0.0),
            vec![0.25, 0.75]
        );
    }

    #[test]
    fn batched_dense_towers_match_scalar_inference() {
        let layer = |input_size: usize, output_size: usize, activation: &str, offset: usize| {
            ValueNetworkLayer {
                input_size,
                output_size,
                activation: activation.to_owned(),
                weights: (0..input_size * output_size)
                    .map(|index| ((index + offset) % 19) as f32 * 0.006 - 0.045)
                    .collect(),
                biases: (0..output_size)
                    .map(|index| ((index + offset) % 5) as f32 * 0.009 - 0.017)
                    .collect(),
                normalization: None,
                normalization_weights: Vec::new(),
                normalization_biases: Vec::new(),
                normalization_epsilon: 0.0,
                residual: false,
            }
        };
        let samples = 4;
        let input_size = 11;
        let input = (0..samples * input_size)
            .map(|index| index as f32 * 0.013 - 0.21)
            .collect::<Vec<_>>();
        let tower = vec![
            layer(input_size, 7, "relu", 6),
            layer(7, 5, "gelu-fast", 7),
            layer(5, 2, "tanh", 8),
        ];
        let scalar = input
            .chunks_exact(input_size)
            .flat_map(|sample| {
                tower
                    .iter()
                    .fold(sample.to_vec(), |values, dense| dense.forward(&values))
            })
            .collect::<Vec<_>>();
        let batched = forward_batch_tower(&tower, &input, samples);
        for (left, right) in scalar.iter().zip(&batched) {
            assert!((left - right).abs() < 1e-6, "{left} != {right}");
        }

        let context = vec![0.13, -0.27, 0.41];
        let head = vec![layer(5, 4, "relu", 9), layer(4, 1, "linear", 10)];
        let scalar = batched
            .chunks_exact(2)
            .flat_map(|query| {
                head.iter().fold(
                    context.iter().chain(query).copied().collect::<Vec<_>>(),
                    |values, dense| dense.forward(&values),
                )
            })
            .collect::<Vec<_>>();
        let batched_head = forward_batch_head(&head, &context, &batched, samples);
        for (left, right) in scalar.iter().zip(&batched_head) {
            assert!((left - right).abs() < 1e-6, "{left} != {right}");
        }
    }

    #[test]
    fn dense_layer_applies_exported_layer_normalization_before_activation() {
        let layer = ValueNetworkLayer {
            input_size: 2,
            output_size: 2,
            activation: "linear".to_owned(),
            weights: vec![1.0, 0.0, 0.0, 1.0],
            biases: vec![0.0, 0.0],
            normalization: Some("layernorm".to_owned()),
            normalization_weights: vec![2.0, 3.0],
            normalization_biases: vec![0.5, -0.5],
            normalization_epsilon: 1e-5,
            residual: false,
        };
        layer.validate(2).unwrap();
        let measured = layer.forward(&[1.0, 3.0]);
        let inverse = 1.0f32 / (1.0f32 + 1e-5f32).sqrt();
        let expected = [-inverse * 2.0 + 0.5, inverse * 3.0 - 0.5];
        for (left, right) in measured.iter().zip(expected) {
            assert!((left - right).abs() < 1e-6, "{left} != {right}");
        }

        let mut invalid = layer;
        invalid.normalization_weights.pop();
        assert!(invalid.validate(2).is_err());

        let residual = ValueNetworkLayer {
            input_size: 2,
            output_size: 2,
            activation: "linear".to_owned(),
            weights: vec![1.0, 0.0, 0.0, 1.0],
            biases: vec![0.0, 0.0],
            normalization: None,
            normalization_weights: Vec::new(),
            normalization_biases: Vec::new(),
            normalization_epsilon: 0.0,
            residual: true,
        };
        assert_eq!(residual.forward(&[1.0, 3.0]), vec![2.0, 6.0]);

        let mut mismatched = residual;
        mismatched.output_size = 1;
        mismatched.weights.truncate(2);
        mismatched.biases.truncate(1);
        assert!(mismatched.validate(2).is_err());
    }

    #[test]
    fn river_range_masks_board_cards_and_normalizes() {
        let board = [0, 5, 10, 15, 20];
        let state = PublicBeliefState::river_start(
            board,
            1,
            [1.0, 1.0],
            [vec![1.0; COMBO_COUNT], vec![1.0; COMBO_COUNT]],
        );
        let normalized = state.validate_and_normalize(&tiny_game()).unwrap();
        for combo in all_combos() {
            if combo.cards().iter().any(|card| board.contains(card)) {
                assert_eq!(normalized.ranges[0][combo.key()], 0.0);
                assert_eq!(normalized.ranges[1][combo.key()], 0.0);
            }
        }
        assert!((normalized.ranges[0].iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn card_marginal_terminal_values_match_conflict_enumeration() {
        let board = [0, 5, 10, 15, 20];
        let combos = all_combos();
        let range = combos
            .iter()
            .enumerate()
            .map(|(index, combo)| {
                if combo.cards().iter().any(|card| board.contains(card)) {
                    0.0
                } else {
                    (index % 23 + 1) as f64 / 10_000.0
                }
            })
            .collect::<Vec<_>>();
        let conflicts = combo_conflicts();
        let fast_masses = compatible_masses_from_card_marginals(&combos, &range);
        for own in 0..COMBO_COUNT {
            let reference = compatible_mass_from_conflicts(&range, &conflicts, own);
            assert!((fast_masses[own] - reference).abs() < 1e-11);
        }

        let strengths = combos
            .iter()
            .map(|combo| {
                let mut cards = board.to_vec();
                cards.extend(combo.cards());
                evaluate(&cards)
            })
            .collect::<Vec<_>>();
        let groups = strengths.iter().copied().collect::<BTreeSet<_>>();
        let rank_by_strength = groups
            .iter()
            .copied()
            .enumerate()
            .map(|(rank, strength)| (strength, rank))
            .collect::<BTreeMap<_, _>>();
        let ranks = strengths
            .iter()
            .map(|strength| rank_by_strength[strength])
            .collect::<Vec<_>>();
        let fast = showdown_values_from_card_strength_marginals(
            &combos,
            &ranks,
            groups.len(),
            &range,
            3.0,
            -2.0,
            0.5,
        );
        for (own, combo) in combos.iter().enumerate() {
            if combo.cards().iter().any(|card| board.contains(card)) {
                continue;
            }
            let mut lower = 0.0;
            let mut equal = 0.0;
            let mut higher = 0.0;
            for opponent in 0..COMBO_COUNT {
                if combos[own].overlaps(combos[opponent]) {
                    continue;
                }
                match strengths[opponent].cmp(&strengths[own]) {
                    std::cmp::Ordering::Less => lower += range[opponent],
                    std::cmp::Ordering::Equal => equal += range[opponent],
                    std::cmp::Ordering::Greater => higher += range[opponent],
                }
            }
            let reference = lower * 3.0 + equal * 0.5 + higher * -2.0;
            assert!((fast[own] - reference).abs() < 1e-10);
        }
    }

    #[test]
    fn exact_showdown_values_respect_blockers_and_zero_sum() {
        let board = [0, 5, 10, 15, 20];
        let solution = solve_river(RiverSolveConfig {
            game: tiny_game(),
            state: PublicBeliefState::uniform_river_start(board, 1, [1.0, 1.0]),
            iterations: 8,
            averaging_delay: 0,
        })
        .unwrap();
        assert!(solution.metrics.joint_compatibility_mass > 0.9);
        assert!(
            solution.metrics.zero_sum_residual_bb < 1e-8,
            "zero-sum residual was {}",
            solution.metrics.zero_sum_residual_bb
        );
        assert!(solution.metrics.maximum_probability_sum_error < 1e-6);
        assert!(solution
            .counterfactual_values_bb
            .iter()
            .flatten()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn river_solve_is_deterministic() {
        let config = RiverSolveConfig {
            game: tiny_game(),
            state: PublicBeliefState::uniform_river_start([0, 5, 10, 15, 20], 1, [1.0, 1.0]),
            iterations: 4,
            averaging_delay: 0,
        };
        assert_eq!(solve_river(config.clone()), solve_river(config));
    }

    #[test]
    fn turn_river_solve_keeps_turn_betting_and_observed_river_chance() {
        let board = [0, 5, 10, 15];
        let ranges = std::array::from_fn(|_| uniform_range(&board));
        let config = TurnRiverSolveConfig {
            game: tiny_game(),
            state: PublicBeliefState::turn_start(board, 1, [1.0, 1.0], ranges),
            iterations: 2,
            averaging_delay: 0,
            river_refinement_iterations: 0,
            regret_matching_plus: false,
        };
        let solution = solve_turn_river(config.clone()).unwrap();
        let values = solve_turn_river_continuation_values(config).unwrap();
        assert_eq!(
            values.counterfactual_values_bb,
            solution.counterfactual_values_bb
        );
        assert_eq!(
            values.opponent_compatible_mass,
            solution.opponent_compatible_mass
        );
        assert_eq!(values.metrics, solution.metrics);
        assert_eq!(
            values.schema,
            "hu-turn-river-public-belief-continuation-values-v2"
        );
        assert_eq!(solution.schema, "hu-turn-river-public-belief-solution-v2");
        assert_eq!(solution.metrics.exact_river_cards, 48);
        assert!(solution.metrics.turn_information_sets > 0);
        assert!(solution.metrics.river_information_sets > 0);
        assert_eq!(
            solution.metrics.information_sets,
            solution.metrics.turn_information_sets + solution.metrics.river_information_sets
        );
        assert!(solution.strategies.iter().any(|node| node
            .public_history
            .iter()
            .any(|part| part.starts_with("Turn:p"))));
        assert!(solution.strategies.iter().any(|node| node
            .public_history
            .iter()
            .any(|part| part.starts_with("River:p"))));
        assert!(solution.strategies.iter().any(|node| node
            .public_history
            .iter()
            .any(|part| part.starts_with("chance:river:"))));
        assert!(
            solution.metrics.zero_sum_residual_bb < 1e-8,
            "zero-sum residual was {}",
            solution.metrics.zero_sum_residual_bb
        );
        assert!(solution.metrics.maximum_probability_sum_error < 1e-6);
        assert!(solution
            .metrics
            .current_strategy_exploitability_bb_per_hand
            .is_finite());
        assert!(solution.metrics.current_strategy_exploitability_bb_per_hand >= 0.0);
        assert!(
            solution.metrics.turn_only_best_response_gain_bb_per_hand
                <= solution.metrics.exact_abstract_exploitability_bb_per_hand + 1e-8
        );
        assert!(
            solution.metrics.river_only_best_response_gain_bb_per_hand
                <= solution.metrics.exact_abstract_exploitability_bb_per_hand + 1e-8
        );
        assert!(solution
            .counterfactual_values_bb
            .iter()
            .flatten()
            .all(|value| value.is_finite()));
        assert!(solution.strategies.iter().all(|node| {
            node.action_values_bb.as_ref().is_some_and(|values| {
                values.len() == COMBO_COUNT * node.action_labels.len()
                    && values.iter().all(|value| value.is_finite())
            })
        }));
        let root_history = solution.state.game_state().public_history;
        let root = solution
            .strategies
            .iter()
            .find(|node| node.public_history == root_history)
            .unwrap();
        let action_count = root.action_labels.len();
        let action_values = root.action_values_bb.as_ref().unwrap();
        let joint_mass = joint_compatibility_mass(&solution.state.ranges);
        let reconstructed = (0..COMBO_COUNT)
            .map(|combo| {
                let offset = combo * action_count;
                let row_ev = (0..action_count)
                    .map(|action| {
                        f64::from(root.probabilities[offset + action])
                            * f64::from(action_values[offset + action])
                    })
                    .sum::<f64>();
                solution.state.ranges[root.actor][combo]
                    * f64::from(solution.opponent_compatible_mass[root.actor][combo])
                    * row_ev
            })
            .sum::<f64>()
            / joint_mass;
        let expected = if root.actor == 0 {
            solution.metrics.profile_value_p0_bb
        } else {
            solution.metrics.profile_value_p1_bb
        };
        assert!((reconstructed - expected).abs() < 1e-5);
    }

    #[test]
    fn turn_all_in_card_marginal_values_are_zero_sum() {
        let board = [0, 5, 10, 15];
        let ranges = std::array::from_fn(|_| uniform_range(&board));
        let config = TurnRiverSolveConfig {
            game: tiny_game(),
            state: PublicBeliefState::turn_start(board, 1, [1.0, 1.0], ranges.clone()),
            iterations: 2,
            averaging_delay: 0,
            river_refinement_iterations: 0,
            regret_matching_plus: false,
        };
        let solver = TurnRiverSolver::new(config).unwrap();
        let mut state = solver.config.state.game_state();
        state.terminal = Some(Terminal::Showdown);
        for river in &solver.river_cards {
            let mut masked = ranges.clone();
            for player in 0..2 {
                for combo in &solver.river_blocked_combos[*river as usize] {
                    masked[player][*combo] = 0.0;
                }
            }
            let river_values = solver.river_showdown_values(&state, &masked, *river);
            let river_total = masked[0]
                .iter()
                .zip(&river_values[0])
                .map(|(reach, value)| reach * value)
                .sum::<f64>()
                + masked[1]
                    .iter()
                    .zip(&river_values[1])
                    .map(|(reach, value)| reach * value)
                    .sum::<f64>();
            assert!(
                river_total.abs() < 1e-8,
                "river {river} zero-sum total was {river_total}"
            );
        }
        let values = solver.turn_all_in_values(&state, &ranges);
        let total = ranges[0]
            .iter()
            .zip(&values[0])
            .map(|(reach, value)| reach * value)
            .sum::<f64>()
            + ranges[1]
                .iter()
                .zip(&values[1])
                .map(|(reach, value)| reach * value)
                .sum::<f64>();
        assert!(total.abs() < 1e-8, "zero-sum total was {total}");
    }

    #[test]
    fn observed_river_chance_counts_exactly_44_cards_per_private_pair() {
        let board = [0, 5, 10, 15];
        let ranges = std::array::from_fn(|_| uniform_range(&board));
        let solver = TurnRiverSolver::new(TurnRiverSolveConfig {
            game: tiny_game(),
            state: PublicBeliefState::turn_start(board, 1, [1.0, 1.0], ranges),
            iterations: 2,
            averaging_delay: 0,
            river_refinement_iterations: 0,
            regret_matching_plus: false,
        })
        .unwrap();
        let player_zero = Combo::new(1, 2);
        let player_one = Combo::new(3, 4);
        let mut values = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        for river in &solver.river_cards {
            let mut child = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
            if !player_one.cards().contains(river) {
                child[0][player_zero.key()] = 1.0;
            }
            if !player_zero.cards().contains(river) {
                child[1][player_one.key()] = 1.0;
            }
            solver.accumulate_compatible_river_child(&mut values, &child, *river);
        }
        assert!((values[0][player_zero.key()] - 1.0).abs() < 1e-12);
        assert!((values[1][player_one.key()] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn turn_river_chance_matches_legacy_river_average_when_turn_is_forced_check() {
        let board = [0, 5, 10, 15];
        let ranges = std::array::from_fn(|_| uniform_range(&board));
        let mut game = tiny_game();
        game.effective_stack_bb = 2.0;
        game.action_abstraction.include_all_in = false;
        let invested = [1.999, 1.999];
        let direct = legacy_turn_target_from_exact_rivers(
            &game,
            board,
            1,
            invested,
            ranges.clone(),
            4,
            0,
            1,
            0,
        )
        .unwrap();
        let solved = solve_turn_river(TurnRiverSolveConfig {
            game,
            state: PublicBeliefState::turn_start(board, 1, invested, ranges),
            iterations: 4,
            averaging_delay: 0,
            river_refinement_iterations: 0,
            regret_matching_plus: false,
        })
        .unwrap();
        for player in 0..2 {
            for (left, right) in direct.counterfactual_values_bb[player]
                .iter()
                .zip(&solved.counterfactual_values_bb[player])
            {
                assert!((left - right).abs() < 1e-5, "{left} != {right}");
            }
        }
    }

    #[test]
    fn river_refinement_freezes_the_exported_average_turn_policy() {
        let board = [0, 5, 10, 15];
        let ranges = std::array::from_fn(|_| uniform_range(&board));
        let base = TurnRiverSolveConfig {
            game: tiny_game(),
            state: PublicBeliefState::turn_start(board, 1, [1.0, 1.0], ranges),
            iterations: 2,
            averaging_delay: 0,
            river_refinement_iterations: 0,
            regret_matching_plus: false,
        };
        let joint = solve_turn_river(base.clone()).unwrap();
        let refined = solve_turn_river(TurnRiverSolveConfig {
            river_refinement_iterations: 2,
            ..base
        })
        .unwrap();
        let turn_strategies = |solution: &TurnRiverSolution| {
            solution
                .strategies
                .iter()
                .filter(|strategy| {
                    !strategy
                        .public_history
                        .iter()
                        .any(|part| part.starts_with("chance:river:"))
                })
                .map(|strategy| {
                    (
                        strategy.public_history.clone(),
                        strategy.probabilities.clone(),
                    )
                })
                .collect::<BTreeMap<_, _>>()
        };
        assert_eq!(turn_strategies(&joint), turn_strategies(&refined));
        assert_eq!(refined.river_refinement_iterations, 2);
        assert!(refined
            .method
            .contains("frozen_average_turn_river_refinement"));
    }

    #[test]
    fn turn_target_upgrade_preserves_provenance_and_changes_continuation_semantics() {
        let board = [0, 5, 10, 15];
        let ranges = std::array::from_fn(|_| uniform_range(&board));
        let mut game = tiny_game();
        game.effective_stack_bb = 2.0;
        game.action_abstraction.include_all_in = false;
        let invested = [1.999, 1.999];
        let mut source =
            legacy_turn_target_from_exact_rivers(&game, board, 1, invested, ranges, 2, 0, 1, 7)
                .unwrap();
        source.belief_method = Some("pinned-test-belief".to_owned());
        source.public_action_line = Some(vec!["pinned-action".to_owned()]);
        let upgraded = upgrade_turn_value_target(&game, &source, 2, 0).unwrap();
        assert_eq!(upgraded.state_id, source.state_id);
        assert_eq!(upgraded.belief_method, source.belief_method);
        assert_eq!(upgraded.public_action_line, source.public_action_line);
        assert!(upgraded.input_sha256.is_some());
        assert!(upgraded.turn_river_exploitability_bb_per_hand.is_some());
        assert!(upgraded
            .current_turn_river_exploitability_bb_per_hand
            .is_some());
        assert!(upgraded.turn_river_maximum_probability_sum_error.is_some());
        assert!(upgraded.turn_only_best_response_gain_bb_per_hand.is_some());
        assert!(upgraded.river_only_best_response_gain_bb_per_hand.is_some());
        assert_eq!(
            upgraded.maximum_river_exploitability_bb_per_hand,
            upgraded.turn_river_exploitability_bb_per_hand.unwrap()
        );
    }

    #[test]
    fn stratified_belief_replicates_are_deterministic_and_close_to_exact() {
        let board = [0, 5, 10, 15];
        let exact = std::array::from_fn(|_| uniform_range(&board));
        let mut first_rng = SplitMix64::new(77);
        let mut repeated_rng = SplitMix64::new(77);
        let mut independent_rng = SplitMix64::new(78);
        let (first, first_ess) =
            particle_reach_factors_from_exact(&exact, board, 4_096, &mut first_rng).unwrap();
        let (repeated, repeated_ess) =
            particle_reach_factors_from_exact(&exact, board, 4_096, &mut repeated_rng).unwrap();
        let (independent, _) =
            particle_reach_factors_from_exact(&exact, board, 4_096, &mut independent_rng).unwrap();
        assert_eq!(first, repeated);
        assert_eq!(first_ess, repeated_ess);
        assert!(first_ess > 4_000.0);
        assert!(maximum_range_total_variation(&exact, &first) < 0.15);
        assert!(maximum_range_total_variation(&first, &independent) < 0.15);
    }

    #[test]
    fn full_vector_value_network_masks_illegal_combos_and_is_finite() {
        let board = [0, 5, 10, 15];
        let ranges = std::array::from_fn(|_| uniform_range(&board));
        let network = zero_value_network();
        network.validate().unwrap();
        let values = network.predict(&board, 1, [2.0, 2.0], &ranges);
        assert_eq!(values[0].len(), COMBO_COUNT);
        assert!(values.iter().flatten().all(|value| *value == 0.0));
    }

    #[test]
    fn shared_combo_value_network_masks_illegal_combos_and_is_finite() {
        let board = [0, 5, 10, 15];
        let ranges = std::array::from_fn(|_| uniform_range(&board));
        let network = zero_shared_value_network();
        network.validate().unwrap();
        let values = network.predict(&board, 1, [2.0, 2.0], &ranges);
        assert_eq!(values[0].len(), COMBO_COUNT);
        assert!(values.iter().flatten().all(|value| value.is_finite()));
        for combo in all_combos() {
            if combo.cards().iter().any(|card| board.contains(card)) {
                assert_eq!(values[0][combo.key()], 0.0);
                assert_eq!(values[1][combo.key()], 0.0);
            }
        }
        let conflicts = combo_conflicts();
        let masses: [Vec<f64>; 2] = std::array::from_fn(|player| {
            (0..COMBO_COUNT)
                .map(|combo| compatible_mass_from_conflicts(&ranges[1 - player], &conflicts, combo))
                .collect()
        });
        let joint = joint_compatibility_mass(&ranges);
        let aggregate = |player: usize| {
            ranges[player]
                .iter()
                .zip(&values[player])
                .zip(&masses[player])
                .map(|((reach, value), mass)| reach * value * mass)
                .sum::<f64>()
                / joint
        };
        assert!((aggregate(0) + aggregate(1)).abs() < 1e-8);
    }

    #[test]
    fn shared_value_network_scores_zero_reach_board_legal_deviations() {
        let board = [0, 5, 10, 15];
        let mut ranges = std::array::from_fn(|_| uniform_range(&board));
        let deviation = Combo::new(47, 51).key();
        ranges[0][deviation] = 0.0;
        let network = zero_shared_value_network();
        let values = network.predict(&board, 1, [1.0, 3.0], &ranges);
        assert!(values[0][deviation].abs() > 1e-6);
    }

    #[test]
    fn shared_combo_features_are_exactly_suit_equivariant() {
        let board = [0u8, 5, 10, 15];
        let ranges = [shaped_range(&board, 3, 0), shaped_range(&board, 3, 1)];
        let permutation = [2u8, 0, 3, 1];
        let permute_card = |card: u8| (card >> 2) * 4 + permutation[(card & 3) as usize];
        let permuted_board = board.map(permute_card);
        let mut permuted_ranges = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        let mut mapping = vec![0usize; COMBO_COUNT];
        for combo in all_combos() {
            let [first, second] = combo.cards();
            let permuted = Combo::new(permute_card(first), permute_card(second));
            mapping[combo.key()] = permuted.key();
            for player in 0..2 {
                permuted_ranges[player][permuted.key()] = ranges[player][combo.key()];
            }
        }
        let conflicts = combo_conflicts();
        let board_queries = board_query_features(&board);
        let permuted_board_queries = board_query_features(&permuted_board);
        for combo in 0..COMBO_COUNT {
            for (left, right) in board_queries.queries[combo]
                .iter()
                .zip(&permuted_board_queries.queries[mapping[combo]])
            {
                assert!((left - right).abs() < 1e-6);
            }
        }
        for schema in [
            SHARED_FEATURE_SCHEMA_V1,
            SHARED_FEATURE_SCHEMA_V2,
            SHARED_FEATURE_SCHEMA_V3,
        ] {
            let (contexts, queries) =
                shared_combo_features(&board, 1, [3.0, 4.0], &ranges, &conflicts, 20.0, schema);
            let (permuted_contexts, permuted_queries) = shared_combo_features(
                &permuted_board,
                1,
                [3.0, 4.0],
                &permuted_ranges,
                &conflicts,
                20.0,
                schema,
            );
            let (context_size, query_size) = shared_feature_sizes(schema).unwrap();
            assert!(contexts.iter().all(|context| context.len() == context_size));
            assert!(queries
                .iter()
                .flatten()
                .all(|query| query.len() == query_size));
            for player in 0..2 {
                for (left, right) in contexts[player].iter().zip(&permuted_contexts[player]) {
                    assert!((left - right).abs() < 1e-6);
                }
                for combo in 0..COMBO_COUNT {
                    for (left, right) in queries[player][combo]
                        .iter()
                        .zip(&permuted_queries[player][mapping[combo]])
                    {
                        assert!((left - right).abs() < 1e-6);
                    }
                }
            }
        }
    }

    #[test]
    fn exact_turn_equity_cache_key_is_suit_canonical() {
        let board = [0u8, 20, 40, 47];
        let permutation = [2u8, 0, 3, 1];
        let permuted = board.map(|card| permute_card_suit(card, permutation));
        let (canonical, canonical_permutation) = canonical_turn_board_suits(board);
        let (permuted_canonical, _) = canonical_turn_board_suits(permuted);
        assert_eq!(canonical, permuted_canonical);
        let mut mapped = board.map(|card| permute_card_suit(card, canonical_permutation));
        mapped.sort_unstable();
        assert_eq!(mapped, canonical);

        let monotone_flop = [0u8, 20, 40];
        let canonical_turns = (0..52u8)
            .filter(|card| !monotone_flop.contains(card))
            .map(|card| {
                canonical_turn_board_suits([
                    monotone_flop[0],
                    monotone_flop[1],
                    monotone_flop[2],
                    card,
                ])
                .0
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(canonical_turns.len(), 23);
    }

    #[test]
    fn dense_turn_equity_cache_evicts_the_least_recently_used_board() {
        let mut cache = DenseTurnEquityCache::default();
        let keys = (0..=DENSE_TURN_EQUITY_CACHE_BOARDS)
            .map(|index| [0, 1, (index / 52) as u8 + 2, (index % 52) as u8])
            .collect::<Vec<_>>();
        for key in &keys[..DENSE_TURN_EQUITY_CACHE_BOARDS] {
            cache.cell(*key);
        }
        cache.cell(keys[0]);
        cache.cell(keys[DENSE_TURN_EQUITY_CACHE_BOARDS]);
        assert!(cache.entries.contains_key(&keys[0]));
        assert!(!cache.entries.contains_key(&keys[1]));
        assert_eq!(cache.entries.len(), DENSE_TURN_EQUITY_CACHE_BOARDS);
    }

    #[test]
    fn flop_resolver_evaluates_all_in_runouts_exactly() {
        let board = [0, 5, 10];
        let first = Combo::new(47, 51);
        let second = Combo::new(4, 9);
        let mut sparse = vec![0.0; COMBO_COUNT];
        sparse[first.key()] = 0.5;
        sparse[second.key()] = 0.5;
        let ranges = [sparse.clone(), sparse];
        let config = FlopResolveConfig {
            game: tiny_game(),
            state: PublicBeliefState::flop_start(board, 1, [1.0, 1.0], ranges),
            iterations: 2,
            averaging_delay: 0,
            regret_matching_plus: false,
            value_network: zero_value_network(),
            auxiliary_value_networks: Vec::new(),
            threads: 1,
        };
        let continuation = solve_flop_continuation_values(config.clone()).unwrap();
        let solution = solve_flop(config).unwrap();
        assert_eq!(
            continuation.counterfactual_values_bb,
            solution.counterfactual_values_bb
        );
        assert!(continuation.exact_all_in_terminal_evaluations > 0);
        assert!(continuation.zero_sum_residual_after_projection_bb < 1e-8);
        assert!(solution.metrics.exact_all_in_terminal_evaluations > 0);
        assert!(solution
            .strategies
            .iter()
            .flat_map(|node| &node.action_labels)
            .any(|label| label.contains("all_in")));
        assert!(solution.strategies.iter().all(|node| {
            node.action_values_bb.as_ref().is_some_and(|values| {
                values.len() == COMBO_COUNT * node.action_labels.len()
                    && values.iter().all(|value| value.is_finite())
            })
        }));
        assert!(solution.metrics.zero_sum_residual_after_projection_bb < 1e-8);
        assert!(!solution
            .validation
            .reasons
            .iter()
            .any(|reason| reason.contains("all-in")));
    }

    #[test]
    fn cross_evaluated_flop_resolver_freezes_strategy_and_records_both_models() {
        let board = [0, 5, 10];
        let ranges = std::array::from_fn(|_| uniform_range(&board));
        let mut resolver_network = zero_value_network();
        resolver_network.seed = 41;
        resolver_network.artifact_sha256 = Some("c".repeat(64));
        resolver_network.source_dataset_sha256 = Some("a".repeat(64));
        let config = FlopResolveConfig {
            game: tiny_game(),
            state: PublicBeliefState::flop_start(board, 1, [1.0, 1.0], ranges),
            iterations: 2,
            averaging_delay: 0,
            regret_matching_plus: false,
            value_network: resolver_network.clone(),
            auxiliary_value_networks: Vec::new(),
            threads: 1,
        };
        let ordinary = solve_flop(config.clone()).unwrap();
        let mut evaluation_network = resolver_network;
        evaluation_network.seed = 42;
        evaluation_network.artifact_sha256 = Some("d".repeat(64));
        evaluation_network.source_dataset_sha256 = Some("b".repeat(64));
        let cross = solve_flop_cross_evaluated(config, evaluation_network.clone()).unwrap();
        let rescored =
            evaluate_frozen_flop_solution(tiny_game(), &ordinary, evaluation_network, 1).unwrap();
        assert_eq!(cross.strategies, ordinary.strategies);
        assert_eq!(rescored.strategies, ordinary.strategies);
        assert_eq!(cross.value_network_seed, 41);
        assert_eq!(cross.effective_stack_bb, 4.0);
        assert_eq!(cross.averaging_delay, 0);
        assert_eq!(cross.threads, 1);
        assert_eq!(rescored.effective_stack_bb, ordinary.effective_stack_bb);
        assert_eq!(rescored.averaging_delay, ordinary.averaging_delay);
        assert_eq!(cross.value_network_sha256, Some("c".repeat(64)));
        assert_eq!(
            cross.value_network_source_dataset_sha256,
            Some("a".repeat(64))
        );
        assert_eq!(cross.evaluation_value_network_seed, Some(42));
        assert_eq!(cross.evaluation_value_network_sha256, Some("d".repeat(64)));
        assert_eq!(
            cross.evaluation_value_network_source_dataset_sha256,
            Some("b".repeat(64))
        );
        assert!(cross.method.contains("scored_by_independent"));
        assert!(cross.evaluation_has_distinct_training_identity);
        assert!(rescored.method.contains("serialized_frozen"));
        assert_eq!(rescored.value_network_sha256, Some("c".repeat(64)));
        assert_eq!(
            rescored.evaluation_value_network_sha256,
            Some("d".repeat(64))
        );
        assert!(
            (cross.metrics.depth_limited_exploitability_bb_per_hand
                - ordinary.metrics.depth_limited_exploitability_bb_per_hand)
                .abs()
                < 1e-10
        );
        assert!(
            (rescored.metrics.depth_limited_exploitability_bb_per_hand
                - cross.metrics.depth_limited_exploitability_bb_per_hand)
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn flop_convergence_diagnostic_reuses_training_across_checkpoints() {
        let board = [0, 5, 10];
        let ranges = std::array::from_fn(|_| uniform_range(&board));
        let mut resolver_network = zero_value_network();
        resolver_network.seed = 41;
        resolver_network.artifact_sha256 = Some("c".repeat(64));
        resolver_network.source_dataset_sha256 = Some("a".repeat(64));
        let mut game = tiny_game();
        game.dcfr.strategy_exponent = 4.0;
        let config = FlopResolveConfig {
            game,
            state: PublicBeliefState::flop_start(board, 1, [1.0, 1.0], ranges),
            iterations: 4,
            averaging_delay: 0,
            regret_matching_plus: false,
            value_network: resolver_network,
            auxiliary_value_networks: Vec::new(),
            threads: 1,
        };
        let mut evaluation_network = zero_value_network();
        evaluation_network.seed = 42;
        evaluation_network.artifact_sha256 = Some("d".repeat(64));
        evaluation_network.source_dataset_sha256 = Some("b".repeat(64));

        let report = diagnose_flop_cross_evaluated_convergence(
            config.clone(),
            evaluation_network.clone(),
            &[2, 4],
        )
        .unwrap();
        let direct =
            solve_flop_cross_evaluated(config.clone(), evaluation_network.clone()).unwrap();

        assert_eq!(report.schema, "hu-flop-resolver-convergence-diagnostic-v3");
        assert!(!report.regret_matching_plus);
        assert_eq!(report.dcfr.strategy_exponent, 4.0);
        assert_eq!(report.final_solution.dcfr, report.dcfr);
        assert_eq!(report.checkpoints.len(), 2);
        assert_eq!(report.checkpoint_solutions.len(), 2);
        assert_eq!(report.checkpoints[0].iterations, 2);
        assert_eq!(report.checkpoints[1].iterations, 4);
        assert_eq!(report.value_network_seed, 41);
        assert_eq!(report.evaluation_value_network_seed, 42);
        assert!(report.evaluation_has_distinct_training_identity);
        assert!(report
            .checkpoint_solutions
            .iter()
            .all(|solution| solution.evaluation_has_distinct_training_identity));
        assert_eq!(report.final_solution.strategies, direct.strategies);
        assert_eq!(report.solution_at_iterations(Some(4)).unwrap(), direct);
        assert_eq!(
            report.solution_at_iterations(Some(2)).unwrap().iterations,
            2
        );
        assert!(report.solution_at_iterations(Some(3)).is_err());
        assert_eq!(
            report.final_strategy_sha256,
            flop_strategy_sha256(&direct.strategies)
        );
        assert_eq!(report.final_solution.value_network_seed, 41);
        assert_eq!(
            report.final_solution.evaluation_value_network_seed,
            Some(42)
        );
        assert_eq!(
            report.final_solution.value_network_sha256,
            Some("c".repeat(64))
        );
        assert_eq!(
            report.final_solution.evaluation_value_network_sha256,
            Some("d".repeat(64))
        );
        assert!(
            (report.checkpoints[1]
                .metrics
                .depth_limited_exploitability_bb_per_hand
                - direct.metrics.depth_limited_exploitability_bb_per_hand)
                .abs()
                < 1e-10
        );
        let mut plus_config = config.clone();
        plus_config.regret_matching_plus = true;
        let plus = diagnose_flop_cross_evaluated_convergence(
            plus_config,
            evaluation_network.clone(),
            &[2, 4],
        )
        .unwrap();
        assert!(plus.regret_matching_plus);
        assert!(plus.method.ends_with("_regret_matching_plus"));
        assert!(plus.final_solution.regret_matching_plus);
        assert!(plus
            .final_solution
            .method
            .ends_with("_regret_matching_plus"));
        assert!(
            diagnose_flop_cross_evaluated_convergence(config, evaluation_network, &[4, 2],)
                .is_err()
        );
    }

    #[test]
    fn flop_range_response_keeps_the_opponent_frozen_and_ranges_consistent() {
        let board = [0, 5, 10];
        let ranges = std::array::from_fn(|_| uniform_range(&board));
        let mut resolver_network = zero_value_network();
        resolver_network.seed = 41;
        resolver_network.artifact_sha256 = Some("c".repeat(64));
        resolver_network.source_dataset_sha256 = Some("a".repeat(64));
        let config = FlopResolveConfig {
            game: tiny_game(),
            state: PublicBeliefState::flop_start(board, 1, [1.0, 1.0], ranges),
            iterations: 4,
            averaging_delay: 0,
            regret_matching_plus: false,
            value_network: resolver_network,
            auxiliary_value_networks: Vec::new(),
            threads: 1,
        };
        let frozen = solve_flop(config.clone()).unwrap();
        let mut game = config.game.clone();
        game.dcfr.strategy_exponent = 3.0;
        let mut evaluation_network = zero_value_network();
        evaluation_network.seed = 42;
        evaluation_network.artifact_sha256 = Some("d".repeat(64));
        evaluation_network.source_dataset_sha256 = Some("b".repeat(64));

        let report = evaluate_frozen_flop_range_response_convergence(
            game.clone(),
            &frozen,
            evaluation_network.clone(),
            &[2, 4],
            0,
            false,
            1,
        )
        .unwrap();
        let repeated = evaluate_frozen_flop_range_response_convergence(
            game.clone(),
            &frozen,
            evaluation_network.clone(),
            &[2, 4],
            0,
            false,
            1,
        )
        .unwrap();

        assert_eq!(report, repeated);
        assert_eq!(report.schema, "hu-flop-range-response-diagnostic-v1");
        assert_eq!(report.response_dcfr.strategy_exponent, 3.0);
        assert!(report.evaluation_has_distinct_training_identity);
        assert_eq!(report.checkpoints.len(), 2);
        assert_eq!(report.frozen_strategy_iterations, 4);
        assert_eq!(
            report.frozen_strategy_sha256,
            flop_strategy_sha256(&frozen.strategies)
        );
        assert_eq!(report.checkpoints[1].iterations, 4);
        assert!(report.checkpoints.iter().all(|checkpoint| {
            checkpoint.response_gain_p0_bb >= 0.0
                && checkpoint.response_gain_p1_bb >= 0.0
                && checkpoint.range_consistent_response_gain_bb_per_hand >= 0.0
        }));
        for responder in 0..2 {
            assert!(!report.final_response_strategies[responder].is_empty());
            assert!(report.final_response_strategies[responder]
                .iter()
                .all(|strategy| strategy.actor == responder));
            assert!(!report.information_set_attribution[responder].is_empty());
            assert!(report.information_set_attribution[responder]
                .iter()
                .all(|node| {
                    node.actor == responder
                        && node.node_reach_probability > 0.0
                        && node.node_reach_probability <= 1.0 + EPSILON
                        && (0.0..=1.0).contains(&node.reach_weighted_combo_policy_total_variation)
                        && (0.0..=1.0).contains(&node.reach_weighted_primary_action_agreement)
                        && (0.0..=1.0).contains(&node.maximum_combo_total_variation)
                        && (node.frozen_action_frequencies.iter().sum::<f64>() - 1.0).abs() < 1e-6
                        && (node.response_action_frequencies.iter().sum::<f64>() - 1.0).abs() < 1e-6
                        && node.action_frequency_deltas.iter().sum::<f64>().abs() < 1e-6
                        && node.conditional_action_ev_bb.len() == node.action_labels.len()
                        && node
                            .conditional_action_ev_bb
                            .iter()
                            .all(|value| value.is_finite())
                        && node.conditional_frozen_strategy_ev_bb.is_finite()
                        && node.conditional_response_strategy_ev_bb.is_finite()
                        && node.conditional_best_action_ev_bb.is_finite()
                        && node.conditional_frozen_strategy_ev_loss_bb >= 0.0
                        && node.conditional_response_strategy_ev_loss_bb >= 0.0
                        && ((node.conditional_best_action_ev_bb
                            - node.conditional_frozen_strategy_ev_bb)
                            .max(0.0)
                            - node.conditional_frozen_strategy_ev_loss_bb)
                            .abs()
                            < 1e-9
                        && ((node.conditional_best_action_ev_bb
                            - node.conditional_response_strategy_ev_bb)
                            .max(0.0)
                            - node.conditional_response_strategy_ev_loss_bb)
                            .abs()
                            < 1e-9
                        && node.top_combo_deviations.iter().all(|combo| {
                            combo.action_ev_bb.len() == node.action_labels.len()
                                && combo.action_ev_bb.iter().all(|value| value.is_finite())
                                && combo.frozen_strategy_ev_bb.is_finite()
                                && combo.response_strategy_ev_bb.is_finite()
                                && combo.best_action_ev_bb.is_finite()
                                && combo.frozen_ev_loss_bb >= 0.0
                                && combo.response_ev_loss_bb >= 0.0
                                && ((combo.best_action_ev_bb - combo.frozen_strategy_ev_bb)
                                    .max(0.0)
                                    - combo.frozen_ev_loss_bb)
                                    .abs()
                                    < 1e-9
                                && ((combo.best_action_ev_bb - combo.response_strategy_ev_bb)
                                    .max(0.0)
                                    - combo.response_ev_loss_bb)
                                    .abs()
                                    < 1e-9
                        })
                }));
            assert!(report.information_set_attribution[responder]
                .windows(2)
                .all(|pair| {
                    pair[0].node_reach_probability
                        * pair[0].reach_weighted_combo_policy_total_variation
                        >= pair[1].node_reach_probability
                            * pair[1].reach_weighted_combo_policy_total_variation
                }));
        }
        assert!(report
            .interpretation
            .contains("not an exploitability upper bound"));
        assert!(!report.response_regret_matching_plus);
        let strengthened = evaluate_frozen_flop_range_response_convergence(
            game.clone(),
            &frozen,
            evaluation_network.clone(),
            &[2, 4],
            0,
            true,
            1,
        )
        .unwrap();
        assert!(strengthened.response_regret_matching_plus);
        assert!(strengthened.method.ends_with("_regret_matching_plus"));

        let mut rare_frozen = frozen.clone();
        let root_strategy = rare_frozen
            .strategies
            .iter_mut()
            .find(|strategy| {
                strategy.actor == 1
                    && strategy.public_history == ["public_belief:flop_start".to_owned()]
            })
            .expect("frozen strategy has the root node");
        let check = root_strategy
            .action_labels
            .iter()
            .position(|label| label == "check")
            .expect("flop root can check");
        let fallback = (0..root_strategy.action_labels.len())
            .find(|action| *action != check)
            .expect("flop root has another action");
        for row in root_strategy
            .probabilities
            .chunks_mut(root_strategy.action_labels.len())
        {
            if row.iter().sum::<f32>() <= 0.0 {
                continue;
            }
            row.fill(0.0);
            row[check] = 1e-8;
            row[fallback] = 1.0;
        }
        let rare_report = evaluate_frozen_flop_range_response_convergence(
            game.clone(),
            &rare_frozen,
            evaluation_network.clone(),
            &[2],
            0,
            false,
            1,
        )
        .unwrap();
        let rare_node = rare_report.information_set_attribution[0]
            .iter()
            .find(|node| {
                node.public_history
                    == [
                        "public_belief:flop_start".to_owned(),
                        "Flop:p1:check".to_owned(),
                    ]
            })
            .unwrap_or_else(|| {
                panic!(
                    "positive rare branch remains attributed: {:?}",
                    rare_report.information_set_attribution[0]
                        .iter()
                        .map(|node| (&node.public_history, node.node_reach_probability))
                        .collect::<Vec<_>>()
                )
            });
        assert!(rare_node.node_reach_probability < 1e-7);
        assert!((rare_node.frozen_action_frequencies.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        assert!((rare_node.response_action_frequencies.iter().sum::<f64>() - 1.0).abs() < 1e-6);

        assert!(evaluate_frozen_flop_range_response_convergence(
            game,
            &frozen,
            evaluation_network,
            &[4, 2],
            0,
            false,
            1,
        )
        .is_err());
    }

    #[test]
    fn flop_strategy_hash_has_a_cross_language_binary_fixture() {
        let strategies = vec![PublicBeliefStrategy {
            public_history: vec!["root".to_owned(), "check".to_owned()],
            actor: 1,
            action_labels: vec!["fold".to_owned(), "call".to_owned()],
            probabilities: vec![0.25, 0.75],
            action_values_bb: None,
        }];
        assert_eq!(
            flop_strategy_sha256(&strategies),
            "caa0399fc945c99975cf5d3466dcd84f395f1fa19d9149622efd07e567e75983"
        );
    }

    #[test]
    fn resolver_leaf_capture_uses_frozen_average_reach_and_exact_turn_blockers() {
        let board = [0, 5, 10];
        let ranges = std::array::from_fn(|_| uniform_range(&board));
        let mut game = tiny_game();
        game.action_abstraction.include_all_in = false;
        let mut solver = FlopSolver::new(FlopResolveConfig {
            game,
            state: PublicBeliefState::flop_start(board, 1, [1.0, 1.0], ranges),
            iterations: 4,
            averaging_delay: 0,
            regret_matching_plus: false,
            value_network: zero_shared_value_network(),
            auxiliary_value_networks: Vec::new(),
            threads: 1,
        })
        .unwrap();
        solver.train();
        let leaves = solver.capture_average_turn_leaves();
        assert!(leaves.len() > 49);
        let probability_mass = leaves
            .iter()
            .map(|leaf| leaf.reach_probability)
            .sum::<f64>();
        assert!(probability_mass > 0.0 && probability_mass <= 1.0 + 1e-9);
        for leaf in &leaves {
            assert_eq!(leaf.root_board, board);
            assert!(leaf.reach_probability.is_finite() && leaf.reach_probability > 0.0);
            for player in 0..2 {
                assert!((leaf.ranges[player].iter().sum::<f64>() - 1.0).abs() < 1e-9);
                for combo in all_combos() {
                    if combo.cards().iter().any(|card| leaf.board.contains(card)) {
                        assert_eq!(leaf.ranges[player][combo.key()], 0.0);
                    }
                }
            }
        }
        let first = sample_resolver_turn_leaves(&leaves, 3, 91).unwrap();
        let repeated = sample_resolver_turn_leaves(&leaves, 3, 91).unwrap();
        assert_eq!(
            first.iter().map(|leaf| leaf.board).collect::<Vec<_>>(),
            repeated.iter().map(|leaf| leaf.board).collect::<Vec<_>>()
        );
        assert_eq!(
            first
                .iter()
                .map(|leaf| &leaf.public_history)
                .collect::<Vec<_>>(),
            repeated
                .iter()
                .map(|leaf| &leaf.public_history)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            first
                .iter()
                .map(|leaf| leaf.board)
                .collect::<BTreeSet<_>>()
                .len(),
            3
        );
        let canonical = canonicalize_resolver_turn_leaf(leaves[0].clone()).unwrap();
        assert_eq!(
            canonical.reach_probability,
            (leaves[0].reach_probability * RESOLVER_REACH_CANONICAL_SCALE).round()
                / RESOLVER_REACH_CANONICAL_SCALE
        );
        for player in 0..2 {
            for weight in &canonical.ranges[player] {
                assert_eq!(*weight, (*weight as f32) as f64);
            }
        }
        let mut perturbed = canonical.clone();
        perturbed.reach_probability += 1e-12;
        let weight = perturbed.ranges[0]
            .iter_mut()
            .find(|weight| **weight > 1e-6)
            .unwrap();
        *weight += 1e-16;
        let perturbed = canonicalize_resolver_turn_leaf(perturbed).unwrap();
        assert_eq!(canonical.reach_probability, perturbed.reach_probability);
        assert_eq!(canonical.ranges, perturbed.ranges);
    }

    #[test]
    fn resolver_root_checkpoint_validates_provenance_ranges_and_pot_bands() {
        let root_board = [0, 5, 10];
        let investments = [[2.0, 2.0], [4.0, 4.0], [8.0, 8.0]];
        let selected_leaves = [15u8, 16, 17]
            .into_iter()
            .zip(investments)
            .enumerate()
            .map(|(index, (turn, invested))| {
                let board = [root_board[0], root_board[1], root_board[2], turn];
                canonicalize_resolver_turn_leaf(ResolverTurnLeaf {
                    root_board,
                    public_history: vec![format!("leaf-{index}")],
                    trajectory: Vec::new(),
                    board,
                    actor: index % 2,
                    invested,
                    ranges: std::array::from_fn(|_| uniform_range(&board)),
                    reach_probability: (index + 1) as f64 / 10.0,
                })
                .unwrap()
            })
            .collect::<Vec<_>>();
        let checkpoint = ResolverRootLeafCheckpoint {
            schema: RESOLVER_ROOT_CHECKPOINT_SCHEMA.to_owned(),
            input_sha256: "fingerprint".to_owned(),
            source_value_network_sha256: "network".to_owned(),
            root_index: 2,
            root_board,
            leaf_population: 30,
            leaf_probability_mass: 0.9,
            selected_leaves,
        };
        validate_resolver_root_checkpoint(
            &checkpoint,
            "fingerprint",
            "network",
            2,
            root_board,
            3,
            20.0,
        )
        .unwrap();
        let round_trip: ResolverRootLeafCheckpoint =
            serde_json::from_slice(&serde_json::to_vec(&checkpoint).unwrap()).unwrap();
        assert_eq!(checkpoint, round_trip);

        let mut missing_band = checkpoint.clone();
        missing_band.selected_leaves[2].invested = [4.0, 4.0];
        assert!(validate_resolver_root_checkpoint(
            &missing_band,
            "fingerprint",
            "network",
            2,
            root_board,
            3,
            20.0,
        )
        .unwrap_err()
        .contains("pot band"));

        let mut wrong_provenance = checkpoint;
        wrong_provenance.source_value_network_sha256 = "other".to_owned();
        assert!(validate_resolver_root_checkpoint(
            &wrong_provenance,
            "fingerprint",
            "network",
            2,
            root_board,
            3,
            20.0,
        )
        .unwrap_err()
        .contains("provenance"));
    }

    #[test]
    fn parallel_turn_leaf_enumeration_matches_single_thread() {
        let board = [0, 5, 10];
        let ranges = std::array::from_fn(|_| uniform_range(&board));
        let mut game = tiny_game();
        game.action_abstraction.include_all_in = false;
        let config = |threads| FlopResolveConfig {
            game: game.clone(),
            state: PublicBeliefState::flop_start(board, 1, [1.0, 1.0], ranges.clone()),
            iterations: 2,
            averaging_delay: 0,
            regret_matching_plus: false,
            value_network: zero_shared_value_network(),
            auxiliary_value_networks: Vec::new(),
            threads,
        };
        let single_solver = FlopSolver::new(config(1)).unwrap();
        let parallel_solver = FlopSolver::new(config(4)).unwrap();
        let state = single_solver.config.state.game_state();
        let single = single_solver.turn_leaf_values(&state, &ranges);
        let parallel = parallel_solver.turn_leaf_values(&state, &ranges);
        for player in 0..2 {
            for (left, right) in single[player].iter().zip(&parallel[player]) {
                assert!((left - right).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn parallel_self_play_target_solves_match_single_worker_order_and_values() {
        let game = tiny_game();
        let boards = [[0, 5, 10, 15], [1, 6, 11, 16]];
        let prepared = boards
            .into_iter()
            .enumerate()
            .map(|(state_index, board)| PreparedSelfPlayTurnTarget {
                state_index,
                board,
                actor: 1,
                invested: [1.0, 1.0],
                ranges: std::array::from_fn(|_| uniform_range(&board)),
                minimum_ess: 2.0,
                maximum_total_variation: 0.0,
                fingerprint: format!("fingerprint-{state_index}"),
                checkpoint_path: None,
                explorer: None,
                exploration_action_line: None,
            })
            .collect::<Vec<_>>();
        let config = |threads| SelfPlayTurnTargetConfig {
            game: game.clone(),
            states: prepared.len(),
            range_particles: 2,
            river_iterations: 2,
            river_averaging_delay: 0,
            seed: 91,
            threads,
            network_path: PathBuf::new(),
            belief_replicates: 2,
            exploration_probability: 0.0,
            minimum_pot_bb: 0.0,
            checkpoint_dir: None,
        };
        let single = solve_prepared_self_play_turn_targets(&config(1), &prepared).unwrap();
        let parallel = solve_prepared_self_play_turn_targets(&config(2), &prepared).unwrap();
        assert_eq!(single, parallel);
        assert_eq!(
            parallel
                .iter()
                .map(|target| target.state_id.as_str())
                .collect::<Vec<_>>(),
            vec!["turn-pbs-000000", "turn-pbs-000001"]
        );
    }

    #[test]
    fn accepted_turn_target_shards_merge_with_unique_provenance_and_ids() {
        let boards = (0..52u8)
            .flat_map(|a| {
                (a + 1..52).flat_map(move |b| {
                    (b + 1..52).flat_map(move |c| (c + 1..52).map(move |d| [a, b, c, d]))
                })
            })
            .take(64)
            .collect::<Vec<_>>();
        let target = |index: usize, board: [u8; 4]| {
            let legal = all_combos()
                .into_iter()
                .map(|combo| !combo.cards().iter().any(|card| board.contains(card)))
                .collect::<Vec<_>>();
            let legal_count = legal.iter().filter(|value| **value).count() as f32;
            let range = legal
                .iter()
                .map(|value| if *value { 1.0 / legal_count } else { 0.0 })
                .collect::<Vec<_>>();
            TurnValueTarget {
                state_id: format!("component-state-{index}"),
                board,
                actor: index % 2,
                invested_bb: [2.0, 2.0],
                ranges: [range.clone(), range],
                counterfactual_values_bb: [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]],
                opponent_compatible_mass: [vec![1.0; COMBO_COUNT], vec![1.0; COMBO_COUNT]],
                exact_river_cards: 48,
                maximum_river_exploitability_bb_per_hand: 0.01,
                turn_river_exploitability_bb_per_hand: Some(0.01),
                current_turn_river_exploitability_bb_per_hand: Some(0.02),
                turn_river_maximum_probability_sum_error: Some(1e-12),
                turn_only_best_response_gain_bb_per_hand: Some(0.005),
                river_only_best_response_gain_bb_per_hand: Some(0.007),
                turn_river_solver_method: Some(
                    "value_only_paired_alternating_complete_turn_river_betting".to_owned(),
                ),
                turn_river_information_sets: Some(3),
                turn_information_sets: Some(1),
                river_information_sets: Some(2),
                zero_sum_residual_bb: 0.0,
                range_particles: Some(4_096),
                range_replicates: Some(2),
                range_effective_sample_size: Some(1_000.0),
                belief_method: Some("exact_per-player_reach_factors_test".to_owned()),
                range_maximum_total_variation: Some(0.1),
                input_sha256: Some(format!("{index:064x}")),
                off_policy_explorer: None,
                sampling_exploration_probability: None,
                public_action_line: Some(vec!["test".to_owned()]),
                resolver_root_board: None,
                resolver_public_history: None,
                resolver_leaf_reach_probability: None,
            }
        };
        let dataset = |seed: u64, targets: Vec<TurnValueTarget>| TurnTargetDataset {
            schema: "hu-turn-public-belief-cfv-dataset-v2".to_owned(),
            method: "test-complete-turn".to_owned(),
            approximate: true,
            game: BlueprintConfig::default(),
            seed,
            river_iterations: 200,
            turn_river_iterations: Some(200),
            turn_river_averaging_delay: Some(20),
            state_distribution: "test-authentic".to_owned(),
            source_policy_sha256: Some("f".repeat(64)),
            sampling_exploration_probability: None,
            exploration_method: None,
            minimum_sampled_pot_bb: None,
            resolver_source_value_network_sha256: None,
            resolver_iterations: None,
            resolver_leaf_population: None,
            resolver_leaf_probability_mass: None,
            component_dataset_sha256: None,
            component_seeds: None,
            component_target_counts: None,
            merge_method: None,
            targets,
            validation: BlueprintValidation {
                status: "accepted".to_owned(),
                reasons: Vec::new(),
            },
        };
        let first = boards[..32]
            .iter()
            .enumerate()
            .map(|(index, board)| target(index, *board))
            .collect();
        let second = boards[32..]
            .iter()
            .enumerate()
            .map(|(offset, board)| target(offset + 32, *board))
            .collect();
        let merged = merge_turn_target_datasets(vec![
            (dataset(101, first), "a".repeat(64)),
            (dataset(102, second), "b".repeat(64)),
        ])
        .unwrap();
        assert_eq!(merged.targets.len(), 64);
        assert_eq!(merged.component_seeds, Some(vec![101, 102]));
        assert_eq!(merged.component_target_counts, Some(vec![32, 32]));
        assert_eq!(merged.targets[0].state_id, "turn-pbs-000000");
        assert_eq!(merged.targets[63].state_id, "turn-pbs-000063");
        assert_eq!(merged.validation.status, "accepted");
    }

    #[test]
    fn action_target_export_preserves_the_live_preflop_trajectory() {
        let mut game = BlueprintConfig::default();
        game.effective_stack_bb = 20.0;
        game.iterations = 2;
        game.averaging_delay = 0;
        let mut state = GameState::initial(&game);
        let limp = state
            .legal_actions(&game)
            .into_iter()
            .find(|action| action.label == "limp")
            .unwrap();
        state = state.apply(&limp, &game);
        let check = state
            .legal_actions(&game)
            .into_iter()
            .find(|action| action.label == "check")
            .unwrap();
        state = state.apply(&check, &game);
        assert_eq!(state.street, Street::Flop);
        assert_eq!(state.trajectory.len(), 2);

        let board = [0, 5, 10];
        let ranges = std::array::from_fn(|_| uniform_range(&board));
        let public = PublicBeliefState::from_game_state(board.to_vec(), &state, ranges.clone());
        let actions = state.legal_actions(&game);
        let action_count = actions.len();
        let strategy = PublicBeliefStrategy {
            public_history: state.public_history.clone(),
            actor: state.actor,
            action_labels: actions.iter().map(|action| action.label.clone()).collect(),
            probabilities: vec![1.0 / action_count as f32; COMBO_COUNT * action_count],
            action_values_bb: Some(vec![0.0; COMBO_COUNT * action_count]),
        };
        let mut records = BoundedActionRecordCollector::new(3, 7);
        let mut range_records = BoundedActionRecordCollector::new(10, 8);
        let added = append_public_belief_policy_records(
            &game,
            None,
            public.game_state(),
            &board,
            ranges,
            &[strategy],
            1.0,
            &mut records,
            Some(&mut range_records),
            true,
        )
        .unwrap();
        assert!(added > 1);
        assert_eq!(records.len(), 1);
        let records = records.into_records();
        let record: serde_json::Value = serde_json::from_slice(&records[0]).unwrap();
        assert_eq!(record["state"]["street"], "flop");
        assert_eq!(record["state"]["trajectory"].as_array().unwrap().len(), 2);
        assert_eq!(
            record["feature_sha256"].as_array().unwrap().len(),
            action_count
        );
        assert_eq!(
            record["action_values_bb"].as_array().unwrap().len(),
            action_count
        );
        assert_eq!(range_records.len(), 1);
        let range_record: RangeConditionedPolicyRecord =
            serde_json::from_slice(&range_records.into_records()[0]).unwrap();
        assert_eq!(range_record.state.street, Street::Flop);
        assert_eq!(range_record.state.trajectory.len(), 2);
        assert_eq!(range_record.action_labels.len(), action_count);
        assert!(range_record
            .action_features
            .iter()
            .all(|features| features.len() == ACTION_FEATURE_COUNT));
        assert_eq!(range_record.probabilities.len(), COMBO_COUNT * action_count);
        for range in &range_record.ranges {
            assert!((range.iter().map(|value| f64::from(*value)).sum::<f64>() - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn postflop_action_target_root_window_uses_absolute_ordinals() {
        assert_eq!(selected_root_ordinal(0, 2, 0).unwrap(), Some(0));
        assert_eq!(selected_root_ordinal(0, 2, 1).unwrap(), Some(1));
        assert_eq!(selected_root_ordinal(0, 2, 2).unwrap(), None);
        assert_eq!(selected_root_ordinal(1, 2, 0).unwrap(), None);
        assert_eq!(selected_root_ordinal(1, 2, 1).unwrap(), Some(0));
        assert_eq!(selected_root_ordinal(1, 2, 2).unwrap(), Some(1));
        assert_eq!(selected_root_ordinal(1, 2, 3).unwrap(), None);
        assert!(selected_root_ordinal(usize::MAX, 2, usize::MAX).is_err());
    }

    #[test]
    fn action_target_export_rejects_missing_action_values() {
        let game = tiny_game();
        let board = [0, 5, 10];
        let ranges = std::array::from_fn(|_| uniform_range(&board));
        let state =
            PublicBeliefState::flop_start(board, 1, [1.0, 1.0], ranges.clone()).game_state();
        let actions = state.legal_actions(&game);
        let strategy = PublicBeliefStrategy {
            public_history: state.public_history.clone(),
            actor: state.actor,
            action_labels: actions.iter().map(|action| action.label.clone()).collect(),
            probabilities: vec![1.0 / actions.len() as f32; COMBO_COUNT * actions.len()],
            action_values_bb: None,
        };
        let mut records = BoundedActionRecordCollector::new(3, 7);
        let error = append_public_belief_policy_records(
            &game,
            None,
            state,
            &board,
            ranges,
            &[strategy],
            1.0,
            &mut records,
            None,
            true,
        )
        .unwrap_err();
        assert_eq!(
            error,
            "solver strategy is missing counterfactual action values"
        );
    }

    #[test]
    fn bounded_action_records_apply_inverse_inclusion_weight_correction() {
        let mut records = BoundedActionRecordCollector::new(3, 19);
        records.consider(Street::Flop, br#"{"weight":1.0,"id":0}"#.to_vec());
        records.consider(Street::Flop, br#"{"weight":1.0,"id":1}"#.to_vec());
        assert_eq!(records.seen, 2);
        assert_eq!(records.len(), 1);
        let selected: serde_json::Value =
            serde_json::from_slice(&records.into_records()[0]).unwrap();
        assert_eq!(selected["weight"], 2.0);
    }

    #[test]
    fn action_target_export_keeps_river_chance_on_descendant_keys() {
        let mut game = BlueprintConfig::default();
        game.effective_stack_bb = 20.0;
        game.iterations = 2;
        game.averaging_delay = 0;
        let board = [20, 25, 30, 35];
        let combos = all_combos();
        let first = combos
            .iter()
            .find(|combo| !combo.cards().iter().any(|card| board.contains(card)))
            .unwrap();
        let second = combos
            .iter()
            .find(|combo| {
                !combo.overlaps(*first) && !combo.cards().iter().any(|card| board.contains(card))
            })
            .unwrap();
        let mut ranges = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        ranges[0][first.key()] = 1.0;
        ranges[1][second.key()] = 1.0;
        let root = PublicBeliefState::turn_start(board, 1, [1.0, 1.0], ranges.clone());
        let root_state = root.game_state();
        let first_check = root_state.legal_actions(&game)[0].clone();
        assert_eq!(first_check.label, "check");
        let checked_once = root_state.apply(&first_check, &game);
        let second_check = checked_once.legal_actions(&game)[0].clone();
        let river_state = checked_once.apply(&second_check, &game);
        assert_eq!(river_state.street, Street::River);
        let river = (0..52u8)
            .find(|card| {
                !board.contains(card)
                    && !first.cards().contains(card)
                    && !second.cards().contains(card)
            })
            .unwrap();
        let river_check = river_state.legal_actions(&game)[0].clone();
        let river_checked_once = river_state.apply(&river_check, &game);
        let make_strategy = |state: &GameState, river: Option<u8>| {
            let actions = state.legal_actions(&game);
            let mut history = state.public_history.clone();
            if let Some(card) = river {
                history.push(format!("chance:river:{card}"));
            }
            PublicBeliefStrategy {
                public_history: history,
                actor: state.actor,
                action_labels: actions.iter().map(|action| action.label.clone()).collect(),
                probabilities: vec![1.0 / actions.len() as f32; COMBO_COUNT * actions.len()],
                action_values_bb: Some(vec![0.0; COMBO_COUNT * actions.len()]),
            }
        };
        let strategies = vec![
            make_strategy(&root_state, None),
            make_strategy(&checked_once, None),
            make_strategy(&river_state, Some(river)),
            make_strategy(&river_checked_once, Some(river)),
        ];
        let mut records = BoundedActionRecordCollector::new(100, 11);
        append_public_belief_policy_records(
            &game,
            None,
            root_state,
            &board,
            ranges,
            &strategies,
            1.0,
            &mut records,
            None,
            true,
        )
        .unwrap();
        let records = records.into_records();
        assert!(records.iter().any(|record| {
            let record: serde_json::Value = serde_json::from_slice(record).unwrap();
            record["state"]["board"].as_array().unwrap().len() == 5
                && record["state"]["trajectory"].as_array().unwrap().len() == 3
        }));
    }
}
