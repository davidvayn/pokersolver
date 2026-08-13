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
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use super::public_belief::{
    PublicBeliefState, RangeConditionedPolicyNetwork, COMBO_COUNT, RANGE_POLICY_CONTEXT_V2_COUNT,
    RANGE_POLICY_FEATURE_SCHEMA_V2,
};

pub const STATE_FEATURE_COUNT: usize = 716;
pub const ACTION_FEATURE_COUNT: usize = 9;
pub const MODEL_INPUT_COUNT: usize = STATE_FEATURE_COUNT + ACTION_FEATURE_COUNT;
pub const MAX_TRAJECTORY_ACTIONS: usize = 32;

const DATASET_SCHEMA: &str = "hu-neural-traversal-jsonl-v7";
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
    pub evaluate_trajectory_values: bool,
    pub value_rollouts_per_action: u32,
    pub enumerate_turn_river_chance: bool,
}

#[derive(Clone, Debug)]
pub struct ExploitabilityCertificateConfig {
    pub game: BlueprintConfig,
    pub deals: u64,
    pub seed: u64,
    pub confidence: f64,
    pub threads: usize,
    pub network_path: PathBuf,
    pub range_policy_path: Option<PathBuf>,
    /// Optional research policy refinement. At every reached river decision,
    /// replace the static network row with an exact-range public-belief CFR
    /// solve rooted at that decision. This changes policy actions, not the
    /// response or confidence-bound evaluator, and remains fail-closed until
    /// an independently measured certificate passes every release gate.
    pub river_resolver: Option<RiverResolverConfig>,
    /// Optional joint turn/river public-belief search. Turn decisions use the
    /// complete exact-river-card subgame; reached river decisions may then use
    /// `river_resolver` as a nested terminal-street solve.
    pub turn_resolver: Option<TurnResolverConfig>,
    /// Optional depth-limited flop public-belief search using a frozen turn
    /// counterfactual-value network at the cutoff.
    pub flop_resolver: Option<FlopResolverConfig>,
}

impl ExploitabilityCertificateConfig {
    fn continual_resolving_enabled(&self) -> bool {
        self.flop_resolver.is_some()
            || self.turn_resolver.is_some()
            || self.river_resolver.is_some()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RiverResolverConfig {
    pub iterations: u64,
    pub averaging_delay: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TurnResolverConfig {
    pub iterations: u64,
    pub averaging_delay: u64,
    pub river_refinement_iterations: u64,
    pub regret_matching_plus: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FlopResolverConfig {
    pub iterations: u64,
    pub averaging_delay: u64,
    pub regret_matching_plus: bool,
    pub threads: usize,
    pub value_network_path: PathBuf,
    pub auxiliary_value_network_paths: Vec<PathBuf>,
    pub continuation_selection: super::public_belief::FlopContinuationSelection,
    /// Fraction of the served action probability contributed by the resolved
    /// strategy. The remainder stays anchored to the frozen blueprint at the
    /// same exact public belief, preventing unconstrained strategy grafting.
    pub resolved_policy_weight: f64,
}

#[derive(Clone, Debug)]
pub struct CausalPolicyAttributionConfig {
    pub game: BlueprintConfig,
    pub deals: u64,
    pub seed: u64,
    pub threads: usize,
    pub network_path: PathBuf,
    pub range_policy_path: Option<PathBuf>,
    pub public_branches_per_street: u32,
    pub opponent_samples_per_runout: u32,
    pub max_records: usize,
    pub output: PathBuf,
}

#[derive(Clone, Debug)]
pub struct RangeSelfPlaySampleConfig {
    pub game: BlueprintConfig,
    pub traversals: u64,
    pub start_iteration: u64,
    pub seed: u64,
    pub max_records: usize,
    pub network_path: PathBuf,
    pub range_policy_path: PathBuf,
    pub value_rollouts_per_action: u32,
    pub enumerate_turn_river_chance: bool,
    pub output: PathBuf,
}

#[derive(Clone, Debug, Serialize)]
pub struct RangeSelfPlaySampleReport {
    pub schema: &'static str,
    pub method: &'static str,
    pub depth_bb: f64,
    pub traversals: u64,
    pub start_iteration: u64,
    pub seed: u64,
    pub network_sha256: String,
    pub range_policy_sha256: String,
    pub value_rollouts_per_action: u32,
    pub candidate_records: usize,
    pub retained_records: usize,
    pub retained_records_by_street: [usize; 3],
    pub truncated: bool,
    pub minimum_policy_action_probability: f64,
    pub maximum_probability_sum_error: f64,
    pub minimum_action_value_bb: f64,
    pub maximum_action_value_bb: f64,
    pub maximum_action_value_standard_error_bb: f64,
    pub output: PathBuf,
    pub output_sha256: String,
    pub validation_status: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct CausalPolicyAttributionReport {
    pub schema: &'static str,
    pub method: &'static str,
    pub depth_bb: f64,
    pub deals: u64,
    pub seed: u64,
    pub network_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub range_policy_sha256: Option<String>,
    pub threads: usize,
    pub public_branches_per_street: u32,
    pub opponent_samples_per_runout: u32,
    pub scenarios_per_deal: u64,
    pub response_tree_nodes: u64,
    pub attribution_tree_nodes: u64,
    pub candidate_records: usize,
    pub retained_records: usize,
    pub candidate_records_by_street: [usize; 3],
    pub retained_records_by_street: [usize; 3],
    pub truncated: bool,
    pub sample_mean_exploitability_bb: f64,
    pub maximum_root_value_reconstruction_error_bb: f64,
    pub minimum_frozen_policy_action_probability: f64,
    pub maximum_target_probability_sum_error: f64,
    pub minimum_policy_action_value_bb: f64,
    pub maximum_policy_action_value_bb: f64,
    pub output: PathBuf,
    pub output_sha256: String,
    pub validation_status: &'static str,
}

#[derive(Clone, Debug)]
pub struct CausalAttributionPolicyEvaluationConfig {
    pub dataset_path: PathBuf,
    pub network_path: PathBuf,
    pub maximum_node_kl: f64,
    pub maximum_weighted_kl: f64,
    pub minimum_policy_value_gain_bb: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct CausalAttributionPolicyEvaluation {
    pub schema: &'static str,
    pub method: &'static str,
    pub depth_bb: f64,
    pub records: usize,
    pub dataset_sha256: String,
    pub source_network_sha256: String,
    pub candidate_network_sha256: String,
    pub total_objective_weight: f64,
    pub weighted_baseline_policy_value_bb: f64,
    pub weighted_candidate_policy_value_bb: f64,
    pub weighted_policy_value_gain_bb: f64,
    pub weighted_reverse_kl_from_frozen: f64,
    pub maximum_reverse_kl_from_frozen: f64,
    pub weighted_forward_kl_from_frozen: f64,
    pub maximum_forward_kl_from_frozen: f64,
    pub weighted_l1_action_delta: f64,
    pub maximum_l1_action_delta: f64,
    pub weighted_primary_action_agreement: f64,
    pub maximum_baseline_probability_sum_error: f64,
    pub maximum_candidate_probability_sum_error: f64,
    pub minimum_policy_value_gain_bb: f64,
    pub feature_hashes_verified: bool,
    pub policy_value_improved: bool,
    pub maximum_node_kl_passed: bool,
    pub weighted_kl_passed: bool,
    pub accepted_for_routed_evaluation: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ExploitabilityCertificate {
    pub schema: &'static str,
    pub method: &'static str,
    pub depth_bb: f64,
    pub deals: u64,
    pub seed: u64,
    pub network_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub range_policy_sha256: Option<String>,
    pub confidence: f64,
    pub threads: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub opponent_samples_per_deal: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub opponent_samples_per_runout: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub public_branches_per_street: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scenarios_per_deal: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub river_resolver_iterations: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub river_resolver_averaging_delay: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_resolver_iterations: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_resolver_averaging_delay: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_resolver_river_refinement_iterations: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_resolver_regret_matching_plus: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flop_resolver_iterations: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flop_resolver_averaging_delay: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flop_resolver_regret_matching_plus: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flop_resolver_resolved_policy_weight: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flop_resolver_value_network_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub flop_resolver_auxiliary_value_network_sha256s: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flop_resolver_continuation_selection:
        Option<super::public_belief::FlopContinuationSelection>,
    pub exact_betting_tree_nodes: u64,
    /// Retained outer-game observations permit paired candidate comparisons,
    /// deterministic replay, and honest inspection of chance-sampling noise.
    pub sample_exploitabilities_bb: Vec<f64>,
    /// Per observation, responder-zero and responder-one values before their
    /// average is clamped into the reported exploitability sample.
    pub sample_response_values_bb: Vec<[f64; 2]>,
    pub sample_mean_exploitability_bb: f64,
    pub sample_standard_error_bb: f64,
    pub hoeffding_margin_bb: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub empirical_bernstein_margin_bb: Option<f64>,
    pub confidence_bound_method: &'static str,
    pub confidence_bound_reference: &'static str,
    pub exploitability_upper_bound_bb: f64,
    pub relaxation: &'static str,
    pub guarantee: &'static str,
    pub assumptions: Vec<&'static str>,
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

/// Frozen inference-only policy shared by the neural sampler, preflop
/// continuation oracle, and response evaluators. Keeping inference in one
/// implementation prevents the offline solver from silently using different
/// feature or action semantics than the browser artifact.
pub(super) struct FrozenPolicy {
    bundle: TrainingNetworkBundle,
    bundle_sha256: String,
    range_policy: Option<RangeConditionedPolicyNetwork>,
    range_cache: Mutex<BTreeMap<[u8; 32], Arc<RangePolicyCachedNode>>>,
    preflop_range_cache: Mutex<BTreeMap<[u8; 32], Arc<RangePolicyCachedNode>>>,
    flop_strategy_cache: Mutex<BTreeMap<[u8; 32], Arc<ResolvedPolicyCachedRow>>>,
    turn_strategy_cache: Mutex<BTreeMap<[u8; 32], Arc<ResolvedPolicyCachedRow>>>,
    river_strategy_cache: Mutex<BTreeMap<[u8; 32], Arc<ResolvedPolicyCachedRow>>>,
    river_resolver: Option<RiverResolverConfig>,
    turn_resolver: Option<TurnResolverConfig>,
    flop_resolver: Option<FlopResolverRuntime>,
}

struct FlopResolverRuntime {
    config: FlopResolverConfig,
    value_network: super::public_belief::PublicValueNetwork,
    auxiliary_value_networks: Vec<super::public_belief::PublicValueNetwork>,
}

struct RangePolicyCachedNode {
    action_labels: Vec<String>,
    probabilities: Vec<f64>,
    ranges: [Vec<f64>; 2],
}

struct ResolvedPolicyCachedRow {
    action_labels: Vec<String>,
    probabilities: Vec<f64>,
}

impl FrozenPolicy {
    pub(super) fn load(path: &std::path::Path) -> Result<Self, Box<dyn Error>> {
        let bytes = fs::read(path)?;
        let bundle_sha256 = format!("{:x}", Sha256::digest(&bytes));
        let bundle: TrainingNetworkBundle = serde_json::from_slice(&bytes)?;
        validate_training_bundle(&bundle)?;
        Ok(Self {
            bundle,
            bundle_sha256,
            range_policy: None,
            range_cache: Mutex::new(BTreeMap::new()),
            preflop_range_cache: Mutex::new(BTreeMap::new()),
            flop_strategy_cache: Mutex::new(BTreeMap::new()),
            turn_strategy_cache: Mutex::new(BTreeMap::new()),
            river_strategy_cache: Mutex::new(BTreeMap::new()),
            river_resolver: None,
            turn_resolver: None,
            flop_resolver: None,
        })
    }

    pub(super) fn load_with_range(
        path: &std::path::Path,
        range_policy_path: Option<&std::path::Path>,
    ) -> Result<Self, Box<dyn Error>> {
        let mut policy = Self::load(path)?;
        policy.range_policy = range_policy_path
            .map(RangeConditionedPolicyNetwork::read)
            .transpose()?;
        if let Some(range_policy) = &policy.range_policy {
            range_policy.validate_source_policy_sha256(&policy.bundle_sha256)?;
        }
        Ok(policy)
    }

    pub(super) fn strategy(
        &self,
        state: &GameState,
        deal: &Deal,
        actions: &[LegalAction],
        config: &BlueprintConfig,
    ) -> Vec<f64> {
        if state.street == Street::Preflop || self.range_policy.is_none() {
            return strategy_from_bundle(&self.bundle, state, deal, actions, config);
        }
        self.range_strategy(state, deal, actions, config)
            .expect("validated range-conditioned policy state")
    }

    fn range_strategy(
        &self,
        state: &GameState,
        deal: &Deal,
        actions: &[LegalAction],
        config: &BlueprintConfig,
    ) -> Result<Vec<f64>, String> {
        self.range_node(state, deal, actions, config)?
            .range_for_deal(state, deal, actions)
    }

    fn range_node(
        &self,
        state: &GameState,
        deal: &Deal,
        actions: &[LegalAction],
        config: &BlueprintConfig,
    ) -> Result<Arc<RangePolicyCachedNode>, String> {
        let range_policy = self
            .range_policy
            .as_ref()
            .ok_or_else(|| "range-conditioned policy is unavailable".to_owned())?;
        let key = range_policy_cache_key(state, deal);
        if let Some(cached) = self
            .range_cache
            .lock()
            .expect("range policy cache")
            .get(&key)
            .cloned()
        {
            return Ok(cached);
        }
        let ranges = self.ranges_for_state(state, deal, config)?;
        let public = PublicBeliefState::from_game_state(
            deal.board[..state.street.board_len()].to_vec(),
            state,
            ranges,
        );
        let probabilities = if let Some(resolver) = self
            .flop_resolver
            .as_ref()
            .filter(|_| state.street == Street::Flop)
        {
            let resolved = self
                .flop_strategy_cache
                .lock()
                .expect("flop strategy cache")
                .get(&key)
                .cloned();
            let resolved = if let Some(resolved) = resolved {
                resolved
            } else {
                let solution =
                    super::public_belief::solve_flop(super::public_belief::FlopResolveConfig {
                        game: config.clone(),
                        state: public.clone(),
                        iterations: resolver.config.iterations,
                        averaging_delay: resolver.config.averaging_delay,
                        regret_matching_plus: resolver.config.regret_matching_plus,
                        value_network: resolver.value_network.clone(),
                        auxiliary_value_networks: resolver.auxiliary_value_networks.clone(),
                        continuation_selection: resolver.config.continuation_selection,
                        threads: resolver.config.threads,
                    })?;
                let mut cache = self
                    .flop_strategy_cache
                    .lock()
                    .expect("flop strategy cache");
                cache_resolved_policy_rows(
                    &mut cache,
                    key,
                    Street::Flop,
                    &public.board,
                    solution.strategies,
                    4_096,
                )?
            };
            let labels = actions
                .iter()
                .map(|action| action.label.as_str())
                .collect::<Vec<_>>();
            if resolved
                .action_labels
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
                != labels
                || resolved.probabilities.len() != COMBO_COUNT * actions.len()
            {
                return Err("flop resolver root strategy is incompatible".to_owned());
            }
            if resolver.config.resolved_policy_weight < 1.0 {
                let source = range_policy
                    .requires_source_policy()
                    .then(|| self.bundle_strategy_matrix(state, &public.board, actions, config))
                    .transpose()?;
                let anchor = range_policy.strategy(&public, config, source.as_deref())?;
                blend_resolved_with_anchor(
                    &resolved.probabilities,
                    &anchor,
                    actions.len(),
                    resolver.config.resolved_policy_weight,
                )?
            } else {
                resolved.probabilities.clone()
            }
        } else if let Some(resolver) = self.turn_resolver.filter(|_| state.street == Street::Turn) {
            let resolved = self
                .turn_strategy_cache
                .lock()
                .expect("turn strategy cache")
                .get(&key)
                .cloned();
            let resolved = if let Some(resolved) = resolved {
                resolved
            } else {
                let solution = super::public_belief::solve_turn_river(
                    super::public_belief::TurnRiverSolveConfig {
                        game: config.clone(),
                        state: public.clone(),
                        iterations: resolver.iterations,
                        averaging_delay: resolver.averaging_delay,
                        river_refinement_iterations: resolver.river_refinement_iterations,
                        regret_matching_plus: resolver.regret_matching_plus,
                    },
                )?;
                let mut cache = self
                    .turn_strategy_cache
                    .lock()
                    .expect("turn strategy cache");
                cache_resolved_policy_rows(
                    &mut cache,
                    key,
                    Street::Turn,
                    &public.board,
                    solution.strategies.into_iter().filter(|strategy| {
                        !strategy
                            .public_history
                            .last()
                            .is_some_and(|part| part.starts_with("chance:river:"))
                    }),
                    4_096,
                )?
            };
            let labels = actions
                .iter()
                .map(|action| action.label.as_str())
                .collect::<Vec<_>>();
            if resolved
                .action_labels
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
                != labels
                || resolved.probabilities.len() != COMBO_COUNT * actions.len()
            {
                return Err("turn resolver root strategy is incompatible".to_owned());
            }
            resolved.probabilities.clone()
        } else if let Some(resolver) = self
            .river_resolver
            .filter(|_| state.street == Street::River)
        {
            let resolved = self
                .river_strategy_cache
                .lock()
                .expect("river strategy cache")
                .get(&key)
                .cloned();
            let resolved = if let Some(resolved) = resolved {
                resolved
            } else {
                let solution =
                    super::public_belief::solve_river(super::public_belief::RiverSolveConfig {
                        game: config.clone(),
                        state: public.clone(),
                        iterations: resolver.iterations,
                        averaging_delay: resolver.averaging_delay,
                    })?;
                let mut cache = self
                    .river_strategy_cache
                    .lock()
                    .expect("river strategy cache");
                cache_resolved_policy_rows(
                    &mut cache,
                    key,
                    Street::River,
                    &public.board,
                    solution.strategies,
                    16_384,
                )?
            };
            let labels = actions
                .iter()
                .map(|action| action.label.as_str())
                .collect::<Vec<_>>();
            if resolved
                .action_labels
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
                != labels
                || resolved.probabilities.len() != COMBO_COUNT * actions.len()
            {
                return Err("river resolver root strategy is incompatible".to_owned());
            }
            resolved.probabilities.clone()
        } else {
            let source = range_policy
                .requires_source_policy()
                .then(|| self.bundle_strategy_matrix(state, &public.board, actions, config))
                .transpose()?;
            range_policy.strategy(&public, config, source.as_deref())?
        };
        let cached = Arc::new(RangePolicyCachedNode {
            action_labels: actions.iter().map(|action| action.label.clone()).collect(),
            probabilities,
            ranges: public.ranges,
        });
        {
            let mut cache = self.range_cache.lock().expect("range policy cache");
            if cache.len() >= 4_096 {
                if let Some(oldest) = cache.keys().next().copied() {
                    cache.remove(&oldest);
                }
            }
            cache.insert(key, cached.clone());
        }
        Ok(cached)
    }

    fn policy_node(
        &self,
        state: &GameState,
        deal: &Deal,
        actions: &[LegalAction],
        config: &BlueprintConfig,
    ) -> Result<Arc<RangePolicyCachedNode>, String> {
        if state.street != Street::Preflop {
            return self.range_node(state, deal, actions, config);
        }
        let key = range_policy_cache_key(state, deal);
        if let Some(cached) = self
            .preflop_range_cache
            .lock()
            .expect("preflop range cache")
            .get(&key)
            .cloned()
        {
            return Ok(cached);
        }
        let ranges = self.ranges_for_state(state, deal, config)?;
        let mut probabilities = vec![0.0; COMBO_COUNT * actions.len()];
        for combo in all_combos() {
            let synthetic = deal_for_policy_combo(combo, state.actor);
            let strategy = strategy_from_bundle(&self.bundle, state, &synthetic, actions, config);
            let offset = combo.key() * actions.len();
            probabilities[offset..offset + actions.len()].copy_from_slice(&strategy);
        }
        let cached = Arc::new(RangePolicyCachedNode {
            action_labels: actions.iter().map(|action| action.label.clone()).collect(),
            probabilities,
            ranges,
        });
        let mut cache = self
            .preflop_range_cache
            .lock()
            .expect("preflop range cache");
        if cache.len() >= 256 {
            if let Some(oldest) = cache.keys().next().copied() {
                cache.remove(&oldest);
            }
        }
        cache.insert(key, cached.clone());
        Ok(cached)
    }

    fn ranges_for_state(
        &self,
        state: &GameState,
        deal: &Deal,
        config: &BlueprintConfig,
    ) -> Result<[Vec<f64>; 2], String> {
        let key = range_policy_cache_key(state, deal);
        let cached = if state.street == Street::Preflop {
            self.preflop_range_cache
                .lock()
                .expect("preflop range cache")
                .get(&key)
                .cloned()
        } else {
            self.range_cache
                .lock()
                .expect("range policy cache")
                .get(&key)
                .cloned()
        };
        if let Some(cached) = cached {
            return Ok(cached.ranges.clone());
        }
        if state.trajectory.is_empty() {
            let uniform = 1.0 / COMBO_COUNT as f64;
            let mut ranges = [vec![uniform; COMBO_COUNT], vec![uniform; COMBO_COUNT]];
            normalize_ranges_for_board(&mut ranges, &deal.board[..state.street.board_len()])?;
            return Ok(ranges);
        }

        let last = state.trajectory.last().expect("nonempty range trajectory");
        let mut parent = GameState::initial(config);
        for observed in &state.trajectory[..state.trajectory.len() - 1] {
            let legal = parent.legal_actions(config);
            let selected = legal
                .iter()
                .position(|action| trajectory_action_matches(&parent, action, observed, config))
                .ok_or_else(|| "range policy could not replay a trajectory prefix".to_owned())?;
            parent = parent.apply(&legal[selected], config);
        }
        let parent_actions = parent.legal_actions(config);
        let selected = parent_actions
            .iter()
            .position(|action| trajectory_action_matches(&parent, action, last, config))
            .ok_or_else(|| {
                "range policy could not replay the final trajectory action".to_owned()
            })?;
        let reconstructed = parent.apply(&parent_actions[selected], config);
        if reconstructed.public_history != state.public_history
            || reconstructed.street != state.street
            || reconstructed.actor != state.actor
        {
            return Err("range policy replay did not reconstruct the requested state".to_owned());
        }
        let parent_node = self.policy_node(&parent, deal, &parent_actions, config)?;
        let mut ranges = parent_node.ranges.clone();
        let actor = parent.actor;
        for combo in 0..COMBO_COUNT {
            ranges[actor][combo] *=
                parent_node.probabilities[combo * parent_actions.len() + selected];
        }
        normalize_ranges_for_board(&mut ranges, &deal.board[..state.street.board_len()])?;
        Ok(ranges)
    }

    #[cfg(test)]
    fn replay_ranges_from_root(
        &self,
        state: &GameState,
        deal: &Deal,
        config: &BlueprintConfig,
    ) -> Result<[Vec<f64>; 2], String> {
        let range_policy = self
            .range_policy
            .as_ref()
            .ok_or_else(|| "range-conditioned policy is unavailable".to_owned())?;
        let mut cursor = GameState::initial(config);
        let uniform = 1.0 / COMBO_COUNT as f64;
        let mut ranges = [vec![uniform; COMBO_COUNT], vec![uniform; COMBO_COUNT]];
        for observed in &state.trajectory {
            let legal = cursor.legal_actions(config);
            let selected = legal
                .iter()
                .position(|action| trajectory_action_matches(&cursor, action, observed, config))
                .ok_or_else(|| "range policy could not replay the public trajectory".to_owned())?;
            if cursor.street == Street::Preflop {
                let actor = cursor.actor;
                for combo in all_combos() {
                    if ranges[actor][combo.key()] <= 0.0 {
                        continue;
                    }
                    let synthetic = deal_for_policy_combo(combo, actor);
                    let strategy =
                        strategy_from_bundle(&self.bundle, &cursor, &synthetic, &legal, config);
                    ranges[actor][combo.key()] *= strategy[selected];
                }
            } else {
                normalize_ranges_for_board(&mut ranges, &deal.board[..cursor.street.board_len()])?;
                let public = PublicBeliefState::from_game_state(
                    deal.board[..cursor.street.board_len()].to_vec(),
                    &cursor,
                    ranges.clone(),
                );
                let source = range_policy
                    .requires_source_policy()
                    .then(|| self.bundle_strategy_matrix(&cursor, &public.board, &legal, config))
                    .transpose()?;
                let matrix = range_policy.strategy(&public, config, source.as_deref())?;
                let actor = cursor.actor;
                for combo in 0..COMBO_COUNT {
                    ranges[actor][combo] *= matrix[combo * legal.len() + selected];
                }
            }
            cursor = cursor.apply(&legal[selected], config);
        }
        if cursor.public_history != state.public_history
            || cursor.street != state.street
            || cursor.actor != state.actor
        {
            return Err("range policy replay did not reconstruct the requested state".to_owned());
        }
        normalize_ranges_for_board(&mut ranges, &deal.board[..state.street.board_len()])?;
        Ok(ranges)
    }

    fn range_public_state(
        &self,
        state: &GameState,
        deal: &Deal,
        actions: &[LegalAction],
        config: &BlueprintConfig,
    ) -> Result<PublicBeliefState, String> {
        let cached = self.range_node(state, deal, actions, config)?;
        Ok(PublicBeliefState::from_game_state(
            deal.board[..state.street.board_len()].to_vec(),
            state,
            cached.ranges.clone(),
        ))
    }

    pub(super) fn bundle_sha256(&self) -> &str {
        &self.bundle_sha256
    }

    fn enable_flop_resolver(&mut self, resolver: FlopResolverConfig) -> Result<(), Box<dyn Error>> {
        if self.range_policy.is_none() {
            return Err("flop resolving requires a range-conditioned policy".into());
        }
        if resolver.iterations < 2
            || resolver.averaging_delay >= resolver.iterations
            || resolver.threads == 0
            || !resolver.resolved_policy_weight.is_finite()
            || !(0.0..=1.0).contains(&resolver.resolved_policy_weight)
        {
            return Err("flop resolver configuration is invalid".into());
        }
        let value_network =
            super::public_belief::PublicValueNetwork::read(&resolver.value_network_path)?;
        let auxiliary_value_networks = resolver
            .auxiliary_value_network_paths
            .iter()
            .map(|path| super::public_belief::PublicValueNetwork::read(path))
            .collect::<Result<Vec<_>, _>>()?;
        self.flop_resolver = Some(FlopResolverRuntime {
            config: resolver,
            value_network,
            auxiliary_value_networks,
        });
        self.range_cache.lock().expect("range policy cache").clear();
        self.flop_strategy_cache
            .lock()
            .expect("flop strategy cache")
            .clear();
        Ok(())
    }

    fn enable_turn_resolver(&mut self, resolver: TurnResolverConfig) -> Result<(), String> {
        if self.range_policy.is_none() {
            return Err("turn resolving requires a range-conditioned policy".to_owned());
        }
        if resolver.iterations < 2 || resolver.averaging_delay >= resolver.iterations {
            return Err("turn resolver configuration is invalid".to_owned());
        }
        self.turn_resolver = Some(resolver);
        self.range_cache.lock().expect("range policy cache").clear();
        self.turn_strategy_cache
            .lock()
            .expect("turn strategy cache")
            .clear();
        Ok(())
    }

    fn enable_river_resolver(&mut self, resolver: RiverResolverConfig) -> Result<(), String> {
        if self.range_policy.is_none() {
            return Err("river resolving requires a range-conditioned policy".to_owned());
        }
        if resolver.iterations < 2 || resolver.averaging_delay >= resolver.iterations {
            return Err("river resolver configuration is invalid".to_owned());
        }
        self.river_resolver = Some(resolver);
        self.range_cache.lock().expect("range policy cache").clear();
        self.river_strategy_cache
            .lock()
            .expect("river strategy cache")
            .clear();
        Ok(())
    }

    pub(super) fn bundle_strategy_matrix(
        &self,
        state: &GameState,
        board: &[u8],
        actions: &[LegalAction],
        config: &BlueprintConfig,
    ) -> Result<Vec<f64>, String> {
        let mut probabilities = vec![0.0; COMBO_COUNT * actions.len()];
        for combo in all_combos() {
            if combo.cards().iter().any(|card| board.contains(card)) {
                continue;
            }
            let deal = deal_for_policy_combo_on_board(combo, state.actor, board)?;
            let strategy = strategy_from_bundle(&self.bundle, state, &deal, actions, config);
            let offset = combo.key() * actions.len();
            probabilities[offset..offset + actions.len()].copy_from_slice(&strategy);
        }
        Ok(probabilities)
    }
}

impl RangePolicyCachedNode {
    fn range_for_deal(
        &self,
        state: &GameState,
        deal: &Deal,
        actions: &[LegalAction],
    ) -> Result<Vec<f64>, String> {
        let labels = actions
            .iter()
            .map(|action| action.label.clone())
            .collect::<Vec<_>>();
        if labels != self.action_labels {
            return Err("cached range policy legal actions changed".to_owned());
        }
        let combo = Combo::new(deal.holes[state.actor][0], deal.holes[state.actor][1]).key();
        let offset = combo * actions.len();
        let row = self.probabilities[offset..offset + actions.len()].to_vec();
        if (row.iter().sum::<f64>() - 1.0).abs() > 1e-6 {
            return Err(
                "range policy cannot score the requested forced-deviation combo".to_owned(),
            );
        }
        Ok(row)
    }
}

fn range_policy_cache_key(state: &GameState, deal: &Deal) -> [u8; 32] {
    range_policy_public_cache_key(
        state.street,
        state.actor,
        &deal.board[..state.street.board_len()],
        &state.public_history,
    )
}

fn range_policy_public_cache_key(
    street: Street,
    actor: usize,
    board: &[u8],
    public_history: &[String],
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"hu-range-policy-public-state-v1");
    digest.update([street as u8, actor as u8]);
    for card in board {
        digest.update([*card]);
    }
    for history in public_history {
        digest.update((history.len() as u64).to_le_bytes());
        digest.update(history.as_bytes());
    }
    digest.finalize().into()
}

fn cache_resolved_policy_rows(
    cache: &mut BTreeMap<[u8; 32], Arc<ResolvedPolicyCachedRow>>,
    requested_key: [u8; 32],
    street: Street,
    board: &[u8],
    strategies: impl IntoIterator<Item = super::public_belief::PublicBeliefStrategy>,
    capacity: usize,
) -> Result<Arc<ResolvedPolicyCachedRow>, String> {
    let mut requested = None;
    for strategy in strategies {
        let strategy_key =
            range_policy_public_cache_key(street, strategy.actor, board, &strategy.public_history);
        if cache.len() >= capacity && !cache.contains_key(&strategy_key) {
            if let Some(evicted) = cache.keys().next().copied() {
                cache.remove(&evicted);
            }
        }
        let action_labels = strategy.action_labels;
        let probabilities = stabilize_resolved_policy(
            strategy.probabilities.into_iter().map(f64::from).collect(),
            action_labels.len(),
        )?;
        let row = Arc::new(ResolvedPolicyCachedRow {
            action_labels,
            probabilities,
        });
        if strategy_key == requested_key {
            // Retain the requested row locally. A later insertion in this same
            // solution may evict it from the bounded shared cache, and another
            // worker can concurrently fill/evict unrelated entries.
            requested = Some(row.clone());
        }
        cache.insert(strategy_key, row);
    }
    requested.ok_or_else(|| "resolver solution omitted its requested root strategy".to_owned())
}

/// Exact CFR averages can assign a legal action zero probability after a
/// short research solve. The response evaluator deliberately explores forced
/// deviations, so retain negligible full support without materially changing
/// the solved mix. Rows for card-blocked combos remain all-zero.
fn stabilize_resolved_policy(
    mut probabilities: Vec<f64>,
    action_count: usize,
) -> Result<Vec<f64>, String> {
    const MINIMUM_ACTION_PROBABILITY: f64 = 1e-9;
    if action_count == 0 || probabilities.len() != COMBO_COUNT * action_count {
        return Err("resolved policy dimensions are incompatible".to_owned());
    }
    for row in probabilities.chunks_exact_mut(action_count) {
        let sum = row.iter().sum::<f64>();
        if sum <= 0.0 {
            continue;
        }
        if !sum.is_finite() || row.iter().any(|probability| !probability.is_finite()) {
            return Err("resolved policy contains non-finite probabilities".to_owned());
        }
        for probability in row.iter_mut() {
            *probability = probability.max(MINIMUM_ACTION_PROBABILITY);
        }
        let stabilized_sum = row.iter().sum::<f64>();
        for probability in row {
            *probability /= stabilized_sum;
        }
    }
    Ok(probabilities)
}

fn blend_resolved_with_anchor(
    resolved: &[f64],
    anchor: &[f64],
    action_count: usize,
    resolved_weight: f64,
) -> Result<Vec<f64>, String> {
    if action_count == 0
        || resolved.len() != COMBO_COUNT * action_count
        || anchor.len() != resolved.len()
        || !resolved_weight.is_finite()
        || !(0.0..=1.0).contains(&resolved_weight)
    {
        return Err("anchored resolved policy dimensions are incompatible".to_owned());
    }
    let anchor_weight = 1.0 - resolved_weight;
    let mut blended = Vec::with_capacity(resolved.len());
    for (resolved_row, anchor_row) in resolved
        .chunks_exact(action_count)
        .zip(anchor.chunks_exact(action_count))
    {
        let resolved_sum = resolved_row.iter().sum::<f64>();
        let anchor_sum = anchor_row.iter().sum::<f64>();
        if resolved_sum <= 0.0 && anchor_sum <= 0.0 {
            blended.extend(std::iter::repeat_n(0.0, action_count));
            continue;
        }
        if !resolved_sum.is_finite()
            || !anchor_sum.is_finite()
            || resolved_row
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            || anchor_row
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err("anchored resolved policy contains invalid probabilities".to_owned());
        }
        for action in 0..action_count {
            let resolved_probability = if resolved_sum > 0.0 {
                resolved_row[action] / resolved_sum
            } else {
                anchor_row[action] / anchor_sum
            };
            let anchor_probability = if anchor_sum > 0.0 {
                anchor_row[action] / anchor_sum
            } else {
                resolved_row[action] / resolved_sum
            };
            blended
                .push(resolved_weight * resolved_probability + anchor_weight * anchor_probability);
        }
    }
    stabilize_resolved_policy(blended, action_count)
}

fn trajectory_action_matches(
    state: &GameState,
    action: &LegalAction,
    observed: &TrajectoryAction,
    config: &BlueprintConfig,
) -> bool {
    let next = state.apply(action, config);
    let Some(candidate) = next.trajectory.last() else {
        return false;
    };
    candidate.actor == observed.actor
        && candidate.street == observed.street
        && candidate.kind == observed.kind
        && (candidate.amount_bb - observed.amount_bb).abs() <= 1e-6
        && match (candidate.amount_to_bb, observed.amount_to_bb) {
            (Some(left), Some(right)) => (left - right).abs() <= 1e-6,
            (None, None) => true,
            _ => false,
        }
}

fn deal_for_policy_combo(combo: Combo, actor: usize) -> Deal {
    deal_for_policy_combo_on_board(combo, actor, &[])
        .expect("an empty public board always has enough remaining cards")
}

fn deal_for_policy_combo_on_board(
    combo: Combo,
    actor: usize,
    visible_board: &[u8],
) -> Result<Deal, String> {
    if visible_board.len() > 5 {
        return Err("source policy board contains more than five cards".to_owned());
    }
    let private = combo.cards();
    if private.iter().any(|card| visible_board.contains(card))
        || visible_board.iter().copied().collect::<BTreeSet<_>>().len() != visible_board.len()
    {
        return Err("source policy private cards conflict with the public board".to_owned());
    }
    let available = (0..52u8)
        .filter(|card| !private.contains(card) && !visible_board.contains(card))
        .take(7 - visible_board.len())
        .collect::<Vec<_>>();
    if available.len() != 7 - visible_board.len() {
        return Err("source policy synthetic deal lacks enough cards".to_owned());
    }
    let mut holes = [[0u8; 2]; 2];
    holes[actor] = private;
    holes[1 - actor] = [available[0], available[1]];
    let mut board = [0u8; 5];
    board[..visible_board.len()].copy_from_slice(visible_board);
    board[visible_board.len()..].copy_from_slice(&available[2..]);
    Ok(Deal::from_sampled_cards(holes, board))
}

fn normalize_ranges_for_board(ranges: &mut [Vec<f64>; 2], board: &[u8]) -> Result<(), String> {
    for player in 0..2 {
        for combo in all_combos() {
            if combo.cards().iter().any(|card| board.contains(card)) {
                ranges[player][combo.key()] = 0.0;
            }
        }
        let total = ranges[player].iter().sum::<f64>();
        // Public-action reach is a scale factor, not a probability after each
        // observed action. Rare but reachable lines may legitimately carry
        // less than the game's comparison epsilon before conditioning.
        if !total.is_finite() || total <= 0.0 {
            return Err("range policy public action has zero conditional reach".to_owned());
        }
        for weight in &mut ranges[player] {
            *weight /= total;
        }
    }
    Ok(())
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

fn validate_training_bundle(bundle: &TrainingNetworkBundle) -> Result<(), Box<dyn Error>> {
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
    Ok(())
}

fn strategy_from_bundle(
    bundle: &TrainingNetworkBundle,
    state: &GameState,
    deal: &Deal,
    actions: &[LegalAction],
    config: &BlueprintConfig,
) -> Vec<f64> {
    let network = bundle.policy_network(state.street, state.actor);
    let state_features = encode_state_features(state, deal, config);
    let action_features = actions
        .iter()
        .map(|action| encode_action_features(state, action, config))
        .collect::<Vec<_>>();
    let scores = network.score_state_actions(&state_features, &action_features);
    match bundle.strategy_transform {
        StrategyTransform::RegretMatching => {
            normalize_or_uniform(scores.into_iter().map(|value| value.max(0.0)).collect())
        }
        StrategyTransform::Softmax => stable_softmax(&scores),
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

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum SampleKind {
    AdvantageP0,
    AdvantageP1,
    AverageStrategy,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CompactTrajectoryAction {
    actor: usize,
    street: Street,
    kind: TrajectoryActionKind,
    amount_bb: f32,
    amount_to_bb: Option<f32>,
    pot_after_bb: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CompactLegalAction {
    kind: TrajectoryActionKind,
    amount_to_bb: Option<f32>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
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
    action_value_standard_errors_bb: Option<Vec<f32>>,
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
    evaluates_trajectory_action_values: bool,
    enumerates_turn_river_chance: bool,
    action_abstraction: &'a ActionAbstraction,
}

struct SampleGenerator {
    config: SampleGenerationConfig,
    networks: Option<FrozenPolicy>,
    rng: SplitMix64,
    records: Vec<TrainingSample>,
    attempted_records: usize,
    range_records: Vec<Vec<u8>>,
    attempted_range_records: usize,
    range_self_play_only: bool,
}

impl SampleGenerator {
    fn new(config: SampleGenerationConfig) -> Result<Self, Box<dyn Error>> {
        Self::new_with_range(config, None)
    }

    fn new_with_range(
        config: SampleGenerationConfig,
        range_policy_path: Option<&std::path::Path>,
    ) -> Result<Self, Box<dyn Error>> {
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
        if config.evaluate_trajectory_values && !config.trajectory_sampling {
            return Err("trajectory action-value evaluation requires trajectory sampling".into());
        }
        if config.evaluate_trajectory_values && config.value_rollouts_per_action < 2 {
            return Err("trajectory action-value evaluation requires at least two rollouts".into());
        }
        let networks = match &config.network_path {
            Some(path) => Some(FrozenPolicy::load_with_range(path, range_policy_path)?),
            None => None,
        };
        Ok(Self {
            rng: SplitMix64::new(config.seed),
            records: Vec::with_capacity(config.max_records.min(65_536)),
            attempted_records: 0,
            range_records: Vec::with_capacity(config.max_records.min(65_536)),
            attempted_range_records: 0,
            range_self_play_only: false,
            config,
            networks,
        })
    }

    fn enable_river_resolver(
        &mut self,
        resolver: Option<RiverResolverConfig>,
    ) -> Result<(), Box<dyn Error>> {
        if let Some(resolver) = resolver {
            self.networks
                .as_mut()
                .ok_or("river resolving requires a frozen policy")?
                .enable_river_resolver(resolver)?;
        }
        Ok(())
    }

    fn enable_turn_resolver(
        &mut self,
        resolver: Option<TurnResolverConfig>,
    ) -> Result<(), Box<dyn Error>> {
        if let Some(resolver) = resolver {
            self.networks
                .as_mut()
                .ok_or("turn resolving requires a frozen policy")?
                .enable_turn_resolver(resolver)?;
        }
        Ok(())
    }

    fn enable_flop_resolver(
        &mut self,
        resolver: Option<FlopResolverConfig>,
    ) -> Result<(), Box<dyn Error>> {
        if let Some(resolver) = resolver {
            self.networks
                .as_mut()
                .ok_or("flop resolving requires a frozen policy")?
                .enable_flop_resolver(resolver)?;
        }
        Ok(())
    }

    fn run(
        mut self,
    ) -> Result<
        (
            SampleGenerationConfig,
            Vec<TrainingSample>,
            usize,
            Vec<Vec<u8>>,
            usize,
        ),
        Box<dyn Error>,
    > {
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
        Ok((
            self.config,
            self.records,
            self.attempted_records,
            self.range_records,
            self.attempted_range_records,
        ))
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
        bundle.strategy(state, deal, actions, &self.config.game)
    }

    fn sampled_value_baseline(
        &self,
        state: &GameState,
        deal: &Deal,
        actions: &[LegalAction],
        traverser: usize,
    ) -> Option<Vec<f64>> {
        let bundle = &self.networks.as_ref()?.bundle;
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
        if !self.range_self_play_only && self.records.len() < self.config.max_records {
            self.records.push(sample);
        }
    }

    fn push_range_record(&mut self, record: Option<Vec<u8>>) {
        self.attempted_range_records += 1;
        if let Some(record) = record {
            debug_assert!(self.range_records.len() < self.config.max_records);
            self.range_records.push(record);
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
                        * self.value_only_after_action(&state, action, deal, traverser, rng)
                })
                .sum()
        } else {
            let selected = sample_index(&strategy, rng);
            let baselines = self.sampled_value_baseline(&state, deal, &actions, traverser);
            let sampled_value =
                self.value_only_after_action(&state, &actions[selected], deal, traverser, rng);
            match baselines {
                Some(values) => {
                    baseline_corrected_sample(&strategy, &values, selected, sampled_value)
                }
                None => sampled_value,
            }
        }
    }

    fn value_only_after_action(
        &self,
        state: &GameState,
        action: &LegalAction,
        deal: &Deal,
        traverser: usize,
        rng: &mut SplitMix64,
    ) -> f64 {
        let next = state.apply(action, &self.config.game);
        if self.config.enumerate_turn_river_chance
            && state.street == Street::Turn
            && next.street == Street::River
            && next.terminal.is_none()
        {
            let rivers = exact_river_deals(deal);
            return rivers
                .iter()
                .map(|river_deal| {
                    self.value_only_external_sampling(next.clone(), river_deal, traverser, rng)
                })
                .sum::<f64>()
                / rivers.len() as f64;
        }
        self.value_only_external_sampling(next, deal, traverser, rng)
    }

    fn action_value_estimates(
        &self,
        state: &GameState,
        deal: &Deal,
        actions: &[LegalAction],
        traverser: usize,
        iteration: u64,
        primary_values: Option<&[f64]>,
    ) -> (Vec<f64>, Option<Vec<f64>>) {
        let samples = self.config.value_rollouts_per_action;
        let estimates = actions
            .iter()
            .enumerate()
            .map(|(action_index, action)| {
                let mut values = Vec::with_capacity(samples as usize);
                if let Some(primary) = primary_values {
                    values.push(primary[action_index]);
                }
                let first_independent_sample = u32::from(primary_values.is_some());
                for sample_index in first_independent_sample..samples {
                    let mut rng = SplitMix64::new(value_rollout_seed(
                        &self.config,
                        state,
                        deal,
                        action,
                        iteration,
                        sample_index,
                    ));
                    values.push(
                        self.value_only_after_action(state, action, deal, traverser, &mut rng),
                    );
                }
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let standard_error = if values.len() >= 2 {
                    let squared_deviations = values
                        .iter()
                        .map(|value| (value - mean).powi(2))
                        .sum::<f64>();
                    (squared_deviations / ((values.len() - 1) * values.len()) as f64).sqrt()
                } else {
                    f64::NAN
                };
                (mean, standard_error)
            })
            .collect::<Vec<_>>();
        let means = estimates.iter().map(|(mean, _)| *mean).collect();
        let standard_errors = (samples >= 2).then(|| {
            estimates
                .iter()
                .map(|(_, standard_error)| *standard_error)
                .collect()
        });
        (means, standard_errors)
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
        let (action_values, action_value_standard_errors) = if self
            .config
            .evaluate_trajectory_values
        {
            let (values, standard_errors) =
                self.action_value_estimates(&state, deal, &actions, state.actor, iteration, None);
            (Some(values), standard_errors)
        } else {
            (None, None)
        };
        self.push_record(training_sample(
            SampleKind::AverageStrategy,
            iteration,
            1.0,
            reach_probability,
            &state,
            deal,
            &actions,
            strategy.clone(),
            action_values,
            action_value_standard_errors,
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
                values.push(self.external_sampling_after_action(
                    &state,
                    action,
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
            let is_range_node = state.street != Street::Preflop
                && self
                    .networks
                    .as_ref()
                    .is_some_and(|policy| policy.range_policy.is_some());
            let retain_range_record =
                is_range_node && self.range_records.len() < self.config.max_records;
            let (action_value_targets, action_value_standard_errors) =
                if !self.range_self_play_only || retain_range_record {
                    self.action_value_estimates(
                        &state,
                        deal,
                        &actions,
                        traverser,
                        iteration,
                        Some(&values),
                    )
                } else {
                    (values.clone(), None)
                };
            if is_range_node {
                let record = retain_range_record.then(|| {
                    range_policy_directional_record_bytes(
                        "range_conditioned_self_play_regret",
                        self,
                        &state,
                        deal,
                        &actions,
                        &strategy,
                        &action_value_targets,
                        action_value_standard_errors.as_deref(),
                        1.0,
                    )
                    .expect("validated range-conditioned self-play record")
                });
                self.push_range_record(record);
            }
            if !self.range_self_play_only {
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
                    action_value_standard_errors,
                    &self.config.game,
                ));
            }
            node_value
        } else {
            let iteration_weight = ((iteration + 1) as f64)
                .powf(self.config.game.dcfr.strategy_exponent)
                .min(f32::MAX as f64) as f32;
            if !self.range_self_play_only {
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
                    None,
                    &self.config.game,
                ));
            }
            let selected = sample_index(&strategy, &mut self.rng);
            let baselines = self.sampled_value_baseline(&state, deal, &actions, traverser);
            let sampled_value = self.external_sampling_after_action(
                &state,
                &actions[selected],
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

    #[allow(clippy::too_many_arguments)]
    fn external_sampling_after_action(
        &mut self,
        state: &GameState,
        action: &LegalAction,
        deal: &Deal,
        traverser: usize,
        iteration: u64,
        reach_probability: f64,
    ) -> f64 {
        let next = state.apply(action, &self.config.game);
        if self.config.enumerate_turn_river_chance
            && state.street == Street::Turn
            && next.street == Street::River
            && next.terminal.is_none()
        {
            let rivers = exact_river_deals(deal);
            let chance_probability = 1.0 / rivers.len() as f64;
            return rivers
                .iter()
                .map(|river_deal| {
                    self.external_sampling(
                        next.clone(),
                        river_deal,
                        traverser,
                        iteration,
                        reach_probability * chance_probability,
                    )
                })
                .sum::<f64>()
                * chance_probability;
        }
        self.external_sampling(next, deal, traverser, iteration, reach_probability)
    }
}

fn exact_river_deals(deal: &Deal) -> Vec<Deal> {
    let mut blocked = [false; 52];
    for card in deal.holes.iter().flatten().chain(&deal.board[..4]) {
        blocked[*card as usize] = true;
    }
    (0..52u8)
        .filter(|river| !blocked[*river as usize])
        .map(|river| {
            let mut board = deal.board;
            board[4] = river;
            Deal::from_sampled_cards(deal.holes, board)
        })
        .collect()
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
    action_value_standard_errors: Option<Vec<f64>>,
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
        action_value_standard_errors_bb: action_value_standard_errors
            .map(|values| values.into_iter().map(|value| value as f32).collect()),
    }
}

pub(super) fn average_strategy_record_json(
    state: &GameState,
    deal: &Deal,
    actions: &[LegalAction],
    targets: Vec<f64>,
    weight: f32,
    config: &BlueprintConfig,
) -> serde_json::Value {
    serde_json::to_value(training_sample(
        SampleKind::AverageStrategy,
        0,
        weight,
        1.0,
        state,
        deal,
        actions,
        targets,
        None,
        None,
        config,
    ))
    .expect("average-strategy sample is serializable")
}

pub(super) fn average_strategy_record_bytes(
    state: &GameState,
    deal: &Deal,
    actions: &[LegalAction],
    targets: Vec<f64>,
    action_values_bb: Vec<f64>,
    weight: f32,
    config: &BlueprintConfig,
) -> Vec<u8> {
    serde_json::to_vec(&training_sample(
        SampleKind::AverageStrategy,
        0,
        weight,
        1.0,
        state,
        deal,
        actions,
        targets,
        Some(action_values_bb),
        None,
        config,
    ))
    .expect("average-strategy sample is serializable")
}

/// Persist solver-produced average-policy records using the exact dataset
/// contract consumed by the MLX action-policy distiller.
pub(super) fn write_average_strategy_dataset(
    game: &BlueprintConfig,
    seed: u64,
    teacher: serde_json::Value,
    records: &[Vec<u8>],
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    if records.is_empty() {
        return Err("average-strategy dataset has no records".into());
    }
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let temporary = output.with_extension("tmp");
    let file = fs::File::create(&temporary)?;
    let buffered = BufWriter::new(file);
    let mut writer = GzEncoder::new(buffered, Compression::fast());
    let metadata = serde_json::json!({
        "record_type": "metadata",
        "schema": DATASET_SCHEMA,
        "state_feature_schema": "hu-cash-trajectory-poker-aware-v4",
        "state_feature_count": STATE_FEATURE_COUNT,
        "action_feature_schema": "hu-cash-legal-action-v1",
        "action_feature_count": ACTION_FEATURE_COUNT,
        "depth_bb": game.effective_stack_bb,
        "seed": seed,
        "start_iteration": 0,
        "traversals": 0,
        "records": records.len(),
        "truncated": false,
        "sampling_mode": "range_conditioned_solver_average_policy",
        "value_rollouts_per_action": 0,
        "evaluates_trajectory_action_values": true,
        "action_value_method": "exact_solver_average_profile_counterfactual_values",
        "enumerates_turn_river_chance": true,
        "action_abstraction": game.action_abstraction,
        "teacher": teacher,
    });
    serde_json::to_writer(&mut writer, &metadata)?;
    writer.write_all(b"\n")?;
    for record in records {
        writer.write_all(record)?;
        writer.write_all(b"\n")?;
    }
    writer.finish()?.flush()?;
    fs::rename(temporary, output)?;
    Ok(())
}

fn feature_sha256(features: &[f32]) -> String {
    let mut digest = Sha256::new();
    for feature in features {
        let canonical_micro_units = (*feature as f64 * 1_000_000.0).round() as i32;
        digest.update(canonical_micro_units.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

pub(super) fn stable_softmax(values: &[f64]) -> Vec<f64> {
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

pub(super) fn encode_action_features(
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
    let (config, records, attempted_records, _, _) = SampleGenerator::new(config)?.run()?;
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
        evaluates_trajectory_action_values: config.evaluate_trajectory_values,
        enumerates_turn_river_chance: config.enumerate_turn_river_chance,
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

/// Export postflop focal-combo action values from alternating-traverser
/// external-sampling self-play. Unlike fixed-response attribution, these rows
/// follow both players' current policy and therefore can be regenerated after
/// every accepted policy update without freezing a stale opponent response.
pub fn generate_range_self_play_samples(
    config: RangeSelfPlaySampleConfig,
) -> Result<RangeSelfPlaySampleReport, Box<dyn Error>> {
    if config.traversals < 2 || config.max_records < 3 {
        return Err("range self-play requires two traversals and three records".into());
    }
    if config.value_rollouts_per_action < 2 {
        return Err("range self-play action values require at least two rollouts".into());
    }
    let network_sha256 = sha256_path(&config.network_path)?;
    let range_policy_sha256 = sha256_path(&config.range_policy_path)?;
    let generator_config = SampleGenerationConfig {
        game: config.game.clone(),
        traversals: config.traversals,
        start_iteration: config.start_iteration,
        seed: config.seed,
        max_records: config.max_records,
        output: PathBuf::from("unused-range-self-play.jsonl.gz"),
        network_path: Some(config.network_path.clone()),
        trajectory_sampling: false,
        evaluate_trajectory_values: false,
        value_rollouts_per_action: config.value_rollouts_per_action,
        enumerate_turn_river_chance: config.enumerate_turn_river_chance,
    };
    let mut generator =
        SampleGenerator::new_with_range(generator_config, Some(&config.range_policy_path))?;
    generator.range_self_play_only = true;
    generator.records = Vec::new();
    let (_, _, _, records, attempted_records) = generator.run()?;
    if records.is_empty() || records.len() > config.max_records {
        return Err("range self-play produced no bounded records".into());
    }

    let mut retained_records_by_street = [0usize; 3];
    let mut minimum_probability = f64::INFINITY;
    let mut maximum_sum_error = 0.0f64;
    let mut minimum_action_value = f64::INFINITY;
    let mut maximum_action_value = f64::NEG_INFINITY;
    let mut maximum_standard_error = 0.0f64;
    for encoded in &records {
        let record: serde_json::Value = serde_json::from_slice(encoded)?;
        if record["record_type"] != "range_conditioned_self_play_regret"
            || record["weight"].as_f64().is_none_or(|weight| weight <= 0.0)
        {
            return Err("range self-play record header is invalid".into());
        }
        let actor = record["state"]["actor"]
            .as_u64()
            .filter(|actor| *actor < 2)
            .ok_or("range self-play actor is invalid")? as usize;
        let focal_combo = record["focal_combo"]
            .as_u64()
            .filter(|combo| *combo < super::public_belief::COMBO_COUNT as u64)
            .ok_or("range self-play focal combo is invalid")? as usize;
        let focal_reach = record["ranges"][actor][focal_combo]
            .as_f64()
            .ok_or("range self-play focal reach is absent")?;
        if !focal_reach.is_finite() || focal_reach <= 0.0 {
            return Err("range self-play focal combo has no reach".into());
        }
        let street = match record["state"]["street"].as_str() {
            Some("flop") => 0,
            Some("turn") => 1,
            Some("river") => 2,
            _ => return Err("range self-play retained a non-postflop street".into()),
        };
        retained_records_by_street[street] += 1;
        let probabilities = record["probabilities"]
            .as_array()
            .ok_or("range self-play probabilities are absent")?;
        let values = record["action_values_bb"]
            .as_array()
            .ok_or("range self-play action values are absent")?;
        let standard_errors = record["action_value_standard_errors_bb"]
            .as_array()
            .ok_or("range self-play standard errors are absent")?;
        if probabilities.is_empty()
            || probabilities.len() != values.len()
            || probabilities.len() != standard_errors.len()
        {
            return Err("range self-play action vectors differ".into());
        }
        let mut sum = 0.0;
        for probability in probabilities {
            let probability = probability
                .as_f64()
                .ok_or("range self-play probability is not numeric")?;
            if !probability.is_finite() || probability <= 0.0 {
                return Err("range self-play policy lacks finite full support".into());
            }
            minimum_probability = minimum_probability.min(probability);
            sum += probability;
        }
        maximum_sum_error = maximum_sum_error.max((sum - 1.0).abs());
        for value in values {
            let value = value
                .as_f64()
                .ok_or("range self-play action value is not numeric")?;
            if !value.is_finite() || value.abs() > config.game.effective_stack_bb + 1e-6 {
                return Err("range self-play action value is invalid".into());
            }
            minimum_action_value = minimum_action_value.min(value);
            maximum_action_value = maximum_action_value.max(value);
        }
        for standard_error in standard_errors {
            let standard_error = standard_error
                .as_f64()
                .ok_or("range self-play standard error is not numeric")?;
            if !standard_error.is_finite() || standard_error < 0.0 {
                return Err("range self-play standard error is invalid".into());
            }
            maximum_standard_error = maximum_standard_error.max(standard_error);
        }
    }
    if retained_records_by_street.contains(&0)
        || maximum_sum_error > 1e-6
        || !minimum_probability.is_finite()
        || !minimum_action_value.is_finite()
        || !maximum_action_value.is_finite()
    {
        return Err("range self-play failed its integrity gates".into());
    }

    if let Some(parent) = config.output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let temporary = config.output.with_extension("tmp");
    let file = fs::File::create(&temporary)?;
    let buffered = BufWriter::new(file);
    let mut writer = GzEncoder::new(buffered, Compression::fast());
    let metadata = serde_json::json!({
        "record_type": "metadata",
        "schema": "hu-range-conditioned-self-play-regret-jsonl-v1",
        "state_feature_schema": RANGE_POLICY_FEATURE_SCHEMA_V2,
        "state_feature_count": RANGE_POLICY_CONTEXT_V2_COUNT,
        "action_feature_schema": "hu-cash-legal-action-v1",
        "action_feature_count": ACTION_FEATURE_COUNT,
        "depth_bb": config.game.effective_stack_bb,
        "seed": config.seed,
        "start_iteration": config.start_iteration,
        "traversals": config.traversals,
        "deals": config.traversals,
        "records": records.len(),
        "candidate_records": attempted_records,
        "truncated": records.len() < attempted_records,
        "sampling_mode": "alternating_traverser_external_sampling_current_profile",
        "evaluates_trajectory_action_values": true,
        "value_rollouts_per_action": config.value_rollouts_per_action,
        "action_value_method": "independent_external_sampling_current_profile_mean",
        "policy_objective": "bilateral_range_conditioned_self_play_mirror_descent",
        "source_network_sha256": network_sha256,
        "source_range_policy_sha256": range_policy_sha256,
        "uses_exact_ranges": true,
        "focal_combo_attribution": true,
        "preflop_policy_frozen": true,
        "postflop_only": true,
        "enumerates_turn_river_chance": config.enumerate_turn_river_chance,
        "action_abstraction": config.game.action_abstraction,
    });
    serde_json::to_writer(&mut writer, &metadata)?;
    writer.write_all(b"\n")?;
    for record in &records {
        writer.write_all(record)?;
        writer.write_all(b"\n")?;
    }
    writer.finish()?.flush()?;
    fs::rename(&temporary, &config.output)?;
    let output_sha256 = sha256_path(&config.output)?;
    Ok(RangeSelfPlaySampleReport {
        schema: "hu-range-conditioned-self-play-regret-report-v1",
        method: "alternating_traverser_external_sampling_with_exact_public_range_reconstruction",
        depth_bb: config.game.effective_stack_bb,
        traversals: config.traversals,
        start_iteration: config.start_iteration,
        seed: config.seed,
        network_sha256,
        range_policy_sha256,
        value_rollouts_per_action: config.value_rollouts_per_action,
        candidate_records: attempted_records,
        retained_records: records.len(),
        retained_records_by_street,
        truncated: records.len() < attempted_records,
        minimum_policy_action_probability: minimum_probability,
        maximum_probability_sum_error: maximum_sum_error,
        minimum_action_value_bb: minimum_action_value,
        maximum_action_value_bb: maximum_action_value,
        maximum_action_value_standard_error_bb: maximum_standard_error,
        output: config.output,
        output_sha256,
        validation_status: "accepted_for_directional_training",
    })
}

fn clairvoyant_response_value(
    generator: &SampleGenerator,
    state: GameState,
    deal: &Deal,
    responder: usize,
    visited_nodes: &mut u64,
) -> f64 {
    *visited_nodes += 1;
    if state.terminal.is_some() {
        let utility_p0 = complete_runout_utility_p0(&state, deal);
        return if responder == 0 {
            utility_p0
        } else {
            -utility_p0
        };
    }
    let actions = state.legal_actions(&generator.config.game);
    let strategy = generator.current_strategy(&state, deal, &actions);
    if state.actor == responder {
        actions
            .iter()
            .map(|action| {
                clairvoyant_response_value(
                    generator,
                    state.apply(action, &generator.config.game),
                    deal,
                    responder,
                    visited_nodes,
                )
            })
            .fold(f64::NEG_INFINITY, f64::max)
    } else {
        actions
            .iter()
            .zip(strategy)
            .map(|(action, probability)| {
                probability
                    * clairvoyant_response_value(
                        generator,
                        state.apply(action, &generator.config.game),
                        deal,
                        responder,
                        visited_nodes,
                    )
            })
            .sum()
    }
}

fn complete_runout_utility_p0(state: &GameState, deal: &Deal) -> f64 {
    match state.terminal.as_ref().expect("terminal utility") {
        Terminal::Fold { winner } => {
            if *winner == 0 {
                state.invested[1]
            } else {
                -state.invested[0]
            }
        }
        Terminal::Showdown => {
            let equity = showdown_result(&deal.holes, &deal.board);
            equity * state.invested[1] - (1.0 - equity) * state.invested[0]
        }
    }
}

fn sample_opponent_hidden_scenarios(
    template: &Deal,
    responder: usize,
    samples: u32,
    rng: &mut SplitMix64,
) -> Vec<Deal> {
    let responder_holes = template.holes[responder];
    let mut blocked = [false; 52];
    for card in responder_holes.into_iter().chain(template.board) {
        blocked[card as usize] = true;
    }
    let available = (0..52u8)
        .filter(|card| !blocked[*card as usize])
        .collect::<Vec<_>>();
    debug_assert_eq!(available.len(), 45);
    (0..samples)
        .map(|_| {
            let first_index = rng.index(available.len());
            let mut second_index = rng.index(available.len() - 1);
            if second_index >= first_index {
                second_index += 1;
            }
            let opponent_holes = [available[first_index], available[second_index]];
            let mut holes = [[0u8; 2]; 2];
            holes[responder] = responder_holes;
            holes[1 - responder] = opponent_holes;
            Deal::from_sampled_cards(holes, template.board)
        })
        .collect()
}

fn opponent_hidden_future_board_response_value(
    generator: &SampleGenerator,
    state: GameState,
    scenarios: &[Deal],
    weights: &[f64],
    responder: usize,
    visited_nodes: &mut u64,
) -> f64 {
    *visited_nodes += 1;
    debug_assert_eq!(scenarios.len(), weights.len());
    if state.terminal.is_some() {
        return scenarios
            .iter()
            .zip(weights)
            .map(|(deal, weight)| {
                let utility_p0 = complete_runout_utility_p0(&state, deal);
                weight
                    * if responder == 0 {
                        utility_p0
                    } else {
                        -utility_p0
                    }
            })
            .sum();
    }
    let actions = state.legal_actions(&generator.config.game);
    if state.actor == responder {
        return actions
            .iter()
            .map(|action| {
                opponent_hidden_future_board_response_value(
                    generator,
                    state.apply(action, &generator.config.game),
                    scenarios,
                    weights,
                    responder,
                    visited_nodes,
                )
            })
            .fold(f64::NEG_INFINITY, f64::max);
    }

    let mut branch_weights = vec![vec![0.0; scenarios.len()]; actions.len()];
    for (scenario_index, (deal, reach)) in scenarios.iter().zip(weights).enumerate() {
        if *reach <= 0.0 {
            continue;
        }
        let strategy = generator.current_strategy(&state, deal, &actions);
        for (action_index, probability) in strategy.into_iter().enumerate() {
            branch_weights[action_index][scenario_index] = reach * probability;
        }
    }
    actions
        .iter()
        .zip(branch_weights)
        .map(|(action, child_weights)| {
            opponent_hidden_future_board_response_value(
                generator,
                state.apply(action, &generator.config.game),
                scenarios,
                &child_weights,
                responder,
                visited_nodes,
            )
        })
        .sum()
}

fn sample_without_replacement(
    available: &[u8],
    count: usize,
    rng: &mut SplitMix64,
) -> (Vec<u8>, Vec<u8>) {
    debug_assert!(count <= available.len());
    let mut deck = available.to_vec();
    for index in 0..count {
        let swap = index + rng.index(deck.len() - index);
        deck.swap(index, swap);
    }
    (deck[..count].to_vec(), deck[count..].to_vec())
}

/// Draw a nested empirical public chance tree conditional only on the
/// responder's private hand.
///
/// Every sampled terminal deal has the exact conditional card distribution.
/// Reusing each sampled flop for several turns, each turn for several rivers,
/// and each river for several hidden hands gives the empirical responder real
/// information sets instead of revealing a complete runout at the root.
fn sample_causal_scenarios(
    responder_holes: [u8; 2],
    responder: usize,
    public_branches_per_street: u32,
    opponent_samples_per_runout: u32,
    rng: &mut SplitMix64,
) -> Vec<Deal> {
    let available = (0..52u8)
        .filter(|card| !responder_holes.contains(card))
        .collect::<Vec<_>>();
    let scenario_count = (public_branches_per_street as usize)
        .checked_pow(3)
        .and_then(|count| count.checked_mul(opponent_samples_per_runout as usize))
        .expect("validated causal scenario count");
    let mut scenarios = Vec::with_capacity(scenario_count);
    for _ in 0..public_branches_per_street {
        let (flop, after_flop) = sample_without_replacement(&available, 3, rng);
        for _ in 0..public_branches_per_street {
            let (turn, after_turn) = sample_without_replacement(&after_flop, 1, rng);
            for _ in 0..public_branches_per_street {
                let (river, after_river) = sample_without_replacement(&after_turn, 1, rng);
                for _ in 0..opponent_samples_per_runout {
                    let (opponent_holes, _) = sample_without_replacement(&after_river, 2, rng);
                    let mut holes = [[0u8; 2]; 2];
                    holes[responder] = responder_holes;
                    holes[1 - responder] = [opponent_holes[0], opponent_holes[1]];
                    scenarios.push(Deal::from_sampled_cards(
                        holes,
                        [flop[0], flop[1], flop[2], turn[0], river[0]],
                    ));
                }
            }
        }
    }
    scenarios
}

fn observed_public_board(deal: &Deal, street: Street) -> Vec<u8> {
    let board_len = street.board_len();
    let mut key = deal.board[..board_len].to_vec();
    // A flop is an unordered public set. Turn and river remain chronological.
    key[..board_len.min(3)].sort_unstable();
    key
}

fn causal_sample_game_response_value(
    generator: &SampleGenerator,
    state: GameState,
    scenarios: &[Deal],
    weights: &[f64],
    responder: usize,
    visited_nodes: &mut u64,
) -> f64 {
    *visited_nodes += 1;
    debug_assert_eq!(scenarios.len(), weights.len());
    if weights.iter().all(|weight| *weight <= 0.0) {
        return 0.0;
    }
    if state.terminal.is_some() {
        return scenarios
            .iter()
            .zip(weights)
            .map(|(deal, weight)| {
                let utility_p0 = complete_runout_utility_p0(&state, deal);
                weight
                    * if responder == 0 {
                        utility_p0
                    } else {
                        -utility_p0
                    }
            })
            .sum();
    }
    let actions = state.legal_actions(&generator.config.game);
    if state.actor == responder {
        let mut information_sets: BTreeMap<Vec<u8>, Vec<f64>> = BTreeMap::new();
        for (index, (scenario, reach)) in scenarios.iter().zip(weights).enumerate() {
            if *reach <= 0.0 {
                continue;
            }
            information_sets
                .entry(observed_public_board(scenario, state.street))
                .or_insert_with(|| vec![0.0; scenarios.len()])[index] = *reach;
        }
        return information_sets
            .into_values()
            .map(|information_set_weights| {
                actions
                    .iter()
                    .map(|action| {
                        causal_sample_game_response_value(
                            generator,
                            state.apply(action, &generator.config.game),
                            scenarios,
                            &information_set_weights,
                            responder,
                            visited_nodes,
                        )
                    })
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .sum();
    }

    let mut branch_weights = vec![vec![0.0; scenarios.len()]; actions.len()];
    for (scenario_index, (deal, reach)) in scenarios.iter().zip(weights).enumerate() {
        if *reach <= 0.0 {
            continue;
        }
        let strategy = generator.current_strategy(&state, deal, &actions);
        for (action_index, probability) in strategy.into_iter().enumerate() {
            branch_weights[action_index][scenario_index] = reach * probability;
        }
    }
    actions
        .iter()
        .zip(branch_weights)
        .map(|(action, child_weights)| {
            causal_sample_game_response_value(
                generator,
                state.apply(action, &generator.config.game),
                scenarios,
                &child_weights,
                responder,
                visited_nodes,
            )
        })
        .sum()
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct CausalResponseInformationSet {
    public_history: Vec<String>,
    observed_board: Vec<u8>,
}

type CausalResponsePlan = BTreeMap<CausalResponseInformationSet, usize>;

fn merge_causal_response_plan(
    target: &mut CausalResponsePlan,
    source: CausalResponsePlan,
) -> Result<(), String> {
    for (information_set, action) in source {
        if let Some(previous) = target.insert(information_set, action) {
            if previous != action {
                return Err("causal response plan selected conflicting actions".to_owned());
            }
        }
    }
    Ok(())
}

/// Solve the same nested sample game as the causal certificate while retaining
/// one deterministic action for every reached responder information set.
/// Descendant plans below unselected responder actions are deliberately
/// discarded: the retained map represents one valid subgradient response, not
/// a collection of independently re-optimized deviations.
fn solve_causal_response_plan(
    generator: &SampleGenerator,
    state: GameState,
    scenarios: &[Deal],
    weights: &[f64],
    responder: usize,
    visited_nodes: &mut u64,
) -> Result<(f64, CausalResponsePlan), String> {
    *visited_nodes += 1;
    if scenarios.len() != weights.len() {
        return Err("causal response scenarios and weights differ".to_owned());
    }
    if weights.iter().all(|weight| *weight <= 0.0) {
        return Ok((0.0, BTreeMap::new()));
    }
    if state.terminal.is_some() {
        let value = scenarios
            .iter()
            .zip(weights)
            .map(|(deal, weight)| {
                let utility_p0 = complete_runout_utility_p0(&state, deal);
                weight
                    * if responder == 0 {
                        utility_p0
                    } else {
                        -utility_p0
                    }
            })
            .sum();
        return Ok((value, BTreeMap::new()));
    }
    let actions = state.legal_actions(&generator.config.game);
    if state.actor == responder {
        let mut information_sets: BTreeMap<Vec<u8>, Vec<f64>> = BTreeMap::new();
        for (index, (scenario, reach)) in scenarios.iter().zip(weights).enumerate() {
            if *reach <= 0.0 {
                continue;
            }
            information_sets
                .entry(observed_public_board(scenario, state.street))
                .or_insert_with(|| vec![0.0; scenarios.len()])[index] = *reach;
        }
        let mut total = 0.0;
        let mut plan = BTreeMap::new();
        for (observed_board, information_set_weights) in information_sets {
            let mut best: Option<(usize, f64, CausalResponsePlan)> = None;
            for (action_index, action) in actions.iter().enumerate() {
                let (value, child_plan) = solve_causal_response_plan(
                    generator,
                    state.apply(action, &generator.config.game),
                    scenarios,
                    &information_set_weights,
                    responder,
                    visited_nodes,
                )?;
                if best
                    .as_ref()
                    .is_none_or(|(_, best_value, _)| value > *best_value)
                {
                    best = Some((action_index, value, child_plan));
                }
            }
            let (selected_action, value, child_plan) =
                best.ok_or_else(|| "causal responder has no legal action".to_owned())?;
            total += value;
            merge_causal_response_plan(&mut plan, child_plan)?;
            let information_set = CausalResponseInformationSet {
                public_history: state.public_history.clone(),
                observed_board,
            };
            if let Some(previous) = plan.insert(information_set, selected_action) {
                if previous != selected_action {
                    return Err("causal response information set is inconsistent".to_owned());
                }
            }
        }
        return Ok((total, plan));
    }

    let mut branch_weights = vec![vec![0.0; scenarios.len()]; actions.len()];
    for (scenario_index, (deal, reach)) in scenarios.iter().zip(weights).enumerate() {
        if *reach <= 0.0 {
            continue;
        }
        let strategy = generator.current_strategy(&state, deal, &actions);
        for (action_index, probability) in strategy.into_iter().enumerate() {
            branch_weights[action_index][scenario_index] = reach * probability;
        }
    }
    let mut total = 0.0;
    let mut plan = BTreeMap::new();
    for (action, child_weights) in actions.iter().zip(branch_weights) {
        let (value, child_plan) = solve_causal_response_plan(
            generator,
            state.apply(action, &generator.config.game),
            scenarios,
            &child_weights,
            responder,
            visited_nodes,
        )?;
        total += value;
        merge_causal_response_plan(&mut plan, child_plan)?;
    }
    Ok((total, plan))
}

type AttributionReservoirKey = (u64, u64, u64, u8, u64);

struct BoundedCausalAttributionCollector {
    capacities: [usize; 3],
    seed: u64,
    seen_by_street: [usize; 3],
    records: [BTreeMap<AttributionReservoirKey, Vec<u8>>; 3],
}

impl BoundedCausalAttributionCollector {
    fn new(capacities: [usize; 3], seed: u64) -> Self {
        Self {
            capacities,
            seed,
            seen_by_street: [0; 3],
            records: std::array::from_fn(|_| BTreeMap::new()),
        }
    }

    fn street_index(street: Street) -> usize {
        match street {
            Street::Flop => 0,
            Street::Turn => 1,
            Street::River => 2,
            Street::Preflop => unreachable!("causal attribution retains only postflop rows"),
        }
    }

    fn consider(
        &mut self,
        street: Street,
        deal_index: u64,
        responder: usize,
        candidate_index: u64,
        record: Vec<u8>,
    ) {
        let street_index = Self::street_index(street);
        self.seen_by_street[street_index] += 1;
        let capacity = self.capacities[street_index];
        if capacity == 0 {
            return;
        }
        let mut digest = Sha256::new();
        digest.update(b"hu-causal-policy-attribution-reservoir-v1");
        digest.update(self.seed.to_le_bytes());
        digest.update(deal_index.to_le_bytes());
        digest.update([responder as u8, street_index as u8]);
        digest.update(candidate_index.to_le_bytes());
        digest.update(&record);
        let bytes = digest.finalize();
        let key = (
            u64::from_le_bytes(bytes[..8].try_into().expect("SHA-256 prefix")),
            u64::from_le_bytes(bytes[8..16].try_into().expect("SHA-256 suffix")),
            deal_index,
            responder as u8,
            candidate_index,
        );
        let records = &mut self.records[street_index];
        if records.len() < capacity {
            records.insert(key, record);
            return;
        }
        let largest = *records
            .last_key_value()
            .expect("positive attribution-record capacity")
            .0;
        if key < largest {
            records.pop_last();
            records.insert(key, record);
        }
    }

    fn into_records(self) -> (Vec<Vec<u8>>, [usize; 3], [usize; 3]) {
        let retained_by_street = self.records.each_ref().map(BTreeMap::len);
        let mut output = Vec::with_capacity(retained_by_street.iter().sum());
        for (street, records) in self.records.into_iter().enumerate() {
            let retained = records.len();
            if retained == 0 {
                continue;
            }
            let inclusion_correction = self.seen_by_street[street] as f64 / retained as f64;
            for record in records.into_values() {
                let mut value: serde_json::Value = serde_json::from_slice(&record)
                    .expect("generated causal attribution remains valid JSON");
                let weight = value["weight"]
                    .as_f64()
                    .expect("causal attribution weight is numeric");
                value["weight"] = serde_json::json!(weight * inclusion_correction);
                output.push(
                    serde_json::to_vec(&value)
                        .expect("corrected causal attribution remains serializable"),
                );
            }
        }
        (output, self.seen_by_street, retained_by_street)
    }
}

struct CausalAttributionWalkStats {
    attribution_nodes: u64,
    minimum_policy_probability: f64,
    maximum_target_sum_error: f64,
    minimum_policy_value: f64,
    maximum_policy_value: f64,
}

#[derive(Serialize)]
struct CausalRangePolicyState {
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

#[derive(Serialize)]
struct CausalRangePolicyAttributionRecord {
    record_type: &'static str,
    weight: f32,
    state: CausalRangePolicyState,
    ranges: [Vec<f32>; 2],
    focal_combo: usize,
    action_labels: Vec<String>,
    action_features: Vec<Vec<f32>>,
    probabilities: Vec<f32>,
    action_values_bb: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    action_value_standard_errors_bb: Option<Vec<f32>>,
}

#[allow(clippy::too_many_arguments)]
fn range_policy_directional_record_bytes(
    record_type: &'static str,
    generator: &SampleGenerator,
    state: &GameState,
    deal: &Deal,
    actions: &[LegalAction],
    probabilities: &[f64],
    action_values_bb: &[f64],
    action_value_standard_errors_bb: Option<&[f64]>,
    weight: f32,
) -> Result<Vec<u8>, String> {
    let policy = generator
        .networks
        .as_ref()
        .ok_or_else(|| "causal range attribution requires a frozen policy".to_owned())?;
    let public = policy.range_public_state(state, deal, actions, &generator.config.game)?;
    let focal_combo = Combo::new(deal.holes[state.actor][0], deal.holes[state.actor][1]).key();
    let record = CausalRangePolicyAttributionRecord {
        record_type,
        weight,
        state: CausalRangePolicyState {
            board: public.board,
            street: public.street,
            actor: public.actor,
            invested_bb: public.invested_bb,
            street_invested_bb: public.street_invested_bb,
            last_full_raise_bb: public.last_full_raise_bb,
            aggressions: public.aggressions,
            checks: public.checks,
            raise_reopened: public.raise_reopened,
            public_history: public.public_history,
            trajectory: public.trajectory,
        },
        ranges: public.ranges.map(|range| {
            range
                .into_iter()
                .map(|value| value as f32)
                .collect::<Vec<_>>()
        }),
        focal_combo,
        action_labels: actions.iter().map(|action| action.label.clone()).collect(),
        action_features: actions
            .iter()
            .map(|action| encode_action_features(state, action, &generator.config.game))
            .collect(),
        probabilities: probabilities.iter().map(|value| *value as f32).collect(),
        action_values_bb: action_values_bb.iter().map(|value| *value as f32).collect(),
        action_value_standard_errors_bb: action_value_standard_errors_bb
            .map(|values| values.iter().map(|value| *value as f32).collect::<Vec<_>>()),
    };
    serde_json::to_vec(&record)
        .map_err(|error| format!("causal range attribution is not serializable: {error}"))
}

impl Default for CausalAttributionWalkStats {
    fn default() -> Self {
        Self {
            attribution_nodes: 0,
            minimum_policy_probability: f64::INFINITY,
            maximum_target_sum_error: 0.0,
            minimum_policy_value: f64::INFINITY,
            maximum_policy_value: f64::NEG_INFINITY,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn fixed_causal_response_policy_values(
    generator: &SampleGenerator,
    state: GameState,
    scenarios: &[Deal],
    weights: &[f64],
    responder: usize,
    response_plan: &CausalResponsePlan,
    deal_index: u64,
    candidate_index: &mut u64,
    objective_scale: f64,
    collector: &mut BoundedCausalAttributionCollector,
    stats: &mut CausalAttributionWalkStats,
) -> Result<Vec<f64>, String> {
    stats.attribution_nodes += 1;
    if scenarios.len() != weights.len() {
        return Err("causal attribution scenarios and weights differ".to_owned());
    }
    if state.terminal.is_some() {
        return Ok(scenarios
            .iter()
            .map(|deal| {
                let utility_p0 = complete_runout_utility_p0(&state, deal);
                if responder == 0 {
                    utility_p0
                } else {
                    -utility_p0
                }
            })
            .collect());
    }
    if weights.iter().all(|weight| *weight <= 0.0) {
        return Ok(vec![0.0; scenarios.len()]);
    }
    let actions = state.legal_actions(&generator.config.game);
    if state.actor == responder {
        let mut information_sets: BTreeMap<Vec<u8>, Vec<f64>> = BTreeMap::new();
        for (index, (scenario, reach)) in scenarios.iter().zip(weights).enumerate() {
            if *reach <= 0.0 {
                continue;
            }
            information_sets
                .entry(observed_public_board(scenario, state.street))
                .or_insert_with(|| vec![0.0; scenarios.len()])[index] = *reach;
        }
        let mut output = vec![0.0; scenarios.len()];
        for (observed_board, information_set_weights) in information_sets {
            let key = CausalResponseInformationSet {
                public_history: state.public_history.clone(),
                observed_board,
            };
            let selected = *response_plan
                .get(&key)
                .ok_or_else(|| "fixed causal response plan is missing a reached node".to_owned())?;
            if selected >= actions.len() {
                return Err("fixed causal response selected an illegal action".to_owned());
            }
            let child = fixed_causal_response_policy_values(
                generator,
                state.apply(&actions[selected], &generator.config.game),
                scenarios,
                &information_set_weights,
                responder,
                response_plan,
                deal_index,
                candidate_index,
                objective_scale,
                collector,
                stats,
            )?;
            for (index, reach) in information_set_weights.iter().enumerate() {
                if *reach > 0.0 {
                    output[index] = child[index];
                }
            }
        }
        return Ok(output);
    }

    let mut strategies = vec![Vec::new(); scenarios.len()];
    let mut branch_weights = vec![vec![0.0; scenarios.len()]; actions.len()];
    for (scenario_index, (deal, reach)) in scenarios.iter().zip(weights).enumerate() {
        if *reach <= 0.0 {
            continue;
        }
        let strategy = generator.current_strategy(&state, deal, &actions);
        if strategy.len() != actions.len()
            || strategy
                .iter()
                .any(|probability| !probability.is_finite() || *probability <= 0.0)
        {
            return Err(
                "causal attribution requires a finite full-support frozen policy".to_owned(),
            );
        }
        let probability_sum = strategy.iter().sum::<f64>();
        stats.maximum_target_sum_error = stats
            .maximum_target_sum_error
            .max((probability_sum - 1.0).abs());
        stats.minimum_policy_probability = stats
            .minimum_policy_probability
            .min(strategy.iter().copied().fold(f64::INFINITY, f64::min));
        for (action_index, probability) in strategy.iter().copied().enumerate() {
            branch_weights[action_index][scenario_index] = reach * probability;
        }
        strategies[scenario_index] = strategy;
    }

    let mut action_values = Vec::with_capacity(actions.len());
    for (action, child_weights) in actions.iter().zip(branch_weights) {
        action_values.push(fixed_causal_response_policy_values(
            generator,
            state.apply(action, &generator.config.game),
            scenarios,
            &child_weights,
            responder,
            response_plan,
            deal_index,
            candidate_index,
            objective_scale,
            collector,
            stats,
        )?);
    }

    let mut output = vec![0.0; scenarios.len()];
    for (scenario_index, ((deal, reach), strategy)) in
        scenarios.iter().zip(weights).zip(&strategies).enumerate()
    {
        if *reach <= 0.0 {
            continue;
        }
        let responder_values = action_values
            .iter()
            .map(|values| values[scenario_index])
            .collect::<Vec<_>>();
        if responder_values.iter().any(|value| {
            !value.is_finite() || value.abs() > generator.config.game.effective_stack_bb + 1e-8
        }) {
            return Err("causal attribution produced an invalid action value".to_owned());
        }
        output[scenario_index] = strategy
            .iter()
            .zip(&responder_values)
            .map(|(probability, value)| probability * value)
            .sum();
        if state.street != Street::Preflop {
            let policy_values = responder_values
                .into_iter()
                .map(|value| -value)
                .collect::<Vec<_>>();
            stats.minimum_policy_value = stats
                .minimum_policy_value
                .min(policy_values.iter().copied().fold(f64::INFINITY, f64::min));
            stats.maximum_policy_value = stats.maximum_policy_value.max(
                policy_values
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max),
            );
            let record_weight = (*reach * objective_scale) as f32;
            let record = if generator
                .networks
                .as_ref()
                .is_some_and(|policy| policy.range_policy.is_some())
            {
                range_policy_directional_record_bytes(
                    "range_conditioned_causal_policy_attribution",
                    generator,
                    &state,
                    deal,
                    &actions,
                    strategy,
                    &policy_values,
                    None,
                    record_weight,
                )?
            } else {
                let sample = training_sample(
                    SampleKind::AverageStrategy,
                    deal_index,
                    record_weight,
                    *reach,
                    &state,
                    deal,
                    &actions,
                    strategy.clone(),
                    Some(policy_values),
                    None,
                    &generator.config.game,
                );
                serde_json::to_vec(&sample).expect("causal policy attribution is serializable")
            };
            collector.consider(
                state.street,
                deal_index,
                responder,
                *candidate_index,
                record,
            );
            *candidate_index += 1;
        }
    }
    Ok(output)
}

fn attribution_street_capacities(capacity: usize) -> [usize; 3] {
    let flop = (capacity / 4).max(1);
    let turn = (capacity * 3 / 10).max(1);
    [flop, turn, capacity - flop - turn]
}

fn split_attribution_capacities(total: [usize; 3], worker: usize, workers: usize) -> [usize; 3] {
    total.map(|capacity| capacity / workers + usize::from(worker < capacity % workers))
}

fn sha256_path(path: &std::path::Path) -> Result<String, Box<dyn Error>> {
    let mut reader = BufReader::new(fs::File::open(path)?);
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

/// Export policy-player action values from the exact causal sample games used
/// by the conservative exploitability certificate. The responder plan is
/// solved once, frozen, and replayed so every row is a valid subgradient of
/// that same sampled maximum. Values are negated into the policy player's
/// utility direction and preflop rows are intentionally omitted.
pub fn generate_causal_policy_attribution(
    config: CausalPolicyAttributionConfig,
) -> Result<CausalPolicyAttributionReport, Box<dyn Error>> {
    if config.deals == 0 {
        return Err("causal attribution requires at least one deal".into());
    }
    if config.threads == 0 {
        return Err("causal attribution thread count must be positive".into());
    }
    if config.public_branches_per_street == 0 || config.opponent_samples_per_runout == 0 {
        return Err("causal attribution requires public and hidden-hand samples".into());
    }
    if config.max_records < 3 {
        return Err("causal attribution requires at least three retained records".into());
    }
    let scenarios_per_deal = u64::from(config.public_branches_per_street)
        .checked_pow(3)
        .and_then(|count| count.checked_mul(u64::from(config.opponent_samples_per_runout)))
        .ok_or("causal attribution scenario count overflows")?;
    if scenarios_per_deal > 1_000_000 {
        return Err("causal attribution exceeds one million scenarios per deal".into());
    }
    let network_sha256 = sha256_path(&config.network_path)?;
    let range_policy_sha256 = config
        .range_policy_path
        .as_ref()
        .map(|path| sha256_path(path))
        .transpose()?;
    let generator = SampleGenerator::new_with_range(
        SampleGenerationConfig {
            game: config.game.clone(),
            traversals: 1,
            start_iteration: 0,
            seed: config.seed,
            max_records: 1,
            output: PathBuf::from("unused-causal-attribution.jsonl.gz"),
            network_path: Some(config.network_path.clone()),
            trajectory_sampling: false,
            evaluate_trajectory_values: false,
            value_rollouts_per_action: 1,
            enumerate_turn_river_chance: false,
        },
        config.range_policy_path.as_deref(),
    )?;
    let total_capacities = attribution_street_capacities(config.max_records);
    if total_capacities[2] == 0 {
        return Err("causal attribution record budget cannot cover every postflop street".into());
    }
    let worker_count = config.threads.min(config.deals as usize).min(
        *total_capacities
            .iter()
            .min()
            .expect("three street capacities"),
    );
    let mut rng = SplitMix64::new(config.seed);
    let responder_holes = (0..config.deals)
        .map(|index| (index, Deal::sample(&mut rng).holes))
        .collect::<Vec<_>>();
    let chunk_size = responder_holes.len().div_ceil(worker_count);
    let chunks = responder_holes
        .chunks(chunk_size)
        .map(<[(u64, [[u8; 2]; 2])]>::to_vec)
        .collect::<Vec<_>>();
    let objective_scale = 1.0 / (2.0 * config.deals as f64);
    let worker_results = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for (worker, chunk) in chunks.into_iter().enumerate() {
            let generator = &generator;
            let game = &config.game;
            let capacities = split_attribution_capacities(total_capacities, worker, worker_count);
            handles.push(scope.spawn(move || -> Result<_, String> {
                let mut collector =
                    BoundedCausalAttributionCollector::new(capacities, config.seed ^ worker as u64);
                let mut response_nodes = 0u64;
                let mut stats = CausalAttributionWalkStats::default();
                let mut deal_results = Vec::with_capacity(chunk.len());
                for (index, sampled_holes) in chunk {
                    let mut responses = [0.0; 2];
                    let mut maximum_reconstruction_error = 0.0f64;
                    for responder in 0..2 {
                        let seed = config.seed
                            ^ index.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                            ^ (responder as u64 + 1).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                        let mut scenario_rng = SplitMix64::new(seed);
                        let scenarios = sample_causal_scenarios(
                            sampled_holes[responder],
                            responder,
                            config.public_branches_per_street,
                            config.opponent_samples_per_runout,
                            &mut scenario_rng,
                        );
                        let weights = vec![1.0 / scenarios.len() as f64; scenarios.len()];
                        let (response, plan) = solve_causal_response_plan(
                            generator,
                            GameState::initial(game),
                            &scenarios,
                            &weights,
                            responder,
                            &mut response_nodes,
                        )?;
                        let mut candidate_index = 0u64;
                        let reconstructed_values = fixed_causal_response_policy_values(
                            generator,
                            GameState::initial(game),
                            &scenarios,
                            &weights,
                            responder,
                            &plan,
                            index,
                            &mut candidate_index,
                            objective_scale,
                            &mut collector,
                            &mut stats,
                        )?;
                        let reconstructed = weights
                            .iter()
                            .zip(reconstructed_values)
                            .map(|(weight, value)| weight * value)
                            .sum::<f64>();
                        maximum_reconstruction_error =
                            maximum_reconstruction_error.max((response - reconstructed).abs());
                        responses[responder] = response;
                    }
                    deal_results.push((
                        index,
                        ((responses[0] + responses[1]) / 2.0).clamp(0.0, game.effective_stack_bb),
                        maximum_reconstruction_error,
                    ));
                }
                Ok((collector, response_nodes, stats, deal_results))
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().expect("causal attribution worker panicked"))
            .collect::<Result<Vec<_>, _>>()
    })?;

    let mut records = Vec::new();
    let mut candidate_records_by_street = [0usize; 3];
    let mut retained_records_by_street = [0usize; 3];
    let mut response_tree_nodes = 0u64;
    let mut attribution_tree_nodes = 0u64;
    let mut minimum_probability = f64::INFINITY;
    let mut maximum_sum_error = 0.0f64;
    let mut minimum_policy_value = f64::INFINITY;
    let mut maximum_policy_value = f64::NEG_INFINITY;
    let mut deal_results = Vec::with_capacity(config.deals as usize);
    for (collector, response_nodes, stats, worker_deals) in worker_results {
        let (mut worker_records, candidate_by_street, retained_by_street) =
            collector.into_records();
        records.append(&mut worker_records);
        for street in 0..3 {
            candidate_records_by_street[street] += candidate_by_street[street];
            retained_records_by_street[street] += retained_by_street[street];
        }
        response_tree_nodes += response_nodes;
        attribution_tree_nodes += stats.attribution_nodes;
        minimum_probability = minimum_probability.min(stats.minimum_policy_probability);
        maximum_sum_error = maximum_sum_error.max(stats.maximum_target_sum_error);
        minimum_policy_value = minimum_policy_value.min(stats.minimum_policy_value);
        maximum_policy_value = maximum_policy_value.max(stats.maximum_policy_value);
        deal_results.extend(worker_deals);
    }
    deal_results.sort_by_key(|(index, _, _)| *index);
    let sample_mean_exploitability =
        deal_results.iter().map(|(_, value, _)| value).sum::<f64>() / config.deals as f64;
    let maximum_reconstruction_error = deal_results
        .iter()
        .map(|(_, _, error)| *error)
        .fold(0.0f64, f64::max);
    if records.is_empty()
        || records.len() > config.max_records
        || maximum_reconstruction_error > 1e-8 * config.game.effective_stack_bb.max(1.0)
        || maximum_sum_error > 1e-6
        || !minimum_probability.is_finite()
        || minimum_probability <= 0.0
        || !minimum_policy_value.is_finite()
        || !maximum_policy_value.is_finite()
    {
        return Err("causal attribution failed its integrity gates".into());
    }
    if let Some(parent) = config.output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let temporary = config.output.with_extension("tmp");
    let file = fs::File::create(&temporary)?;
    let buffered = BufWriter::new(file);
    let mut writer = GzEncoder::new(buffered, Compression::fast());
    let candidate_records = candidate_records_by_street.iter().sum::<usize>();
    let range_conditioned = range_policy_sha256.is_some();
    let metadata = serde_json::json!({
        "record_type": "metadata",
        "schema": if range_conditioned {
            "hu-range-conditioned-causal-policy-attribution-jsonl-v1"
        } else {
            "hu-neural-causal-policy-attribution-jsonl-v1"
        },
        "state_feature_schema": if range_conditioned {
            RANGE_POLICY_FEATURE_SCHEMA_V2
        } else {
            "hu-cash-trajectory-poker-aware-v4"
        },
        "state_feature_count": if range_conditioned {
            RANGE_POLICY_CONTEXT_V2_COUNT
        } else {
            STATE_FEATURE_COUNT
        },
        "action_feature_schema": "hu-cash-legal-action-v1",
        "action_feature_count": ACTION_FEATURE_COUNT,
        "depth_bb": config.game.effective_stack_bb,
        "seed": config.seed,
        "deals": config.deals,
        "records": records.len(),
        "candidate_records": candidate_records,
        "truncated": records.len() < candidate_records,
        "sampling_mode": "thread_and_street_stratified_bottom_hash_reservoir",
        "evaluates_trajectory_action_values": true,
        "action_value_method": "negative_fixed_causal_sample_game_best_response_utility_subgradient",
        "policy_objective": "maximize_negated_responder_utility_with_a_trust_region",
        "public_branches_per_street": config.public_branches_per_street,
        "opponent_samples_per_runout": config.opponent_samples_per_runout,
        "scenarios_per_deal": scenarios_per_deal,
        "source_network_sha256": network_sha256,
        "source_range_policy_sha256": range_policy_sha256,
        "uses_exact_ranges": range_conditioned,
        "focal_combo_attribution": range_conditioned,
        "preflop_policy_frozen": true,
        "postflop_only": true,
        "maximum_root_value_reconstruction_error_bb": maximum_reconstruction_error,
        "action_abstraction": config.game.action_abstraction,
    });
    serde_json::to_writer(&mut writer, &metadata)?;
    writer.write_all(b"\n")?;
    for record in records {
        writer.write_all(&record)?;
        writer.write_all(b"\n")?;
    }
    writer.finish()?.flush()?;
    fs::rename(&temporary, &config.output)?;
    let output_sha256 = sha256_path(&config.output)?;
    Ok(CausalPolicyAttributionReport {
        schema: "hu-neural-causal-policy-attribution-report-v1",
        method: "fixed_information_set_causal_best_response_policy_action_subgradient",
        depth_bb: config.game.effective_stack_bb,
        deals: config.deals,
        seed: config.seed,
        network_sha256,
        range_policy_sha256,
        threads: worker_count,
        public_branches_per_street: config.public_branches_per_street,
        opponent_samples_per_runout: config.opponent_samples_per_runout,
        scenarios_per_deal,
        response_tree_nodes,
        attribution_tree_nodes,
        candidate_records,
        retained_records: retained_records_by_street.iter().sum(),
        candidate_records_by_street,
        retained_records_by_street,
        truncated: retained_records_by_street.iter().sum::<usize>() < candidate_records,
        sample_mean_exploitability_bb: sample_mean_exploitability,
        maximum_root_value_reconstruction_error_bb: maximum_reconstruction_error,
        minimum_frozen_policy_action_probability: minimum_probability,
        maximum_target_probability_sum_error: maximum_sum_error,
        minimum_policy_action_value_bb: minimum_policy_value,
        maximum_policy_action_value_bb: maximum_policy_value,
        output: config.output,
        output_sha256,
        validation_status: "accepted_for_trust_region_training",
    })
}

fn reconstruct_attribution_state(
    compact: &CompactState,
    game: &BlueprintConfig,
) -> Result<(GameState, Deal), String> {
    if compact.actor > 1
        || compact.button != 0
        || compact.board.len() != compact.street.board_len()
        || compact.trajectory.len() > MAX_TRAJECTORY_ACTIONS
    {
        return Err("causal attribution state is structurally invalid".to_owned());
    }
    let state = GameState {
        street: compact.street,
        actor: compact.actor,
        invested: compact.total_committed_bb.map(f64::from),
        street_invested: compact.street_bets_bb.map(f64::from),
        last_full_raise: f64::from(compact.last_full_raise_bb),
        aggressions: 0,
        checks: 0,
        raise_reopened: compact.raise_reopened,
        public_history: Vec::new(),
        trajectory: compact
            .trajectory
            .iter()
            .map(|action| TrajectoryAction {
                actor: action.actor,
                street: action.street,
                kind: action.kind,
                amount_bb: f64::from(action.amount_bb),
                amount_to_bb: action.amount_to_bb.map(f64::from),
                pot_after_bb: f64::from(action.pot_after_bb),
            })
            .collect(),
        terminal: None,
    };
    let settled_pot = state.pot() - state.street_invested[0] - state.street_invested[1];
    if (settled_pot - f64::from(compact.pot_bb)).abs() > 1e-4
        || (state.to_call() - f64::from(compact.to_call_bb)).abs() > 1e-4
        || (0..2).any(|player| {
            (state.remaining(player, game) - f64::from(compact.stacks_bb[player])).abs() > 1e-4
        })
    {
        return Err("causal attribution chip accounting is inconsistent".to_owned());
    }

    let mut used = [false; 52];
    for card in compact
        .private_cards
        .into_iter()
        .chain(compact.board.iter().copied())
    {
        if card >= 52 || used[card as usize] {
            return Err("causal attribution cards are invalid or duplicated".to_owned());
        }
        used[card as usize] = true;
    }
    let mut filler = (0..52u8).filter(|card| !used[*card as usize]);
    let opponent = [
        filler
            .next()
            .ok_or_else(|| "causal attribution has no filler card".to_owned())?,
        filler
            .next()
            .ok_or_else(|| "causal attribution has no filler card".to_owned())?,
    ];
    let mut holes = [[0u8; 2]; 2];
    holes[compact.actor] = compact.private_cards;
    holes[1 - compact.actor] = opponent;
    let mut board = [0u8; 5];
    board[..compact.board.len()].copy_from_slice(&compact.board);
    for card in &mut board[compact.board.len()..] {
        *card = filler
            .next()
            .ok_or_else(|| "causal attribution has no future-board filler".to_owned())?;
    }
    Ok((state, Deal::from_sampled_cards(holes, board)))
}

fn compact_attribution_features(
    compact: &CompactState,
    compact_actions: &[CompactLegalAction],
    state: &GameState,
    deal: &Deal,
    game: &BlueprintConfig,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let mut state_features = encode_state_features(state, deal, game);
    let depth = game.effective_stack_bb as f32;
    let actor = compact.actor;
    let opponent = 1 - actor;
    let scalars = [
        compact.pot_bb / depth,
        compact.stacks_bb[actor] / depth,
        compact.stacks_bb[opponent] / depth,
        compact.street_bets_bb[actor] / depth,
        compact.street_bets_bb[opponent] / depth,
        compact.total_committed_bb[actor] / depth,
        compact.total_committed_bb[opponent] / depth,
        compact.to_call_bb / depth,
        compact.last_full_raise_bb / depth,
        f32::from(compact.raise_reopened),
        compact.board.len() as f32 / 5.0,
        compact.trajectory.len() as f32 / MAX_TRAJECTORY_ACTIONS as f32,
    ];
    state_features[112..124].copy_from_slice(&scalars);
    let milliblind = |value: f64| (value * 1_000.0).round() / 1_000.0;
    let current = milliblind(f64::from(compact.street_bets_bb[actor]));
    let highest = compact
        .street_bets_bb
        .iter()
        .copied()
        .map(f64::from)
        .fold(0.0f64, f64::max);
    let highest = milliblind(highest);
    let pot = milliblind(
        f64::from(compact.pot_bb)
            + compact
                .street_bets_bb
                .iter()
                .copied()
                .map(f64::from)
                .sum::<f64>(),
    );
    let action_features = compact_actions
        .iter()
        .map(|action| {
            let mut features = vec![0.0f32; ACTION_FEATURE_COUNT];
            features[trajectory_kind_index(action.kind)] = 1.0;
            let target = match action.kind {
                TrajectoryActionKind::Call => highest,
                TrajectoryActionKind::Bet
                | TrajectoryActionKind::Raise
                | TrajectoryActionKind::AllIn => {
                    f64::from(action.amount_to_bb.expect("validated sized action target"))
                }
                TrajectoryActionKind::Fold | TrajectoryActionKind::Check => current,
            };
            let paid = if action.kind == TrajectoryActionKind::Call {
                milliblind(f64::from(compact.to_call_bb))
            } else {
                milliblind((target - current).max(0.0))
            };
            features[6] = target as f32 / depth;
            features[7] = paid as f32 / depth;
            features[8] = (paid / pot.max(1.0)) as f32;
            features
        })
        .collect();
    (state_features, action_features)
}

fn frozen_policy_from_features(
    policy: &FrozenPolicy,
    street: Street,
    actor: usize,
    state_features: &[f32],
    action_features: &[Vec<f32>],
) -> Vec<f64> {
    let scores = policy
        .bundle
        .policy_network(street, actor)
        .score_state_actions(state_features, action_features);
    match policy.bundle.strategy_transform {
        StrategyTransform::RegretMatching => {
            normalize_or_uniform(scores.into_iter().map(|value| value.max(0.0)).collect())
        }
        StrategyTransform::Softmax => stable_softmax(&scores),
    }
}

fn categorical_kl(first: &[f64], second: &[f64]) -> Result<f64, String> {
    if first.len() != second.len() || first.is_empty() {
        return Err("categorical distributions have incompatible shapes".to_owned());
    }
    let mut value = 0.0;
    for (probability, reference) in first.iter().zip(second) {
        if !probability.is_finite()
            || !reference.is_finite()
            || *probability <= 0.0
            || *reference <= 0.0
        {
            return Err("categorical distributions must have finite full support".to_owned());
        }
        value += probability * (probability / reference).ln();
    }
    Ok(value.max(0.0))
}

/// Score a causal-attribution corpus with the same deterministic Rust dense
/// inference used by the full-game certificate. This is the authoritative
/// pre-routing trust-region check; ML framework metrics remain diagnostics.
pub fn evaluate_causal_attribution_policy(
    config: CausalAttributionPolicyEvaluationConfig,
) -> Result<CausalAttributionPolicyEvaluation, Box<dyn Error>> {
    if !config.maximum_node_kl.is_finite()
        || config.maximum_node_kl <= 0.0
        || !config.maximum_weighted_kl.is_finite()
        || config.maximum_weighted_kl <= 0.0
        || !config.minimum_policy_value_gain_bb.is_finite()
        || config.minimum_policy_value_gain_bb <= 0.0
    {
        return Err("causal attribution KL bounds must be finite and positive".into());
    }
    let dataset_sha256 = sha256_path(&config.dataset_path)?;
    let candidate_network_sha256 = sha256_path(&config.network_path)?;
    let policy = FrozenPolicy::load(&config.network_path)?;
    let file = fs::File::open(&config.dataset_path)?;
    let mut lines = BufReader::new(flate2::read::GzDecoder::new(file)).lines();
    let metadata_line = lines
        .next()
        .ok_or("causal attribution dataset is empty")??;
    let metadata: serde_json::Value = serde_json::from_str(&metadata_line)?;
    if metadata["schema"] != "hu-neural-causal-policy-attribution-jsonl-v1"
        || metadata["state_feature_schema"] != "hu-cash-trajectory-poker-aware-v4"
        || metadata["state_feature_count"] != STATE_FEATURE_COUNT
        || metadata["action_feature_count"] != ACTION_FEATURE_COUNT
        || metadata["postflop_only"] != true
        || metadata["preflop_policy_frozen"] != true
    {
        return Err("causal attribution metadata is incompatible".into());
    }
    let records = metadata["records"]
        .as_u64()
        .ok_or("causal attribution metadata omits its record count")? as usize;
    let source_network_sha256 = metadata["source_network_sha256"]
        .as_str()
        .ok_or("causal attribution metadata omits its source network")?
        .to_owned();
    let depth = metadata["depth_bb"]
        .as_f64()
        .ok_or("causal attribution metadata omits its depth")?;
    let action_abstraction: ActionAbstraction =
        serde_json::from_value(metadata["action_abstraction"].clone())?;
    let mut game = BlueprintConfig::default();
    game.effective_stack_bb = depth;
    game.action_abstraction = action_abstraction;
    game.validate()
        .map_err(|error| format!("causal attribution game is invalid: {error}"))?;

    let mut observed = 0usize;
    let mut total_weight = 0.0;
    let mut baseline_value_sum = 0.0;
    let mut candidate_value_sum = 0.0;
    let mut reverse_kl_sum = 0.0;
    let mut forward_kl_sum = 0.0;
    let mut l1_sum = 0.0;
    let mut primary_agreement_sum = 0.0;
    let mut maximum_reverse_kl = 0.0f64;
    let mut maximum_forward_kl = 0.0f64;
    let mut maximum_l1 = 0.0f64;
    let mut maximum_baseline_sum_error = 0.0f64;
    let mut maximum_candidate_sum_error = 0.0f64;
    for line in lines {
        let sample: TrainingSample = serde_json::from_str(&line?)?;
        if sample.kind != SampleKind::AverageStrategy
            || sample.state.street == Street::Preflop
            || sample.actions.is_empty()
            || sample.actions.len() != sample.targets.len()
            || sample.actions.len() != sample.feature_sha256.len()
        {
            return Err("causal attribution record is structurally invalid".into());
        }
        let action_values = sample
            .action_values_bb
            .as_ref()
            .ok_or("causal attribution record omits policy action values")?;
        if action_values.len() != sample.actions.len()
            || action_values
                .iter()
                .any(|value| !value.is_finite() || f64::from(value.abs()) > depth + 1e-6)
            || !sample.weight.is_finite()
            || sample.weight <= 0.0
        {
            return Err("causal attribution values or weight are invalid".into());
        }
        let (state, deal) = reconstruct_attribution_state(&sample.state, &game)?;
        let (state_features, action_features) =
            compact_attribution_features(&sample.state, &sample.actions, &state, &deal, &game);
        for (action_index, ((features, expected_hash), compact_action)) in action_features
            .iter()
            .zip(&sample.feature_sha256)
            .zip(&sample.actions)
            .enumerate()
        {
            let mut complete_features = state_features.clone();
            complete_features.extend(features);
            let measured_hash = feature_sha256(&complete_features);
            if measured_hash != *expected_hash {
                return Err(format!(
                    "causal attribution feature hash differs at record {observed}, action {action_index}: expected {expected_hash}, measured {measured_hash}"
                )
                .into());
            }
            if matches!(
                compact_action.kind,
                TrajectoryActionKind::Bet
                    | TrajectoryActionKind::Raise
                    | TrajectoryActionKind::AllIn
            ) && compact_action.amount_to_bb.is_none()
            {
                return Err("causal attribution sized action has no target".into());
            }
        }
        let mut baseline = sample
            .targets
            .iter()
            .map(|value| f64::from(*value))
            .collect::<Vec<_>>();
        let baseline_sum = baseline.iter().sum::<f64>();
        maximum_baseline_sum_error = maximum_baseline_sum_error.max((baseline_sum - 1.0).abs());
        if baseline_sum <= 0.0
            || baseline
                .iter()
                .any(|probability| !probability.is_finite() || *probability <= 0.0)
        {
            return Err("causal attribution baseline probabilities are invalid".into());
        }
        for probability in &mut baseline {
            *probability /= baseline_sum;
        }
        let candidate = frozen_policy_from_features(
            &policy,
            sample.state.street,
            sample.state.actor,
            &state_features,
            &action_features,
        );
        let candidate_sum = candidate.iter().sum::<f64>();
        maximum_candidate_sum_error = maximum_candidate_sum_error.max((candidate_sum - 1.0).abs());
        let reverse_kl = categorical_kl(&candidate, &baseline)?;
        let forward_kl = categorical_kl(&baseline, &candidate)?;
        let l1 = candidate
            .iter()
            .zip(&baseline)
            .map(|(first, second)| (first - second).abs())
            .sum::<f64>();
        let baseline_value = baseline
            .iter()
            .zip(action_values)
            .map(|(probability, value)| probability * f64::from(*value))
            .sum::<f64>();
        let candidate_value = candidate
            .iter()
            .zip(action_values)
            .map(|(probability, value)| probability * f64::from(*value))
            .sum::<f64>();
        let weight = f64::from(sample.weight);
        total_weight += weight;
        baseline_value_sum += weight * baseline_value;
        candidate_value_sum += weight * candidate_value;
        reverse_kl_sum += weight * reverse_kl;
        forward_kl_sum += weight * forward_kl;
        l1_sum += weight * l1;
        let baseline_primary = baseline
            .iter()
            .enumerate()
            .max_by(|(_, first), (_, second)| first.total_cmp(second))
            .map(|(index, _)| index)
            .expect("nonempty baseline");
        let candidate_primary = candidate
            .iter()
            .enumerate()
            .max_by(|(_, first), (_, second)| first.total_cmp(second))
            .map(|(index, _)| index)
            .expect("nonempty candidate");
        if baseline_primary == candidate_primary {
            primary_agreement_sum += weight;
        }
        maximum_reverse_kl = maximum_reverse_kl.max(reverse_kl);
        maximum_forward_kl = maximum_forward_kl.max(forward_kl);
        maximum_l1 = maximum_l1.max(l1);
        observed += 1;
    }
    if observed != records || records == 0 || !total_weight.is_finite() || total_weight <= 0.0 {
        return Err("causal attribution observed record count or weight is invalid".into());
    }
    let weighted_baseline_value = baseline_value_sum / total_weight;
    let weighted_candidate_value = candidate_value_sum / total_weight;
    let weighted_gain = weighted_candidate_value - weighted_baseline_value;
    let weighted_reverse_kl = reverse_kl_sum / total_weight;
    let weighted_forward_kl = forward_kl_sum / total_weight;
    let probability_sums_valid =
        maximum_baseline_sum_error <= 1e-6 && maximum_candidate_sum_error <= 1e-12;
    let policy_value_improved = weighted_gain > config.minimum_policy_value_gain_bb;
    let maximum_node_kl_passed = maximum_reverse_kl <= config.maximum_node_kl;
    let weighted_kl_passed = weighted_reverse_kl <= config.maximum_weighted_kl;
    Ok(CausalAttributionPolicyEvaluation {
        schema: "hu-neural-causal-attribution-policy-evaluation-v1",
        method: "exact_rust_dense_inference_on_fixed_causal_response_action_values",
        depth_bb: depth,
        records,
        dataset_sha256,
        source_network_sha256,
        candidate_network_sha256,
        total_objective_weight: total_weight,
        weighted_baseline_policy_value_bb: weighted_baseline_value,
        weighted_candidate_policy_value_bb: weighted_candidate_value,
        weighted_policy_value_gain_bb: weighted_gain,
        weighted_reverse_kl_from_frozen: weighted_reverse_kl,
        maximum_reverse_kl_from_frozen: maximum_reverse_kl,
        weighted_forward_kl_from_frozen: weighted_forward_kl,
        maximum_forward_kl_from_frozen: maximum_forward_kl,
        weighted_l1_action_delta: l1_sum / total_weight,
        maximum_l1_action_delta: maximum_l1,
        weighted_primary_action_agreement: primary_agreement_sum / total_weight,
        maximum_baseline_probability_sum_error: maximum_baseline_sum_error,
        maximum_candidate_probability_sum_error: maximum_candidate_sum_error,
        minimum_policy_value_gain_bb: config.minimum_policy_value_gain_bb,
        feature_hashes_verified: true,
        policy_value_improved,
        maximum_node_kl_passed,
        weighted_kl_passed,
        accepted_for_routed_evaluation: policy_value_improved
            && maximum_node_kl_passed
            && weighted_kl_passed
            && probability_sums_valid,
    })
}

fn one_sided_empirical_bernstein_margin(
    sample_variance: f64,
    range: f64,
    samples: u64,
    confidence: f64,
) -> f64 {
    debug_assert!(sample_variance.is_finite() && sample_variance >= 0.0);
    debug_assert!(range.is_finite() && range > 0.0);
    debug_assert!(samples >= 2);
    debug_assert!((0.0..1.0).contains(&confidence));
    let log_term = (2.0 / (1.0 - confidence)).ln();
    (2.0 * sample_variance * log_term / samples as f64).sqrt()
        + 7.0 * range * log_term / (3.0 * (samples - 1) as f64)
}

/// Compute a conservative upper bound by relaxing the responder's information.
///
/// For every sampled complete deal, the responder observes both private hands
/// and the full runout, then solves the entire betting tree exactly against the
/// frozen policy. This responder contains every legal imperfect-information
/// response, so its expected value upper-bounds true exploitability. Hoeffding's
/// inequality bounds the remaining i.i.d. chance-sampling error.
fn flop_resolver_value_network_sha256(
    config: &ExploitabilityCertificateConfig,
) -> Result<Option<String>, Box<dyn Error>> {
    config
        .flop_resolver
        .as_ref()
        .map(|resolver| {
            fs::read(&resolver.value_network_path)
                .map(|bytes| format!("{:x}", Sha256::digest(bytes)))
        })
        .transpose()
        .map_err(Into::into)
}

fn flop_resolver_auxiliary_value_network_sha256s(
    config: &ExploitabilityCertificateConfig,
) -> Result<Vec<String>, Box<dyn Error>> {
    config
        .flop_resolver
        .as_ref()
        .map(|resolver| {
            resolver
                .auxiliary_value_network_paths
                .iter()
                .map(|path| fs::read(path).map(|bytes| format!("{:x}", Sha256::digest(bytes))))
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()
        .map(Option::unwrap_or_default)
        .map_err(Into::into)
}

fn enable_certificate_resolvers(
    generator: &mut SampleGenerator,
    config: &ExploitabilityCertificateConfig,
) -> Result<(), Box<dyn Error>> {
    generator.enable_flop_resolver(config.flop_resolver.clone())?;
    generator.enable_turn_resolver(config.turn_resolver)?;
    generator.enable_river_resolver(config.river_resolver)?;
    Ok(())
}

pub fn certify_exploitability_upper_bound(
    config: ExploitabilityCertificateConfig,
) -> Result<ExploitabilityCertificate, Box<dyn Error>> {
    if config.deals < 2 {
        return Err("exploitability certification requires at least two deals".into());
    }
    if !config.confidence.is_finite() || !(0.0..1.0).contains(&config.confidence) {
        return Err("certificate confidence must be strictly between zero and one".into());
    }
    if config.threads == 0 {
        return Err("certificate thread count must be positive".into());
    }
    let network_sha256 = format!("{:x}", Sha256::digest(fs::read(&config.network_path)?));
    let range_policy_sha256 = config
        .range_policy_path
        .as_ref()
        .map(|path| fs::read(path).map(|bytes| format!("{:x}", Sha256::digest(bytes))))
        .transpose()?;
    let flop_resolver_value_network_sha256 = flop_resolver_value_network_sha256(&config)?;
    let flop_resolver_auxiliary_value_network_sha256s =
        flop_resolver_auxiliary_value_network_sha256s(&config)?;
    let mut generator = SampleGenerator::new_with_range(
        SampleGenerationConfig {
            game: config.game.clone(),
            traversals: 1,
            start_iteration: 0,
            seed: config.seed,
            max_records: 1,
            output: PathBuf::from("unused-certificate.jsonl.gz"),
            network_path: Some(config.network_path.clone()),
            trajectory_sampling: false,
            evaluate_trajectory_values: false,
            value_rollouts_per_action: 1,
            enumerate_turn_river_chance: false,
        },
        config.range_policy_path.as_deref(),
    )?;
    enable_certificate_resolvers(&mut generator, &config)?;
    if generator.networks.is_none() {
        return Err("exploitability certification requires a frozen policy".into());
    }
    let mut rng = SplitMix64::new(config.seed);
    let deals = (0..config.deals)
        .map(|_| Deal::sample(&mut rng))
        .collect::<Vec<_>>();
    let worker_count = config.threads.min(deals.len());
    let chunk_size = deals.len().div_ceil(worker_count);
    // `Deal` owns mutable rollout caches and is Send but intentionally not
    // Sync. Give every worker an owned chunk so no cache crosses a thread.
    let deal_chunks = deals
        .chunks(chunk_size)
        .map(<[Deal]>::to_vec)
        .collect::<Vec<_>>();
    let evaluated = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for chunk in deal_chunks {
            let generator = &generator;
            let game = &config.game;
            handles.push(scope.spawn(move || {
                chunk
                    .into_iter()
                    .map(|deal| {
                        let mut visited_nodes = 0u64;
                        let response_p0 = clairvoyant_response_value(
                            generator,
                            GameState::initial(game),
                            &deal,
                            0,
                            &mut visited_nodes,
                        );
                        let response_p1 = clairvoyant_response_value(
                            generator,
                            GameState::initial(game),
                            &deal,
                            1,
                            &mut visited_nodes,
                        );
                        (
                            ((response_p0 + response_p1) / 2.0).clamp(0.0, game.effective_stack_bb),
                            [response_p0, response_p1],
                            visited_nodes,
                        )
                    })
                    .collect::<Vec<_>>()
            }));
        }
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("certificate worker panicked"))
            .collect::<Vec<_>>()
    });
    let mut visited_nodes = 0u64;
    let mut mean = 0.0;
    let mut squared_deviation_sum = 0.0;
    let mut sample_exploitabilities = Vec::with_capacity(config.deals as usize);
    let mut sample_responses = Vec::with_capacity(config.deals as usize);
    let depth = config.game.effective_stack_bb;
    for (index, (exploitability, responses, nodes)) in evaluated.into_iter().enumerate() {
        visited_nodes += nodes;
        sample_exploitabilities.push(exploitability);
        sample_responses.push(responses);
        let sample_index = index + 1;
        let delta = exploitability - mean;
        mean += delta / sample_index as f64;
        squared_deviation_sum += delta * (exploitability - mean);
    }
    let sample_variance = squared_deviation_sum / (config.deals - 1) as f64;
    let sample_standard_error = (sample_variance.max(0.0) / config.deals as f64).sqrt();
    let alpha = 1.0 - config.confidence;
    let margin = depth * ((1.0 / alpha).ln() / (2.0 * config.deals as f64)).sqrt();
    let mut assumptions = vec![
        "complete deals are independent uniform samples from the exact card-removal distribution",
        "the frozen opponent network is evaluated exactly on every reached betting action",
        "utilities are zero-sum and bounded to the effective stack in absolute value",
    ];
    if config.river_resolver.is_some() {
        assumptions.push(
            "every reached river action is replaced by deterministic exact-range public-belief CFR",
        );
    }
    if config.turn_resolver.is_some() {
        assumptions.push(
            "every reached turn action is replaced by deterministic exact-range joint turn/river public-belief CFR",
        );
    }
    if config.flop_resolver.is_some() {
        assumptions.push(
            "every reached flop action is replaced by deterministic exact-range depth-limited public-belief CFR using a frozen turn value network",
        );
        if config.flop_resolver.as_ref().is_some_and(|resolver| {
            resolver.continuation_selection
                == super::public_belief::FlopContinuationSelection::OpponentPublicChoice
        }) {
            assumptions.push(
                "each flop regret update retains distinct accepted turn-value hypotheses from different frozen continuation policies and permits the opposing continuation to select the traverser's worst hypothesis at each public turn leaf",
            );
        }
    }
    Ok(ExploitabilityCertificate {
        schema: if config.continual_resolving_enabled() {
            "hu-neural-continual-resolved-clairvoyant-upper-bound-v1"
        } else {
            "hu-neural-clairvoyant-upper-bound-v1"
        },
        method: if config.continual_resolving_enabled() {
            "complete_deal_clairvoyant_best_response_against_exact_range_continual_resolved_policy_with_hoeffding_ucb"
        } else {
            "complete_deal_clairvoyant_best_response_with_hoeffding_ucb"
        },
        depth_bb: depth,
        deals: config.deals,
        seed: config.seed,
        network_sha256,
        range_policy_sha256,
        confidence: config.confidence,
        threads: worker_count,
        opponent_samples_per_deal: None,
        opponent_samples_per_runout: None,
        public_branches_per_street: None,
        scenarios_per_deal: None,
        river_resolver_iterations: config.river_resolver.map(|resolver| resolver.iterations),
        river_resolver_averaging_delay: config
            .river_resolver
            .map(|resolver| resolver.averaging_delay),
        turn_resolver_iterations: config.turn_resolver.map(|resolver| resolver.iterations),
        turn_resolver_averaging_delay: config
            .turn_resolver
            .map(|resolver| resolver.averaging_delay),
        turn_resolver_river_refinement_iterations: config
            .turn_resolver
            .map(|resolver| resolver.river_refinement_iterations),
        turn_resolver_regret_matching_plus: config
            .turn_resolver
            .map(|resolver| resolver.regret_matching_plus),
        flop_resolver_iterations: config
            .flop_resolver
            .as_ref()
            .map(|resolver| resolver.iterations),
        flop_resolver_averaging_delay: config
            .flop_resolver
            .as_ref()
            .map(|resolver| resolver.averaging_delay),
        flop_resolver_regret_matching_plus: config
            .flop_resolver
            .as_ref()
            .map(|resolver| resolver.regret_matching_plus),
        flop_resolver_resolved_policy_weight: config
            .flop_resolver
            .as_ref()
            .map(|resolver| resolver.resolved_policy_weight),
        flop_resolver_value_network_sha256,
        flop_resolver_auxiliary_value_network_sha256s,
        flop_resolver_continuation_selection: config
            .flop_resolver
            .as_ref()
            .map(|resolver| resolver.continuation_selection),
        exact_betting_tree_nodes: visited_nodes,
        sample_exploitabilities_bb: sample_exploitabilities,
        sample_response_values_bb: sample_responses,
        sample_mean_exploitability_bb: mean,
        sample_standard_error_bb: sample_standard_error,
        hoeffding_margin_bb: margin,
        empirical_bernstein_margin_bb: None,
        confidence_bound_method: "one_sided_hoeffding",
        confidence_bound_reference: "https://doi.org/10.1007/BF02288859",
        exploitability_upper_bound_bb: (mean + margin).min(depth),
        relaxation: "each responder observes both private hands and the complete board runout",
        guarantee: "the relaxed responder contains all legal imperfect-information responses, so its i.i.d. chance expectation upper-bounds true exploitability",
        assumptions,
    })
}

/// Compute a tighter conservative upper bound without revealing the
/// opponent's private cards.
///
/// Each outer sample reveals the responder's cards and the complete public
/// runout. Conditional opponent hands remain hidden and are represented by a
/// common sample-average game. The responder must select one action for all
/// opponent particles reaching the same public history. Maximizing a finite
/// unbiased sample average has non-negative optimization bias, so the
/// expectation of this relaxed empirical best response remains an upper bound
/// on the true imperfect-information best response. The outer Hoeffding bound
/// covers both deal and opponent-particle sampling error.
pub fn certify_opponent_hidden_exploitability_upper_bound(
    config: ExploitabilityCertificateConfig,
    opponent_samples_per_deal: u32,
) -> Result<ExploitabilityCertificate, Box<dyn Error>> {
    if config.deals < 2 {
        return Err("exploitability certification requires at least two deals".into());
    }
    if opponent_samples_per_deal == 0 {
        return Err("opponent-hidden certification requires opponent samples".into());
    }
    if !config.confidence.is_finite() || !(0.0..1.0).contains(&config.confidence) {
        return Err("certificate confidence must be strictly between zero and one".into());
    }
    if config.threads == 0 {
        return Err("certificate thread count must be positive".into());
    }
    let network_sha256 = format!("{:x}", Sha256::digest(fs::read(&config.network_path)?));
    let range_policy_sha256 = config
        .range_policy_path
        .as_ref()
        .map(|path| fs::read(path).map(|bytes| format!("{:x}", Sha256::digest(bytes))))
        .transpose()?;
    let flop_resolver_value_network_sha256 = flop_resolver_value_network_sha256(&config)?;
    let flop_resolver_auxiliary_value_network_sha256s =
        flop_resolver_auxiliary_value_network_sha256s(&config)?;
    let mut generator = SampleGenerator::new_with_range(
        SampleGenerationConfig {
            game: config.game.clone(),
            traversals: 1,
            start_iteration: 0,
            seed: config.seed,
            max_records: 1,
            output: PathBuf::from("unused-opponent-hidden-certificate.jsonl.gz"),
            network_path: Some(config.network_path.clone()),
            trajectory_sampling: false,
            evaluate_trajectory_values: false,
            value_rollouts_per_action: 1,
            enumerate_turn_river_chance: false,
        },
        config.range_policy_path.as_deref(),
    )?;
    enable_certificate_resolvers(&mut generator, &config)?;
    if generator.networks.is_none() {
        return Err("exploitability certification requires a frozen policy".into());
    }
    let mut rng = SplitMix64::new(config.seed);
    let deals = (0..config.deals)
        .map(|index| (index, Deal::sample(&mut rng)))
        .collect::<Vec<_>>();
    let worker_count = config.threads.min(deals.len());
    let chunk_size = deals.len().div_ceil(worker_count);
    let deal_chunks = deals
        .chunks(chunk_size)
        .map(<[(u64, Deal)]>::to_vec)
        .collect::<Vec<_>>();
    let evaluated = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for chunk in deal_chunks {
            let generator = &generator;
            let game = &config.game;
            handles.push(scope.spawn(move || {
                chunk
                    .into_iter()
                    .map(|(index, template)| {
                        let mut visited_nodes = 0u64;
                        let responses: [f64; 2] = std::array::from_fn(|responder| {
                            let seed = config.seed
                                ^ index.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                                ^ (responder as u64 + 1).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                            let mut scenario_rng = SplitMix64::new(seed);
                            let scenarios = sample_opponent_hidden_scenarios(
                                &template,
                                responder,
                                opponent_samples_per_deal,
                                &mut scenario_rng,
                            );
                            let weights = vec![
                                1.0 / opponent_samples_per_deal as f64;
                                opponent_samples_per_deal as usize
                            ];
                            opponent_hidden_future_board_response_value(
                                generator,
                                GameState::initial(game),
                                &scenarios,
                                &weights,
                                responder,
                                &mut visited_nodes,
                            )
                        });
                        (
                            ((responses[0] + responses[1]) / 2.0)
                                .clamp(0.0, game.effective_stack_bb),
                            responses,
                            visited_nodes,
                        )
                    })
                    .collect::<Vec<_>>()
            }));
        }
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("certificate worker panicked"))
            .collect::<Vec<_>>()
    });
    let mut visited_nodes = 0u64;
    let mut mean = 0.0;
    let mut squared_deviation_sum = 0.0;
    let mut sample_exploitabilities = Vec::with_capacity(config.deals as usize);
    let mut sample_responses = Vec::with_capacity(config.deals as usize);
    let depth = config.game.effective_stack_bb;
    for (index, (exploitability, responses, nodes)) in evaluated.into_iter().enumerate() {
        visited_nodes += nodes;
        sample_exploitabilities.push(exploitability);
        sample_responses.push(responses);
        let sample_index = index + 1;
        let delta = exploitability - mean;
        mean += delta / sample_index as f64;
        squared_deviation_sum += delta * (exploitability - mean);
    }
    let sample_variance = squared_deviation_sum / (config.deals - 1) as f64;
    let sample_standard_error = (sample_variance.max(0.0) / config.deals as f64).sqrt();
    let alpha = 1.0 - config.confidence;
    let hoeffding_margin = depth * ((1.0 / alpha).ln() / (2.0 * config.deals as f64)).sqrt();
    let empirical_bernstein_margin = one_sided_empirical_bernstein_margin(
        sample_variance,
        depth,
        config.deals,
        config.confidence,
    );
    let mut assumptions = vec![
        "outer responder-card and board samples are independent and exact under card removal",
        "conditional opponent particles are independent uniform samples with replacement and shared across every candidate response in an outer game",
        "the frozen opponent network is evaluated exactly on every reached betting action",
        "complete sampled runouts settle every showdown exactly",
        "utilities are zero-sum and bounded to the effective stack in absolute value",
    ];
    if config.river_resolver.is_some() {
        assumptions.push(
            "every reached river action is replaced by deterministic exact-range public-belief CFR",
        );
    }
    if config.turn_resolver.is_some() {
        assumptions.push(
            "every reached turn action is replaced by deterministic exact-range joint turn/river public-belief CFR",
        );
    }
    if config.flop_resolver.is_some() {
        assumptions.push(
            "every reached flop action is replaced by deterministic exact-range depth-limited public-belief CFR using a frozen turn value network",
        );
        if config.flop_resolver.as_ref().is_some_and(|resolver| {
            resolver.continuation_selection
                == super::public_belief::FlopContinuationSelection::OpponentPublicChoice
        }) {
            assumptions.push(
                "each flop regret update retains distinct accepted turn-value hypotheses from different frozen continuation policies and permits the opposing continuation to select the traverser's worst hypothesis at each public turn leaf",
            );
        }
    }
    Ok(ExploitabilityCertificate {
        schema: if config.continual_resolving_enabled() {
            "hu-neural-continual-resolved-opponent-hidden-upper-bound-v1"
        } else {
            "hu-neural-opponent-hidden-upper-bound-v1"
        },
        method: if config.continual_resolving_enabled() {
            "future_public_runout_relaxation_with_hidden_opponent_sample_average_best_response_against_exact_range_continual_resolved_policy_and_empirical_bernstein_ucb"
        } else {
            "future_public_runout_relaxation_with_hidden_opponent_sample_average_best_response_and_empirical_bernstein_ucb"
        },
        depth_bb: depth,
        deals: config.deals,
        seed: config.seed,
        network_sha256,
        range_policy_sha256,
        confidence: config.confidence,
        threads: worker_count,
        opponent_samples_per_deal: Some(opponent_samples_per_deal),
        opponent_samples_per_runout: None,
        public_branches_per_street: None,
        scenarios_per_deal: Some(opponent_samples_per_deal as u64),
        river_resolver_iterations: config.river_resolver.map(|resolver| resolver.iterations),
        river_resolver_averaging_delay: config
            .river_resolver
            .map(|resolver| resolver.averaging_delay),
        turn_resolver_iterations: config.turn_resolver.map(|resolver| resolver.iterations),
        turn_resolver_averaging_delay: config
            .turn_resolver
            .map(|resolver| resolver.averaging_delay),
        turn_resolver_river_refinement_iterations: config
            .turn_resolver
            .map(|resolver| resolver.river_refinement_iterations),
        turn_resolver_regret_matching_plus: config
            .turn_resolver
            .map(|resolver| resolver.regret_matching_plus),
        flop_resolver_iterations: config
            .flop_resolver
            .as_ref()
            .map(|resolver| resolver.iterations),
        flop_resolver_averaging_delay: config
            .flop_resolver
            .as_ref()
            .map(|resolver| resolver.averaging_delay),
        flop_resolver_regret_matching_plus: config
            .flop_resolver
            .as_ref()
            .map(|resolver| resolver.regret_matching_plus),
        flop_resolver_resolved_policy_weight: config
            .flop_resolver
            .as_ref()
            .map(|resolver| resolver.resolved_policy_weight),
        flop_resolver_value_network_sha256,
        flop_resolver_auxiliary_value_network_sha256s,
        flop_resolver_continuation_selection: config
            .flop_resolver
            .as_ref()
            .map(|resolver| resolver.continuation_selection),
        exact_betting_tree_nodes: visited_nodes,
        sample_exploitabilities_bb: sample_exploitabilities,
        sample_response_values_bb: sample_responses,
        sample_mean_exploitability_bb: mean,
        sample_standard_error_bb: sample_standard_error,
        hoeffding_margin_bb: hoeffding_margin,
        empirical_bernstein_margin_bb: Some(empirical_bernstein_margin),
        confidence_bound_method: "maurer_pontil_2009_theorem_4_one_sided_empirical_bernstein",
        confidence_bound_reference: "https://arxiv.org/abs/0907.3740",
        exploitability_upper_bound_bb: (mean + empirical_bernstein_margin).min(depth),
        relaxation: "each responder observes its own private cards and the complete public runout, while opponent private cards remain hidden behind a common sample-average belief",
        guarantee: "future-board revelation contains every legal response; the expected sample-average optimum upper-bounds the relaxed best response by convexity, and the one-sided empirical Bernstein bound covers its outer i.i.d. expectation",
        assumptions,
    })
}

/// Compute a conservative sample-game upper bound with causal public chance.
///
/// An outer sample fixes only the responder's private cards. Each nested
/// empirical game samples flop, turn, and river branches, then hidden opponent
/// hands. The responder may condition on its private cards, the betting history,
/// and only the public cards revealed on the current street. For every fixed
/// legal response the nested estimator is unbiased; maximizing the empirical
/// game has non-negative sample-optimization bias, so its expectation remains
/// an upper bound on the true best response.
pub fn certify_causal_sample_game_exploitability_upper_bound(
    config: ExploitabilityCertificateConfig,
    public_branches_per_street: u32,
    opponent_samples_per_runout: u32,
) -> Result<ExploitabilityCertificate, Box<dyn Error>> {
    if config.deals < 2 {
        return Err("exploitability certification requires at least two deals".into());
    }
    if public_branches_per_street == 0 {
        return Err("causal certification requires public chance branches".into());
    }
    if opponent_samples_per_runout == 0 {
        return Err("causal certification requires hidden-hand samples".into());
    }
    if !config.confidence.is_finite() || !(0.0..1.0).contains(&config.confidence) {
        return Err("certificate confidence must be strictly between zero and one".into());
    }
    if config.threads == 0 {
        return Err("certificate thread count must be positive".into());
    }
    let scenarios_per_deal = u64::from(public_branches_per_street)
        .checked_pow(3)
        .and_then(|count| count.checked_mul(u64::from(opponent_samples_per_runout)))
        .ok_or("causal certificate scenario count overflows")?;
    if scenarios_per_deal > 1_000_000 {
        return Err("causal certificate exceeds one million scenarios per deal".into());
    }
    let network_sha256 = format!("{:x}", Sha256::digest(fs::read(&config.network_path)?));
    let range_policy_sha256 = config
        .range_policy_path
        .as_ref()
        .map(|path| fs::read(path).map(|bytes| format!("{:x}", Sha256::digest(bytes))))
        .transpose()?;
    let flop_resolver_value_network_sha256 = flop_resolver_value_network_sha256(&config)?;
    let flop_resolver_auxiliary_value_network_sha256s =
        flop_resolver_auxiliary_value_network_sha256s(&config)?;
    let mut generator = SampleGenerator::new_with_range(
        SampleGenerationConfig {
            game: config.game.clone(),
            traversals: 1,
            start_iteration: 0,
            seed: config.seed,
            max_records: 1,
            output: PathBuf::from("unused-causal-certificate.jsonl.gz"),
            network_path: Some(config.network_path.clone()),
            trajectory_sampling: false,
            evaluate_trajectory_values: false,
            value_rollouts_per_action: 1,
            enumerate_turn_river_chance: false,
        },
        config.range_policy_path.as_deref(),
    )?;
    enable_certificate_resolvers(&mut generator, &config)?;
    if generator.networks.is_none() {
        return Err("exploitability certification requires a frozen policy".into());
    }
    let mut rng = SplitMix64::new(config.seed);
    let responder_holes = (0..config.deals)
        .map(|index| (index, Deal::sample(&mut rng).holes))
        .collect::<Vec<_>>();
    let worker_count = config.threads.min(responder_holes.len());
    let chunk_size = responder_holes.len().div_ceil(worker_count);
    let chunks = responder_holes
        .chunks(chunk_size)
        .map(<[(u64, [[u8; 2]; 2])]>::to_vec)
        .collect::<Vec<_>>();
    let evaluated = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for chunk in chunks {
            let generator = &generator;
            let game = &config.game;
            handles.push(scope.spawn(move || {
                chunk
                    .into_iter()
                    .map(|(index, sampled_holes)| {
                        let mut visited_nodes = 0u64;
                        let responses: [f64; 2] = std::array::from_fn(|responder| {
                            let seed = config.seed
                                ^ index.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                                ^ (responder as u64 + 1).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                            let mut scenario_rng = SplitMix64::new(seed);
                            let scenarios = sample_causal_scenarios(
                                sampled_holes[responder],
                                responder,
                                public_branches_per_street,
                                opponent_samples_per_runout,
                                &mut scenario_rng,
                            );
                            let weights = vec![1.0 / scenarios.len() as f64; scenarios.len()];
                            causal_sample_game_response_value(
                                generator,
                                GameState::initial(game),
                                &scenarios,
                                &weights,
                                responder,
                                &mut visited_nodes,
                            )
                        });
                        (
                            ((responses[0] + responses[1]) / 2.0)
                                .clamp(0.0, game.effective_stack_bb),
                            responses,
                            visited_nodes,
                        )
                    })
                    .collect::<Vec<_>>()
            }));
        }
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("certificate worker panicked"))
            .collect::<Vec<_>>()
    });
    let mut visited_nodes = 0u64;
    let mut mean = 0.0;
    let mut squared_deviation_sum = 0.0;
    let mut sample_exploitabilities = Vec::with_capacity(config.deals as usize);
    let mut sample_responses = Vec::with_capacity(config.deals as usize);
    let depth = config.game.effective_stack_bb;
    for (index, (exploitability, responses, nodes)) in evaluated.into_iter().enumerate() {
        visited_nodes += nodes;
        sample_exploitabilities.push(exploitability);
        sample_responses.push(responses);
        let sample_index = index + 1;
        let delta = exploitability - mean;
        mean += delta / sample_index as f64;
        squared_deviation_sum += delta * (exploitability - mean);
    }
    let sample_variance = squared_deviation_sum / (config.deals - 1) as f64;
    let sample_standard_error = (sample_variance.max(0.0) / config.deals as f64).sqrt();
    let alpha = 1.0 - config.confidence;
    let hoeffding_margin = depth * ((1.0 / alpha).ln() / (2.0 * config.deals as f64)).sqrt();
    let empirical_bernstein_margin = one_sided_empirical_bernstein_margin(
        sample_variance,
        depth,
        config.deals,
        config.confidence,
    );
    let mut assumptions = vec![
        "outer responder-card samples are independent and exact under card removal",
        "nested flop, turn, river, and hidden-hand branches have the exact conditional card distribution",
        "identical observed public boards share one responder action within each sampled betting history",
        "the frozen opponent network is evaluated exactly on every reached betting action",
        "complete sampled runouts settle every showdown exactly",
        "utilities are zero-sum and bounded to the effective stack in absolute value",
    ];
    if config.river_resolver.is_some() {
        assumptions.push(
            "every reached river action is replaced by deterministic exact-range public-belief CFR",
        );
    }
    if config.turn_resolver.is_some() {
        assumptions.push(
            "every reached turn action is replaced by deterministic exact-range joint turn/river public-belief CFR",
        );
    }
    if config.flop_resolver.is_some() {
        assumptions.push(
            "every reached flop action is replaced by deterministic exact-range depth-limited public-belief CFR using a frozen turn value network",
        );
        if config.flop_resolver.as_ref().is_some_and(|resolver| {
            resolver.continuation_selection
                == super::public_belief::FlopContinuationSelection::OpponentPublicChoice
        }) {
            assumptions.push(
                "each flop regret update retains distinct accepted turn-value hypotheses from different frozen continuation policies and permits the opposing continuation to select the traverser's worst hypothesis at each public turn leaf",
            );
        }
    }
    Ok(ExploitabilityCertificate {
        schema: if config.continual_resolving_enabled() {
            "hu-neural-continual-resolved-causal-sample-game-upper-bound-v1"
        } else {
            "hu-neural-causal-sample-game-upper-bound-v1"
        },
        method: if config.continual_resolving_enabled() {
            "nested_public_chance_and_hidden_hand_sample_game_best_response_against_exact_range_continual_resolved_policy_with_empirical_bernstein_ucb"
        } else {
            "nested_public_chance_and_hidden_hand_sample_game_best_response_with_empirical_bernstein_ucb"
        },
        depth_bb: depth,
        deals: config.deals,
        seed: config.seed,
        network_sha256,
        range_policy_sha256,
        confidence: config.confidence,
        threads: worker_count,
        opponent_samples_per_deal: None,
        opponent_samples_per_runout: Some(opponent_samples_per_runout),
        public_branches_per_street: Some(public_branches_per_street),
        scenarios_per_deal: Some(scenarios_per_deal),
        river_resolver_iterations: config.river_resolver.map(|resolver| resolver.iterations),
        river_resolver_averaging_delay: config
            .river_resolver
            .map(|resolver| resolver.averaging_delay),
        turn_resolver_iterations: config.turn_resolver.map(|resolver| resolver.iterations),
        turn_resolver_averaging_delay: config
            .turn_resolver
            .map(|resolver| resolver.averaging_delay),
        turn_resolver_river_refinement_iterations: config
            .turn_resolver
            .map(|resolver| resolver.river_refinement_iterations),
        turn_resolver_regret_matching_plus: config
            .turn_resolver
            .map(|resolver| resolver.regret_matching_plus),
        flop_resolver_iterations: config
            .flop_resolver
            .as_ref()
            .map(|resolver| resolver.iterations),
        flop_resolver_averaging_delay: config
            .flop_resolver
            .as_ref()
            .map(|resolver| resolver.averaging_delay),
        flop_resolver_regret_matching_plus: config
            .flop_resolver
            .as_ref()
            .map(|resolver| resolver.regret_matching_plus),
        flop_resolver_resolved_policy_weight: config
            .flop_resolver
            .as_ref()
            .map(|resolver| resolver.resolved_policy_weight),
        flop_resolver_value_network_sha256,
        flop_resolver_auxiliary_value_network_sha256s,
        flop_resolver_continuation_selection: config
            .flop_resolver
            .as_ref()
            .map(|resolver| resolver.continuation_selection),
        exact_betting_tree_nodes: visited_nodes,
        sample_exploitabilities_bb: sample_exploitabilities,
        sample_response_values_bb: sample_responses,
        sample_mean_exploitability_bb: mean,
        sample_standard_error_bb: sample_standard_error,
        hoeffding_margin_bb: hoeffding_margin,
        empirical_bernstein_margin_bb: Some(empirical_bernstein_margin),
        confidence_bound_method: "maurer_pontil_2009_theorem_4_one_sided_empirical_bernstein",
        confidence_bound_reference: "https://arxiv.org/abs/0907.3740",
        exploitability_upper_bound_bb: (mean + empirical_bernstein_margin).min(depth),
        relaxation: "each responder observes its own cards, betting history, and only the currently revealed public board in a nested public-chance and hidden-hand sample game",
        guarantee: "every fixed legal response has an unbiased nested sample-game value; the expected empirical optimum upper-bounds the legal best response by convexity, and the one-sided empirical Bernstein bound covers the independent outer games",
        assumptions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolved_policy_stabilization_preserves_blocked_rows_and_full_support() {
        let mut probabilities = vec![0.0; COMBO_COUNT * 3];
        probabilities[3..6].copy_from_slice(&[0.0, 0.25, 0.75]);
        let stabilized = stabilize_resolved_policy(probabilities, 3).unwrap();
        assert_eq!(&stabilized[..3], &[0.0, 0.0, 0.0]);
        assert!(stabilized[3..6]
            .iter()
            .all(|probability| *probability > 0.0));
        assert!((stabilized[3..6].iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bounded_resolver_cache_returns_requested_root_after_self_eviction() {
        let board = [0, 5, 10, 15, 20];
        let root_history = vec!["root".to_owned()];
        let root_key = range_policy_public_cache_key(Street::River, 0, &board, &root_history);
        let strategies = (0..32)
            .map(|index| {
                let history = if index == 0 {
                    root_history.clone()
                } else {
                    vec![format!("child:{index}")]
                };
                let mut probabilities = vec![0.0f32; COMBO_COUNT * 2];
                probabilities[2..4].copy_from_slice(&[0.25, 0.75]);
                super::public_belief::PublicBeliefStrategy {
                    public_history: history,
                    actor: 0,
                    action_labels: vec!["check".to_owned(), "bet".to_owned()],
                    probabilities,
                    action_values_bb: None,
                }
            })
            .collect::<Vec<_>>();
        let mut cache = BTreeMap::new();
        let root =
            cache_resolved_policy_rows(&mut cache, root_key, Street::River, &board, strategies, 1)
                .unwrap();
        assert_eq!(root.action_labels, ["check", "bet"]);
        assert_eq!(root.probabilities.len(), COMBO_COUNT * 2);
        assert!(!cache.contains_key(&root_key));
    }

    #[test]
    fn anchored_resolver_blend_preserves_rows_and_endpoints() {
        let mut resolved = vec![0.0; COMBO_COUNT * 2];
        let mut anchor = vec![0.0; COMBO_COUNT * 2];
        resolved[2..4].copy_from_slice(&[1.0, 0.0]);
        anchor[2..4].copy_from_slice(&[0.0, 1.0]);
        let midpoint = blend_resolved_with_anchor(&resolved, &anchor, 2, 0.5).unwrap();
        assert_eq!(&midpoint[..2], &[0.0, 0.0]);
        assert!((midpoint[2] - 0.5).abs() < 1e-8);
        assert!((midpoint[3] - 0.5).abs() < 1e-8);
        let anchored = blend_resolved_with_anchor(&resolved, &anchor, 2, 0.0).unwrap();
        let solved = blend_resolved_with_anchor(&resolved, &anchor, 2, 1.0).unwrap();
        assert!(anchored[3] > 1.0 - 1e-8);
        assert!(solved[2] > 1.0 - 1e-8);
    }

    #[test]
    fn public_range_normalization_is_scale_invariant_for_rare_lines() {
        let mut ranges = [vec![1.0e-16; COMBO_COUNT], vec![2.0e-16; COMBO_COUNT]];
        normalize_ranges_for_board(&mut ranges, &[0, 5, 10])
            .expect("positive rare-line reach remains conditionable");

        for range in ranges {
            assert!((range.iter().sum::<f64>() - 1.0).abs() <= 1.0e-12);
            for combo in all_combos() {
                if combo.cards().iter().any(|card| [0, 5, 10].contains(card)) {
                    assert_eq!(range[combo.key()], 0.0);
                }
            }
        }
    }

    #[test]
    fn public_range_normalization_still_rejects_impossible_lines() {
        let mut ranges = [vec![0.0; COMBO_COUNT], vec![1.0; COMBO_COUNT]];
        assert_eq!(
            normalize_ranges_for_board(&mut ranges, &[0, 5, 10]),
            Err("range policy public action has zero conditional reach".to_owned())
        );
    }

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
    fn exact_turn_river_chance_enumerates_every_unblocked_card() {
        let deal = Deal::from_cards([[0, 1], [2, 3]], [4, 5, 6, 7, 51]);
        let rivers = exact_river_deals(&deal);
        assert_eq!(rivers.len(), 44);
        let cards = rivers
            .iter()
            .map(|candidate| candidate.board[4])
            .collect::<BTreeSet<_>>();
        assert_eq!(cards.len(), 44);
        assert!(cards
            .iter()
            .all(|card| ![0, 1, 2, 3, 4, 5, 6, 7].contains(card)));
        assert!(cards.contains(&51));
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
    fn frozen_policy_replays_public_ranges_and_routes_postflop_inference() {
        let dense = serde_json::json!({
            "layers": [{
                "input_size": MODEL_INPUT_COUNT,
                "output_size": 1,
                "activation": "linear",
                "weights": vec![0.0f32; MODEL_INPUT_COUNT],
                "biases": [0.0]
            }]
        });
        let bundle = serde_json::json!({
            "schema": TRAINING_NETWORK_SCHEMA,
            "input_size": MODEL_INPUT_COUNT,
            "strategy_transform": "softmax",
            "networks": [dense.clone(), dense]
        });
        let layer = |input_size: usize, weights: Vec<f32>| {
            serde_json::json!({
                "inputSize": input_size,
                "outputSize": 1,
                "activation": "linear",
                "weights": weights,
                "biases": [0.0]
            })
        };
        let mut action_weights = vec![0.0f32; ACTION_FEATURE_COUNT];
        action_weights[1] = 1.0;
        let mut head_weights = vec![0.0f32; 5];
        head_weights[4] = 1.0;
        let range = serde_json::json!({
            "schema": "hu-public-belief-combo-policy-network-v1",
            "seed": 91,
            "depthBb": 20.0,
            "usesExactRanges": true,
            "featureSchema": "rank-suit-invariant-combo-policy-query-v1",
            "contextSize": 417,
            "querySize": 124,
            "actionFeatureSchema": "hu-cash-legal-action-v1",
            "actionFeatureSize": ACTION_FEATURE_COUNT,
            "contextTower": [layer(417, vec![0.0f32; 417])],
            "queryTower": [layer(124, vec![0.0f32; 124])],
            "actionTower": [layer(ACTION_FEATURE_COUNT, action_weights)],
            "head": [layer(5, head_weights)],
            "sourceDatasetSha256": "1".repeat(64),
            "sourceDatasetSchema": "hu-range-conditioned-postflop-policy-dataset-v1",
            "sourceValidationStatus": "accepted_for_training"
        });
        let prefix = format!("pokersolver-range-route-{}", std::process::id());
        let bundle_path = std::env::temp_dir().join(format!("{prefix}-bundle.json"));
        let range_path = std::env::temp_dir().join(format!("{prefix}-range.json"));
        fs::write(&bundle_path, serde_json::to_vec(&bundle).unwrap()).unwrap();
        fs::write(&range_path, serde_json::to_vec(&range).unwrap()).unwrap();
        let policy = FrozenPolicy::load_with_range(&bundle_path, Some(&range_path)).unwrap();
        let mut game = BlueprintConfig::default();
        game.effective_stack_bb = 20.0;
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
        let deal = Deal::from_sampled_cards([[48, 49], [44, 45]], [0, 5, 10, 15, 20]);
        let actions = state.legal_actions(&game);
        let first = policy.strategy(&state, &deal, &actions, &game);
        let second = policy.strategy(&state, &deal, &actions, &game);
        assert_eq!(first, second);
        let expected = std::f64::consts::E / (std::f64::consts::E + actions.len() as f64 - 1.0);
        assert!((first[0] - expected).abs() < 1e-6);
        for _ in 0..2 {
            let legal = state.legal_actions(&game);
            let check = legal.iter().find(|action| action.label == "check").unwrap();
            state = state.apply(check, &game);
        }
        assert_eq!(state.street, Street::Turn);
        let turn_actions = state.legal_actions(&game);
        let cached_public = policy
            .range_public_state(&state, &deal, &turn_actions, &game)
            .unwrap();
        let replayed_ranges = policy
            .replay_ranges_from_root(&state, &deal, &game)
            .unwrap();
        let maximum_range_difference = cached_public
            .ranges
            .iter()
            .flatten()
            .zip(replayed_ranges.iter().flatten())
            .map(|(cached, replayed)| (cached - replayed).abs())
            .fold(0.0f64, f64::max);
        assert!(maximum_range_difference < 1e-15);
        let cached_turn = policy.strategy(&state, &deal, &turn_actions, &game);
        let replay_public = PublicBeliefState::from_game_state(
            deal.board[..state.street.board_len()].to_vec(),
            &state,
            replayed_ranges,
        );
        let replay_matrix = policy
            .range_policy
            .as_ref()
            .unwrap()
            .strategy(&replay_public, &game, None)
            .unwrap();
        let combo = Combo::new(deal.holes[state.actor][0], deal.holes[state.actor][1]).key();
        assert_eq!(
            cached_turn,
            replay_matrix[combo * turn_actions.len()..(combo + 1) * turn_actions.len()]
        );
        assert!(policy.range_cache.lock().unwrap().len() >= 3);
        fs::remove_file(bundle_path).unwrap();
        fs::remove_file(range_path).unwrap();
    }

    #[test]
    fn frozen_residual_policy_pins_and_preserves_its_source_bundle() {
        let dense = serde_json::json!({
            "layers": [{
                "input_size": MODEL_INPUT_COUNT,
                "output_size": 1,
                "activation": "linear",
                "weights": vec![0.0f32; MODEL_INPUT_COUNT],
                "biases": [0.0]
            }]
        });
        let bundle = serde_json::json!({
            "schema": TRAINING_NETWORK_SCHEMA,
            "input_size": MODEL_INPUT_COUNT,
            "strategy_transform": "softmax",
            "networks": [dense.clone(), dense]
        });
        let bundle_bytes = serde_json::to_vec(&bundle).unwrap();
        let bundle_sha256 = format!("{:x}", Sha256::digest(&bundle_bytes));
        let layer = |input_size: usize| {
            serde_json::json!({
                "inputSize": input_size,
                "outputSize": 1,
                "activation": "linear",
                "weights": vec![0.0f32; input_size],
                "biases": [0.0]
            })
        };
        let residual = serde_json::json!({
            "schema": "hu-public-belief-combo-policy-network-v1",
            "seed": 92,
            "depthBb": 20.0,
            "usesExactRanges": true,
            "featureSchema": "rank-suit-invariant-combo-policy-query-v1",
            "contextSize": 417,
            "querySize": 124,
            "actionFeatureSchema": "hu-cash-legal-action-v1",
            "actionFeatureSize": ACTION_FEATURE_COUNT,
            "contextTower": [layer(417)],
            "queryTower": [layer(124)],
            "actionTower": [layer(ACTION_FEATURE_COUNT)],
            "head": [layer(5)],
            "sourceDatasetSha256": "1".repeat(64),
            "sourceDatasetSchema": "hu-range-conditioned-postflop-policy-dataset-v1",
            "sourceValidationStatus": "accepted_for_training",
            "policyComposition": "source_bundle_logit_residual",
            "sourcePolicySha256": bundle_sha256,
        });
        let prefix = format!("pokersolver-residual-route-{}", std::process::id());
        let bundle_path = std::env::temp_dir().join(format!("{prefix}-bundle.json"));
        let residual_path = std::env::temp_dir().join(format!("{prefix}-residual.json"));
        fs::write(&bundle_path, bundle_bytes).unwrap();
        fs::write(&residual_path, serde_json::to_vec(&residual).unwrap()).unwrap();
        let policy = FrozenPolicy::load_with_range(&bundle_path, Some(&residual_path)).unwrap();
        let mut game = BlueprintConfig::default();
        game.effective_stack_bb = 20.0;
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
        let deal = Deal::from_sampled_cards([[48, 49], [44, 45]], [0, 5, 10, 15, 20]);
        let actions = state.legal_actions(&game);
        let probabilities = policy.strategy(&state, &deal, &actions, &game);
        assert!(probabilities
            .iter()
            .all(|probability| (*probability - 1.0 / actions.len() as f64).abs() < 1e-12));

        let mut mismatched = residual;
        mismatched["sourcePolicySha256"] = serde_json::json!("f".repeat(64));
        fs::write(&residual_path, serde_json::to_vec(&mismatched).unwrap()).unwrap();
        assert!(FrozenPolicy::load_with_range(&bundle_path, Some(&residual_path)).is_err());
        fs::remove_file(bundle_path).unwrap();
        fs::remove_file(residual_path).unwrap();
    }

    #[test]
    fn clairvoyant_certificate_is_deterministic_and_upper_bounded() {
        let network = serde_json::json!({
            "layers": [{
                "input_size": MODEL_INPUT_COUNT,
                "output_size": 1,
                "activation": "linear",
                "weights": vec![0.0f32; MODEL_INPUT_COUNT],
                "biases": [0.0]
            }]
        });
        let bundle = serde_json::json!({
            "schema": TRAINING_NETWORK_SCHEMA,
            "input_size": MODEL_INPUT_COUNT,
            "strategy_transform": "softmax",
            "networks": [network.clone(), network]
        });
        let path = std::env::temp_dir().join(format!(
            "pokersolver-neural-certificate-{}.json",
            std::process::id()
        ));
        let payload = serde_json::to_vec(&bundle).unwrap();
        let expected_network_sha256 = format!("{:x}", Sha256::digest(&payload));
        fs::write(&path, payload).unwrap();
        let mut game = BlueprintConfig::default();
        game.effective_stack_bb = 2.0;
        let make = || ExploitabilityCertificateConfig {
            game: game.clone(),
            deals: 8,
            seed: 73,
            confidence: 0.99,
            threads: 2,
            network_path: path.clone(),
            range_policy_path: None,
            river_resolver: None,
            turn_resolver: None,
            flop_resolver: None,
        };
        let first = certify_exploitability_upper_bound(make()).unwrap();
        let second = certify_exploitability_upper_bound(make()).unwrap();
        let hidden_first = certify_opponent_hidden_exploitability_upper_bound(make(), 4).unwrap();
        let hidden_second = certify_opponent_hidden_exploitability_upper_bound(make(), 4).unwrap();
        let causal_first =
            certify_causal_sample_game_exploitability_upper_bound(make(), 2, 2).unwrap();
        let causal_second =
            certify_causal_sample_game_exploitability_upper_bound(make(), 2, 2).unwrap();
        fs::remove_file(path).unwrap();
        assert_eq!(
            serde_json::to_vec(&first).unwrap(),
            serde_json::to_vec(&second).unwrap()
        );
        assert!(first.exact_betting_tree_nodes > 0);
        assert_eq!(first.network_sha256, expected_network_sha256);
        assert!(first.sample_mean_exploitability_bb >= 0.0);
        for certificate in [&first, &hidden_first, &causal_first] {
            assert_eq!(certificate.sample_exploitabilities_bb.len(), 8);
            assert_eq!(certificate.sample_response_values_bb.len(), 8);
            let reconstructed_mean =
                certificate.sample_exploitabilities_bb.iter().sum::<f64>() / 8.0;
            assert!((reconstructed_mean - certificate.sample_mean_exploitability_bb).abs() < 1e-12);
            for (sample, responses) in certificate
                .sample_exploitabilities_bb
                .iter()
                .zip(&certificate.sample_response_values_bb)
            {
                assert!(responses.iter().all(|value| value.is_finite()));
                assert_eq!(
                    *sample,
                    ((responses[0] + responses[1]) / 2.0).clamp(0.0, 2.0)
                );
            }
        }
        assert!(first.exploitability_upper_bound_bb >= first.sample_mean_exploitability_bb);
        assert!(first.exploitability_upper_bound_bb <= 2.0);
        assert_eq!(
            serde_json::to_vec(&hidden_first).unwrap(),
            serde_json::to_vec(&hidden_second).unwrap()
        );
        assert_eq!(hidden_first.opponent_samples_per_deal, Some(4));
        assert_eq!(hidden_first.network_sha256, first.network_sha256);
        assert_eq!(
            hidden_first.schema,
            "hu-neural-opponent-hidden-upper-bound-v1"
        );
        assert!(hidden_first.exact_betting_tree_nodes > 0);
        assert!(hidden_first.sample_mean_exploitability_bb >= 0.0);
        assert!(
            hidden_first.exploitability_upper_bound_bb
                >= hidden_first.sample_mean_exploitability_bb
        );
        assert!(hidden_first.exploitability_upper_bound_bb <= 2.0);
        assert_eq!(
            serde_json::to_vec(&causal_first).unwrap(),
            serde_json::to_vec(&causal_second).unwrap()
        );
        assert_eq!(
            causal_first.schema,
            "hu-neural-causal-sample-game-upper-bound-v1"
        );
        assert_eq!(causal_first.opponent_samples_per_runout, Some(2));
        assert_eq!(causal_first.public_branches_per_street, Some(2));
        assert_eq!(causal_first.scenarios_per_deal, Some(16));
        assert!(causal_first.exact_betting_tree_nodes > 0);
        assert!(causal_first.sample_mean_exploitability_bb >= 0.0);
        assert!(
            causal_first.exploitability_upper_bound_bb
                >= causal_first.sample_mean_exploitability_bb
        );
        assert!(causal_first.exploitability_upper_bound_bb <= 2.0);
    }

    #[test]
    fn causal_scenarios_preserve_cards_and_nested_public_information() {
        let responder_holes = [48, 49];
        let scenarios = sample_causal_scenarios(responder_holes, 0, 2, 3, &mut SplitMix64::new(91));
        assert_eq!(scenarios.len(), 24);
        for deal in &scenarios {
            assert_eq!(deal.holes[0], responder_holes);
            let cards = deal
                .holes
                .into_iter()
                .flatten()
                .chain(deal.board)
                .collect::<BTreeSet<_>>();
            assert_eq!(cards.len(), 9);
        }
        let flop_groups = scenarios
            .iter()
            .map(|deal| observed_public_board(deal, Street::Flop))
            .collect::<BTreeSet<_>>();
        let turn_groups = scenarios
            .iter()
            .map(|deal| observed_public_board(deal, Street::Turn))
            .collect::<BTreeSet<_>>();
        let river_groups = scenarios
            .iter()
            .map(|deal| observed_public_board(deal, Street::River))
            .collect::<BTreeSet<_>>();
        assert!(flop_groups.len() <= 2);
        assert!(turn_groups.len() <= 4);
        assert!(river_groups.len() <= 8);
        assert!(flop_groups.len() <= turn_groups.len());
        assert!(turn_groups.len() <= river_groups.len());
    }

    #[test]
    fn causal_policy_attribution_reconstructs_the_certificate_and_is_deterministic() {
        use flate2::read::GzDecoder;
        use std::io::BufRead;

        let network = serde_json::json!({
            "layers": [{
                "input_size": MODEL_INPUT_COUNT,
                "output_size": 1,
                "activation": "linear",
                "weights": vec![0.0f32; MODEL_INPUT_COUNT],
                "biases": [0.0]
            }]
        });
        let bundle = serde_json::json!({
            "schema": TRAINING_NETWORK_SCHEMA,
            "input_size": MODEL_INPUT_COUNT,
            "strategy_transform": "softmax",
            "networks": [network.clone(), network]
        });
        let prefix = format!("pokersolver-causal-attribution-{}", std::process::id());
        let directory = std::env::temp_dir();
        let network_path = directory.join(format!("{prefix}-network.json"));
        let first_path = directory.join(format!("{prefix}-first.jsonl.gz"));
        let second_path = directory.join(format!("{prefix}-second.jsonl.gz"));
        fs::write(&network_path, serde_json::to_vec(&bundle).unwrap()).unwrap();
        let mut game = BlueprintConfig::default();
        game.effective_stack_bb = 2.0;
        let make = |output| CausalPolicyAttributionConfig {
            game: game.clone(),
            deals: 2,
            seed: 79,
            threads: 2,
            network_path: network_path.clone(),
            range_policy_path: None,
            public_branches_per_street: 1,
            opponent_samples_per_runout: 1,
            max_records: 60,
            output,
        };
        let first = generate_causal_policy_attribution(make(first_path.clone())).unwrap();
        let second = generate_causal_policy_attribution(make(second_path.clone())).unwrap();
        let certificate = certify_causal_sample_game_exploitability_upper_bound(
            ExploitabilityCertificateConfig {
                game,
                deals: 2,
                seed: 79,
                confidence: 0.99,
                threads: 2,
                network_path: network_path.clone(),
                range_policy_path: None,
                river_resolver: None,
                turn_resolver: None,
                flop_resolver: None,
            },
            1,
            1,
        )
        .unwrap();
        let source_evaluation =
            evaluate_causal_attribution_policy(CausalAttributionPolicyEvaluationConfig {
                dataset_path: first_path.clone(),
                network_path: network_path.clone(),
                maximum_node_kl: 0.005,
                maximum_weighted_kl: 0.0015,
                minimum_policy_value_gain_bb: 0.000001,
            })
            .unwrap();
        assert_eq!(
            fs::read(&first_path).unwrap(),
            fs::read(&second_path).unwrap()
        );
        assert_eq!(first.output_sha256, second.output_sha256);
        assert!(
            (first.sample_mean_exploitability_bb - certificate.sample_mean_exploitability_bb).abs()
                < 1e-12
        );
        assert!(first.maximum_root_value_reconstruction_error_bb < 1e-12);
        assert!(first.candidate_records >= first.retained_records);
        assert!(first.retained_records > 0);
        assert!(first.minimum_frozen_policy_action_probability > 0.0);
        assert!(first.maximum_target_probability_sum_error < 1e-12);
        assert_eq!(source_evaluation.records, first.retained_records);
        assert!(source_evaluation.feature_hashes_verified);
        assert!(source_evaluation.weighted_policy_value_gain_bb.abs() < 1e-12);
        assert!(source_evaluation.maximum_reverse_kl_from_frozen < 1e-12);
        assert!(source_evaluation.maximum_candidate_probability_sum_error < 1e-12);
        assert!(!source_evaluation.accepted_for_routed_evaluation);

        let reader = BufReader::new(GzDecoder::new(fs::File::open(&first_path).unwrap()));
        let mut lines = reader.lines();
        let metadata: serde_json::Value =
            serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
        assert_eq!(
            metadata["schema"],
            "hu-neural-causal-policy-attribution-jsonl-v1"
        );
        assert_eq!(metadata["preflop_policy_frozen"], true);
        let records = lines
            .map(|line| serde_json::from_str::<serde_json::Value>(&line.unwrap()).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(records.len(), first.retained_records);
        for record in records {
            assert_ne!(record["state"]["street"], "preflop");
            let actions = record["actions"].as_array().unwrap();
            let targets = record["targets"].as_array().unwrap();
            let values = record["action_values_bb"].as_array().unwrap();
            let hashes = record["feature_sha256"].as_array().unwrap();
            assert!(!actions.is_empty());
            assert_eq!(actions.len(), targets.len());
            assert_eq!(actions.len(), values.len());
            assert_eq!(actions.len(), hashes.len());
            let probability_sum = targets
                .iter()
                .map(|value| value.as_f64().unwrap())
                .sum::<f64>();
            assert!((probability_sum - 1.0).abs() < 1e-6);
            assert!(record["weight"].as_f64().unwrap() > 0.0);
            assert!(values
                .iter()
                .all(|value| value.as_f64().unwrap().is_finite()));
            assert!(hashes.iter().all(|hash| hash.as_str().unwrap().len() == 64));
        }
        fs::remove_file(network_path).unwrap();
        fs::remove_file(first_path).unwrap();
        fs::remove_file(second_path).unwrap();
    }

    #[test]
    fn range_conditioned_causal_attribution_records_beliefs_and_focal_combos() {
        use flate2::read::GzDecoder;
        use std::io::BufRead;

        let direct_layer = serde_json::json!({
            "layers": [{
                "input_size": MODEL_INPUT_COUNT,
                "output_size": 1,
                "activation": "linear",
                "weights": vec![0.0f32; MODEL_INPUT_COUNT],
                "biases": [0.0]
            }]
        });
        let bundle = serde_json::json!({
            "schema": TRAINING_NETWORK_SCHEMA,
            "input_size": MODEL_INPUT_COUNT,
            "strategy_transform": "softmax",
            "networks": [direct_layer.clone(), direct_layer]
        });
        let layer = |input_size: usize| {
            serde_json::json!({
                "inputSize": input_size,
                "outputSize": 1,
                "activation": "linear",
                "weights": vec![0.0f32; input_size],
                "biases": [0.0]
            })
        };
        let range_policy = serde_json::json!({
            "schema": "hu-public-belief-combo-policy-network-v1",
            "seed": 83,
            "depthBb": 2.0,
            "usesExactRanges": true,
            "featureSchema": RANGE_POLICY_FEATURE_SCHEMA_V2,
            "contextSize": RANGE_POLICY_CONTEXT_V2_COUNT,
            "querySize": 124,
            "actionFeatureSchema": "hu-cash-legal-action-v1",
            "actionFeatureSize": ACTION_FEATURE_COUNT,
            "contextTower": [layer(RANGE_POLICY_CONTEXT_V2_COUNT)],
            "queryTower": [layer(124)],
            "actionTower": [layer(ACTION_FEATURE_COUNT)],
            "head": [layer(5)],
            "sourceDatasetSha256": "a".repeat(64),
            "sourceDatasetSchema": "hu-range-conditioned-postflop-policy-dataset-v1",
            "sourceValidationStatus": "accepted_for_training",
            "policyComposition": "replace"
        });
        let value_network = serde_json::json!({
            "schema": "hu-public-belief-combo-value-network-v3",
            "seed": 83,
            "usesExactRanges": true,
            "targetScaleBb": 20.0,
            "rangeScale": COMBO_COUNT,
            "residualScaleBb": 5.0,
            "sourceDatasetSha256": "b".repeat(64),
            "sourcePolicySha256": "c".repeat(64),
            "sourceValidationStatus": "accepted",
            "featureSchema": "rank-suit-invariant-combo-query-v1",
            "contextPublicCount": 21,
            "contextSize": 359,
            "queryStructuralCount": 76,
            "querySize": 95,
            "contextTower": [layer(359)],
            "queryTower": [layer(95)],
            "head": [layer(2)]
        });
        let prefix = format!(
            "pokersolver-range-causal-attribution-{}",
            std::process::id()
        );
        let directory = std::env::temp_dir();
        let network_path = directory.join(format!("{prefix}-network.json"));
        let range_path = directory.join(format!("{prefix}-range.json"));
        let value_path = directory.join(format!("{prefix}-value.json"));
        let output_path = directory.join(format!("{prefix}.jsonl.gz"));
        let self_play_first_path = directory.join(format!("{prefix}-self-play-first.jsonl.gz"));
        let self_play_second_path = directory.join(format!("{prefix}-self-play-second.jsonl.gz"));
        let self_play_tiny_reach_path =
            directory.join(format!("{prefix}-self-play-tiny-reach.jsonl.gz"));
        fs::write(&network_path, serde_json::to_vec(&bundle).unwrap()).unwrap();
        let range_bytes = serde_json::to_vec(&range_policy).unwrap();
        fs::write(&range_path, &range_bytes).unwrap();
        fs::write(&value_path, serde_json::to_vec(&value_network).unwrap()).unwrap();
        let mut game = BlueprintConfig::default();
        game.effective_stack_bb = 2.0;
        let resolver_config = ExploitabilityCertificateConfig {
            game: game.clone(),
            deals: 2,
            seed: 83,
            confidence: 0.99,
            threads: 2,
            network_path: network_path.clone(),
            range_policy_path: Some(range_path.clone()),
            river_resolver: None,
            turn_resolver: None,
            flop_resolver: Some(FlopResolverConfig {
                iterations: 2,
                averaging_delay: 0,
                regret_matching_plus: false,
                threads: 1,
                value_network_path: value_path.clone(),
                auxiliary_value_network_paths: Vec::new(),
                continuation_selection: super::public_belief::FlopContinuationSelection::Mean,
                resolved_policy_weight: 0.5,
            }),
        };
        let mut resolver_generator = SampleGenerator::new_with_range(
            SampleGenerationConfig {
                game: game.clone(),
                traversals: 1,
                start_iteration: 0,
                seed: 83,
                max_records: 1,
                output: directory.join(format!("{prefix}-unused.jsonl.gz")),
                network_path: Some(network_path.clone()),
                trajectory_sampling: false,
                evaluate_trajectory_values: false,
                value_rollouts_per_action: 1,
                enumerate_turn_river_chance: false,
            },
            Some(&range_path),
        )
        .unwrap();
        enable_certificate_resolvers(&mut resolver_generator, &resolver_config).unwrap();
        assert!(resolver_generator
            .networks
            .as_ref()
            .is_some_and(|policy| policy.flop_resolver.is_some()));
        assert_eq!(
            resolver_generator
                .networks
                .as_ref()
                .and_then(|policy| policy.flop_resolver.as_ref())
                .map(|resolver| resolver.config.resolved_policy_weight),
            Some(0.5)
        );
        let attribution = generate_causal_policy_attribution(CausalPolicyAttributionConfig {
            game: game.clone(),
            deals: 2,
            seed: 83,
            threads: 2,
            network_path: network_path.clone(),
            range_policy_path: Some(range_path.clone()),
            public_branches_per_street: 1,
            opponent_samples_per_runout: 1,
            max_records: 60,
            output: output_path.clone(),
        })
        .unwrap();
        let certificate = certify_causal_sample_game_exploitability_upper_bound(
            ExploitabilityCertificateConfig {
                game: game.clone(),
                deals: 2,
                seed: 83,
                confidence: 0.99,
                threads: 2,
                network_path: network_path.clone(),
                range_policy_path: Some(range_path.clone()),
                river_resolver: None,
                turn_resolver: None,
                flop_resolver: None,
            },
            1,
            1,
        )
        .unwrap();
        let resolved_certificate = certify_causal_sample_game_exploitability_upper_bound(
            ExploitabilityCertificateConfig {
                game: game.clone(),
                deals: 2,
                seed: 83,
                confidence: 0.99,
                threads: 2,
                network_path: network_path.clone(),
                range_policy_path: Some(range_path.clone()),
                river_resolver: Some(RiverResolverConfig {
                    iterations: 2,
                    averaging_delay: 0,
                }),
                turn_resolver: Some(TurnResolverConfig {
                    iterations: 2,
                    averaging_delay: 0,
                    river_refinement_iterations: 0,
                    regret_matching_plus: false,
                }),
                flop_resolver: None,
            },
            1,
            1,
        )
        .unwrap();
        assert_eq!(
            attribution.range_policy_sha256,
            Some(format!("{:x}", Sha256::digest(&range_bytes)))
        );
        assert!(
            (attribution.sample_mean_exploitability_bb - certificate.sample_mean_exploitability_bb)
                .abs()
                < 1e-12
        );
        assert!(attribution.maximum_root_value_reconstruction_error_bb < 1e-12);
        assert_eq!(
            resolved_certificate.schema,
            "hu-neural-continual-resolved-causal-sample-game-upper-bound-v1"
        );
        assert_eq!(resolved_certificate.river_resolver_iterations, Some(2));
        assert_eq!(resolved_certificate.river_resolver_averaging_delay, Some(0));
        assert_eq!(resolved_certificate.turn_resolver_iterations, Some(2));
        assert_eq!(resolved_certificate.turn_resolver_averaging_delay, Some(0));
        assert!(resolved_certificate
            .sample_mean_exploitability_bb
            .is_finite());

        let reader = BufReader::new(GzDecoder::new(fs::File::open(&output_path).unwrap()));
        let mut lines = reader.lines();
        let metadata: serde_json::Value =
            serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
        assert_eq!(
            metadata["schema"],
            "hu-range-conditioned-causal-policy-attribution-jsonl-v1"
        );
        assert_eq!(metadata["uses_exact_ranges"], true);
        assert_eq!(metadata["focal_combo_attribution"], true);
        let records = lines
            .map(|line| serde_json::from_str::<serde_json::Value>(&line.unwrap()).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(records.len(), attribution.retained_records);
        assert!(!records.is_empty());
        for record in records {
            assert_eq!(
                record["record_type"],
                "range_conditioned_causal_policy_attribution"
            );
            assert_ne!(record["state"]["street"], "preflop");
            assert!(record["focal_combo"].as_u64().unwrap() < COMBO_COUNT as u64);
            let ranges = record["ranges"].as_array().unwrap();
            assert_eq!(ranges.len(), 2);
            for range in ranges {
                assert_eq!(range.as_array().unwrap().len(), COMBO_COUNT);
                let sum = range
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_f64().unwrap())
                    .sum::<f64>();
                assert!((sum - 1.0).abs() < 1e-5);
            }
            let actions = record["action_labels"].as_array().unwrap();
            let probabilities = record["probabilities"].as_array().unwrap();
            let values = record["action_values_bb"].as_array().unwrap();
            assert!(!actions.is_empty());
            assert_eq!(actions.len(), probabilities.len());
            assert_eq!(actions.len(), values.len());
            let probability_sum = probabilities
                .iter()
                .map(|value| value.as_f64().unwrap())
                .sum::<f64>();
            assert!((probability_sum - 1.0).abs() < 1e-6);
            assert!(record["weight"].as_f64().unwrap() > 0.0);
        }
        let evaluation = crate::blueprint::public_belief::evaluate_causal_range_policy(
            crate::blueprint::public_belief::CausalRangePolicyEvaluationConfig {
                network_path: range_path.clone(),
                frozen_network_path: range_path.clone(),
                attribution_network_path: range_path.clone(),
                dataset_path: output_path.clone(),
                source_policy_path: None,
                minimum_policy_value_gain_bb: 0.000001,
                maximum_node_kl: 0.005,
                maximum_weighted_kl: 0.0015,
            },
        )
        .unwrap();
        assert_eq!(evaluation.records, attribution.retained_records);
        assert!(evaluation.weighted_policy_value_gain_bb.abs() < 1e-12);
        assert!(evaluation.maximum_reverse_kl_from_frozen < 1e-12);
        assert!(evaluation.maximum_stored_source_probability_difference < 1e-6);
        assert_eq!(evaluation.validation.status, "rejected");

        let self_play_config = |output| RangeSelfPlaySampleConfig {
            game: game.clone(),
            traversals: 8,
            start_iteration: 0,
            seed: 89,
            max_records: 200,
            network_path: network_path.clone(),
            range_policy_path: range_path.clone(),
            value_rollouts_per_action: 2,
            enumerate_turn_river_chance: false,
            output,
        };
        let first_self_play =
            generate_range_self_play_samples(self_play_config(self_play_first_path.clone()))
                .unwrap();
        let second_self_play =
            generate_range_self_play_samples(self_play_config(self_play_second_path.clone()))
                .unwrap();
        assert_eq!(
            fs::read(&self_play_first_path).unwrap(),
            fs::read(&self_play_second_path).unwrap()
        );
        assert_eq!(
            first_self_play.output_sha256,
            second_self_play.output_sha256
        );
        assert_eq!(first_self_play.retained_records_by_street.len(), 3);
        assert!(first_self_play
            .retained_records_by_street
            .iter()
            .all(|count| *count > 0));
        assert!(first_self_play.minimum_policy_action_probability > 0.0);
        assert!(first_self_play.maximum_probability_sum_error < 1e-6);
        assert!(first_self_play.maximum_action_value_standard_error_bb >= 0.0);
        let reader = BufReader::new(GzDecoder::new(
            fs::File::open(&self_play_first_path).unwrap(),
        ));
        let mut lines = reader.lines();
        let self_play_metadata: serde_json::Value =
            serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
        assert_eq!(
            self_play_metadata["schema"],
            "hu-range-conditioned-self-play-regret-jsonl-v1"
        );
        let first_record: serde_json::Value =
            serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
        assert_eq!(
            first_record["record_type"],
            "range_conditioned_self_play_regret"
        );
        assert_eq!(
            first_record["action_values_bb"].as_array().unwrap().len(),
            first_record["action_value_standard_errors_bb"]
                .as_array()
                .unwrap()
                .len()
        );
        let self_play_evaluation = crate::blueprint::public_belief::evaluate_causal_range_policy(
            crate::blueprint::public_belief::CausalRangePolicyEvaluationConfig {
                network_path: range_path.clone(),
                frozen_network_path: range_path.clone(),
                attribution_network_path: range_path.clone(),
                dataset_path: self_play_first_path.clone(),
                source_policy_path: None,
                minimum_policy_value_gain_bb: 0.000001,
                maximum_node_kl: 0.005,
                maximum_weighted_kl: 0.0015,
            },
        )
        .unwrap();
        assert_eq!(
            self_play_evaluation.records,
            first_self_play.retained_records
        );
        assert!(self_play_evaluation.weighted_policy_value_gain_bb.abs() < 1e-12);
        assert_eq!(self_play_evaluation.validation.status, "rejected");
        let reader = BufReader::new(GzDecoder::new(
            fs::File::open(&self_play_first_path).unwrap(),
        ));
        let mut payloads = reader
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(&line.unwrap()).unwrap())
            .collect::<Vec<_>>();
        let actor = payloads[1]["state"]["actor"].as_u64().unwrap() as usize;
        let focal = payloads[1]["focal_combo"].as_u64().unwrap() as usize;
        payloads[1]["ranges"][actor][focal] = serde_json::json!(1e-13);
        let file = fs::File::create(&self_play_tiny_reach_path).unwrap();
        let mut writer = GzEncoder::new(BufWriter::new(file), Compression::fast());
        for payload in &payloads {
            serde_json::to_writer(&mut writer, payload).unwrap();
            writer.write_all(b"\n").unwrap();
        }
        writer.finish().unwrap().flush().unwrap();
        let tiny_reach_evaluation = crate::blueprint::public_belief::evaluate_causal_range_policy(
            crate::blueprint::public_belief::CausalRangePolicyEvaluationConfig {
                network_path: range_path.clone(),
                frozen_network_path: range_path.clone(),
                attribution_network_path: range_path.clone(),
                dataset_path: self_play_tiny_reach_path.clone(),
                source_policy_path: None,
                minimum_policy_value_gain_bb: 0.000001,
                maximum_node_kl: 0.005,
                maximum_weighted_kl: 0.0015,
            },
        )
        .unwrap();
        assert_eq!(
            tiny_reach_evaluation.records,
            first_self_play.retained_records
        );
        fs::remove_file(network_path).unwrap();
        fs::remove_file(value_path).unwrap();
        fs::remove_file(range_path).unwrap();
        fs::remove_file(output_path).unwrap();
        fs::remove_file(self_play_first_path).unwrap();
        fs::remove_file(self_play_second_path).unwrap();
        fs::remove_file(self_play_tiny_reach_path).unwrap();
    }

    #[test]
    fn empirical_bernstein_margin_matches_the_bounded_variable_theorem() {
        let confidence: f64 = 0.99;
        let range = 20.0;
        let samples = 100;
        let expected = 7.0 * range * (2.0 / (1.0 - confidence)).ln() / (3.0 * (samples - 1) as f64);
        let measured = one_sided_empirical_bernstein_margin(0.0, range, samples, confidence);
        assert!((measured - expected).abs() < 1e-12);
        assert!(one_sided_empirical_bernstein_margin(0.0, range, 1_000, confidence) < measured);
        assert!(one_sided_empirical_bernstein_margin(0.5, range, samples, confidence) > measured);
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
            evaluate_trajectory_values: false,
            value_rollouts_per_action: 1,
            enumerate_turn_river_chance: false,
        };
        let (_, first, attempted, _, _) = SampleGenerator::new(make()).unwrap().run().unwrap();
        let (_, second, _, _, _) = SampleGenerator::new(make()).unwrap().run().unwrap();
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
            evaluate_trajectory_values: false,
            value_rollouts_per_action,
            enumerate_turn_river_chance: false,
        };
        let (_, primary, _, _, _) = SampleGenerator::new(make(1)).unwrap().run().unwrap();
        let (_, averaged, _, _, _) = SampleGenerator::new(make(4)).unwrap().run().unwrap();
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
            first_without_values
                .as_object_mut()
                .unwrap()
                .remove("action_value_standard_errors_bb");
            second_without_values
                .as_object_mut()
                .unwrap()
                .remove("action_value_standard_errors_bb");
            assert_eq!(first_without_values, second_without_values);
        }
        assert!(value_target_changed);
        assert!(averaged.iter().any(|sample| {
            sample
                .action_value_standard_errors_bb
                .as_ref()
                .is_some_and(|values| values.iter().all(|value| value.is_finite()))
        }));
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
            evaluate_trajectory_values: false,
            value_rollouts_per_action: 1,
            enumerate_turn_river_chance: false,
        };
        let (_, records, attempted, _, _) = SampleGenerator::new(config).unwrap().run().unwrap();
        assert_eq!(attempted, records.len());
        assert!(records.len() >= 4);
        assert!(records
            .iter()
            .all(|sample| matches!(sample.kind, SampleKind::AverageStrategy)));
        assert!(records
            .iter()
            .all(|sample| sample.weight == 1.0 && sample.reach_probability > 0.0));
    }

    #[test]
    fn trajectory_action_value_evaluation_is_deterministic_and_reports_standard_errors() {
        let mut game = BlueprintConfig::default();
        game.effective_stack_bb = 20.0;
        game.showdown_evaluation.preflop_runout_samples = 4;
        game.showdown_evaluation.flop_runout_samples = 4;
        game.showdown_evaluation.exact_turn_rivers = false;
        let make = || SampleGenerationConfig {
            game: game.clone(),
            traversals: 2,
            start_iteration: 0,
            seed: 31,
            max_records: 128,
            output: PathBuf::from("unused.jsonl.gz"),
            network_path: None,
            trajectory_sampling: true,
            evaluate_trajectory_values: true,
            value_rollouts_per_action: 3,
            enumerate_turn_river_chance: false,
        };
        let (_, first, _, _, _) = SampleGenerator::new(make()).unwrap().run().unwrap();
        let (_, second, _, _, _) = SampleGenerator::new(make()).unwrap().run().unwrap();
        assert_eq!(
            serde_json::to_vec(&first).unwrap(),
            serde_json::to_vec(&second).unwrap()
        );
        for sample in first {
            let values = sample.action_values_bb.unwrap();
            let errors = sample.action_value_standard_errors_bb.unwrap();
            assert_eq!(values.len(), sample.actions.len());
            assert_eq!(errors.len(), sample.actions.len());
            assert!(values.iter().all(|value| value.is_finite()));
            assert!(errors
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0));
        }
    }
}
