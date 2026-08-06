//! Range-consistent preflop solving and evaluation.
//!
//! The full neural game samples a complete deal at the root.  This module
//! keeps those hidden chance outcomes together at responder decisions: one
//! action is chosen for every world in the same observable preflop
//! information set.  Flop continuations are frozen in a deterministic cache,
//! turning the preflop portion into a finite, reproducible zero-sum game that
//! can be solved tabularly and used as a neural distillation oracle.

use super::neural::{average_strategy_record_json, FrozenPolicy};
use super::*;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const CONTINUATION_SCHEMA: &str = "hu-preflop-continuation-cache-v2";
const LEGACY_CONTINUATION_SCHEMA: &str = "hu-preflop-continuation-cache-v1";
const POLICY_SCHEMA: &str = "hu-tabular-preflop-dcfr-v1";
const EVALUATION_SCHEMA: &str = "hu-preflop-information-set-response-v2";
const ATTRIBUTION_SCHEMA: &str = "hu-preflop-local-leak-attribution-v1";
const EXTERNAL_SAMPLING_EXPLORATION: f64 = 0.05;
const RESOLVER_RANGE_PROBABILITY_FLOOR: f64 = 1e-9;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PreflopSolverVariant {
    #[default]
    Dcfr,
    MccfrPlus,
}

#[derive(Clone, Debug)]
pub struct PreflopSolveOptions {
    pub iterations: u64,
    pub seed: u64,
    pub model_version: String,
    pub dcfr: DcfrParameters,
    pub exploration_probability: f64,
    pub variant: PreflopSolverVariant,
}

#[derive(Clone, Debug)]
pub struct ContinuationCacheConfig {
    pub game: BlueprintConfig,
    pub deals: usize,
    pub seed: u64,
    pub rollouts_per_leaf: u32,
    pub network_paths: Vec<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct ResolverContinuationCacheConfig {
    pub deal_offset: usize,
    pub deals: usize,
    pub resolver_iterations: u64,
    pub resolver_averaging_delay: u64,
    pub resolver_regret_matching_plus: bool,
    pub value_uncertainty_bb: f64,
    pub value_network_path: PathBuf,
    pub evaluation_value_network_path: Option<PathBuf>,
    pub range_policy_path: Option<PathBuf>,
    pub source_cache_path: PathBuf,
    pub threads: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ContinuationEstimate {
    pub mean_utility_p0_bb: f64,
    #[serde(default)]
    pub conditional_utilities_bb: Option<[f64; 2]>,
    pub action_standard_error_bb: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CachedDeal {
    pub holes: [[u8; 2]; 2],
    pub board: [u8; 5],
    pub continuations: BTreeMap<u64, ContinuationEstimate>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ContinuationValidation {
    pub complete: bool,
    pub probability_values_finite: bool,
    pub utilities_within_stack: bool,
    pub leaf_values: usize,
    pub fraction_action_se_at_most_0_02bb: f64,
    pub maximum_action_standard_error_bb: f64,
    #[serde(default)]
    pub fraction_history_mean_se_at_most_0_02bb: f64,
    #[serde(default)]
    pub maximum_history_mean_standard_error_bb: f64,
    #[serde(default)]
    pub fraction_information_group_mean_se_at_most_0_25bb: f64,
    #[serde(default)]
    pub maximum_information_group_mean_standard_error_bb: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ContinuationCache {
    pub schema: String,
    pub depth_bb: f64,
    pub seed: u64,
    pub rollouts_per_leaf: u32,
    pub chance_sampling: String,
    pub complete_exact_combo_cycles: usize,
    #[serde(alias = "uniform_joint_deal_design")]
    pub balanced_exact_combo_marginals: bool,
    pub network_sha256: String,
    #[serde(default)]
    pub network_sha256s: Vec<String>,
    #[serde(default)]
    pub policy_mixture: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_cache_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_deal_indices: Option<Vec<usize>>,
    pub game: BlueprintConfig,
    pub public_histories: BTreeMap<u64, Vec<String>>,
    pub deals: Vec<CachedDeal>,
    pub validation: ContinuationValidation,
}

fn complete_source_deal_cycles(indices: &[usize]) -> usize {
    let cycle_size = 2 * all_combos().len();
    let mut cycles = BTreeMap::<usize, BTreeSet<usize>>::new();
    for index in indices {
        cycles
            .entry(index / cycle_size)
            .or_default()
            .insert(index % cycle_size);
    }
    cycles
        .values()
        .filter(|positions| positions.len() == cycle_size)
        .count()
}

fn balanced_source_deal_indices(indices: &[usize]) -> bool {
    complete_source_deal_cycles(indices) * 2 * all_combos().len() == indices.len()
}

impl ContinuationCache {
    pub fn read(path: &Path) -> Result<Self, Box<dyn Error>> {
        let file = fs::File::open(path)?;
        let reader = BufReader::new(file);
        let cache: Self = if path.extension().is_some_and(|extension| extension == "gz") {
            serde_json::from_reader(GzDecoder::new(reader))?
        } else {
            serde_json::from_reader(reader)?
        };
        cache.validate()?;
        Ok(cache)
    }

    pub fn write(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let temporary = path.with_extension("tmp");
        let file = fs::File::create(&temporary)?;
        let writer = BufWriter::new(file);
        if path.extension().is_some_and(|extension| extension == "gz") {
            let mut gzip = GzEncoder::new(writer, Compression::fast());
            serde_json::to_writer(&mut gzip, self)?;
            gzip.finish()?.flush()?;
        } else {
            let mut writer = writer;
            serde_json::to_writer(&mut writer, self)?;
            writer.flush()?;
        }
        fs::rename(temporary, path)?;
        Ok(())
    }

    pub fn validate(&self) -> Result<(), Box<dyn Error>> {
        if ![CONTINUATION_SCHEMA, LEGACY_CONTINUATION_SCHEMA].contains(&self.schema.as_str())
            || self.deals.len() < 2
        {
            return Err("preflop continuation cache is incompatible".into());
        }
        if self.game.effective_stack_bb != self.depth_bb || self.rollouts_per_leaf < 2 {
            return Err("preflop continuation cache metadata is invalid".into());
        }
        self.game.validate()?;
        let expected = self.public_histories.len();
        let provenance_is_valid = match (&self.source_cache_sha256, &self.source_deal_indices) {
            (None, None) => true,
            (Some(digest), Some(indices)) => {
                digest.len() == 64
                    && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
                    && indices.len() == self.deals.len()
                    && indices.iter().collect::<BTreeSet<_>>().len() == indices.len()
            }
            _ => false,
        };
        let expected_cycles = self
            .source_deal_indices
            .as_deref()
            .map(complete_source_deal_cycles)
            .unwrap_or_else(|| self.deals.len() / (2 * all_combos().len()));
        let expected_balanced = self
            .source_deal_indices
            .as_deref()
            .map(balanced_source_deal_indices)
            .unwrap_or_else(|| self.deals.len().is_multiple_of(2 * all_combos().len()));
        if expected == 0
            || !provenance_is_valid
            || self.complete_exact_combo_cycles != expected_cycles
            || self.balanced_exact_combo_marginals != expected_balanced
            || self.network_sha256.len() != 64
            || !self
                .network_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
            || self.network_sha256s.iter().any(|digest| {
                digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit())
            })
            || (self.network_sha256s.len() > 1 && self.policy_mixture.is_empty())
            || self.deals.iter().any(|deal| {
                deal.continuations.len() != expected
                    || deal
                        .continuations
                        .keys()
                        .any(|key| !self.public_histories.contains_key(key))
            })
        {
            return Err("preflop continuation cache is incomplete".into());
        }
        let depth = self.depth_bb;
        if self
            .deals
            .iter()
            .flat_map(|deal| deal.continuations.values())
            .any(|value| {
                !value.mean_utility_p0_bb.is_finite()
                    || !value.action_standard_error_bb.is_finite()
                    || value.action_standard_error_bb < 0.0
                    || value.mean_utility_p0_bb.abs() > depth + EPSILON
                    || value.conditional_utilities_bb.is_some_and(|utilities| {
                        utilities
                            .iter()
                            .any(|utility| !utility.is_finite() || utility.abs() > depth + EPSILON)
                    })
            })
        {
            return Err("preflop continuation cache contains invalid values".into());
        }
        if !self.validation.complete
            || !self.validation.probability_values_finite
            || !self.validation.utilities_within_stack
            || self.validation.leaf_values != self.deals.len() * expected
            || !in_unit_interval(self.validation.fraction_action_se_at_most_0_02bb)
            || !in_unit_interval(self.validation.fraction_history_mean_se_at_most_0_02bb)
            || !in_unit_interval(
                self.validation
                    .fraction_information_group_mean_se_at_most_0_25bb,
            )
            || !self.validation.maximum_action_standard_error_bb.is_finite()
            || !self
                .validation
                .maximum_history_mean_standard_error_bb
                .is_finite()
            || !self
                .validation
                .maximum_information_group_mean_standard_error_bb
                .is_finite()
        {
            return Err("preflop continuation cache validation summary is invalid".into());
        }
        Ok(())
    }

    fn deal(&self, index: usize) -> Deal {
        let deal = &self.deals[index];
        Deal::from_sampled_cards(deal.holes, deal.board)
    }

    fn continuation_utility(&self, deal_index: usize, state: &GameState, player: usize) -> f64 {
        let history = history_key(state);
        let estimate = &self.deals[deal_index].continuations[&history];
        estimate
            .conditional_utilities_bb
            .map(|utilities| utilities[player])
            .unwrap_or_else(|| {
                if player == 0 {
                    estimate.mean_utility_p0_bb
                } else {
                    -estimate.mean_utility_p0_bb
                }
            })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PreflopPolicyEntry {
    pub key: String,
    pub actor: usize,
    pub hand_class: String,
    pub public_history: Vec<String>,
    pub action_labels: Vec<String>,
    pub probabilities: Vec<f64>,
    pub positive_regret_sum_bb: f64,
    pub regret_updates: u64,
    pub average_visits: u64,
    #[serde(default)]
    pub average_reach_weight: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PreflopPolicyArtifact {
    pub schema: String,
    pub model_version: String,
    pub depth_bb: f64,
    pub seed: u64,
    pub iterations: u64,
    pub sampling_exploration_probability: f64,
    #[serde(default)]
    pub solver_dcfr: DcfrParameters,
    #[serde(default)]
    pub solver_variant: PreflopSolverVariant,
    pub continuation_cache_sha256: String,
    pub game: BlueprintConfig,
    pub strategies: Vec<PreflopPolicyEntry>,
    pub training_evaluation: PreflopEvaluation,
}

impl PreflopPolicyArtifact {
    pub fn read(path: &Path) -> Result<Self, Box<dyn Error>> {
        let artifact: Self = serde_json::from_reader(BufReader::new(fs::File::open(path)?))?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn write(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let temporary = path.with_extension("tmp");
        let mut writer = BufWriter::new(fs::File::create(&temporary)?);
        serde_json::to_writer_pretty(&mut writer, self)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        fs::rename(temporary, path)?;
        Ok(())
    }

    pub fn validate(&self) -> Result<(), Box<dyn Error>> {
        if self.schema != POLICY_SCHEMA || self.iterations == 0 || self.strategies.is_empty() {
            return Err("tabular preflop policy artifact is incompatible".into());
        }
        self.game.validate()?;
        if self.game.effective_stack_bb != self.depth_bb
            || !(0.0..1.0).contains(&self.sampling_exploration_probability)
            || !self.solver_dcfr.positive_regret_exponent.is_finite()
            || !self.solver_dcfr.negative_regret_exponent.is_finite()
            || !self.solver_dcfr.strategy_exponent.is_finite()
            || self.solver_dcfr.positive_regret_exponent < 0.0
            || self.solver_dcfr.negative_regret_exponent < 0.0
            || self.solver_dcfr.strategy_exponent < 0.0
            || self.continuation_cache_sha256.len() != 64
            || !self
                .continuation_cache_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err("tabular preflop policy metadata is invalid".into());
        }
        let mut keys = BTreeSet::new();
        for entry in &self.strategies {
            if !keys.insert(&entry.key)
                || entry.actor > 1
                || entry.action_labels.is_empty()
                || entry.action_labels.len() != entry.probabilities.len()
                || entry
                    .probabilities
                    .iter()
                    .any(|value| !value.is_finite() || *value < 0.0)
                || (entry.probabilities.iter().sum::<f64>() - 1.0).abs() > 1e-9
                || !entry.average_reach_weight.is_finite()
                || entry.average_reach_weight < 0.0
            {
                return Err("tabular preflop policy has invalid probabilities".into());
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PreflopEvaluation {
    pub schema: String,
    pub corpus_deals: usize,
    pub policy_value_p0_bb: f64,
    #[serde(default)]
    pub policy_value_p1_bb: f64,
    #[serde(default)]
    pub policy_value_zero_sum_residual_bb: f64,
    pub player_zero_best_response_bb: f64,
    pub player_one_best_response_bb: f64,
    pub nash_conv_bb: f64,
    pub exploitability_bb_per_hand: f64,
    pub responder_information_sets: [usize; 2],
    pub policy_lookup_coverage: f64,
    pub interpretation: String,
}

/// A policy-reach-weighted one-step deviation diagnostic.  The action values
/// hold every later decision at the evaluated policy, so these rows identify
/// concrete local leaks without pretending that their sum is a full best
/// response or an exploitability decomposition.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PreflopLeakEntry {
    pub key: String,
    pub player: usize,
    pub hand_class: String,
    pub public_history: Vec<String>,
    pub reach_probability: f64,
    pub action_labels: Vec<String>,
    pub policy_probabilities: Vec<f64>,
    pub action_values_bb: Vec<f64>,
    pub policy_value_bb: f64,
    pub best_action: String,
    pub best_action_value_bb: f64,
    pub conditional_ev_gain_bb: f64,
    pub reach_weighted_ev_gain_bb_per_hand: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PreflopPublicHistoryLeak {
    pub player: usize,
    pub public_history: Vec<String>,
    pub information_sets: usize,
    pub policy_reach_probability: f64,
    pub reach_weighted_ev_gain_bb_per_hand: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PreflopLeakAttribution {
    pub schema: String,
    pub corpus_deals: usize,
    pub policy_model_version: String,
    pub top_per_player: usize,
    pub evaluated_information_sets: [usize; 2],
    pub total_policy_reach_weighted_local_gain_bb: [f64; 2],
    pub policy_lookup_coverage: f64,
    pub players: [Vec<PreflopLeakEntry>; 2],
    pub public_histories: [Vec<PreflopPublicHistoryLeak>; 2],
    pub interpretation: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct DistillationDatasetSummary {
    pub schema: &'static str,
    pub records: usize,
    pub policy_information_sets: usize,
    pub covered_policy_information_sets: usize,
    pub coverage: f64,
    pub output: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CompactPreflopPolicySummary {
    pub schema: &'static str,
    pub format_version: u16,
    pub model_version: String,
    pub information_sets: usize,
    pub probabilities: usize,
    pub bytes: u64,
    pub sha256: String,
    pub maximum_probability_quantization_error: f64,
    pub quantized_probability_sums_valid: bool,
    pub output: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ContinuationPlayerComparison {
    pub player: usize,
    pub common_groups: usize,
    pub union_groups: usize,
    pub group_coverage: f64,
    pub sample_weighted_mean_absolute_delta_bb: f64,
    pub sample_weighted_root_mean_squared_delta_bb: f64,
    pub maximum_group_delta_bb: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ContinuationCacheComparison {
    pub schema: String,
    pub depth_bb: f64,
    pub cache_seeds: [u64; 2],
    pub deal_counts: [usize; 2],
    pub complete_exact_combo_cycles: [usize; 2],
    pub network_mixtures_match: bool,
    pub fraction_information_group_mean_se_at_most_0_25bb: [f64; 2],
    pub maximum_information_group_mean_standard_error_bb: [f64; 2],
    pub players: [ContinuationPlayerComparison; 2],
    pub interpretation: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct SolverNode {
    actor: usize,
    hand_class: String,
    public_history: Vec<String>,
    action_labels: Vec<String>,
    regrets: Vec<f64>,
    strategy_sum: Vec<f64>,
    regret_updates: u64,
    average_visits: u64,
    last_discount_iteration: u64,
    last_discount_cumulative_logs: [f64; 3],
}

impl SolverNode {
    fn new(state: &GameState, deal: &Deal, actions: &[LegalAction]) -> Self {
        Self {
            actor: state.actor,
            hand_class: hand_class(deal, state.actor),
            public_history: state.public_history.clone(),
            action_labels: actions.iter().map(|action| action.label.clone()).collect(),
            regrets: vec![0.0; actions.len()],
            strategy_sum: vec![0.0; actions.len()],
            regret_updates: 0,
            average_visits: 0,
            last_discount_iteration: 0,
            last_discount_cumulative_logs: [0.0; 3],
        }
    }

    fn current_strategy(&self) -> Vec<f64> {
        normalize_or_uniform(self.regrets.iter().map(|value| value.max(0.0)).collect())
    }

    fn average_strategy(&self) -> Vec<f64> {
        normalize_or_uniform(self.strategy_sum.clone())
    }

    fn apply_dcfr_discount(&mut self, iteration: u64, discounts: &DiscountAccumulator) {
        if iteration == 0 || self.last_discount_iteration == iteration {
            return;
        }
        let factors = [
            (discounts.cumulative_logs[0] - self.last_discount_cumulative_logs[0]).exp(),
            (discounts.cumulative_logs[1] - self.last_discount_cumulative_logs[1]).exp(),
            (discounts.cumulative_logs[2] - self.last_discount_cumulative_logs[2]).exp(),
        ];
        for regret in &mut self.regrets {
            *regret *= if *regret >= 0.0 {
                factors[0]
            } else {
                factors[1]
            };
        }
        for probability in &mut self.strategy_sum {
            *probability *= factors[2];
        }
        self.last_discount_iteration = iteration;
        self.last_discount_cumulative_logs = discounts.cumulative_logs;
    }
}

struct DiscountAccumulator {
    parameters: DcfrParameters,
    iteration: u64,
    cumulative_logs: [f64; 3],
}

impl DiscountAccumulator {
    fn new(parameters: DcfrParameters) -> Self {
        Self {
            parameters,
            iteration: 0,
            cumulative_logs: [0.0; 3],
        }
    }

    fn advance(&mut self, iteration: u64) {
        assert_eq!(iteration, self.iteration + 1);
        let time = iteration as f64;
        let positive_power = time.powf(self.parameters.positive_regret_exponent);
        let negative_power = time.powf(self.parameters.negative_regret_exponent);
        let factors = [
            positive_power / (positive_power + 1.0),
            negative_power / (negative_power + 1.0),
            (time / (time + 1.0)).powf(self.parameters.strategy_exponent),
        ];
        for (cumulative, factor) in self.cumulative_logs.iter_mut().zip(factors) {
            *cumulative += factor.ln();
        }
        self.iteration = iteration;
    }
}

struct PreflopDcfrSolver<'a> {
    cache: &'a ContinuationCache,
    nodes: BTreeMap<String, SolverNode>,
    rng: SplitMix64,
    iterations: u64,
    discounts: DiscountAccumulator,
    exploration_probability: f64,
    variant: PreflopSolverVariant,
}

impl<'a> PreflopDcfrSolver<'a> {
    #[cfg(test)]
    fn new(cache: &'a ContinuationCache, seed: u64) -> Self {
        Self::with_parameters(
            cache,
            seed,
            cache.game.dcfr.clone(),
            EXTERNAL_SAMPLING_EXPLORATION,
        )
    }

    #[cfg(test)]
    fn with_parameters(
        cache: &'a ContinuationCache,
        seed: u64,
        dcfr: DcfrParameters,
        exploration_probability: f64,
    ) -> Self {
        Self::with_variant(
            cache,
            seed,
            dcfr,
            exploration_probability,
            PreflopSolverVariant::Dcfr,
        )
    }

    fn with_variant(
        cache: &'a ContinuationCache,
        seed: u64,
        dcfr: DcfrParameters,
        exploration_probability: f64,
        variant: PreflopSolverVariant,
    ) -> Self {
        Self {
            cache,
            nodes: BTreeMap::new(),
            rng: SplitMix64::new(seed),
            iterations: 0,
            discounts: DiscountAccumulator::new(dcfr),
            exploration_probability,
            variant,
        }
    }

    fn train(&mut self, iterations: u64) {
        while self.iterations < iterations {
            if self.variant == PreflopSolverVariant::Dcfr {
                self.discounts.advance(self.iterations + 1);
            }
            let traverser = self.iterations as usize % 2;
            let deal_index = self.rng.index(self.cache.deals.len());
            let deal = self.cache.deal(deal_index);
            self.external_sampling(
                GameState::initial(&self.cache.game),
                &deal,
                deal_index,
                traverser,
                1.0,
            );
            self.iterations += 1;
        }
    }

    fn external_sampling(
        &mut self,
        state: GameState,
        deal: &Deal,
        deal_index: usize,
        traverser: usize,
        sampled_opponent_reach_ratio: f64,
    ) -> f64 {
        if state.terminal.is_some() {
            let utility = realized_utility_p0(&state, deal);
            return if traverser == 0 { utility } else { -utility };
        }
        if state.street != Street::Preflop {
            return self
                .cache
                .continuation_utility(deal_index, &state, traverser);
        }

        let actions = state.legal_actions(&self.cache.game);
        let key = information_key(&state, deal);
        let strategy = {
            let node = self
                .nodes
                .entry(key.clone())
                .or_insert_with(|| SolverNode::new(&state, deal, &actions));
            assert_eq!(
                node.action_labels,
                actions
                    .iter()
                    .map(|action| action.label.clone())
                    .collect::<Vec<_>>()
            );
            if self.variant == PreflopSolverVariant::Dcfr {
                node.apply_dcfr_discount(self.iterations + 1, &self.discounts);
            }
            node.current_strategy()
        };

        if state.actor == traverser {
            let values = actions
                .iter()
                .map(|action| {
                    self.external_sampling(
                        state.apply(action, &self.cache.game),
                        deal,
                        deal_index,
                        traverser,
                        sampled_opponent_reach_ratio,
                    )
                })
                .collect::<Vec<_>>();
            let node_value = strategy
                .iter()
                .zip(&values)
                .map(|(probability, value)| probability * value)
                .sum::<f64>();
            let node = self.nodes.get_mut(&key).expect("preflop node inserted");
            for (regret, value) in node.regrets.iter_mut().zip(values) {
                *regret += sampled_opponent_reach_ratio * (value - node_value);
                if self.variant == PreflopSolverVariant::MccfrPlus {
                    *regret = regret.max(0.0);
                }
            }
            node.regret_updates += 1;
            node_value
        } else {
            let node = self.nodes.get_mut(&key).expect("preflop node inserted");
            let averaging_weight = if self.variant == PreflopSolverVariant::MccfrPlus {
                (self.iterations + 1) as f64
            } else {
                1.0
            };
            for (sum, probability) in node.strategy_sum.iter_mut().zip(&strategy) {
                // The opponent's earlier actions were drawn from the exploratory
                // behavior policy. Correct their sampled reach back to the target
                // policy before accumulating the average strategy.
                *sum += averaging_weight * sampled_opponent_reach_ratio * probability;
            }
            node.average_visits += 1;
            let behavior = strategy
                .iter()
                .map(|probability| {
                    (1.0 - self.exploration_probability) * probability
                        + self.exploration_probability / strategy.len() as f64
                })
                .collect::<Vec<_>>();
            let selected = sample_index(&behavior, &mut self.rng);
            let importance_ratio = strategy[selected] / behavior[selected];
            importance_ratio
                * self.external_sampling(
                    state.apply(&actions[selected], &self.cache.game),
                    deal,
                    deal_index,
                    traverser,
                    sampled_opponent_reach_ratio * importance_ratio,
                )
        }
    }

    fn artifact(
        self,
        model_version: String,
        seed: u64,
        cache_sha256: String,
    ) -> PreflopPolicyArtifact {
        let solver_dcfr = self.discounts.parameters.clone();
        let solver_variant = self.variant;
        let strategies = self
            .nodes
            .into_iter()
            .map(|(key, node)| {
                let probabilities = node.average_strategy();
                let positive_regret_sum_bb = node.regrets.iter().map(|value| value.max(0.0)).sum();
                let average_reach_weight = node.strategy_sum.iter().sum();
                PreflopPolicyEntry {
                    key,
                    actor: node.actor,
                    hand_class: node.hand_class,
                    public_history: node.public_history,
                    action_labels: node.action_labels,
                    probabilities,
                    positive_regret_sum_bb,
                    regret_updates: node.regret_updates,
                    average_visits: node.average_visits,
                    average_reach_weight,
                }
            })
            .collect::<Vec<_>>();
        let mut artifact = PreflopPolicyArtifact {
            schema: POLICY_SCHEMA.to_owned(),
            model_version,
            depth_bb: self.cache.depth_bb,
            seed,
            iterations: self.iterations,
            sampling_exploration_probability: self.exploration_probability,
            solver_dcfr,
            solver_variant,
            continuation_cache_sha256: cache_sha256,
            game: self.cache.game.clone(),
            strategies,
            training_evaluation: empty_evaluation(self.cache.deals.len()),
        };
        artifact.training_evaluation = evaluate_policy(self.cache, &artifact);
        artifact
    }
}

trait Policy {
    fn strategy(
        &self,
        state: &GameState,
        deal: &Deal,
        actions: &[LegalAction],
        counters: &mut LookupCounters,
    ) -> Vec<f64>;
}

struct ArtifactPolicy {
    entries: BTreeMap<String, (Vec<String>, Vec<f64>)>,
}

impl ArtifactPolicy {
    fn new(artifact: &PreflopPolicyArtifact) -> Self {
        Self {
            entries: artifact
                .strategies
                .iter()
                .map(|entry| {
                    (
                        entry.key.clone(),
                        (entry.action_labels.clone(), entry.probabilities.clone()),
                    )
                })
                .collect(),
        }
    }
}

impl Policy for ArtifactPolicy {
    fn strategy(
        &self,
        state: &GameState,
        deal: &Deal,
        actions: &[LegalAction],
        counters: &mut LookupCounters,
    ) -> Vec<f64> {
        counters.queries += 1;
        let key = information_key(state, deal);
        let Some((labels, probabilities)) = self.entries.get(&key) else {
            counters.misses += 1;
            return vec![1.0 / actions.len() as f64; actions.len()];
        };
        assert_eq!(
            labels,
            &actions
                .iter()
                .map(|action| action.label.clone())
                .collect::<Vec<_>>()
        );
        probabilities.clone()
    }
}

struct NeuralPolicy<'a> {
    frozen: FrozenPolicy,
    config: &'a BlueprintConfig,
    strategies: RefCell<StrategyCache>,
}

type StrategyCache = BTreeMap<String, (Vec<String>, Vec<f64>)>;

impl Policy for NeuralPolicy<'_> {
    fn strategy(
        &self,
        state: &GameState,
        deal: &Deal,
        actions: &[LegalAction],
        counters: &mut LookupCounters,
    ) -> Vec<f64> {
        counters.queries += 1;
        let key = information_key(state, deal);
        if let Some((labels, probabilities)) = self.strategies.borrow().get(&key) {
            assert_eq!(
                labels,
                &actions
                    .iter()
                    .map(|action| action.label.clone())
                    .collect::<Vec<_>>()
            );
            return probabilities.clone();
        }
        let probabilities = self.frozen.strategy(state, deal, actions, self.config);
        self.strategies.borrow_mut().insert(
            key,
            (
                actions.iter().map(|action| action.label.clone()).collect(),
                probabilities.clone(),
            ),
        );
        probabilities
    }
}

#[derive(Default)]
struct LookupCounters {
    queries: usize,
    misses: usize,
}

#[derive(Clone)]
struct WeightedWorld {
    state: GameState,
    deal_index: usize,
    weight: f64,
}

#[derive(Default)]
struct ResponseStats {
    information_sets: usize,
}

pub fn build_continuation_cache(
    config: ContinuationCacheConfig,
) -> Result<ContinuationCache, Box<dyn Error>> {
    config.game.validate()?;
    if config.deals < 2 || config.rollouts_per_leaf < 2 {
        return Err("continuation cache requires at least two deals and rollouts".into());
    }
    if config.network_paths.is_empty()
        || config.rollouts_per_leaf < config.network_paths.len() as u32
        || !config
            .rollouts_per_leaf
            .is_multiple_of(config.network_paths.len() as u32)
    {
        return Err(
            "continuation cache requires an equal positive rollout count per frozen policy".into(),
        );
    }
    let policies = config
        .network_paths
        .iter()
        .map(|path| FrozenPolicy::load(path))
        .collect::<Result<Vec<_>, _>>()?;
    let leaves = enumerate_flop_leaves(&config.game);
    if leaves.is_empty() {
        return Err("preflop abstraction reaches no flop leaves".into());
    }
    let public_histories = leaves
        .iter()
        .map(|(key, state)| (*key, state.public_history.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut chance = SplitMix64::new(config.seed);
    let combos = all_combos();
    let complete_exact_combo_cycles = config.deals / (2 * combos.len());
    let balanced_exact_combo_marginals = config.deals.is_multiple_of(2 * combos.len());
    let mut deals = Vec::with_capacity(config.deals);
    let mut accepted_standard_errors = 0usize;
    let mut maximum_standard_error = 0.0f64;
    for deal_index in 0..config.deals {
        let deal = stratified_deal(&mut chance, deal_index, &combos);
        let mut continuations = BTreeMap::new();
        for (history, state) in &leaves {
            let values = (0..config.rollouts_per_leaf)
                .map(|rollout| {
                    let mut rng = SplitMix64::new(continuation_seed(
                        config.seed,
                        deal_index,
                        *history,
                        rollout,
                    ));
                    let policy = &policies[rollout as usize % policies.len()];
                    rollout_policy_value(policy, state.clone(), &deal, &config.game, &mut rng)
                })
                .collect::<Vec<_>>();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let squared = values
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>();
            let standard_error = (squared / ((values.len() - 1) * values.len()) as f64).sqrt();
            maximum_standard_error = maximum_standard_error.max(standard_error);
            accepted_standard_errors += usize::from(standard_error <= 0.02);
            continuations.insert(
                *history,
                ContinuationEstimate {
                    mean_utility_p0_bb: mean,
                    conditional_utilities_bb: None,
                    action_standard_error_bb: standard_error,
                },
            );
        }
        deals.push(CachedDeal {
            holes: deal.holes,
            board: deal.board,
            continuations,
        });
    }
    let leaf_values = deals.len() * leaves.len();
    let history_standard_errors = leaves
        .keys()
        .map(|history| {
            let values = deals
                .iter()
                .map(|deal| deal.continuations[history].mean_utility_p0_bb)
                .collect::<Vec<_>>();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let squared = values
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>();
            (squared / ((values.len() - 1) * values.len()) as f64).sqrt()
        })
        .collect::<Vec<_>>();
    let information_group_standard_errors = information_group_standard_errors(&deals);
    let cache = ContinuationCache {
        schema: CONTINUATION_SCHEMA.to_owned(),
        depth_bb: config.game.effective_stack_bb,
        seed: config.seed,
        rollouts_per_leaf: config.rollouts_per_leaf,
        chance_sampling:
            "alternating_seat_exact_combo_stratified_with_uniform_compatible_opponent_and_board"
                .to_owned(),
        complete_exact_combo_cycles,
        balanced_exact_combo_marginals,
        network_sha256: combined_file_sha256(&config.network_paths)?,
        network_sha256s: config
            .network_paths
            .iter()
            .map(|path| sha256_file(path))
            .collect::<Result<Vec<_>, _>>()?,
        policy_mixture: "frozen_v26_model_selected_round_robin_per_rollout".to_owned(),
        source_cache_sha256: None,
        source_deal_indices: None,
        game: config.game,
        public_histories,
        deals,
        validation: ContinuationValidation {
            complete: true,
            probability_values_finite: true,
            utilities_within_stack: true,
            leaf_values,
            fraction_action_se_at_most_0_02bb: accepted_standard_errors as f64 / leaf_values as f64,
            maximum_action_standard_error_bb: maximum_standard_error,
            fraction_history_mean_se_at_most_0_02bb: history_standard_errors
                .iter()
                .filter(|standard_error| **standard_error <= 0.02)
                .count() as f64
                / history_standard_errors.len() as f64,
            maximum_history_mean_standard_error_bb: history_standard_errors
                .into_iter()
                .fold(0.0, f64::max),
            fraction_information_group_mean_se_at_most_0_25bb: information_group_standard_errors
                .iter()
                .filter(|standard_error| **standard_error <= 0.25)
                .count() as f64
                / information_group_standard_errors.len().max(1) as f64,
            maximum_information_group_mean_standard_error_bb: information_group_standard_errors
                .into_iter()
                .fold(0.0, f64::max),
        },
    };
    cache.validate()?;
    Ok(cache)
}

/// Revalues an existing exact-deal design with the depth-limited public-belief
/// flop resolver. Player-conditional CFVs remain separate for the downstream
/// information-set solve; the legacy scalar is retained only for cache-format
/// compatibility. This is a research bridge, and model uncertainty remains
/// attached to every leaf, so it still fails the precision release gate.
pub fn build_resolver_continuation_cache(
    base: &ContinuationCache,
    config: ResolverContinuationCacheConfig,
) -> Result<ContinuationCache, Box<dyn Error>> {
    if config.deals < 2
        || config
            .deal_offset
            .checked_add(config.deals)
            .is_none_or(|end| end > base.deals.len())
        || config.resolver_iterations < 2
        || config.resolver_averaging_delay >= config.resolver_iterations
        || !config.value_uncertainty_bb.is_finite()
        || config.value_uncertainty_bb < 0.0
        || config.threads == 0
    {
        return Err("resolver continuation configuration is invalid".into());
    }
    let network = super::public_belief::PublicValueNetwork::read(&config.value_network_path)?;
    let evaluation_network = config
        .evaluation_value_network_path
        .as_ref()
        .map(|path| super::public_belief::PublicValueNetwork::read(path))
        .transpose()?;
    if evaluation_network
        .as_ref()
        .is_some_and(|evaluation| !network.has_distinct_training_identity(evaluation))
    {
        return Err(
            "resolver continuation cross-scoring requires an independent value network".into(),
        );
    }
    let (leaf_ranges, range_policy_sha256) = if let Some(path) = &config.range_policy_path {
        let policy = PreflopPolicyArtifact::read(path)?;
        if policy.game != base.game {
            return Err("range policy and continuation cache use different games".into());
        }
        (resolver_leaf_ranges(&policy)?, Some(sha256_file(path)?))
    } else {
        (
            enumerate_flop_leaves(&base.game)
                .keys()
                .map(|history| {
                    (
                        *history,
                        std::array::from_fn(|_| vec![1.0; super::public_belief::COMBO_COUNT]),
                    )
                })
                .collect(),
            None,
        )
    };
    let leaf_ranges = std::sync::Arc::new(leaf_ranges);
    let leaves = enumerate_flop_leaves(&base.game);
    let public_histories = leaves
        .iter()
        .map(|(key, state)| (*key, state.public_history.clone()))
        .collect::<BTreeMap<_, _>>();
    if public_histories != base.public_histories {
        return Err("base cache public leaves do not match the current game".into());
    }
    let source_deals = base
        .deals
        .iter()
        .skip(config.deal_offset)
        .take(config.deals)
        .cloned()
        .collect::<Vec<_>>();
    let source_deal_indices =
        (config.deal_offset..config.deal_offset + config.deals).collect::<Vec<_>>();
    let worker_count = config.threads.min(source_deals.len()).max(1);
    let resolver_threads = (config.threads / worker_count).max(1);
    let solved = std::thread::scope(|scope| {
        let mut workers = Vec::with_capacity(worker_count);
        for worker in 0..worker_count {
            let assigned = source_deals
                .iter()
                .enumerate()
                .skip(worker)
                .step_by(worker_count)
                .collect::<Vec<_>>();
            let network = network.clone();
            let evaluation_network = evaluation_network.clone();
            let game = base.game.clone();
            let leaf_ranges = leaf_ranges.clone();
            let leaves = &leaves;
            workers.push(scope.spawn(move || {
                let mut results = Vec::with_capacity(assigned.len() * leaves.len());
                for (deal_index, cached) in assigned {
                    for (history, leaf) in leaves {
                        let flop = [cached.board[0], cached.board[1], cached.board[2]];
                        let ranges = leaf_ranges
                            .get(history)
                            .expect("validated resolver leaf range")
                            .clone();
                        let resolver_config = super::public_belief::FlopResolveConfig {
                            game: game.clone(),
                            state: super::public_belief::PublicBeliefState::flop_start(
                                flop,
                                leaf.actor,
                                leaf.invested,
                                ranges,
                            ),
                            iterations: config.resolver_iterations,
                            averaging_delay: config.resolver_averaging_delay,
                            regret_matching_plus: config.resolver_regret_matching_plus,
                            value_network: network.clone(),
                            threads: resolver_threads,
                        };
                        let counterfactual_values_bb = if let Some(evaluation) = &evaluation_network
                        {
                            super::public_belief::solve_flop_cross_evaluated(
                                resolver_config,
                                evaluation.clone(),
                            )?
                            .counterfactual_values_bb
                        } else {
                            super::public_belief::solve_flop_continuation_values(resolver_config)?
                                .counterfactual_values_bb
                        };
                        let first = Combo::new(cached.holes[0][0], cached.holes[0][1]).key();
                        let second = Combo::new(cached.holes[1][0], cached.holes[1][1]).key();
                        let reconstructed = (counterfactual_values_bb[0][first] as f64
                            - counterfactual_values_bb[1][second] as f64)
                            / 2.0;
                        results.push((
                            deal_index,
                            *history,
                            ContinuationEstimate {
                                mean_utility_p0_bb: reconstructed
                                    .clamp(-game.effective_stack_bb, game.effective_stack_bb),
                                conditional_utilities_bb: Some([
                                    counterfactual_values_bb[0][first] as f64,
                                    counterfactual_values_bb[1][second] as f64,
                                ]),
                                action_standard_error_bb: config.value_uncertainty_bb,
                            },
                        ));
                    }
                }
                Ok::<_, String>(results)
            }));
        }
        let mut values = Vec::with_capacity(source_deals.len() * leaves.len());
        for worker in workers {
            values.extend(
                worker
                    .join()
                    .map_err(|_| "resolver continuation worker panicked".to_owned())??,
            );
        }
        Ok::<_, String>(values)
    })?;
    let mut deals = source_deals
        .into_iter()
        .map(|cached| CachedDeal {
            holes: cached.holes,
            board: cached.board,
            continuations: BTreeMap::new(),
        })
        .collect::<Vec<_>>();
    for (deal_index, history, estimate) in solved {
        deals[deal_index].continuations.insert(history, estimate);
    }
    let validation = continuation_validation(&deals, &public_histories);
    let network_sha256 = sha256_file(&config.value_network_path)?;
    let evaluation_network_sha256 = config
        .evaluation_value_network_path
        .as_ref()
        .map(|path| sha256_file(path))
        .transpose()?;
    let source_cache_sha256 = sha256_file(&config.source_cache_path)?;
    let range_method = range_policy_sha256
        .as_deref()
        .map(|digest| format!("tabular_preflop_policy_{digest}"))
        .unwrap_or_else(|| "uniform_range_control".to_owned());
    let cache = ContinuationCache {
        schema: CONTINUATION_SCHEMA.to_owned(),
        depth_bb: base.depth_bb,
        seed: base.seed ^ 0xF10F_C0F5,
        rollouts_per_leaf: 2,
        chance_sampling: format!(
            "{}_revalued_by_depth_limited_flop_public_belief_resolver",
            base.chance_sampling
        ),
        complete_exact_combo_cycles: complete_source_deal_cycles(&source_deal_indices),
        balanced_exact_combo_marginals: balanced_source_deal_indices(&source_deal_indices),
        network_sha256: network_sha256.clone(),
        network_sha256s: std::iter::once(network_sha256.clone())
            .chain(evaluation_network_sha256.clone())
            .collect(),
        policy_mixture: format!(
            "turn_cfv_network_{}_plus_{}_iteration_exact_all_in_flop_{}_ranges_from_{}",
            if evaluation_network_sha256.is_some() {
                "cross_scored"
            } else {
                "self_scored"
            },
            config.resolver_iterations,
            if config.resolver_regret_matching_plus {
                "dcfr_plus"
            } else {
                "dcfr"
            },
            range_method
        ),
        source_cache_sha256: Some(source_cache_sha256),
        source_deal_indices: Some(source_deal_indices),
        game: base.game.clone(),
        public_histories,
        deals,
        validation,
    };
    cache.validate()?;
    Ok(cache)
}

type ResolverRangePolicy = BTreeMap<(usize, String, Vec<String>), (Vec<String>, Vec<f64>)>;
type ResolverLeafRanges = BTreeMap<u64, [Vec<f64>; 2]>;

fn resolver_leaf_ranges(
    artifact: &PreflopPolicyArtifact,
) -> Result<ResolverLeafRanges, Box<dyn Error>> {
    let policy = artifact
        .strategies
        .iter()
        .map(|entry| {
            (
                (
                    entry.actor,
                    entry.hand_class.clone(),
                    entry.public_history.clone(),
                ),
                (entry.action_labels.clone(), entry.probabilities.clone()),
            )
        })
        .collect::<ResolverRangePolicy>();
    let combo_classes = all_combos()
        .iter()
        .map(|combo| combo.label())
        .collect::<Vec<_>>();

    fn visit(
        state: GameState,
        reaches: [Vec<f64>; 2],
        game: &BlueprintConfig,
        policy: &ResolverRangePolicy,
        combo_classes: &[String],
        leaves: &mut BTreeMap<u64, [Vec<f64>; 2]>,
    ) -> Result<(), String> {
        if state.terminal.is_some() {
            return Ok(());
        }
        if state.street != Street::Preflop {
            let key = history_key(&state);
            if let Some(existing) = leaves.insert(key, reaches.clone()) {
                if existing != reaches {
                    return Err("public history produced inconsistent range factors".to_owned());
                }
            }
            return Ok(());
        }
        let actions = state.legal_actions(game);
        let labels = actions
            .iter()
            .map(|action| action.label.clone())
            .collect::<Vec<_>>();
        for (action_index, action) in actions.iter().enumerate() {
            let mut child_reaches = reaches.clone();
            condition_range_for_action(
                &state,
                &labels,
                action_index,
                policy,
                combo_classes,
                &mut child_reaches[state.actor],
            )?;
            visit(
                state.apply(action, game),
                child_reaches,
                game,
                policy,
                combo_classes,
                leaves,
            )?;
        }
        Ok(())
    }

    let mut leaves = BTreeMap::new();
    visit(
        GameState::initial(&artifact.game),
        std::array::from_fn(|_| vec![1.0; super::public_belief::COMBO_COUNT]),
        &artifact.game,
        &policy,
        &combo_classes,
        &mut leaves,
    )?;
    let expected = enumerate_flop_leaves(&artifact.game);
    if leaves.keys().copied().collect::<BTreeSet<_>>()
        != expected.keys().copied().collect::<BTreeSet<_>>()
    {
        return Err("range policy did not cover every preflop continuation leaf".into());
    }
    Ok(leaves)
}

fn condition_range_for_action(
    state: &GameState,
    action_labels: &[String],
    action_index: usize,
    policy: &ResolverRangePolicy,
    combo_classes: &[String],
    range: &mut [f64],
) -> Result<(), String> {
    for (combo, class) in combo_classes.iter().enumerate() {
        let Some((stored_labels, probabilities)) =
            policy.get(&(state.actor, class.clone(), state.public_history.clone()))
        else {
            return Err(format!(
                "range policy is missing p{} {} at {}",
                state.actor,
                class,
                state.public_history.join("/")
            ));
        };
        if stored_labels != action_labels || probabilities.len() != action_labels.len() {
            return Err("range policy action labels do not match the resolver game".to_owned());
        }
        range[combo] *= probabilities[action_index].max(RESOLVER_RANGE_PROBABILITY_FLOOR);
    }
    Ok(())
}

pub fn solve_preflop(
    cache: &ContinuationCache,
    iterations: u64,
    seed: u64,
    model_version: String,
    cache_path: &Path,
) -> Result<PreflopPolicyArtifact, Box<dyn Error>> {
    solve_preflop_with_parameters(
        cache,
        iterations,
        seed,
        model_version,
        cache_path,
        cache.game.dcfr.clone(),
        EXTERNAL_SAMPLING_EXPLORATION,
    )
}

pub fn solve_preflop_with_parameters(
    cache: &ContinuationCache,
    iterations: u64,
    seed: u64,
    model_version: String,
    cache_path: &Path,
    dcfr: DcfrParameters,
    exploration_probability: f64,
) -> Result<PreflopPolicyArtifact, Box<dyn Error>> {
    solve_preflop_with_options(
        cache,
        cache_path,
        PreflopSolveOptions {
            iterations,
            seed,
            model_version,
            dcfr,
            exploration_probability,
            variant: PreflopSolverVariant::Dcfr,
        },
    )
}

pub fn solve_preflop_with_options(
    cache: &ContinuationCache,
    cache_path: &Path,
    options: PreflopSolveOptions,
) -> Result<PreflopPolicyArtifact, Box<dyn Error>> {
    if options.iterations == 0 {
        return Err("preflop DCFR iterations must be positive".into());
    }
    if !options.dcfr.positive_regret_exponent.is_finite()
        || !options.dcfr.negative_regret_exponent.is_finite()
        || !options.dcfr.strategy_exponent.is_finite()
        || options.dcfr.positive_regret_exponent < 0.0
        || options.dcfr.negative_regret_exponent < 0.0
        || options.dcfr.strategy_exponent < 0.0
        || !(0.0..1.0).contains(&options.exploration_probability)
    {
        return Err("preflop DCFR parameters are invalid".into());
    }
    let mut solver = PreflopDcfrSolver::with_variant(
        cache,
        options.seed,
        options.dcfr,
        options.exploration_probability,
        options.variant,
    );
    solver.train(options.iterations);
    let artifact = solver.artifact(
        options.model_version,
        options.seed,
        sha256_file(cache_path)?,
    );
    artifact.validate()?;
    Ok(artifact)
}

pub fn export_compact_preflop_policy(
    artifact: &PreflopPolicyArtifact,
    output: &Path,
) -> Result<CompactPreflopPolicySummary, Box<dyn Error>> {
    const MAGIC: &[u8; 8] = b"HUPFTAB1";
    const FORMAT_VERSION: u16 = 1;
    artifact.validate()?;
    if artifact.model_version.len() > u16::MAX as usize
        || artifact.strategies.len() > u32::MAX as usize
    {
        return Err("compact preflop policy metadata is too large".into());
    }
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let temporary = output.with_extension("tmp");
    let mut writer = BufWriter::new(fs::File::create(&temporary)?);
    writer.write_all(MAGIC)?;
    writer.write_all(&FORMAT_VERSION.to_le_bytes())?;
    writer.write_all(&artifact.depth_bb.to_le_bytes())?;
    writer.write_all(&(artifact.model_version.len() as u16).to_le_bytes())?;
    writer.write_all(artifact.model_version.as_bytes())?;
    writer.write_all(&(artifact.strategies.len() as u32).to_le_bytes())?;
    let mut probabilities = 0usize;
    let mut maximum_error = 0.0f64;
    let mut sums_valid = true;
    for entry in &artifact.strategies {
        if entry.key.len() > u16::MAX as usize
            || entry.action_labels.len() > u8::MAX as usize
            || entry
                .action_labels
                .iter()
                .any(|label| label.len() > u8::MAX as usize)
        {
            return Err("compact preflop policy entry is too large".into());
        }
        let quantized = quantize_probabilities_u16(&entry.probabilities);
        sums_valid &= quantized.iter().map(|value| *value as u64).sum::<u64>() == u16::MAX as u64;
        for (original, encoded) in entry.probabilities.iter().zip(&quantized) {
            maximum_error = maximum_error.max((original - *encoded as f64 / u16::MAX as f64).abs());
        }
        probabilities += quantized.len();
        writer.write_all(&(entry.key.len() as u16).to_le_bytes())?;
        writer.write_all(entry.key.as_bytes())?;
        writer.write_all(&[entry.action_labels.len() as u8])?;
        for label in &entry.action_labels {
            writer.write_all(&[label.len() as u8])?;
            writer.write_all(label.as_bytes())?;
        }
        for probability in quantized {
            writer.write_all(&probability.to_le_bytes())?;
        }
    }
    writer.flush()?;
    drop(writer);
    fs::rename(&temporary, output)?;
    let bytes = fs::metadata(output)?.len();
    Ok(CompactPreflopPolicySummary {
        schema: "hu-compact-tabular-preflop-policy-summary-v1",
        format_version: FORMAT_VERSION,
        model_version: artifact.model_version.clone(),
        information_sets: artifact.strategies.len(),
        probabilities,
        bytes,
        sha256: sha256_file(output)?,
        maximum_probability_quantization_error: maximum_error,
        quantized_probability_sums_valid: sums_valid,
        output: output.display().to_string(),
    })
}

fn quantize_probabilities_u16(probabilities: &[f64]) -> Vec<u16> {
    let scale = u16::MAX as f64;
    let scaled = probabilities
        .iter()
        .map(|probability| probability * scale)
        .collect::<Vec<_>>();
    let mut quantized = scaled
        .iter()
        .map(|value| value.floor() as u16)
        .collect::<Vec<_>>();
    let assigned = quantized.iter().map(|value| *value as u64).sum::<u64>();
    let remaining = u16::MAX as u64 - assigned;
    let mut order = scaled
        .iter()
        .enumerate()
        .map(|(index, value)| (index, value - value.floor()))
        .collect::<Vec<_>>();
    order.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    for (index, _) in order.into_iter().take(remaining as usize) {
        quantized[index] += 1;
    }
    quantized
}

pub fn compare_continuation_caches(
    first: &ContinuationCache,
    second: &ContinuationCache,
) -> Result<ContinuationCacheComparison, Box<dyn Error>> {
    if first.game != second.game || first.public_histories != second.public_histories {
        return Err("continuation caches use different abstract games".into());
    }
    let players = [0usize, 1usize].map(|player| {
        let first_groups = continuation_groups(first, player);
        let second_groups = continuation_groups(second, player);
        let union = first_groups
            .keys()
            .chain(second_groups.keys())
            .collect::<BTreeSet<_>>();
        let common = first_groups
            .keys()
            .filter(|key| second_groups.contains_key(*key))
            .collect::<Vec<_>>();
        let mut total_weight = 0.0;
        let mut absolute = 0.0;
        let mut squared = 0.0;
        let mut maximum = 0.0f64;
        for key in &common {
            let (first_sum, first_count) = first_groups[*key];
            let (second_sum, second_count) = second_groups[*key];
            let delta = (first_sum / first_count as f64 - second_sum / second_count as f64).abs();
            let weight = first_count.min(second_count) as f64;
            total_weight += weight;
            absolute += weight * delta;
            squared += weight * delta * delta;
            maximum = maximum.max(delta);
        }
        ContinuationPlayerComparison {
            player,
            common_groups: common.len(),
            union_groups: union.len(),
            group_coverage: common.len() as f64 / union.len().max(1) as f64,
            sample_weighted_mean_absolute_delta_bb: absolute / total_weight.max(1.0),
            sample_weighted_root_mean_squared_delta_bb: (squared / total_weight.max(1.0)).sqrt(),
            maximum_group_delta_bb: maximum,
        }
    });
    let first_group_standard_errors = information_group_standard_errors(&first.deals);
    let second_group_standard_errors = information_group_standard_errors(&second.deals);
    let group_fraction = |values: &[f64]| {
        values.iter().filter(|value| **value <= 0.25).count() as f64 / values.len().max(1) as f64
    };
    Ok(ContinuationCacheComparison {
        schema: "hu-preflop-continuation-cache-comparison-v1".to_owned(),
        depth_bb: first.depth_bb,
        cache_seeds: [first.seed, second.seed],
        deal_counts: [first.deals.len(), second.deals.len()],
        complete_exact_combo_cycles: [
            first.complete_exact_combo_cycles,
            second.complete_exact_combo_cycles,
        ],
        network_mixtures_match: first.network_sha256 == second.network_sha256,
        fraction_information_group_mean_se_at_most_0_25bb: [
            group_fraction(&first_group_standard_errors),
            group_fraction(&second_group_standard_errors),
        ],
        maximum_information_group_mean_standard_error_bb: [
            first_group_standard_errors
                .into_iter()
                .fold(0.0, f64::max),
            second_group_standard_errors
                .into_iter()
                .fold(0.0, f64::max),
        ],
        players,
        interpretation: "independent-chance agreement of frozen continuation utility means grouped by observable preflop hand class and public leaf and weighted by group sample count, not strategic reach; this measures oracle stability, not equilibrium exploitability"
            .to_owned(),
    })
}

pub fn merge_continuation_caches(
    first: &ContinuationCache,
    second: &ContinuationCache,
) -> Result<ContinuationCache, Box<dyn Error>> {
    if first.game != second.game
        || first.public_histories != second.public_histories
        || first.rollouts_per_leaf != second.rollouts_per_leaf
        || first.network_sha256 != second.network_sha256
        || first.network_sha256s != second.network_sha256s
        || first.policy_mixture != second.policy_mixture
        || first.source_cache_sha256 != second.source_cache_sha256
        || (first.source_cache_sha256.is_some() && first.seed != second.seed)
    {
        return Err(
            "only compatible continuation caches with the same frozen policy mixture can be merged"
                .into(),
        );
    }
    let (deals, source_deal_indices) =
        match (&first.source_deal_indices, &second.source_deal_indices) {
            (Some(first_indices), Some(second_indices)) => {
                let mut indexed = first_indices
                    .iter()
                    .copied()
                    .zip(first.deals.iter().cloned())
                    .chain(
                        second_indices
                            .iter()
                            .copied()
                            .zip(second.deals.iter().cloned()),
                    )
                    .collect::<Vec<_>>();
                indexed.sort_by_key(|(index, _)| *index);
                if indexed.windows(2).any(|pair| pair[0].0 == pair[1].0) {
                    return Err("continuation cache chunks overlap source deal indices".into());
                }
                let (indices, deals): (Vec<_>, Vec<_>) = indexed.into_iter().unzip();
                (deals, Some(indices))
            }
            (None, None) => {
                let mut deals = first.deals.clone();
                deals.extend(second.deals.clone());
                (deals, None)
            }
            _ => {
                return Err("continuation cache provenance is incompatible".into());
            }
        };
    let validation = continuation_validation(&deals, &first.public_histories);
    let complete_exact_combo_cycles = source_deal_indices
        .as_deref()
        .map(complete_source_deal_cycles)
        .unwrap_or_else(|| deals.len() / (2 * all_combos().len()));
    let balanced_exact_combo_marginals = source_deal_indices
        .as_deref()
        .map(balanced_source_deal_indices)
        .unwrap_or_else(|| deals.len().is_multiple_of(2 * all_combos().len()));
    let provenance_merged = source_deal_indices.is_some();
    let cache = ContinuationCache {
        schema: CONTINUATION_SCHEMA.to_owned(),
        depth_bb: first.depth_bb,
        seed: if provenance_merged {
            first.seed
        } else {
            first.seed ^ second.seed.rotate_left(1)
        },
        rollouts_per_leaf: first.rollouts_per_leaf,
        chance_sampling: if provenance_merged {
            "concatenated_provenance_checked_continuation_chunks".to_owned()
        } else {
            "concatenated_independent_balanced_exact_combo_cycles".to_owned()
        },
        complete_exact_combo_cycles,
        balanced_exact_combo_marginals,
        network_sha256: first.network_sha256.clone(),
        network_sha256s: first.network_sha256s.clone(),
        policy_mixture: first.policy_mixture.clone(),
        source_cache_sha256: first.source_cache_sha256.clone(),
        source_deal_indices,
        game: first.game.clone(),
        public_histories: first.public_histories.clone(),
        deals,
        validation,
    };
    cache.validate()?;
    Ok(cache)
}

pub fn refresh_continuation_cache_validation(
    mut cache: ContinuationCache,
) -> Result<ContinuationCache, Box<dyn Error>> {
    cache.validation = continuation_validation(&cache.deals, &cache.public_histories);
    cache.validate()?;
    Ok(cache)
}

fn continuation_validation(
    deals: &[CachedDeal],
    public_histories: &BTreeMap<u64, Vec<String>>,
) -> ContinuationValidation {
    let estimates = deals
        .iter()
        .flat_map(|deal| deal.continuations.values())
        .collect::<Vec<_>>();
    let history_standard_errors = public_histories
        .keys()
        .flat_map(|history| {
            (0..2).map(move |player| {
                let values = deals
                    .iter()
                    .map(|deal| continuation_estimate_utility(&deal.continuations[history], player))
                    .collect::<Vec<_>>();
                sample_standard_error(&values)
            })
        })
        .collect::<Vec<_>>();
    let information_group_standard_errors = information_group_standard_errors(deals);
    ContinuationValidation {
        complete: true,
        probability_values_finite: true,
        utilities_within_stack: true,
        leaf_values: estimates.len(),
        fraction_action_se_at_most_0_02bb: estimates
            .iter()
            .filter(|estimate| estimate.action_standard_error_bb <= 0.02)
            .count() as f64
            / estimates.len().max(1) as f64,
        maximum_action_standard_error_bb: estimates
            .iter()
            .map(|estimate| estimate.action_standard_error_bb)
            .fold(0.0, f64::max),
        fraction_history_mean_se_at_most_0_02bb: history_standard_errors
            .iter()
            .filter(|standard_error| **standard_error <= 0.02)
            .count() as f64
            / history_standard_errors.len().max(1) as f64,
        maximum_history_mean_standard_error_bb: history_standard_errors
            .into_iter()
            .fold(0.0, f64::max),
        fraction_information_group_mean_se_at_most_0_25bb: information_group_standard_errors
            .iter()
            .filter(|standard_error| **standard_error <= 0.25)
            .count() as f64
            / information_group_standard_errors.len().max(1) as f64,
        maximum_information_group_mean_standard_error_bb: information_group_standard_errors
            .into_iter()
            .fold(0.0, f64::max),
    }
}

fn information_group_standard_errors(deals: &[CachedDeal]) -> Vec<f64> {
    let mut groups = BTreeMap::<(usize, String, u64), (usize, f64, f64)>::new();
    for cached in deals {
        let deal = Deal::from_sampled_cards(cached.holes, cached.board);
        for player in 0..2 {
            let class = hand_class(&deal, player);
            for (history, estimate) in &cached.continuations {
                let utility = continuation_estimate_utility(estimate, player);
                let group = groups
                    .entry((player, class.clone(), *history))
                    .or_insert((0, 0.0, 0.0));
                group.0 += 1;
                group.1 += utility;
                group.2 += utility * utility;
            }
        }
    }
    groups
        .into_values()
        .map(|(count, sum, squared_sum)| {
            if count < 2 {
                return 0.0;
            }
            let count = count as f64;
            let variance = ((squared_sum - sum * sum / count) / (count - 1.0)).max(0.0);
            (variance / count).sqrt()
        })
        .collect()
}

fn sample_standard_error(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let squared = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>();
    (squared / ((values.len() - 1) * values.len()) as f64).sqrt()
}

fn continuation_estimate_utility(estimate: &ContinuationEstimate, player: usize) -> f64 {
    estimate
        .conditional_utilities_bb
        .map(|utilities| utilities[player])
        .unwrap_or_else(|| {
            if player == 0 {
                estimate.mean_utility_p0_bb
            } else {
                -estimate.mean_utility_p0_bb
            }
        })
}

fn continuation_groups(
    cache: &ContinuationCache,
    player: usize,
) -> BTreeMap<(String, u64), (f64, usize)> {
    let mut groups = BTreeMap::<(String, u64), (f64, usize)>::new();
    for (deal_index, cached) in cache.deals.iter().enumerate() {
        let deal = cache.deal(deal_index);
        let class = hand_class(&deal, player);
        for (history, estimate) in &cached.continuations {
            let utility = continuation_estimate_utility(estimate, player);
            let group = groups.entry((class.clone(), *history)).or_insert((0.0, 0));
            group.0 += utility;
            group.1 += 1;
        }
    }
    groups
}

pub fn evaluate_policy(
    cache: &ContinuationCache,
    artifact: &PreflopPolicyArtifact,
) -> PreflopEvaluation {
    let policy = ArtifactPolicy::new(artifact);
    let worlds = root_worlds(cache);
    let mut counters = LookupCounters::default();
    let policy_value = policy_value_worlds(cache, &policy, worlds.clone(), 0, &mut counters);
    let policy_value_p1 = policy_value_worlds(cache, &policy, worlds.clone(), 1, &mut counters);
    let mut first_stats = ResponseStats::default();
    let first = best_response_worlds(
        cache,
        &policy,
        worlds.clone(),
        0,
        &mut first_stats,
        &mut counters,
    );
    let mut second_stats = ResponseStats::default();
    let second = best_response_worlds(cache, &policy, worlds, 1, &mut second_stats, &mut counters);
    let nash_conv = (first + second - policy_value - policy_value_p1).max(0.0);
    PreflopEvaluation {
        schema: EVALUATION_SCHEMA.to_owned(),
        corpus_deals: cache.deals.len(),
        policy_value_p0_bb: policy_value,
        policy_value_p1_bb: policy_value_p1,
        policy_value_zero_sum_residual_bb: (policy_value + policy_value_p1).abs(),
        player_zero_best_response_bb: first,
        player_one_best_response_bb: second,
        nash_conv_bb: nash_conv,
        exploitability_bb_per_hand: nash_conv / 2.0,
        responder_information_sets: [first_stats.information_sets, second_stats.information_sets],
        policy_lookup_coverage: if counters.queries == 0 {
            0.0
        } else {
            1.0 - counters.misses as f64 / counters.queries as f64
        },
        interpretation: "exact information-set best response in the sampled preflop game with player-conditional frozen postflop continuation values when available; not full-game exploitability".to_owned(),
    }
}

pub fn attribute_policy_leaks(
    cache: &ContinuationCache,
    artifact: &PreflopPolicyArtifact,
    top_per_player: usize,
) -> Result<PreflopLeakAttribution, Box<dyn Error>> {
    if cache.game != artifact.game {
        return Err("preflop policy and continuation cache use different games".into());
    }
    if top_per_player == 0 {
        return Err("preflop leak attribution requires at least one row per player".into());
    }
    let policy = ArtifactPolicy::new(artifact);
    let mut counters = LookupCounters::default();
    let mut players: [Vec<PreflopLeakEntry>; 2] = std::array::from_fn(|_| Vec::new());
    collect_local_leak_entries(
        cache,
        &policy,
        root_worlds(cache),
        &mut players,
        &mut counters,
    );
    let evaluated_information_sets = [players[0].len(), players[1].len()];
    let total_policy_reach_weighted_local_gain_bb = std::array::from_fn(|player| {
        players[player]
            .iter()
            .map(|entry| entry.reach_weighted_ev_gain_bb_per_hand)
            .sum()
    });
    let public_histories = std::array::from_fn(|player| {
        let mut groups = BTreeMap::<Vec<String>, (usize, f64, f64)>::new();
        for entry in &players[player] {
            let group = groups
                .entry(entry.public_history.clone())
                .or_insert((0, 0.0, 0.0));
            group.0 += 1;
            group.1 += entry.reach_probability;
            group.2 += entry.reach_weighted_ev_gain_bb_per_hand;
        }
        let mut groups = groups
            .into_iter()
            .map(
                |(public_history, (information_sets, policy_reach_probability, gain))| {
                    PreflopPublicHistoryLeak {
                        player,
                        public_history,
                        information_sets,
                        policy_reach_probability,
                        reach_weighted_ev_gain_bb_per_hand: gain,
                    }
                },
            )
            .collect::<Vec<_>>();
        groups.sort_by(|left, right| {
            right
                .reach_weighted_ev_gain_bb_per_hand
                .total_cmp(&left.reach_weighted_ev_gain_bb_per_hand)
                .then_with(|| left.public_history.cmp(&right.public_history))
        });
        groups
    });
    for entries in &mut players {
        entries.sort_by(|left, right| {
            right
                .reach_weighted_ev_gain_bb_per_hand
                .total_cmp(&left.reach_weighted_ev_gain_bb_per_hand)
                .then_with(|| left.key.cmp(&right.key))
        });
        entries.truncate(top_per_player);
    }
    Ok(PreflopLeakAttribution {
        schema: ATTRIBUTION_SCHEMA.to_owned(),
        corpus_deals: cache.deals.len(),
        policy_model_version: artifact.model_version.clone(),
        top_per_player,
        evaluated_information_sets,
        total_policy_reach_weighted_local_gain_bb,
        policy_lookup_coverage: if counters.queries == 0 {
            0.0
        } else {
            1.0 - counters.misses as f64 / counters.queries as f64
        },
        players,
        public_histories,
        interpretation: "policy-reach-weighted one-step deviation gains with all later decisions fixed to the evaluated policy; rows localize leaks but are not additive best-response contributions or a full-game exploitability estimate".to_owned(),
    })
}

pub fn evaluate_neural_policy(
    cache: &ContinuationCache,
    network_path: &Path,
) -> Result<PreflopEvaluation, Box<dyn Error>> {
    let policy = NeuralPolicy {
        frozen: FrozenPolicy::load(network_path)?,
        config: &cache.game,
        strategies: RefCell::new(BTreeMap::new()),
    };
    let worlds = root_worlds(cache);
    let mut counters = LookupCounters::default();
    let policy_value = policy_value_worlds(cache, &policy, worlds.clone(), 0, &mut counters);
    let policy_value_p1 = policy_value_worlds(cache, &policy, worlds.clone(), 1, &mut counters);
    let mut first_stats = ResponseStats::default();
    let first = best_response_worlds(
        cache,
        &policy,
        worlds.clone(),
        0,
        &mut first_stats,
        &mut counters,
    );
    let mut second_stats = ResponseStats::default();
    let second = best_response_worlds(cache, &policy, worlds, 1, &mut second_stats, &mut counters);
    let nash_conv = (first + second - policy_value - policy_value_p1).max(0.0);
    Ok(PreflopEvaluation {
        schema: EVALUATION_SCHEMA.to_owned(),
        corpus_deals: cache.deals.len(),
        policy_value_p0_bb: policy_value,
        policy_value_p1_bb: policy_value_p1,
        policy_value_zero_sum_residual_bb: (policy_value + policy_value_p1).abs(),
        player_zero_best_response_bb: first,
        player_one_best_response_bb: second,
        nash_conv_bb: nash_conv,
        exploitability_bb_per_hand: nash_conv / 2.0,
        responder_information_sets: [first_stats.information_sets, second_stats.information_sets],
        policy_lookup_coverage: 1.0,
        interpretation: "exact information-set best response to a neural preflop policy in the sampled preflop game with player-conditional frozen postflop continuation values when available; not full-game exploitability".to_owned(),
    })
}

pub fn export_distillation_dataset(
    cache: &ContinuationCache,
    artifact: &PreflopPolicyArtifact,
    output: &Path,
) -> Result<DistillationDatasetSummary, Box<dyn Error>> {
    if cache.game != artifact.game {
        return Err("preflop policy and continuation cache use different games".into());
    }
    let entries = artifact
        .strategies
        .iter()
        .map(|entry| (entry.key.clone(), entry))
        .collect::<BTreeMap<_, _>>();
    let mut records = BTreeMap::<String, serde_json::Value>::new();
    let mut covered_information_sets = BTreeSet::new();
    for deal_index in 0..cache.deals.len() {
        let deal = cache.deal(deal_index);
        collect_distillation_records(
            GameState::initial(&cache.game),
            &deal,
            &cache.game,
            &entries,
            &mut records,
            &mut covered_information_sets,
        );
        if covered_information_sets.len() == entries.len()
            && cache.balanced_exact_combo_marginals
            && deal_index + 1 >= 2 * all_combos().len()
        {
            break;
        }
    }
    if records.is_empty() {
        return Err("preflop policy produced no distillation records".into());
    }
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let temporary = output.with_extension("tmp");
    let writer = BufWriter::new(fs::File::create(&temporary)?);
    let mut gzip = GzEncoder::new(writer, Compression::fast());
    let metadata = serde_json::json!({
        "record_type": "metadata",
        "schema": "hu-neural-traversal-jsonl-v7",
        "state_feature_schema": "hu-cash-trajectory-poker-aware-v4",
        "state_feature_count": super::neural::STATE_FEATURE_COUNT,
        "action_feature_schema": "hu-cash-legal-action-v1",
        "action_feature_count": super::neural::ACTION_FEATURE_COUNT,
        "depth_bb": cache.depth_bb,
        "seed": artifact.seed,
        "start_iteration": 0,
        "traversals": 0,
        "records": records.len(),
        "truncated": covered_information_sets.len() != entries.len(),
        "sampling_mode": "external_sampling",
        "distillation_sampling_mode": "enumerated_preflop_tree_with_exact_private_combo_expansion",
        "value_rollouts_per_action": 1,
        "evaluates_trajectory_action_values": false,
        "action_abstraction": cache.game.action_abstraction,
        "teacher": artifact.model_version,
    });
    serde_json::to_writer(&mut gzip, &metadata)?;
    gzip.write_all(b"\n")?;
    for record in records.values() {
        serde_json::to_writer(&mut gzip, record)?;
        gzip.write_all(b"\n")?;
    }
    gzip.finish()?.flush()?;
    fs::rename(temporary, output)?;
    Ok(DistillationDatasetSummary {
        schema: "hu-preflop-distillation-dataset-summary-v1",
        records: records.len(),
        policy_information_sets: entries.len(),
        covered_policy_information_sets: covered_information_sets.len(),
        coverage: covered_information_sets.len() as f64 / entries.len() as f64,
        output: output.display().to_string(),
    })
}

fn empty_evaluation(deals: usize) -> PreflopEvaluation {
    PreflopEvaluation {
        schema: EVALUATION_SCHEMA.to_owned(),
        corpus_deals: deals,
        policy_value_p0_bb: 0.0,
        policy_value_p1_bb: 0.0,
        policy_value_zero_sum_residual_bb: 0.0,
        player_zero_best_response_bb: 0.0,
        player_one_best_response_bb: 0.0,
        nash_conv_bb: 0.0,
        exploitability_bb_per_hand: 0.0,
        responder_information_sets: [0, 0],
        policy_lookup_coverage: 0.0,
        interpretation: "not evaluated".to_owned(),
    }
}

fn collect_distillation_records(
    state: GameState,
    deal: &Deal,
    config: &BlueprintConfig,
    entries: &BTreeMap<String, &PreflopPolicyEntry>,
    records: &mut BTreeMap<String, serde_json::Value>,
    covered_information_sets: &mut BTreeSet<String>,
) {
    if state.terminal.is_some() || state.street != Street::Preflop {
        return;
    }
    let actions = state.legal_actions(config);
    let key = information_key(&state, deal);
    if let Some(entry) = entries.get(&key) {
        covered_information_sets.insert(key.clone());
        let mut private_cards = deal.holes[state.actor];
        private_cards.sort_unstable();
        let record_key = format!("{key}|{:02}:{:02}", private_cards[0], private_cards[1]);
        if let Entry::Vacant(slot) = records.entry(record_key) {
            assert_eq!(
                entry.action_labels,
                actions
                    .iter()
                    .map(|action| action.label.clone())
                    .collect::<Vec<_>>()
            );
            let weight = (entry.average_reach_weight.max(1e-6)
                / exact_combo_count(&entry.hand_class) as f64) as f32;
            slot.insert(average_strategy_record_json(
                &state,
                deal,
                &actions,
                entry.probabilities.clone(),
                weight,
                config,
            ));
        }
    }
    for action in &actions {
        collect_distillation_records(
            state.apply(action, config),
            deal,
            config,
            entries,
            records,
            covered_information_sets,
        );
    }
}

fn exact_combo_count(hand_class: &str) -> usize {
    if hand_class.len() == 2 {
        6
    } else if hand_class.ends_with('s') {
        4
    } else {
        12
    }
}

fn in_unit_interval(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn root_worlds(cache: &ContinuationCache) -> Vec<WeightedWorld> {
    let weight = 1.0 / cache.deals.len() as f64;
    cache
        .deals
        .iter()
        .enumerate()
        .map(|(deal_index, _)| WeightedWorld {
            state: GameState::initial(&cache.game),
            deal_index,
            weight,
        })
        .collect()
}

fn collect_local_leak_entries<P: Policy>(
    cache: &ContinuationCache,
    policy: &P,
    worlds: Vec<WeightedWorld>,
    players: &mut [Vec<PreflopLeakEntry>; 2],
    counters: &mut LookupCounters,
) {
    if worlds.is_empty()
        || worlds[0].state.terminal.is_some()
        || worlds[0].state.street != Street::Preflop
    {
        return;
    }
    let actor = worlds[0].state.actor;
    debug_assert!(worlds.iter().all(|world| world.state.actor == actor));
    let mut information_sets = BTreeMap::<String, Vec<WeightedWorld>>::new();
    for world in &worlds {
        let deal = cache.deal(world.deal_index);
        information_sets
            .entry(information_key(&world.state, &deal))
            .or_default()
            .push(world.clone());
    }
    for (key, group) in information_sets {
        let actions = group[0].state.legal_actions(&cache.game);
        let deal = cache.deal(group[0].deal_index);
        let probabilities = policy.strategy(&group[0].state, &deal, &actions, counters);
        let reach_probability = group.iter().map(|world| world.weight).sum::<f64>();
        if reach_probability <= 0.0 {
            continue;
        }
        let action_weighted_values = actions
            .iter()
            .map(|action| {
                policy_value_worlds(
                    cache,
                    policy,
                    group
                        .iter()
                        .map(|world| WeightedWorld {
                            state: world.state.apply(action, &cache.game),
                            deal_index: world.deal_index,
                            weight: world.weight,
                        })
                        .collect(),
                    actor,
                    counters,
                )
            })
            .collect::<Vec<_>>();
        let policy_weighted_value = probabilities
            .iter()
            .zip(&action_weighted_values)
            .map(|(probability, value)| probability * value)
            .sum::<f64>();
        let (best_index, best_weighted_value) = action_weighted_values
            .iter()
            .copied()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .expect("preflop information set has a legal action");
        let reach_weighted_gain = (best_weighted_value - policy_weighted_value).max(0.0);
        players[actor].push(PreflopLeakEntry {
            key,
            player: actor,
            hand_class: hand_class(&deal, actor),
            public_history: group[0].state.public_history.clone(),
            reach_probability,
            action_labels: actions.iter().map(|action| action.label.clone()).collect(),
            policy_probabilities: probabilities,
            action_values_bb: action_weighted_values
                .iter()
                .map(|value| value / reach_probability)
                .collect(),
            policy_value_bb: policy_weighted_value / reach_probability,
            best_action: actions[best_index].label.clone(),
            best_action_value_bb: best_weighted_value / reach_probability,
            conditional_ev_gain_bb: reach_weighted_gain / reach_probability,
            reach_weighted_ev_gain_bb_per_hand: reach_weighted_gain,
        });
    }

    let actions = worlds[0].state.legal_actions(&cache.game);
    for (action_index, action) in actions.iter().enumerate() {
        let branch = worlds
            .iter()
            .filter_map(|world| {
                let deal = cache.deal(world.deal_index);
                let strategy = policy.strategy(&world.state, &deal, &actions, counters);
                let probability = strategy[action_index];
                (probability > 0.0).then(|| WeightedWorld {
                    state: world.state.apply(action, &cache.game),
                    deal_index: world.deal_index,
                    weight: world.weight * probability,
                })
            })
            .collect();
        collect_local_leak_entries(cache, policy, branch, players, counters);
    }
}

fn best_response_worlds<P: Policy>(
    cache: &ContinuationCache,
    policy: &P,
    worlds: Vec<WeightedWorld>,
    responder: usize,
    stats: &mut ResponseStats,
    counters: &mut LookupCounters,
) -> f64 {
    if worlds.is_empty() {
        return 0.0;
    }
    if worlds[0].state.terminal.is_some() || worlds[0].state.street != Street::Preflop {
        return worlds
            .iter()
            .map(|world| world.weight * world_utility(cache, world, responder))
            .sum();
    }
    let actor = worlds[0].state.actor;
    debug_assert!(worlds.iter().all(|world| world.state.actor == actor));
    if actor == responder {
        let mut information_sets = BTreeMap::<String, Vec<WeightedWorld>>::new();
        for world in worlds {
            let deal = cache.deal(world.deal_index);
            information_sets
                .entry(information_key(&world.state, &deal))
                .or_default()
                .push(world);
        }
        stats.information_sets += information_sets.len();
        information_sets
            .into_values()
            .map(|group| {
                let actions = group[0].state.legal_actions(&cache.game);
                actions
                    .iter()
                    .map(|action| {
                        best_response_worlds(
                            cache,
                            policy,
                            group
                                .iter()
                                .map(|world| WeightedWorld {
                                    state: world.state.apply(action, &cache.game),
                                    deal_index: world.deal_index,
                                    weight: world.weight,
                                })
                                .collect(),
                            responder,
                            stats,
                            counters,
                        )
                    })
                    .max_by(f64::total_cmp)
                    .expect("preflop responder has a legal action")
            })
            .sum()
    } else {
        let actions = worlds[0].state.legal_actions(&cache.game);
        actions
            .iter()
            .enumerate()
            .map(|(action_index, action)| {
                let branch = worlds
                    .iter()
                    .filter_map(|world| {
                        let deal = cache.deal(world.deal_index);
                        let strategy = policy.strategy(&world.state, &deal, &actions, counters);
                        let probability = strategy[action_index];
                        (probability > 0.0).then(|| WeightedWorld {
                            state: world.state.apply(action, &cache.game),
                            deal_index: world.deal_index,
                            weight: world.weight * probability,
                        })
                    })
                    .collect();
                best_response_worlds(cache, policy, branch, responder, stats, counters)
            })
            .sum()
    }
}

fn policy_value_worlds<P: Policy>(
    cache: &ContinuationCache,
    policy: &P,
    worlds: Vec<WeightedWorld>,
    player: usize,
    counters: &mut LookupCounters,
) -> f64 {
    if worlds.is_empty() {
        return 0.0;
    }
    if worlds[0].state.terminal.is_some() || worlds[0].state.street != Street::Preflop {
        return worlds
            .iter()
            .map(|world| world.weight * world_utility(cache, world, player))
            .sum();
    }
    let actions = worlds[0].state.legal_actions(&cache.game);
    actions
        .iter()
        .enumerate()
        .map(|(action_index, action)| {
            let branch = worlds
                .iter()
                .filter_map(|world| {
                    let deal = cache.deal(world.deal_index);
                    let strategy = policy.strategy(&world.state, &deal, &actions, counters);
                    let probability = strategy[action_index];
                    (probability > 0.0).then(|| WeightedWorld {
                        state: world.state.apply(action, &cache.game),
                        deal_index: world.deal_index,
                        weight: world.weight * probability,
                    })
                })
                .collect();
            policy_value_worlds(cache, policy, branch, player, counters)
        })
        .sum()
}

fn world_utility(cache: &ContinuationCache, world: &WeightedWorld, player: usize) -> f64 {
    let deal = cache.deal(world.deal_index);
    if world.state.terminal.is_some() {
        let utility_p0 = realized_utility_p0(&world.state, &deal);
        if player == 0 {
            utility_p0
        } else {
            -utility_p0
        }
    } else {
        cache.continuation_utility(world.deal_index, &world.state, player)
    }
}

fn enumerate_flop_leaves(config: &BlueprintConfig) -> BTreeMap<u64, GameState> {
    fn visit(state: GameState, config: &BlueprintConfig, leaves: &mut BTreeMap<u64, GameState>) {
        if state.terminal.is_some() {
            return;
        }
        if state.street != Street::Preflop {
            let key = history_key(&state);
            match leaves.get(&key) {
                Some(existing) => assert_eq!(existing.public_history, state.public_history),
                None => {
                    leaves.insert(key, state);
                }
            }
            return;
        }
        for action in state.legal_actions(config) {
            visit(state.apply(&action, config), config, leaves);
        }
    }
    let mut leaves = BTreeMap::new();
    visit(GameState::initial(config), config, &mut leaves);
    leaves
}

fn rollout_policy_value(
    policy: &FrozenPolicy,
    state: GameState,
    deal: &Deal,
    config: &BlueprintConfig,
    rng: &mut SplitMix64,
) -> f64 {
    if state.terminal.is_some() {
        return realized_utility_p0(&state, deal);
    }
    let actions = state.legal_actions(config);
    let strategy = policy.strategy(&state, deal, &actions, config);
    let selected = sample_index(&strategy, rng);
    rollout_policy_value(
        policy,
        state.apply(&actions[selected], config),
        deal,
        config,
        rng,
    )
}

fn realized_utility_p0(state: &GameState, deal: &Deal) -> f64 {
    match state.terminal.as_ref().expect("realized terminal utility") {
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

fn information_key(state: &GameState, deal: &Deal) -> String {
    format!(
        "p{}|{}|{}",
        state.actor,
        hand_class(deal, state.actor),
        state.public_history.join("/")
    )
}

fn hand_class(deal: &Deal, player: usize) -> String {
    Combo::new(deal.holes[player][0], deal.holes[player][1]).label()
}

fn history_key(state: &GameState) -> u64 {
    stable_hash(state.public_history.join("/").as_bytes())
}

fn continuation_seed(seed: u64, deal: usize, history: u64, rollout: u32) -> u64 {
    let mut digest = Sha256::new();
    digest.update(b"hu-v26-preflop-continuation-v1");
    digest.update(seed.to_le_bytes());
    digest.update((deal as u64).to_le_bytes());
    digest.update(history.to_le_bytes());
    digest.update(rollout.to_le_bytes());
    let bytes = digest.finalize();
    u64::from_le_bytes(bytes[..8].try_into().expect("SHA-256 prefix"))
}

fn stratified_deal(rng: &mut SplitMix64, index: usize, combos: &[Combo]) -> Deal {
    let block = index / combos.len();
    let fixed_player = block % 2;
    let fixed = combos[index % combos.len()];
    let compatible = combos
        .iter()
        .copied()
        .filter(|candidate| !fixed.overlaps(*candidate))
        .collect::<Vec<_>>();
    let opponent = compatible[rng.index(compatible.len())];
    let mut holes = [[0u8; 2]; 2];
    holes[fixed_player] = fixed.cards();
    holes[1 - fixed_player] = opponent.cards();
    let mut available = (0..52u8)
        .filter(|card| !holes.iter().flatten().any(|known| known == card))
        .collect::<Vec<_>>();
    for board_index in 0..5 {
        let swap = board_index + rng.index(available.len() - board_index);
        available.swap(board_index, swap);
    }
    Deal::from_sampled_cards(
        holes,
        [
            available[0],
            available[1],
            available[2],
            available[3],
            available[4],
        ],
    )
}

fn sha256_file(path: &Path) -> Result<String, Box<dyn Error>> {
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn combined_file_sha256(paths: &[PathBuf]) -> Result<String, Box<dyn Error>> {
    let mut digest = Sha256::new();
    digest.update(b"hu-frozen-policy-mixture-v1");
    for path in paths {
        digest.update(sha256_file(path)?.as_bytes());
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_cache(seed: u64) -> ContinuationCache {
        let game = BlueprintConfig {
            effective_stack_bb: 20.0,
            ..BlueprintConfig::default()
        };
        let leaves = enumerate_flop_leaves(&game);
        let raw_deals = [
            ([[51, 50], [47, 46]], [0, 5, 10, 15, 20]),
            ([[49, 48], [45, 44]], [1, 6, 11, 16, 21]),
            ([[47, 43], [39, 35]], [2, 7, 12, 17, 22]),
            ([[46, 42], [38, 34]], [3, 8, 13, 18, 23]),
        ];
        let deals = raw_deals
            .into_iter()
            .enumerate()
            .map(|(deal_index, (holes, board))| CachedDeal {
                holes,
                board,
                continuations: leaves
                    .keys()
                    .map(|history| {
                        (
                            *history,
                            ContinuationEstimate {
                                mean_utility_p0_bb: if deal_index < 2 { 0.25 } else { -0.25 },
                                conditional_utilities_bb: None,
                                action_standard_error_bb: 0.0,
                            },
                        )
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let leaf_values = deals.len() * leaves.len();
        ContinuationCache {
            schema: CONTINUATION_SCHEMA.to_owned(),
            depth_bb: 20.0,
            seed,
            rollouts_per_leaf: 2,
            chance_sampling: "synthetic_test".to_owned(),
            complete_exact_combo_cycles: 0,
            balanced_exact_combo_marginals: false,
            network_sha256: "0".repeat(64),
            network_sha256s: vec!["0".repeat(64)],
            policy_mixture: "synthetic_test".to_owned(),
            source_cache_sha256: None,
            source_deal_indices: None,
            game,
            public_histories: leaves
                .into_iter()
                .map(|(key, state)| (key, state.public_history))
                .collect(),
            deals,
            validation: ContinuationValidation {
                complete: true,
                probability_values_finite: true,
                utilities_within_stack: true,
                leaf_values,
                fraction_action_se_at_most_0_02bb: 1.0,
                maximum_action_standard_error_bb: 0.0,
                fraction_history_mean_se_at_most_0_02bb: 0.0,
                maximum_history_mean_standard_error_bb: 0.25,
                fraction_information_group_mean_se_at_most_0_25bb: 0.0,
                maximum_information_group_mean_standard_error_bb: 0.25,
            },
        }
    }

    #[test]
    fn conditional_continuation_utilities_override_legacy_scalar_fallback() {
        let mut cache = synthetic_cache(7);
        let state = enumerate_flop_leaves(&cache.game)
            .into_values()
            .next()
            .unwrap();
        let history = history_key(&state);
        assert_eq!(cache.continuation_utility(0, &state, 0), 0.25);
        assert_eq!(cache.continuation_utility(0, &state, 1), -0.25);
        cache.deals[0]
            .continuations
            .get_mut(&history)
            .unwrap()
            .conditional_utilities_bb = Some([0.75, -0.10]);
        assert_eq!(cache.continuation_utility(0, &state, 0), 0.75);
        assert_eq!(cache.continuation_utility(0, &state, 1), -0.10);
    }

    #[test]
    fn resolver_ranges_follow_only_own_class_and_public_action() {
        let game = BlueprintConfig {
            effective_stack_bb: 20.0,
            ..BlueprintConfig::default()
        };
        let state = GameState::initial(&game);
        let actions = state.legal_actions(&game);
        let labels = actions
            .iter()
            .map(|action| action.label.clone())
            .collect::<Vec<_>>();
        let combo_classes = all_combos()
            .iter()
            .map(|combo| combo.label())
            .collect::<Vec<_>>();
        let classes = combo_classes.iter().cloned().collect::<BTreeSet<_>>();
        let mut policy = ResolverRangePolicy::new();
        for class in classes {
            let mut probabilities = vec![1.0 / actions.len() as f64; actions.len()];
            if class == "AA" {
                probabilities.fill(0.1 / (actions.len() - 1) as f64);
                probabilities[0] = 0.9;
            }
            policy.insert(
                (state.actor, class, state.public_history.clone()),
                (labels.clone(), probabilities),
            );
        }
        let mut range = vec![1.0; super::public_belief::COMBO_COUNT];
        condition_range_for_action(&state, &labels, 0, &policy, &combo_classes, &mut range)
            .unwrap();
        let first_aces = Combo::new(51, 50).key();
        let second_aces = Combo::new(49, 48).key();
        let kings = Combo::new(47, 46).key();
        assert_eq!(range[first_aces], 0.9);
        assert_eq!(range[first_aces], range[second_aces]);
        assert_eq!(range[kings], 1.0 / actions.len() as f64);
    }

    #[test]
    fn preflop_information_key_hides_opponent_cards_and_future_board() {
        let game = BlueprintConfig::default();
        let state = GameState::initial(&game);
        let first = Deal::from_cards([[51, 47], [50, 46]], [0, 1, 2, 3, 4]);
        let second = Deal::from_cards([[50, 46], [45, 41]], [5, 6, 7, 8, 9]);
        assert_eq!(hand_class(&first, 0), "AKs");
        assert_eq!(
            information_key(&state, &first),
            information_key(&state, &second)
        );
    }

    #[test]
    fn flop_leaf_enumeration_is_public_and_deterministic() {
        let game = BlueprintConfig {
            effective_stack_bb: 20.0,
            ..BlueprintConfig::default()
        };
        let first = enumerate_flop_leaves(&game);
        let second = enumerate_flop_leaves(&game);
        assert!(!first.is_empty());
        assert_eq!(
            first.keys().collect::<Vec<_>>(),
            second.keys().collect::<Vec<_>>()
        );
        assert!(first.values().all(|state| state.street == Street::Flop));
    }

    #[test]
    fn complete_stratified_cycle_covers_every_exact_combo_for_both_seats() {
        let mut rng = SplitMix64::new(17);
        let combos = all_combos();
        let count = combos.len();
        let deals = (0..2 * count)
            .map(|index| stratified_deal(&mut rng, index, &combos))
            .collect::<Vec<_>>();
        for player in 0..2 {
            let observed = deals
                .iter()
                .map(|deal| Combo::new(deal.holes[player][0], deal.holes[player][1]).key())
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(observed.len(), count);
        }
        assert_eq!(exact_combo_count("AA"), 6);
        assert_eq!(exact_combo_count("AKs"), 4);
        assert_eq!(exact_combo_count("AKo"), 12);
    }

    #[test]
    fn lazy_dcfr_discount_matches_every_iteration_discounting() {
        let parameters = DcfrParameters::default();
        let mut lazy_discounts = DiscountAccumulator::new(parameters.clone());
        let mut eager_discounts = DiscountAccumulator::new(parameters);
        let mut lazy = SolverNode {
            actor: 0,
            hand_class: "AKs".to_owned(),
            public_history: Vec::new(),
            action_labels: vec!["fold".to_owned(), "call".to_owned()],
            regrets: vec![3.0, -2.0],
            strategy_sum: vec![4.0, 1.0],
            regret_updates: 0,
            average_visits: 0,
            last_discount_iteration: 0,
            last_discount_cumulative_logs: [0.0; 3],
        };
        let mut eager = lazy.clone();
        for iteration in 1..=12 {
            lazy_discounts.advance(iteration);
            eager_discounts.advance(iteration);
            eager.apply_dcfr_discount(iteration, &eager_discounts);
        }
        lazy.apply_dcfr_discount(12, &lazy_discounts);
        for (actual, expected) in lazy
            .regrets
            .iter()
            .chain(&lazy.strategy_sum)
            .zip(eager.regrets.iter().chain(&eager.strategy_sum))
        {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn distillation_expands_teacher_information_sets_across_exact_private_combos() {
        let cache = synthetic_cache(7);
        let mut solver = PreflopDcfrSolver::new(&cache, 11);
        solver.train(500);
        let artifact = solver.artifact("test-v27".to_owned(), 11, "a".repeat(64));
        let entries = artifact
            .strategies
            .iter()
            .map(|entry| (entry.key.clone(), entry))
            .collect::<BTreeMap<_, _>>();
        let mut records = BTreeMap::new();
        let mut covered = BTreeSet::new();
        collect_distillation_records(
            GameState::initial(&cache.game),
            &cache.deal(0),
            &cache.game,
            &entries,
            &mut records,
            &mut covered,
        );
        let first_record_count = records.len();
        collect_distillation_records(
            GameState::initial(&cache.game),
            &cache.deal(1),
            &cache.game,
            &entries,
            &mut records,
            &mut covered,
        );
        assert!(records.len() > first_record_count);
    }

    #[test]
    fn short_tabular_solve_is_deterministic_and_normalized() {
        let cache = synthetic_cache(7);
        cache.validate().unwrap();
        let mut first = PreflopDcfrSolver::new(&cache, 11);
        first.train(200);
        let first = first.artifact("test-v27".to_owned(), 11, "a".repeat(64));
        let mut second = PreflopDcfrSolver::new(&cache, 11);
        second.train(200);
        let second = second.artifact("test-v27".to_owned(), 11, "a".repeat(64));
        assert_eq!(
            serde_json::to_vec(&first).unwrap(),
            serde_json::to_vec(&second).unwrap()
        );
        first.validate().unwrap();
        assert!(first.training_evaluation.policy_lookup_coverage > 0.5);
        assert!(first
            .strategies
            .iter()
            .all(|entry| (entry.probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-9));
    }

    #[test]
    fn custom_solver_parameters_are_recorded_and_validated() {
        let cache = synthetic_cache(7);
        let dcfr = DcfrParameters {
            positive_regret_exponent: 2.0,
            negative_regret_exponent: 0.5,
            strategy_exponent: 3.0,
        };
        let mut solver = PreflopDcfrSolver::with_parameters(&cache, 11, dcfr.clone(), 0.02);
        solver.train(200);
        let artifact = solver.artifact("test-v28".to_owned(), 11, "a".repeat(64));
        artifact.validate().unwrap();
        assert_eq!(artifact.solver_dcfr, dcfr);
        assert_eq!(artifact.sampling_exploration_probability, 0.02);
    }

    #[test]
    fn mccfr_plus_clamps_regrets_and_records_variant() {
        let cache = synthetic_cache(7);
        let mut solver = PreflopDcfrSolver::with_variant(
            &cache,
            11,
            DcfrParameters::default(),
            0.05,
            PreflopSolverVariant::MccfrPlus,
        );
        solver.train(200);
        assert!(solver
            .nodes
            .values()
            .flat_map(|node| &node.regrets)
            .all(|regret| *regret >= 0.0));
        let artifact = solver.artifact("test-v28-plus".to_owned(), 11, "a".repeat(64));
        assert_eq!(artifact.solver_variant, PreflopSolverVariant::MccfrPlus);
        artifact.validate().unwrap();
    }

    #[test]
    fn compact_policy_quantization_preserves_probability_sums() {
        let cache = synthetic_cache(7);
        let mut solver = PreflopDcfrSolver::new(&cache, 11);
        solver.train(200);
        let artifact = solver.artifact("test-v28".to_owned(), 11, "a".repeat(64));
        let output = std::env::temp_dir().join(format!(
            "pokersolver-compact-preflop-{}.bin",
            std::process::id()
        ));
        let summary = export_compact_preflop_policy(&artifact, &output).unwrap();
        let bytes = fs::read(&output).unwrap();
        fs::remove_file(&output).unwrap();
        assert_eq!(&bytes[..8], b"HUPFTAB1");
        assert_eq!(summary.information_sets, artifact.strategies.len());
        assert!(summary.quantized_probability_sums_valid);
        assert!(summary.maximum_probability_quantization_error <= 1.0 / u16::MAX as f64);
    }

    #[test]
    fn leak_attribution_is_reach_weighted_sorted_and_bounded() {
        let cache = synthetic_cache(7);
        let mut solver = PreflopDcfrSolver::new(&cache, 11);
        solver.train(200);
        let artifact = solver.artifact("test-v27".to_owned(), 11, "a".repeat(64));
        let attribution = attribute_policy_leaks(&cache, &artifact, 3).unwrap();
        assert_eq!(attribution.schema, ATTRIBUTION_SCHEMA);
        assert_eq!(attribution.players[0].len(), 3);
        assert_eq!(attribution.players[1].len(), 3);
        assert!(attribution.policy_lookup_coverage > 0.5);
        assert!(attribution
            .public_histories
            .iter()
            .all(|histories| !histories.is_empty()
                && histories.windows(2).all(|pair| {
                    pair[0].reach_weighted_ev_gain_bb_per_hand
                        >= pair[1].reach_weighted_ev_gain_bb_per_hand
                })));
        for entries in &attribution.players {
            assert!(entries.windows(2).all(|pair| {
                pair[0].reach_weighted_ev_gain_bb_per_hand
                    >= pair[1].reach_weighted_ev_gain_bb_per_hand
            }));
            assert!(entries.iter().all(|entry| {
                entry.reach_probability > 0.0
                    && entry.conditional_ev_gain_bb >= 0.0
                    && entry.action_labels.len() == entry.action_values_bb.len()
                    && entry.action_labels.len() == entry.policy_probabilities.len()
            }));
        }
    }

    #[test]
    fn identical_continuation_caches_have_zero_group_delta() {
        let cache = synthetic_cache(7);
        let comparison = compare_continuation_caches(&cache, &cache).unwrap();
        assert!(comparison
            .players
            .iter()
            .all(|player| player.group_coverage == 1.0
                && player.sample_weighted_mean_absolute_delta_bb == 0.0));
        let merged = merge_continuation_caches(&cache, &cache).unwrap();
        assert_eq!(merged.deals.len(), 2 * cache.deals.len());
        assert_eq!(
            merged.validation.leaf_values,
            2 * cache.validation.leaf_values
        );
        let refreshed = refresh_continuation_cache_validation(cache).unwrap();
        assert!(refreshed
            .validation
            .maximum_information_group_mean_standard_error_bb
            .is_finite());
    }

    #[test]
    fn source_deal_cycles_require_every_unique_position() {
        let cycle_size = 2 * all_combos().len();
        let complete = (0..cycle_size).collect::<Vec<_>>();
        assert_eq!(complete_source_deal_cycles(&complete), 1);
        assert!(balanced_source_deal_indices(&complete));

        let mut split_across_cycles = (0..cycle_size / 2).collect::<Vec<_>>();
        split_across_cycles.extend(cycle_size..cycle_size + cycle_size / 2);
        assert_eq!(complete_source_deal_cycles(&split_across_cycles), 0);
        assert!(!balanced_source_deal_indices(&split_across_cycles));
    }

    #[test]
    fn provenance_chunks_merge_in_source_order_and_reject_overlap() {
        let mut first = synthetic_cache(7);
        first.source_cache_sha256 = Some("1".repeat(64));
        first.source_deal_indices = Some(vec![0, 1, 2, 3]);
        first.validate().unwrap();
        let mut second = synthetic_cache(7);
        second.source_cache_sha256 = first.source_cache_sha256.clone();
        second.source_deal_indices = Some(vec![4, 5, 6, 7]);
        second.validate().unwrap();

        let merged = merge_continuation_caches(&second, &first).unwrap();
        assert_eq!(merged.source_deal_indices, Some((0..8).collect()));
        assert_eq!(merged.deals.len(), 8);

        second.source_deal_indices = Some(vec![3, 4, 5, 6]);
        assert!(merge_continuation_caches(&first, &second).is_err());
    }
}
