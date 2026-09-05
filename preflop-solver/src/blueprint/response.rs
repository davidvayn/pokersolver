//! Information-set local best response and range-conditioned resolver.
//!
//! This is deliberately a lower-bound/red-team evaluator.  It aggregates
//! action rollouts by the responder's observable abstract information set,
//! freezes the resulting full-game response, and evaluates it on independent
//! deals.  Sampling uncertainty is reported explicitly: a profitable point
//! estimate is evidence of a leak, while a zero result cannot certify a Nash
//! equilibrium.

use super::neural::FrozenPolicy;
use super::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

const RESPONSE_SCHEMA: &str = "hu-full-game-information-set-lbr-v5";
const RESOLVER_SCHEMA: &str = "hu-range-conditioned-postflop-resolver-v5";

fn default_response_workers() -> usize {
    1
}
fn serial_response_workers(workers: &usize) -> bool {
    *workers == 1
}

mod backoff;
mod flop;
pub use backoff::FlopBackoffOptions;
mod flop_allin;
pub use flop_allin::TerminalFlopOptions;
mod parallel;
mod recheck;
pub use recheck::{recheck_full_game_response, ResponseRecheckConfig};
pub use flop::{evaluate_flop_patch, FlopPatchEvaluationConfig};
mod table;
mod terminal;
mod turn;
#[cfg(test)]
use table::AverageNode;
use table::InferenceTable;
pub use turn::TurnResolveOptions;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolverGranularity {
    ExactTrajectory,
    ObservableBackoff,
    CoarseObservableBackoff,
    StrategicObservableBackoff,
}

fn granularity_rank(granularity: ResolverGranularity) -> u8 {
    match granularity {
        ResolverGranularity::ExactTrajectory => 0,
        ResolverGranularity::ObservableBackoff => 1,
        ResolverGranularity::CoarseObservableBackoff => 2,
        ResolverGranularity::StrategicObservableBackoff => 3,
    }
}

#[derive(Clone, Debug)]
pub struct ResponseEvaluationConfig {
    pub game: BlueprintConfig,
    pub training_deals: u64,
    pub calibration_deals: u64,
    pub evaluation_deals: u64,
    pub rollouts_per_action: u32,
    pub minimum_range_particles: u64,
    pub maximum_granularity: ResolverGranularity,
    pub seed: u64,
    pub source: ResponsePolicySource,
    pub turn_resolver: Option<TurnResolveOptions>,
    pub terminal_flop: Option<TerminalFlopOptions>,
    pub flop_backoff: Option<FlopBackoffOptions>,
    pub exact_terminal_training_values: bool,
    pub postflop_only_response: bool,
    pub response_workers: usize,
}

#[derive(Clone, Debug)]
pub enum ResponsePolicySource {
    Neural(PathBuf),
    TabularCheckpoint(PathBuf),
}

trait ResponsePolicy {
    fn strategy(
        &self,
        state: &GameState,
        deal: &Deal,
        actions: &[LegalAction],
        game: &BlueprintConfig,
    ) -> Vec<f64>;
    fn take_coverage(&self) -> Vec<StreetPolicyCoverage> {
        Vec::new()
    }
    fn take_resolution_diagnostics(&self) -> Option<serde_json::Value> {
        None
    }
    fn parallel_copy(&self) -> Option<Box<dyn ResponsePolicy + Send>> {
        None
    }
    fn take_raw_coverage(&self) -> [CoverageCounter; 4] {
        std::array::from_fn(|_| CoverageCounter::default())
    }
    fn take_completion_coverage(&self) -> backoff::CompletionCoverage {
        backoff::CompletionCoverage::default()
    }
    fn absorb_worker(&self, _worker: &dyn ResponsePolicy) {}
}

impl ResponsePolicy for FrozenPolicy {
    fn strategy(
        &self,
        state: &GameState,
        deal: &Deal,
        actions: &[LegalAction],
        game: &BlueprintConfig,
    ) -> Vec<f64> {
        FrozenPolicy::strategy(self, state, deal, actions, game)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StreetPolicyCoverage {
    pub street: Street,
    pub coverage: PolicyCoverage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    completion: Option<backoff::CompletionCoverage>,
}

struct TabularResponsePolicy {
    table: Arc<InferenceTable>,
    coverage: RefCell<[CoverageCounter; 4]>,
    flop_patch: Option<Arc<flop::FlopPatch>>,
    flop_backoff: Option<Arc<backoff::FlopBackoff>>,
    completion_coverage: RefCell<backoff::CompletionCoverage>,
}

impl TabularResponsePolicy {
    fn isolated_copy(&self) -> Self {
        Self {
            table: Arc::clone(&self.table),
            coverage: RefCell::default(),
            flop_patch: self.flop_patch.clone(),
            flop_backoff: self.flop_backoff.clone(),
            completion_coverage: RefCell::default(),
        }
    }
    fn frozen_strategy(
        &self,
        state: &GameState,
        deal: &Deal,
        actions: &[LegalAction],
        game: &BlueprintConfig,
    ) -> Vec<f64> {
        let (key, descriptor, _) = information_set(state, deal, game);
        let node = self.table.nodes.get(&key);
        let mut strategy = match node {
            Some(node) => {
                assert_eq!(
                    node.descriptor, descriptor,
                    "checkpoint information-set collision"
                );
                assert!(
                    node.action_labels
                        .iter()
                        .map(|s| s.as_ref())
                        .eq(actions.iter().map(|a| a.label.as_str())),
                    "checkpoint action grid mismatch"
                );
                node.average_strategy()
            }
            None => vec![1.0 / actions.len() as f64; actions.len()],
        };
        if node.is_none_or(|n| n.average_visits == 0) {
            if let Some(backoff) = &self.flop_backoff {
                backoff.complete(&descriptor, actions, &mut strategy);
            }
        }
        self.apply_flop_patch(state, deal, actions, game, strategy)
    }

    fn apply_flop_patch(
        &self,
        state: &GameState,
        deal: &Deal,
        actions: &[LegalAction],
        game: &BlueprintConfig,
        strategy: Vec<f64>,
    ) -> Vec<f64> {
        match &self.flop_patch {
            Some(patch) => patch.strategy(self, state, deal, actions, game, strategy),
            None => strategy,
        }
    }
}

impl ResponsePolicy for TabularResponsePolicy {
    fn parallel_copy(&self) -> Option<Box<dyn ResponsePolicy + Send>> {
        Some(Box::new(self.isolated_copy()))
    }

    fn take_raw_coverage(&self) -> [CoverageCounter; 4] {
        std::mem::take(&mut *self.coverage.borrow_mut())
    }

    fn take_completion_coverage(&self) -> backoff::CompletionCoverage {
        std::mem::take(&mut *self.completion_coverage.borrow_mut())
    }

    fn absorb_worker(&self, worker: &dyn ResponsePolicy) {
        self.completion_coverage
            .borrow_mut()
            .add(worker.take_completion_coverage());
        for (counter, incoming) in self
            .coverage
            .borrow_mut()
            .iter_mut()
            .zip(worker.take_raw_coverage())
        {
            counter.add(&incoming);
        }
    }
    fn strategy(
        &self,
        state: &GameState,
        deal: &Deal,
        actions: &[LegalAction],
        game: &BlueprintConfig,
    ) -> Vec<f64> {
        let (key, descriptor, _) = information_set(state, deal, game);
        let street = match state.street {
            Street::Preflop => 0,
            Street::Flop => 1,
            Street::Turn => 2,
            Street::River => 3,
        };
        let mut counters = self.coverage.borrow_mut();
        let counter = &mut counters[street];
        counter.decisions += 1;
        let node = self.table.nodes.get(&key);
        let mut strategy = match node {
            Some(node) => {
                assert_eq!(
                    node.descriptor, descriptor,
                    "checkpoint information-set collision"
                );
                assert!(
                    node.action_labels
                        .iter()
                        .map(|s| s.as_ref())
                        .eq(actions.iter().map(|a| a.label.as_str())),
                    "checkpoint action grid mismatch"
                );
                if node.average_visits == 0 {
                    counter.untrained += 1;
                }
                node.average_strategy()
            }
            None => {
                // The same explicit profile completion as the trainer's
                // held-out evaluator. Never serve or hide these missing rows.
                counter.unknown += 1;
                vec![1.0 / actions.len() as f64; actions.len()]
            }
        };
        if state.street == Street::Flop && node.is_none_or(|n| n.average_visits == 0) {
            if let Some(backoff) = &self.flop_backoff {
                let mut counts = self.completion_coverage.borrow_mut();
                counts.eligible_missing_or_untrained_queries += 1;
                counts.matched_queries +=
                    u64::from(backoff.complete(&descriptor, actions, &mut strategy));
            }
        }
        self.apply_flop_patch(state, deal, actions, game, strategy)
    }

    fn take_coverage(&self) -> Vec<StreetPolicyCoverage> {
        let counters = self.take_raw_coverage();
        let completion = self.take_completion_coverage();
        [Street::Preflop, Street::Flop, Street::Turn, Street::River]
            .into_iter()
            .zip(counters)
            .map(|(street, counter)| StreetPolicyCoverage {
                street,
                coverage: counter.report(),
                completion: (street == Street::Flop && self.flop_backoff.is_some())
                    .then(|| completion.clone()),
            })
            .collect()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ResolverDecision {
    pub information_set: u64,
    pub granularity: ResolverGranularity,
    pub actor: usize,
    pub street: Street,
    pub hand_bucket_trajectory: Vec<String>,
    pub public_bucket_trajectory: Vec<String>,
    pub public_history: Vec<String>,
    pub action_labels: Vec<String>,
    pub action_values_bb: Vec<f64>,
    pub action_standard_errors_bb: Vec<f64>,
    pub selected_action: usize,
    pub selected_action_mean_gap_bb: f64,
    pub approximate_selected_action_gap_lower_bound_99_5_percent_bb: f64,
    /// Paired improvement over the evaluated profile, rather than a margin
    /// over the runner-up action. Absent in legacy retained reports.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_advantage: Option<ResponseAdvantage>,
    /// Legacy confidence in a unique best action, not in beating the baseline.
    pub low_confidence: bool,
    pub range_particles: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ResponseAdvantage {
    pub baseline_mean_ev_bb: f64,
    pub selected_mean_gain_bb: f64,
    pub selected_gain_standard_error_bb: f64,
    pub approximate_gain_lower_bound_99_5_percent_bb: f64,
}

impl ResolverDecision {
    fn is_profitable_response(&self) -> bool {
        self.response_advantage
            .as_ref()
            .map_or(!self.low_confidence, |advantage| {
                advantage
                    .approximate_gain_lower_bound_99_5_percent_bb
                    .is_finite()
                    && advantage.approximate_gain_lower_bound_99_5_percent_bb > 0.0
            })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RangeConditionedResolver {
    pub schema: String,
    pub responder: usize,
    pub training_deals: u64,
    pub rollouts_per_action: u32,
    pub minimum_range_particles: u64,
    pub decisions: Vec<ResolverDecision>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ResponsePlayerEvaluation {
    pub responder: usize,
    pub response_deployed: bool,
    pub baseline_utility_bb: f64,
    pub response_utility_bb: f64,
    pub estimated_gain_bb: f64,
    pub gain_standard_error_bb: f64,
    pub approximate_one_sided_99_5_percent_gain_lower_bound_bb: f64,
    pub resolver_lookup_coverage: f64,
    pub exact_lookup_coverage: f64,
    pub observable_backoff_lookup_coverage: f64,
    pub coarse_observable_backoff_lookup_coverage: f64,
    pub strategic_observable_backoff_lookup_coverage: f64,
    pub preflop_lookup_coverage: f64,
    pub postflop_lookup_coverage: f64,
    pub postflop_exact_lookup_coverage: f64,
    pub postflop_observable_backoff_lookup_coverage: f64,
    pub postflop_coarse_observable_backoff_lookup_coverage: f64,
    pub postflop_strategic_observable_backoff_lookup_coverage: f64,
    pub learned_information_sets: usize,
    pub confident_information_sets: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FullGameResponseEvaluation {
    pub schema: String,
    pub method: String,
    pub depth_bb: f64,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub network_sha256: String,
    #[serde(default)]
    pub policy_sha256: String,
    #[serde(default)]
    pub policy_source_kind: String,
    #[serde(default)]
    pub checkpoint_training_iterations: Option<u64>,
    #[serde(default)]
    pub source_policy_coverage: BTreeMap<String, Vec<StreetPolicyCoverage>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_resolver: Option<TurnResolveOptions>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal_flop: Option<TerminalFlopOptions>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flop_backoff: Option<FlopBackoffOptions>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub exact_terminal_training_values: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub postflop_only_response: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retained_training: Option<RetainedResponseTraining>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub resolution_diagnostics: BTreeMap<String, serde_json::Value>,
    /// Sum across seats, not the legacy seat-average/NashConv-over-two scale.
    #[serde(default)]
    pub total_response_gain_bb_per_hand: f64,
    #[serde(default)]
    pub total_response_gain_lower_confidence_bound_99_percent_bb_per_hand: f64,
    pub seed: u64,
    pub training_deals: u64,
    #[serde(
        default = "default_response_workers",
        skip_serializing_if = "serial_response_workers"
    )]
    pub response_workers: usize,
    pub calibration_deals: u64,
    pub evaluation_deals: u64,
    pub rollouts_per_action: u32,
    pub minimum_range_particles: u64,
    pub maximum_granularity: ResolverGranularity,
    pub players: [ResponsePlayerEvaluation; 2],
    pub calibration_players: [ResponsePlayerEvaluation; 2],
    pub response_deployed: [bool; 2],
    pub approximate_exploitability_lower_bound_bb_per_hand: f64,
    pub approximate_exploitability_lower_confidence_bound_99_percent_bb_per_hand: f64,
    pub interpretation: String,
    pub preflop_responses: [Vec<ResolverDecision>; 2],
    pub resolvers: [RangeConditionedResolver; 2],
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RetainedResponseTraining {
    pub report_sha256: String,
    pub seed: u64,
}

impl FullGameResponseEvaluation {
    fn uses_seed(&self, seed: u64) -> bool {
        self.seed == seed || self.retained_training.as_ref().is_some_and(|p| p.seed == seed)
    }
}

#[derive(Clone)]
struct DecisionAccumulator {
    actor: usize,
    street: Street,
    hand_bucket_trajectory: Vec<String>,
    public_bucket_trajectory: Vec<String>,
    public_history: Vec<String>,
    action_labels: Vec<String>,
    count: u64,
    sums: Vec<f64>,
    squared_sums: Vec<f64>,
    // Common-random action rollouts are paired. Track each upper-triangular
    // difference directly with Welford's update, avoiding cancellation and
    // the incorrect independence assumption for marginal EV errors.
    gap_means: Vec<f64>,
    gap_m2: Vec<f64>,
    advantage_count: u64,
    baseline_sum: f64,
    advantage_means: Vec<f64>,
    advantage_m2: Vec<f64>,
}

impl DecisionAccumulator {
    fn new(descriptor: &NodeDescriptor, history: Vec<String>, actions: &[LegalAction]) -> Self {
        Self {
            actor: match descriptor.actor {
                Position::ButtonSmallBlind => 0,
                Position::BigBlind => 1,
            },
            street: descriptor.street,
            hand_bucket_trajectory: descriptor
                .hand_bucket_trajectory
                .iter()
                .map(ToString::to_string)
                .collect(),
            public_bucket_trajectory: descriptor
                .public_bucket_trajectory
                .iter()
                .map(ToString::to_string)
                .collect(),
            public_history: history,
            action_labels: actions.iter().map(|action| action.label.clone()).collect(),
            count: 0,
            sums: vec![0.0; actions.len()],
            squared_sums: vec![0.0; actions.len()],
            gap_means: vec![0.0; actions.len() * actions.len()],
            gap_m2: vec![0.0; actions.len() * actions.len()],
            advantage_count: 0,
            baseline_sum: 0.0,
            advantage_means: vec![0.0; actions.len()],
            advantage_m2: vec![0.0; actions.len()],
        }
    }

    fn add_with_strategy(&mut self, values: &[f64], baseline_strategy: &[f64]) {
        assert_eq!(values.len(), baseline_strategy.len());
        assert!(baseline_strategy.iter().all(|p| p.is_finite() && *p >= 0.0));
        assert!((baseline_strategy.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        assert_eq!(
            self.advantage_count, self.count,
            "incomplete baseline observations"
        );
        self.add(values);
        self.advantage_count += 1;
        let baseline = values
            .iter()
            .zip(baseline_strategy)
            .map(|(v, p)| v * p)
            .sum::<f64>();
        self.baseline_sum += baseline;
        for (action, value) in values.iter().enumerate() {
            let difference = value - baseline;
            let delta = difference - self.advantage_means[action];
            self.advantage_means[action] += delta / self.advantage_count as f64;
            self.advantage_m2[action] += delta * (difference - self.advantage_means[action]);
        }
    }

    fn add(&mut self, values: &[f64]) {
        assert_eq!(values.len(), self.sums.len());
        self.count += 1;
        for first in 0..values.len() {
            for second in first + 1..values.len() {
                let index = first * values.len() + second;
                let difference = values[first] - values[second];
                let delta = difference - self.gap_means[index];
                self.gap_means[index] += delta / self.count as f64;
                self.gap_m2[index] += delta * (difference - self.gap_means[index]);
            }
        }
        for ((sum, squared), value) in self.sums.iter_mut().zip(&mut self.squared_sums).zip(values)
        {
            *sum += value;
            *squared += value * value;
        }
    }

    fn finish(
        self,
        key: u64,
        granularity: ResolverGranularity,
        unevaluated_standard_error_bb: f64,
    ) -> ResolverDecision {
        let count = self.count.max(1) as f64;
        let means = self.sums.iter().map(|sum| sum / count).collect::<Vec<_>>();
        let standard_errors = self
            .sums
            .iter()
            .zip(&self.squared_sums)
            .map(|(sum, squared)| {
                if self.count < 2 {
                    return unevaluated_standard_error_bb;
                }
                let variance = ((squared - sum * sum / count) / (count - 1.0)).max(0.0);
                (variance / count).sqrt()
            })
            .collect::<Vec<_>>();
        let selected_action = means
            .iter()
            .enumerate()
            .max_by(|(_, first), (_, second)| first.total_cmp(second))
            .map(|(index, _)| index)
            .unwrap_or(0);
        let runner_up = means
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != selected_action)
            .max_by(|(_, first), (_, second)| first.total_cmp(second))
            .map(|(index, value)| (index, *value));
        let (selected_action_mean_gap_bb, gap_lower_bound) = runner_up
            .map(|(runner_up_index, runner_up_value)| {
                let gap = means[selected_action] - runner_up_value;
                let first = selected_action.min(runner_up_index);
                let second = selected_action.max(runner_up_index);
                let gap_standard_error = if self.count < 2 {
                    unevaluated_standard_error_bb
                } else {
                    (self.gap_m2[first * means.len() + second].max(0.0) / ((count - 1.0) * count))
                        .sqrt()
                };
                (gap, gap - 2.575_829_303_548_900_4 * gap_standard_error)
            })
            .unwrap_or((0.0, 0.0));
        let response_advantage = (self.advantage_count > 0).then(|| {
            assert_eq!(self.advantage_count, self.count);
            let error = if self.count < 2 {
                unevaluated_standard_error_bb
            } else {
                (self.advantage_m2[selected_action].max(0.0) / ((count - 1.0) * count)).sqrt()
            };
            ResponseAdvantage {
                baseline_mean_ev_bb: self.baseline_sum / count,
                selected_mean_gain_bb: self.advantage_means[selected_action],
                selected_gain_standard_error_bb: error,
                approximate_gain_lower_bound_99_5_percent_bb: self.advantage_means[selected_action]
                    - 2.575_829_303_548_900_4 * error,
            }
        });
        ResolverDecision {
            information_set: key,
            granularity,
            actor: self.actor,
            street: self.street,
            hand_bucket_trajectory: self.hand_bucket_trajectory,
            public_bucket_trajectory: self.public_bucket_trajectory,
            public_history: self.public_history,
            action_labels: self.action_labels,
            action_values_bb: means,
            action_standard_errors_bb: standard_errors,
            selected_action,
            selected_action_mean_gap_bb,
            approximate_selected_action_gap_lower_bound_99_5_percent_bb: gap_lower_bound,
            response_advantage,
            low_confidence: runner_up.is_some() && gap_lower_bound <= 0.0,
            range_particles: self.count,
        }
    }
}

#[derive(Default)]
struct ResolverLookup {
    queries: u64,
    hits: u64,
    preflop_queries: u64,
    preflop_hits: u64,
    postflop_queries: u64,
    postflop_hits: u64,
    exact_hits: u64,
    observable_backoff_hits: u64,
    coarse_observable_backoff_hits: u64,
    strategic_observable_backoff_hits: u64,
    postflop_exact_hits: u64,
    postflop_observable_backoff_hits: u64,
    postflop_coarse_observable_backoff_hits: u64,
    postflop_strategic_observable_backoff_hits: u64,
}

impl ResolverLookup {
    fn add(&mut self, other: Self) {
        self.queries += other.queries;
        self.hits += other.hits;
        self.preflop_queries += other.preflop_queries;
        self.preflop_hits += other.preflop_hits;
        self.postflop_queries += other.postflop_queries;
        self.postflop_hits += other.postflop_hits;
        self.exact_hits += other.exact_hits;
        self.observable_backoff_hits += other.observable_backoff_hits;
        self.coarse_observable_backoff_hits += other.coarse_observable_backoff_hits;
        self.strategic_observable_backoff_hits += other.strategic_observable_backoff_hits;
        self.postflop_exact_hits += other.postflop_exact_hits;
        self.postflop_observable_backoff_hits += other.postflop_observable_backoff_hits;
        self.postflop_coarse_observable_backoff_hits +=
            other.postflop_coarse_observable_backoff_hits;
        self.postflop_strategic_observable_backoff_hits +=
            other.postflop_strategic_observable_backoff_hits;
    }
}

pub fn evaluate_full_game_response(
    config: ResponseEvaluationConfig,
) -> Result<FullGameResponseEvaluation, Box<dyn Error>> {
    evaluate_full_game_response_inner(config, None)
}

fn response_method(config: &ResponseEvaluationConfig) -> String {
    "calibrated_one_step_common_random_full_game_rollout_response_with_exact_fine_coarse_and_strategic_observable_information_sets_and_paired_action_gap_errors_and_aligned_intervention_draws_and_paired_baseline_advantages"
        .to_owned() + if config.exact_terminal_training_values { "_with_exact_postflop_terminal_training_values" } else { "" }
        + if config.postflop_only_response { "_postflop_response_only" } else { "" }
}

fn evaluate_full_game_response_inner(
    mut config: ResponseEvaluationConfig,
    retained: Option<(FullGameResponseEvaluation, String)>,
) -> Result<FullGameResponseEvaluation, Box<dyn Error>> {
    if let Some(options) = &config.flop_backoff {
        options.validate()?;
        if !matches!(&config.source, ResponsePolicySource::TabularCheckpoint(_)) {
            return Err("flop pooling requires a frozen tabular checkpoint".into());
        }
    }
    if let Some(options) = &config.terminal_flop {
        options.validate()?;
        if !matches!(&config.source, ResponsePolicySource::TabularCheckpoint(_)) {
            return Err("terminal flop range correction requires a tabular checkpoint".into());
        }
    }
    if let Some(options) = &config.turn_resolver {
        options.validate()?;
        if !matches!(&config.source, ResponsePolicySource::TabularCheckpoint(_)) {
            return Err("tabular turn resolving requires a tabular checkpoint".into());
        }
    }
    if !(1..=4).contains(&config.response_workers)
        || (config.response_workers > 1
            && !matches!(&config.source, ResponsePolicySource::TabularCheckpoint(_)))
    {
        return Err("response evaluation requires 1..=4 workers; parallel workers require a tabular checkpoint".into());
    }
    if config.training_deals == 0
        || config.calibration_deals < 2
        || config.evaluation_deals < 2
        || config.rollouts_per_action < 2
        || config.minimum_range_particles < 2
    {
        return Err(
            "response evaluation requires training deals and at least two calibration deals, evaluation deals, rollouts, and range particles"
                .into(),
        );
    }
    let (policy, policy_sha256, policy_source_kind, checkpoint_training_iterations): (
        Box<dyn ResponsePolicy>,
        String,
        String,
        Option<u64>,
    ) = match &config.source {
        ResponsePolicySource::Neural(path) => (
            Box::new(FrozenPolicy::load(path)?),
            sha256_file(path)?,
            "frozen_neural".to_owned(),
            None,
        ),
        ResponsePolicySource::TabularCheckpoint(path) => {
            let digest = sha256_file(path)?;
            let table = InferenceTable::read(path)?;
            config.game = table.config.clone();
            let rounds = table.rounds;
            let flop_backoff = config
                .flop_backoff
                .clone()
                .map(|o| backoff::FlopBackoff::build(&table, o).map(Arc::new))
                .transpose()?;
            if let Some(pooled) = &flop_backoff {
                eprintln!("flop-pooling {}", pooled.summary());
            }
            let base = TabularResponsePolicy {
                table: Arc::new(table),
                coverage: RefCell::default(),
                flop_backoff,
                completion_coverage: RefCell::default(),
                flop_patch: config
                    .terminal_flop
                    .as_ref()
                    .map(|options| Arc::new(flop::FlopPatch::terminal(options))),
            };
            let policy: Box<dyn ResponsePolicy> =
                if let Some(options) = config.turn_resolver.clone() {
                    Box::new(turn::TabularTurnPolicy::new(base, options))
                } else {
                    Box::new(base)
                };
            (
                policy,
                digest,
                if config.turn_resolver.is_some() {
                    "tabular_trunk_with_joint_turn_river_resolving"
                } else {
                    "frozen_tabular_average_with_explicit_uniform_completion"
                }
                .to_owned(),
                Some(rounds),
            )
        }
    };
    config.game.validate()?;
    let policy_source_kind = if config.flop_backoff.is_some() {
        format!("{policy_source_kind}_with_frozen_flop_mass_backoff")
    } else {
        policy_source_kind
    };
    let policy_source_kind = if config.terminal_flop.is_some() {
        format!("{policy_source_kind}_with_terminal_flop_range_correction")
    } else {
        policy_source_kind
    };
    if let Some((report, _)) = &retained {
        flop::validate_report(report, &policy_sha256, config.game.effective_stack_bb)?;
        if report.checkpoint_training_iterations != checkpoint_training_iterations
            || report.policy_source_kind != policy_source_kind
            || report.method != response_method(&config)
        {
            return Err("retained response must match the frozen profile, iterations, and response method".into());
        }
    }
    let policy = policy.as_ref();
    let network_sha256 = if checkpoint_training_iterations.is_none() {
        policy_sha256.clone()
    } else {
        String::new()
    };
    let mut source_policy_coverage = BTreeMap::new();
    let mut resolution_diagnostics = BTreeMap::new();
    let (preflop_responses, resolvers, retained_training) = if let Some((report, digest)) = retained {
        // Freeze the exact trained rows, including rejected seats. Only fresh
        // calibration may admit them. No old calibration payoff is pooled in.
        let provenance = RetainedResponseTraining { report_sha256: digest, seed: report.seed };
        (report.preflop_responses, report.resolvers, Some(provenance))
    } else {
        let [first, second] = [
            train_learned_response(policy, &config, 0),
            train_learned_response(policy, &config, 1),
        ];
        source_policy_coverage.insert("response_training".to_owned(), policy.take_coverage());
        if let Some(value) = policy.take_resolution_diagnostics() {
            resolution_diagnostics.insert("response_training".to_owned(), value);
        }
        ([first.0, second.0], [first.1, second.1], None)
    };
    let calibration_players = [
        evaluate_resolver(
            policy,
            &preflop_responses[0],
            &resolvers[0],
            &config,
            0,
            config.calibration_deals,
            u64::MAX - 1,
            true,
        ),
        evaluate_resolver(
            policy,
            &preflop_responses[1],
            &resolvers[1],
            &config,
            1,
            config.calibration_deals,
            u64::MAX - 1,
            true,
        ),
    ];
    source_policy_coverage.insert("response_calibration".to_owned(), policy.take_coverage());
    if let Some(value) = policy.take_resolution_diagnostics() {
        resolution_diagnostics.insert("response_calibration".to_owned(), value);
    }
    let response_deployed = [
        response_lower_bound_passes_calibration(
            calibration_players[0].approximate_one_sided_99_5_percent_gain_lower_bound_bb,
        ),
        response_lower_bound_passes_calibration(
            calibration_players[1].approximate_one_sided_99_5_percent_gain_lower_bound_bb,
        ),
    ];
    let players = [
        evaluate_resolver(
            policy,
            &preflop_responses[0],
            &resolvers[0],
            &config,
            0,
            config.evaluation_deals,
            u64::MAX,
            response_deployed[0],
        ),
        evaluate_resolver(
            policy,
            &preflop_responses[1],
            &resolvers[1],
            &config,
            1,
            config.evaluation_deals,
            u64::MAX,
            response_deployed[1],
        ),
    ];
    source_policy_coverage.insert("independent_evaluation".to_owned(), policy.take_coverage());
    if let Some(value) = policy.take_resolution_diagnostics() {
        resolution_diagnostics.insert("independent_evaluation".to_owned(), value);
    }
    let lower_bound =
        (players[0].estimated_gain_bb.max(0.0) + players[1].estimated_gain_bb.max(0.0)) / 2.0;
    let confidence_lower_bound = (players[0]
        .approximate_one_sided_99_5_percent_gain_lower_bound_bb
        .max(0.0)
        + players[1]
            .approximate_one_sided_99_5_percent_gain_lower_bound_bb
            .max(0.0))
        / 2.0;
    Ok(FullGameResponseEvaluation {
        schema: if checkpoint_training_iterations.is_some() { "hu-tabular-checkpoint-information-set-response-v1" } else { RESPONSE_SCHEMA }.to_owned(),
        method: response_method(&config),
        depth_bb: config.game.effective_stack_bb,
        network_sha256,
        policy_sha256,
        policy_source_kind,
        checkpoint_training_iterations,
        source_policy_coverage,
        turn_resolver: config.turn_resolver,
        terminal_flop: config.terminal_flop,
        flop_backoff: config.flop_backoff,
        exact_terminal_training_values: config.exact_terminal_training_values,
        postflop_only_response: config.postflop_only_response,
        retained_training,
        resolution_diagnostics,
        total_response_gain_bb_per_hand: players.iter().map(|player| player.estimated_gain_bb).sum(),
        total_response_gain_lower_confidence_bound_99_percent_bb_per_hand: 2.0 * confidence_lower_bound,
        seed: config.seed,
        training_deals: config.training_deals,
        response_workers: config.response_workers,
        calibration_deals: config.calibration_deals,
        evaluation_deals: config.evaluation_deals,
        rollouts_per_action: config.rollouts_per_action,
        minimum_range_particles: config.minimum_range_particles,
        maximum_granularity: config.maximum_granularity,
        players,
        calibration_players,
        response_deployed,
        approximate_exploitability_lower_bound_bb_per_hand: lower_bound,
        approximate_exploitability_lower_confidence_bound_99_percent_bb_per_hand:
            confidence_lower_bound,
        interpretation: "a fixed legal imperfect-information learned response is trained, accepted only when a disjoint calibration corpus has a positive one-sided 99.5% gain lower bound, and measured on a third independent corpus; rejected players deploy the frozen baseline with zero claimed gain; total_response_gain sums the seats (legacy approximate_exploitability fields use half that scale and clamp negative estimates); tabular missing/untrained lookups are disclosed by street and phase, never silently treated as trained; optional flop pooling borrows frozen average mass and reports matches separately, otherwise retaining the trainer's explicit uniform completion; expected response gain is a lower bound, not an exploitability upper-bound certificate; low response coverage can miss leaks"
            .to_owned(),
        preflop_responses,
        resolvers,
    })
}

fn response_lower_bound_passes_calibration(lower_bound_bb: f64) -> bool {
    lower_bound_bb.is_finite() && lower_bound_bb > 0.0
}

fn train_learned_response(
    policy: &dyn ResponsePolicy,
    config: &ResponseEvaluationConfig,
    responder: usize,
) -> (Vec<ResolverDecision>, RangeConditionedResolver) {
    let mut chance = SplitMix64::new(derived_seed(config.seed, responder as u64, 0));
    let mut exact_accumulators = BTreeMap::<u64, DecisionAccumulator>::new();
    let mut backoff_accumulators = BTreeMap::<u64, DecisionAccumulator>::new();
    let mut coarse_backoff_accumulators = BTreeMap::<u64, DecisionAccumulator>::new();
    let mut strategic_backoff_accumulators = BTreeMap::<u64, DecisionAccumulator>::new();
    parallel::for_each_deal(
        policy,
        config.response_workers,
        &mut chance,
        config.training_deals,
        |local, deal, deal_index| {
            if deal_index % 128 == 0 {
                eprintln!(
                    "response-training player={responder} deals={deal_index}/{}",
                    config.training_deals
                );
            }
            let mut trajectory_rng =
                SplitMix64::new(derived_seed(config.seed, responder as u64, deal_index + 1));
            let mut observations = Vec::new();
            collect_trajectory_decisions(
                local,
                GameState::initial(&config.game),
                deal,
                &config.game,
                responder,
                config.rollouts_per_action,
                config.exact_terminal_training_values,
                config.postflop_only_response,
                config.seed,
                deal_index,
                &mut trajectory_rng,
                &mut observations,
            );
            observations
        },
        |observations| {
            for observation in observations {
                let labels: Vec<_> = observation
                    .actions
                    .iter()
                    .map(|action| action.label.clone())
                    .collect();
                for (layer, ((key, descriptor, history), bank)) in observation
                    .keys
                    .into_iter()
                    .zip([
                        &mut exact_accumulators,
                        &mut backoff_accumulators,
                        &mut coarse_backoff_accumulators,
                        &mut strategic_backoff_accumulators,
                    ])
                    .enumerate()
                {
                    let expected = if layer == 3 {
                        &observation.strategic_labels
                    } else {
                        &labels
                    };
                    let accumulator = bank.entry(key).or_insert_with(|| {
                        let mut value =
                            DecisionAccumulator::new(&descriptor, history, &observation.actions);
                        value.action_labels = expected.clone();
                        value
                    });
                    assert_eq!(&accumulator.action_labels, expected);
                    accumulator
                        .add_with_strategy(&observation.values, &observation.baseline_strategy);
                }
            }
        },
    );
    let exact_decisions = exact_accumulators
        .into_iter()
        .filter(|(_, accumulator)| accumulator.count >= config.minimum_range_particles)
        .map(|(key, accumulator)| {
            accumulator.finish(
                key,
                ResolverGranularity::ExactTrajectory,
                config.game.effective_stack_bb,
            )
        });
    let backoff_decisions = backoff_accumulators
        .into_iter()
        .filter(|(_, accumulator)| accumulator.count >= config.minimum_range_particles)
        .map(|(key, accumulator)| {
            accumulator.finish(
                key,
                ResolverGranularity::ObservableBackoff,
                config.game.effective_stack_bb,
            )
        });
    let coarse_backoff_decisions = coarse_backoff_accumulators
        .into_iter()
        .filter(|(_, accumulator)| accumulator.count >= config.minimum_range_particles)
        .map(|(key, accumulator)| {
            accumulator.finish(
                key,
                ResolverGranularity::CoarseObservableBackoff,
                config.game.effective_stack_bb,
            )
        });
    let strategic_backoff_decisions = strategic_backoff_accumulators
        .into_iter()
        .filter(|(_, accumulator)| accumulator.count >= config.minimum_range_particles)
        .map(|(key, accumulator)| {
            accumulator.finish(
                key,
                ResolverGranularity::StrategicObservableBackoff,
                config.game.effective_stack_bb,
            )
        });
    let decisions = exact_decisions
        .chain(backoff_decisions)
        .chain(coarse_backoff_decisions)
        .chain(strategic_backoff_decisions)
        .collect::<Vec<_>>();
    let preflop = decisions
        .iter()
        .filter(|decision| decision.street == Street::Preflop)
        .cloned()
        .collect();
    let resolver = RangeConditionedResolver {
        schema: RESOLVER_SCHEMA.to_owned(),
        responder,
        training_deals: config.training_deals,
        rollouts_per_action: config.rollouts_per_action,
        minimum_range_particles: config.minimum_range_particles,
        decisions: decisions
            .into_iter()
            .filter(|decision| decision.street != Street::Preflop)
            .collect(),
    };
    (preflop, resolver)
}

struct DecisionObservation {
    baseline_strategy: Vec<f64>,
    keys: [(u64, NodeDescriptor, Vec<String>); 4],
    strategic_labels: Vec<String>,
    actions: Vec<LegalAction>,
    values: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
fn collect_trajectory_decisions(
    policy: &dyn ResponsePolicy,
    state: GameState,
    deal: &Deal,
    game: &BlueprintConfig,
    responder: usize,
    rollouts_per_action: u32,
    exact_terminal_training_values: bool,
    postflop_only_response: bool,
    response_seed: u64,
    deal_index: u64,
    trajectory_rng: &mut SplitMix64,
    observations: &mut Vec<DecisionObservation>,
) {
    if state.terminal.is_some() {
        return;
    }
    let actions = state.legal_actions(game);
    // This is the profile actually being attacked, including exact/private
    // distinctions forgotten by coarser response keys. Average Q-values alone
    // cannot establish that a coarser response beats that informed profile.
    let strategy = policy.strategy(&state, deal, &actions, game);
    if state.actor == responder && (!postflop_only_response || state.street != Street::Preflop) {
        let (key, descriptor, history) = information_set(&state, deal, game);
        let (backoff_key, backoff_descriptor, backoff_history) =
            observable_backoff_information_set(&state, deal, game, &actions);
        let (coarse_key, coarse_descriptor, coarse_history) =
            coarse_observable_backoff_information_set(&state, deal, game, &actions);
        let (strategic_key, strategic_descriptor, strategic_history, strategic_labels) =
            strategic_observable_backoff_information_set(&state, deal, game, &actions);
        let values = actions
            .iter()
            .map(|action| {
                let next = state.apply(action, game);
                if exact_terminal_training_values {
                    if let Some(utility) = terminal::expectation(&next, deal) {
                        return if responder == 0 { utility } else { -utility };
                    }
                }
                (0..rollouts_per_action)
                    .map(|rollout| {
                        let mut rng = SplitMix64::new(derived_seed(
                            response_seed ^ ((responder as u64 + 1) << 61),
                            deal_index ^ key,
                            rollout as u64,
                        ));
                        let utility = baseline_rollout(policy, next.clone(), deal, game, &mut rng);
                        if responder == 0 {
                            utility
                        } else {
                            -utility
                        }
                    })
                    .sum::<f64>()
                    / rollouts_per_action as f64
            })
            .collect::<Vec<_>>();
        observations.push(DecisionObservation {
            baseline_strategy: strategy.clone(),
            keys: [
                (key, descriptor, history),
                (backoff_key, backoff_descriptor, backoff_history),
                (coarse_key, coarse_descriptor, coarse_history),
                (strategic_key, strategic_descriptor, strategic_history),
            ],
            strategic_labels,
            actions: actions.clone(),
            values,
        });
    }
    let selected = sample_index(&strategy, trajectory_rng);
    collect_trajectory_decisions(
        policy,
        state.apply(&actions[selected], game),
        deal,
        game,
        responder,
        rollouts_per_action,
        exact_terminal_training_values,
        postflop_only_response,
        response_seed,
        deal_index,
        trajectory_rng,
        observations,
    );
}

fn observable_backoff_information_set(
    state: &GameState,
    deal: &Deal,
    game: &BlueprintConfig,
    actions: &[LegalAction],
) -> (u64, NodeDescriptor, Vec<String>) {
    let (_, mut descriptor, _) = information_set(state, deal, game);
    descriptor.hand_bucket_trajectory = descriptor
        .hand_bucket_trajectory
        .last()
        .cloned()
        .into_iter()
        .collect::<Vec<_>>()
        .into();
    descriptor.public_bucket_trajectory = descriptor
        .public_bucket_trajectory
        .last()
        .cloned()
        .into_iter()
        .collect::<Vec<_>>()
        .into();
    let price = if state.to_call() <= EPSILON {
        0.0
    } else {
        quantize(
            state.to_call() / (state.pot() + state.to_call()).max(EPSILON),
            0.05,
        )
    };
    let spr = quantize(
        state.remaining(state.actor, game) / state.pot().max(game.big_blind_bb),
        0.5,
    );
    let action_identity = actions
        .iter()
        .map(|action| action.label.as_str())
        .collect::<Vec<_>>()
        .join(">");
    let identity = format!(
        "observable-response-v1|{:?}|p{}|h:{}|b:{}|price:{price:.2}|spr:{spr:.1}|a:{action_identity}",
        state.street,
        state.actor,
        descriptor.hand_bucket_trajectory.join(">"),
        descriptor.public_bucket_trajectory.join(">"),
    );
    let key = stable_hash(identity.as_bytes());
    descriptor.public_history_id = key;
    (
        key,
        descriptor,
        vec![format!("observable_backoff:{identity}")],
    )
}

fn coarse_observable_backoff_information_set(
    state: &GameState,
    deal: &Deal,
    game: &BlueprintConfig,
    actions: &[LegalAction],
) -> (u64, NodeDescriptor, Vec<String>) {
    let (_, mut descriptor, _) = information_set(state, deal, game);
    descriptor.hand_bucket_trajectory = descriptor
        .hand_bucket_trajectory
        .last()
        .map(|bucket| {
            if state.street == Street::Preflop {
                return bucket.clone();
            }
            let parts = bucket.split(':').collect::<Vec<_>>();
            let category = parts.first().copied().unwrap_or("unknown");
            let flush_draw = parts
                .iter()
                .find(|part| part.starts_with("fd"))
                .copied()
                .unwrap_or("fd0");
            let straight_draw = parts
                .iter()
                .find(|part| part.starts_with("sd"))
                .copied()
                .unwrap_or("sd0");
            format!("{category}:{flush_draw}:{straight_draw}").into()
        })
        .into_iter()
        .collect::<Vec<_>>()
        .into();
    descriptor.public_bucket_trajectory = descriptor
        .public_bucket_trajectory
        .last()
        .map(|bucket| {
            bucket
                .split(':')
                .take(3)
                .collect::<Vec<_>>()
                .join(":")
                .into()
        })
        .into_iter()
        .collect::<Vec<_>>()
        .into();
    let price = if state.to_call() <= EPSILON {
        0.0
    } else {
        quantize(
            state.to_call() / (state.pot() + state.to_call()).max(EPSILON),
            0.10,
        )
    };
    let spr = quantize(
        state.remaining(state.actor, game) / state.pot().max(game.big_blind_bb),
        1.0,
    );
    let action_identity = actions
        .iter()
        .map(|action| action.label.as_str())
        .collect::<Vec<_>>()
        .join(">");
    let identity = format!(
        "coarse-observable-response-v1|{:?}|p{}|h:{}|b:{}|price:{price:.1}|spr:{spr:.0}|a:{action_identity}",
        state.street,
        state.actor,
        descriptor.hand_bucket_trajectory.join(">"),
        descriptor.public_bucket_trajectory.join(">"),
    );
    let key = stable_hash(identity.as_bytes());
    descriptor.public_history_id = key;
    (
        key,
        descriptor,
        vec![format!("coarse_observable_backoff:{identity}")],
    )
}

fn strategic_action_labels(
    state: &GameState,
    actions: &[LegalAction],
    game: &BlueprintConfig,
) -> Vec<String> {
    let facing_bet = state.to_call() > EPSILON;
    let all_in_target = state.street_invested[state.actor] + state.remaining(state.actor, game);
    let mut sized_action = 0usize;
    actions
        .iter()
        .map(|action| match action.kind {
            ActionKind::Fold => "fold".to_owned(),
            ActionKind::Check => "check".to_owned(),
            ActionKind::Call => "call".to_owned(),
            ActionKind::RaiseTo(target) if (target - all_in_target).abs() <= EPSILON => {
                "all_in".to_owned()
            }
            ActionKind::RaiseTo(_) => {
                sized_action += 1;
                if facing_bet {
                    format!("raise_{sized_action}")
                } else {
                    format!("bet_{sized_action}")
                }
            }
        })
        .collect()
}

fn strategic_observable_backoff_information_set(
    state: &GameState,
    deal: &Deal,
    game: &BlueprintConfig,
    actions: &[LegalAction],
) -> (u64, NodeDescriptor, Vec<String>, Vec<String>) {
    let (_, mut descriptor, _) = information_set(state, deal, game);
    let strength = descriptor
        .hand_bucket_trajectory
        .last()
        .map(|bucket| {
            if state.street == Street::Preflop {
                bucket.clone()
            } else {
                bucket
                    .split(':')
                    .next()
                    .unwrap_or("unknown")
                    .to_owned()
                    .into()
            }
        })
        .unwrap_or_else(|| "unknown".into());
    descriptor.hand_bucket_trajectory = vec![strength.clone()].into();
    descriptor.public_bucket_trajectory = Vec::new().into();
    let price = if state.to_call() <= EPSILON {
        "free"
    } else {
        let pot_odds = state.to_call() / (state.pot() + state.to_call()).max(EPSILON);
        if pot_odds <= 0.25 {
            "low"
        } else if pot_odds <= 0.5 {
            "medium"
        } else {
            "high"
        }
    };
    let spr = state.remaining(state.actor, game) / state.pot().max(game.big_blind_bb);
    let spr_band = if spr <= 1.0 {
        "low"
    } else if spr <= 4.0 {
        "medium"
    } else {
        "high"
    };
    let action_labels = strategic_action_labels(state, actions, game);
    let action_identity = action_labels.join(">");
    let identity = format!(
        "strategic-observable-response-v1|{:?}|p{}|h:{strength}|price:{price}|spr:{spr_band}|a:{action_identity}",
        state.street, state.actor,
    );
    let key = stable_hash(identity.as_bytes());
    descriptor.public_history_id = key;
    (
        key,
        descriptor,
        vec![format!("strategic_observable_backoff:{identity}")],
        action_labels,
    )
}

fn evaluate_resolver(
    policy: &dyn ResponsePolicy,
    preflop: &[ResolverDecision],
    resolver: &RangeConditionedResolver,
    config: &ResponseEvaluationConfig,
    responder: usize,
    deals: u64,
    seed_domain: u64,
    deploy_response: bool,
) -> ResponsePlayerEvaluation {
    let confident = preflop
        .iter()
        .chain(&resolver.decisions)
        .filter(|decision| {
            deploy_response
                && decision.is_profitable_response()
                && granularity_rank(decision.granularity)
                    <= granularity_rank(config.maximum_granularity)
        });
    let exact_decisions = confident
        .clone()
        .filter(|decision| decision.granularity == ResolverGranularity::ExactTrajectory)
        .map(|decision| (decision.information_set, decision))
        .collect::<BTreeMap<_, _>>();
    let backoff_decisions = confident
        .clone()
        .filter(|decision| decision.granularity == ResolverGranularity::ObservableBackoff)
        .map(|decision| (decision.information_set, decision))
        .collect::<BTreeMap<_, _>>();
    let coarse_backoff_decisions = confident
        .filter(|decision| decision.granularity == ResolverGranularity::CoarseObservableBackoff)
        .map(|decision| (decision.information_set, decision))
        .collect::<BTreeMap<_, _>>();
    let strategic_backoff_decisions = preflop
        .iter()
        .chain(&resolver.decisions)
        .filter(|decision| {
            deploy_response
                && decision.is_profitable_response()
                && decision.granularity == ResolverGranularity::StrategicObservableBackoff
                && granularity_rank(decision.granularity)
                    <= granularity_rank(config.maximum_granularity)
        })
        .map(|decision| (decision.information_set, decision))
        .collect::<BTreeMap<_, _>>();
    let phase_seed = derived_seed(config.seed, responder as u64, seed_domain);
    let mut chance = SplitMix64::new(phase_seed);
    let mut differences = Vec::with_capacity(deals as usize);
    let mut baseline_total = 0.0;
    let mut response_total = 0.0;
    let mut lookup = ResolverLookup::default();
    parallel::for_each_deal(
        policy,
        config.response_workers,
        &mut chance,
        deals,
        |local, deal, deal_index| {
            if deal_index % 512 == 0 {
                eprintln!("response-evaluation domain={seed_domain} player={responder} deals={deal_index}/{deals} deployed={deploy_response}");
            }
            let mut hand_lookup = ResolverLookup::default();
            let rollout_seed = derived_seed(phase_seed, deal_index, 11);
            let mut baseline_rng = SplitMix64::new(rollout_seed);
            let mut response_rng = SplitMix64::new(rollout_seed);
            let baseline_p0 = baseline_rollout(
                local,
                GameState::initial(&config.game),
                deal,
                &config.game,
                &mut baseline_rng,
            );
            let response_p0 = response_rollout(
                local,
                &exact_decisions,
                &backoff_decisions,
                &coarse_backoff_decisions,
                &strategic_backoff_decisions,
                GameState::initial(&config.game),
                deal,
                &config.game,
                responder,
                false,
                &mut response_rng,
                &mut hand_lookup,
            );
            let baseline = if responder == 0 {
                baseline_p0
            } else {
                -baseline_p0
            };
            let response = if responder == 0 {
                response_p0
            } else {
                -response_p0
            };
            (baseline, response, hand_lookup)
        },
        |(baseline, response, hand_lookup)| {
            baseline_total += baseline;
            response_total += response;
            differences.push(response - baseline);
            lookup.add(hand_lookup);
        },
    );
    let count = differences.len() as f64;
    let mean = differences.iter().sum::<f64>() / count;
    let squared = differences
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>();
    ResponsePlayerEvaluation {
        responder,
        response_deployed: deploy_response,
        baseline_utility_bb: baseline_total / count,
        response_utility_bb: response_total / count,
        estimated_gain_bb: mean,
        gain_standard_error_bb: (squared / ((count - 1.0) * count)).sqrt(),
        approximate_one_sided_99_5_percent_gain_lower_bound_bb: mean
            - 2.575_829_303_548_900_4 * (squared / ((count - 1.0) * count)).sqrt(),
        resolver_lookup_coverage: if lookup.queries == 0 {
            0.0
        } else {
            lookup.hits as f64 / lookup.queries as f64
        },
        exact_lookup_coverage: ratio(lookup.exact_hits, lookup.queries),
        observable_backoff_lookup_coverage: ratio(lookup.observable_backoff_hits, lookup.queries),
        coarse_observable_backoff_lookup_coverage: ratio(
            lookup.coarse_observable_backoff_hits,
            lookup.queries,
        ),
        strategic_observable_backoff_lookup_coverage: ratio(
            lookup.strategic_observable_backoff_hits,
            lookup.queries,
        ),
        preflop_lookup_coverage: ratio(lookup.preflop_hits, lookup.preflop_queries),
        postflop_lookup_coverage: ratio(lookup.postflop_hits, lookup.postflop_queries),
        postflop_exact_lookup_coverage: ratio(lookup.postflop_exact_hits, lookup.postflop_queries),
        postflop_observable_backoff_lookup_coverage: ratio(
            lookup.postflop_observable_backoff_hits,
            lookup.postflop_queries,
        ),
        postflop_coarse_observable_backoff_lookup_coverage: ratio(
            lookup.postflop_coarse_observable_backoff_hits,
            lookup.postflop_queries,
        ),
        postflop_strategic_observable_backoff_lookup_coverage: ratio(
            lookup.postflop_strategic_observable_backoff_hits,
            lookup.postflop_queries,
        ),
        learned_information_sets: preflop.len() + resolver.decisions.len(),
        confident_information_sets: exact_decisions.len()
            + backoff_decisions.len()
            + coarse_backoff_decisions.len()
            + strategic_backoff_decisions.len(),
    }
}

fn baseline_rollout(
    policy: &dyn ResponsePolicy,
    state: GameState,
    deal: &Deal,
    game: &BlueprintConfig,
    rng: &mut SplitMix64,
) -> f64 {
    if state.terminal.is_some() {
        return realized_utility_p0(&state, deal);
    }
    let actions = state.legal_actions(game);
    let strategy = policy.strategy(&state, deal, &actions, game);
    let selected = sample_index(&strategy, rng);
    baseline_rollout(
        policy,
        state.apply(&actions[selected], game),
        deal,
        game,
        rng,
    )
}

#[allow(clippy::too_many_arguments)]
fn response_rollout(
    policy: &dyn ResponsePolicy,
    exact_decisions: &BTreeMap<u64, &ResolverDecision>,
    backoff_decisions: &BTreeMap<u64, &ResolverDecision>,
    coarse_backoff_decisions: &BTreeMap<u64, &ResolverDecision>,
    strategic_backoff_decisions: &BTreeMap<u64, &ResolverDecision>,
    state: GameState,
    deal: &Deal,
    game: &BlueprintConfig,
    responder: usize,
    deviation_taken: bool,
    rng: &mut SplitMix64,
    lookup: &mut ResolverLookup,
) -> f64 {
    if state.terminal.is_some() {
        return realized_utility_p0(&state, deal);
    }
    let actions = state.legal_actions(game);
    let (selected, deviated_here) = if state.actor == responder && !deviation_taken {
        lookup.queries += 1;
        if state.street == Street::Preflop {
            lookup.preflop_queries += 1;
        } else {
            lookup.postflop_queries += 1;
        }
        let labels = actions
            .iter()
            .map(|action| action.label.clone())
            .collect::<Vec<_>>();
        let (key, _, _) = information_set(&state, deal, game);
        let (backoff_key, _, _) = observable_backoff_information_set(&state, deal, game, &actions);
        let (coarse_key, _, _) =
            coarse_observable_backoff_information_set(&state, deal, game, &actions);
        let (strategic_key, _, _, strategic_labels) =
            strategic_observable_backoff_information_set(&state, deal, game, &actions);
        let selected_decision = exact_decisions
            .get(&key)
            .filter(|decision| decision.action_labels == labels)
            .map(|decision| (*decision, ResolverGranularity::ExactTrajectory))
            .or_else(|| {
                backoff_decisions
                    .get(&backoff_key)
                    .filter(|decision| decision.action_labels == labels)
                    .map(|decision| (*decision, ResolverGranularity::ObservableBackoff))
            })
            .or_else(|| {
                coarse_backoff_decisions
                    .get(&coarse_key)
                    .filter(|decision| decision.action_labels == labels)
                    .map(|decision| (*decision, ResolverGranularity::CoarseObservableBackoff))
            })
            .or_else(|| {
                strategic_backoff_decisions
                    .get(&strategic_key)
                    .filter(|decision| decision.action_labels == strategic_labels)
                    .map(|decision| (*decision, ResolverGranularity::StrategicObservableBackoff))
            });
        match selected_decision {
            Some((decision, granularity)) => {
                // A forced action replaces a sampled action, including its
                // random draw. Keep later common-random samples aligned when
                // the intervention takes the same action as the baseline.
                rng.next_f64();
                lookup.hits += 1;
                match granularity {
                    ResolverGranularity::ExactTrajectory => lookup.exact_hits += 1,
                    ResolverGranularity::ObservableBackoff => lookup.observable_backoff_hits += 1,
                    ResolverGranularity::CoarseObservableBackoff => {
                        lookup.coarse_observable_backoff_hits += 1
                    }
                    ResolverGranularity::StrategicObservableBackoff => {
                        lookup.strategic_observable_backoff_hits += 1
                    }
                }
                if state.street == Street::Preflop {
                    lookup.preflop_hits += 1;
                } else {
                    lookup.postflop_hits += 1;
                    match granularity {
                        ResolverGranularity::ExactTrajectory => lookup.postflop_exact_hits += 1,
                        ResolverGranularity::ObservableBackoff => {
                            lookup.postflop_observable_backoff_hits += 1
                        }
                        ResolverGranularity::CoarseObservableBackoff => {
                            lookup.postflop_coarse_observable_backoff_hits += 1
                        }
                        ResolverGranularity::StrategicObservableBackoff => {
                            lookup.postflop_strategic_observable_backoff_hits += 1
                        }
                    }
                }
                (decision.selected_action, true)
            }
            _ => {
                let strategy = policy.strategy(&state, deal, &actions, game);
                (sample_index(&strategy, rng), false)
            }
        }
    } else {
        let strategy = policy.strategy(&state, deal, &actions, game);
        (sample_index(&strategy, rng), false)
    };
    response_rollout(
        policy,
        exact_decisions,
        backoff_decisions,
        coarse_backoff_decisions,
        strategic_backoff_decisions,
        state.apply(&actions[selected], game),
        deal,
        game,
        responder,
        deviation_taken || deviated_here,
        rng,
        lookup,
    )
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
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

fn derived_seed(first: u64, second: u64, third: u64) -> u64 {
    let mut digest = Sha256::new();
    digest.update(b"hu-information-set-lbr-v1");
    digest.update(first.to_le_bytes());
    digest.update(second.to_le_bytes());
    digest.update(third.to_le_bytes());
    let bytes = digest.finalize();
    u64::from_le_bytes(bytes[..8].try_into().expect("SHA-256 prefix"))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tied_good_actions_can_still_exploit_a_lossy_baseline_mix() {
        let (trainer, deal) = fixture_trainer();
        let state = GameState::initial(&trainer.config);
        let (key, descriptor, history) = information_set(&state, &deal, &trainer.config);
        let actions = state.legal_actions(&trainer.config);
        let mut acc = DecisionAccumulator::new(&descriptor, history, &actions);
        let mut values = vec![-10.0; actions.len()];
        values[0] = 1.0;
        values[1] = 1.0;
        assert!(values.iter().sum::<f64>() / (values.len() as f64) < 1.0);
        for _ in 0..8 {
            acc.add_with_strategy(&values, &vec![1.0 / actions.len() as f64; actions.len()]);
        }
        let decision = acc.finish(key, ResolverGranularity::ExactTrajectory, 20.0);
        assert_eq!(decision.selected_action_mean_gap_bb, 0.0);
        assert!(decision.is_profitable_response(),
            "either tied winner strictly improves the uniform baseline; a unique winner is unnecessary");
        let advantage = decision.response_advantage.unwrap();
        assert_eq!(advantage.selected_gain_standard_error_bb, 0.0);
        assert!(
            (advantage.selected_mean_gain_bb
                - (1.0 - values.iter().sum::<f64>() / values.len() as f64))
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn response_compares_against_the_informed_baseline_not_an_averaged_mix() {
        let (trainer, deal) = fixture_trainer();
        let state = GameState::initial(&trainer.config);
        let (_, descriptor, history) = information_set(&state, &deal, &trainer.config);
        let actions = state.legal_actions(&trainer.config);
        let mut acc = DecisionAccumulator::new(&descriptor, history, &actions);
        for sample in 0..1_000 {
            let mut values = vec![-10.0; actions.len()];
            let mut baseline = vec![0.0; actions.len()];
            if sample % 10 < 6 {
                values[0] = 10.0;
                values[1] = 0.0;
                baseline[0] = 1.0;
            } else {
                values[0] = 0.0;
                values[1] = 9.0;
                baseline[1] = 1.0;
            }
            acc.add_with_strategy(&values, &baseline);
        }
        let decision = acc.finish(0, ResolverGranularity::StrategicObservableBackoff, 20.0);
        assert_eq!(decision.selected_action, 0);
        assert!(
            !decision.low_confidence,
            "the old unique-action rule accepts this losing response"
        );
        assert!(!decision.is_profitable_response());
        let advantage = decision.response_advantage.unwrap();
        assert!((advantage.baseline_mean_ev_bb - 9.6).abs() < 1e-12);
        assert!((advantage.selected_mean_gain_bb + 3.6).abs() < 1e-12);
    }

    #[test]
    fn paired_response_advantages_cancel_common_noise_and_reject_identity_updates() {
        let (trainer, deal) = fixture_trainer();
        let state = GameState::initial(&trainer.config);
        let (_, descriptor, history) = information_set(&state, &deal, &trainer.config);
        let actions = state.legal_actions(&trainer.config);
        for baseline_action in [0, 1] {
            let mut acc = DecisionAccumulator::new(&descriptor, history.clone(), &actions);
            let mut baseline = vec![0.0; actions.len()];
            baseline[baseline_action] = 1.0;
            for sample in 0..100 {
                let noise = if sample % 2 == 0 { 10.0 } else { -10.0 };
                let mut values = vec![noise - 5.0; actions.len()];
                values[0] = noise + 1.0;
                values[1] = noise;
                acc.add_with_strategy(&values, &baseline);
            }
            let decision = acc.finish(0, ResolverGranularity::ExactTrajectory, 20.0);
            assert_eq!(decision.is_profitable_response(), baseline_action == 1);
            let advantage = decision.response_advantage.unwrap();
            assert_eq!(advantage.selected_mean_gain_bb, baseline_action as f64);
            assert_eq!(advantage.selected_gain_standard_error_bb, 0.0);
        }
    }

    #[test]
    fn forcing_the_same_action_preserves_paired_continuation_draws() {
        let (mut policy, deal) = tabular_fixture();
        let game = policy.table.config.clone();
        let state = GameState::initial(&game);
        let actions = state.legal_actions(&game);
        let (key, descriptor, history) = information_set(&state, &deal, &game);
        let node = Arc::get_mut(&mut policy.table)
            .unwrap()
            .nodes
            .get_mut(&key)
            .unwrap();
        node.strategy_sum.fill(0.0);
        node.strategy_sum[1] = 1.0;
        let mut accumulator = DecisionAccumulator::new(&descriptor, history, &actions);
        let mut values = vec![0.0; actions.len()];
        values[1] = 1.0;
        for _ in 0..4 {
            accumulator.add(&values);
        }
        let decision = accumulator.finish(key, ResolverGranularity::ExactTrajectory, 20.0);
        let exact = BTreeMap::from([(key, &decision)]);
        let empty = BTreeMap::new();
        // Real full-hand recursion: identical initial action, stochastic later
        // actions, folds and showdowns. An identity intervention has zero noise.
        for seed in 0..128 {
            let mut baseline_rng = SplitMix64::new(seed);
            let mut response_rng = SplitMix64::new(seed);
            let baseline =
                baseline_rollout(&policy, state.clone(), &deal, &game, &mut baseline_rng);
            let mut lookup = ResolverLookup::default();
            let response = response_rollout(
                &policy,
                &exact,
                &empty,
                &empty,
                &empty,
                state.clone(),
                &deal,
                &game,
                0,
                false,
                &mut response_rng,
                &mut lookup,
            );
            assert_eq!(lookup.hits, 1);
            assert_eq!(
                baseline, response,
                "identity response changed payoff for seed {seed}"
            );
            assert_eq!(baseline_rng.state(), response_rng.state());
        }
    }

    #[test]
    fn response_gap_uncertainty_uses_paired_samples_not_marginal_variances() {
        let (trainer, deal) = fixture_trainer();
        let state = GameState::initial(&trainer.config);
        let (_, descriptor, history) = information_set(&state, &deal, &trainer.config);
        let actions = state.legal_actions(&trainer.config);
        let mut paired = DecisionAccumulator::new(&descriptor, history.clone(), &actions);
        let mut anticorrelated = DecisionAccumulator::new(&descriptor, history, &actions);
        for index in 0..100 {
            let common = if index % 2 == 0 { 10.0 } else { -10.0 };
            let mut values = vec![common - 5.0; actions.len()];
            values[0] = common + 1.0;
            values[1] = common;
            paired.add(&values);
            values[1] = -common;
            anticorrelated.add(&values);
        }
        let paired = paired.finish(0, ResolverGranularity::ExactTrajectory, 20.0);
        assert_eq!(paired.selected_action, 0);
        assert_eq!(paired.selected_action_mean_gap_bb, 1.0);
        assert!(
            !paired.low_confidence,
            "identical paired differences have no sampling variance"
        );
        assert_eq!(
            paired.approximate_selected_action_gap_lower_bound_99_5_percent_bb,
            1.0
        );
        assert!(
            anticorrelated
                .finish(0, ResolverGranularity::ExactTrajectory, 20.0)
                .low_confidence
        );
    }

    pub(super) fn fixture_trainer() -> (Trainer, Deal) {
        let game = BlueprintConfig {
            effective_stack_bb: 20.0,
            iterations: 4,
            averaging_delay: 0,
            ..BlueprintConfig::default()
        };
        let deal = Deal::from_cards([[51, 50], [45, 44]], [0, 5, 10, 27, 28]);
        let state = GameState::initial(&game);
        let (key, descriptor, history) = information_set(&state, &deal, &game);
        let actions = state.legal_actions(&game);
        let mut trainer = Trainer::fresh(game);
        trainer
            .public_histories
            .insert(descriptor.public_history_id, history);
        let mut node = Node::new(descriptor, &actions, &mut trainer.string_interner);
        node.regrets[0] = 100.0;
        node.strategy_sum[0] = 1.0;
        node.strategy_sum[1] = 7.0;
        node.average_visits = 1;
        trainer.nodes.insert(key, node);
        (trainer, deal)
    }

    pub(super) fn tabular_fixture() -> (TabularResponsePolicy, Deal) {
        let (trainer, deal) = fixture_trainer();
        (
            TabularResponsePolicy {
                table: Arc::new(InferenceTable::from_trainer(trainer)),
                coverage: RefCell::default(),
                flop_patch: None,
                flop_backoff: None,
                completion_coverage: RefCell::default(),
            },
            deal,
        )
    }

    #[test]
    fn inference_discards_training_allocations_without_changing_frozen_policy() {
        let (mut trainer, _) = fixture_trainer();
        let extra = trainer.nodes.first_key_value().unwrap().1.clone();
        trainer.nodes.insert(123, extra);
        assert!(std::mem::size_of::<AverageNode>() < std::mem::size_of::<Node>());
        for codec in ["json.gz", "msgpack.gz"] {
            let path = std::env::temp_dir()
                .join(format!("inference-reader-{}.{codec}", std::process::id()));
            trainer.write_checkpoint(&path).unwrap();
            let table = InferenceTable::read(&path).unwrap();
            assert_eq!(table.nodes.len(), trainer.nodes.len());
            for (key, original) in &trainer.nodes {
                let loaded = &table.nodes[key];
                assert_eq!(loaded.descriptor, original.descriptor);
                assert_eq!(loaded.strategy_sum, original.strategy_sum);
                assert_eq!(loaded.average_strategy(), original.average_strategy());
                assert_eq!(loaded.average_visits, original.average_visits);
            }
            let rows: Vec<_> = table.nodes.values().collect();
            assert!(Arc::ptr_eq(&rows[0].action_labels, &rows[1].action_labels));
            assert!(Arc::ptr_eq(
                &rows[0].descriptor.hand_bucket_trajectory,
                &rows[1].descriptor.hand_bucket_trajectory
            ));
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn tabular_response_uses_average_not_regrets_and_no_hidden_cards() {
        let (policy, deal) = tabular_fixture();
        let game = &policy.table.config;
        let state = GameState::initial(game);
        let actions = state.legal_actions(game);
        let mix = policy.strategy(&state, &deal, &actions, game);
        assert_eq!(&mix[..2], &[0.125, 0.875]);
        let other_hidden = Deal::from_cards([[51, 50], [41, 40]], [1, 6, 11, 26, 29]);
        assert_eq!(mix, policy.strategy(&state, &other_hidden, &actions, game));
        let mut state = state;
        let mut visited = BTreeSet::new();
        while state.terminal.is_none() {
            let actions = state.legal_actions(game);
            let strategy = policy.strategy(&state, &deal, &actions, game);
            assert_eq!(strategy.len(), actions.len());
            assert!((strategy.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            visited.insert(format!("{:?}", state.street));
            let visible = state.street.board_len();
            let mut remaining = (0..52u8).filter(|card| {
                !deal.holes[state.actor].contains(card) && !deal.board[..visible].contains(card)
            });
            let mut holes = deal.holes;
            holes[1 - state.actor] = [remaining.next().unwrap(), remaining.next().unwrap()];
            let mut board = deal.board;
            for card in &mut board[visible..] {
                *card = remaining.next().unwrap();
            }
            let alternative = Deal::from_cards(holes, board);
            assert_eq!(
                information_set(&state, &deal, game).0,
                information_set(&state, &alternative, game).0,
                "hidden cards changed {:?} key",
                state.street
            );
            let action = actions
                .iter()
                .find(|a| matches!(a.kind, ActionKind::Check | ActionKind::Call))
                .unwrap();
            state = state.apply(action, game);
        }
        assert_eq!(visited.len(), 4);
        let coverage = policy.take_coverage();
        assert!(coverage.iter().all(|c| c.coverage.decisions > 0));
        assert!(coverage
            .iter()
            .skip(1)
            .all(|c| c.coverage.unknown_information_set_fraction == 1.0));
        assert!(policy
            .take_coverage()
            .iter()
            .all(|c| c.coverage.decisions == 0));
    }

    #[test]
    fn tabular_full_hand_rollouts_match_frozen_action_paths_and_realized_settlement() {
        let (policy, _) = tabular_fixture();
        let game = &policy.table.config;
        let mut chance = SplitMix64::new(197);
        for seed in 0..100 {
            let deal = Deal::sample(&mut chance);
            let mut reference = GameState::initial(game);
            let mut rng = SplitMix64::new(seed);
            while reference.terminal.is_none() {
                let actions = reference.legal_actions(game);
                let (key, _, _) = information_set(&reference, &deal, game);
                let mix = policy
                    .table
                    .nodes
                    .get(&key)
                    .map(AverageNode::average_strategy)
                    .unwrap_or_else(|| vec![1.0 / actions.len() as f64; actions.len()]);
                reference = reference.apply(&actions[sample_index(&mix, &mut rng)], game);
            }
            // Root diagnostics integrate early all-ins over extra runouts;
            // full-game response evaluation settles this exact sampled board.
            // At a terminal state, River selects the latter in the reference.
            reference.street = Street::River;
            let expected = reference.utility_p0(&deal, game);
            let actual = baseline_rollout(
                &policy,
                GameState::initial(game),
                &deal,
                game,
                &mut SplitMix64::new(seed),
            );
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn tabular_checkpoint_response_is_deterministic_and_discloses_completion() {
        let (trainer, _) = fixture_trainer();
        let path = std::env::temp_dir().join(format!(
            "tabular-response-{}.msgpack.gz",
            std::process::id()
        ));
        trainer.write_checkpoint(&path).unwrap();
        let config = ResponseEvaluationConfig {
            game: BlueprintConfig::default(),
            response_workers: 1,
            training_deals: 8,
            calibration_deals: 8,
            evaluation_deals: 16,
            rollouts_per_action: 2,
            minimum_range_particles: 2,
            maximum_granularity: ResolverGranularity::StrategicObservableBackoff,
            seed: 915,
            source: ResponsePolicySource::TabularCheckpoint(path.clone()),
            turn_resolver: None,
            terminal_flop: None,
            flop_backoff: None,
            exact_terminal_training_values: false,
            postflop_only_response: false,
        };
        let first = evaluate_full_game_response(config.clone()).unwrap();
        let second = evaluate_full_game_response(config.clone()).unwrap();
        assert_eq!(
            serde_json::to_vec(&first).unwrap(),
            serde_json::to_vec(&second).unwrap()
        );
        for workers in [2, 4] {
            let mut parallel_config = config.clone();
            parallel_config.response_workers = workers;
            parallel_config.training_deals = 71;
            parallel_config.calibration_deals = 71;
            parallel_config.evaluation_deals = 71;
            let mut serial_config = parallel_config.clone();
            serial_config.response_workers = 1;
            let serial = evaluate_full_game_response(serial_config).unwrap();
            let mut parallel = evaluate_full_game_response(parallel_config).unwrap();
            assert_eq!(parallel.response_workers, workers);
            parallel.response_workers = 1;
            assert_eq!(
                serde_json::to_vec(&serial).unwrap(),
                serde_json::to_vec(&parallel).unwrap()
            );
        }
        assert_eq!(first.depth_bb, 20.0);
        let mut corrected = config.clone();
        corrected.exact_terminal_training_values = true;
        corrected.postflop_only_response = true;
        corrected.flop_backoff = Some(FlopBackoffOptions {
            minimum_average_visits: 1,
            weight: 1.0,
        });
        corrected.terminal_flop = Some(TerminalFlopOptions {
            equity_samples: 128,
            weight: 0.25,
        });
        let serial_correction = evaluate_full_game_response(corrected.clone()).unwrap();
        corrected.response_workers = 2;
        let mut parallel_correction = evaluate_full_game_response(corrected).unwrap();
        parallel_correction.response_workers = 1;
        assert_eq!(
            serde_json::to_vec(&serial_correction).unwrap(),
            serde_json::to_vec(&parallel_correction).unwrap()
        );
        assert!(serial_correction
            .policy_source_kind
            .contains("terminal_flop"));
        assert_eq!(serial_correction.terminal_flop.unwrap().weight, 0.25);
        assert!(serial_correction
            .preflop_responses
            .iter()
            .all(Vec::is_empty));
        assert!(serial_correction.postflop_only_response);
        assert!(first.network_sha256.is_empty());
        assert_eq!(first.policy_sha256, sha256_file(&path).unwrap());
        assert_eq!(first.source_policy_coverage.len(), 3);
        assert!(first.policy_source_kind.contains("uniform_completion"));
        assert_eq!(
            first.total_response_gain_bb_per_hand,
            first
                .players
                .iter()
                .map(|p| p.estimated_gain_bb)
                .sum::<f64>()
        );
        assert_eq!(
            first.total_response_gain_lower_confidence_bound_99_percent_bb_per_hand,
            2.0 * first.approximate_exploitability_lower_confidence_bound_99_percent_bb_per_hand
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn range_aggregate_selects_one_action_for_all_particles() {
        let descriptor = NodeDescriptor {
            actor: Position::BigBlind,
            street: Street::Flop,
            hand_bucket_trajectory: vec!["preflop:AKs".into(), "pair".into()].into(),
            public_bucket_trajectory: vec!["dry".into()].into(),
            public_history_id: 1,
            pot_bb: 4.0,
            to_call_bb: 1.0,
            effective_stack_remaining_bb: 18.0,
        };
        let actions = vec![
            LegalAction {
                label: "fold".to_owned(),
                kind: ActionKind::Fold,
            },
            LegalAction {
                label: "call".to_owned(),
                kind: ActionKind::Call,
            },
        ];
        let mut aggregate =
            DecisionAccumulator::new(&descriptor, vec!["history".to_owned()], &actions);
        aggregate.add(&[0.0, 2.0]);
        aggregate.add(&[3.0, 0.0]);
        let decision = aggregate.finish(7, ResolverGranularity::ObservableBackoff, 20.0);
        assert_eq!(decision.range_particles, 2);
        assert_eq!(decision.granularity, ResolverGranularity::ObservableBackoff);
        assert_eq!(decision.selected_action, 0);
        assert_eq!(decision.action_values_bb, vec![1.5, 1.0]);
        assert_eq!(decision.selected_action_mean_gap_bb, 0.5);
        assert!(decision.low_confidence);
    }

    #[test]
    fn derived_response_seeds_are_deterministic_and_separated() {
        assert_eq!(derived_seed(1, 2, 3), derived_seed(1, 2, 3));
        assert_ne!(derived_seed(1, 2, 3), derived_seed(1, 2, 4));
    }

    #[test]
    fn response_deployment_requires_a_strict_finite_calibration_bound() {
        assert!(response_lower_bound_passes_calibration(1e-9));
        assert!(!response_lower_bound_passes_calibration(0.0));
        assert!(!response_lower_bound_passes_calibration(-1e-9));
        assert!(!response_lower_bound_passes_calibration(f64::NAN));
    }

    #[test]
    fn observable_backoff_forgets_public_trajectory_without_observing_opponent_cards() {
        let mut game = BlueprintConfig::default();
        game.effective_stack_bb = 20.0;
        let first = Deal::from_cards([[51, 47], [0, 5]], [10, 15, 20, 25, 30]);
        let second = Deal::from_cards([[51, 47], [1, 6]], [10, 15, 20, 25, 30]);
        let state = GameState::initial(&game);
        let actions = state.legal_actions(&game);
        let (first_key, _, _) = observable_backoff_information_set(&state, &first, &game, &actions);
        let (second_key, _, _) =
            observable_backoff_information_set(&state, &second, &game, &actions);
        assert_eq!(first_key, second_key);
        let (first_strategic_key, _, _, _) =
            strategic_observable_backoff_information_set(&state, &first, &game, &actions);
        let (second_strategic_key, _, _, _) =
            strategic_observable_backoff_information_set(&state, &second, &game, &actions);
        assert_eq!(first_strategic_key, second_strategic_key);

        let mut forgotten_history = state.clone();
        forgotten_history
            .public_history
            .push("public-but-forgotten-marker".to_owned());
        let exact_before = information_set(&state, &first, &game).0;
        let exact_after = information_set(&forgotten_history, &first, &game).0;
        let backoff_after =
            observable_backoff_information_set(&forgotten_history, &first, &game, &actions).0;
        assert_ne!(exact_before, exact_after);
        assert_eq!(first_key, backoff_after);

        let call = actions
            .iter()
            .find(|action| action.kind == ActionKind::Call)
            .unwrap();
        let after_call = state.apply(call, &game);
        let checks = after_call.legal_actions(&game);
        let check = checks
            .iter()
            .find(|action| action.kind == ActionKind::Check)
            .unwrap();
        let flop = after_call.apply(check, &game);
        assert_eq!(flop.street, Street::Flop);
        let flop_actions = flop.legal_actions(&game);
        let (_, coarse, _) =
            coarse_observable_backoff_information_set(&flop, &first, &game, &flop_actions);
        let coarse_hand = coarse.hand_bucket_trajectory.last().unwrap();
        assert!(!coarse_hand.contains("eq"));
        assert!(!coarse_hand.contains("pot"));

        let connected = Deal::from_cards([[1, 6], [51, 47]], [0, 5, 10, 15, 20]);
        let disconnected = Deal::from_cards([[1, 6], [51, 47]], [0, 21, 38, 15, 20]);
        let connected_coarse =
            coarse_observable_backoff_information_set(&flop, &connected, &game, &flop_actions).0;
        let disconnected_coarse =
            coarse_observable_backoff_information_set(&flop, &disconnected, &game, &flop_actions).0;
        assert_ne!(connected_coarse, disconnected_coarse);
        let connected_strategic =
            strategic_observable_backoff_information_set(&flop, &connected, &game, &flop_actions).0;
        let disconnected_strategic = strategic_observable_backoff_information_set(
            &flop,
            &disconnected,
            &game,
            &flop_actions,
        )
        .0;
        assert_eq!(connected_strategic, disconnected_strategic);
    }

    #[test]
    fn strategic_action_shape_forgets_absolute_bet_amounts() {
        let mut game = BlueprintConfig::default();
        game.effective_stack_bb = 20.0;
        let first_state = GameState::initial(&game);
        let first_actions = vec![
            LegalAction {
                label: "fold".to_owned(),
                kind: ActionKind::Fold,
            },
            LegalAction {
                label: "call".to_owned(),
                kind: ActionKind::Call,
            },
            LegalAction {
                label: "raise_to_2.000bb".to_owned(),
                kind: ActionKind::RaiseTo(2.0),
            },
            LegalAction {
                label: "raise_to_3.000bb".to_owned(),
                kind: ActionKind::RaiseTo(3.0),
            },
            LegalAction {
                label: "raise_all_in_to_20.000bb".to_owned(),
                kind: ActionKind::RaiseTo(20.0),
            },
        ];
        let mut second_state = first_state.clone();
        second_state.invested = [1.0, 2.0];
        second_state.street_invested = [1.0, 2.0];
        let second_actions = vec![
            LegalAction {
                label: "fold".to_owned(),
                kind: ActionKind::Fold,
            },
            LegalAction {
                label: "call".to_owned(),
                kind: ActionKind::Call,
            },
            LegalAction {
                label: "raise_to_4.000bb".to_owned(),
                kind: ActionKind::RaiseTo(4.0),
            },
            LegalAction {
                label: "raise_to_6.000bb".to_owned(),
                kind: ActionKind::RaiseTo(6.0),
            },
            LegalAction {
                label: "raise_all_in_to_20.000bb".to_owned(),
                kind: ActionKind::RaiseTo(20.0),
            },
        ];
        let expected = vec!["fold", "call", "raise_1", "raise_2", "all_in"];
        assert_eq!(
            strategic_action_labels(&first_state, &first_actions, &game),
            expected
        );
        assert_eq!(
            strategic_action_labels(&second_state, &second_actions, &game),
            expected
        );
    }
}
