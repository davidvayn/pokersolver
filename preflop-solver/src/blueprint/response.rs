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
    pub network_path: PathBuf,
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
    pub low_confidence: bool,
    pub range_particles: u64,
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
    pub network_sha256: String,
    pub seed: u64,
    pub training_deals: u64,
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
        }
    }

    fn add(&mut self, values: &[f64]) {
        assert_eq!(values.len(), self.sums.len());
        self.count += 1;
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
                let gap_standard_error = (standard_errors[selected_action].powi(2)
                    + standard_errors[runner_up_index].powi(2))
                .sqrt();
                (gap, gap - 2.575_829_303_548_900_4 * gap_standard_error)
            })
            .unwrap_or((0.0, 0.0));
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

pub fn evaluate_full_game_response(
    config: ResponseEvaluationConfig,
) -> Result<FullGameResponseEvaluation, Box<dyn Error>> {
    config.game.validate()?;
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
    let network_sha256 = sha256_file(&config.network_path)?;
    let policy = FrozenPolicy::load(&config.network_path)?;
    let [first, second] = [
        train_learned_response(&policy, &config, 0),
        train_learned_response(&policy, &config, 1),
    ];
    let preflop_responses = [first.0, second.0];
    let resolvers = [first.1, second.1];
    let calibration_players = [
        evaluate_resolver(
            &policy,
            &preflop_responses[0],
            &resolvers[0],
            &config,
            0,
            config.calibration_deals,
            u64::MAX - 1,
            true,
        ),
        evaluate_resolver(
            &policy,
            &preflop_responses[1],
            &resolvers[1],
            &config,
            1,
            config.calibration_deals,
            u64::MAX - 1,
            true,
        ),
    ];
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
            &policy,
            &preflop_responses[0],
            &resolvers[0],
            &config,
            0,
            config.evaluation_deals,
            u64::MAX,
            response_deployed[0],
        ),
        evaluate_resolver(
            &policy,
            &preflop_responses[1],
            &resolvers[1],
            &config,
            1,
            config.evaluation_deals,
            u64::MAX,
            response_deployed[1],
        ),
    ];
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
        schema: RESPONSE_SCHEMA.to_owned(),
        method: "calibrated_one_step_common_random_full_game_rollout_response_with_exact_fine_coarse_and_strategic_observable_information_sets"
            .to_owned(),
        depth_bb: config.game.effective_stack_bb,
        network_sha256,
        seed: config.seed,
        training_deals: config.training_deals,
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
        interpretation: "a fixed legal imperfect-information learned response is trained, accepted only when a disjoint calibration corpus has a positive one-sided 99.5% gain lower bound, and measured on a third independent corpus; rejected players deploy the frozen baseline with zero claimed gain; expected gain remains a lower bound on exploitability, never a release upper-bound certificate"
            .to_owned(),
        preflop_responses,
        resolvers,
    })
}

fn response_lower_bound_passes_calibration(lower_bound_bb: f64) -> bool {
    lower_bound_bb.is_finite() && lower_bound_bb > 0.0
}

fn train_learned_response(
    policy: &FrozenPolicy,
    config: &ResponseEvaluationConfig,
    responder: usize,
) -> (Vec<ResolverDecision>, RangeConditionedResolver) {
    let mut chance = SplitMix64::new(derived_seed(config.seed, responder as u64, 0));
    let mut exact_accumulators = BTreeMap::<u64, DecisionAccumulator>::new();
    let mut backoff_accumulators = BTreeMap::<u64, DecisionAccumulator>::new();
    let mut coarse_backoff_accumulators = BTreeMap::<u64, DecisionAccumulator>::new();
    let mut strategic_backoff_accumulators = BTreeMap::<u64, DecisionAccumulator>::new();
    for deal_index in 0..config.training_deals {
        let deal = Deal::sample(&mut chance);
        let mut trajectory_rng =
            SplitMix64::new(derived_seed(config.seed, responder as u64, deal_index + 1));
        collect_trajectory_decisions(
            policy,
            GameState::initial(&config.game),
            &deal,
            &config.game,
            responder,
            config.rollouts_per_action,
            config.seed,
            deal_index,
            &mut trajectory_rng,
            &mut exact_accumulators,
            &mut backoff_accumulators,
            &mut coarse_backoff_accumulators,
            &mut strategic_backoff_accumulators,
        );
    }
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

#[allow(clippy::too_many_arguments)]
fn collect_trajectory_decisions(
    policy: &FrozenPolicy,
    state: GameState,
    deal: &Deal,
    game: &BlueprintConfig,
    responder: usize,
    rollouts_per_action: u32,
    response_seed: u64,
    deal_index: u64,
    trajectory_rng: &mut SplitMix64,
    exact_accumulators: &mut BTreeMap<u64, DecisionAccumulator>,
    backoff_accumulators: &mut BTreeMap<u64, DecisionAccumulator>,
    coarse_backoff_accumulators: &mut BTreeMap<u64, DecisionAccumulator>,
    strategic_backoff_accumulators: &mut BTreeMap<u64, DecisionAccumulator>,
) {
    if state.terminal.is_some() {
        return;
    }
    let actions = state.legal_actions(game);
    if state.actor == responder {
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
                (0..rollouts_per_action)
                    .map(|rollout| {
                        let mut rng = SplitMix64::new(derived_seed(
                            response_seed ^ ((responder as u64 + 1) << 61),
                            deal_index ^ key,
                            rollout as u64,
                        ));
                        let utility = baseline_rollout(
                            policy,
                            state.apply(action, game),
                            deal,
                            game,
                            &mut rng,
                        );
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
        let accumulator = exact_accumulators
            .entry(key)
            .or_insert_with(|| DecisionAccumulator::new(&descriptor, history, &actions));
        assert_eq!(
            accumulator.action_labels,
            actions
                .iter()
                .map(|action| action.label.clone())
                .collect::<Vec<_>>()
        );
        accumulator.add(&values);
        let backoff_accumulator = backoff_accumulators.entry(backoff_key).or_insert_with(|| {
            DecisionAccumulator::new(&backoff_descriptor, backoff_history, &actions)
        });
        assert_eq!(
            backoff_accumulator.action_labels,
            actions
                .iter()
                .map(|action| action.label.clone())
                .collect::<Vec<_>>()
        );
        backoff_accumulator.add(&values);
        let coarse_accumulator = coarse_backoff_accumulators
            .entry(coarse_key)
            .or_insert_with(|| {
                DecisionAccumulator::new(&coarse_descriptor, coarse_history, &actions)
            });
        assert_eq!(
            coarse_accumulator.action_labels,
            actions
                .iter()
                .map(|action| action.label.clone())
                .collect::<Vec<_>>()
        );
        coarse_accumulator.add(&values);
        let strategic_accumulator = strategic_backoff_accumulators
            .entry(strategic_key)
            .or_insert_with(|| {
                let mut accumulator =
                    DecisionAccumulator::new(&strategic_descriptor, strategic_history, &actions);
                accumulator.action_labels = strategic_labels.clone();
                accumulator
            });
        assert_eq!(strategic_accumulator.action_labels, strategic_labels);
        strategic_accumulator.add(&values);
    }
    let strategy = policy.strategy(&state, deal, &actions, game);
    let selected = sample_index(&strategy, trajectory_rng);
    collect_trajectory_decisions(
        policy,
        state.apply(&actions[selected], game),
        deal,
        game,
        responder,
        rollouts_per_action,
        response_seed,
        deal_index,
        trajectory_rng,
        exact_accumulators,
        backoff_accumulators,
        coarse_backoff_accumulators,
        strategic_backoff_accumulators,
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
    policy: &FrozenPolicy,
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
                && !decision.low_confidence
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
                && !decision.low_confidence
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
    for deal_index in 0..deals {
        let deal = Deal::sample(&mut chance);
        let rollout_seed = derived_seed(phase_seed, deal_index, 11);
        let mut baseline_rng = SplitMix64::new(rollout_seed);
        let mut response_rng = SplitMix64::new(rollout_seed);
        let baseline_p0 = baseline_rollout(
            policy,
            GameState::initial(&config.game),
            &deal,
            &config.game,
            &mut baseline_rng,
        );
        let response_p0 = response_rollout(
            policy,
            &exact_decisions,
            &backoff_decisions,
            &coarse_backoff_decisions,
            &strategic_backoff_decisions,
            GameState::initial(&config.game),
            &deal,
            &config.game,
            responder,
            false,
            &mut response_rng,
            &mut lookup,
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
        baseline_total += baseline;
        response_total += response;
        differences.push(response - baseline);
    }
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
    policy: &FrozenPolicy,
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
    policy: &FrozenPolicy,
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
