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

const RESPONSE_SCHEMA: &str = "hu-full-game-information-set-lbr-v1";
const RESOLVER_SCHEMA: &str = "hu-range-conditioned-postflop-resolver-v1";

#[derive(Clone, Debug)]
pub struct ResponseEvaluationConfig {
    pub game: BlueprintConfig,
    pub training_deals: u64,
    pub evaluation_deals: u64,
    pub rollouts_per_action: u32,
    pub minimum_range_particles: u64,
    pub seed: u64,
    pub network_path: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ResolverDecision {
    pub information_set: u64,
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
    pub baseline_utility_bb: f64,
    pub response_utility_bb: f64,
    pub estimated_gain_bb: f64,
    pub gain_standard_error_bb: f64,
    pub approximate_one_sided_99_5_percent_gain_lower_bound_bb: f64,
    pub resolver_lookup_coverage: f64,
    pub preflop_lookup_coverage: f64,
    pub postflop_lookup_coverage: f64,
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
    pub evaluation_deals: u64,
    pub rollouts_per_action: u32,
    pub minimum_range_particles: u64,
    pub players: [ResponsePlayerEvaluation; 2],
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
            hand_bucket_trajectory: descriptor.hand_bucket_trajectory.clone(),
            public_bucket_trajectory: descriptor.public_bucket_trajectory.clone(),
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

    fn finish(self, key: u64, unevaluated_standard_error_bb: f64) -> ResolverDecision {
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
}

pub fn evaluate_full_game_response(
    config: ResponseEvaluationConfig,
) -> Result<FullGameResponseEvaluation, Box<dyn Error>> {
    config.game.validate()?;
    if config.training_deals == 0
        || config.evaluation_deals < 2
        || config.rollouts_per_action < 2
        || config.minimum_range_particles < 2
    {
        return Err(
            "response evaluation requires training deals and at least two evaluation deals, rollouts, and range particles"
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
    let players = [
        evaluate_resolver(&policy, &preflop_responses[0], &resolvers[0], &config, 0),
        evaluate_resolver(&policy, &preflop_responses[1], &resolvers[1], &config, 1),
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
        method: "one_step_common_random_full_game_rollout_response_grouped_by_abstract_information_set"
            .to_owned(),
        depth_bb: config.game.effective_stack_bb,
        network_sha256,
        seed: config.seed,
        training_deals: config.training_deals,
        evaluation_deals: config.evaluation_deals,
        rollouts_per_action: config.rollouts_per_action,
        minimum_range_particles: config.minimum_range_particles,
        players,
        approximate_exploitability_lower_bound_bb_per_hand: lower_bound,
        approximate_exploitability_lower_confidence_bound_99_percent_bb_per_hand:
            confidence_lower_bound,
        interpretation: "an independently evaluated fixed legal imperfect-information learned response; its expected gain is a lower bound on exploitability, but the reported confidence bound uses a normal approximation and is not a release upper-bound certificate"
            .to_owned(),
        preflop_responses,
        resolvers,
    })
}

fn train_learned_response(
    policy: &FrozenPolicy,
    config: &ResponseEvaluationConfig,
    responder: usize,
) -> (Vec<ResolverDecision>, RangeConditionedResolver) {
    let mut chance = SplitMix64::new(derived_seed(config.seed, responder as u64, 0));
    let mut accumulators = BTreeMap::<u64, DecisionAccumulator>::new();
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
            &mut accumulators,
        );
    }
    let decisions = accumulators
        .into_iter()
        .filter(|(_, accumulator)| accumulator.count >= config.minimum_range_particles)
        .map(|(key, accumulator)| accumulator.finish(key, config.game.effective_stack_bb))
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
    accumulators: &mut BTreeMap<u64, DecisionAccumulator>,
) {
    if state.terminal.is_some() {
        return;
    }
    let actions = state.legal_actions(game);
    if state.actor == responder {
        let (key, descriptor, history) = information_set(&state, deal, game);
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
        let accumulator = accumulators
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
        accumulators,
    );
}

fn evaluate_resolver(
    policy: &FrozenPolicy,
    preflop: &[ResolverDecision],
    resolver: &RangeConditionedResolver,
    config: &ResponseEvaluationConfig,
    responder: usize,
) -> ResponsePlayerEvaluation {
    let decisions = preflop
        .iter()
        .chain(&resolver.decisions)
        .filter(|decision| !decision.low_confidence)
        .map(|decision| (decision.information_set, decision))
        .collect::<BTreeMap<_, _>>();
    let mut chance = SplitMix64::new(derived_seed(config.seed, responder as u64, u64::MAX));
    let mut differences = Vec::with_capacity(config.evaluation_deals as usize);
    let mut baseline_total = 0.0;
    let mut response_total = 0.0;
    let mut lookup = ResolverLookup::default();
    for deal_index in 0..config.evaluation_deals {
        let deal = Deal::sample(&mut chance);
        let mut baseline_rng = SplitMix64::new(derived_seed(config.seed, deal_index, 11));
        let mut response_rng = SplitMix64::new(derived_seed(config.seed, deal_index, 11));
        let baseline_p0 = baseline_rollout(
            policy,
            GameState::initial(&config.game),
            &deal,
            &config.game,
            &mut baseline_rng,
        );
        let response_p0 = response_rollout(
            policy,
            &decisions,
            GameState::initial(&config.game),
            &deal,
            &config.game,
            responder,
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
        preflop_lookup_coverage: ratio(lookup.preflop_hits, lookup.preflop_queries),
        postflop_lookup_coverage: ratio(lookup.postflop_hits, lookup.postflop_queries),
        learned_information_sets: preflop.len() + resolver.decisions.len(),
        confident_information_sets: decisions.len(),
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
    decisions: &BTreeMap<u64, &ResolverDecision>,
    state: GameState,
    deal: &Deal,
    game: &BlueprintConfig,
    responder: usize,
    rng: &mut SplitMix64,
    lookup: &mut ResolverLookup,
) -> f64 {
    if state.terminal.is_some() {
        return realized_utility_p0(&state, deal);
    }
    let actions = state.legal_actions(game);
    let selected = if state.actor == responder {
        lookup.queries += 1;
        if state.street == Street::Preflop {
            lookup.preflop_queries += 1;
        } else {
            lookup.postflop_queries += 1;
        }
        let (key, _, _) = information_set(&state, deal, game);
        match decisions.get(&key) {
            Some(decision)
                if decision.action_labels
                    == actions
                        .iter()
                        .map(|action| action.label.clone())
                        .collect::<Vec<_>>() =>
            {
                lookup.hits += 1;
                if state.street == Street::Preflop {
                    lookup.preflop_hits += 1;
                } else {
                    lookup.postflop_hits += 1;
                }
                decision.selected_action
            }
            _ => {
                let strategy = policy.strategy(&state, deal, &actions, game);
                sample_index(&strategy, rng)
            }
        }
    } else {
        let strategy = policy.strategy(&state, deal, &actions, game);
        sample_index(&strategy, rng)
    };
    response_rollout(
        policy,
        decisions,
        state.apply(&actions[selected], game),
        deal,
        game,
        responder,
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
            hand_bucket_trajectory: vec!["preflop:AKs".to_owned(), "pair".to_owned()],
            public_bucket_trajectory: vec!["dry".to_owned()],
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
        let decision = aggregate.finish(7, 20.0);
        assert_eq!(decision.range_particles, 2);
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
}
