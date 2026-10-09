//! Full-hand smoke evaluation, never an unrestricted exploitability certificate.
//!
//! Fixed deviations inspect only the public betting state. The exact same
//! uniformly dealt cards and per-decision random numbers are used for the
//! profile and each deviation. This makes accounting and matched candidate
//! comparisons inexpensive without leaking future boards or hidden cards.
use super::*;
use std::time::Instant;

pub struct CashFullHandEvaluationConfig {
    pub game: BlueprintConfig,
    pub network_path: PathBuf,
    pub comparison_path: Option<PathBuf>,
    pub deals: u64,
    pub seed: u64,
    pub threads: usize,
    pub confidence: f64,
}

#[derive(Clone, Copy, Debug)]
enum FixedResponse {
    CheckCall,
    FoldToWager,
    MaximumPressure,
    MinimumProbe,
}
const RESPONSES: [FixedResponse; 4] = [
    FixedResponse::CheckCall,
    FixedResponse::FoldToWager,
    FixedResponse::MaximumPressure,
    FixedResponse::MinimumProbe,
];
impl FixedResponse {
    fn label(self) -> &'static str {
        match self {
            Self::CheckCall => "check-call",
            Self::FoldToWager => "fold-to-wager",
            Self::MaximumPressure => "maximum-pressure",
            Self::MinimumProbe => "minimum-probe",
        }
    }
}

/// No Deal argument: a fixed response cannot read opponent cards or a runout.
fn response_action(response: FixedResponse, actions: &[LegalAction]) -> usize {
    let passive = || {
        actions
            .iter()
            .position(|a| matches!(a.kind, ActionKind::Check | ActionKind::Call))
            .unwrap_or(0)
    };
    match response {
        FixedResponse::CheckCall => passive(),
        FixedResponse::FoldToWager => actions
            .iter()
            .position(|a| matches!(a.kind, ActionKind::Check | ActionKind::Fold))
            .unwrap_or_else(passive),
        FixedResponse::MaximumPressure => actions
            .iter()
            .rposition(|a| matches!(a.kind, ActionKind::RaiseTo(_)))
            .unwrap_or_else(passive),
        FixedResponse::MinimumProbe => actions
            .iter()
            .position(|a| matches!(a.kind, ActionKind::RaiseTo(_)))
            .unwrap_or_else(passive),
    }
}

#[derive(Clone, Default, Serialize)]
struct PolicyDiagnostics {
    decisions_by_street: [u64; 4],
    valid_probability_rows: u64,
    near_uniform_rows: u64,
    comparison_rows: u64,
    comparison_action_count: u64,
    comparison_absolute_action_delta_sum: f64,
    comparison_primary_agreements: u64,
    #[serde(skip)]
    aggregate_action_delta: BTreeMap<String, f64>,
}
impl PolicyDiagnostics {
    fn merge(&mut self, other: Self) {
        for i in 0..4 {
            self.decisions_by_street[i] += other.decisions_by_street[i];
        }
        self.valid_probability_rows += other.valid_probability_rows;
        self.near_uniform_rows += other.near_uniform_rows;
        self.comparison_rows += other.comparison_rows;
        self.comparison_action_count += other.comparison_action_count;
        self.comparison_absolute_action_delta_sum += other.comparison_absolute_action_delta_sum;
        self.comparison_primary_agreements += other.comparison_primary_agreements;
        for (label, value) in other.aggregate_action_delta {
            *self.aggregate_action_delta.entry(label).or_default() += value;
        }
    }
}

struct HandOutcome {
    payoffs: [f64; 2],
    house_rake: f64,
    accounting_residual: f64,
    diagnostics: PolicyDiagnostics,
}
pub(super) fn checked_strategy(
    policy: &FrozenPolicy,
    state: &GameState,
    deal: &Deal,
    actions: &[LegalAction],
    game: &BlueprintConfig,
) -> Result<Vec<f64>, String> {
    let probabilities = policy.strategy(state, deal, actions, game);
    if probabilities.len() != actions.len()
        || probabilities
            .iter()
            .any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
        || (probabilities.iter().sum::<f64>() - 1.0).abs() > 1e-9
    {
        return Err("cash policy returned invalid legal-action probabilities".into());
    }
    Ok(probabilities)
}
fn primary_action(probabilities: &[f64]) -> usize {
    // First action wins ties, consistently with the comparison row.
    (0..probabilities.len())
        .max_by(|a, b| {
            probabilities[*a]
                .total_cmp(&probabilities[*b])
                .then_with(|| b.cmp(a))
        })
        .unwrap()
}

fn play_hand(
    policy: &FrozenPolicy,
    comparison: Option<&FrozenPolicy>,
    game: &BlueprintConfig,
    deal: &Deal,
    seed: u64,
    deviation: Option<(usize, FixedResponse)>,
) -> Result<HandOutcome, String> {
    let mut state = GameState::initial(game);
    let mut rng = SplitMix64::new(seed);
    let mut diagnostics = PolicyDiagnostics::default();
    for _ in 0..128 {
        if state.terminal.is_some() {
            let payoffs = state.complete_runout_utilities(deal, game);
            let rules = game
                .cash_rules
                .as_ref()
                .ok_or("cash evaluator requires rules")?;
            // Independent of the paired utility routine: refund unmatched
            // wagers, then calculate the actual terminal house deduction.
            let gross = 2 * cash::cash_units(state.invested[0].min(state.invested[1]), rules)?;
            let flop_dealt =
                state.street != Street::Preflop || state.terminal == Some(Terminal::Showdown);
            let house_rake =
                rules.rake_units(gross, flop_dealt)? as f64 / rules.units_per_bb as f64;
            let accounting_residual = (payoffs[0] + payoffs[1] + house_rake).abs();
            if accounting_residual > 1e-8 {
                return Err("cash full-hand accounting failed".into());
            }
            return Ok(HandOutcome {
                payoffs,
                house_rake,
                accounting_residual,
                diagnostics,
            });
        }
        let actions = state.legal_actions(game);
        if actions.is_empty() {
            return Err("nonterminal cash hand has no legal action".into());
        }
        let uniform = rng.next_f64();
        let choice = if let Some((responder, response)) =
            deviation.filter(|(actor, _)| *actor == state.actor)
        {
            debug_assert_eq!(responder, state.actor);
            response_action(response, &actions)
        } else {
            let probabilities = checked_strategy(policy, &state, deal, &actions, game)?;
            if deviation.is_none() {
                let street = match state.street {
                    Street::Preflop => 0,
                    Street::Flop => 1,
                    Street::Turn => 2,
                    Street::River => 3,
                };
                diagnostics.decisions_by_street[street] += 1;
                diagnostics.valid_probability_rows += 1;
                if actions.len() > 1
                    && probabilities
                        .iter()
                        .all(|p| (*p - 1.0 / actions.len() as f64).abs() <= 0.005)
                {
                    diagnostics.near_uniform_rows += 1;
                }
                if let Some(other) = comparison {
                    let alternate = checked_strategy(other, &state, deal, &actions, game)?;
                    diagnostics.comparison_rows += 1;
                    diagnostics.comparison_action_count += actions.len() as u64;
                    diagnostics.comparison_primary_agreements +=
                        u64::from(primary_action(&probabilities) == primary_action(&alternate));
                    for ((action, a), b) in actions.iter().zip(&probabilities).zip(&alternate) {
                        diagnostics.comparison_absolute_action_delta_sum += (a - b).abs();
                        *diagnostics
                            .aggregate_action_delta
                            .entry(action.label.clone())
                            .or_default() += a - b;
                    }
                }
            }
            let mut cumulative = 0.0;
            let mut chosen = actions.len() - 1;
            for (index, probability) in probabilities.iter().enumerate() {
                cumulative += probability;
                if uniform < cumulative {
                    chosen = index;
                    break;
                }
            }
            chosen
        };
        state = state.apply(&actions[choice], game);
    }
    Err(
        "cash full-hand trajectory exceeded its decision bound; no interrupted hand is scored"
            .into(),
    )
}

#[derive(Serialize)]
pub struct CashFullHandEvaluation {
    schema: &'static str,
    validation_status: &'static str,
    game: BlueprintConfig,
    rules_sha256: String,
    policy_sha256: String,
    comparison_sha256: Option<String>,
    deals: u64,
    seed: u64,
    threads: usize,
    elapsed_seconds: f64,
    baseline: OutcomeSummary,
    comparison_baseline: Option<OutcomeSummary>,
    restricted_responses: Vec<DeviationSummary>,
    restricted_gain_estimate_sum_bb: f64,
    confidence: f64,
    diagnostics: PolicyDiagnostics,
    comparison_action_frequency_mae: Option<f64>,
    comparison_primary_agreement: Option<f64>,
    comparison_aggregate_action_deltas: BTreeMap<String, f64>,
    limitations: Vec<&'static str>,
}
#[derive(Default, Serialize)]
struct OutcomeSummary {
    mean_own_payoff_bb: [f64; 2],
    mean_house_rake_bb: f64,
    maximum_accounting_residual_bb: f64,
}
impl OutcomeSummary {
    fn add(&mut self, outcome: &HandOutcome, count: f64) {
        for i in 0..2 {
            self.mean_own_payoff_bb[i] += outcome.payoffs[i] / count;
        }
        self.mean_house_rake_bb += outcome.house_rake / count;
        self.maximum_accounting_residual_bb = self
            .maximum_accounting_residual_bb
            .max(outcome.accounting_residual);
    }
}
#[derive(Serialize)]
struct DeviationSummary {
    seat: usize,
    response: &'static str,
    mean_gain_bb: f64,
    paired_sampling_standard_error_bb: f64,
    simultaneous_sampling_margin_bb: f64,
    mean_response_own_payoff_bb: f64,
    mean_response_house_rake_bb: f64,
    maximum_accounting_residual_bb: f64,
}
struct EvaluatedDeal {
    baseline: HandOutcome,
    comparison: Option<HandOutcome>,
    deviations: Vec<HandOutcome>,
}

pub fn evaluate_cash_full_hands(
    config: CashFullHandEvaluationConfig,
) -> Result<CashFullHandEvaluation, Box<dyn Error>> {
    let started = Instant::now();
    config.game.validate()?;
    let rules = config
        .game
        .cash_rules
        .as_ref()
        .ok_or("cash full-hand evaluation requires explicit cash rules")?;
    if !(2..=65_536).contains(&config.deals)
        || !(1..=16).contains(&config.threads)
        || !config.confidence.is_finite()
        || !(0.0..1.0).contains(&config.confidence)
    {
        return Err("cash evaluation requires 2..65536 deals, 1..16 threads and confidence strictly between 0 and 1".into());
    }
    let policy = FrozenPolicy::load(&config.network_path)?;
    policy.bundle.validate_game(&config.game)?;
    if !matches!(policy.bundle.strategy_transform, StrategyTransform::Softmax) {
        return Err("cash full-hand evaluation requires frozen average-policy weights, not regret/current strategy".into());
    }
    let comparison = config
        .comparison_path
        .as_ref()
        .map(|path| FrozenPolicy::load(path))
        .transpose()?;
    if let Some(other) = &comparison {
        other.bundle.validate_game(&config.game)?;
        if !matches!(other.bundle.strategy_transform, StrategyTransform::Softmax) {
            return Err("cash comparison requires frozen average policy".into());
        }
    }
    let mut rng = SplitMix64::new(config.seed);
    let deals = (0..config.deals)
        .map(|i| (i, Deal::sample(&mut rng)))
        .collect::<Vec<_>>();
    let workers = config.threads.min(deals.len());
    let chunks = deals
        .chunks(deals.len().div_ceil(workers))
        .map(<[(u64, Deal)]>::to_vec)
        .collect::<Vec<_>>();
    let evaluated = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for chunk in chunks {
            let policy = policy.fork_with_fresh_caches();
            let other = comparison
                .as_ref()
                .map(FrozenPolicy::fork_with_fresh_caches);
            let game = &config.game;
            handles.push(scope.spawn(move || -> Result<Vec<EvaluatedDeal>, String> {
                chunk
                    .into_iter()
                    .map(|(index, deal)| {
                        let seed = config.seed
                            ^ index.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                            ^ 0xa24b_aed4_963e_e407;
                        let baseline = play_hand(&policy, other.as_ref(), game, &deal, seed, None)?;
                        let alternate = other
                            .as_ref()
                            .map(|other| play_hand(other, None, game, &deal, seed, None))
                            .transpose()?;
                        let mut deviations = Vec::new();
                        for seat in 0..2 {
                            for response in RESPONSES {
                                deviations.push(play_hand(
                                    &policy,
                                    None,
                                    game,
                                    &deal,
                                    seed,
                                    Some((seat, response)),
                                )?);
                            }
                        }
                        Ok(EvaluatedDeal {
                            baseline,
                            comparison: alternate,
                            deviations,
                        })
                    })
                    .collect()
            }));
        }
        let mut values = Vec::new();
        for handle in handles {
            values.extend(
                handle
                    .join()
                    .map_err(|_| "cash evaluation worker panicked".to_owned())??,
            );
        }
        Ok::<_, String>(values)
    })?;
    let count = config.deals as f64;
    let mut baseline = OutcomeSummary::default();
    let mut alternate = comparison.as_ref().map(|_| OutcomeSummary::default());
    let mut diagnostics = PolicyDiagnostics::default();
    let mut gains = [0.0; 8];
    let mut m2 = [0.0; 8];
    let mut response_outcomes: [OutcomeSummary; 8] =
        std::array::from_fn(|_| OutcomeSummary::default());
    for (index, outcome) in evaluated.into_iter().enumerate() {
        baseline.add(&outcome.baseline, count);
        if let (Some(summary), Some(other)) = (&mut alternate, &outcome.comparison) {
            summary.add(other, count);
        }
        for (response_index, deviation) in outcome.deviations.iter().enumerate() {
            let seat = response_index / 4;
            let gain = deviation.payoffs[seat] - outcome.baseline.payoffs[seat];
            let delta = gain - gains[response_index];
            gains[response_index] += delta / (index + 1) as f64;
            m2[response_index] += delta * (gain - gains[response_index]);
            response_outcomes[response_index].add(deviation, count);
        }
        diagnostics.merge(outcome.baseline.diagnostics);
    }
    // Bonferroni over eight fixed deviations; each paired difference lies in
    // [-2*stack,2*stack]. This interval covers sampling only, not response search.
    let margin = 4.0
        * config.game.effective_stack_bb
        * ((8.0 / (1.0 - config.confidence)).ln() / (2.0 * count)).sqrt();
    let responses = (0..8)
        .map(|i| DeviationSummary {
            seat: i / 4,
            response: RESPONSES[i % 4].label(),
            mean_gain_bb: gains[i],
            paired_sampling_standard_error_bb: (m2[i].max(0.0) / (count - 1.0) / count).sqrt(),
            simultaneous_sampling_margin_bb: margin,
            mean_response_own_payoff_bb: response_outcomes[i].mean_own_payoff_bb[i / 4],
            mean_response_house_rake_bb: response_outcomes[i].mean_house_rake_bb,
            maximum_accounting_residual_bb: response_outcomes[i].maximum_accounting_residual_bb,
        })
        .collect();
    let comparison_action_frequency_mae = (diagnostics.comparison_action_count > 0).then(|| {
        diagnostics.comparison_absolute_action_delta_sum
            / diagnostics.comparison_action_count as f64
    });
    let comparison_primary_agreement = (diagnostics.comparison_rows > 0).then(|| {
        diagnostics.comparison_primary_agreements as f64 / diagnostics.comparison_rows as f64
    });
    let aggregate = diagnostics
        .aggregate_action_delta
        .iter()
        .map(|(label, delta)| (label.clone(), delta / diagnostics.comparison_rows as f64))
        .collect();
    Ok(CashFullHandEvaluation { schema: "hu-cash-full-hand-restricted-evaluation-v1", validation_status: "research_only",
        rules_sha256: rules.sha256()?, policy_sha256: policy.bundle_sha256, comparison_sha256: comparison.map(|p| p.bundle_sha256),
        deals: config.deals, seed: config.seed, threads: workers, elapsed_seconds: started.elapsed().as_secs_f64(),
        baseline, comparison_baseline: alternate, restricted_responses: responses,
        restricted_gain_estimate_sum_bb: gains[..4].iter().copied().fold(0.0, f64::max) + gains[4..].iter().copied().fold(0.0, f64::max),
        confidence: config.confidence, diagnostics, comparison_action_frequency_mae, comparison_primary_agreement,
        comparison_aggregate_action_deltas: aggregate, game: config.game,
        limitations: vec!["Fixed public-state-only deviations are information-set consistent but are not a best-response search",
            "Restricted gains and sampling intervals cannot certify an exploitability upper bound or a release gate",
            "Uniform legal complete deals; probabilities are observed on the candidate's authentic trajectories",
            "Valid inference rows are not evidence of training support or 99.99% held-out serving coverage",
            "No range-conditioned resolver, action-EV confidence or browser-serving qualification"] })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn policy(game: &BlueprintConfig) -> FrozenPolicy {
        let scorer = DenseScorer {
            layers: vec![DenseLayer {
                input_size: CASH_MODEL_INPUT_COUNT,
                output_size: 1,
                activation: DenseActivation::Linear,
                weights: vec![0.0; CASH_MODEL_INPUT_COUNT],
                biases: vec![0.0],
            }],
        };
        let bundle = TrainingNetworkBundle {
            schema: CASH_TRAINING_NETWORK_SCHEMA.into(),
            input_size: CASH_MODEL_INPUT_COUNT,
            strategy_transform: StrategyTransform::Softmax,
            networks: vec![scorer.clone(), scorer],
            postflop_networks: None,
            sampling_baseline: None,
            sampling_baseline_scale: None,
            cash_rules: game.cash_rules.clone(),
            cash_depth_bb: Some(game.effective_stack_bb),
            cash_action_abstraction: Some(game.action_abstraction.clone()),
        };
        FrozenPolicy::from_bundle(bundle, "test-only".into()).unwrap()
    }
    fn game() -> BlueprintConfig {
        let mut game = BlueprintConfig::default();
        game.effective_stack_bb = 20.0;
        game.small_blind_bb = 0.4;
        game.cash_rules = Some(crate::cash_game::study_rules("nl25").unwrap());
        game.action_abstraction.open_sizes_bb = vec![2.0];
        game
    }
    #[test]
    fn complete_hands_and_every_fixed_deviation_conserve_house_rake() {
        let game = game();
        let policy = policy(&game);
        let mut rng = SplitMix64::new(51);
        let mut showdown = false;
        for seed in 0..40 {
            let deal = Deal::sample(&mut rng);
            for response in std::iter::once(None)
                .chain((0..2).flat_map(|seat| RESPONSES.map(|r| Some((seat, r)))))
            {
                let result =
                    play_hand(&policy, Some(&policy), &game, &deal, seed, response).unwrap();
                assert!(result.accounting_residual < 1e-10);
                assert!((result.payoffs[0] + result.payoffs[1] + result.house_rake).abs() < 1e-10);
                showdown |= result.house_rake > 0.0;
                if response.is_none() {
                    assert_eq!(
                        result.diagnostics.comparison_primary_agreements,
                        result.diagnostics.comparison_rows
                    );
                }
            }
        }
        assert!(showdown);
    }
    #[test]
    fn fixed_deviations_cannot_read_hidden_cards_or_future_boards() {
        let game = game();
        let actions = GameState::initial(&game).legal_actions(&game);
        assert_eq!(response_action(FixedResponse::CheckCall, &actions), 1);
        assert!(matches!(
            actions[response_action(FixedResponse::FoldToWager, &actions)].kind,
            ActionKind::Fold
        ));
        let min = response_action(FixedResponse::MinimumProbe, &actions);
        let max = response_action(FixedResponse::MaximumPressure, &actions);
        assert!(min < max);
        // Observation-only responses have no Deal parameter, by construction.
    }
}
