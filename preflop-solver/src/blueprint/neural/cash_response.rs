//! General-sum adapter for the existing information-set-consistent sample game.
//! Baseline and deviation use each seat's own payoff, never opponent negation.
use super::*;
use std::time::Instant;

pub struct CashCausalResponseConfig {
    pub game: BlueprintConfig,
    pub network_path: PathBuf,
    pub deals: u64,
    pub seed: u64,
    pub threads: usize,
    pub public_branches_per_street: u32,
    pub opponent_samples_per_runout: u32,
    pub node_limit_per_seat: u64,
    pub confidence: f64,
}

#[derive(Clone, Default, Serialize)]
struct JointValue {
    own_payoffs_bb: [f64; 2],
    house_rake_bb: f64,
    terminal_reach: f64,
    maximum_accounting_residual_bb: f64,
}
impl JointValue {
    fn merge(&mut self, other: Self) {
        for i in 0..2 {
            self.own_payoffs_bb[i] += other.own_payoffs_bb[i];
        }
        self.house_rake_bb += other.house_rake_bb;
        self.terminal_reach += other.terminal_reach;
        self.maximum_accounting_residual_bb = self
            .maximum_accounting_residual_bb
            .max(other.maximum_accounting_residual_bb);
    }
    fn validate(&self) -> Result<(), String> {
        if self.maximum_accounting_residual_bb > 1e-8
            || (self.terminal_reach - 1.0).abs() > 1e-8
            || (self.own_payoffs_bb.iter().sum::<f64>() + self.house_rake_bb).abs() > 1e-8
        {
            return Err("cash response joint-belief accounting or terminal reach failed".into());
        }
        Ok(())
    }
}

/// Replay an entire frozen policy pair, or a retained response plan, without
/// coupling the two own-player value heads. Both are weighted by the same joint
/// scenario distribution, making the independent house-rake check meaningful.
#[allow(clippy::too_many_arguments)]
fn joint_walk(
    generator: &SampleGenerator,
    state: GameState,
    scenarios: &[Deal],
    weights: &[f64],
    response: Option<(usize, &CausalResponsePlan)>,
    nodes: &mut u64,
    limit: u64,
) -> Result<JointValue, String> {
    if *nodes >= limit {
        return Err(
            "cash baseline/replay node budget exhausted; no partial response is scored".into(),
        );
    }
    *nodes += 1;
    if scenarios.len() != weights.len() {
        return Err("cash response joint scenarios and weights differ".into());
    }
    if weights.iter().all(|w| *w <= 0.0) {
        return Ok(JointValue::default());
    }
    if state.terminal.is_some() {
        let rules = generator
            .config
            .game
            .cash_rules
            .as_ref()
            .ok_or("cash response requires rules")?;
        let gross = 2 * cash::cash_units(state.invested[0].min(state.invested[1]), rules)?;
        let flop = state.street != Street::Preflop || state.terminal == Some(Terminal::Showdown);
        let rake = rules.rake_units(gross, flop)? as f64 / rules.units_per_bb as f64;
        let mut result = JointValue::default();
        for (deal, weight) in scenarios.iter().zip(weights) {
            if *weight <= 0.0 {
                continue;
            }
            let payoffs = state.complete_runout_utilities(deal, &generator.config.game);
            for i in 0..2 {
                result.own_payoffs_bb[i] += weight * payoffs[i];
            }
            result.house_rake_bb += weight * rake;
            result.terminal_reach += weight;
            result.maximum_accounting_residual_bb = result
                .maximum_accounting_residual_bb
                .max((payoffs.iter().sum::<f64>() + rake).abs());
        }
        return Ok(result);
    }
    let actions = state.legal_actions(&generator.config.game);
    let mut branches = vec![vec![0.0; weights.len()]; actions.len()];
    for (index, (deal, reach)) in scenarios.iter().zip(weights).enumerate() {
        if *reach <= 0.0 {
            continue;
        }
        if let Some((actor, plan)) = response.filter(|(actor, _)| *actor == state.actor) {
            debug_assert_eq!(actor, state.actor);
            let key = CausalResponseInformationSet {
                public_history: state.public_history.clone(),
                observed_board: observed_public_board(deal, state.street),
            };
            let action = *plan
                .get(&key)
                .ok_or("cash response plan omitted a reached information set")?;
            if action >= actions.len() {
                return Err("cash response plan has an illegal action".into());
            }
            branches[action][index] = *reach;
        } else {
            let probabilities = generator.current_strategy(&state, deal, &actions);
            if probabilities.len() != actions.len()
                || probabilities.iter().any(|p| !p.is_finite() || *p < 0.0)
                || (probabilities.iter().sum::<f64>() - 1.0).abs() > 1e-9
            {
                return Err("cash response policy probabilities invalid".into());
            }
            for (action, probability) in probabilities.iter().enumerate() {
                branches[action][index] = reach * probability;
            }
        }
    }
    let mut total = JointValue::default();
    for (action, weights) in actions.iter().zip(branches) {
        if weights.iter().all(|w| *w == 0.0) {
            continue;
        }
        total.merge(joint_walk(
            generator,
            state.apply(action, &generator.config.game),
            scenarios,
            &weights,
            response,
            nodes,
            limit,
        )?);
    }
    Ok(total)
}

#[derive(Serialize)]
struct SeatSample {
    baseline: JointValue,
    response: JointValue,
    own_gain_bb: f64,
    searched_nodes: u64,
    replayed_nodes: u64,
}
#[derive(Serialize)]
pub struct CashCausalResponseReport {
    schema: &'static str,
    validation_status: &'static str,
    game: BlueprintConfig,
    rules_sha256: String,
    policy_sha256: String,
    seed: u64,
    deals: u64,
    threads: usize,
    public_branches_per_street: u32,
    opponent_samples_per_runout: u32,
    scenarios_per_seat: u64,
    node_limit_per_seat: u64,
    elapsed_seconds: f64,
    mean_unilateral_gains_bb: [f64; 2],
    sample_mean_total_nash_conv_bb: f64,
    sampling_standard_error_total_bb: f64,
    confidence: f64,
    one_sided_hoeffding_margin_bb: f64,
    abstract_game_optimistic_upper_bound_bb: f64,
    samples: Vec<[SeatSample; 2]>,
    assumptions: Vec<&'static str>,
    limitations: Vec<&'static str>,
}

pub fn evaluate_cash_causal_response(
    config: CashCausalResponseConfig,
) -> Result<CashCausalResponseReport, Box<dyn Error>> {
    let started = Instant::now();
    config.game.validate()?;
    let rules = config
        .game
        .cash_rules
        .as_ref()
        .ok_or("cash response requires explicit cash rules")?;
    let scenario_count = u64::from(config.public_branches_per_street)
        .checked_pow(3)
        .and_then(|n| n.checked_mul(u64::from(config.opponent_samples_per_runout)))
        .ok_or("cash response scenario count overflow")?;
    if !(2..=4096).contains(&config.deals)
        || !(1..=4).contains(&config.threads)
        || !(1..=4096).contains(&scenario_count)
        || !(100..=2_000_000).contains(&config.node_limit_per_seat)
        || !config.confidence.is_finite()
        || !(0.0..1.0).contains(&config.confidence)
    {
        return Err("cash response requires 2..4096 deals, 1..4 threads, 1..4096 scenarios and 100..2M nodes per seat".into());
    }
    let policy = FrozenPolicy::load(&config.network_path)?;
    policy.bundle.validate_game(&config.game)?;
    if !matches!(policy.bundle.strategy_transform, StrategyTransform::Softmax) {
        return Err("cash response requires frozen average-policy weights".into());
    }
    let policy_sha256 = policy.bundle_sha256.clone();
    let generator = SampleGenerator::new(SampleGenerationConfig {
        game: config.game.clone(),
        traversals: 1,
        start_iteration: 0,
        seed: config.seed,
        max_records: 1,
        output: PathBuf::from("unused-cash-response.jsonl.gz"),
        network_path: Some(config.network_path.clone()),
        trajectory_sampling: false,
        evaluate_trajectory_values: false,
        value_rollouts_per_action: 1,
        enumerate_turn_river_chance: false,
    })?;
    let mut rng = SplitMix64::new(config.seed);
    let holes = (0..config.deals)
        .map(|index| (index, Deal::sample(&mut rng).holes))
        .collect::<Vec<_>>();
    let workers = config.threads.min(holes.len());
    let chunks = holes
        .chunks(holes.len().div_ceil(workers))
        .map(<[(u64, [[u8; 2]; 2])]>::to_vec)
        .collect::<Vec<_>>();
    let samples = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for chunk in chunks {
            let generator = generator.fork_for_response_evaluation();
            let config = &config;
            handles.push(scope.spawn(move || -> Result<Vec<[SeatSample; 2]>, String> {
                chunk.into_iter().map(|(index, holes)| {
                    let mut seats = Vec::new();
                    for seat in 0..2 {
                        let seed = config.seed ^ index.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (seat as u64 + 1).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                        let mut rng = SplitMix64::new(seed);
                        let scenarios = sample_causal_scenarios(holes[seat], seat, config.public_branches_per_street, config.opponent_samples_per_runout, &mut rng);
                        let weights = vec![1.0 / scenarios.len() as f64; scenarios.len()];
                        let mut searched = 0;
                        let (value, plan, _) = solve_causal_response_plan_limited(&generator, GameState::initial(&config.game), &scenarios, &weights, seat, &mut searched, config.node_limit_per_seat)?;
                        let mut replayed = 0;
                        let baseline = joint_walk(&generator, GameState::initial(&config.game), &scenarios, &weights, None, &mut replayed, config.node_limit_per_seat)?;
                        baseline.validate()?;
                        replayed = 0;
                        let response = joint_walk(&generator, GameState::initial(&config.game), &scenarios, &weights, Some((seat, &plan)), &mut replayed, config.node_limit_per_seat)?;
                        response.validate()?;
                        let gain = response.own_payoffs_bb[seat] - baseline.own_payoffs_bb[seat];
                        if gain < -1e-8 || (response.own_payoffs_bb[seat] - value).abs() > 1e-8 { return Err("cash response does not dominate/reconstruct its own frozen-policy baseline".into()); }
                        seats.push(SeatSample { baseline, response, own_gain_bb: gain.max(0.0), searched_nodes: searched, replayed_nodes: replayed });
                    }
                    Ok(seats.try_into().ok().expect("two seats"))
                }).collect()
            }));
        }
        let mut output = Vec::new();
        for handle in handles {
            output.extend(
                handle
                    .join()
                    .map_err(|_| "cash response worker panicked".to_owned())??,
            );
        }
        Ok::<_, String>(output)
    })?;
    let mut gains = [0.0; 2];
    let mut mean = 0.0;
    let mut m2 = 0.0;
    for (index, sample) in samples.iter().enumerate() {
        for seat in 0..2 {
            gains[seat] += sample[seat].own_gain_bb / config.deals as f64;
        }
        let total = sample[0].own_gain_bb + sample[1].own_gain_bb;
        let delta = total - mean;
        mean += delta / (index + 1) as f64;
        m2 += delta * (total - mean);
    }
    // Each complete own payoff is in [-stack,stack], so each gain is <=2*stack
    // and their nonnegative total is <=4*stack. Independent outer samples
    // permit a one-sided Hoeffding bound on the optimistic sample-game mean.
    let range = 4.0 * config.game.effective_stack_bb;
    let margin =
        range * ((1.0 / (1.0 - config.confidence)).ln() / (2.0 * config.deals as f64)).sqrt();
    Ok(CashCausalResponseReport { schema: "hu-cash-causal-sample-game-nash-conv-v1", validation_status: "research_only",
        rules_sha256: rules.sha256()?, policy_sha256, seed: config.seed, deals: config.deals, threads: workers,
        public_branches_per_street: config.public_branches_per_street, opponent_samples_per_runout: config.opponent_samples_per_runout,
        scenarios_per_seat: scenario_count, node_limit_per_seat: config.node_limit_per_seat, elapsed_seconds: started.elapsed().as_secs_f64(),
        mean_unilateral_gains_bb: gains, sample_mean_total_nash_conv_bb: mean,
        sampling_standard_error_total_bb: (m2.max(0.0) / (config.deals - 1) as f64 / config.deals as f64).sqrt(), confidence: config.confidence,
        one_sided_hoeffding_margin_bb: margin, abstract_game_optimistic_upper_bound_bb: (mean + margin).min(range), samples, game: config.game,
        assumptions: vec!["Independent uniform outer private-card samples and unbiased nested conditional chance/opponent-hand samples",
            "Exact exhaustive legal betting-tree optimization within each sampled game; no budget-truncated values",
            "One responder action for all hidden-hand and future-card scenarios sharing an observed information set",
            "Frozen scalar average policy observes only acting-seat private cards and already dealt public cards",
            "Expected sampled maximum is optimistic by Jensen; subtracting the unbiased on-policy own baseline preserves an abstract-game upper bound"],
        limitations: vec!["Optimistic finite sample-game bound, not an exact full-card-game NashConv measurement",
            "Bound covers the pinned sizing/raise abstraction, not unrestricted no-limit actions",
            "No inherited zero-sum CFR convergence or safe-resolving guarantee; metric is total NashConv, not half the total",
            "No accepted raked continuation/resolver route or normal serving-gate qualification",
            "Per-seat joint baselines reconcile rake separately; differently conditioned sample baselines must not be zero-sum projected"] })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cash_hidden_hand_response_chooses_one_action_and_reconstructs_own_house_payoffs() {
        let generator = SampleGenerator::new(super::super::cash_training_tests::config()).unwrap();
        let mut state = GameState::initial(&generator.config.game);
        state.street = Street::River;
        state.invested = [2.0, 3.0];
        state.street_invested = [0.0, 1.0];
        state.raise_reopened = false;
        let scenarios = [
            Deal::from_sampled_cards([[32, 33], [48, 49]], [0, 5, 10, 19, 43]),
            Deal::from_sampled_cards([[32, 33], [24, 25]], [0, 5, 10, 19, 43]),
        ];
        let weights = [0.5, 0.5];
        let mut nodes = 0;
        let (value, plan, _) = solve_causal_response_plan_limited(
            &generator,
            state.clone(),
            &scenarios,
            &weights,
            0,
            &mut nodes,
            100,
        )
        .unwrap();
        assert_eq!(plan.len(), 1);
        assert!((value + 0.14).abs() < 1e-10);
        let response = joint_walk(
            &generator,
            state.clone(),
            &scenarios,
            &weights,
            Some((0, &plan)),
            &mut 0,
            100,
        )
        .unwrap();
        response.validate().unwrap();
        assert!((response.own_payoffs_bb[0] - value).abs() < 1e-10);
        assert!((response.house_rake_bb - 0.28).abs() < 1e-10);
        assert!(solve_causal_response_plan_limited(
            &generator, state, &scenarios, &weights, 0, &mut 0, 1
        )
        .err()
        .unwrap()
        .contains("no partial response"));
    }
}
