//! Small policy-action experiment, not a deployment or equilibrium certificate.
//! Correct the first confident flop opportunity per player, then compare against
//! fixed retained opponents on paired fresh hands. Opponents retain their own
//! original continuation policy, not the candidate's changed completion.

use super::*;
use crate::blueprint::neural::trajectory_action_matches;
mod terminal_pair;

#[derive(Clone, Default)]
struct DecisionBank {
    rows: [BTreeMap<u64, ResolverDecision>; 4],
}

impl DecisionBank {
    fn from_decisions<'a>(
        decisions: impl Iterator<Item = &'a ResolverDecision>,
        maximum: ResolverGranularity,
    ) -> Result<Self, String> {
        let mut bank = Self::default();
        for decision in decisions {
            let n = decision.action_labels.len();
            if decision.actor > 1
                || n == 0
                || decision.selected_action >= n
                || decision.action_values_bb.len() != n
                || decision.action_standard_errors_bb.len() != n
                || decision.action_values_bb.iter().any(|v| !v.is_finite())
                || decision
                    .action_standard_errors_bb
                    .iter()
                    .any(|v| !v.is_finite() || *v < 0.0)
                || !decision.selected_action_mean_gap_bb.is_finite()
                || !decision
                    .approximate_selected_action_gap_lower_bound_99_5_percent_bb
                    .is_finite()
                || decision.response_advantage.as_ref().is_some_and(|a| {
                    !a.baseline_mean_ev_bb.is_finite()
                        || !a.selected_mean_gain_bb.is_finite()
                        || !a.selected_gain_standard_error_bb.is_finite()
                        || a.selected_gain_standard_error_bb < 0.0
                        || !a.approximate_gain_lower_bound_99_5_percent_bb.is_finite()
                })
            {
                return Err("invalid retained response decision".to_owned());
            }
            let rank = granularity_rank(decision.granularity) as usize;
            if !decision.is_profitable_response() || rank > granularity_rank(maximum) as usize {
                continue;
            }
            if bank.rows[rank]
                .insert(decision.information_set, decision.clone())
                .is_some()
            {
                return Err("duplicate retained response information set".to_owned());
            }
        }
        Ok(bank)
    }

    fn find(
        &self,
        state: &GameState,
        deal: &Deal,
        actions: &[LegalAction],
        game: &BlueprintConfig,
    ) -> Option<&ResolverDecision> {
        for (rank, rows) in self.rows.iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            let (key, labels) = match rank {
                0 => (information_set(state, deal, game).0, None),
                1 => (
                    observable_backoff_information_set(state, deal, game, actions).0,
                    None,
                ),
                2 => (
                    coarse_observable_backoff_information_set(state, deal, game, actions).0,
                    None,
                ),
                _ => {
                    let (key, _, _, labels) =
                        strategic_observable_backoff_information_set(state, deal, game, actions);
                    (key, Some(labels))
                }
            };
            if let Some(row) = rows.get(&key) {
                let compatible = match labels {
                    Some(labels) => row.action_labels == labels,
                    None => row
                        .action_labels
                        .iter()
                        .map(String::as_str)
                        .eq(actions.iter().map(|a| a.label.as_str())),
                };
                if compatible && row.actor == state.actor && row.street == state.street {
                    return Some(row);
                }
            }
        }
        None
    }

    fn len(&self) -> usize {
        self.rows.iter().map(BTreeMap::len).sum()
    }
}

pub(super) struct FlopPatch {
    bank: DecisionBank,
    weight: f64,
    all_in_samples: Option<u32>,
    // Saved nonterminal decisions must not discard the control's terminal
    // correction, including when their own blend weight is zero.
    prior_terminal: Option<TerminalFlopOptions>,
}

impl FlopPatch {
    pub(super) fn terminal(options: &TerminalFlopOptions) -> Self {
        Self {
            bank: DecisionBank::default(),
            weight: options.weight,
            all_in_samples: Some(options.equity_samples),
            prior_terminal: None,
        }
    }
    pub(super) fn strategy(
        &self,
        base: &TabularResponsePolicy,
        state: &GameState,
        deal: &Deal,
        actions: &[LegalAction],
        game: &BlueprintConfig,
        mut baseline: Vec<f64>,
    ) -> Vec<f64> {
        if state.street != Street::Flop {
            return baseline;
        }
        if let Some(prior) = &self.prior_terminal {
            if actions.iter().all(|action| state.apply(action, game).terminal.is_some()) {
                // Preserve the entire terminal rule, including its abstention
                // on uncertain values. A coarse saved decision cannot override it.
                return Self::terminal(prior).strategy(base, state, deal, actions, game, baseline);
            }
        }
        if self.weight == 0.0 {
            return baseline;
        }
        if let Some(samples) = self.all_in_samples {
            if let Some(selected) =
                super::flop_allin::correction(base, state, deal, actions, game, samples)
            {
                for p in &mut baseline {
                    *p *= 1.0 - self.weight;
                }
                baseline[selected] += self.weight;
            }
            return baseline;
        }
        let Some(decision) = self.bank.find(state, deal, actions, game) else {
            return baseline;
        };
        // Pure public-history replay provides perfect recall of whether this
        // player already had a matching opportunity. No per-hand hidden state.
        let mut cursor = GameState::initial(game);
        for observed in &state.trajectory {
            let prior_actions = cursor.legal_actions(game);
            if cursor.street == Street::Flop
                && cursor.actor == state.actor
                && self
                    .bank
                    .find(&cursor, deal, &prior_actions, game)
                    .is_some()
            {
                return baseline;
            }
            let action = prior_actions
                .iter()
                .find(|a| trajectory_action_matches(&cursor, a, observed, game))
                .expect("flop correction can replay its public history");
            cursor = cursor.apply(action, game);
        }
        for probability in &mut baseline {
            *probability *= 1.0 - self.weight;
        }
        baseline[decision.selected_action] += self.weight;
        baseline
    }
}

#[derive(Clone, Debug)]
pub struct FlopPatchEvaluationConfig {
    pub checkpoint: PathBuf,
    pub proposal_response: PathBuf,
    pub opponent_responses: Vec<PathBuf>,
    pub weight: f64,
    pub evaluation_deals: u64,
    pub seed: u64,
    pub workers: usize,
    pub all_in_samples: Option<u32>,
    pub integrate_terminal: bool,
    pub flop_backoff: Option<FlopBackoffOptions>,
}

pub(super) fn validate_report(
    report: &FullGameResponseEvaluation,
    digest: &str,
    depth: f64,
) -> Result<(), String> {
    if report.policy_sha256 != digest
        || report.depth_bb != depth
        || report.checkpoint_training_iterations.is_none()
    {
        return Err("retained response must pin the same tabular checkpoint and depth".to_owned());
    }
    if let Some(options) = &report.turn_resolver {
        options.validate()?;
    }
    if let Some(options) = &report.terminal_flop {
        options.validate()?;
    }
    if let Some(options) = &report.flop_backoff {
        options.validate()?;
    }
    for seat in 0..2 {
        if report.resolvers[seat].responder != seat
            || report.preflop_responses[seat]
                .iter()
                .chain(&report.resolvers[seat].decisions)
                .any(|d| d.actor != seat)
        {
            return Err("retained response seat mismatch".to_owned());
        }
        DecisionBank::from_decisions(
            report.preflop_responses[seat].iter().chain(&report.resolvers[seat].decisions),
            report.maximum_granularity,
        )?;
    }
    Ok(())
}

fn make_policy(
    table: Arc<InferenceTable>,
    options: Option<TurnResolveOptions>,
    patch: Option<Arc<FlopPatch>>,
    flop_backoff: Option<Arc<backoff::FlopBackoff>>,
) -> Box<dyn ResponsePolicy> {
    let base = TabularResponsePolicy {
        table,
        coverage: RefCell::default(),
        flop_patch: patch,
        flop_backoff,
        completion_coverage: RefCell::default(),
    };
    match options {
        Some(options) => Box::new(turn::TabularTurnPolicy::new(base, options)),
        None => Box::new(base),
    }
}

// Explicit two-policy rollout: the attacker always uses its frozen completion.
// Only its first matching action is overridden by the retained response bank.
fn panel_rollout(
    defender: &dyn ResponsePolicy,
    attacker: &dyn ResponsePolicy,
    bank: &DecisionBank,
    mut state: GameState,
    deal: &Deal,
    game: &BlueprintConfig,
    responder: usize,
    mut rng: SplitMix64,
) -> (f64, bool) {
    let mut deviated = false;
    while state.terminal.is_none() {
        let actions = state.legal_actions(game);
        let selected = if state.actor == responder {
            match (!deviated)
                .then(|| bank.find(&state, deal, &actions, game))
                .flatten()
            {
                Some(decision) => {
                    rng.next_f64();
                    deviated = true;
                    decision.selected_action
                }
                None => sample_index(&attacker.strategy(&state, deal, &actions, game), &mut rng),
            }
        } else {
            sample_index(&defender.strategy(&state, deal, &actions, game), &mut rng)
        };
        state = state.apply(&actions[selected], game);
    }
    let p0 = realized_utility_p0(&state, deal);
    (if responder == 0 { -p0 } else { p0 }, deviated)
}

fn summarize(samples: &[[f64; 4]]) -> serde_json::Value {
    let n = samples.len() as f64;
    let control = samples.iter().map(|s| s[0]).sum::<f64>() / n;
    let candidate = samples.iter().map(|s| s[1]).sum::<f64>() / n;
    let mean = samples.iter().map(|s| s[1] - s[0]).sum::<f64>() / n;
    let se = (samples
        .iter()
        .map(|s| (s[1] - s[0] - mean).powi(2))
        .sum::<f64>()
        / (n * (n - 1.0)))
        .sqrt();
    serde_json::json!({
        "control_defender_utility_bb":control, "candidate_defender_utility_bb":candidate,
        "paired_improvement_bb_per_hand":mean, "paired_standard_error_bb":se,
        "approximate_paired_99_percent_interval_bb":[mean - 2.575_829_303_548_900_4 * se, mean + 2.575_829_303_548_900_4 * se],
        "control_attacked_hand_fraction":samples.iter().map(|s| s[2]).sum::<f64>() / n,
        "candidate_attacked_hand_fraction":samples.iter().map(|s| s[3]).sum::<f64>() / n,
        "changed_payoff_hands":samples.iter().filter(|s| s[0] != s[1]).count(),
    })
}

pub fn evaluate_flop_patch(
    config: FlopPatchEvaluationConfig,
) -> Result<serde_json::Value, Box<dyn Error>> {
    if !config.weight.is_finite()
        || !(0.0..=if config.all_in_samples.is_some() { 1.0 } else { 0.5 }).contains(&config.weight)
        || config.evaluation_deals < 2
        || config.evaluation_deals > 1_000_000
        || !(1..=4).contains(&config.workers)
        || config.opponent_responses.is_empty()
        || config
            .all_in_samples
            .is_some_and(|samples| !(128..=16384).contains(&samples))
        || (config.integrate_terminal && config.all_in_samples.is_none())
        || (config.flop_backoff.is_some()
            && (config.integrate_terminal || config.all_in_samples.is_some()))
    {
        return Err("flop pilot requires weight 0..1 for terminal-only corrections (0..0.5 otherwise), 2..1000000 hands, 1..4 workers and retained opponents".into());
    }
    if let Some(options) = &config.flop_backoff {
        options.validate()?;
    }
    let digest = sha256_file(&config.checkpoint)?;
    let proposal: FullGameResponseEvaluation =
        serde_json::from_slice(&fs::read(&config.proposal_response)?)?;
    let proposal_digest = sha256_file(&config.proposal_response)?;
    let mut opponents = Vec::new();
    for path in &config.opponent_responses {
        let report: FullGameResponseEvaluation = serde_json::from_slice(&fs::read(path)?)?;
        if report.uses_seed(config.seed) {
            return Err("fresh evaluation seed must differ from opponent training seed".into());
        }
        opponents.push((sha256_file(path)?, report));
    }
    if proposal.uses_seed(config.seed) {
        return Err("fresh evaluation seed must differ from proposal training seed".into());
    }
    let table = Arc::new(InferenceTable::read(&config.checkpoint)?);
    let game = &table.config;
    validate_report(&proposal, &digest, game.effective_stack_bb)?;
    for (_, report) in &opponents {
        validate_report(report, &digest, game.effective_stack_bb)?;
    }
    // A rejected seat is not sufficient evidence for a policy correction.
    let patch = Arc::new(FlopPatch {
        bank: if config.all_in_samples.is_some() || config.flop_backoff.is_some() {
            DecisionBank::default()
        } else {
            DecisionBank::from_decisions(
                proposal
                    .resolvers
                    .iter()
                    .enumerate()
                    .filter(|(seat, _)| proposal.response_deployed[*seat])
                    .flat_map(|(_, resolver)| resolver.decisions.iter())
                    .filter(|d| d.street == Street::Flop && d.is_profitable_response()),
                proposal.maximum_granularity,
            )?
        },
        weight: config.weight,
        all_in_samples: config.all_in_samples,
        prior_terminal: if config.all_in_samples.is_none() {
            proposal.terminal_flop.clone()
        } else {
            None
        },
    });
    if patch.bank.len() == 0 && config.all_in_samples.is_none() && config.flop_backoff.is_none() {
        return Err("no calibrated confident flop decisions for this proposal".into());
    }
    let patch_counts: Vec<_> = (0..2)
        .map(|seat| {
            patch
                .bank
                .rows
                .iter()
                .flat_map(|m| m.values())
                .filter(|d| d.actor == seat)
                .count()
        })
        .collect();
    let mut results = Vec::new();
    let build_backoff =
        |options: Option<FlopBackoffOptions>| -> Result<Option<Arc<backoff::FlopBackoff>>, String> {
            options
                .map(|o| backoff::FlopBackoff::build(&table, o).map(Arc::new))
                .transpose()
        };
    let control_backoff = build_backoff(proposal.flop_backoff.clone())?;
    let candidate_backoff = if config.flop_backoff.is_some() {
        build_backoff(config.flop_backoff.clone())?
    } else {
        control_backoff.clone()
    };
    if let Some(pooled) = &candidate_backoff {
        eprintln!("flop-pooling {}", pooled.summary());
    }
    for (opponent_index, (opponent_digest, opponent)) in opponents.iter().enumerate() {
        let opponent_backoff = if opponent.flop_backoff == proposal.flop_backoff {
            control_backoff.clone()
        } else {
            build_backoff(opponent.flop_backoff.clone())?
        };
        for responder in 0..2 {
            let bank = DecisionBank::from_decisions(
                opponent.preflop_responses[responder]
                    .iter()
                    .chain(&opponent.resolvers[responder].decisions),
                opponent.maximum_granularity,
            )?;
            let phase_seed = derived_seed(
                config.seed,
                responder as u64,
                0xF10F_0000 + opponent_index as u64,
            );
            let mut chance = SplitMix64::new(phase_seed);
            let cards: Vec<_> = (0..config.evaluation_deals)
                .map(|index| {
                    let deal = Deal::sample(&mut chance);
                    (index, deal.holes, deal.board)
                })
                .collect();
            let chunk_size = cards.len().div_ceil(config.workers);
            let (samples, completion): (Vec<[f64; 4]>, backoff::CompletionCoverage) =
                std::thread::scope(|scope| {
                    let handles: Vec<_> = cards.chunks(chunk_size).map(|chunk| {
                    let table = Arc::clone(&table);
                    let patch = Arc::clone(&patch);
                    let bank = &bank;
                    let defender_options = proposal.turn_resolver.clone();
                    let attacker_options = opponent.turn_resolver.clone();
                    let control_patch = proposal.terminal_flop.as_ref().map(|o| Arc::new(FlopPatch::terminal(o)));
                    let attacker_patch = opponent.terminal_flop.as_ref().map(|o| Arc::new(FlopPatch::terminal(o)));
                    let candidate_patch = if config.flop_backoff.is_some() { control_patch.clone() } else { Some(patch) };
                    let control_backoff = control_backoff.clone();
                    let candidate_backoff = candidate_backoff.clone();
                    let opponent_backoff = opponent_backoff.clone();
                    scope.spawn(move || {
                        let control = make_policy(Arc::clone(&table), defender_options.clone(), control_patch, control_backoff);
                        let candidate = make_policy(Arc::clone(&table), defender_options, candidate_patch, candidate_backoff);
                        let attacker = make_policy(Arc::clone(&table), attacker_options, attacker_patch, opponent_backoff);
                        let samples = chunk.iter().map(|(index, holes, board)| {
                            if index % 128 == 0 { eprintln!("flop-panel opponent={opponent_index} responder={responder} hand={index}/{}", config.evaluation_deals); }
                            let deal = Deal::from_sampled_cards(*holes, *board);
                            let seed = derived_seed(phase_seed, *index, 11);
                            if config.integrate_terminal {
                                return terminal_pair::rollout(control.as_ref(), candidate.as_ref(), attacker.as_ref(), bank,
                                    GameState::initial(&table.config), &deal, &table.config, responder, SplitMix64::new(seed));
                            }
                            let a = panel_rollout(control.as_ref(), attacker.as_ref(), bank, GameState::initial(&table.config), &deal, &table.config, responder, SplitMix64::new(seed));
                            let b = panel_rollout(candidate.as_ref(), attacker.as_ref(), bank, GameState::initial(&table.config), &deal, &table.config, responder, SplitMix64::new(seed));
                            [a.0, b.0, u8::from(a.1) as f64, u8::from(b.1) as f64]
                        }).collect::<Vec<_>>();
                        (samples, candidate.take_completion_coverage())
                    })
                }).collect();
                    let mut samples = Vec::new();
                    let mut completion = backoff::CompletionCoverage::default();
                    for handle in handles {
                        let (local, counts) = handle
                            .join()
                            .unwrap_or_else(|e| std::panic::resume_unwind(e));
                        samples.extend(local);
                        completion.add(counts);
                    }
                    (samples, completion)
                });
            results.push(serde_json::json!({
                "opponent_report_sha256":opponent_digest, "opponent_turn_resolver":opponent.turn_resolver,
                "opponent_terminal_flop":opponent.terminal_flop,
                "opponent_flop_backoff":opponent.flop_backoff,
                "opponent_originally_calibrated":opponent.response_deployed[responder],
                "responder":responder, "defender":1-responder, "confident_opponent_decisions":bank.len(),
                "evaluation_seed":phase_seed, "summary":summarize(&samples),
                "candidate_flop_completion":completion,
                "paired_samples_control_candidate_and_attack_flags":samples,
            }));
        }
    }
    Ok(serde_json::json!({
        "schema":"tabular-first-confident-flop-correction-panel-v1",
        "policy_sha256":digest, "proposal_report_sha256":proposal_digest,
        "checkpoint_training_iterations":table.rounds, "depth_bb":game.effective_stack_bb,
        "defender_turn_resolver":proposal.turn_resolver, "patch_weight":config.weight,
        "defender_control_terminal_flop":proposal.terminal_flop,
        "defender_candidate_prior_terminal_flop":patch.prior_terminal,
        "defender_control_flop_backoff":proposal.flop_backoff,
        "defender_candidate_flop_backoff":config.flop_backoff.as_ref().or(proposal.flop_backoff.as_ref()),
        "flop_backoff_summary":candidate_backoff.as_ref().map(|b| b.summary()),
        "patch_rule":if config.flop_backoff.is_some() { "missing_flop_frozen_strategy_mass_pooling" } else if config.all_in_samples.is_some() { "range_conditioned_terminal_flop" } else { "first_confident_saved_flop_action" },
        "all_in_equity_samples":config.all_in_samples,
        "payoff_assessment":if config.integrate_terminal { "exact_terminal_action_and_runout_conditional_mean" } else { "paired_realized_actions_and_runouts" },
        "patch_decisions_by_seat":patch_counts, "seed":config.seed,
        "evaluation_deals_per_seat_per_opponent":config.evaluation_deals, "workers":config.workers,
        "results":results,
        "interpretation":"Positive paired values mean the candidate defender earns more against the identical frozen opponent on fresh hands. Opponent actions use only observable information; its original continuation never changes with the defender. Saved-action mode blends the first confident flop opportunity from calibrated proposal seats. Terminal-range mode conditions on exact hero cards and opponent action likelihoods, samples legal runouts and requires a 99.5% Hoeffding equity margin before blending a call/fold correction. Pooling mode changes only missing/untrained flop rows using frozen strategy mass matched by current hand/board buckets and public betting history, while preserving the control terminal correction; borrowed rows are not newly trained exact coverage. No mode guarantees safety against arbitrary opponents. Rejected retained opponents remain disclosed raw diagnostic challenges, never certified responses. Per-comparison normal-approximation intervals are not multiple-comparison-corrected or exploitability upper bounds. This experiment does not promote a policy or modify the source checkpoint."
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flop_fixture() -> (TabularResponsePolicy, Deal, GameState, ResolverDecision) {
        let (policy, deal) = super::super::tests::tabular_fixture();
        let game = &policy.table.config;
        let root = GameState::initial(game);
        let called = root.apply(&root.legal_actions(game)[1], game);
        let flop = called.apply(&called.legal_actions(game)[0], game);
        assert_eq!(flop.street, Street::Flop);
        let actions = flop.legal_actions(game);
        let (key, descriptor, history) = information_set(&flop, &deal, game);
        let mut acc = DecisionAccumulator::new(&descriptor, history, &actions);
        let mut values = vec![0.0; actions.len()];
        values[0] = 1.0;
        for _ in 0..4 {
            acc.add(&values);
        }
        (
            policy,
            deal,
            flop,
            acc.finish(key, ResolverGranularity::ExactTrajectory, 20.0),
        )
    }

    #[test]
    fn saved_decisions_preserve_terminal_correction_and_abstention() {
        let (policy, deal, flop, _) = flop_fixture();
        let game = &policy.table.config;
        let facing = flop.apply(flop.legal_actions(game).last().unwrap(), game);
        let actions = facing.legal_actions(game);
        assert!(actions.iter().all(|a| facing.apply(a, game).terminal.is_some()));
        let baseline = policy.frozen_strategy(&facing, &deal, &actions, game);
        let (key, descriptor, history) = information_set(&facing, &deal, game);
        for selected in 0..actions.len() {
            let mut acc = DecisionAccumulator::new(&descriptor, history.clone(), &actions);
            let mut values = vec![0.0; actions.len()];
            values[selected] = 1.0;
            for _ in 0..4 {
                acc.add(&values);
            }
            let decision = acc.finish(key, ResolverGranularity::ExactTrajectory, 20.0);
            for equity_samples in [128, 2048] {
                let prior = TerminalFlopOptions { equity_samples, weight: 0.25 };
                let expected = FlopPatch::terminal(&prior)
                    .strategy(&policy, &facing, &deal, &actions, game, baseline.clone());
                for weight in [0.0, 0.25] {
                    let saved = FlopPatch {
                        bank: DecisionBank::from_decisions(
                            std::iter::once(&decision), ResolverGranularity::ExactTrajectory,
                        ).unwrap(),
                        weight,
                        all_in_samples: None,
                        prior_terminal: Some(prior.clone()),
                    };
                    assert_eq!(expected, saved.strategy(
                        &policy, &facing, &deal, &actions, game, baseline.clone(),
                    ));
                }
            }
        }
    }

    #[test]
    fn patch_is_bounded_and_identical_for_serving_and_range_replay() {
        let (mut policy, deal, flop, decision) = flop_fixture();
        let game = policy.table.config.clone();
        let actions = flop.legal_actions(&game);
        let baseline = policy.frozen_strategy(&flop, &deal, &actions, &game);
        policy.flop_patch = Some(Arc::new(FlopPatch {
            bank: DecisionBank::from_decisions(
                std::iter::once(&decision),
                ResolverGranularity::ExactTrajectory,
            )
            .unwrap(),
            weight: 0.25,
            all_in_samples: None,
            prior_terminal: None,
        }));
        let changed = policy.strategy(&flop, &deal, &actions, &game);
        assert_eq!(
            changed,
            policy.frozen_strategy(&flop, &deal, &actions, &game)
        );
        assert_eq!(changed[0], 0.75 * baseline[0] + 0.25);
        assert!((changed.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let hidden_changed = Deal::from_cards(
            [[36, 37], deal.holes[1]],
            [deal.board[0], deal.board[1], deal.board[2], 31, 32],
        );
        assert_eq!(
            changed,
            policy.frozen_strategy(&flop, &hidden_changed, &actions, &game)
        );
        let root = GameState::initial(&game);
        assert_eq!(
            policy.strategy(&root, &deal, &root.legal_actions(&game), &game),
            policy
                .isolated_copy()
                .frozen_strategy(&root, &deal, &root.legal_actions(&game), &game)
        );
    }

    #[test]
    fn invalid_or_duplicate_actions_are_rejected() {
        let (_, _, _, mut decision) = flop_fixture();
        assert!(DecisionBank::from_decisions(
            [&decision, &decision].into_iter(),
            ResolverGranularity::ExactTrajectory
        )
        .is_err());
        decision.selected_action = decision.action_labels.len();
        assert!(DecisionBank::from_decisions(
            std::iter::once(&decision),
            ResolverGranularity::ExactTrajectory
        )
        .is_err());
    }

    #[test]
    fn only_the_first_matching_flop_opportunity_is_corrected() {
        let (mut policy, deal, flop, first) = flop_fixture();
        let game = policy.table.config.clone();
        let checked = flop.apply(&flop.legal_actions(&game)[0], &game);
        let facing_bet = checked.apply(&checked.legal_actions(&game)[1], &game);
        assert_eq!(facing_bet.actor, flop.actor);
        assert_eq!(facing_bet.street, Street::Flop);
        let actions = facing_bet.legal_actions(&game);
        let (key, descriptor, history) = information_set(&facing_bet, &deal, &game);
        let mut accumulator = DecisionAccumulator::new(&descriptor, history, &actions);
        let mut values = vec![0.0; actions.len()];
        values[0] = 1.0;
        for _ in 0..4 {
            accumulator.add(&values);
        }
        let second = accumulator.finish(key, ResolverGranularity::ExactTrajectory, 20.0);
        let baseline = policy.frozen_strategy(&facing_bet, &deal, &actions, &game);
        policy.flop_patch = Some(Arc::new(FlopPatch {
            bank: DecisionBank::from_decisions(
                [&first, &second].into_iter(),
                ResolverGranularity::ExactTrajectory,
            )
            .unwrap(),
            weight: 0.25,
            all_in_samples: None,
            prior_terminal: None,
        }));
        assert_eq!(
            baseline,
            policy.frozen_strategy(&facing_bet, &deal, &actions, &game)
        );
        let root = GameState::initial(&game);
        let patched_preflop =
            policy.frozen_strategy(&root, &deal, &root.legal_actions(&game), &game);
        policy.flop_patch = None;
        assert_eq!(
            patched_preflop,
            policy.frozen_strategy(&root, &deal, &root.legal_actions(&game), &game)
        );
    }

    #[test]
    fn fixed_panel_matches_the_one_step_response_and_identity_comparison() {
        let (policy, deal, _, decision) = flop_fixture();
        let game = &policy.table.config;
        let bank = DecisionBank::from_decisions(
            std::iter::once(&decision),
            ResolverGranularity::ExactTrajectory,
        )
        .unwrap();
        let exact = BTreeMap::from([(decision.information_set, &decision)]);
        let empty = BTreeMap::new();
        let mut samples = Vec::new();
        for seed in 0..128 {
            let a = panel_rollout(
                &policy,
                &policy,
                &bank,
                GameState::initial(game),
                &deal,
                game,
                1,
                SplitMix64::new(seed),
            );
            let mut lookup = ResolverLookup::default();
            let expected = response_rollout(
                &policy,
                &exact,
                &empty,
                &empty,
                &empty,
                GameState::initial(game),
                &deal,
                game,
                1,
                false,
                &mut SplitMix64::new(seed),
                &mut lookup,
            );
            assert_eq!(a.0, expected);
            assert_eq!(a.1, lookup.hits > 0);
            let copy = policy.isolated_copy();
            let b = panel_rollout(
                &copy,
                &policy,
                &bank,
                GameState::initial(game),
                &deal,
                game,
                1,
                SplitMix64::new(seed),
            );
            samples.push([a.0, b.0, u8::from(a.1) as f64, u8::from(b.1) as f64]);
        }
        assert_eq!(summarize(&samples)["paired_improvement_bb_per_hand"], 0.0);
        assert_eq!(summarize(&samples)["paired_standard_error_bb"], 0.0);
    }

    #[test]
    fn paired_summary_has_zero_error_for_constant_improvement() {
        let result = summarize(&[[-20.0, -19.0, 1.0, 1.0], [10.0, 11.0, 0.0, 0.0]]);
        assert_eq!(result["paired_improvement_bb_per_hand"], 1.0);
        assert_eq!(result["paired_standard_error_bb"], 0.0);
    }

    #[test]
    fn whole_panel_keeps_order_across_workers_and_rejects_reused_seeds() {
        let (trainer, _) = super::super::tests::fixture_trainer();
        let directory =
            std::env::temp_dir().join(format!("flop-panel-fixture-{}", std::process::id()));
        fs::create_dir(&directory).unwrap();
        let checkpoint = directory.join("checkpoint.msgpack.gz");
        let response = directory.join("response.json");
        trainer.write_checkpoint(&checkpoint).unwrap();
        let mut report = evaluate_full_game_response(ResponseEvaluationConfig {
            game: trainer.config.clone(),
            source: ResponsePolicySource::TabularCheckpoint(checkpoint.clone()),
            training_deals: 8,
            calibration_deals: 8,
            evaluation_deals: 16,
            rollouts_per_action: 2,
            minimum_range_particles: 2,
            maximum_granularity: ResolverGranularity::StrategicObservableBackoff,
            seed: 781,
            response_workers: 1,
            turn_resolver: None,
            terminal_flop: None,
            flop_backoff: None,
            exact_terminal_training_values: false,
            conditional_preflop_runouts: false,
            postflop_only_response: false,
        })
        .unwrap();
        // Synthetic unit-test proposal; never used by a training pilot.
        let (_, _, _, decision) = flop_fixture();
        report.resolvers[1].decisions = vec![decision];
        report.response_deployed[1] = true;
        fs::write(&response, serde_json::to_vec(&report).unwrap()).unwrap();
        let mut config = FlopPatchEvaluationConfig {
            checkpoint,
            proposal_response: response.clone(),
            opponent_responses: vec![response],
            weight: 0.25,
            evaluation_deals: 71,
            seed: 913,
            workers: 1,
            all_in_samples: None,
            integrate_terminal: false,
            flop_backoff: None,
        };
        let reference = evaluate_flop_patch(config.clone()).unwrap();
        for workers in [2, 4] {
            config.workers = workers;
            let mut parallel = evaluate_flop_patch(config.clone()).unwrap();
            parallel["workers"] = serde_json::json!(1);
            assert_eq!(
                serde_json::to_vec(&reference).unwrap(),
                serde_json::to_vec(&parallel).unwrap()
            );
        }
        config.workers = 1;
        config.all_in_samples = Some(2048);
        config.integrate_terminal = true;
        let integrated = evaluate_flop_patch(config.clone()).unwrap();
        config.workers = 2;
        let mut parallel = evaluate_flop_patch(config.clone()).unwrap();
        parallel["workers"] = serde_json::json!(1);
        assert_eq!(
            serde_json::to_vec(&integrated).unwrap(),
            serde_json::to_vec(&parallel).unwrap()
        );
        let mut pooling = config.clone();
        pooling.all_in_samples = None;
        pooling.integrate_terminal = false;
        pooling.flop_backoff = Some(FlopBackoffOptions {
            minimum_average_visits: 1,
            weight: 1.0,
        });
        report.terminal_flop = Some(TerminalFlopOptions {
            equity_samples: 128,
            weight: 0.25,
        });
        fs::write(
            &pooling.proposal_response,
            serde_json::to_vec(&report).unwrap(),
        )
        .unwrap();
        let unchanged = evaluate_flop_patch(pooling.clone()).unwrap();
        // This checkpoint fixture has no trained flop rows to borrow. The
        // candidate must therefore retain the entire terminal-corrected control.
        for row in unchanged["results"].as_array().unwrap() {
            for pair in row["paired_samples_control_candidate_and_attack_flags"]
                .as_array()
                .unwrap()
            {
                assert_eq!(pair[0], pair[1]);
            }
        }
        let mut zero_saved = pooling.clone();
        zero_saved.flop_backoff = None;
        zero_saved.weight = 0.0;
        let unchanged = evaluate_flop_patch(zero_saved).unwrap();
        for row in unchanged["results"].as_array().unwrap() {
            for pair in row["paired_samples_control_candidate_and_attack_flags"]
                .as_array()
                .unwrap()
            {
                assert_eq!(pair[0], pair[1], "zero saved-action weight must preserve the control policy");
            }
        }
        // Terminal-only interpolation changes no upstream decisions or range
        // likelihoods. Its exact conditional payoff is affine in blend weight.
        let mut terminal_curve = pooling.clone();
        terminal_curve.flop_backoff = None;
        terminal_curve.all_in_samples = Some(128);
        terminal_curve.integrate_terminal = true;
        terminal_curve.weight = 0.5;
        let half = evaluate_flop_patch(terminal_curve.clone()).unwrap();
        terminal_curve.weight = 1.0;
        let full = evaluate_flop_patch(terminal_curve.clone()).unwrap();
        let mut changed = 0;
        for (a, b) in half["results"].as_array().unwrap().iter().zip(full["results"].as_array().unwrap()) {
            let samples = "paired_samples_control_candidate_and_attack_flags";
            for (a, b) in a[samples].as_array().unwrap().iter().zip(b[samples].as_array().unwrap()) {
                assert_eq!(a[0], b[0]);
                let half_gain = a[1].as_f64().unwrap() - a[0].as_f64().unwrap();
                let full_gain = b[1].as_f64().unwrap() - b[0].as_f64().unwrap();
                // Control is weight .25: .25 -> 1 is three times .25 -> .5.
                assert!((full_gain - 3.0 * half_gain).abs() < 1e-10);
                changed += usize::from(half_gain.abs() > 1e-10);
            }
        }
        assert!(changed > 0, "linearity check must exercise changed terminal decisions");
        terminal_curve.all_in_samples = None;
        terminal_curve.integrate_terminal = false;
        assert!(evaluate_flop_patch(terminal_curve).is_err(), "saved nonterminal corrections remain capped at .5");
        pooling.integrate_terminal = true;
        assert!(evaluate_flop_patch(pooling).is_err());
        config.all_in_samples = None;
        assert!(evaluate_flop_patch(config.clone()).is_err());
        config.all_in_samples = Some(2048);
        config.seed = report.seed;
        assert!(evaluate_flop_patch(config)
            .unwrap_err()
            .to_string()
            .contains("fresh"));
        fs::remove_dir_all(directory).unwrap();
    }
}
