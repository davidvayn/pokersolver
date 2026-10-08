//! Research-only cash query boundary. No legacy resolver or zero-sum values.
//! Opponent beliefs use public action likelihoods; unrevealed cards are sampled
//! conditionally, never supplied by the browser. All EV rollouts use the same
//! frozen average policy returned to the caller, without an online update.
use super::*;
mod closed_all_in;

pub const CASH_QUERY_SCHEMA: &str = "hu-cash-average-policy-query-v1";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CashPolicyQuery {
    pub schema: String,
    pub rules_sha256: String,
    pub network_sha256: String,
    pub query: PracticePolicyQuery,
}

pub struct CashPolicyEngineConfig {
    pub game: BlueprintConfig,
    pub model_version: String,
    pub network_path: PathBuf,
    pub rollout_samples: u32,
    pub seed: u64,
}

pub struct CashPolicyEngine {
    config: CashPolicyEngineConfig,
    policy: FrozenPolicy,
    rules_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CashPolicyQueryResult {
    pub schema: &'static str,
    pub validation_status: &'static str,
    pub request_id: String,
    pub state_hash: String,
    pub model_version: String,
    pub depth_bb: f64,
    pub rules_sha256: String,
    pub network_sha256: String,
    pub actions: Vec<PracticePolicyAction>,
    pub rollout_samples_per_action: u32,
    pub ev_evaluation: &'static str,
    pub exact_policy_tree_nodes: u64,
    pub exact_all_in_chance_outcomes: u64,
    pub action_ev_evaluations: Vec<&'static str>,
    pub opponent_combo_count: usize,
    pub maximum_probability_sum_error: f64,
    pub maximum_accounting_residual_bb: f64,
    pub limitations: Vec<&'static str>,
}

#[derive(Default)]
struct Moment {
    n: u32,
    mean: f64,
    m2: f64,
}
impl Moment {
    fn add(&mut self, value: f64) {
        self.n += 1;
        let delta = value - self.mean;
        self.mean += delta / f64::from(self.n);
        self.m2 += delta * (value - self.mean);
    }
    fn standard_error(&self) -> f64 {
        (self.m2.max(0.0) / f64::from(self.n - 1) / f64::from(self.n)).sqrt()
    }
}

fn seat_label(actor: usize) -> &'static str {
    if actor == 0 {
        "button-small-blind"
    } else {
        "big-blind"
    }
}
fn kind_label(kind: PracticePolicyActionKind) -> &'static str {
    match kind {
        PracticePolicyActionKind::Fold => "fold",
        PracticePolicyActionKind::Check => "check",
        PracticePolicyActionKind::Call => "call",
        PracticePolicyActionKind::Bet => "bet",
        PracticePolicyActionKind::Raise => "raise",
        PracticePolicyActionKind::AllIn => "all-in",
    }
}
fn street_label(street: Street) -> &'static str {
    match street {
        Street::Preflop => "preflop",
        Street::Flop => "flop",
        Street::Turn => "turn",
        Street::River => "river",
    }
}

/// Cash v3 parity with canonicalPolicyState in TypeScript, reconstructed from
/// the legal replay rather than trusting a caller's pot/history string.
pub fn canonical_cash_query_state(
    query: &PracticePolicyQuery,
    game: &BlueprintConfig,
) -> Result<String, String> {
    let rules = game
        .cash_rules
        .as_ref()
        .ok_or("cash query requires pinned rules")?;
    let final_state = replay_practice_query(query, game)?;
    validate_practice_state_snapshot(query, &final_state, game)?;
    let mut history = Vec::new();
    let mut state = GameState::initial(game);
    for observed in &query.actions {
        let amount = observed.amount_to_bb.unwrap_or_else(|| {
            if observed.kind == PracticePolicyActionKind::Call {
                state.to_call().min(state.remaining(state.actor, game))
            } else {
                0.0
            }
        });
        // Replay already established a unique match. Round derived payments
        // to integer cents before stringification (the browser ledger does so).
        let amount = cash::cash_units(amount, rules)? as f64 / rules.units_per_bb as f64;
        history.push(format!(
            "{}:{}:{}:{}",
            street_label(observed.street),
            seat_label(observed.actor),
            kind_label(observed.kind),
            amount
        ));
        let action = state
            .legal_actions(game)
            .into_iter()
            .find(|a| {
                let (kind, target) = practice_legal_action_identity(&state, a, game);
                kind == observed.kind
                    && match (target, observed.amount_to_bb) {
                        (Some(a), Some(b)) => (a - b).abs() < 1e-8,
                        (None, None) => true,
                        _ => false,
                    }
            })
            .ok_or("cash canonical replay action differs")?;
        state = state.apply(&action, game);
    }
    let mut private = query.private_cards;
    private.sort_unstable();
    let mut board = query.board.clone();
    // Preserve the exact public reveal trajectory, modulo flop permutations.
    // All-card sorting conflates lines with different opponent posteriors.
    let flop_length = board.len().min(3);
    board[..flop_length].sort_unstable();
    let card_string = |cards: &[u8]| {
        cards
            .iter()
            .map(u8::to_string)
            .collect::<Vec<_>>()
            .join(",")
    };
    let settled_pot = final_state.pot() - final_state.street_invested.iter().sum::<f64>();
    Ok(format!(
        "hu-cash-v3|{}|{}|{:.3}|{}|{}|{}|{}|{:.3}|{:.3}|{:.3}|{:.3}|{:.3}|{}",
        rules.canonical_string()?,
        query.model_version,
        game.effective_stack_bb,
        seat_label(query.actor),
        street_label(query.street),
        card_string(&private),
        card_string(&board),
        settled_pot,
        final_state.street_invested[0],
        final_state.street_invested[1],
        final_state.remaining(0, game),
        final_state.remaining(1, game),
        history.join("/")
    ))
}

impl CashPolicyEngine {
    pub fn load(config: CashPolicyEngineConfig) -> Result<Self, Box<dyn Error>> {
        config.game.validate()?;
        let rules = config
            .game
            .cash_rules
            .as_ref()
            .ok_or("cash engine cannot load a Home game")?;
        if config.model_version.trim().is_empty()
            || config.model_version.len() > 128
            || config.model_version.contains('|')
            || !(16..=4096).contains(&config.rollout_samples)
        {
            return Err("cash queries require a pinned model and 16..4096 rollout samples".into());
        }
        let policy = FrozenPolicy::load(&config.network_path)?;
        policy.bundle.validate_game(&config.game)?;
        if !matches!(policy.bundle.strategy_transform, StrategyTransform::Softmax) {
            return Err(
                "cash query engine requires frozen average weights, not training regrets".into(),
            );
        }
        Ok(Self {
            rules_sha256: rules.sha256()?,
            config,
            policy,
        })
    }

    fn validate(&self, input: &CashPolicyQuery) -> Result<(GameState, Deal), String> {
        let query = &input.query;
        if input.schema != CASH_QUERY_SCHEMA
            || input.rules_sha256 != self.rules_sha256
            || input.network_sha256 != self.policy.bundle_sha256()
            || query.model_version != self.config.model_version
            || query.depth_bb != self.config.game.effective_stack_bb
            || query.actor > 1
            || query.request_id.trim().is_empty()
            || query.request_id.len() > 128
            || query.actions.len() > MAX_TRAJECTORY_ACTIONS
            || query.state_hash.len() != 64
        {
            return Err("cash query identity differs from pinned rules, depth or weights".into());
        }
        let rules = self.config.game.cash_rules.as_ref().expect("cash rules");
        for amount in [query.total_pot_bb, query.last_full_raise_bb]
            .into_iter()
            .chain(query.stacks_bb)
            .chain(query.street_bets_bb)
            .chain(query.total_committed_bb)
            .chain(query.actions.iter().filter_map(|a| a.amount_to_bb))
        {
            cash::cash_units(amount, rules)?;
        }
        let deal = practice_query_deal(query)?;
        let state = replay_practice_query(query, &self.config.game)?;
        validate_practice_state_snapshot(query, &state, &self.config.game)?;
        let canonical = canonical_cash_query_state(query, &self.config.game)?;
        let digest = Sha256::digest(canonical.as_bytes());
        let expected = digest
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        if query.state_hash != expected {
            return Err("cash canonical state hash differs from exact replay".into());
        }
        Ok((state, deal))
    }

    /// Exact posterior over all compatible opponent combos given the public
    /// line. Likelihoods stay in log space; zero reach fails, never becomes a
    /// uniform fallback. Hero action likelihoods cancel for the known hand.
    fn opponent_belief(
        &self,
        query: &PracticePolicyQuery,
        placeholder: &Deal,
    ) -> Result<Vec<(Combo, f64)>, String> {
        let opponent = 1 - query.actor;
        let mut log_weights = Vec::new();
        for combo in all_combos() {
            if combo
                .cards()
                .iter()
                .any(|c| query.private_cards.contains(c) || query.board.contains(c))
            {
                continue;
            }
            let mut holes = placeholder.holes;
            holes[opponent] = combo.cards();
            let deal = Deal::from_sampled_cards(holes, placeholder.board);
            let mut state = GameState::initial(&self.config.game);
            let mut log_weight = 0.0;
            for observed in &query.actions {
                let actions = state.legal_actions(&self.config.game);
                let index = actions
                    .iter()
                    .position(|action| {
                        let (kind, target) =
                            practice_legal_action_identity(&state, action, &self.config.game);
                        kind == observed.kind
                            && match (target, observed.amount_to_bb) {
                                (Some(a), Some(b)) => (a - b).abs() < 1e-8,
                                (None, None) => true,
                                _ => false,
                            }
                    })
                    .ok_or("cash opponent posterior history changed")?;
                if state.actor == opponent {
                    let probabilities = cash_evaluation::checked_strategy(
                        &self.policy,
                        &state,
                        &deal,
                        &actions,
                        &self.config.game,
                    )?;
                    log_weight += probabilities[index].ln();
                }
                state = state.apply(&actions[index], &self.config.game);
            }
            if log_weight.is_finite() {
                log_weights.push((combo, log_weight));
            }
        }
        let maximum = log_weights
            .iter()
            .map(|(_, w)| *w)
            .fold(f64::NEG_INFINITY, f64::max);
        if !maximum.is_finite() {
            return Err(
                "cash query has no compatible positive-reach opponent belief; no fallback".into(),
            );
        }
        let total: f64 = log_weights.iter().map(|(_, w)| (*w - maximum).exp()).sum();
        Ok(log_weights
            .into_iter()
            .map(|(c, w)| (c, (w - maximum).exp() / total))
            .collect())
    }

    fn terminal_values(&self, state: &GameState, deal: &Deal) -> Result<[f64; 3], String> {
        let own = state.complete_runout_utilities(deal, &self.config.game);
        let rules = self.config.game.cash_rules.as_ref().expect("cash rules");
        let gross = 2 * cash::cash_units(state.invested[0].min(state.invested[1]), rules)?;
        let flop = state.street != Street::Preflop || state.terminal == Some(Terminal::Showdown);
        let rake = rules.rake_units(gross, flop)? as f64 / rules.units_per_bb as f64;
        if (own.iter().sum::<f64>() + rake).abs() > 1e-8 {
            return Err("cash query own payoffs disagree with house ledger".into());
        }
        Ok([own[0], own[1], rake])
    }

    /// Evaluate a fixed policy, NOT a best response: every future decision
    /// uses the frozen mix for that actor's own cards and public information.
    /// At the river there is no remaining chance, so finite enumeration has
    /// zero sampling error. Model/range error remains a separate limitation.
    fn exact_river_values(
        &self,
        state: &GameState,
        deal: &Deal,
        nodes: &mut u64,
    ) -> Result<[f64; 3], String> {
        *nodes += 1;
        if *nodes > 2_000_000 {
            return Err(
                "cash exact river exceeded node budget; no sampled fallback or partial score"
                    .into(),
            );
        }
        if state.terminal.is_some() {
            return self.terminal_values(state, deal);
        }
        if state.street != Street::River {
            return Err("cash exact river cannot evaluate an earlier street".into());
        }
        let actions = state.legal_actions(&self.config.game);
        let probabilities = cash_evaluation::checked_strategy(
            &self.policy,
            state,
            deal,
            &actions,
            &self.config.game,
        )?;
        let mut value = [0.; 3];
        for (action, probability) in actions.iter().zip(probabilities) {
            if probability == 0. {
                continue;
            }
            let child =
                self.exact_river_values(&state.apply(action, &self.config.game), deal, nodes)?;
            for i in 0..3 {
                value[i] += probability * child[i];
            }
        }
        Ok(value)
    }

    pub fn query(&self, input: CashPolicyQuery) -> Result<CashPolicyQueryResult, String> {
        let (state, placeholder) = self.validate(&input)?;
        let query = &input.query;
        let legal = state.legal_actions(&self.config.game);
        let probabilities = cash_evaluation::checked_strategy(
            &self.policy,
            &state,
            &placeholder,
            &legal,
            &self.config.game,
        )?;
        let belief = self.opponent_belief(query, &placeholder)?;
        let mut hash = Sha256::new();
        hash.update(query.state_hash.as_bytes());
        hash.update(input.network_sha256.as_bytes());
        let digest: [u8; 32] = hash.finalize().into();
        let mut rng = SplitMix64::new(
            self.config.seed
                ^ u64::from_le_bytes(digest[..8].try_into().expect("eight seed bytes")),
        );
        let mut moments = (0..legal.len())
            .map(|_| Moment::default())
            .collect::<Vec<_>>();
        let mut residual: f64 = 0.0;
        let exact = state.street == Street::River;
        let mut exact_nodes = 0;
        let mut all_in_outcomes = 0;
        let mut exact_actions = vec![exact; legal.len()];
        if !exact && matches!(state.street, Street::Flop | Street::Turn) {
            for (index, action) in legal.iter().enumerate() {
                let child = state.apply(action, &self.config.game);
                // Folds are already noiseless; this optimization targets
                // showdown leaves and their immediate call/fold responses.
                if matches!(child.terminal, Some(Terminal::Fold { .. })) {
                    continue;
                }
                let mut aggregate = [0.; 3];
                let mut closed = true;
                for (opponent, weight) in &belief {
                    let mut holes = placeholder.holes;
                    holes[1 - query.actor] = opponent.cards();
                    let deal = Deal::from_sampled_cards(holes, placeholder.board);
                    let Some(value) = self.exact_closed_all_in(
                        &child,
                        &deal,
                        &query.board,
                        &mut all_in_outcomes,
                    )?
                    else {
                        closed = false;
                        break;
                    };
                    for i in 0..3 {
                        aggregate[i] += weight * value[i];
                    }
                }
                if closed {
                    exact_actions[index] = true;
                    moments[index].mean = aggregate[query.actor];
                    residual = residual.max((aggregate[0] + aggregate[1] + aggregate[2]).abs());
                    if residual > 1e-8 {
                        return Err("closed all-in aggregate house accounting failed".into());
                    }
                }
            }
        }
        if exact {
            let mut values = vec![[0.; 3]; legal.len()];
            for (opponent, weight) in &belief {
                let mut holes = placeholder.holes;
                holes[1 - query.actor] = opponent.cards();
                let deal = Deal::from_sampled_cards(holes, placeholder.board);
                for (index, action) in legal.iter().enumerate() {
                    let result = self.exact_river_values(
                        &state.apply(action, &self.config.game),
                        &deal,
                        &mut exact_nodes,
                    )?;
                    for player in 0..3 {
                        values[index][player] += weight * result[player];
                    }
                }
            }
            for (moment, value) in moments.iter_mut().zip(values) {
                residual = residual.max((value[0] + value[1] + value[2]).abs());
                moment.mean = value[query.actor];
            }
            if residual > 1e-8 {
                return Err("cash exact river aggregate accounting failed".into());
            }
        }
        for _ in 0..if exact {
            0
        } else {
            self.config.rollout_samples
        } {
            let mut draw = rng.next_f64();
            let opponent = belief
                .iter()
                .find_map(|(c, p)| {
                    draw -= p;
                    (draw < 0.0).then_some(*c)
                })
                .unwrap_or(belief.last().expect("positive belief").0);
            let mut holes = placeholder.holes;
            holes[1 - query.actor] = opponent.cards();
            let mut deck = (0..52u8)
                .filter(|c| !holes.iter().flatten().any(|h| h == c) && !query.board.contains(c))
                .collect::<Vec<_>>();
            let mut board = placeholder.board;
            for index in query.board.len()..5 {
                let offset = index - query.board.len();
                let selected = offset + rng.index(deck.len() - offset);
                deck.swap(offset, selected);
                board[index] = deck[offset];
            }
            let deal = Deal::from_sampled_cards(holes, board);
            let uniforms = (0..128).map(|_| rng.next_f64()).collect::<Vec<_>>();
            for (index, action) in legal.iter().enumerate() {
                if exact_actions[index] {
                    continue;
                }
                let mut child = state.apply(action, &self.config.game);
                let mut step = 0;
                while child.terminal.is_none() {
                    let actions = child.legal_actions(&self.config.game);
                    let strategy = cash_evaluation::checked_strategy(
                        &self.policy,
                        &child,
                        &deal,
                        &actions,
                        &self.config.game,
                    )?;
                    let uniform = *uniforms
                        .get(step)
                        .ok_or("cash query rollout exceeded decision budget; no partial score")?;
                    let mut cumulative = 0.0;
                    let chosen = strategy
                        .iter()
                        .position(|p| {
                            cumulative += p;
                            uniform < cumulative
                        })
                        .unwrap_or(actions.len() - 1);
                    child = child.apply(&actions[chosen], &self.config.game);
                    step += 1;
                }
                let own = self.terminal_values(&child, &deal)?;
                residual = residual.max((own[0] + own[1] + own[2]).abs());
                moments[index].add(own[query.actor]);
            }
        }
        let probability_error = (probabilities.iter().sum::<f64>() - 1.0).abs();
        let action_ev_evaluations = exact_actions
            .iter()
            .map(|a| {
                if exact {
                    "exact-river-policy-expectation"
                } else if *a {
                    "exact-closed-all-in-expectation"
                } else {
                    "paired-full-hand-monte-carlo"
                }
            })
            .collect();
        let mixed = !exact && exact_actions.iter().any(|a| *a);
        let actions = legal
            .iter()
            .zip(probabilities)
            .zip(moments)
            .zip(exact_actions)
            .map(|(((action, probability), moment), exact_action)| {
                let (kind, amount_to_bb) =
                    practice_legal_action_identity(&state, action, &self.config.game);
                PracticePolicyAction {
                    kind,
                    amount_to_bb,
                    probability,
                    ev_bb: Some(moment.mean),
                    standard_error_bb: Some(if exact_action {
                        0.
                    } else {
                        moment.standard_error()
                    }),
                    confidence: "low",
                }
            })
            .collect();
        Ok(CashPolicyQueryResult { schema: "hu-cash-average-policy-result-v1", validation_status: "research_only",
            request_id: query.request_id.clone(), state_hash: query.state_hash.clone(), model_version: self.config.model_version.clone(),
            depth_bb: self.config.game.effective_stack_bb, rules_sha256: self.rules_sha256.clone(), network_sha256: input.network_sha256,
            actions, rollout_samples_per_action: if exact { 0 } else { self.config.rollout_samples },
            ev_evaluation: if exact { "exact-river-policy-expectation" } else if mixed { "paired-monte-carlo-with-exact-all-ins" } else { "paired-full-hand-monte-carlo" },
            exact_all_in_chance_outcomes: all_in_outcomes, action_ev_evaluations,
            exact_policy_tree_nodes: exact_nodes, opponent_combo_count: belief.len(), maximum_probability_sum_error: probability_error,
            maximum_accounting_residual_bb: residual, limitations: vec!["inactive research adapter; not an accepted website model", "EV standard error measures chance and policy sampling, not opponent-model error or equilibrium strength", "no online updates, legacy Home continuations, zero-sum projection or safe-resolving guarantee"] })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_typescript_cash_snapshots_have_exact_native_hashes() {
        let engine = engine();
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../data/practice/cash-query-fixtures.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let mut json = case["query"].clone();
            json["stateHash"] = case["stateHash"].clone();
            let query: PracticePolicyQuery = serde_json::from_value(json).unwrap();
            let canonical = canonical_cash_query_state(&query, &engine.config.game).unwrap();
            assert_eq!(
                format!("{:x}", Sha256::digest(canonical.as_bytes())),
                case["stateHash"].as_str().unwrap()
            );
            engine
                .query(CashPolicyQuery {
                    schema: CASH_QUERY_SCHEMA.into(),
                    rules_sha256: engine.rules_sha256.clone(),
                    network_sha256: engine.policy.bundle_sha256().into(),
                    query,
                })
                .unwrap();
        }
    }
    pub(super) fn engine() -> CashPolicyEngine {
        let config = super::super::cash_training_tests::config();
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
            cash_rules: config.game.cash_rules.clone(),
            cash_depth_bb: Some(20.0),
            cash_action_abstraction: Some(config.game.action_abstraction.clone()),
        };
        let policy = FrozenPolicy::from_bundle(bundle, "a".repeat(64)).unwrap();
        let rules_sha256 = config.game.cash_rules.as_ref().unwrap().sha256().unwrap();
        CashPolicyEngine {
            config: CashPolicyEngineConfig {
                game: config.game,
                model_version: "cash-pilot-test".into(),
                network_path: "unused".into(),
                rollout_samples: 32,
                seed: 7,
            },
            policy,
            rules_sha256,
        }
    }
    fn input(
        engine: &CashPolicyEngine,
        actions: Vec<PracticeObservedAction>,
        board: Vec<u8>,
    ) -> CashPolicyQuery {
        let mut query = PracticePolicyQuery {
            request_id: "research-query".into(),
            state_hash: String::new(),
            model_version: engine.config.model_version.clone(),
            depth_bb: 20.0,
            private_cards: [48, 49],
            board,
            street: Street::Preflop,
            actor: 0,
            total_pot_bb: 1.4,
            stacks_bb: [19.6, 19.0],
            street_bets_bb: [0.4, 1.],
            total_committed_bb: [0.4, 1.],
            last_full_raise_bb: 1.0,
            raise_reopened: true,
            actions,
        };
        let state = replay_practice_query(&query, &engine.config.game).unwrap();
        query.actor = state.actor;
        query.street = state.street;
        query.total_pot_bb = state.pot();
        query.stacks_bb = std::array::from_fn(|p| state.remaining(p, &engine.config.game));
        query.street_bets_bb = state.street_invested;
        query.total_committed_bb = state.invested;
        query.last_full_raise_bb = state.last_full_raise;
        query.raise_reopened = state.raise_reopened;
        query.state_hash = format!(
            "{:x}",
            Sha256::digest(
                canonical_cash_query_state(&query, &engine.config.game)
                    .unwrap()
                    .as_bytes()
            )
        );
        CashPolicyQuery {
            schema: CASH_QUERY_SCHEMA.into(),
            rules_sha256: engine.rules_sha256.clone(),
            network_sha256: engine.policy.bundle_sha256().into(),
            query,
        }
    }
    #[test]
    fn cash_query_replays_all_streets_returns_own_values_and_rejects_cross_profile_or_snapshot() {
        let engine = engine();
        let mut query = input(&engine, vec![], vec![]);
        let first = engine.query(query.clone()).unwrap();
        let fold = first
            .actions
            .iter()
            .find(|a| a.kind == PracticePolicyActionKind::Fold)
            .unwrap();
        assert_eq!(fold.ev_bb, Some(-0.4));
        assert_eq!(fold.standard_error_bb, Some(0.0));
        assert!(first.maximum_accounting_residual_bb < 1e-8);
        query.query.request_id = "different-request".into();
        assert_eq!(
            serde_json::to_value(&first.actions).unwrap(),
            serde_json::to_value(engine.query(query.clone()).unwrap().actions).unwrap()
        );
        query.rules_sha256 = "0".repeat(64);
        assert!(engine.query(query).is_err());
        let mut query = input(&engine, vec![], vec![]);
        query.query.total_pot_bb += 0.04;
        assert!(engine.query(query).is_err());
        let mut actions = vec![
            PracticeObservedAction {
                actor: 0,
                street: Street::Preflop,
                kind: PracticePolicyActionKind::Call,
                amount_to_bb: None,
            },
            PracticeObservedAction {
                actor: 1,
                street: Street::Preflop,
                kind: PracticePolicyActionKind::Check,
                amount_to_bb: None,
            },
        ];
        for (street, board) in [
            (Street::Flop, vec![8, 13, 22]),
            (Street::Turn, vec![8, 13, 22, 31]),
            (Street::River, vec![8, 13, 22, 31, 40]),
        ] {
            let result = engine
                .query(input(&engine, actions.clone(), board))
                .unwrap();
            assert!(result.actions.iter().all(|a| a.ev_bb.unwrap().is_finite()
                && a.standard_error_bb.unwrap().is_finite()
                && a.confidence == "low"));
            actions.push(PracticeObservedAction {
                actor: 1,
                street,
                kind: PracticePolicyActionKind::Check,
                amount_to_bb: None,
            });
            actions.push(PracticeObservedAction {
                actor: 0,
                street,
                kind: PracticePolicyActionKind::Check,
                amount_to_bb: None,
            });
        }
    }
    #[test]
    fn cash_canonical_hash_has_shared_blinds_and_call_payment_not_home_identity() {
        let engine = engine();
        let root = input(&engine, vec![], vec![]);
        let rules = engine
            .config
            .game
            .cash_rules
            .as_ref()
            .unwrap()
            .canonical_string()
            .unwrap();
        assert_eq!(canonical_cash_query_state(&root.query,&engine.config.game).unwrap(),format!("hu-cash-v3|{rules}|cash-pilot-test|20.000|button-small-blind|preflop|48,49||0.000|0.400|1.000|19.600|19.000|"));
        let query = input(
            &engine,
            vec![PracticeObservedAction {
                actor: 0,
                street: Street::Preflop,
                kind: PracticePolicyActionKind::Call,
                amount_to_bb: None,
            }],
            vec![],
        );
        assert!(
            canonical_cash_query_state(&query.query, &engine.config.game)
                .unwrap()
                .ends_with("preflop:button-small-blind:call:0.6")
        );
        let mut bad = query;
        bad.query.private_cards = [48, 48];
        assert!(engine.query(bad).is_err());
    }

    #[test]
    fn exact_river_call_accounts_for_odd_cent_and_is_independent_of_sampling_budget() {
        let mut engine = engine();
        let mut actions = vec![
            PracticeObservedAction {
                actor: 0,
                street: Street::Preflop,
                kind: PracticePolicyActionKind::Call,
                amount_to_bb: None,
            },
            PracticeObservedAction {
                actor: 1,
                street: Street::Preflop,
                kind: PracticePolicyActionKind::Check,
                amount_to_bb: None,
            },
        ];
        for street in [Street::Flop, Street::Turn] {
            for actor in [1, 0] {
                actions.push(PracticeObservedAction {
                    actor,
                    street,
                    kind: PracticePolicyActionKind::Check,
                    amount_to_bb: None,
                });
            }
        }
        actions.push(PracticeObservedAction {
            actor: 1,
            street: Street::River,
            kind: PracticePolicyActionKind::AllIn,
            amount_to_bb: Some(19.),
        });
        let mut query = input(&engine, actions, vec![32, 36, 40, 44, 48]);
        query.query.private_cards = [49, 50]; // Board is the club royal flush: every showdown ties.
        query.query.state_hash = format!(
            "{:x}",
            Sha256::digest(
                canonical_cash_query_state(&query.query, &engine.config.game)
                    .unwrap()
                    .as_bytes()
            )
        );
        let result = engine.query(query.clone()).unwrap();
        assert_eq!(result.ev_evaluation, "exact-river-policy-expectation");
        assert_eq!(result.rollout_samples_per_action, 0);
        assert_eq!(result.opponent_combo_count, 990);
        assert!(result.maximum_accounting_residual_bb < 1e-8);
        let call = result
            .actions
            .iter()
            .find(|a| a.kind == PracticePolicyActionKind::Call)
            .unwrap();
        assert!((call.ev_bb.unwrap() - -0.92).abs() < 1e-9);
        assert_eq!(call.standard_error_bb, Some(0.));
        engine.config.seed = 123;
        engine.config.rollout_samples = 4096;
        assert_eq!(
            serde_json::to_value(result.actions).unwrap(),
            serde_json::to_value(engine.query(query).unwrap().actions).unwrap()
        );
    }

    #[test]
    fn exact_turn_all_in_call_has_no_sampling_noise_and_keeps_model_confidence_low() {
        let mut engine = engine();
        let mut actions = vec![
            PracticeObservedAction {
                actor: 0,
                street: Street::Preflop,
                kind: PracticePolicyActionKind::Call,
                amount_to_bb: None,
            },
            PracticeObservedAction {
                actor: 1,
                street: Street::Preflop,
                kind: PracticePolicyActionKind::Check,
                amount_to_bb: None,
            },
        ];
        for actor in [1, 0] {
            actions.push(PracticeObservedAction {
                actor,
                street: Street::Flop,
                kind: PracticePolicyActionKind::Check,
                amount_to_bb: None,
            });
        }
        actions.push(PracticeObservedAction {
            actor: 1,
            street: Street::Turn,
            kind: PracticePolicyActionKind::AllIn,
            amount_to_bb: Some(19.),
        });
        let query = input(&engine, actions, vec![8, 13, 22, 31]);
        let first = engine.query(query.clone()).unwrap();
        assert_eq!(first.ev_evaluation, "paired-monte-carlo-with-exact-all-ins");
        assert_eq!(first.exact_all_in_chance_outcomes, 1035 * 44);
        let index = first
            .actions
            .iter()
            .position(|a| a.kind == PracticePolicyActionKind::Call)
            .unwrap();
        assert_eq!(
            first.action_ev_evaluations[index],
            "exact-closed-all-in-expectation"
        );
        assert_eq!(first.actions[index].standard_error_bb, Some(0.));
        assert_eq!(first.actions[index].confidence, "low");
        engine.config.rollout_samples = 4096;
        engine.config.seed = 123;
        let second = engine.query(query).unwrap();
        assert_eq!(first.actions[index].ev_bb, second.actions[index].ev_bb);
        assert!(second.maximum_accounting_residual_bb < 1e-8);
    }

    #[test]
    fn cash_query_hash_preserves_public_card_arrivals_not_flop_permutations() {
        let engine = engine();
        let mut actions = vec![
            PracticeObservedAction {
                actor: 0,
                street: Street::Preflop,
                kind: PracticePolicyActionKind::Call,
                amount_to_bb: None,
            },
            PracticeObservedAction {
                actor: 1,
                street: Street::Preflop,
                kind: PracticePolicyActionKind::Check,
                amount_to_bb: None,
            },
        ];
        for street in [Street::Flop, Street::Turn] {
            for actor in [1, 0] {
                actions.push(PracticeObservedAction {
                    actor,
                    street,
                    kind: PracticePolicyActionKind::Check,
                    amount_to_bb: None,
                });
            }
        }
        let mut query = input(&engine, actions, vec![8, 13, 22, 31, 40]).query;
        let first = canonical_cash_query_state(&query, &engine.config.game).unwrap();
        query.board.swap(2, 3);
        assert_ne!(
            first,
            canonical_cash_query_state(&query, &engine.config.game).unwrap()
        );
        query.board = vec![22, 8, 13, 31, 40];
        assert_eq!(
            first,
            canonical_cash_query_state(&query, &engine.config.game).unwrap()
        );
    }
}
