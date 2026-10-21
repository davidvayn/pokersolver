//! Offline qualification only. Never changes a deployed policy or resolver.
use super::*;

pub struct FrozenValueConfig {
    pub game: BlueprintConfig,
    pub networks: PathBuf,
    pub snapshots: Option<PathBuf>,
    pub input: PathBuf,
    pub seed: u64,
    pub hidden_blocks: usize,
    pub rollouts_per_block: usize,
    pub policy_only: bool,
    pub integrate_terminal_responses: bool,
    pub exact_river_values: bool,
    pub exact_closed_values: bool,
    pub action_expectation_depth: usize,
}

#[derive(Deserialize)]
struct SnapshotSpec {
    path: PathBuf,
    sha256: String,
    weight: f64,
    #[serde(default)]
    weights_by_player: Option<[f64; 2]>,
}

struct StudyPolicy {
    models: Vec<(FrozenPolicy, [f64; 2])>,
    source_hashes: Vec<String>,
    snapshot_average: bool,
}

fn matching_action(
    state: &GameState,
    observed: &CompactTrajectoryAction,
    game: &BlueprintConfig,
) -> Result<LegalAction, String> {
    if state.actor != observed.actor || state.street != observed.street || state.terminal.is_some()
    {
        return Err("qualification history has an invalid actor/street".into());
    }
    state
        .legal_actions(game)
        .into_iter()
        .find(|action| {
            let actual = compact_action(state, action, game);
            actual.kind == observed.kind
                && match (actual.amount_to_bb, observed.amount_to_bb) {
                    (None, None) => true,
                    (Some(a), Some(b)) => (a - b).abs() <= 0.001,
                    _ => false,
                }
        })
        .ok_or_else(|| "qualification history contains an unmodeled action".into())
}

impl StudyPolicy {
    fn load(config: &FrozenValueConfig) -> Result<Self, Box<dyn Error>> {
        let specs: Vec<SnapshotSpec> = if let Some(path) = &config.snapshots {
            serde_json::from_slice(&fs::read(path)?)?
        } else {
            vec![SnapshotSpec {
                path: config.networks.clone(),
                sha256: sha256_path(&config.networks)?,
                weight: 1.0,
                weights_by_player: None,
            }]
        };
        if specs.is_empty() || specs.len() > 128 {
            return Err("snapshot reference requires 1..128 frozen models".into());
        }
        let mut models = Vec::new();
        let mut source_hashes = Vec::new();
        for spec in specs {
            let source: serde_json::Value = serde_json::from_slice(&fs::read(&spec.path)?)?;
            let identity = source
                .get("qualification_game")
                .ok_or("qualification network lacks game identity")?;
            let declared: ActionAbstraction = serde_json::from_value(
                identity
                    .get("action_abstraction")
                    .cloned()
                    .ok_or("qualification network lacks action grid")?,
            )?;
            // JSON integer and float spellings represent the same numeric game.
            // Normalize through the typed grid before exact structural comparison.
            if identity.get("depth_bb").and_then(|depth| depth.as_f64())
                != Some(config.game.effective_stack_bb)
                || serde_json::to_value(&declared)?
                    != serde_json::to_value(&config.game.action_abstraction)?
            {
                return Err(
                    "qualification network lacks matching depth/action-grid identity".into(),
                );
            }
            let weights = spec.weights_by_player.unwrap_or([spec.weight; 2]);
            if weights
                .iter()
                .any(|weight| !weight.is_finite() || *weight <= 0.0)
                || sha256_path(&spec.path)? != spec.sha256
            {
                return Err("snapshot reference weight/hash is invalid".into());
            }
            let policy = FrozenPolicy::load(&spec.path)?;
            policy.bundle.validate_game(&config.game)?;
            // Qualification must evaluate the requested policy, not a training
            // control variate, range resolver, or an accidental hidden route.
            if policy.bundle.sampling_baseline.is_some() {
                return Err("qualification requires policy-only networks".into());
            }
            source_hashes.push(spec.sha256);
            models.push((policy, weights));
        }
        Ok(Self {
            models,
            source_hashes,
            snapshot_average: config.snapshots.is_some(),
        })
    }

    fn strategy(
        &self,
        state: &GameState,
        deal: &Deal,
        actions: &[LegalAction],
        game: &BlueprintConfig,
    ) -> Result<Vec<f64>, String> {
        let mut result = vec![0.0; actions.len()];
        let mut masses = Vec::with_capacity(self.models.len());
        for (model, weight) in &self.models {
            let mut log_mass = weight[state.actor].ln();
            if self.snapshot_average {
                let mut prefix = GameState::initial(game);
                for observed in &state.trajectory {
                    let action = matching_action(
                        &prefix,
                        &CompactTrajectoryAction {
                            actor: observed.actor,
                            street: observed.street,
                            kind: observed.kind,
                            amount_bb: observed.amount_bb as f32,
                            amount_to_bb: observed.amount_to_bb.map(|v| v as f32),
                            pot_after_bb: observed.pot_after_bb as f32,
                        },
                        game,
                    )?;
                    if prefix.actor == state.actor {
                        let legal = prefix.legal_actions(game);
                        let index = legal
                            .iter()
                            .position(|candidate| candidate.label == action.label)
                            .ok_or("snapshot action missing")?;
                        let probability = model.strategy(&prefix, deal, &legal, game)[index];
                        log_mass += probability.ln();
                    }
                    prefix = prefix.apply(&action, game);
                }
            }
            masses.push(log_mass);
        }
        let maximum = masses.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !maximum.is_finite() {
            return Err("snapshot average has zero own reach at this information set".into());
        }
        let mut total = 0.0;
        for ((model, _), log_mass) in self.models.iter().zip(masses) {
            let mass = (log_mass - maximum).exp();
            let row = model.strategy(state, deal, actions, game);
            for (out, probability) in result.iter_mut().zip(row) {
                *out += mass * probability;
            }
            total += mass;
        }
        for probability in &mut result {
            *probability /= total;
        }
        if result.iter().any(|p| !p.is_finite() || *p < 0.0)
            || (result.iter().sum::<f64>() - 1.0).abs() > 1e-6
        {
            return Err("qualification policy probabilities are invalid".into());
        }
        Ok(result)
    }
}

fn complete_deal(compact: &CompactState, opponent: [u8; 2], rng: &mut SplitMix64) -> Deal {
    let mut holes = [[0; 2]; 2];
    holes[compact.actor] = compact.private_cards;
    holes[1 - compact.actor] = opponent;
    let mut deck = (0..52u8)
        .filter(|card| {
            !holes
                .iter()
                .flatten()
                .chain(&compact.board)
                .any(|used| used == card)
        })
        .collect::<Vec<_>>();
    let mut board = [0; 5];
    board[..compact.board.len()].copy_from_slice(&compact.board);
    for (offset, slot) in board[compact.board.len()..].iter_mut().enumerate() {
        let next = offset + rng.index(deck.len() - offset);
        deck.swap(offset, next);
        *slot = deck[offset];
    }
    Deal::from_sampled_cards(holes, board)
}

fn replay(compact: &CompactState, game: &BlueprintConfig) -> Result<GameState, String> {
    let mut state = GameState::initial(game);
    for observed in &compact.trajectory {
        let action = matching_action(&state, observed, game)?;
        state = state.apply(&action, game);
        if (state.pot() - f64::from(observed.pot_after_bb)).abs() > 0.001 {
            return Err("qualification history pot accounting differs".into());
        }
    }
    if state.terminal.is_some()
        || state.actor != compact.actor
        || state.street != compact.street
        || state
            .invested
            .iter()
            .zip(compact.total_committed_bb)
            .any(|(a, b)| (a - f64::from(b)).abs() > 0.001)
        || state
            .street_invested
            .iter()
            .zip(compact.street_bets_bb)
            .any(|(a, b)| (a - f64::from(b)).abs() > 0.001)
        || state.raise_reopened != compact.raise_reopened
        || (state.last_full_raise - f64::from(compact.last_full_raise_bb)).abs() > 0.001
    {
        return Err("qualification replay does not reconstruct the requested state".into());
    }
    reconstruct_attribution_state(compact, game)?; // Cards and redundant chip snapshot.
    Ok(state)
}

fn opponent_posterior(
    compact: &CompactState,
    policy: &StudyPolicy,
    game: &BlueprintConfig,
) -> Result<(Vec<[u8; 2]>, Vec<f64>), String> {
    let mut combos = Vec::new();
    let mut log_weights = Vec::new();
    for combo in all_combos() {
        let cards = combo.cards();
        if cards
            .iter()
            .any(|card| compact.private_cards.contains(card) || compact.board.contains(card))
        {
            continue;
        }
        let deal = complete_deal(compact, cards, &mut SplitMix64::new(0));
        let mut prefix = GameState::initial(game);
        let mut log_weight = 0.0;
        for observed in &compact.trajectory {
            let action = matching_action(&prefix, observed, game)?;
            if prefix.actor != compact.actor {
                let legal = prefix.legal_actions(game);
                let index = legal
                    .iter()
                    .position(|candidate| candidate.label == action.label)
                    .ok_or("posterior action missing")?;
                log_weight += policy.strategy(&prefix, &deal, &legal, game)?[index].ln();
                if !log_weight.is_finite() {
                    break;
                }
            }
            prefix = prefix.apply(&action, game);
        }
        combos.push(cards);
        log_weights.push(log_weight);
    }
    let maximum = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !maximum.is_finite() {
        return Err("observed history has zero opponent reach under the frozen policy".into());
    }
    let mut weights = log_weights
        .iter()
        .map(|w| (w - maximum).exp())
        .collect::<Vec<_>>();
    let sum = weights.iter().sum::<f64>();
    for weight in &mut weights {
        *weight /= sum;
    }
    Ok((combos, weights))
}

fn rollout(
    mut state: GameState,
    deal: &Deal,
    policy: &StudyPolicy,
    game: &BlueprintConfig,
    actor: usize,
    rng: &mut SplitMix64,
    integrate_terminal_responses: bool,
) -> Result<f64, String> {
    for _ in 0..MAX_TRAJECTORY_ACTIONS + 1 {
        if state.terminal.is_some() {
            return Ok(state.complete_runout_utilities(deal, game)[actor]);
        }
        let actions = state.legal_actions(game);
        let strategy = policy.strategy(&state, deal, &actions, game)?;
        if integrate_terminal_responses {
            if let Some(value) = closed_terminal_value(&state, deal, &actions, &strategy, game, actor) {
                // Match the legacy draw at this node. There is no downstream
                // information set to skip; only terminal action noise is removed.
                let _ = sample_index(&strategy, rng);
                return Ok(value);
            }
        }
        state = state.apply(&actions[sample_index(&strategy, rng)], game);
    }
    Err("qualification rollout exceeded the modeled trajectory".into())
}

fn closed_terminal_value(state: &GameState, deal: &Deal, actions: &[LegalAction],
                         strategy: &[f64], game: &BlueprintConfig, actor: usize) -> Option<f64> {
    let mut value = 0.;
    for (action, probability) in actions.iter().zip(strategy) {
        let child = state.apply(action, game);
        child.terminal.as_ref()?;
        value += probability * child.complete_runout_utilities(deal, game)[actor];
    }
    Some(value)
}

fn prefix_branch_seed(seed: u64, index: usize) -> u64 {
    seed ^ (index as u64 + 1).wrapping_mul(0xa0761d6478bd642f)
}

/// Rao-Blackwellize a bounded prefix of future action draws, conditional on
/// the sampled hidden deal. Future boards/private cards remain sampled; this
/// never turns a sampled open-street value into an exact information-set EV.
fn prefix_expectation(state: GameState, deal: &Deal, policy: &StudyPolicy,
                      game: &BlueprintConfig, actor: usize, seed: u64, depth: usize,
                      integrate_terminal_responses: bool) -> Result<f64,String> {
    if depth > 2 { return Err("action expectation depth must be 0..2".into()); }
    if depth == 0 {
        return rollout(state,deal,policy,game,actor,&mut SplitMix64::new(seed),integrate_terminal_responses);
    }
    if state.terminal.is_some() { return Ok(state.complete_runout_utilities(deal,game)[actor]); }
    let actions = state.legal_actions(game);
    let probabilities = policy.strategy(&state,deal,&actions,game)?;
    let mut value = 0.;
    for (index,(action,p)) in actions.iter().zip(probabilities).enumerate() {
        if p > 0. {
            value += p * prefix_expectation(state.apply(action,game),deal,policy,game,actor,
                prefix_branch_seed(seed,index),depth-1,integrate_terminal_responses)?;
        }
    }
    Ok(value)
}

fn column_summary(blocks: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let count = blocks.len() as f64;
    let means = (0..blocks[0].len())
        .map(|a| blocks.iter().map(|b| b[a]).sum::<f64>() / count)
        .collect::<Vec<_>>();
    let errors = means
        .iter()
        .enumerate()
        .map(|(a, mean)| {
            (blocks.iter().map(|b| (b[a] - mean).powi(2)).sum::<f64>() / (count * (count - 1.0)))
                .sqrt()
        })
        .collect();
    (means, errors)
}

/// Exhaustive action expectation on a known river. No private information is
/// passed into an actor's policy beyond its own ordinary information set.
fn exact_river_tree(state: GameState, deal: &Deal, policy: &StudyPolicy,
                    game: &BlueprintConfig, actor: usize, nodes: &mut usize) -> Result<f64, String> {
    *nodes += 1;
    if *nodes > 2_000_000 { return Err("exact river label exceeded two million nodes".into()); }
    if state.terminal.is_some() {
        return Ok(state.complete_runout_utilities(deal, game)[actor]);
    }
    if state.street != Street::River { return Err("exact river evaluator received an open chance node".into()); }
    let actions = state.legal_actions(game);
    let probabilities = policy.strategy(&state, deal, &actions, game)?;
    let mut value = 0.;
    for (action, probability) in actions.iter().zip(probabilities) {
        if probability > 0. {
            value += probability * exact_river_tree(state.apply(action, game), deal, policy, game, actor, nodes)?;
        }
    }
    Ok(value)
}

fn exact_river_action_values(state: &GameState, compact: &CompactState, actions: &[LegalAction],
    combos: &[[u8;2]], weights: &[f64], policy: &StudyPolicy, game: &BlueprintConfig) -> Result<(Vec<f64>,usize),String> {
    if state.street != Street::River { return Err("exact river labels require a complete visible board".into()); }
    let mut nodes = 0;
    let mut values = vec![0.;actions.len()];
    for (opponent, weight) in combos.iter().zip(weights) {
        if *weight <= 0. { continue; }
        let deal = complete_deal(compact, *opponent, &mut SplitMix64::new(0));
        for (index, action) in actions.iter().enumerate() {
            values[index] += weight * exact_river_tree(state.apply(action,game), &deal, policy, game, state.actor, &mut nodes)?;
        }
    }
    Ok((values,nodes))
}

fn is_closed_postflop(state: &GameState, visible: usize, game: &BlueprintConfig) -> bool {
    match state.terminal {
        Some(Terminal::Fold { .. }) => true,
        Some(Terminal::Showdown) => matches!(visible,3|4|5),
        None => state.legal_actions(game).iter().all(|a| {
            match state.apply(a,game).terminal {
                Some(Terminal::Fold { .. }) => true,
                Some(Terminal::Showdown) => matches!(visible,3|4|5),
                None => false,
            }
        }),
    }
}

/// Only finite closed leaves. Caller probabilities are evaluated before future
/// cards arrive, then every legal runout is integrated with exact card removal.
fn exact_closed_tree(state: &GameState, deal: &Deal, visible: &[u8], policy: &StudyPolicy,
    game: &BlueprintConfig, actor: usize, outcomes: &mut usize) -> Result<f64,String> {
    if !is_closed_postflop(state,visible.len(),game) { return Err("exact closed label received an open tree".into()); }
    match state.terminal {
        Some(Terminal::Fold { .. }) => Ok(state.complete_runout_utilities(deal,game)[actor]),
        Some(Terminal::Showdown) => {
            let available = (0..52u8).filter(|c| !visible.contains(c) && !deal.holes.iter().flatten().any(|h| h==c)).collect::<Vec<_>>();
            let count = match visible.len() { 3 => available.len()*(available.len()-1)/2, 4 => available.len(), 5 => 1, _ => return Err("unsupported exact chance fanout".into()) };
            *outcomes += count;
            if *outcomes > 2_000_000 { return Err("exact closed label exceeded two million runouts".into()); }
            let mut board = [0u8;5]; board[..visible.len()].copy_from_slice(visible);
            let mut value = 0.;
            let mut observe = |board| {
                let complete = Deal::from_sampled_cards(deal.holes,board);
                value += state.complete_runout_utilities(&complete,game)[actor] / count as f64;
            };
            match visible.len() {
                3 => for i in 0..available.len() { for j in i+1..available.len() {
                    board[3]=available[i]; board[4]=available[j]; observe(board);
                }},
                4 => for card in available { board[4]=card; observe(board); },
                5 => observe(board),
                _ => unreachable!(),
            }
            Ok(value)
        },
        None => {
            let actions = state.legal_actions(game);
            let probabilities = policy.strategy(state,deal,&actions,game)?;
            let mut value = 0.;
            for (action,p) in actions.iter().zip(probabilities) {
                if p>0. { value += p * exact_closed_tree(&state.apply(action,game),deal,visible,policy,game,actor,outcomes)?; }
            }
            Ok(value)
        },
    }
}

fn exact_closed_action_values(state: &GameState, compact: &CompactState, actions: &[LegalAction],
    combos: &[[u8;2]], weights: &[f64], policy: &StudyPolicy, game: &BlueprintConfig) -> Result<(Vec<Option<f64>>,usize),String> {
    let mut outcomes = 0;
    let mut values = Vec::new();
    for action in actions {
        let child = state.apply(action,game);
        if !is_closed_postflop(&child,compact.board.len(),game) { values.push(None); continue; }
        let mut value = 0.;
        for (opponent,p) in combos.iter().zip(weights) {
            if *p>0. {
                let deal = complete_deal(compact,*opponent,&mut SplitMix64::new(0));
                value += p * exact_closed_tree(&child,&deal,&compact.board,policy,game,state.actor,&mut outcomes)?;
            }
        }
        values.push(Some(value));
    }
    Ok((values,outcomes))
}

pub fn qualify_frozen_values(
    config: FrozenValueConfig,
) -> Result<serde_json::Value, Box<dyn Error>> {
    config.game.validate()?;
    config.game.require_legacy_home()?;
    if config.action_expectation_depth > 2 {
        return Err("action expectation depth must be 0..2".into());
    }
    if !config.policy_only
        && (config.hidden_blocks < 4
            || config.hidden_blocks > 1_000_000
            || config.rollouts_per_block == 0
            || config.rollouts_per_block > 1024)
    {
        return Err(
            "qualification requires 4..1000000 hidden blocks and 1..1024 inner rollouts".into(),
        );
    }
    let policy = StudyPolicy::load(&config)?;
    let input_bytes = fs::read(&config.input)?;
    let records: Vec<TrainingSample> = serde_json::from_slice(&input_bytes)?;
    if records.is_empty() || records.len() > 10000 {
        return Err("qualification requires 1..10000 decisions".into());
    }
    if !config.policy_only
        && records
            .len()
            .checked_mul(config.hidden_blocks)
            .is_none_or(|n| n > 1_000_000)
    {
        return Err("qualification output exceeds one million retained hidden blocks".into());
    }
    let mut output = Vec::new();
    for (index, record) in records.iter().enumerate() {
        let state = replay(&record.state, &config.game)?;
        let legal = state.legal_actions(&config.game);
        let synthetic = reconstruct_attribution_state(&record.state, &config.game)?.1;
        let expected = legal
            .iter()
            .map(|action| compact_action(&state, action, &config.game))
            .collect::<Vec<_>>();
        if serde_json::to_value(&expected)? != serde_json::to_value(&record.actions)? {
            return Err("qualification legal actions differ from the sampled grid".into());
        }
        let probabilities = policy.strategy(&state, &synthetic, &legal, &config.game)?;
        let mut blocks = Vec::new();
        let mut posterior_combos = 0;
        let mut exact = None;
        let mut closed = vec![None;legal.len()];
        let mut closed_outcomes = 0;
        if !config.policy_only {
            let (combos, weights) = opponent_posterior(&record.state, &policy, &config.game)?;
            posterior_combos = weights.iter().filter(|weight| **weight > 0.0).count();
            if config.exact_river_values && state.street == Street::River {
                exact = Some(exact_river_action_values(&state, &record.state, &legal, &combos, &weights, &policy, &config.game)?);
            } else if config.exact_closed_values {
                (closed,closed_outcomes) = exact_closed_action_values(&state,&record.state,&legal,&combos,&weights,&policy,&config.game)?;
            }
            for block in 0..if exact.is_some() || closed.iter().all(Option::is_some) { 0 } else { config.hidden_blocks } {
                let block_seed = config
                    .seed
                    .wrapping_add((index as u64).wrapping_mul(0x9e3779b97f4a7c15))
                    .wrapping_add((block as u64).wrapping_mul(0xd1b54a32d192ed03));
                let mut rng = SplitMix64::new(block_seed);
                let opponent = combos[sample_index(&weights, &mut rng)];
                let deal = complete_deal(&record.state, opponent, &mut rng);
                let mut values = vec![0.0; legal.len()];
                for inner in 0..config.rollouts_per_block {
                    // Shared hidden scenarios and random streams reduce noise
                    // in action differences without correlating outer blocks.
                    let seed = block_seed ^ (inner as u64 + 1).wrapping_mul(0x94d049bb133111eb);
                    for (action_index, action) in legal.iter().enumerate() {
                        if let Some(value) = closed[action_index] { values[action_index]=value; continue; }
                        values[action_index] += prefix_expectation(
                            state.apply(action, &config.game),
                            &deal,
                            &policy,
                            &config.game,
                            state.actor,
                            seed,
                            config.action_expectation_depth,
                            config.integrate_terminal_responses,
                        )? / config.rollouts_per_block as f64;
                    }
                }
                blocks.push(values);
            }
        }
        let (mut means, mut errors) = if let Some((values,_)) = &exact {
            (values.clone(), vec![0.;values.len()])
        } else if !config.policy_only && closed.iter().all(Option::is_some) {
            (closed.iter().map(|v| v.unwrap()).collect(),vec![0.;closed.len()])
        } else if blocks.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            column_summary(&blocks)
        };
        if !config.policy_only && exact.is_none() {
            for (index,value) in closed.iter().enumerate() {
                if let Some(value) = value { means[index]=*value; errors[index]=0.; }
            }
        }
        output.push(serde_json::json!({"state":record.state,"actions":record.actions,"probabilities":probabilities,
            "action_values_bb":means,"standard_errors_bb":errors,"independent_hidden_block_values_bb":blocks,
            "estimator":if exact.is_some() { "exact_river_posterior_and_action_tree_v1" }
                else if closed.iter().all(Option::is_some) { "exact_closed_posterior_runouts_v1" }
                else if closed.iter().any(Option::is_some) { "mixed_exact_closed_and_hidden_blocks_v1" }
                else { "independent_hidden_blocks_v1" },
            "exact_tree_nodes":exact.as_ref().map(|(_,nodes)| *nodes),
            "action_estimators":closed.iter().map(|v| if exact.is_some() { "exact_river_posterior_and_action_tree_v1" }
                else if v.is_some() { "exact_closed_posterior_runouts_v1" } else { "independent_hidden_blocks_v1" }).collect::<Vec<_>>(),
            "exact_closed_runouts":closed_outcomes,
            "posterior_supported_combos":posterior_combos}));
    }
    Ok(
        serde_json::json!({"schema":"hu-frozen-information-set-values-v1","depth_bb":config.game.effective_stack_bb,
        "action_abstraction":config.game.action_abstraction,"source_network_sha256s":policy.source_hashes,
        "input_sha256":format!("{:x}",Sha256::digest(&input_bytes)),"seed":config.seed,
        "hidden_blocks":config.hidden_blocks,"rollouts_per_block":config.rollouts_per_block,
        "policy_only":config.policy_only,"snapshot_average":policy.snapshot_average,"records":output,
        "integrates_terminal_responses":config.integrate_terminal_responses,
        "exact_river_values":config.exact_river_values,
        "exact_closed_values":config.exact_closed_values,
        "action_expectation_depth":config.action_expectation_depth,
        "uncertainty_contract":"Independent outer opponent-posterior/future-board blocks for sampled actions; SE includes inner rollout noise and excludes neural approximation error. Explicit exact action estimators enumerate the full posterior and applicable action/chance trees; they have zero Monte Carlo SE, not zero neural error. Fully exact decisions contain no sampled blocks.",
        "qualification":"research_only_not_an_exploitability_certificate"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_policy() -> StudyPolicy {
        let scorer = DenseScorer {
            layers: vec![DenseLayer {
                input_size: MODEL_INPUT_COUNT,
                output_size: 1,
                activation: DenseActivation::Linear,
                weights: vec![0.; MODEL_INPUT_COUNT],
                biases: vec![0.],
            }],
        };
        let bundle: TrainingNetworkBundle = serde_json::from_value(serde_json::json!({"schema":TRAINING_NETWORK_SCHEMA,
            "input_size":MODEL_INPUT_COUNT,"strategy_transform":"softmax","networks":[scorer,scorer]})).unwrap();
        StudyPolicy {
            models: vec![(
                FrozenPolicy::from_bundle(bundle, "fixture".into()).unwrap(),
                [1.; 2],
            )],
            source_hashes: vec![],
            snapshot_average: false,
        }
    }

    #[test]
    fn exact_river_values_enumerate_the_posterior_and_terminal_policy_mix() {
        let game = BlueprintConfig::default();
        let deal = Deal::sample(&mut SplitMix64::new(2919011));
        let policy = uniform_policy();
        let mut state = GameState::initial(&game);
        while state.street != Street::River {
            let passive = state.legal_actions(&game).into_iter()
                .find(|a| matches!(a.kind, ActionKind::Call | ActionKind::Check)).unwrap();
            state = state.apply(&passive,&game);
        }
        let shove = state.legal_actions(&game).into_iter().find(|a| matches!(a.kind, ActionKind::RaiseTo(v)
            if (v - state.street_invested[state.actor] - state.remaining(state.actor,&game)).abs() < EPSILON)).unwrap();
        state = state.apply(&shove,&game);
        let compact = compact_state(&state,&deal,&game);
        let actions = state.legal_actions(&game);
        let (combos,weights) = opponent_posterior(&compact,&policy,&game).unwrap();
        assert_eq!(combos.len(),990);
        let (values,nodes) = exact_river_action_values(&state,&compact,&actions,&combos,&weights,&policy,&game).unwrap();
        assert_eq!(nodes,990*actions.len());
        for (i,action) in actions.iter().enumerate() {
            let expected:f64 = combos.iter().zip(&weights).map(|(cards,p)| {
                let exact_deal = complete_deal(&compact,*cards,&mut SplitMix64::new(0));
                p*state.apply(action,&game).complete_runout_utilities(&exact_deal,&game)[state.actor]
            }).sum();
            assert!((values[i]-expected).abs() < 1e-12);
        }
        let expected_mix:f64 = actions.iter().map(|a|
            state.apply(a,&game).complete_runout_utilities(&deal,&game)[state.actor] / actions.len() as f64).sum();
        let exact_mix = exact_river_tree(state.clone(),&deal,&policy,&game,state.actor,&mut 0).unwrap();
        assert!((exact_mix-expected_mix).abs() < 1e-12);
        let initial = GameState::initial(&game);
        assert!(exact_river_tree(initial,&deal,&policy,&game,0,&mut 0).is_err());
        assert!(exact_river_tree(state.clone(),&deal,&policy,&game,0,&mut 2_000_000).is_err());
    }

    #[test]
    fn closed_turn_labels_integrate_runouts_without_future_leakage_or_open_tree_shortcuts() {
        let game = BlueprintConfig::default();
        let deal = Deal::sample(&mut SplitMix64::new(2919051));
        let policy = uniform_policy();
        let mut state = GameState::initial(&game);
        while state.street != Street::Turn {
            let action=state.legal_actions(&game).into_iter().find(|a| matches!(a.kind,ActionKind::Call|ActionKind::Check)).unwrap();
            state=state.apply(&action,&game);
        }
        assert!(!is_closed_postflop(&state,4,&game));
        assert!(exact_closed_tree(&state,&deal,&deal.board[..4],&policy,&game,0,&mut 0).is_err());
        let shove=state.legal_actions(&game).into_iter().find(|a| matches!(a.kind,ActionKind::RaiseTo(v)
            if (v-state.street_invested[state.actor]-state.remaining(state.actor,&game)).abs()<EPSILON)).unwrap();
        let response=state.apply(&shove,&game);
        let actor=state.actor;
        let mut outcomes=0;
        let actual=exact_closed_tree(&response,&deal,&deal.board[..4],&policy,&game,actor,&mut outcomes).unwrap();
        assert_eq!(outcomes,44);
        let actions=response.legal_actions(&game);
        let mut expected=0.;
        for river in 0..52u8 {
            if deal.holes.iter().flatten().any(|h| *h==river) || deal.board[..4].contains(&river) { continue; }
            let mut board=deal.board; board[4]=river;
            let complete=Deal::from_sampled_cards(deal.holes,board);
            for action in &actions {
                expected += response.apply(action,&game).complete_runout_utilities(&complete,&game)[actor]/(44.*actions.len() as f64);
            }
            assert!((actual-exact_closed_tree(&response,&complete,&deal.board[..4],&policy,&game,actor,&mut 0).unwrap()).abs()<1e-10);
        }
        assert!((actual-expected).abs()<1e-10);
        assert!(exact_closed_tree(&response,&deal,&deal.board[..4],&policy,&game,actor,&mut 2_000_000).is_err());
        let initial=GameState::initial(&game);
        let preflop_shove=initial.legal_actions(&game).into_iter().find(|a| matches!(a.kind,ActionKind::RaiseTo(v) if (v-game.effective_stack_bb).abs()<EPSILON)).unwrap();
        assert!(!is_closed_postflop(&initial.apply(&preflop_shove,&game),0,&game));
    }

    #[test]
    fn root_posterior_removes_cards_and_complete_runouts_never_collide() {
        let game = BlueprintConfig::default();
        let deal = Deal::sample(&mut SplitMix64::new(17));
        let compact = compact_state(&GameState::initial(&game), &deal, &game);
        let (combos, weights) = opponent_posterior(&compact, &uniform_policy(), &game).unwrap();
        assert_eq!(combos.len(), 1225);
        assert!((weights.iter().sum::<f64>() - 1.).abs() < 1e-10);
        for cards in combos.iter().step_by(17) {
            let deal = complete_deal(&compact, *cards, &mut SplitMix64::new(19));
            let mut unique = deal
                .holes
                .iter()
                .flatten()
                .chain(&deal.board)
                .copied()
                .collect::<Vec<_>>();
            unique.sort();
            unique.dedup();
            assert_eq!(unique.len(), 9);
        }
    }

    #[test]
    fn action_prefix_expectation_matches_manual_policy_average_and_zero_is_legacy() {
        let game = BlueprintConfig::default();
        let deal = Deal::sample(&mut SplitMix64::new(2919061));
        let policy = uniform_policy();
        let state = GameState::initial(&game);
        let actions = state.legal_actions(&game);
        let p = policy.strategy(&state,&deal,&actions,&game).unwrap();
        let seed = 2919063;
        let expected:f64 = actions.iter().zip(p).enumerate().map(|(index,(action,p))|
            p * rollout(state.apply(action,&game),&deal,&policy,&game,0,
                &mut SplitMix64::new(prefix_branch_seed(seed,index)),true).unwrap()).sum();
        let actual = prefix_expectation(state.clone(),&deal,&policy,&game,0,seed,1,true).unwrap();
        assert!((actual-expected).abs()<1e-12);
        assert_eq!(prefix_expectation(state.clone(),&deal,&policy,&game,0,seed,0,true).unwrap(),
            rollout(state.clone(),&deal,&policy,&game,0,&mut SplitMix64::new(seed),true).unwrap());
        assert!(prefix_expectation(state,&deal,&policy,&game,0,seed,3,true).is_err());
    }

    #[test]
    fn outer_hidden_variance_is_not_lost_when_inner_rollouts_are_deterministic() {
        let (means, errors) =
            column_summary(&[vec![-10., 0.], vec![10., 0.], vec![-10., 0.], vec![10., 0.]]);
        assert_eq!(means, vec![0., 0.]);
        assert!(errors[0] > 5.);
        assert_eq!(errors[1], 0.);
        let p = vec![0.5, 0.5];
        let a = instantaneous_regrets(&p, &[10., 0.]);
        let b = instantaneous_regrets(&p, &[-10., 0.]);
        let averaged = instantaneous_regrets(&p, &[0., 0.]);
        assert_eq!(averaged, vec![0., 0.]);
        assert_eq!(a[0] + b[0], 0.);
    }

    #[test]
    fn frozen_terminal_integration_is_exact_and_does_not_close_open_trees() {
        let game = BlueprintConfig::default();
        let deal = Deal::sample(&mut SplitMix64::new(2818011));
        let policy = uniform_policy();
        let initial = GameState::initial(&game);
        let actions = initial.legal_actions(&game);
        assert!(closed_terminal_value(&initial, &deal, &actions,
            &vec![1. / actions.len() as f64; actions.len()], &game, 0).is_none());
        let shove = actions.iter().find(|a| matches!(a.kind, ActionKind::RaiseTo(v)
            if (v - game.effective_stack_bb).abs() < EPSILON)).unwrap();
        let state = initial.apply(shove, &game);
        let responses = state.legal_actions(&game);
        let p = policy.strategy(&state, &deal, &responses, &game).unwrap();
        let expected: f64 = responses.iter().zip(&p).map(|(a, p)|
            p * state.apply(a, &game).complete_runout_utilities(&deal, &game)[0]).sum();
        let values: Vec<_> = (0..32).map(|seed|
            rollout(state.clone(), &deal, &policy, &game, 0,
                &mut SplitMix64::new(seed), true).unwrap()).collect();
        assert!(values.iter().all(|v| (*v - expected).abs() < 1e-12));
        assert!((0..32).any(|seed| (rollout(state.clone(), &deal, &policy, &game, 0,
            &mut SplitMix64::new(seed), false).unwrap() - expected).abs() > 1e-6));
    }

    #[test]
    fn home_terminal_response_integration_preserves_mean_without_closing_open_trees() {
        let game = BlueprintConfig::default();
        let deal = Deal::sample(&mut SplitMix64::new(17331));
        let mut state = GameState::initial(&game);
        while state.street != Street::River {
            let action = state
                .legal_actions(&game)
                .into_iter()
                .find(|action| matches!(action.kind, ActionKind::Call | ActionKind::Check))
                .unwrap();
            state = state.apply(&action, &game);
        }
        let all_in_to = state.street_invested[state.actor] + state.remaining(state.actor, &game);
        let allin = state.legal_actions(&game).into_iter().find(|action| {
            matches!(action.kind, ActionKind::RaiseTo(amount) if (amount - all_in_to).abs() < 1e-8)
        }).unwrap();
        state = state.apply(&allin, &game);
        let config = SampleGenerationConfig {
            game: game.clone(),
            traversals: 2,
            start_iteration: 0,
            seed: 17,
            max_records: 1000,
            output: PathBuf::new(),
            network_path: None,
            trajectory_sampling: false,
            evaluate_trajectory_values: false,
            value_rollouts_per_action: 2,
            enumerate_turn_river_chance: false,
        };
        let mut generator = SampleGenerator::new(config).unwrap();
        let actions = state.legal_actions(&game);
        let strategy = vec![1. / actions.len() as f64; actions.len()];
        assert!(generator
            .cash_closed_terminal_expectation(&state, &deal, &actions, &strategy, state.actor)
            .is_none());
        generator.integrate_home_closed_actions = true;
        let exact = generator
            .cash_closed_terminal_expectation(&state, &deal, &actions, &strategy, state.actor)
            .unwrap();
        let expected = actions
            .iter()
            .zip(&strategy)
            .map(|(action, p)| {
                p * state.apply(action, &game).own_utilities(&deal, &game)[state.actor]
            })
            .sum::<f64>();
        assert_eq!(exact, expected);
        let open = GameState::initial(&game);
        let legal = open.legal_actions(&game);
        let p = vec![1. / legal.len() as f64; legal.len()];
        assert!(generator
            .cash_closed_terminal_expectation(&open, &deal, &legal, &p, 0)
            .is_none());
    }

    #[test]
    fn replay_checks_the_complete_legal_betting_state() {
        let game = BlueprintConfig::default();
        let deal = Deal::sample(&mut SplitMix64::new(21));
        let mut state = GameState::initial(&game);
        while state.street == Street::Preflop {
            let action = state
                .legal_actions(&game)
                .into_iter()
                .find(|a| matches!(a.kind, ActionKind::Call | ActionKind::Check))
                .unwrap();
            state = state.apply(&action, &game);
        }
        let mut compact = compact_state(&state, &deal, &game);
        let reconstructed = replay(&compact, &game).unwrap();
        assert_eq!(
            reconstructed.legal_actions(&game).len(),
            state.legal_actions(&game).len()
        );
        compact.trajectory[0].pot_after_bb += 1.;
        assert!(replay(&compact, &game).is_err());
    }

    #[test]
    fn posterior_conditions_on_opponent_actions_not_uniform_hidden_hands() {
        let game = BlueprintConfig::default();
        let mut policy = uniform_policy();
        let mut weights = vec![0.; MODEL_INPUT_COUNT];
        weights[48] = 1.;
        weights[STATE_FEATURE_COUNT + 2] = 1.;
        policy.models[0].0.bundle.networks[0] = DenseScorer {
            layers: vec![
                DenseLayer {
                    input_size: MODEL_INPUT_COUNT,
                    output_size: 1,
                    activation: DenseActivation::Relu,
                    weights,
                    biases: vec![-1.],
                },
                DenseLayer {
                    input_size: 1,
                    output_size: 1,
                    activation: DenseActivation::Linear,
                    weights: vec![4.],
                    biases: vec![0.],
                },
            ],
        };
        let deal = Deal::from_sampled_cards([[48, 49], [0, 1]], [4, 8, 12, 16, 20]);
        let root = GameState::initial(&game);
        let call = root
            .legal_actions(&game)
            .into_iter()
            .find(|a| matches!(a.kind, ActionKind::Call))
            .unwrap();
        let state = root.apply(&call, &game);
        let compact = compact_state(&state, &deal, &game);
        let (combos, weights) = opponent_posterior(&compact, &policy, &game).unwrap();
        // Suit canonicalization maps an ace's first suit to clubs (48).
        let ace = combos
            .iter()
            .position(|cards| cards.iter().any(|c| c / 4 == 12))
            .unwrap();
        let low = combos
            .iter()
            .position(|cards| cards.iter().all(|c| c / 4 < 10))
            .unwrap();
        assert!(weights[ace] > weights[low] * 2.);
    }

    #[test]
    fn snapshot_distribution_uses_own_reach_not_pointwise_probability_averaging() {
        let game = BlueprintConfig::default();
        let mut first = uniform_policy();
        let mut second = uniform_policy();
        let mut weights = vec![0.; MODEL_INPUT_COUNT];
        weights[STATE_FEATURE_COUNT + 2] = 4.;
        let scorer = DenseScorer {
            layers: vec![DenseLayer {
                input_size: MODEL_INPUT_COUNT,
                output_size: 1,
                activation: DenseActivation::Linear,
                weights,
                biases: vec![0.],
            }],
        };
        second.models[0].0.bundle.networks = vec![scorer.clone(), scorer];
        let deal = Deal::sample(&mut SplitMix64::new(42));
        let root = GameState::initial(&game);
        let legal = root.legal_actions(&game);
        let call_index = legal
            .iter()
            .position(|a| matches!(a.kind, ActionKind::Call))
            .unwrap();
        let reach0 = first.strategy(&root, &deal, &legal, &game).unwrap()[call_index];
        let reach1 = second.strategy(&root, &deal, &legal, &game).unwrap()[call_index];
        let bb = root.apply(&legal[call_index], &game);
        let raise = bb
            .legal_actions(&game)
            .into_iter()
            .find(|a| matches!(a.kind, ActionKind::RaiseTo(_)))
            .unwrap();
        let state = bb.apply(&raise, &game);
        let actions = state.legal_actions(&game);
        let row0 = first.strategy(&state, &deal, &actions, &game).unwrap();
        let row1 = second.strategy(&state, &deal, &actions, &game).unwrap();
        first.models.append(&mut second.models);
        first.snapshot_average = true;
        let actual = first.strategy(&state, &deal, &actions, &game).unwrap();
        for i in 0..actions.len() {
            assert!(
                (actual[i] - (row0[i] * reach0 + row1[i] * reach1) / (reach0 + reach1)).abs()
                    < 1e-12
            );
        }
        assert!(actual
            .iter()
            .zip(row0.iter().zip(&row1))
            .any(|(a, (b, c))| (a - (b + c) / 2.).abs() > 0.01));
    }
}
