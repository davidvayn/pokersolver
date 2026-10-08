//! Authentic public ranges for bounded, inactive cash turn-value pilots.
//! Scalar frozen policies factor their public action likelihoods by seat.
//! Exact blockers are retained by the compatible joint belief downstream.
use super::super::public_belief::TurnRiverSolveConfig;
use super::*;

pub struct CashTurnRootSampleConfig {
    pub game: BlueprintConfig,
    pub network_path: PathBuf,
    pub seed: u64,
    pub roots: usize,
    pub max_deals: u64,
    pub solve_iterations: u64,
}

#[derive(Serialize)]
struct CashTurnRoot {
    root_sha256: String,
    source_deal_index: u64,
    source_public_actions: Vec<TrajectoryAction>,
    compatible_joint_mass: f64,
    solve_input: TurnRiverSolveConfig,
}

#[derive(Serialize)]
pub struct CashTurnRootCorpus {
    schema: &'static str,
    root_hash_schema: &'static str,
    validation_status: &'static str,
    rules_sha256: String,
    policy_sha256: String,
    sampling_seed: u64,
    sampled_deals: u64,
    roots: Vec<CashTurnRoot>,
    limitations: Vec<&'static str>,
}

/// Synthetic hidden cards only make a well-formed Deal for a known acting
/// combo. The encoder sees only that combo and the board already revealed.
fn public_combo_deal(combo: Combo, actor: usize, board: &[u8]) -> Deal {
    let mut spare = (0..52u8).filter(|c| !combo.cards().contains(c) && !board.contains(c));
    let mut holes = [[0; 2]; 2];
    holes[actor] = combo.cards();
    holes[1 - actor] = [spare.next().unwrap(), spare.next().unwrap()];
    let mut runout = [0; 5];
    runout[..board.len()].copy_from_slice(board);
    for card in &mut runout[board.len()..] {
        *card = spare.next().unwrap();
    }
    Deal::from_sampled_cards(holes, runout)
}

/// No actual private hands or future river are accepted by this boundary.
fn public_reaches(
    policy: &FrozenPolicy,
    game: &BlueprintConfig,
    board: &[u8],
    trajectory: &[TrajectoryAction],
) -> Result<[Vec<f64>; 2], String> {
    if game.cash_rules.is_none()
        || board.len() != 4
        || board.iter().any(|c| *c >= 52)
        || board.iter().collect::<BTreeSet<_>>().len() != 4
        || trajectory.len() > MAX_TRAJECTORY_ACTIONS
    {
        return Err(
            "public cash reaches require four distinct revealed cards and pinned rules".into(),
        );
    }
    let mut state = GameState::initial(game);
    let mut line = Vec::new();
    for observed in trajectory {
        if state.actor != observed.actor
            || state.street != observed.street
            || state.terminal.is_some()
        {
            return Err("cash public range history is not a legal prior line".into());
        }
        let actions = state.legal_actions(game);
        let selected = actions
            .iter()
            .position(|action| {
                let child = state.apply(action, game);
                child.trajectory.last() == Some(observed)
            })
            .ok_or("cash public range action differs from the frozen abstraction")?;
        let next = state.apply(&actions[selected], game);
        line.push((state, actions, selected));
        state = next;
    }
    if state.street != Street::Turn
        || state.terminal.is_some()
        || state.street_invested != [0.; 2]
        || state.checks != 0
        || state.aggressions != 0
    {
        return Err("cash reach line does not end at a fresh live turn".into());
    }
    let mut ranges = [vec![0.; COMBO_COUNT], vec![0.; COMBO_COUNT]];
    for actor in 0..2 {
        let mut log_weights = vec![f64::NEG_INFINITY; COMBO_COUNT];
        for combo in all_combos() {
            if combo.cards().iter().any(|c| board.contains(c)) {
                continue;
            }
            let deal = public_combo_deal(combo, actor, board);
            let mut likelihood = 0.;
            for (prior, actions, selected) in &line {
                if prior.actor == actor {
                    let mix =
                        cash_evaluation::checked_strategy(policy, prior, &deal, actions, game)?;
                    likelihood += mix[*selected].ln();
                }
            }
            log_weights[combo.key()] = likelihood;
        }
        let maximum = log_weights
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if !maximum.is_finite() {
            return Err("cash public range has zero reach; no uniform fallback".into());
        }
        let mass: f64 = log_weights.iter().map(|v| (v - maximum).exp()).sum();
        for (weight, log_weight) in ranges[actor].iter_mut().zip(log_weights) {
            *weight = (log_weight - maximum).exp() / mass;
        }
    }
    Ok(ranges)
}

fn compatible_joint_mass(ranges: &[Vec<f64>; 2]) -> f64 {
    let mut card_mass = [0.; 52];
    for combo in all_combos() {
        for card in combo.cards() {
            card_mass[card as usize] += ranges[1][combo.key()];
        }
    }
    let total: f64 = ranges[1].iter().sum();
    all_combos()
        .iter()
        .map(|combo| {
            ranges[0][combo.key()]
                * (total - card_mass[combo.high as usize] - card_mass[combo.low as usize]
                    + ranges[1][combo.key()])
                .max(0.)
        })
        .sum()
}

fn root_fingerprint(
    policy_sha256: &str,
    actions: &[TrajectoryAction],
    input: &TurnRiverSolveConfig,
) -> Result<String, Box<dyn Error>> {
    let range_hashes: Vec<String> = input
        .state
        .ranges
        .iter()
        .map(|range| {
            let mut hash = Sha256::new();
            for weight in range {
                hash.update(weight.to_le_bytes());
            }
            format!("{:x}", hash.finalize())
        })
        .collect();
    let mut normalized_input = serde_json::to_value(input)?;
    normalized_input["state"]["ranges"] = serde_json::json!(range_hashes);
    // Sorted JSON keys plus little-endian f64 range hashes avoid cross-language
    // exponent-formatting ambiguity for small public reach probabilities.
    let payload = serde_json::json!({"schema":"hu-cash-public-range-root-v1",
        "policy_sha256":policy_sha256,"source_public_actions":actions,"solve_input":normalized_input});
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&payload)?)
    ))
}

fn sample_with_policy(
    config: CashTurnRootSampleConfig,
    policy: FrozenPolicy,
) -> Result<CashTurnRootCorpus, Box<dyn Error>> {
    config.game.validate()?;
    if config.game.cash_rules.is_none()
        || config.game.effective_stack_bb != 20.
        || !(1..=64).contains(&config.roots)
        || !(1..=10_000).contains(&config.max_deals)
        || !(2..=128).contains(&config.solve_iterations)
    {
        return Err("cash turn preflight requires 20bb, 1..64 roots, <=10000 deals and 2..128 solve updates".into());
    }
    policy.bundle.validate_game(&config.game)?;
    if !matches!(policy.bundle.strategy_transform, StrategyTransform::Softmax)
        || policy.range_policy.is_some()
    {
        return Err("cash public reach sampling requires frozen scalar average policies".into());
    }
    let mut rng = SplitMix64::new(config.seed);
    let mut roots = Vec::new();
    let mut seen = BTreeSet::new();
    let mut sampled_deals = 0;
    for index in 0..config.max_deals {
        sampled_deals = index + 1;
        let deal = Deal::sample(&mut rng);
        let mut state = GameState::initial(&config.game);
        for _ in 0..MAX_TRAJECTORY_ACTIONS {
            if state.terminal.is_some() || state.street == Street::Turn {
                break;
            }
            let actions = state.legal_actions(&config.game);
            let mix =
                cash_evaluation::checked_strategy(&policy, &state, &deal, &actions, &config.game)?;
            state = state.apply(&actions[sample_index(&mix, &mut rng)], &config.game);
        }
        if state.terminal.is_some() || state.street != Street::Turn {
            continue;
        }
        let board: [u8; 4] = deal.board[..4].try_into().unwrap();
        let ranges = public_reaches(&policy, &config.game, &board, &state.trajectory)?;
        let joint = compatible_joint_mass(&ranges);
        if !joint.is_finite() || joint <= 0. {
            return Err("cash turn root lacks compatible public joint reach".into());
        }
        let mut game = config.game.clone();
        game.iterations = 2;
        game.averaging_delay = 0;
        let input = TurnRiverSolveConfig {
            game,
            state: PublicBeliefState::turn_start(board, state.actor, state.invested, ranges),
            iterations: config.solve_iterations,
            averaging_delay: 0,
            river_refinement_iterations: 0,
            regret_matching_plus: false,
        };
        let root_sha256 = root_fingerprint(&policy.bundle_sha256, &state.trajectory, &input)?;
        if !seen.insert(root_sha256.clone()) {
            continue;
        }
        roots.push(CashTurnRoot {
            root_sha256,
            source_deal_index: index,
            source_public_actions: state.trajectory,
            compatible_joint_mass: joint,
            solve_input: input,
        });
        if roots.len() == config.roots {
            break;
        }
    }
    if roots.len() != config.roots {
        return Err("cash turn sampling exhausted its deal budget; no partial corpus".into());
    }
    Ok(CashTurnRootCorpus { schema: "hu-cash-authentic-turn-roots-v1", root_hash_schema: "hu-cash-public-range-root-v1", validation_status: "research_only",
        rules_sha256: config.game.cash_rules.as_ref().unwrap().sha256()?,
        policy_sha256: policy.bundle_sha256, sampling_seed: config.seed, sampled_deals, roots,
        limitations: vec!["Conditional authentic turn-start distribution under one frozen research policy, not full serving coverage",
            "Public action likelihoods factor by seat; exact compatible joint card removal remains required",
            "Fresh turn references re-solve the stated subgame; they do not evaluate the earlier frozen continuation or provide a safety guarantee",
            "No website activation, unrestricted exploitability certificate, or accepted continuation oracle"] })
}

pub fn sample_cash_turn_roots(
    config: CashTurnRootSampleConfig,
) -> Result<CashTurnRootCorpus, Box<dyn Error>> {
    let policy = FrozenPolicy::load(&config.network_path)?;
    sample_with_policy(config, policy)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(game: &BlueprintConfig, ace_call: bool) -> FrozenPolicy {
        let scorer = if ace_call {
            let mut weights = vec![0.; CASH_MODEL_INPUT_COUNT];
            weights[48] = 1.;
            weights[CASH_STATE_FEATURE_COUNT + 2] = 1.;
            DenseScorer {
                layers: vec![
                    DenseLayer {
                        input_size: CASH_MODEL_INPUT_COUNT,
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
            }
        } else {
            DenseScorer {
                layers: vec![DenseLayer {
                    input_size: CASH_MODEL_INPUT_COUNT,
                    output_size: 1,
                    activation: DenseActivation::Linear,
                    weights: vec![0.; CASH_MODEL_INPUT_COUNT],
                    biases: vec![0.],
                }],
            }
        };
        FrozenPolicy::from_bundle(
            TrainingNetworkBundle {
                schema: CASH_TRAINING_NETWORK_SCHEMA.into(),
                input_size: CASH_MODEL_INPUT_COUNT,
                strategy_transform: StrategyTransform::Softmax,
                networks: vec![scorer.clone(), scorer],
                postflop_networks: None,
                sampling_baseline: None,
                sampling_baseline_scale: None,
                cash_rules: game.cash_rules.clone(),
                cash_depth_bb: Some(20.),
                cash_action_abstraction: Some(game.action_abstraction.clone()),
            },
            "a".repeat(64),
        )
        .unwrap()
    }

    fn passive_line(game: &BlueprintConfig) -> Vec<TrajectoryAction> {
        let mut state = GameState::initial(game);
        for _ in 0..4 {
            let action = state
                .legal_actions(game)
                .into_iter()
                .find(|a| matches!(a.kind, ActionKind::Call | ActionKind::Check))
                .unwrap();
            state = state.apply(&action, game);
        }
        assert_eq!(state.street, Street::Turn);
        state.trajectory
    }

    #[test]
    fn public_ranges_use_action_likelihoods_not_sampled_hidden_hands() {
        let game = cash_training_tests::config().game;
        let board = [8, 13, 22, 31];
        let line = passive_line(&game);
        let uniform = public_reaches(&policy(&game, false), &game, &board, &line).unwrap();
        let informed = public_reaches(&policy(&game, true), &game, &board, &line).unwrap();
        for player in 0..2 {
            assert!((uniform[player].iter().sum::<f64>() - 1.).abs() < 1e-12);
            assert!((informed[player].iter().sum::<f64>() - 1.).abs() < 1e-12);
            for combo in all_combos() {
                if combo.cards().iter().any(|c| board.contains(c)) {
                    assert_eq!(informed[player][combo.key()], 0.);
                } else {
                    assert!((uniform[player][combo.key()] - 1. / 1128.).abs() < 1e-12);
                }
            }
        }
        let ace = Combo::new(48, 0);
        let other = Combo::new(40, 0);
        assert!(informed[0][ace.key()] > informed[0][other.key()]);
        assert_eq!(informed[1], uniform[1]);
        let mut illegal = line;
        illegal[0].actor = 1;
        assert!(public_reaches(&policy(&game, false), &game, &board, &illegal).is_err());
    }

    #[test]
    fn synthetic_hidden_cards_and_future_rivers_cannot_change_visible_policy_features() {
        let game = cash_training_tests::config().game;
        let combo = Combo::new(48, 0);
        let deal = public_combo_deal(combo, 0, &[8, 13, 22, 31]);
        let other = Deal::from_sampled_cards([combo.cards(), [44, 45]], [8, 13, 22, 31, 40]);
        let mut state = GameState::initial(&game);
        for street in [Street::Preflop, Street::Flop, Street::Turn] {
            state.street = street;
            assert_eq!(
                encode_state_features(&state, &deal, &game),
                encode_state_features(&state, &other, &game)
            );
        }
    }

    #[test]
    fn bounded_root_corpus_is_deterministic_public_only_and_fail_closed() {
        let game = cash_training_tests::config().game;
        let config = || CashTurnRootSampleConfig {
            game: game.clone(),
            network_path: "unused".into(),
            seed: 931,
            roots: 2,
            max_deals: 2000,
            solve_iterations: 8,
        };
        let one = sample_with_policy(config(), policy(&game, false)).unwrap();
        let two = sample_with_policy(config(), policy(&game, false)).unwrap();
        let bytes = serde_json::to_vec(&one).unwrap();
        assert_eq!(bytes, serde_json::to_vec(&two).unwrap());
        let json = String::from_utf8(bytes).unwrap();
        assert!(!json.contains("private_cards"));
        assert!(!json.contains("holes"));
        for root in one.roots {
            assert_eq!(root.solve_input.state.board.len(), 4);
            assert!(root.compatible_joint_mass > 0.);
            assert_eq!(
                root.solve_input.state.public_history,
                ["public_belief:turn_start"]
            );
            assert!(!root.source_public_actions.is_empty());
        }
        let mut invalid = config();
        invalid.max_deals = 0;
        assert!(sample_with_policy(invalid, policy(&game, false)).is_err());
        let mut insufficient = config();
        insufficient.roots = 64;
        insufficient.max_deals = 1;
        assert!(sample_with_policy(insufficient, policy(&game, false)).is_err());
    }
}
