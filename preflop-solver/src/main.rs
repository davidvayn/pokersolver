use preflop_solver::blueprint::{self, BlueprintConfig, RecallMode, RunControl};
use preflop_solver::kuhn;
use preflop_solver::push_fold::{self, PushFoldConfig};
use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let command = args.first().map(String::as_str).unwrap_or("help");
    match command {
        "kuhn" => run_kuhn(&args[1..]),
        "solve" => run_push_fold(&args[1..]),
        "blueprint" => run_blueprint(&args[1..]),
        "neural-samples" => run_neural_samples(&args[1..]),
        "neural-certificate" => run_neural_certificate(&args[1..]),
        "preflop-cache" => run_preflop_cache(&args[1..]),
        "preflop-cache-resolver" => run_preflop_cache_resolver(&args[1..]),
        "preflop-cache-compare" => run_preflop_cache_compare(&args[1..]),
        "preflop-cache-merge" => run_preflop_cache_merge(&args[1..]),
        "preflop-cache-refresh" => run_preflop_cache_refresh(&args[1..]),
        "preflop-cache-inspect" => run_preflop_cache_inspect(&args[1..]),
        "preflop-dcfr" => run_preflop_dcfr(&args[1..]),
        "preflop-evaluate" => run_preflop_evaluate(&args[1..]),
        "preflop-attribution" => run_preflop_attribution(&args[1..]),
        "preflop-compact" => run_preflop_compact(&args[1..]),
        "preflop-distill-samples" => run_preflop_distill_samples(&args[1..]),
        "preflop-evaluate-neural" => run_preflop_evaluate_neural(&args[1..]),
        "full-game-lbr" => run_full_game_lbr(&args[1..]),
        "river-pbs-solve" => run_river_pbs_solve(&args[1..]),
        "turn-pbs-targets" => run_turn_pbs_targets(&args[1..]),
        "flop-pbs-resolve" => run_flop_pbs_resolve(&args[1..]),
        "flop-pbs-leaf-targets" => run_flop_pbs_leaf_targets(&args[1..]),
        "turn-pbs-self-play-targets" => run_turn_pbs_self_play_targets(&args[1..]),
        "turn-pbs-value-predict" => run_turn_pbs_value_predict(&args[1..]),
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        unknown => Err(format!("unknown command: {unknown}").into()),
    }
}

fn run_preflop_cache_resolver(args: &[String]) -> Result<(), Box<dyn Error>> {
    let base_path = value(args, "--base-cache")
        .map(PathBuf::from)
        .ok_or("--base-cache is required")?;
    let network_path = value(args, "--value-network")
        .map(PathBuf::from)
        .ok_or("--value-network is required")?;
    let base = blueprint::preflop::ContinuationCache::read(&base_path)?;
    let iterations = parse_or(args, "--resolver-iterations", 10u64)?;
    let cache = blueprint::preflop::build_resolver_continuation_cache(
        &base,
        blueprint::preflop::ResolverContinuationCacheConfig {
            deals: parse_or(args, "--deals", 2usize)?,
            resolver_iterations: iterations,
            resolver_averaging_delay: parse_or(
                args,
                "--resolver-averaging-delay",
                iterations / 10,
            )?,
            value_uncertainty_bb: parse_or(args, "--value-uncertainty-bb", 1.0f64)?,
            value_network_path: network_path,
            threads: parse_or(
                args,
                "--threads",
                std::thread::available_parallelism()
                    .map(usize::from)
                    .unwrap_or(1),
            )?,
        },
    )?;
    let output = value(args, "--output")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("preflop-continuation-resolver.json.gz"));
    cache.write(&output)?;
    println!("{}", serde_json::to_string_pretty(&cache.validation)?);
    eprintln!(
        "wrote resolver-derived continuation cache {}",
        output.display()
    );
    Ok(())
}

fn run_turn_pbs_self_play_targets(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut game = BlueprintConfig::default();
    game.effective_stack_bb = parse_or(args, "--effective-stack-bb", 20.0)?;
    game.iterations = 2;
    game.averaging_delay = 0;
    let network_path = value(args, "--networks")
        .map(PathBuf::from)
        .ok_or("--networks is required for self-play public beliefs")?;
    let river_iterations = parse_or(args, "--river-iterations", 200u64)?;
    let dataset = blueprint::public_belief::generate_self_play_turn_targets(
        blueprint::public_belief::SelfPlayTurnTargetConfig {
            game,
            states: parse_or(args, "--states", 4usize)?,
            range_particles: parse_or(args, "--range-particles", 512u64)?,
            river_iterations,
            river_averaging_delay: parse_or(
                args,
                "--river-averaging-delay",
                river_iterations / 10,
            )?,
            seed: parse_or(args, "--seed", 0x5E1F_91A7u64)?,
            threads: parse_or(
                args,
                "--threads",
                std::thread::available_parallelism()
                    .map(usize::from)
                    .unwrap_or(1),
            )?,
            network_path,
            belief_replicates: parse_or(args, "--belief-replicates", 2u32)?,
            exploration_probability: parse_or(args, "--exploration", 0.0f64)?,
            minimum_pot_bb: parse_or(args, "--minimum-pot-bb", 0.0f64)?,
            checkpoint_dir: value(args, "--checkpoint-dir").map(PathBuf::from),
        },
    )?;
    let path = value(args, "--output")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("turn-pbs-self-play-targets.json"));
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(&path, format!("{}\n", serde_json::to_string(&dataset)?))?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": dataset.schema,
            "stateDistribution": dataset.state_distribution,
            "sourcePolicySha256": dataset.source_policy_sha256,
            "explorationProbability": dataset.sampling_exploration_probability,
            "minimumSampledPotBb": dataset.minimum_sampled_pot_bb,
            "states": dataset.targets.len(),
            "minimumRangeEffectiveSampleSize": dataset.targets.iter().filter_map(|target| target.range_effective_sample_size).fold(f64::INFINITY, f64::min),
            "minimumRangeReplicates": dataset.targets.iter().filter_map(|target| target.range_replicates).min(),
            "maximumRangeTotalVariation": dataset.targets.iter().filter_map(|target| target.range_maximum_total_variation).fold(0.0f64, f64::max),
            "maximumRiverExploitabilityBbPerHand": dataset.targets.iter().map(|target| target.maximum_river_exploitability_bb_per_hand).fold(0.0f64, f64::max),
            "validation": dataset.validation,
            "output": path,
        }))?
    );
    Ok(())
}

fn run_flop_pbs_leaf_targets(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut game = BlueprintConfig::default();
    game.effective_stack_bb = parse_or(args, "--effective-stack-bb", 20.0)?;
    game.iterations = 2;
    game.averaging_delay = 0;
    let network_path = value(args, "--value-network")
        .map(PathBuf::from)
        .ok_or("--value-network is required")?;
    let root_boards = parse_flop_boards(
        &value(args, "--boards")
            .ok_or("--boards is required, for example 2c,7d,Th;As,Kd,7c;9h,Th,Jh")?,
    )?;
    let resolver_iterations = parse_or(args, "--resolver-iterations", 20u64)?;
    let river_iterations = parse_or(args, "--river-iterations", 200u64)?;
    let dataset = blueprint::public_belief::generate_resolver_leaf_turn_targets(
        blueprint::public_belief::ResolverLeafTurnTargetConfig {
            game,
            root_boards,
            states_per_board: parse_or(args, "--states-per-board", 3usize)?,
            root_pot_bb: parse_or(args, "--pot-bb", 4.0f64)?,
            root_actor: parse_or(args, "--actor", 1usize)?,
            resolver_iterations,
            resolver_averaging_delay: parse_or(
                args,
                "--resolver-averaging-delay",
                resolver_iterations / 10,
            )?,
            river_iterations,
            river_averaging_delay: parse_or(
                args,
                "--river-averaging-delay",
                river_iterations / 10,
            )?,
            seed: parse_or(args, "--seed", 0x1EAF_C0DEu64)?,
            threads: parse_or(
                args,
                "--threads",
                std::thread::available_parallelism()
                    .map(usize::from)
                    .unwrap_or(1),
            )?,
            value_network_path: network_path,
            checkpoint_dir: value(args, "--checkpoint-dir").map(PathBuf::from),
        },
    )?;
    let path = value(args, "--output")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("flop-resolver-leaf-targets.json"));
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(&path, format!("{}\n", serde_json::to_string(&dataset)?))?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": dataset.schema,
            "states": dataset.targets.len(),
            "resolverIterations": dataset.resolver_iterations,
            "resolverLeafPopulation": dataset.resolver_leaf_population,
            "meanResolverLeafProbabilityMass": dataset.resolver_leaf_probability_mass,
            "maximumRiverExploitabilityBbPerHand": dataset.targets.iter().map(|target| target.maximum_river_exploitability_bb_per_hand).fold(0.0f64, f64::max),
            "maximumZeroSumResidualBb": dataset.targets.iter().map(|target| target.zero_sum_residual_bb.abs()).fold(0.0f64, f64::max),
            "validation": dataset.validation,
            "output": path,
        }))?
    );
    Ok(())
}

fn run_flop_pbs_resolve(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut game = BlueprintConfig::default();
    game.effective_stack_bb = parse_or(args, "--effective-stack-bb", 20.0)?;
    game.iterations = 2;
    game.averaging_delay = 0;
    // The pilot is intentionally fail-closed until exact all-in flop runouts
    // are vectorized in the public-belief solver.
    game.action_abstraction.include_all_in = false;
    let board = parse_board::<3>(
        &value(args, "--board").ok_or("--board is required, for example 2c,7d,Th")?,
    )?;
    let network_path = value(args, "--value-network")
        .map(PathBuf::from)
        .ok_or("--value-network is required")?;
    let network = blueprint::public_belief::PublicValueNetwork::read(&network_path)?;
    let ranges = std::array::from_fn(|_| blueprint::public_belief::uniform_range(&board));
    let pot_bb = parse_or(args, "--pot-bb", 4.0f64)?;
    let iterations = parse_or(args, "--iterations", 20u64)?;
    let solution =
        blueprint::public_belief::solve_flop(blueprint::public_belief::FlopResolveConfig {
            game,
            state: blueprint::public_belief::PublicBeliefState::flop_start(
                board,
                parse_or(args, "--actor", 1usize)?,
                [pot_bb / 2.0, pot_bb / 2.0],
                ranges,
            ),
            iterations,
            averaging_delay: parse_or(args, "--averaging-delay", iterations / 10)?,
            value_network: network,
            threads: parse_or(
                args,
                "--threads",
                std::thread::available_parallelism()
                    .map(usize::from)
                    .unwrap_or(1),
            )?,
        })?;
    let output = serde_json::to_string_pretty(&solution)?;
    if let Some(path) = value(args, "--output").map(PathBuf::from) {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(&path, format!("{output}\n"))?;
        println!("{}", serde_json::to_string_pretty(&solution.metrics)?);
        eprintln!("wrote depth-limited flop pilot {}", path.display());
    } else {
        println!("{output}");
    }
    Ok(())
}

fn run_turn_pbs_value_predict(args: &[String]) -> Result<(), Box<dyn Error>> {
    let network_path = value(args, "--value-network")
        .map(PathBuf::from)
        .ok_or("--value-network is required")?;
    let dataset_path = value(args, "--dataset")
        .map(PathBuf::from)
        .ok_or("--dataset is required")?;
    let dataset: blueprint::public_belief::TurnTargetDataset =
        serde_json::from_slice(&fs::read(dataset_path)?)?;
    let state_index = parse_or(args, "--state-index", 0usize)?;
    let target = dataset
        .targets
        .get(state_index)
        .ok_or("--state-index is outside the target dataset")?;
    let network = blueprint::public_belief::PublicValueNetwork::read(&network_path)?;
    let ranges: [Vec<f64>; 2] = std::array::from_fn(|player| {
        target.ranges[player]
            .iter()
            .map(|value| *value as f64)
            .collect()
    });
    let prediction = network.predict(&target.board, target.actor, target.invested_bb, &ranges);
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "schema": "hu-public-belief-value-prediction-v1",
            "stateIndex": state_index,
            "counterfactualValuesBb": prediction,
        }))?
    );
    Ok(())
}

fn run_river_pbs_solve(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut game = BlueprintConfig::default();
    game.effective_stack_bb = parse_or(args, "--effective-stack-bb", 20.0)?;
    game.iterations = 2;
    game.averaging_delay = 0;
    if args
        .iter()
        .any(|argument| argument == "--compact-serving-grid")
    {
        game.action_abstraction = blueprint::ActionAbstraction::compact_serving_candidate();
    }
    let board = parse_board::<5>(
        &value(args, "--board").ok_or("--board is required, for example 2c,7d,Th,Js,Ac")?,
    )?;
    let pot_bb = parse_or(args, "--pot-bb", 4.0f64)?;
    let iterations = parse_or(args, "--iterations", 2_000u64)?;
    let averaging_delay = parse_or(args, "--averaging-delay", iterations / 10)?;
    let solution =
        blueprint::public_belief::solve_river(blueprint::public_belief::RiverSolveConfig {
            game,
            state: blueprint::public_belief::PublicBeliefState::uniform_river_start(
                board,
                parse_or(args, "--actor", 1usize)?,
                [pot_bb / 2.0, pot_bb / 2.0],
            ),
            iterations,
            averaging_delay,
        })?;
    let output = serde_json::to_string_pretty(&solution)?;
    if let Some(path) = value(args, "--output").map(PathBuf::from) {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(&path, format!("{output}\n"))?;
        println!("{}", serde_json::to_string_pretty(&solution.metrics)?);
        eprintln!("wrote exact-card-removal river solution {}", path.display());
    } else {
        println!("{output}");
    }
    Ok(())
}

fn run_turn_pbs_targets(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut game = BlueprintConfig::default();
    game.effective_stack_bb = parse_or(args, "--effective-stack-bb", 20.0)?;
    game.iterations = 2;
    game.averaging_delay = 0;
    if args
        .iter()
        .any(|argument| argument == "--compact-serving-grid")
    {
        game.action_abstraction = blueprint::ActionAbstraction::compact_serving_candidate();
    }
    let river_iterations = parse_or(args, "--river-iterations", 500u64)?;
    let dataset = blueprint::public_belief::generate_turn_targets(
        blueprint::public_belief::TurnTargetGenerationConfig {
            game,
            states: parse_or(args, "--states", 16usize)?,
            river_iterations,
            river_averaging_delay: parse_or(
                args,
                "--river-averaging-delay",
                river_iterations / 10,
            )?,
            seed: parse_or(args, "--seed", 0xB311_EF5u64)?,
            threads: parse_or(
                args,
                "--threads",
                std::thread::available_parallelism()
                    .map(usize::from)
                    .unwrap_or(1),
            )?,
        },
    )?;
    let path = value(args, "--output")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("turn-pbs-targets.json"));
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(&path, format!("{}\n", serde_json::to_string(&dataset)?))?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": dataset.schema,
            "states": dataset.targets.len(),
            "riverIterations": dataset.river_iterations,
            "maximumRiverExploitabilityBbPerHand": dataset.targets.iter().map(|target| target.maximum_river_exploitability_bb_per_hand).fold(0.0f64, f64::max),
            "maximumZeroSumResidualBb": dataset.targets.iter().map(|target| target.zero_sum_residual_bb).fold(0.0f64, f64::max),
            "validation": dataset.validation,
            "output": path,
        }))?
    );
    Ok(())
}

fn parse_board<const N: usize>(value: &str) -> Result<[u8; N], Box<dyn Error>> {
    let cards = value
        .split(',')
        .map(|token| parse_card(token.trim()))
        .collect::<Result<Vec<_>, _>>()?;
    let board: [u8; N] = cards
        .try_into()
        .map_err(|cards: Vec<u8>| format!("expected {N} board cards, received {}", cards.len()))?;
    let unique = board
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    if unique.len() != N {
        return Err("board cards must be unique".into());
    }
    Ok(board)
}

fn parse_flop_boards(value: &str) -> Result<Vec<[u8; 3]>, Box<dyn Error>> {
    value
        .split(';')
        .map(|board| parse_board::<3>(board.trim()))
        .collect()
}

fn parse_card(token: &str) -> Result<u8, Box<dyn Error>> {
    let bytes = token.as_bytes();
    if bytes.len() != 2 {
        return Err(format!("invalid card {token:?}").into());
    }
    let rank = preflop_solver::cards::RANKS
        .iter()
        .position(|candidate| candidate.eq_ignore_ascii_case(&bytes[0]))
        .ok_or_else(|| format!("invalid card rank in {token:?}"))?;
    let suit = preflop_solver::cards::SUITS
        .iter()
        .position(|candidate| candidate.eq_ignore_ascii_case(&bytes[1]))
        .ok_or_else(|| format!("invalid card suit in {token:?}"))?;
    Ok((rank * 4 + suit) as u8)
}

fn run_preflop_cache_refresh(args: &[String]) -> Result<(), Box<dyn Error>> {
    let cache_path = value(args, "--cache")
        .map(PathBuf::from)
        .ok_or("--cache is required for continuation validation refresh")?;
    let output = value(args, "--output")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("preflop-continuation-refreshed.json.gz"));
    let cache = blueprint::preflop::ContinuationCache::read(&cache_path)?;
    let refreshed = blueprint::preflop::refresh_continuation_cache_validation(cache)?;
    refreshed.write(&output)?;
    println!("{}", serde_json::to_string_pretty(&refreshed.validation)?);
    eprintln!("wrote refreshed continuation cache {}", output.display());
    Ok(())
}

fn run_preflop_cache_inspect(args: &[String]) -> Result<(), Box<dyn Error>> {
    let cache_path = value(args, "--cache")
        .map(PathBuf::from)
        .ok_or("--cache is required for continuation inspection")?;
    let cache = blueprint::preflop::ContinuationCache::read(&cache_path)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": "hu-preflop-continuation-cache-summary-v1",
            "path": cache_path.display().to_string(),
            "depthBb": cache.depth_bb,
            "seed": cache.seed,
            "deals": cache.deals.len(),
            "completeExactComboCycles": cache.complete_exact_combo_cycles,
            "rolloutsPerLeaf": cache.rollouts_per_leaf,
            "networkSha256s": cache.network_sha256s,
            "policyMixture": cache.policy_mixture,
            "publicHistories": cache.public_histories.len(),
            "validation": cache.validation,
        }))?
    );
    Ok(())
}

fn run_preflop_cache_merge(args: &[String]) -> Result<(), Box<dyn Error>> {
    let first_path = value(args, "--cache-a")
        .map(PathBuf::from)
        .ok_or("--cache-a is required for continuation merge")?;
    let second_path = value(args, "--cache-b")
        .map(PathBuf::from)
        .ok_or("--cache-b is required for continuation merge")?;
    let output = value(args, "--output")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("preflop-continuation-merged.json.gz"));
    let first = blueprint::preflop::ContinuationCache::read(&first_path)?;
    let second = blueprint::preflop::ContinuationCache::read(&second_path)?;
    let merged = blueprint::preflop::merge_continuation_caches(&first, &second)?;
    merged.write(&output)?;
    println!("{}", serde_json::to_string_pretty(&merged.validation)?);
    eprintln!("wrote merged continuation cache {}", output.display());
    Ok(())
}

fn run_preflop_cache_compare(args: &[String]) -> Result<(), Box<dyn Error>> {
    let first_path = value(args, "--cache-a")
        .map(PathBuf::from)
        .ok_or("--cache-a is required for continuation comparison")?;
    let second_path = value(args, "--cache-b")
        .map(PathBuf::from)
        .ok_or("--cache-b is required for continuation comparison")?;
    let first = blueprint::preflop::ContinuationCache::read(&first_path)?;
    let second = blueprint::preflop::ContinuationCache::read(&second_path)?;
    let comparison = blueprint::preflop::compare_continuation_caches(&first, &second)?;
    let output = serde_json::to_string_pretty(&comparison)?;
    if let Some(path) = value(args, "--output").map(PathBuf::from) {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, format!("{output}\n"))?;
    }
    println!("{output}");
    Ok(())
}

fn run_full_game_lbr(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut game = BlueprintConfig::default();
    game.effective_stack_bb = parse_or(args, "--effective-stack-bb", 20.0)?;
    if args
        .iter()
        .any(|argument| argument == "--compact-serving-grid")
    {
        game.action_abstraction = blueprint::ActionAbstraction::compact_serving_candidate();
    }
    let network_path = value(args, "--networks")
        .map(PathBuf::from)
        .ok_or("--networks is required for full-game LBR")?;
    let evaluation = blueprint::response::evaluate_full_game_response(
        blueprint::response::ResponseEvaluationConfig {
            game,
            training_deals: parse_or(args, "--training-deals", 10_000u64)?,
            evaluation_deals: parse_or(args, "--evaluation-deals", 10_000u64)?,
            rollouts_per_action: parse_or(args, "--rollouts-per-action", 8u32)?,
            minimum_range_particles: parse_or(args, "--minimum-range-particles", 4u64)?,
            seed: parse_or(args, "--seed", 0x1B12_E5A1u64)?,
            network_path,
        },
    )?;
    let output = serde_json::to_string_pretty(&evaluation)?;
    if let Some(path) = value(args, "--output").map(PathBuf::from) {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, format!("{output}\n"))?;
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "schema": evaluation.schema,
                "approximateExploitabilityLowerBoundBbPerHand": evaluation.approximate_exploitability_lower_bound_bb_per_hand,
                "approximateExploitabilityLowerConfidenceBound99PercentBbPerHand": evaluation.approximate_exploitability_lower_confidence_bound_99_percent_bb_per_hand,
                "players": evaluation.players,
                "preflopDecisionCounts": evaluation.preflop_responses.iter().map(Vec::len).collect::<Vec<_>>(),
                "resolverDecisionCounts": evaluation.resolvers.iter().map(|resolver| resolver.decisions.len()).collect::<Vec<_>>(),
            }))?
        );
    } else {
        println!("{output}");
    }
    Ok(())
}

fn run_preflop_evaluate_neural(args: &[String]) -> Result<(), Box<dyn Error>> {
    let cache_path = value(args, "--cache")
        .map(PathBuf::from)
        .ok_or("--cache is required for neural preflop evaluation")?;
    let network_path = value(args, "--networks")
        .map(PathBuf::from)
        .ok_or("--networks is required for neural preflop evaluation")?;
    let cache = blueprint::preflop::ContinuationCache::read(&cache_path)?;
    let evaluation = blueprint::preflop::evaluate_neural_policy(&cache, &network_path)?;
    let output = serde_json::to_string_pretty(&evaluation)?;
    if let Some(path) = value(args, "--output").map(PathBuf::from) {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, format!("{output}\n"))?;
    }
    println!("{output}");
    Ok(())
}

fn run_preflop_distill_samples(args: &[String]) -> Result<(), Box<dyn Error>> {
    let cache_path = value(args, "--cache")
        .map(PathBuf::from)
        .ok_or("--cache is required for preflop distillation samples")?;
    let policy_path = value(args, "--policy")
        .map(PathBuf::from)
        .ok_or("--policy is required for preflop distillation samples")?;
    let output = value(args, "--output")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("preflop-distillation.jsonl.gz"));
    let cache = blueprint::preflop::ContinuationCache::read(&cache_path)?;
    let policy = blueprint::preflop::PreflopPolicyArtifact::read(&policy_path)?;
    let summary = blueprint::preflop::export_distillation_dataset(&cache, &policy, &output)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn run_preflop_cache(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut game = BlueprintConfig::default();
    game.effective_stack_bb = parse_or(args, "--effective-stack-bb", 20.0)?;
    if args
        .iter()
        .any(|argument| argument == "--compact-serving-grid")
    {
        game.action_abstraction = blueprint::ActionAbstraction::compact_serving_candidate();
    }
    let network_path = value(args, "--networks")
        .map(PathBuf::from)
        .ok_or("--networks is required for the continuation cache")?;
    let mut network_paths = vec![network_path];
    if let Some(second) = value(args, "--networks-b") {
        network_paths.push(PathBuf::from(second));
    }
    let output = value(args, "--output")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("preflop-continuation-cache.json.gz"));
    let cache = blueprint::preflop::build_continuation_cache(
        blueprint::preflop::ContinuationCacheConfig {
            game,
            deals: parse_or(args, "--deals", 2_652usize)?,
            seed: parse_or(args, "--seed", 0xC01A_71A7u64)?,
            rollouts_per_leaf: parse_or(args, "--rollouts-per-leaf", 8u32)?,
            network_paths,
        },
    )?;
    cache.write(&output)?;
    println!("{}", serde_json::to_string_pretty(&cache.validation)?);
    eprintln!(
        "wrote deterministic continuation cache {}",
        output.display()
    );
    Ok(())
}

fn run_preflop_dcfr(args: &[String]) -> Result<(), Box<dyn Error>> {
    let cache_path = value(args, "--cache")
        .map(PathBuf::from)
        .ok_or("--cache is required for preflop DCFR")?;
    let cache = blueprint::preflop::ContinuationCache::read(&cache_path)?;
    let seed = parse_or(args, "--seed", 1u64)?;
    let iterations = parse_or(args, "--iterations", 100_000u64)?;
    let model_version = value(args, "--model-version")
        .unwrap_or_else(|| format!("hu-{}bb-tabular-preflop-v1", cache.depth_bb));
    let output = value(args, "--output")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("preflop-policy.json"));
    let dcfr = blueprint::DcfrParameters {
        positive_regret_exponent: parse_or(
            args,
            "--dcfr-alpha",
            cache.game.dcfr.positive_regret_exponent,
        )?,
        negative_regret_exponent: parse_or(
            args,
            "--dcfr-beta",
            cache.game.dcfr.negative_regret_exponent,
        )?,
        strategy_exponent: parse_or(args, "--dcfr-gamma", cache.game.dcfr.strategy_exponent)?,
    };
    let variant = match value(args, "--solver").as_deref().unwrap_or("dcfr") {
        "dcfr" => blueprint::preflop::PreflopSolverVariant::Dcfr,
        "mccfr-plus" => blueprint::preflop::PreflopSolverVariant::MccfrPlus,
        other => return Err(format!("unsupported preflop solver variant: {other}").into()),
    };
    let artifact = blueprint::preflop::solve_preflop_with_options(
        &cache,
        &cache_path,
        blueprint::preflop::PreflopSolveOptions {
            iterations,
            seed,
            model_version,
            dcfr,
            exploration_probability: parse_or(args, "--exploration", 0.05f64)?,
            variant,
        },
    )?;
    artifact.write(&output)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&artifact.training_evaluation)?
    );
    eprintln!("wrote tabular preflop policy {}", output.display());
    Ok(())
}

fn run_preflop_evaluate(args: &[String]) -> Result<(), Box<dyn Error>> {
    let cache_path = value(args, "--cache")
        .map(PathBuf::from)
        .ok_or("--cache is required for preflop evaluation")?;
    let policy_path = value(args, "--policy")
        .map(PathBuf::from)
        .ok_or("--policy is required for preflop evaluation")?;
    let cache = blueprint::preflop::ContinuationCache::read(&cache_path)?;
    let policy = blueprint::preflop::PreflopPolicyArtifact::read(&policy_path)?;
    let evaluation = blueprint::preflop::evaluate_policy(&cache, &policy);
    let output = serde_json::to_string_pretty(&evaluation)?;
    if let Some(path) = value(args, "--output").map(PathBuf::from) {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, format!("{output}\n"))?;
    }
    println!("{output}");
    Ok(())
}

fn run_preflop_attribution(args: &[String]) -> Result<(), Box<dyn Error>> {
    let cache_path = value(args, "--cache")
        .map(PathBuf::from)
        .ok_or("--cache is required for preflop attribution")?;
    let policy_path = value(args, "--policy")
        .map(PathBuf::from)
        .ok_or("--policy is required for preflop attribution")?;
    let cache = blueprint::preflop::ContinuationCache::read(&cache_path)?;
    let policy = blueprint::preflop::PreflopPolicyArtifact::read(&policy_path)?;
    let attribution = blueprint::preflop::attribute_policy_leaks(
        &cache,
        &policy,
        parse_or(args, "--top", 50usize)?,
    )?;
    let output = serde_json::to_string_pretty(&attribution)?;
    if let Some(path) = value(args, "--output").map(PathBuf::from) {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, format!("{output}\n"))?;
    }
    println!("{output}");
    Ok(())
}

fn run_preflop_compact(args: &[String]) -> Result<(), Box<dyn Error>> {
    let policy_path = value(args, "--policy")
        .map(PathBuf::from)
        .ok_or("--policy is required for compact preflop export")?;
    let output = value(args, "--output")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("preflop-policy.bin"));
    let policy = blueprint::preflop::PreflopPolicyArtifact::read(&policy_path)?;
    let summary = blueprint::preflop::export_compact_preflop_policy(&policy, &output)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn run_neural_certificate(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut game = BlueprintConfig::default();
    game.effective_stack_bb = parse_or(args, "--effective-stack-bb", 20.0)?;
    if args
        .iter()
        .any(|argument| argument == "--compact-serving-grid")
    {
        game.action_abstraction = blueprint::ActionAbstraction::compact_serving_candidate();
    }
    let network_path = value(args, "--networks")
        .map(PathBuf::from)
        .ok_or("--networks is required for neural certification")?;
    let certificate = blueprint::neural::certify_exploitability_upper_bound(
        blueprint::neural::ExploitabilityCertificateConfig {
            game,
            deals: parse_or(args, "--deals", 10_000u64)?,
            seed: parse_or(args, "--seed", 0xA11CE5EEDu64)?,
            confidence: parse_or(args, "--confidence", 0.99f64)?,
            threads: parse_or(
                args,
                "--threads",
                std::thread::available_parallelism()
                    .map(usize::from)
                    .unwrap_or(1),
            )?,
            network_path,
        },
    )?;
    let output = serde_json::to_string_pretty(&certificate)?;
    if let Some(path) = value(args, "--output").map(PathBuf::from) {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, format!("{output}\n"))?;
    }
    println!("{output}");
    Ok(())
}

fn run_neural_samples(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut game = BlueprintConfig::default();
    game.effective_stack_bb = parse_or(args, "--effective-stack-bb", 20.0)?;
    let traversals = parse_or(args, "--traversals", 100u64)?;
    let start_iteration = parse_or(args, "--start-iteration", 0u64)?;
    game.iterations = traversals.max(2);
    game.averaging_delay = 0;
    game.showdown_evaluation.preflop_runout_samples = parse_or(
        args,
        "--preflop-runout-samples",
        game.showdown_evaluation.preflop_runout_samples,
    )?;
    game.showdown_evaluation.flop_runout_samples = parse_or(
        args,
        "--flop-runout-samples",
        game.showdown_evaluation.flop_runout_samples,
    )?;
    if args
        .iter()
        .any(|argument| argument == "--sample-turn-rivers")
    {
        game.showdown_evaluation.exact_turn_rivers = false;
    }
    if args
        .iter()
        .any(|argument| argument == "--compact-serving-grid")
    {
        game.action_abstraction = blueprint::ActionAbstraction::compact_serving_candidate();
    }
    let output = value(args, "--output")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("neural-samples.jsonl.gz"));
    let max_records = parse_or(args, "--max-records", 50_000usize)?;
    let seed = parse_or(args, "--seed", 1u64)?;
    let network_path = value(args, "--networks").map(PathBuf::from);
    let trajectory_sampling = args
        .iter()
        .any(|argument| argument == "--sample-trajectories");
    let evaluate_trajectory_values = args
        .iter()
        .any(|argument| argument == "--evaluate-action-values");
    let value_rollouts_per_action = parse_or(args, "--value-rollouts-per-action", 1u32)?;
    blueprint::neural::generate_samples(blueprint::neural::SampleGenerationConfig {
        game,
        traversals,
        start_iteration,
        seed,
        max_records,
        output: output.clone(),
        network_path,
        trajectory_sampling,
        evaluate_trajectory_values,
        value_rollouts_per_action,
        enumerate_turn_river_chance: args
            .iter()
            .any(|argument| argument == "--enumerate-turn-river-chance"),
    })?;
    eprintln!("wrote compact neural traversal batch {}", output.display());
    Ok(())
}

fn run_blueprint(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut config = BlueprintConfig::default();
    config.small_blind_bb = parse_or(args, "--small-blind-bb", config.small_blind_bb)?;
    config.big_blind_bb = parse_or(args, "--big-blind-bb", config.big_blind_bb)?;
    config.effective_stack_bb = parse_or(args, "--effective-stack-bb", config.effective_stack_bb)?;
    config.iterations = parse_or(args, "--iterations", config.iterations)?;
    config.max_information_sets =
        parse_or(args, "--max-information-sets", config.max_information_sets)?;
    config.seed = parse_or(args, "--seed", config.seed)?;
    config.averaging_delay = parse_or(args, "--averaging-delay", config.averaging_delay)?;
    config.evaluation_controls.held_out_deals = parse_or(
        args,
        "--held-out-deals",
        config.evaluation_controls.held_out_deals,
    )?;
    config.evaluation_controls.held_out_seed = parse_or(
        args,
        "--held-out-seed",
        config.evaluation_controls.held_out_seed,
    )?;
    config.evaluation_controls.root_deviation_samples_per_class = parse_or(
        args,
        "--root-deviation-samples",
        config.evaluation_controls.root_deviation_samples_per_class,
    )?;
    config.evaluation_controls.root_deviation_seed = parse_or(
        args,
        "--root-deviation-seed",
        config.evaluation_controls.root_deviation_seed,
    )?;
    config.evaluation_controls.action_value_deals = parse_or(
        args,
        "--action-value-deals",
        config.evaluation_controls.action_value_deals,
    )?;
    config.evaluation_controls.action_value_seed = parse_or(
        args,
        "--action-value-seed",
        config.evaluation_controls.action_value_seed,
    )?;
    config.dcfr.positive_regret_exponent =
        parse_or(args, "--dcfr-alpha", config.dcfr.positive_regret_exponent)?;
    config.dcfr.negative_regret_exponent =
        parse_or(args, "--dcfr-beta", config.dcfr.negative_regret_exponent)?;
    config.dcfr.strategy_exponent = parse_or(args, "--dcfr-gamma", config.dcfr.strategy_exponent)?;
    config.hand_abstraction.distribution_samples = parse_or(
        args,
        "--distribution-samples",
        config.hand_abstraction.distribution_samples,
    )?;
    config.hand_abstraction.equity_bins =
        parse_or(args, "--equity-bins", config.hand_abstraction.equity_bins)?;
    config.hand_abstraction.potential_bins = parse_or(
        args,
        "--potential-bins",
        config.hand_abstraction.potential_bins,
    )?;
    config.showdown_evaluation.preflop_runout_samples = parse_or(
        args,
        "--preflop-runout-samples",
        config.showdown_evaluation.preflop_runout_samples,
    )?;
    config.showdown_evaluation.flop_runout_samples = parse_or(
        args,
        "--flop-runout-samples",
        config.showdown_evaluation.flop_runout_samples,
    )?;
    if args
        .iter()
        .any(|argument| argument == "--sample-turn-rivers")
    {
        config.showdown_evaluation.exact_turn_rivers = false;
    }
    config.export_postflop_strategies = args
        .iter()
        .any(|argument| argument == "--export-postflop-strategies");
    if args
        .iter()
        .any(|argument| argument == "--current-street-recall")
    {
        config.recall_mode = RecallMode::CurrentStreet;
    }
    if args
        .iter()
        .any(|argument| argument == "--compact-serving-grid")
    {
        config.action_abstraction = blueprint::ActionAbstraction::compact_serving_candidate();
    }
    replace_list(
        args,
        "--open-sizes-bb",
        &mut config.action_abstraction.open_sizes_bb,
    )?;
    replace_list(
        args,
        "--limp-raise-sizes-bb",
        &mut config.action_abstraction.limp_raise_sizes_bb,
    )?;
    replace_list(
        args,
        "--three-bet-sizes-bb",
        &mut config.action_abstraction.three_bet_sizes_bb,
    )?;
    replace_list(
        args,
        "--four-bet-sizes-bb",
        &mut config.action_abstraction.four_bet_sizes_bb,
    )?;
    replace_list(
        args,
        "--deeper-raise-pot-fractions",
        &mut config.action_abstraction.deeper_raise_pot_fractions,
    )?;
    replace_list(
        args,
        "--flop-bet-fractions",
        &mut config.action_abstraction.flop_bet_pot_fractions,
    )?;
    replace_list(
        args,
        "--postflop-raise-fractions",
        &mut config.action_abstraction.postflop_raise_pot_fractions,
    )?;
    config.action_abstraction.preflop_raise_cap = parse_or(
        args,
        "--preflop-raise-cap",
        config.action_abstraction.preflop_raise_cap,
    )?;
    config.action_abstraction.postflop_raise_cap = parse_or(
        args,
        "--postflop-raise-cap",
        config.action_abstraction.postflop_raise_cap,
    )?;
    if args.iter().any(|argument| argument == "--no-all-in") {
        config.action_abstraction.include_all_in = false;
    }
    replace_list(
        args,
        "--turn-river-bet-fractions",
        &mut config.action_abstraction.turn_river_bet_pot_fractions,
    )?;

    let output = value(args, "--output")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("blueprint-artifact.json"));
    let checkpoint = value(args, "--checkpoint");
    let control = RunControl {
        checkpoint_path: checkpoint,
        checkpoint_every: parse_or(args, "--checkpoint-every", 0u64)?,
        resume_path: value(args, "--resume"),
    };
    let artifact = blueprint::solve_controlled(config, control)?;
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(&output, serde_json::to_string_pretty(&artifact)?)?;
    eprintln!(
        "wrote {} ({} infosets, approximate model; no exploitability claim)",
        output.display(),
        artifact.metrics.information_sets
    );
    Ok(())
}

fn replace_list(args: &[String], name: &str, target: &mut Vec<f64>) -> Result<(), Box<dyn Error>> {
    if let Some(raw) = value(args, name) {
        *target = raw
            .split(',')
            .map(str::trim)
            .map(str::parse)
            .collect::<Result<Vec<_>, _>>()?;
    }
    Ok(())
}

fn run_kuhn(args: &[String]) -> Result<(), Box<dyn Error>> {
    let iterations = value(args, "--iterations")
        .unwrap_or_else(|| "20000".to_owned())
        .parse::<u64>()?;
    let result = kuhn::solve(iterations);
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn run_push_fold(args: &[String]) -> Result<(), Box<dyn Error>> {
    let defaults = PushFoldConfig::default();
    let config = PushFoldConfig {
        small_blind_bb: parse_or(args, "--small-blind-bb", defaults.small_blind_bb)?,
        big_blind_bb: parse_or(args, "--big-blind-bb", defaults.big_blind_bb)?,
        effective_stack_bb: parse_or(args, "--effective-stack-bb", defaults.effective_stack_bb)?,
        iterations: parse_or(args, "--iterations", defaults.iterations)?,
        equity_samples: parse_or(args, "--equity-samples", defaults.equity_samples)?,
        seed: parse_or(args, "--seed", defaults.seed)?,
    };
    let output = value(args, "--output")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("push-fold-artifact.json"));
    let artifact = push_fold::solve(config).map_err(|error| format!("invalid solve: {error}"))?;
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(&output, serde_json::to_string_pretty(&artifact)?)?;
    eprintln!(
        "wrote {} (exploitability {:.6} bb, validation {})",
        output.display(),
        artifact.metrics.exploitability_bb,
        artifact.validation.status
    );
    Ok(())
}

fn parse_or<T>(args: &[String], name: &str, default: T) -> Result<T, Box<dyn Error>>
where
    T: std::str::FromStr,
    T::Err: Error + 'static,
{
    match value(args, name) {
        Some(raw) => Ok(raw.parse::<T>()?),
        None => Ok(default),
    }
}

fn value(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|argument| argument == name)
        .and_then(|index| args.get(index + 1))
        .cloned()
}

fn print_help() {
    println!(
        "Offline preflop solver

Usage:
  preflop-solver kuhn [--iterations 20000]
  preflop-solver solve [options]
  preflop-solver blueprint [options]
  preflop-solver neural-samples [options]
  preflop-solver neural-certificate [options]
  preflop-solver preflop-cache [options]
  preflop-solver preflop-cache-resolver [options]
  preflop-solver preflop-cache-compare [options]
  preflop-solver preflop-cache-merge [options]
  preflop-solver preflop-cache-refresh [options]
  preflop-solver preflop-dcfr [options]
  preflop-solver preflop-evaluate [options]
  preflop-solver preflop-distill-samples [options]
  preflop-solver preflop-evaluate-neural [options]
  preflop-solver full-game-lbr [options]
  preflop-solver river-pbs-solve [options]
  preflop-solver turn-pbs-targets [options]
  preflop-solver turn-pbs-self-play-targets [options]
  preflop-solver turn-pbs-value-predict [options]
  preflop-solver flop-pbs-resolve [options]
  preflop-solver flop-pbs-leaf-targets [options]

Solve options:
  --small-blind-bb <number>       Default: 0.5
  --big-blind-bb <number>         Default: 1
  --effective-stack-bb <number>   Default: 10
  --iterations <integer>          Default: 25000000
  --equity-samples <integer>      Default: 1024
  --seed <integer>                Default: 1
  --output <path>                 Default: push-fold-artifact.json

Blueprint options:
  --effective-stack-bb <number>   Default: 100
  --iterations <integer>          Default: 100000
  --max-information-sets <int>    Default: 5000000 memory guard
  --seed <integer>                Default: 1
  --averaging-delay <integer>     Default: 1000
  --held-out-deals <integer>      Default: 10000
  --held-out-seed <integer>       Fixed evaluation seed
  --root-deviation-samples <int>  Default: 256 deals per hand class
  --root-deviation-seed <integer> Fixed local-deviation seed
  --action-value-deals <integer>  Default: 10000 independent policy deals
  --action-value-seed <integer>   Fixed per-action evaluation seed
  --dcfr-alpha <number>           Positive-regret exponent (default: 1.5)
  --dcfr-beta <number>            Negative-regret exponent (default: 0)
  --dcfr-gamma <number>           Average-strategy exponent (default: 2)
  --distribution-samples <int>    Default: 128 per visible-card bucket
  --equity-bins <int>             Default: 10
  --potential-bins <int>          Default: 3
  --preflop-runout-samples <int>  Default: 256
  --flop-runout-samples <int>     Default: 128
  --sample-turn-rivers            Sample instead of enumerating turn rivers
  --enumerate-turn-river-chance   Enumerate 44 legal rivers in neural traversals
  --compact-serving-grid          Remove 4bb/5bb opens; retain other sizes
  --open-sizes-bb <csv>           Default: 2,2.5,3,4,5
  --limp-raise-sizes-bb <csv>     Default: 3,4,5
  --three-bet-sizes-bb <csv>      Default: 7.5,9,11
  --four-bet-sizes-bb <csv>       Default: 18,22,26
  --deeper-raise-pot-fractions    Default: .75,1,1.25
  --flop-bet-fractions <csv>      Default: .333,.75,1.25
  --turn-river-bet-fractions      Default: .5,1
  --postflop-raise-fractions      Default: 1
  --preflop-raise-cap <int>       Default: 4
  --postflop-raise-cap <int>      Default: 1
  --no-all-in                     Remove all-in from the action grid
  --checkpoint <path>             Optional resumable checkpoint
  --checkpoint-every <integer>    Default: 0 (final only)
  --resume <path>                 Resume compatible checkpoint
  --export-postflop-strategies    Include trained postflop profile (large)
  --current-street-recall         Opt into imperfect recall (not publishable)
  --output <path>                 Default: blueprint-artifact.json

Neural sample options:
  --effective-stack-bb <number>   Default: 20
  --traversals <integer>          Default: 100 alternating traversers
  --start-iteration <integer>     Global iteration used for weighting/parity
  --seed <integer>                Deterministic batch chance seed
  --max-records <integer>         Default: 50000 bounded output guard
  --networks <path>               Frozen advantage-network JSON (uniform if absent)
  --sample-trajectories           Sample both players; evaluation only
  --evaluate-action-values       Evaluate every reached trajectory action; requires 2+ rollouts
  --value-rollouts-per-action <N> Independent external samples per value target; default: 1
  --preflop-runout-samples <int>  Default: 256
  --flop-runout-samples <int>     Default: 128
  --sample-turn-rivers            Sample instead of enumerating turn rivers
  --compact-serving-grid          Opt-in reduced open grid
  --output <path>                 Default: neural-samples.jsonl.gz

Neural certificate options:
  --effective-stack-bb <number>   Default: 20
  --networks <path>               Required frozen policy-network JSON
  --deals <integer>               Default: 10000 exact-card chance samples
  --seed <integer>                Default: deterministic evaluation seed
  --confidence <number>           Default: 0.99 one-sided bound
  --threads <integer>             Default: available logical CPUs
  --compact-serving-grid          Match an opt-in reduced-open model
  --output <path>                 Optional JSON certificate file

Preflop continuation/solve options:
  preflop-cache --networks <json> [--networks-b <json>] [--deals 2652] [--rollouts-per-leaf 8]
  preflop-cache-compare --cache-a <json.gz> --cache-b <json.gz> [--output <json>]
  preflop-cache-merge --cache-a <json.gz> --cache-b <json.gz> --output <json.gz>
  preflop-cache-refresh --cache <json.gz> --output <json.gz>
  preflop-cache-inspect --cache <json.gz>
  preflop-dcfr --cache <json.gz> [--iterations 100000] [--seed 1]
    [--solver dcfr|mccfr-plus] [--dcfr-alpha 1.5] [--dcfr-beta 0]
    [--dcfr-gamma 2] [--exploration 0.05]
  preflop-evaluate --cache <json.gz> --policy <json> [--output <json>]
  preflop-attribution --cache <json.gz> --policy <json> [--top 50] [--output <json>]
  preflop-compact --policy <json> [--output <bin>]
  preflop-distill-samples --cache <json.gz> --policy <json> --output <jsonl.gz>
  preflop-evaluate-neural --cache <json.gz> --networks <json> [--output <json>]

Full-game learned-response options:
  --effective-stack-bb <number>   Default: 20
  --networks <path>               Required frozen routed policy JSON
  --training-deals <integer>      Default: 10000 response-training deals
  --evaluation-deals <integer>    Default: 10000 independent paired deals
  --rollouts-per-action <integer> Default: 8 common-random action rollouts
  --minimum-range-particles <N>   Default: 4; minimum: 2
  --seed <integer>                Deterministic training/evaluation root seed
  --output <path>                 Optional full resolver/evaluation JSON"
    );
}
