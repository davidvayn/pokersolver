use preflop_solver::blueprint::{self, BlueprintConfig, RecallMode, RunControl};
use preflop_solver::kuhn;
use preflop_solver::push_fold::{self, PushFoldConfig};
use sha2::{Digest, Sha256};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let command = args.first().map(String::as_str).unwrap_or("help");
    match command {
        "kuhn" => run_kuhn(&args[1..]),
        "solve" => run_push_fold(&args[1..]),
        "blueprint" => run_blueprint(&args[1..]),
        "neural-samples" => run_neural_samples(&args[1..]),
        "neural-certificate" => run_neural_certificate(&args[1..]),
        "neural-causal-attribution" => run_neural_causal_attribution(&args[1..]),
        "neural-causal-attribution-evaluate" => run_neural_causal_attribution_evaluate(&args[1..]),
        "range-policy-self-play-samples" => run_range_policy_self_play_samples(&args[1..]),
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
        "turn-river-pbs-solve" => run_turn_river_pbs_solve(&args[1..]),
        "turn-pbs-targets" => run_turn_pbs_targets(&args[1..]),
        "turn-pbs-upgrade-targets" => run_turn_pbs_upgrade_targets(&args[1..]),
        "turn-pbs-compose-upgrade" => run_turn_pbs_compose_upgrade(&args[1..]),
        "flop-pbs-resolve" => run_flop_pbs_resolve(&args[1..]),
        "flop-pbs-convergence" => run_flop_pbs_convergence(&args[1..]),
        "flop-pbs-evaluate" => run_flop_pbs_evaluate(&args[1..]),
        "flop-pbs-range-response" => run_flop_pbs_range_response(&args[1..]),
        "flop-pbs-leaf-targets" => run_flop_pbs_leaf_targets(&args[1..]),
        "postflop-action-targets" => run_postflop_action_targets(&args[1..]),
        "range-policy-add-baseline" => run_range_policy_add_baseline(&args[1..]),
        "range-policy-evaluate" => run_range_policy_evaluate(&args[1..]),
        "range-policy-causal-evaluate" => run_range_policy_causal_evaluate(&args[1..]),
        "range-policy-compare" => run_range_policy_compare(&args[1..]),
        "turn-pbs-self-play-targets" => run_turn_pbs_self_play_targets(&args[1..]),
        "turn-pbs-merge-targets" => run_turn_pbs_merge_targets(&args[1..]),
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
    let mut resolver_game = BlueprintConfig::default();
    apply_dcfr_args(&mut resolver_game, args)?;
    let cache = blueprint::preflop::build_resolver_continuation_cache(
        &base,
        blueprint::preflop::ResolverContinuationCacheConfig {
            deal_offset: parse_or(args, "--deal-offset", 0usize)?,
            deals: parse_or(args, "--deals", 2usize)?,
            resolver_iterations: iterations,
            resolver_averaging_delay: parse_or(
                args,
                "--resolver-averaging-delay",
                iterations / 10,
            )?,
            resolver_regret_matching_plus: args
                .iter()
                .any(|argument| argument == "--regret-matching-plus"),
            resolver_dcfr: resolver_game.dcfr,
            value_uncertainty_bb: parse_or(args, "--value-uncertainty-bb", 1.0f64)?,
            value_network_path: network_path,
            evaluation_value_network_path: value(args, "--evaluation-value-network")
                .map(PathBuf::from),
            range_policy_path: value(args, "--range-policy").map(PathBuf::from),
            source_cache_path: base_path,
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
            "maximumTurnRiverExploitabilityBbPerHand": dataset.targets.iter().map(|target| target.maximum_river_exploitability_bb_per_hand).fold(0.0f64, f64::max),
            "validation": dataset.validation,
            "output": path,
        }))?
    );
    Ok(())
}

fn run_turn_pbs_merge_targets(args: &[String]) -> Result<(), Box<dyn Error>> {
    let paths = values(args, "--dataset")
        .into_iter()
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    if paths.len() < 2 {
        return Err("turn-pbs-merge-targets requires at least two --dataset paths".into());
    }
    let mut components = Vec::with_capacity(paths.len());
    for path in &paths {
        let bytes = fs::read(path)?;
        let hash = format!("{:x}", Sha256::digest(&bytes));
        let dataset: blueprint::public_belief::TurnTargetDataset = serde_json::from_slice(&bytes)?;
        components.push((dataset, hash));
    }
    let merged = blueprint::public_belief::merge_turn_target_datasets(components)?;
    let output = value(args, "--output")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("turn-pbs-targets-merged.json"));
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let temporary = output.with_extension("tmp");
    fs::write(&temporary, serde_json::to_vec(&merged)?)?;
    fs::rename(temporary, &output)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": merged.schema,
            "states": merged.targets.len(),
            "componentDatasetSha256": merged.component_dataset_sha256,
            "componentSeeds": merged.component_seeds,
            "componentTargetCounts": merged.component_target_counts,
            "validation": merged.validation,
            "output": output,
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
            "maximumTurnRiverExploitabilityBbPerHand": dataset.targets.iter().map(|target| target.maximum_river_exploitability_bb_per_hand).fold(0.0f64, f64::max),
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
    apply_dcfr_args(&mut game, args)?;
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
    let resolve_config = blueprint::public_belief::FlopResolveConfig {
        game,
        state: blueprint::public_belief::PublicBeliefState::flop_start(
            board,
            parse_or(args, "--actor", 1usize)?,
            [pot_bb / 2.0, pot_bb / 2.0],
            ranges,
        ),
        iterations,
        averaging_delay: parse_or(args, "--averaging-delay", iterations / 10)?,
        regret_matching_plus: args
            .iter()
            .any(|argument| argument == "--regret-matching-plus"),
        value_network: network,
        auxiliary_value_networks: values(args, "--auxiliary-value-network")
            .iter()
            .map(|path| blueprint::public_belief::PublicValueNetwork::read(Path::new(path)))
            .collect::<Result<Vec<_>, _>>()?,
        threads: parse_or(
            args,
            "--threads",
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
        )?,
    };
    let solution = if let Some(path) = value(args, "--evaluation-value-network") {
        let evaluation_network =
            blueprint::public_belief::PublicValueNetwork::read(Path::new(&path))?;
        blueprint::public_belief::solve_flop_cross_evaluated(resolve_config, evaluation_network)?
    } else {
        blueprint::public_belief::solve_flop(resolve_config)?
    };
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

fn run_flop_pbs_convergence(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut game = BlueprintConfig::default();
    game.effective_stack_bb = parse_or(args, "--effective-stack-bb", 20.0)?;
    game.iterations = 2;
    game.averaging_delay = 0;
    apply_dcfr_args(&mut game, args)?;
    let board = parse_board::<3>(
        &value(args, "--board").ok_or("--board is required, for example 2c,7d,Th")?,
    )?;
    let network_path = value(args, "--value-network")
        .map(PathBuf::from)
        .ok_or("--value-network is required")?;
    let evaluation_path = value(args, "--evaluation-value-network")
        .map(PathBuf::from)
        .ok_or("--evaluation-value-network is required")?;
    let checkpoints = value(args, "--checkpoints")
        .ok_or("--checkpoints is required, for example 100,200,400")?
        .split(',')
        .map(str::parse::<u64>)
        .collect::<Result<Vec<_>, _>>()?;
    let iterations = checkpoints
        .last()
        .copied()
        .ok_or("--checkpoints must not be empty")?;
    let ranges = std::array::from_fn(|_| blueprint::public_belief::uniform_range(&board));
    let report = blueprint::public_belief::diagnose_flop_cross_evaluated_convergence(
        blueprint::public_belief::FlopResolveConfig {
            game,
            state: blueprint::public_belief::PublicBeliefState::flop_start(
                board,
                parse_or(args, "--actor", 1usize)?,
                {
                    let pot_bb = parse_or(args, "--pot-bb", 4.0f64)?;
                    [pot_bb / 2.0, pot_bb / 2.0]
                },
                ranges,
            ),
            iterations,
            averaging_delay: parse_or(args, "--averaging-delay", iterations / 10)?,
            regret_matching_plus: args
                .iter()
                .any(|argument| argument == "--regret-matching-plus"),
            value_network: blueprint::public_belief::PublicValueNetwork::read(&network_path)?,
            auxiliary_value_networks: values(args, "--auxiliary-value-network")
                .iter()
                .map(|path| blueprint::public_belief::PublicValueNetwork::read(Path::new(path)))
                .collect::<Result<Vec<_>, _>>()?,
            threads: parse_or(
                args,
                "--threads",
                std::thread::available_parallelism()
                    .map(usize::from)
                    .unwrap_or(1),
            )?,
        },
        blueprint::public_belief::PublicValueNetwork::read(&evaluation_path)?,
        &checkpoints,
    )?;
    let output = serde_json::to_string_pretty(&report)?;
    if let Some(path) = value(args, "--output").map(PathBuf::from) {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(&path, format!("{output}\n"))?;
        eprintln!("wrote flop convergence diagnostic {}", path.display());
    } else {
        println!("{output}");
    }
    Ok(())
}

fn run_flop_pbs_range_response(args: &[String]) -> Result<(), Box<dyn Error>> {
    let evaluation_path = value(args, "--evaluation-value-network")
        .map(PathBuf::from)
        .ok_or("--evaluation-value-network is required")?;
    let frozen: blueprint::public_belief::FlopSolution =
        if let Some(path) = value(args, "--solution").map(PathBuf::from) {
            serde_json::from_slice(&fs::read(path)?)?
        } else if let Some(path) = value(args, "--convergence-report").map(PathBuf::from) {
            let report: blueprint::public_belief::FlopConvergenceReport =
                serde_json::from_slice(&fs::read(path)?)?;
            report.solution_at_iterations(
                value(args, "--strategy-iterations")
                    .map(|raw| raw.parse::<u64>())
                    .transpose()?,
            )?
        } else {
            return Err("--solution or --convergence-report is required".into());
        };
    let evaluation = blueprint::public_belief::PublicValueNetwork::read(&evaluation_path)?;
    let checkpoints = value(args, "--checkpoints")
        .ok_or("--checkpoints is required, for example 25,50,100")?
        .split(',')
        .map(str::parse::<u64>)
        .collect::<Result<Vec<_>, _>>()?;
    let mut game = BlueprintConfig::default();
    game.effective_stack_bb = parse_or(args, "--effective-stack-bb", 20.0)?;
    apply_dcfr_args(&mut game, args)?;
    let report = blueprint::public_belief::evaluate_frozen_flop_range_response_convergence(
        game,
        &frozen,
        evaluation,
        &checkpoints,
        parse_or(args, "--averaging-delay", 0u64)?,
        args.iter()
            .any(|argument| argument == "--regret-matching-plus"),
        parse_or(
            args,
            "--threads",
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
        )?,
    )?;
    let output = serde_json::to_string_pretty(&report)?;
    if let Some(path) = value(args, "--output").map(PathBuf::from) {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(&path, format!("{output}\n"))?;
        eprintln!("wrote flop range-response diagnostic {}", path.display());
    } else {
        println!("{output}");
    }
    Ok(())
}

fn run_flop_pbs_evaluate(args: &[String]) -> Result<(), Box<dyn Error>> {
    let solution_path = value(args, "--solution")
        .map(PathBuf::from)
        .ok_or("--solution is required")?;
    let evaluation_path = value(args, "--evaluation-value-network")
        .map(PathBuf::from)
        .ok_or("--evaluation-value-network is required")?;
    let frozen: blueprint::public_belief::FlopSolution =
        serde_json::from_slice(&fs::read(solution_path)?)?;
    let evaluation = blueprint::public_belief::PublicValueNetwork::read(&evaluation_path)?;
    let mut game = BlueprintConfig::default();
    game.effective_stack_bb = parse_or(args, "--effective-stack-bb", 20.0)?;
    let scored = blueprint::public_belief::evaluate_frozen_flop_solution(
        game,
        &frozen,
        evaluation,
        parse_or(
            args,
            "--threads",
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
        )?,
    )?;
    let output = serde_json::to_string_pretty(&scored)?;
    if let Some(path) = value(args, "--output").map(PathBuf::from) {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(&path, format!("{output}\n"))?;
        println!("{}", serde_json::to_string_pretty(&scored.metrics)?);
        eprintln!(
            "wrote cross-evaluated frozen flop solution {}",
            path.display()
        );
    } else {
        println!("{output}");
    }
    Ok(())
}

fn run_postflop_action_targets(args: &[String]) -> Result<(), Box<dyn Error>> {
    let source_policy_path = value(args, "--networks")
        .map(PathBuf::from)
        .ok_or("--networks is required")?;
    let value_network_path = value(args, "--value-network")
        .map(PathBuf::from)
        .ok_or("--value-network is required")?;
    let output = value(args, "--output")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("postflop-action-targets.jsonl.gz"));
    let mut game = BlueprintConfig::default();
    game.effective_stack_bb = parse_or(args, "--effective-stack-bb", 20.0)?;
    game.iterations = 2;
    game.averaging_delay = 0;
    apply_dcfr_args(&mut game, args)?;
    if args
        .iter()
        .any(|argument| argument == "--compact-serving-grid")
    {
        game.action_abstraction = blueprint::ActionAbstraction::compact_serving_candidate();
    }
    let requested_flop_iterations = parse_or(args, "--flop-iterations", 400u64)?;
    let flop_iteration_checkpoints = value(args, "--flop-checkpoints")
        .map(|values| {
            values
                .split(',')
                .map(str::parse::<u64>)
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_else(|| vec![requested_flop_iterations]);
    let flop_iterations = *flop_iteration_checkpoints
        .last()
        .ok_or("--flop-checkpoints must not be empty")?;
    let turn_river_iterations = parse_or(args, "--turn-river-iterations", 400u64)?;
    let flop_response_checkpoints = value(args, "--flop-response-checkpoints")
        .map(|values| {
            values
                .split(',')
                .map(str::parse::<u64>)
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_default();
    let default_response_delay = flop_response_checkpoints.first().copied().unwrap_or(1) / 4;
    let report = blueprint::public_belief::generate_postflop_action_targets(
        blueprint::public_belief::PostflopActionTargetConfig {
            game,
            roots: parse_or(args, "--roots", 1usize)?,
            root_offset: parse_or(args, "--root-offset", 0usize)?,
            turn_leaves_per_root: parse_or(args, "--turn-leaves-per-root", 1usize)?,
            flop_iterations,
            flop_iteration_checkpoints,
            flop_averaging_delay: parse_or(args, "--flop-averaging-delay", flop_iterations / 4)?,
            flop_regret_matching_plus: args
                .iter()
                .any(|argument| argument == "--flop-regret-matching-plus"),
            require_accepted_flop_teachers: args
                .iter()
                .any(|argument| argument == "--require-accepted-flop-teachers"),
            require_range_consistent_flop_teachers: args
                .iter()
                .any(|argument| argument == "--require-range-consistent-flop-teachers"),
            flop_response_checkpoints,
            flop_response_averaging_delay: parse_or(
                args,
                "--flop-response-averaging-delay",
                default_response_delay,
            )?,
            flop_response_regret_matching_plus: args
                .iter()
                .any(|argument| argument == "--flop-response-regret-matching-plus"),
            maximum_flop_range_response_gain_bb_per_hand: parse_or(
                args,
                "--maximum-flop-range-response-gain-bb",
                0.05f64,
            )?,
            require_accepted_turn_river_teachers: args
                .iter()
                .any(|argument| argument == "--require-accepted-turn-river-teachers"),
            turn_river_iterations,
            turn_river_averaging_delay: parse_or(
                args,
                "--turn-river-averaging-delay",
                turn_river_iterations / 10,
            )?,
            seed: parse_or(args, "--seed", 16_001u64)?,
            threads: parse_or(
                args,
                "--threads",
                std::thread::available_parallelism()
                    .map(usize::from)
                    .unwrap_or(1),
            )?,
            exploration_probability: parse_or(args, "--exploration", 0.05f64)?,
            max_records: parse_or(args, "--max-records", 100_000usize)?,
            source_policy_path,
            value_network_path,
            auxiliary_value_network_paths: values(args, "--auxiliary-value-network")
                .into_iter()
                .map(PathBuf::from)
                .collect(),
            evaluation_value_network_path: value(args, "--evaluation-value-network")
                .map(PathBuf::from),
            output,
            range_output: value(args, "--range-output").map(PathBuf::from),
            range_only: args.iter().any(|argument| argument == "--range-only"),
        },
    )?;
    let serialized = serde_json::to_string_pretty(&report)?;
    if let Some(path) = value(args, "--report").map(PathBuf::from) {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, format!("{serialized}\n"))?;
    }
    println!("{serialized}");
    Ok(())
}

fn run_range_policy_evaluate(args: &[String]) -> Result<(), Box<dyn Error>> {
    let network = value(args, "--network")
        .map(PathBuf::from)
        .ok_or("--network is required")?;
    let dataset = value(args, "--dataset")
        .map(PathBuf::from)
        .ok_or("--dataset is required")?;
    let report = blueprint::public_belief::evaluate_range_conditioned_policy_dataset(
        &network,
        &dataset,
        args.iter()
            .any(|argument| argument == "--allow-independent-dataset"),
        value(args, "--source-network")
            .map(PathBuf::from)
            .as_deref(),
    )?;
    let serialized = serde_json::to_string_pretty(&report)?;
    if let Some(path) = value(args, "--output").map(PathBuf::from) {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(&path, format!("{serialized}\n"))?;
        eprintln!(
            "wrote exact Rust range-policy evaluation {}",
            path.display()
        );
    } else {
        println!("{serialized}");
    }
    Ok(())
}

fn run_range_policy_causal_evaluate(args: &[String]) -> Result<(), Box<dyn Error>> {
    let frozen_network_path = value(args, "--frozen-network")
        .map(PathBuf::from)
        .ok_or("--frozen-network is required")?;
    let attribution_network_path = value(args, "--attribution-network")
        .map(PathBuf::from)
        .unwrap_or_else(|| frozen_network_path.clone());
    let report = blueprint::public_belief::evaluate_causal_range_policy(
        blueprint::public_belief::CausalRangePolicyEvaluationConfig {
            network_path: value(args, "--network")
                .map(PathBuf::from)
                .ok_or("--network is required")?,
            frozen_network_path,
            attribution_network_path,
            dataset_path: value(args, "--dataset")
                .map(PathBuf::from)
                .ok_or("--dataset is required")?,
            source_policy_path: value(args, "--source-network").map(PathBuf::from),
            minimum_policy_value_gain_bb: parse_or(
                args,
                "--minimum-policy-value-gain-bb",
                0.000001f64,
            )?,
            maximum_node_kl: parse_or(args, "--maximum-node-kl", 0.005f64)?,
            maximum_weighted_kl: parse_or(args, "--maximum-weighted-kl", 0.0015f64)?,
        },
    )?;
    let serialized = serde_json::to_string_pretty(&report)?;
    if let Some(path) = value(args, "--output").map(PathBuf::from) {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(&path, format!("{serialized}\n"))?;
        eprintln!(
            "wrote exact Rust causal range-policy evaluation {}",
            path.display()
        );
    } else {
        println!("{serialized}");
    }
    Ok(())
}

fn run_range_policy_compare(args: &[String]) -> Result<(), Box<dyn Error>> {
    let network_a = value(args, "--network-a")
        .map(PathBuf::from)
        .ok_or("--network-a is required")?;
    let network_b = value(args, "--network-b")
        .map(PathBuf::from)
        .ok_or("--network-b is required")?;
    let datasets = values(args, "--dataset")
        .into_iter()
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    let source_a = value(args, "--source-network-a").map(PathBuf::from);
    let source_b = value(args, "--source-network-b").map(PathBuf::from);
    let report = blueprint::public_belief::compare_range_conditioned_policies(
        [&network_a, &network_b],
        [source_a.as_deref(), source_b.as_deref()],
        &datasets,
    )?;
    let serialized = serde_json::to_string_pretty(&report)?;
    if let Some(path) = value(args, "--output").map(PathBuf::from) {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(&path, format!("{serialized}\n"))?;
        eprintln!(
            "wrote exact Rust range-policy comparison {}",
            path.display()
        );
    } else {
        println!("{serialized}");
    }
    Ok(())
}

fn run_range_policy_add_baseline(args: &[String]) -> Result<(), Box<dyn Error>> {
    let source = value(args, "--source-network")
        .map(PathBuf::from)
        .ok_or("--source-network is required")?;
    let input = value(args, "--dataset")
        .map(PathBuf::from)
        .ok_or("--dataset is required")?;
    let output = value(args, "--output")
        .map(PathBuf::from)
        .ok_or("--output is required")?;
    let report = blueprint::public_belief::attach_source_policy_baseline(&source, &input, &output)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
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

fn run_turn_river_pbs_solve(args: &[String]) -> Result<(), Box<dyn Error>> {
    let dataset_and_state = if let Some(path) = value(args, "--dataset").map(PathBuf::from) {
        let dataset: blueprint::public_belief::TurnTargetDataset =
            serde_json::from_slice(&fs::read(path)?)?;
        let state_index = parse_or(args, "--state-index", 0usize)?;
        let target = dataset
            .targets
            .get(state_index)
            .ok_or("--state-index is outside the target dataset")?;
        let ranges = std::array::from_fn(|player| {
            target.ranges[player]
                .iter()
                .map(|value| *value as f64)
                .collect::<Vec<_>>()
        });
        Some((
            dataset.game,
            blueprint::public_belief::PublicBeliefState::turn_start(
                target.board,
                target.actor,
                target.invested_bb,
                ranges,
            ),
        ))
    } else {
        None
    };
    let mut game = dataset_and_state
        .as_ref()
        .map(|(game, _)| game.clone())
        .unwrap_or_default();
    game.effective_stack_bb = parse_or(args, "--effective-stack-bb", game.effective_stack_bb)?;
    game.iterations = 2;
    game.averaging_delay = 0;
    apply_dcfr_args(&mut game, args)?;
    if args
        .iter()
        .any(|argument| argument == "--compact-serving-grid")
    {
        game.action_abstraction = blueprint::ActionAbstraction::compact_serving_candidate();
    }
    let iterations = parse_or(args, "--iterations", 500u64)?;
    let averaging_delay = parse_or(args, "--averaging-delay", iterations / 10)?;
    let state = if let Some((_, state)) = dataset_and_state {
        state
    } else {
        let board =
            parse_board::<4>(&value(args, "--board").ok_or("--board or --dataset is required")?)?;
        let pot_bb = parse_or(args, "--pot-bb", 4.0f64)?;
        let ranges = std::array::from_fn(|_| blueprint::public_belief::uniform_range(&board));
        blueprint::public_belief::PublicBeliefState::turn_start(
            board,
            parse_or(args, "--actor", 1usize)?,
            [pot_bb / 2.0, pot_bb / 2.0],
            ranges,
        )
    };
    let config = blueprint::public_belief::TurnRiverSolveConfig {
        game,
        state,
        iterations,
        averaging_delay,
        river_refinement_iterations: parse_or(args, "--river-refinement-iterations", 0u64)?,
        regret_matching_plus: args
            .iter()
            .any(|argument| argument == "--regret-matching-plus"),
    };
    let export_strategies = args
        .iter()
        .any(|argument| argument == "--export-strategies");
    let (output, metrics) = if export_strategies {
        let solution = blueprint::public_belief::solve_turn_river(config)?;
        (
            serde_json::to_string_pretty(&solution)?,
            serde_json::to_string_pretty(&solution.metrics)?,
        )
    } else {
        let values = blueprint::public_belief::solve_turn_river_continuation_values(config)?;
        (
            serde_json::to_string_pretty(&values)?,
            serde_json::to_string_pretty(&values.metrics)?,
        )
    };
    if let Some(path) = value(args, "--output").map(PathBuf::from) {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(&path, format!("{output}\n"))?;
        println!("{metrics}");
        eprintln!(
            "wrote exact-card-removal turn-river solution {}",
            path.display()
        );
    } else {
        println!("{output}");
    }
    Ok(())
}

fn turn_upgrade_checkpoint_path(directory: &Path, index: usize, fingerprint: &str) -> PathBuf {
    directory.join(format!("turn-{index:06}-{fingerprint}.json"))
}

fn turn_upgrade_fingerprint(
    game: &BlueprintConfig,
    target: &blueprint::public_belief::TurnValueTarget,
    iterations: u64,
    averaging_delay: u64,
) -> Result<String, Box<dyn Error>> {
    let ranges = std::array::from_fn(|player| {
        target.ranges[player]
            .iter()
            .map(|value| *value as f64)
            .collect::<Vec<_>>()
    });
    Ok(blueprint::public_belief::turn_target_input_sha256(
        game,
        target.board,
        target.actor,
        target.invested_bb,
        &ranges,
        iterations,
        averaging_delay,
    )?)
}

fn run_turn_pbs_upgrade_targets(args: &[String]) -> Result<(), Box<dyn Error>> {
    let dataset_path = value(args, "--dataset")
        .map(PathBuf::from)
        .ok_or("--dataset is required")?;
    let checkpoint_dir = value(args, "--checkpoint-dir")
        .map(PathBuf::from)
        .ok_or("--checkpoint-dir is required")?;
    let mut dataset: blueprint::public_belief::TurnTargetDataset =
        serde_json::from_slice(&fs::read(&dataset_path)?)?;
    apply_dcfr_args(&mut dataset.game, args)?;
    let iterations = parse_or(
        args,
        "--iterations",
        dataset
            .turn_river_iterations
            .unwrap_or(dataset.river_iterations),
    )?;
    let averaging_delay = parse_or(args, "--averaging-delay", iterations / 10)?;
    if iterations < 2 || averaging_delay >= iterations {
        return Err("upgrade iterations and averaging delay are invalid".into());
    }
    let start = parse_or(args, "--start-index", 0usize)?;
    let end = parse_or(args, "--end-index", dataset.targets.len())?;
    if start >= end || end > dataset.targets.len() {
        return Err(format!(
            "upgrade range [{start}, {end}) is outside {} targets",
            dataset.targets.len()
        )
        .into());
    }
    fs::create_dir_all(&checkpoint_dir)?;
    for index in start..end {
        let source = &dataset.targets[index];
        let fingerprint =
            turn_upgrade_fingerprint(&dataset.game, source, iterations, averaging_delay)?;
        let path = turn_upgrade_checkpoint_path(&checkpoint_dir, index, &fingerprint);
        if path.exists() {
            let cached: blueprint::public_belief::TurnValueTarget =
                serde_json::from_slice(&fs::read(&path)?)?;
            if cached.input_sha256.as_deref() != Some(fingerprint.as_str())
                || cached.state_id != source.state_id
                || cached.board != source.board
                || cached.actor != source.actor
                || cached.invested_bb != source.invested_bb
                || cached.ranges != source.ranges
                || cached.turn_river_exploitability_bb_per_hand.is_none()
                || cached
                    .current_turn_river_exploitability_bb_per_hand
                    .is_none()
                || cached.turn_river_maximum_probability_sum_error.is_none()
                || cached.turn_only_best_response_gain_bb_per_hand.is_none()
                || cached.river_only_best_response_gain_bb_per_hand.is_none()
                || cached.turn_river_solver_method.is_none()
                || cached.turn_river_information_sets.is_none()
                || cached.turn_information_sets.is_none()
                || cached.river_information_sets.is_none()
            {
                return Err(format!("invalid upgrade checkpoint {}", path.display()).into());
            }
            println!("cached {index} {}", path.display());
            continue;
        }
        let upgraded = blueprint::public_belief::upgrade_turn_value_target(
            &dataset.game,
            source,
            iterations,
            averaging_delay,
        )?;
        let temporary = path.with_extension("json.tmp");
        fs::write(&temporary, serde_json::to_vec(&upgraded)?)?;
        fs::rename(&temporary, &path)?;
        println!(
            "solved {index} exploitability={:.9}bb/hand {}",
            upgraded
                .turn_river_exploitability_bb_per_hand
                .expect("upgraded target records exploitability"),
            path.display()
        );
    }
    Ok(())
}

fn run_turn_pbs_compose_upgrade(args: &[String]) -> Result<(), Box<dyn Error>> {
    let dataset_path = value(args, "--dataset")
        .map(PathBuf::from)
        .ok_or("--dataset is required")?;
    let checkpoint_dir = value(args, "--checkpoint-dir")
        .map(PathBuf::from)
        .ok_or("--checkpoint-dir is required")?;
    let output_path = value(args, "--output")
        .map(PathBuf::from)
        .ok_or("--output is required")?;
    let mut dataset: blueprint::public_belief::TurnTargetDataset =
        serde_json::from_slice(&fs::read(&dataset_path)?)?;
    apply_dcfr_args(&mut dataset.game, args)?;
    let iterations = parse_or(
        args,
        "--iterations",
        dataset
            .turn_river_iterations
            .unwrap_or(dataset.river_iterations),
    )?;
    let averaging_delay = parse_or(args, "--averaging-delay", iterations / 10)?;
    let mut upgraded = Vec::with_capacity(dataset.targets.len());
    for (index, source) in dataset.targets.iter().enumerate() {
        let fingerprint =
            turn_upgrade_fingerprint(&dataset.game, source, iterations, averaging_delay)?;
        let path = turn_upgrade_checkpoint_path(&checkpoint_dir, index, &fingerprint);
        let target: blueprint::public_belief::TurnValueTarget =
            serde_json::from_slice(&fs::read(&path).map_err(|error| {
                format!("missing upgrade checkpoint {}: {error}", path.display())
            })?)?;
        if target.input_sha256.as_deref() != Some(fingerprint.as_str())
            || target.state_id != source.state_id
            || target.board != source.board
            || target.actor != source.actor
            || target.invested_bb != source.invested_bb
            || target.ranges != source.ranges
            || target.range_particles != source.range_particles
            || target.range_replicates != source.range_replicates
            || target.range_effective_sample_size != source.range_effective_sample_size
            || target.belief_method != source.belief_method
            || target.range_maximum_total_variation != source.range_maximum_total_variation
            || target.off_policy_explorer != source.off_policy_explorer
            || target.sampling_exploration_probability != source.sampling_exploration_probability
            || target.public_action_line != source.public_action_line
            || target.resolver_root_board != source.resolver_root_board
            || target.resolver_public_history != source.resolver_public_history
            || target.resolver_leaf_reach_probability != source.resolver_leaf_reach_probability
            || target.turn_river_exploitability_bb_per_hand.is_none()
            || target
                .current_turn_river_exploitability_bb_per_hand
                .is_none()
            || target.turn_river_maximum_probability_sum_error.is_none()
            || target.turn_only_best_response_gain_bb_per_hand.is_none()
            || target.river_only_best_response_gain_bb_per_hand.is_none()
            || target.turn_river_solver_method.is_none()
            || target.turn_river_information_sets.is_none()
            || target.turn_information_sets.is_none()
            || target.river_information_sets.is_none()
        {
            return Err(format!(
                "upgrade checkpoint {} does not match source",
                path.display()
            )
            .into());
        }
        upgraded.push(target);
    }
    dataset.schema = "hu-turn-public-belief-cfv-dataset-v2".to_owned();
    dataset.method = format!(
        "complete_turn_river_public_belief_upgrade_of_{}",
        dataset.method
    );
    dataset.river_iterations = iterations;
    dataset.turn_river_iterations = Some(iterations);
    dataset.turn_river_averaging_delay = Some(averaging_delay);
    dataset.targets = upgraded;
    let maximum_exploitability = dataset
        .targets
        .iter()
        .filter_map(|target| target.turn_river_exploitability_bb_per_hand)
        .fold(0.0f64, f64::max);
    let maximum_zero_sum_residual = dataset
        .targets
        .iter()
        .map(|target| target.zero_sum_residual_bb.abs())
        .fold(0.0f64, f64::max);
    let maximum_probability_sum_error = dataset
        .targets
        .iter()
        .filter_map(|target| target.turn_river_maximum_probability_sum_error)
        .fold(0.0f64, f64::max);
    let mut reasons = if dataset.validation.status == "accepted" {
        Vec::new()
    } else {
        dataset
            .validation
            .reasons
            .iter()
            .filter(|reason| {
                !reason.contains("river solve") && !reason.contains("zero-sum residual")
            })
            .cloned()
            .collect()
    };
    for (index, target) in dataset.targets.iter().enumerate() {
        let exploitability = target
            .turn_river_exploitability_bb_per_hand
            .expect("checked complete target");
        let current_exploitability = target
            .current_turn_river_exploitability_bb_per_hand
            .expect("checked complete target");
        let probability_error = target
            .turn_river_maximum_probability_sum_error
            .expect("checked complete target");
        let turn_gain = target
            .turn_only_best_response_gain_bb_per_hand
            .expect("checked complete target");
        let river_gain = target
            .river_only_best_response_gain_bb_per_hand
            .expect("checked complete target");
        let method = target
            .turn_river_solver_method
            .as_deref()
            .expect("checked complete target");
        let information_sets = target
            .turn_river_information_sets
            .expect("checked complete target");
        let turn_information_sets = target
            .turn_information_sets
            .expect("checked complete target");
        let river_information_sets = target
            .river_information_sets
            .expect("checked complete target");
        if !exploitability.is_finite()
            || exploitability < 0.0
            || !current_exploitability.is_finite()
            || current_exploitability < 0.0
        {
            reasons.push(format!("target {index} has invalid exploitability metrics"));
        }
        if !probability_error.is_finite() || probability_error > 1e-6 {
            reasons.push(format!("target {index} probability-sum error exceeds 1e-6"));
        }
        if !turn_gain.is_finite()
            || turn_gain < 0.0
            || turn_gain > exploitability + 1e-8
            || !river_gain.is_finite()
            || river_gain < 0.0
            || river_gain > exploitability + 1e-8
        {
            reasons.push(format!(
                "target {index} has invalid street best-response attribution"
            ));
        }
        if !method.contains("complete_turn_river_betting") || !method.contains("paired_alternating")
        {
            reasons.push(format!(
                "target {index} lacks corrected paired-alternating solver provenance"
            ));
        }
        if target.exact_river_cards != 48
            || turn_information_sets == 0
            || river_information_sets == 0
            || turn_information_sets.checked_add(river_information_sets) != Some(information_sets)
        {
            reasons.push(format!(
                "target {index} has incomplete river or information-set provenance"
            ));
        }
        let vector_shapes_and_values_valid = (0..2).all(|player| {
            let range_len = target.ranges[player].len();
            range_len > 0
                && target.counterfactual_values_bb[player].len() == range_len
                && target.opponent_compatible_mass[player].len() == range_len
                && target.ranges[player]
                    .iter()
                    .all(|value| value.is_finite() && *value >= 0.0)
                && target.counterfactual_values_bb[player]
                    .iter()
                    .all(|value| value.is_finite())
                && target.opponent_compatible_mass[player]
                    .iter()
                    .all(|value| value.is_finite() && *value >= 0.0)
        });
        if !vector_shapes_and_values_valid {
            reasons.push(format!("target {index} has invalid continuation vectors"));
        }
    }
    if maximum_exploitability > 0.05 {
        reasons.push(format!(
            "maximum complete turn-river exploitability {maximum_exploitability:.6}bb/hand exceeds 0.05bb/hand"
        ));
    }
    if maximum_zero_sum_residual > 1e-7 {
        reasons.push(format!(
            "maximum complete turn-river zero-sum residual {maximum_zero_sum_residual:.3e}bb exceeds 1e-7bb"
        ));
    }
    if maximum_probability_sum_error > 1e-6 {
        reasons.push(format!(
            "maximum complete turn-river probability-sum error {maximum_probability_sum_error:.3e} exceeds 1e-6"
        ));
    }
    reasons.sort();
    reasons.dedup();
    dataset.validation = blueprint::BlueprintValidation {
        status: if reasons.is_empty() {
            "accepted"
        } else {
            "rejected"
        }
        .to_owned(),
        reasons,
    };
    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(&output_path, serde_json::to_vec(&dataset)?)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": dataset.schema,
            "states": dataset.targets.len(),
            "iterations": iterations,
            "averagingDelay": averaging_delay,
            "maximumTurnRiverExploitabilityBbPerHand": maximum_exploitability,
            "maximumZeroSumResidualBb": maximum_zero_sum_residual,
            "maximumProbabilitySumError": maximum_probability_sum_error,
            "validation": dataset.validation,
            "output": output_path,
        }))?
    );
    Ok(())
}

fn run_turn_pbs_targets(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut game = BlueprintConfig::default();
    game.effective_stack_bb = parse_or(args, "--effective-stack-bb", 20.0)?;
    game.iterations = 2;
    game.averaging_delay = 0;
    apply_dcfr_args(&mut game, args)?;
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
            "maximumTurnRiverExploitabilityBbPerHand": dataset.targets.iter().map(|target| target.maximum_river_exploitability_bb_per_hand).fold(0.0f64, f64::max),
            "maximumZeroSumResidualBb": dataset.targets.iter().map(|target| target.zero_sum_residual_bb).fold(0.0f64, f64::max),
            "validation": dataset.validation,
            "output": path,
        }))?
    );
    Ok(())
}

fn apply_dcfr_args(game: &mut BlueprintConfig, args: &[String]) -> Result<(), Box<dyn Error>> {
    game.dcfr.positive_regret_exponent =
        parse_or(args, "--dcfr-alpha", game.dcfr.positive_regret_exponent)?;
    game.dcfr.negative_regret_exponent =
        parse_or(args, "--dcfr-beta", game.dcfr.negative_regret_exponent)?;
    game.dcfr.strategy_exponent = parse_or(args, "--dcfr-gamma", game.dcfr.strategy_exponent)?;
    if !game.dcfr.positive_regret_exponent.is_finite()
        || !game.dcfr.negative_regret_exponent.is_finite()
        || !game.dcfr.strategy_exponent.is_finite()
        || game.dcfr.positive_regret_exponent < 0.0
        || game.dcfr.negative_regret_exponent < 0.0
        || game.dcfr.strategy_exponent < 0.0
    {
        return Err("DCFR exponents must be finite and non-negative".into());
    }
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
    let maximum_granularity = match value(args, "--maximum-response-granularity")
        .as_deref()
        .unwrap_or("exact")
    {
        "exact" => blueprint::response::ResolverGranularity::ExactTrajectory,
        "fine" => blueprint::response::ResolverGranularity::ObservableBackoff,
        "coarse" => blueprint::response::ResolverGranularity::CoarseObservableBackoff,
        "strategic" => blueprint::response::ResolverGranularity::StrategicObservableBackoff,
        value => {
            return Err(format!(
            "unsupported response granularity {value}; expected exact, fine, coarse, or strategic"
        )
            .into())
        }
    };
    let evaluation = blueprint::response::evaluate_full_game_response(
        blueprint::response::ResponseEvaluationConfig {
            game,
            training_deals: parse_or(args, "--training-deals", 10_000u64)?,
            calibration_deals: parse_or(args, "--calibration-deals", 2_000u64)?,
            evaluation_deals: parse_or(args, "--evaluation-deals", 10_000u64)?,
            rollouts_per_action: parse_or(args, "--rollouts-per-action", 8u32)?,
            minimum_range_particles: parse_or(args, "--minimum-range-particles", 4u64)?,
            maximum_granularity,
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
                "calibrationPlayers": evaluation.calibration_players,
                "responseDeployed": evaluation.response_deployed,
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
    let config = blueprint::neural::ExploitabilityCertificateConfig {
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
        range_policy_path: value(args, "--range-policy").map(PathBuf::from),
    };
    let opponent_samples = value(args, "--opponent-samples-per-deal")
        .map(|samples| samples.parse())
        .transpose()?;
    let opponent_samples_per_runout = value(args, "--opponent-samples-per-runout")
        .map(|samples| samples.parse())
        .transpose()?;
    let public_branches = value(args, "--public-branches-per-street")
        .map(|branches| branches.parse())
        .transpose()?;
    let certificate = match (
        public_branches,
        opponent_samples,
        opponent_samples_per_runout,
    ) {
        (Some(branches), None, Some(samples)) => {
            blueprint::neural::certify_causal_sample_game_exploitability_upper_bound(
                config, branches, samples,
            )?
        }
        (Some(_), _, _) => {
            return Err(
                "--public-branches-per-street requires only --opponent-samples-per-runout".into(),
            )
        }
        (None, Some(samples), None) => {
            blueprint::neural::certify_opponent_hidden_exploitability_upper_bound(config, samples)?
        }
        (None, None, None) => blueprint::neural::certify_exploitability_upper_bound(config)?,
        (None, _, Some(_)) => {
            return Err(
                "--opponent-samples-per-runout requires --public-branches-per-street".into(),
            )
        }
    };
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

fn run_neural_causal_attribution(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut game = BlueprintConfig::default();
    game.effective_stack_bb = parse_or(args, "--effective-stack-bb", 20.0)?;
    if args
        .iter()
        .any(|argument| argument == "--compact-serving-grid")
    {
        game.action_abstraction = blueprint::ActionAbstraction::compact_serving_candidate();
    }
    let output = value(args, "--output")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("causal-policy-attribution.jsonl.gz"));
    let report = blueprint::neural::generate_causal_policy_attribution(
        blueprint::neural::CausalPolicyAttributionConfig {
            game,
            deals: parse_or(args, "--deals", 8u64)?,
            seed: parse_or(args, "--seed", 0xA11CE5EEDu64)?,
            threads: parse_or(
                args,
                "--threads",
                std::thread::available_parallelism()
                    .map(usize::from)
                    .unwrap_or(1),
            )?,
            network_path: value(args, "--networks")
                .map(PathBuf::from)
                .ok_or("--networks is required for causal attribution")?,
            range_policy_path: value(args, "--range-policy").map(PathBuf::from),
            public_branches_per_street: parse_or(args, "--public-branches-per-street", 2u32)?,
            opponent_samples_per_runout: parse_or(args, "--opponent-samples-per-runout", 4u32)?,
            max_records: parse_or(args, "--max-records", 100_000usize)?,
            output,
        },
    )?;
    let serialized = serde_json::to_string_pretty(&report)?;
    if let Some(path) = value(args, "--report").map(PathBuf::from) {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, format!("{serialized}\n"))?;
    }
    println!("{serialized}");
    Ok(())
}

fn run_range_policy_self_play_samples(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut game = BlueprintConfig::default();
    game.effective_stack_bb = parse_or(args, "--effective-stack-bb", 20.0)?;
    if args
        .iter()
        .any(|argument| argument == "--compact-serving-grid")
    {
        game.action_abstraction = blueprint::ActionAbstraction::compact_serving_candidate();
    }
    let output = value(args, "--output")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("range-policy-self-play.jsonl.gz"));
    let report = blueprint::neural::generate_range_self_play_samples(
        blueprint::neural::RangeSelfPlaySampleConfig {
            game,
            traversals: parse_or(args, "--traversals", 100u64)?,
            start_iteration: parse_or(args, "--start-iteration", 0u64)?,
            seed: parse_or(args, "--seed", 0x5E1F_91A7u64)?,
            max_records: parse_or(args, "--max-records", 50_000usize)?,
            network_path: value(args, "--networks")
                .map(PathBuf::from)
                .ok_or("--networks is required for range self-play")?,
            range_policy_path: value(args, "--range-policy")
                .map(PathBuf::from)
                .ok_or("--range-policy is required for range self-play")?,
            value_rollouts_per_action: parse_or(args, "--value-rollouts-per-action", 4u32)?,
            enumerate_turn_river_chance: args
                .iter()
                .any(|argument| argument == "--enumerate-turn-river-chance"),
            output,
        },
    )?;
    let serialized = serde_json::to_string_pretty(&report)?;
    if let Some(path) = value(args, "--report").map(PathBuf::from) {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, format!("{serialized}\n"))?;
    }
    println!("{serialized}");
    Ok(())
}

fn run_neural_causal_attribution_evaluate(args: &[String]) -> Result<(), Box<dyn Error>> {
    let evaluation = blueprint::neural::evaluate_causal_attribution_policy(
        blueprint::neural::CausalAttributionPolicyEvaluationConfig {
            dataset_path: value(args, "--dataset")
                .map(PathBuf::from)
                .ok_or("--dataset is required for causal attribution evaluation")?,
            network_path: value(args, "--networks")
                .map(PathBuf::from)
                .ok_or("--networks is required for causal attribution evaluation")?,
            maximum_node_kl: parse_or(args, "--maximum-node-kl", 0.005f64)?,
            maximum_weighted_kl: parse_or(args, "--maximum-weighted-kl", 0.0015f64)?,
            minimum_policy_value_gain_bb: parse_or(
                args,
                "--minimum-policy-value-gain-bb",
                0.000001f64,
            )?,
        },
    )?;
    let serialized = serde_json::to_string_pretty(&evaluation)?;
    if let Some(path) = value(args, "--output").map(PathBuf::from) {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, format!("{serialized}\n"))?;
    }
    println!("{serialized}");
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

fn values(args: &[String], name: &str) -> Vec<String> {
    args.windows(2)
        .filter(|pair| pair[0] == name)
        .map(|pair| pair[1].clone())
        .collect()
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
  preflop-solver neural-causal-attribution [options]
  preflop-solver neural-causal-attribution-evaluate [options]
  preflop-solver range-policy-self-play-samples [options]
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
  preflop-solver turn-river-pbs-solve [options]
  preflop-solver turn-pbs-targets [options]
  preflop-solver turn-pbs-upgrade-targets [options]
  preflop-solver turn-pbs-compose-upgrade [options]
  preflop-solver turn-pbs-self-play-targets [options]
  preflop-solver turn-pbs-merge-targets --dataset <json> --dataset <json> [options]
  preflop-solver turn-pbs-value-predict [options]
  preflop-solver flop-pbs-resolve [options]
  preflop-solver flop-pbs-convergence [options]
  preflop-solver flop-pbs-range-response [options]
  preflop-solver flop-pbs-leaf-targets [options]
  preflop-solver postflop-action-targets [options]
  preflop-solver range-policy-add-baseline [options]
  preflop-solver range-policy-evaluate [options]
  preflop-solver range-policy-causal-evaluate [options]
  preflop-solver range-policy-compare [options]

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

Range-policy self-play sample options:
  --effective-stack-bb <number>   Default: 20
  --traversals <integer>          Default: 100 alternating traversers
  --start-iteration <integer>     Global iteration used for deterministic weighting
  --seed <integer>                Deterministic chance and rollout seed
  --max-records <integer>         Default: 50000 bounded output guard
  --networks <path>               Required frozen routed policy JSON
  --range-policy <path>           Required frozen range-policy JSON
  --value-rollouts-per-action <N> Default: 4 independent value samples
  --enumerate-turn-river-chance   Enumerate later-street chance during rollouts
  --compact-serving-grid          Match an opt-in reduced-open model
  --output <path>                 Default: range-policy-self-play.jsonl.gz
  --report <path>                 Optional generation report JSON

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
  preflop-cache-resolver --base-cache <json.gz> --value-network <json>
    [--evaluation-value-network <independent-json>]
    [--range-policy <tabular-policy.json>] [--deal-offset 0] [--deals 2]
    [--resolver-iterations 10] [--resolver-averaging-delay 1]
    [--regret-matching-plus]
    [--dcfr-alpha 1.5] [--dcfr-beta 0] [--dcfr-gamma 2]
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
  --calibration-deals <integer>   Default: 2000 disjoint response-selection deals
  --evaluation-deals <integer>    Default: 10000 independent paired deals
  --rollouts-per-action <integer> Default: 8 common-random action rollouts
  --minimum-range-particles <N>   Default: 4; minimum: 2
  --maximum-response-granularity <exact|fine|coarse|strategic>
                                  Default: exact; broader layers require calibration
  --seed <integer>                Deterministic training/evaluation root seed
  --output <path>                 Optional full resolver/evaluation JSON

Neural exploitability-certificate options:
  --opponent-samples-per-deal <N> Hide opponent cards and use N common
                                  conditional particles per future-board game
  --opponent-samples-per-runout <N>
                                  Hide N opponent hands under each causal runout
  --public-branches-per-street <N> With opponent samples, build N nested flop,
                                  turn, and river branches and reveal cards only
                                  when their street is reached

Complete turn/river label options:
  postflop-action-targets --networks <source-policy.json>
    --value-network <turn-value.json> --range-output <targets.jsonl.gz>
    [--root-offset 0] [--roots 1]
    [--evaluation-value-network <independent-turn-value.json>]
    [--flop-checkpoints <csv>] [--flop-response-checkpoints <csv>]
    [--require-range-consistent-flop-teachers]
    [--maximum-flop-range-response-gain-bb 0.05]
  range-policy-add-baseline --source-network <source-policy.json>
    --dataset <targets.jsonl.gz> --output <augmented.jsonl.gz>
  range-policy-evaluate --network <residual-policy.json>
    --source-network <source-policy.json> --dataset <targets.jsonl.gz>
  range-policy-causal-evaluate --network <candidate.json>
    --frozen-network <candidate-parent.json>
    [--attribution-network <dataset-source.json>]
    --dataset <causal-attribution.jsonl.gz>
    [--minimum-policy-value-gain-bb 0.000001]
    [--maximum-node-kl 0.005] [--maximum-weighted-kl 0.0015]
  range-policy-compare --network-a <policy-a.json> --network-b <policy-b.json>
    --dataset <heldout-a.jsonl.gz> --dataset <heldout-b.jsonl.gz>
    [--source-network-a <source-a.json> --source-network-b <source-b.json>]
  flop-pbs-convergence --board <3-card-csv> --value-network <json>
    --evaluation-value-network <json> --checkpoints <csv>
    [--pot-bb 4] [--actor 1] [--averaging-delay N] [--threads N]
    [--regret-matching-plus]
    [--dcfr-alpha 1.5] [--dcfr-beta 0] [--dcfr-gamma 2]
    [--output <json>]
  flop-pbs-range-response (--solution <flop-solution.json> |
    --convergence-report <flop-convergence.json>)
    --evaluation-value-network <json> --checkpoints <csv>
    [--strategy-iterations N]
    [--averaging-delay 0] [--regret-matching-plus] [--threads N]
    [--dcfr-alpha 1.5] [--dcfr-beta 0] [--dcfr-gamma 2]
    [--output <json>]
  turn-river-pbs-solve --board <4-card-csv> [--pot-bb 4] [--actor 1]
    [--dataset <targets.json> --state-index 0]
    [--iterations 500] [--averaging-delay 50] [--export-strategies]
    [--river-refinement-iterations 0]
    [--regret-matching-plus] [--dcfr-alpha 1.5] [--dcfr-beta 0]
    [--dcfr-gamma 2] [--output <json>]
  turn-pbs-upgrade-targets --dataset <legacy-v1.json>
    --checkpoint-dir <directory> [--start-index 0] [--end-index N]
    [--iterations N] [--averaging-delay N] [--dcfr-alpha 1.5]
    [--dcfr-beta 0] [--dcfr-gamma 2]
  turn-pbs-compose-upgrade --dataset <legacy-v1.json>
    --checkpoint-dir <directory> --output <complete-v2.json>
    [--iterations N] [--averaging-delay N] [--dcfr-alpha 1.5]
    [--dcfr-beta 0] [--dcfr-gamma 2]
  turn-pbs-merge-targets --dataset <v2.json> --dataset <v2.json>
    [--dataset <v2.json> ...] --output <merged-v2.json>"
    );
}
