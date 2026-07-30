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
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        unknown => Err(format!("unknown command: {unknown}").into()),
    }
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
        .any(|argument| argument == "--trajectory-recall")
    {
        config.recall_mode = RecallMode::Trajectory;
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
  --distribution-samples <int>    Default: 128 per visible-card bucket
  --equity-bins <int>             Default: 10
  --potential-bins <int>          Default: 3
  --preflop-runout-samples <int>  Default: 256
  --flop-runout-samples <int>     Default: 128
  --sample-turn-rivers            Sample instead of enumerating turn rivers
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
  --trajectory-recall             Retain all prior street buckets (large)
  --output <path>                 Default: blueprint-artifact.json"
    );
}
