//! Explicit bounded research pilot; saves average root policies, never regrets.
use super::*;
use std::io::Write;

fn save(name: &str, value: &impl Serialize) {
    let directory = PathBuf::from(std::env::var("POKER_EXACT_FLOP_PILOT_DIR").unwrap());
    let encoded = serde_json::to_vec(value).unwrap();
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join(name))
        .unwrap();
    file.write_all(&encoded).unwrap();
    file.sync_all().unwrap();
}

fn train(config: SampledFlopConfig, exact: bool) -> SampledFlopSolution {
    if exact {
        solve_with_exact_terminals(config)
    } else {
        solve(config)
    }
    .unwrap()
}

fn pilot_game() -> BlueprintConfig {
    BlueprintConfig {
        effective_stack_bb: 20.0,
        ..BlueprintConfig::default()
    }
}

#[test]
fn terminal_training_pilot_pins_twenty_bb_before_solving() {
    assert_eq!(pilot_game().effective_stack_bb, 20.0);
}

fn terminal_loss(
    solution: &SampledFlopSolution,
    state: &GameState,
    board: [u8; 3],
    game: &BlueprintConfig,
    ranges: &[Vec<f64>; 2],
) -> f64 {
    let actions = state.legal_actions(game);
    assert_eq!(actions.len(), 2);
    assert!(actions
        .iter()
        .all(|a| state.apply(a, game).terminal.is_some()));
    let call = actions
        .iter()
        .position(|a| a.kind == ActionKind::Call)
        .unwrap();
    let fold = actions
        .iter()
        .position(|a| a.kind == ActionKind::Fold)
        .unwrap();
    let called = state.apply(&actions[call], game);
    let legal = std::array::from_fn(|_| {
        all_combos()
            .iter()
            .map(|c| !c.cards().iter().any(|card| board.contains(card)))
            .collect::<Vec<_>>()
    });
    let equities = exact_flop_all_in_equities(board, &legal, 1);
    let combos = all_combos();
    let mut weighted_loss = 0.0;
    let mut joint_mass = 0.0;
    for (hero, own) in combos.iter().enumerate() {
        if ranges[state.actor][hero] == 0.0 {
            continue;
        }
        let mut mass = 0.0;
        let mut value = 0.0;
        for (other, opponent) in combos.iter().enumerate() {
            let weight = ranges[1 - state.actor][other];
            if weight == 0.0 || own.overlaps(*opponent) {
                continue;
            }
            let equity = equities[hero * COMBO_COUNT + other];
            assert!(equity.is_finite());
            let units = (equity as f64 * 1980.0).round();
            value += weight * (called.pot() * units / 1980.0 - called.invested[state.actor]);
            mass += weight;
        }
        if mass == 0.0 {
            continue;
        }
        let ev = [-state.invested[state.actor], value / mass];
        let row = &solution.root.probabilities[hero * 2..hero * 2 + 2];
        let total: f64 = row.iter().map(|p| *p as f64).sum();
        assert!(total > 0.0 && (total - 1.0).abs() < 1e-6);
        let current = (row[fold] as f64 * ev[0] + row[call] as f64 * ev[1]) / total;
        let weight = ranges[state.actor][hero] * mass;
        weighted_loss += weight * (ev[0].max(ev[1]) - current);
        joint_mass += weight;
    }
    assert!(joint_mass > 0.0 && weighted_loss >= -1e-12);
    weighted_loss / joint_mass
}

#[test]
#[ignore = "explicit new output directory and external 2GiB/time/disk guard required; not full-game qualification"]
fn fixed_flop_exact_terminal_training_pair() {
    let output = PathBuf::from(std::env::var("POKER_EXACT_FLOP_PILOT_DIR").unwrap());
    assert!(output.is_dir());
    let game = pilot_game();
    assert_eq!(game.effective_stack_bb, 20.0);
    for (board_index, board) in [[48, 21, 2], [27, 2, 9], [42, 34, 25]]
        .into_iter()
        .enumerate()
    {
        let ranges = std::array::from_fn(|_| uniform_range(&board));
        let mut terminal = GameState::initial(&game);
        while terminal.street != Street::Flop {
            let action = terminal
                .legal_actions(&game)
                .into_iter()
                .find(|a| matches!(a.kind, ActionKind::Call | ActionKind::Check))
                .unwrap();
            terminal = terminal.apply(&action, &game);
        }
        let shove = terminal
            .legal_actions(&game)
            .into_iter()
            .find(|a| a.label.contains("all_in"))
            .unwrap();
        terminal = terminal.apply(&shove, &game);
        for seed in [97001, 97002] {
            for exact in [false, true] {
                let config = SampledFlopConfig {
                    game: game.clone(),
                    state: PublicBeliefState::from_game_state(
                        board.to_vec(),
                        &terminal,
                        ranges.clone(),
                    ),
                    iterations: 32,
                    seed,
                    maximum_information_sets: 2_000_000,
                };
                let started = std::time::Instant::now();
                let solution = train(config, exact);
                let seconds = started.elapsed().as_secs_f64();
                let loss = terminal_loss(&solution, &terminal, board, &game, &ranges);
                save(
                    &format!("terminal-{board_index}-{seed}-{exact}.json"),
                    &solution,
                );
                println!(
                    "{}",
                    serde_json::json!({"stage":"exact_terminal_training_loss",
                    "board":board, "seed":seed, "exactTerminalChance":exact, "iterations":32,
                    "exactTerminalSubgameLossBb":loss, "seconds":seconds,
                    "inputSha256":solution.input_sha256, "informationSets":solution.information_sets,
                    "interpretation":"uniform public ranges, one terminal decision; not full-game exploitability or authentic reach"})
                );
            }
        }
    }
    // Full nonterminal flop trees, unchanged sizing and fixed chance seeds.
    // These are cost/policy-change screens, not root payoff qualifications.
    for pot in [4.0, 10.0, 20.0] {
        let board = [48, 21, 2];
        for seed in [97101, 97102] {
            for exact in [false, true] {
                let config = SampledFlopConfig {
                    game: game.clone(),
                    state: PublicBeliefState::flop_start(
                        board,
                        1,
                        [pot / 2.0; 2],
                        std::array::from_fn(|_| uniform_range(&board)),
                    ),
                    iterations: 32,
                    seed,
                    maximum_information_sets: 2_000_000,
                };
                let started = std::time::Instant::now();
                let solution = train(config, exact);
                let seconds = started.elapsed().as_secs_f64();
                save(&format!("nonterminal-{pot}-{seed}-{exact}.json"), &solution);
                println!(
                    "{}",
                    serde_json::json!({"stage":"exact_terminal_training_cost",
                    "potBb":pot, "board":board, "seed":seed, "exactTerminalChance":exact,
                    "iterations":32, "seconds":seconds, "informationSets":solution.information_sets,
                    "trainedRootCombos":solution.trained_root_combos,
                    "inputSha256":solution.input_sha256,
                    "interpretation":"cost and root policy only; process-local exact equity cache already warm; no payoff or exploitability claim"})
                );
            }
        }
    }
}
