//! Bounded research capture of the native values actually used by flop CFR.
//! No additional solve, synthetic range, or policy mutation is performed here.
use super::*;

const SCHEMA: &str = "hu-native-turn-cfv-dataset-v1";
const SEMANTICS: &str = "profile-positive-own-reach-cbr-zero-own-reach-v1";

#[derive(Debug, Serialize)]
pub(super) struct Target {
    iteration: u64,
    board: Vec<u8>,
    actor: usize,
    invested_bb: [f64; 2],
    public_state: PublicBeliefState,
    /// Independently normalized ranges used by the reference solve. A zero
    /// total stays zero, including for the other player's counterfactuals.
    ranges: [Vec<f64>; 2],
    raw_reach_totals: [f64; 2],
    opponent_compatible_mass: [Vec<f64>; 2],
    /// Conditional values (legacy trainer naming), NOT raw CFVs. Multiplying
    /// by compatible opponent mass and its raw total recovers the raw CFV.
    counterfactual_values_bb: [Vec<f64>; 2],
    raw_counterfactual_bb: [Vec<f64>; 2],
    raw_profile_counterfactual_bb: [Vec<f64>; 2],
    raw_best_response_counterfactual_bb: [Vec<f64>; 2],
    completed_zero_own_reach: [usize; 2],
    conditional_response_gain_bb: Option<[f64; 2]>,
    policy_sha256: String,
    policy_rows: usize,
    turn_iterations: u64,
    value_semantics: String,
    state_distribution: String,
}

impl Target {
    pub(super) fn new(
        round: u64,
        config: &TurnRiverSolveConfig,
        values: &Values,
    ) -> Result<Self, String> {
        let raw = &config.state.ranges;
        let totals: [f64; 2] = std::array::from_fn(|p| raw[p].iter().sum());
        let ranges: [Vec<f64>; 2] = std::array::from_fn(|p| {
            raw[p]
                .iter()
                .map(|v| if totals[p] > 0.0 { v / totals[p] } else { 0.0 })
                .collect()
        });
        let combos = all_combos();
        let masses: [Vec<f64>; 2] =
            std::array::from_fn(|p| compatible_masses_from_card_marginals(&combos, &ranges[1 - p]));
        let conditional: [Vec<f64>; 2] = std::array::from_fn(|p| {
            (0..COMBO_COUNT)
                .map(|c| {
                    let mass = masses[p][c] * totals[1 - p];
                    if mass > 0.0 {
                        values.counterfactual_bb[p][c] / mass
                    } else {
                        0.0
                    }
                })
                .collect()
        });
        if conditional.iter().flatten().any(|v| !v.is_finite()) {
            return Err("native target contains nonfinite conditional values".into());
        }
        Ok(Self {
            iteration: round,
            board: config.state.board.clone(),
            actor: config.state.actor,
            invested_bb: config.state.invested_bb,
            public_state: config.state.clone(),
            ranges,
            raw_reach_totals: totals,
            opponent_compatible_mass: masses,
            counterfactual_values_bb: conditional,
            raw_counterfactual_bb: values.counterfactual_bb.clone(),
            raw_profile_counterfactual_bb: values.profile_counterfactual_bb.clone(),
            raw_best_response_counterfactual_bb: values.best_response_counterfactual_bb.clone(),
            completed_zero_own_reach: values.completed_zero_own_reach,
            conditional_response_gain_bb: values.conditional_response_gain_bb,
            policy_sha256: values.policy_sha256.clone(),
            policy_rows: values.policy_rows,
            turn_iterations: config.iterations,
            value_semantics: SEMANTICS.into(),
            state_distribution: "native_flop_intermediate_iteration_exact_reaches".into(),
        })
    }

    pub(super) fn with_distribution(mut self, distribution: &str) -> Self {
        self.state_distribution = distribution.to_owned();
        self
    }
}

#[derive(Default)]
struct Capture {
    maximum_states: usize,
    observed_queries: usize,
    targets: Vec<Target>,
}

impl Capture {
    fn observe(
        &mut self,
        round: u64,
        config: &TurnRiverSolveConfig,
        values: &Values,
    ) -> Result<(), String> {
        self.observed_queries += 1;
        if self.targets.len() < self.maximum_states {
            self.targets.push(Target::new(round, config, values)?);
        }
        Ok(())
    }
}

#[test]
fn capture_preserves_exact_policy_and_is_worker_order_independent() {
    let mut game = super::super::tests::config().game;
    game.effective_stack_bb = 4.0;
    let board = [0, 5, 10];
    let state = PublicBeliefState::flop_start(
        board,
        1,
        [1.0, 1.0],
        std::array::from_fn(|_| uniform_range(&board)),
    );
    let control = train(game.clone(), state.clone(), 100101, 2, 4).unwrap();
    let mut expected = None;
    for workers in [1, 4] {
        let mut capture = Capture {
            maximum_states: 3,
            ..Default::default()
        };
        let observed = train_with_observer(
            game.clone(),
            state.clone(),
            100101,
            2,
            4,
            false,
            1,
            workers,
            Some(&mut |round, config, values| capture.observe(round, config, values)),
        )
        .unwrap();
        assert_eq!(
            serde_json::to_vec(&observed).unwrap(),
            serde_json::to_vec(&control).unwrap()
        );
        assert_eq!(capture.observed_queries as u64, control.turn_queries);
        assert_eq!(capture.targets.len(), 3);
        let encoded = serde_json::to_vec(&capture.targets).unwrap();
        if let Some(ref previous) = expected {
            assert_eq!(&encoded, previous);
        }
        expected = Some(encoded);
    }
    let error = train_with_observer(
        game,
        state,
        100101,
        2,
        4,
        false,
        1,
        1,
        Some(&mut |_, _, _| Err("capture failed".into())),
    )
    .err()
    .unwrap();
    assert_eq!(error, "capture failed");
}

#[test]
fn captured_targets_preserve_zero_reach_completion_and_raw_scaling() {
    let mut config = super::super::tests::config();
    config.state.ranges[0].fill(0.0);
    for v in &mut config.state.ranges[1] {
        *v *= 0.125;
    }
    let values = solve(config.clone()).unwrap();
    let target = Target::new(2, &config, &values).unwrap();
    assert!(target.conditional_response_gain_bb.is_none());
    assert!(target.completed_zero_own_reach[0] > 0);
    assert!(target.ranges[0].iter().all(|v| *v == 0.0));
    assert!(target.raw_counterfactual_bb[1].iter().all(|v| *v == 0.0));
    for p in 0..2 {
        for c in 0..COMBO_COUNT {
            let reconstructed = target.counterfactual_values_bb[p][c]
                * target.opponent_compatible_mass[p][c]
                * target.raw_reach_totals[1 - p];
            assert!((reconstructed - target.raw_counterfactual_bb[p][c]).abs() < 1e-12);
        }
    }
}

#[test]
#[ignore = "explicit pinned input/output and external resource guard; research target capture, not release"]
fn saved_native_value_preflight() {
    let input_path = PathBuf::from(std::env::var("POKER_NATIVE_VALUE_INPUT").unwrap());
    assert!(fs::metadata(&input_path).unwrap().len() <= 1024 * 1024);
    let input = fs::read(input_path).unwrap();
    let input_sha = std::env::var("POKER_NATIVE_VALUE_INPUT_SHA").unwrap();
    let (game, state) = root_input::decode(&input, &input_sha).unwrap();
    let output = PathBuf::from(std::env::var("POKER_NATIVE_VALUE_OUTPUT").unwrap());
    assert!(!output.exists(), "never overwrite a corpus");
    let maximum_states: usize = std::env::var("POKER_NATIVE_VALUE_MAX_STATES")
        .unwrap()
        .parse()
        .unwrap();
    assert!((1..=512).contains(&maximum_states));
    let iterations: u64 = std::env::var("POKER_NATIVE_VALUE_ITERATIONS")
        .unwrap()
        .parse()
        .unwrap();
    assert!((2..=32).contains(&iterations));
    let turn_iterations: u64 = std::env::var("POKER_NATIVE_VALUE_TURN_ITERATIONS")
        .unwrap()
        .parse()
        .unwrap();
    assert!([4, 64, 128].contains(&turn_iterations));
    let seed: u64 = std::env::var("POKER_NATIVE_VALUE_SEED")
        .unwrap()
        .parse()
        .unwrap();
    assert!([100101, 100102].contains(&seed));
    let workers: usize = std::env::var("POKER_NATIVE_VALUE_WORKERS")
        .unwrap()
        .parse()
        .unwrap();
    let mut capture = Capture {
        maximum_states,
        ..Default::default()
    };
    let started = std::time::Instant::now();
    let policy = train_with_observer(
        game.clone(),
        state,
        seed,
        iterations,
        turn_iterations,
        false,
        1,
        workers,
        Some(&mut |round, config, values| capture.observe(round, config, values)),
    )
    .unwrap();
    let capture_all = std::env::var("POKER_NATIVE_VALUE_CAPTURE_ALL").as_deref() == Ok("1");
    if capture_all {
        assert_eq!(
            capture.targets.len(),
            capture.observed_queries,
            "complete capture exceeded its bound"
        );
    } else {
        assert_eq!(
            capture.targets.len(),
            maximum_states,
            "incomplete capture is not exported"
        );
    }
    let captured_states = capture.targets.len();
    let policy_sha = format!("{:x}", Sha256::digest(serde_json::to_vec(&policy).unwrap()));
    let corpus = serde_json::json!({
        "schema": SCHEMA, "game": game, "source_public_input_sha256": input_sha,
        "source_policy_sha256": policy_sha, "seed": seed,
        "flop_iterations": iterations, "turn_iterations": turn_iterations,
        "maximum_states": captured_states, "observed_queries": capture.observed_queries,
        "capture_selection": if capture_all { "all_intermediate_queries_bounded_feasibility" } else { "first_n_in_public_leaf_chance_order_cost_preflight_only" },
        "validation": {"status": "research_only", "reasons": ["finite-budget reference pilot; no release or full-game qualification"]},
        "targets": capture.targets,
    });
    let encoded = serde_json::to_vec(&corpus).unwrap();
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)
        .unwrap();
    let mut compressed = GzEncoder::new(file, Compression::default());
    compressed.write_all(&encoded).unwrap();
    compressed.finish().unwrap().sync_all().unwrap();
    println!(
        "{}",
        serde_json::json!({
            "stage": "native_value_preflight", "seconds": started.elapsed().as_secs_f64(),
            "capturedStates": captured_states, "observedQueries": capture.observed_queries,
            "policySha256": policy_sha, "decodedDatasetSha256": format!("{:x}", Sha256::digest(&encoded)),
            "compressedBytes": fs::metadata(output).unwrap().len(), "releaseAccepted": false,
        })
    );
}
