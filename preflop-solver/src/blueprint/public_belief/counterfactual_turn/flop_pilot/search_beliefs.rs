//! Bounded search-distribution sampling. The learned model proposes beliefs;
//! a separate native solve supplies every label. This is not ReBeL or safe CFR-D.
use super::*;

#[derive(Clone)]
struct Query {
    order: usize,
    round: u64,
    config: TurnRiverSolveConfig,
    distribution: &'static str,
}

struct Sampler {
    iterations: u64,
    capacity: usize,
    final_count: usize,
    quotas: [usize; 3],
    seen: [usize; 3],
    bins: [Vec<Query>; 3],
    rng: [SplitMix64; 3],
    observed: usize,
}

impl Sampler {
    fn new(
        capacity: usize,
        final_count: usize,
        seed: u64,
        iterations: u64,
    ) -> Result<Self, String> {
        if ![32, 128].contains(&iterations)
            || capacity > 64
            || final_count == 0
            || capacity < final_count + 3
        {
            return Err("invalid bounded search-belief quota".into());
        }
        let remaining = capacity - final_count;
        Ok(Self {
            iterations,
            capacity,
            final_count,
            quotas: [(remaining + 2) / 3, (remaining + 1) / 3, remaining / 3],
            seen: [0; 3],
            bins: std::array::from_fn(|_| Vec::new()),
            rng: std::array::from_fn(|b| SplitMix64::new(seed ^ (0x51A7E + b as u64))),
            observed: 0,
        })
    }

    fn observe(&mut self, round: u64, config: &TurnRiverSolveConfig) -> Result<(), String> {
        if !(1..=self.iterations).contains(&round) {
            return Err("unplanned learned-search iteration".into());
        }
        let band = ((round - 1) * 3 / self.iterations) as usize;
        self.observed += 1;
        self.seen[band] += 1;
        let position = if self.bins[band].len() < self.quotas[band] {
            self.bins[band].len()
        } else {
            self.rng[band].index(self.seen[band])
        };
        if position < self.quotas[band] {
            let query = Query {
                order: self.observed,
                round,
                config: config.clone(),
                distribution: [
                    "learned_flop_search_early_belief_native_label",
                    "learned_flop_search_middle_belief_native_label",
                    "learned_flop_search_late_belief_native_label",
                ][band],
            };
            if position == self.bins[band].len() {
                self.bins[band].push(query);
            } else {
                self.bins[band][position] = query;
            }
        }
        Ok(())
    }

    fn finish(self, final_queries: Vec<TurnRiverSolveConfig>) -> Result<Vec<Query>, String> {
        if final_queries.len() != self.final_count
            || (0..3).any(|b| self.bins[b].len() != self.quotas[b])
        {
            return Err("incomplete search-belief sample; no padding".into());
        }
        let mut queries = self.bins.into_iter().flatten().collect::<Vec<_>>();
        queries.sort_by_key(|q| q.order);
        queries.extend(
            final_queries
                .into_iter()
                .enumerate()
                .map(|(index, config)| Query {
                    order: self.observed + index + 1,
                    round: self.iterations,
                    config,
                    distribution: "learned_flop_final_average_belief_native_label",
                }),
        );
        if queries.len() != self.capacity {
            return Err("search-belief count drift".into());
        }
        Ok(queries)
    }
}

fn label(queries: &[Query], workers: usize) -> Result<Vec<value_targets::Target>, String> {
    if !(1..=4).contains(&workers) {
        return Err("native label workers must be 1..4".into());
    }
    let mut output = std::thread::scope(|scope| {
        let handles = (0..workers)
            .map(|worker| {
                scope.spawn(move || {
                    queries
                        .iter()
                        .enumerate()
                        .skip(worker)
                        .step_by(workers)
                        .map(|(index, query)| {
                            let values = solve(query.config.clone())?;
                            let target =
                                value_targets::Target::new(query.round, &query.config, &values)?
                                    .with_distribution(query.distribution);
                            Ok((index, target))
                        })
                        .collect::<Result<Vec<_>, String>>()
                })
            })
            .collect::<Vec<_>>();
        let mut output = Vec::new();
        for handle in handles {
            output.extend(
                handle
                    .join()
                    .map_err(|_| "native belief-label worker panicked")??,
            );
        }
        Ok::<_, String>(output)
    })?;
    output.sort_by_key(|(index, _)| *index);
    if output.len() != queries.len() {
        return Err("incomplete native belief labels".into());
    }
    Ok(output.into_iter().map(|(_, target)| target).collect())
}

#[test]
fn sampler_is_bounded_deterministic_and_keeps_three_search_bands_and_final_beliefs() {
    let config = super::super::tests::config();
    let mut expected = None;
    for _ in 0..2 {
        let mut sample = Sampler::new(16, 1, 917, 32).unwrap();
        for round in 1..=32 {
            for _ in 0..9 {
                sample.observe(round, &config).unwrap();
            }
        }
        assert_eq!(sample.observed, 288);
        let queries = sample.finish(vec![config.clone()]).unwrap();
        let counts = queries.iter().fold(BTreeMap::new(), |mut m, q| {
            *m.entry(q.distribution).or_insert(0) += 1;
            m
        });
        assert_eq!(counts.len(), 4);
        assert_eq!(counts["learned_flop_search_early_belief_native_label"], 5);
        assert_eq!(counts["learned_flop_search_middle_belief_native_label"], 5);
        assert_eq!(counts["learned_flop_search_late_belief_native_label"], 5);
        let identity = queries
            .iter()
            .map(|q| (q.order, q.round))
            .collect::<Vec<_>>();
        if let Some(previous) = &expected {
            assert_eq!(&identity, previous);
        }
        expected = Some(identity);
    }
    assert!(Sampler::new(16, 1, 917, 32)
        .unwrap()
        .finish(vec![config])
        .is_err());
    assert!(Sampler::new(3, 1, 917, 32).is_err());
}

#[test]
fn search_refresh_samples_all_128_updates_without_changing_the_label_budget() {
    let config = super::super::tests::config();
    let mut sample = Sampler::new(16, 1, 917, 128).unwrap();
    assert!(sample.observe(0, &config).is_err());
    assert!(sample.observe(129, &config).is_err());
    for round in 1..=128 {
        sample.observe(round, &config).unwrap();
    }
    let queries = sample.finish(vec![config]).unwrap();
    assert_eq!(queries.len(), 16);
    for (band, interval) in [(0, 1..=43), (1, 44..=86), (2, 87..=128)] {
        assert_eq!(
            queries[band * 5..(band + 1) * 5]
                .iter()
                .filter(|q| interval.contains(&q.round))
                .count(),
            5
        );
    }
    assert_eq!(queries[15].round, 128);
    assert_eq!(
        queries[15].distribution,
        "learned_flop_final_average_belief_native_label"
    );
    assert!(Sampler::new(16, 1, 917, 64).is_err());
}

#[test]
fn learned_belief_observation_does_not_change_updates_and_cannot_capture_native_values() {
    let mut model = crate::blueprint::public_belief::tests::zero_shared_value_network();
    model.schema = "hu-public-belief-combo-value-network-v4".into();
    model.value_normalization = Some("payoff-exposure".into());
    model.prediction_contract = Some("native-turn-cfv-full-stack-v1".into());
    model.artifact_sha256 = Some("d".repeat(64));
    model.validate().unwrap();
    let mut game = BlueprintConfig::default();
    game.effective_stack_bb = 20.0;
    let board = [0, 5, 10];
    let mut ranges = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
    ranges[0][Combo::new(51, 50).key()] = 1.0;
    ranges[1][Combo::new(49, 48).key()] = 1.0;
    let state = PublicBeliefState::flop_start(board, 1, [1.0, 1.0], ranges);
    let evaluator = continuation::Evaluator::Learned(&model);
    let control = train_with_evaluator(
        game.clone(),
        state.clone(),
        100101,
        2,
        4,
        false,
        1,
        1,
        None,
        evaluator,
    )
    .unwrap();
    let mut observed = 0;
    let mut callback = |_: u64, config: &TurnRiverSolveConfig| {
        assert_eq!(config.state.board.len(), 4);
        observed += 1;
        Ok(())
    };
    let captured = train_with_evaluator(
        game.clone(),
        state.clone(),
        100101,
        2,
        4,
        false,
        1,
        4,
        Some(LeafObservation::Beliefs(&mut callback)),
        evaluator,
    )
    .unwrap();
    assert_eq!(observed, control.turn_queries);
    assert_eq!(
        serde_json::to_vec(&control).unwrap(),
        serde_json::to_vec(&captured).unwrap()
    );
    assert!(train_with_evaluator(
        game,
        state,
        100101,
        2,
        4,
        false,
        1,
        1,
        Some(LeafObservation::Native(&mut |_, _, _| Ok(()))),
        evaluator
    )
    .is_err());
    let frozen = frozen_response::Frozen::new(&captured).unwrap();
    assert!(frozen.belief_queries(0).is_err());
    for config in frozen.belief_queries(11).unwrap() {
        assert_eq!(config.state.board, [0, 5, 10, 11]);
        for combo in all_combos() {
            if combo.cards().contains(&11) {
                assert_eq!(config.state.ranges[0][combo.key()], 0.0);
                assert_eq!(config.state.ranges[1][combo.key()], 0.0);
            }
        }
    }
}

#[test]
#[ignore = "pinned proposer/root, exclusive output and external resource guard; research-only native labels"]
fn saved_search_distribution_native_capture() {
    let root = fs::read(std::env::var("POKER_SEARCH_ROOT").unwrap()).unwrap();
    assert!(root.len() <= 1024 * 1024);
    let root_sha = std::env::var("POKER_SEARCH_ROOT_SHA").unwrap();
    let (game, state) = root_input::decode(&root, &root_sha).unwrap();
    let model =
        PublicValueNetwork::read(Path::new(&std::env::var("POKER_SEARCH_MODEL").unwrap())).unwrap();
    assert_eq!(
        model.artifact_sha256().unwrap(),
        std::env::var("POKER_SEARCH_MODEL_SHA").unwrap()
    );
    let seed: u64 = std::env::var("POKER_SEARCH_TRUNK_SEED")
        .unwrap()
        .parse()
        .unwrap();
    assert!([100101, 100102].contains(&seed));
    let sample_seed: u64 = std::env::var("POKER_SEARCH_SAMPLE_SEED")
        .unwrap()
        .parse()
        .unwrap();
    let count: usize = std::env::var("POKER_SEARCH_LABEL_COUNT")
        .unwrap()
        .parse()
        .unwrap();
    assert!((16..=64).contains(&count));
    let iterations: u64 = std::env::var("POKER_SEARCH_FLOP_ITERATIONS")
        .unwrap_or_else(|_| "32".into())
        .parse()
        .unwrap();
    assert!([32, 128].contains(&iterations));
    let workers: usize = std::env::var("POKER_SEARCH_WORKERS")
        .unwrap()
        .parse()
        .unwrap();
    let output = PathBuf::from(std::env::var("POKER_SEARCH_OUTPUT").unwrap());
    let policy_path = PathBuf::from(std::env::var("POKER_SEARCH_POLICY_OUTPUT").unwrap());
    assert!(
        !output.exists() && !policy_path.exists(),
        "never overwrite research artifacts"
    );
    let prepared = Trunk::new(game.clone(), state.clone()).unwrap();
    // Count complete live histories without evaluating or updating any policy.
    fn count_leaves(state: GameState, game: &BlueprintConfig) -> usize {
        if state.terminal.is_some() {
            return 0;
        }
        if state.street == Street::Turn {
            return 1;
        }
        state
            .legal_actions(game)
            .iter()
            .map(|a| count_leaves(state.apply(a, game), game))
            .sum()
    }
    let final_count = count_leaves(prepared.state.game_state(), &game);
    let mut sampler = Sampler::new(count, final_count, sample_seed, iterations).unwrap();
    let started = std::time::Instant::now();
    let mut callback = |round, config: &TurnRiverSolveConfig| sampler.observe(round, config);
    let policy = train_with_evaluator(
        game.clone(),
        state.clone(),
        seed,
        iterations,
        64,
        false,
        1,
        workers,
        Some(LeafObservation::Beliefs(&mut callback)),
        continuation::Evaluator::Learned(&model),
    )
    .unwrap();
    let policy_bytes = serde_json::to_vec(&policy).unwrap();
    let verify = std::env::var("POKER_SEARCH_VERIFY_POLICY_PARITY").as_deref() == Ok("1");
    if verify {
        let control = train_with_evaluator(
            game.clone(),
            state,
            seed,
            iterations,
            64,
            false,
            1,
            workers,
            None,
            continuation::Evaluator::Learned(&model),
        )
        .unwrap();
        assert_eq!(policy_bytes, serde_json::to_vec(&control).unwrap());
    }
    let proposal_seconds = started.elapsed().as_secs_f64();
    let frozen = frozen_response::Frozen::new(&policy).unwrap();
    let turns = (0..52u8)
        .filter(|c| !policy.state.board.contains(c))
        .collect::<Vec<_>>();
    let turn = turns[SplitMix64::new(sample_seed ^ 0xF1A1).index(turns.len())];
    let observed = sampler.observed + final_count;
    let queries = sampler
        .finish(frozen.belief_queries(turn).unwrap())
        .unwrap();
    let targets = label(&queries, workers).unwrap();
    let policy_sha = format!("{:x}", Sha256::digest(&policy_bytes));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(policy_path)
        .unwrap();
    file.write_all(&policy_bytes).unwrap();
    file.sync_all().unwrap();
    let corpus = serde_json::json!({"schema":"hu-native-turn-cfv-dataset-v1", "game":game,
        "source_public_input_sha256":root_sha, "source_policy_sha256":policy_sha,
        "proposal_model_sha256":model.artifact_sha256(), "proposal_policy_kind":"frozen_learned_leaf_search_only",
        "seed":seed,"sampling_seed":sample_seed,"flop_iterations":iterations,"turn_iterations":64,
        "maximum_states":targets.len(),"observed_queries":observed,"native_label_queries":targets.len(),
        "capture_selection":"stratified_learned_search_and_final_average_beliefs_with_native_labels",
        "policy_observation_parity_checked":verify,
        "validation":{"status":"research_only","reasons":["finite-budget native labels on learned-search beliefs; not release qualification"]},
        "targets":targets});
    let encoded = serde_json::to_vec(&corpus).unwrap();
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)
        .unwrap();
    let mut gzip = GzEncoder::new(file, Compression::default());
    gzip.write_all(&encoded).unwrap();
    gzip.finish().unwrap().sync_all().unwrap();
    println!(
        "{}",
        serde_json::json!({"capturedStates":count,"observedQueries":observed,
        "proposalSeconds":proposal_seconds,"seconds":started.elapsed().as_secs_f64(),
        "policySha256":policy_sha,"compressedBytes":fs::metadata(output).unwrap().len(),"releaseAccepted":false})
    );
}
