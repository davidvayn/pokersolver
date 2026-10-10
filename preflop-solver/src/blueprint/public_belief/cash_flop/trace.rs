//! Bounded capture of the actual current-policy turn beliefs seen in training.
//! Observation only: no extra prediction, target generation, or CFR updates.
use super::*;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Mutex,
};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CashFlopTraceInput {
    pub solve: CashFlopPilotInput,
    pub sample_rounds: Vec<u64>,
    pub leaves_per_round: usize,
    /// Absent preserves the original smallest-hash observer and identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reach_sampling_seed: Option<u64>,
}

impl CashFlopTraceInput {
    pub fn validate(&self) -> Result<(), String> {
        self.solve.validate()?;
        if self.sample_rounds.is_empty()
            || self.sample_rounds.len() > 4
            || self
                .sample_rounds
                .iter()
                .any(|round| *round == 0 || *round > self.solve.iterations)
            || self.sample_rounds.windows(2).any(|pair| pair[0] >= pair[1])
            || !(1..=4).contains(&self.leaves_per_round)
        {
            return Err(
                "cash trace needs 1..4 sorted unique in-budget rounds and 1..4 leaves per round"
                    .into(),
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CashTurnQueryCapture {
    pub round: u64,
    pub query_sha256: String,
    pub flop_public_history: Vec<String>,
    pub reach_probability_from_flop_root: f64,
    pub input: TurnRiverSolveConfig,
}

#[derive(Clone, Debug, Serialize)]
pub struct CashFlopTraceReport {
    pub schema: String,
    pub status: String,
    pub active: bool,
    pub trace_input_sha256: String,
    pub trace_input: CashFlopTraceInput,
    pub solution: CashFlopPilotSolution,
    pub observed_turn_query_events_per_round: BTreeMap<u64, u64>,
    pub queries: Vec<CashTurnQueryCapture>,
    pub selection_method: String,
    pub limitations: Vec<String>,
}

#[derive(Default)]
struct CaptureRows {
    observed: BTreeMap<u64, u64>,
    rows: BTreeMap<u64, BTreeMap<String, CashTurnQueryCapture>>,
}

pub(in super::super) struct CashLeafCapture {
    sample_rounds: Vec<u64>,
    limit: usize,
    reach_sampling_seed: Option<u64>,
    round: AtomicU64,
    rows: Mutex<CaptureRows>,
}

impl CashLeafCapture {
    fn new(input: &CashFlopTraceInput) -> Self {
        Self {
            sample_rounds: input.sample_rounds.clone(),
            limit: input.leaves_per_round,
            reach_sampling_seed: input.reach_sampling_seed,
            round: AtomicU64::new(0),
            rows: Mutex::new(CaptureRows::default()),
        }
    }

    pub(in super::super) fn start_round(&self, round: u64) {
        self.round.store(round, Ordering::Relaxed);
    }

    pub(in super::super) fn finish_training(&self) {
        self.round.store(0, Ordering::Relaxed);
    }

    fn compare_queries(
        &self,
        left: &CashTurnQueryCapture,
        right: &CashTurnQueryCapture,
    ) -> std::cmp::Ordering {
        if let Some(seed) = self.reach_sampling_seed {
            // Only finite positive weights enter the retained reservoir.
            weighted_priority(
                seed,
                &left.query_sha256,
                left.reach_probability_from_flop_root,
            )
            .unwrap()
            .total_cmp(
                &weighted_priority(
                    seed,
                    &right.query_sha256,
                    right.reach_probability_from_flop_root,
                )
                .unwrap(),
            )
            .then_with(|| left.query_sha256.cmp(&right.query_sha256))
        } else {
            left.query_sha256.cmp(&right.query_sha256)
        }
    }

    fn retain_query(&self, rows: &mut CaptureRows, query: CashTurnQueryCapture) {
        *rows.observed.entry(query.round).or_default() += 1;
        if self.reach_sampling_seed.is_some_and(|seed| {
            weighted_priority(
                seed,
                &query.query_sha256,
                query.reach_probability_from_flop_root,
            )
            .is_none()
        }) {
            return;
        }
        let retained = rows.rows.entry(query.round).or_default();
        // Repeated logical cache events are observations, not extra tickets.
        if retained.contains_key(&query.query_sha256) {
            return;
        }
        if retained.len() == self.limit {
            let worst = retained
                .values()
                .max_by(|a, b| self.compare_queries(a, b))
                .unwrap();
            if !self.compare_queries(&query, worst).is_lt() {
                return;
            }
            let remove = worst.query_sha256.clone();
            retained.remove(&remove);
        }
        retained.insert(query.query_sha256.clone(), query);
    }

    pub(in super::super) fn record_leaf(
        &self,
        config: &FlopResolveConfig,
        state: &GameState,
        reaches: &[Vec<f64>; 2],
    ) {
        let round = self.round.load(Ordering::Relaxed);
        if self.sample_rounds.binary_search(&round).is_err() {
            return;
        }
        let root_joint = joint_compatibility_mass(&config.state.ranges);
        for turn in (0..52u8).filter(|card| !config.state.board.contains(card)) {
            let Some((input, _, joint_mass)) =
                turn_prediction_input(&config.game, &config.state.board, state, reaches, turn)
            else {
                continue;
            };
            let query_sha256 = format!(
                "{:x}",
                Sha256::digest(
                    serde_json::to_vec(&(
                        "cash-current-policy-turn-query-v1",
                        round,
                        &state.public_history,
                        &input
                    ))
                    .expect("finite cash query")
                )
            );
            let mut rows = self.rows.lock().expect("cash query capture");
            let reach_probability_from_flop_root = joint_mass / (45. * root_joint);
            self.retain_query(
                &mut rows,
                CashTurnQueryCapture {
                    round,
                    query_sha256,
                    flop_public_history: state.public_history.clone(),
                    reach_probability_from_flop_root,
                    input,
                },
            );
        }
    }
}

/// Exponential-race ordering implements sequential weighted sampling without
/// replacement over unique queries. Log space avoids dividing by tiny reach.
fn weighted_priority(seed: u64, query_sha256: &str, weight: f64) -> Option<f64> {
    if !weight.is_finite() || weight <= 0. {
        return None;
    }
    let mut hash = Sha256::new();
    hash.update(b"cash-query-reach-reservoir-v1");
    hash.update(seed.to_be_bytes());
    hash.update(query_sha256.as_bytes());
    let digest = hash.finalize();
    // 52 bits plus an exactly representable midpoint keep U strictly in
    // (0,1), including both extreme hashes. 53 bits + .5 can round to 1.
    let bits = u64::from_be_bytes(digest[..8].try_into().unwrap()) >> 12;
    let uniform = (bits as f64 + 0.5) / (1u64 << 52) as f64;
    let priority = (-uniform.ln()).ln() - weight.ln();
    priority.is_finite().then_some(priority)
}

pub fn solve(
    input: CashFlopTraceInput,
    network: PublicValueNetwork,
) -> Result<CashFlopTraceReport, String> {
    input.validate()?;
    let trace_input_sha256 = format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(&(
                "cash-current-policy-turn-trace-v1",
                &input,
                network.artifact_sha256()
            ))
            .map_err(|e| e.to_string())?
        )
    );
    let capture = Arc::new(CashLeafCapture::new(&input));
    let (mut solver, solution_input_sha) = prepare_solver(input.solve.clone(), network)?;
    solver.cash_leaf_capture = Some(capture.clone());
    solver.train_frozen_public_chance_pairs()?;
    // Disable capture before average-policy scoring: those are different beliefs.
    let solution = solver.finish_cash_flop(solution_input_sha)?;
    let rows = capture.rows.lock().map_err(|_| "cash capture poisoned")?;
    let queries = rows
        .rows
        .values()
        .flat_map(|group| group.values().cloned())
        .collect();
    let reach_weighted = input.reach_sampling_seed.is_some();
    Ok(CashFlopTraceReport {
        schema: "cash-current-policy-turn-trace-v1".into(), status: "research_only".into(), active: false,
        trace_input_sha256, trace_input: input, solution,
        observed_turn_query_events_per_round: rows.observed.clone(), queries,
        selection_method: if reach_weighted { "seeded-exponential-race-unique-queries-weighted-by-joint-root-reach" }
            else { "smallest-sha256-unique-queries-per-selected-training-round" }.into(),
        limitations: vec![if reach_weighted {
            "Seeded unique-query sampling without replacement; weights determine sequential selections, not independent inclusion probabilities. No importance-corrected loss or untouched release validation."
        } else { "Deterministic bounded diagnostic, not reach-proportional sampling or untouched release validation." }.into(),
            "Current-policy beliefs include leaf-cache hits; no predictions or regret updates are added.".into(),
            "Captured 2-update inputs specify prediction queries, not equilibrium teacher labels.".into(),
            "No opponent hole cards or future river; no complete routed policy or exploitability bound.".into()],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optional_reach_sampling_captures_without_changing_the_solve() {
        let (input, network) = super::super::tests::fixture();
        let options: CashFlopTraceInput = serde_json::from_value(serde_json::json!({
            "solve": input.clone(), "sample_rounds": [1, 2],
            "leaves_per_round": 2, "reach_sampling_seed": 917
        }))
        .unwrap();
        let original = super::super::solve(input, network.clone()).unwrap();
        let traced = solve(options, network).unwrap();
        assert_eq!(
            serde_json::to_vec(&original).unwrap(),
            serde_json::to_vec(&traced.solution).unwrap()
        );
        assert!(!traced.queries.is_empty() && traced.queries.len() <= 4);
        assert_eq!(
            traced.selection_method,
            "seeded-exponential-race-unique-queries-weighted-by-joint-root-reach"
        );
    }

    #[test]
    fn reach_priorities_are_finite_seeded_and_scale_invariant() {
        for weight in [0., -1., f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            assert!(weighted_priority(917, "query", weight).is_none());
        }
        for weight in [f64::from_bits(1), 1e-300, 1e-100, 0.01, 1., f64::MAX] {
            assert!(weighted_priority(917, "query", weight).unwrap().is_finite());
        }
        let priority = weighted_priority(917, "query", 0.01).unwrap();
        assert_eq!(priority, weighted_priority(917, "query", 0.01).unwrap());
        assert_ne!(priority, weighted_priority(918, "query", 0.01).unwrap());
        assert_ne!(priority, weighted_priority(917, "other", 0.01).unwrap());
        let mut ordinary: Vec<_> = (0..32)
            .map(|id| {
                let hash = format!("{id:064x}");
                (
                    weighted_priority(917, &hash, (id + 1) as f64 / 100.).unwrap(),
                    id,
                )
            })
            .collect();
        let mut scaled: Vec<_> = (0..32)
            .map(|id| {
                let hash = format!("{id:064x}");
                (
                    weighted_priority(917, &hash, (id + 1) as f64 / 100. * 1e-280).unwrap(),
                    id,
                )
            })
            .collect();
        ordinary.sort_by(|a, b| a.0.total_cmp(&b.0));
        scaled.sort_by(|a, b| a.0.total_cmp(&b.0));
        assert_eq!(
            ordinary.iter().map(|row| row.1).collect::<Vec<_>>(),
            scaled.iter().map(|row| row.1).collect::<Vec<_>>()
        );
        let heavier_wins = (0..4096)
            .filter(|seed| {
                weighted_priority(*seed, "heavy", 9.).unwrap()
                    < weighted_priority(*seed, "light", 1.).unwrap()
            })
            .count();
        // Fixed seed enumeration, not an unseeded/flaky statistical test.
        assert!((3500..3900).contains(&heavier_wins), "{heavier_wins}");
    }

    #[test]
    fn reach_reservoir_matches_offline_priority_order_and_ignores_repeat_tickets() {
        let (input, _) = super::super::tests::fixture();
        let query_input = turn_prediction_input(
            &input.game,
            &input.state.board,
            &input.state.game_state(),
            &input.state.ranges,
            0,
        )
        .unwrap()
        .0;
        let observer = CashLeafCapture::new(&CashFlopTraceInput {
            solve: input,
            sample_rounds: vec![1],
            leaves_per_round: 3,
            reach_sampling_seed: Some(917),
        });
        let candidates: Vec<_> = (0..24)
            .map(|id| CashTurnQueryCapture {
                round: 1,
                query_sha256: format!("{id:064x}"),
                flop_public_history: vec![format!("branch-{id}")],
                reach_probability_from_flop_root: (id + 1) as f64 / 1000.,
                input: query_input.clone(),
            })
            .collect();
        let mut forward = CaptureRows::default();
        let mut reverse = CaptureRows::default();
        for row in &candidates {
            observer.retain_query(&mut forward, row.clone());
        }
        for row in candidates.iter().rev() {
            observer.retain_query(&mut reverse, row.clone());
            observer.retain_query(&mut reverse, row.clone());
        }
        assert_eq!(
            serde_json::to_vec(&forward.rows).unwrap(),
            serde_json::to_vec(&reverse.rows).unwrap()
        );
        assert_eq!(forward.observed[&1], 24);
        assert_eq!(reverse.observed[&1], 48);
        let mut expected = candidates.clone();
        expected.sort_by(|a, b| observer.compare_queries(a, b));
        let mut expected_hashes: Vec<_> = expected[..3]
            .iter()
            .map(|row| row.query_sha256.clone())
            .collect();
        expected_hashes.sort();
        assert_eq!(
            forward.rows[&1].keys().cloned().collect::<Vec<_>>(),
            expected_hashes
        );
        let before = serde_json::to_vec(&forward.rows).unwrap();
        for weight in [0., -1., f64::NAN, f64::INFINITY] {
            let mut invalid = candidates[0].clone();
            invalid.reach_probability_from_flop_root = weight;
            observer.retain_query(&mut forward, invalid);
        }
        assert_eq!(before, serde_json::to_vec(&forward.rows).unwrap());
    }

    #[test]
    fn legacy_trace_serialization_and_malformed_seed_rejection() {
        let (input, _) = super::super::tests::fixture();
        let legacy =
            serde_json::json!({"solve": input, "sample_rounds": [1], "leaves_per_round": 2});
        let parsed: CashFlopTraceInput = serde_json::from_value(legacy.clone()).unwrap();
        assert_eq!(parsed.reach_sampling_seed, None);
        assert_eq!(serde_json::to_value(parsed).unwrap(), legacy);
        for invalid in [
            serde_json::json!(-1),
            serde_json::json!(0.5),
            serde_json::json!("917"),
            serde_json::json!(true),
            serde_json::json!([]),
        ] {
            let mut payload = legacy.clone();
            payload["reach_sampling_seed"] = invalid;
            assert!(serde_json::from_value::<CashFlopTraceInput>(payload).is_err());
        }
        let mut payload = legacy;
        payload["reach_sampling_seed"] = serde_json::json!(u64::MAX);
        let parsed: CashFlopTraceInput = serde_json::from_value(payload).unwrap();
        assert_eq!(parsed.reach_sampling_seed, Some(u64::MAX));
    }

    #[test]
    fn tracing_actual_current_beliefs_preserves_entire_solve() {
        let (input, network) = super::super::tests::fixture();
        let original = super::super::solve(input.clone(), network.clone()).unwrap();
        let options = CashFlopTraceInput {
            solve: input,
            sample_rounds: vec![1, 2],
            leaves_per_round: 2,
            reach_sampling_seed: None,
        };
        let traced = solve(options, network).unwrap();
        assert_eq!(
            serde_json::to_vec(&original).unwrap(),
            serde_json::to_vec(&traced.solution).unwrap()
        );
        // This near-all-in fixture can stop reaching live turns after the
        // first update. A capture is a bound, not fabricated missing states.
        assert!(!traced.queries.is_empty() && traced.queries.len() <= 4);
        assert!(traced.observed_turn_query_events_per_round.contains_key(&1));
        assert!(traced
            .queries
            .iter()
            .all(|query| [1, 2].contains(&query.round)));
        for query in &traced.queries {
            query.input.game.validate().unwrap();
            query
                .input
                .state
                .validate_street_and_normalize_impl(&query.input.game, Street::Turn, 4, true)
                .unwrap();
            assert_eq!(query.input.iterations, 2);
            assert_eq!(query.input.state.board.len(), 4);
            assert!(query.flop_public_history.len() > 1);
            assert!(
                query.reach_probability_from_flop_root.is_finite()
                    && query.reach_probability_from_flop_root > 0.
            );
        }
        assert!(!traced.active);
    }

    #[test]
    fn logical_turn_cache_hits_are_observed_in_each_selected_round() {
        let (input, network) = super::super::tests::fixture();
        let options = CashFlopTraceInput {
            solve: input.clone(),
            sample_rounds: vec![1, 2],
            leaves_per_round: 2,
            reach_sampling_seed: None,
        };
        let (mut solver, _) = prepare_solver(input, network).unwrap();
        let capture = Arc::new(CashLeafCapture::new(&options));
        solver.cash_leaf_capture = Some(capture.clone());
        let mut state = solver.config.state.game_state();
        for _ in 0..2 {
            let check = state
                .legal_actions(&solver.config.game)
                .into_iter()
                .find(|action| action.label == "check")
                .unwrap();
            state = state.apply(&check, &solver.config.game);
        }
        assert_eq!(state.street, Street::Turn);
        let reaches = solver.config.state.ranges.clone();
        capture.start_round(1);
        let first = solver.turn_leaf_values(&state, &reaches, None);
        assert_eq!(first, solver.turn_leaf_values(&state, &reaches, None));
        capture.start_round(2);
        assert_eq!(first, solver.turn_leaf_values(&state, &reaches, None));
        let rows = capture.rows.lock().unwrap();
        assert_eq!(rows.observed[&1], 98);
        assert_eq!(rows.observed[&2], 49);
        assert_eq!(rows.rows[&1].len(), 2);
        assert_eq!(rows.rows[&2].len(), 2);
        drop(rows);
        capture.finish_training();
        assert_eq!(first, solver.turn_leaf_values(&state, &reaches, None));
        assert_eq!(capture.rows.lock().unwrap().observed[&2], 49);
    }

    #[test]
    fn capture_is_bounded_order_independent_and_stops_before_average_scoring() {
        let (input, _) = super::super::tests::fixture();
        let mut network = super::super::tests::fixture().1;
        network.artifact_sha256 = Some("b".repeat(64));
        let (mut solver, _) = prepare_solver(input.clone(), network).unwrap();
        let options = CashFlopTraceInput {
            solve: input,
            sample_rounds: vec![1],
            leaves_per_round: 2,
            reach_sampling_seed: None,
        };
        let forward = CashLeafCapture::new(&options);
        let reverse = CashLeafCapture::new(&options);
        forward.start_round(1);
        reverse.start_round(1);
        let state = solver.config.state.game_state();
        let ranges = solver.config.state.ranges.clone();
        forward.record_leaf(&solver.config, &state, &ranges);
        let mut changed = state.clone();
        changed
            .public_history
            .push("diagnostic-second-branch".into());
        forward.record_leaf(&solver.config, &changed, &ranges);
        reverse.record_leaf(&solver.config, &changed, &ranges);
        reverse.record_leaf(&solver.config, &state, &ranges);
        let before = serde_json::to_vec(&forward.rows.lock().unwrap().rows).unwrap();
        assert_eq!(
            before,
            serde_json::to_vec(&reverse.rows.lock().unwrap().rows).unwrap()
        );
        assert_eq!(forward.rows.lock().unwrap().rows[&1].len(), 2);
        forward.finish_training();
        forward.record_leaf(&solver.config, &state, &ranges);
        assert_eq!(
            before,
            serde_json::to_vec(&forward.rows.lock().unwrap().rows).unwrap()
        );
        solver.cash_leaf_capture = None;
    }

    #[test]
    fn invalid_trace_options_fail_before_solving() {
        let (input, network) = super::super::tests::fixture();
        for rounds in [vec![], vec![0], vec![3], vec![1, 1], vec![2, 1]] {
            assert!(solve(
                CashFlopTraceInput {
                    solve: input.clone(),
                    sample_rounds: rounds,
                    leaves_per_round: 2,
                    reach_sampling_seed: None,
                },
                network.clone()
            )
            .is_err());
        }
        for limit in [0, 5, usize::MAX] {
            assert!(solve(
                CashFlopTraceInput {
                    solve: input.clone(),
                    sample_rounds: vec![1],
                    leaves_per_round: limit,
                    reach_sampling_seed: None,
                },
                network.clone()
            )
            .is_err());
        }
    }
}
