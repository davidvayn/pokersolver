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
    round: AtomicU64,
    rows: Mutex<CaptureRows>,
}

impl CashLeafCapture {
    fn new(input: &CashFlopTraceInput) -> Self {
        Self {
            sample_rounds: input.sample_rounds.clone(),
            limit: input.leaves_per_round,
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
            *rows.observed.entry(round).or_default() += 1;
            let retained = rows.rows.entry(round).or_default();
            if retained.len() == self.limit
                && retained
                    .last_key_value()
                    .is_some_and(|(key, _)| key < &query_sha256)
            {
                continue;
            }
            let reach_probability_from_flop_root = joint_mass / (45. * root_joint);
            retained
                .entry(query_sha256.clone())
                .or_insert(CashTurnQueryCapture {
                    round,
                    query_sha256,
                    flop_public_history: state.public_history.clone(),
                    reach_probability_from_flop_root,
                    input,
                });
            if retained.len() > self.limit {
                retained.pop_last();
            }
        }
    }
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
    Ok(CashFlopTraceReport {
        schema: "cash-current-policy-turn-trace-v1".into(), status: "research_only".into(), active: false,
        trace_input_sha256, trace_input: input, solution,
        observed_turn_query_events_per_round: rows.observed.clone(), queries,
        selection_method: "smallest-sha256-unique-queries-per-selected-training-round".into(),
        limitations: vec!["Deterministic bounded diagnostic, not reach-proportional sampling or untouched release validation.".into(),
            "Current-policy beliefs include leaf-cache hits; no predictions or regret updates are added.".into(),
            "Captured 2-update inputs specify prediction queries, not equilibrium teacher labels.".into(),
            "No opponent hole cards or future river; no complete routed policy or exploitability bound.".into()],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracing_actual_current_beliefs_preserves_entire_solve() {
        let (input, network) = super::super::tests::fixture();
        let original = super::super::solve(input.clone(), network.clone()).unwrap();
        let options = CashFlopTraceInput {
            solve: input,
            sample_rounds: vec![1, 2],
            leaves_per_round: 2,
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
                    leaves_per_round: 2
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
                    leaves_per_round: limit
                },
                network.clone()
            )
            .is_err());
        }
    }
}
