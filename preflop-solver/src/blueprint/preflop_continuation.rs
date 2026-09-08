//! Research boundary between compact preflop regret training and public-belief
//! continuations. These are true policy reaches, never PCS estimator weights.
use super::*;
use public_belief::counterfactual_turn::{
    exact_flop_kernel, NativeFlopOptions, NativePostflopPolicy,
};
use public_belief::PublicBeliefState;
use public_belief::PublicValueNetwork;
use std::time::Instant;
mod checkpoint;
mod exact_checkdown;
mod fixed_control;
mod fixed_noise;
mod frozen_response;
mod history_baseline;
mod sampling;
mod target_diagnostic;
use sampling::EndpointSampling;

type History = Vec<String>;

struct PublicRow {
    state: GameState,
    reaches: [Vec<f64>; 2],
    keys: Vec<u64>,
    // Action-major, matching the existing exact-private-hand update helpers.
    probabilities: Vec<Vec<f64>>,
}

struct Snapshot {
    rows: BTreeMap<History, PublicRow>,
    endpoints: BTreeMap<History, (GameState, [Vec<f64>; 2])>,
}

impl Snapshot {
    fn capture_frozen(policy: &response::FrozenPreflopPolicy) -> Result<Self, String> {
        let mut snapshot = Self { rows: BTreeMap::new(), endpoints: BTreeMap::new() };
        snapshot.visit_frozen(policy, GameState::initial(&policy.game),
            [vec![1.0; 1326], vec![1.0; 1326]])?;
        Ok(snapshot)
    }

    fn visit_frozen(&mut self, policy: &response::FrozenPreflopPolicy,
        state: GameState, reaches: [Vec<f64>; 2]) -> Result<(), String> {
        if state.terminal.is_some() || state.street != Street::Preflop {
            self.endpoints.insert(state.public_history.clone(), (state, reaches));
            return Ok(());
        }
        let actions = state.legal_actions(&policy.game);
        let mut classes = BTreeMap::new();
        let mut keys = Vec::with_capacity(1326);
        let mut probabilities = vec![vec![0.0; 1326]; actions.len()];
        for combo in all_combos() {
            let class = format!("preflop:{}", combo.label());
            if !classes.contains_key(&class) {
                let (key, _, _) = information_set_from_bucket_trajectories(
                    &state, &policy.game, vec![Arc::from(class.as_str())], Vec::new());
                classes.insert(class.clone(), (key, policy.strategy(&state, combo)?));
            }
            let (key, mix) = &classes[&class];
            keys.push(*key);
            for (a, p) in mix.iter().enumerate() { probabilities[a][combo.key()] = *p; }
        }
        for (a, action) in actions.iter().enumerate() {
            let mut child = reaches.clone();
            for c in 0..1326 { child[state.actor][c] *= probabilities[a][c]; }
            self.visit_frozen(policy, state.apply(action, &policy.game), child)?;
        }
        self.rows.insert(state.public_history.clone(), PublicRow { state, reaches, keys, probabilities });
        Ok(())
    }

    /// Observe one iteration-wide policy without changing regrets, averages,
    /// lazy discounts or RNG. The existing exact sweep initializes all rows.
    fn capture(trainer: &Trainer) -> Result<Self, String> {
        Self::capture_policy(trainer, false)
    }

    fn capture_policy(trainer: &Trainer, average: bool) -> Result<Self, String> {
        let mut snapshot = Self {
            rows: BTreeMap::new(),
            endpoints: BTreeMap::new(),
        };
        snapshot.visit(
            trainer,
            GameState::initial(&trainer.config),
            [vec![1.0; 1326], vec![1.0; 1326]],
            average,
        )?;
        Ok(snapshot)
    }

    fn visit(
        &mut self,
        trainer: &Trainer,
        state: GameState,
        reaches: [Vec<f64>; 2],
        average: bool,
    ) -> Result<(), String> {
        if state.terminal.is_some() || state.street != Street::Preflop {
            self.endpoints
                .insert(state.public_history.clone(), (state, reaches));
            return Ok(());
        }
        let actions = state.legal_actions(&trainer.config);
        let mut classes = BTreeMap::new();
        let mut keys = Vec::with_capacity(1326);
        let mut probabilities = vec![vec![0.0; 1326]; actions.len()];
        for combo in all_combos() {
            let class = format!("preflop:{}", combo.label());
            if !classes.contains_key(&class) {
                let (key, descriptor, _) = information_set_from_bucket_trajectories(
                    &state,
                    &trainer.config,
                    vec![Arc::from(class.as_str())],
                    Vec::new(),
                );
                let mut node = trainer
                    .nodes
                    .get(&key)
                    .ok_or("missing preflop snapshot row")?
                    .clone();
                if node.descriptor != descriptor
                    || node
                        .action_labels
                        .iter()
                        .map(AsRef::as_ref)
                        .ne(actions.iter().map(|a| a.label.as_str()))
                {
                    return Err("preflop snapshot identity/action mismatch".into());
                }
                node.apply_dcfr_regret_discount(
                    trainer.completed_iterations + 1,
                    &trainer.discounts,
                );
                classes.insert(
                    class.clone(),
                    (
                        key,
                        if average {
                            node.average_strategy()
                        } else {
                            node.current_strategy()
                        },
                    ),
                );
            }
            let (key, mix) = &classes[&class];
            keys.push(*key);
            for (a, p) in mix.iter().enumerate() {
                probabilities[a][combo.key()] = *p;
            }
        }
        for (a, action) in actions.iter().enumerate() {
            let mut child = reaches.clone();
            for c in 0..1326 {
                child[state.actor][c] *= probabilities[a][c];
            }
            self.visit(
                trainer,
                state.apply(action, &trainer.config),
                child,
                average,
            )?;
        }
        self.rows.insert(
            state.public_history.clone(),
            PublicRow {
                state,
                reaches,
                keys,
                probabilities,
            },
        );
        Ok(())
    }

    fn flop_input(&self, history: &History, board: [u8; 3]) -> Result<PublicBeliefState, String> {
        let (state, prior) = self
            .endpoints
            .get(history)
            .ok_or("unknown preflop endpoint")?;
        if state.terminal.is_some() || state.street != Street::Flop {
            return Err("continuation requires a live flop endpoint".into());
        }
        // Only the revealed flop is permitted here. No sampled turn/river,
        // private holding, exploration floor or proposal correction enters.
        PublicBeliefState::from_preflop_reaches(state, board, prior.clone())
    }
}

fn mask_ranges(ranges: &mut [Vec<f64>; 2], board: [u8; 3]) {
    for combo in all_combos() {
        if combo.cards().iter().any(|c| board.contains(c)) {
            ranges[0][combo.key()] = 0.0;
            ranges[1][combo.key()] = 0.0;
        }
    }
}

// A public flop has C(52,3) proposals but C(48,3) possibilities given a
// compatible private pair. Private CFVs average over C(50,2) opponents.
const FLOP_CHANCE_CORRECTION: f64 = 22_100.0 / 17_296.0;
const OPPONENT_HANDS: f64 = 1_225.0;

fn begin_update(trainer: &mut Trainer) -> Result<Snapshot, String> {
    let round = trainer.completed_iterations + 1;
    trainer.discounts.advance(round);
    for node in trainer.nodes.values_mut() {
        node.apply_dcfr_regret_discount(round, &trainer.discounts);
    }
    trainer.sweep_preflop_average()?;
    // The first sweep creates nodes after the eager pass above. Stamp their
    // zero-regret state NOW, before adding this round's update; otherwise the
    // next round applies historical discounts to a newly accumulated regret.
    // Existing nodes are already stamped, so this second pass is idempotent.
    for node in trainer.nodes.values_mut() {
        node.apply_dcfr_regret_discount(round, &trainer.discounts);
    }
    Snapshot::capture(trainer)
}

fn back_up_preflop(
    trainer: &mut Trainer,
    snapshot: &Snapshot,
    state: &GameState,
    endpoints: &BTreeMap<History, [Vec<f64>; 2]>,
    traverser: usize,
) -> Result<[Vec<f64>; 2], String> {
    if state.terminal.is_some() || state.street != Street::Preflop {
        return endpoints
            .get(&state.public_history)
            .cloned()
            .ok_or("missing preflop value endpoint".into());
    }
    let row = &snapshot.rows[&state.public_history];
    let actions = state.legal_actions(&trainer.config);
    let children = actions
        .iter()
        .map(|a| {
            back_up_preflop(
                trainer,
                snapshot,
                &state.apply(a, &trainer.config),
                endpoints,
                traverser,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let actor = state.actor;
    let mut result = [vec![0.0; 1326], vec![0.0; 1326]];
    for c in 0..1326 {
        for a in 0..actions.len() {
            result[actor][c] += row.probabilities[a][c] * children[a][actor][c];
            result[1 - actor][c] += children[a][1 - actor][c];
        }
    }
    if actor == traverser {
        let evaluation = range_vector::RangeActionEvaluation {
            actor_action_values_bb: children.iter().map(|v| v[actor].clone()).collect(),
            expected_counterfactual_values_bb: result.clone(),
        };
        let keys = row.keys.iter().map(|key| Some(*key)).collect::<Vec<_>>();
        let updates = range_vector::aggregate_information_set_updates(
            actor,
            &keys,
            &vec![1.0 / 1326.0; 1326],
            &row.reaches[actor],
            &row.probabilities,
            &evaluation,
            None,
        )?;
        trainer.apply_range_information_set_updates(updates, true, false)?;
    }
    Ok(result)
}

/// Both raw-CFV vectors already exist. Optional simultaneous updates reuse
/// this one frozen policy/chance sample; no second oracle or averaging sweep.
fn apply_snapshot_updates(
    trainer: &mut Trainer,
    snapshot: &Snapshot,
    values: &BTreeMap<History, [Vec<f64>; 2]>,
    simultaneous: bool,
) -> Result<[Vec<f64>; 2], String> {
    let root = GameState::initial(&trainer.config);
    let traverser = trainer.completed_iterations as usize % 2;
    let result = back_up_preflop(trainer,snapshot,&root,values,traverser)?;
    if simultaneous {
        let second = back_up_preflop(trainer,snapshot,&root,values,1-traverser)?;
        if second != result {
            return Err("simultaneous backup changed the frozen profile values".into());
        }
    }
    Ok(result)
}

/// Select the research target without changing the frozen continuation policy.
/// Played-profile labels include its actual behavior at zero-own-reach hands;
/// the legacy control substitutes best-response completions there.
pub(super) fn continuation_target(
    policy: &NativePostflopPolicy,
    turn: u8,
    model: &PublicValueNetwork,
    played_profile: bool,
) -> Result<[Vec<f64>; 2], String> {
    if played_profile {
        policy.sampled_profile_values_with_turn_baseline(turn, model)
    } else {
        policy.sampled_training_values_with_turn_baseline(turn, model)
    }
}

/// One compact, exact-private-hand preflop update. Enumerate the small public
/// preflop tree; sample one expensive flop endpoint with an exact-checkdown
/// control variate. The correction is unbiased for this finite-budget oracle,
/// not proof that the oracle supplies equilibrium continuation values.
fn continuation_step(
    trainer: &mut Trainer,
    model: &PublicValueNetwork,
    continuation_seed: u64,
    exact_checkdown: Option<&exact_checkdown::ExactCheckdown>,
    sampling: EndpointSampling,
    fixed_proposal: Option<&sampling::FixedProposal>,
    mut history_baseline: Option<&mut history_baseline::HistoryBaseline>,
    turn_baseline: bool,
    flop_checkdown_scale: f64,
    simultaneous: bool,
    root_turn_averages: bool,
    diagnose_targets: bool,
    played_profile: bool,
    trace_root_updates: bool,
) -> Result<serde_json::Value, String> {
    let started = Instant::now();
    let round = trainer.completed_iterations + 1;
    let snapshot = begin_update(trainer)?;
    let live: Vec<_> = snapshot
        .endpoints
        .iter()
        .filter(|(_, (s, _))| s.terminal.is_none() && s.street == Street::Flop)
        .map(|(h, _)| h.clone())
        .collect();
    if live.is_empty() {
        return Err("preflop pilot has no live flop endpoints".into());
    }
    let selected = if sampling == EndpointSampling::FixedImportance {
        fixed_proposal.ok_or("missing fixed endpoint proposal")?.select(
            &live, trainer.completed_iterations as usize % 2, round, &mut trainer.rng)?
    } else if sampling == EndpointSampling::OpponentReach {
        sampling::select_opponent_reach(
            &snapshot,
            &live,
            trainer.completed_iterations as usize % 2,
            &mut trainer.rng,
        )?
    } else {
        sampling.select(
            &live,
            &GameState::initial(&trainer.config).public_history,
            &mut trainer.rng,
        )?
    };
    // Shared across paired solver seeds; independent from endpoint sampling.
    let mut chance = SplitMix64::new(batch_iteration_seed(continuation_seed, round));
    let deal = Deal::sample(&mut chance);
    let board: [u8; 3] = deal.board[..3].try_into().unwrap();
    let turns: Vec<_> = (0..52).filter(|c| !board.contains(c)).collect();
    let turn = turns[chance.index(turns.len())];
    eprintln!(
        "{}",
        serde_json::json!({"stage":"compact_preflop_round_started","round":round,
        "board":board,"turn":turn,"selectedHistory":selected.keys().next().unwrap(),
        "sampledEndpointCount":selected.len(),"endpointSampling":sampling.label(),
        "liveFlopEndpoints":live.len()})
    );
    let kernel = exact_flop_kernel(board)?;
    let kernel_seconds = started.elapsed().as_secs_f64();
    let mut values = BTreeMap::new();
    let mut samples = Vec::new();
    let mut baseline_observations = Vec::new();
    let mut target_observation = None;
    let conflicts = public_belief::combo_conflicts();
    for (history, (state, prior)) in &snapshot.endpoints {
        if let Some(Terminal::Fold { winner }) = state.terminal {
            let p0 = if winner == 0 {
                state.invested[1]
            } else {
                -state.invested[0]
            };
            values.insert(
                history.clone(),
                std::array::from_fn(|p| {
                    (0..1326)
                        .map(|c| {
                            (if p == 0 { p0 } else { -p0 })
                                * public_belief::compatible_mass_from_conflicts(
                                    &prior[1 - p],
                                    &conflicts,
                                    c,
                                )
                                / OPPONENT_HANDS
                        })
                        .collect()
                }),
            );
            continue;
        }
        let mut masked = prior.clone();
        mask_ranges(&mut masked, board);
        let mut sampled_baseline = [vec![0.0; 1326], vec![0.0; 1326]];
        if exact_checkdown.is_none() || selected.contains_key(history) {
            for p in 0..2 {
                sampled_baseline[p] = kernel.values_on_flop(state.invested, &masked[1 - p], p)?;
                for v in &mut sampled_baseline[p] {
                    *v *= FLOP_CHANCE_CORRECTION / OPPONENT_HANDS;
                }
            }
        }
        let mut baseline = if let Some(exact) = exact_checkdown {
            exact.values(state.invested, prior)?
        } else {
            sampled_baseline.clone()
        };
        scale_live_checkdown_baseline(
            state,
            &mut baseline,
            &mut sampled_baseline,
            flop_checkdown_scale,
        );
        let predicted = history_baseline.as_ref().map(|b| b.predict(history, prior));
        if let Some(predicted) = &predicted {
            for p in 0..2 {
                for c in 0..1326 {
                    baseline[p][c] += predicted[p][c];
                }
            }
        }
        if let Some(&probability) = selected.get(history) {
            let before = Instant::now();
            let mut selected_policy = None;
            let mut residual_mse = None;
            let totals: [f64; 2] = std::array::from_fn(|p| masked[p].iter().sum());
            if totals.iter().any(|v| *v > 0.0) {
                let input = snapshot.flop_input(history, board)?;
                let digest = Sha256::digest(serde_json::to_vec(&input).map_err(|e| e.to_string())?);
                let options = NativeFlopOptions {
                    seed: 100101 ^ u64::from_le_bytes(digest[..8].try_into().unwrap()),
                    iterations: 128,
                    training_turn_iterations: 64,
                    response_turn_iterations: 64,
                };
                let policy = NativePostflopPolicy::solve_pinned_compact_continuation(
                    trainer.config.clone(),
                    input,
                    &options,
                    model,
                    root_turn_averages,
                )?;
                selected_policy = Some(policy.identity().to_owned());
                let mut observed = if diagnose_targets {
                    let (training, profile) = policy.sampled_training_profile_comparison(turn, model)?;
                    target_observation = Some(target_diagnostic::Observation::new(
                        history.clone(), policy.root().clone(), &training, &profile, totals, probability));
                    training
                } else if turn_baseline {
                    continuation_target(&policy, turn, model, played_profile)?
                } else {
                    policy.sampled_training_values(turn)?
                };
                let mut residual = [vec![0.0; 1326], vec![0.0; 1326]];
                let mut before_mse = 0.0;
                let mut after_mse = 0.0;
                for p in 0..2 {
                    for v in &mut observed[p] {
                        *v *= totals[1 - p] * FLOP_CHANCE_CORRECTION / OPPONENT_HANDS;
                    }
                    for c in 0..1326 {
                        residual[p][c] = observed[p][c] - sampled_baseline[p][c];
                        let prediction = predicted.as_ref().map_or(0.0, |v| v[p][c]);
                        if p == trainer.completed_iterations as usize % 2 {
                            before_mse += residual[p][c].powi(2) / 1326.0;
                            after_mse += (residual[p][c] - prediction).powi(2) / 1326.0;
                        }
                        if predicted.is_some() {
                            sampled_baseline[p][c] += prediction;
                        }
                    }
                    // The exact expectation and sampled baseline are distinct:
                    // E[B] + (sampled native value - sampled B) / endpoint q.
                    // Using E[B] in the residual would restore the chance noise.
                    baseline[p] = if exact_checkdown.is_some() {
                        corrected_known_expectation(
                            &baseline[p],
                            &sampled_baseline[p],
                            &observed[p],
                            probability,
                        )?
                    } else {
                        corrected_endpoint(&sampled_baseline[p], Some(&observed[p]), probability)?
                    };
                }
                residual_mse = Some(serde_json::json!({"before":before_mse,"after":after_mse,
                    "interpretation":"Observed traverser raw-CFV correction squared mean, before endpoint importance weighting; not action-EV error or a variance certificate."}));
                if history_baseline.is_some() {
                    baseline_observations.push((history.clone(), residual));
                }
            }
            samples.push(serde_json::json!({"selectedHistory":history,
                "endpointProposal":probability,"selectedRawRangeTotals":totals,
                "strategicResidualSquaredMean":residual_mse,
                "flopPolicySha256":selected_policy,"seconds":before.elapsed().as_secs_f64()}));
        }
        values.insert(history.clone(), baseline);
    }
    let target_report = target_observation.map(|v| v.describe(&snapshot,&trainer.config,
        &values,trainer.completed_iterations as usize%2)).transpose()?;
    let mut root_report = trace_root_updates.then(||
        target_diagnostic::root_trace(trainer,&snapshot,&values)).transpose()?;
    let result = apply_snapshot_updates(trainer,&snapshot,&values,simultaneous)?;
    if let Some(report) = &mut root_report {
        report["regretsAfter"] = serde_json::json!(target_diagnostic::root_regrets(trainer,&snapshot));
    }
    if result.iter().flatten().any(|v| !v.is_finite()) {
        return Err("nonfinite preflop backup".into());
    }
    let residual = result
        .iter()
        .map(|v| v.iter().sum::<f64>() / 1326.0)
        .sum::<f64>();
    if residual.abs() > 1e-5 {
        return Err(format!(
            "preflop backup lost zero-sum accounting: {residual}"
        ));
    }
    if let Some(cache) = history_baseline.as_mut() {
        for (history, value) in baseline_observations {
            cache.observe(history.clone(), &snapshot.endpoints[&history].1, &value)?;
        }
    }
    trainer.completed_iterations += 1;
    assert_eq!(
        trainer.nodes.len(),
        16_900,
        "preflop-only table must not grow postflop rows"
    );
    let first = samples.first().ok_or("no sampled continuation values")?;
    Ok(
        serde_json::json!({"round":round,"seconds":started.elapsed().as_secs_f64(),
        "kernelSeconds":kernel_seconds,"board":board,"turn":turn,
        "selectedHistory":first["selectedHistory"],"endpointProposal":first["endpointProposal"],
        "selectedRawRangeTotals":first["selectedRawRangeTotals"],"flopPolicySha256":first["flopPolicySha256"],
        "endpointSamples":samples,"endpointSampling":sampling.label(),
        "targetDiagnostic":target_report,
        "rootUpdateTrace":root_report,
        "simultaneousUpdates":simultaneous,
        "rootRealizationTurnAverages":root_turn_averages,
        "historyBaseline":history_baseline.is_some(),
        "completeTurnBaseline":turn_baseline,
        "flopCheckdownScale":flop_checkdown_scale,
        "historyBaselineEntries":history_baseline.as_ref().map_or(0,|b| b.len()),
        "liveFlopEndpoints":live.len(),"preflopNodes":trainer.nodes.len(),
        "continuationSeed":continuation_seed,"valueModelSha256":model.artifact_sha256(),
        "exactPreflopCheckdown":exact_checkdown.is_some(),
        "zeroSumResidual":residual,
        "interpretation":"Finite-budget counterfactual oracle training estimate; corrected public endpoints with a shared sampled turn, not a policy-response or EV-confidence result."}),
    )
}

#[test]
#[ignore = "bounded hash-pinned compact preflop continuation pilot; external resource guard required"]
fn compact_preflop_continuation_pilot() {
    let seed: u64 = std::env::var("POKER_COMPACT_SEED")
        .unwrap()
        .parse()
        .unwrap();
    let continuation_seed: u64 = std::env::var("POKER_COMPACT_CHANCE_SEED")
        .unwrap()
        .parse()
        .unwrap();
    let rounds: u64 = std::env::var("POKER_COMPACT_ROUNDS")
        .unwrap()
        .parse()
        .unwrap();
    assert!([27001, 27002].contains(&seed));
    assert!([28001, 28002].contains(&continuation_seed));
    assert!([2, 4, 8, 16, 32, 128].contains(&rounds));
    let model = PublicValueNetwork::read(Path::new(&std::env::var("POKER_COMPACT_MODEL").unwrap()))
        .unwrap();
    assert_eq!(
        model.artifact_sha256().unwrap(),
        std::env::var("POKER_COMPACT_MODEL_SHA").unwrap()
    );
    let checkdown_sha = std::env::var("POKER_COMPACT_CHECKDOWN_SHA").ok();
    let checkdown = std::env::var("POKER_COMPACT_CHECKDOWN").ok().map(|path| {
        exact_checkdown::ExactCheckdown::read(Path::new(&path), checkdown_sha.as_deref().unwrap())
            .unwrap()
    });
    assert_eq!(checkdown.is_some(), checkdown_sha.is_some());
    let sampling = EndpointSampling::parse(
        &std::env::var("POKER_COMPACT_ENDPOINT_SAMPLING").unwrap_or_else(|_| "uniform_one".into()),
    )
    .unwrap();
    assert!(sampling == EndpointSampling::UniformOne || checkdown.is_some());
    let proposal_sha = std::env::var("POKER_COMPACT_PROPOSAL_SHA").ok();
    let fixed_proposal = std::env::var("POKER_COMPACT_PROPOSAL").ok().map(|path| {
        sampling::FixedProposal::read(Path::new(&path), proposal_sha.as_deref().unwrap()).unwrap()
    });
    assert_eq!(fixed_proposal.is_some(), proposal_sha.is_some());
    assert_eq!(fixed_proposal.is_some(), sampling == EndpointSampling::FixedImportance);
    let use_history_baseline = std::env::var("POKER_COMPACT_HISTORY_BASELINE")
        .map(|v| {
            assert!(v == "0" || v == "1");
            v == "1"
        })
        .unwrap_or(false);
    assert!(
        !use_history_baseline || (checkdown.is_some() && sampling == EndpointSampling::UniformOne)
    );
    let mut history_baseline = use_history_baseline.then(history_baseline::HistoryBaseline::new);
    let simultaneous = std::env::var("POKER_COMPACT_SIMULTANEOUS")
        .map(|v| { assert!(v == "0" || v == "1"); v == "1" }).unwrap_or(false);
    let root_turn_averages = std::env::var("POKER_COMPACT_TURN_ROOT_AVERAGES")
        .map(|v| { assert!(v == "0" || v == "1"); v == "1" }).unwrap_or(false);
    let turn_baseline = std::env::var("POKER_COMPACT_TURN_BASELINE")
        .map(|v| {
            assert!(v == "0" || v == "1");
            v == "1"
        })
        .unwrap_or(false);
    assert!(
        !turn_baseline
            || (!use_history_baseline
                && checkdown.is_some()
                && matches!(sampling, EndpointSampling::UniformOne | EndpointSampling::FixedImportance))
    );
    let flop_checkdown_scale: f64 = std::env::var("POKER_COMPACT_FLOP_CHECKDOWN_SCALE")
        .unwrap_or_else(|_| "1".into())
        .parse()
        .unwrap();
    assert!([1.0, 2.0].contains(&flop_checkdown_scale));
    assert!(flop_checkdown_scale == 1.0 || turn_baseline);
    assert!(!simultaneous || (turn_baseline && sampling == EndpointSampling::UniformOne
        && !use_history_baseline && flop_checkdown_scale == 1.0));
    assert!(!root_turn_averages || (!simultaneous && turn_baseline
        && !use_history_baseline && flop_checkdown_scale == 1.0));
    let diagnose_targets = std::env::var("POKER_COMPACT_DIAGNOSE_TARGETS")
        .map(|v| { assert!(v=="0" || v=="1"); v=="1" }).unwrap_or(false);
    let played_profile = std::env::var("POKER_COMPACT_PLAYED_TARGETS")
        .map(|v| { assert!(v=="0" || v=="1"); v=="1" }).unwrap_or(false);
    let trace_root_updates = std::env::var("POKER_COMPACT_TRACE_ROOT")
        .map(|v| { assert!(v=="0" || v=="1"); v=="1" }).unwrap_or(false);
    assert!(!trace_root_updates || (rounds<=32 && played_profile && turn_baseline
        && !root_turn_averages && !simultaneous && !diagnose_targets
        && !use_history_baseline && flop_checkdown_scale==1.0));
    assert!(fixed_proposal.is_none() || (rounds<=128 && played_profile && turn_baseline
        && !simultaneous && !root_turn_averages && !diagnose_targets
        && !use_history_baseline && flop_checkdown_scale==1.0));
    assert!(!played_profile || (rounds<=128 && turn_baseline && !simultaneous
        && !use_history_baseline && !diagnose_targets
        && flop_checkdown_scale==1.0));
    assert!(!diagnose_targets || (rounds<=8 && turn_baseline && !simultaneous
        && !root_turn_averages && !use_history_baseline && flop_checkdown_scale==1.0));
    // The cost-audited extension changes only the number of updates of the
    // retained control. Do not silently extend rejected sampling experiments.
    assert!(rounds <= 32 || (turn_baseline && !simultaneous && !root_turn_averages
        && !use_history_baseline && matches!(sampling, EndpointSampling::UniformOne | EndpointSampling::FixedImportance)
        && flop_checkdown_scale == 1.0));
    let output = PathBuf::from(std::env::var("POKER_COMPACT_OUTPUT").unwrap());
    let frozen = output.with_extension("preflop.json.gz");
    assert!(!output.exists() && !frozen.exists());
    let regret_schedule = std::env::var("POKER_COMPACT_REGRET_SCHEDULE")
        .unwrap_or_else(|_| "dcfr".into());
    assert!(["dcfr","lcfr"].contains(&regret_schedule.as_str()));
    assert!(fixed_proposal.as_ref().and_then(|p|p.refresh_after_round()).is_none()
        || regret_schedule=="lcfr");
    assert!(regret_schedule!="lcfr" || (rounds<=128 && played_profile && turn_baseline
        && sampling==EndpointSampling::FixedImportance && !simultaneous
        && !root_turn_averages && !use_history_baseline && !diagnose_targets
        && flop_checkdown_scale==1.0));
    let mut config = BlueprintConfig {
        seed,
        effective_stack_bb: 20.0,
        iterations: rounds,
        averaging_delay: 0,
        exact_preflop_averaging: true,
        max_information_sets: 20_000,
        traversal: BlueprintTraversal::PublicChanceSampling,
        ..BlueprintConfig::default()
    };
    if regret_schedule=="lcfr" {
        config.dcfr_schedule = DcfrSchedule::Lcfr;
        config.dcfr = DcfrParameters { positive_regret_exponent:1.0,
            negative_regret_exponent:1.0, strategy_exponent:1.0 };
    }
    config.validate().unwrap();
    let checkpoint_interval: u64 = std::env::var("POKER_COMPACT_CHECKPOINT_INTERVAL")
        .unwrap_or_else(|_| "0".into()).parse().unwrap();
    assert!([0,8,16,32].contains(&checkpoint_interval));
    let resume_path = std::env::var("POKER_COMPACT_RESUME_RECEIPT").ok();
    let resume_sha = std::env::var("POKER_COMPACT_RESUME_SHA").ok();
    assert_eq!(resume_path.is_some(),resume_sha.is_some());
    assert!((checkpoint_interval==0 && resume_path.is_none()) ||
        (played_profile && turn_baseline && !use_history_baseline && !simultaneous
            && !root_turn_averages && !diagnose_targets && flop_checkdown_scale==1.0));
    let recovery_identity = serde_json::json!({
        "binarySha256":std::env::var("POKER_COMPACT_BINARY_SHA").ok(),
        "modelSha256":model.artifact_sha256(),"checkdownSha256":checkdown_sha,
        "proposalSha256":proposal_sha,"continuationSeed":continuation_seed,
        "sampling":sampling.label(),"playedProfile":played_profile,
        "turnBaseline":turn_baseline,"rootTrace":trace_root_updates,"regretSchedule":regret_schedule,
        "flopIterations":128,"turnIterations":64});
    if checkpoint_interval>0 || resume_path.is_some() {
        assert!(recovery_identity["binarySha256"].as_str().is_some_and(|s|
            s.len()==64 && s.bytes().all(|c|c.is_ascii_hexdigit())));
    }
    let (mut trainer, mut progress) = if let Some(path) = &resume_path {
        checkpoint::restore(Path::new(path),resume_sha.as_deref().unwrap(),&recovery_identity,&config).unwrap()
    } else { (Trainer::fresh(config.clone()),Vec::new()) };
    let resumed_from_round = trainer.completed_iterations;
    let started = Instant::now();
    while trainer.completed_iterations < rounds {
        let tick = continuation_step(
            &mut trainer,
            &model,
            continuation_seed,
            checkdown.as_ref(),
            sampling,
            fixed_proposal.as_ref(),
            history_baseline.as_mut(),
            turn_baseline,
            flop_checkdown_scale,
            simultaneous,
            root_turn_averages,
            diagnose_targets,
            played_profile,
            trace_root_updates,
        )
        .unwrap();
        eprintln!("{tick}");
        progress.push(tick);
        if checkpoint_interval>0 && (trainer.completed_iterations%checkpoint_interval==0
            || trainer.completed_iterations==rounds) {
            let receipt = checkpoint::save(&trainer,&output.with_extension("checkpoints"),
                &recovery_identity,&progress).unwrap();
            eprintln!("{}",serde_json::json!({"stage":"compact_checkpoint_complete",
                "round":trainer.completed_iterations,"receipt":receipt,
                "receiptSha256":format!("{:x}",Sha256::digest(fs::read(&receipt).unwrap()))}));
        }
    }
    trainer.write_frozen_preflop_average(&frozen).unwrap();
    let snapshot = Snapshot::capture(&trainer).unwrap();
    let mut classes = BTreeMap::new();
    for combo in all_combos() {
        classes.entry(combo.label()).or_insert((combo, 0)).1 += 1;
    }
    let mut rows = Vec::new();
    for row in snapshot.rows.values() {
        for (class, (combo, multiplicity)) in &classes {
            let key = row.keys[combo.key()];
            let node = &trainer.nodes[&key];
            rows.push(
                serde_json::json!({"key":key.to_string(),"actor":row.state.actor,
                "history":row.state.public_history,"hand":class,"comboWeight":multiplicity,
                "actions":node.action_labels,"probabilities":node.average_strategy(),
                "averageVisits":node.average_visits,"regretUpdates":node.regret_updates,
                "trained":node.average_visits > 0 && node.regret_updates > 0}),
            );
        }
    }
    let payload = serde_json::json!({"schema":"compact-preflop-continuation-pilot-v1",
        "config":config,"continuationSeed":continuation_seed,"rows":rows,"progress":progress,
        "trainingSeconds":started.elapsed().as_secs_f64(),"totalNodes":trainer.nodes.len(),
        "valueModelSha256":model.artifact_sha256(),"frozenPolicy":frozen,
        "exactCheckdownSha256":checkdown_sha,
        "endpointSampling":sampling.label(),
        "endpointProposalSha256":proposal_sha,
        "endpointProposalRefreshAfterRound":fixed_proposal.as_ref().and_then(|p|p.refresh_after_round()),
        "regretSchedule":regret_schedule,
        "historyBaseline":use_history_baseline,
        "completeTurnBaseline":turn_baseline,
        "flopCheckdownScale":flop_checkdown_scale,
        "frozenPolicySha256":format!("{:x}", Sha256::digest(fs::read(&frozen).unwrap())),"releaseAccepted":false,
        "simultaneousUpdates":simultaneous,
        "rootRealizationTurnAverages":root_turn_averages,
        "targetDiagnosticEnabled":diagnose_targets,
        "rootUpdateTraceEnabled":trace_root_updates,
        "checkpointInterval":checkpoint_interval,"resumedFromRound":resumed_from_round,
        "resumeReceiptSha256":resume_sha,
        "playedProfileTargets":played_profile,
        "interpretation":"Compact exact-private-hand DCFR with exact preflop averages, endpoint checkdown control variate and evolving counterfactual public beliefs. Approximate native continuation oracle; not full-game exploitability."});
    let bytes = serde_json::to_vec(&payload).unwrap();
    assert!(bytes.len() < 32 * 1024 * 1024);
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap();
    file.write_all(&bytes).unwrap();
    file.sync_all().unwrap();
}

/// Only live sampled continuations may use a scaled control. Exact terminals
/// have no correcting native observation, so scaling them would change poker.
fn scale_live_checkdown_baseline(
    state: &GameState,
    mean: &mut [Vec<f64>; 2],
    sampled: &mut [Vec<f64>; 2],
    scale: f64,
) {
    assert!([1.0, 2.0].contains(&scale));
    if state.terminal.is_none() && state.street == Street::Flop && scale != 1.0 {
        for pair in [mean, sampled] {
            for row in pair {
                for v in row {
                    *v *= scale;
                }
            }
        }
    }
}

#[test]
fn scaled_control_changes_both_live_baselines_but_never_exact_terminal_payoffs() {
    let game = BlueprintConfig {
        effective_stack_bb: 20.0,
        ..BlueprintConfig::default()
    };
    let mut pending = vec![GameState::initial(&game)];
    let mut live = 0;
    let mut folds = 0;
    let mut showdowns = 0;
    while let Some(state) = pending.pop() {
        if state.terminal.is_some() || state.street != Street::Preflop {
            let mut mean = [vec![2.0], vec![-2.0]];
            let mut sampled = [vec![3.0], vec![-3.0]];
            scale_live_checkdown_baseline(&state, &mut mean, &mut sampled, 2.0);
            if state.terminal.is_none() {
                live += 1;
                assert_eq!(mean[0][0], 4.0);
                assert_eq!(sampled[0][0], 6.0);
                let q = 1.0 / 49.0;
                // Two equally likely chance values whose scaled mean is four.
                let expectation = (1.0 - q) * mean[0][0]
                    + [2.0, 6.0]
                        .iter()
                        .map(|b| {
                            q * 0.5
                                * corrected_known_expectation(&mean[0], &[*b], &[7.0], q).unwrap()
                                    [0]
                        })
                        .sum::<f64>();
                assert!((expectation - 7.0).abs() < 1e-10);
            } else {
                if matches!(state.terminal, Some(Terminal::Fold { .. })) {
                    folds += 1;
                } else {
                    showdowns += 1;
                }
                assert_eq!(mean, [vec![2.0], vec![-2.0]]);
                assert_eq!(sampled, [vec![3.0], vec![-3.0]]);
            }
        } else {
            pending.extend(
                state
                    .legal_actions(&game)
                    .iter()
                    .map(|a| state.apply(a, &game)),
            );
        }
    }
    assert_eq!(live, 49);
    assert!(folds > 0 && showdowns > 0);
}

/// Horvitz-Thompson baseline correction for an explicitly sampled endpoint.
/// Baselines must be fixed before drawing the endpoint and are not labels.
fn corrected_endpoint(
    baseline: &[f64],
    sampled: Option<&[f64]>,
    probability: f64,
) -> Result<Vec<f64>, String> {
    if !probability.is_finite()
        || probability <= 0.0
        || probability > 1.0
        || baseline.iter().any(|v| !v.is_finite())
        || sampled.is_some_and(|v| v.len() != baseline.len() || v.iter().any(|x| !x.is_finite()))
    {
        return Err("invalid endpoint baseline/proposal".into());
    }
    let result: Vec<_> = baseline
        .iter()
        .enumerate()
        .map(|(i, b)| sampled.map_or(*b, |v| b + (v[i] - b) / probability))
        .collect();
    if result.iter().any(|v| !v.is_finite()) {
        return Err("endpoint correction overflow".into());
    }
    Ok(result)
}

fn corrected_known_expectation(
    mean: &[f64],
    sampled_baseline: &[f64],
    observed: &[f64],
    probability: f64,
) -> Result<Vec<f64>, String> {
    if !probability.is_finite()
        || probability <= 0.0
        || probability > 1.0
        || mean.len() != sampled_baseline.len()
        || mean.len() != observed.len()
        || mean
            .iter()
            .chain(sampled_baseline)
            .chain(observed)
            .any(|v| !v.is_finite())
    {
        return Err("invalid exact-mean baseline correction".into());
    }
    let result: Vec<_> = mean
        .iter()
        .zip(sampled_baseline)
        .zip(observed)
        .map(|((m, b), v)| m + (v - b) / probability)
        .collect();
    if result.iter().any(|v| !v.is_finite()) {
        return Err("exact-mean correction overflow".into());
    }
    Ok(result)
}

#[test]
fn known_checkdown_expectation_removes_baseline_chance_noise_without_bias() {
    for q in [0.2, 1.0] {
        let mut expected = (1.0 - q) * 1.0;
        for b in [-10.0, 12.0] {
            let result = corrected_known_expectation(&[1.0], &[b], &[b + 1.0], q).unwrap();
            expected += q * 0.5 * result[0];
            if q == 1.0 {
                assert_eq!(result[0], 2.0);
            }
        }
        assert!((expected - 2.0).abs() < 1e-12);
    }
    assert!(corrected_known_expectation(&[1.0], &[], &[2.0], 1.0).is_err());
    assert!(corrected_known_expectation(&[1.0], &[1.0], &[2.0], 0.0).is_err());
}

#[test]
fn public_flop_proposal_has_exact_compatible_private_pair_correction() {
    let private = [51, 50, 47, 46];
    let mut proposed = 0;
    let mut compatible = 0;
    for a in 0..52u8 {
        for b in a + 1..52 {
            for c in b + 1..52 {
                proposed += 1;
                if [a, b, c].iter().all(|v| !private.contains(v)) {
                    compatible += 1;
                }
            }
        }
    }
    assert_eq!((proposed, compatible), (22_100, 17_296));
    assert!((compatible as f64 / proposed as f64 * FLOP_CHANCE_CORRECTION - 1.0).abs() < 1e-15);
}

#[test]
fn simultaneous_updates_reuse_frozen_values_without_extra_chance_or_averaging() {
    let config = BlueprintConfig { effective_stack_bb:20.0, iterations:2,
        averaging_delay:0,exact_preflop_averaging:true,
        traversal:BlueprintTraversal::PublicChanceSampling,..BlueprintConfig::default() };
    let mut alternating = Trainer::fresh(config.clone());
    let mut simultaneous = Trainer::fresh(config);
    let snapshot = begin_update(&mut alternating).unwrap();
    let other = begin_update(&mut simultaneous).unwrap();
    let conflicts = public_belief::combo_conflicts();
    let values = snapshot.endpoints.iter().map(|(h,(_,r))| {
        // Synthetic varying terminal utilities, not a poker strategy fixture.
        let payoff = h.len() as f64 / 10.0;
        let cfv = std::array::from_fn(|p| (0..1326).map(|c|
            (if p==0 {payoff} else {-payoff})
            * public_belief::compatible_mass_from_conflicts(&r[1-p],&conflicts,c)/OPPONENT_HANDS
        ).collect());
        (h.clone(),cfv)
    }).collect();
    let rng = simultaneous.rng.state();
    let a = apply_snapshot_updates(&mut alternating,&snapshot,&values,false).unwrap();
    let b = apply_snapshot_updates(&mut simultaneous,&other,&values,true).unwrap();
    assert_eq!(a,b);
    assert_eq!(rng,simultaneous.rng.state());
    assert_eq!(simultaneous.completed_iterations,0);
    let mut extra_updates = 0;
    for (key,node) in &simultaneous.nodes {
        let old = &alternating.nodes[key];
        assert_eq!(node.strategy_sum,old.strategy_sum);
        assert_eq!(node.average_visits,old.average_visits);
        assert!(node.regret_updates>0);
        if old.regret_updates>0 {
            assert_eq!(node.regrets,old.regrets);
            assert_eq!(node.regret_updates,old.regret_updates);
        } else { extra_updates += 1; }
    }
    assert!(extra_updates>0 && extra_updates<16900);
}

#[test]
fn compact_backup_reuses_exact_private_regrets_without_own_reach_weighting() {
    let config = BlueprintConfig {
        effective_stack_bb: 20.0,
        iterations: 2,
        averaging_delay: 0,
        exact_preflop_averaging: true,
        traversal: BlueprintTraversal::PublicChanceSampling,
        ..BlueprintConfig::default()
    };
    let mut trainer = Trainer::fresh(config.clone());
    trainer.discounts.advance(1);
    trainer.sweep_preflop_average().unwrap();
    let snapshot = Snapshot::capture(&trainer).unwrap();
    let endpoints = snapshot
        .endpoints
        .iter()
        .map(|(h, (_, r))| {
            // Synthetic constant per-endpoint zero-sum game, not a poker model.
            let conflicts = public_belief::combo_conflicts();
            (
                h.clone(),
                std::array::from_fn(|p| {
                    (0..1326)
                        .map(|c| {
                            (if p == 0 { 1.0 } else { -1.0 })
                                * public_belief::compatible_mass_from_conflicts(
                                    &r[1 - p],
                                    &conflicts,
                                    c,
                                )
                                / OPPONENT_HANDS
                        })
                        .collect()
                }),
            )
        })
        .collect();
    let root = GameState::initial(&config);
    let result = back_up_preflop(&mut trainer, &snapshot, &root, &endpoints, 0).unwrap();
    for p in 0..2 {
        for v in &result[p] {
            assert!((v - if p == 0 { 1.0 } else { -1.0 }).abs() < 1e-12);
        }
    }
    assert_eq!(trainer.nodes.len(), 16_900);
    for node in trainer.nodes.values() {
        assert_eq!(
            node.regret_updates > 0,
            node.descriptor.actor == Position::for_player(0)
        );
        assert!(node.regrets.iter().all(|r| r.abs() < 1e-12));
    }
}

#[test]
fn snapshot_preserves_training_and_conditions_both_players_without_future_cards() {
    let config = BlueprintConfig {
        effective_stack_bb: 20.0,
        iterations: 2,
        averaging_delay: 0,
        exact_preflop_averaging: true,
        traversal: BlueprintTraversal::PublicChanceSampling,
        ..BlueprintConfig::default()
    };
    let mut trainer = Trainer::fresh(config.clone());
    trainer.discounts.advance(1);
    trainer.sweep_preflop_average().unwrap();
    let before = rmp_serde::to_vec_named(&trainer.nodes).unwrap();
    let rng = trainer.rng.state();
    let snapshot = Snapshot::capture(&trainer).unwrap();
    assert_eq!(before, rmp_serde::to_vec_named(&trainer.nodes).unwrap());
    assert_eq!(rng, trainer.rng.state());
    assert_eq!(snapshot.rows.len(), 100);
    let root = GameState::initial(&config);
    let hero = Combo::new(51, 50).key();
    let row = &snapshot.rows[&root.public_history];
    let actions = root.legal_actions(&config);
    let limp = actions
        .iter()
        .position(|a| a.kind == ActionKind::Call)
        .unwrap();
    let child = root.apply(&actions[limp], &config);
    let child_row = &snapshot.rows[&child.public_history];
    assert_eq!(row.state.actor, 0);
    assert_eq!(child_row.reaches[0][hero], row.probabilities[limp][hero]);
    assert_eq!(child_row.reaches[1][hero], 1.0);
    let check = child
        .legal_actions(&config)
        .into_iter()
        .find(|a| a.kind == ActionKind::Check)
        .unwrap();
    let flop = child.apply(&check, &config);
    let input = snapshot
        .flop_input(&flop.public_history, [0, 5, 10])
        .unwrap();
    assert!(input.ranges[0][hero] > 0.0);
    assert_eq!(input.ranges[0][Combo::new(0, 51).key()], 0.0);
    assert!(
        input.ranges[0][Combo::new(15, 20).key()] > 0.0,
        "unrevealed cards must stay legal"
    );
    let node = trainer.nodes.get_mut(&row.keys[hero]).unwrap();
    node.regrets.fill(0.0);
    node.regrets[limp] = 1.0;
    let changed = Snapshot::capture(&trainer).unwrap();
    assert_eq!(changed.rows[&child.public_history].reaches[0][hero], 1.0);
    assert_eq!(
        changed.rows[&child.public_history].reaches[1],
        child_row.reaches[1]
    );
    assert_ne!(
        changed
            .flop_input(&flop.public_history, [0, 5, 10])
            .unwrap()
            .ranges,
        input.ranges
    );
}

#[test]
fn endpoint_baseline_is_unbiased_for_each_leaf_and_arbitrary_stale_baselines() {
    let truth = [[3.0, -2.0], [-7.0, 1.0], [2.0, 4.0]];
    let baseline = [[8.0, -9.0], [0.0, 0.0], [2.0, 4.0]];
    let proposal = [0.1, 0.3, 0.6];
    for leaf in 0..3 {
        let mut mean = [0.0; 2];
        for draw in 0..3 {
            let result = corrected_endpoint(
                &baseline[leaf],
                (draw == leaf).then_some(truth[leaf].as_slice()),
                proposal[leaf],
            )
            .unwrap();
            for c in 0..2 {
                mean[c] += proposal[draw] * result[c];
            }
        }
        for c in 0..2 {
            assert!((mean[c] - truth[leaf][c]).abs() < 1e-12);
        }
    }
    assert!(corrected_endpoint(&[1.0], Some(&[2.0]), 0.0).is_err());
    assert!(corrected_endpoint(&[1.0], Some(&[]), 0.5).is_err());
}
