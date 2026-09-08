//! Observe one unchanged update. Compare targets using the same native solves,
//! chance draw and importance correction; never update from the diagnostic.
use super::*;

fn root_classes() -> BTreeMap<String, Vec<usize>> {
    let mut classes: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for c in all_combos() { classes.entry(c.label()).or_default().push(c.key()); }
    classes
}

pub(super) fn root_regrets(trainer: &Trainer, snapshot: &Snapshot) -> Vec<Vec<f64>> {
    let root = GameState::initial(&trainer.config);
    let row = &snapshot.rows[&root.public_history];
    let classes = root_classes();
    (0..row.probabilities.len()).map(|a| classes.values().map(|indices|
        trainer.nodes[&row.keys[indices[0]]].regrets[a]).collect()).collect()
}

/// Observe the EXACT estimates fed to the root update. No extra native solves,
/// RNG calls, policy queries that apply lazy discounts, or training mutations.
pub(super) fn root_trace(trainer: &Trainer, snapshot: &Snapshot,
    endpoints: &BTreeMap<History, [Vec<f64>; 2]>) -> Result<serde_json::Value, String> {
    let root = GameState::initial(&trainer.config);
    let actions = root.legal_actions(&trainer.config);
    let values = actions.iter().map(|a| value(snapshot, &trainer.config,
        &root.apply(a, &trainer.config), endpoints, None).map(|v| v[0].clone()))
        .collect::<Result<Vec<_>, _>>()?;
    let classes = root_classes();
    let reduce = |matrix: &[Vec<f64>]| -> Vec<Vec<f64>> {
        matrix.iter().map(|row| classes.values().map(|indices|
            indices.iter().map(|c| row[*c]).sum::<f64>()/indices.len() as f64).collect()).collect()
    };
    Ok(serde_json::json!({"rootUpdatedThisRound":trainer.completed_iterations%2==0,
        "classes":classes.keys().collect::<Vec<_>>(),
        "multiplicities":classes.values().map(Vec::len).collect::<Vec<_>>(),
        "actions":actions.iter().map(|a|&a.label).collect::<Vec<_>>(),
        "probabilities":reduce(&snapshot.rows[&root.public_history].probabilities),
        "actionValuesBb":reduce(&values), "regretsBefore":root_regrets(trainer,snapshot),
        "interpretation":"Actual importance-weighted root estimates and discounted regrets, not converged action EVs. Read-only training trace."}))
}

pub(super) struct Observation {
    history: History,
    input: PublicBeliefState,
    native_gap: [Vec<f64>; 2],
    endpoint_gap: [Vec<f64>; 2],
}

impl Observation {
    pub(super) fn new(history: History, input: PublicBeliefState,
        training: &[Vec<f64>; 2], profile: &[Vec<f64>; 2], totals: [f64; 2], q: f64) -> Self {
        let native_gap: [Vec<f64>; 2] = std::array::from_fn(|p|
            training[p].iter().zip(&profile[p]).map(|(a,b)| a-b).collect());
        let endpoint_gap = std::array::from_fn(|p| native_gap[p].iter()
            .map(|v| v*totals[1-p]*FLOP_CHANCE_CORRECTION/OPPONENT_HANDS/q).collect());
        Self { history, input, native_gap, endpoint_gap }
    }

    pub(super) fn describe(&self, snapshot: &Snapshot, game: &BlueprintConfig,
        endpoints: &BTreeMap<History, [Vec<f64>; 2]>, traverser: usize)
        -> Result<serde_json::Value, String> {
        let root = GameState::initial(game);
        let actions = root.legal_actions(game);
        let training = actions.iter().map(|a| value(snapshot, game,
            &root.apply(a, game), endpoints, None).map(|v| v[0].clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let profile = actions.iter().map(|a| value(snapshot, game,
            &root.apply(a, game), endpoints, Some(self)).map(|v| v[0].clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let mut classes: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for c in all_combos() { classes.entry(c.label()).or_default().push(c.key()); }
        let reduce = |matrix: &[Vec<f64>]| -> Vec<Vec<f64>> {
            matrix.iter().map(|row| classes.values().map(|indices|
                indices.iter().map(|c| row[*c]).sum::<f64>()/indices.len() as f64).collect()).collect()
        };
        let mut on_support = [0.0f64; 2];
        let mut off_support = [0.0f64; 2];
        let mut zero_count = [0; 2];
        for p in 0..2 {
            for c in all_combos() {
                if c.cards().iter().any(|v| self.input.board.contains(v)) { continue; }
                let gap = self.native_gap[p][c.key()].abs();
                if self.input.ranges[p][c.key()] == 0.0 {
                    zero_count[p] += 1;
                    off_support[p] = off_support[p].max(gap);
                } else { on_support[p] = on_support[p].max(gap); }
            }
        }
        Ok(serde_json::json!({"input":self.input,"selectedHistory":self.history,
            "zeroOwnRootHoldings":zero_count,"maxNativeGapOnPositiveRootReachBb":on_support,
            "maxNativeGapOnZeroRootReachBb":off_support,"rootUpdatedThisRound":traverser==0,
            "classes":classes.keys().collect::<Vec<_>>(),
            "multiplicities":classes.values().map(Vec::len).collect::<Vec<_>>(),
            "actions":actions.iter().map(|a|&a.label).collect::<Vec<_>>(),
            "rootProbabilities":reduce(&snapshot.rows[&root.public_history].probabilities),
            "trainingRootActionValuesBb":reduce(&training),
            "profileRootActionValuesBb":reduce(&profile),
            "interpretation":"Same sampled update, differing only in zero-own-reach BR completion versus actual frozen profile at the selected endpoint. These importance-weighted root estimates are not converged action EVs or release grades. Diagnostic does not alter training."}))
    }
}

fn value(snapshot: &Snapshot, game: &BlueprintConfig, state: &GameState,
    endpoints: &BTreeMap<History, [Vec<f64>; 2]>, alternative: Option<&Observation>)
    -> Result<[Vec<f64>; 2], String> {
    if let Some(values) = endpoints.get(&state.public_history) {
        return Ok(std::array::from_fn(|p| values[p].iter().enumerate().map(|(c,v)| {
            v-alternative.filter(|a| a.history==state.public_history).map_or(0.0,|a|a.endpoint_gap[p][c])
        }).collect()));
    }
    let row = snapshot.rows.get(&state.public_history).ok_or("missing diagnostic row")?;
    let mut result = [vec![0.0; 1326], vec![0.0; 1326]];
    for (a, action) in state.legal_actions(game).iter().enumerate() {
        let child = value(snapshot, game, &state.apply(action, game), endpoints, alternative)?;
        for c in 0..1326 {
            result[state.actor][c] += row.probabilities[a][c]*child[state.actor][c];
            result[1-state.actor][c] += child[1-state.actor][c];
        }
    }
    Ok(result)
}

#[test]
fn diagnostic_backup_matches_actual_preflop_backup_without_mutating_policy() {
    let game = BlueprintConfig { effective_stack_bb: 20.0, exact_preflop_averaging: true,
        averaging_delay: 0, ..BlueprintConfig::default() };
    let mut trainer = Trainer::fresh(game.clone());
    let snapshot = begin_update(&mut trainer).unwrap();
    let endpoints = snapshot.endpoints.keys().enumerate().map(|(i,h)|
        (h.clone(), [vec![i as f64/10.0;1326], vec![-(i as f64)/10.0;1326]])).collect();
    let root = GameState::initial(&game);
    let expected = value(&snapshot,&game,&root,&endpoints,None).unwrap();
    let actual = back_up_preflop(&mut trainer,&snapshot,&root,&endpoints,0).unwrap();
    assert_eq!(expected,actual);
    assert_eq!(expected,value(&snapshot,&game,&root,&endpoints,None).unwrap());
}

#[test]
fn root_trace_matches_applied_regret_updates_without_changing_training() {
    let game = BlueprintConfig { effective_stack_bb: 20.0, exact_preflop_averaging: true,
        averaging_delay: 0, ..BlueprintConfig::default() };
    let mut trainer = Trainer::fresh(game.clone());
    let snapshot = begin_update(&mut trainer).unwrap();
    let endpoints = snapshot.endpoints.keys().enumerate().map(|(i,h)|
        (h.clone(), [vec![i as f64/10.0;1326], vec![-(i as f64)/10.0;1326]])).collect();
    let expected_before = root_regrets(&trainer, &snapshot);
    let trace = root_trace(&trainer, &snapshot, &endpoints).unwrap();
    assert_eq!(expected_before, root_regrets(&trainer, &snapshot));
    apply_snapshot_updates(&mut trainer, &snapshot, &endpoints, false).unwrap();
    let after = root_regrets(&trainer, &snapshot);
    let mix: Vec<Vec<f64>> = serde_json::from_value(trace["probabilities"].clone()).unwrap();
    let q: Vec<Vec<f64>> = serde_json::from_value(trace["actionValuesBb"].clone()).unwrap();
    let weights: Vec<usize> = serde_json::from_value(trace["multiplicities"].clone()).unwrap();
    for c in 0..169 {
        let ev: f64 = (0..q.len()).map(|a| q[a][c]*mix[a][c]).sum();
        for a in 0..q.len() {
            let expected = (q[a][c]-ev)*weights[c] as f64/1326.0;
            assert!((after[a][c]-expected_before[a][c]-expected).abs()<1e-10);
        }
    }
    // On the other traverser's iteration, these root regrets are untouched.
    trainer.completed_iterations = 1;
    let _ = root_trace(&trainer, &snapshot, &endpoints).unwrap();
    apply_snapshot_updates(&mut trainer, &snapshot, &endpoints, false).unwrap();
    assert_eq!(after, root_regrets(&trainer, &snapshot));
}

#[test]
fn first_update_is_not_retroactively_discounted_when_sweep_creates_nodes() {
    let game = BlueprintConfig { effective_stack_bb: 20.0, exact_preflop_averaging: true,
        averaging_delay: 0, ..BlueprintConfig::default() };
    let mut trainer = Trainer::fresh(game);
    let first = begin_update(&mut trainer).unwrap();
    let endpoints = first.endpoints.keys().enumerate().map(|(i,h)|
        (h.clone(), [vec![i as f64/10.0;1326], vec![-(i as f64)/10.0;1326]])).collect();
    apply_snapshot_updates(&mut trainer, &first, &endpoints, false).unwrap();
    let before = root_regrets(&trainer, &first);
    assert!(before.iter().flatten().any(|v| *v>0.0));
    assert!(before.iter().flatten().any(|v| *v<0.0));
    trainer.completed_iterations = 1;
    let second = begin_update(&mut trainer).unwrap();
    let after = root_regrets(&trainer, &second);
    for (a,b) in before.iter().flatten().zip(after.iter().flatten()) {
        let factor = if *a>=0.0 { 2.0f64.powf(1.5)/(2.0f64.powf(1.5)+1.0) } else { 0.5 };
        assert!((b-a*factor).abs()<1e-12, "first update must receive round2 discount only");
    }
}
