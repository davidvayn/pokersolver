//! Offline exact-chance pilot. No neural leaves, independently solved street
//! policies, sampled hidden hands, serving activation or exported regrets.
use super::*;
use std::time::{Duration, Instant};

pub mod bundle;
mod check;
#[cfg(test)]
mod tests;
mod tree;
mod values;
use tree::{Game, Kind};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PilotLimits {
    pub max_decision_nodes: u64,
    pub max_seconds: u64,
}

struct Guard {
    deadline: Instant,
    visits: Cell<u64>,
}

impl Guard {
    fn tick(&self) -> Result<(), String> {
        let visits = self.visits.get() + 1;
        self.visits.set(visits);
        if visits % 4096 == 0 {
            self.check()?;
        }
        Ok(())
    }

    fn check(&self) -> Result<(), String> {
        if Instant::now() >= self.deadline {
            Err("exact-flop pilot time budget exceeded; no policy exported".into())
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PilotReport {
    pub schema: String,
    pub input: SpotInput,
    pub input_sha256: String,
    pub limits: PilotLimits,
    pub iterations_completed: u64,
    pub public_decision_nodes: usize,
    pub tree_nodes_including_chance_and_terminals: usize,
    pub board_contexts: usize,
    pub regret_average_array_bytes: usize,
    pub equity_cache_element_bytes: usize,
    pub frozen_policy_sha256: String,
    pub frozen_action_values_sha256: String,
    pub action_value_rows_checked: usize,
    pub action_values_checked: usize,
    pub rows_without_opponent_reach: usize,
    pub maximum_action_ev_error_bb: f64,
    pub profile_value_bb: [f64; 2],
    pub best_response_value_bb: [f64; 2],
    pub total_best_response_gain_bb: f64,
    pub maximum_probability_sum_error: f64,
    pub maximum_independent_profile_difference_bb: f64,
    pub zero_sum_residual_bb: f64,
    pub conditional_response_gain_gate_passed: bool,
    pub initialization_seconds: f64,
    pub training_seconds: f64,
    pub checking_seconds: f64,
    pub action_ev_validation_completed: bool,
    pub serving_artifact_produced: bool,
    pub full_hand_gto_claim: bool,
}

pub fn run(input: SpotInput, limits: PilotLimits) -> Result<PilotReport, String> {
    Ok(solve_checked(input, limits)?.0)
}

fn solve_checked(
    input: SpotInput,
    limits: PilotLimits,
) -> Result<(PilotReport, Game, Vec<Vec<f32>>, Vec<values::NodeValues>), String> {
    if input.board.len() != 3
        || !(20..=1024).contains(&input.iterations)
        || !(4..=500_000).contains(&limits.max_decision_nodes)
        || !(1..=600).contains(&limits.max_seconds)
    {
        return Err(
            "exact-flop pilot requires a flop, 20..1024 rounds, 4..500000 nodes and 1..600 seconds"
                .into(),
        );
    }
    let start = Instant::now();
    let guard = Guard {
        deadline: start + Duration::from_secs(limits.max_seconds),
        visits: Cell::new(0),
    };
    let config = config_for_audit(&input, true)?;
    let footprint = audit(&input)?;
    if footprint.public_decision_nodes > limits.max_decision_nodes
        || footprint.sparse_regret_average_array_bytes > 512 * 1024 * 1024
        || footprint.starting_combos.iter().any(|n| *n > 32)
    {
        return Err("exact-flop pilot structural budget exceeded before tree allocation".into());
    }
    let mut game = Game::new(&config, &guard)?;
    if game.decisions() as u64 != footprint.public_decision_nodes
        || game.array_bytes() as u64 != footprint.sparse_regret_average_array_bytes
    {
        return Err("exact-flop tree does not match independent structural census".into());
    }
    guard.check()?;
    let initialization_seconds = start.elapsed().as_secs_f64();
    let train_start = Instant::now();
    game.train(input.iterations, &config.game.dcfr, &guard)?;
    let training_seconds = train_start.elapsed().as_secs_f64();
    let checking_start = Instant::now();
    let frozen = game.freeze()?;
    let maximum_probability_sum_error = game.check_probabilities(&frozen)?;
    let (profile_value_bb, best_response_value_bb) = check::verify(&game, &frozen, &guard)?;
    let action_values = values::collect(&game, &frozen, &guard)?;
    let ev_check = check::verify_values(&game, &frozen, &action_values, &guard)?;
    let vector = [
        game.profile_value(&frozen, 0, &guard)?,
        game.profile_value(&frozen, 1, &guard)?,
    ];
    let difference = (0..2)
        .map(|p| (profile_value_bb[p] - vector[p]).abs())
        .fold(0.0, f64::max);
    let zero_sum = (profile_value_bb[0] + profile_value_bb[1]).abs();
    let gain = (0..2)
        .map(|p| best_response_value_bb[p] - profile_value_bb[p])
        .sum::<f64>();
    if vector.iter().any(|v| !v.is_finite())
        || difference > 1e-8
        || zero_sum > 1e-8
        || !gain.is_finite()
        || (0..2).any(|p| best_response_value_bb[p] + 1e-8 < profile_value_bb[p])
    {
        return Err("exact-flop independent policy verification disagreed".into());
    }
    let input_sha256 = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&input).map_err(|e| e.to_string())?)
    );
    let mut digest = Sha256::new();
    digest.update(b"exact-flop-pilot-f32-policy-v1");
    digest.update(input_sha256.as_bytes());
    for (id, row) in frozen.iter().enumerate() {
        digest.update((id as u64).to_le_bytes());
        digest.update((row.len() as u64).to_le_bytes());
        for probability in row {
            digest.update(probability.to_le_bytes());
        }
    }
    guard.check()?;
    let frozen_policy_sha256 = format!("{:x}", digest.finalize());
    let report = PilotReport {
        schema: "exact-flop-pilot-v2".into(),
        input_sha256,
        iterations_completed: input.iterations,
        input,
        limits,
        public_decision_nodes: game.decisions(),
        tree_nodes_including_chance_and_terminals: game.nodes.len(),
        board_contexts: game.boards.len(),
        regret_average_array_bytes: game.array_bytes(),
        equity_cache_element_bytes: game.boards.iter().map(|b| b.equity.len() * 8).sum(),
        frozen_action_values_sha256: values::fingerprint(&action_values, &frozen_policy_sha256),
        frozen_policy_sha256,
        action_value_rows_checked: ev_check.rows,
        action_values_checked: ev_check.actions,
        rows_without_opponent_reach: ev_check.unavailable_rows,
        maximum_action_ev_error_bb: ev_check.maximum_error,
        profile_value_bb,
        best_response_value_bb,
        total_best_response_gain_bb: gain,
        maximum_probability_sum_error,
        maximum_independent_profile_difference_bb: difference,
        zero_sum_residual_bb: zero_sum,
        conditional_response_gain_gate_passed: gain <= GAIN_LIMIT,
        initialization_seconds,
        training_seconds,
        checking_seconds: checking_start.elapsed().as_secs_f64(),
        action_ev_validation_completed: true,
        serving_artifact_produced: false,
        full_hand_gto_claim: false,
    };
    Ok((report, game, frozen, action_values))
}
