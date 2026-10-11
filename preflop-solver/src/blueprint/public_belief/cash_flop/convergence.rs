//! Bounded search-budget diagnosis with frozen critics and average-policy
//! checkpoints. No regret state is exported and no cash model is activated.
use super::*;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CashFlopConvergenceInput {
    pub solve: CashFlopPilotInput,
    pub checkpoint_rounds: Vec<u64>,
}

impl CashFlopConvergenceInput {
    pub fn validate(&self) -> Result<(), String> {
        self.solve.validate()?;
        if self.checkpoint_rounds.is_empty()
            || self.checkpoint_rounds.len() > 4
            || self.checkpoint_rounds.last() != Some(&self.solve.iterations)
            || self.checkpoint_rounds.iter().any(|round| {
                *round < 2
                    || *round > self.solve.iterations
                    || *round <= self.solve.averaging_delay
            })
            || self.checkpoint_rounds.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err("cash convergence needs 1..4 sorted checkpoints ending at the bounded solve budget, each after averaging starts".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CashFlopConvergenceCheckpoint {
    pub round: u64,
    pub solution: CashFlopPilotSolution,
}

#[derive(Clone, Debug, Serialize)]
pub struct CashFlopConvergenceReport {
    pub schema: String,
    pub status: String,
    pub active: bool,
    pub input: CashFlopConvergenceInput,
    pub input_sha256: String,
    pub value_network_sha256: String,
    pub updates_performed: u64,
    pub checkpoints: Vec<CashFlopConvergenceCheckpoint>,
    pub full_game_exploitability: String,
    pub limitations: Vec<String>,
}

pub fn solve(
    input: CashFlopConvergenceInput,
    network: PublicValueNetwork,
) -> Result<CashFlopConvergenceReport, String> {
    input.validate()?;
    let (mut solver, _) = prepare_solver(input.solve.clone(), network)?;
    let network_sha = solver
        .config
        .value_network
        .artifact_sha256()
        .ok_or("cash convergence requires immutable hashed weights")?
        .to_owned();
    let input_sha256 = format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(&(
                "hu-cash-flop-convergence-pilot-v1",
                &input,
                &network_sha,
            ))
            .map_err(|e| e.to_string())?
        )
    );
    let mut previous = 0;
    let mut checkpoints = Vec::with_capacity(input.checkpoint_rounds.len());
    for &round in &input.checkpoint_rounds {
        // Re-enter the existing trainer without resetting nodes or DCFR time.
        // Tree preparation uses entry/or_insert; discounts retain global round.
        solver.training_round_offset = previous;
        solver.config.iterations = round - previous;
        solver.train_frozen_public_chance_pairs()?;
        solver.config.iterations = round;
        let mut checkpoint_input = input.solve.clone();
        checkpoint_input.iterations = round;
        let checkpoint_sha = format!(
            "{:x}",
            Sha256::digest(
                serde_json::to_vec(&(
                    SCHEMA,
                    &checkpoint_input,
                    Some(network_sha.as_str()),
                ))
                .map_err(|e| e.to_string())?
            )
        );
        let solution = solver.finish_cash_flop(checkpoint_sha)?;
        checkpoints.push(CashFlopConvergenceCheckpoint { round, solution });
        previous = round;
    }
    Ok(CashFlopConvergenceReport {
        schema: "hu-cash-flop-convergence-pilot-v1".into(),
        status: "research_only".into(),
        active: false,
        value_network_sha256: network_sha,
        updates_performed: previous,
        input,
        input_sha256,
        checkpoints,
        full_game_exploitability: "unmeasured".into(),
        limitations: vec![
            "Only average policies and critic-dependent diagnostics are exported, not regrets or resumable training state.".into(),
            "Checkpoint scoring adds value-query work but no regret/average updates; query counters include that work.".into(),
            "Frozen learned continuations retain bias. General-sum cash convergence and full-game exploitability are not certified.".into(),
            "This offline budget extension is not a serving mode, qualified model, or cash activation.".into(),
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_preserve_prefix_and_do_not_reset_discount_or_averages() {
        let (mut input, network) = super::super::tests::fixture();
        input.iterations = 4;
        let direct = super::super::solve(input.clone(), network.clone()).unwrap();
        let mut prefix_input = input.clone();
        prefix_input.iterations = 2;
        let prefix = super::super::solve(prefix_input, network.clone()).unwrap();
        let report = solve(
            CashFlopConvergenceInput {
                solve: input,
                checkpoint_rounds: vec![2, 4],
            },
            network,
        )
        .unwrap();
        assert_eq!(report.updates_performed, 4);
        assert!(!report.active);
        assert_eq!(
            serde_json::to_vec(&report.checkpoints[0].solution).unwrap(),
            serde_json::to_vec(&prefix).unwrap()
        );
        let final_solution = &report.checkpoints[1].solution;
        assert_eq!(final_solution.input_sha256, direct.input_sha256);
        assert_eq!(
            serde_json::to_vec(&final_solution.strategies).unwrap(),
            serde_json::to_vec(&direct.strategies).unwrap()
        );
        assert_eq!(final_solution.profile_net_bb, direct.profile_net_bb);
        assert_eq!(final_solution.iterations, 4);
    }

    #[test]
    fn checkpoints_reject_invalid_budget_order_and_averaging_windows() {
        let (mut input, _) = super::super::tests::fixture();
        input.iterations = 8;
        for rounds in [
            vec![],
            vec![1, 8],
            vec![4],
            vec![4, 4, 8],
            vec![8, 4],
            vec![2, 4, 6, 8, 10],
            vec![2, 9],
        ] {
            assert!(CashFlopConvergenceInput {
                solve: input.clone(),
                checkpoint_rounds: rounds,
            }
            .validate()
            .is_err());
        }
        input.averaging_delay = 4;
        assert!(CashFlopConvergenceInput {
            solve: input,
            checkpoint_rounds: vec![4, 8],
        }
        .validate()
        .is_err());
        assert!(serde_json::from_value::<CashFlopConvergenceInput>(serde_json::json!({
            "solve": super::super::tests::fixture().0,
            "checkpoint_rounds": [2],
            "resume_regrets": true,
        }))
        .is_err());
    }
}
