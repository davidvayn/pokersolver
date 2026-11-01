//! Exact structural census without allocating regrets, evaluating cards or
//! training a strategy. Betting rules do not depend on the unrevealed cards:
//! walk each action history once and count its ordered public chance copies.
use super::*;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StreetFootprint {
    pub street: Street,
    pub public_decision_nodes: u64,
    pub public_action_edges: u64,
    pub served_combo_rows: u64,
    pub served_action_values: u64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpotFootprint {
    pub schema: String,
    pub input_sha256: String,
    pub id: String,
    pub board: Vec<u8>,
    pub depth_bb: f64,
    pub pot_bb: f64,
    pub starting_combos: [u64; 2],
    pub compatible_private_deals: u64,
    pub ordered_runouts_per_private_deal: u64,
    pub compatible_deal_runouts: u64,
    pub streets: Vec<StreetFootprint>,
    pub public_decision_nodes: u64,
    pub public_action_edges: u64,
    pub public_chance_edges: u64,
    pub fold_leaves: u64,
    pub showdown_leaves_before_all_in_runout: u64,
    pub served_combo_rows: u64,
    pub served_action_values: u64,
    /// Dense reference has two COMBO_COUNT * actions arrays of f64. Does not
    /// include history strings/maps, rank layouts, caches or traversal scratch.
    pub dense_regret_average_bytes_lower_bound: u64,
    /// Positive-support-only arrays (implemented for saved turn studies).
    /// For flop this remains a projection, not an implemented full-flop solve.
    pub sparse_regret_average_array_bytes: u64,
    /// Hypothetical fixed layout: u16 combo id + two f32 arrays per row. JSON,
    /// node/action metadata, hash indexes and off-path null EVs are not modeled.
    pub sparse_full_value_payload_bytes: u64,
    pub action_history_states_walked: u64,
    pub maximum_action_depth: u64,
    pub current_reader_node_limit: u64,
    pub exceeds_current_reader_node_limit: bool,
    pub strategy_strength_evaluated: bool,
}

fn product(available: u64, draws: usize) -> Result<u64, String> {
    (0..draws).try_fold(1u64, |n, i| {
        n.checked_mul(
            available
                .checked_sub(i as u64)
                .ok_or("invalid chance count")?,
        )
        .ok_or_else(|| "footprint count overflow".into())
    })
}

fn add(target: &mut u64, amount: u64) -> Result<(), String> {
    *target = target
        .checked_add(amount)
        .ok_or("footprint count overflow")?;
    Ok(())
}

fn multiply(a: u64, b: u64) -> Result<u64, String> {
    a.checked_mul(b)
        .ok_or_else(|| "footprint count overflow".into())
}

struct Census<'a> {
    game: &'a BlueprintConfig,
    root_board_len: usize,
    combos: [u64; 2],
    report: SpotFootprint,
}

impl Census<'_> {
    fn walk(&mut self, state: GameState, depth: u64) -> Result<(), String> {
        add(&mut self.report.action_history_states_walked, 1)?;
        if self.report.action_history_states_walked > 1_000_000 {
            return Err("footprint action-history work limit exceeded".into());
        }
        self.report.maximum_action_depth = self.report.maximum_action_depth.max(depth);
        let drawn = state.street.board_len() - self.root_board_len;
        let copies = product(52 - self.root_board_len as u64, drawn)?;
        if let Some(terminal) = &state.terminal {
            match terminal {
                Terminal::Fold { .. } => add(&mut self.report.fold_leaves, copies)?,
                Terminal::Showdown => add(
                    &mut self.report.showdown_leaves_before_all_in_runout,
                    copies,
                )?,
            }
            return Ok(());
        }
        let actions = state.legal_actions(self.game);
        if actions.is_empty() {
            return Err("nonterminal state has no actions".into());
        }
        let action_count = actions.len() as u64;
        // Every root-legal private combo excludes exactly two future public
        // candidates. Count rows across ordered boards, without clairvoyance.
        let rows = multiply(
            self.combos[state.actor],
            product(50 - self.root_board_len as u64, drawn)?,
        )?;
        let cells = multiply(rows, action_count)?;
        let edges = multiply(copies, action_count)?;
        let i = state.street.board_len() - 3;
        add(&mut self.report.streets[i].public_decision_nodes, copies)?;
        add(&mut self.report.streets[i].public_action_edges, edges)?;
        add(&mut self.report.streets[i].served_combo_rows, rows)?;
        add(&mut self.report.streets[i].served_action_values, cells)?;
        add(&mut self.report.public_decision_nodes, copies)?;
        add(&mut self.report.public_action_edges, edges)?;
        add(&mut self.report.served_combo_rows, rows)?;
        add(&mut self.report.served_action_values, cells)?;
        for action in actions {
            let next = state.apply(&action, self.game);
            if next.terminal.is_none() && next.street != state.street {
                add(
                    &mut self.report.public_chance_edges,
                    multiply(copies, 52 - state.street.board_len() as u64)?,
                )?;
            }
            self.walk(next, depth + 1)?;
        }
        Ok(())
    }
}

pub fn audit(input: &SpotInput) -> Result<SpotFootprint, String> {
    let config = config_for_audit(input, true)?;
    let ranges = sparse_ranges(&config.state.ranges);
    let combos = all_combos();
    let compatible_private_deals = ranges[0]
        .iter()
        .map(|(a, _)| {
            ranges[1]
                .iter()
                .filter(|(b, _)| {
                    !combos[*a]
                        .cards()
                        .iter()
                        .any(|c| combos[*b].cards().contains(c))
                })
                .count() as u64
        })
        .sum::<u64>();
    if compatible_private_deals == 0 {
        return Err("no compatible private deals".into());
    }
    let ordered_runouts_per_private_deal =
        product(48 - input.board.len() as u64, 5 - input.board.len())?;
    let counts = ranges.each_ref().map(|range| range.len() as u64);
    let report = SpotFootprint {
        schema: "solved-spot-footprint-v1".into(),
        input_sha256: format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(input).map_err(|e| e.to_string())?)
        ),
        id: input.id.clone(),
        board: input.board.clone(),
        depth_bb: input.depth_bb,
        pot_bb: input.pot_bb,
        starting_combos: counts,
        compatible_private_deals,
        ordered_runouts_per_private_deal,
        compatible_deal_runouts: multiply(
            compatible_private_deals,
            ordered_runouts_per_private_deal,
        )?,
        streets: [Street::Flop, Street::Turn, Street::River]
            .into_iter()
            .map(|street| StreetFootprint {
                street,
                public_decision_nodes: 0,
                public_action_edges: 0,
                served_combo_rows: 0,
                served_action_values: 0,
            })
            .collect(),
        public_decision_nodes: 0,
        public_action_edges: 0,
        public_chance_edges: 0,
        fold_leaves: 0,
        showdown_leaves_before_all_in_runout: 0,
        served_combo_rows: 0,
        served_action_values: 0,
        dense_regret_average_bytes_lower_bound: 0,
        sparse_regret_average_array_bytes: 0,
        sparse_full_value_payload_bytes: 0,
        action_history_states_walked: 0,
        maximum_action_depth: 0,
        current_reader_node_limit: 4096,
        exceeds_current_reader_node_limit: false,
        strategy_strength_evaluated: false,
    };
    let mut census = Census {
        game: &config.game,
        root_board_len: input.board.len(),
        combos: counts,
        report,
    };
    census.walk(config.state.game_state(), 0)?;
    let r = &mut census.report;
    r.dense_regret_average_bytes_lower_bound =
        multiply(multiply(r.public_action_edges, COMBO_COUNT as u64)?, 16)?;
    r.sparse_regret_average_array_bytes = multiply(r.served_action_values, 16)?;
    r.sparse_full_value_payload_bytes = multiply(r.served_action_values, 8)?
        .checked_add(multiply(r.served_combo_rows, 2)?)
        .ok_or("footprint count overflow")?;
    r.exceeds_current_reader_node_limit = r.public_decision_nodes > r.current_reader_node_limit;
    Ok(census.report)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn census_matches_every_published_river_and_turn_artifact() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let catalog: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("data/solved-spots/catalog.json")).unwrap(),
        )
        .unwrap();
        for entry in catalog.as_array().unwrap() {
            let path = root
                .join("public")
                .join(entry["url"].as_str().unwrap().trim_start_matches('/'));
            let spot: SolvedSpot = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
            let r = audit(&spot.input).unwrap();
            assert_eq!(
                r.public_decision_nodes,
                spot.nodes.len() as u64,
                "{}",
                spot.input.id
            );
            assert_eq!(
                r.served_combo_rows,
                spot.nodes.iter().map(|n| n.rows.len() as u64).sum::<u64>()
            );
            assert_eq!(
                r.served_action_values,
                spot.nodes
                    .iter()
                    .map(|n| (n.rows.len() * n.actions.len()) as u64)
                    .sum::<u64>()
            );
            assert_eq!(
                r.public_action_edges,
                spot.nodes
                    .iter()
                    .map(|n| n.actions.len() as u64)
                    .sum::<u64>()
            );
            assert_eq!(
                r.public_chance_edges,
                spot.nodes
                    .iter()
                    .flat_map(|n| &n.actions)
                    .filter_map(|a| a.next_by_river.as_ref())
                    .map(|e| e.len() as u64)
                    .sum::<u64>()
            );
            assert_eq!(
                r.compatible_private_deals,
                spot.verification.compatible_deals as u64
            );
            assert!(!r.strategy_strength_evaluated);
        }
    }
    #[test]
    fn flop_audit_never_enables_flop_serving_or_trains_a_policy() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let mut input: SpotInput = serde_json::from_slice(
            &std::fs::read(root.join("data/solved-spots/river-50bb-paired.json")).unwrap(),
        )
        .unwrap();
        input.board.truncate(3);
        input.pot_bb = 12.0;
        assert!(solve(input.clone()).is_err());
        let r = audit(&input).unwrap();
        assert_eq!(r.ordered_runouts_per_private_deal, 45 * 44);
        assert!(r.public_decision_nodes > 4096);
        assert!(r.exceeds_current_reader_node_limit);
        assert!(r.action_history_states_walked < 1_000_000);
        assert!(r.dense_regret_average_bytes_lower_bound > r.sparse_regret_average_array_bytes);
        assert_eq!(
            serde_json::to_vec(&r).unwrap(),
            serde_json::to_vec(&audit(&input).unwrap()).unwrap()
        );
        input.board[1] = input.board[0];
        assert!(audit(&input).is_err());
    }
}
