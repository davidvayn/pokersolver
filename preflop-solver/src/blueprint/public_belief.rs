//! Exact-card-removal public-belief solving for late-street subgames.
//!
//! The full HUNL game is too large to solve tabularly. River public subgames,
//! however, have no remaining chance events. This module keeps one probability
//! for every exact two-card combination, solves the configured river betting
//! abstraction with alternating CFR, and exports per-combination
//! counterfactual values. No private cards are sampled by the solver.

use super::neural::FrozenPolicy;
use super::*;
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::path::PathBuf;

pub const COMBO_COUNT: usize = 1_326;
const RIVER_SCHEMA: &str = "hu-river-public-belief-solution-v1";

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PublicBeliefState {
    pub street: Street,
    pub board: Vec<u8>,
    pub actor: usize,
    pub invested_bb: [f64; 2],
    pub street_invested_bb: [f64; 2],
    pub last_full_raise_bb: f64,
    pub aggressions: u8,
    pub checks: u8,
    pub raise_reopened: bool,
    pub public_history: Vec<String>,
    pub ranges: [Vec<f64>; 2],
}

impl PublicBeliefState {
    pub fn flop_start(
        board: [u8; 3],
        actor: usize,
        invested_bb: [f64; 2],
        ranges: [Vec<f64>; 2],
    ) -> Self {
        Self {
            street: Street::Flop,
            board: board.to_vec(),
            actor,
            invested_bb,
            street_invested_bb: [0.0, 0.0],
            last_full_raise_bb: 1.0,
            aggressions: 0,
            checks: 0,
            raise_reopened: true,
            public_history: vec!["public_belief:flop_start".to_owned()],
            ranges,
        }
    }

    pub fn river_start(
        board: [u8; 5],
        actor: usize,
        invested_bb: [f64; 2],
        ranges: [Vec<f64>; 2],
    ) -> Self {
        Self {
            street: Street::River,
            board: board.to_vec(),
            actor,
            invested_bb,
            street_invested_bb: [0.0, 0.0],
            last_full_raise_bb: 1.0,
            aggressions: 0,
            checks: 0,
            raise_reopened: true,
            public_history: vec!["public_belief:river_start".to_owned()],
            ranges,
        }
    }

    pub fn uniform_river_start(board: [u8; 5], actor: usize, invested_bb: [f64; 2]) -> Self {
        let ranges = std::array::from_fn(|_| uniform_range(&board));
        Self::river_start(board, actor, invested_bb, ranges)
    }

    fn validate_and_normalize(&self, game: &BlueprintConfig) -> Result<Self, String> {
        self.validate_street_and_normalize(game, Street::River, 5)
    }

    fn validate_street_and_normalize(
        &self,
        game: &BlueprintConfig,
        street: Street,
        board_len: usize,
    ) -> Result<Self, String> {
        if self.street != street || self.board.len() != board_len {
            return Err(format!(
                "public-belief state must be {street:?} with {board_len} board cards"
            ));
        }
        if self.actor > 1 {
            return Err("public-belief actor must be player zero or one".to_owned());
        }
        let unique = self.board.iter().copied().collect::<BTreeSet<_>>();
        if unique.len() != self.board.len() || self.board.iter().any(|card| *card >= 52) {
            return Err("public board must contain five unique cards".to_owned());
        }
        if self
            .invested_bb
            .iter()
            .chain(&self.street_invested_bb)
            .any(|value| !value.is_finite() || *value < 0.0 || *value > game.effective_stack_bb)
        {
            return Err("public-belief commitments must be finite and within the stack".to_owned());
        }
        if !self.last_full_raise_bb.is_finite() || self.last_full_raise_bb <= 0.0 {
            return Err("last full raise must be positive".to_owned());
        }
        let mut normalized = self.clone();
        for player in 0..2 {
            if normalized.ranges[player].len() != COMBO_COUNT {
                return Err(format!(
                    "player {player} range must contain {COMBO_COUNT} exact combinations"
                ));
            }
            let mut total = 0.0;
            for combo in all_combos() {
                let weight = &mut normalized.ranges[player][combo.key()];
                if !weight.is_finite() || *weight < 0.0 {
                    return Err(format!(
                        "player {player} range contains a non-finite or negative weight"
                    ));
                }
                if combo.cards().iter().any(|card| unique.contains(card)) {
                    *weight = 0.0;
                }
                total += *weight;
            }
            if total <= EPSILON {
                return Err(format!(
                    "player {player} range has no legal river combinations"
                ));
            }
            for weight in &mut normalized.ranges[player] {
                *weight /= total;
            }
        }
        let joint_mass = joint_compatibility_mass(&normalized.ranges);
        if joint_mass <= EPSILON {
            return Err("the two ranges have no mutually compatible deals".to_owned());
        }
        Ok(normalized)
    }

    fn game_state(&self) -> GameState {
        GameState {
            street: self.street,
            actor: self.actor,
            invested: self.invested_bb,
            street_invested: self.street_invested_bb,
            last_full_raise: self.last_full_raise_bb,
            aggressions: self.aggressions,
            checks: self.checks,
            raise_reopened: self.raise_reopened,
            public_history: self.public_history.clone(),
            trajectory: Vec::new(),
            terminal: None,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ValueNetworkLayer {
    input_size: usize,
    output_size: usize,
    activation: String,
    weights: Vec<f32>,
    biases: Vec<f32>,
}

impl ValueNetworkLayer {
    fn validate(&self, expected_input: usize) -> Result<usize, String> {
        if self.input_size != expected_input
            || self.output_size == 0
            || self.weights.len() != self.input_size * self.output_size
            || self.biases.len() != self.output_size
            || self
                .weights
                .iter()
                .chain(&self.biases)
                .any(|value| !value.is_finite())
            || !matches!(self.activation.as_str(), "relu" | "linear" | "tanh")
        {
            return Err("public value network contains an invalid dense layer".to_owned());
        }
        Ok(self.output_size)
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        debug_assert_eq!(input.len(), self.input_size);
        let mut output = self.biases.clone();
        for (out, row) in output
            .iter_mut()
            .zip(self.weights.chunks_exact(self.input_size))
        {
            *out += row
                .iter()
                .zip(input)
                .map(|(weight, value)| weight * value)
                .sum::<f32>();
            *out = match self.activation.as_str() {
                "relu" => out.max(0.0),
                "tanh" => out.tanh(),
                _ => *out,
            };
        }
        output
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PublicValueNetwork {
    schema: String,
    seed: u64,
    uses_exact_ranges: bool,
    target_scale_bb: f64,
    range_scale: f64,
    #[serde(default)]
    source_dataset_sha256: Option<String>,
    #[serde(default)]
    source_validation_status: Option<String>,
    public_tower: Vec<ValueNetworkLayer>,
    range_tower: Vec<ValueNetworkLayer>,
    head: Vec<ValueNetworkLayer>,
}

impl PublicValueNetwork {
    pub fn read(path: &Path) -> Result<Self, Box<dyn Error>> {
        let network: Self = serde_json::from_slice(&fs::read(path)?)?;
        network.validate()?;
        Ok(network)
    }

    fn validate(&self) -> Result<(), String> {
        if self.schema != "hu-public-belief-value-network-v2"
            || !self.target_scale_bb.is_finite()
            || self.target_scale_bb <= 0.0
            || !self.range_scale.is_finite()
            || self.range_scale <= 0.0
            || self.public_tower.is_empty()
            || self.range_tower.is_empty()
            || self.head.is_empty()
        {
            return Err("public value network header is incompatible".to_owned());
        }
        let mut public_size = 56;
        for layer in &self.public_tower {
            public_size = layer.validate(public_size)?;
        }
        let mut range_size = COMBO_COUNT * 2;
        for layer in &self.range_tower {
            range_size = layer.validate(range_size)?;
        }
        let mut head_size = public_size + range_size;
        for layer in &self.head {
            head_size = layer.validate(head_size)?;
        }
        if head_size != COMBO_COUNT * 2 {
            return Err("public value network must output two exact-combo CFV vectors".to_owned());
        }
        Ok(())
    }

    fn predict(
        &self,
        board: &[u8],
        actor: usize,
        invested: [f64; 2],
        ranges: &[Vec<f64>; 2],
    ) -> [Vec<f64>; 2] {
        let mut range_features = ranges
            .iter()
            .flatten()
            .map(|weight| (*weight * self.range_scale) as f32)
            .collect::<Vec<_>>();
        if !self.uses_exact_ranges {
            range_features.fill(0.0);
        }
        let range_embedding = self
            .range_tower
            .iter()
            .fold(range_features, |values, layer| layer.forward(&values));
        let mut public_features = vec![0.0f32; 52];
        for card in board {
            public_features[*card as usize] = 1.0;
        }
        public_features.extend([
            if actor == 0 { 1.0 } else { 0.0 },
            if actor == 1 { 1.0 } else { 0.0 },
        ]);
        public_features.extend([
            (invested[0] / self.target_scale_bb) as f32,
            (invested[1] / self.target_scale_bb) as f32,
        ]);
        let public_embedding = self
            .public_tower
            .iter()
            .fold(public_features, |values, layer| layer.forward(&values));
        let mut head = public_embedding;
        head.extend(range_embedding);
        let output = self
            .head
            .iter()
            .fold(head, |values, layer| layer.forward(&values));
        std::array::from_fn(|player| {
            output[player * COMBO_COUNT..(player + 1) * COMBO_COUNT]
                .iter()
                .enumerate()
                .map(|(combo, value)| {
                    if ranges[player][combo] > 0.0 {
                        *value as f64 * self.target_scale_bb
                    } else {
                        0.0
                    }
                })
                .collect()
        })
    }
}

#[derive(Clone, Debug)]
pub struct FlopResolveConfig {
    pub game: BlueprintConfig,
    pub state: PublicBeliefState,
    pub iterations: u64,
    pub averaging_delay: u64,
    pub value_network: PublicValueNetwork,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FlopResolveMetrics {
    pub information_sets: usize,
    pub turn_leaf_evaluations: u64,
    pub profile_value_p0_bb: f64,
    pub profile_value_p1_bb: f64,
    pub depth_limited_best_response_value_p0_bb: f64,
    pub depth_limited_best_response_value_p1_bb: f64,
    pub depth_limited_exploitability_bb_per_hand: f64,
    pub unresolved_uniform_exploitability_bb_per_hand: f64,
    pub resolver_relative_exploitability_improvement: f64,
    pub maximum_leaf_zero_sum_residual_before_projection_bb: f64,
    pub zero_sum_residual_after_projection_bb: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FlopSolution {
    pub schema: String,
    pub method: String,
    pub approximate: bool,
    pub value_network_seed: u64,
    pub uses_exact_ranges: bool,
    pub value_network_source_dataset_sha256: Option<String>,
    pub state: PublicBeliefState,
    pub iterations: u64,
    pub strategies: Vec<PublicBeliefStrategy>,
    pub counterfactual_values_bb: [Vec<f32>; 2],
    pub opponent_compatible_mass: [Vec<f32>; 2],
    pub metrics: FlopResolveMetrics,
    pub validation: BlueprintValidation,
}

struct FlopSolver {
    config: FlopResolveConfig,
    legal: [Vec<bool>; 2],
    conflicts: Vec<Vec<usize>>,
    nodes: BTreeMap<Vec<String>, RangeNode>,
    turn_leaf_evaluations: Cell<u64>,
    maximum_leaf_zero_sum_residual: Cell<f64>,
}

impl FlopSolver {
    fn new(mut config: FlopResolveConfig) -> Result<Self, String> {
        config.game.validate()?;
        config.value_network.validate()?;
        if config.game.action_abstraction.include_all_in {
            return Err(
                "the pilot flop resolver requires include_all_in=false until exact flop all-in runouts are vectorized"
                    .to_owned(),
            );
        }
        if config.iterations < 2 || config.averaging_delay >= config.iterations {
            return Err(
                "flop resolving requires alternating iterations and a valid averaging delay"
                    .to_owned(),
            );
        }
        config.state = config
            .state
            .validate_street_and_normalize(&config.game, Street::Flop, 3)?;
        let legal = std::array::from_fn(|player| {
            config.state.ranges[player]
                .iter()
                .map(|weight| *weight > 0.0)
                .collect()
        });
        Ok(Self {
            config,
            legal,
            conflicts: combo_conflicts(),
            nodes: BTreeMap::new(),
            turn_leaf_evaluations: Cell::new(0),
            maximum_leaf_zero_sum_residual: Cell::new(0.0),
        })
    }

    fn train(&mut self) {
        let root = self.config.state.game_state();
        let reaches = self.config.state.ranges.clone();
        for offset in 0..self.config.iterations {
            self.walk(
                root.clone(),
                reaches.clone(),
                offset as usize % 2,
                offset + 1,
            );
        }
    }

    fn walk(
        &mut self,
        state: GameState,
        reaches: [Vec<f64>; 2],
        traverser: usize,
        iteration: u64,
    ) -> [Vec<f64>; 2] {
        if state.street == Street::Turn && state.terminal.is_none() {
            return self.turn_leaf_values(&state, &reaches);
        }
        if state.terminal.is_some() {
            return self.fold_terminal_values(&state, &reaches);
        }
        let actions = state.legal_actions(&self.config.game);
        let key = state.public_history.clone();
        let actor = state.actor;
        let strategy = {
            let node = self
                .nodes
                .entry(key.clone())
                .or_insert_with(|| RangeNode::new(actor, &actions));
            node.discount(iteration, &self.config.game.dcfr);
            node.strategy(&self.legal[actor])
        };
        let action_count = actions.len();
        let mut children = Vec::with_capacity(action_count);
        for (action_index, action) in actions.iter().enumerate() {
            let mut child_reaches = reaches.clone();
            for combo in 0..COMBO_COUNT {
                child_reaches[actor][combo] *= strategy[combo * action_count + action_index];
            }
            children.push(self.walk(
                state.apply(action, &self.config.game),
                child_reaches,
                traverser,
                iteration,
            ));
        }
        let opponent = 1 - actor;
        let mut values = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        for combo in 0..COMBO_COUNT {
            for action in 0..action_count {
                values[actor][combo] +=
                    strategy[combo * action_count + action] * children[action][actor][combo];
                values[opponent][combo] += children[action][opponent][combo];
            }
        }
        let node = self.nodes.get_mut(&key).expect("flop range node inserted");
        if actor == traverser {
            for combo in 0..COMBO_COUNT {
                if !self.legal[actor][combo] {
                    continue;
                }
                let offset = combo * action_count;
                for action in 0..action_count {
                    node.regrets[offset + action] +=
                        children[action][actor][combo] - values[actor][combo];
                }
            }
        }
        if iteration > self.config.averaging_delay {
            for combo in 0..COMBO_COUNT {
                let offset = combo * action_count;
                for action in 0..action_count {
                    node.strategy_sum[offset + action] +=
                        reaches[actor][combo] * strategy[offset + action];
                }
            }
        }
        values
    }

    fn fold_terminal_values(&self, state: &GameState, reaches: &[Vec<f64>; 2]) -> [Vec<f64>; 2] {
        let Terminal::Fold { winner } = state.terminal.as_ref().expect("terminal") else {
            unreachable!("all-in actions are disabled in the flop pilot")
        };
        let utility_p0 = if *winner == 0 {
            state.invested[1]
        } else {
            -state.invested[0]
        };
        std::array::from_fn(|player| {
            let utility = if player == 0 { utility_p0 } else { -utility_p0 };
            (0..COMBO_COUNT)
                .map(|combo| {
                    utility
                        * compatible_mass_from_conflicts(
                            &reaches[1 - player],
                            &self.conflicts,
                            combo,
                        )
                })
                .collect()
        })
    }

    fn turn_leaf_values(&self, state: &GameState, reaches: &[Vec<f64>; 2]) -> [Vec<f64>; 2] {
        self.turn_leaf_evaluations
            .set(self.turn_leaf_evaluations.get() + 1);
        let mut result = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        for turn in 0..52u8 {
            if self.config.state.board.contains(&turn) {
                continue;
            }
            let mut masked = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
            let mut totals = [0.0; 2];
            for player in 0..2 {
                for combo in all_combos() {
                    let weight = if combo.cards().contains(&turn) {
                        0.0
                    } else {
                        reaches[player][combo.key()]
                    };
                    masked[player][combo.key()] = weight;
                    totals[player] += weight;
                }
                if totals[player] <= EPSILON {
                    continue;
                }
                for weight in &mut masked[player] {
                    *weight /= totals[player];
                }
            }
            if totals.iter().any(|total| *total <= EPSILON) {
                continue;
            }
            let mut board = self.config.state.board.clone();
            board.push(turn);
            let mut predicted =
                self.config
                    .value_network
                    .predict(&board, state.actor, state.invested, &masked);
            let masses: [Vec<f64>; 2] = std::array::from_fn(|player| {
                (0..COMBO_COUNT)
                    .map(|combo| {
                        compatible_mass_from_conflicts(&masked[1 - player], &self.conflicts, combo)
                    })
                    .collect()
            });
            let joint = joint_compatibility_mass(&masked);
            let aggregate = |player: usize| {
                masked[player]
                    .iter()
                    .zip(&predicted[player])
                    .zip(&masses[player])
                    .map(|((reach, value), mass)| reach * value * mass)
                    .sum::<f64>()
                    / joint.max(EPSILON)
            };
            let residual = aggregate(0) + aggregate(1);
            self.maximum_leaf_zero_sum_residual.set(
                self.maximum_leaf_zero_sum_residual
                    .get()
                    .max(residual.abs()),
            );
            for values in &mut predicted {
                for value in values {
                    *value -= residual / 2.0;
                }
            }
            for player in 0..2 {
                for combo in 0..COMBO_COUNT {
                    result[player][combo] +=
                        predicted[player][combo] * masses[player][combo] * totals[1 - player]
                            / 45.0;
                }
            }
        }
        result
    }

    fn profile_walk(
        &self,
        state: GameState,
        reaches: [Vec<f64>; 2],
        best_responder: Option<usize>,
        unresolved_uniform: bool,
    ) -> [Vec<f64>; 2] {
        if state.street == Street::Turn && state.terminal.is_none() {
            return self.turn_leaf_values(&state, &reaches);
        }
        if state.terminal.is_some() {
            return self.fold_terminal_values(&state, &reaches);
        }
        let actions = state.legal_actions(&self.config.game);
        let actor = state.actor;
        let action_count = actions.len();
        let strategy = if unresolved_uniform {
            let mut values = vec![0.0; COMBO_COUNT * action_count];
            for combo in 0..COMBO_COUNT {
                if self.legal[actor][combo] {
                    values[combo * action_count..(combo + 1) * action_count]
                        .fill(1.0 / action_count as f64);
                }
            }
            values
        } else {
            self.nodes
                .get(&state.public_history)
                .expect("trained flop node")
                .average_strategy(&self.legal[actor])
        };
        let mut children = Vec::with_capacity(action_count);
        for (action_index, action) in actions.iter().enumerate() {
            let mut child_reaches = reaches.clone();
            if best_responder != Some(actor) {
                for combo in 0..COMBO_COUNT {
                    child_reaches[actor][combo] *= strategy[combo * action_count + action_index];
                }
            }
            children.push(self.profile_walk(
                state.apply(action, &self.config.game),
                child_reaches,
                best_responder,
                unresolved_uniform,
            ));
        }
        let opponent = 1 - actor;
        let mut values = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        for combo in 0..COMBO_COUNT {
            if best_responder == Some(actor) {
                values[actor][combo] = children
                    .iter()
                    .map(|child| child[actor][combo])
                    .fold(f64::NEG_INFINITY, f64::max);
                for child in &children {
                    values[opponent][combo] += child[opponent][combo];
                }
            } else {
                for action in 0..action_count {
                    values[actor][combo] +=
                        strategy[combo * action_count + action] * children[action][actor][combo];
                    values[opponent][combo] += children[action][opponent][combo];
                }
            }
        }
        values
    }

    fn finish(self) -> FlopSolution {
        let root = self.config.state.game_state();
        let reaches = self.config.state.ranges.clone();
        let joint = joint_compatibility_mass(&reaches);
        let mut profile = self.profile_walk(root.clone(), reaches.clone(), None, false);
        let br0 = self.profile_walk(root.clone(), reaches.clone(), Some(0), false);
        let br1 = self.profile_walk(root.clone(), reaches.clone(), Some(1), false);
        let unresolved_profile = self.profile_walk(root.clone(), reaches.clone(), None, true);
        let unresolved_br0 = self.profile_walk(root.clone(), reaches.clone(), Some(0), true);
        let unresolved_br1 = self.profile_walk(root, reaches.clone(), Some(1), true);
        let aggregate = |values: &[f64], player: usize| {
            reaches[player]
                .iter()
                .zip(values)
                .map(|(reach, value)| reach * value)
                .sum::<f64>()
                / joint
        };
        let unprojected_profile_p0 = aggregate(&profile[0], 0);
        let unprojected_profile_p1 = aggregate(&profile[1], 1);
        let root_residual = unprojected_profile_p0 + unprojected_profile_p1;
        for player in 0..2 {
            for combo in 0..COMBO_COUNT {
                let mass =
                    compatible_mass_from_conflicts(&reaches[1 - player], &self.conflicts, combo);
                profile[player][combo] -= root_residual / 2.0 * mass;
            }
        }
        let profile_p0 = aggregate(&profile[0], 0);
        let profile_p1 = aggregate(&profile[1], 1);
        let best_p0 = aggregate(&br0[0], 0);
        let best_p1 = aggregate(&br1[1], 1);
        let exploitability =
            ((best_p0 - unprojected_profile_p0) + (best_p1 - unprojected_profile_p1)) / 2.0;
        let unresolved_p0 = aggregate(&unresolved_profile[0], 0);
        let unresolved_p1 = aggregate(&unresolved_profile[1], 1);
        let unresolved_best_p0 = aggregate(&unresolved_br0[0], 0);
        let unresolved_best_p1 = aggregate(&unresolved_br1[1], 1);
        let unresolved_exploitability =
            ((unresolved_best_p0 - unresolved_p0) + (unresolved_best_p1 - unresolved_p1)) / 2.0;
        let strategies = self
            .nodes
            .iter()
            .map(|(history, node)| PublicBeliefStrategy {
                public_history: history.clone(),
                actor: node.actor,
                action_labels: node.action_labels.clone(),
                probabilities: node
                    .average_strategy(&self.legal[node.actor])
                    .into_iter()
                    .map(|value| value as f32)
                    .collect(),
            })
            .collect::<Vec<_>>();
        let opponent_compatible_mass: [Vec<f32>; 2] = std::array::from_fn(|player| {
            (0..COMBO_COUNT)
                .map(|combo| {
                    compatible_mass_from_conflicts(&reaches[1 - player], &self.conflicts, combo)
                        as f32
                })
                .collect()
        });
        let counterfactual_values_bb = std::array::from_fn(|player| {
            profile[player]
                .iter()
                .zip(&opponent_compatible_mass[player])
                .map(|(value, mass)| {
                    if *mass > 0.0 {
                        (*value / *mass as f64) as f32
                    } else {
                        0.0
                    }
                })
                .collect()
        });
        let metrics = FlopResolveMetrics {
            information_sets: strategies.len(),
            turn_leaf_evaluations: self.turn_leaf_evaluations.get(),
            profile_value_p0_bb: profile_p0,
            profile_value_p1_bb: profile_p1,
            depth_limited_best_response_value_p0_bb: best_p0,
            depth_limited_best_response_value_p1_bb: best_p1,
            depth_limited_exploitability_bb_per_hand: exploitability.max(0.0),
            unresolved_uniform_exploitability_bb_per_hand: unresolved_exploitability.max(0.0),
            resolver_relative_exploitability_improvement: (unresolved_exploitability
                - exploitability)
                / unresolved_exploitability.max(EPSILON),
            maximum_leaf_zero_sum_residual_before_projection_bb: self
                .maximum_leaf_zero_sum_residual
                .get(),
            zero_sum_residual_after_projection_bb: (profile_p0 + profile_p1).abs(),
        };
        let mut reasons = vec![
            "pilot flop abstraction omits all-in actions pending vectorized exact flop all-in runouts"
                .to_owned(),
        ];
        if self
            .config
            .value_network
            .source_validation_status
            .as_deref()
            != Some("accepted")
        {
            reasons.push("turn value network was trained from a rejected source corpus".to_owned());
        }
        if !self.config.value_network.uses_exact_ranges {
            reasons
                .push("range-blind value network is an ablation and cannot be promoted".to_owned());
        }
        if metrics.depth_limited_exploitability_bb_per_hand > 0.05 {
            reasons.push(format!(
                "depth-limited exploitability {:.6}bb/hand exceeds 0.05bb/hand",
                metrics.depth_limited_exploitability_bb_per_hand
            ));
        }
        if metrics.resolver_relative_exploitability_improvement <= 0.0 {
            reasons.push("resolver did not improve over the unresolved uniform control".to_owned());
        }
        if metrics.zero_sum_residual_after_projection_bb > 1e-6 {
            reasons.push(format!(
                "root zero-sum residual {:.3e} exceeds 1e-6",
                metrics.zero_sum_residual_after_projection_bb
            ));
        }
        FlopSolution {
            schema: "hu-depth-limited-flop-public-belief-solution-v1".to_owned(),
            method: "exact_turn_chance_enumeration_with_full_vector_turn_cfv_network_and_alternating_dcfr"
                .to_owned(),
            approximate: true,
            value_network_seed: self.config.value_network.seed,
            uses_exact_ranges: self.config.value_network.uses_exact_ranges,
            value_network_source_dataset_sha256: self
                .config
                .value_network
                .source_dataset_sha256,
            state: self.config.state,
            iterations: self.config.iterations,
            strategies,
            counterfactual_values_bb,
            opponent_compatible_mass,
            metrics,
            validation: BlueprintValidation {
                status: "rejected".to_owned(),
                reasons,
            },
        }
    }
}

pub fn solve_flop(config: FlopResolveConfig) -> Result<FlopSolution, String> {
    let mut solver = FlopSolver::new(config)?;
    solver.train();
    Ok(solver.finish())
}

#[derive(Clone, Debug)]
pub struct RiverSolveConfig {
    pub game: BlueprintConfig,
    pub state: PublicBeliefState,
    pub iterations: u64,
    pub averaging_delay: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PublicBeliefStrategy {
    pub public_history: Vec<String>,
    pub actor: usize,
    pub action_labels: Vec<String>,
    /// Row-major `[combo][action]` frozen average policy.
    pub probabilities: Vec<f32>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RiverSolveMetrics {
    pub information_sets: usize,
    pub legal_combinations: [usize; 2],
    pub joint_compatibility_mass: f64,
    pub profile_value_p0_bb: f64,
    pub profile_value_p1_bb: f64,
    pub best_response_value_p0_bb: f64,
    pub best_response_value_p1_bb: f64,
    pub exact_abstract_exploitability_bb_per_hand: f64,
    pub zero_sum_residual_bb: f64,
    pub maximum_probability_sum_error: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RiverSolution {
    pub schema: String,
    pub method: String,
    pub approximate: bool,
    pub game: BlueprintConfig,
    pub state: PublicBeliefState,
    pub iterations: u64,
    pub averaging_delay: u64,
    pub counterfactual_values_bb: [Vec<f32>; 2],
    pub opponent_compatible_mass: [Vec<f32>; 2],
    pub strategies: Vec<PublicBeliefStrategy>,
    pub metrics: RiverSolveMetrics,
    pub validation: BlueprintValidation,
}

#[derive(Clone)]
struct RangeNode {
    actor: usize,
    action_labels: Vec<String>,
    regrets: Vec<f64>,
    strategy_sum: Vec<f64>,
    last_discount_iteration: u64,
}

impl RangeNode {
    fn new(actor: usize, actions: &[LegalAction]) -> Self {
        let action_count = actions.len();
        Self {
            actor,
            action_labels: actions.iter().map(|action| action.label.clone()).collect(),
            regrets: vec![0.0; COMBO_COUNT * action_count],
            strategy_sum: vec![0.0; COMBO_COUNT * action_count],
            last_discount_iteration: 0,
        }
    }

    fn strategy(&self, legal: &[bool]) -> Vec<f64> {
        let action_count = self.action_labels.len();
        let mut result = vec![0.0; COMBO_COUNT * action_count];
        for combo in 0..COMBO_COUNT {
            if !legal[combo] {
                continue;
            }
            let offset = combo * action_count;
            let positive_sum = self.regrets[offset..offset + action_count]
                .iter()
                .map(|regret| regret.max(0.0))
                .sum::<f64>();
            if positive_sum > EPSILON {
                for action in 0..action_count {
                    result[offset + action] = self.regrets[offset + action].max(0.0) / positive_sum;
                }
            } else {
                result[offset..offset + action_count].fill(1.0 / action_count as f64);
            }
        }
        result
    }

    fn average_strategy(&self, legal: &[bool]) -> Vec<f64> {
        let action_count = self.action_labels.len();
        let mut result = vec![0.0; COMBO_COUNT * action_count];
        for combo in 0..COMBO_COUNT {
            if !legal[combo] {
                continue;
            }
            let offset = combo * action_count;
            let total = self.strategy_sum[offset..offset + action_count]
                .iter()
                .sum::<f64>();
            if total > EPSILON {
                for action in 0..action_count {
                    result[offset + action] = self.strategy_sum[offset + action] / total;
                }
            } else {
                result[offset..offset + action_count].fill(1.0 / action_count as f64);
            }
        }
        result
    }

    fn discount(&mut self, iteration: u64, parameters: &DcfrParameters) {
        if iteration == 0 || self.last_discount_iteration == iteration {
            return;
        }
        let time = iteration as f64;
        let positive_power = time.powf(parameters.positive_regret_exponent);
        let negative_power = time.powf(parameters.negative_regret_exponent);
        let positive_factor = positive_power / (positive_power + 1.0);
        let negative_factor = negative_power / (negative_power + 1.0);
        let strategy_factor = (time / (time + 1.0)).powf(parameters.strategy_exponent);
        for regret in &mut self.regrets {
            *regret *= if *regret >= 0.0 {
                positive_factor
            } else {
                negative_factor
            };
        }
        for weight in &mut self.strategy_sum {
            *weight *= strategy_factor;
        }
        self.last_discount_iteration = iteration;
    }
}

struct RiverSolver {
    config: RiverSolveConfig,
    legal: [Vec<bool>; 2],
    strengths: Vec<u32>,
    conflicts: Vec<Vec<usize>>,
    nodes: BTreeMap<Vec<String>, RangeNode>,
}

impl RiverSolver {
    fn new(config: RiverSolveConfig) -> Result<Self, String> {
        config.game.validate()?;
        if config.iterations < 2 {
            return Err("river solving requires at least two alternating iterations".to_owned());
        }
        if config.averaging_delay >= config.iterations {
            return Err("river averaging delay must be smaller than iterations".to_owned());
        }
        let state = config.state.validate_and_normalize(&config.game)?;
        let combos = all_combos();
        let legal = std::array::from_fn(|player| {
            state.ranges[player]
                .iter()
                .map(|weight| *weight > 0.0)
                .collect()
        });
        let strengths = combos
            .iter()
            .map(|combo| {
                let mut cards = state.board.clone();
                cards.extend(combo.cards());
                evaluate(&cards)
            })
            .collect();
        let conflicts = combos
            .iter()
            .map(|own| {
                combos
                    .iter()
                    .enumerate()
                    .filter_map(|(index, other)| own.overlaps(*other).then_some(index))
                    .collect()
            })
            .collect();
        Ok(Self {
            config: RiverSolveConfig { state, ..config },
            legal,
            strengths,
            conflicts,
            nodes: BTreeMap::new(),
        })
    }

    fn train(&mut self) {
        let root = self.config.state.game_state();
        let reaches = self.config.state.ranges.clone();
        for offset in 0..self.config.iterations {
            let traverser = offset as usize % 2;
            let iteration = offset + 1;
            self.walk(root.clone(), reaches.clone(), traverser, iteration);
        }
    }

    fn walk(
        &mut self,
        state: GameState,
        reaches: [Vec<f64>; 2],
        traverser: usize,
        iteration: u64,
    ) -> [Vec<f64>; 2] {
        if state.terminal.is_some() {
            return self.terminal_values(&state, &reaches);
        }
        let actions = state.legal_actions(&self.config.game);
        let key = state.public_history.clone();
        let actor = state.actor;
        let strategy = {
            let node = self
                .nodes
                .entry(key.clone())
                .or_insert_with(|| RangeNode::new(actor, &actions));
            assert_eq!(node.actor, actor);
            assert_eq!(
                node.action_labels,
                actions
                    .iter()
                    .map(|action| action.label.clone())
                    .collect::<Vec<_>>()
            );
            node.discount(iteration, &self.config.game.dcfr);
            node.strategy(&self.legal[actor])
        };
        let action_count = actions.len();
        let mut children = Vec::with_capacity(action_count);
        for (action_index, action) in actions.iter().enumerate() {
            let mut child_reaches = reaches.clone();
            for combo in 0..COMBO_COUNT {
                child_reaches[actor][combo] *= strategy[combo * action_count + action_index];
            }
            children.push(self.walk(
                state.apply(action, &self.config.game),
                child_reaches,
                traverser,
                iteration,
            ));
        }
        let opponent = 1 - actor;
        let mut values = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        for combo in 0..COMBO_COUNT {
            for action in 0..action_count {
                values[actor][combo] +=
                    strategy[combo * action_count + action] * children[action][actor][combo];
                values[opponent][combo] += children[action][opponent][combo];
            }
        }
        let node = self.nodes.get_mut(&key).expect("range node inserted");
        if actor == traverser {
            for combo in 0..COMBO_COUNT {
                if !self.legal[actor][combo] {
                    continue;
                }
                let offset = combo * action_count;
                for action in 0..action_count {
                    node.regrets[offset + action] +=
                        children[action][actor][combo] - values[actor][combo];
                }
            }
        }
        if iteration > self.config.averaging_delay {
            for combo in 0..COMBO_COUNT {
                if !self.legal[actor][combo] {
                    continue;
                }
                let offset = combo * action_count;
                for action in 0..action_count {
                    node.strategy_sum[offset + action] +=
                        reaches[actor][combo] * strategy[offset + action];
                }
            }
        }
        values
    }

    fn terminal_values(&self, state: &GameState, reaches: &[Vec<f64>; 2]) -> [Vec<f64>; 2] {
        match state.terminal.as_ref().expect("terminal public state") {
            Terminal::Fold { winner } => {
                let utility_p0 = if *winner == 0 {
                    state.invested[1]
                } else {
                    -state.invested[0]
                };
                [
                    self.constant_terminal_values(&reaches[1], utility_p0),
                    self.constant_terminal_values(&reaches[0], -utility_p0),
                ]
            }
            Terminal::Showdown => [
                self.showdown_terminal_values(
                    0,
                    &reaches[1],
                    state.invested[1],
                    -state.invested[0],
                    (state.invested[1] - state.invested[0]) / 2.0,
                ),
                self.showdown_terminal_values(
                    1,
                    &reaches[0],
                    state.invested[0],
                    -state.invested[1],
                    (state.invested[0] - state.invested[1]) / 2.0,
                ),
            ],
        }
    }

    fn constant_terminal_values(&self, opponent_reach: &[f64], utility: f64) -> Vec<f64> {
        (0..COMBO_COUNT)
            .map(|combo| utility * self.compatible_mass(opponent_reach, combo))
            .collect()
    }

    fn showdown_terminal_values(
        &self,
        _player: usize,
        opponent_reach: &[f64],
        win: f64,
        loss: f64,
        tie: f64,
    ) -> Vec<f64> {
        let mut by_strength = BTreeMap::<u32, f64>::new();
        for (strength, weight) in self.strengths.iter().zip(opponent_reach) {
            *by_strength.entry(*strength).or_default() += *weight;
        }
        let mut lower_by_strength = BTreeMap::new();
        let mut running = 0.0;
        for (strength, weight) in &by_strength {
            lower_by_strength.insert(*strength, running);
            running += weight;
        }
        let total = running;
        let mut values = vec![0.0; COMBO_COUNT];
        for own in 0..COMBO_COUNT {
            let strength = self.strengths[own];
            let mut lower = *lower_by_strength.get(&strength).unwrap_or(&0.0);
            let mut equal = *by_strength.get(&strength).unwrap_or(&0.0);
            let mut higher = total - lower - equal;
            for opponent in &self.conflicts[own] {
                let weight = opponent_reach[*opponent];
                match self.strengths[*opponent].cmp(&strength) {
                    std::cmp::Ordering::Less => lower -= weight,
                    std::cmp::Ordering::Equal => equal -= weight,
                    std::cmp::Ordering::Greater => higher -= weight,
                }
            }
            values[own] = lower.max(0.0) * win + equal.max(0.0) * tie + higher.max(0.0) * loss;
        }
        values
    }

    fn compatible_mass(&self, range: &[f64], own: usize) -> f64 {
        let total = range.iter().sum::<f64>();
        let blocked = self.conflicts[own]
            .iter()
            .map(|index| range[*index])
            .sum::<f64>();
        (total - blocked).max(0.0)
    }

    fn profile_walk(
        &self,
        state: GameState,
        reaches: [Vec<f64>; 2],
        best_responder: Option<usize>,
    ) -> [Vec<f64>; 2] {
        if state.terminal.is_some() {
            return self.terminal_values(&state, &reaches);
        }
        let actions = state.legal_actions(&self.config.game);
        let key = state.public_history.clone();
        let actor = state.actor;
        let node = self.nodes.get(&key).expect("trained range node");
        let strategy = node.average_strategy(&self.legal[actor]);
        let action_count = actions.len();
        let mut children = Vec::with_capacity(action_count);
        for (action_index, action) in actions.iter().enumerate() {
            let mut child_reaches = reaches.clone();
            if best_responder != Some(actor) {
                for combo in 0..COMBO_COUNT {
                    child_reaches[actor][combo] *= strategy[combo * action_count + action_index];
                }
            }
            children.push(self.profile_walk(
                state.apply(action, &self.config.game),
                child_reaches,
                best_responder,
            ));
        }
        let opponent = 1 - actor;
        let mut values = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        for combo in 0..COMBO_COUNT {
            if best_responder == Some(actor) {
                let best = children
                    .iter()
                    .map(|child| child[actor][combo])
                    .fold(f64::NEG_INFINITY, f64::max);
                values[actor][combo] = best;
                for child in &children {
                    values[opponent][combo] += child[opponent][combo];
                }
            } else {
                for action in 0..action_count {
                    values[actor][combo] +=
                        strategy[combo * action_count + action] * children[action][actor][combo];
                    values[opponent][combo] += children[action][opponent][combo];
                }
            }
        }
        values
    }

    fn finish(self) -> RiverSolution {
        let root = self.config.state.game_state();
        let reaches = self.config.state.ranges.clone();
        let joint_mass = joint_compatibility_mass(&reaches);
        let profile = self.profile_walk(root.clone(), reaches.clone(), None);
        let br0 = self.profile_walk(root.clone(), reaches.clone(), Some(0));
        let br1 = self.profile_walk(root, reaches.clone(), Some(1));
        let aggregate = |values: &[f64], player: usize| {
            reaches[player]
                .iter()
                .zip(values)
                .map(|(reach, value)| reach * value)
                .sum::<f64>()
                / joint_mass
        };
        let profile_p0 = aggregate(&profile[0], 0);
        let profile_p1 = aggregate(&profile[1], 1);
        let best_p0 = aggregate(&br0[0], 0);
        let best_p1 = aggregate(&br1[1], 1);
        let exploitability = ((best_p0 - profile_p0) + (best_p1 - profile_p1)) / 2.0;
        let compatible_mass: [Vec<f32>; 2] = std::array::from_fn(|player| {
            (0..COMBO_COUNT)
                .map(|combo| self.compatible_mass(&reaches[1 - player], combo) as f32)
                .collect()
        });
        let counterfactual_values_bb = std::array::from_fn(|player| {
            profile[player]
                .iter()
                .zip(&compatible_mass[player])
                .map(|(value, mass)| {
                    if *mass > 0.0 {
                        (*value / *mass as f64) as f32
                    } else {
                        0.0
                    }
                })
                .collect()
        });
        let mut maximum_probability_sum_error = 0.0f64;
        let strategies = self
            .nodes
            .iter()
            .map(|(history, node)| {
                let probabilities = node.average_strategy(&self.legal[node.actor]);
                let action_count = node.action_labels.len();
                for combo in 0..COMBO_COUNT {
                    if self.legal[node.actor][combo] {
                        let sum = probabilities[combo * action_count..(combo + 1) * action_count]
                            .iter()
                            .sum::<f64>();
                        maximum_probability_sum_error =
                            maximum_probability_sum_error.max((sum - 1.0).abs());
                    }
                }
                PublicBeliefStrategy {
                    public_history: history.clone(),
                    actor: node.actor,
                    action_labels: node.action_labels.clone(),
                    probabilities: probabilities
                        .into_iter()
                        .map(|value| value as f32)
                        .collect(),
                }
            })
            .collect::<Vec<_>>();
        let zero_sum_residual = (profile_p0 + profile_p1).abs();
        let metrics = RiverSolveMetrics {
            information_sets: strategies.len(),
            legal_combinations: std::array::from_fn(|player| {
                self.legal[player].iter().filter(|legal| **legal).count()
            }),
            joint_compatibility_mass: joint_mass,
            profile_value_p0_bb: profile_p0,
            profile_value_p1_bb: profile_p1,
            best_response_value_p0_bb: best_p0,
            best_response_value_p1_bb: best_p1,
            exact_abstract_exploitability_bb_per_hand: exploitability.max(0.0),
            zero_sum_residual_bb: zero_sum_residual,
            maximum_probability_sum_error,
        };
        let mut reasons = Vec::new();
        if metrics.zero_sum_residual_bb > 1e-8 {
            reasons.push(format!(
                "zero-sum residual {:.3e} exceeds 1e-8",
                metrics.zero_sum_residual_bb
            ));
        }
        if metrics.maximum_probability_sum_error > 1e-6 {
            reasons.push(format!(
                "probability sum error {:.3e} exceeds 1e-6",
                metrics.maximum_probability_sum_error
            ));
        }
        if metrics.exact_abstract_exploitability_bb_per_hand > 0.05 {
            reasons.push(format!(
                "river abstraction exploitability {:.6}bb/hand exceeds 0.05bb/hand",
                metrics.exact_abstract_exploitability_bb_per_hand
            ));
        }
        RiverSolution {
            schema: RIVER_SCHEMA.to_owned(),
            method: "alternating_vectorized_dcfr_exact_private-card_and_river_chance_enumeration"
                .to_owned(),
            approximate: true,
            game: self.config.game,
            state: self.config.state,
            iterations: self.config.iterations,
            averaging_delay: self.config.averaging_delay,
            counterfactual_values_bb,
            opponent_compatible_mass: compatible_mass,
            strategies,
            validation: BlueprintValidation {
                status: if reasons.is_empty() {
                    "accepted"
                } else {
                    "rejected"
                }
                .to_owned(),
                reasons,
            },
            metrics,
        }
    }
}

pub fn solve_river(config: RiverSolveConfig) -> Result<RiverSolution, String> {
    let mut solver = RiverSolver::new(config)?;
    solver.train();
    Ok(solver.finish())
}

#[derive(Clone, Debug)]
pub struct TurnTargetGenerationConfig {
    pub game: BlueprintConfig,
    pub states: usize,
    pub river_iterations: u64,
    pub river_averaging_delay: u64,
    pub seed: u64,
    pub threads: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TurnValueTarget {
    pub state_id: String,
    pub board: [u8; 4],
    pub actor: usize,
    pub invested_bb: [f64; 2],
    pub ranges: [Vec<f32>; 2],
    pub counterfactual_values_bb: [Vec<f32>; 2],
    pub opponent_compatible_mass: [Vec<f32>; 2],
    pub exact_river_cards: usize,
    pub maximum_river_exploitability_bb_per_hand: f64,
    pub zero_sum_residual_bb: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range_particles: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range_effective_sample_size: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TurnTargetDataset {
    pub schema: String,
    pub method: String,
    pub approximate: bool,
    pub game: BlueprintConfig,
    pub seed: u64,
    pub river_iterations: u64,
    pub state_distribution: String,
    pub targets: Vec<TurnValueTarget>,
    pub validation: BlueprintValidation,
}

/// Generates a small, deterministic pilot corpus. Range shapes deliberately
/// vary by state and player so a range-aware network can be compared against a
/// range-blind ablation. These are synthetic reachable-like public beliefs;
/// promotion remains blocked until the same generator is fed beliefs sampled
/// from frozen-policy self play.
pub fn generate_turn_targets(
    config: TurnTargetGenerationConfig,
) -> Result<TurnTargetDataset, String> {
    config.game.validate()?;
    if config.states == 0 {
        return Err("turn target generation requires at least one state".to_owned());
    }
    if config.river_iterations < 2 {
        return Err("turn target generation requires at least two river iterations".to_owned());
    }
    if config.threads == 0 {
        return Err("turn target generation requires at least one worker thread".to_owned());
    }
    let mut rng = SplitMix64::new(config.seed);
    let mut targets = Vec::with_capacity(config.states);
    for state_index in 0..config.states {
        let board = sample_unique_board4(&mut rng);
        let pot_options: [f64; 5] = [2.0, 4.0, 6.0, 10.0, 16.0];
        let pot =
            pot_options[state_index % pot_options.len()].min(config.game.effective_stack_bb * 1.5);
        let invested = [pot / 2.0, pot / 2.0];
        let ranges = [
            shaped_range(&board, state_index, 0),
            shaped_range(&board, state_index, 1),
        ];
        targets.push(turn_target_from_exact_rivers(
            &config.game,
            board,
            1,
            invested,
            ranges,
            config.river_iterations,
            config.river_averaging_delay,
            config.threads,
            state_index,
        )?);
    }
    let mut reasons = vec![
        "pilot public beliefs are deterministic synthetic reachable-like ranges, not frozen-policy self-play samples"
            .to_owned(),
    ];
    let maximum_zero_sum_residual = targets
        .iter()
        .map(|target| target.zero_sum_residual_bb)
        .fold(0.0f64, f64::max);
    if maximum_zero_sum_residual > 1e-7 {
        reasons.push(format!(
            "turn target zero-sum residual {maximum_zero_sum_residual:.3e} exceeds 1e-7"
        ));
    }
    let maximum_river_exploitability = targets
        .iter()
        .map(|target| target.maximum_river_exploitability_bb_per_hand)
        .fold(0.0f64, f64::max);
    if maximum_river_exploitability > 0.05 {
        reasons.push(format!(
            "at least one river solve has {:.6}bb/hand abstract exploitability",
            maximum_river_exploitability
        ));
    }
    Ok(TurnTargetDataset {
        schema: "hu-turn-public-belief-cfv-dataset-v1".to_owned(),
        method:
            "exact_legal_river_enumeration_with_exact_card_removal_and_solved_river_betting_leaves"
                .to_owned(),
        approximate: true,
        game: config.game,
        seed: config.seed,
        river_iterations: config.river_iterations,
        state_distribution: "synthetic_reachable_like_pilot".to_owned(),
        targets,
        validation: BlueprintValidation {
            status: "rejected".to_owned(),
            reasons,
        },
    })
}

#[allow(clippy::too_many_arguments)]
fn turn_target_from_exact_rivers(
    game: &BlueprintConfig,
    board: [u8; 4],
    actor: usize,
    invested: [f64; 2],
    ranges: [Vec<f64>; 2],
    river_iterations: u64,
    river_averaging_delay: u64,
    threads: usize,
    state_index: usize,
) -> Result<TurnValueTarget, String> {
    let original_ranges = std::array::from_fn(|player| normalize_masked(&ranges[player], &board));
    let conflicts = combo_conflicts();
    let compatible_mass: [Vec<f64>; 2] = std::array::from_fn(|player| {
        (0..COMBO_COUNT)
            .map(|own| {
                compatible_mass_from_conflicts(&original_ranges[1 - player], &conflicts, own)
            })
            .collect()
    });
    let mut raw_values = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
    let mut exact_river_cards = 0usize;
    let mut maximum_river_exploitability = 0.0f64;
    let river_cards = (0..52u8)
        .filter(|river| !board.contains(river))
        .collect::<Vec<_>>();
    let worker_count = threads.min(river_cards.len()).max(1);
    let solved = std::thread::scope(|scope| {
        let mut workers = Vec::with_capacity(worker_count);
        for worker in 0..worker_count {
            let assigned = river_cards
                .iter()
                .copied()
                .skip(worker)
                .step_by(worker_count)
                .collect::<Vec<_>>();
            let game = game.clone();
            let original_ranges = original_ranges.clone();
            workers.push(scope.spawn(move || {
                assigned
                    .into_iter()
                    .map(|river| {
                        let river_board = [board[0], board[1], board[2], board[3], river];
                        let mut masked = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
                        let mut totals = [0.0; 2];
                        for player in 0..2 {
                            for combo in all_combos() {
                                let weight = if combo.cards().contains(&river) {
                                    0.0
                                } else {
                                    original_ranges[player][combo.key()]
                                };
                                masked[player][combo.key()] = weight;
                                totals[player] += weight;
                            }
                            if totals[player] <= EPSILON {
                                return Err("river removal exhausted a turn range".to_owned());
                            }
                            for weight in &mut masked[player] {
                                *weight /= totals[player];
                            }
                        }
                        let solution = solve_river(RiverSolveConfig {
                            game: game.clone(),
                            state: PublicBeliefState::river_start(
                                river_board,
                                actor,
                                invested,
                                masked,
                            ),
                            iterations: river_iterations,
                            averaging_delay: river_averaging_delay,
                        })?;
                        Ok((river, totals, solution))
                    })
                    .collect::<Result<Vec<_>, String>>()
            }));
        }
        let mut leaves = Vec::with_capacity(river_cards.len());
        for worker in workers {
            leaves.extend(
                worker
                    .join()
                    .map_err(|_| "river solve worker panicked".to_owned())??,
            );
        }
        leaves.sort_by_key(|(river, _, _)| *river);
        Ok::<_, String>(leaves)
    })?;
    for (_, totals, solution) in solved {
        exact_river_cards += 1;
        maximum_river_exploitability = maximum_river_exploitability
            .max(solution.metrics.exact_abstract_exploitability_bb_per_hand);
        for player in 0..2 {
            let opponent_total = totals[1 - player];
            for combo in 0..COMBO_COUNT {
                raw_values[player][combo] += solution.counterfactual_values_bb[player][combo]
                    as f64
                    * solution.opponent_compatible_mass[player][combo] as f64
                    * opponent_total;
            }
        }
    }
    // Every compatible pair of hole cards has 44 remaining river cards.
    let chance_denominator = 44.0;
    let counterfactual_values_bb: [Vec<f32>; 2] = std::array::from_fn(|player| {
        raw_values[player]
            .iter()
            .zip(&compatible_mass[player])
            .map(|(raw, mass)| {
                if *mass > EPSILON {
                    (raw / (chance_denominator * mass)) as f32
                } else {
                    0.0
                }
            })
            .collect()
    });
    let joint_mass = joint_compatibility_mass(&original_ranges);
    let aggregate = |player: usize| {
        original_ranges[player]
            .iter()
            .zip(&counterfactual_values_bb[player])
            .zip(&compatible_mass[player])
            .map(|((reach, value), opponent_mass)| reach * *value as f64 * opponent_mass)
            .sum::<f64>()
            / joint_mass
    };
    let zero_sum_residual = (aggregate(0) + aggregate(1)).abs();
    Ok(TurnValueTarget {
        state_id: format!("turn-pbs-{state_index:06}"),
        board,
        actor,
        invested_bb: invested,
        ranges: std::array::from_fn(|player| {
            original_ranges[player]
                .iter()
                .map(|value| *value as f32)
                .collect()
        }),
        counterfactual_values_bb,
        opponent_compatible_mass: std::array::from_fn(|player| {
            compatible_mass[player]
                .iter()
                .map(|value| *value as f32)
                .collect()
        }),
        exact_river_cards,
        maximum_river_exploitability_bb_per_hand: maximum_river_exploitability,
        zero_sum_residual_bb: zero_sum_residual,
        range_particles: None,
        range_effective_sample_size: None,
    })
}

#[derive(Clone, Debug)]
pub struct SelfPlayTurnTargetConfig {
    pub game: BlueprintConfig,
    pub states: usize,
    pub range_particles: u64,
    pub river_iterations: u64,
    pub river_averaging_delay: u64,
    pub seed: u64,
    pub threads: usize,
    pub network_path: PathBuf,
}

pub fn generate_self_play_turn_targets(
    config: SelfPlayTurnTargetConfig,
) -> Result<TurnTargetDataset, Box<dyn Error>> {
    config.game.validate()?;
    if config.states == 0 || config.range_particles < 2 || config.threads == 0 {
        return Err("self-play targets require states, range particles, and threads".into());
    }
    let policy = FrozenPolicy::load(&config.network_path)?;
    let mut chance = SplitMix64::new(config.seed);
    let mut targets = Vec::with_capacity(config.states);
    let mut attempts = 0usize;
    while targets.len() < config.states {
        attempts += 1;
        if attempts > config.states * 1_000 {
            return Err("could not sample enough nonterminal self-play turn states".into());
        }
        let true_deal = Deal::sample(&mut chance);
        let Some((turn_state, action_line)) =
            sample_turn_line(&policy, &config.game, &true_deal, &mut chance)
        else {
            continue;
        };
        let board = [
            true_deal.board[0],
            true_deal.board[1],
            true_deal.board[2],
            true_deal.board[3],
        ];
        let (ranges, effective_sample_size) = particle_belief_for_line(
            &policy,
            &config.game,
            board,
            &action_line,
            config.range_particles,
            &mut chance,
        );
        if effective_sample_size < 2.0 {
            continue;
        }
        let mut target = turn_target_from_exact_rivers(
            &config.game,
            board,
            turn_state.actor,
            turn_state.invested,
            ranges,
            config.river_iterations,
            config.river_averaging_delay,
            config.threads,
            targets.len(),
        )?;
        target.range_particles = Some(config.range_particles);
        target.range_effective_sample_size = Some(effective_sample_size);
        targets.push(target);
    }
    let maximum_river_exploitability = targets
        .iter()
        .map(|target| target.maximum_river_exploitability_bb_per_hand)
        .fold(0.0f64, f64::max);
    let minimum_ess = targets
        .iter()
        .filter_map(|target| target.range_effective_sample_size)
        .fold(f64::INFINITY, f64::min);
    let mut reasons = Vec::new();
    if config.range_particles < 100_000 {
        reasons.push(format!(
            "self-play beliefs use {} importance particles; release requires 100000 plus independent replicate agreement",
            config.range_particles
        ));
    }
    if maximum_river_exploitability > 0.05 {
        reasons.push(format!(
            "at least one river solve has {maximum_river_exploitability:.6}bb/hand abstract exploitability"
        ));
    }
    if minimum_ess < config.range_particles as f64 * 0.1 {
        reasons.push(format!(
            "minimum range effective sample size {minimum_ess:.1} is below 10% of particles"
        ));
    }
    Ok(TurnTargetDataset {
        schema: "hu-turn-public-belief-cfv-dataset-v1".to_owned(),
        method: "frozen_policy_self_play_public_states_with_importance_particle_exact_combo_beliefs_and_exact_river_enumeration"
            .to_owned(),
        approximate: true,
        game: config.game,
        seed: config.seed,
        river_iterations: config.river_iterations,
        state_distribution: "frozen_v26_self_play_importance_particle_public_beliefs".to_owned(),
        targets,
        validation: BlueprintValidation {
            status: if reasons.is_empty() { "accepted" } else { "rejected" }.to_owned(),
            reasons,
        },
    })
}

fn sample_turn_line(
    policy: &FrozenPolicy,
    game: &BlueprintConfig,
    deal: &Deal,
    rng: &mut SplitMix64,
) -> Option<(GameState, Vec<String>)> {
    let mut state = GameState::initial(game);
    let mut line = Vec::new();
    while state.terminal.is_none() && state.street != Street::Turn {
        let actions = state.legal_actions(game);
        let strategy = policy.strategy(&state, deal, &actions, game);
        let selected = sample_index(&strategy, rng);
        line.push(actions[selected].label.clone());
        state = state.apply(&actions[selected], game);
    }
    (state.street == Street::Turn && state.terminal.is_none()).then_some((state, line))
}

fn particle_belief_for_line(
    policy: &FrozenPolicy,
    game: &BlueprintConfig,
    board: [u8; 4],
    line: &[String],
    particles: u64,
    rng: &mut SplitMix64,
) -> ([Vec<f64>; 2], f64) {
    let mut ranges = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
    let mut sum = 0.0;
    let mut squared_sum = 0.0;
    for _ in 0..particles {
        let deal = sample_deal_conditioned_on_board4(board, rng);
        let mut state = GameState::initial(game);
        let mut likelihood = 1.0;
        for selected_label in line {
            let actions = state.legal_actions(game);
            let Some(selected) = actions
                .iter()
                .position(|action| &action.label == selected_label)
            else {
                likelihood = 0.0;
                break;
            };
            let strategy = policy.strategy(&state, &deal, &actions, game);
            likelihood *= strategy[selected];
            state = state.apply(&actions[selected], game);
        }
        if likelihood <= 0.0 || !likelihood.is_finite() {
            continue;
        }
        sum += likelihood;
        squared_sum += likelihood * likelihood;
        for player in 0..2 {
            ranges[player][Combo::new(deal.holes[player][0], deal.holes[player][1]).key()] +=
                likelihood;
        }
    }
    for range in &mut ranges {
        let total = range.iter().sum::<f64>();
        if total > EPSILON {
            for weight in range {
                *weight /= total;
            }
        }
    }
    let effective_sample_size = sum * sum / squared_sum.max(EPSILON);
    (ranges, effective_sample_size)
}

fn sample_deal_conditioned_on_board4(board: [u8; 4], rng: &mut SplitMix64) -> Deal {
    let mut available = (0..52u8)
        .filter(|card| !board.contains(card))
        .collect::<Vec<_>>();
    for index in 0..5 {
        let swap = index + rng.index(available.len() - index);
        available.swap(index, swap);
    }
    Deal::from_sampled_cards(
        [[available[0], available[1]], [available[2], available[3]]],
        [board[0], board[1], board[2], board[3], available[4]],
    )
}

fn sample_unique_board4(rng: &mut SplitMix64) -> [u8; 4] {
    let mut deck = [0u8; 52];
    for (index, card) in deck.iter_mut().enumerate() {
        *card = index as u8;
    }
    for index in 0..4 {
        let swap = index + rng.index(52 - index);
        deck.swap(index, swap);
    }
    [deck[0], deck[1], deck[2], deck[3]]
}

fn shaped_range(board: &[u8], state_index: usize, player: usize) -> Vec<f64> {
    let blocked = board.iter().copied().collect::<BTreeSet<_>>();
    let direction = if (state_index + player).is_multiple_of(2) {
        1.0
    } else {
        -0.7
    };
    let mut weights = all_combos()
        .into_iter()
        .map(|combo| {
            if combo.cards().iter().any(|card| blocked.contains(card)) {
                return 0.0;
            }
            let cards = combo.cards();
            let high = (cards[0] >> 2).max(cards[1] >> 2) as f64 / 12.0;
            let low = (cards[0] >> 2).min(cards[1] >> 2) as f64 / 12.0;
            let pair = f64::from((cards[0] >> 2) == (cards[1] >> 2));
            let suited = f64::from((cards[0] & 3) == (cards[1] & 3));
            let board_match = cards
                .iter()
                .filter(|card| board.iter().any(|public| public >> 2 == **card >> 2))
                .count() as f64;
            let phase =
                ((combo.key() * 17 + state_index * 31 + player * 13) % 97) as f64 / 96.0 - 0.5;
            (direction * (1.4 * high + 0.6 * low + 1.0 * pair + 0.2 * suited)
                + 0.35 * board_match
                + 0.25 * phase)
                .exp()
        })
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f64>();
    for weight in &mut weights {
        *weight /= total;
    }
    weights
}

fn normalize_masked(range: &[f64], board: &[u8]) -> Vec<f64> {
    let blocked = board.iter().copied().collect::<BTreeSet<_>>();
    let mut normalized = all_combos()
        .into_iter()
        .map(|combo| {
            if combo.cards().iter().any(|card| blocked.contains(card)) {
                0.0
            } else {
                range[combo.key()].max(0.0)
            }
        })
        .collect::<Vec<_>>();
    let total = normalized.iter().sum::<f64>();
    for weight in &mut normalized {
        *weight /= total;
    }
    normalized
}

fn combo_conflicts() -> Vec<Vec<usize>> {
    let combos = all_combos();
    combos
        .iter()
        .map(|own| {
            combos
                .iter()
                .enumerate()
                .filter_map(|(index, other)| own.overlaps(*other).then_some(index))
                .collect()
        })
        .collect()
}

fn compatible_mass_from_conflicts(range: &[f64], conflicts: &[Vec<usize>], own: usize) -> f64 {
    let total = range.iter().sum::<f64>();
    (total
        - conflicts[own]
            .iter()
            .map(|index| range[*index])
            .sum::<f64>())
    .max(0.0)
}

pub fn uniform_range(board: &[u8]) -> Vec<f64> {
    let blocked = board.iter().copied().collect::<BTreeSet<_>>();
    let mut range = all_combos()
        .into_iter()
        .map(|combo| {
            if combo.cards().iter().any(|card| blocked.contains(card)) {
                0.0
            } else {
                1.0
            }
        })
        .collect::<Vec<_>>();
    let total = range.iter().sum::<f64>();
    for weight in &mut range {
        *weight /= total;
    }
    range
}

fn joint_compatibility_mass(ranges: &[Vec<f64>; 2]) -> f64 {
    let combos = all_combos();
    ranges[0]
        .iter()
        .enumerate()
        .map(|(first, first_weight)| {
            first_weight
                * ranges[1]
                    .iter()
                    .enumerate()
                    .filter(|(second, _)| !combos[first].overlaps(combos[*second]))
                    .map(|(_, weight)| weight)
                    .sum::<f64>()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_game() -> BlueprintConfig {
        let mut game = BlueprintConfig::default();
        game.effective_stack_bb = 4.0;
        game.iterations = 10;
        game.averaging_delay = 0;
        game.action_abstraction.turn_river_bet_pot_fractions = vec![1.0];
        game.action_abstraction.postflop_raise_pot_fractions = vec![1.0];
        game
    }

    fn zero_value_network() -> PublicValueNetwork {
        let layer = |input_size, output_size, activation: &str| ValueNetworkLayer {
            input_size,
            output_size,
            activation: activation.to_owned(),
            weights: vec![0.0; input_size * output_size],
            biases: vec![0.0; output_size],
        };
        PublicValueNetwork {
            schema: "hu-public-belief-value-network-v2".to_owned(),
            seed: 1,
            uses_exact_ranges: true,
            target_scale_bb: 20.0,
            range_scale: COMBO_COUNT as f64,
            source_dataset_sha256: Some("0".repeat(64)),
            source_validation_status: Some("accepted".to_owned()),
            public_tower: vec![layer(56, 1, "linear")],
            range_tower: vec![layer(COMBO_COUNT * 2, 1, "linear")],
            head: vec![layer(2, COMBO_COUNT * 2, "tanh")],
        }
    }

    #[test]
    fn river_range_masks_board_cards_and_normalizes() {
        let board = [0, 5, 10, 15, 20];
        let state = PublicBeliefState::river_start(
            board,
            1,
            [1.0, 1.0],
            [vec![1.0; COMBO_COUNT], vec![1.0; COMBO_COUNT]],
        );
        let normalized = state.validate_and_normalize(&tiny_game()).unwrap();
        for combo in all_combos() {
            if combo.cards().iter().any(|card| board.contains(card)) {
                assert_eq!(normalized.ranges[0][combo.key()], 0.0);
                assert_eq!(normalized.ranges[1][combo.key()], 0.0);
            }
        }
        assert!((normalized.ranges[0].iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn exact_showdown_values_respect_blockers_and_zero_sum() {
        let board = [0, 5, 10, 15, 20];
        let solution = solve_river(RiverSolveConfig {
            game: tiny_game(),
            state: PublicBeliefState::uniform_river_start(board, 1, [1.0, 1.0]),
            iterations: 8,
            averaging_delay: 0,
        })
        .unwrap();
        assert!(solution.metrics.joint_compatibility_mass > 0.9);
        assert!(solution.metrics.zero_sum_residual_bb < 1e-8);
        assert!(solution.metrics.maximum_probability_sum_error < 1e-6);
        assert!(solution
            .counterfactual_values_bb
            .iter()
            .flatten()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn river_solve_is_deterministic() {
        let config = RiverSolveConfig {
            game: tiny_game(),
            state: PublicBeliefState::uniform_river_start([0, 5, 10, 15, 20], 1, [1.0, 1.0]),
            iterations: 4,
            averaging_delay: 0,
        };
        assert_eq!(solve_river(config.clone()), solve_river(config));
    }

    #[test]
    fn full_vector_value_network_masks_illegal_combos_and_is_finite() {
        let board = [0, 5, 10, 15];
        let ranges = std::array::from_fn(|_| uniform_range(&board));
        let network = zero_value_network();
        network.validate().unwrap();
        let values = network.predict(&board, 1, [2.0, 2.0], &ranges);
        assert_eq!(values[0].len(), COMBO_COUNT);
        assert!(values.iter().flatten().all(|value| *value == 0.0));
    }

    #[test]
    fn flop_pilot_refuses_to_silently_drop_all_in() {
        let board = [0, 5, 10];
        let ranges = std::array::from_fn(|_| uniform_range(&board));
        let error = solve_flop(FlopResolveConfig {
            game: BlueprintConfig::default(),
            state: PublicBeliefState::flop_start(board, 1, [2.0, 2.0], ranges),
            iterations: 2,
            averaging_delay: 0,
            value_network: zero_value_network(),
        })
        .unwrap_err();
        assert!(error.contains("include_all_in=false"));
    }
}
