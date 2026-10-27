//! Small, complete river or turn/river games. No neural leaves or online solves.
//! Verification enumerates compatible opponent hands directly, independently of
//! RiverSolver's card-marginal terminal evaluator and vectorized BR traversal.
use super::*;

pub const SCHEMA: &str = "solved-river-spot-v1";
pub const TURN_SCHEMA: &str = "solved-turn-river-spot-v1";
const GAIN_LIMIT: f64 = 0.05; // SUM of both unilateral gains, not their average.
const EV_TOLERANCE: f64 = 0.0001;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SpotInput {
    pub id: String,
    pub title: String,
    pub description: String,
    pub range_assumptions: String,
    pub board: Vec<u8>,
    pub depth_bb: f64,
    pub pot_bb: f64,
    /// Exact hand-class labels. These are declared study ranges, NOT certified
    /// full-game preflop ranges. Unspecified class weights default to one.
    pub range_classes: [Vec<String>; 2],
    /// Inclusion weight per combo, in the same order as range_classes. Omit
    /// entirely for legacy equal weights; omission preserves old artifact bytes.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_class_weights"
    )]
    pub range_class_weights: Option<[Vec<f64>; 2]>,
    pub bet_pot_fractions: Vec<f64>,
    pub raise_pot_fractions: Vec<f64>,
    pub raise_cap: u8,
    pub iterations: u64,
}

fn deserialize_class_weights<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<[Vec<f64>; 2]>, D::Error> {
    // Missing means legacy defaults; an explicitly null/malformed prior is not
    // permission to silently substitute equal weights.
    <[Vec<f64>; 2]>::deserialize(deserializer).map(Some)
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpotAction {
    pub id: String,
    pub kind: String,
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub amount_to_bb: Option<f64>,
    pub next_node_id: Option<String>,
    /// Public river chance, only on a non-all-in turn round closure. All 48
    /// public candidates are present; the dealt private cards exclude four.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_by_river: Option<Vec<(u8, String)>>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpotRow {
    pub combo: usize,
    pub probabilities: Vec<f32>,
    /// None means no compatible opponent reach at this off-path information set.
    pub ev_bb: Option<Vec<f32>>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpotNode {
    pub id: String,
    pub actor: usize,
    pub invested_bb: [f64; 2],
    pub street_bets_bb: [f64; 2],
    pub actions: Vec<SpotAction>,
    pub rows: Vec<SpotRow>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub board: Option<Vec<u8>>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Verification {
    pub method: String,
    pub compatible_deals: usize,
    pub profile_value_bb: [f64; 2],
    pub best_response_value_bb: [f64; 2],
    pub unilateral_gain_bb: [f64; 2],
    pub total_best_response_gain_bb: f64,
    pub maximum_action_ev_error_bb: f64,
    pub maximum_probability_sum_error: f64,
    pub zero_sum_residual_bb: f64,
    pub accepted: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub river_outcomes: Option<usize>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SolvedSpot {
    pub schema: String,
    pub input_sha256: String,
    pub input: SpotInput,
    pub root_node_id: String,
    pub ranges: [Vec<(usize, f64)>; 2],
    pub nodes: Vec<SpotNode>,
    pub verification: Verification,
}

struct SpotConfig {
    game: BlueprintConfig,
    state: PublicBeliefState,
}

fn config(input: &SpotInput) -> Result<SpotConfig, String> {
    if input.id.is_empty()
        || input.id.len() > 80
        || !input
            .id
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        || input.title.is_empty()
        || input.range_assumptions.is_empty()
        || !matches!(input.board.len(), 4 | 5)
        || !input.depth_bb.is_finite()
        || input.depth_bb < 2.0
        || input.depth_bb > 1000.0
        || !input.pot_bb.is_finite()
        || input.pot_bb < 2.0
        || input.pot_bb >= 2.0 * input.depth_bb
        || !(20..=100_000).contains(&input.iterations)
        || (input.pot_bb * 500.0).fract().abs() > 1e-8
        || input.raise_cap > 1
        || [&input.bet_pot_fractions, &input.raise_pot_fractions]
            .iter()
            .any(|sizes| {
                sizes.is_empty()
                    || sizes.len() > 2
                    || sizes
                        .iter()
                        .any(|f| !f.is_finite() || *f <= 0.0 || *f > 4.0)
                    || sizes.windows(2).any(|pair| pair[0] >= pair[1])
            })
    {
        return Err("invalid or unbounded solved-spot input".into());
    }
    let mut game = BlueprintConfig::default();
    game.effective_stack_bb = input.depth_bb;
    game.iterations = 2;
    game.averaging_delay = 0;
    game.action_abstraction.turn_river_bet_pot_fractions = input.bet_pot_fractions.clone();
    game.action_abstraction.postflop_raise_pot_fractions = input.raise_pot_fractions.clone();
    game.action_abstraction.postflop_raise_cap = input.raise_cap;
    game.action_abstraction.include_all_in = true;
    game.dcfr = DcfrParameters {
        positive_regret_exponent: 1.5,
        negative_regret_exponent: 0.0,
        strategy_exponent: 2.0,
    };
    let initial_ranges = std::array::from_fn(|_| uniform_range(&input.board));
    let mut state = if input.board.len() == 5 {
        PublicBeliefState::river_start(
            input.board.clone().try_into().unwrap(),
            1,
            [input.pot_bb / 2.0; 2],
            initial_ranges,
        )
    } else {
        PublicBeliefState::turn_start(
            input.board.clone().try_into().unwrap(),
            1,
            [input.pot_bb / 2.0; 2],
            initial_ranges,
        )
    };
    state.public_history = vec![format!("solved-spot:{}", input.id)];
    let combos = all_combos();
    for player in 0..2 {
        let labels = input.range_classes[player].iter().collect::<BTreeSet<_>>();
        if labels.len() != input.range_classes[player].len()
            || labels.is_empty()
            || labels
                .iter()
                .any(|label| !combos.iter().any(|c| &c.label() == *label))
        {
            return Err("ranges require unique exact hand-class labels".into());
        }
        let weights = input.range_class_weights.as_ref().map(|w| &w[player]);
        if weights.is_some_and(|w| {
            w.len() != input.range_classes[player].len()
                || w.iter().any(|x| !x.is_finite() || *x <= 0.0 || *x > 1.0)
        }) {
            return Err("class weights must match the labels and be finite in (0, 1]".into());
        }
        let weighted_labels: BTreeMap<_, _> = input.range_classes[player]
            .iter()
            .enumerate()
            .map(|(i, label)| (label.clone(), weights.map_or(1.0, |w| w[i])))
            .collect();
        state.ranges[player] = combos
            .iter()
            .map(|combo| weighted_labels.get(&combo.label()).copied().unwrap_or(0.0))
            .collect();
    }
    let support: [Vec<usize>; 2] = std::array::from_fn(|p| {
        combos
            .iter()
            .filter(|combo| {
                state.ranges[p][combo.key()] > 0.0
                    && !combo.cards().iter().any(|c| input.board.contains(c))
            })
            .map(|combo| combo.key())
            .collect()
    });
    state = state.validate_street_and_normalize(&game, state.street, input.board.len())?;
    if support
        .iter()
        .enumerate()
        .any(|(p, keys)| keys.iter().any(|key| state.ranges[p][*key] <= 0.0))
    {
        return Err("class weights underflow after normalization".into());
    }
    Ok(SpotConfig { game, state })
}

fn node_id(history: &[String]) -> String {
    history.join("/")
}

fn key(state: &GameState, board: &[u8], turn_root: bool) -> Vec<String> {
    if turn_root {
        TurnRiverSolver::node_key(state, board.get(4).copied())
    } else {
        state.public_history.clone()
    }
}

fn river_edges(state: &GameState, board: &[u8], turn_root: bool) -> Option<Vec<(u8, String)>> {
    if state.terminal.is_some() || state.street != Street::River || board.len() != 4 {
        return None;
    }
    Some(
        (0..52u8)
            .filter(|card| !board.contains(card))
            .map(|card| {
                let mut next_board = board.to_vec();
                next_board.push(card);
                (card, node_id(&key(state, &next_board, turn_root)))
            })
            .collect(),
    )
}

fn export_nodes(
    state: GameState,
    board: Vec<u8>,
    turn_root: bool,
    game: &BlueprintConfig,
    ranges: [Vec<f64>; 2],
    policies: &BTreeMap<Vec<String>, &PublicBeliefStrategy>,
    output: &mut Vec<SpotNode>,
) -> Result<(), String> {
    if state.terminal.is_some() {
        return Ok(());
    }
    if let Some(edges) = river_edges(&state, &board, turn_root) {
        for (card, _) in edges {
            let mut next_board = board.clone();
            next_board.push(card);
            let mut next_ranges = ranges.clone();
            for player in &mut next_ranges {
                for combo in all_combos() {
                    if combo.cards().contains(&card) {
                        player[combo.key()] = 0.0;
                    }
                }
            }
            export_nodes(
                state.clone(),
                next_board,
                turn_root,
                game,
                next_ranges,
                policies,
                output,
            )?;
        }
        return Ok(());
    }
    let public_key = key(&state, &board, turn_root);
    let policy = policies
        .get(&public_key)
        .ok_or("missing frozen river policy")?;
    let legal = state.legal_actions(game);
    let mut actions = Vec::new();
    for (index, action) in legal.iter().enumerate() {
        let next = state.apply(action, game);
        let (kind, label, amount_to_bb) = match action.kind {
            ActionKind::Fold => ("fold", "Fold".into(), None),
            ActionKind::Check => ("check", "Check".into(), None),
            ActionKind::Call => ("call", format!("Call {:.1}bb", state.to_call()), None),
            ActionKind::RaiseTo(target) => {
                let all_in = (target
                    - state.street_invested[state.actor]
                    - state.remaining(state.actor, game))
                .abs()
                    < 1e-8;
                let kind = if all_in {
                    "all-in"
                } else if state.to_call() > 0.0 {
                    "raise"
                } else {
                    "bet"
                };
                (
                    kind,
                    format!(
                        "{} {:.1}bb",
                        if all_in {
                            "All-in"
                        } else if kind == "raise" {
                            "Raise to"
                        } else {
                            "Bet"
                        },
                        target
                    ),
                    Some(target),
                )
            }
        };
        actions.push(SpotAction {
            id: action.label.clone(),
            kind: kind.into(),
            label,
            amount_to_bb,
            next_node_id: if next.terminal.is_some()
                || river_edges(&next, &board, turn_root).is_some()
            {
                None
            } else {
                Some(node_id(&key(&next, &board, turn_root)))
            },
            next_by_river: river_edges(&next, &board, turn_root),
        });
        let mut next_ranges = ranges.clone();
        for combo in 0..COMBO_COUNT {
            next_ranges[state.actor][combo] *=
                policy.probabilities[combo * legal.len() + index] as f64;
        }
        export_nodes(
            next,
            board.clone(),
            turn_root,
            game,
            next_ranges,
            policies,
            output,
        )?;
    }
    let rows = all_combos()
        .iter()
        .filter(|combo| {
            policy.probabilities[combo.key() * legal.len()..(combo.key() + 1) * legal.len()]
                .iter()
                .any(|p| *p > 0.0)
        })
        .map(|combo| {
            let offset = combo.key() * legal.len();
            let mass: f64 = all_combos()
                .iter()
                .filter(|other| !combo.overlaps(**other))
                .map(|other| ranges[1 - state.actor][other.key()])
                .sum();
            Ok(SpotRow {
                combo: combo.key(),
                probabilities: policy.probabilities[offset..offset + legal.len()].to_vec(),
                ev_bb: if mass > 0.0 {
                    Some(
                        policy
                            .action_values_bb
                            .as_ref()
                            .ok_or("missing action EVs")?[offset..offset + legal.len()]
                            .to_vec(),
                    )
                } else {
                    None
                },
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    output.push(SpotNode {
        id: node_id(&public_key),
        actor: state.actor,
        invested_bb: state.invested,
        street_bets_bb: state.street_invested,
        actions,
        rows,
        board: turn_root.then_some(board),
    });
    Ok(())
}

// Direct pair enumeration: do not reuse the optimized marginal evaluator.
struct PairVerifier<'a> {
    nodes: BTreeMap<String, &'a SpotNode>,
    rows: BTreeMap<(String, usize), &'a SpotRow>,
    combos: Vec<crate::cards::Combo>,
    strengths: Vec<Vec<u32>>,
    turn_root: bool,
    root_board: Vec<u8>,
    all_in_equities: BTreeMap<(usize, usize), f64>,
    maximum_ev_error: f64,
    visited: BTreeSet<String>,
}

impl PairVerifier<'_> {
    fn equity(&mut self, combo: usize, other: usize, river: Option<u8>) -> f64 {
        let comparison = |a: u32, b: u32| match a.cmp(&b) {
            std::cmp::Ordering::Greater => 1.0,
            std::cmp::Ordering::Less => -1.0,
            std::cmp::Ordering::Equal => 0.0,
        };
        if !self.turn_root {
            return comparison(self.strengths[52][combo], self.strengths[52][other]);
        }
        if let Some(card) = river {
            return comparison(
                self.strengths[card as usize][combo],
                self.strengths[card as usize][other],
            );
        }
        if let Some(value) = self.all_in_equities.get(&(combo, other)) {
            return *value;
        }
        let mut value = 0.0;
        let mut count = 0;
        for card in 0..52u8 {
            if self.root_board.contains(&card)
                || self.combos[combo].cards().contains(&card)
                || self.combos[other].cards().contains(&card)
            {
                continue;
            }
            value += comparison(
                self.strengths[card as usize][combo],
                self.strengths[card as usize][other],
            );
            count += 1;
        }
        assert_eq!(count, 44);
        value /= 44.0;
        self.all_in_equities.insert((combo, other), value);
        value
    }
    fn probability(row: &SpotRow, action: usize) -> f64 {
        // Match actual categorical serving: renormalize stored f32 rows in f64.
        row.probabilities[action] as f64 / row.probabilities.iter().map(|p| *p as f64).sum::<f64>()
    }
    fn walk(
        &mut self,
        state: &GameState,
        board: &[u8],
        game: &BlueprintConfig,
        own: usize,
        combo: usize,
        opponents: &[(usize, f64)],
        best_response: bool,
    ) -> Result<f64, String> {
        if let Some(terminal) = &state.terminal {
            return Ok(opponents
                .iter()
                .map(|(other, weight)| {
                    let value = match terminal {
                        Terminal::Fold { winner } => {
                            if *winner == own {
                                state.invested[1 - own]
                            } else {
                                -state.invested[own]
                            }
                        }
                        Terminal::Showdown => {
                            self.equity(combo, *other, board.get(4).copied()) * state.invested[own]
                        }
                    };
                    weight * value
                })
                .sum());
        }
        if let Some(edges) = river_edges(state, board, self.turn_root) {
            let mut total = 0.0;
            for (card, _) in edges {
                // Empty opponents is a separate structural pass, covering even
                // public boards impossible for a particular private holding.
                if !opponents.is_empty() && self.combos[combo].cards().contains(&card) {
                    continue;
                }
                let masked = opponents
                    .iter()
                    .copied()
                    .filter(|(other, _)| !self.combos[*other].cards().contains(&card))
                    .map(|(other, w)| (other, w / 44.0))
                    .collect::<Vec<_>>();
                let mut next_board = board.to_vec();
                next_board.push(card);
                total += self.walk(state, &next_board, game, own, combo, &masked, best_response)?;
            }
            return Ok(total);
        }
        let id = node_id(&key(state, board, self.turn_root));
        self.visited.insert(id.clone());
        let node = *self.nodes.get(&id).ok_or("incomplete exported tree")?;
        let actions = state.legal_actions(game);
        if node.actor != state.actor
            || node.invested_bb != state.invested
            || node.street_bets_bb != state.street_invested
            || node.actions.len() != actions.len()
            || node.board != self.turn_root.then_some(board.to_vec())
        {
            return Err("exported node/state mismatch".into());
        }
        let mut values = Vec::new();
        for (index, action) in actions.iter().enumerate() {
            let next = state.apply(action, game);
            let chance = river_edges(&next, board, self.turn_root);
            let expected = if next.terminal.is_some() || chance.is_some() {
                None
            } else {
                Some(node_id(&key(&next, board, self.turn_root)))
            };
            let (kind, target) = match action.kind {
                ActionKind::Fold => ("fold", None),
                ActionKind::Check => ("check", None),
                ActionKind::Call => ("call", None),
                ActionKind::RaiseTo(target) => {
                    let all_in = (target
                        - state.street_invested[state.actor]
                        - state.remaining(state.actor, game))
                    .abs()
                        < 1e-8;
                    (
                        if all_in {
                            "all-in"
                        } else if state.to_call() > 0.0 {
                            "raise"
                        } else {
                            "bet"
                        },
                        Some(target),
                    )
                }
            };
            if node.actions[index].id != action.label
                || node.actions[index].next_node_id != expected
                || node.actions[index].kind != kind
                || node.actions[index].amount_to_bb != target
                || node.actions[index].next_by_river != chance
            {
                return Err("exported tree edge mismatch".into());
            }
            let next_opponents = if state.actor != own {
                opponents
                    .iter()
                    .map(|(other, weight)| {
                        let row = self
                            .rows
                            .get(&(id.clone(), *other))
                            .ok_or("missing opponent row")?;
                        Ok((*other, weight * Self::probability(row, index)))
                    })
                    .collect::<Result<Vec<_>, String>>()?
            } else {
                opponents.to_vec()
            };
            values.push(self.walk(
                &next,
                board,
                game,
                own,
                combo,
                &next_opponents,
                best_response,
            )?);
        }
        if opponents.is_empty() {
            return Ok(0.0);
        }
        if state.actor != own {
            return Ok(values.iter().sum());
        }
        let row = self.rows.get(&(id, combo)).ok_or("missing own row")?;
        if best_response {
            // Max AFTER summing hidden opponent hands: no clairvoyant response.
            return Ok(values.into_iter().fold(f64::NEG_INFINITY, f64::max));
        }
        let mass: f64 = opponents.iter().map(|(_, w)| w).sum();
        if mass > 0.0 {
            let labels = row
                .ev_bb
                .as_ref()
                .ok_or("missing reachable action values")?;
            for (index, value) in values.iter().enumerate() {
                self.maximum_ev_error = self
                    .maximum_ev_error
                    .max((value / mass - labels[index] as f64).abs());
            }
        }
        Ok(values
            .iter()
            .enumerate()
            .map(|(index, value)| value * Self::probability(row, index))
            .sum())
    }
}

pub fn verify(spot: &SolvedSpot) -> Result<Verification, String> {
    let config = config(&spot.input)?;
    let turn_root = spot.input.board.len() == 4;
    if spot.schema != if turn_root { TURN_SCHEMA } else { SCHEMA }
        || spot.input_sha256
            != format!(
                "{:x}",
                Sha256::digest(serde_json::to_vec(&spot.input).map_err(|e| e.to_string())?)
            )
        || spot.root_node_id != node_id(&config.state.public_history)
        || spot.ranges != sparse_ranges(&config.state.ranges)
    {
        return Err("spot identity mismatch".into());
    }
    let combos = all_combos();
    let mut verifier = PairVerifier {
        nodes: BTreeMap::new(),
        rows: BTreeMap::new(),
        strengths: (0..53)
            .map(|river| {
                if (turn_root && (river == 52 || config.state.board.contains(&(river as u8))))
                    || (!turn_root && river != 52)
                {
                    return Vec::new();
                }
                combos
                    .iter()
                    .map(|combo| {
                        let mut cards = config.state.board.clone();
                        if turn_root {
                            cards.push(river as u8);
                        }
                        cards.extend(combo.cards());
                        evaluate(&cards)
                    })
                    .collect()
            })
            .collect(),
        turn_root,
        root_board: config.state.board.clone(),
        all_in_equities: BTreeMap::new(),
        combos,
        maximum_ev_error: 0.0,
        visited: BTreeSet::new(),
    };
    let mut max_sum_error = 0.0f64;
    for node in &spot.nodes {
        if verifier.nodes.insert(node.id.clone(), node).is_some() {
            return Err("duplicate node".into());
        }
        if node.actor > 1 {
            return Err("invalid actor".into());
        }
        let expected_rows = config.state.ranges[node.actor]
            .iter()
            .enumerate()
            .filter(|(combo, p)| {
                **p > 0.0
                    && !verifier.combos[*combo]
                        .cards()
                        .iter()
                        .any(|c| node.board.as_ref().unwrap_or(&spot.input.board).contains(c))
            })
            .count();
        if node.rows.len() != expected_rows {
            return Err("incomplete combo rows".into());
        }
        for row in &node.rows {
            if row.combo >= COMBO_COUNT
                || config.state.ranges[node.actor][row.combo] <= 0.0
                || verifier.combos[row.combo]
                    .cards()
                    .iter()
                    .any(|c| node.board.as_ref().unwrap_or(&spot.input.board).contains(c))
                || row.probabilities.len() != node.actions.len()
                || row
                    .probabilities
                    .iter()
                    .any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
                || row.ev_bb.as_ref().is_some_and(|v| {
                    v.len() != node.actions.len() || v.iter().any(|x| !x.is_finite())
                })
                || verifier
                    .rows
                    .insert((node.id.clone(), row.combo), row)
                    .is_some()
            {
                return Err("invalid combo row".into());
            }
            max_sum_error = max_sum_error
                .max((row.probabilities.iter().map(|p| *p as f64).sum::<f64>() - 1.0).abs());
        }
    }
    let mut report = Verification {
        method: if turn_root {
            "direct-compatible-pair-and-river-enumeration-of-exported-f32-policy-v1"
        } else {
            "direct-compatible-pair-enumeration-of-exported-f32-policy-v1"
        }
        .into(),
        maximum_probability_sum_error: max_sum_error,
        ..Verification::default()
    };
    if max_sum_error > 1e-6 {
        return Err("invalid stored probability sums".into());
    }
    verifier.walk(
        &config.state.game_state(),
        &config.state.board,
        &config.game,
        0,
        0,
        &[],
        false,
    )?;
    let mut joint = 0.0;
    for own in 0..2 {
        for (combo, own_weight) in &spot.ranges[own] {
            let opponents = spot.ranges[1 - own]
                .iter()
                .copied()
                .filter(|(other, _)| !verifier.combos[*combo].overlaps(verifier.combos[*other]))
                .collect::<Vec<_>>();
            if own == 0 {
                report.compatible_deals += opponents.len();
                joint += own_weight * opponents.iter().map(|(_, w)| w).sum::<f64>();
            }
            report.profile_value_bb[own] += own_weight
                * verifier.walk(
                    &config.state.game_state(),
                    &config.state.board,
                    &config.game,
                    own,
                    *combo,
                    &opponents,
                    false,
                )?;
            report.best_response_value_bb[own] += own_weight
                * verifier.walk(
                    &config.state.game_state(),
                    &config.state.board,
                    &config.game,
                    own,
                    *combo,
                    &opponents,
                    true,
                )?;
        }
    }
    for own in 0..2 {
        report.profile_value_bb[own] /= joint;
        report.best_response_value_bb[own] /= joint;
        report.unilateral_gain_bb[own] =
            (report.best_response_value_bb[own] - report.profile_value_bb[own]).max(0.0);
    }
    report.total_best_response_gain_bb = report.unilateral_gain_bb.iter().sum();
    report.river_outcomes = turn_root.then_some(report.compatible_deals * 44);
    if verifier.visited.len() != spot.nodes.len() {
        return Err("unreachable extra nodes in exported tree".into());
    }
    report.maximum_action_ev_error_bb = verifier.maximum_ev_error;
    report.zero_sum_residual_bb = report.profile_value_bb.iter().sum::<f64>().abs();
    report.accepted = report.total_best_response_gain_bb <= GAIN_LIMIT
        && report.maximum_action_ev_error_bb <= EV_TOLERANCE
        && max_sum_error <= 1e-6
        && report.zero_sum_residual_bb <= 1e-5;
    Ok(report)
}

fn sparse_ranges(ranges: &[Vec<f64>; 2]) -> [Vec<(usize, f64)>; 2] {
    std::array::from_fn(|p| {
        ranges[p]
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, w)| *w > 0.0)
            .collect()
    })
}

pub fn solve(input: SpotInput) -> Result<SolvedSpot, String> {
    let config = config(&input)?;
    let turn_root = input.board.len() == 4;
    let policies = if turn_root {
        let mut solver = TurnRiverSolver::new(TurnRiverSolveConfig {
            game: config.game.clone(),
            state: config.state.clone(),
            iterations: input.iterations,
            averaging_delay: input.iterations / 10,
            river_refinement_iterations: 0,
            regret_matching_plus: false,
        })?;
        solver.train();
        solver.policy_strategies_with_action_values()
    } else {
        let mut solver = RiverSolver::new(RiverSolveConfig {
            game: config.game.clone(),
            state: config.state.clone(),
            iterations: input.iterations,
            averaging_delay: input.iterations / 10,
        })?;
        solver.train();
        solver.policy_strategies_with_action_values()
    };
    let map = policies
        .iter()
        .map(|p| (p.public_history.clone(), p))
        .collect();
    let mut nodes = Vec::new();
    export_nodes(
        config.state.game_state(),
        config.state.board.clone(),
        turn_root,
        &config.game,
        config.state.ranges.clone(),
        &map,
        &mut nodes,
    )?;
    let mut spot = SolvedSpot {
        schema: if turn_root { TURN_SCHEMA } else { SCHEMA }.into(),
        input_sha256: format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&input).map_err(|e| e.to_string())?)
        ),
        input,
        root_node_id: node_id(&config.state.public_history),
        ranges: sparse_ranges(&config.state.ranges),
        nodes,
        verification: Verification::default(),
    };
    spot.verification = verify(&spot)?;
    Ok(spot)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn input() -> SpotInput {
        SpotInput {
            id: "river-test".into(),
            title: "River test".into(),
            description: "Declared test ranges".into(),
            range_assumptions: "Illustrative equal-weight combos".into(),
            board: vec![43, 30, 16, 1, 3],
            depth_bb: 50.0,
            pot_bb: 12.0,
            range_classes: [
                vec!["QQ".into(), "AKs".into()],
                vec!["99".into(), "JTs".into()],
            ],
            range_class_weights: None,
            bet_pot_fractions: vec![0.5, 1.0],
            raise_pot_fractions: vec![1.0],
            raise_cap: 1,
            iterations: 100,
        }
    }
    #[test]
    fn exported_river_is_complete_and_direct_values_match() {
        let spot = solve(input()).unwrap();
        assert!(spot.nodes.len() > 4);
        assert!(spot.verification.maximum_action_ev_error_bb < EV_TOLERANCE);
        assert!(spot.verification.zero_sum_residual_bb < 1e-5);
        assert_eq!(
            serde_json::to_vec(&spot).unwrap(),
            serde_json::to_vec(&solve(input()).unwrap()).unwrap()
        );
    }
    #[test]
    fn corrupt_frozen_rows_and_missing_edges_fail_verification() {
        let mut spot = solve(input()).unwrap();
        spot.nodes[0].rows[0].probabilities[0] = -1.0;
        assert!(verify(&spot).is_err());
        let mut spot = solve(input()).unwrap();
        spot.nodes.pop();
        assert!(verify(&spot).is_err());
        let mut spot = solve(input()).unwrap();
        for node in &mut spot.nodes {
            for row in &mut node.rows {
                if let Some(evs) = &mut row.ev_bb {
                    evs[0] += 1.0;
                }
            }
        }
        assert!(!verify(&spot).unwrap().accepted);
    }
    #[test]
    fn malformed_ranges_and_board_are_rejected() {
        let mut value = input();
        value.range_classes[0].push("nonsense".into());
        assert!(solve(value).is_err());
        let mut value = input();
        value.board[1] = value.board[0];
        assert!(solve(value).is_err());
    }
    #[test]
    fn weighted_ranges_are_normalized_and_verified_without_changing_legacy_defaults() {
        let original = solve(input()).unwrap();
        assert!(serde_json::to_value(&original.input)
            .unwrap()
            .get("rangeClassWeights")
            .is_none());
        let mut equal = input();
        equal.range_class_weights = Some([vec![1.0, 1.0], vec![1.0, 1.0]]);
        let equal = solve(equal).unwrap();
        assert_eq!(original.ranges, equal.ranges);
        assert_eq!(
            serde_json::to_vec(&original.nodes).unwrap(),
            serde_json::to_vec(&equal.nodes).unwrap()
        );

        let mut weighted = input();
        weighted.range_class_weights = Some([vec![1.0, 0.25], vec![0.5, 1.0]]);
        let mut spot = solve(weighted).unwrap();
        for (combo, weight) in &spot.ranges[0] {
            let expected = if all_combos()[*combo].label() == "QQ" {
                0.25
            } else {
                0.0625
            };
            assert!((weight - expected).abs() < 1e-12);
        }
        assert!(spot.verification.maximum_action_ev_error_bb < EV_TOLERANCE);
        assert!(spot.verification.zero_sum_residual_bb < 1e-5);
        assert_eq!(
            serde_json::to_vec(&spot).unwrap(),
            serde_json::to_vec(&solve(spot.input.clone()).unwrap()).unwrap()
        );
        // A new prior cannot borrow an earlier artifact's frozen rows/identity.
        spot.input.range_class_weights.as_mut().unwrap()[0][0] = 0.5;
        assert!(verify(&spot).is_err());
    }
    #[test]
    fn invalid_class_weights_fail_closed() {
        let mut null = serde_json::to_value(input()).unwrap();
        null["rangeClassWeights"] = serde_json::Value::Null;
        assert!(serde_json::from_value::<SpotInput>(null).is_err());
        for bad in [
            vec![],
            vec![1.0],
            vec![1.0, 0.0],
            vec![1.0, -0.1],
            vec![1.0, 1.1],
            vec![1.0, f64::NAN],
            vec![1.0, f64::INFINITY],
            vec![1.0, f64::from_bits(1)],
        ] {
            let mut value = input();
            value.range_class_weights = Some([bad, vec![1.0, 1.0]]);
            assert!(solve(value).is_err());
        }
    }
    #[test]
    fn complete_turn_game_exports_every_river_and_detects_missing_chance() {
        let mut value = input();
        value.board.pop();
        value.pot_bb = 32.0;
        value.iterations = 20;
        let mut spot = solve(value).unwrap();
        assert_eq!(spot.schema, TURN_SCHEMA);
        assert_eq!(
            spot.verification.river_outcomes,
            Some(spot.verification.compatible_deals * 44)
        );
        assert!(spot.verification.maximum_action_ev_error_bb < EV_TOLERANCE);
        assert!(spot
            .nodes
            .iter()
            .any(|n| n.board.as_ref().unwrap().len() == 4));
        assert!(spot
            .nodes
            .iter()
            .any(|n| n.board.as_ref().unwrap().len() == 5));
        let chance = spot
            .nodes
            .iter_mut()
            .flat_map(|n| &mut n.actions)
            .find_map(|a| a.next_by_river.as_mut())
            .unwrap();
        assert_eq!(chance.len(), 48);
        chance.pop();
        assert!(verify(&spot).is_err());
    }
    #[test]
    fn verifier_checks_served_sizes_and_known_showdown_ordering() {
        let mut spot = solve(input()).unwrap();
        let wager = spot
            .nodes
            .iter_mut()
            .flat_map(|n| &mut n.actions)
            .find(|a| a.amount_to_bb.is_some())
            .unwrap();
        wager.amount_to_bb = Some(999.0);
        assert!(verify(&spot).is_err());
        let mut known = input();
        known.iterations = 300;
        known.range_classes = [vec!["AA".into()], vec!["KK".into()]];
        let spot = solve(known).unwrap();
        assert!(spot.verification.accepted);
        assert!((spot.verification.profile_value_bb[0] - 6.0).abs() < 0.05);
        assert!((spot.verification.best_response_value_bb[1] + 6.0).abs() < 0.05);
    }
}
