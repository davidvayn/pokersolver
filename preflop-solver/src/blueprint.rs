//! Coarse heads-up no-limit hold'em blueprint trained with external-sampling
//! Monte Carlo Discounted CFR.
//!
//! This is intentionally a research-grade approximation, not a claim of a
//! solved 100bb game. It samples exact cards while merging strategically
//! similar private/public states into reusable information sets.

use crate::cards::{all_combos, Combo};
use crate::evaluator::evaluate;
use crate::rng::SplitMix64;
use flate2::read::GzDecoder;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

pub mod neural;
pub mod preflop;
pub mod public_belief;
pub mod response;

/// Read an immutable JSON artifact, accepting either plain JSON or a gzip
/// transport wrapper. Hashes are intentionally computed by callers over the
/// decoded JSON bytes so compression cannot change a model's pinned identity.
fn read_json_artifact(path: &Path) -> Result<Vec<u8>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    if path.extension().and_then(|extension| extension.to_str()) != Some("gz") {
        return Ok(bytes);
    }
    let mut decoded = Vec::new();
    GzDecoder::new(bytes.as_slice()).read_to_end(&mut decoded)?;
    Ok(decoded)
}

const MODEL_BINARY_MAGIC: &[u8; 8] = b"PKRMODL2";
const MODEL_BINARY_HEADER_BYTES: usize = 8 + 32 + 32 + 8;
const BLUEPRINT_CHECKPOINT_SCHEMA_VERSION: u32 = 3;

fn model_binary_path(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.bin", path.display()))
}

/// Load a typed immutable model. A verified binary sidecar avoids gzip and
/// JSON parsing while preserving the SHA-256 identity of the canonical JSON.
fn read_model_artifact<T: DeserializeOwned>(path: &Path) -> Result<(T, String), Box<dyn Error>> {
    let binary_path = model_binary_path(path);
    if binary_path.is_file() {
        let bytes = fs::read(&binary_path)?;
        if bytes.len() < MODEL_BINARY_HEADER_BYTES
            || &bytes[..MODEL_BINARY_MAGIC.len()] != MODEL_BINARY_MAGIC
        {
            return Err(format!("invalid model binary header: {}", binary_path.display()).into());
        }
        let canonical_digest = &bytes[8..40];
        let expected_payload_digest = &bytes[40..72];
        let payload_len = u64::from_le_bytes(bytes[72..80].try_into()?) as usize;
        let payload = &bytes[80..];
        if payload.len() != payload_len
            || Sha256::digest(payload).as_slice() != expected_payload_digest
        {
            return Err(format!("corrupt model binary payload: {}", binary_path.display()).into());
        }
        let value = rmp_serde::from_slice(payload)?;
        return Ok((
            value,
            canonical_digest
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect(),
        ));
    }
    let bytes = read_json_artifact(path)?;
    let digest = format!("{:x}", Sha256::digest(&bytes));
    Ok((serde_json::from_slice(&bytes)?, digest))
}

/// Compile an adjacent, versioned binary sidecar from canonical JSON. The
/// sidecar carries both the canonical JSON identity and its own payload hash.
fn write_model_binary_cache<T>(path: &Path) -> Result<PathBuf, Box<dyn Error>>
where
    T: DeserializeOwned + Serialize,
{
    let json = read_json_artifact(path)?;
    let value: T = serde_json::from_slice(&json)?;
    let payload = rmp_serde::to_vec_named(&value)?;
    let mut bytes = Vec::with_capacity(MODEL_BINARY_HEADER_BYTES + payload.len());
    bytes.extend_from_slice(MODEL_BINARY_MAGIC);
    bytes.extend_from_slice(&Sha256::digest(&json));
    bytes.extend_from_slice(&Sha256::digest(&payload));
    bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&payload);
    let target = model_binary_path(path);
    let temporary = target.with_extension("tmp");
    fs::write(&temporary, bytes)?;
    fs::rename(temporary, &target)?;
    Ok(target)
}

const EPSILON: f64 = 1e-9;
const MODEL: &str = "hu-abstracted-external-sampling-dcfr-trajectory-v3";
const SOLVER_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Street {
    Preflop,
    Flop,
    Turn,
    River,
}

impl Street {
    fn next(self) -> Option<Self> {
        match self {
            Self::Preflop => Some(Self::Flop),
            Self::Flop => Some(Self::Turn),
            Self::Turn => Some(Self::River),
            Self::River => None,
        }
    }

    fn board_len(self) -> usize {
        match self {
            Self::Preflop => 0,
            Self::Flop => 3,
            Self::Turn => 4,
            Self::River => 5,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Position {
    ButtonSmallBlind,
    BigBlind,
}

impl Position {
    fn for_player(player: usize) -> Self {
        match player {
            0 => Self::ButtonSmallBlind,
            1 => Self::BigBlind,
            _ => unreachable!("heads-up player index"),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ActionAbstraction {
    pub open_sizes_bb: Vec<f64>,
    pub limp_raise_sizes_bb: Vec<f64>,
    pub three_bet_sizes_bb: Vec<f64>,
    pub four_bet_sizes_bb: Vec<f64>,
    pub deeper_raise_pot_fractions: Vec<f64>,
    pub preflop_raise_cap: u8,
    pub flop_bet_pot_fractions: Vec<f64>,
    pub turn_river_bet_pot_fractions: Vec<f64>,
    pub postflop_raise_pot_fractions: Vec<f64>,
    pub postflop_raise_cap: u8,
    pub include_all_in: bool,
}

impl Default for ActionAbstraction {
    fn default() -> Self {
        Self {
            open_sizes_bb: vec![2.0, 2.5, 3.0, 4.0, 5.0],
            limp_raise_sizes_bb: vec![3.0, 4.0, 5.0],
            three_bet_sizes_bb: vec![7.5, 9.0, 11.0],
            four_bet_sizes_bb: vec![18.0, 22.0, 26.0],
            deeper_raise_pot_fractions: vec![0.75, 1.0, 1.25],
            preflop_raise_cap: 4,
            // A compact continuation tree leaves enough repeated samples for
            // preflop decisions. Callers can opt into richer grids offline.
            flop_bet_pot_fractions: vec![1.0 / 3.0, 0.75, 1.25],
            turn_river_bet_pot_fractions: vec![0.5, 1.0],
            postflop_raise_pot_fractions: vec![1.0],
            postflop_raise_cap: 1,
            include_all_in: true,
        }
    }
}

impl ActionAbstraction {
    pub fn compact_serving_candidate() -> Self {
        Self {
            open_sizes_bb: vec![2.0, 2.5, 3.0],
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct BlueprintConfig {
    pub small_blind_bb: f64,
    pub big_blind_bb: f64,
    pub effective_stack_bb: f64,
    pub iterations: u64,
    pub max_information_sets: usize,
    pub seed: u64,
    pub averaging_delay: u64,
    pub export_postflop_strategies: bool,
    pub recall_mode: RecallMode,
    pub dcfr: DcfrParameters,
    pub evaluation_controls: EvaluationControls,
    pub hand_abstraction: HandAbstraction,
    pub showdown_evaluation: ShowdownEvaluation,
    pub action_abstraction: ActionAbstraction,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecallMode {
    CurrentStreet,
    Trajectory,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DcfrParameters {
    pub positive_regret_exponent: f64,
    pub negative_regret_exponent: f64,
    pub strategy_exponent: f64,
}

impl Default for DcfrParameters {
    fn default() -> Self {
        Self {
            positive_regret_exponent: 1.5,
            negative_regret_exponent: 0.0,
            strategy_exponent: 2.0,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EvaluationControls {
    pub held_out_deals: u64,
    pub held_out_seed: u64,
    pub root_deviation_samples_per_class: u64,
    pub root_deviation_seed: u64,
    pub action_value_deals: u64,
    pub action_value_seed: u64,
}

impl Default for EvaluationControls {
    fn default() -> Self {
        Self {
            held_out_deals: 10_000,
            held_out_seed: 0xd1b5_4a32_d192_ed03,
            root_deviation_samples_per_class: 256,
            root_deviation_seed: 0xa24b_aed4_963e_e407,
            action_value_deals: 10_000,
            action_value_seed: 0x8a5c_d789_635d_2dff,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct HandAbstraction {
    pub distribution_samples: u32,
    pub equity_bins: u8,
    pub potential_bins: u8,
}

impl Default for HandAbstraction {
    fn default() -> Self {
        Self {
            distribution_samples: 128,
            equity_bins: 10,
            potential_bins: 3,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ShowdownEvaluation {
    pub preflop_runout_samples: u32,
    pub flop_runout_samples: u32,
    pub exact_turn_rivers: bool,
}

impl Default for ShowdownEvaluation {
    fn default() -> Self {
        Self {
            preflop_runout_samples: 256,
            flop_runout_samples: 128,
            exact_turn_rivers: true,
        }
    }
}

impl Default for BlueprintConfig {
    fn default() -> Self {
        Self {
            small_blind_bb: 0.5,
            big_blind_bb: 1.0,
            effective_stack_bb: 100.0,
            iterations: 100_000,
            max_information_sets: 5_000_000,
            seed: 1,
            averaging_delay: 1_000,
            export_postflop_strategies: false,
            recall_mode: RecallMode::Trajectory,
            dcfr: DcfrParameters::default(),
            evaluation_controls: EvaluationControls::default(),
            hand_abstraction: HandAbstraction::default(),
            showdown_evaluation: ShowdownEvaluation::default(),
            action_abstraction: ActionAbstraction::default(),
        }
    }
}

impl BlueprintConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !(self.small_blind_bb > 0.0
            && self.big_blind_bb > self.small_blind_bb
            && self.effective_stack_bb > self.big_blind_bb)
        {
            return Err("blinds and effective stack must satisfy 0 < SB < BB < stack".to_owned());
        }
        if self.iterations == 0 {
            return Err("iterations must be positive".to_owned());
        }
        if self.max_information_sets == 0 {
            return Err("max information sets must be positive".to_owned());
        }
        if self.averaging_delay >= self.iterations {
            return Err("averaging delay must be smaller than iterations".to_owned());
        }
        if self.evaluation_controls.held_out_deals == 0 {
            return Err("held-out deals must be positive".to_owned());
        }
        if self.evaluation_controls.root_deviation_samples_per_class == 0 {
            return Err("root-deviation samples per class must be positive".to_owned());
        }
        if self.evaluation_controls.action_value_deals == 0 {
            return Err("action-value evaluation deals must be positive".to_owned());
        }
        if !self.dcfr.positive_regret_exponent.is_finite()
            || !self.dcfr.negative_regret_exponent.is_finite()
            || !self.dcfr.strategy_exponent.is_finite()
            || self.dcfr.positive_regret_exponent < 0.0
            || self.dcfr.negative_regret_exponent < 0.0
            || self.dcfr.strategy_exponent < 0.0
        {
            return Err("DCFR exponents must be finite and non-negative".to_owned());
        }
        if self.hand_abstraction.distribution_samples == 0
            || self.hand_abstraction.equity_bins == 0
            || self.hand_abstraction.potential_bins == 0
        {
            return Err("hand-abstraction sampling and bins must be positive".to_owned());
        }
        if self.showdown_evaluation.preflop_runout_samples == 0
            || self.showdown_evaluation.flop_runout_samples == 0
        {
            return Err("showdown runout sample counts must be positive".to_owned());
        }
        let abstraction = &self.action_abstraction;
        let preflop_grids = [
            &abstraction.open_sizes_bb,
            &abstraction.limp_raise_sizes_bb,
            &abstraction.three_bet_sizes_bb,
            &abstraction.four_bet_sizes_bb,
            &abstraction.deeper_raise_pot_fractions,
        ];
        if preflop_grids.iter().any(|grid| {
            grid.is_empty()
                || grid.iter().any(|size| !size.is_finite() || *size <= 0.0)
                || grid.windows(2).any(|window| window[0] >= window[1])
        }) {
            return Err(
                "preflop action grids must contain strictly increasing positive sizes".to_owned(),
            );
        }
        if abstraction.open_sizes_bb[0] <= self.big_blind_bb
            || abstraction.limp_raise_sizes_bb[0] <= self.big_blind_bb
            || abstraction.three_bet_sizes_bb[0]
                <= *abstraction.open_sizes_bb.last().expect("nonempty")
            || abstraction.four_bet_sizes_bb[0]
                <= *abstraction.three_bet_sizes_bb.last().expect("nonempty")
        {
            return Err("preflop raise-to grids overlap or start below a legal raise".to_owned());
        }
        if abstraction.preflop_raise_cap == 0 || abstraction.postflop_raise_cap == 0 {
            return Err("raise caps must be positive".to_owned());
        }
        for fraction in abstraction
            .flop_bet_pot_fractions
            .iter()
            .chain(abstraction.turn_river_bet_pot_fractions.iter())
        {
            if !fraction.is_finite() || *fraction <= 0.0 {
                return Err("postflop bet fractions must be positive".to_owned());
            }
        }
        if abstraction.flop_bet_pot_fractions.is_empty()
            || abstraction.turn_river_bet_pot_fractions.is_empty()
            || abstraction.postflop_raise_pot_fractions.is_empty()
            || abstraction
                .postflop_raise_pot_fractions
                .iter()
                .any(|fraction| !fraction.is_finite() || *fraction <= 0.0)
        {
            return Err("postflop action grids and raise fraction must be positive".to_owned());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct RunControl {
    pub checkpoint_path: Option<String>,
    pub checkpoint_every: u64,
    pub resume_path: Option<String>,
}

#[derive(Clone, Debug)]
struct Deal {
    holes: [[u8; 2]; 2],
    board: [u8; 5],
    hand_bucket_cache: RefCell<BTreeMap<(usize, usize), String>>,
    showdown_equity_cache: RefCell<BTreeMap<usize, f64>>,
}

impl Deal {
    fn sample(rng: &mut SplitMix64) -> Self {
        let mut deck = [0u8; 52];
        for (index, card) in deck.iter_mut().enumerate() {
            *card = index as u8;
        }
        for index in 0..9 {
            let swap = index + rng.index(52 - index);
            deck.swap(index, swap);
        }
        Self::from_sampled_cards(
            [[deck[0], deck[1]], [deck[2], deck[3]]],
            [deck[4], deck[5], deck[6], deck[7], deck[8]],
        )
    }

    fn from_sampled_cards(holes: [[u8; 2]; 2], board: [u8; 5]) -> Self {
        Self {
            holes,
            board,
            hand_bucket_cache: RefCell::new(BTreeMap::new()),
            showdown_equity_cache: RefCell::new(BTreeMap::new()),
        }
    }

    #[cfg(test)]
    fn from_cards(holes: [[u8; 2]; 2], board: [u8; 5]) -> Self {
        Self::from_sampled_cards(holes, board)
    }

    fn hand_bucket(&self, player: usize, street: Street, abstraction: &HandAbstraction) -> String {
        let key = (player, street.board_len());
        if let Some(bucket) = self.hand_bucket_cache.borrow().get(&key) {
            return bucket.clone();
        }
        let bucket = postflop_hand_bucket(self, player, street, abstraction);
        self.hand_bucket_cache
            .borrow_mut()
            .insert(key, bucket.clone());
        bucket
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Terminal {
    Fold { winner: usize },
    Showdown,
}

#[derive(Clone, Debug, PartialEq)]
enum ActionKind {
    Fold,
    Check,
    Call,
    RaiseTo(f64),
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TrajectoryActionKind {
    Fold,
    Check,
    Call,
    Bet,
    Raise,
    AllIn,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct TrajectoryAction {
    actor: usize,
    street: Street,
    kind: TrajectoryActionKind,
    amount_bb: f64,
    amount_to_bb: Option<f64>,
    pot_after_bb: f64,
}

#[derive(Clone, Debug, PartialEq)]
struct LegalAction {
    label: String,
    kind: ActionKind,
}

#[derive(Clone, Debug)]
struct GameState {
    street: Street,
    actor: usize,
    invested: [f64; 2],
    street_invested: [f64; 2],
    last_full_raise: f64,
    aggressions: u8,
    checks: u8,
    raise_reopened: bool,
    public_history: Vec<String>,
    trajectory: Vec<TrajectoryAction>,
    terminal: Option<Terminal>,
}

impl GameState {
    fn initial(config: &BlueprintConfig) -> Self {
        Self {
            street: Street::Preflop,
            actor: 0,
            invested: [config.small_blind_bb, config.big_blind_bb],
            street_invested: [config.small_blind_bb, config.big_blind_bb],
            last_full_raise: config.big_blind_bb,
            aggressions: 0,
            checks: 0,
            raise_reopened: true,
            public_history: vec![format!(
                "blinds:{:.3}/{:.3}",
                config.small_blind_bb, config.big_blind_bb
            )],
            trajectory: Vec::new(),
            terminal: None,
        }
    }

    fn pot(&self) -> f64 {
        self.invested[0] + self.invested[1]
    }

    fn remaining(&self, player: usize, config: &BlueprintConfig) -> f64 {
        (config.effective_stack_bb - self.invested[player]).max(0.0)
    }

    fn to_call(&self) -> f64 {
        let opponent = 1 - self.actor;
        (self.street_invested[opponent] - self.street_invested[self.actor]).max(0.0)
    }

    fn legal_actions(&self, config: &BlueprintConfig) -> Vec<LegalAction> {
        if self.terminal.is_some() {
            return Vec::new();
        }
        let abstraction = &config.action_abstraction;
        let player = self.actor;
        let opponent = 1 - player;
        let remaining = self.remaining(player, config);
        let opponent_remaining = self.remaining(opponent, config);
        let to_call = self.to_call();
        let current_bet = self.street_invested[opponent].max(self.street_invested[player]);
        let mut actions = Vec::new();

        if to_call > EPSILON {
            actions.push(LegalAction {
                label: "fold".to_owned(),
                kind: ActionKind::Fold,
            });
            actions.push(LegalAction {
                label: if to_call + EPSILON >= remaining {
                    "call_all_in".to_owned()
                } else if self.street == Street::Preflop && self.aggressions == 0 && player == 0 {
                    "limp".to_owned()
                } else {
                    "call".to_owned()
                },
                kind: ActionKind::Call,
            });
        } else {
            actions.push(LegalAction {
                label: "check".to_owned(),
                kind: ActionKind::Check,
            });
        }

        let opponent_all_in = opponent_remaining <= EPSILON;
        let cap = if self.street == Street::Preflop {
            abstraction.preflop_raise_cap
        } else {
            // One postflop raise means a bet plus one subsequent raise.
            abstraction.postflop_raise_cap + 1
        };
        if remaining > to_call + EPSILON
            && !opponent_all_in
            && self.raise_reopened
            && self.aggressions < cap
        {
            let targets = if self.street == Street::Preflop {
                self.preflop_raise_targets(config)
            } else {
                self.postflop_raise_targets(config)
            };
            let max_to = self.street_invested[player] + remaining;
            let minimum_to = current_bet + self.last_full_raise.max(config.big_blind_bb);
            let mut seen = BTreeSet::new();
            for target in targets {
                // The browser engine and canonical policy hash both use
                // milliblind chip accounting. Keep traversal semantics at the
                // same boundary instead of retaining invisible sub-milliblind
                // pot-fraction differences.
                let target = quantize(target.min(max_to), 0.001);
                if target + EPSILON < minimum_to || target >= max_to - EPSILON {
                    continue;
                }
                let milli = (target * 1000.0).round() as i64;
                if seen.insert(milli) {
                    actions.push(LegalAction {
                        label: format!(
                            "{}_to_{:.3}bb",
                            if current_bet <= EPSILON {
                                "bet"
                            } else {
                                "raise"
                            },
                            target
                        ),
                        kind: ActionKind::RaiseTo(target),
                    });
                }
            }
            if abstraction.include_all_in {
                let max_to = quantize(max_to, 0.001);
                actions.push(LegalAction {
                    label: format!(
                        "{}_all_in_to_{:.3}bb",
                        if current_bet <= EPSILON {
                            "bet"
                        } else {
                            "raise"
                        },
                        max_to
                    ),
                    kind: ActionKind::RaiseTo(max_to),
                });
            }
        }
        actions
    }

    fn preflop_raise_targets(&self, config: &BlueprintConfig) -> Vec<f64> {
        let abstraction = &config.action_abstraction;
        match self.aggressions {
            0 if self.actor == 0 => abstraction.open_sizes_bb.clone(),
            0 => abstraction.limp_raise_sizes_bb.clone(),
            1 => abstraction.three_bet_sizes_bb.clone(),
            2 => abstraction.four_bet_sizes_bb.clone(),
            _ => {
                let current = self.street_invested[0].max(self.street_invested[1]);
                let pot_after_call = self.pot() + self.to_call();
                abstraction
                    .deeper_raise_pot_fractions
                    .iter()
                    .map(|fraction| current + pot_after_call * fraction)
                    .collect()
            }
        }
    }

    fn postflop_raise_targets(&self, config: &BlueprintConfig) -> Vec<f64> {
        let fractions = if self.street == Street::Flop {
            &config.action_abstraction.flop_bet_pot_fractions
        } else {
            &config.action_abstraction.turn_river_bet_pot_fractions
        };
        let player_commit = self.street_invested[self.actor];
        let opponent_commit = self.street_invested[1 - self.actor];
        if self.to_call() <= EPSILON {
            return fractions
                .iter()
                .map(|fraction| player_commit + self.pot() * fraction)
                .collect();
        }

        let pot_after_call = self.pot() + self.to_call();
        config
            .action_abstraction
            .postflop_raise_pot_fractions
            .iter()
            .map(|fraction| opponent_commit + pot_after_call * fraction)
            .collect()
    }

    fn apply(&self, action: &LegalAction, config: &BlueprintConfig) -> Self {
        let mut next = self.clone();
        let player = self.actor;
        let opponent = 1 - player;
        let before_invested = self.invested[player];
        let before_street_invested = self.street_invested[player];
        let before_highest = self.street_invested[player].max(self.street_invested[opponent]);
        let maximum_to = before_street_invested + self.remaining(player, config);
        next.public_history
            .push(format!("{:?}:p{}:{}", self.street, player, action.label));
        match action.kind {
            ActionKind::Fold => {
                next.terminal = Some(Terminal::Fold { winner: opponent });
            }
            ActionKind::Check => {
                if self.checks == 1 {
                    next.close_street(config);
                } else {
                    next.actor = opponent;
                    next.checks = 1;
                }
            }
            ActionKind::Call => {
                let paid = self.to_call().min(self.remaining(player, config));
                next.street_invested[player] += paid;
                next.invested[player] += paid;
                let is_opening_limp =
                    self.street == Street::Preflop && self.aggressions == 0 && player == 0;
                if is_opening_limp {
                    next.actor = opponent;
                    // The completion is the first passive action in this
                    // round, so a BB check closes preflop.
                    next.checks = 1;
                } else {
                    next.close_street(config);
                }
            }
            ActionKind::RaiseTo(target) => {
                let old_bet = self.street_invested[player].max(self.street_invested[opponent]);
                let delta = target - self.street_invested[player];
                next.street_invested[player] = target;
                next.invested[player] += delta;
                let raise_increment = target - old_bet;
                let is_full_raise = raise_increment + EPSILON >= self.last_full_raise;
                if is_full_raise {
                    next.last_full_raise = raise_increment;
                }
                next.raise_reopened = is_full_raise;
                next.aggressions += 1;
                next.checks = 0;
                next.actor = opponent;
            }
        }
        let paid = (next.invested[player] - before_invested).max(0.0);
        let (kind, amount_to_bb) = match action.kind {
            ActionKind::Fold => (TrajectoryActionKind::Fold, None),
            ActionKind::Check => (TrajectoryActionKind::Check, None),
            ActionKind::Call => (TrajectoryActionKind::Call, None),
            ActionKind::RaiseTo(target) if (target - maximum_to).abs() <= EPSILON => {
                (TrajectoryActionKind::AllIn, Some(target))
            }
            ActionKind::RaiseTo(target) if before_highest <= EPSILON => {
                (TrajectoryActionKind::Bet, Some(target))
            }
            ActionKind::RaiseTo(target) => (TrajectoryActionKind::Raise, Some(target)),
        };
        next.trajectory.push(TrajectoryAction {
            actor: player,
            street: self.street,
            kind,
            amount_bb: paid,
            amount_to_bb,
            pot_after_bb: next.pot(),
        });
        next
    }

    fn close_street(&mut self, config: &BlueprintConfig) {
        let equal = (self.street_invested[0] - self.street_invested[1]).abs() <= EPSILON;
        assert!(equal, "a street closes only after equal commitments");
        if self.street == Street::River
            || self.remaining(0, config) <= EPSILON
            || self.remaining(1, config) <= EPSILON
        {
            self.terminal = Some(Terminal::Showdown);
            return;
        }
        self.street = self.street.next().expect("river handled above");
        self.actor = 1;
        self.street_invested = [0.0, 0.0];
        self.last_full_raise = config.big_blind_bb;
        self.aggressions = 0;
        self.checks = 0;
        self.raise_reopened = true;
        self.public_history.push(format!("deal:{:?}", self.street));
    }

    fn utility_p0(&self, deal: &Deal, config: &BlueprintConfig) -> f64 {
        match self.terminal.as_ref().expect("terminal utility") {
            Terminal::Fold { winner } => {
                if *winner == 0 {
                    self.invested[1]
                } else {
                    -self.invested[0]
                }
            }
            Terminal::Showdown => {
                let equity = conditional_showdown_equity_p0(deal, self.street, config);
                equity * self.invested[1] - (1.0 - equity) * self.invested[0]
            }
        }
    }
}

fn conditional_showdown_equity_p0(deal: &Deal, street: Street, config: &BlueprintConfig) -> f64 {
    let board_len = street.board_len();
    if let Some(equity) = deal.showdown_equity_cache.borrow().get(&board_len) {
        return *equity;
    }
    let visible_board = &deal.board[..board_len];
    let equity = if street == Street::River {
        showdown_result(&deal.holes, visible_board)
    } else {
        let mut known = [false; 52];
        for card in deal.holes.iter().flatten().chain(visible_board.iter()) {
            known[*card as usize] = true;
        }
        let mut available = (0..52u8)
            .filter(|card| !known[*card as usize])
            .collect::<Vec<_>>();
        if street == Street::Turn && config.showdown_evaluation.exact_turn_rivers {
            available
                .iter()
                .map(|river| {
                    let mut board = visible_board.to_vec();
                    board.push(*river);
                    showdown_result(&deal.holes, &board)
                })
                .sum::<f64>()
                / available.len() as f64
        } else {
            let samples = if street == Street::Preflop {
                config.showdown_evaluation.preflop_runout_samples
            } else {
                config.showdown_evaluation.flop_runout_samples
            } as usize;
            let missing = 5 - board_len;
            let mut rng = SplitMix64::new(showdown_seed(&deal.holes, visible_board));
            let mut total = 0.0;
            for _ in 0..samples {
                for index in 0..missing {
                    let swap = index + rng.index(available.len() - index);
                    available.swap(index, swap);
                }
                let mut board = visible_board.to_vec();
                board.extend_from_slice(&available[..missing]);
                total += showdown_result(&deal.holes, &board);
            }
            total / samples as f64
        }
    };
    deal.showdown_equity_cache
        .borrow_mut()
        .insert(board_len, equity);
    equity
}

fn showdown_result(holes: &[[u8; 2]; 2], board: &[u8]) -> f64 {
    debug_assert_eq!(board.len(), 5);
    let mut first_cards = Vec::with_capacity(7);
    first_cards.extend_from_slice(&holes[0]);
    first_cards.extend_from_slice(board);
    let mut second_cards = Vec::with_capacity(7);
    second_cards.extend_from_slice(&holes[1]);
    second_cards.extend_from_slice(board);
    match evaluate(&first_cards).cmp(&evaluate(&second_cards)) {
        std::cmp::Ordering::Greater => 1.0,
        std::cmp::Ordering::Equal => 0.5,
        std::cmp::Ordering::Less => 0.0,
    }
}

fn showdown_seed(holes: &[[u8; 2]; 2], board: &[u8]) -> u64 {
    let mut bytes = Vec::with_capacity(4 + board.len());
    for hole in holes {
        let mut cards = *hole;
        cards.sort_unstable();
        bytes.extend_from_slice(&cards);
        bytes.push(0xfe);
    }
    let mut canonical_board = board.to_vec();
    canonical_board.sort_unstable();
    bytes.extend(canonical_board);
    stable_hash(&bytes)
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct InfoSetDescriptor {
    pub actor: Position,
    pub street: Street,
    pub hand_bucket_trajectory: Vec<String>,
    pub public_bucket_trajectory: Vec<String>,
    pub public_history: Vec<String>,
    pub pot_bb: f64,
    pub to_call_bb: f64,
    pub effective_stack_remaining_bb: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct NodeDescriptor {
    actor: Position,
    street: Street,
    hand_bucket_trajectory: Vec<String>,
    public_bucket_trajectory: Vec<String>,
    public_history_id: u64,
    pot_bb: f64,
    to_call_bb: f64,
    effective_stack_remaining_bb: f64,
}

impl NodeDescriptor {
    fn canonicalize_money(&mut self) {
        self.pot_bb = quantize(self.pot_bb, 0.001);
        self.to_call_bb = quantize(self.to_call_bb, 0.001);
        self.effective_stack_remaining_bb = quantize(self.effective_stack_remaining_bb, 0.001);
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Node {
    descriptor: NodeDescriptor,
    action_labels: Vec<String>,
    regrets: Vec<f64>,
    strategy_sum: Vec<f64>,
    regret_updates: u64,
    average_visits: u64,
    #[serde(default)]
    last_discount_iteration: u64,
    #[serde(default)]
    last_regret_discount_cumulative_logs: [f64; 2],
}

impl Node {
    fn new(descriptor: NodeDescriptor, actions: &[LegalAction]) -> Self {
        let action_labels = actions
            .iter()
            .map(|action| action.label.clone())
            .collect::<Vec<_>>();
        Self {
            regrets: vec![0.0; action_labels.len()],
            strategy_sum: vec![0.0; action_labels.len()],
            action_labels,
            descriptor,
            regret_updates: 0,
            average_visits: 0,
            last_discount_iteration: 0,
            last_regret_discount_cumulative_logs: [0.0; 2],
        }
    }

    fn current_strategy(&self) -> Vec<f64> {
        let positive = self
            .regrets
            .iter()
            .map(|regret| regret.max(0.0))
            .collect::<Vec<_>>();
        normalize_or_uniform(positive)
    }

    fn average_strategy(&self) -> Vec<f64> {
        normalize_or_uniform(self.strategy_sum.clone())
    }

    fn apply_dcfr_regret_discount(
        &mut self,
        iteration: u64,
        discounts: &BlueprintDiscountAccumulator,
    ) {
        if iteration == 0 || self.last_discount_iteration == iteration {
            return;
        }
        let positive_factor =
            (discounts.cumulative_logs[0] - self.last_regret_discount_cumulative_logs[0]).exp();
        let negative_factor =
            (discounts.cumulative_logs[1] - self.last_regret_discount_cumulative_logs[1]).exp();
        for regret in &mut self.regrets {
            *regret *= if *regret >= 0.0 {
                positive_factor
            } else {
                negative_factor
            };
        }
        self.last_discount_iteration = iteration;
        self.last_regret_discount_cumulative_logs = discounts.cumulative_logs;
    }
}

struct BlueprintDiscountAccumulator {
    parameters: DcfrParameters,
    iteration: u64,
    cumulative_logs: [f64; 2],
}

impl BlueprintDiscountAccumulator {
    fn new(parameters: DcfrParameters) -> Self {
        Self {
            parameters,
            iteration: 0,
            cumulative_logs: [0.0; 2],
        }
    }

    fn advance(&mut self, iteration: u64) {
        assert_eq!(iteration, self.iteration + 1);
        let time = iteration as f64;
        let positive_power = time.powf(self.parameters.positive_regret_exponent);
        let negative_power = time.powf(self.parameters.negative_regret_exponent);
        let factors = [
            positive_power / (positive_power + 1.0),
            negative_power / (negative_power + 1.0),
        ];
        for (cumulative, factor) in self.cumulative_logs.iter_mut().zip(factors) {
            *cumulative += factor.ln();
        }
        self.iteration = iteration;
    }
}

/// DCFR's repeated strategy discount telescopes to a final iteration weight
/// of `t^gamma`. Accumulating that weight directly is essential under external
/// sampling because an information set is not necessarily visited every round.
fn dcfr_strategy_averaging_weight(iteration: u64, parameters: &DcfrParameters) -> f64 {
    (iteration as f64).powf(parameters.strategy_exponent)
}

fn normalize_or_uniform(mut weights: Vec<f64>) -> Vec<f64> {
    let total = weights.iter().sum::<f64>();
    if total > EPSILON {
        for weight in &mut weights {
            *weight /= total;
        }
        weights
    } else {
        let probability = 1.0 / weights.len() as f64;
        weights.fill(probability);
        weights
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct BlueprintCheckpoint {
    schema_version: u32,
    model: String,
    approximate: bool,
    config: BlueprintConfig,
    completed_iterations: u64,
    rng_state: u64,
    #[serde(default)]
    regret_discount_cumulative_logs: [f64; 2],
    terminal_evaluations: u64,
    public_histories: BTreeMap<u64, Vec<String>>,
    nodes: BTreeMap<u64, Node>,
}

#[derive(Serialize)]
struct BlueprintCheckpointRef<'a> {
    schema_version: u32,
    model: &'a str,
    approximate: bool,
    config: &'a BlueprintConfig,
    completed_iterations: u64,
    rng_state: u64,
    regret_discount_cumulative_logs: [f64; 2],
    terminal_evaluations: u64,
    public_histories: &'a BTreeMap<u64, Vec<String>>,
    nodes: &'a BTreeMap<u64, Node>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct StrategyAction {
    pub action: String,
    pub probability: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ExportedInfoSet {
    pub key: String,
    pub actor: Position,
    pub street: Street,
    pub hand_bucket_trajectory: Vec<String>,
    pub public_bucket_trajectory: Vec<String>,
    pub public_history: Vec<String>,
    pub pot_bb: f64,
    pub to_call_bb: f64,
    pub effective_stack_remaining_bb: f64,
    pub actions: Vec<StrategyAction>,
    pub action_values: Vec<ActionValueEstimate>,
    pub best_action: Option<String>,
    pub best_action_ev_bb: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ActionValueEstimate {
    pub action: String,
    pub samples: u64,
    pub mean_ev_bb: f64,
    pub standard_error_bb: f64,
    pub best_action_ev_loss_bb: f64,
    pub low_confidence: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ActionValueEvaluationMetrics {
    pub deals: u64,
    pub evaluated_information_sets: usize,
    pub exported_information_set_coverage: f64,
    pub reach_weighted_standard_error_coverage: f64,
    pub standard_error_threshold_bb: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct HeldOutMetrics {
    pub deals: u64,
    pub button_mean_net_bb: f64,
    pub button_net_standard_error_bb: f64,
    pub fold_terminal_fraction: f64,
    pub showdown_terminal_fraction: f64,
    pub unknown_information_set_fraction: f64,
    pub untrained_information_set_fraction: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PolicyCoverage {
    pub decisions: u64,
    pub unknown_information_set_fraction: f64,
    pub untrained_information_set_fraction: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RootActionValue {
    pub action: String,
    pub samples: u64,
    pub mean_net_bb: f64,
    pub standard_error_bb: f64,
    pub continuation_coverage: PolicyCoverage,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RootClassDeviation {
    pub hand_class: String,
    pub exact_combo_count: usize,
    pub root_policy_trained: bool,
    pub root_policy: Vec<StrategyAction>,
    pub action_values: Vec<RootActionValue>,
    pub chosen_average_ev_bb: f64,
    pub best_action: String,
    pub best_action_ev_bb: f64,
    pub local_deviation_gain_bb: f64,
    pub local_deviation_gain_standard_error_bb: f64,
    pub local_deviation_gain_99pct_lower_bound_bb: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RootLocalDeviation {
    pub kind: String,
    pub samples_per_class: u64,
    pub seed: u64,
    pub classes: Vec<RootClassDeviation>,
    pub aggregate_chosen_average_ev_bb: f64,
    pub aggregate_best_action_ev_bb: f64,
    pub aggregate_local_deviation_gain_bb: f64,
    pub aggregate_local_deviation_gain_standard_error_bb: f64,
    pub aggregate_local_deviation_gain_99pct_lower_bound_bb: f64,
    pub trained_root_combo_fraction: f64,
    pub continuation_coverage: PolicyCoverage,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct BlueprintMetrics {
    pub requested_iterations: u64,
    pub training_iterations: u64,
    pub stopped_early: bool,
    pub stop_reason: Option<String>,
    pub sampled_deals: u64,
    pub terminal_evaluations: u64,
    pub information_sets: usize,
    pub preflop_information_sets: usize,
    pub postflop_information_sets: usize,
    pub trained_information_sets: usize,
    pub exported_information_sets: usize,
    pub clipped_cumulative_regret_per_update_diagnostic_bb: f64,
    pub held_out: HeldOutMetrics,
    pub root_local_deviation: RootLocalDeviation,
    pub action_value_evaluation: ActionValueEvaluationMetrics,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct BlueprintArtifact {
    pub schema_version: u32,
    pub solver_version: String,
    pub artifact_id: String,
    pub config_hash: String,
    pub training_config_hash: String,
    pub model: String,
    pub approximate: bool,
    pub provenance: Vec<String>,
    pub validation: BlueprintValidation,
    pub config: BlueprintConfig,
    pub metrics: BlueprintMetrics,
    pub strategies: Vec<ExportedInfoSet>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct BlueprintValidation {
    pub status: String,
    pub reasons: Vec<String>,
}

struct Trainer {
    config: BlueprintConfig,
    completed_iterations: u64,
    rng: SplitMix64,
    discounts: BlueprintDiscountAccumulator,
    terminal_evaluations: u64,
    public_histories: BTreeMap<u64, Vec<String>>,
    nodes: BTreeMap<u64, Node>,
}

impl Trainer {
    fn fresh(config: BlueprintConfig) -> Self {
        Self {
            rng: SplitMix64::new(config.seed),
            discounts: BlueprintDiscountAccumulator::new(config.dcfr.clone()),
            config,
            completed_iterations: 0,
            terminal_evaluations: 0,
            public_histories: BTreeMap::new(),
            nodes: BTreeMap::new(),
        }
    }

    fn from_checkpoint(
        mut checkpoint: BlueprintCheckpoint,
        target: &BlueprintConfig,
    ) -> Result<Self, String> {
        if checkpoint.schema_version != BLUEPRINT_CHECKPOINT_SCHEMA_VERSION {
            return Err(format!(
                "unsupported checkpoint schema {}; expected {} after the exact sampled-DCFR weighting upgrade",
                checkpoint.schema_version, BLUEPRINT_CHECKPOINT_SCHEMA_VERSION
            ));
        }
        let mut comparable = target.clone();
        comparable.iterations = checkpoint.config.iterations;
        comparable.max_information_sets = checkpoint.config.max_information_sets;
        comparable.evaluation_controls = checkpoint.config.evaluation_controls.clone();
        comparable.export_postflop_strategies = checkpoint.config.export_postflop_strategies;
        if comparable != checkpoint.config {
            return Err(
                "resume configuration differs from checkpoint; only target iterations, max information sets, evaluation controls, and postflop export selection may change".to_owned(),
            );
        }
        if target.iterations < checkpoint.completed_iterations {
            return Err("target iterations precede checkpoint progress".to_owned());
        }
        if checkpoint
            .regret_discount_cumulative_logs
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err("checkpoint has invalid DCFR discount state".to_owned());
        }
        // Reapply the same chip quantization used during traversal. Decimal
        // JSON can otherwise round-trip values such as 29.333 to a neighboring
        // f64 and trip the information-set collision guard after resume.
        for node in checkpoint.nodes.values_mut() {
            node.descriptor.canonicalize_money();
        }
        for (key, node) in &checkpoint.nodes {
            let length = node.action_labels.len();
            if length == 0 || node.regrets.len() != length || node.strategy_sum.len() != length {
                return Err(format!(
                    "checkpoint node {key} has inconsistent action vectors"
                ));
            }
            if node
                .regrets
                .iter()
                .chain(node.strategy_sum.iter())
                .chain(node.last_regret_discount_cumulative_logs.iter())
                .any(|value| !value.is_finite())
            {
                return Err(format!("checkpoint node {key} has non-finite values"));
            }
            if !checkpoint
                .public_histories
                .contains_key(&node.descriptor.public_history_id)
            {
                return Err(format!(
                    "checkpoint node {key} references a missing history"
                ));
            }
        }
        Ok(Self {
            config: target.clone(),
            completed_iterations: checkpoint.completed_iterations,
            rng: SplitMix64::from_state(checkpoint.rng_state),
            discounts: BlueprintDiscountAccumulator {
                parameters: target.dcfr.clone(),
                iteration: checkpoint.completed_iterations,
                cumulative_logs: checkpoint.regret_discount_cumulative_logs,
            },
            terminal_evaluations: checkpoint.terminal_evaluations,
            public_histories: checkpoint.public_histories,
            nodes: checkpoint.nodes,
        })
    }

    fn write_checkpoint(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        let checkpoint = BlueprintCheckpointRef {
            schema_version: BLUEPRINT_CHECKPOINT_SCHEMA_VERSION,
            model: MODEL,
            approximate: true,
            config: &self.config,
            completed_iterations: self.completed_iterations,
            rng_state: self.rng.state(),
            regret_discount_cumulative_logs: self.discounts.cumulative_logs,
            terminal_evaluations: self.terminal_evaluations,
            public_histories: &self.public_histories,
            nodes: &self.nodes,
        };
        write_json_atomic(path, &checkpoint)
    }

    fn train(&mut self, control: &RunControl) -> Result<(), Box<dyn Error>> {
        while self.completed_iterations < self.config.iterations {
            self.discounts.advance(self.completed_iterations + 1);
            let traverser = self.completed_iterations as usize % 2;
            let deal = Deal::sample(&mut self.rng);
            self.external_sampling(GameState::initial(&self.config), &deal, traverser);
            self.completed_iterations += 1;
            if self.nodes.len() >= self.config.max_information_sets {
                break;
            }
            if control.checkpoint_every > 0
                && self
                    .completed_iterations
                    .is_multiple_of(control.checkpoint_every)
            {
                if let Some(path) = &control.checkpoint_path {
                    self.write_checkpoint(Path::new(path))?;
                }
            }
        }
        if let Some(path) = &control.checkpoint_path {
            self.write_checkpoint(Path::new(path))?;
        }
        Ok(())
    }

    fn external_sampling(&mut self, state: GameState, deal: &Deal, traverser: usize) -> f64 {
        if state.terminal.is_some() {
            self.terminal_evaluations += 1;
            let utility = state.utility_p0(deal, &self.config);
            return if traverser == 0 { utility } else { -utility };
        }

        let actions = state.legal_actions(&self.config);
        debug_assert!(!actions.is_empty());
        let (key, descriptor, public_history) = information_set(&state, deal, &self.config);
        match self.public_histories.get(&descriptor.public_history_id) {
            Some(existing) => assert_eq!(
                existing, &public_history,
                "public-history hash collision detected"
            ),
            None => {
                self.public_histories
                    .insert(descriptor.public_history_id, public_history);
            }
        }
        let strategy = {
            let node = self
                .nodes
                .entry(key)
                .or_insert_with(|| Node::new(descriptor.clone(), &actions));
            assert_eq!(
                node.descriptor, descriptor,
                "information-set hash collision detected"
            );
            assert_eq!(
                node.action_labels,
                actions
                    .iter()
                    .map(|action| action.label.clone())
                    .collect::<Vec<_>>(),
                "one abstraction key produced incompatible action sets"
            );
            node.apply_dcfr_regret_discount(self.completed_iterations + 1, &self.discounts);
            node.current_strategy()
        };

        if state.actor == traverser {
            let mut action_values = Vec::with_capacity(actions.len());
            for action in &actions {
                action_values.push(self.external_sampling(
                    state.apply(action, &self.config),
                    deal,
                    traverser,
                ));
            }
            let node_value = strategy
                .iter()
                .zip(&action_values)
                .map(|(probability, value)| probability * value)
                .sum::<f64>();
            let node = self.nodes.get_mut(&key).expect("node inserted");
            for (regret, action_value) in node.regrets.iter_mut().zip(action_values) {
                *regret += action_value - node_value;
            }
            node.regret_updates += 1;
            node_value
        } else {
            // OpenSpiel-style external sampling: opponent actions are sampled
            // directly from sigma. Because behavior == target, no importance
            // ratio is needed. Simple averaging happens at these opponent
            // nodes during the other player's traversal.
            if self.completed_iterations >= self.config.averaging_delay {
                let node = self.nodes.get_mut(&key).expect("node inserted");
                let averaging_weight = dcfr_strategy_averaging_weight(
                    self.completed_iterations + 1,
                    &self.config.dcfr,
                );
                for (sum, probability) in node.strategy_sum.iter_mut().zip(&strategy) {
                    *sum += averaging_weight * probability;
                }
                node.average_visits += 1;
            }
            let selected = sample_index(&strategy, &mut self.rng);
            self.external_sampling(
                state.apply(&actions[selected], &self.config),
                deal,
                traverser,
            )
        }
    }

    fn artifact(&self) -> BlueprintArtifact {
        let mut hash_input = SOLVER_VERSION.as_bytes().to_vec();
        hash_input.extend_from_slice(MODEL.as_bytes());
        hash_input.extend(
            serde_json::to_vec(&self.config).expect("serializable blueprint configuration"),
        );
        let config_hash = stable_hash(&hash_input);
        let training_hash_input = serde_json::to_vec(&serde_json::json!({
            "solver_version": SOLVER_VERSION,
            "model": MODEL,
            "small_blind_bb": self.config.small_blind_bb,
            "big_blind_bb": self.config.big_blind_bb,
            "effective_stack_bb": self.config.effective_stack_bb,
            "iterations": self.config.iterations,
            "max_information_sets": self.config.max_information_sets,
            "seed": self.config.seed,
            "averaging_delay": self.config.averaging_delay,
            "recall_mode": self.config.recall_mode,
            "dcfr": &self.config.dcfr,
            "hand_abstraction": &self.config.hand_abstraction,
            "showdown_evaluation": &self.config.showdown_evaluation,
            "action_abstraction": &self.config.action_abstraction,
        }))
        .expect("serializable training configuration");
        let training_config_hash = stable_hash(&training_hash_input);
        let action_value_pass = evaluate_information_set_actions(&self.config, &self.nodes);
        let strategies = self
            .nodes
            .iter()
            .filter(|(_, node)| {
                node.average_visits > 0
                    && (node.descriptor.street == Street::Preflop
                        || self.config.export_postflop_strategies)
            })
            .map(|(key, node)| {
                let evaluated = action_value_pass.information_sets.get(key);
                let (action_values, best_action, best_action_ev_bb) =
                    finalize_action_values(node, evaluated);
                ExportedInfoSet {
                    key: format!("{key:016x}"),
                    actor: node.descriptor.actor,
                    street: node.descriptor.street,
                    hand_bucket_trajectory: node.descriptor.hand_bucket_trajectory.clone(),
                    public_bucket_trajectory: node.descriptor.public_bucket_trajectory.clone(),
                    public_history: self
                        .public_histories
                        .get(&node.descriptor.public_history_id)
                        .expect("node history interned")
                        .clone(),
                    pot_bb: node.descriptor.pot_bb,
                    to_call_bb: node.descriptor.to_call_bb,
                    effective_stack_remaining_bb: node.descriptor.effective_stack_remaining_bb,
                    actions: node
                        .action_labels
                        .iter()
                        .cloned()
                        .zip(node.average_strategy())
                        .map(|(action, probability)| StrategyAction {
                            action,
                            probability,
                        })
                        .collect(),
                    action_values,
                    best_action,
                    best_action_ev_bb,
                }
            })
            .collect::<Vec<_>>();
        let evaluated_exported = self
            .nodes
            .iter()
            .filter(|(_, node)| {
                node.average_visits > 0
                    && (node.descriptor.street == Street::Preflop
                        || self.config.export_postflop_strategies)
            })
            .filter(|(key, _)| action_value_pass.information_sets.contains_key(key))
            .count();
        let mut evaluated_weight = 0.0;
        let mut precise_weight = 0.0;
        for (key, evaluation) in &action_value_pass.information_sets {
            let Some(node) = self.nodes.get(key) else {
                continue;
            };
            for (probability, accumulator) in
                node.average_strategy().iter().zip(&evaluation.actions)
            {
                let weight = evaluation.visits as f64 * probability;
                evaluated_weight += weight;
                if accumulator.samples >= 2 && accumulator.standard_error() <= 0.02 {
                    precise_weight += weight;
                }
            }
        }
        let action_value_evaluation = ActionValueEvaluationMetrics {
            deals: self.config.evaluation_controls.action_value_deals,
            evaluated_information_sets: action_value_pass.information_sets.len(),
            exported_information_set_coverage: evaluated_exported as f64
                / strategies.len().max(1) as f64,
            reach_weighted_standard_error_coverage: precise_weight / evaluated_weight.max(EPSILON),
            standard_error_threshold_bb: 0.02,
        };
        let total_updates = self
            .nodes
            .values()
            .map(|node| node.regret_updates)
            .sum::<u64>();
        let positive_regret = self
            .nodes
            .values()
            .flat_map(|node| node.regrets.iter())
            .map(|regret| regret.max(0.0))
            .sum::<f64>();
        let held_out = evaluate_held_out(
            &self.config,
            &self.nodes,
            self.config.evaluation_controls.held_out_seed,
        );
        let root_local_deviation = evaluate_root_local_deviation(&self.config, &self.nodes);
        let mut provenance = vec![
            "External-sampling Discounted CFR with alternating traverser updates; traverser actions are enumerated and opponent/chance actions are sampled deterministically from the seeded current strategy.".to_owned(),
            "Sampled information sets lazily receive every skipped global DCFR regret discount, and each average-strategy visit is weighted by the exact global iteration^gamma contribution.".to_owned(),
            "Exact private cards and boards are sampled without replacement; information sets use lossy coarse rollout-derived strength/potential and public buckets plus an abstract no-limit action grid.".to_owned(),
            "All-in showdowns before the river use deterministic conditional-expectation runout evaluation to reduce chance variance; sample counts and exact-turn behavior are recorded in config.showdown_evaluation.".to_owned(),
            "Rake-free, equal-stack, heads-up cash model with no ante; button posts the small blind, acts first preflop, and acts last postflop.".to_owned(),
            "Reported regret is a training diagnostic, not exploitability or a Nash-distance certificate.".to_owned(),
            "Root local-deviation evaluation forces one button action at a time against the fixed average continuation/opponent policy. It is a one-step local best response with sampling error and is not exploitability or a full best response.".to_owned(),
            "A separate seeded evaluation pass records counterfactual values and standard errors for every action at each reached served information set. Low-sample or >0.02bb-standard-error values are flagged low confidence.".to_owned(),
        ];
        if self.config.recall_mode == RecallMode::CurrentStreet {
            provenance.push(
                "Postflop abstraction retains only the current street's private/public bucket while preserving full public action history. This is imperfect recall, so standard perfect-recall CFR convergence guarantees do not transfer.".to_owned(),
            );
        }
        BlueprintArtifact {
            schema_version: 1,
            solver_version: SOLVER_VERSION.to_owned(),
            artifact_id: format!(
                "hu-blueprint-{:.0}bb-i{}-s{}-{config_hash:016x}",
                self.config.effective_stack_bb, self.completed_iterations, self.config.seed,
            ),
            config_hash: format!("{config_hash:016x}"),
            training_config_hash: format!("{training_config_hash:016x}"),
            model: MODEL.to_owned(),
            approximate: true,
            provenance,
            metrics: BlueprintMetrics {
                requested_iterations: self.config.iterations,
                training_iterations: self.completed_iterations,
                stopped_early: self.completed_iterations < self.config.iterations,
                stop_reason: (self.completed_iterations < self.config.iterations).then(|| {
                    format!(
                        "information-set limit reached ({} >= {})",
                        self.nodes.len(),
                        self.config.max_information_sets
                    )
                }),
                sampled_deals: self.completed_iterations,
                terminal_evaluations: self.terminal_evaluations,
                information_sets: self.nodes.len(),
                preflop_information_sets: self
                    .nodes
                    .values()
                    .filter(|node| node.descriptor.street == Street::Preflop)
                    .count(),
                postflop_information_sets: self
                    .nodes
                    .values()
                    .filter(|node| node.descriptor.street != Street::Preflop)
                    .count(),
                trained_information_sets: self
                    .nodes
                    .values()
                    .filter(|node| node.average_visits > 0)
                    .count(),
                exported_information_sets: strategies.len(),
                clipped_cumulative_regret_per_update_diagnostic_bb: positive_regret
                    / total_updates.max(1) as f64,
                held_out,
                root_local_deviation,
                action_value_evaluation,
            },
            config: self.config.clone(),
            validation: BlueprintValidation {
                status: if self.completed_iterations < self.config.iterations {
                    "incomplete_advisory"
                } else {
                    "advisory_only"
                }
                .to_owned(),
                reasons: {
                    let mut reasons = vec![
                    "No exploitability or Nash-distance certificate is computed for the abstract full game.".to_owned(),
                    "Cross-seed strategy stability and a full or stronger local-best-response validation must pass before publishing ranges as solver-backed recommendations.".to_owned(),
                    "Unvisited or pre-averaging information sets are omitted from the strategy export; their coverage remains visible in metrics and the resumable checkpoint.".to_owned(),
                    ];
                    if self.completed_iterations < self.config.iterations {
                        reasons.push("Training stopped at the configured information-set memory guard before reaching the requested iteration count.".to_owned());
                    }
                    reasons
                },
            },
            strategies,
        }
    }
}

fn stable_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

pub fn solve(config: BlueprintConfig) -> Result<BlueprintArtifact, Box<dyn Error>> {
    solve_controlled(config, RunControl::default())
}

pub fn solve_controlled(
    config: BlueprintConfig,
    control: RunControl,
) -> Result<BlueprintArtifact, Box<dyn Error>> {
    config
        .validate()
        .map_err(|error| format!("invalid blueprint config: {error}"))?;
    let mut trainer = if let Some(path) = &control.resume_path {
        let checkpoint_file = fs::File::open(path)?;
        let checkpoint: BlueprintCheckpoint =
            serde_json::from_reader(BufReader::new(checkpoint_file))?;
        if checkpoint.model != MODEL || !checkpoint.approximate {
            return Err("checkpoint model is incompatible".into());
        }
        Trainer::from_checkpoint(checkpoint, &config)
            .map_err(|error| format!("invalid checkpoint: {error}"))?
    } else {
        Trainer::fresh(config)
    };
    trainer.train(&control)?;
    Ok(trainer.artifact())
}

fn write_json_atomic<T: Serialize + ?Sized>(path: &Path, value: &T) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let temporary = path.with_extension("tmp");
    let file = fs::File::create(&temporary)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, value)?;
    writer.flush()?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn sample_index(strategy: &[f64], rng: &mut SplitMix64) -> usize {
    let roll = rng.next_f64();
    let mut cumulative = 0.0;
    for (index, probability) in strategy.iter().enumerate() {
        cumulative += probability;
        if roll < cumulative {
            return index;
        }
    }
    strategy.len() - 1
}

fn information_set(
    state: &GameState,
    deal: &Deal,
    config: &BlueprintConfig,
) -> (u64, NodeDescriptor, Vec<String>) {
    let hand_bucket_trajectory =
        hand_bucket_trajectory(deal, state.actor, state.street, &config.hand_abstraction);
    let mut hand_bucket_trajectory = hand_bucket_trajectory;
    let mut public_bucket_trajectory = public_bucket_trajectory(deal, state.street);
    if config.recall_mode == RecallMode::CurrentStreet && state.street != Street::Preflop {
        hand_bucket_trajectory = hand_bucket_trajectory.last().cloned().into_iter().collect();
        public_bucket_trajectory = public_bucket_trajectory
            .last()
            .cloned()
            .into_iter()
            .collect();
    }
    let history_identity = state.public_history.join("/");
    let public_history_id = stable_hash(history_identity.as_bytes());
    let descriptor = NodeDescriptor {
        actor: Position::for_player(state.actor),
        street: state.street,
        hand_bucket_trajectory,
        public_bucket_trajectory,
        public_history_id,
        pot_bb: quantize(state.pot(), 0.001),
        to_call_bb: quantize(state.to_call(), 0.001),
        effective_stack_remaining_bb: quantize(state.remaining(state.actor, config), 0.001),
    };
    let identity = format!(
        "{:?}|p{}|h:{}|b:{}|pot:{:.3}|call:{:.3}|stack:{:.3}|{}",
        descriptor.street,
        state.actor,
        descriptor.hand_bucket_trajectory.join(">"),
        descriptor.public_bucket_trajectory.join(">"),
        descriptor.pot_bb,
        descriptor.to_call_bb,
        descriptor.effective_stack_remaining_bb,
        history_identity
    );
    (
        stable_hash(identity.as_bytes()),
        descriptor,
        state.public_history.clone(),
    )
}

fn hand_bucket_trajectory(
    deal: &Deal,
    player: usize,
    through: Street,
    abstraction: &HandAbstraction,
) -> Vec<String> {
    let mut buckets = vec![format!(
        "preflop:{}",
        Combo::new(deal.holes[player][0], deal.holes[player][1]).label()
    )];
    for street in [Street::Flop, Street::Turn, Street::River] {
        if street.board_len() > through.board_len() {
            break;
        }
        buckets.push(deal.hand_bucket(player, street, abstraction));
    }
    buckets
}

fn public_bucket_trajectory(deal: &Deal, through: Street) -> Vec<String> {
    let mut buckets = Vec::new();
    for street in [Street::Flop, Street::Turn, Street::River] {
        if street.board_len() > through.board_len() {
            break;
        }
        buckets.push(public_board_bucket(&deal.board[..street.board_len()]));
    }
    buckets
}

fn postflop_hand_bucket(
    deal: &Deal,
    player: usize,
    street: Street,
    abstraction: &HandAbstraction,
) -> String {
    let board = &deal.board[..street.board_len()];
    let mut cards = Vec::with_capacity(board.len() + 2);
    cards.extend_from_slice(&deal.holes[player]);
    cards.extend_from_slice(board);
    let score = evaluate(&cards);
    let category = score >> 24;
    let hole_high = (deal.holes[player][0] >> 2).max(deal.holes[player][1] >> 2);
    let board_high = board.iter().map(|card| card >> 2).max().unwrap_or(0);
    let rank_relation = if (deal.holes[player][0] >> 2) == (deal.holes[player][1] >> 2) {
        if hole_high > board_high {
            "overpair"
        } else {
            "pocket_pair"
        }
    } else if hole_high > board_high {
        "overcard"
    } else {
        "no_overcard"
    };
    let flush_draw = has_flush_draw(&cards, street);
    let straight_draw = has_straight_draw(&cards, category, street);
    let (equity_quantile, potential_quantile, future_mode) =
        visible_card_distribution_features(&deal.holes[player], board, abstraction);
    let kicker_band = match hole_high {
        10..=12 => "high",
        7..=9 => "middle",
        _ => "low",
    };
    format!(
        "c{category}:{rank_relation}:k{kicker_band}:fd{}:sd{}:eq{equity_quantile}:pot{potential_quantile}:future{future_mode}",
        u8::from(flush_draw),
        u8::from(straight_draw)
    )
}

fn visible_card_distribution_features(
    hole: &[u8; 2],
    board: &[u8],
    abstraction: &HandAbstraction,
) -> (u8, u8, u8) {
    let samples = abstraction.distribution_samples as usize;
    let mut known = [false; 52];
    for card in hole.iter().chain(board.iter()) {
        known[*card as usize] = true;
    }
    let mut available = (0..52u8)
        .filter(|card| !known[*card as usize])
        .collect::<Vec<_>>();
    let mut seed = 0xcbf2_9ce4_8422_2325u64;
    let mut visible = hole.to_vec();
    visible.sort_unstable();
    visible.push(0xff);
    let mut canonical_board = board.to_vec();
    canonical_board.sort_unstable();
    visible.extend_from_slice(&canonical_board);
    for card in visible {
        seed ^= card as u64 + 1;
        seed = seed.wrapping_mul(0x100_0000_01b3);
    }
    let mut rng = SplitMix64::new(seed);
    let missing_board = 5 - board.len();
    let mut equity = 0.0;
    let mut improved = 0u64;
    let mut future_categories = [0u32; 9];
    let current_category = {
        let mut current = Vec::with_capacity(2 + board.len());
        current.extend_from_slice(hole);
        current.extend_from_slice(board);
        evaluate(&current) >> 24
    };

    for _ in 0..samples {
        let needed = 2 + missing_board;
        for index in 0..needed {
            let swap = index + rng.index(available.len() - index);
            available.swap(index, swap);
        }
        let opponent = [available[0], available[1]];
        let mut completed_board = board.to_vec();
        completed_board.extend_from_slice(&available[2..needed]);
        let mut hero_cards = Vec::with_capacity(7);
        hero_cards.extend_from_slice(hole);
        hero_cards.extend_from_slice(&completed_board);
        let mut opponent_cards = Vec::with_capacity(7);
        opponent_cards.extend_from_slice(&opponent);
        opponent_cards.extend_from_slice(&completed_board);
        let hero_score = evaluate(&hero_cards);
        let opponent_score = evaluate(&opponent_cards);
        equity += match hero_score.cmp(&opponent_score) {
            std::cmp::Ordering::Greater => 1.0,
            std::cmp::Ordering::Equal => 0.5,
            std::cmp::Ordering::Less => 0.0,
        };
        let final_category = (hero_score >> 24) as usize;
        future_categories[final_category] += 1;
        improved += u64::from(final_category as u32 > current_category);
    }
    let equity_quantile = ((equity / samples as f64 * abstraction.equity_bins as f64).floor()
        as u8)
        .min(abstraction.equity_bins - 1);
    let potential_quantile =
        (((improved as f64 / samples as f64) * abstraction.potential_bins as f64).floor() as u8)
            .min(abstraction.potential_bins - 1);
    let future_mode = future_categories
        .iter()
        .enumerate()
        .max_by_key(|(_, count)| *count)
        .map(|(category, _)| category as u8)
        .unwrap_or(0);
    (equity_quantile, potential_quantile, future_mode)
}

fn public_board_bucket(board: &[u8]) -> String {
    let mut rank_counts = [0u8; 13];
    let mut suit_counts = [0u8; 4];
    let mut rank_mask = 0u16;
    for card in board {
        let rank = (card >> 2) as usize;
        rank_counts[rank] += 1;
        suit_counts[(card & 3) as usize] += 1;
        rank_mask |= 1 << rank;
    }
    let paired = match rank_counts.iter().copied().max().unwrap_or(0) {
        1 => "unpaired",
        2 => "paired",
        3 => "trips",
        _ => "quads",
    };
    let max_suit = suit_counts.iter().copied().max().unwrap_or(0);
    let suit_texture = match max_suit {
        0 | 1 => "rainbow",
        2 => "two_tone",
        3 => "monotone",
        _ => "four_flush",
    };
    let connected = max_window_rank_count(rank_mask) >= 3;
    let high = board.iter().map(|card| card >> 2).max().unwrap_or(0);
    let high_band = match high {
        10..=12 => "broadway",
        7..=9 => "middle",
        _ => "low",
    };
    format!(
        "{paired}:{suit_texture}:conn{}:{high_band}",
        u8::from(connected)
    )
}

fn has_flush_draw(cards: &[u8], street: Street) -> bool {
    if street == Street::River {
        return false;
    }
    let mut suits = [0u8; 4];
    for card in cards {
        suits[(card & 3) as usize] += 1;
    }
    suits.into_iter().any(|count| count == 4)
}

fn has_straight_draw(cards: &[u8], made_category: u32, street: Street) -> bool {
    if street == Street::River || made_category >= 4 {
        return false;
    }
    let mut mask = 0u16;
    for card in cards {
        mask |= 1 << (card >> 2);
    }
    // Treat the ace as low for wheel draws.
    let wheel_mask = if mask & (1 << 12) != 0 {
        mask | (1 << 13)
    } else {
        mask
    };
    (0..=9).any(|low| ((wheel_mask >> low) & 0b1_1111).count_ones() == 4)
        || (mask & ((1 << 12) | 0b1111)).count_ones() == 4
}

fn max_window_rank_count(mask: u16) -> u32 {
    let mut best = 0;
    for low in 0..=8 {
        best = best.max(((mask >> low) & 0b1_1111).count_ones());
    }
    best
}

fn quantize(value: f64, step: f64) -> f64 {
    (value / step).round() * step
}

fn evaluate_held_out(
    config: &BlueprintConfig,
    nodes: &BTreeMap<u64, Node>,
    seed: u64,
) -> HeldOutMetrics {
    let mut rng = SplitMix64::new(seed);
    let mut sum = 0.0;
    let mut square_sum = 0.0;
    let mut folds = 0u64;
    let mut showdowns = 0u64;
    let mut decisions = 0u64;
    let mut unknown = 0u64;
    let mut untrained = 0u64;
    for _ in 0..config.evaluation_controls.held_out_deals {
        let deal = Deal::sample(&mut rng);
        let mut state = GameState::initial(config);
        while state.terminal.is_none() {
            let actions = state.legal_actions(config);
            let (key, _, _) = information_set(&state, &deal, config);
            decisions += 1;
            let strategy = match nodes.get(&key) {
                Some(node) if node.average_visits > 0 => node.average_strategy(),
                Some(node) => {
                    untrained += 1;
                    node.average_strategy()
                }
                None => {
                    unknown += 1;
                    vec![1.0 / actions.len() as f64; actions.len()]
                }
            };
            let selected = sample_index(&strategy, &mut rng);
            state = state.apply(&actions[selected], config);
        }
        match state.terminal {
            Some(Terminal::Fold { .. }) => folds += 1,
            Some(Terminal::Showdown) => showdowns += 1,
            None => unreachable!(),
        }
        let utility = state.utility_p0(&deal, config);
        sum += utility;
        square_sum += utility * utility;
    }
    let count = config.evaluation_controls.held_out_deals as f64;
    let mean = sum / count;
    let variance = (square_sum / count - mean * mean).max(0.0);
    HeldOutMetrics {
        deals: config.evaluation_controls.held_out_deals,
        button_mean_net_bb: mean,
        button_net_standard_error_bb: (variance / count).sqrt(),
        fold_terminal_fraction: folds as f64 / count,
        showdown_terminal_fraction: showdowns as f64 / count,
        unknown_information_set_fraction: unknown as f64 / decisions.max(1) as f64,
        untrained_information_set_fraction: untrained as f64 / decisions.max(1) as f64,
    }
}

#[derive(Clone, Debug, Default)]
struct CoverageCounter {
    decisions: u64,
    unknown: u64,
    untrained: u64,
}

impl CoverageCounter {
    fn add(&mut self, other: &Self) {
        self.decisions += other.decisions;
        self.unknown += other.unknown;
        self.untrained += other.untrained;
    }

    fn report(&self) -> PolicyCoverage {
        PolicyCoverage {
            decisions: self.decisions,
            unknown_information_set_fraction: self.unknown as f64 / self.decisions.max(1) as f64,
            untrained_information_set_fraction: self.untrained as f64
                / self.decisions.max(1) as f64,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct ValueAccumulator {
    samples: u64,
    sum: f64,
    square_sum: f64,
    coverage: CoverageCounter,
}

impl ValueAccumulator {
    fn observe(&mut self, value: f64, coverage: &CoverageCounter) {
        self.samples += 1;
        self.sum += value;
        self.square_sum += value * value;
        self.coverage.add(coverage);
    }

    fn mean(&self) -> f64 {
        self.sum / self.samples.max(1) as f64
    }

    fn standard_error(&self) -> f64 {
        if self.samples < 2 {
            return 0.0;
        }
        let count = self.samples as f64;
        let sample_variance =
            ((self.square_sum - self.sum * self.sum / count) / (count - 1.0)).max(0.0);
        (sample_variance / count).sqrt()
    }
}

#[derive(Clone, Debug)]
struct InformationSetActionEvaluation {
    visits: u64,
    action_labels: Vec<String>,
    actions: Vec<ValueAccumulator>,
}

#[derive(Clone, Debug, Default)]
struct ActionValuePass {
    information_sets: BTreeMap<u64, InformationSetActionEvaluation>,
}

fn evaluate_information_set_actions(
    config: &BlueprintConfig,
    nodes: &BTreeMap<u64, Node>,
) -> ActionValuePass {
    let mut chance_rng = SplitMix64::new(config.evaluation_controls.action_value_seed);
    let mut pass = ActionValuePass::default();
    for deal_index in 0..config.evaluation_controls.action_value_deals {
        let deal = Deal::sample(&mut chance_rng);
        let mut state = GameState::initial(config);
        while state.terminal.is_none() {
            let actions = state.legal_actions(config);
            let (key, _, _) = information_set(&state, &deal, config);
            let strategy = match nodes.get(&key) {
                Some(node) => node.average_strategy(),
                None => vec![1.0 / actions.len() as f64; actions.len()],
            };

            if let Some(node) = nodes.get(&key).filter(|node| node.average_visits > 0) {
                let entry = pass.information_sets.entry(key).or_insert_with(|| {
                    InformationSetActionEvaluation {
                        visits: 0,
                        action_labels: node.action_labels.clone(),
                        actions: vec![ValueAccumulator::default(); actions.len()],
                    }
                });
                debug_assert_eq!(entry.action_labels, node.action_labels);
                entry.visits += 1;
                for (action_index, action) in actions.iter().enumerate() {
                    let rollout_seed = config.evaluation_controls.action_value_seed
                        ^ deal_index.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        ^ key.rotate_left(17)
                        ^ (action_index as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    let mut rollout_rng = SplitMix64::new(rollout_seed);
                    let mut coverage = CoverageCounter::default();
                    let button_value = rollout_average_policy(
                        state.apply(action, config),
                        &deal,
                        config,
                        nodes,
                        &mut rollout_rng,
                        &mut coverage,
                    );
                    let actor_value = if state.actor == 0 {
                        button_value
                    } else {
                        -button_value
                    };
                    entry.actions[action_index].observe(actor_value, &coverage);
                }
            }

            let selected = sample_index(&strategy, &mut chance_rng);
            state = state.apply(&actions[selected], config);
        }
    }
    pass
}

fn finalize_action_values(
    node: &Node,
    evaluation: Option<&InformationSetActionEvaluation>,
) -> (Vec<ActionValueEstimate>, Option<String>, Option<f64>) {
    let Some(evaluation) = evaluation else {
        return (Vec::new(), None, None);
    };
    debug_assert_eq!(node.action_labels, evaluation.action_labels);
    let best_ev = evaluation
        .actions
        .iter()
        .map(ValueAccumulator::mean)
        .max_by(f64::total_cmp)
        .expect("information set has actions");
    let best_index = evaluation
        .actions
        .iter()
        .position(|accumulator| (accumulator.mean() - best_ev).abs() <= EPSILON)
        .expect("best action value is present");
    let values = node
        .action_labels
        .iter()
        .cloned()
        .zip(&evaluation.actions)
        .map(|(action, accumulator)| ActionValueEstimate {
            action,
            samples: accumulator.samples,
            mean_ev_bb: accumulator.mean(),
            standard_error_bb: accumulator.standard_error(),
            best_action_ev_loss_bb: (best_ev - accumulator.mean()).max(0.0),
            low_confidence: accumulator.samples < 2 || accumulator.standard_error() > 0.02,
        })
        .collect();
    (
        values,
        Some(node.action_labels[best_index].clone()),
        Some(best_ev),
    )
}

fn evaluate_root_local_deviation(
    config: &BlueprintConfig,
    nodes: &BTreeMap<u64, Node>,
) -> RootLocalDeviation {
    let mut class_combos = BTreeMap::<String, Vec<Combo>>::new();
    for combo in all_combos() {
        class_combos.entry(combo.label()).or_default().push(combo);
    }
    debug_assert_eq!(class_combos.len(), 169);

    let initial = GameState::initial(config);
    let root_actions = initial.legal_actions(config);
    let samples_per_class = config.evaluation_controls.root_deviation_samples_per_class;
    let base_seed = config.evaluation_controls.root_deviation_seed;
    let mut classes = Vec::with_capacity(class_combos.len());
    let mut aggregate_chosen = 0.0;
    let mut aggregate_best = 0.0;
    let mut aggregate_gain_variance = 0.0;
    let mut trained_combo_weight = 0.0;
    let mut aggregate_coverage = CoverageCounter::default();

    for (class_index, (hand_class, combos)) in class_combos.into_iter().enumerate() {
        let combo_weight = combos.len() as f64 / 1326.0;
        let class_seed =
            base_seed ^ stable_hash(hand_class.as_bytes()) ^ (class_index as u64).rotate_left(23);
        let mut chance_rng = SplitMix64::new(class_seed);
        let mut action_accumulators = vec![ValueAccumulator::default(); root_actions.len()];
        let mut paired_action_values =
            vec![Vec::with_capacity(samples_per_class as usize); root_actions.len()];
        let mut root_policy = None;
        let mut root_policy_trained = false;

        for sample in 0..samples_per_class {
            let hero = combos[chance_rng.index(combos.len())];
            let deal = sample_deal_conditioned_on_hero(hero, &mut chance_rng);
            if root_policy.is_none() {
                let (key, _, _) = information_set(&initial, &deal, config);
                root_policy = Some(match nodes.get(&key) {
                    Some(node) if node.average_visits > 0 => {
                        root_policy_trained = true;
                        node.average_strategy()
                    }
                    Some(node) => node.average_strategy(),
                    None => vec![1.0 / root_actions.len() as f64; root_actions.len()],
                });
            }

            for (action_index, root_action) in root_actions.iter().enumerate() {
                let policy_seed = class_seed
                    ^ sample.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    ^ (action_index as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                let mut policy_rng = SplitMix64::new(policy_seed);
                let mut coverage = CoverageCounter::default();
                let value = rollout_average_policy(
                    initial.apply(root_action, config),
                    &deal,
                    config,
                    nodes,
                    &mut policy_rng,
                    &mut coverage,
                );
                action_accumulators[action_index].observe(value, &coverage);
                paired_action_values[action_index].push(value);
            }
        }

        let policy = root_policy
            .unwrap_or_else(|| vec![1.0 / root_actions.len() as f64; root_actions.len()]);
        let action_values = root_actions
            .iter()
            .zip(&action_accumulators)
            .map(|(action, accumulator)| RootActionValue {
                action: action.label.clone(),
                samples: accumulator.samples,
                mean_net_bb: accumulator.mean(),
                standard_error_bb: accumulator.standard_error(),
                continuation_coverage: accumulator.coverage.report(),
            })
            .collect::<Vec<_>>();
        let chosen_average_ev = policy
            .iter()
            .zip(&action_values)
            .map(|(probability, value)| probability * value.mean_net_bb)
            .sum::<f64>();
        let (best_index, best) = action_values
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.mean_net_bb.total_cmp(&right.mean_net_bb))
            .expect("root has legal actions");
        let best_action = best.action.clone();
        let best_action_ev = best.mean_net_bb;
        let gain = (best_action_ev - chosen_average_ev).max(0.0);
        let mut paired_gain = ValueAccumulator::default();
        for sample_index in 0..samples_per_class as usize {
            let chosen_sample = policy
                .iter()
                .zip(&paired_action_values)
                .map(|(probability, values)| probability * values[sample_index])
                .sum::<f64>();
            paired_gain.observe(
                paired_action_values[best_index][sample_index] - chosen_sample,
                &CoverageCounter::default(),
            );
        }
        let gain_standard_error = paired_gain.standard_error();
        let gain_lower_bound = (gain - 2.326_347_874 * gain_standard_error).max(0.0);
        aggregate_chosen += combo_weight * chosen_average_ev;
        aggregate_best += combo_weight * best_action_ev;
        aggregate_gain_variance += (combo_weight * gain_standard_error).powi(2);
        if root_policy_trained {
            trained_combo_weight += combo_weight;
        }
        for accumulator in &action_accumulators {
            aggregate_coverage.add(&accumulator.coverage);
        }
        classes.push(RootClassDeviation {
            hand_class,
            exact_combo_count: combos.len(),
            root_policy_trained,
            root_policy: root_actions
                .iter()
                .map(|action| action.label.clone())
                .zip(policy)
                .map(|(action, probability)| StrategyAction {
                    action,
                    probability,
                })
                .collect(),
            action_values,
            chosen_average_ev_bb: chosen_average_ev,
            best_action,
            best_action_ev_bb: best_action_ev,
            local_deviation_gain_bb: gain,
            local_deviation_gain_standard_error_bb: gain_standard_error,
            local_deviation_gain_99pct_lower_bound_bb: gain_lower_bound,
        });
    }

    let aggregate_gain = (aggregate_best - aggregate_chosen).max(0.0);
    let aggregate_gain_standard_error = aggregate_gain_variance.sqrt();
    RootLocalDeviation {
        kind: "button-root-one-step-local-best-response-v1".to_owned(),
        samples_per_class,
        seed: base_seed,
        classes,
        aggregate_chosen_average_ev_bb: aggregate_chosen,
        aggregate_best_action_ev_bb: aggregate_best,
        aggregate_local_deviation_gain_bb: aggregate_gain,
        aggregate_local_deviation_gain_standard_error_bb: aggregate_gain_standard_error,
        aggregate_local_deviation_gain_99pct_lower_bound_bb: (aggregate_gain
            - 2.326_347_874 * aggregate_gain_standard_error)
            .max(0.0),
        trained_root_combo_fraction: trained_combo_weight,
        continuation_coverage: aggregate_coverage.report(),
    }
}

fn sample_deal_conditioned_on_hero(hero: Combo, rng: &mut SplitMix64) -> Deal {
    let hero_cards = hero.cards();
    let mut deck = (0..52u8)
        .filter(|card| *card != hero_cards[0] && *card != hero_cards[1])
        .collect::<Vec<_>>();
    for index in 0..7 {
        let swap = index + rng.index(deck.len() - index);
        deck.swap(index, swap);
    }
    Deal::from_sampled_cards(
        [hero_cards, [deck[0], deck[1]]],
        [deck[2], deck[3], deck[4], deck[5], deck[6]],
    )
}

fn rollout_average_policy(
    mut state: GameState,
    deal: &Deal,
    config: &BlueprintConfig,
    nodes: &BTreeMap<u64, Node>,
    rng: &mut SplitMix64,
    coverage: &mut CoverageCounter,
) -> f64 {
    while state.terminal.is_none() {
        let actions = state.legal_actions(config);
        let (key, _, _) = information_set(&state, deal, config);
        coverage.decisions += 1;
        let strategy = match nodes.get(&key) {
            Some(node) if node.average_visits > 0 => node.average_strategy(),
            Some(node) => {
                coverage.untrained += 1;
                node.average_strategy()
            }
            None => {
                coverage.unknown += 1;
                vec![1.0 / actions.len() as f64; actions.len()]
            }
        };
        let selected = sample_index(&strategy, rng);
        state = state.apply(&actions[selected], config);
    }
    state.utility_p0(deal, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Deserialize, PartialEq, Serialize)]
    struct BinaryCacheFixture {
        label: String,
        values: Vec<f32>,
    }

    #[test]
    fn model_binary_cache_preserves_canonical_identity_and_value() {
        let path = std::env::temp_dir().join(format!(
            "poker-model-cache-{}-{}.json",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let fixture = BinaryCacheFixture {
            label: "frozen".to_owned(),
            values: vec![0.125, 0.875],
        };
        let json = serde_json::to_vec(&fixture).expect("serialize fixture");
        fs::write(&path, &json).expect("write canonical fixture");
        let binary = write_model_binary_cache::<BinaryCacheFixture>(&path)
            .expect("compile binary model cache");
        let (loaded, sha256) =
            read_model_artifact::<BinaryCacheFixture>(&path).expect("load binary model cache");
        assert_eq!(loaded, fixture);
        assert_eq!(sha256, format!("{:x}", Sha256::digest(&json)));
        fs::remove_file(binary).expect("remove binary fixture");
        fs::remove_file(path).expect("remove canonical fixture");
    }

    fn action<'a>(state: &GameState, config: &BlueprintConfig, needle: &str) -> LegalAction {
        state
            .legal_actions(config)
            .into_iter()
            .find(|candidate| candidate.label == needle)
            .unwrap_or_else(|| panic!("missing action {needle}"))
    }

    fn fixed_deal() -> Deal {
        Deal::from_cards([[51, 50], [45, 44]], [0, 5, 10, 27, 28])
    }

    fn tiny_config() -> BlueprintConfig {
        BlueprintConfig {
            effective_stack_bb: 6.0,
            iterations: 3,
            averaging_delay: 0,
            evaluation_controls: EvaluationControls {
                held_out_deals: 4,
                held_out_seed: 41,
                root_deviation_samples_per_class: 1,
                root_deviation_seed: 43,
                action_value_deals: 4,
                action_value_seed: 47,
            },
            hand_abstraction: HandAbstraction {
                distribution_samples: 4,
                equity_bins: 4,
                potential_bins: 2,
            },
            action_abstraction: ActionAbstraction {
                open_sizes_bb: vec![2.0],
                limp_raise_sizes_bb: vec![2.0],
                three_bet_sizes_bb: vec![4.0],
                four_bet_sizes_bb: vec![5.0],
                deeper_raise_pot_fractions: vec![1.0],
                preflop_raise_cap: 2,
                flop_bet_pot_fractions: vec![0.5],
                turn_river_bet_pot_fractions: vec![0.5],
                postflop_raise_pot_fractions: vec![1.0],
                postflop_raise_cap: 1,
                include_all_in: true,
            },
            ..BlueprintConfig::default()
        }
    }

    #[test]
    fn initial_state_has_correct_blinds_positions_and_rich_actions() {
        let config = BlueprintConfig::default();
        let state = GameState::initial(&config);
        assert_eq!(state.actor, 0);
        assert_eq!(state.invested, [0.5, 1.0]);
        assert_eq!(state.pot(), 1.5);
        let labels = state
            .legal_actions(&config)
            .into_iter()
            .map(|candidate| candidate.label)
            .collect::<Vec<_>>();
        assert!(labels.contains(&"fold".to_owned()));
        assert!(labels.contains(&"limp".to_owned()));
        for size in ["2.000", "2.500", "3.000", "4.000", "5.000"] {
            assert!(labels.iter().any(|label| label.contains(size)));
        }
        assert!(labels.iter().any(|label| label.contains("all_in")));
    }

    #[test]
    fn limp_check_and_open_call_reach_flop_with_big_blind_acting() {
        let config = BlueprintConfig::default();
        let initial = GameState::initial(&config);
        let limped = initial.apply(&action(&initial, &config, "limp"), &config);
        assert_eq!(limped.actor, 1);
        assert_eq!(limped.invested, [1.0, 1.0]);
        let checked = limped.apply(&action(&limped, &config, "check"), &config);
        assert_eq!(checked.street, Street::Flop);
        assert_eq!(checked.actor, 1);

        let opened = initial.apply(&action(&initial, &config, "raise_to_2.500bb"), &config);
        let called = opened.apply(&action(&opened, &config, "call"), &config);
        assert_eq!(called.street, Street::Flop);
        assert_eq!(called.actor, 1);
        assert_eq!(called.invested, [2.5, 2.5]);
    }

    #[test]
    fn fold_and_all_in_showdown_pay_exact_net_contributions() {
        let config = BlueprintConfig::default();
        let initial = GameState::initial(&config);
        let folded = initial.apply(&action(&initial, &config, "fold"), &config);
        assert_eq!(folded.utility_p0(&fixed_deal(), &config), -0.5);

        let shoved = initial.apply(
            &action(&initial, &config, "raise_all_in_to_100.000bb"),
            &config,
        );
        let called = shoved.apply(&action(&shoved, &config, "call_all_in"), &config);
        assert_eq!(called.terminal, Some(Terminal::Showdown));
        assert_eq!(called.invested, [100.0, 100.0]);
        let all_in_utility = called.utility_p0(&fixed_deal(), &config);
        let equity = conditional_showdown_equity_p0(&fixed_deal(), Street::Preflop, &config);
        assert!((all_in_utility - (equity * 100.0 - (1.0 - equity) * 100.0)).abs() < EPSILON);
        assert!(all_in_utility > 40.0 && all_in_utility < 100.0);
    }

    #[test]
    fn all_tiny_abstract_paths_preserve_stack_and_terminal_accounting() {
        fn visit(state: GameState, config: &BlueprintConfig, deal: &Deal, terminals: &mut usize) {
            assert!(state.invested.iter().all(|amount| {
                *amount >= 0.0 && *amount <= config.effective_stack_bb + EPSILON
            }));
            if state.terminal.is_some() {
                *terminals += 1;
                let utility = state.utility_p0(deal, config);
                assert!(utility.abs() <= config.effective_stack_bb + EPSILON);
                if state.terminal == Some(Terminal::Showdown) {
                    assert!((state.invested[0] - state.invested[1]).abs() <= EPSILON);
                }
                return;
            }
            for next_action in state.legal_actions(config) {
                visit(state.apply(&next_action, config), config, deal, terminals);
            }
        }

        let config = tiny_config();
        let mut terminals = 0;
        visit(
            GameState::initial(&config),
            &config,
            &fixed_deal(),
            &mut terminals,
        );
        assert!(terminals > 100);
    }

    #[test]
    fn short_training_is_deterministic_and_exports_probability_sums() {
        let first = solve(tiny_config()).expect("first deterministic solve");
        let second = solve(tiny_config()).expect("second deterministic solve");
        assert_eq!(first, second);
        assert!(!first.strategies.is_empty());
        for info_set in &first.strategies {
            let sum = info_set
                .actions
                .iter()
                .map(|action| action.probability)
                .sum::<f64>();
            assert!((sum - 1.0).abs() < 1e-9);
            assert_eq!(info_set.street, Street::Preflop);
            assert!(!info_set.actions.is_empty());
            assert!(info_set.action_values.len() <= info_set.actions.len());
        }
        assert_eq!(first.validation.status, "advisory_only");
        assert!(first.artifact_id.starts_with("hu-blueprint-6bb-i3-s1-"));
        let deviation = &first.metrics.root_local_deviation;
        assert_eq!(deviation.classes.len(), 169);
        assert_eq!(
            deviation
                .classes
                .iter()
                .map(|class| class.exact_combo_count)
                .sum::<usize>(),
            1326
        );
        assert_eq!(
            deviation
                .classes
                .iter()
                .find(|class| class.hand_class == "AA")
                .expect("AA class")
                .exact_combo_count,
            6
        );
        assert_eq!(
            deviation
                .classes
                .iter()
                .find(|class| class.hand_class == "AKs")
                .expect("AKs class")
                .exact_combo_count,
            4
        );
        assert_eq!(
            deviation
                .classes
                .iter()
                .find(|class| class.hand_class == "AKo")
                .expect("AKo class")
                .exact_combo_count,
            12
        );
        assert!(deviation.aggregate_local_deviation_gain_bb >= 0.0);
        assert_eq!(first.metrics.action_value_evaluation.deals, 4);
        assert!(
            first
                .metrics
                .action_value_evaluation
                .evaluated_information_sets
                > 0
        );
        assert!(
            deviation
                .aggregate_local_deviation_gain_standard_error_bb
                .is_finite()
                && deviation.aggregate_local_deviation_gain_standard_error_bb >= 0.0
        );
        assert!(
            deviation
                .aggregate_local_deviation_gain_99pct_lower_bound_bb
                .is_finite()
                && deviation.aggregate_local_deviation_gain_99pct_lower_bound_bb >= 0.0
        );
        for class in &deviation.classes {
            assert_eq!(class.action_values.len(), 4);
            assert!(class.action_values.iter().all(|action| action.samples == 1
                && action.mean_net_bb.is_finite()
                && action.standard_error_bb == 0.0));
            assert!(
                class.local_deviation_gain_standard_error_bb.is_finite()
                    && class.local_deviation_gain_standard_error_bb >= 0.0
            );
            assert!(
                class.local_deviation_gain_99pct_lower_bound_bb.is_finite()
                    && class.local_deviation_gain_99pct_lower_bound_bb >= 0.0
            );
        }
    }

    #[test]
    fn dcfr_discounting_is_applied_only_to_regrets() {
        let mut node = Node {
            descriptor: NodeDescriptor {
                actor: Position::ButtonSmallBlind,
                street: Street::Preflop,
                hand_bucket_trajectory: vec!["preflop:AA".to_owned()],
                public_bucket_trajectory: Vec::new(),
                public_history_id: 1,
                pot_bb: 1.5,
                to_call_bb: 0.5,
                effective_stack_remaining_bb: 99.5,
            },
            action_labels: vec!["fold".to_owned(), "call".to_owned()],
            regrets: vec![4.0, -4.0],
            strategy_sum: vec![4.0, 4.0],
            regret_updates: 1,
            average_visits: 1,
            last_discount_iteration: 0,
            last_regret_discount_cumulative_logs: [0.0; 2],
        };
        let mut discounts = BlueprintDiscountAccumulator::new(DcfrParameters::default());
        discounts.advance(1);
        node.apply_dcfr_regret_discount(1, &discounts);
        assert_eq!(node.regrets, vec![2.0, -2.0]);
        assert_eq!(node.strategy_sum, vec![4.0, 4.0]);
        node.apply_dcfr_regret_discount(1, &discounts);
        assert_eq!(node.regrets, vec![2.0, -2.0]);
    }

    #[test]
    fn lazy_full_game_dcfr_regret_discount_matches_eager_updates() {
        let parameters = DcfrParameters::default();
        let mut lazy_discounts = BlueprintDiscountAccumulator::new(parameters.clone());
        let mut eager_discounts = BlueprintDiscountAccumulator::new(parameters);
        let descriptor = NodeDescriptor {
            actor: Position::ButtonSmallBlind,
            street: Street::Preflop,
            hand_bucket_trajectory: vec!["preflop:AA".to_owned()],
            public_bucket_trajectory: Vec::new(),
            public_history_id: 1,
            pot_bb: 1.5,
            to_call_bb: 0.5,
            effective_stack_remaining_bb: 99.5,
        };
        let mut lazy = Node {
            descriptor: descriptor.clone(),
            action_labels: vec!["fold".to_owned(), "call".to_owned()],
            regrets: vec![3.0, -2.0],
            strategy_sum: vec![4.0, 1.0],
            regret_updates: 0,
            average_visits: 0,
            last_discount_iteration: 0,
            last_regret_discount_cumulative_logs: [0.0; 2],
        };
        let mut eager = lazy.clone();
        for iteration in 1..=12 {
            lazy_discounts.advance(iteration);
            eager_discounts.advance(iteration);
            eager.apply_dcfr_regret_discount(iteration, &eager_discounts);
        }
        lazy.apply_dcfr_regret_discount(12, &lazy_discounts);
        for (actual, expected) in lazy.regrets.iter().zip(&eager.regrets) {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert_eq!(lazy.strategy_sum, [4.0, 1.0]);
        assert_eq!(eager.strategy_sum, [4.0, 1.0]);
    }

    #[test]
    fn dcfr_strategy_average_uses_exact_iteration_to_gamma_weight() {
        let mut parameters = DcfrParameters::default();
        assert_eq!(dcfr_strategy_averaging_weight(2, &parameters), 4.0);
        parameters.strategy_exponent = 3.0;
        assert_eq!(dcfr_strategy_averaging_weight(2, &parameters), 8.0);
    }

    #[test]
    fn publishable_defaults_use_trajectory_recall() {
        let config = BlueprintConfig::default();
        assert_eq!(config.recall_mode, RecallMode::Trajectory);
        assert_eq!(config.dcfr, DcfrParameters::default());
    }

    #[test]
    fn compact_grid_only_removes_four_and_five_big_blind_opens() {
        let compact = ActionAbstraction::compact_serving_candidate();
        assert_eq!(compact.open_sizes_bb, vec![2.0, 2.5, 3.0]);
        assert_eq!(
            compact.flop_bet_pot_fractions,
            ActionAbstraction::default().flop_bet_pot_fractions
        );
        assert_eq!(compact.postflop_raise_cap, 1);
        assert!(compact.include_all_in);
        let config = BlueprintConfig {
            action_abstraction: compact,
            ..BlueprintConfig::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn visible_card_rollout_bucket_is_invariant_to_board_order() {
        let abstraction = HandAbstraction::default();
        let first = visible_card_distribution_features(&[51, 46], &[0, 5, 10], &abstraction);
        let second = visible_card_distribution_features(&[51, 46], &[10, 0, 5], &abstraction);
        assert_eq!(first, second);
    }

    #[test]
    fn information_set_guard_stops_cleanly_and_marks_artifact_incomplete() {
        let mut config = tiny_config();
        config.iterations = 20;
        config.max_information_sets = 1;
        let artifact = solve(config).expect("guarded solve");
        assert!(artifact.metrics.stopped_early);
        assert!(artifact.metrics.training_iterations < artifact.metrics.requested_iterations);
        assert_eq!(artifact.validation.status, "incomplete_advisory");
        assert!(artifact.metrics.stop_reason.is_some());
    }

    #[test]
    fn checkpoint_resume_matches_uninterrupted_training() {
        let path = std::env::temp_dir().join(format!(
            "blueprint-checkpoint-{}-{}.json",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let mut partial = tiny_config();
        partial.iterations = 2;
        solve_controlled(
            partial,
            RunControl {
                checkpoint_path: Some(path.to_string_lossy().into_owned()),
                checkpoint_every: 1,
                resume_path: None,
            },
        )
        .expect("write partial checkpoint");
        let checkpoint_json: serde_json::Value =
            serde_json::from_reader(BufReader::new(fs::File::open(&path).unwrap())).unwrap();
        assert_eq!(
            checkpoint_json["schema_version"],
            BLUEPRINT_CHECKPOINT_SCHEMA_VERSION
        );
        assert_eq!(
            checkpoint_json["regret_discount_cumulative_logs"]
                .as_array()
                .map(Vec::len),
            Some(2)
        );

        let resumed = solve_controlled(
            tiny_config(),
            RunControl {
                checkpoint_path: None,
                checkpoint_every: 0,
                resume_path: Some(path.to_string_lossy().into_owned()),
            },
        )
        .expect("resume checkpoint");
        let uninterrupted = solve(tiny_config()).expect("uninterrupted solve");
        assert_eq!(resumed, uninterrupted);
        fs::remove_file(path).expect("remove checkpoint");
    }

    #[test]
    fn checkpoint_money_is_recanonicalized_after_decimal_round_trip() {
        let mut descriptor = NodeDescriptor {
            actor: Position::BigBlind,
            street: Street::Flop,
            hand_bucket_trajectory: vec!["bucket".to_owned()],
            public_bucket_trajectory: vec!["board".to_owned()],
            public_history_id: 1,
            pot_bb: serde_json::from_str("29.333").expect("decimal pot"),
            to_call_bb: serde_json::from_str("7.333").expect("decimal call"),
            effective_stack_remaining_bb: serde_json::from_str("89.0").expect("decimal stack"),
        };

        descriptor.canonicalize_money();

        assert_eq!(descriptor.pot_bb, quantize(29.333, 0.001));
        assert_eq!(descriptor.to_call_bb, quantize(7.333, 0.001));
        assert_eq!(
            descriptor.effective_stack_remaining_bb,
            quantize(89.0, 0.001)
        );
    }
}
