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
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub mod neural;
pub mod preflop;
pub mod public_belief;
pub mod range_vector;
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
// Version 5 rejects training state accumulated under the old absolute
// probability/regret cutoff. A deterministic new run must use one recurrence.
const BLUEPRINT_CHECKPOINT_SCHEMA_VERSION: u32 = 5;
const MAX_HS_DCFR_HORIZON: u64 = 10_000_000;

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
    /// Number of compatible active-traverser hole-card combinations trained
    /// against one sampled public board and opponent hand. One preserves the
    /// legacy external-sampling stream exactly.
    #[serde(default = "default_traverser_hand_batch_size")]
    pub traverser_hand_batch_size: usize,
    /// Number of compatible opponent hands sampled for every active-traverser
    /// hand in the contiguous research traversal. One preserves the scalar
    /// trainer and all established artifact identities.
    #[serde(
        default = "default_opponent_hand_batch_size",
        skip_serializing_if = "is_default_hand_batch_size"
    )]
    pub opponent_hand_batch_size: usize,
    /// Research traversal for policy-changing pilots. The default preserves
    /// the established scalar external-sampling stream and artifact identity.
    #[serde(default, skip_serializing_if = "is_default_blueprint_traversal")]
    pub traversal: BlueprintTraversal,
    /// Integrate cheap terminal opponent actions exactly and sample only a
    /// continuation, with its conditional proposal probability. Research-only.
    #[serde(default, skip_serializing_if = "is_false")]
    pub integrate_terminal_actions: bool,
    /// Stateless check/call-to-showdown control variate. Samples the original
    /// opponent proposal, retaining its stopping frequency. Research-only.
    #[serde(default, skip_serializing_if = "is_false")]
    pub opponent_checkdown_baseline: bool,
    pub export_postflop_strategies: bool,
    pub recall_mode: RecallMode,
    pub dcfr: DcfrParameters,
    #[serde(default)]
    pub dcfr_schedule: DcfrSchedule,
    #[serde(default)]
    pub dcfr_schedule_horizon: u64,
    pub evaluation_controls: EvaluationControls,
    pub hand_abstraction: HandAbstraction,
    pub showdown_evaluation: ShowdownEvaluation,
    pub action_abstraction: ActionAbstraction,
}

const fn default_traverser_hand_batch_size() -> usize {
    1
}

const fn default_opponent_hand_batch_size() -> usize {
    1
}

fn is_default_hand_batch_size(size: &usize) -> bool {
    *size == 1
}

fn is_default_blueprint_traversal(traversal: &BlueprintTraversal) -> bool {
    *traversal == BlueprintTraversal::ExternalSampling
}

fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BlueprintTraversal {
    #[default]
    ExternalSampling,
    PublicChanceSampling,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DcfrSchedule {
    #[default]
    Fixed,
    /// HS-DCFR(30) from Zhang, McAleer, and Sandholm (AAAI 2026).
    Hs30,
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
            traverser_hand_batch_size: 1,
            opponent_hand_batch_size: 1,
            traversal: BlueprintTraversal::ExternalSampling,
            integrate_terminal_actions: false,
            opponent_checkdown_baseline: false,
            export_postflop_strategies: false,
            recall_mode: RecallMode::Trajectory,
            dcfr: DcfrParameters::default(),
            dcfr_schedule: DcfrSchedule::Fixed,
            dcfr_schedule_horizon: 0,
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
        if (self.integrate_terminal_actions || self.opponent_checkdown_baseline)
            && self.traversal != BlueprintTraversal::PublicChanceSampling
        {
            return Err("opponent variance reduction requires public-chance sampling".to_owned());
        }
        if self.integrate_terminal_actions && self.opponent_checkdown_baseline {
            return Err(
                "choose terminal integration or the checkdown baseline, not both".to_owned(),
            );
        }
        if self.averaging_delay >= self.iterations {
            return Err("averaging delay must be smaller than iterations".to_owned());
        }
        if !(1..=990).contains(&self.traverser_hand_batch_size) {
            return Err("traverser hand batch size must be between 1 and 990".to_owned());
        }
        if !(1..=990).contains(&self.opponent_hand_batch_size) {
            return Err("opponent hand batch size must be between 1 and 990".to_owned());
        }
        if self.traversal == BlueprintTraversal::PublicChanceSampling
            && (self.traverser_hand_batch_size != 1 || self.opponent_hand_batch_size != 1)
        {
            return Err(
                "public-chance sampling cannot be combined with finite hand batches".to_owned(),
            );
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
        match self.dcfr_schedule {
            DcfrSchedule::Fixed if self.dcfr_schedule_horizon != 0 => {
                return Err("fixed DCFR cannot declare a schedule horizon".to_owned());
            }
            DcfrSchedule::Hs30
                if self.dcfr != DcfrParameters::default()
                    || self.dcfr_schedule_horizon < self.iterations
                    || self.dcfr_schedule_horizon < 2 =>
            {
                return Err(
                    "HS-DCFR requires default base parameters and a horizon at least as large as the requested iterations"
                        .to_owned(),
                );
            }
            DcfrSchedule::Hs30 if self.dcfr_schedule_horizon > MAX_HS_DCFR_HORIZON => {
                return Err(format!(
                    "HS-DCFR horizon may not exceed {MAX_HS_DCFR_HORIZON} iterations"
                ));
            }
            _ => {}
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
    hand_bucket_cache: RefCell<BTreeMap<(usize, usize), Arc<str>>>,
    public_bucket_cache: RefCell<BTreeMap<usize, Arc<str>>>,
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
            public_bucket_cache: RefCell::new(BTreeMap::new()),
            showdown_equity_cache: RefCell::new(BTreeMap::new()),
        }
    }

    #[cfg(test)]
    fn from_cards(holes: [[u8; 2]; 2], board: [u8; 5]) -> Self {
        Self::from_sampled_cards(holes, board)
    }

    fn hand_bucket(
        &self,
        player: usize,
        street: Street,
        abstraction: &HandAbstraction,
    ) -> Arc<str> {
        let key = (player, street.board_len());
        if let Some(bucket) = self.hand_bucket_cache.borrow().get(&key) {
            return bucket.clone();
        }
        let bucket: Arc<str> = match street {
            Street::Preflop => format!(
                "preflop:{}",
                Combo::new(self.holes[player][0], self.holes[player][1]).label()
            )
            .into(),
            _ => postflop_hand_bucket(self, player, street, abstraction).into(),
        };
        self.hand_bucket_cache
            .borrow_mut()
            .insert(key, bucket.clone());
        bucket
    }

    fn public_bucket(&self, street: Street) -> Arc<str> {
        let board_len = street.board_len();
        if let Some(bucket) = self.public_bucket_cache.borrow().get(&board_len) {
            return bucket.clone();
        }
        let bucket: Arc<str> = public_board_bucket(&self.board[..board_len]).into();
        self.public_bucket_cache
            .borrow_mut()
            .insert(board_len, bucket.clone());
        bucket
    }
}

fn sample_traverser_hand_batch(
    template: &Deal,
    traverser: usize,
    requested: usize,
    rng: &mut SplitMix64,
) -> Vec<Deal> {
    if requested == 1 {
        // Preserve the established seeded stream and exact artifact identity
        // for the default scalar trainer.
        return vec![template.clone()];
    }
    let opponent = 1 - traverser;
    let blocked = template
        .board
        .iter()
        .copied()
        .chain(template.holes[opponent]);
    let mut blocked_cards = [false; 52];
    for card in blocked {
        blocked_cards[card as usize] = true;
    }
    let mut candidates = all_combos()
        .into_iter()
        .filter(|combo| {
            combo
                .cards()
                .iter()
                .all(|card| !blocked_cards[*card as usize])
        })
        .collect::<Vec<_>>();
    debug_assert_eq!(candidates.len(), 990);
    let selected = requested.min(candidates.len());
    for index in 0..selected {
        let swap = index + rng.index(candidates.len() - index);
        candidates.swap(index, swap);
    }
    candidates
        .into_iter()
        .take(selected)
        .map(|combo| {
            let mut holes = template.holes;
            holes[traverser] = combo.cards();
            Deal::from_sampled_cards(holes, template.board)
        })
        .collect()
}

fn sample_joint_hand_batch(
    template: &Deal,
    traverser: usize,
    traverser_requested: usize,
    opponent_requested: usize,
    rng: &mut SplitMix64,
) -> Vec<Deal> {
    if opponent_requested == 1 {
        return sample_traverser_hand_batch(template, traverser, traverser_requested, rng);
    }
    let opponent = 1 - traverser;
    let traverser_deals =
        sample_traverser_hand_batch(template, traverser, traverser_requested, rng);
    let mut deals = Vec::with_capacity(traverser_deals.len() * opponent_requested);
    for traverser_deal in traverser_deals {
        let traverser_cards = traverser_deal.holes[traverser];
        let mut candidates = all_combos()
            .into_iter()
            .filter(|combo| {
                combo.cards().iter().all(|card| {
                    !traverser_deal.board.contains(card) && !traverser_cards.contains(card)
                })
            })
            .collect::<Vec<_>>();
        debug_assert_eq!(candidates.len(), 990);
        let selected = opponent_requested.min(candidates.len());
        for index in 0..selected {
            let swap = index + rng.index(candidates.len() - index);
            candidates.swap(index, swap);
        }
        for combo in candidates.into_iter().take(selected) {
            let mut holes = traverser_deal.holes;
            holes[opponent] = combo.cards();
            deals.push(Deal::from_sampled_cards(holes, traverser_deal.board));
        }
    }
    deals
}

fn batch_iteration_seed(base_seed: u64, iteration: u64) -> u64 {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&base_seed.to_le_bytes());
    bytes[8..].copy_from_slice(&iteration.to_le_bytes());
    stable_hash(&bytes)
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

/// Cheap legal rollout used only as a control variate, without policy lookup
/// or table mutation. No bets are added beyond calling the existing amount.
fn checkdown_terminal(mut state: GameState, config: &BlueprintConfig) -> Result<GameState, String> {
    while state.terminal.is_none() {
        let action = state
            .legal_actions(config)
            .into_iter()
            .find(|action| matches!(action.kind, ActionKind::Check | ActionKind::Call))
            .ok_or_else(|| "nonterminal checkdown has no check/call action".to_owned())?;
        state = state.apply(&action, config);
    }
    Ok(state)
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
    hand_bucket_trajectory: Arc<[Arc<str>]>,
    public_bucket_trajectory: Arc<[Arc<str>]>,
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
    action_labels: Arc<[Arc<str>]>,
    regrets: Box<[f64]>,
    strategy_sum: Box<[f64]>,
    regret_updates: u64,
    average_visits: u64,
    #[serde(default)]
    last_discount_iteration: u64,
    #[serde(default)]
    last_regret_discount_cumulative_logs: [f64; 2],
}

impl Node {
    fn new(
        descriptor: NodeDescriptor,
        actions: &[LegalAction],
        string_interner: &mut NodeStorageInterner,
    ) -> Self {
        let action_labels = actions
            .iter()
            .map(|action| {
                if let Some(canonical) = string_interner.strings.get(action.label.as_str()) {
                    canonical.clone()
                } else {
                    let label: Arc<str> = action.label.as_str().into();
                    string_interner
                        .strings
                        .insert(action.label.clone(), label.clone());
                    label
                }
            })
            .collect::<Vec<_>>();
        let action_labels = string_interner.intern_slice(&action_labels);
        Self {
            regrets: vec![0.0; action_labels.len()].into_boxed_slice(),
            strategy_sum: vec![0.0; action_labels.len()].into_boxed_slice(),
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
        let mut strategy = self.strategy_sum.to_vec();
        let maximum = strategy.iter().copied().fold(0.0f64, f64::max);
        if maximum > 0.0 {
            for probability in &mut strategy {
                *probability /= maximum;
            }
            let total = strategy.iter().sum::<f64>();
            for probability in &mut strategy {
                *probability /= total;
            }
            strategy
        } else {
            normalize_or_uniform(strategy)
        }
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
    schedule: DcfrSchedule,
    schedule_horizon: u64,
    iteration: u64,
    cumulative_logs: [f64; 2],
}

impl BlueprintDiscountAccumulator {
    fn new(parameters: DcfrParameters, schedule: DcfrSchedule, schedule_horizon: u64) -> Self {
        Self {
            parameters,
            schedule,
            schedule_horizon,
            iteration: 0,
            cumulative_logs: [0.0; 2],
        }
    }

    fn parameters_at(&self, iteration: u64) -> DcfrParameters {
        match self.schedule {
            DcfrSchedule::Fixed => self.parameters.clone(),
            DcfrSchedule::Hs30 => {
                let progress = iteration as f64 / self.schedule_horizon as f64;
                DcfrParameters {
                    positive_regret_exponent: 1.0 + 3.0 * progress,
                    negative_regret_exponent: -1.0 - 2.0 * progress,
                    strategy_exponent: 30.0 - 5.0 * progress,
                }
            }
        }
    }

    fn advance(&mut self, iteration: u64) {
        assert_eq!(iteration, self.iteration + 1);
        let time = iteration as f64;
        let parameters = self.parameters_at(iteration);
        let positive_power = time.powf(parameters.positive_regret_exponent);
        let negative_power = time.powf(parameters.negative_regret_exponent);
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

fn hs_dcfr_strategy_averaging_weights(horizon: u64) -> Vec<f64> {
    let mut weights = vec![0.0; horizon as usize + 1];
    weights[horizon as usize] = 1.0;
    for iteration in (1..horizon).rev() {
        let gamma = 30.0 - 5.0 * iteration as f64 / horizon as f64;
        let discount = (iteration as f64 / (iteration + 1) as f64).powf(gamma);
        weights[iteration as usize] = weights[iteration as usize + 1] * discount;
    }
    weights
}

fn normalize_or_uniform(mut weights: Vec<f64>) -> Vec<f64> {
    let total = weights.iter().sum::<f64>();
    if total > 0.0 && total.is_finite() {
        for weight in &mut weights {
            *weight /= total;
        }
        weights
    } else if total.is_infinite() {
        // Regret matching is homogeneous: a chip-accounting tolerance must
        // never erase positive regret support. Scale only on overflow so
        // ordinary finite inputs retain their established rounding order.
        let maximum = weights.iter().copied().fold(0.0f64, f64::max);
        for weight in &mut weights {
            *weight /= maximum;
        }
        normalize_or_uniform(weights)
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
    #[serde(default)]
    sampled_deals: u64,
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
    sampled_deals: u64,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regret_updates: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub average_visits: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trained_average: Option<bool>,
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

/// Immutable descriptor sequences repeat at millions of betting histories.
/// Share whole sequences as well as their strings; numerical arrays remain
/// private to each information set. Serde still stores ordinary sequences.
#[derive(Default)]
struct NodeStorageInterner {
    strings: BTreeMap<String, Arc<str>>,
    slices: BTreeMap<Vec<Arc<str>>, Arc<[Arc<str>]>>,
}

impl NodeStorageInterner {
    fn intern_slice(&mut self, values: &[Arc<str>]) -> Arc<[Arc<str>]> {
        if let Some(canonical) = self.slices.get(values) {
            return canonical.clone();
        }
        let canonical = values
            .iter()
            .map(|value| {
                self.strings
                    .entry(value.to_string())
                    .or_insert_with(|| value.clone())
                    .clone()
            })
            .collect::<Vec<_>>();
        let shared: Arc<[Arc<str>]> = canonical.clone().into();
        self.slices.insert(canonical, shared.clone());
        shared
    }
}

fn rebuild_string_interner(nodes: &mut BTreeMap<u64, Node>) -> NodeStorageInterner {
    let mut interner = NodeStorageInterner::default();
    for node in nodes.values_mut() {
        node.descriptor.hand_bucket_trajectory =
            interner.intern_slice(&node.descriptor.hand_bucket_trajectory);
        node.descriptor.public_bucket_trajectory =
            interner.intern_slice(&node.descriptor.public_bucket_trajectory);
        node.action_labels = interner.intern_slice(&node.action_labels);
    }
    interner
}

struct Trainer {
    config: BlueprintConfig,
    completed_iterations: u64,
    rng: SplitMix64,
    discounts: BlueprintDiscountAccumulator,
    strategy_averaging_weights: Option<Vec<f64>>,
    sampled_deals: u64,
    terminal_evaluations: u64,
    string_interner: NodeStorageInterner,
    public_histories: BTreeMap<u64, Vec<String>>,
    nodes: BTreeMap<u64, Node>,
}

impl Trainer {
    fn fresh(config: BlueprintConfig) -> Self {
        let strategy_averaging_weights = match config.dcfr_schedule {
            DcfrSchedule::Fixed => None,
            DcfrSchedule::Hs30 => Some(hs_dcfr_strategy_averaging_weights(
                config.dcfr_schedule_horizon,
            )),
        };
        Self {
            rng: SplitMix64::new(config.seed),
            discounts: BlueprintDiscountAccumulator::new(
                config.dcfr.clone(),
                config.dcfr_schedule,
                config.dcfr_schedule_horizon,
            ),
            strategy_averaging_weights,
            config,
            completed_iterations: 0,
            sampled_deals: 0,
            terminal_evaluations: 0,
            string_interner: NodeStorageInterner::default(),
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
                "unsupported checkpoint schema {}; expected {} for this solver binary",
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
        let sampled_deals = if checkpoint.sampled_deals == 0 {
            checkpoint
                .completed_iterations
                .saturating_mul(target.traverser_hand_batch_size as u64)
                .saturating_mul(target.opponent_hand_batch_size as u64)
        } else {
            checkpoint.sampled_deals
        };
        let string_interner = rebuild_string_interner(&mut checkpoint.nodes);
        Ok(Self {
            config: target.clone(),
            completed_iterations: checkpoint.completed_iterations,
            rng: SplitMix64::from_state(checkpoint.rng_state),
            discounts: BlueprintDiscountAccumulator {
                parameters: target.dcfr.clone(),
                schedule: target.dcfr_schedule,
                schedule_horizon: target.dcfr_schedule_horizon,
                iteration: checkpoint.completed_iterations,
                cumulative_logs: checkpoint.regret_discount_cumulative_logs,
            },
            strategy_averaging_weights: match target.dcfr_schedule {
                DcfrSchedule::Fixed => None,
                DcfrSchedule::Hs30 => Some(hs_dcfr_strategy_averaging_weights(
                    target.dcfr_schedule_horizon,
                )),
            },
            sampled_deals,
            terminal_evaluations: checkpoint.terminal_evaluations,
            string_interner,
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
            sampled_deals: self.sampled_deals,
            terminal_evaluations: self.terminal_evaluations,
            public_histories: &self.public_histories,
            nodes: &self.nodes,
        };
        if is_message_pack_checkpoint(path) {
            write_message_pack_atomic(path, &checkpoint)
        } else {
            write_json_atomic(path, &checkpoint)
        }
    }

    fn strategy_averaging_weight(&self, iteration: u64) -> f64 {
        self.strategy_averaging_weights.as_ref().map_or_else(
            || dcfr_strategy_averaging_weight(iteration, &self.config.dcfr),
            |weights| weights[iteration as usize],
        )
    }

    fn intern_descriptor_buckets(&mut self, descriptor: &mut NodeDescriptor) {
        descriptor.hand_bucket_trajectory = self
            .string_interner
            .intern_slice(&descriptor.hand_bucket_trajectory);
        descriptor.public_bucket_trajectory = self
            .string_interner
            .intern_slice(&descriptor.public_bucket_trajectory);
    }

    fn train(&mut self, control: &RunControl) -> Result<(), Box<dyn Error>> {
        let starting_iteration = self.completed_iterations;
        let mut last_checkpoint_iteration = None;
        while self.completed_iterations < self.config.iterations {
            self.discounts.advance(self.completed_iterations + 1);
            let traverser = self.completed_iterations as usize % 2;
            let template = Deal::sample(&mut self.rng);
            match self.config.traversal {
                BlueprintTraversal::ExternalSampling => {
                    let batched = self.config.traverser_hand_batch_size > 1
                        || self.config.opponent_hand_batch_size > 1;
                    let mut batch_rng = SplitMix64::new(batch_iteration_seed(
                        self.config.seed,
                        self.completed_iterations + 1,
                    ));
                    let deals = sample_joint_hand_batch(
                        &template,
                        traverser,
                        self.config.traverser_hand_batch_size,
                        self.config.opponent_hand_batch_size,
                        if batched {
                            &mut batch_rng
                        } else {
                            &mut self.rng
                        },
                    );
                    let sample_weight = 1.0 / deals.len() as f64;
                    if deals.len() == 1 {
                        self.external_sampling(
                            GameState::initial(&self.config),
                            &deals[0],
                            traverser,
                            sample_weight,
                        );
                    } else {
                        self.external_sampling_batch(
                            GameState::initial(&self.config),
                            &deals,
                            traverser,
                            sample_weight,
                            &mut batch_rng,
                        );
                    }
                    self.sampled_deals += deals.len() as u64;
                }
                BlueprintTraversal::PublicChanceSampling => {
                    self.public_chance_iteration(template.board, traverser)
                        .map_err(|error| format!("public-chance traversal failed: {error}"))?;
                }
            }
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
                    eprintln!(
                        "blueprint checkpoint: iteration={} information_sets={} path={path}",
                        self.completed_iterations,
                        self.nodes.len()
                    );
                    last_checkpoint_iteration = Some(self.completed_iterations);
                }
            }
        }
        if self.completed_iterations > starting_iteration
            && last_checkpoint_iteration != Some(self.completed_iterations)
        {
            if let Some(path) = &control.checkpoint_path {
                self.write_checkpoint(Path::new(path))?;
                eprintln!(
                    "blueprint checkpoint: iteration={} information_sets={} path={path}",
                    self.completed_iterations,
                    self.nodes.len()
                );
            }
        }
        Ok(())
    }

    fn current_strategy_at(
        &mut self,
        state: &GameState,
        deal: &Deal,
        actions: &[LegalAction],
    ) -> (u64, Vec<f64>) {
        let (key, mut descriptor, public_history) = information_set(state, deal, &self.config);
        self.intern_descriptor_buckets(&mut descriptor);
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
        let string_interner = &mut self.string_interner;
        let node = self
            .nodes
            .entry(key)
            .or_insert_with(|| Node::new(descriptor.clone(), actions, string_interner));
        assert_eq!(
            node.descriptor, descriptor,
            "information-set hash collision detected"
        );
        assert!(
            node.action_labels
                .iter()
                .map(AsRef::as_ref)
                .eq(actions.iter().map(|action| action.label.as_str())),
            "one abstraction key produced incompatible action sets"
        );
        node.apply_dcfr_regret_discount(self.completed_iterations + 1, &self.discounts);
        (key, node.current_strategy())
    }

    fn current_range_strategies_at(
        &mut self,
        state: &GameState,
        cache: &mut range_vector::PublicInformationSetCache,
        active_combos: &[bool],
        actions: &[LegalAction],
    ) -> Result<(Vec<Option<u64>>, Vec<Vec<f64>>), String> {
        let mapping = cache.map(state, &self.config, active_combos)?;
        let mut strategies = BTreeMap::<u64, Vec<f64>>::new();
        for (key, (mut descriptor, public_history)) in mapping.descriptors {
            if !self.nodes.contains_key(&key)
                && self.nodes.len() >= self.config.max_information_sets
            {
                return Err("public-chance traversal reached the information-set guard".to_owned());
            }
            self.intern_descriptor_buckets(&mut descriptor);
            match self.public_histories.get(&descriptor.public_history_id) {
                Some(existing) if existing != &public_history => {
                    return Err("public-history hash collision in public-range lookup".to_owned());
                }
                Some(_) => {}
                None => {
                    self.public_histories
                        .insert(descriptor.public_history_id, public_history);
                }
            }
            let string_interner = &mut self.string_interner;
            let node = self
                .nodes
                .entry(key)
                .or_insert_with(|| Node::new(descriptor.clone(), actions, string_interner));
            if node.descriptor != descriptor
                || !node
                    .action_labels
                    .iter()
                    .map(AsRef::as_ref)
                    .eq(actions.iter().map(|action| action.label.as_str()))
            {
                return Err(
                    "one public-range abstraction key produced incompatible node data".to_owned(),
                );
            }
            node.apply_dcfr_regret_discount(self.completed_iterations + 1, &self.discounts);
            strategies.insert(key, node.current_strategy());
        }

        let uniform = 1.0 / actions.len() as f64;
        let mut action_probabilities =
            vec![vec![uniform; range_vector::EXACT_COMBO_COUNT]; actions.len()];
        for (combo, key) in mapping.keys.iter().enumerate() {
            let Some(key) = key else {
                continue;
            };
            let strategy = &strategies[key];
            for (action, probability) in strategy.iter().enumerate() {
                action_probabilities[action][combo] = *probability;
            }
        }
        Ok((mapping.keys, action_probabilities))
    }

    fn apply_range_information_set_updates(
        &mut self,
        updates: BTreeMap<u64, range_vector::RangeInformationSetUpdate>,
        apply_regrets: bool,
        apply_average: bool,
    ) -> Result<(), String> {
        for (key, update) in updates {
            let node = self
                .nodes
                .get_mut(&key)
                .ok_or_else(|| "public-range update references a missing node".to_owned())?;
            if node.regrets.len() != update.regret_deltas_bb.len()
                || node.strategy_sum.len() != update.strategy_deltas.len()
            {
                return Err("public-range update action count differs from its node".to_owned());
            }
            if apply_regrets {
                for (regret, delta) in node.regrets.iter_mut().zip(update.regret_deltas_bb) {
                    *regret += delta;
                }
                node.regret_updates += update.regret_contributions;
            }
            if apply_average {
                for (sum, delta) in node.strategy_sum.iter_mut().zip(update.strategy_deltas) {
                    *sum += delta;
                }
                node.average_visits += update.average_contributions;
            }
        }
        Ok(())
    }

    fn public_chance_iteration(&mut self, board: [u8; 5], traverser: usize) -> Result<(), String> {
        let combos = all_combos();
        debug_assert_eq!(combos.len(), range_vector::EXACT_COMBO_COUNT);
        let active = combos
            .iter()
            .map(|combo| !combo.cards().iter().any(|card| board.contains(card)))
            .collect::<Vec<_>>();
        let active_count = active.iter().filter(|active| **active).count();
        if active_count != 1_081 {
            return Err(format!(
                "sampled board has {active_count} compatible exact combos; expected 1081"
            ));
        }
        // For any board-compatible hero combo exactly C(45, 2) opponent
        // combos remain. This normalization makes each terminal CFV an
        // expected bb value rather than an unscaled sum over opponent hands.
        let realization_weight = 1.0 / 990.0;
        let private_chance_weight = 1.0 / active_count as f64;
        let ranges = std::array::from_fn(|_| {
            active
                .iter()
                .map(|active| if *active { realization_weight } else { 0.0 })
                .collect::<Vec<_>>()
        });
        let private_chance = active
            .iter()
            .map(|active| if *active { private_chance_weight } else { 0.0 })
            .collect::<Vec<_>>();
        let mut action_rng = SplitMix64::new(batch_iteration_seed(
            self.config.seed ^ 0x7063_732d_726e_6701,
            self.completed_iterations + 1,
        ));
        let mut information_set_cache = range_vector::PublicInformationSetCache::new(board)?;
        self.public_chance_external_sampling(
            GameState::initial(&self.config),
            board,
            &ranges,
            &private_chance,
            traverser,
            &mut action_rng,
            &mut information_set_cache,
        )?;
        self.sampled_deals += active_count as u64;
        Ok(())
    }

    fn public_chance_external_sampling(
        &mut self,
        state: GameState,
        board: [u8; 5],
        ranges: &[Vec<f64>; 2],
        private_chance: &[f64],
        traverser: usize,
        rng: &mut SplitMix64,
        information_set_cache: &mut range_vector::PublicInformationSetCache,
    ) -> Result<Vec<f64>, String> {
        if let Some(terminal) = &state.terminal {
            self.terminal_evaluations += 1;
            let terminal = match terminal {
                Terminal::Fold { winner } => {
                    range_vector::RangeTerminalKind::Fold { winner: *winner }
                }
                Terminal::Showdown => range_vector::RangeTerminalKind::Showdown,
            };
            let evaluation = information_set_cache.terminal_values(
                state.invested,
                &ranges[1 - traverser],
                traverser,
                terminal,
            )?;
            return Ok(evaluation);
        }

        let actions = state.legal_actions(&self.config);
        if actions.is_empty() {
            return Err("non-terminal public state has no legal actions".to_owned());
        }
        let actor = state.actor;
        let active = if actor == traverser {
            private_chance
                .iter()
                .map(|weight| *weight > 0.0)
                .collect::<Vec<_>>()
        } else {
            ranges[actor]
                .iter()
                .map(|reach| *reach > 0.0)
                .collect::<Vec<_>>()
        };
        let (keys, probabilities) =
            self.current_range_strategies_at(&state, information_set_cache, &active, &actions)?;

        if actor == traverser {
            let mut action_values = Vec::with_capacity(actions.len());
            for action in &actions {
                action_values.push(self.public_chance_external_sampling(
                    state.apply(action, &self.config),
                    board,
                    ranges,
                    private_chance,
                    traverser,
                    rng,
                    information_set_cache,
                )?);
            }
            let mut expected = std::array::from_fn(|_| vec![0.0; range_vector::EXACT_COMBO_COUNT]);
            for combo in 0..range_vector::EXACT_COMBO_COUNT {
                for action in 0..actions.len() {
                    expected[traverser][combo] +=
                        probabilities[action][combo] * action_values[action][combo];
                }
            }
            let evaluation = range_vector::RangeActionEvaluation {
                actor_action_values_bb: action_values,
                expected_counterfactual_values_bb: expected,
            };
            let updates = range_vector::aggregate_information_set_updates(
                actor,
                &keys,
                private_chance,
                &ranges[actor],
                &probabilities,
                &evaluation,
                None,
            )?;
            self.apply_range_information_set_updates(updates, true, false)?;
            return Ok(evaluation.expected_counterfactual_values_bb[traverser].clone());
        }

        if self.completed_iterations >= self.config.averaging_delay {
            let updates = range_vector::aggregate_average_strategy_updates(
                &keys,
                &ranges[actor],
                &probabilities,
                self.strategy_averaging_weight(self.completed_iterations + 1),
            )?;
            self.apply_range_information_set_updates(updates, false, true)?;
        }

        // Sample one shared public action from the reached range mixture.
        // Multiplying each exact combo by sigma_i(a) / q(a) makes the returned
        // traverser CFV unbiased while avoiding one recursive branch per
        // opponent information set.
        let proposal = range_vector::opponent_action_proposal(&ranges[actor], &probabilities)?;
        if self.config.opponent_checkdown_baseline {
            let children = actions
                .iter()
                .map(|action| state.apply(action, &self.config))
                .collect::<Vec<_>>();
            let mut baselines = Vec::with_capacity(actions.len());
            for (action, child) in children.iter().enumerate() {
                if proposal[action] == 0.0 {
                    baselines.push(vec![0.0; range_vector::EXACT_COMBO_COUNT]);
                    continue;
                }
                let branch_ranges = range_vector::importance_sample_opponent_range(
                    ranges,
                    actor,
                    &probabilities,
                    action,
                    1.0,
                )?;
                // This is a training control variate, never a policy/value
                // replacement. It may use the sampled runout but creates no
                // information sets and makes no hidden-information decisions.
                let terminal = checkdown_terminal(child.clone(), &self.config)?;
                let kind = match terminal.terminal {
                    Some(Terminal::Fold { winner }) => {
                        range_vector::RangeTerminalKind::Fold { winner }
                    }
                    Some(Terminal::Showdown) => range_vector::RangeTerminalKind::Showdown,
                    None => unreachable!("checkdown must terminate"),
                };
                baselines.push(information_set_cache.terminal_values(
                    terminal.invested,
                    &branch_ranges[actor],
                    traverser,
                    kind,
                )?);
            }
            let mut values = vec![0.0; range_vector::EXACT_COMBO_COUNT];
            for baseline in &baselines {
                for (value, contribution) in values.iter_mut().zip(baseline) {
                    *value += contribution;
                }
            }
            let selected = sample_index(&proposal, rng);
            if children[selected].terminal.is_none() {
                let child_ranges = range_vector::importance_sample_opponent_range(
                    ranges,
                    actor,
                    &probabilities,
                    selected,
                    proposal[selected],
                )?;
                let sampled = self.public_chance_external_sampling(
                    children[selected].clone(),
                    board,
                    &child_ranges,
                    private_chance,
                    traverser,
                    rng,
                    information_set_cache,
                )?;
                // Control-variate identity: Davis et al. (ICML 2020), Eq. 4:
                // https://proceedings.mlr.press/v119/davis20a.html
                // Child CFVs already include 1/q through the opponent range.
                // Dividing them again would bias this estimator.
                range_vector::add_baseline_residual(
                    &mut values,
                    &sampled,
                    &baselines[selected],
                    proposal[selected],
                )?;
            }
            // A selected terminal has an exact baseline and zero residual.
            return Ok(values);
        }
        if self.config.integrate_terminal_actions {
            let children = actions
                .iter()
                .map(|action| state.apply(action, &self.config))
                .collect::<Vec<_>>();
            let terminal_mask = children
                .iter()
                .map(|child| child.terminal.is_some())
                .collect::<Vec<_>>();
            let mut values = vec![0.0; range_vector::EXACT_COMBO_COUNT];
            for (action, child) in children.iter().enumerate() {
                if !terminal_mask[action] || proposal[action] == 0.0 {
                    continue;
                }
                let terminal_ranges = range_vector::importance_sample_opponent_range(
                    ranges,
                    actor,
                    &probabilities,
                    action,
                    1.0,
                )?;
                let terminal_values = self.public_chance_external_sampling(
                    child.clone(),
                    board,
                    &terminal_ranges,
                    private_chance,
                    traverser,
                    rng,
                    information_set_cache,
                )?;
                for (value, contribution) in values.iter_mut().zip(terminal_values) {
                    *value += contribution;
                }
            }
            if let Some(conditional) =
                range_vector::continuation_action_proposal(&proposal, &terminal_mask)?
            {
                let selected = sample_index(&conditional, rng);
                let child_ranges = range_vector::importance_sample_opponent_range(
                    ranges,
                    actor,
                    &probabilities,
                    selected,
                    conditional[selected],
                )?;
                let continuation = self.public_chance_external_sampling(
                    children[selected].clone(),
                    board,
                    &child_ranges,
                    private_chance,
                    traverser,
                    rng,
                    information_set_cache,
                )?;
                for (value, contribution) in values.iter_mut().zip(continuation) {
                    *value += contribution;
                }
            }
            return Ok(values);
        }
        let selected = sample_index(&proposal, rng);
        let child_ranges = range_vector::importance_sample_opponent_range(
            ranges,
            actor,
            &probabilities,
            selected,
            proposal[selected],
        )?;
        self.public_chance_external_sampling(
            state.apply(&actions[selected], &self.config),
            board,
            &child_ranges,
            private_chance,
            traverser,
            rng,
            information_set_cache,
        )
    }

    /// Traverse a compatible active-player hand population as one public
    /// tree. Every hand remains an independent external-sampling lane, but
    /// lanes selecting the same opponent action share one contiguous recursive
    /// call. Traverser actions remain fully enumerated for every private hand.
    fn external_sampling_batch(
        &mut self,
        state: GameState,
        deals: &[Deal],
        traverser: usize,
        sample_weight: f64,
        rng: &mut SplitMix64,
    ) -> Vec<f64> {
        debug_assert!(deals.len() > 1);
        debug_assert!(deals.iter().all(|deal| deal.board == deals[0].board));
        let deal_refs = deals.iter().collect::<Vec<_>>();
        self.external_sampling_batch_refs(state, &deal_refs, traverser, sample_weight, rng)
    }

    fn external_sampling_batch_refs(
        &mut self,
        state: GameState,
        deals: &[&Deal],
        traverser: usize,
        sample_weight: f64,
        rng: &mut SplitMix64,
    ) -> Vec<f64> {
        debug_assert!(!deals.is_empty());
        if state.terminal.is_some() {
            self.terminal_evaluations += deals.len() as u64;
            return deals
                .iter()
                .map(|deal| {
                    let utility = state.utility_p0(deal, &self.config);
                    if traverser == 0 {
                        utility
                    } else {
                        -utility
                    }
                })
                .collect();
        }

        let actions = state.legal_actions(&self.config);
        debug_assert!(!actions.is_empty());
        if state.actor != traverser {
            // Policy lookup varies with the sampled opponent hand. Lanes that
            // select the same public action still share one recursive call.
            let accumulate_average = self.completed_iterations >= self.config.averaging_delay;
            let averaging_weight = accumulate_average
                .then(|| self.strategy_averaging_weight(self.completed_iterations + 1));
            let mut average_deltas = BTreeMap::<u64, (Vec<f64>, u64)>::new();
            let mut grouped_deals = vec![Vec::<&Deal>::new(); actions.len()];
            let mut grouped_positions = vec![Vec::<usize>::new(); actions.len()];
            for (position, deal) in deals.iter().enumerate() {
                let (key, strategy) = self.current_strategy_at(&state, deal, &actions);
                if let Some(averaging_weight) = averaging_weight {
                    let delta = average_deltas
                        .entry(key)
                        .or_insert_with(|| (vec![0.0; actions.len()], 0));
                    for (sum, probability) in delta.0.iter_mut().zip(&strategy) {
                        *sum += sample_weight * averaging_weight * probability;
                    }
                    delta.1 += 1;
                }
                let selected = sample_index(&strategy, rng);
                grouped_deals[selected].push(*deal);
                grouped_positions[selected].push(position);
            }
            for (key, (deltas, visits)) in average_deltas {
                let node = self.nodes.get_mut(&key).expect("batch node inserted");
                for (sum, delta) in node.strategy_sum.iter_mut().zip(deltas) {
                    *sum += delta;
                }
                node.average_visits += visits;
            }
            let mut values = vec![0.0; deals.len()];
            for action in 0..actions.len() {
                if grouped_deals[action].is_empty() {
                    continue;
                }
                let child = self.external_sampling_batch_refs(
                    state.apply(&actions[action], &self.config),
                    &grouped_deals[action],
                    traverser,
                    sample_weight,
                    rng,
                );
                for (position, value) in grouped_positions[action].iter().zip(child) {
                    values[*position] = value;
                }
            }
            return values;
        }

        let mut keys = Vec::with_capacity(deals.len());
        let mut strategies = Vec::with_capacity(deals.len());
        for deal in deals {
            let (key, strategy) = self.current_strategy_at(&state, deal, &actions);
            keys.push(key);
            strategies.push(strategy);
        }
        let children = actions
            .iter()
            .map(|action| {
                self.external_sampling_batch_refs(
                    state.apply(action, &self.config),
                    deals,
                    traverser,
                    sample_weight,
                    rng,
                )
            })
            .collect::<Vec<_>>();
        let mut values = vec![0.0; deals.len()];
        let mut regret_deltas = BTreeMap::<u64, (Vec<f64>, u64)>::new();
        for deal_index in 0..deals.len() {
            let strategy = &strategies[deal_index];
            let node_value = strategy
                .iter()
                .enumerate()
                .map(|(action, probability)| probability * children[action][deal_index])
                .sum::<f64>();
            values[deal_index] = node_value;
            let delta = regret_deltas
                .entry(keys[deal_index])
                .or_insert_with(|| (vec![0.0; actions.len()], 0));
            for (action, regret) in delta.0.iter_mut().enumerate() {
                *regret += sample_weight * (children[action][deal_index] - node_value);
            }
            delta.1 += 1;
        }
        for (key, (deltas, updates)) in regret_deltas {
            let node = self.nodes.get_mut(&key).expect("batch node inserted");
            for (regret, delta) in node.regrets.iter_mut().zip(deltas) {
                *regret += delta;
            }
            node.regret_updates += updates;
        }
        values
    }

    fn external_sampling(
        &mut self,
        state: GameState,
        deal: &Deal,
        traverser: usize,
        sample_weight: f64,
    ) -> f64 {
        if state.terminal.is_some() {
            self.terminal_evaluations += 1;
            let utility = state.utility_p0(deal, &self.config);
            return if traverser == 0 { utility } else { -utility };
        }

        let actions = state.legal_actions(&self.config);
        debug_assert!(!actions.is_empty());
        let (key, strategy) = self.current_strategy_at(&state, deal, &actions);

        if state.actor == traverser {
            let mut action_values = Vec::with_capacity(actions.len());
            for action in &actions {
                action_values.push(self.external_sampling(
                    state.apply(action, &self.config),
                    deal,
                    traverser,
                    sample_weight,
                ));
            }
            let node_value = strategy
                .iter()
                .zip(&action_values)
                .map(|(probability, value)| probability * value)
                .sum::<f64>();
            let node = self.nodes.get_mut(&key).expect("node inserted");
            for (regret, action_value) in node.regrets.iter_mut().zip(action_values) {
                *regret += sample_weight * (action_value - node_value);
            }
            node.regret_updates += 1;
            node_value
        } else {
            // OpenSpiel-style external sampling: opponent actions are sampled
            // directly from sigma. Because behavior == target, no importance
            // ratio is needed. Simple averaging happens at these opponent
            // nodes during the other player's traversal.
            if self.completed_iterations >= self.config.averaging_delay {
                let averaging_weight =
                    self.strategy_averaging_weight(self.completed_iterations + 1);
                let node = self.nodes.get_mut(&key).expect("node inserted");
                for (sum, probability) in node.strategy_sum.iter_mut().zip(&strategy) {
                    *sum += sample_weight * averaging_weight * probability;
                }
                node.average_visits += 1;
            }
            let selected = sample_index(&strategy, &mut self.rng);
            self.external_sampling(
                state.apply(&actions[selected], &self.config),
                deal,
                traverser,
                sample_weight,
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
        let mut training_config = serde_json::json!({
            "solver_version": SOLVER_VERSION,
            "model": MODEL,
            "small_blind_bb": self.config.small_blind_bb,
            "big_blind_bb": self.config.big_blind_bb,
            "effective_stack_bb": self.config.effective_stack_bb,
            "iterations": self.config.iterations,
            "max_information_sets": self.config.max_information_sets,
            "seed": self.config.seed,
            "averaging_delay": self.config.averaging_delay,
            "traverser_hand_batch_size": self.config.traverser_hand_batch_size,
            "recall_mode": self.config.recall_mode,
            "dcfr": &self.config.dcfr,
            "dcfr_schedule": self.config.dcfr_schedule,
            "dcfr_schedule_horizon": self.config.dcfr_schedule_horizon,
            "hand_abstraction": &self.config.hand_abstraction,
            "showdown_evaluation": &self.config.showdown_evaluation,
            "action_abstraction": &self.config.action_abstraction,
        });
        if self.config.opponent_hand_batch_size > 1 {
            training_config
                .as_object_mut()
                .expect("training config object")
                .insert(
                    "opponent_hand_batch_size".to_owned(),
                    self.config.opponent_hand_batch_size.into(),
                );
        }
        if self.config.traversal != BlueprintTraversal::ExternalSampling {
            training_config
                .as_object_mut()
                .expect("training config object")
                .insert(
                    "traversal".to_owned(),
                    serde_json::to_value(self.config.traversal)
                        .expect("serializable blueprint traversal"),
                );
        }
        let training_hash_input =
            serde_json::to_vec(&training_config).expect("serializable training configuration");
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
                let public_history = self
                    .public_histories
                    .get(&node.descriptor.public_history_id)
                    .expect("node history interned")
                    .clone();
                let is_root = node.descriptor.actor == Position::ButtonSmallBlind
                    && node.descriptor.street == Street::Preflop
                    && public_history.len() == 1
                    && public_history[0].starts_with("blinds:");
                ExportedInfoSet {
                    key: format!("{key:016x}"),
                    actor: node.descriptor.actor,
                    street: node.descriptor.street,
                    hand_bucket_trajectory: node
                        .descriptor
                        .hand_bucket_trajectory
                        .iter()
                        .map(ToString::to_string)
                        .collect(),
                    public_bucket_trajectory: node
                        .descriptor
                        .public_bucket_trajectory
                        .iter()
                        .map(ToString::to_string)
                        .collect(),
                    public_history,
                    pot_bb: node.descriptor.pot_bb,
                    to_call_bb: node.descriptor.to_call_bb,
                    effective_stack_remaining_bb: node.descriptor.effective_stack_remaining_bb,
                    regret_updates: is_root.then_some(node.regret_updates),
                    average_visits: is_root.then_some(node.average_visits),
                    trained_average: is_root.then_some(node.average_visits > 0),
                    actions: node
                        .action_labels
                        .iter()
                        .map(ToString::to_string)
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
        let traversal_provenance = match self.config.traversal {
            BlueprintTraversal::ExternalSampling => "External-sampling Discounted CFR with alternating traverser updates; traverser actions are enumerated and opponent/chance actions are sampled deterministically from the seeded current strategy.".to_owned(),
            BlueprintTraversal::PublicChanceSampling => "Research public-chance external-sampling Discounted CFR with alternating traverser updates. Each round samples one complete public board, updates all 1,081 board-compatible exact private combos, enumerates traverser actions, and samples one shared opponent action from a reach-weighted proposal with exact per-combo importance correction.".to_owned(),
        };
        let showdown_provenance = match self.config.traversal {
            BlueprintTraversal::ExternalSampling => "All-in showdowns before the river use deterministic conditional-expectation runout evaluation to reduce chance variance; sample counts and exact-turn behavior are recorded in config.showdown_evaluation.".to_owned(),
            BlueprintTraversal::PublicChanceSampling => "All terminal showdowns use the exact five-card public board sampled for that public-chance round; exact compatible-hand settlement preserves card removal and zero-sum utilities.".to_owned(),
        };
        let mut provenance = vec![
            traversal_provenance,
            "Sampled information sets lazily receive every skipped global DCFR regret discount, and average-strategy visits follow the configured fixed or scheduled DCFR weighting recurrence.".to_owned(),
            "Exact private cards and boards are sampled without replacement; information sets use lossy coarse rollout-derived strength/potential and public buckets plus an abstract no-limit action grid.".to_owned(),
            showdown_provenance,
            "Rake-free, equal-stack, heads-up cash model with no ante; button posts the small blind, acts first preflop, and acts last postflop.".to_owned(),
            "Reported regret is a training diagnostic, not exploitability or a Nash-distance certificate.".to_owned(),
            "Root local-deviation evaluation forces one button action at a time against the fixed average continuation/opponent policy. It is a one-step local best response with sampling error and is not exploitability or a full best response.".to_owned(),
            "Root best-action selection and reporting reuse samples. Its maximum has selection bias; reported root confidence bounds are descriptive, not selection-corrected exploitability certificates.".to_owned(),
            "A separate seeded evaluation pass records counterfactual values and standard errors for every action at each reached served information set. Low-sample or >0.02bb-standard-error values are flagged low confidence.".to_owned(),
        ];
        if self.config.integrate_terminal_actions {
            provenance.push("Opponent terminal actions are integrated exactly; remaining public actions use a conditional proposal with per-combo importance correction. This reduces terminal-action sampling variance without changing legal actions or average-policy weights.".to_owned());
        }
        if self.config.opponent_checkdown_baseline {
            provenance.push("Research stateless check/call-to-showdown opponent control variates use the original public-action proposal and importance-corrected residuals. Baselines use the sampled board only during training, do not replace policy values, and do not create persistent nodes. Baseline accuracy affects variance, not the estimator expectation.".to_owned());
        }
        if self.config.dcfr_schedule != DcfrSchedule::Fixed {
            provenance.push(format!(
                "HS-DCFR(30) varies alpha=1+3t/n, beta=-1-2t/n, and gamma=30-5t/n over a pinned {}-iteration horizon. Strategy contributions use the exact terminal-relative product of the changing gamma discounts, and lazy regret discounts preserve every skipped global factor.",
                self.config.dcfr_schedule_horizon
            ));
        } else {
            provenance.push(
                "For fixed DCFR, the repeated average-strategy discount telescopes to the exact global iteration^gamma contribution used at each sampled visit."
                    .to_owned(),
            );
        }
        if self.config.traverser_hand_batch_size > 1 || self.config.opponent_hand_batch_size > 1 {
            provenance.push(format!(
                "Each alternating iteration samples one public board, then trains a deterministic batch of {} active-traverser by {} compatible opponent hands with normalized regret and average-policy weight. Independent opponent-action lanes selecting the same action share contiguous public-tree recursion.",
                self.config.traverser_hand_batch_size,
                self.config.opponent_hand_batch_size,
            ));
        }
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
                sampled_deals: self.sampled_deals,
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
    let mut hash = StableHasher::new();
    hash.update(bytes);
    hash.finish()
}

struct StableHasher {
    state: u64,
}

impl StableHasher {
    const fn new() -> Self {
        Self {
            state: 0xcbf2_9ce4_8422_2325u64,
        }
    }

    fn update(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.state ^= *byte as u64;
            self.state = self.state.wrapping_mul(0x100_0000_01b3);
        }
    }

    fn update_joined<T: AsRef<str>>(&mut self, values: &[T], separator: u8) {
        for (index, value) in values.iter().enumerate() {
            if index > 0 {
                self.update(&[separator]);
            }
            self.update(value.as_ref().as_bytes());
        }
    }

    fn update_unsigned(&mut self, mut value: u64) {
        if value == 0 {
            self.update(b"0");
            return;
        }
        let mut digits = [0u8; 20];
        let mut offset = digits.len();
        while value > 0 {
            offset -= 1;
            digits[offset] = b'0' + (value % 10) as u8;
            value /= 10;
        }
        self.update(&digits[offset..]);
    }

    fn update_fixed_three(&mut self, value: f64) {
        debug_assert!(value.is_finite() && value >= 0.0);
        let milli = (value * 1_000.0).round() as u64;
        self.update_unsigned(milli / 1_000);
        self.update(b".");
        let fraction = milli % 1_000;
        self.update(&[
            b'0' + (fraction / 100) as u8,
            b'0' + ((fraction / 10) % 10) as u8,
            b'0' + (fraction % 10) as u8,
        ]);
    }

    const fn finish(self) -> u64 {
        self.state
    }
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
        let checkpoint = read_checkpoint(Path::new(path))?;
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

fn is_message_pack_checkpoint(path: &Path) -> bool {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    name.ends_with(".msgpack")
        || name.ends_with(".msgpack.gz")
        || name.ends_with(".mpk")
        || name.ends_with(".mpk.gz")
}

fn read_checkpoint(path: &Path) -> Result<BlueprintCheckpoint, Box<dyn Error>> {
    let compressed = path.extension().and_then(|extension| extension.to_str()) == Some("gz");
    let message_pack = is_message_pack_checkpoint(path);
    let file = fs::File::open(path)?;
    if compressed {
        let reader = GzDecoder::new(BufReader::new(file));
        if message_pack {
            Ok(rmp_serde::from_read(reader)?)
        } else {
            Ok(serde_json::from_reader(reader)?)
        }
    } else {
        let reader = BufReader::new(file);
        if message_pack {
            Ok(rmp_serde::from_read(reader)?)
        } else {
            Ok(serde_json::from_reader(reader)?)
        }
    }
}

fn write_message_pack_atomic<T: Serialize + ?Sized>(
    path: &Path,
    value: &T,
) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let temporary = path.with_extension("tmp");
    let file = fs::File::create(&temporary)?;
    if path.extension().and_then(|extension| extension.to_str()) == Some("gz") {
        let buffered = BufWriter::new(file);
        let mut gzip = GzEncoder::new(buffered, Compression::fast());
        rmp_serde::encode::write_named(&mut gzip, value)?;
        let mut buffered = gzip.finish()?;
        buffered.flush()?;
    } else {
        let mut writer = BufWriter::new(file);
        rmp_serde::encode::write_named(&mut writer, value)?;
        writer.flush()?;
    }
    fs::rename(temporary, path)?;
    Ok(())
}

pub fn write_json_atomic<T: Serialize + ?Sized>(
    path: &Path,
    value: &T,
) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let temporary = path.with_extension("tmp");
    let file = fs::File::create(&temporary)?;
    if path.extension().and_then(|extension| extension.to_str()) == Some("gz") {
        let buffered = BufWriter::new(file);
        let mut gzip = GzEncoder::new(buffered, Compression::fast());
        serde_json::to_writer(&mut gzip, value)?;
        let mut buffered = gzip.finish()?;
        buffered.flush()?;
    } else {
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, value)?;
        writer.flush()?;
    }
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
    let public_bucket_trajectory = public_bucket_trajectory(deal, state.street);
    information_set_from_bucket_trajectories(
        state,
        config,
        hand_bucket_trajectory,
        public_bucket_trajectory,
    )
}

fn information_set_from_bucket_trajectories(
    state: &GameState,
    config: &BlueprintConfig,
    mut hand_bucket_trajectory: Vec<Arc<str>>,
    mut public_bucket_trajectory: Vec<Arc<str>>,
) -> (u64, NodeDescriptor, Vec<String>) {
    if config.recall_mode == RecallMode::CurrentStreet && state.street != Street::Preflop {
        hand_bucket_trajectory = hand_bucket_trajectory.last().cloned().into_iter().collect();
        public_bucket_trajectory = public_bucket_trajectory
            .last()
            .cloned()
            .into_iter()
            .collect();
    }
    let mut public_history_hasher = StableHasher::new();
    public_history_hasher.update_joined(&state.public_history, b'/');
    let public_history_id = public_history_hasher.finish();
    let descriptor = NodeDescriptor {
        actor: Position::for_player(state.actor),
        street: state.street,
        hand_bucket_trajectory: hand_bucket_trajectory.into(),
        public_bucket_trajectory: public_bucket_trajectory.into(),
        public_history_id,
        pot_bb: quantize(state.pot(), 0.001),
        to_call_bb: quantize(state.to_call(), 0.001),
        effective_stack_remaining_bb: quantize(state.remaining(state.actor, config), 0.001),
    };
    let mut identity = StableHasher::new();
    identity.update(match descriptor.street {
        Street::Preflop => b"Preflop",
        Street::Flop => b"Flop",
        Street::Turn => b"Turn",
        Street::River => b"River",
    });
    identity.update(b"|p");
    identity.update_unsigned(state.actor as u64);
    identity.update(b"|h:");
    identity.update_joined(&descriptor.hand_bucket_trajectory, b'>');
    identity.update(b"|b:");
    identity.update_joined(&descriptor.public_bucket_trajectory, b'>');
    identity.update(b"|pot:");
    identity.update_fixed_three(descriptor.pot_bb);
    identity.update(b"|call:");
    identity.update_fixed_three(descriptor.to_call_bb);
    identity.update(b"|stack:");
    identity.update_fixed_three(descriptor.effective_stack_remaining_bb);
    identity.update(b"|");
    identity.update_joined(&state.public_history, b'/');
    (identity.finish(), descriptor, state.public_history.clone())
}

fn hand_bucket_trajectory(
    deal: &Deal,
    player: usize,
    through: Street,
    abstraction: &HandAbstraction,
) -> Vec<Arc<str>> {
    let mut buckets = vec![deal.hand_bucket(player, Street::Preflop, abstraction)];
    for street in [Street::Flop, Street::Turn, Street::River] {
        if street.board_len() > through.board_len() {
            break;
        }
        buckets.push(deal.hand_bucket(player, street, abstraction));
    }
    buckets
}

fn public_bucket_trajectory(deal: &Deal, through: Street) -> Vec<Arc<str>> {
    let mut buckets = Vec::new();
    for street in [Street::Flop, Street::Turn, Street::River] {
        if street.board_len() > through.board_len() {
            break;
        }
        buckets.push(deal.public_bucket(street));
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
    let mut cards = [0u8; 7];
    cards[..2].copy_from_slice(&deal.holes[player]);
    cards[2..2 + board.len()].copy_from_slice(board);
    let cards = &cards[..2 + board.len()];
    let score = evaluate(cards);
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
    let mut available = [0u8; 52];
    let mut available_len = 0usize;
    for card in 0..52u8 {
        if !known[card as usize] {
            available[available_len] = card;
            available_len += 1;
        }
    }
    let mut seed = 0xcbf2_9ce4_8422_2325u64;
    let mut visible = [0u8; 8];
    visible[..2].copy_from_slice(hole);
    visible[..2].sort_unstable();
    visible[2] = 0xff;
    visible[3..3 + board.len()].copy_from_slice(board);
    visible[3..3 + board.len()].sort_unstable();
    for &card in &visible[..3 + board.len()] {
        seed ^= card as u64 + 1;
        seed = seed.wrapping_mul(0x100_0000_01b3);
    }
    let mut rng = SplitMix64::new(seed);
    let missing_board = 5 - board.len();
    let mut equity = 0.0;
    let mut improved = 0u64;
    let mut future_categories = [0u32; 9];
    let current_category = {
        let mut current = [0u8; 7];
        current[..2].copy_from_slice(hole);
        current[2..2 + board.len()].copy_from_slice(board);
        evaluate(&current[..2 + board.len()]) >> 24
    };

    for _ in 0..samples {
        let needed = 2 + missing_board;
        for index in 0..needed {
            let swap = index + rng.index(available_len - index);
            available.swap(index, swap);
        }
        let opponent = [available[0], available[1]];
        let mut completed_board = [0u8; 5];
        completed_board[..board.len()].copy_from_slice(board);
        completed_board[board.len()..].copy_from_slice(&available[2..needed]);
        let mut hero_cards = [0u8; 7];
        hero_cards[..2].copy_from_slice(hole);
        hero_cards[2..].copy_from_slice(&completed_board);
        let mut opponent_cards = [0u8; 7];
        opponent_cards[..2].copy_from_slice(&opponent);
        opponent_cards[2..].copy_from_slice(&completed_board);
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
                        action_labels: node.action_labels.iter().map(ToString::to_string).collect(),
                        actions: vec![ValueAccumulator::default(); actions.len()],
                    }
                });
                debug_assert!(entry
                    .action_labels
                    .iter()
                    .map(String::as_str)
                    .eq(node.action_labels.iter().map(AsRef::as_ref)));
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
    debug_assert!(node
        .action_labels
        .iter()
        .map(AsRef::as_ref)
        .eq(evaluation.action_labels.iter().map(String::as_str)));
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
        .map(ToString::to_string)
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
        Some(node.action_labels[best_index].to_string()),
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

    #[test]
    fn traverser_hand_batch_is_unique_compatible_and_deterministic() {
        let template = fixed_deal();
        let mut first_rng = SplitMix64::new(73);
        let mut second_rng = SplitMix64::new(73);
        let first = sample_traverser_hand_batch(&template, 0, 32, &mut first_rng);
        let second = sample_traverser_hand_batch(&template, 0, 32, &mut second_rng);
        assert_eq!(first.len(), 32);
        assert_eq!(
            first.iter().map(|deal| deal.holes).collect::<Vec<_>>(),
            second.iter().map(|deal| deal.holes).collect::<Vec<_>>()
        );
        let unique = first
            .iter()
            .map(|deal| Combo::new(deal.holes[0][0], deal.holes[0][1]).key())
            .collect::<BTreeSet<_>>();
        assert_eq!(unique.len(), first.len());
        for deal in first {
            assert_eq!(deal.board, template.board);
            assert_eq!(deal.holes[1], template.holes[1]);
            assert!(deal.holes[0]
                .iter()
                .all(|card| !deal.board.contains(card) && !deal.holes[1].contains(card)));
        }
        let mut legacy_rng = SplitMix64::new(73);
        let legacy_state = legacy_rng.state();
        let legacy = sample_traverser_hand_batch(&template, 0, 1, &mut legacy_rng);
        assert_eq!(legacy[0].holes, template.holes);
        assert_eq!(legacy_rng.state(), legacy_state);
    }

    #[test]
    fn joint_hand_batch_samples_compatible_opponent_ranges_deterministically() {
        let template = fixed_deal();
        let mut first_rng = SplitMix64::new(97);
        let mut second_rng = SplitMix64::new(97);
        let first = sample_joint_hand_batch(&template, 0, 3, 4, &mut first_rng);
        let second = sample_joint_hand_batch(&template, 0, 3, 4, &mut second_rng);
        assert_eq!(first.len(), 12);
        assert_eq!(
            first.iter().map(|deal| deal.holes).collect::<Vec<_>>(),
            second.iter().map(|deal| deal.holes).collect::<Vec<_>>()
        );
        for group in first.chunks_exact(4) {
            assert!(group
                .iter()
                .all(|deal| deal.holes[0] == group[0].holes[0] && deal.board == template.board));
            let opponents = group
                .iter()
                .map(|deal| Combo::new(deal.holes[1][0], deal.holes[1][1]).key())
                .collect::<BTreeSet<_>>();
            assert_eq!(opponents.len(), 4);
            assert!(group.iter().all(|deal| {
                deal.holes[0]
                    .iter()
                    .chain(&deal.holes[1])
                    .all(|card| !deal.board.contains(card))
                    && !deal.holes[0]
                        .iter()
                        .any(|card| deal.holes[1].contains(card))
            }));
        }

        let mut direct_rng = SplitMix64::new(101);
        let mut joint_rng = SplitMix64::new(101);
        let direct = sample_traverser_hand_batch(&template, 1, 8, &mut direct_rng);
        let joint = sample_joint_hand_batch(&template, 1, 8, 1, &mut joint_rng);
        assert_eq!(direct_rng.state(), joint_rng.state());
        assert_eq!(
            direct.iter().map(|deal| deal.holes).collect::<Vec<_>>(),
            joint.iter().map(|deal| deal.holes).collect::<Vec<_>>()
        );
    }

    #[test]
    fn contiguous_batch_matches_scalar_regrets_when_every_action_is_terminal() {
        let config = tiny_config();
        let state = GameState {
            street: Street::River,
            actor: 0,
            invested: [5.0, 6.0],
            street_invested: [0.0, 1.0],
            last_full_raise: 1.0,
            aggressions: 1,
            checks: 0,
            raise_reopened: true,
            public_history: vec!["test:river:facing_all_in".to_owned()],
            trajectory: Vec::new(),
            terminal: None,
        };
        let mut deal_rng = SplitMix64::new(79);
        let deals = sample_traverser_hand_batch(&fixed_deal(), 0, 8, &mut deal_rng);
        let sample_weight = 1.0 / deals.len() as f64;

        let mut scalar = Trainer::fresh(config.clone());
        scalar.discounts.advance(1);
        for deal in &deals {
            scalar.external_sampling(state.clone(), deal, 0, sample_weight);
        }
        let mut contiguous = Trainer::fresh(config);
        contiguous.discounts.advance(1);
        let mut contiguous_rng = SplitMix64::new(103);
        contiguous.external_sampling_batch(state, &deals, 0, sample_weight, &mut contiguous_rng);

        assert_eq!(scalar.terminal_evaluations, contiguous.terminal_evaluations);
        assert_eq!(scalar.public_histories, contiguous.public_histories);
        assert_eq!(scalar.nodes.len(), contiguous.nodes.len());
        for (key, scalar_node) in scalar.nodes {
            let contiguous_node = contiguous.nodes.get(&key).expect("matching batch node");
            assert_eq!(scalar_node.descriptor, contiguous_node.descriptor);
            assert_eq!(scalar_node.action_labels, contiguous_node.action_labels);
            assert_eq!(scalar_node.regrets, contiguous_node.regrets);
            assert_eq!(scalar_node.strategy_sum, contiguous_node.strategy_sum);
            assert_eq!(scalar_node.regret_updates, contiguous_node.regret_updates);
            assert_eq!(scalar_node.average_visits, contiguous_node.average_visits);
        }
    }

    #[test]
    fn contiguous_batch_keeps_independent_opponent_sampling_lanes() {
        let config = tiny_config();
        let state = GameState {
            street: Street::River,
            actor: 1,
            invested: [6.0, 5.0],
            street_invested: [1.0, 0.0],
            last_full_raise: 1.0,
            aggressions: 1,
            checks: 0,
            raise_reopened: true,
            public_history: vec!["test:river:opponent_facing_all_in".to_owned()],
            trajectory: Vec::new(),
            terminal: None,
        };
        let mut deal_rng = SplitMix64::new(83);
        let deals = sample_traverser_hand_batch(&fixed_deal(), 0, 8, &mut deal_rng);
        let mut trainer = Trainer::fresh(config);
        trainer.discounts.advance(1);
        let mut batch_rng = SplitMix64::new(107);
        let mut expected_rng = SplitMix64::new(107);
        for _ in &deals {
            sample_index(&[0.5, 0.5], &mut expected_rng);
        }

        trainer.external_sampling_batch(state, &deals, 0, 1.0 / deals.len() as f64, &mut batch_rng);

        assert_eq!(batch_rng.state(), expected_rng.state());
        assert_eq!(trainer.nodes.len(), 1);
        let node = trainer.nodes.values().next().expect("opponent batch node");
        assert_eq!(node.regret_updates, 0);
        assert_eq!(node.average_visits, deals.len() as u64);
        assert_eq!(node.strategy_sum.as_ref(), &[0.5, 0.5]);
        assert_eq!(trainer.terminal_evaluations, deals.len() as u64);
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
        let exported_roots = first
            .strategies
            .iter()
            .filter(|info_set| {
                info_set.actor == Position::ButtonSmallBlind
                    && info_set.public_history.len() == 1
                    && info_set.public_history[0].starts_with("blinds:")
            })
            .collect::<Vec<_>>();
        assert!(!exported_roots.is_empty());
        assert!(exported_roots.iter().all(|info_set| {
            info_set.trained_average == Some(true)
                && info_set.average_visits.is_some_and(|visits| visits > 0)
                && info_set.regret_updates.is_some()
        }));
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
    fn traverser_hand_batch_expands_active_private_coverage_deterministically() {
        let mut scalar_config = tiny_config();
        scalar_config.iterations = 1;
        scalar_config.traverser_hand_batch_size = 1;
        let mut scalar = Trainer::fresh(scalar_config);
        scalar.train(&RunControl::default()).unwrap();

        let mut batch_config = tiny_config();
        batch_config.iterations = 1;
        batch_config.traverser_hand_batch_size = 8;
        let mut first = Trainer::fresh(batch_config.clone());
        first.train(&RunControl::default()).unwrap();
        let mut second = Trainer::fresh(batch_config);
        second.train(&RunControl::default()).unwrap();

        let trained_root_classes = |trainer: &Trainer| {
            trainer
                .nodes
                .values()
                .filter(|node| {
                    node.descriptor.actor == Position::ButtonSmallBlind
                        && node.descriptor.street == Street::Preflop
                        && node.regret_updates > 0
                        && trainer
                            .public_histories
                            .get(&node.descriptor.public_history_id)
                            .is_some_and(|history| history.len() == 1)
                })
                .count()
        };
        assert_eq!(scalar.sampled_deals, 1);
        assert_eq!(first.sampled_deals, 8);
        assert!(trained_root_classes(&first) > trained_root_classes(&scalar));
        assert_eq!(first.rng.state(), second.rng.state());
        assert_eq!(first.sampled_deals, second.sampled_deals);
        assert_eq!(
            serde_json::to_value(BlueprintCheckpointRef {
                schema_version: BLUEPRINT_CHECKPOINT_SCHEMA_VERSION,
                model: MODEL,
                approximate: true,
                config: &first.config,
                completed_iterations: first.completed_iterations,
                rng_state: first.rng.state(),
                regret_discount_cumulative_logs: first.discounts.cumulative_logs,
                sampled_deals: first.sampled_deals,
                terminal_evaluations: first.terminal_evaluations,
                public_histories: &first.public_histories,
                nodes: &first.nodes,
            })
            .unwrap(),
            serde_json::to_value(BlueprintCheckpointRef {
                schema_version: BLUEPRINT_CHECKPOINT_SCHEMA_VERSION,
                model: MODEL,
                approximate: true,
                config: &second.config,
                completed_iterations: second.completed_iterations,
                rng_state: second.rng.state(),
                regret_discount_cumulative_logs: second.discounts.cumulative_logs,
                sampled_deals: second.sampled_deals,
                terminal_evaluations: second.terminal_evaluations,
                public_histories: &second.public_histories,
                nodes: &second.nodes,
            })
            .unwrap()
        );
    }

    #[test]
    fn dcfr_discounting_is_applied_only_to_regrets() {
        let mut node = Node {
            descriptor: NodeDescriptor {
                actor: Position::ButtonSmallBlind,
                street: Street::Preflop,
                hand_bucket_trajectory: vec!["preflop:AA".into()].into(),
                public_bucket_trajectory: Vec::new().into(),
                public_history_id: 1,
                pot_bb: 1.5,
                to_call_bb: 0.5,
                effective_stack_remaining_bb: 99.5,
            },
            action_labels: vec!["fold".into(), "call".into()].into(),
            regrets: vec![4.0, -4.0].into_boxed_slice(),
            strategy_sum: vec![4.0, 4.0].into_boxed_slice(),
            regret_updates: 1,
            average_visits: 1,
            last_discount_iteration: 0,
            last_regret_discount_cumulative_logs: [0.0; 2],
        };
        let mut discounts =
            BlueprintDiscountAccumulator::new(DcfrParameters::default(), DcfrSchedule::Fixed, 0);
        discounts.advance(1);
        node.apply_dcfr_regret_discount(1, &discounts);
        assert_eq!(node.regrets.as_ref(), [2.0, -2.0]);
        assert_eq!(node.strategy_sum.as_ref(), [4.0, 4.0]);
        node.apply_dcfr_regret_discount(1, &discounts);
        assert_eq!(node.regrets.as_ref(), [2.0, -2.0]);
    }

    #[test]
    fn lazy_full_game_dcfr_regret_discount_matches_eager_updates() {
        let parameters = DcfrParameters::default();
        let mut lazy_discounts =
            BlueprintDiscountAccumulator::new(parameters.clone(), DcfrSchedule::Fixed, 0);
        let mut eager_discounts =
            BlueprintDiscountAccumulator::new(parameters, DcfrSchedule::Fixed, 0);
        let descriptor = NodeDescriptor {
            actor: Position::ButtonSmallBlind,
            street: Street::Preflop,
            hand_bucket_trajectory: vec!["preflop:AA".into()].into(),
            public_bucket_trajectory: Vec::new().into(),
            public_history_id: 1,
            pot_bb: 1.5,
            to_call_bb: 0.5,
            effective_stack_remaining_bb: 99.5,
        };
        let mut lazy = Node {
            descriptor: descriptor.clone(),
            action_labels: vec!["fold".into(), "call".into()].into(),
            regrets: vec![3.0, -2.0].into_boxed_slice(),
            strategy_sum: vec![4.0, 1.0].into_boxed_slice(),
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
        assert_eq!(lazy.strategy_sum.as_ref(), [4.0, 1.0]);
        assert_eq!(eager.strategy_sum.as_ref(), [4.0, 1.0]);
    }

    #[test]
    fn dcfr_strategy_average_uses_exact_iteration_to_gamma_weight() {
        let mut parameters = DcfrParameters::default();
        assert_eq!(dcfr_strategy_averaging_weight(2, &parameters), 4.0);
        parameters.strategy_exponent = 3.0;
        assert_eq!(dcfr_strategy_averaging_weight(2, &parameters), 8.0);
    }

    #[test]
    fn hs_dcfr_30_matches_published_linear_schedules() {
        let discounts =
            BlueprintDiscountAccumulator::new(DcfrParameters::default(), DcfrSchedule::Hs30, 1_000);
        assert_eq!(
            discounts.parameters_at(1_000),
            DcfrParameters {
                positive_regret_exponent: 4.0,
                negative_regret_exponent: -3.0,
                strategy_exponent: 25.0,
            }
        );
        assert_eq!(
            discounts.parameters_at(500),
            DcfrParameters {
                positive_regret_exponent: 2.5,
                negative_regret_exponent: -2.0,
                strategy_exponent: 27.5,
            }
        );

        let weights = hs_dcfr_strategy_averaging_weights(1_000);
        assert_eq!(weights[1_000], 1.0);
        let gamma = 30.0 - 5.0 * 500.0 / 1_000.0;
        let expected_ratio = (500.0f64 / 501.0).powf(gamma);
        assert!((weights[500] / weights[501] - expected_ratio).abs() < 1e-14);
        assert!(weights.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn hs_dcfr_average_weights_match_the_published_eager_recurrence() {
        let horizon = 16;
        let lazy = hs_dcfr_strategy_averaging_weights(horizon);
        let mut eager = vec![0.0; horizon as usize + 1];
        for current in 1..=horizon {
            if current > 1 {
                let previous = current - 1;
                let gamma = 30.0 - 5.0 * previous as f64 / horizon as f64;
                let discount = (previous as f64 / current as f64).powf(gamma);
                for weight in &mut eager[1..current as usize] {
                    *weight *= discount;
                }
            }
            eager[current as usize] = 1.0;
        }
        for iteration in 1..=horizon as usize {
            assert!((lazy[iteration] - eager[iteration]).abs() < 1e-15);
        }
    }

    #[test]
    fn hs_dcfr_terminal_relative_weights_preserve_interim_prefix_averages() {
        let horizon = 16;
        let cutoff = 8;
        let terminal_weights = hs_dcfr_strategy_averaging_weights(horizon);
        let direct_numerator = (1..=cutoff)
            .map(|iteration| terminal_weights[iteration as usize] * iteration as f64)
            .sum::<f64>();
        let direct_denominator = (1..=cutoff)
            .map(|iteration| terminal_weights[iteration as usize])
            .sum::<f64>();

        let mut eager_numerator = 0.0;
        let mut eager_denominator = 0.0;
        for current in 1..=cutoff {
            if current > 1 {
                let previous = current - 1;
                let gamma = 30.0 - 5.0 * previous as f64 / horizon as f64;
                let discount = (previous as f64 / current as f64).powf(gamma);
                eager_numerator *= discount;
                eager_denominator *= discount;
            }
            eager_numerator += current as f64;
            eager_denominator += 1.0;
        }
        assert!(
            (direct_numerator / direct_denominator - eager_numerator / eager_denominator).abs()
                < 1e-13
        );
    }

    #[test]
    fn publishable_defaults_use_trajectory_recall() {
        let config = BlueprintConfig::default();
        assert_eq!(config.recall_mode, RecallMode::Trajectory);
        assert_eq!(config.traversal, BlueprintTraversal::ExternalSampling);
        assert_eq!(config.dcfr, DcfrParameters::default());
        assert_eq!(config.dcfr_schedule, DcfrSchedule::Fixed);
        assert_eq!(config.dcfr_schedule_horizon, 0);
        assert!(
            serde_json::to_value(config)
                .expect("serializable default config")
                .get("traversal")
                .is_none(),
            "the default scalar config must preserve its canonical artifact identity"
        );
    }

    #[test]
    fn regret_matching_is_invariant_to_positive_weight_scale() {
        for scale in [1.0, 1e-12, 1e-200, 1e300] {
            let actual = normalize_or_uniform(vec![scale, 3.0 * scale, 0.0]);
            assert_eq!(actual, vec![0.25, 0.75, 0.0], "scale={scale}");
        }
        assert_eq!(normalize_or_uniform(vec![0.0, 0.0]), vec![0.5, 0.5]);
        assert_eq!(normalize_or_uniform(vec![1e308, 1e308]), vec![0.5, 0.5]);
    }

    #[test]
    fn shared_descriptor_storage_survives_checkpoint_decode_without_aliasing_regrets() {
        let config = tiny_config();
        let deal = Deal::from_cards([[51, 50], [45, 44]], [0, 5, 10, 27, 28]);
        let state = GameState::initial(&config);
        let (_, descriptor, _) = information_set(&state, &deal, &config);
        let actions = state.legal_actions(&config);
        let mut interner = NodeStorageInterner::default();
        let mut nodes = BTreeMap::new();
        for key in [1, 2] {
            nodes.insert(key, Node::new(descriptor.clone(), &actions, &mut interner));
        }
        let encoded = rmp_serde::to_vec_named(&nodes).unwrap();
        let mut decoded: BTreeMap<u64, Node> = rmp_serde::from_slice(&encoded).unwrap();
        let _interner = rebuild_string_interner(&mut decoded);
        assert!(Arc::ptr_eq(
            &decoded[&1].descriptor.hand_bucket_trajectory,
            &decoded[&2].descriptor.hand_bucket_trajectory
        ));
        assert!(Arc::ptr_eq(
            &decoded[&1].descriptor.public_bucket_trajectory,
            &decoded[&2].descriptor.public_bucket_trajectory
        ));
        assert!(Arc::ptr_eq(
            &decoded[&1].action_labels,
            &decoded[&2].action_labels
        ));
        assert_eq!(rmp_serde::to_vec_named(&decoded).unwrap(), encoded);
        decoded.get_mut(&1).unwrap().regrets[0] = 7.0;
        assert_eq!(decoded[&2].regrets[0], 0.0);
    }

    #[test]
    fn integrated_opponent_all_in_response_is_exact_without_action_sampling() {
        let mut config = tiny_config();
        config.traversal = BlueprintTraversal::PublicChanceSampling;
        config.integrate_terminal_actions = true;
        let initial = GameState::initial(&config);
        let shove = initial
            .legal_actions(&config)
            .into_iter()
            .find(|action| action.label.contains("all_in"))
            .unwrap();
        let response = initial.apply(&shove, &config);
        assert_eq!(response.actor, 1);
        let actions = response.legal_actions(&config);
        assert_eq!(actions.len(), 2);
        assert!(actions
            .iter()
            .all(|action| response.apply(action, &config).terminal.is_some()));
        let board = [0, 5, 10, 27, 28];
        let combos = all_combos();
        let private: Vec<f64> = combos
            .iter()
            .map(|combo| {
                if combo.cards().iter().any(|c| board.contains(c)) {
                    0.0
                } else {
                    1.0 / 1081.0
                }
            })
            .collect();
        let ranges = [private.clone(), private.clone()];
        let mut trainer = Trainer::fresh(config.clone());
        trainer.discounts.advance(1);
        let mut rng = SplitMix64::new(29);
        let rng_before = rng.state();
        let mut cache = range_vector::PublicInformationSetCache::new(board).unwrap();
        let actual = trainer
            .public_chance_external_sampling(
                response.clone(),
                board,
                &ranges,
                &private,
                0,
                &mut rng,
                &mut cache,
            )
            .unwrap();
        assert_eq!(
            rng.state(),
            rng_before,
            "all actions were integrated exactly"
        );
        let mut expected = vec![0.0; range_vector::EXACT_COMBO_COUNT];
        for action in &actions {
            let child = response.apply(action, &config);
            let terminal = match child.terminal.unwrap() {
                Terminal::Fold { winner } => range_vector::RangeTerminalKind::Fold { winner },
                Terminal::Showdown => range_vector::RangeTerminalKind::Showdown,
            };
            let reference =
                range_vector::evaluate_terminal_ranges(board, child.invested, &ranges, terminal)
                    .unwrap();
            for (value, cfv) in expected
                .iter_mut()
                .zip(&reference.counterfactual_values_bb[0])
            {
                *value += 0.5 * cfv;
            }
        }
        for (key, mass) in private.iter().enumerate() {
            if *mass > 0.0 {
                assert!((actual[key] - expected[key]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn public_chance_checkpoint_resume_matches_uninterrupted_training() {
        for mode in 0..3 {
            let path = std::env::temp_dir().join(format!(
                "blueprint-public-chance-checkpoint-{}-{}-{mode}.msgpack.gz",
                std::process::id(),
                std::thread::current().name().unwrap_or("test")
            ));
            let mut target = tiny_config();
            target.traversal = BlueprintTraversal::PublicChanceSampling;
            target.integrate_terminal_actions = mode == 1;
            target.opponent_checkdown_baseline = mode == 2;
            target.max_information_sets = 500_000;
            let mut partial = target.clone();
            partial.iterations = 1;
            solve_controlled(
                partial,
                RunControl {
                    checkpoint_path: Some(path.to_string_lossy().into_owned()),
                    checkpoint_every: 1,
                    resume_path: None,
                },
            )
            .expect("write public-chance partial checkpoint");
            let mut old_schema = read_checkpoint(&path).expect("read partial checkpoint");
            old_schema.schema_version = BLUEPRINT_CHECKPOINT_SCHEMA_VERSION - 1;
            assert!(Trainer::from_checkpoint(old_schema, &target).is_err());
            let resumed = solve_controlled(
                target.clone(),
                RunControl {
                    checkpoint_path: None,
                    checkpoint_every: 0,
                    resume_path: Some(path.to_string_lossy().into_owned()),
                },
            )
            .expect("resume public-chance checkpoint");
            let mut changed_mode = target.clone();
            changed_mode.integrate_terminal_actions = mode != 1;
            changed_mode.opponent_checkdown_baseline = false;
            assert!(
                solve_controlled(
                    changed_mode,
                    RunControl {
                        checkpoint_path: None,
                        checkpoint_every: 0,
                        resume_path: Some(path.to_string_lossy().into_owned()),
                    }
                )
                .is_err(),
                "resuming must pin the estimator, not merely the action grid"
            );
            let uninterrupted = solve(target).expect("uninterrupted public-chance solve");
            assert_eq!(resumed, uninterrupted);
            fs::remove_file(path).expect("remove public-chance checkpoint");
        }
    }

    #[test]
    fn checkdown_baseline_preserves_terminal_accounting_and_requires_pcs() {
        let mut config = tiny_config();
        config.opponent_checkdown_baseline = true;
        assert!(config.validate().is_err());
        config.traversal = BlueprintTraversal::PublicChanceSampling;
        assert!(config.validate().is_ok());
        config.integrate_terminal_actions = true;
        assert!(config.validate().is_err());
        config.integrate_terminal_actions = false;
        let initial = GameState::initial(&config);
        for action in initial.legal_actions(&config) {
            let child = initial.apply(&action, &config);
            let terminal = checkdown_terminal(child.clone(), &config).unwrap();
            if child.terminal.is_some() {
                assert_eq!(terminal.invested, child.invested);
                assert_eq!(terminal.terminal, child.terminal);
            } else {
                assert_eq!(terminal.terminal, Some(Terminal::Showdown));
                let called_amount = child.invested[0].max(child.invested[1]);
                assert_eq!(terminal.invested, [called_amount; 2]);
            }
        }
    }

    #[test]
    fn hs_dcfr_configuration_pins_a_compatible_training_horizon() {
        let mut config = tiny_config();
        config.dcfr_schedule_horizon = config.iterations;
        assert!(config.validate().is_err());

        config.dcfr_schedule = DcfrSchedule::Hs30;
        assert!(config.validate().is_ok());

        config.dcfr_schedule_horizon = config.iterations - 1;
        assert!(config.validate().is_err());
        config.dcfr_schedule_horizon = config.iterations;
        config.dcfr.strategy_exponent = 1.0;
        assert!(config.validate().is_err());
        config.dcfr = DcfrParameters::default();
        config.dcfr_schedule_horizon = MAX_HS_DCFR_HORIZON + 1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn hs_dcfr_checkpoint_resume_matches_uninterrupted_training() {
        let path = std::env::temp_dir().join(format!(
            "blueprint-hs30-checkpoint-{}-{}.json.gz",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let mut target = tiny_config();
        target.dcfr_schedule = DcfrSchedule::Hs30;
        target.dcfr_schedule_horizon = target.iterations;
        let mut partial = target.clone();
        partial.iterations = 2;
        solve_controlled(
            partial,
            RunControl {
                checkpoint_path: Some(path.to_string_lossy().into_owned()),
                checkpoint_every: 1,
                resume_path: None,
            },
        )
        .expect("write HS-DCFR partial checkpoint");
        let resumed = solve_controlled(
            target.clone(),
            RunControl {
                checkpoint_path: None,
                checkpoint_every: 0,
                resume_path: Some(path.to_string_lossy().into_owned()),
            },
        )
        .expect("resume HS-DCFR checkpoint");
        let uninterrupted = solve(target).expect("uninterrupted HS-DCFR solve");
        assert_eq!(resumed.config, uninterrupted.config);
        assert_eq!(resumed.config_hash, uninterrupted.config_hash);
        assert_eq!(
            resumed.training_config_hash,
            uninterrupted.training_config_hash
        );
        assert_eq!(resumed.strategies, uninterrupted.strategies);
        assert_eq!(resumed.metrics, uninterrupted.metrics);
        assert_eq!(resumed, uninterrupted);
        fs::remove_file(path).expect("remove HS-DCFR checkpoint");
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
    fn allocation_free_identity_fragments_match_formatted_hash_inputs() {
        for milli in 0..=100_000u64 {
            let value = milli as f64 / 1_000.0;
            let mut direct = StableHasher::new();
            direct.update_fixed_three(value);
            assert_eq!(
                direct.finish(),
                stable_hash(format!("{value:.3}").as_bytes())
            );
        }

        let history = vec!["blinds:0.500/1.000".to_owned(), "call".to_owned()];
        let mut direct = StableHasher::new();
        direct.update_joined(&history, b'/');
        assert_eq!(direct.finish(), stable_hash(history.join("/").as_bytes()));
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
    fn joint_hand_batch_checkpoint_preserves_deal_count_and_replay() {
        let path = std::env::temp_dir().join(format!(
            "blueprint-batch-checkpoint-{}-{}.json",
            std::process::id(),
            std::thread::current().name().unwrap_or("batch-test")
        ));
        let mut partial = tiny_config();
        partial.iterations = 2;
        partial.traverser_hand_batch_size = 4;
        partial.opponent_hand_batch_size = 2;
        let partial_artifact = solve_controlled(
            partial,
            RunControl {
                checkpoint_path: Some(path.to_string_lossy().into_owned()),
                checkpoint_every: 1,
                resume_path: None,
            },
        )
        .expect("write batched checkpoint");
        assert_eq!(partial_artifact.metrics.sampled_deals, 16);

        let mut target = tiny_config();
        target.traverser_hand_batch_size = 4;
        target.opponent_hand_batch_size = 2;
        let resumed = solve_controlled(
            target.clone(),
            RunControl {
                checkpoint_path: None,
                checkpoint_every: 0,
                resume_path: Some(path.to_string_lossy().into_owned()),
            },
        )
        .expect("resume batched checkpoint");
        let uninterrupted = solve(target).expect("uninterrupted batched solve");
        assert_eq!(resumed.metrics.sampled_deals, 24);
        assert_eq!(resumed, uninterrupted);
        fs::remove_file(path).expect("remove batched checkpoint");
    }

    #[test]
    fn gzip_checkpoint_resume_matches_uninterrupted_training() {
        let path = std::env::temp_dir().join(format!(
            "blueprint-checkpoint-{}-{}.json.gz",
            std::process::id(),
            std::thread::current().name().unwrap_or("gzip-test")
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
        .expect("write compressed checkpoint");
        let resumed = solve_controlled(
            tiny_config(),
            RunControl {
                checkpoint_path: None,
                checkpoint_every: 0,
                resume_path: Some(path.to_string_lossy().into_owned()),
            },
        )
        .expect("resume compressed checkpoint");
        assert_eq!(resumed, solve(tiny_config()).expect("uninterrupted solve"));
        fs::remove_file(path).expect("remove compressed checkpoint");
    }

    #[test]
    fn message_pack_checkpoint_preserves_exact_float_state_and_resume() {
        let path = std::env::temp_dir().join(format!(
            "blueprint-checkpoint-{}-{}.msgpack.gz",
            std::process::id(),
            std::thread::current().name().unwrap_or("msgpack-test")
        ));
        let mut target = tiny_config();
        target.iterations = 20;
        let mut partial = target.clone();
        partial.iterations = 10;
        solve_controlled(
            partial,
            RunControl {
                checkpoint_path: Some(path.to_string_lossy().into_owned()),
                checkpoint_every: 10,
                resume_path: None,
            },
        )
        .expect("write MessagePack checkpoint");

        let checkpoint = read_checkpoint(&path).expect("decode MessagePack checkpoint");
        assert_eq!(
            checkpoint.schema_version,
            BLUEPRINT_CHECKPOINT_SCHEMA_VERSION
        );
        let resumed = solve_controlled(
            target.clone(),
            RunControl {
                checkpoint_path: None,
                checkpoint_every: 0,
                resume_path: Some(path.to_string_lossy().into_owned()),
            },
        )
        .expect("resume MessagePack checkpoint");
        assert_eq!(resumed, solve(target).expect("uninterrupted solve"));
        fs::remove_file(path).expect("remove MessagePack checkpoint");
    }

    #[test]
    fn completed_resume_does_not_rewrite_checkpoint() {
        let stem = format!(
            "blueprint-checkpoint-{}-{}-{}-no-rewrite",
            std::process::id(),
            std::thread::current().name().unwrap_or("resume-test"),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock after epoch")
                .as_nanos()
        );
        let source = std::env::temp_dir().join(format!("{stem}.json.gz"));
        let invalid_destination = std::env::temp_dir().join(format!("{stem}-directory"));
        fs::create_dir(&invalid_destination).expect("create invalid checkpoint destination");
        let mut complete = tiny_config();
        complete.iterations = 2;
        solve_controlled(
            complete.clone(),
            RunControl {
                checkpoint_path: Some(source.to_string_lossy().into_owned()),
                checkpoint_every: 1,
                resume_path: None,
            },
        )
        .expect("write completed checkpoint");
        solve_controlled(
            complete,
            RunControl {
                checkpoint_path: Some(invalid_destination.to_string_lossy().into_owned()),
                checkpoint_every: 1,
                resume_path: Some(source.to_string_lossy().into_owned()),
            },
        )
        .expect("completed resume must not attempt a checkpoint rewrite");
        fs::remove_file(source).expect("remove checkpoint source");
        fs::remove_dir(invalid_destination).expect("remove checkpoint test directory");
    }

    #[test]
    fn checkpoint_money_is_recanonicalized_after_decimal_round_trip() {
        let mut descriptor = NodeDescriptor {
            actor: Position::BigBlind,
            street: Street::Flop,
            hand_bucket_trajectory: vec!["bucket".into()].into(),
            public_bucket_trajectory: vec!["board".into()].into(),
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
