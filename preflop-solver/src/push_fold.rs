use crate::cards::{all_combos, Combo, ComboIdentity};
use crate::evaluator::evaluate;
use crate::rng::SplitMix64;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::time::{SystemTime, UNIX_EPOCH};

const SCHEMA_VERSION: u32 = 1;
const SOLVER_VERSION: &str = env!("CARGO_PKG_VERSION");
const MODEL: &str = "heads-up-push-fold-monte-carlo-v1";
const VALIDATION_VERSION: &str = "push-fold-validation-v1";
const ADVISORY_EXPLOITABILITY_BB: f64 = 0.01;
const HIGH_PRECISION_EXPLOITABILITY_BB: f64 = 0.002;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PushFoldConfig {
    pub small_blind_bb: f64,
    pub big_blind_bb: f64,
    pub effective_stack_bb: f64,
    pub iterations: u64,
    pub equity_samples: u32,
    pub seed: u64,
}

impl Default for PushFoldConfig {
    fn default() -> Self {
        Self {
            small_blind_bb: 0.5,
            big_blind_bb: 1.0,
            effective_stack_bb: 10.0,
            iterations: 25_000_000,
            equity_samples: 1_024,
            seed: 1,
        }
    }
}

impl PushFoldConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !self.small_blind_bb.is_finite() || self.small_blind_bb <= 0.0 {
            return Err("small_blind_bb must be finite and positive".into());
        }
        if !self.big_blind_bb.is_finite() || self.big_blind_bb <= 0.0 {
            return Err("big_blind_bb must be finite and positive".into());
        }
        if self.small_blind_bb >= self.big_blind_bb {
            return Err("small_blind_bb must be less than big_blind_bb".into());
        }
        if !self.effective_stack_bb.is_finite() || self.effective_stack_bb <= self.big_blind_bb {
            return Err("effective_stack_bb must exceed big_blind_bb".into());
        }
        if self.iterations == 0 {
            return Err("iterations must be positive".into());
        }
        if self.equity_samples == 0 {
            return Err("equity_samples must be positive".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct ActionFrequency {
    pub fold: f64,
    pub shove: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct ResponseFrequency {
    pub fold: f64,
    pub call: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ExactComboStrategy {
    #[serde(flatten)]
    pub identity: ComboIdentity,
    pub small_blind: ActionFrequency,
    pub big_blind_vs_shove: ResponseFrequency,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HandClassStrategy {
    pub label: String,
    pub combo_count: usize,
    pub small_blind: ActionFrequency,
    pub big_blind_vs_shove: ResponseFrequency,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Strategies {
    pub exact_combos: Vec<ExactComboStrategy>,
    pub hand_classes: Vec<HandClassStrategy>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PushFoldMetrics {
    pub profile_small_blind_ev_bb: f64,
    pub small_blind_best_response_ev_bb: f64,
    pub small_blind_ev_vs_big_blind_best_response_bb: f64,
    pub nash_conv_bb: f64,
    pub exploitability_bb: f64,
    pub small_blind_best_response_equity_interval_bb: EstimateInterval,
    pub small_blind_ev_vs_big_blind_best_response_equity_interval_bb: EstimateInterval,
    pub nash_conv_equity_interval_bb: EstimateInterval,
    pub equity_standard_error_upper_bound: f64,
    pub called_payoff_standard_error_upper_bound_bb: f64,
    pub compatible_deals: u64,
    pub training_equity_cache_entries: usize,
    pub evaluation_equity_cache_entries: usize,
    pub evaluation_seed: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct EstimateInterval {
    pub low: f64,
    pub high: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ValidationCheck {
    pub name: String,
    pub passed: bool,
    pub value: f64,
    pub threshold: f64,
    pub comparison: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Validation {
    pub status: String,
    pub quality: String,
    pub validation_version: String,
    pub note: String,
    pub checks: Vec<ValidationCheck>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PushFoldArtifact {
    pub schema_version: u32,
    pub solver_version: String,
    pub model: String,
    pub artifact_id: String,
    pub config_hash: String,
    pub generated_at_unix_seconds: u64,
    pub payoff_convention: String,
    pub config: PushFoldConfig,
    pub metrics: PushFoldMetrics,
    pub validation: Validation,
    pub strategies: Strategies,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct SmallBlindActionValues {
    pub fold_ev_bb: f64,
    pub shove_ev_bb: f64,
    pub fold_standard_error_bb: f64,
    pub shove_standard_error_upper_bound_bb: f64,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct BigBlindActionValues {
    pub fold_ev_bb: f64,
    pub call_ev_bb: f64,
    pub fold_standard_error_bb: f64,
    pub call_standard_error_upper_bound_bb: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct HandClassActionValues {
    pub label: String,
    pub combo_count: usize,
    pub small_blind: SmallBlindActionValues,
    pub big_blind_vs_shove: BigBlindActionValues,
}

#[derive(Clone, Debug, Serialize)]
pub struct PushFoldActionValuesArtifact {
    pub schema_version: u32,
    pub model: String,
    pub source_artifact_id: String,
    pub source_config_hash: String,
    pub source_artifact_sha256: String,
    pub evaluation_seed: u64,
    pub equity_samples: u32,
    pub called_payoff_standard_error_upper_bound_bb: f64,
    pub hand_classes: Vec<HandClassActionValues>,
}

#[derive(Clone, Copy, Debug, Default)]
struct RegretNode {
    regrets: [f64; 2],
    strategy_sum: [f64; 2],
}

impl RegretNode {
    fn strategy(self) -> [f64; 2] {
        let positive = [self.regrets[0].max(0.0), self.regrets[1].max(0.0)];
        let total = positive[0] + positive[1];
        if total > 0.0 {
            [positive[0] / total, positive[1] / total]
        } else {
            [0.5, 0.5]
        }
    }

    fn average(self) -> [f64; 2] {
        let total = self.strategy_sum[0] + self.strategy_sum[1];
        if total > 0.0 {
            [
                (self.strategy_sum[0] / total).clamp(0.0, 1.0),
                (self.strategy_sum[1] / total).clamp(0.0, 1.0),
            ]
        } else {
            [0.5, 0.5]
        }
    }
}

pub fn solve(config: PushFoldConfig) -> Result<PushFoldArtifact, String> {
    config.validate()?;
    let combos = all_combos();
    let mut sb_nodes = vec![RegretNode::default(); combos.len()];
    let mut bb_nodes = vec![RegretNode::default(); combos.len()];
    let mut rng = SplitMix64::new(config.seed);
    let mut equities = EquityCache::new(config.seed, config.equity_samples);

    for iteration in 1..=config.iterations {
        let sb_index = rng.index(combos.len());
        let sb_combo = combos[sb_index];
        let mut bb_index = rng.index(combos.len());
        while sb_combo.overlaps(combos[bb_index]) {
            bb_index = rng.index(combos.len());
        }
        let bb_combo = combos[bb_index];

        let sb_strategy = sb_nodes[sb_index].strategy();
        let bb_strategy = bb_nodes[bb_index].strategy();
        let fold_utility = -config.small_blind_bb;
        let shove_win_utility = config.big_blind_bb;
        let equity = equities.equity(sb_combo, bb_combo);
        let called_utility = (2.0 * equity - 1.0) * config.effective_stack_bb;

        let sb_action_utility = [
            fold_utility,
            bb_strategy[0] * shove_win_utility + bb_strategy[1] * called_utility,
        ];
        let sb_utility =
            sb_strategy[0] * sb_action_utility[0] + sb_strategy[1] * sb_action_utility[1];

        let bb_action_utility = [-shove_win_utility, -called_utility];
        let bb_utility =
            bb_strategy[0] * bb_action_utility[0] + bb_strategy[1] * bb_action_utility[1];

        let average_weight = iteration as f64;
        for action in 0..2 {
            sb_nodes[sb_index].regrets[action] =
                (sb_nodes[sb_index].regrets[action] + sb_action_utility[action] - sb_utility)
                    .max(0.0);
            bb_nodes[bb_index].regrets[action] = (bb_nodes[bb_index].regrets[action]
                + sb_strategy[1] * (bb_action_utility[action] - bb_utility))
                .max(0.0);
            sb_nodes[sb_index].strategy_sum[action] += average_weight * sb_strategy[action];
            bb_nodes[bb_index].strategy_sum[action] += average_weight * bb_strategy[action];
        }
    }

    let sb_average = sb_nodes
        .iter()
        .map(|node| node.average())
        .collect::<Vec<_>>();
    let bb_average = bb_nodes
        .iter()
        .map(|node| node.average())
        .collect::<Vec<_>>();
    let training_equity_cache_entries = equities.values.len();
    let evaluation_seed = config.seed ^ 0xd1b54a32d192ed03;
    let mut evaluation_equities = EquityCache::new(evaluation_seed, config.equity_samples);
    let metrics = evaluate_metrics(
        &config,
        &combos,
        &sb_average,
        &bb_average,
        &mut evaluation_equities,
        training_equity_cache_entries,
        evaluation_seed,
    );
    let strategies = build_strategies(&combos, &sb_average, &bb_average);
    let validation = validate_metrics(&metrics, &strategies);
    let config_hash = stable_config_hash(&config);

    Ok(PushFoldArtifact {
        schema_version: SCHEMA_VERSION,
        solver_version: SOLVER_VERSION.to_owned(),
        model: MODEL.to_owned(),
        artifact_id: format!("hu-push-fold-{config_hash}"),
        config_hash,
        generated_at_unix_seconds: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_secs(),
        payoff_convention: "Small-blind chip EV relative to the start of the hand: fold = -small blind; uncalled shove = +big blind; called shove = (2 * equity - 1) * effective stack. Equal stacks, no ante, no rake."
            .to_owned(),
        config,
        metrics,
        validation,
        strategies,
    })
}

/// Evaluate every served push/fold action against the same 169-class policy
/// that the browser samples. BB values are conditioned on the SB having
/// shoved, so the opponent range is weighted by its class-level shove rate.
pub fn estimate_action_values(
    artifact: &PushFoldArtifact,
    source_artifact_sha256: String,
) -> Result<PushFoldActionValuesArtifact, String> {
    if artifact.schema_version != SCHEMA_VERSION || artifact.model != MODEL {
        return Err("unsupported push/fold artifact".into());
    }
    if artifact.validation.status != "approximate" {
        return Err("push/fold artifact is not accepted".into());
    }
    if source_artifact_sha256.len() != 64
        || !source_artifact_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("source artifact SHA-256 must contain 64 hexadecimal characters".into());
    }
    artifact.config.validate()?;
    let standard_error = artifact.metrics.called_payoff_standard_error_upper_bound_bb;
    if !standard_error.is_finite() || standard_error < 0.0 {
        return Err("invalid called-payoff standard-error upper bound".into());
    }

    let combos = all_combos();
    if artifact.strategies.exact_combos.len() != combos.len() {
        return Err("push/fold artifact must contain all 1,326 exact combos".into());
    }
    for (expected, stored) in combos.iter().zip(&artifact.strategies.exact_combos) {
        if stored.identity.combo_key != expected.key()
            || stored.identity.cards != expected.cards()
            || stored.identity.label != expected.label()
        {
            return Err(format!(
                "exact-combo identity mismatch at key {}",
                expected.key()
            ));
        }
    }

    if artifact.strategies.hand_classes.len() != 169 {
        return Err("push/fold artifact must contain all 169 hand classes".into());
    }
    let mut class_indexes = HashMap::with_capacity(169);
    for (index, strategy) in artifact.strategies.hand_classes.iter().enumerate() {
        let valid_pair = |first: f64, second: f64| {
            first.is_finite()
                && second.is_finite()
                && first >= 0.0
                && second >= 0.0
                && first <= 1.0
                && second <= 1.0
                && (first + second - 1.0).abs() <= 1e-6
        };
        if strategy.combo_count == 0
            || !valid_pair(strategy.small_blind.fold, strategy.small_blind.shove)
            || !valid_pair(
                strategy.big_blind_vs_shove.fold,
                strategy.big_blind_vs_shove.call,
            )
            || class_indexes
                .insert(strategy.label.clone(), index)
                .is_some()
        {
            return Err(format!("invalid hand-class policy for {}", strategy.label));
        }
    }

    let combo_class_indexes = combos
        .iter()
        .map(|combo| {
            class_indexes
                .get(&combo.label())
                .copied()
                .ok_or_else(|| format!("missing hand-class policy for {}", combo.label()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let class_count = artifact.strategies.hand_classes.len();
    let mut class_combo_counts = vec![0usize; class_count];
    let mut sb_shove_value_sums = vec![0.0f64; class_count];
    let mut sb_compatible_deals = vec![0u64; class_count];
    let mut bb_call_value_sums = vec![0.0f64; class_count];
    let mut bb_shove_reach_sums = vec![0.0f64; class_count];
    for &class_index in &combo_class_indexes {
        class_combo_counts[class_index] += 1;
    }

    let mut equities = EquityCache::new(
        artifact.metrics.evaluation_seed,
        artifact.config.equity_samples,
    );
    for (sb_index, &sb_combo) in combos.iter().enumerate() {
        let sb_class_index = combo_class_indexes[sb_index];
        let sb_strategy = artifact.strategies.hand_classes[sb_class_index].small_blind;
        for (bb_index, &bb_combo) in combos.iter().enumerate() {
            if sb_combo.overlaps(bb_combo) {
                continue;
            }
            let bb_class_index = combo_class_indexes[bb_index];
            let bb_strategy = artifact.strategies.hand_classes[bb_class_index].big_blind_vs_shove;
            let equity = equities.equity(sb_combo, bb_combo);
            let called_sb_utility = (2.0 * equity - 1.0) * artifact.config.effective_stack_bb;

            sb_shove_value_sums[sb_class_index] += bb_strategy.fold * artifact.config.big_blind_bb
                + bb_strategy.call * called_sb_utility;
            sb_compatible_deals[sb_class_index] += 1;

            bb_call_value_sums[bb_class_index] += sb_strategy.shove * -called_sb_utility;
            bb_shove_reach_sums[bb_class_index] += sb_strategy.shove;
        }
    }

    let hand_classes = artifact
        .strategies
        .hand_classes
        .iter()
        .enumerate()
        .map(|(index, strategy)| {
            if class_combo_counts[index] != strategy.combo_count {
                return Err(format!(
                    "hand-class combo count mismatch for {}",
                    strategy.label
                ));
            }
            if sb_compatible_deals[index] == 0 || bb_shove_reach_sums[index] <= 0.0 {
                return Err(format!(
                    "unreachable hand-class value for {}",
                    strategy.label
                ));
            }
            let shove_ev_bb = sb_shove_value_sums[index] / sb_compatible_deals[index] as f64;
            let call_ev_bb = bb_call_value_sums[index] / bb_shove_reach_sums[index];
            if !shove_ev_bb.is_finite() || !call_ev_bb.is_finite() {
                return Err(format!(
                    "non-finite hand-class value for {}",
                    strategy.label
                ));
            }
            Ok(HandClassActionValues {
                label: strategy.label.clone(),
                combo_count: strategy.combo_count,
                small_blind: SmallBlindActionValues {
                    fold_ev_bb: -artifact.config.small_blind_bb,
                    shove_ev_bb,
                    fold_standard_error_bb: 0.0,
                    shove_standard_error_upper_bound_bb: standard_error,
                },
                big_blind_vs_shove: BigBlindActionValues {
                    fold_ev_bb: -artifact.config.big_blind_bb,
                    call_ev_bb,
                    fold_standard_error_bb: 0.0,
                    call_standard_error_upper_bound_bb: standard_error,
                },
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

    Ok(PushFoldActionValuesArtifact {
        schema_version: 1,
        model: "heads-up-push-fold-action-values-v1".to_owned(),
        source_artifact_id: artifact.artifact_id.clone(),
        source_config_hash: artifact.config_hash.clone(),
        source_artifact_sha256: source_artifact_sha256.to_ascii_lowercase(),
        evaluation_seed: artifact.metrics.evaluation_seed,
        equity_samples: artifact.config.equity_samples,
        called_payoff_standard_error_upper_bound_bb: standard_error,
        hand_classes,
    })
}

fn validate_metrics(metrics: &PushFoldMetrics, strategies: &Strategies) -> Validation {
    let finite = metrics.profile_small_blind_ev_bb.is_finite()
        && metrics.small_blind_best_response_ev_bb.is_finite()
        && metrics
            .small_blind_ev_vs_big_blind_best_response_bb
            .is_finite()
        && metrics.nash_conv_bb.is_finite();
    let ordering_gap = metrics.small_blind_best_response_ev_bb
        - metrics.small_blind_ev_vs_big_blind_best_response_bb;
    let probability_error = strategies
        .exact_combos
        .iter()
        .map(|strategy| {
            (strategy.small_blind.fold + strategy.small_blind.shove - 1.0)
                .abs()
                .max(
                    (strategy.big_blind_vs_shove.fold + strategy.big_blind_vs_shove.call - 1.0)
                        .abs(),
                )
        })
        .fold(0.0, f64::max);
    let aces = strategies
        .hand_classes
        .iter()
        .find(|strategy| strategy.label == "AA")
        .expect("all hand classes are exported");
    let aces_floor = aces.small_blind.shove.min(aces.big_blind_vs_shove.call);
    let checks = vec![
        ValidationCheck {
            name: "finite_metrics".to_owned(),
            passed: finite,
            value: if finite { 1.0 } else { 0.0 },
            threshold: 1.0,
            comparison: "==".to_owned(),
        },
        ValidationCheck {
            name: "best_response_ordering".to_owned(),
            passed: ordering_gap >= -1e-9,
            value: ordering_gap,
            threshold: 0.0,
            comparison: ">=".to_owned(),
        },
        ValidationCheck {
            name: "strategy_probability_sums".to_owned(),
            passed: probability_error <= 1e-9,
            value: probability_error,
            threshold: 1e-9,
            comparison: "<=".to_owned(),
        },
        ValidationCheck {
            name: "aces_shove_and_call_sanity".to_owned(),
            passed: aces_floor >= 0.99,
            value: aces_floor,
            threshold: 0.99,
            comparison: ">=".to_owned(),
        },
        ValidationCheck {
            name: "exploitability_advisory".to_owned(),
            passed: metrics.exploitability_bb <= ADVISORY_EXPLOITABILITY_BB,
            value: metrics.exploitability_bb,
            threshold: ADVISORY_EXPLOITABILITY_BB,
            comparison: "<=".to_owned(),
        },
        ValidationCheck {
            name: "exploitability_high_precision".to_owned(),
            passed: metrics.exploitability_bb <= HIGH_PRECISION_EXPLOITABILITY_BB,
            value: metrics.exploitability_bb,
            threshold: HIGH_PRECISION_EXPLOITABILITY_BB,
            comparison: "<=".to_owned(),
        },
    ];
    let required_pass = checks
        .iter()
        .filter(|check| check.name != "exploitability_high_precision")
        .all(|check| check.passed);
    let high_precision = metrics.exploitability_bb <= HIGH_PRECISION_EXPLOITABILITY_BB;
    let status = if required_pass {
        "approximate"
    } else {
        "rejected"
    };
    Validation {
        status: status.to_owned(),
        quality: if high_precision {
            "high-precision"
        } else if required_pass {
            "advisory"
        } else {
            "insufficient"
        }
        .to_owned(),
        validation_version: VALIDATION_VERSION.to_owned(),
        note: "Status remains approximate because showdown equities are Monte Carlo estimates. NashConv is an exact best-response certificate only for the independently sampled evaluation payoff matrix; equity intervals are separate, non-simultaneous sampling advisories."
            .to_owned(),
        checks,
    }
}

fn evaluate_metrics(
    config: &PushFoldConfig,
    combos: &[Combo],
    sb_strategy: &[[f64; 2]],
    bb_strategy: &[[f64; 2]],
    equities: &mut EquityCache,
    training_equity_cache_entries: usize,
    evaluation_seed: u64,
) -> PushFoldMetrics {
    let fold_utility = -config.small_blind_bb;
    let shove_win_utility = config.big_blind_bb;
    let mut profile_total = 0.0;
    let mut sb_best_total = 0.0;
    let mut bb_fold_totals = vec![0.0; combos.len()];
    let mut bb_call_totals = vec![0.0; combos.len()];
    let mut bb_fold_branch_totals = vec![0.0; combos.len()];
    let mut compatible_deals = 0u64;

    for (sb_index, &sb_combo) in combos.iter().enumerate() {
        let mut profile_for_sb = 0.0;
        let mut shove_for_sb = 0.0;
        let mut opponents = 0u64;
        for (bb_index, &bb_combo) in combos.iter().enumerate() {
            if sb_combo.overlaps(bb_combo) {
                continue;
            }
            compatible_deals += 1;
            opponents += 1;
            let equity = equities.equity(sb_combo, bb_combo);
            let called_utility = (2.0 * equity - 1.0) * config.effective_stack_bb;
            let shove_utility = bb_strategy[bb_index][0] * shove_win_utility
                + bb_strategy[bb_index][1] * called_utility;
            profile_for_sb +=
                sb_strategy[sb_index][0] * fold_utility + sb_strategy[sb_index][1] * shove_utility;
            shove_for_sb += shove_utility;

            bb_fold_branch_totals[bb_index] += sb_strategy[sb_index][0] * fold_utility;
            bb_fold_totals[bb_index] += sb_strategy[sb_index][1] * shove_win_utility;
            bb_call_totals[bb_index] += sb_strategy[sb_index][1] * called_utility;
        }
        profile_total += profile_for_sb / opponents as f64;
        sb_best_total += fold_utility.max(shove_for_sb / opponents as f64);
    }

    let profile_ev = profile_total / combos.len() as f64;
    let sb_best_response = sb_best_total / combos.len() as f64;
    let opponents_per_combo = 1225.0;
    let sb_ev_vs_bb_best_response = bb_fold_branch_totals
        .iter()
        .zip(bb_fold_totals.iter())
        .zip(bb_call_totals.iter())
        .map(|((fold_branch, fold_response), call_response)| {
            (fold_branch + fold_response.min(*call_response)) / opponents_per_combo
        })
        .sum::<f64>()
        / combos.len() as f64;
    let nash_conv = (sb_best_response - sb_ev_vs_bb_best_response).max(0.0);
    let equity_standard_error_upper_bound = 0.5 / (config.equity_samples as f64).sqrt();
    let payoff_standard_error_upper_bound =
        2.0 * config.effective_stack_bb * equity_standard_error_upper_bound;

    PushFoldMetrics {
        profile_small_blind_ev_bb: profile_ev,
        small_blind_best_response_ev_bb: sb_best_response,
        small_blind_ev_vs_big_blind_best_response_bb: sb_ev_vs_bb_best_response,
        nash_conv_bb: nash_conv,
        exploitability_bb: nash_conv / 2.0,
        small_blind_best_response_equity_interval_bb: EstimateInterval {
            low: sb_best_response - payoff_standard_error_upper_bound,
            high: sb_best_response + payoff_standard_error_upper_bound,
        },
        small_blind_ev_vs_big_blind_best_response_equity_interval_bb: EstimateInterval {
            low: sb_ev_vs_bb_best_response - payoff_standard_error_upper_bound,
            high: sb_ev_vs_bb_best_response + payoff_standard_error_upper_bound,
        },
        nash_conv_equity_interval_bb: EstimateInterval {
            low: (nash_conv - 2.0 * payoff_standard_error_upper_bound).max(0.0),
            high: nash_conv + 2.0 * payoff_standard_error_upper_bound,
        },
        equity_standard_error_upper_bound,
        called_payoff_standard_error_upper_bound_bb: payoff_standard_error_upper_bound,
        compatible_deals,
        training_equity_cache_entries,
        evaluation_equity_cache_entries: equities.values.len(),
        evaluation_seed,
    }
}

fn stable_config_hash(config: &PushFoldConfig) -> String {
    let input = format!(
        "{MODEL}|{SOLVER_VERSION}|{:.12}|{:.12}|{:.12}|{}|{}|{}",
        config.small_blind_bb,
        config.big_blind_bb,
        config.effective_stack_bb,
        config.iterations,
        config.equity_samples,
        config.seed
    );
    let mut hash = 0xcbf29ce484222325u64;
    for byte in input.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn build_strategies(
    combos: &[Combo],
    sb_strategy: &[[f64; 2]],
    bb_strategy: &[[f64; 2]],
) -> Strategies {
    let exact_combos = combos
        .iter()
        .enumerate()
        .map(|(index, &combo)| ExactComboStrategy {
            identity: combo.into(),
            small_blind: ActionFrequency {
                fold: sb_strategy[index][0],
                shove: sb_strategy[index][1],
            },
            big_blind_vs_shove: ResponseFrequency {
                fold: bb_strategy[index][0],
                call: bb_strategy[index][1],
            },
        })
        .collect();

    let mut classes: BTreeMap<String, (usize, f64, f64)> = BTreeMap::new();
    for (index, combo) in combos.iter().enumerate() {
        let entry = classes.entry(combo.label()).or_insert((0, 0.0, 0.0));
        entry.0 += 1;
        entry.1 += sb_strategy[index][1];
        entry.2 += bb_strategy[index][1];
    }
    let hand_classes = classes
        .into_iter()
        .map(
            |(label, (combo_count, shove_sum, call_sum))| HandClassStrategy {
                label,
                combo_count,
                small_blind: ActionFrequency {
                    fold: 1.0 - shove_sum / combo_count as f64,
                    shove: shove_sum / combo_count as f64,
                },
                big_blind_vs_shove: ResponseFrequency {
                    fold: 1.0 - call_sum / combo_count as f64,
                    call: call_sum / combo_count as f64,
                },
            },
        )
        .collect();

    Strategies {
        exact_combos,
        hand_classes,
    }
}

struct EquityCache {
    seed: u64,
    samples: u32,
    values: HashMap<u64, f64>,
}

impl EquityCache {
    fn new(seed: u64, samples: u32) -> Self {
        Self {
            seed,
            samples,
            values: HashMap::new(),
        }
    }

    fn equity(&mut self, hero: Combo, villain: Combo) -> f64 {
        let forward_key = canonical_matchup_key(hero, villain);
        let reverse_key = canonical_matchup_key(villain, hero);
        if forward_key == reverse_key {
            return 0.5;
        }
        let (key, complement) = if forward_key < reverse_key {
            (forward_key, false)
        } else {
            (reverse_key, true)
        };
        if let Some(&value) = self.values.get(&key) {
            return if complement { 1.0 - value } else { value };
        }
        let canonical = unpack_matchup(key);
        let value = estimate_equity(
            Combo::new(canonical[0], canonical[1]),
            Combo::new(canonical[2], canonical[3]),
            self.samples,
            self.seed ^ key.wrapping_mul(0x9e3779b97f4a7c15),
        );
        self.values.insert(key, value);
        if complement {
            1.0 - value
        } else {
            value
        }
    }
}

fn estimate_equity(hero: Combo, villain: Combo, samples: u32, seed: u64) -> f64 {
    debug_assert!(!hero.overlaps(villain));
    let blocked = [hero.high, hero.low, villain.high, villain.low];
    let mut deck = Vec::with_capacity(48);
    for card in 0..52u8 {
        if !blocked.contains(&card) {
            deck.push(card);
        }
    }
    let mut rng = SplitMix64::new(seed);
    let mut score = 0.0;
    for _ in 0..samples {
        for index in 0..5 {
            let selected = index + rng.index(deck.len() - index);
            deck.swap(index, selected);
        }
        let board = &deck[..5];
        let hero_score = evaluate(&[
            hero.high, hero.low, board[0], board[1], board[2], board[3], board[4],
        ]);
        let villain_score = evaluate(&[
            villain.high,
            villain.low,
            board[0],
            board[1],
            board[2],
            board[3],
            board[4],
        ]);
        score += match hero_score.cmp(&villain_score) {
            std::cmp::Ordering::Greater => 1.0,
            std::cmp::Ordering::Equal => 0.5,
            std::cmp::Ordering::Less => 0.0,
        };
    }
    score / samples as f64
}

fn canonical_matchup_key(hero: Combo, villain: Combo) -> u64 {
    debug_assert!(!hero.overlaps(villain));
    let mut best = u64::MAX;
    for a in 0..4u8 {
        for b in 0..4u8 {
            if b == a {
                continue;
            }
            for c in 0..4u8 {
                if c == a || c == b {
                    continue;
                }
                let d = 6 - a - b - c;
                let permutation = [a, b, c, d];
                let map_card = |card: u8| (card & !3) | permutation[(card & 3) as usize];
                let mapped_hero = Combo::new(map_card(hero.high), map_card(hero.low));
                let mapped_villain = Combo::new(map_card(villain.high), map_card(villain.low));
                let key = pack_matchup([
                    mapped_hero.high,
                    mapped_hero.low,
                    mapped_villain.high,
                    mapped_villain.low,
                ]);
                best = best.min(key);
            }
        }
    }
    best
}

fn pack_matchup(cards: [u8; 4]) -> u64 {
    (cards[0] as u64)
        | ((cards[1] as u64) << 6)
        | ((cards[2] as u64) << 12)
        | ((cards[3] as u64) << 18)
}

fn unpack_matchup(key: u64) -> [u8; 4] {
    [
        (key & 63) as u8,
        ((key >> 6) & 63) as u8,
        ((key >> 12) & 63) as u8,
        ((key >> 18) & 63) as u8,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_game_configuration() {
        let config = PushFoldConfig {
            small_blind_bb: 1.0,
            big_blind_bb: 1.0,
            ..PushFoldConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn canonical_matchup_is_invariant_to_global_suit_permutation() {
        let first = canonical_matchup_key(Combo::new(51, 46), Combo::new(43, 36));
        let second = canonical_matchup_key(Combo::new(48, 45), Combo::new(40, 39));
        assert_eq!(first, second);
    }

    #[test]
    fn deterministic_equity_estimate_respects_clear_favorite() {
        let aces = Combo::new(51, 50);
        let deuces = Combo::new(3, 2);
        let first = estimate_equity(aces, deuces, 10_000, 17);
        let second = estimate_equity(aces, deuces, 10_000, 17);
        assert_eq!(first, second);
        assert!(first > 0.75, "equity={first}");
    }

    #[test]
    fn cached_equity_is_exactly_antisymmetric() {
        let hero = Combo::new(51, 46);
        let villain = Combo::new(43, 36);
        let mut cache = EquityCache::new(12, 1_000);
        let forward = cache.equity(hero, villain);
        let reverse = cache.equity(villain, hero);
        assert_eq!(forward + reverse, 1.0);
    }

    #[test]
    fn short_solver_exports_complete_finite_strategy() {
        let mut artifact = solve(PushFoldConfig {
            effective_stack_bb: 5.0,
            iterations: 20_000,
            equity_samples: 32,
            seed: 9,
            ..PushFoldConfig::default()
        })
        .expect("valid solve");
        assert_eq!(artifact.model, MODEL);
        assert_eq!(artifact.strategies.exact_combos.len(), 1326);
        assert_eq!(artifact.strategies.hand_classes.len(), 169);
        assert_eq!(artifact.metrics.compatible_deals, 1326 * 1225);
        assert!(artifact.metrics.nash_conv_bb.is_finite());
        let aces = artifact
            .strategies
            .hand_classes
            .iter()
            .find(|strategy| strategy.label == "AA")
            .expect("aces strategy");
        assert!(aces.small_blind.shove > 0.99);
        assert!(aces.big_blind_vs_shove.call > 0.99);
        for strategy in &artifact.strategies.exact_combos {
            assert!((strategy.small_blind.fold + strategy.small_blind.shove - 1.0).abs() < 1e-9);
            assert!(
                (strategy.big_blind_vs_shove.fold + strategy.big_blind_vs_shove.call - 1.0).abs()
                    < 1e-9
            );
        }
        artifact.validation.status = "approximate".to_owned();
        let values = estimate_action_values(&artifact, "a".repeat(64))
            .expect("valid serving-policy action values");
        assert_eq!(values.hand_classes.len(), 169);
        assert_eq!(values.source_artifact_sha256, "a".repeat(64));
        for hand in values.hand_classes {
            assert_eq!(hand.small_blind.fold_ev_bb, -0.5);
            assert_eq!(hand.big_blind_vs_shove.fold_ev_bb, -1.0);
            assert!(hand.small_blind.shove_ev_bb.is_finite());
            assert!(hand.big_blind_vs_shove.call_ev_bb.is_finite());
        }
    }

    #[test]
    fn scaling_all_chip_values_preserves_strategy_and_scales_ev() {
        let base = solve(PushFoldConfig {
            effective_stack_bb: 5.0,
            iterations: 30_000,
            equity_samples: 16,
            seed: 22,
            ..PushFoldConfig::default()
        })
        .expect("base solve");
        let scaled = solve(PushFoldConfig {
            small_blind_bb: 1.0,
            big_blind_bb: 2.0,
            effective_stack_bb: 10.0,
            iterations: 30_000,
            equity_samples: 16,
            seed: 22,
        })
        .expect("scaled solve");
        for (left, right) in base
            .strategies
            .exact_combos
            .iter()
            .zip(&scaled.strategies.exact_combos)
        {
            assert!((left.small_blind.shove - right.small_blind.shove).abs() < 1e-10);
            assert!((left.big_blind_vs_shove.call - right.big_blind_vs_shove.call).abs() < 1e-10);
        }
        assert!(
            (scaled.metrics.profile_small_blind_ev_bb
                - 2.0 * base.metrics.profile_small_blind_ev_bb)
                .abs()
                < 1e-10
        );
        assert!((scaled.metrics.nash_conv_bb - 2.0 * base.metrics.nash_conv_bb).abs() < 1e-10);
    }
}
