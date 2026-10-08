//! Immutable cash rules and exact terminal accounting shared with TypeScript.
//!
//! This module does not enable raked training or serving. Callers must first
//! remove their zero-sum assumptions and pin compatible artifacts.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const MAX_SAFE_UNITS: u64 = 9_007_199_254_740_991;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub enum Currency {
    #[serde(rename = "BB")]
    BigBlinds,
    #[serde(rename = "USD")]
    Usd,
}

impl Currency {
    fn as_str(self) -> &'static str {
        match self {
            Self::BigBlinds => "BB",
            Self::Usd => "USD",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MoneyRounding {
    HalfUp,
    HalfToEven,
}

impl MoneyRounding {
    fn as_str(self) -> &'static str {
        match self {
            Self::HalfUp => "half-up",
            Self::HalfToEven => "half-to-even",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum SplitPotRule {
    FirstLeftOfButton,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RakeRules {
    pub rate_basis_points: u64,
    pub cap_units: u64,
    pub no_flop_no_drop: bool,
    pub rounding: MoneyRounding,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CashGameRules {
    pub schema: String,
    pub id: String,
    pub currency: Currency,
    pub units_per_bb: u64,
    pub blinds_units: [u64; 2],
    pub players_dealt: u8,
    pub ante_units: u64,
    pub rake: RakeRules,
    pub split_pot_rule: SplitPotRule,
    pub bet_rounding: MoneyRounding,
}

fn money_units(value: u64) -> Result<(), String> {
    if value > MAX_SAFE_UNITS {
        return Err("money exceeds safe integer units".into());
    }
    Ok(())
}

impl CashGameRules {
    pub fn validate(&self) -> Result<(), String> {
        let valid_id = !self.id.is_empty()
            && self.id.len() <= 96
            && self.id.as_bytes()[0].is_ascii_lowercase()
            && self
                .id
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-');
        if self.schema != "hu-cash-rules-v1"
            || !valid_id
            || self.players_dealt != 2
            || self.ante_units != 0
        {
            return Err("unsupported heads-up cash rules".into());
        }
        for amount in [
            self.units_per_bb,
            self.blinds_units[0],
            self.blinds_units[1],
            self.rake.rate_basis_points,
            self.rake.cap_units,
        ] {
            money_units(amount)?;
        }
        if self.units_per_bb == 0
            || self.blinds_units[0] == 0
            || self.blinds_units[0] >= self.blinds_units[1]
            || self.blinds_units[1] != self.units_per_bb
            || self.rake.rate_basis_points > 10_000
            || (self.rake.rate_basis_points == 0) != (self.rake.cap_units == 0)
            || self.rake.rounding != MoneyRounding::HalfToEven
            || (self.currency == Currency::Usd
                && (self.id == "home-game-v1" || self.bet_rounding != MoneyRounding::HalfToEven))
            || (self.currency == Currency::BigBlinds
                && (self.id != "home-game-v1"
                    || self.units_per_bb != 1000
                    || self.blinds_units[0] != 500
                    || self.rake.rate_basis_points != 0
                    || self.rake.no_flop_no_drop
                    || self.bet_rounding != MoneyRounding::HalfUp))
        {
            return Err("inconsistent heads-up cash rules".into());
        }
        Ok(())
    }

    pub fn canonical_string(&self) -> Result<String, String> {
        self.validate()?;
        Ok(format!(
            "{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|first-left-of-button|{}",
            self.schema,
            self.id,
            self.currency.as_str(),
            self.units_per_bb,
            self.blinds_units[0],
            self.blinds_units[1],
            self.players_dealt,
            self.ante_units,
            self.rake.rate_basis_points,
            self.rake.cap_units,
            u8::from(self.rake.no_flop_no_drop),
            self.rake.rounding.as_str(),
            self.bet_rounding.as_str()
        ))
    }

    pub fn sha256(&self) -> Result<String, String> {
        let digest = Sha256::digest(self.canonical_string()?.as_bytes());
        Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
    }

    pub fn rake_units(&self, gross_pot_units: u64, flop_dealt: bool) -> Result<u64, String> {
        self.validate()?;
        money_units(gross_pot_units)?;
        if self.rake.no_flop_no_drop && !flop_dealt {
            return Ok(0);
        }
        let rounded = round_ratio(
            u128::from(gross_pot_units) * u128::from(self.rake.rate_basis_points),
            10_000,
            MoneyRounding::HalfToEven,
        );
        Ok(rounded.min(u128::from(self.rake.cap_units)) as u64)
    }
}

/// Frozen CLI profiles. The rake-off control retains NL25 blinds and cents.
pub fn study_rules(name: &str) -> Result<CashGameRules, String> {
    let registry: serde_json::Value =
        serde_json::from_str(include_str!("../../data/practice/cash-game-rules.json"))
            .map_err(|error| error.to_string())?;
    let key = match name {
        "home" => "home",
        "nl25" | "nl25-rake-off-control" => "nl25",
        _ => return Err("unknown cash study profile".into()),
    };
    let mut rules: CashGameRules =
        serde_json::from_value(registry[key].clone()).map_err(|error| error.to_string())?;
    if name == "nl25-rake-off-control" {
        rules.id = "nl25-rake-off-control-v1".into();
        rules.rake.rate_basis_points = 0;
        rules.rake.cap_units = 0;
    }
    rules.validate()?;
    Ok(rules)
}

fn round_ratio(numerator: u128, denominator: u128, rounding: MoneyRounding) -> u128 {
    let whole = numerator / denominator;
    let twice_remainder = 2 * (numerator % denominator);
    whole
        + u128::from(
            twice_remainder > denominator
                || (twice_remainder == denominator
                    && (rounding == MoneyRounding::HalfUp || whole % 2 == 1)),
        )
}

pub fn round_money_ratio(
    numerator: u64,
    denominator: u64,
    rounding: MoneyRounding,
) -> Result<u64, String> {
    money_units(numerator)?;
    money_units(denominator)?;
    if denominator == 0 {
        return Err("invalid money rounding ratio".into());
    }
    Ok(round_ratio(u128::from(numerator), u128::from(denominator), rounding) as u64)
}

/// Bounds are supplied by the betting engine; this does not generate a legal tree.
pub fn quantize_raise_amounts(
    rules: &CashGameRules,
    ratios: &[[u64; 2]],
    minimum_to_units: u64,
    maximum_to_units: u64,
) -> Result<Vec<u64>, String> {
    rules.validate()?;
    money_units(minimum_to_units)?;
    money_units(maximum_to_units)?;
    if minimum_to_units > maximum_to_units {
        return Err("invalid raise bounds".into());
    }
    let mut amounts = ratios
        .iter()
        .map(|ratio| {
            round_money_ratio(ratio[0], ratio[1], rules.bet_rounding)
                .map(|amount| amount.clamp(minimum_to_units, maximum_to_units))
        })
        .collect::<Result<Vec<_>, _>>()?;
    amounts.sort_unstable();
    amounts.dedup();
    Ok(amounts)
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Outcome {
    PlayerZero,
    PlayerOne,
    Split,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TerminalReason {
    Fold,
    Showdown,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CashTerminal {
    pub committed_units: [u64; 2],
    pub button: u8,
    pub reason: TerminalReason,
    pub outcome: Outcome,
    pub board_cards_dealt: u8,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CashSettlement {
    pub uncalled_refund_units: [u64; 2],
    pub gross_pot_units: u64,
    pub rake_units: u64,
    pub net_pot_units: u64,
    pub awards_units: [u64; 2],
    pub net_payoff_units: [i64; 2],
}

/// Settlement is only for completed hands, never partial practice review stops.
pub fn settle_cash_hand(
    rules: &CashGameRules,
    terminal: &CashTerminal,
) -> Result<CashSettlement, String> {
    rules.validate()?;
    if terminal.button > 1
        || ![0, 3, 4, 5].contains(&terminal.board_cards_dealt)
        || (terminal.reason == TerminalReason::Showdown && terminal.board_cards_dealt != 5)
        || (terminal.reason == TerminalReason::Fold && terminal.outcome == Outcome::Split)
    {
        return Err("settlement requires a completed heads-up hand".into());
    }
    for amount in terminal.committed_units {
        money_units(amount)?;
    }
    let total = terminal.committed_units[0]
        .checked_add(terminal.committed_units[1])
        .ok_or("committed money overflow")?;
    money_units(total)?;
    let winner = usize::from(terminal.outcome == Outcome::PlayerOne);
    if terminal.reason == TerminalReason::Fold
        && terminal.committed_units[winner] < terminal.committed_units[1 - winner]
    {
        return Err("fold winner cannot have the unmatched losing wager".into());
    }
    let matched = terminal.committed_units[0].min(terminal.committed_units[1]);
    let uncalled_refund_units = terminal.committed_units.map(|amount| amount - matched);
    let gross_pot_units = 2 * matched;
    let rake_units = rules.rake_units(gross_pot_units, terminal.board_cards_dealt >= 3)?;
    let net_pot_units = gross_pot_units - rake_units;
    let mut awards_units = [0, 0];
    if terminal.outcome == Outcome::Split {
        awards_units = [net_pot_units / 2; 2];
        awards_units[1 - usize::from(terminal.button)] += net_pot_units % 2;
    } else {
        awards_units[winner] = net_pot_units;
    }
    let net_payoff_units = std::array::from_fn(|player| {
        awards_units[player] as i64 + uncalled_refund_units[player] as i64
            - terminal.committed_units[player] as i64
    });
    Ok(CashSettlement {
        uncalled_refund_units,
        gross_pot_units,
        rake_units,
        net_pot_units,
        awards_units,
        net_payoff_units,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn rules(name: &str) -> CashGameRules {
        let data: Value =
            serde_json::from_str(include_str!("../../data/practice/cash-game-rules.json")).unwrap();
        serde_json::from_value(data[name].clone()).unwrap()
    }

    fn fixtures() -> Value {
        serde_json::from_str(include_str!(
            "../../data/practice/cash-settlement-fixtures.json"
        ))
        .unwrap()
    }

    #[test]
    fn shares_frozen_rules_hashes_with_typescript() {
        for (name, identity) in fixtures()["identities"].as_object().unwrap() {
            let profile = rules(name);
            assert_eq!(
                profile.canonical_string().unwrap(),
                identity["canonical"].as_str().unwrap()
            );
            assert_eq!(
                profile.sha256().unwrap(),
                identity["sha256"].as_str().unwrap()
            );
        }
        let mut changed = rules("nl25");
        changed.rake.rate_basis_points = 500;
        assert_ne!(changed.sha256().unwrap(), rules("nl25").sha256().unwrap());
    }

    #[test]
    fn shared_money_rounding_and_betting_grid_fixtures() {
        for fixture in fixtures()["rounding"].as_array().unwrap() {
            let rounding: MoneyRounding = serde_json::from_value(fixture["mode"].clone()).unwrap();
            assert_eq!(
                round_money_ratio(
                    fixture["numerator"].as_u64().unwrap(),
                    fixture["denominator"].as_u64().unwrap(),
                    rounding
                )
                .unwrap(),
                fixture["expected"].as_u64().unwrap()
            );
        }
        for fixture in fixtures()["sizing"].as_array().unwrap() {
            let ratios: Vec<[u64; 2]> = serde_json::from_value(fixture["ratios"].clone()).unwrap();
            let actual = quantize_raise_amounts(
                &rules(fixture["rules"].as_str().unwrap()),
                &ratios,
                fixture["minimum"].as_u64().unwrap(),
                fixture["maximum"].as_u64().unwrap(),
            )
            .unwrap();
            assert_eq!(serde_json::to_value(actual).unwrap(), fixture["expected"]);
        }
        assert!(round_money_ratio(1, 0, MoneyRounding::HalfToEven).is_err());
        assert!(round_money_ratio(MAX_SAFE_UNITS + 1, 2, MoneyRounding::HalfToEven).is_err());
        assert!(quantize_raise_amounts(&rules("nl25"), &[[10, 1]], 100, 50).is_err());
        assert_eq!(rules("nl25").rake_units(MAX_SAFE_UNITS, true).unwrap(), 50);
    }

    #[test]
    fn shares_all_terminal_accounting_fixtures_with_typescript() {
        for fixture in fixtures()["settlements"].as_array().unwrap() {
            let terminal: CashTerminal =
                serde_json::from_value(fixture["terminal"].clone()).unwrap();
            let actual =
                settle_cash_hand(&rules(fixture["rules"].as_str().unwrap()), &terminal).unwrap();
            assert_eq!(
                serde_json::to_value(&actual).unwrap(),
                fixture["expected"],
                "{}",
                fixture["name"]
            );
            assert_eq!(
                actual.net_payoff_units.iter().sum::<i64>(),
                -(actual.rake_units as i64)
            );
            assert_eq!(
                actual.awards_units.iter().sum::<u64>() + actual.rake_units,
                actual.gross_pot_units
            );
            assert_eq!(
                actual.gross_pot_units + actual.uncalled_refund_units.iter().sum::<u64>(),
                terminal.committed_units.iter().sum::<u64>()
            );
        }
    }

    #[test]
    fn validates_rules_and_rejects_unknown_fields() {
        let mut invalid = rules("nl25");
        invalid.id = "home-game-v1".into();
        assert!(invalid.validate().is_err());
        invalid = rules("nl25");
        invalid.rake.rate_basis_points = 10_001;
        assert!(invalid.validate().is_err());
        invalid = rules("nl25");
        invalid.blinds_units = [10, 20];
        assert!(invalid.validate().is_err());
        invalid = rules("nl25");
        invalid.rake.rounding = MoneyRounding::HalfUp;
        assert!(invalid.validate().is_err());
        invalid = rules("home");
        invalid.rake = rules("nl25").rake;
        assert!(invalid.validate().is_err());
        let mut extended = serde_json::to_value(rules("nl25")).unwrap();
        extended["unexpected"] = json!(true);
        assert!(serde_json::from_value::<CashGameRules>(extended).is_err());
    }

    #[test]
    fn rejects_partial_reviews_unfinished_runouts_and_invalid_money() {
        let base = json!({"committedUnits": [125, 125], "button": 0, "reason": "showdown",
            "outcome": "player-zero", "boardCardsDealt": 5});
        for (field, value) in [
            ("unexpected", json!(true)),
            ("reason", json!("preflop-complete")),
            ("reason", json!("review-complete")),
            ("boardCardsDealt", json!(0)),
            ("boardCardsDealt", json!(3)),
            ("boardCardsDealt", json!(2)),
            ("button", json!(2)),
            ("committedUnits", json!([-1, 125])),
            ("committedUnits", json!([1.5, 125])),
            ("committedUnits", json!([125])),
            ("committedUnits", json!([MAX_SAFE_UNITS, MAX_SAFE_UNITS])),
        ] {
            let mut input = base.clone();
            input[field] = value;
            let failed = serde_json::from_value::<CashTerminal>(input)
                .and_then(|terminal| {
                    settle_cash_hand(&rules("nl25"), &terminal).map_err(serde::de::Error::custom)
                })
                .is_err();
            assert!(failed, "invalid {field}");
        }
        let mut folded: CashTerminal = serde_json::from_value(base).unwrap();
        folded.reason = TerminalReason::Fold;
        folded.outcome = Outcome::Split;
        assert!(settle_cash_hand(&rules("nl25"), &folded).is_err());
        folded.outcome = Outcome::PlayerZero;
        folded.committed_units = [50, 125];
        assert!(settle_cash_hand(&rules("nl25"), &folded).is_err());
    }

    #[test]
    fn conserves_chips_for_many_amounts_and_both_buttons() {
        for profile in [rules("home"), rules("nl25")] {
            for committed in (1..=1200).step_by(7) {
                for button in 0..=1 {
                    for outcome in [Outcome::PlayerZero, Outcome::PlayerOne, Outcome::Split] {
                        let result = settle_cash_hand(
                            &profile,
                            &CashTerminal {
                                committed_units: [committed; 2],
                                button,
                                reason: TerminalReason::Showdown,
                                outcome,
                                board_cards_dealt: 5,
                            },
                        )
                        .unwrap();
                        assert_eq!(
                            result.net_payoff_units.iter().sum::<i64>(),
                            -(result.rake_units as i64)
                        );
                        assert!(result.rake_units <= profile.rake.cap_units);
                        assert_eq!(
                            result.awards_units.iter().sum::<u64>() + result.rake_units,
                            2 * committed
                        );
                    }
                }
            }
        }
    }
}
