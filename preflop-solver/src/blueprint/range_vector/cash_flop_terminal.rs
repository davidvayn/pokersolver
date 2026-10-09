//! Exact own-payoff flop runouts; scalar equity cannot encode odd-cent ties.
use super::*;

/// An immutable rules/board-local showdown kernel. Rank each completed board
/// once, then reuse the blocker-aware win/tie/loss prefix kernels as ranges and
/// commitments change. No private hands, rounded beliefs or policy fallback.
pub struct ExactCashFlopTerminal {
    flop: [u8; 3],
    rules: CashGameRules,
    runouts: Vec<([u8; 2], PublicTerminalCache)>,
}

impl ExactCashFlopTerminal {
    pub fn new(flop: [u8; 3], rules: CashGameRules) -> Result<Self, String> {
        rules.validate()?;
        if flop.iter().any(|c| *c >= 52) || flop.iter().copied().collect::<BTreeSet<_>>().len() != 3
        {
            return Err("cash flop terminal requires three unique public cards".into());
        }
        // There are C(49,2)=1176 proposals, of which C(45,2)=990 survive for
        // every compatible private pair. A showdown is invariant to arrival
        // order, so unordered enumeration is exact here (not for betting).
        let mut runouts = Vec::with_capacity(1176);
        for turn in 0..52u8 {
            if flop.contains(&turn) {
                continue;
            }
            for river in turn + 1..52u8 {
                if flop.contains(&river) {
                    continue;
                }
                runouts.push((
                    [turn, river],
                    PublicTerminalCache::new([flop[0], flop[1], flop[2], turn, river]),
                ));
            }
        }
        Ok(Self {
            flop,
            rules,
            runouts,
        })
    }

    fn validate_ranges(&self, invested: [f64; 2], reaches: &[Vec<f64>; 2]) -> Result<(), String> {
        for amount in invested {
            cash::cash_units(amount, &self.rules)?;
        }
        for range in reaches {
            if range.len() != EXACT_COMBO_COUNT || range.iter().any(|w| !w.is_finite() || *w < 0.0)
            {
                return Err(
                    "cash flop terminal requires finite nonnegative exact-combo reaches".into(),
                );
            }
            if range
                .iter()
                .zip(exact_combos())
                .any(|(w, c)| *w > 0.0 && c.cards().iter().any(|card| self.flop.contains(card)))
            {
                return Err("cash flop terminal reach includes a board-blocked hand".into());
            }
        }
        Ok(())
    }

    /// CFVs exclude own reach, including when a counterfactual branch has no
    /// own behavioral mass. Never normalize/resample away blocked runouts.
    pub fn counterfactual_values(
        &self,
        invested: [f64; 2],
        reaches: &[Vec<f64>; 2],
    ) -> Result<[Vec<f64>; 2], String> {
        self.validate_ranges(invested, reaches)?;
        let mut result = std::array::from_fn(|_| vec![0.0; EXACT_COMBO_COUNT]);
        let mut masked = reaches.clone();
        for (cards, cache) in &self.runouts {
            for p in 0..2 {
                for combo in exact_combos() {
                    masked[p][combo.key()] = if combo.cards().iter().any(|c| cards.contains(c)) {
                        0.0
                    } else {
                        reaches[p][combo.key()]
                    };
                }
                let values = cache.values_with_rules(
                    invested,
                    &masked[p],
                    1 - p,
                    RangeTerminalKind::Showdown,
                    Some(&self.rules),
                    5,
                )?;
                for (sum, value) in result[1 - p].iter_mut().zip(values) {
                    *sum += value / 990.0;
                }
            }
        }
        Ok(result)
    }

    pub fn evaluate(
        &self,
        invested: [f64; 2],
        reaches: &[Vec<f64>; 2],
    ) -> Result<CashRangeTerminalEvaluation, String> {
        self.validate_ranges(invested, reaches)?;
        let masses = compatible_masses(&reaches[1], &public_belief::combo_conflicts());
        let joint_reach_mass = reaches[0]
            .iter()
            .zip(masses)
            .map(|(r, m)| r * m)
            .sum::<f64>();
        if joint_reach_mass <= 0.0 {
            return Err("cash flop evaluation has no compatible joint reach".into());
        }
        let values = self.counterfactual_values(invested, reaches)?;
        let profile_net_bb: [f64; 2] = std::array::from_fn(|p| {
            reaches[p]
                .iter()
                .zip(&values[p])
                .map(|(r, v)| r * v)
                .sum::<f64>()
                / joint_reach_mass
        });
        let gross = 2 * cash::cash_units(invested[0].min(invested[1]), &self.rules)?;
        let house_rake_bb =
            self.rules.rake_units(gross, true)? as f64 / self.rules.units_per_bb as f64;
        Ok(CashRangeTerminalEvaluation {
            counterfactual_values_bb: values,
            joint_reach_mass,
            profile_net_bb,
            house_rake_bb,
            conservation_residual_bb: (profile_net_bb[0] + profile_net_bb[1] + house_rake_bb).abs(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::cash_game::{Outcome, TerminalReason};

    #[test]
    fn cash_flop_runouts_match_independent_settlements_with_odd_cent_ties() {
        let rules = crate::cash_game::study_rules("nl25").unwrap();
        let flop = [48, 45, 42];
        let holes = [[0, 4], [1, 5]];
        let mut ranges = [vec![0.0; EXACT_COMBO_COUNT], vec![0.0; EXACT_COMBO_COUNT]];
        for p in 0..2 {
            ranges[p][Combo::new(holes[p][0], holes[p][1]).key()] = 1.0;
        }
        let kernel = ExactCashFlopTerminal::new(flop, rules.clone()).unwrap();
        let mut expected = [0.0; 2];
        let mut ties = 0;
        let mut outcomes = 0;
        for turn in 0..52u8 {
            for river in turn + 1..52u8 {
                if [turn, river].iter().any(|card| {
                    flop.contains(card) || holes.iter().flatten().any(|hole| hole == card)
                }) {
                    continue;
                }
                let board = [flop[0], flop[1], flop[2], turn, river];
                let outcome = match showdown_result(&holes, &board) {
                    1.0 => Outcome::PlayerZero,
                    0.5 => {
                        ties += 1;
                        Outcome::Split
                    }
                    _ => Outcome::PlayerOne,
                };
                let settled = cash::cash_terminal_payoffs(
                    &rules,
                    [1.2; 2],
                    TerminalReason::Showdown,
                    outcome,
                    5,
                )
                .unwrap();
                for p in 0..2 {
                    expected[p] += settled[p] / 990.0;
                }
                outcomes += 1;
            }
        }
        assert_eq!(outcomes, 990);
        assert!(
            ties > 0,
            "fixture must distinguish equity from cent tie payouts"
        );
        let evaluated = kernel.evaluate([1.2; 2], &ranges).unwrap();
        for p in 0..2 {
            assert!((evaluated.profile_net_bb[p] - expected[p]).abs() < 1e-10);
            let key = Combo::new(holes[p][0], holes[p][1]).key();
            assert!((evaluated.counterfactual_values_bb[p][key] - expected[p]).abs() < 1e-10);
        }
        assert!((evaluated.house_rake_bb - 0.12).abs() < 1e-12);
        assert!(evaluated.conservation_residual_bb < 1e-10);
        // The odd split cent is awarded to BB, not erased by zero-sum shifting.
        assert!((expected[0] + expected[1] + 0.12).abs() < 1e-10);
        assert!((expected[0] - expected[1]).abs() > 0.0);

        let mut no_own_reach = ranges.clone();
        no_own_reach[0].fill(0.0);
        let cfv = kernel
            .counterfactual_values([1.2; 2], &no_own_reach)
            .unwrap();
        assert_eq!(cfv[0], evaluated.counterfactual_values_bb[0]);
        assert!(cfv[1].iter().all(|v| *v == 0.0));
        assert!(kernel.evaluate([1.2; 2], &no_own_reach).is_err());
        assert!(kernel.counterfactual_values([1.21; 2], &ranges).is_err());
        let mut blocked = ranges;
        blocked[0][Combo::new(flop[0], 3).key()] = 1.0;
        assert!(kernel.counterfactual_values([1.2; 2], &blocked).is_err());
    }

    #[test]
    fn cash_flop_kernel_rejects_invalid_public_cards_and_rules() {
        let rules = crate::cash_game::study_rules("nl25").unwrap();
        assert!(ExactCashFlopTerminal::new([0, 0, 2], rules.clone()).is_err());
        assert!(ExactCashFlopTerminal::new([0, 1, 52], rules.clone()).is_err());
        let mut wrong = rules;
        wrong.units_per_bb = 0;
        assert!(ExactCashFlopTerminal::new([0, 1, 2], wrong).is_err());
    }
}
