//! Exact all-in chance values for the fixed-flop research trainer. This does
//! not normalize reaches, unmask sampled future blockers, or change decisions.
use super::*;

const RUNOUT_UNITS: u16 = 1_980; // 990 legal unordered runouts, two units per win.
const BLOCKED: u16 = u16::MAX;

pub(crate) struct ExactFlopTerminal {
    pub(super) flop: [u8; 3],
    units: Vec<u16>,
}

impl ExactFlopTerminal {
    #[cfg(test)]
    pub(crate) fn class_totals(&self, class_by_combo: &[usize], count: usize) -> Result<(Vec<u64>, Vec<u64>), String> {
        if class_by_combo.len() != EXACT_COMBO_COUNT || count != 169
            || class_by_combo.iter().any(|c| *c >= count) {
            return Err("invalid preflop class mapping".into());
        }
        let mut wins = vec![0u64; count*count];
        let mut pairs = vec![0u64; count*count];
        for (i, units) in self.units.iter().enumerate() {
            if *units == BLOCKED { continue; }
            let key = class_by_combo[i/EXACT_COMBO_COUNT]*count + class_by_combo[i%EXACT_COMBO_COUNT];
            wins[key] += *units as u64;
            pairs[key] += 1;
        }
        Ok((wins, pairs))
    }

    pub(crate) fn from_equities(flop: [u8; 3], equities: &[f32]) -> Result<Self, String> {
        if flop.iter().any(|c| *c >= 52)
            || flop.iter().collect::<BTreeSet<_>>().len() != 3
            || equities.len() != EXACT_COMBO_COUNT * EXACT_COMBO_COUNT
        {
            return Err("invalid exact flop matrix dimensions or board".to_owned());
        }
        let combos = exact_combos();
        let legal = combos
            .iter()
            .map(|c| !c.cards().iter().any(|card| flop.contains(card)))
            .collect::<Vec<_>>();
        let mut units = vec![BLOCKED; equities.len()];
        for (i, first) in combos.iter().enumerate() {
            for (j, second) in combos.iter().enumerate() {
                let key = i * EXACT_COMBO_COUNT + j;
                let equity = equities[key];
                if !legal[i] || !legal[j] || first.overlaps(*second) {
                    if !equity.is_nan() {
                        return Err("blocked flop matrix entry must be absent".to_owned());
                    }
                    continue;
                }
                if !equity.is_finite() || !(0.0..=1.0).contains(&equity) {
                    return Err("compatible flop matrix entry is missing or invalid".to_owned());
                }
                // The existing matrix stores integer showdown counts / 1980
                // as f32. Recover the exact integer count; never carry f32
                // rounding into an allegedly exact terminal expectation.
                let count = (equity as f64 * RUNOUT_UNITS as f64).round() as u16;
                if count > RUNOUT_UNITS || count as f32 / RUNOUT_UNITS as f32 != equity {
                    return Err("flop equity is not on the exact 990-runout lattice".to_owned());
                }
                units[key] = count;
            }
        }
        for i in 0..EXACT_COMBO_COUNT {
            for j in 0..i {
                let a = units[i * EXACT_COMBO_COUNT + j];
                let b = units[j * EXACT_COMBO_COUNT + i];
                if (a == BLOCKED) != (b == BLOCKED) || (a != BLOCKED && a + b != RUNOUT_UNITS) {
                    return Err("flop equity matrix is not symmetric zero-sum".to_owned());
                }
            }
        }
        Ok(Self { flop, units })
    }

    pub(super) fn values(
        &self,
        board: [u8; 5],
        invested: [f64; 2],
        opponent_reach: &[f64],
        player: usize,
    ) -> Result<Vec<f64>, String> {
        self.values_for_board(&board, invested, opponent_reach, player)
    }

    #[cfg(test)]
    pub(crate) fn values_on_flop(
        &self, invested: [f64; 2], opponent_reach: &[f64], player: usize,
    ) -> Result<Vec<f64>, String> {
        self.values_for_board(&self.flop, invested, opponent_reach, player)
    }

    fn values_for_board(
        &self, board: &[u8], invested: [f64; 2], opponent_reach: &[f64], player: usize,
    ) -> Result<Vec<f64>, String> {
        if board[..3] != self.flop
            || player > 1
            || opponent_reach.len() != EXACT_COMBO_COUNT
            || invested.iter().any(|v| !v.is_finite() || *v < 0.0)
            || opponent_reach.iter().any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err("invalid exact flop terminal input".to_owned());
        }
        let combos = exact_combos();
        let active = combos
            .iter()
            .map(|c| !c.cards().iter().any(|card| board.contains(card)))
            .collect::<Vec<_>>();
        if opponent_reach
            .iter()
            .zip(&active)
            .any(|(reach, active)| *reach > 0.0 && !active)
        {
            return Err("exact flop terminal reach overlaps sampled future cards".to_owned());
        }
        let reached = opponent_reach
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, reach)| *reach > 0.0)
            .collect::<Vec<_>>();
        let pot = invested[0] + invested[1];
        let mut result = vec![0.0; EXACT_COMBO_COUNT];
        for (hero, value) in result.iter_mut().enumerate() {
            if !active[hero] {
                continue;
            }
            let row = hero * EXACT_COMBO_COUNT;
            for &(opponent, reach) in &reached {
                let count = self.units[row + opponent];
                if count != BLOCKED {
                    *value += reach * (pot * count as f64 / RUNOUT_UNITS as f64 - invested[player]);
                }
            }
        }
        Ok(result)
    }
}
