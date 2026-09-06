//! Exact-card Bayesian LBR, with a sampled checkdown heuristic on early streets.
//! The attack is a legal online strategy, not a learned bucket lookup. Its
//! measured payoff is lower-bound evidence, never an exploitability upper bound.
use super::*;
use crate::blueprint::neural::deal_for_policy_combo_on_board;
mod pilot;
mod tests;

struct Belief {
    seat: usize,
    own: [u8; 2],
    weights: Vec<f64>,
}

impl Belief {
    fn new(seat: usize, mut own: [u8; 2]) -> Result<Self, String> {
        if seat > 1 || own[0] == own[1] || own.iter().any(|c| *c >= 52) {
            return Err("LBR requires a legal seat and two distinct own cards".into());
        }
        own.sort_unstable();
        let mut value = Self {
            seat,
            own,
            weights: vec![1.0; 1326],
        };
        value.reveal(&[])?;
        Ok(value)
    }

    fn normalize(&mut self) -> Result<(), String> {
        let mass: f64 = self.weights.iter().sum();
        if !mass.is_finite()
            || mass <= 0.0
            || self.weights.iter().any(|w| *w < 0.0 || !w.is_finite())
        {
            return Err("LBR posterior has no finite compatible mass; no replacement range".into());
        }
        self.weights.iter_mut().for_each(|w| *w /= mass);
        Ok(())
    }

    fn reveal(&mut self, board: &[u8]) -> Result<(), String> {
        let unique = board.iter().copied().collect::<BTreeSet<_>>();
        if ![0, 3, 4, 5].contains(&board.len())
            || unique.len() != board.len()
            || board.iter().any(|c| *c >= 52 || self.own.contains(c))
        {
            return Err("LBR visible board is invalid".into());
        }
        for combo in all_combos() {
            if combo
                .cards()
                .iter()
                .any(|c| self.own.contains(c) || board.contains(c))
            {
                self.weights[combo.key()] = 0.0;
            }
        }
        self.normalize()
    }

    fn observe(
        &mut self,
        policy: &dyn ResponsePolicy,
        game: &BlueprintConfig,
        state: &GameState,
        board: &[u8],
        actions: &[LegalAction],
        selected: usize,
    ) -> Result<(), String> {
        if state.actor == self.seat || selected >= actions.len() {
            return Err(
                "LBR posterior may only condition on the opponent's observed action".into(),
            );
        }
        self.reveal(board)?;
        for combo in all_combos() {
            if self.weights[combo.key()] > 0.0 {
                self.weights[combo.key()] *=
                    query(policy, game, state, board, actions, combo)?[selected];
            }
        }
        self.normalize()
    }
}

fn query(
    policy: &dyn ResponsePolicy,
    game: &BlueprintConfig,
    state: &GameState,
    board: &[u8],
    actions: &[LegalAction],
    combo: Combo,
) -> Result<Vec<f64>, String> {
    if board.len() != state.street.board_len() {
        return Err("LBR policy query must contain exactly the visible board".into());
    }
    // This synthetic deal contains only the queried actor's holding and public
    // board. No actual opponent holding or unrevealed community card is passed.
    let deal = deal_for_policy_combo_on_board(combo, state.actor, board)?;
    let mix = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        policy.strategy(state, &deal, actions, game)
    }))
    .map_err(|_| {
        format!(
            "LBR defender query failed at {:?}, board={board:?}, combo={}, history={:?}",
            state.street,
            combo.key(),
            state.public_history
        )
    })?;
    if mix.len() != actions.len()
        || mix.iter().any(|p| !p.is_finite() || *p < 0.0)
        || (mix.iter().sum::<f64>() - 1.0).abs() > 1e-6
    {
        return Err("LBR queried an invalid defender policy; no uniform substitution".into());
    }
    Ok(mix)
}

struct Lbr {
    seed: u64,
    early_runouts_per_combo: u32,
}

impl Lbr {
    fn equity_by_combo(
        &self,
        belief: &Belief,
        board: &[u8],
        state: &GameState,
    ) -> Result<Vec<f64>, String> {
        if self.early_runouts_per_combo == 0 || board.len() != state.street.board_len() {
            return Err("LBR equity requires positive sampling and the visible board".into());
        }
        let mut identity = b"exact-card-lbr-checkdown-equity-v1".to_vec();
        identity.extend_from_slice(&self.seed.to_le_bytes());
        identity.extend_from_slice(&belief.own);
        identity.extend_from_slice(board);
        identity.extend_from_slice(state.public_history.join("|").as_bytes());
        let seed = stable_hash(&identity);
        let mut equity = vec![0.0; 1326];
        for combo in all_combos() {
            if belief.weights[combo.key()] == 0.0 {
                continue;
            }
            let opponent = combo.cards();
            let available: Vec<u8> = (0..52)
                .filter(|c| !belief.own.contains(c) && !opponent.contains(c) && !board.contains(c))
                .collect();
            let mut full = [0; 5];
            full[..board.len()].copy_from_slice(board);
            let holes = [belief.own, opponent];
            equity[combo.key()] = match board.len() {
                5 => showdown_result(&holes, &full),
                4 => {
                    assert_eq!(available.len(), 44);
                    available
                        .iter()
                        .map(|c| {
                            full[4] = *c;
                            showdown_result(&holes, &full)
                        })
                        .sum::<f64>()
                        / 44.0
                }
                _ => {
                    let mut rng = SplitMix64::new(derived_seed(seed, combo.key() as u64, 0));
                    let mut sum = 0.0;
                    for _ in 0..self.early_runouts_per_combo {
                        let mut deck = available.clone();
                        for next in board.len()..5 {
                            full[next] = deck.swap_remove(rng.index(deck.len()));
                        }
                        sum += showdown_result(&holes, &full);
                    }
                    sum / self.early_runouts_per_combo as f64
                }
            };
        }
        Ok(equity)
    }

    fn values(
        &self,
        belief: &Belief,
        policy: &dyn ResponsePolicy,
        game: &BlueprintConfig,
        state: &GameState,
        board: &[u8],
        actions: &[LegalAction],
    ) -> Result<Vec<f64>, String> {
        if state.actor != belief.seat || state.terminal.is_some() {
            return Err("LBR decision must belong to the responder and be nonterminal".into());
        }
        let equity = self.equity_by_combo(belief, board, state)?;
        let mean_equity: f64 = belief.weights.iter().zip(&equity).map(|(w, e)| w * e).sum();
        let hero = belief.seat;
        let opponent = 1 - hero;
        actions
            .iter()
            .map(|action| {
                let after = state.apply(action, game);
                match action.kind {
                    ActionKind::Fold => Ok(-state.invested[hero]),
                    ActionKind::Call | ActionKind::Check => {
                        // Heuristic: after matching the current bet, check down.
                        // Correct net-chip accounting includes already invested chips.
                        Ok(mean_equity * after.pot() - after.invested[hero])
                    }
                    ActionKind::RaiseTo(_) => {
                        let replies = after.legal_actions(game);
                        let fold = replies
                            .iter()
                            .position(|a| a.kind == ActionKind::Fold)
                            .ok_or("LBR bet must offer an opponent fold")?;
                        if after.actor != opponent || after.street != state.street {
                            return Err("LBR bet unexpectedly changed street/actor".into());
                        }
                        let mut value = 0.0;
                        for combo in all_combos() {
                            let weight = belief.weights[combo.key()];
                            if weight == 0.0 {
                                continue;
                            }
                            let fold_probability =
                                query(policy, game, &after, board, &replies, combo)?[fold];
                            // All nonfold replies are treated as calls only in the
                            // heuristic. Actual play still samples every legal reply,
                            // including raises, from the complete defender policy.
                            let called = (2.0 * equity[combo.key()] - 1.0) * after.invested[hero];
                            value += weight
                                * (fold_probability * state.invested[opponent]
                                    + (1.0 - fold_probability) * called);
                        }
                        Ok(value)
                    }
                }
            })
            .collect()
    }
}

fn best_index(values: &[f64]) -> usize {
    assert!(!values.is_empty() && values.iter().all(|v| v.is_finite()));
    let mut selected = 0;
    for index in 1..values.len() {
        if values[index] > values[selected] {
            selected = index;
        }
    }
    selected
}
