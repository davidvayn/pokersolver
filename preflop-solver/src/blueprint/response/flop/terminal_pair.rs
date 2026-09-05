//! Conditional-mean evaluation for the terminal-only correction. The profiles
//! have identical behavior until a defender flop fold/call ends the hand.
//! Integrate that last known-strategy action and every legal runout. This is
//! ordinary Rao-Blackwellization, not a full AIVAT implementation. Hidden cards
//! are used only for payout assessment, never for either player's strategy.

use super::*;

pub(super) fn rollout(
    control: &dyn ResponsePolicy,
    candidate: &dyn ResponsePolicy,
    attacker: &dyn ResponsePolicy,
    bank: &DecisionBank,
    mut state: GameState,
    deal: &Deal,
    game: &BlueprintConfig,
    responder: usize,
    mut rng: SplitMix64,
) -> [f64; 4] {
    let mut deviated = false;
    while state.terminal.is_none() {
        let actions = state.legal_actions(game);
        let selected = if state.actor == responder {
            match (!deviated)
                .then(|| bank.find(&state, deal, &actions, game))
                .flatten()
            {
                Some(decision) => {
                    rng.next_f64();
                    deviated = true;
                    decision.selected_action
                }
                None => sample_index(&attacker.strategy(&state, deal, &actions, game), &mut rng),
            }
        } else {
            let baseline = control.strategy(&state, deal, &actions, game);
            if state.street == Street::Flop
                && actions.len() == 2
                && state.remaining(responder, game) <= EPSILON
                && actions
                    .iter()
                    .all(|a| state.apply(a, game).terminal.is_some())
            {
                let proposal = candidate.strategy(&state, deal, &actions, game);
                let values = terminal_values(&state, deal, &actions, game);
                let a = baseline
                    .iter()
                    .zip(&values)
                    .map(|(p, v)| p * v)
                    .sum::<f64>();
                let b = proposal
                    .iter()
                    .zip(&values)
                    .map(|(p, v)| p * v)
                    .sum::<f64>();
                let sign = if responder == 0 { -1.0 } else { 1.0 };
                return [
                    sign * a,
                    sign * b,
                    u8::from(deviated) as f64,
                    u8::from(deviated) as f64,
                ];
            }
            sample_index(&baseline, &mut rng)
        };
        state = state.apply(&actions[selected], game);
    }
    let p0 = realized_utility_p0(&state, deal);
    let value = if responder == 0 { -p0 } else { p0 };
    [
        value,
        value,
        u8::from(deviated) as f64,
        u8::from(deviated) as f64,
    ]
}

fn terminal_values(
    state: &GameState,
    deal: &Deal,
    actions: &[LegalAction],
    game: &BlueprintConfig,
) -> Vec<f64> {
    assert_eq!(state.street, Street::Flop);
    actions
        .iter()
        .map(|action| {
            super::super::terminal::expectation(&state.apply(action, game), deal)
                .expect("postflop terminal action")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integrated_values_match_all_legal_realized_payoffs_and_ignore_future_cards() {
        let (policy, _) = super::super::super::tests::tabular_fixture();
        let game = &policy.table.config;
        let root = GameState::initial(game);
        let called = root.apply(&root.legal_actions(game)[1], game);
        let flop = called.apply(&called.legal_actions(game)[0], game);
        let facing = flop.apply(flop.legal_actions(game).last().unwrap(), game);
        let actions = facing.legal_actions(game);
        let deal = Deal::from_cards([[51, 50], [36, 37]], [48, 5, 10, 0, 1]);
        let values = terminal_values(&facing, &deal, &actions, game);
        let hidden_future = Deal::from_cards(deal.holes, [48, 5, 10, 12, 13]);
        assert_eq!(
            values,
            terminal_values(&facing, &hidden_future, &actions, game)
        );
        let mut reference = vec![0.0; actions.len()];
        let mut count = 0;
        for a in 0..52u8 {
            for b in a + 1..52u8 {
                if [a, b].iter().any(|c| {
                    deal.holes.iter().flatten().any(|h| h == c) || deal.board[..3].contains(c)
                }) {
                    continue;
                }
                let next = Deal::from_cards(deal.holes, [48, 5, 10, a, b]);
                for (index, action) in actions.iter().enumerate() {
                    reference[index] += realized_utility_p0(&facing.apply(action, game), &next);
                }
                count += 1;
            }
        }
        assert_eq!(count, 990);
        for (value, sum) in values.iter().zip(reference) {
            assert!((value - sum / 990.0).abs() < 1e-10);
        }
        let candidate = make_policy(
            Arc::clone(&policy.table),
            None,
            Some(Arc::new(FlopPatch {
                bank: DecisionBank::default(),
                weight: 0.25,
                all_in_samples: Some(2048),
                prior_terminal: None,
            })),
            None,
        );
        let a = rollout(
            &policy,
            candidate.as_ref(),
            &policy,
            &DecisionBank::default(),
            facing.clone(),
            &deal,
            game,
            1,
            SplitMix64::new(1),
        );
        let b = rollout(
            &policy,
            candidate.as_ref(),
            &policy,
            &DecisionBank::default(),
            facing.clone(),
            &hidden_future,
            game,
            1,
            SplitMix64::new(9),
        );
        assert_eq!(a, b);
        let expected: f64 = candidate
            .strategy(&facing, &deal, &actions, game)
            .iter()
            .zip(&values)
            .map(|(p, v)| p * v)
            .sum();
        assert_eq!(a[1], expected);
    }
}
