//! Read-only research inspection of the actual local frozen average policy.
//! No regrets, fallback actions, resumable state or serving artifact is exposed.
use super::*;

struct Row {
    labels: Arc<[Arc<str>]>,
    average: Option<Vec<f64>>,
    average_visits: u64,
    regret_updates: u64,
}

pub(in crate::blueprint) struct FrozenContinuation {
    game: BlueprintConfig,
    flop: Vec<u8>,
    root_history: Vec<String>,
    rows: BTreeMap<u64, Row>,
}

impl FrozenContinuation {
    pub(in crate::blueprint) fn query(
        &self,
        state: &GameState,
        deal: &Deal,
        actions: &[LegalAction],
    ) -> Result<&[f64], &'static str> {
        if state.terminal.is_some()
            || state.street == Street::Preflop
            || deal.board[..3] != self.flop
            || !state.public_history.starts_with(&self.root_history)
        {
            return Err("outside_captured_subgame");
        }
        let key = information_set(state, deal, &self.game).0;
        let row = self.rows.get(&key).ok_or("missing_information_set")?;
        if !row
            .labels
            .iter()
            .map(AsRef::as_ref)
            .eq(actions.iter().map(|a| a.label.as_str()))
        {
            return Err("incompatible_legal_actions");
        }
        let average = row
            .average
            .as_ref()
            .filter(|_| row.average_visits > 0)
            .ok_or("no_average_contributions")?;
        if row.regret_updates == 0 {
            return Err("no_regret_updates");
        }
        if average.iter().any(|p| !p.is_finite() || *p < 0.0)
            || (average.iter().sum::<f64>() - 1.0).abs() > 1e-12
        {
            return Err("invalid_average_probability");
        }
        Ok(average)
    }
}

pub(in crate::blueprint) fn capture(
    config: SampledFlopConfig,
    exact: bool,
) -> Result<(SampledFlopSolution, FrozenContinuation), String> {
    let flop = config.state.board.clone();
    let root_history = config.state.public_history.clone();
    solve_observed(config, exact, |trainer| FrozenContinuation {
        game: trainer.config.clone(),
        flop,
        root_history,
        rows: trainer
            .nodes
            .iter()
            .map(|(key, node)| {
                (
                    *key,
                    Row {
                        labels: node.action_labels.clone(),
                        average: (node.average_visits > 0
                            && node.strategy_sum.iter().any(|p| *p > 0.0))
                        .then(|| node.average_strategy()),
                        average_visits: node.average_visits,
                        regret_updates: node.regret_updates,
                    },
                )
            })
            .collect(),
    })
}

#[test]
fn reproduces_unavailable_turn_average_without_hidden_cards_or_lookup_mismatch() {
    #[derive(Deserialize)]
    struct Fixture {
        game: BlueprintConfig,
        public: PublicBeliefState,
    }
    let fixture: Fixture = serde_json::from_slice(include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/flop-continuation-public-root-a.json"
    )))
    .unwrap();
    assert_eq!(fixture.game.effective_stack_bb, 20.0);
    for exact in [false, true] {
        let config = SampledFlopConfig {
            game: fixture.game.clone(),
            state: fixture.public.clone(),
            iterations: 32,
            seed: 3454475668975051736,
            maximum_information_sets: 2_000_000,
        };
        let (_, frozen) = capture(config, exact).unwrap();
        let mut state = fixture.public.game_state();
        let labels = if exact {
            vec!["check", "bet_to_2.500bb", "raise_to_9.500bb", "call"]
        } else {
            vec!["check", "bet_to_2.500bb", "call"]
        };
        for label in labels {
            let action = state
                .legal_actions(&fixture.game)
                .into_iter()
                .find(|a| a.label == label)
                .unwrap();
            state = state.apply(&action, &fixture.game);
        }
        assert_eq!(state.street, Street::Turn);
        assert_eq!(state.actor, 1);
        let visible = [24, 14, 5, 50];
        let deal =
            deal_for_policy_combo_on_board(Combo::new(35, 26), state.actor, &visible).unwrap();
        let actions = state.legal_actions(&fixture.game);
        let key = information_set(&state, &deal, &frozen.game).0;
        let row = frozen
            .rows
            .get(&key)
            .expect("actual training key exists; this is not a lookup mismatch");
        let expected = if exact {
            assert_eq!(row.average_visits, 0);
            assert!(row.regret_updates > 0);
            "no_average_contributions"
        } else {
            assert!(row.average_visits > 0);
            assert_eq!(row.regret_updates, 0);
            "no_regret_updates"
        };
        assert_eq!(frozen.query(&state, &deal, &actions), Err(expected));
        // Fill hidden cards differently. The visible information set and the
        // missing-update diagnosis must remain identical.
        let changed = Deal::from_sampled_cards([[3, 4], [35, 26]], [24, 14, 5, 50, 6]);
        assert_ne!(deal.holes[0], changed.holes[0]);
        assert_ne!(deal.board[4], changed.board[4]);
        assert_eq!(information_set(&state, &changed, &frozen.game).0, key);
        assert_eq!(frozen.query(&state, &changed, &actions), Err(expected));
    }
}

#[test]
fn captured_average_matches_root_and_refuses_missing_or_untrained_rows() {
    let config = super::tests::fixture();
    let (solution, mut frozen) = capture(config.clone(), false).unwrap();
    assert_eq!(solution, solve(config.clone()).unwrap());
    let state = config.state.game_state();
    let actions = state.legal_actions(&config.game);
    let combo = Combo::new(51, 50);
    let deal = deal_for_policy_combo_on_board(combo, state.actor, &config.state.board).unwrap();
    let key = information_set(&state, &deal, &frozen.game).0;
    let mix = frozen.query(&state, &deal, &actions).unwrap();
    for (i, p) in mix.iter().enumerate() {
        assert_eq!(
            *p as f32,
            solution.root.probabilities[combo.key() * actions.len() + i]
        );
    }
    frozen.rows.get_mut(&key).unwrap().regret_updates = 0;
    assert_eq!(
        frozen.query(&state, &deal, &actions),
        Err("no_regret_updates")
    );
    frozen.rows.get_mut(&key).unwrap().average = None;
    assert_eq!(
        frozen.query(&state, &deal, &actions),
        Err("no_average_contributions")
    );
    frozen.rows.remove(&key);
    assert_eq!(
        frozen.query(&state, &deal, &actions),
        Err("missing_information_set")
    );
}
