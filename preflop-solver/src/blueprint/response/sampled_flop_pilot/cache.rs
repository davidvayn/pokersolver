//! Frozen development action samples. Never a serving value oracle or a fresh
//! holdout when reused to choose another proposal.
use super::*;

const SCHEMA: &str = "hu-conditional-flop-action-cache-v1";

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(super) struct Context {
    pub checkpoint_sha256: String,
    pub game: BlueprintConfig,
    pub public: PublicBeliefState,
    pub turn_resolver: TurnResolveOptions,
    pub terminal_flop: TerminalFlopOptions,
    pub evaluation_seed: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct ActionSample {
    actor_combo: usize,
    baseline: Vec<f64>,
    action_values_bb: Vec<f64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(super) struct RootActionCache {
    schema: String,
    pub context: Context,
    action_labels: Vec<String>,
    samples: Vec<ActionSample>,
}

impl Context {
    pub(super) fn root(&self) -> Result<GameState, String> {
        self.game.validate()?;
        self.turn_resolver.validate()?;
        self.terminal_flop.validate()?;
        let public = &self.public;
        if self.checkpoint_sha256.len() != 64
            || !self
                .checkpoint_sha256
                .bytes()
                .all(|c| c.is_ascii_hexdigit())
            || public.street != Street::Flop
            || public.actor > 1
            || public.board.len() != 3
            || public.board.iter().any(|c| *c >= 52)
            || public.board.iter().collect::<BTreeSet<_>>().len() != 3
        {
            return Err("invalid frozen root context".to_owned());
        }
        for range in &public.ranges {
            if range.len() != 1326
                || range.iter().any(|v| !v.is_finite() || *v < 0.0)
                || (range.iter().sum::<f64>() - 1.0).abs() > 1e-10
            {
                return Err("invalid frozen root range".to_owned());
            }
            for combo in all_combos() {
                if combo.cards().iter().any(|c| public.board.contains(c))
                    && range[combo.key()] != 0.0
                {
                    return Err("blocked combo in frozen root range".to_owned());
                }
            }
        }
        let mut root = GameState::initial(&self.game);
        for observed in &public.trajectory {
            if root.terminal.is_some() {
                return Err("frozen root line continued after terminal".to_owned());
            }
            let action = root
                .legal_actions(&self.game)
                .into_iter()
                .find(|a| trajectory_action_matches(&root, a, observed, &self.game))
                .ok_or("frozen root has an illegal public line")?;
            root = root.apply(&action, &self.game);
        }
        if root.terminal.is_some()
            || PublicBeliefState::from_game_state(
                public.board.clone(),
                &root,
                public.ranges.clone(),
            ) != *public
        {
            return Err("frozen root line/accounting mismatch".to_owned());
        }
        Ok(root)
    }
}

impl RootActionCache {
    pub(super) fn sample_count(&self) -> u64 {
        self.samples.len() as u64
    }

    pub(super) fn collect(
        policy: &dyn ResponsePolicy,
        context: Context,
        count: u64,
    ) -> Result<Self, String> {
        if !(2..=4096).contains(&count) {
            return Err("cache requires 2..4096 development samples".to_owned());
        }
        let state = context.root()?;
        let actions = state.legal_actions(&context.game);
        let mut rng = SplitMix64::new(context.evaluation_seed);
        let mut samples = Vec::new();
        for sample in 0..count {
            let deal = conditional_deal(&context.public.board, &context.public.ranges, &mut rng);
            let combo = Combo::new(deal.holes[state.actor][0], deal.holes[state.actor][1]);
            let baseline = policy.strategy(&state, &deal, &actions, &context.game);
            let action_values_bb = actions
                .iter()
                .map(|action| {
                    let mut continuation_rng =
                        SplitMix64::new(derived_seed(context.evaluation_seed, 0x666c6f70, sample));
                    let p0 = baseline_rollout(
                        policy,
                        state.apply(action, &context.game),
                        &deal,
                        &context.game,
                        &mut continuation_rng,
                    );
                    if state.actor == 0 {
                        p0
                    } else {
                        -p0
                    }
                })
                .collect();
            samples.push(ActionSample {
                actor_combo: combo.key(),
                baseline,
                action_values_bb,
            });
        }
        let cache = Self {
            schema: SCHEMA.to_owned(),
            context,
            action_labels: actions.into_iter().map(|a| a.label).collect(),
            samples,
        };
        cache.validate()?;
        Ok(cache)
    }

    /// For frozen policies only: stop each rollout on entry to the turn,
    /// group identical public turn roots, then restore original sample/action
    /// order. Every request carries its unchanged continuation RNG. The live
    /// resolver still stores just one generation, not an unbounded row cache.
    pub(super) fn collect_grouped(
        policy: &dyn ResponsePolicy,
        context: Context,
        count: u64,
    ) -> Result<(Self, usize), String> {
        struct Request {
            sample: usize,
            action: usize,
            state: GameState,
            deal: Deal,
            rng: SplitMix64,
        }
        if !(2..=4096).contains(&count) {
            return Err("cache requires 2..4096 development samples".to_owned());
        }
        let state = context.root()?;
        let game = &context.game;
        let actions = state.legal_actions(game);
        let mut chance = SplitMix64::new(context.evaluation_seed);
        let mut samples = Vec::new();
        let mut groups = BTreeMap::<(Vec<u8>, Vec<String>), Vec<Request>>::new();
        for sample in 0..count as usize {
            let deal = conditional_deal(&context.public.board, &context.public.ranges, &mut chance);
            let combo = Combo::new(deal.holes[state.actor][0], deal.holes[state.actor][1]);
            samples.push(ActionSample {
                actor_combo: combo.key(),
                baseline: policy.strategy(&state, &deal, &actions, game),
                action_values_bb: vec![f64::NAN; actions.len()],
            });
            for (action, first) in actions.iter().enumerate() {
                let mut cursor = state.apply(first, game);
                let mut rng = SplitMix64::new(derived_seed(
                    context.evaluation_seed,
                    0x666c6f70,
                    sample as u64,
                ));
                while cursor.terminal.is_none() && cursor.street == Street::Flop {
                    let legal = cursor.legal_actions(game);
                    let mix = policy.strategy(&cursor, &deal, &legal, game);
                    cursor = cursor.apply(&legal[sample_index(&mix, &mut rng)], game);
                }
                if cursor.terminal.is_some() {
                    let p0 = realized_utility_p0(&cursor, &deal);
                    samples[sample].action_values_bb[action] =
                        if state.actor == 0 { p0 } else { -p0 };
                } else {
                    if cursor.street != Street::Turn {
                        return Err("grouped rollout did not stop at turn entry".to_owned());
                    }
                    groups
                        .entry((deal.board[..4].to_vec(), cursor.public_history.clone()))
                        .or_default()
                        .push(Request {
                            sample,
                            action,
                            state: cursor,
                            deal: deal.clone(),
                            rng,
                        });
                }
            }
        }
        let unique_turn_roots = groups.len();
        for requests in groups.into_values() {
            for mut request in requests {
                let p0 =
                    baseline_rollout(policy, request.state, &request.deal, game, &mut request.rng);
                samples[request.sample].action_values_bb[request.action] =
                    if state.actor == 0 { p0 } else { -p0 };
            }
        }
        let cache = Self {
            schema: SCHEMA.to_owned(),
            context,
            action_labels: actions.into_iter().map(|a| a.label).collect(),
            samples,
        };
        cache.validate()?;
        Ok((cache, unique_turn_roots))
    }

    fn validate(&self) -> Result<(), String> {
        let root = self.context.root()?;
        let actions = root.legal_actions(&self.context.game);
        if self.schema != SCHEMA
            || !(2..=4096).contains(&self.samples.len())
            || !self
                .action_labels
                .iter()
                .map(String::as_str)
                .eq(actions.iter().map(|a| a.label.as_str()))
        {
            return Err("incompatible frozen action cache".to_owned());
        }
        for sample in &self.samples {
            if sample.actor_combo >= 1326
                || self.context.public.ranges[root.actor][sample.actor_combo] == 0.0
                || sample.baseline.len() != actions.len()
                || sample.action_values_bb.len() != actions.len()
                || sample.baseline.iter().any(|p| !p.is_finite() || *p < 0.0)
                || (sample.baseline.iter().sum::<f64>() - 1.0).abs() > 1e-10
                || sample.action_values_bb.iter().any(|q| {
                    !q.is_finite() || q.abs() > self.context.game.effective_stack_bb + 1e-9
                })
            {
                return Err("invalid frozen action sample".to_owned());
            }
        }
        Ok(())
    }

    pub(super) fn encode(&self) -> Result<Vec<u8>, String> {
        self.validate()?;
        rmp_serde::to_vec_named(self).map_err(|e| e.to_string())
    }

    pub(super) fn decode(bytes: &[u8]) -> Result<Self, String> {
        let cache: Self = rmp_serde::from_slice(bytes).map_err(|e| e.to_string())?;
        cache.validate()?;
        Ok(cache)
    }

    // Return the paired differences as well as gains against the baseline so
    // callers do not compare two separate SEs and discard their covariance.
    pub(super) fn score(
        &self,
        rows: &[PublicBeliefStrategy],
    ) -> Result<(Vec<ValueAccumulator>, Vec<ValueAccumulator>), String> {
        self.validate()?;
        if rows.is_empty() {
            return Err("no proposed rows to score".to_owned());
        }
        let root = self.context.root()?;
        let actions = root.legal_actions(&self.context.game);
        let combos = all_combos();
        let mut values = vec![ValueAccumulator::default(); rows.len()];
        let mut vs_first = vec![ValueAccumulator::default(); rows.len()];
        for sample in &self.samples {
            let gains: Vec<f64> = rows
                .iter()
                .map(|row| {
                    row_mix(row, combos[sample.actor_combo], &root, &actions)
                        .iter()
                        .zip(&sample.baseline)
                        .zip(&sample.action_values_bb)
                        .map(|((candidate, control), q)| (candidate - control) * q)
                        .sum()
                })
                .collect();
            for (index, gain) in gains.iter().enumerate() {
                values[index].observe(*gain, &CoverageCounter::default());
                vs_first[index].observe(gain - gains[0], &CoverageCounter::default());
            }
        }
        Ok((values, vs_first))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_action_samples_round_trip_and_match_direct_paired_evaluation() {
        let (base, _) = super::super::super::tests::tabular_fixture();
        let game = base.table.config.clone();
        let mut root = GameState::initial(&game);
        while root.street != Street::Flop {
            let action = root
                .legal_actions(&game)
                .into_iter()
                .find(|a| matches!(a.kind, ActionKind::Call | ActionKind::Check))
                .unwrap();
            root = root.apply(&action, &game);
        }
        let board = vec![0, 5, 10];
        let ranges = public_ranges(&base, &root, &board).unwrap();
        let context = Context {
            checkpoint_sha256: "a".repeat(64),
            game: game.clone(),
            public: PublicBeliefState::from_game_state(board.clone(), &root, ranges.clone()),
            turn_resolver: TurnResolveOptions {
                iterations: 4,
                safe_bilateral: false,
                maximum_policy_rows: 20000,
            },
            terminal_flop: TerminalFlopOptions {
                equity_samples: 2048,
                weight: 0.5,
            },
            evaluation_seed: 96,
        };
        let cache = RootActionCache::collect(&base, context, 32).unwrap();
        assert_eq!(
            cache,
            RootActionCache::decode(&cache.encode().unwrap()).unwrap()
        );
        let actions = root.legal_actions(&game);
        let mut rows = Vec::new();
        for selected in 0..actions.len() {
            let mut row = PublicBeliefStrategy {
                public_history: root.public_history.clone(),
                actor: root.actor,
                action_labels: actions.iter().map(|a| a.label.clone()).collect(),
                probabilities: vec![0.0; 1326 * actions.len()],
                action_values_bb: None,
            };
            for combo in all_combos() {
                if ranges[root.actor][combo.key()] > 0.0 {
                    row.probabilities[combo.key() * actions.len() + selected] = 1.0;
                }
            }
            rows.push(row);
        }
        let direct = compare(&base, &game, &root, &board, &ranges, &rows, 96, 32);
        let (cached, differences) = cache.score(&rows).unwrap();
        for (a, b) in direct.iter().zip(&cached) {
            assert_eq!(a.mean(), b.mean());
            assert_eq!(a.standard_error(), b.standard_error());
        }
        assert!(differences[0].mean() == 0.0 && differences[0].standard_error() == 0.0);
        for (i, delta) in differences.iter().enumerate() {
            assert!((delta.mean() - (cached[i].mean() - cached[0].mean())).abs() < 1e-12);
        }
        let mut invalid = cache.clone();
        invalid.context.public.invested_bb[0] += 0.25;
        assert!(invalid.encode().is_err());
        invalid = cache.clone();
        invalid.samples[0].actor_combo = 1326;
        assert!(invalid.encode().is_err());
        invalid = cache.clone();
        invalid.samples[0].action_values_bb[0] = f64::NAN;
        assert!(invalid.encode().is_err());
        invalid = cache.clone();
        invalid.samples[0].baseline.fill(0.0);
        assert!(invalid.encode().is_err());
        invalid = cache.clone();
        invalid.action_labels[0] = "illegal".to_owned();
        assert!(invalid.encode().is_err());
        let bytes = cache.encode().unwrap();
        assert!(RootActionCache::decode(&bytes[..bytes.len() / 2]).is_err());
    }

    #[test]
    fn grouped_turn_requests_preserve_every_sample_and_reduce_actual_solves() {
        let (mut base, _) = super::super::super::tests::tabular_fixture();
        let table = Arc::get_mut(&mut base.table).unwrap();
        table.config.effective_stack_bb = 2.0;
        table.nodes.clear();
        let game = base.table.config.clone();
        let mut root = GameState::initial(&game);
        while root.street != Street::Flop {
            let action = root
                .legal_actions(&game)
                .into_iter()
                .find(|a| matches!(a.kind, ActionKind::Call | ActionKind::Check))
                .unwrap();
            root = root.apply(&action, &game);
        }
        let board = vec![0, 5, 10];
        let context = Context {
            checkpoint_sha256: "b".repeat(64),
            game,
            public: PublicBeliefState::from_game_state(
                board.clone(),
                &root,
                public_ranges(&base, &root, &board).unwrap(),
            ),
            turn_resolver: TurnResolveOptions {
                iterations: 2,
                safe_bilateral: false,
                maximum_policy_rows: 20000,
            },
            terminal_flop: TerminalFlopOptions {
                equity_samples: 2048,
                weight: 0.5,
            },
            evaluation_seed: 97,
        };
        let direct_policy =
            turn::TabularTurnPolicy::new(base.isolated_copy(), context.turn_resolver.clone());
        let grouped_policy = turn::TabularTurnPolicy::new(base, context.turn_resolver.clone());
        let direct = RootActionCache::collect(&direct_policy, context.clone(), 64).unwrap();
        let (grouped, unique) =
            RootActionCache::collect_grouped(&grouped_policy, context, 64).unwrap();
        assert_eq!(direct, grouped);
        assert_eq!(direct.encode().unwrap(), grouped.encode().unwrap());
        let direct_solves = direct_policy.take_resolution_diagnostics().unwrap()["solved_roots"]
            .as_u64()
            .unwrap();
        let grouped_solves = grouped_policy.take_resolution_diagnostics().unwrap()["solved_roots"]
            .as_u64()
            .unwrap();
        assert_eq!(grouped_solves, unique as u64);
        assert!(grouped_solves > 0 && grouped_solves < direct_solves);
    }
}
