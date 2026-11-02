use super::*;

pub(super) struct Board {
    pub cards: Vec<u8>,
    pub legal: [Vec<usize>; 2],
    pub rows: [Vec<Option<usize>>; 2],
    pub equity: Vec<f64>, // p0 sign: win 1, tie 0, loss -1; incompatible pairs zero.
}

pub(super) enum Kind {
    Decision {
        actor: usize,
        children: Vec<usize>,
    },
    Chance {
        children: Vec<(u8, usize)>,
        divisor: f64,
    },
    Terminal {
        terminal: Terminal,
        invested: [f64; 2],
    },
}

pub(super) struct Node {
    pub board: usize,
    pub kind: Kind,
    pub regrets: Vec<f64>,
    pub sums: Vec<f64>,
}

pub(super) struct Game {
    pub combos: [Vec<crate::cards::Combo>; 2],
    pub prior: [Vec<f64>; 2],
    pub boards: Vec<Board>,
    pub nodes: Vec<Node>,
    pub root: usize,
    pub compatible: Vec<bool>,
}

impl Game {
    pub fn new(config: &SpotConfig, guard: &Guard) -> Result<Self, String> {
        let all = all_combos();
        let combos: [Vec<_>; 2] = std::array::from_fn(|p| {
            all.iter()
                .copied()
                .filter(|c| config.state.ranges[p][c.key()] > 0.0)
                .collect()
        });
        let prior = std::array::from_fn(|p| {
            combos[p]
                .iter()
                .map(|c| config.state.ranges[p][c.key()])
                .collect()
        });
        let compatible = combos[0]
            .iter()
            .flat_map(|a| combos[1].iter().map(move |b| !a.overlaps(*b)))
            .collect();
        let mut game = Self {
            combos,
            prior,
            compatible,
            boards: Vec::new(),
            nodes: Vec::new(),
            root: 0,
        };
        let mut boards = BTreeMap::new();
        let root_board = game.build_board(config.state.board.clone(), &mut boards, guard)?;
        game.root = game.build_node(
            config.state.game_state(),
            root_board,
            &config.game,
            &boards,
            guard,
        )?;
        Ok(game)
    }

    fn build_board(
        &mut self,
        cards: Vec<u8>,
        indexes: &mut BTreeMap<Vec<u8>, usize>,
        guard: &Guard,
    ) -> Result<usize, String> {
        guard.tick()?;
        let id = self.boards.len();
        indexes.insert(cards.clone(), id);
        let legal: [Vec<usize>; 2] = std::array::from_fn(|p| {
            self.combos[p]
                .iter()
                .enumerate()
                .filter(|(_, c)| !c.cards().iter().any(|c| cards.contains(c)))
                .map(|(i, _)| i)
                .collect()
        });
        let rows = std::array::from_fn(|p| {
            let mut rows = vec![None; self.combos[p].len()];
            for (row, i) in legal[p].iter().enumerate() {
                rows[*i] = Some(row);
            }
            rows
        });
        let mut equity = vec![0.0; self.combos[0].len() * self.combos[1].len()];
        self.boards.push(Board {
            cards: cards.clone(),
            legal: legal.clone(),
            rows,
            equity: vec![],
        });
        if cards.len() == 5 {
            let ranks: [Vec<_>; 2] = std::array::from_fn(|p| {
                self.combos[p]
                    .iter()
                    .map(|combo| {
                        if combo.cards().iter().any(|c| cards.contains(c)) {
                            return 0;
                        }
                        let mut hand = cards.clone();
                        hand.extend(combo.cards());
                        evaluate(&hand)
                    })
                    .collect()
            });
            for a in &legal[0] {
                for b in &legal[1] {
                    let pair = self.pair(*a, *b);
                    if self.compatible[pair] {
                        equity[pair] = match ranks[0][*a].cmp(&ranks[1][*b]) {
                            std::cmp::Ordering::Greater => 1.0,
                            std::cmp::Ordering::Less => -1.0,
                            std::cmp::Ordering::Equal => 0.0,
                        };
                    }
                }
            }
        } else {
            // Both exact private hands remove four cards. There are 45 legal
            // turns per pair, then 44 rivers; not 49/48 public candidates.
            let divisor = (48 - cards.len()) as f64;
            for card in 0..52u8 {
                if cards.contains(&card) {
                    continue;
                }
                let mut next = cards.clone();
                next.push(card);
                let child = self.build_board(next, indexes, guard)?;
                for (value, child) in equity.iter_mut().zip(&self.boards[child].equity) {
                    *value += child / divisor;
                }
            }
        }
        self.boards[id].equity = equity;
        Ok(id)
    }

    fn build_node(
        &mut self,
        state: GameState,
        board: usize,
        game: &BlueprintConfig,
        indexes: &BTreeMap<Vec<u8>, usize>,
        guard: &Guard,
    ) -> Result<usize, String> {
        guard.tick()?;
        if self.nodes.len() >= 2_000_000 {
            return Err("exact-flop total tree node limit exceeded".into());
        }
        let kind = if let Some(terminal) = &state.terminal {
            Kind::Terminal {
                terminal: terminal.clone(),
                invested: state.invested,
            }
        } else if state.street.board_len() > self.boards[board].cards.len() {
            let mut children = Vec::new();
            let cards = self.boards[board].cards.clone();
            for card in 0..52u8 {
                if cards.contains(&card) {
                    continue;
                }
                let mut next = cards.clone();
                next.push(card);
                children.push((
                    card,
                    self.build_node(state.clone(), indexes[&next], game, indexes, guard)?,
                ));
            }
            Kind::Chance {
                children,
                divisor: (48 - cards.len()) as f64,
            }
        } else {
            let children = state
                .legal_actions(game)
                .iter()
                .map(|a| self.build_node(state.apply(a, game), board, game, indexes, guard))
                .collect::<Result<Vec<_>, _>>()?;
            if children.is_empty() {
                return Err("nonterminal exact-flop node has no actions".into());
            }
            Kind::Decision {
                actor: state.actor,
                children,
            }
        };
        let cells = match &kind {
            Kind::Decision { actor, children } => {
                self.boards[board].legal[*actor].len() * children.len()
            }
            _ => 0,
        };
        let id = self.nodes.len();
        self.nodes.push(Node {
            board,
            kind,
            regrets: vec![0.0; cells],
            sums: vec![0.0; cells],
        });
        Ok(id)
    }

    pub fn pair(&self, a: usize, b: usize) -> usize {
        a * self.combos[1].len() + b
    }

    pub fn joint_mass(&self) -> f64 {
        self.prior[0]
            .iter()
            .enumerate()
            .flat_map(|(a, w)| {
                self.prior[1]
                    .iter()
                    .enumerate()
                    .filter(move |(b, _)| self.compatible[self.pair(a, *b)])
                    .map(move |(_, v)| w * v)
            })
            .sum()
    }

    pub fn decisions(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| matches!(n.kind, Kind::Decision { .. }))
            .count()
    }
    pub fn array_bytes(&self) -> usize {
        self.nodes
            .iter()
            .map(|n| (n.regrets.len() + n.sums.len()) * 8)
            .sum()
    }

    fn strategy(&self, id: usize) -> Vec<f64> {
        let node = &self.nodes[id];
        let Kind::Decision { children, .. } = &node.kind else {
            return vec![];
        };
        let n = children.len();
        node.regrets
            .chunks_exact(n)
            .flat_map(|row| {
                let sum: f64 = row.iter().map(|v| v.max(0.0)).sum();
                row.iter().map(move |v| {
                    if sum > EPSILON {
                        v.max(0.0) / sum
                    } else {
                        1.0 / n as f64
                    }
                })
            })
            .collect()
    }

    pub fn freeze(&self) -> Result<Vec<Vec<f32>>, String> {
        self.nodes
            .iter()
            .map(|node| {
                let Kind::Decision { children, .. } = &node.kind else {
                    return Ok(vec![]);
                };
                let n = children.len();
                node.sums
                    .chunks_exact(n)
                    .map(|row| {
                        let sum: f64 = row.iter().sum();
                        if !sum.is_finite() || row.iter().any(|v| !v.is_finite() || *v < 0.0) {
                            return Err("invalid exact-flop average accumulators".to_owned());
                        }
                        Ok(row
                            .iter()
                            .map(|v| {
                                if sum > EPSILON {
                                    (v / sum) as f32
                                } else {
                                    (1.0 / n as f64) as f32
                                }
                            })
                            .collect::<Vec<_>>())
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map(|rows| rows.into_iter().flatten().collect())
            })
            .collect()
    }

    pub fn check_probabilities(&self, frozen: &[Vec<f32>]) -> Result<f64, String> {
        if frozen.len() != self.nodes.len() {
            return Err("incomplete exact-flop policy".into());
        }
        let mut error = 0.0f64;
        for (node, values) in self.nodes.iter().zip(frozen) {
            if let Kind::Decision { children, .. } = &node.kind {
                if values.len() != node.sums.len() {
                    return Err("exact-flop row dimensions disagree".into());
                }
                for row in values.chunks_exact(children.len()) {
                    let sum: f64 = row.iter().map(|v| f64::from(*v)).sum();
                    if row.iter().any(|v| !v.is_finite() || *v < 0.0) || (sum - 1.0).abs() > 1e-6 {
                        return Err("invalid frozen exact-flop probabilities".into());
                    }
                    error = error.max((sum - 1.0).abs());
                }
            } else if !values.is_empty() {
                return Err("policy row on a nondecision node".into());
            }
        }
        Ok(error)
    }

    pub fn probability(&self, frozen: &[Vec<f32>], id: usize, combo: usize, action: usize) -> f64 {
        let node = &self.nodes[id];
        let Kind::Decision { actor, children } = &node.kind else {
            panic!("not a decision");
        };
        let row = self.boards[node.board].rows[*actor][combo].expect("board-legal private hand");
        let values = &frozen[id][row * children.len()..(row + 1) * children.len()];
        f64::from(values[action]) / values.iter().map(|v| f64::from(*v)).sum::<f64>()
    }

    pub fn train(
        &mut self,
        rounds: u64,
        dcfr: &DcfrParameters,
        guard: &Guard,
    ) -> Result<(), String> {
        for round in 1..=rounds {
            self.walk(
                self.root,
                self.prior.clone(),
                0,
                round,
                rounds / 10,
                dcfr,
                guard,
            )?;
            self.walk(
                self.root,
                self.prior.clone(),
                1,
                round,
                rounds / 10,
                dcfr,
                guard,
            )?;
            guard.check()?;
        }
        Ok(())
    }

    fn walk(
        &mut self,
        id: usize,
        reaches: [Vec<f64>; 2],
        hero: usize,
        round: u64,
        delay: u64,
        dcfr: &DcfrParameters,
        guard: &Guard,
    ) -> Result<Vec<f64>, String> {
        guard.tick()?;
        let node = &self.nodes[id];
        let board = node.board;
        let mut value = vec![0.0; self.combos[hero].len()];
        match &node.kind {
            Kind::Terminal { terminal, invested } => {
                return Ok(self.terminal(board, terminal, *invested, &reaches[1 - hero], hero))
            }
            Kind::Chance { children, divisor } => {
                let (children, divisor) = (children.clone(), *divisor);
                for (card, child) in children {
                    let mut masked = reaches.clone();
                    for p in 0..2 {
                        for (i, c) in self.combos[p].iter().enumerate() {
                            if c.cards().contains(&card) {
                                masked[p][i] = 0.0;
                            }
                        }
                    }
                    let values = self.walk(child, masked, hero, round, delay, dcfr, guard)?;
                    for (v, child) in value.iter_mut().zip(values) {
                        *v += child / divisor;
                    }
                }
                return Ok(value);
            }
            Kind::Decision { .. } => {}
        }
        let Kind::Decision { actor, children } = &self.nodes[id].kind else {
            unreachable!()
        };
        let (actor, children) = (*actor, children.clone());
        let n = children.len();
        if actor == hero {
            let t = round as f64;
            let positive = t.powf(dcfr.positive_regret_exponent);
            let negative = t.powf(dcfr.negative_regret_exponent);
            for r in &mut self.nodes[id].regrets {
                *r *= if *r >= 0.0 {
                    positive / (positive + 1.0)
                } else {
                    negative / (negative + 1.0)
                };
            }
        }
        let strategy = self.strategy(id);
        let mut values = Vec::with_capacity(n);
        for (action, child) in children.iter().enumerate() {
            let mut next = reaches.clone();
            for (row, combo) in self.boards[board].legal[actor].iter().enumerate() {
                next[actor][*combo] *= strategy[row * n + action];
            }
            values.push(self.walk(*child, next, hero, round, delay, dcfr, guard)?);
        }
        for combo in &self.boards[board].legal[hero] {
            for action in 0..n {
                value[*combo] += values[action][*combo]
                    * if actor == hero {
                        strategy[self.boards[board].rows[actor][*combo].unwrap() * n + action]
                    } else {
                        1.0
                    };
            }
        }
        let node = &mut self.nodes[id];
        if actor == hero {
            for (row, combo) in self.boards[board].legal[actor].iter().enumerate() {
                for action in 0..n {
                    node.regrets[row * n + action] += values[action][*combo] - value[*combo];
                }
            }
        }
        // One average update after the first traverser's policy update, just
        // as in the existing joint turn solver. Own realization, not joint reach.
        if hero == 1 && round > delay {
            let factor = (round as f64 / (round as f64 + 1.0)).powf(dcfr.strategy_exponent);
            for (row, combo) in self.boards[board].legal[actor].iter().enumerate() {
                for action in 0..n {
                    node.sums[row * n + action] +=
                        reaches[actor][*combo] * strategy[row * n + action];
                    node.sums[row * n + action] *= factor;
                }
            }
        }
        Ok(value)
    }

    fn terminal(
        &self,
        board: usize,
        terminal: &Terminal,
        invested: [f64; 2],
        opponents: &[f64],
        hero: usize,
    ) -> Vec<f64> {
        let mut values = vec![0.0; self.combos[hero].len()];
        for own in &self.boards[board].legal[hero] {
            for other in &self.boards[board].legal[1 - hero] {
                let pair = if hero == 0 {
                    self.pair(*own, *other)
                } else {
                    self.pair(*other, *own)
                };
                if !self.compatible[pair] {
                    continue;
                }
                let utility = match terminal {
                    Terminal::Fold { winner } => {
                        if *winner == hero {
                            invested[1 - hero]
                        } else {
                            -invested[hero]
                        }
                    }
                    Terminal::Showdown => {
                        let sign = if hero == 0 { 1.0 } else { -1.0 };
                        sign * self.boards[board].equity[pair] * (invested[0] + invested[1]) / 2.0
                            + (invested[1 - hero] - invested[hero]) / 2.0
                    }
                };
                values[*own] += opponents[*other] * utility;
            }
        }
        values
    }

    pub fn profile_value(
        &self,
        frozen: &[Vec<f32>],
        hero: usize,
        guard: &Guard,
    ) -> Result<f64, String> {
        let values = self.profile_walk(self.root, self.prior.clone(), frozen, hero, guard)?;
        Ok(values
            .iter()
            .zip(&self.prior[hero])
            .map(|(v, w)| v * w)
            .sum::<f64>()
            / self.joint_mass())
    }

    fn profile_walk(
        &self,
        id: usize,
        reaches: [Vec<f64>; 2],
        frozen: &[Vec<f32>],
        hero: usize,
        guard: &Guard,
    ) -> Result<Vec<f64>, String> {
        guard.tick()?;
        let node = &self.nodes[id];
        let mut values = vec![0.0; self.combos[hero].len()];
        Ok(match &node.kind {
            Kind::Terminal { terminal, invested } => {
                self.terminal(node.board, terminal, *invested, &reaches[1 - hero], hero)
            }
            Kind::Chance { children, divisor } => {
                for (card, child) in children {
                    let mut masked = reaches.clone();
                    for p in 0..2 {
                        for (i, c) in self.combos[p].iter().enumerate() {
                            if c.cards().contains(card) {
                                masked[p][i] = 0.0;
                            }
                        }
                    }
                    for (v, c) in values
                        .iter_mut()
                        .zip(self.profile_walk(*child, masked, frozen, hero, guard)?)
                    {
                        *v += c / divisor;
                    }
                }
                values
            }
            Kind::Decision { actor, children } => {
                for (action, child) in children.iter().enumerate() {
                    let mut masked = reaches.clone();
                    for c in &self.boards[node.board].legal[*actor] {
                        masked[*actor][*c] *= self.probability(frozen, id, *c, action);
                    }
                    let child = self.profile_walk(*child, masked, frozen, hero, guard)?;
                    for c in &self.boards[node.board].legal[hero] {
                        values[*c] += child[*c]
                            * if *actor == hero {
                                self.probability(frozen, id, *c, action)
                            } else {
                                1.0
                            };
                    }
                }
                values
            }
        })
    }
}
