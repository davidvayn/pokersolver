//! Scalar information-set response, independent of training's vector terminal
//! cache. It enumerates private opponents before maximizing an own action.
use super::*;

struct Checker<'a> {
    game: &'a Game,
    frozen: &'a [Vec<f32>],
    equities: BTreeMap<(usize, usize, usize), f64>,
    guard: &'a Guard,
}

pub(super) fn verify(
    game: &Game,
    frozen: &[Vec<f32>],
    guard: &Guard,
) -> Result<([f64; 2], [f64; 2]), String> {
    game.check_probabilities(frozen)?;
    let mut checker = Checker {
        game,
        frozen,
        equities: BTreeMap::new(),
        guard,
    };
    let mut profile = [0.0; 2];
    let mut best = [0.0; 2];
    for hero in 0..2 {
        for own in 0..game.combos[hero].len() {
            let opponents = game.prior[1 - hero]
                .iter()
                .enumerate()
                .map(|(other, w)| {
                    let pair = if hero == 0 {
                        game.pair(own, other)
                    } else {
                        game.pair(other, own)
                    };
                    if game.compatible[pair] {
                        *w
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>();
            profile[hero] +=
                game.prior[hero][own] * checker.walk(game.root, hero, own, &opponents, false)?;
            best[hero] +=
                game.prior[hero][own] * checker.walk(game.root, hero, own, &opponents, true)?;
        }
        profile[hero] /= game.joint_mass();
        best[hero] /= game.joint_mass();
    }
    Ok((profile, best))
}

impl Checker<'_> {
    fn equity(&mut self, board: usize, a: usize, b: usize) -> Result<f64, String> {
        if let Some(value) = self.equities.get(&(board, a, b)) {
            return Ok(*value);
        }
        let cards = &self.game.boards[board].cards;
        let left = self.game.combos[0][a].cards();
        let right = self.game.combos[1][b].cards();
        let remaining = (0..52u8)
            .filter(|c| !cards.contains(c) && !left.contains(c) && !right.contains(c))
            .collect::<Vec<_>>();
        let mut sum = 0.0;
        let mut count = 0;
        let mut settle = |future: &[u8]| {
            let mut x = cards.clone();
            x.extend(future);
            x.extend(left);
            let mut y = cards.clone();
            y.extend(future);
            y.extend(right);
            sum += match evaluate(&x).cmp(&evaluate(&y)) {
                std::cmp::Ordering::Greater => 1.0,
                std::cmp::Ordering::Less => -1.0,
                std::cmp::Ordering::Equal => 0.0,
            };
            count += 1;
        };
        match cards.len() {
            5 => settle(&[]),
            4 => {
                for card in &remaining {
                    settle(&[*card]);
                }
            }
            3 => {
                for i in 0..remaining.len() {
                    for j in i + 1..remaining.len() {
                        settle(&[remaining[i], remaining[j]]);
                    }
                }
            }
            _ => return Err("invalid checker board".into()),
        }
        let expected = match cards.len() {
            3 => 990,
            4 => 44,
            _ => 1,
        };
        if count != expected {
            return Err("independent all-in card-removal count disagreed".into());
        }
        self.guard.check()?;
        let equity = sum / count as f64;
        self.equities.insert((board, a, b), equity);
        Ok(equity)
    }

    fn walk(
        &mut self,
        id: usize,
        hero: usize,
        own: usize,
        opponents: &[f64],
        best: bool,
    ) -> Result<f64, String> {
        self.guard.tick()?;
        if opponents.iter().all(|w| *w == 0.0) {
            return Ok(0.0);
        }
        let node = &self.game.nodes[id];
        match &node.kind {
            Kind::Terminal { terminal, invested } => {
                let mut value = 0.0;
                for (other, w) in opponents.iter().enumerate().filter(|(_, w)| **w > 0.0) {
                    let utility = match terminal {
                        Terminal::Fold { winner } => {
                            if *winner == hero {
                                invested[1 - hero]
                            } else {
                                -invested[hero]
                            }
                        }
                        Terminal::Showdown => {
                            let (a, b, sign) = if hero == 0 {
                                (own, other, 1.0)
                            } else {
                                (other, own, -1.0)
                            };
                            sign * self.equity(node.board, a, b)? * (invested[0] + invested[1])
                                / 2.0
                                + (invested[1 - hero] - invested[hero]) / 2.0
                        }
                    };
                    value += w * utility;
                }
                Ok(value)
            }
            Kind::Chance { children, divisor } => {
                let mut value = 0.0;
                for (card, child) in children {
                    if self.game.combos[hero][own].cards().contains(card) {
                        continue;
                    }
                    let masked = opponents
                        .iter()
                        .enumerate()
                        .map(|(i, w)| {
                            if self.game.combos[1 - hero][i].cards().contains(card) {
                                0.0
                            } else {
                                w / divisor
                            }
                        })
                        .collect::<Vec<_>>();
                    value += self.walk(*child, hero, own, &masked, best)?;
                }
                Ok(value)
            }
            Kind::Decision { actor, children } => {
                let mut values = Vec::new();
                for (action, child) in children.iter().enumerate() {
                    let masked = if *actor == hero {
                        opponents.to_vec()
                    } else {
                        opponents
                            .iter()
                            .enumerate()
                            .map(|(i, w)| {
                                if *w == 0.0 {
                                    0.0
                                } else {
                                    w * self.game.probability(self.frozen, id, i, action)
                                }
                            })
                            .collect()
                    };
                    values.push(self.walk(*child, hero, own, &masked, best)?);
                }
                Ok(if *actor != hero {
                    values.iter().sum()
                } else if best {
                    values.into_iter().fold(f64::NEG_INFINITY, f64::max)
                } else {
                    values
                        .iter()
                        .enumerate()
                        .map(|(a, v)| v * self.game.probability(self.frozen, id, own, a))
                        .sum()
                })
            }
        }
    }
}
