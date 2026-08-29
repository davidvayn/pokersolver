//! A small, exact two-player limit Leduc benchmark.
//!
//! The production blueprint is much larger and uses sampled chance/opponent
//! actions.  This module deliberately enumerates Leduc's 120 physical deals so
//! tests can catch regressions in information-set aggregation, regret updates,
//! average-policy export, and best-response evaluation against a tractable
//! imperfect-information game.

use std::collections::BTreeMap;

const DECK_SIZE: u8 = 6;
const MAX_AGGRESSIVE_ACTIONS: u8 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Action {
    Check,
    Bet,
    Fold,
    Call,
    Raise,
}

impl Action {
    fn code(self) -> char {
        match self {
            Self::Check => 'k',
            Self::Bet => 'b',
            Self::Fold => 'f',
            Self::Call => 'c',
            Self::Raise => 'r',
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Deal {
    private: [u8; 2],
    board: u8,
}

#[derive(Clone, Debug)]
struct State {
    round: u8,
    actor: usize,
    invested: [i32; 2],
    round_invested: [i32; 2],
    aggressive_actions: u8,
    checks: u8,
    history: String,
    terminal: Option<Terminal>,
}

#[derive(Clone, Copy, Debug)]
enum Terminal {
    Fold { winner: usize },
    Showdown,
}

impl State {
    fn initial() -> Self {
        Self {
            round: 0,
            actor: 0,
            invested: [1, 1],
            round_invested: [0, 0],
            aggressive_actions: 0,
            checks: 0,
            history: String::new(),
            terminal: None,
        }
    }

    fn legal_actions(&self) -> Vec<Action> {
        let to_call = self.to_call();
        if to_call > 0 {
            let mut actions = vec![Action::Fold, Action::Call];
            if self.aggressive_actions < MAX_AGGRESSIVE_ACTIONS {
                actions.push(Action::Raise);
            }
            actions
        } else {
            let mut actions = vec![Action::Check];
            if self.aggressive_actions < MAX_AGGRESSIVE_ACTIONS {
                actions.push(Action::Bet);
            }
            actions
        }
    }

    fn to_call(&self) -> i32 {
        self.round_invested[0].max(self.round_invested[1]) - self.round_invested[self.actor]
    }

    fn apply(&self, action: Action) -> Self {
        let mut next = self.clone();
        next.history.push(action.code());
        match action {
            Action::Fold => {
                next.terminal = Some(Terminal::Fold {
                    winner: 1 - self.actor,
                });
            }
            Action::Check => {
                debug_assert_eq!(self.to_call(), 0);
                next.checks += 1;
                if next.checks == 2 {
                    next.finish_round();
                } else {
                    next.actor = 1 - self.actor;
                }
            }
            Action::Call => {
                let payment = self.to_call();
                debug_assert!(payment > 0);
                next.invested[self.actor] += payment;
                next.round_invested[self.actor] += payment;
                next.finish_round();
            }
            Action::Bet | Action::Raise => {
                let highest = self.round_invested[0].max(self.round_invested[1]);
                let target = highest + self.bet_size();
                let payment = target - self.round_invested[self.actor];
                debug_assert!(payment > 0);
                next.invested[self.actor] += payment;
                next.round_invested[self.actor] = target;
                next.aggressive_actions += 1;
                next.checks = 0;
                next.actor = 1 - self.actor;
            }
        }
        next
    }

    fn bet_size(&self) -> i32 {
        if self.round == 0 {
            2
        } else {
            4
        }
    }

    fn finish_round(&mut self) {
        if self.round == 1 {
            self.terminal = Some(Terminal::Showdown);
            return;
        }
        self.round = 1;
        self.actor = 0;
        self.round_invested = [0, 0];
        self.aggressive_actions = 0;
        self.checks = 0;
        self.history.push('/');
    }

    fn utility_p0(&self, deal: Deal) -> f64 {
        match self.terminal.expect("terminal Leduc state") {
            Terminal::Fold { winner: 0 } => self.invested[1] as f64,
            Terminal::Fold { winner: 1 } => -(self.invested[0] as f64),
            Terminal::Fold { .. } => unreachable!(),
            Terminal::Showdown => match showdown_order(deal) {
                std::cmp::Ordering::Greater => self.invested[1] as f64,
                std::cmp::Ordering::Less => -(self.invested[0] as f64),
                std::cmp::Ordering::Equal => (self.invested[1] - self.invested[0]) as f64 / 2.0,
            },
        }
    }
}

fn rank(card: u8) -> u8 {
    card / 2
}

fn showdown_order(deal: Deal) -> std::cmp::Ordering {
    let board = rank(deal.board);
    let first = rank(deal.private[0]);
    let second = rank(deal.private[1]);
    let first_pair = first == board;
    let second_pair = second == board;
    first_pair
        .cmp(&second_pair)
        .then_with(|| first.cmp(&second))
}

fn all_deals() -> Vec<Deal> {
    let mut deals = Vec::with_capacity(120);
    for first in 0..DECK_SIZE {
        for second in 0..DECK_SIZE {
            if second == first {
                continue;
            }
            for board in 0..DECK_SIZE {
                if board != first && board != second {
                    deals.push(Deal {
                        private: [first, second],
                        board,
                    });
                }
            }
        }
    }
    debug_assert_eq!(deals.len(), 120);
    deals
}

fn information_set(state: &State, deal: Deal) -> String {
    let actor_rank = rank(deal.private[state.actor]);
    let board = if state.round == 0 {
        "-".to_owned()
    } else {
        rank(deal.board).to_string()
    };
    format!(
        "p{}:r{}:b{}:{}",
        state.actor, actor_rank, board, state.history
    )
}

#[derive(Clone, Debug)]
struct Node {
    actions: Vec<Action>,
    regrets: Vec<f64>,
    strategy_sum: Vec<f64>,
}

impl Node {
    fn new(actions: Vec<Action>) -> Self {
        Self {
            regrets: vec![0.0; actions.len()],
            strategy_sum: vec![0.0; actions.len()],
            actions,
        }
    }

    fn strategy(&self) -> Vec<f64> {
        normalize(self.regrets.iter().map(|regret| regret.max(0.0)).collect())
    }

    fn average_strategy(&self) -> Vec<f64> {
        normalize(self.strategy_sum.clone())
    }
}

fn normalize(mut values: Vec<f64>) -> Vec<f64> {
    let total = values.iter().sum::<f64>();
    if total > 1e-15 {
        for value in &mut values {
            *value /= total;
        }
    } else {
        let probability = 1.0 / values.len() as f64;
        values.fill(probability);
    }
    values
}

type AveragePolicy = BTreeMap<String, (Vec<Action>, Vec<f64>)>;

#[derive(Clone, Debug)]
pub struct LeducResult {
    pub iterations: u64,
    pub expected_first_player_value: f64,
    pub first_player_best_response: f64,
    pub second_player_best_response_utility: f64,
    pub nash_conv: f64,
    pub exploitability: f64,
    pub information_sets: usize,
}

pub fn solve(iterations: u64) -> LeducResult {
    let deals = all_deals();
    let mut nodes = BTreeMap::<String, Node>::new();
    for iteration in 1..=iterations {
        for deal in &deals {
            cfr(
                State::initial(),
                *deal,
                1.0,
                1.0,
                iteration as f64,
                &mut nodes,
            );
        }
    }
    let policy = nodes
        .iter()
        .map(|(key, node)| (key.clone(), (node.actions.clone(), node.average_strategy())))
        .collect::<AveragePolicy>();
    let expected_first_player_value = evaluate_profile(&deals, &policy);
    let first_player_best_response = best_response(&deals, &policy, 0);
    let second_player_best_response_utility = best_response(&deals, &policy, 1);
    let nash_conv = first_player_best_response + second_player_best_response_utility;
    LeducResult {
        iterations,
        expected_first_player_value,
        first_player_best_response,
        second_player_best_response_utility,
        nash_conv,
        exploitability: nash_conv / 2.0,
        information_sets: nodes.len(),
    }
}

fn cfr(
    state: State,
    deal: Deal,
    reach_first: f64,
    reach_second: f64,
    average_weight: f64,
    nodes: &mut BTreeMap<String, Node>,
) -> f64 {
    if state.terminal.is_some() {
        return state.utility_p0(deal);
    }
    let actor = state.actor;
    let key = information_set(&state, deal);
    let legal = state.legal_actions();
    let strategy = {
        let node = nodes
            .entry(key.clone())
            .or_insert_with(|| Node::new(legal.clone()));
        assert_eq!(node.actions, legal, "Leduc information-set action mismatch");
        node.strategy()
    };
    let mut action_values = Vec::with_capacity(legal.len());
    for (index, action) in legal.iter().enumerate() {
        action_values.push(if actor == 0 {
            cfr(
                state.apply(*action),
                deal,
                reach_first * strategy[index],
                reach_second,
                average_weight,
                nodes,
            )
        } else {
            cfr(
                state.apply(*action),
                deal,
                reach_first,
                reach_second * strategy[index],
                average_weight,
                nodes,
            )
        });
    }
    let node_value = strategy
        .iter()
        .zip(&action_values)
        .map(|(probability, value)| probability * value)
        .sum::<f64>();
    let opponent_reach = if actor == 0 {
        reach_second
    } else {
        reach_first
    };
    let own_reach = if actor == 0 {
        reach_first
    } else {
        reach_second
    };
    let node = nodes.get_mut(&key).expect("Leduc node inserted");
    for index in 0..legal.len() {
        let regret = if actor == 0 {
            action_values[index] - node_value
        } else {
            node_value - action_values[index]
        };
        node.regrets[index] = (node.regrets[index] + opponent_reach * regret).max(0.0);
        node.strategy_sum[index] += average_weight * own_reach * strategy[index];
    }
    node_value
}

fn evaluate_profile(deals: &[Deal], policy: &AveragePolicy) -> f64 {
    deals
        .iter()
        .map(|deal| evaluate_state(State::initial(), *deal, policy))
        .sum::<f64>()
        / deals.len() as f64
}

fn evaluate_state(state: State, deal: Deal, policy: &AveragePolicy) -> f64 {
    if state.terminal.is_some() {
        return state.utility_p0(deal);
    }
    let key = information_set(&state, deal);
    let (actions, strategy) = &policy[&key];
    actions
        .iter()
        .zip(strategy)
        .map(|(action, probability)| {
            probability * evaluate_state(state.apply(*action), deal, policy)
        })
        .sum()
}

#[derive(Clone)]
struct WeightedWorld {
    state: State,
    deal: Deal,
    weight: f64,
}

fn best_response(deals: &[Deal], policy: &AveragePolicy, responder: usize) -> f64 {
    let probability = 1.0 / deals.len() as f64;
    let worlds = deals
        .iter()
        .map(|deal| WeightedWorld {
            state: State::initial(),
            deal: *deal,
            weight: probability,
        })
        .collect();
    best_response_worlds(worlds, policy, responder)
}

fn best_response_worlds(
    worlds: Vec<WeightedWorld>,
    policy: &AveragePolicy,
    responder: usize,
) -> f64 {
    if worlds.is_empty() {
        return 0.0;
    }
    if worlds[0].state.terminal.is_some() {
        return worlds
            .iter()
            .map(|world| {
                let utility = world.state.utility_p0(world.deal);
                world.weight * if responder == 0 { utility } else { -utility }
            })
            .sum();
    }
    let actor = worlds[0].state.actor;
    debug_assert!(worlds.iter().all(|world| world.state.actor == actor));
    if actor == responder {
        let mut information_sets = BTreeMap::<String, Vec<WeightedWorld>>::new();
        for world in worlds {
            information_sets
                .entry(information_set(&world.state, world.deal))
                .or_default()
                .push(world);
        }
        information_sets
            .into_values()
            .map(|group| {
                let actions = group[0].state.legal_actions();
                actions
                    .iter()
                    .map(|action| {
                        best_response_worlds(
                            group
                                .iter()
                                .map(|world| WeightedWorld {
                                    state: world.state.apply(*action),
                                    deal: world.deal,
                                    weight: world.weight,
                                })
                                .collect(),
                            policy,
                            responder,
                        )
                    })
                    .max_by(f64::total_cmp)
                    .expect("responder has a legal Leduc action")
            })
            .sum()
    } else {
        let actions = worlds[0].state.legal_actions();
        actions
            .iter()
            .enumerate()
            .map(|(action_index, action)| {
                let branch = worlds
                    .iter()
                    .filter_map(|world| {
                        let key = information_set(&world.state, world.deal);
                        let (policy_actions, strategy) = &policy[&key];
                        debug_assert_eq!(policy_actions, &actions);
                        let probability = strategy[action_index];
                        (probability > 0.0).then(|| WeightedWorld {
                            state: world.state.apply(*action),
                            deal: world.deal,
                            weight: world.weight * probability,
                        })
                    })
                    .collect();
                best_response_worlds(branch, policy, responder)
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converges_in_two_round_leduc() {
        let result = solve(2_000);
        assert_eq!(result.information_sets, 288);
        assert!(
            result.exploitability < 0.08,
            "value={}, nash_conv={}, exploitability={}",
            result.expected_first_player_value,
            result.nash_conv,
            result.exploitability
        );
    }

    #[test]
    fn terminal_values_use_net_contributions_and_pairs_beat_high_cards() {
        let first_pairs = Deal {
            private: [0, 4],
            board: 1,
        };
        assert_eq!(showdown_order(first_pairs), std::cmp::Ordering::Greater);
        let mut state = State::initial();
        state.invested = [3, 3];
        state.terminal = Some(Terminal::Showdown);
        assert_eq!(state.utility_p0(first_pairs), 3.0);
        state.terminal = Some(Terminal::Fold { winner: 1 });
        assert_eq!(state.utility_p0(first_pairs), -3.0);
    }
}
