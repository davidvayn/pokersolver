use serde::Serialize;
use std::collections::BTreeMap;

const ACTIONS: usize = 2;

#[derive(Clone, Debug, Default)]
struct Node {
    regrets: [f64; ACTIONS],
    strategy_sum: [f64; ACTIONS],
}

impl Node {
    fn strategy(&self) -> [f64; ACTIONS] {
        let positive = [self.regrets[0].max(0.0), self.regrets[1].max(0.0)];
        let total = positive[0] + positive[1];
        if total > 0.0 {
            [positive[0] / total, positive[1] / total]
        } else {
            [0.5, 0.5]
        }
    }

    fn average_strategy(&self) -> [f64; ACTIONS] {
        let total = self.strategy_sum[0] + self.strategy_sum[1];
        if total > 0.0 {
            [self.strategy_sum[0] / total, self.strategy_sum[1] / total]
        } else {
            [0.5, 0.5]
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct KuhnResult {
    pub iterations: u64,
    pub expected_first_player_value: f64,
    pub known_equilibrium_value: f64,
    pub value_error: f64,
    pub first_player_best_response: f64,
    pub value_against_second_player_best_response: f64,
    pub nash_conv: f64,
    pub exploitability: f64,
    pub strategies: BTreeMap<String, [f64; 2]>,
}

pub fn solve(iterations: u64) -> KuhnResult {
    let mut nodes: BTreeMap<String, Node> = BTreeMap::new();
    for iteration in 1..=iterations {
        for first in 0..3u8 {
            for second in 0..3u8 {
                if first == second {
                    continue;
                }
                let cards = [first, second];
                cfr(&cards, "", 1.0, 1.0, iteration as f64, &mut nodes);
            }
        }
    }

    let strategies = nodes
        .iter()
        .map(|(key, node)| (key.clone(), node.average_strategy()))
        .collect::<BTreeMap<_, _>>();
    let expected = evaluate_profile(&strategies, None, None);

    let mut first_best = f64::NEG_INFINITY;
    let mut second_best = f64::INFINITY;
    for policy in 0..64u64 {
        first_best = first_best.max(evaluate_profile(&strategies, Some(policy), None));
        second_best = second_best.min(evaluate_profile(&strategies, None, Some(policy)));
    }
    let nash_conv = first_best - second_best;

    KuhnResult {
        iterations,
        expected_first_player_value: expected,
        known_equilibrium_value: -1.0 / 18.0,
        value_error: (expected + 1.0 / 18.0).abs(),
        first_player_best_response: first_best,
        value_against_second_player_best_response: second_best,
        nash_conv,
        exploitability: nash_conv / 2.0,
        strategies,
    }
}

fn cfr(
    cards: &[u8; 2],
    history: &str,
    reach_first: f64,
    reach_second: f64,
    weight: f64,
    nodes: &mut BTreeMap<String, Node>,
) -> f64 {
    if let Some(utility) = terminal_first_player_utility(cards, history) {
        return utility;
    }

    let player = acting_player(history);
    let key = info_set_key(player, cards[player], history);
    let strategy = nodes.entry(key.clone()).or_default().strategy();
    let mut action_utility = [0.0; ACTIONS];
    let mut node_utility = 0.0;

    for action in 0..ACTIONS {
        let next = next_history(history, action);
        action_utility[action] = if player == 0 {
            cfr(
                cards,
                next,
                reach_first * strategy[action],
                reach_second,
                weight,
                nodes,
            )
        } else {
            cfr(
                cards,
                next,
                reach_first,
                reach_second * strategy[action],
                weight,
                nodes,
            )
        };
        node_utility += strategy[action] * action_utility[action];
    }

    let node = nodes.get_mut(&key).expect("node inserted above");
    let own_reach = if player == 0 {
        reach_first
    } else {
        reach_second
    };
    let opponent_reach = if player == 0 {
        reach_second
    } else {
        reach_first
    };
    for action in 0..ACTIONS {
        let player_action_utility = if player == 0 {
            action_utility[action]
        } else {
            -action_utility[action]
        };
        let player_node_utility = if player == 0 {
            node_utility
        } else {
            -node_utility
        };
        node.regrets[action] = (node.regrets[action]
            + opponent_reach * (player_action_utility - player_node_utility))
            .max(0.0);
        node.strategy_sum[action] += weight * own_reach * strategy[action];
    }
    node_utility
}

fn acting_player(history: &str) -> usize {
    match history {
        "" | "pb" => 0,
        "p" | "b" => 1,
        _ => unreachable!("terminal histories have already returned"),
    }
}

fn info_set_key(player: usize, card: u8, history: &str) -> String {
    format!("p{player}:{}:{history}", card_name(card))
}

fn card_name(card: u8) -> char {
    match card {
        0 => 'J',
        1 => 'Q',
        2 => 'K',
        _ => unreachable!(),
    }
}

fn next_history(history: &str, action: usize) -> &'static str {
    match (history, action) {
        ("", 0) => "p",
        ("", 1) => "b",
        ("p", 0) => "pp",
        ("p", 1) => "pb",
        ("b", 0) => "bp",
        ("b", 1) => "bb",
        ("pb", 0) => "pbp",
        ("pb", 1) => "pbb",
        _ => unreachable!(),
    }
}

fn terminal_first_player_utility(cards: &[u8; 2], history: &str) -> Option<f64> {
    let first_wins = cards[0] > cards[1];
    match history {
        "pp" => Some(if first_wins { 1.0 } else { -1.0 }),
        "bp" => Some(1.0),
        "pbp" => Some(-1.0),
        "bb" | "pbb" => Some(if first_wins { 2.0 } else { -2.0 }),
        _ => None,
    }
}

fn evaluate_profile(
    strategies: &BTreeMap<String, [f64; 2]>,
    first_policy: Option<u64>,
    second_policy: Option<u64>,
) -> f64 {
    let mut total = 0.0;
    for first in 0..3u8 {
        for second in 0..3u8 {
            if first == second {
                continue;
            }
            total += evaluate_deal(
                &[first, second],
                "",
                strategies,
                first_policy,
                second_policy,
            ) / 6.0;
        }
    }
    total
}

fn evaluate_deal(
    cards: &[u8; 2],
    history: &str,
    strategies: &BTreeMap<String, [f64; 2]>,
    first_policy: Option<u64>,
    second_policy: Option<u64>,
) -> f64 {
    if let Some(utility) = terminal_first_player_utility(cards, history) {
        return utility;
    }
    let player = acting_player(history);
    let strategy = match if player == 0 {
        first_policy
    } else {
        second_policy
    } {
        Some(policy) => {
            let bit = policy_bit(player, cards[player], history);
            if policy & (1 << bit) == 0 {
                [1.0, 0.0]
            } else {
                [0.0, 1.0]
            }
        }
        None => strategies[&info_set_key(player, cards[player], history)],
    };
    (0..ACTIONS)
        .map(|action| {
            strategy[action]
                * evaluate_deal(
                    cards,
                    next_history(history, action),
                    strategies,
                    first_policy,
                    second_policy,
                )
        })
        .sum()
}

fn policy_bit(player: usize, card: u8, history: &str) -> u32 {
    let context = match (player, history) {
        (0, "") | (1, "p") => 0,
        (0, "pb") | (1, "b") => 1,
        _ => unreachable!(),
    };
    card as u32 * 2 + context
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converges_toward_known_kuhn_value() {
        let result = solve(20_000);
        assert!(
            result.value_error < 0.01,
            "value={}, error={}",
            result.expected_first_player_value,
            result.value_error
        );
        assert!(
            result.exploitability < 0.02,
            "exploitability={}",
            result.exploitability
        );
    }

    #[test]
    fn terminal_payoffs_use_net_ante_convention() {
        assert_eq!(terminal_first_player_utility(&[2, 0], "pp"), Some(1.0));
        assert_eq!(terminal_first_player_utility(&[2, 0], "pbb"), Some(2.0));
        assert_eq!(terminal_first_player_utility(&[0, 2], "bp"), Some(1.0));
        assert_eq!(terminal_first_player_utility(&[2, 0], "pbp"), Some(-1.0));
    }
}
