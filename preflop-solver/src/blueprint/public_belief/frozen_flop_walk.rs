//! Shared immutable-policy traversal. Leaf adapters vary; regret and average
//! updates keep one implementation for the neural and native continuation pilots.
use super::*;
type Ranges = [Vec<f64>; 2];

pub(super) struct Walker<'a> {
    pub game: &'a BlueprintConfig,
    pub legal: &'a [Vec<bool>; 2],
    pub nodes: &'a BTreeMap<Vec<String>, RangeNode>,
    pub terminal: &'a dyn Fn(&GameState, &Ranges) -> Ranges,
    pub turn: &'a dyn Fn(&GameState, &Ranges, Option<usize>) -> Ranges,
}

impl Walker<'_> {
    pub(super) fn walk(
        &self,
        state: GameState,
        reaches: [Vec<f64>; 2],
        traverser: Option<usize>,
        accumulate_average: bool,
        deltas: &mut BTreeMap<Vec<String>, FrozenRangeNodeDelta>,
    ) -> [Vec<f64>; 2] {
        if state.street == Street::Turn && state.terminal.is_none() {
            return (self.turn)(&state, &reaches, traverser);
        }
        if state.terminal.is_some() {
            return (self.terminal)(&state, &reaches);
        }
        let actions = state.legal_actions(self.game);
        let key = state.public_history.clone();
        let actor = state.actor;
        let node = self
            .nodes
            .get(&key)
            .expect("frozen public tree contains every reachable flop node");
        let strategy = node.strategy(&self.legal[actor]);
        let action_count = actions.len();
        let mut children = Vec::with_capacity(action_count);
        for (action_index, action) in actions.iter().enumerate() {
            let mut child_reaches = reaches.clone();
            for combo in 0..COMBO_COUNT {
                child_reaches[actor][combo] *= strategy[combo * action_count + action_index];
            }
            children.push(self.walk(
                state.apply(action, self.game),
                child_reaches,
                traverser,
                accumulate_average,
                deltas,
            ));
        }
        let opponent = 1 - actor;
        let mut values = [vec![0.0; COMBO_COUNT], vec![0.0; COMBO_COUNT]];
        for combo in 0..COMBO_COUNT {
            for action in 0..action_count {
                values[actor][combo] +=
                    strategy[combo * action_count + action] * children[action][actor][combo];
                values[opponent][combo] += children[action][opponent][combo];
            }
        }
        let update_actor = traverser.is_none_or(|traverser| actor == traverser);
        if update_actor || accumulate_average {
            let delta = deltas
                .entry(key)
                .or_insert_with(|| FrozenRangeNodeDelta::new(node.regrets.len()));
            for combo in 0..COMBO_COUNT {
                if !self.legal[actor][combo] {
                    continue;
                }
                let offset = combo * action_count;
                if update_actor {
                    for action in 0..action_count {
                        delta.regrets[offset + action] +=
                            children[action][actor][combo] - values[actor][combo];
                    }
                }
                if accumulate_average {
                    for action in 0..action_count {
                        delta.strategy_sum[offset + action] +=
                            reaches[actor][combo] * strategy[offset + action];
                    }
                }
            }
        }
        values
    }
}
