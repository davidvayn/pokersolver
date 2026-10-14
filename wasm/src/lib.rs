//! Single-street postflop solver using vectorized CFR+ or full-width
//! extensive-form fictitious play (XFP) with an all-in-equity terminal model.
//!
//! Model: OOP and IP act on the current street (flop or turn) with a
//! discretized bet/raise tree. Any line ending in a call or check-check goes to
//! an "all-in" showdown whose value is the two hands' equity over the remaining
//! runout — precomputed once into an equity matrix that is independent of
//! strategy. This is a legitimate, well-known simplification that yields real
//! GTO betting/checking/calling/folding frequencies and EVs, and converges fast
//! (CFR+). Exploitability is reported so convergence is observable.

mod eval;
use eval::evaluate;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

const RANKS: &[u8; 13] = b"23456789TJQKA";
const MAX_RAISES: u32 = 3;

#[derive(Clone, Copy, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum Algorithm {
    CfrPlus,
    FictitiousPlay,
}

fn default_algorithm() -> Algorithm {
    Algorithm::CfrPlus
}

#[derive(Deserialize)]
struct Input {
    board: Vec<u8>,
    oop: Vec<(u8, u8, f64)>,
    ip: Vec<(u8, u8, f64)>,
    pot: f64,
    stack: f64,
    bet_sizes: Vec<f64>,
    raise_sizes: Vec<f64>,
    iterations: u32,
    #[serde(default = "default_algorithm")]
    algorithm: Algorithm,
    #[serde(default = "default_max_combos")]
    max_combos: usize,
}
fn default_max_combos() -> usize {
    200
}

#[derive(Serialize, Clone)]
struct ActionStrategy {
    action: String,
    freq: f64,
    ev: f64,
}
#[derive(Serialize, Clone)]
struct ComboStrategy {
    card0: u8,
    card1: u8,
    weight: f64,
    actions: Vec<ActionStrategy>,
    ev: f64,
    equity: f64,
}
#[derive(Serialize)]
struct ClassRow {
    class: String,
    combos: f64,
    actions: Vec<ActionStrategy>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    combos_data: Vec<ComboStrategy>,
}
#[derive(Serialize)]
struct NodeStrategy {
    title: String,
    actions: Vec<String>,
    rows: Vec<ClassRow>,
}

#[derive(Serialize)]
struct ActionTree {
    root: usize,
    max_raises: u32,
    nodes: Vec<ActionTreeNode>,
}

#[derive(Serialize)]
struct ActionTreeNode {
    id: usize,
    kind: &'static str,
    player: Option<&'static str>,
    pot: f64,
    stacks: [f64; 2],
    committed: [f64; 2],
    to_call: f64,
    reach_probability: f64,
    children: Vec<usize>,
    strategy: Option<NodeStrategy>,
    reason: Option<&'static str>,
    winner: Option<&'static str>,
}

fn player_label(player: usize) -> &'static str {
    if player == 0 {
        "oop"
    } else {
        "ip"
    }
}
#[derive(Serialize)]
struct CfrComparison {
    cfr_iterations: u32,
    fp_oop_vs_cfr_ip_ev: f64,
    fp_ip_vs_cfr_oop_ev: f64,
}
#[derive(Serialize)]
struct Output {
    algorithm: String,
    iterations: u32,
    exploitability_pct: f64,
    oop_ev: f64,
    ip_ev: f64,
    pot: f64,
    oop_combos: f64,
    ip_combos: f64,
    truncated: bool,
    oop: NodeStrategy,
    ip: NodeStrategy,
    action_tree: ActionTree,
    exploitability_history: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cfr_comparison: Option<CfrComparison>,
}

// ---------------------------------------------------------------------------
// Game tree
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum TermKind {
    Showdown,
    Fold(usize), // player who folded
}

#[derive(Clone)]
struct Terminal {
    kind: TermKind,
    dead_pot: f64,
    oop_paid: f64,
    ip_paid: f64,
}

enum Node {
    Decision {
        player: usize,
        labels: Vec<String>,
        children: Vec<usize>,
        state: BState,
    },
    Terminal(Terminal),
}

#[derive(Clone)]
struct BState {
    to_act: usize,
    dead_pot: f64,
    oop_paid: f64,
    ip_paid: f64,
    facing: bool,
    raises: u32,
    prev_check: bool,
}

struct Tree {
    nodes: Vec<Node>,
    bet_sizes: Vec<f64>,
    raise_sizes: Vec<f64>,
    stack: f64,
}

impl Tree {
    fn build(&mut self, s: BState) -> usize {
        let (my_paid, opp_paid) = if s.to_act == 0 {
            (s.oop_paid, s.ip_paid)
        } else {
            (s.ip_paid, s.oop_paid)
        };
        let to_call = opp_paid - my_paid;
        let cur_pot = s.dead_pot + s.oop_paid + s.ip_paid;
        let my_committed = my_paid;
        let my_remaining = self.stack - my_committed;

        let mut labels: Vec<String> = Vec::new();
        let mut children: Vec<usize> = Vec::new();

        if s.facing && to_call > 1e-9 {
            // Fold
            labels.push("Fold".into());
            children.push(self.terminal(Terminal {
                kind: TermKind::Fold(s.to_act),
                dead_pot: s.dead_pot,
                oop_paid: s.oop_paid,
                ip_paid: s.ip_paid,
            }));

            // Call (matches, possibly all-in for min) -> showdown
            let call_amt = to_call.min(my_remaining);
            let mut cs = s.clone();
            add_paid(&mut cs, s.to_act, call_amt);
            labels.push("Call".into());
            children.push(self.terminal(Terminal {
                kind: TermKind::Showdown,
                dead_pot: cs.dead_pot,
                oop_paid: cs.oop_paid,
                ip_paid: cs.ip_paid,
            }));

            // Raises
            if s.raises < MAX_RAISES && my_remaining > to_call + 1e-9 {
                let pot_after_call = cur_pot + to_call;
                let sizes = self.raise_sizes.clone();
                let mut added: Vec<f64> = Vec::new();
                for r in sizes {
                    let raise_extra = r * pot_after_call;
                    let total = (to_call + raise_extra).min(my_remaining);
                    if total <= to_call + 1e-9 {
                        continue;
                    }
                    // Skip sizes that collapse to the same amount (e.g. multiple
                    // sizes that all cap to the same all-in).
                    if added.iter().any(|&a| (a - total).abs() < 1e-6) {
                        continue;
                    }
                    added.push(total);
                    let mut rs = s.clone();
                    add_paid(&mut rs, s.to_act, total);
                    rs.to_act = 1 - s.to_act;
                    rs.facing = true;
                    rs.raises = s.raises + 1;
                    rs.prev_check = false;
                    let all_in = total >= my_remaining - 1e-9;
                    let pct = (r * 100.0).round() as i64;
                    labels.push(if all_in {
                        "Raise all-in".into()
                    } else {
                        format!("Raise {}%", pct)
                    });
                    children.push(self.build(rs));
                }
            }
        } else {
            // No bet to face: Check or Bet
            // Check
            if s.prev_check {
                // second check -> showdown
                labels.push("Check".into());
                children.push(self.terminal(Terminal {
                    kind: TermKind::Showdown,
                    dead_pot: s.dead_pot,
                    oop_paid: s.oop_paid,
                    ip_paid: s.ip_paid,
                }));
            } else {
                let mut cs = s.clone();
                cs.to_act = 1 - s.to_act;
                cs.prev_check = true;
                cs.facing = false;
                labels.push("Check".into());
                children.push(self.build(cs));
            }

            // Bets
            if my_remaining > 1e-9 {
                let sizes = self.bet_sizes.clone();
                let mut added: Vec<f64> = Vec::new();
                for b in sizes {
                    let amt = (b * cur_pot).min(my_remaining);
                    if amt < 1e-9 {
                        continue;
                    }
                    // Skip sizes that collapse to the same amount (e.g. multiple
                    // sizes that all cap to the same all-in).
                    if added.iter().any(|&a| (a - amt).abs() < 1e-6) {
                        continue;
                    }
                    added.push(amt);
                    let mut bs = s.clone();
                    add_paid(&mut bs, s.to_act, amt);
                    bs.to_act = 1 - s.to_act;
                    bs.facing = true;
                    bs.prev_check = false;
                    let all_in = amt >= my_remaining - 1e-9;
                    let pct = (b * 100.0).round() as i64;
                    labels.push(if all_in {
                        "Bet all-in".into()
                    } else {
                        format!("Bet {}%", pct)
                    });
                    children.push(self.build(bs));
                }
            }
        }

        let id = self.nodes.len();
        self.nodes.push(Node::Decision {
            player: s.to_act,
            labels,
            children,
            state: s,
        });
        id
    }

    fn terminal(&mut self, t: Terminal) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node::Terminal(t));
        id
    }
}

fn add_paid(s: &mut BState, player: usize, amt: f64) {
    if player == 0 {
        s.oop_paid += amt;
    } else {
        s.ip_paid += amt;
    }
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

struct Solver {
    tree: Tree,
    n_oop: usize,
    n_ip: usize,
    oop_cards: Vec<(u8, u8)>,
    ip_cards: Vec<(u8, u8)>,
    oop_prob: Vec<f64>,
    ip_prob: Vec<f64>,
    oop_combos: f64,
    ip_combos: f64,
    // eq[i*n_ip + j] = OOP hand i equity share (win + tie/2) vs IP hand j
    eq: Vec<f32>,
    // compat[i*n_ip + j] = hands share no card
    compat: Vec<bool>,
    // Probability mass of compatible private-hand pairs before normalization.
    joint_mass: f64,
    // per decision node: regret and strategy sums, indexed by [hand*na + a]
    regret: Vec<Vec<f64>>,
    strat_sum: Vec<Vec<f64>>,
}

impl Solver {
    fn current_strategy(&self, node: usize, n: usize, na: usize) -> Vec<f64> {
        // regret matching (CFR+): positive part normalized
        let r = &self.regret[node];
        let mut strat = vec![0.0f64; n * na];
        for i in 0..n {
            let base = i * na;
            let mut sum = 0.0;
            for a in 0..na {
                let v = r[base + a].max(0.0);
                strat[base + a] = v;
                sum += v;
            }
            if sum > 1e-12 {
                for a in 0..na {
                    strat[base + a] /= sum;
                }
            } else {
                for a in 0..na {
                    strat[base + a] = 1.0 / na as f64;
                }
            }
        }
        strat
    }

    fn average_strategy(&self, node: usize, n: usize, na: usize) -> Vec<f64> {
        let s = &self.strat_sum[node];
        let mut strat = vec![0.0f64; n * na];
        for i in 0..n {
            let base = i * na;
            let mut sum = 0.0;
            for a in 0..na {
                sum += s[base + a];
            }
            if sum > 1e-12 {
                for a in 0..na {
                    strat[base + a] = s[base + a] / sum;
                }
            } else {
                for a in 0..na {
                    strat[base + a] = 1.0 / na as f64;
                }
            }
        }
        strat
    }

    fn average_profile(&self) -> Vec<Vec<f64>> {
        self.tree
            .nodes
            .iter()
            .enumerate()
            .map(|(node, value)| match value {
                Node::Decision {
                    player, children, ..
                } => {
                    let n = if *player == 0 { self.n_oop } else { self.n_ip };
                    self.average_strategy(node, n, children.len())
                }
                Node::Terminal(_) => Vec::new(),
            })
            .collect()
    }

    fn profile_strategy(profile: &[Vec<f64>], node: usize, n: usize, na: usize) -> &[f64] {
        let strategy = &profile[node];
        debug_assert_eq!(strategy.len(), n * na);
        strategy
    }

    /// Terminal value vector for `player`, weighted by opponent reach.
    fn terminal_value(&self, t: &Terminal, player: usize, opp_reach: &[f64]) -> Vec<f64> {
        let final_pot = t.dead_pot + t.oop_paid + t.ip_paid;
        if player == 0 {
            let n = self.n_oop;
            let mut v = vec![0.0f64; n];
            for i in 0..n {
                let mut mass = 0.0;
                let mut eqsum = 0.0;
                let base = i * self.n_ip;
                for j in 0..self.n_ip {
                    if !self.compat[base + j] {
                        continue;
                    }
                    let rj = opp_reach[j];
                    if rj == 0.0 {
                        continue;
                    }
                    mass += rj;
                    match t.kind {
                        TermKind::Showdown => {
                            eqsum += rj * self.eq[base + j] as f64;
                        }
                        _ => {}
                    }
                }
                v[i] = match t.kind {
                    TermKind::Showdown => final_pot * eqsum - t.oop_paid * mass,
                    TermKind::Fold(f) => {
                        if f == 1 {
                            // IP folded, OOP wins
                            (t.dead_pot + t.ip_paid) * mass
                        } else {
                            // OOP folded
                            -t.oop_paid * mass
                        }
                    }
                };
            }
            v
        } else {
            let n = self.n_ip;
            let mut v = vec![0.0f64; n];
            for j in 0..n {
                let mut mass = 0.0;
                let mut eqsum = 0.0;
                for i in 0..self.n_oop {
                    let idx = i * self.n_ip + j;
                    if !self.compat[idx] {
                        continue;
                    }
                    let ri = opp_reach[i];
                    if ri == 0.0 {
                        continue;
                    }
                    mass += ri;
                    match t.kind {
                        TermKind::Showdown => {
                            eqsum += ri * (1.0 - self.eq[idx] as f64);
                        }
                        _ => {}
                    }
                }
                v[j] = match t.kind {
                    TermKind::Showdown => final_pot * eqsum - t.ip_paid * mass,
                    TermKind::Fold(f) => {
                        if f == 0 {
                            (t.dead_pot + t.oop_paid) * mass
                        } else {
                            -t.ip_paid * mass
                        }
                    }
                };
            }
            v
        }
    }

    /// Compute an exact deterministic best response at every information set
    /// for `player` against `profile`.
    fn best_response_profile(
        &self,
        root: usize,
        player: usize,
        profile: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        let mut response: Vec<Vec<f64>> = self
            .strat_sum
            .iter()
            .map(|row| vec![0.0; row.len()])
            .collect();
        let opponent_reach = if player == 0 {
            &self.ip_prob
        } else {
            &self.oop_prob
        };
        self.best_response_value(root, player, opponent_reach, profile, &mut response);
        response
    }

    fn best_response_value(
        &self,
        node: usize,
        player: usize,
        opp_reach: &[f64],
        profile: &[Vec<f64>],
        response: &mut [Vec<f64>],
    ) -> Vec<f64> {
        match &self.tree.nodes[node] {
            Node::Terminal(t) => self.terminal_value(t, player, opp_reach),
            Node::Decision {
                player: decision_player,
                children,
                ..
            } => {
                let decision_player = *decision_player;
                let children = children.clone();
                let na = children.len();
                if decision_player == player {
                    let n = if player == 0 { self.n_oop } else { self.n_ip };
                    let action_values: Vec<Vec<f64>> = children
                        .iter()
                        .map(|child| {
                            self.best_response_value(*child, player, opp_reach, profile, response)
                        })
                        .collect();
                    let mut value = vec![0.0; n];
                    for hand in 0..n {
                        let mut best_action = 0;
                        let mut best_value = f64::NEG_INFINITY;
                        for action in 0..na {
                            if action_values[action][hand] > best_value {
                                best_action = action;
                                best_value = action_values[action][hand];
                            }
                        }
                        response[node][hand * na + best_action] = 1.0;
                        value[hand] = best_value;
                    }
                    value
                } else {
                    let n_opp = opp_reach.len();
                    let strategy = Self::profile_strategy(profile, node, n_opp, na);
                    let n_player = if player == 0 { self.n_oop } else { self.n_ip };
                    let mut value = vec![0.0; n_player];
                    for action in 0..na {
                        let child_opp_reach: Vec<f64> = (0..n_opp)
                            .map(|hand| opp_reach[hand] * strategy[hand * na + action])
                            .collect();
                        let child_value = self.best_response_value(
                            children[action],
                            player,
                            &child_opp_reach,
                            profile,
                            response,
                        );
                        for hand in 0..n_player {
                            value[hand] += child_value[hand];
                        }
                    }
                    value
                }
            }
        }
    }

    /// Apply one XFP average-policy update for one player. The reach-corrected
    /// mixture is Theorem 7 of Heinrich, Lanctot & Silver (2015):
    /// it averages realization plans, not behavior probabilities directly.
    fn xfp_update_player(
        &self,
        root: usize,
        player: usize,
        old_profile: &[Vec<f64>],
        best_response: &[Vec<f64>],
        old_count: f64,
        next_profile: &mut [Vec<f64>],
    ) {
        let n = if player == 0 { self.n_oop } else { self.n_ip };
        let reach = vec![1.0; n];
        self.xfp_update_node(
            root,
            player,
            old_profile,
            best_response,
            old_count,
            &reach,
            &reach,
            next_profile,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn xfp_update_node(
        &self,
        node: usize,
        player: usize,
        old_profile: &[Vec<f64>],
        best_response: &[Vec<f64>],
        old_count: f64,
        old_reach: &[f64],
        response_reach: &[f64],
        next_profile: &mut [Vec<f64>],
    ) {
        let Node::Decision {
            player: decision_player,
            children,
            ..
        } = &self.tree.nodes[node]
        else {
            return;
        };
        let decision_player = *decision_player;
        let children = children.clone();

        if decision_player == player {
            let na = children.len();
            let old_strategy = &old_profile[node];
            let response_strategy = &best_response[node];
            for hand in 0..old_reach.len() {
                let denominator = old_count * old_reach[hand] + response_reach[hand];
                if denominator <= 1e-15 {
                    continue;
                }
                let base = hand * na;
                for action in 0..na {
                    next_profile[node][base + action] = xfp_probability(
                        old_strategy[base + action],
                        response_strategy[base + action],
                        old_reach[hand],
                        response_reach[hand],
                        old_count,
                    );
                }
            }

            for action in 0..na {
                let child_old_reach: Vec<f64> = (0..old_reach.len())
                    .map(|hand| old_reach[hand] * old_strategy[hand * na + action])
                    .collect();
                let child_response_reach: Vec<f64> = (0..response_reach.len())
                    .map(|hand| response_reach[hand] * response_strategy[hand * na + action])
                    .collect();
                self.xfp_update_node(
                    children[action],
                    player,
                    old_profile,
                    best_response,
                    old_count,
                    &child_old_reach,
                    &child_response_reach,
                    next_profile,
                );
            }
        } else {
            // Opponent actions do not change this player's realization reach.
            for child in children {
                self.xfp_update_node(
                    child,
                    player,
                    old_profile,
                    best_response,
                    old_count,
                    old_reach,
                    response_reach,
                    next_profile,
                );
            }
        }
    }

    /// CFR+ traversal updating `player`. Returns player's counterfactual value.
    fn cfr(
        &mut self,
        node: usize,
        player: usize,
        p_reach: &[f64],
        opp_reach: &[f64],
        iter: u32,
    ) -> Vec<f64> {
        match &self.tree.nodes[node] {
            Node::Terminal(t) => {
                let t = t.clone();
                self.terminal_value(&t, player, opp_reach)
            }
            Node::Decision {
                player: dp,
                children,
                ..
            } => {
                let dp = *dp;
                let children = children.clone();
                let na = children.len();

                if dp == player {
                    let n = p_reach.len();
                    let strat = self.current_strategy(node, n, na);
                    let mut node_val = vec![0.0f64; n];
                    let mut action_vals: Vec<Vec<f64>> = Vec::with_capacity(na);

                    for a in 0..na {
                        let mut child_reach = vec![0.0f64; n];
                        for i in 0..n {
                            child_reach[i] = p_reach[i] * strat[i * na + a];
                        }
                        let cv = self.cfr(children[a], player, &child_reach, opp_reach, iter);
                        for i in 0..n {
                            node_val[i] += strat[i * na + a] * cv[i];
                        }
                        action_vals.push(cv);
                    }

                    // regret + strategy sum updates
                    let w = iter as f64; // linear (CFR+) averaging
                    let reg = &mut self.regret[node];
                    for i in 0..n {
                        let base = i * na;
                        for a in 0..na {
                            reg[base + a] =
                                (reg[base + a] + action_vals[a][i] - node_val[i]).max(0.0);
                        }
                    }
                    let ss = &mut self.strat_sum[node];
                    for i in 0..n {
                        let base = i * na;
                        for a in 0..na {
                            ss[base + a] += w * p_reach[i] * strat[base + a];
                        }
                    }
                    node_val
                } else {
                    // opponent decides: scale opponent reach by their strategy
                    let n_opp = opp_reach.len();
                    let strat = self.current_strategy(node, n_opp, na);
                    let n_p = p_reach.len();
                    let mut val = vec![0.0f64; n_p];
                    for a in 0..na {
                        let mut child_opp = vec![0.0f64; n_opp];
                        for j in 0..n_opp {
                            child_opp[j] = opp_reach[j] * strat[j * na + a];
                        }
                        let cv = self.cfr(children[a], player, p_reach, &child_opp, iter);
                        for i in 0..n_p {
                            val[i] += cv[i];
                        }
                    }
                    val
                }
            }
        }
    }

    /// Evaluate value for `player` under a fixed strategy profile. If
    /// `best_response` is set, maximize independently at each of the player's
    /// private-hand information sets.
    fn evaluate_profile_value(
        &self,
        node: usize,
        player: usize,
        opp_reach: &[f64],
        n_p: usize,
        profile: &[Vec<f64>],
        best_response: bool,
    ) -> Vec<f64> {
        match &self.tree.nodes[node] {
            Node::Terminal(t) => self.terminal_value(t, player, opp_reach),
            Node::Decision {
                player: dp,
                children,
                ..
            } => {
                let dp = *dp;
                let na = children.len();
                if dp == player {
                    let strategy = Self::profile_strategy(profile, node, n_p, na);
                    let mut action_vals: Vec<Vec<f64>> = Vec::with_capacity(na);
                    for a in 0..na {
                        action_vals.push(self.evaluate_profile_value(
                            children[a],
                            player,
                            opp_reach,
                            n_p,
                            profile,
                            best_response,
                        ));
                    }
                    let mut val = vec![0.0f64; n_p];
                    if best_response {
                        for i in 0..n_p {
                            let mut best = f64::NEG_INFINITY;
                            for a in 0..na {
                                if action_vals[a][i] > best {
                                    best = action_vals[a][i];
                                }
                            }
                            val[i] = best;
                        }
                    } else {
                        for i in 0..n_p {
                            for a in 0..na {
                                val[i] += strategy[i * na + a] * action_vals[a][i];
                            }
                        }
                    }
                    val
                } else {
                    let n_opp = opp_reach.len();
                    let strategy = Self::profile_strategy(profile, node, n_opp, na);
                    let mut val = vec![0.0f64; n_p];
                    for a in 0..na {
                        let mut child_opp = vec![0.0f64; n_opp];
                        for j in 0..n_opp {
                            child_opp[j] = opp_reach[j] * strategy[j * na + a];
                        }
                        let cv = self.evaluate_profile_value(
                            children[a],
                            player,
                            &child_opp,
                            n_p,
                            profile,
                            best_response,
                        );
                        for i in 0..n_p {
                            val[i] += cv[i];
                        }
                    }
                    val
                }
            }
        }
    }

    fn scalar(&self, vals: &[f64], own_prob: &[f64]) -> f64 {
        vals.iter().zip(own_prob).map(|(v, p)| v * p).sum::<f64>() / self.joint_mass
    }

    fn opponent_compatible_mass(&self, player: usize, hand: usize, reach: &[f64]) -> f64 {
        if player == 0 {
            (0..self.n_ip)
                .filter(|opponent| self.compat[hand * self.n_ip + opponent])
                .map(|opponent| reach[opponent])
                .sum()
        } else {
            (0..self.n_oop)
                .filter(|opponent| self.compat[opponent * self.n_ip + hand])
                .map(|opponent| reach[opponent])
                .sum()
        }
    }

    fn exploitability_for_profile(&self, root: usize, profile: &[Vec<f64>]) -> f64 {
        let br_oop = self.evaluate_profile_value(root, 0, &self.ip_prob, self.n_oop, profile, true);
        let br_ip = self.evaluate_profile_value(root, 1, &self.oop_prob, self.n_ip, profile, true);
        let ev_oop =
            self.evaluate_profile_value(root, 0, &self.ip_prob, self.n_oop, profile, false);
        let ev_ip = self.evaluate_profile_value(root, 1, &self.oop_prob, self.n_ip, profile, false);
        let nc = (self.scalar(&br_oop, &self.oop_prob) - self.scalar(&ev_oop, &self.oop_prob))
            + (self.scalar(&br_ip, &self.ip_prob) - self.scalar(&ev_ip, &self.ip_prob));
        nc.max(0.0)
    }

    fn exploitability(&self, root: usize) -> f64 {
        self.exploitability_for_profile(root, &self.average_profile())
    }
}

// ---------------------------------------------------------------------------
// Equity precompute
// ---------------------------------------------------------------------------

fn enumerate_runouts(board: &[u8], dead: &[bool; 52], need: usize) -> Vec<Vec<u8>> {
    let deck: Vec<u8> = (0..52u8).filter(|c| !dead[*c as usize]).collect();
    let mut out = Vec::new();
    if need == 0 {
        out.push(board.to_vec());
        return out;
    }
    let m = deck.len();
    let mut idx: Vec<usize> = (0..need).collect();
    loop {
        let mut b = board.to_vec();
        for &i in &idx {
            b.push(deck[i]);
        }
        out.push(b);
        let mut i = need as i32 - 1;
        while i >= 0 && idx[i as usize] == m - need + i as usize {
            i -= 1;
        }
        if i < 0 {
            break;
        }
        idx[i as usize] += 1;
        for j in (i as usize + 1)..need {
            idx[j] = idx[j - 1] + 1;
        }
    }
    out
}

fn pair_equity(a: (u8, u8), b: (u8, u8), board: &[u8], need: usize) -> f32 {
    let mut dead = [false; 52];
    for &c in board {
        dead[c as usize] = true;
    }
    dead[a.0 as usize] = true;
    dead[a.1 as usize] = true;
    dead[b.0 as usize] = true;
    dead[b.1 as usize] = true;
    let runouts = enumerate_runouts(board, &dead, need);
    let mut win = 0.0f64;
    let mut total = 0.0f64;
    for full in &runouts {
        let mut ha = full.clone();
        ha.push(a.0);
        ha.push(a.1);
        let mut hb = full.clone();
        hb.push(b.0);
        hb.push(b.1);
        let sa = evaluate(&ha);
        let sb = evaluate(&hb);
        if sa > sb {
            win += 1.0;
        } else if sa == sb {
            win += 0.5;
        }
        total += 1.0;
    }
    (win / total) as f32
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn hand_class(a: u8, b: u8) -> String {
    let (r1, s1) = (a >> 2, a & 3);
    let (r2, s2) = (b >> 2, b & 3);
    let (hi, lo, hs, ls) = if r1 >= r2 {
        (r1, r2, s1, s2)
    } else {
        (r2, r1, s2, s1)
    };
    let hc = RANKS[hi as usize] as char;
    let lc = RANKS[lo as usize] as char;
    if hi == lo {
        format!("{}{}", hc, lc)
    } else if hs == ls {
        format!("{}{}s", hc, lc)
    } else {
        format!("{}{}o", hc, lc)
    }
}

fn take_top(mut hands: Vec<(u8, u8, f64)>, max: usize) -> (Vec<(u8, u8, f64)>, bool) {
    if hands.len() <= max {
        return (hands, false);
    }
    hands.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    hands.truncate(max);
    (hands, true)
}

fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

fn xfp_probability(
    old_probability: f64,
    response_probability: f64,
    old_reach: f64,
    response_reach: f64,
    old_count: f64,
) -> f64 {
    let denominator = old_count * old_reach + response_reach;
    if denominator <= 1e-15 {
        old_probability
    } else {
        (old_count * old_reach * old_probability + response_reach * response_probability)
            / denominator
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub fn solve(input_json: &str) -> String {
    let input: Input = match serde_json::from_str(input_json) {
        Ok(v) => v,
        Err(e) => return format!("{{\"error\":\"parse: {}\"}}", e),
    };
    match run(input) {
        Ok(out) => {
            serde_json::to_string(&out).unwrap_or_else(|e| format!("{{\"error\":\"{}\"}}", e))
        }
        Err(e) => format!("{{\"error\":\"{}\"}}", e),
    }
}

fn run(input: Input) -> Result<Output, String> {
    let need = 5usize.saturating_sub(input.board.len());
    if input.board.len() < 3 {
        return Err("board must have at least 3 cards".into());
    }
    if input.pot <= 0.0 {
        return Err("pot must be greater than zero".into());
    }

    // Remove any hand whose hole cards collide with a board card — such combos
    // are impossible and would otherwise be scored with a duplicated card.
    let mut on_board = [false; 52];
    for &c in &input.board {
        if (c as usize) < 52 {
            on_board[c as usize] = true;
        }
    }
    let board_free =
        |h: &(u8, u8, f64)| !on_board[h.0 as usize] && !on_board[h.1 as usize] && h.0 != h.1;
    let oop_in: Vec<_> = input.oop.into_iter().filter(|h| board_free(h)).collect();
    let ip_in: Vec<_> = input.ip.into_iter().filter(|h| board_free(h)).collect();

    let (oop_h, trunc_o) = take_top(oop_in, input.max_combos);
    let (ip_h, trunc_i) = take_top(ip_in, input.max_combos);
    if oop_h.is_empty() || ip_h.is_empty() {
        return Err("both ranges need combos that don't collide with the board".into());
    }

    let oop_cards: Vec<(u8, u8)> = oop_h.iter().map(|h| (h.0, h.1)).collect();
    let ip_cards: Vec<(u8, u8)> = ip_h.iter().map(|h| (h.0, h.1)).collect();
    let oop_combos: f64 = oop_h.iter().map(|h| h.2).sum();
    let ip_combos: f64 = ip_h.iter().map(|h| h.2).sum();

    // Normalize reach to probabilities.
    let oop_prob: Vec<f64> = oop_h.iter().map(|h| h.2 / oop_combos).collect();
    let ip_prob: Vec<f64> = ip_h.iter().map(|h| h.2 / ip_combos).collect();

    let n_oop = oop_cards.len();
    let n_ip = ip_cards.len();

    // Equity matrix + compatibility.
    let mut eq = vec![0.0f32; n_oop * n_ip];
    let mut compat = vec![false; n_oop * n_ip];
    for i in 0..n_oop {
        let a = oop_cards[i];
        for j in 0..n_ip {
            let b = ip_cards[j];
            let conflict = a.0 == b.0 || a.0 == b.1 || a.1 == b.0 || a.1 == b.1;
            if conflict {
                continue;
            }
            compat[i * n_ip + j] = true;
            eq[i * n_ip + j] = pair_equity(a, b, &input.board, need);
        }
    }
    let joint_mass: f64 = (0..n_oop)
        .flat_map(|i| {
            let oop_weight = oop_prob[i];
            let ip_prob_ref = &ip_prob;
            let compat_ref = &compat;
            (0..n_ip).map(move |j| {
                if compat_ref[i * n_ip + j] {
                    oop_weight * ip_prob_ref[j]
                } else {
                    0.0
                }
            })
        })
        .sum();
    if joint_mass <= 1e-15 {
        return Err("the two ranges contain no compatible private-hand pairs".into());
    }

    // Build tree.
    let mut tree = Tree {
        nodes: Vec::new(),
        bet_sizes: input.bet_sizes.clone(),
        raise_sizes: input.raise_sizes.clone(),
        stack: input.stack,
    };
    let root = tree.build(BState {
        to_act: 0,
        dead_pot: input.pot,
        oop_paid: 0.0,
        ip_paid: 0.0,
        facing: false,
        raises: 0,
        prev_check: false,
    });

    let num_nodes = tree.nodes.len();
    let mut regret: Vec<Vec<f64>> = Vec::with_capacity(num_nodes);
    let mut strat_sum: Vec<Vec<f64>> = Vec::with_capacity(num_nodes);
    for node in 0..num_nodes {
        let (p, na) = match &tree.nodes[node] {
            Node::Decision {
                player, children, ..
            } => (*player, children.len()),
            _ => (0, 0),
        };
        let n = if na == 0 {
            0
        } else if p == 0 {
            n_oop
        } else {
            n_ip
        };
        regret.push(vec![0.0; n * na]);
        strat_sum.push(vec![0.0; n * na]);
    }

    let mut solver = Solver {
        tree,
        n_oop,
        n_ip,
        oop_cards,
        ip_cards,
        oop_prob: oop_prob.clone(),
        ip_prob: ip_prob.clone(),
        oop_combos,
        ip_combos,
        eq,
        compat,
        joint_mass,
        regret,
        strat_sum,
    };

    // Always produce the existing CFR+ model. For XFP it becomes the first
    // member of the empirical opponent population (a mathematically valid
    // arbitrary initialization) and the fixed cross-play baseline.
    let iters = input.iterations.max(1);
    let mut cfr_history = Vec::new();
    for t in 1..=iters {
        let oop_reach = oop_prob.clone();
        let ip_reach = ip_prob.clone();
        solver.cfr(root, 0, &oop_reach, &ip_reach, t);
        solver.cfr(root, 1, &ip_reach, &oop_reach, t);
        if t % (iters / 10).max(1) == 0 || t == iters {
            let e = solver.exploitability(root);
            let pct = e / input.pot * 100.0;
            cfr_history.push((pct * 100.0).round() / 100.0);
        }
    }
    let cfr_profile = solver.average_profile();

    let (algorithm, history, cfr_comparison) = if input.algorithm == Algorithm::CfrPlus {
        ("cfr_plus".to_string(), cfr_history, None)
    } else {
        // Give the CFR+ initialization one unit of prior weight per CFR
        // training iteration. This is a generalized weakened-FP stepsize
        // schedule: α still tends to zero and its infinite sum diverges, so it
        // preserves convergence while avoiding a destructive first pure-BR
        // swing away from an already strong CFR profile.
        solver.strat_sum = cfr_profile.clone();
        let mut xfp_history = Vec::new();
        for step in 1..=iters {
            let old_profile = solver.average_profile();
            let oop_response = solver.best_response_profile(root, 0, &old_profile);
            let ip_response = solver.best_response_profile(root, 1, &old_profile);
            let mut next_profile = old_profile.clone();
            solver.xfp_update_player(
                root,
                0,
                &old_profile,
                &oop_response,
                iters as f64 + step as f64 - 1.0,
                &mut next_profile,
            );
            solver.xfp_update_player(
                root,
                1,
                &old_profile,
                &ip_response,
                iters as f64 + step as f64 - 1.0,
                &mut next_profile,
            );
            solver.strat_sum = next_profile;

            if step % (iters / 10).max(1) == 0 || step == iters {
                let pct = solver.exploitability(root) / input.pot * 100.0;
                xfp_history.push((pct * 100.0).round() / 100.0);
            }
        }

        let xfp_profile = solver.average_profile();
        let mut fp_oop_vs_cfr_ip = cfr_profile.clone();
        let mut cfr_oop_vs_fp_ip = cfr_profile.clone();
        for (node, tree_node) in solver.tree.nodes.iter().enumerate() {
            match tree_node {
                Node::Decision { player: 0, .. } => {
                    fp_oop_vs_cfr_ip[node] = xfp_profile[node].clone();
                }
                Node::Decision { player: 1, .. } => {
                    cfr_oop_vs_fp_ip[node] = xfp_profile[node].clone();
                }
                _ => {}
            }
        }
        let fp_oop_values =
            solver.evaluate_profile_value(root, 0, &ip_prob, n_oop, &fp_oop_vs_cfr_ip, false);
        let fp_ip_values =
            solver.evaluate_profile_value(root, 1, &oop_prob, n_ip, &cfr_oop_vs_fp_ip, false);
        let comparison = CfrComparison {
            cfr_iterations: iters,
            fp_oop_vs_cfr_ip_ev: round2(solver.scalar(&fp_oop_values, &oop_prob)),
            fp_ip_vs_cfr_oop_ev: round2(solver.scalar(&fp_ip_values, &ip_prob)),
        };
        ("fictitious_play".to_string(), xfp_history, Some(comparison))
    };

    let final_profile = solver.average_profile();
    let final_expl = solver.exploitability_for_profile(root, &final_profile) / input.pot * 100.0;

    // Overall EVs (chips) under average strategy.
    let ev_oop_vec = solver.evaluate_profile_value(root, 0, &ip_prob, n_oop, &final_profile, false);
    let ev_ip_vec = solver.evaluate_profile_value(root, 1, &oop_prob, n_ip, &final_profile, false);
    let oop_ev = solver.scalar(&ev_oop_vec, &oop_prob);
    let ip_ev = solver.scalar(&ev_ip_vec, &ip_prob);

    // Strategy outputs: OOP root, and IP node after OOP checks.
    let ip_vs_check = ip_node_after_oop_check(&solver, root);
    let oop_strat =
        node_strategy_root(&solver, &final_profile, root, root, 0, "OOP — first to act");
    let ip_strat = match ip_vs_check {
        Some(n) => node_strategy_root(&solver, &final_profile, n, root, 1, "IP — vs check"),
        None => NodeStrategy {
            title: "IP".into(),
            actions: vec![],
            rows: vec![],
        },
    };

    Ok(Output {
        algorithm,
        iterations: iters,
        exploitability_pct: round2(final_expl),
        oop_ev: round2(oop_ev),
        ip_ev: round2(ip_ev),
        pot: input.pot,
        oop_combos,
        ip_combos,
        truncated: trunc_o || trunc_i,
        oop: oop_strat,
        ip: ip_strat,
        action_tree: export_action_tree(&solver, &final_profile, root),
        exploitability_history: history,
        cfr_comparison,
    })
}

/// The IP decision node reached when OOP's first action is "Check".
fn ip_node_after_oop_check(solver: &Solver, root: usize) -> Option<usize> {
    if let Node::Decision {
        labels, children, ..
    } = &solver.tree.nodes[root]
    {
        for (i, l) in labels.iter().enumerate() {
            if l == "Check" {
                if let Node::Decision { .. } = &solver.tree.nodes[children[i]] {
                    return Some(children[i]);
                }
            }
        }
    }
    None
}

/// Report the average strategy at `node`, aggregated by hand class, using the
/// game `root` for the per-hand EV computation.
fn node_strategy_root(
    solver: &Solver,
    profile: &[Vec<f64>],
    node: usize,
    root: usize,
    player: usize,
    title: &str,
) -> NodeStrategy {
    node_strategy(
        solver,
        profile,
        node,
        root,
        player,
        title,
        &solver.oop_prob,
        &solver.ip_prob,
    )
}

/// Conditional strategy and range at an actual public betting history. Reaches
/// include both players' preceding actions, with exact private-card removal.
fn node_strategy(
    solver: &Solver,
    profile: &[Vec<f64>],
    node: usize,
    ev_node: usize,
    player: usize,
    title: &str,
    oop_reach: &[f64],
    ip_reach: &[f64],
) -> NodeStrategy {
    let (cards, probs, n) = if player == 0 {
        (&solver.oop_cards, oop_reach, solver.n_oop)
    } else {
        (&solver.ip_cards, ip_reach, solver.n_ip)
    };
    let (labels, na) = match &solver.tree.nodes[node] {
        Node::Decision {
            labels, children, ..
        } => (labels.clone(), children.len()),
        _ => (vec![], 0),
    };
    let joint_mass: f64 = oop_reach
        .iter()
        .enumerate()
        .map(|(i, reach)| reach * solver.opponent_compatible_mass(0, i, ip_reach))
        .sum();
    if na == 0 || joint_mass <= 1e-15 {
        return NodeStrategy {
            title: title.into(),
            actions: labels,
            rows: vec![],
        };
    }
    let avg = Solver::profile_strategy(profile, node, n, na);
    let opp = if player == 0 { ip_reach } else { oop_reach };
    let ev_vec = solver.evaluate_profile_value(ev_node, player, opp, n, profile, false);

    use std::collections::BTreeMap;
    let mut classes: BTreeMap<String, (f64, Vec<f64>, f64, Vec<ComboStrategy>)> = BTreeMap::new();
    let total_combos = if player == 0 {
        solver.oop_combos
    } else {
        solver.ip_combos
    };
    for i in 0..n {
        let cls = hand_class(cards[i].0, cards[i].1);
        let entry = classes
            .entry(cls)
            .or_insert((0.0, vec![0.0; na], 0.0, Vec::new()));
        let opponent_mass = solver.opponent_compatible_mass(player, i, opp);
        if opponent_mass <= 1e-15 || probs[i] <= 0.0 {
            continue;
        }
        // Weight private hands by their true marginal probability after card
        // removal, rather than by the independent input range alone.
        let w = probs[i] * opponent_mass / joint_mass;
        entry.0 += w;
        for a in 0..na {
            entry.1[a] += w * avg[i * na + a];
        }
        let combo_ev = ev_vec[i] / opponent_mass;
        entry.2 += w * combo_ev;

        // Combo equity vs opponent range
        let mut eq_sum = 0.0;
        let mut opp_mass = 0.0;
        if player == 0 {
            let base = i * solver.n_ip;
            for j in 0..solver.n_ip {
                if solver.compat[base + j] && opp[j] > 0.0 {
                    eq_sum += opp[j] * (solver.eq[base + j] as f64);
                    opp_mass += opp[j];
                }
            }
        } else {
            for j in 0..solver.n_oop {
                let idx = j * solver.n_ip + i;
                if solver.compat[idx] && opp[j] > 0.0 {
                    eq_sum += opp[j] * (1.0 - solver.eq[idx] as f64);
                    opp_mass += opp[j];
                }
            }
        }
        let equity = if opp_mass > 1e-12 {
            eq_sum / opp_mass
        } else {
            0.0
        };

        let combo_actions: Vec<ActionStrategy> = (0..na)
            .map(|a| ActionStrategy {
                action: labels[a].clone(),
                freq: (avg[i * na + a] * 10000.0).round() / 10000.0,
                ev: round2(combo_ev),
            })
            .collect();

        let input_weight = probs[i] * total_combos;
        entry.3.push(ComboStrategy {
            card0: cards[i].0,
            card1: cards[i].1,
            weight: input_weight,
            actions: combo_actions,
            ev: round2(combo_ev),
            equity: (equity * 10000.0).round() / 10000.0,
        });
    }
    let mut rows = Vec::new();
    for (cls, (mass, acts, evsum, combos_data)) in classes {
        if mass < 1e-12 {
            continue;
        }
        let ev = evsum / mass;
        let actions = (0..na)
            .map(|a| ActionStrategy {
                action: labels[a].clone(),
                freq: acts[a] / mass,
                ev,
            })
            .collect();
        rows.push(ClassRow {
            class: cls,
            combos: mass,
            actions,
            combos_data,
        });
    }
    NodeStrategy {
        title: title.into(),
        actions: labels,
        rows,
    }
}

/// Export the existing solved tree, not a new game or independently re-solved
/// branches. Zero-reach histories retain legal actions but have no fabricated
/// conditional range. IDs and child order are the solver's original IDs/order.
fn export_action_tree(solver: &Solver, profile: &[Vec<f64>], root: usize) -> ActionTree {
    fn visit(
        solver: &Solver,
        profile: &[Vec<f64>],
        node: usize,
        oop_reach: &[f64],
        ip_reach: &[f64],
        output: &mut [Option<ActionTreeNode>],
    ) {
        let joint_mass: f64 = oop_reach
            .iter()
            .enumerate()
            .map(|(i, reach)| reach * solver.opponent_compatible_mass(0, i, ip_reach))
            .sum();
        let reach_probability = joint_mass / solver.joint_mass;
        output[node] = Some(match &solver.tree.nodes[node] {
            Node::Decision {
                player,
                labels,
                children,
                state,
            } => {
                let title = format!(
                    "{} — {}",
                    player_label(*player).to_uppercase(),
                    if state.facing {
                        "facing a bet"
                    } else {
                        "to act"
                    }
                );
                let strategy = node_strategy(
                    solver, profile, node, node, *player, &title, oop_reach, ip_reach,
                );
                let own = if *player == 0 { oop_reach } else { ip_reach };
                let avg = Solver::profile_strategy(profile, node, own.len(), labels.len());
                for (action, child) in children.iter().enumerate() {
                    let child_reach: Vec<f64> = own
                        .iter()
                        .enumerate()
                        .map(|(hand, reach)| reach * avg[hand * labels.len() + action])
                        .collect();
                    if *player == 0 {
                        visit(solver, profile, *child, &child_reach, ip_reach, output);
                    } else {
                        visit(solver, profile, *child, oop_reach, &child_reach, output);
                    }
                }
                let paid = [state.oop_paid, state.ip_paid];
                ActionTreeNode {
                    id: node,
                    kind: "decision",
                    player: Some(player_label(*player)),
                    pot: state.dead_pot + state.oop_paid + state.ip_paid,
                    stacks: paid.map(|value| (solver.tree.stack - value).max(0.0)),
                    committed: paid,
                    to_call: (paid[1 - player] - paid[*player]).max(0.0),
                    reach_probability,
                    children: children.clone(),
                    strategy: Some(strategy),
                    reason: None,
                    winner: None,
                }
            }
            Node::Terminal(terminal) => {
                let (reason, winner) = match terminal.kind {
                    TermKind::Showdown => ("showdown", None),
                    TermKind::Fold(player) => ("fold", Some(player_label(1 - player))),
                };
                let paid = [terminal.oop_paid, terminal.ip_paid];
                ActionTreeNode {
                    id: node,
                    kind: "terminal",
                    player: None,
                    pot: terminal.dead_pot + terminal.oop_paid + terminal.ip_paid,
                    stacks: paid.map(|value| (solver.tree.stack - value).max(0.0)),
                    committed: paid,
                    to_call: 0.0,
                    reach_probability,
                    children: Vec::new(),
                    strategy: None,
                    reason: Some(reason),
                    winner,
                }
            }
        });
    }
    let mut nodes: Vec<Option<ActionTreeNode>> =
        (0..solver.tree.nodes.len()).map(|_| None).collect();
    visit(
        solver,
        profile,
        root,
        &solver.oop_prob,
        &solver.ip_prob,
        &mut nodes,
    );
    ActionTree {
        root,
        max_raises: MAX_RAISES,
        nodes: nodes
            .into_iter()
            .map(|node| node.expect("reachable tree node"))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(r: u8, s: u8) -> u8 {
        r * 4 + s
    }

    fn export_fixture() -> (Solver, usize, Vec<Vec<f64>>) {
        let mut tree = Tree {
            nodes: Vec::new(),
            bet_sizes: vec![0.5],
            raise_sizes: vec![0.1],
            stack: 100.0,
        };
        let root = tree.build(BState {
            to_act: 0,
            dead_pot: 6.0,
            oop_paid: 0.0,
            ip_paid: 0.0,
            facing: false,
            raises: 0,
            prev_check: false,
        });
        let strategy: Vec<Vec<f64>> = tree
            .nodes
            .iter()
            .map(|node| match node {
                Node::Decision { children, .. } => vec![1.0; 2 * children.len()],
                _ => Vec::new(),
            })
            .collect();
        let solver = Solver {
            tree,
            n_oop: 2,
            n_ip: 2,
            oop_cards: vec![(48, 49), (44, 45)],
            ip_cards: vec![(48, 40), (36, 37)],
            oop_prob: vec![0.5, 0.5],
            ip_prob: vec![0.5, 0.5],
            oop_combos: 2.0,
            ip_combos: 2.0,
            eq: vec![0.0, 0.8, 0.7, 0.6],
            compat: vec![false, true, true, true],
            joint_mass: 0.75,
            regret: strategy
                .iter()
                .map(|values| vec![0.0; values.len()])
                .collect(),
            strat_sum: strategy,
        };
        let mut profile = solver.average_profile();
        // AA checks, KK bets. Later nodes have a uniform average policy.
        profile[root] = vec![1.0, 0.0, 0.0, 1.0];
        (solver, root, profile)
    }

    #[test]
    fn exports_every_action_with_accounting_and_parent_child_reach() {
        let (solver, root, profile) = export_fixture();
        let export = export_action_tree(&solver, &profile, root);
        assert_eq!(export.nodes.len(), solver.tree.nodes.len());
        assert!((export.nodes[root].reach_probability - 1.0).abs() < 1e-12);
        for (id, node) in export.nodes.iter().enumerate() {
            assert_eq!(node.id, id);
            assert!((node.pot + node.stacks.iter().sum::<f64>() - 206.0).abs() < 1e-9);
            for player in 0..2 {
                assert!((node.stacks[player] + node.committed[player] - 100.0).abs() < 1e-9);
            }
            if let Some(strategy) = &node.strategy {
                assert_eq!(node.children.len(), strategy.actions.len());
                let child_mass: f64 = node
                    .children
                    .iter()
                    .map(|child| export.nodes[*child].reach_probability)
                    .sum();
                assert!((child_mass - node.reach_probability).abs() < 1e-12);
                for row in &strategy.rows {
                    assert!(
                        (row.actions.iter().map(|action| action.freq).sum::<f64>() - 1.0).abs()
                            < 1e-9
                    );
                    assert!(row.actions.iter().all(|action| action.ev.is_finite()));
                }
            } else {
                assert!(node.children.is_empty());
                assert!(node.reason.is_some());
            }
        }
        // Follow the modeled bet plus all three raises. No fourth raise is
        // manufactured by the exporter, and every final response is retained.
        let mut id = export.nodes[root].children[1];
        for _ in 0..MAX_RAISES {
            let node = &export.nodes[id];
            assert_eq!(
                node.strategy.as_ref().unwrap().actions,
                vec!["Fold", "Call", "Raise 10%"]
            );
            id = node.children[2];
        }
        assert_eq!(
            export.nodes[id].strategy.as_ref().unwrap().actions,
            vec!["Fold", "Call"]
        );
    }

    #[test]
    fn exported_ranges_and_values_condition_on_actions_and_card_removal() {
        let (solver, root, profile) = export_fixture();
        let export = export_action_tree(&solver, &profile, root);
        let after_check = &export.nodes[export.nodes[root].children[0]];
        let rows = &after_check.strategy.as_ref().unwrap().rows;
        // OOP checked only AA. IP's AsQc collides, so only JJ can reach.
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].class, "JJ");
        assert!((rows[0].combos - 1.0).abs() < 1e-12);
        assert!((rows[0].combos_data[0].equity - 0.2).abs() < 1e-4);
        assert!((after_check.reach_probability - 1.0 / 3.0).abs() < 1e-12);
        let values =
            solver.evaluate_profile_value(after_check.id, 1, &[0.5, 0.0], 2, &profile, false);
        assert!((rows[0].actions[0].ev - values[1] / 0.5).abs() < 1e-9);
        let after_bet = &export.nodes[export.nodes[root].children[1]];
        assert!((after_bet.pot - 9.0).abs() < 1e-9);
        assert!((after_bet.to_call - 3.0).abs() < 1e-9);
        assert!((after_bet.reach_probability - 2.0 / 3.0).abs() < 1e-12);
        let raise = &export.nodes[after_bet.children[2]];
        let raised_rows = &raise.strategy.as_ref().unwrap().rows;
        assert_eq!(raised_rows.len(), 1);
        assert_eq!(raised_rows[0].class, "KK");
    }

    #[test]
    fn zero_reach_nodes_keep_modeled_actions_without_inventing_ranges() {
        let (solver, root, mut profile) = export_fixture();
        profile[root] = vec![1.0, 0.0, 1.0, 0.0];
        let export = export_action_tree(&solver, &profile, root);
        let bet = &export.nodes[export.nodes[root].children[1]];
        assert_eq!(bet.reach_probability, 0.0);
        let strategy = bet.strategy.as_ref().unwrap();
        assert!(strategy.rows.is_empty());
        assert_eq!(strategy.actions, vec!["Fold", "Call", "Raise 10%"]);
        assert_eq!(bet.children.len(), 3);
        assert!(bet
            .children
            .iter()
            .all(|id| export.nodes[*id].reach_probability == 0.0));
    }

    #[test]
    fn xfp_averages_realization_reach_not_behavior_directly() {
        assert!((xfp_probability(0.25, 1.0, 1.0, 1.0, 1.0) - 0.625).abs() < 1e-12);
        // At a downstream information set the old strategy reaches the state
        // half as often as the pure best response, so its local update receives
        // half the weight that a naive 50/50 behavior average would assign.
        let downstream = xfp_probability(0.8, 1.0, 0.5, 1.0, 1.0);
        assert!((downstream - (1.4 / 1.5)).abs() < 1e-12);
        assert_eq!(xfp_probability(0.3, 1.0, 0.0, 0.0, 4.0), 0.3);
    }

    fn test_input(algorithm: Algorithm) -> Input {
        // Board Kh 7d 2c (dry). OOP has nut overpair + air; IP top pair + air.
        let board = vec![card(11, 2), card(5, 1), card(0, 0)];
        Input {
            board,
            oop: vec![
                (card(12, 3), card(12, 2), 1.0), // AsAh overpair (nuts here)
                (card(3, 3), card(2, 3), 1.0),   // 5s4s air
            ],
            ip: vec![
                (card(11, 3), card(10, 3), 1.0), // KsQs top pair
                (card(7, 2), card(6, 2), 1.0),   // 9h8h air
            ],
            pot: 6.0,
            stack: 100.0,
            bet_sizes: vec![0.75],
            raise_sizes: vec![1.0],
            iterations: 400,
            algorithm,
            max_combos: 200,
        }
    }

    #[test]
    fn cfr_plus_converges_and_is_constant_sum() {
        let out = run(test_input(Algorithm::CfrPlus)).expect("solve");
        eprintln!(
            "exploitability%={} oop_ev={} ip_ev={}",
            out.exploitability_pct, out.oop_ev, out.ip_ev
        );
        eprintln!("history={:?}", out.exploitability_history);
        for r in &out.oop.rows {
            let fr: Vec<String> = r
                .actions
                .iter()
                .map(|a| format!("{}={:.2}", a.action, a.freq))
                .collect();
            eprintln!(
                "OOP {} [{}] ev={:.2}",
                r.class,
                fr.join(","),
                r.actions[0].ev
            );
        }
        for r in &out.ip.rows {
            let fr: Vec<String> = r
                .actions
                .iter()
                .map(|a| format!("{}={:.2}", a.action, a.freq))
                .collect();
            eprintln!("IP(vs check) {} [{}]", r.class, fr.join(","));
        }
        // Exploitability should be driven low.
        assert!(
            out.exploitability_pct < 5.0,
            "expl too high: {}",
            out.exploitability_pct
        );
        assert!((out.oop_ev + out.ip_ev - out.pot).abs() < 0.02);
        assert!(out.cfr_comparison.is_none());
    }

    #[test]
    fn fictitious_play_converges_from_cfr_population_and_cross_plays_it() {
        let out = run(test_input(Algorithm::FictitiousPlay)).expect("solve");
        eprintln!(
            "xfp exploitability%={} oop_ev={} ip_ev={} history={:?}",
            out.exploitability_pct, out.oop_ev, out.ip_ev, out.exploitability_history
        );
        assert!(
            out.exploitability_pct < 5.0,
            "expl too high: {}",
            out.exploitability_pct
        );
        assert!((out.oop_ev + out.ip_ev - out.pot).abs() < 0.02);
        let comparison = out.cfr_comparison.expect("CFR cross-play report");
        assert_eq!(comparison.cfr_iterations, 400);
        assert!(comparison.fp_oop_vs_cfr_ip_ev.is_finite());
        assert!(comparison.fp_ip_vs_cfr_oop_ev.is_finite());
        for row in out.oop.rows.iter().chain(out.ip.rows.iter()) {
            let probability: f64 = row.actions.iter().map(|action| action.freq).sum();
            assert!((probability - 1.0).abs() < 1e-9);
        }
    }
}
