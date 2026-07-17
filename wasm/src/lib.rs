//! Single-street postflop solver using vectorized CFR+ with an all-in-equity
//! terminal model.
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
    #[serde(default = "default_max_combos")]
    max_combos: usize,
}
fn default_max_combos() -> usize {
    200
}

#[derive(Serialize)]
struct ActionStrategy {
    action: String,
    freq: f64,
    ev: f64,
}
#[derive(Serialize)]
struct ClassRow {
    class: String,
    combos: f64,
    actions: Vec<ActionStrategy>,
}
#[derive(Serialize)]
struct NodeStrategy {
    title: String,
    actions: Vec<String>,
    rows: Vec<ClassRow>,
}
#[derive(Serialize)]
struct Output {
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
    exploitability_history: Vec<f64>,
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
    // eq[i*n_ip + j] = OOP hand i equity share (win + tie/2) vs IP hand j
    eq: Vec<f32>,
    // compat[i*n_ip + j] = hands share no card
    compat: Vec<bool>,
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

    /// Evaluate value for `player` under average strategies (no updates).
    /// If `best_response` is set for `player`'s nodes, take the max action.
    fn evaluate_val(
        &self,
        node: usize,
        player: usize,
        opp_reach: &[f64],
        n_p: usize,
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
                    let strat = self.average_strategy(node, n_p, na);
                    let mut action_vals: Vec<Vec<f64>> = Vec::with_capacity(na);
                    for a in 0..na {
                        action_vals.push(self.evaluate_val(
                            children[a],
                            player,
                            opp_reach,
                            n_p,
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
                                val[i] += strat[i * na + a] * action_vals[a][i];
                            }
                        }
                    }
                    val
                } else {
                    let n_opp = opp_reach.len();
                    let strat = self.average_strategy(node, n_opp, na);
                    let mut val = vec![0.0f64; n_p];
                    for a in 0..na {
                        let mut child_opp = vec![0.0f64; n_opp];
                        for j in 0..n_opp {
                            child_opp[j] = opp_reach[j] * strat[j * na + a];
                        }
                        let cv =
                            self.evaluate_val(children[a], player, &child_opp, n_p, best_response);
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
        vals.iter().zip(own_prob).map(|(v, p)| v * p).sum()
    }

    fn exploitability(&self, root: usize) -> f64 {
        // Best response value for each player vs opponent's average strategy.
        let br_oop = self.evaluate_val(root, 0, &self.ip_prob, self.n_oop, true);
        let br_ip = self.evaluate_val(root, 1, &self.oop_prob, self.n_ip, true);
        let ev_oop = self.evaluate_val(root, 0, &self.ip_prob, self.n_oop, false);
        let ev_ip = self.evaluate_val(root, 1, &self.oop_prob, self.n_ip, false);
        let nc = (self.scalar(&br_oop, &self.oop_prob) - self.scalar(&ev_oop, &self.oop_prob))
            + (self.scalar(&br_ip, &self.ip_prob) - self.scalar(&ev_ip, &self.ip_prob));
        nc.max(0.0)
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
        Ok(out) => serde_json::to_string(&out).unwrap_or_else(|e| format!("{{\"error\":\"{}\"}}", e)),
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
    let board_free = |h: &(u8, u8, f64)| {
        !on_board[h.0 as usize] && !on_board[h.1 as usize] && h.0 != h.1
    };
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
            Node::Decision { player, children, .. } => (*player, children.len()),
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
        eq,
        compat,
        regret,
        strat_sum,
    };

    // CFR+ iterations (both players each iteration).
    let iters = input.iterations.max(1);
    let mut history = Vec::new();
    for t in 1..=iters {
        let oop_reach = oop_prob.clone();
        let ip_reach = ip_prob.clone();
        solver.cfr(root, 0, &oop_reach, &ip_reach, t);
        solver.cfr(root, 1, &ip_reach, &oop_reach, t);
        if t % (iters / 10).max(1) == 0 || t == iters {
            let e = solver.exploitability(root);
            let pct = e / input.pot * 100.0;
            history.push((pct * 100.0).round() / 100.0);
        }
    }

    let final_expl = solver.exploitability(root) / input.pot * 100.0;

    // Overall EVs (chips) under average strategy.
    let ev_oop_vec = solver.evaluate_val(root, 0, &ip_prob, n_oop, false);
    let ev_ip_vec = solver.evaluate_val(root, 1, &oop_prob, n_ip, false);
    let oop_ev = solver.scalar(&ev_oop_vec, &oop_prob);
    let ip_ev = solver.scalar(&ev_ip_vec, &ip_prob);

    // Strategy outputs: OOP root, and IP node after OOP checks.
    let ip_vs_check = ip_node_after_oop_check(&solver, root);
    let oop_strat = node_strategy_root(&solver, root, root, 0, "OOP — first to act");
    let ip_strat = match ip_vs_check {
        Some(n) => node_strategy_root(&solver, n, root, 1, "IP — vs check"),
        None => NodeStrategy {
            title: "IP".into(),
            actions: vec![],
            rows: vec![],
        },
    };

    Ok(Output {
        iterations: iters,
        exploitability_pct: (final_expl * 100.0).round() / 100.0,
        oop_ev: (oop_ev * 100.0).round() / 100.0,
        ip_ev: (ip_ev * 100.0).round() / 100.0,
        pot: input.pot,
        oop_combos,
        ip_combos,
        truncated: trunc_o || trunc_i,
        oop: oop_strat,
        ip: ip_strat,
        exploitability_history: history.into_iter().map(|x| x).collect(),
    })
}

/// The IP decision node reached when OOP's first action is "Check".
fn ip_node_after_oop_check(solver: &Solver, root: usize) -> Option<usize> {
    if let Node::Decision { labels, children, .. } = &solver.tree.nodes[root] {
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
    node: usize,
    root: usize,
    player: usize,
    title: &str,
) -> NodeStrategy {
    let (cards, probs, n) = if player == 0 {
        (&solver.oop_cards, &solver.oop_prob, solver.n_oop)
    } else {
        (&solver.ip_cards, &solver.ip_prob, solver.n_ip)
    };
    let (labels, na) = match &solver.tree.nodes[node] {
        Node::Decision { labels, children, .. } => (labels.clone(), children.len()),
        _ => (vec![], 0),
    };
    if na == 0 {
        return NodeStrategy { title: title.into(), actions: vec![], rows: vec![] };
    }
    let avg = solver.average_strategy(node, n, na);
    let opp = if player == 0 { &solver.ip_prob } else { &solver.oop_prob };
    let ev_vec = solver.evaluate_val(root, player, opp, n, false);

    use std::collections::BTreeMap;
    let mut classes: BTreeMap<String, (f64, Vec<f64>, f64)> = BTreeMap::new();
    for i in 0..n {
        let cls = hand_class(cards[i].0, cards[i].1);
        let entry = classes.entry(cls).or_insert((0.0, vec![0.0; na], 0.0));
        let w = probs[i];
        entry.0 += w;
        for a in 0..na {
            entry.1[a] += w * avg[i * na + a];
        }
        entry.2 += w * ev_vec[i];
    }
    let mut rows = Vec::new();
    for (cls, (mass, acts, evsum)) in classes {
        if mass < 1e-12 {
            continue;
        }
        let ev = evsum / mass;
        let actions = (0..na)
            .map(|a| ActionStrategy { action: labels[a].clone(), freq: acts[a] / mass, ev })
            .collect();
        rows.push(ClassRow { class: cls, combos: mass, actions });
    }
    NodeStrategy { title: title.into(), actions: labels, rows }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(r: u8, s: u8) -> u8 { r * 4 + s }

    #[test]
    fn converges_and_sane() {
        // Board Kh 7d 2c (dry). OOP has nut overpair + air; IP top pair + air.
        let board = vec![card(11,2), card(5,1), card(0,0)];
        let input = Input {
            board,
            oop: vec![
                (card(12,3), card(12,2), 1.0), // AsAh overpair (nuts here)
                (card(3,3),  card(2,3),  1.0), // 5s4s air
            ],
            ip: vec![
                (card(11,3), card(10,3), 1.0), // KsQs top pair
                (card(7,2),  card(6,2),  1.0), // 9h8h air
            ],
            pot: 6.0,
            stack: 100.0,
            bet_sizes: vec![0.75],
            raise_sizes: vec![1.0],
            iterations: 400,
            max_combos: 200,
        };
        let out = run(input).expect("solve");
        eprintln!("exploitability%={} oop_ev={} ip_ev={}", out.exploitability_pct, out.oop_ev, out.ip_ev);
        eprintln!("history={:?}", out.exploitability_history);
        for r in &out.oop.rows {
            let fr: Vec<String> = r.actions.iter().map(|a| format!("{}={:.2}", a.action, a.freq)).collect();
            eprintln!("OOP {} [{}] ev={:.2}", r.class, fr.join(","), r.actions[0].ev);
        }
        for r in &out.ip.rows {
            let fr: Vec<String> = r.actions.iter().map(|a| format!("{}={:.2}", a.action, a.freq)).collect();
            eprintln!("IP(vs check) {} [{}]", r.class, fr.join(","));
        }
        // Exploitability should be driven low.
        assert!(out.exploitability_pct < 5.0, "expl too high: {}", out.exploitability_pct);
    }
}
