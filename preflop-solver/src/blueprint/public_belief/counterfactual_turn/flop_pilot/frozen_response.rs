//! Conditional postflop response, with the opponent's continuation frozen at
//! the candidate's own public ranges. Turn packets parallelize independently.
//! Only all 49 legal public turns may produce an aggregate. Turn chance is
//! integrated BEFORE a flop action is maximized; no future-card observation.
use super::*;
pub(super) mod playback;

type Ranges = [Vec<f64>; 2];
type History = Vec<String>;

#[derive(Clone, Serialize, Deserialize)]
struct Leaf {
    history: History,
    profile_bb: Ranges,
    best_response_bb: Ranges,
    policy_sha256: String,
}

#[derive(Clone, Serialize, Deserialize)]
struct TurnPacket {
    schema: String,
    candidate_sha256: String,
    turn: u8,
    turn_iterations: u64,
    leaves: Vec<Leaf>,
}

#[derive(Debug, Serialize)]
struct Response {
    schema: &'static str,
    candidate_sha256: String,
    public_turns: usize,
    turn_policy_count: usize,
    profile_bb: [f64; 2],
    best_response_bb: [f64; 2],
    gain_bb: [f64; 2],
    half_summed_gain_bb: f64,
    interpretation: &'static str,
}

pub(super) struct Frozen {
    trunk: Trunk,
    candidate_sha256: String,
    turn_iterations: u64,
    strategies: BTreeMap<History, Vec<f64>>,
    turns: BTreeMap<History, (GameState, Ranges)>,
}

impl Frozen {
    pub(super) fn new(input: &Solution) -> Result<Self, String> {
        if input.schema != "hu-native-counterfactual-turn-flop-pilot-v1"
            || input.iterations < 2
            || input.turn_iterations < 2
            || input.response_turn_iterations.is_some_and(|n| n < 2)
            || input
                .turn_samples_per_iteration
                .is_some_and(|n| !(2..=49).contains(&n))
            || (input.turn_samples_per_iteration.is_some() && input.chance_baseline.is_some())
            || input
                .chance_baseline
                .as_deref()
                .is_some_and(|mode| mode != "learned_conditional_turn_v1")
            || input
                .state
                .ranges
                .iter()
                .any(|r| (r.iter().sum::<f64>() - 1.0).abs() > 1e-10)
        {
            return Err("invalid frozen native flop candidate".into());
        }
        let mut trunk = Trunk::new(input.game.clone(), input.state.clone())?;
        // Validation may normalize. Preserve the already-normalized exported
        // prior exactly; do not silently change the evaluated input posterior.
        trunk.state = input.state.clone();
        if input.strategies.len() != trunk.nodes.len() {
            return Err("frozen flop candidate omits or adds public nodes".into());
        }
        let mut strategies = BTreeMap::new();
        for row in &input.strategies {
            let node = trunk
                .nodes
                .get_mut(&row.public_history)
                .ok_or("unknown frozen flop public node")?;
            if row.actor != node.actor
                || row.action_labels != node.action_labels
                || row.probabilities.len() != node.strategy_sum.len()
                || row
                    .probabilities
                    .iter()
                    .any(|v| !v.is_finite() || *v < 0.0 || *v > 1.0)
            {
                return Err("invalid frozen flop actions or probabilities".into());
            }
            let n = row.action_labels.len();
            for c in 0..COMBO_COUNT {
                let total: f64 = row.probabilities[c * n..(c + 1) * n]
                    .iter()
                    .map(|v| *v as f64)
                    .sum();
                if (trunk.legal[row.actor][c] && (total - 1.0).abs() > 1e-6)
                    || (!trunk.legal[row.actor][c] && total != 0.0)
                {
                    return Err("invalid frozen flop probability sum or support".into());
                }
            }
            node.strategy_sum = row.probabilities.iter().map(|p| *p as f64).collect();
            if strategies
                .insert(
                    row.public_history.clone(),
                    node.average_strategy(&trunk.legal[row.actor]),
                )
                .is_some()
            {
                return Err("duplicate frozen flop public node".into());
            }
        }
        let mut result = Self {
            trunk,
            strategies,
            turn_iterations: input
                .response_turn_iterations
                .unwrap_or(input.turn_iterations),
            candidate_sha256: format!("{:x}", Sha256::digest(serde_json::to_vec(input).unwrap())),
            turns: BTreeMap::new(),
        };
        result.collect_turns(
            result.trunk.state.game_state(),
            result.trunk.state.ranges.clone(),
        );
        Ok(result)
    }

    fn collect_turns(&mut self, state: GameState, reaches: Ranges) {
        if state.terminal.is_some() {
            return;
        }
        if state.street == Street::Turn {
            self.turns
                .insert(state.public_history.clone(), (state, reaches));
            return;
        }
        let actions = state.legal_actions(&self.trunk.game);
        let strategy = self.strategies[&state.public_history].clone();
        for (a, action) in actions.iter().enumerate() {
            let mut child = reaches.clone();
            for c in 0..COMBO_COUNT {
                child[state.actor][c] *= strategy[c * actions.len() + a];
            }
            self.collect_turns(state.apply(action, &self.trunk.game), child);
        }
    }

    fn turn_packet(&self, turn: u8) -> Result<TurnPacket, String> {
        if turn >= 52 || self.trunk.state.board.contains(&turn) {
            return Err("turn packet has invalid or already-visible card".into());
        }
        let mut board = self.trunk.state.board.clone();
        board.push(turn);
        let mut leaves = Vec::new();
        for (history, (state, prior)) in &self.turns {
            let mut ranges = prior.clone();
            for c in all_combos() {
                if c.cards().contains(&turn) {
                    ranges[0][c.key()] = 0.0;
                    ranges[1][c.key()] = 0.0;
                }
            }
            // This call receives only the frozen profile's reaches, never a
            // best responder's changed reaches. Freeze once, evaluate both BRs.
            let values = solve(TurnRiverSolveConfig {
                game: self.trunk.game.clone(),
                state: PublicBeliefState::from_game_state(board.clone(), state, ranges),
                iterations: self.turn_iterations,
                averaging_delay: 0,
                river_refinement_iterations: 0,
                regret_matching_plus: false,
            })?;
            leaves.push(Leaf {
                history: history.clone(),
                profile_bb: values.profile_counterfactual_bb,
                best_response_bb: values.best_response_counterfactual_bb,
                policy_sha256: values.policy_sha256,
            });
        }
        Ok(TurnPacket {
            schema: "hu-native-flop-frozen-turn-packet-v1".into(),
            candidate_sha256: self.candidate_sha256.clone(),
            turn,
            turn_iterations: self.turn_iterations,
            leaves,
        })
    }

    fn aggregate(&self, packets: &[TurnPacket]) -> Result<Response, String> {
        let expected: BTreeSet<_> = (0..52u8)
            .filter(|c| !self.trunk.state.board.contains(c))
            .collect();
        if packets.len() != 49
            || packets.iter().map(|p| p.turn).collect::<BTreeSet<_>>() != expected
        {
            return Err("response requires all 49 unique legal public turns; partial results are not an estimate".into());
        }
        let mut summed: BTreeMap<History, (Ranges, Ranges)> = self
            .turns
            .keys()
            .map(|key| {
                (
                    key.clone(),
                    (
                        std::array::from_fn(|_| vec![0.0; COMBO_COUNT]),
                        std::array::from_fn(|_| vec![0.0; COMBO_COUNT]),
                    ),
                )
            })
            .collect();
        // Stable reduction independent of process completion/order.
        let mut ordered = packets.iter().collect::<Vec<_>>();
        ordered.sort_by_key(|p| p.turn);
        for packet in ordered {
            if packet.schema != "hu-native-flop-frozen-turn-packet-v1"
                || packet.candidate_sha256 != self.candidate_sha256
                || packet.turn_iterations != self.turn_iterations
                || packet.leaves.len() != self.turns.len()
                || packet
                    .leaves
                    .iter()
                    .map(|l| &l.history)
                    .collect::<BTreeSet<_>>()
                    != self.turns.keys().collect()
            {
                return Err("mismatched candidate, schema or incomplete turn continuations".into());
            }
            for leaf in &packet.leaves {
                if leaf.policy_sha256.len() != 64
                    || !leaf.policy_sha256.bytes().all(|b| b.is_ascii_hexdigit())
                {
                    return Err("missing frozen continuation identity".into());
                }
                for p in 0..2 {
                    for values in [&leaf.profile_bb[p], &leaf.best_response_bb[p]] {
                        if values.len() != COMBO_COUNT
                            || values.iter().any(|v| {
                                !v.is_finite()
                                    || v.abs() > self.trunk.game.effective_stack_bb + 1e-8
                            })
                        {
                            return Err("invalid frozen turn counterfactual values".into());
                        }
                    }
                    for c in all_combos() {
                        let k = c.key();
                        if c.cards().contains(&packet.turn)
                            || c.cards().iter().any(|v| self.trunk.state.board.contains(v))
                        {
                            if leaf.profile_bb[p][k] != 0.0 || leaf.best_response_bb[p][k] != 0.0 {
                                return Err("board-blocked turn counterfactual value".into());
                            }
                        }
                        let entry = summed.get_mut(&leaf.history).unwrap();
                        // An exact compatible private pair has 45 legal turns.
                        entry.0[p][k] += leaf.profile_bb[p][k] / 45.0;
                        entry.1[p][k] += leaf.best_response_bb[p][k] / 45.0;
                    }
                }
            }
        }
        let root = self.trunk.state.game_state();
        let prior = &self.trunk.state.ranges;
        let joint = joint_compatibility_mass(prior);
        if joint <= 0.0 {
            return Err("frozen flop prior has no compatible private deals".into());
        }
        let profile = self.back_up(root.clone(), prior.clone(), None, &summed);
        let best: [Ranges; 2] =
            std::array::from_fn(|p| self.back_up(root.clone(), prior.clone(), Some(p), &summed));
        let profile_bb = std::array::from_fn(|p| {
            (0..COMBO_COUNT)
                .map(|c| prior[p][c] * profile[p][c])
                .sum::<f64>()
                / joint
        });
        let best_response_bb = std::array::from_fn(|p| {
            (0..COMBO_COUNT)
                .map(|c| prior[p][c] * best[p][p][c])
                .sum::<f64>()
                / joint
        });
        let gain_bb: [f64; 2] = std::array::from_fn(|p| best_response_bb[p] - profile_bb[p]);
        if profile_bb
            .iter()
            .chain(best_response_bb.iter())
            .any(|v| !v.is_finite())
            || (profile_bb[0] + profile_bb[1]).abs() > 1e-8
            || gain_bb.iter().any(|v| *v < -1e-8)
        {
            return Err("invalid postflop profile balance or response gain".into());
        }
        Ok(Response { schema: "hu-native-flop-frozen-response-v1",
            candidate_sha256:self.candidate_sha256.clone(), public_turns:49,
            turn_policy_count:49*self.turns.len(),profile_bb,best_response_bb,gain_bb,
            half_summed_gain_bb:(gain_bb[0]+gain_bb[1])/2.0,
            interpretation:"Conditional fixed-root flop/turn/river response with all public chance and exact-combo information sets; f32 frozen policies/equity arithmetic. Not full-game exploitability, a runtime safety proof, or an activation certificate." })
    }

    fn back_up(
        &self,
        state: GameState,
        reaches: Ranges,
        responder: Option<usize>,
        leaves: &BTreeMap<History, (Ranges, Ranges)>,
    ) -> Ranges {
        if state.terminal.is_some() {
            return self.trunk.terminal(&state, &reaches);
        }
        if state.street == Street::Turn {
            let leaf = &leaves[&state.public_history];
            return if responder.is_some() {
                leaf.1.clone()
            } else {
                leaf.0.clone()
            };
        }
        let actor = state.actor;
        let actions = state.legal_actions(&self.trunk.game);
        let strategy = &self.strategies[&state.public_history];
        let mut children = Vec::new();
        for (a, action) in actions.iter().enumerate() {
            let mut child = reaches.clone();
            if responder != Some(actor) {
                for c in 0..COMBO_COUNT {
                    child[actor][c] *= strategy[c * actions.len() + a];
                }
            }
            children.push(self.back_up(
                state.apply(action, &self.trunk.game),
                child,
                responder,
                leaves,
            ));
        }
        let mut result: Ranges = std::array::from_fn(|_| vec![0.0; COMBO_COUNT]);
        for c in 0..COMBO_COUNT {
            if responder == Some(actor) {
                result[actor][c] = children
                    .iter()
                    .map(|v| v[actor][c])
                    .fold(f64::NEG_INFINITY, f64::max);
            } else {
                result[actor][c] = children
                    .iter()
                    .enumerate()
                    .map(|(a, v)| strategy[c * actions.len() + a] * v[actor][c])
                    .sum();
            }
            result[1 - actor][c] = children.iter().map(|v| v[1 - actor][c]).sum();
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn royal_fixture() -> Solution {
        let mut game = BlueprintConfig::default();
        game.effective_stack_bb = 2.0;
        let mut ranges = std::array::from_fn(|_| vec![0.0; COMBO_COUNT]);
        ranges[0][Combo::new(48, 44).key()] = 1.0; // AcKc on TcJcQc: always royal.
        ranges[1][Combo::new(29, 30).key()] = 1.0; // 9d9h.
        let state = PublicBeliefState::flop_start([32, 36, 40], 1, [1.0, 1.0], ranges);
        let trunk = Trunk::new(game.clone(), state.clone()).unwrap();
        let strategies = trunk
            .nodes
            .iter()
            .map(|(history, node)| PublicBeliefStrategy {
                public_history: history.clone(),
                actor: node.actor,
                action_labels: node.action_labels.clone(),
                probabilities: node
                    .average_strategy(&trunk.legal[node.actor])
                    .into_iter()
                    .map(|v| v as f32)
                    .collect(),
                action_values_bb: None,
            })
            .collect();
        // Test-only uniform flop policy, not a learned/exported candidate.
        Solution {
            schema: "hu-native-counterfactual-turn-flop-pilot-v1".into(),
            game,
            state,
            seed: 0,
            iterations: 2,
            turn_iterations: 4,
            response_turn_iterations: None,
            strategies,
            turn_queries: 0,
            zero_own_reach_completions: [0, 0],
            maximum_conditional_turn_response_gain_bb: 0.0,
            zero_joint_turn_queries: 0,
            chance_baseline: None,
            turn_samples_per_iteration: None,
        }
    }

    #[test]
    fn reconstruction_budget_is_explicit_and_does_not_change_flop_training() {
        let original = royal_fixture();
        let original_bytes = serde_json::to_vec(&original).unwrap();
        assert!(!String::from_utf8_lossy(&original_bytes).contains("response_turn_iterations"));
        let original_frozen = Frozen::new(&original).unwrap();
        let mut candidate = original.clone();
        candidate.response_turn_iterations = Some(8);
        let reconstructed = Frozen::new(&candidate).unwrap();
        assert_eq!(candidate.turn_iterations, 4);
        assert_eq!(reconstructed.turn_iterations, 8);
        assert_eq!(original_frozen.strategies, reconstructed.strategies);
        assert_ne!(
            original_frozen.candidate_sha256,
            reconstructed.candidate_sha256
        );
        let packet = reconstructed.turn_packet(0).unwrap();
        assert_eq!(packet.turn_iterations, 8);
        assert_eq!(packet.candidate_sha256, reconstructed.candidate_sha256);
        for invalid in [0, 1] {
            candidate.response_turn_iterations = Some(invalid);
            assert!(Frozen::new(&candidate).is_err());
        }
    }

    #[test]
    fn frozen_response_preserves_information_sets_and_requires_complete_public_chance() {
        let fixture = royal_fixture();
        let frozen = Frozen::new(&fixture).unwrap();
        assert_eq!(frozen.turns.len(), 1);
        let mut packets = (0..52u8)
            .filter(|c| !fixture.state.board.contains(c))
            .map(|c| frozen.turn_packet(c).unwrap())
            .collect::<Vec<_>>();
        let response = frozen.aggregate(&packets).unwrap();
        let nuts = Combo::new(48, 44).key();
        let turn_profile = packets
            .iter()
            .map(|p| p.leaves[0].profile_bb[0][nuts] / 45.0)
            .sum::<f64>();
        let turn_best = packets
            .iter()
            .map(|p| p.leaves[0].best_response_bb[0][nuts] / 45.0)
            .sum::<f64>();
        // Uniform flop: BB jams half; hero calls/folds equally. If BB checks,
        // hero bets/checks equally; BB calls/folds equally against the bet.
        // Terminal payoffs are exact because hero always has a royal flush.
        assert!((response.profile_bb[0] - (0.625 + 0.5 * turn_profile)).abs() < 1e-12);
        assert!((response.best_response_bb[0] - (1.0 + 0.75f64.max(turn_best))).abs() < 1e-12);
        assert!((response.best_response_bb[1] - (-0.5)).abs() < 1e-12);
        let first = serde_json::to_vec(&response).unwrap();
        packets.reverse();
        assert_eq!(
            first,
            serde_json::to_vec(&frozen.aggregate(&packets).unwrap()).unwrap()
        );
        assert!(frozen
            .aggregate(&packets[..48])
            .unwrap_err()
            .contains("49 unique"));
        let mut invalid = packets.clone();
        invalid[0] = invalid[1].clone();
        assert!(frozen.aggregate(&invalid).is_err());
        let mut invalid = packets.clone();
        invalid[0].candidate_sha256 = "different".into();
        assert!(frozen.aggregate(&invalid).is_err());
        let mut invalid = packets.clone();
        invalid[0].leaves.clear();
        assert!(frozen.aggregate(&invalid).is_err());
        let mut invalid = packets.clone();
        let blocked = Combo::new(invalid[0].turn, 32).key();
        invalid[0].leaves[0].profile_bb[0][blocked] = 1.0;
        assert!(frozen
            .aggregate(&invalid)
            .unwrap_err()
            .contains("board-blocked"));

        // Red-capable clairvoyance regression at the real aggregation seam:
        // a turn continuation is better than betting on only 22/45 compatible
        // cards. A flop responder must choose before knowing which card comes.
        let mut high_turns = 0;
        for packet in &mut packets {
            let leaf = &mut packet.leaves[0];
            for p in 0..2 {
                leaf.profile_bb[p].fill(0.0);
                leaf.best_response_bb[p].fill(0.0);
            }
            if ![48, 44, 29, 30].contains(&packet.turn) && high_turns < 22 {
                leaf.best_response_bb[0][nuts] = 1.0;
                high_turns += 1;
            }
        }
        assert_eq!(high_turns, 22);
        let consistent = frozen.aggregate(&packets).unwrap();
        assert!((consistent.best_response_bb[0] - 1.75).abs() < 1e-12);
        let clairvoyant = 1.0 + (22.0 * 1.0 + 23.0 * 0.75) / 45.0;
        assert!(clairvoyant - consistent.best_response_bb[0] > 0.12);
    }

    #[test]
    fn frozen_candidate_rejects_missing_illegal_and_corrupt_probability_rows() {
        let fixture = royal_fixture();
        let mut invalid = fixture.clone();
        invalid.strategies.pop();
        assert!(Frozen::new(&invalid).is_err());
        let mut invalid = fixture.clone();
        invalid.chance_baseline = Some("unknown control variate".into());
        assert!(Frozen::new(&invalid).is_err());
        for count in [0, 1, 50] {
            let mut invalid = fixture.clone();
            invalid.turn_samples_per_iteration = Some(count);
            assert!(Frozen::new(&invalid).is_err());
        }
        let mut invalid = fixture.clone();
        invalid.turn_samples_per_iteration = Some(4);
        invalid.chance_baseline = Some("learned_conditional_turn_v1".into());
        assert!(Frozen::new(&invalid).is_err());
        let mut invalid = fixture.clone();
        invalid.strategies[0].actor = 1 - invalid.strategies[0].actor;
        assert!(Frozen::new(&invalid).is_err());
        let mut invalid = fixture.clone();
        invalid.strategies[0].probabilities[0] = f32::NAN;
        assert!(Frozen::new(&invalid).is_err());
        let mut invalid = fixture.clone();
        invalid.strategies[0].probabilities.fill(0.0);
        assert!(Frozen::new(&invalid).is_err());
        let frozen = Frozen::new(&fixture).unwrap();
        assert!(frozen.turn_packet(52).is_err());
        assert!(frozen.turn_packet(32).is_err());
        let prior = frozen.trunk.state.ranges.clone();
        let packet = frozen.turn_packet(0).unwrap();
        assert_eq!(prior, frozen.trunk.state.ranges);
        assert_eq!(packet.leaves.len(), frozen.turns.len());
    }

    fn read_candidate() -> Frozen {
        let path = std::env::var("POKER_NATIVE_FLOP_CANDIDATE").unwrap();
        let bytes = fs::read(path).unwrap();
        let digest = format!("{:x}", Sha256::digest(&bytes));
        assert_eq!(
            digest,
            std::env::var("POKER_NATIVE_FLOP_CANDIDATE_SHA").unwrap()
        );
        let candidate: Solution = serde_json::from_slice(&bytes).unwrap();
        let frozen = Frozen::new(&candidate).unwrap();
        assert_eq!(
            digest, frozen.candidate_sha256,
            "candidate must round-trip without identity drift"
        );
        frozen
    }

    fn exclusive_output<T: Serialize>(value: &T) -> String {
        let bytes = serde_json::to_vec(value).unwrap();
        let path = std::env::var("POKER_NATIVE_FLOP_OUTPUT").unwrap();
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();
        file.write_all(&bytes).unwrap();
        file.sync_all().unwrap();
        format!("{:x}", Sha256::digest(&bytes))
    }

    #[test]
    #[ignore = "hash-pinned candidate and guarded export for independent response-backup audit"]
    fn saved_native_flop_equity_audit_input() {
        let frozen = read_candidate();
        let equity = exact_flop_all_in_equities(
            frozen.trunk.state.board.clone().try_into().unwrap(),
            &frozen.trunk.legal,
            1,
        );
        // Share the actual f32 terminal inputs, not Rust's traversal or backed-up
        // values. The independent audit does not claim a second hand evaluator.
        let bytes: Vec<u8> = equity.iter().flat_map(|v| v.to_le_bytes()).collect();
        let path = PathBuf::from(std::env::var("POKER_NATIVE_FLOP_OUTPUT").unwrap())
            .with_extension("f32le");
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.write_all(&bytes).unwrap();
        file.sync_all().unwrap();
        let metadata = serde_json::json!({
            "schema":"native-flop-equity-audit-input-v1",
            "candidateSha256":frozen.candidate_sha256,
            "board":frozen.trunk.state.board,
            "legal":frozen.trunk.legal,
            "comboOrder":"high-card-major: high=1..51, low=0..high-1",
            "encoding":"1326x1326 row-major little-endian f32; NaN for incompatible pairs",
            "bytes":bytes.len(),
            "sha256":format!("{:x}",Sha256::digest(&bytes)),
            "interpretation":"Shared native terminal-equity inputs for independent flop backup/accounting audit; not an independent equity evaluator or a model."
        });
        let digest = exclusive_output(&metadata);
        println!(
            "{}",
            serde_json::json!({"stage":"native_flop_equity_audit_input", "metadataSha256":digest})
        );
    }

    #[test]
    #[ignore = "explicit hash-pinned candidate/turn/output and external resource guard"]
    fn saved_native_flop_turn_packet() {
        let started = std::time::Instant::now();
        let frozen = read_candidate();
        let turn = std::env::var("POKER_NATIVE_FLOP_TURN")
            .unwrap()
            .parse()
            .unwrap();
        let packet = frozen.turn_packet(turn).unwrap();
        let digest = exclusive_output(&packet);
        println!(
            "{}",
            serde_json::json!({"stage":"native_flop_frozen_turn_packet","turn":turn,
            "seconds":started.elapsed().as_secs_f64(),"turnPolicies":packet.leaves.len(),
            "candidateSha256":frozen.candidate_sha256,"outputSha256":digest,
            "interpretation":"One frozen candidate turn packet; no aggregate response until all 49 public turns complete"})
        );
    }

    #[test]
    #[ignore = "requires all 49 hash-pinned frozen candidate turn packets and an external resource guard"]
    fn saved_native_flop_response_aggregate() {
        let frozen = read_candidate();
        let directory = PathBuf::from(std::env::var("POKER_NATIVE_FLOP_PACKET_DIRECTORY").unwrap());
        let packets = (0..52u8)
            .filter(|c| !frozen.trunk.state.board.contains(c))
            .map(|c| {
                serde_json::from_slice(&fs::read(directory.join(format!("turn-{c}.json"))).unwrap())
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let response = frozen.aggregate(&packets).unwrap();
        let digest = exclusive_output(&response);
        println!(
            "{}",
            serde_json::json!({"stage":"native_flop_frozen_response","response":response,"outputSha256":digest})
        );
    }
}
