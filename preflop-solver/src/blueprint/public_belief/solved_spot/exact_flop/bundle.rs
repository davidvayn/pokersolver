//! Versioned static binary tree: no regrets, partial training state or guessed
//! continuations. Re-import checks the actual bytes, not an earlier report.
use super::*;
use std::{fs, io::Write, path::Path};

pub const SHARD_SIZE: usize = 128;
pub const MAX_SHARD_BYTES: usize = 65_536;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Shard {
    pub start: usize,
    pub count: usize,
    pub sha256: String,
    pub bytes: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Manifest {
    pub schema: String,
    pub input: SpotInput,
    pub input_sha256: String,
    pub frozen_policy_sha256: String,
    pub frozen_action_values_sha256: String,
    pub root: usize,
    pub tree_nodes: usize,
    pub decision_nodes: usize,
    pub shard_size: usize,
    pub ranges: [Vec<(usize, f64)>; 2],
    pub shards: Vec<Shard>,
    pub verification: Verification,
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn hash_ok(hash: &str) -> bool {
    hash.len() == 64
        && hash
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .and_then(|mut file| file.write_all(bytes))
        .map_err(|e| e.to_string())
}

// Prefixes contain all public state and edges. Replaying native rules creates
// the expected prefixes again during import; rows cannot alter the game tree.
fn prefixes(game: &Game, config: &SpotConfig) -> Result<Vec<Vec<u8>>, String> {
    fn walk(
        game: &Game,
        config: &SpotConfig,
        id: usize,
        state: GameState,
        out: &mut [Vec<u8>],
    ) -> Result<(), String> {
        let node = &game.nodes[id];
        let board = &game.boards[node.board];
        let (tag, actor, edges, rows) = match &node.kind {
            Kind::Terminal {
                terminal: Terminal::Fold { winner },
                ..
            } => (0, *winner, 0, 0),
            Kind::Terminal { .. } => (1, 255, 0, 0),
            Kind::Chance { children, .. } => (2, 255, children.len(), 0),
            Kind::Decision { actor, children } => {
                (3, *actor, children.len(), board.legal[*actor].len())
            }
        };
        let mut bytes = vec![tag, actor as u8, board.cards.len() as u8];
        for i in 0..5 {
            bytes.push(board.cards.get(i).copied().unwrap_or(255));
        }
        for value in state.invested.iter().chain(&state.street_invested) {
            bytes.extend(value.to_le_bytes());
        }
        bytes.extend((edges as u16).to_le_bytes());
        bytes.extend((rows as u16).to_le_bytes());
        match &node.kind {
            Kind::Terminal { .. } => {}
            Kind::Chance { children, .. } => {
                for (card, child) in children {
                    bytes.push(*card);
                    bytes.extend((*child as u32).to_le_bytes());
                    walk(game, config, *child, state.clone(), out)?;
                }
            }
            Kind::Decision { children, .. } => {
                let actions = state.legal_actions(&config.game);
                if actions.len() != children.len() {
                    return Err("export action count mismatch".into());
                }
                for (action, child) in actions.iter().zip(children) {
                    if action.label.is_empty() || action.label.len() > 80 {
                        return Err("unbounded action label".into());
                    }
                    bytes.push(action.label.len() as u8);
                    bytes.extend(action.label.as_bytes());
                    let (kind, target) = match action.kind {
                        ActionKind::Fold => (0, 0.0),
                        ActionKind::Check => (1, 0.0),
                        ActionKind::Call => (2, 0.0),
                        ActionKind::RaiseTo(target) => {
                            let all_in = (target
                                - state.street_invested[state.actor]
                                - state.remaining(state.actor, &config.game))
                            .abs()
                                < 1e-8;
                            (
                                if all_in {
                                    5
                                } else if state.to_call() > 0.0 {
                                    4
                                } else {
                                    3
                                },
                                target,
                            )
                        }
                    };
                    bytes.push(kind);
                    bytes.extend(target.to_le_bytes());
                    bytes.extend((*child as u32).to_le_bytes());
                    walk(game, config, *child, state.apply(action, &config.game), out)?;
                }
            }
        }
        out[id] = bytes;
        Ok(())
    }
    let mut result = vec![Vec::new(); game.nodes.len()];
    walk(
        game,
        config,
        game.root,
        config.state.game_state(),
        &mut result,
    )?;
    Ok(result)
}

fn encode(
    game: &Game,
    frozen: &[Vec<f32>],
    values: &[values::NodeValues],
    prefixes: &[Vec<u8>],
    start: usize,
    end: usize,
) -> Vec<u8> {
    let mut bytes = b"PSF1".to_vec();
    bytes.extend((start as u32).to_le_bytes());
    bytes.extend(((end - start) as u16).to_le_bytes());
    for id in start..end {
        let mut record = prefixes[id].clone();
        if let Kind::Decision { actor, children } = &game.nodes[id].kind {
            for (row, combo) in game.boards[game.nodes[id].board].legal[*actor]
                .iter()
                .enumerate()
            {
                record.extend((game.combos[*actor][*combo].key() as u16).to_le_bytes());
                record.push(u8::from(values[id].available[row]));
                for value in &frozen[id][row * children.len()..(row + 1) * children.len()] {
                    record.extend(value.to_le_bytes());
                }
                if values[id].available[row] {
                    for value in &values[id].cells[row * children.len()..(row + 1) * children.len()]
                    {
                        record.extend(value.to_le_bytes());
                    }
                }
            }
        }
        bytes.extend((record.len() as u32).to_le_bytes());
        bytes.extend(record);
    }
    bytes
}

fn verification(
    input: &SpotInput,
    profile: [f64; 2],
    best: [f64; 2],
    ev: f64,
    prob: f64,
) -> Result<Verification, String> {
    let footprint = audit(input)?;
    let gain = [best[0] - profile[0], best[1] - profile[1]];
    let total = gain.iter().sum::<f64>();
    if profile.iter().chain(&best).any(|v| !v.is_finite())
        || gain.iter().any(|v| *v < -1e-8)
        || !total.is_finite()
        || total < 0.0
        || total > GAIN_LIMIT
        || ev > EV_TOLERANCE
        || prob > 1e-6
        || (profile[0] + profile[1]).abs() > 1e-8
    {
        return Err("flop bundle rejected by unchanged response/EV/probability gates".into());
    }
    Ok(Verification {
        method: "direct-compatible-pair-and-runout-enumeration-of-exported-f32-policy-v1".into(),
        compatible_deals: footprint.compatible_private_deals as usize,
        profile_value_bb: profile,
        best_response_value_bb: best,
        unilateral_gain_bb: gain,
        total_best_response_gain_bb: total,
        maximum_action_ev_error_bb: ev,
        maximum_probability_sum_error: prob,
        zero_sum_residual_bb: (profile[0] + profile[1]).abs(),
        accepted: true,
        river_outcomes: Some(footprint.compatible_deal_runouts as usize),
    })
}

pub fn export(input: SpotInput, limits: PilotLimits, directory: &Path) -> Result<String, String> {
    if directory.exists() {
        return Err("bundle directory must be new".into());
    }
    let (report, game, frozen, values) = solve_checked(input.clone(), limits)?;
    let checked = verification(
        &input,
        report.profile_value_bb,
        report.best_response_value_bb,
        report.maximum_action_ev_error_bb,
        report.maximum_probability_sum_error,
    )?;
    let config = config_for_audit(&input, true)?;
    let prefixes = prefixes(&game, &config)?;
    fs::create_dir(directory).map_err(|e| e.to_string())?;
    let mut shards = Vec::new();
    for start in (0..game.nodes.len()).step_by(SHARD_SIZE) {
        let end = (start + SHARD_SIZE).min(game.nodes.len());
        let bytes = encode(&game, &frozen, &values, &prefixes, start, end);
        if bytes.len() > MAX_SHARD_BYTES {
            return Err("binary shard exceeds fixed byte limit".into());
        }
        let sha = digest(&bytes);
        let path = directory.join(format!("{sha}.bin"));
        write_new(&path, &bytes)?;
        shards.push(Shard {
            start,
            count: end - start,
            sha256: sha,
            bytes: bytes.len(),
        });
    }
    let manifest = Manifest {
        schema: "solved-flop-sharded-v1".into(),
        input: input.clone(),
        input_sha256: report.input_sha256,
        frozen_policy_sha256: report.frozen_policy_sha256,
        frozen_action_values_sha256: report.frozen_action_values_sha256,
        root: game.root,
        tree_nodes: game.nodes.len(),
        decision_nodes: game.decisions(),
        shard_size: SHARD_SIZE,
        ranges: sparse_ranges(&config.state.ranges),
        shards,
        verification: checked,
    };
    let bytes = serde_json::to_vec(&manifest).map_err(|e| e.to_string())?;
    if bytes.len() > 2_000_000 {
        return Err("flop manifest exceeds byte limit".into());
    }
    let name = format!("{}.json", digest(&bytes));
    write_new(&directory.join(&name), &bytes)?;
    // No catalog can point at this bundle until a fresh read of all bytes passes.
    verify(&directory.join(&name))?;
    Ok(name)
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.at.checked_add(n).ok_or("binary length overflow")?;
        let v = self
            .bytes
            .get(self.at..end)
            .ok_or("truncated binary shard")?;
        self.at = end;
        Ok(v)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<usize, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as usize)
    }
    fn u32(&mut self) -> Result<usize, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()) as usize)
    }
    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
}

pub fn verify(path: &Path) -> Result<Manifest, String> {
    if fs::metadata(path).map_err(|e| e.to_string())?.len() > 2_000_000 {
        return Err("oversized flop manifest".into());
    }
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    if path.file_name().and_then(|n| n.to_str())
        != Some(format!("{}.json", digest(&bytes)).as_str())
    {
        return Err("manifest checksum mismatch".into());
    }
    let manifest: Manifest = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    if manifest.schema != "solved-flop-sharded-v1"
        || manifest.input.board.len() != 3
        || manifest.shard_size != SHARD_SIZE
        || manifest.tree_nodes > 2_000_000
        || manifest.tree_nodes == 0
        || manifest.shards.len() != manifest.tree_nodes.div_ceil(SHARD_SIZE)
        || manifest.input_sha256
            != digest(&serde_json::to_vec(&manifest.input).map_err(|e| e.to_string())?)
    {
        return Err("invalid flop manifest".into());
    }
    let footprint = audit(&manifest.input)?;
    if footprint.starting_combos.iter().any(|n| *n > 32)
        || footprint.public_decision_nodes > 500_000
        || footprint.sparse_regret_average_array_bytes > 512 * 1024 * 1024
    {
        return Err("bundle structural budget exceeded".into());
    }
    let guard = Guard {
        deadline: Instant::now() + Duration::from_secs(600),
        visits: Cell::new(0),
    };
    let config = config_for_audit(&manifest.input, true)?;
    let game = Game::new(&config, &guard)?;
    if game.nodes.len() != manifest.tree_nodes
        || game.root != manifest.root
        || game.decisions() != manifest.decision_nodes
        || sparse_ranges(&config.state.ranges) != manifest.ranges
    {
        return Err("bundle game mismatch".into());
    }
    let prefixes = prefixes(&game, &config)?;
    let mut frozen = game
        .nodes
        .iter()
        .map(|n| vec![0.0; n.sums.len()])
        .collect::<Vec<_>>();
    let mut values = game
        .nodes
        .iter()
        .map(|n| values::NodeValues {
            cells: vec![0.0; n.sums.len()],
            available: match &n.kind {
                Kind::Decision { actor, .. } => {
                    vec![false; game.boards[n.board].legal[*actor].len()]
                }
                _ => vec![],
            },
        })
        .collect::<Vec<_>>();
    for (index, shard) in manifest.shards.iter().enumerate() {
        let start = index * SHARD_SIZE;
        let count = (manifest.tree_nodes - start).min(SHARD_SIZE);
        if shard.start != start
            || shard.count != count
            || !hash_ok(&shard.sha256)
            || shard.bytes > MAX_SHARD_BYTES
        {
            return Err("invalid shard index".into());
        }
        let file = path.parent().unwrap().join(format!("{}.bin", shard.sha256));
        if fs::metadata(&file).map_err(|e| e.to_string())?.len() != shard.bytes as u64 {
            return Err("shard byte count mismatch".into());
        }
        let bytes = fs::read(file).map_err(|e| e.to_string())?;
        if digest(&bytes) != shard.sha256 {
            return Err("shard checksum mismatch".into());
        }
        let mut r = Reader {
            bytes: &bytes,
            at: 0,
        };
        if r.take(4)? != b"PSF1" || r.u32()? != start || r.u16()? != count {
            return Err("invalid shard header".into());
        }
        for id in start..start + count {
            let size = r.u32()?;
            let bytes = r.take(size)?;
            let mut row = Reader { bytes, at: 0 };
            if row.take(prefixes[id].len())? != prefixes[id] {
                return Err("binary public state/edges disagree with native rules".into());
            }
            if let Kind::Decision { actor, children } = &game.nodes[id].kind {
                for (i, combo) in game.boards[game.nodes[id].board].legal[*actor]
                    .iter()
                    .enumerate()
                {
                    if row.u16()? != game.combos[*actor][*combo].key() {
                        return Err("binary combo identity mismatch".into());
                    }
                    let available = row.u8()?;
                    if available > 1 {
                        return Err("invalid EV availability flag".into());
                    }
                    values[id].available[i] = available == 1;
                    for a in 0..children.len() {
                        frozen[id][i * children.len() + a] = row.f32()?;
                    }
                    if available == 1 {
                        for a in 0..children.len() {
                            values[id].cells[i * children.len() + a] = row.f32()?;
                        }
                    }
                }
            }
            if row.at != row.bytes.len() {
                return Err("trailing binary node bytes".into());
            }
        }
        if r.at != bytes.len() {
            return Err("trailing binary shard bytes".into());
        }
    }
    let prob = game.check_probabilities(&frozen)?;
    let (profile, best) = check::verify(&game, &frozen, &guard)?;
    let ev = check::verify_values(&game, &frozen, &values, &guard)?;
    let actual = verification(&manifest.input, profile, best, ev.maximum_error, prob)?;
    if serde_json::to_value(&actual).map_err(|e| e.to_string())?
        != serde_json::to_value(&manifest.verification).map_err(|e| e.to_string())?
    {
        return Err("bundle verification summary mismatch".into());
    }
    let mut d = Sha256::new();
    d.update(b"exact-flop-pilot-f32-policy-v1");
    d.update(manifest.input_sha256.as_bytes());
    for (id, row) in frozen.iter().enumerate() {
        d.update((id as u64).to_le_bytes());
        d.update((row.len() as u64).to_le_bytes());
        for p in row {
            d.update(p.to_le_bytes());
        }
    }
    let policy = format!("{:x}", d.finalize());
    if policy != manifest.frozen_policy_sha256
        || values::fingerprint(&values, &policy) != manifest.frozen_action_values_sha256
    {
        return Err("bundle frozen identity mismatch".into());
    }
    guard.check()?;
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn binary_round_trip_is_immutable_and_rejects_damaged_shards() {
        let input: SpotInput = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../data/solved-spots/pilots/flop-50bb-exact-small.json"
        )))
        .unwrap();
        let directory = std::env::temp_dir().join(format!(
            "poker-flop-bundle-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let name = export(
            input.clone(),
            PilotLimits {
                max_decision_nodes: 10_000,
                max_seconds: 120,
            },
            &directory,
        )
        .unwrap();
        let manifest = verify(&directory.join(&name)).unwrap();
        assert_eq!(manifest.decision_nodes, 9608);
        assert!(manifest.verification.accepted);
        assert!(manifest.shards.iter().all(|s| s.bytes <= MAX_SHARD_BYTES));
        assert!(export(
            input,
            PilotLimits {
                max_decision_nodes: 10_000,
                max_seconds: 120
            },
            &directory
        )
        .is_err());
        let shard = directory.join(format!("{}.bin", manifest.shards[0].sha256));
        let mut bytes = fs::read(&shard).unwrap();
        bytes[0] ^= 1;
        fs::write(&shard, bytes).unwrap();
        assert!(verify(&directory.join(name))
            .unwrap_err()
            .contains("checksum"));
        fs::remove_dir_all(&directory).unwrap();
    }
}
