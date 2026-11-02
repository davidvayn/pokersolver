//! Conditional action values of the frozen policy, not best-response labels.
use super::*;

pub(super) struct NodeValues {
    pub cells: Vec<f32>,
    pub available: Vec<bool>,
}

pub(super) fn collect(
    game: &Game,
    frozen: &[Vec<f32>],
    guard: &Guard,
) -> Result<Vec<NodeValues>, String> {
    let mut values = game
        .nodes
        .iter()
        .map(|node| {
            let rows = match &node.kind {
                Kind::Decision { actor, .. } => game.boards[node.board].legal[*actor].len(),
                _ => 0,
            };
            NodeValues {
                cells: vec![0.0; node.sums.len()],
                available: vec![false; rows],
            }
        })
        .collect::<Vec<_>>();
    for hero in 0..2 {
        game.profile_walk(
            game.root,
            game.prior.clone(),
            frozen,
            hero,
            guard,
            Some(&mut values),
        )?;
    }
    Ok(values)
}

pub(super) fn fingerprint(values: &[NodeValues], policy_sha256: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(b"exact-flop-f32-action-values-v1");
    digest.update(policy_sha256.as_bytes());
    for (id, node) in values.iter().enumerate() {
        digest.update((id as u64).to_le_bytes());
        digest.update((node.available.len() as u64).to_le_bytes());
        for available in &node.available {
            digest.update([u8::from(*available)]);
        }
        for value in &node.cells {
            digest.update(value.to_le_bytes());
        }
    }
    format!("{:x}", digest.finalize())
}
