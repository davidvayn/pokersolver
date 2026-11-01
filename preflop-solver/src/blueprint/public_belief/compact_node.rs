//! Lossless accumulator storage for declared-range saved turn studies.
//! Only root-zero or board-blocked rows are omitted, never low-reach rows.
//! Canonical combo IDs and dense traversal/export vectors remain unchanged.
use super::*;

const ABSENT: u16 = u16::MAX;

pub(super) struct ComboLayout {
    dense_to_row: Vec<u16>,
    count: usize,
}

impl ComboLayout {
    pub(super) fn new(legal: &[bool]) -> Self {
        assert_eq!(legal.len(), COMBO_COUNT);
        let mut count = 0;
        let dense_to_row = legal
            .iter()
            .map(|present| {
                if *present {
                    let row = u16::try_from(count).expect("combo index fits u16");
                    assert_ne!(row, ABSENT);
                    count += 1;
                    row
                } else {
                    ABSENT
                }
            })
            .collect();
        Self {
            dense_to_row,
            count,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.count
    }

    pub(super) fn row(&self, combo: usize) -> Option<usize> {
        self.dense_to_row
            .get(combo)
            .copied()
            .filter(|row| *row != ABSENT)
            .map(usize::from)
    }
}

/// Shared by all histories with the same public board and actor. A layout
/// depends on the declared root range, NOT the current policy's action reach.
pub(super) struct CompactNodeLayouts {
    turn: [Arc<ComboLayout>; 2],
    river: Vec<Option<[Arc<ComboLayout>; 2]>>,
}

impl CompactNodeLayouts {
    fn new(turn: &[Vec<bool>; 2], river_data: &[Option<RiverBoardData>]) -> Self {
        Self {
            turn: std::array::from_fn(|actor| Arc::new(ComboLayout::new(&turn[actor]))),
            river: river_data
                .iter()
                .map(|data| {
                    data.as_ref().map(|data| {
                        std::array::from_fn(|actor| Arc::new(ComboLayout::new(&data.legal[actor])))
                    })
                })
                .collect(),
        }
    }

    pub(super) fn for_node(&self, river: Option<u8>, actor: usize) -> &Arc<ComboLayout> {
        match river {
            None => &self.turn[actor],
            Some(card) => &self.river[card as usize]
                .as_ref()
                .expect("legal public river has a combo layout")[actor],
        }
    }
}

impl TurnRiverSolver {
    /// Opt-in for offline saved studies only. Existing live/safe/prior paths
    /// keep the dense default and cannot retrofit compact storage mid-run.
    pub(super) fn enable_compact_nodes(&mut self) -> Result<(), String> {
        if !self.nodes.is_empty()
            || self.safe_root.is_some()
            || self.training_round_offset != 0
            || self.compact_node_layouts.is_some()
        {
            return Err("compact nodes require a fresh unsafed turn-river solver".into());
        }
        self.compact_node_layouts = Some(CompactNodeLayouts::new(&self.legal, &self.river_data));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_combo_layout_preserves_noncontiguous_canonical_ids() {
        let mut legal = vec![false; COMBO_COUNT];
        for combo in [0, 17, COMBO_COUNT - 1] {
            legal[combo] = true;
        }
        let layout = ComboLayout::new(&legal);
        assert_eq!(layout.len(), 3);
        assert_eq!(layout.row(0), Some(0));
        assert_eq!(layout.row(17), Some(1));
        assert_eq!(layout.row(COMBO_COUNT - 1), Some(2));
        assert_eq!(layout.row(1), None);
        assert_eq!(layout.row(COMBO_COUNT), None);
        let empty = ComboLayout::new(&vec![false; COMBO_COUNT]);
        assert_eq!(empty.len(), 0);
        assert!((0..COMBO_COUNT).all(|combo| empty.row(combo).is_none()));
    }

    #[test]
    fn compact_nodes_preserve_discounting_and_dense_strategy_vectors_exactly() {
        let legal: Vec<bool> = (0..COMBO_COUNT).map(|combo| combo % 7 == 0).collect();
        let actions = vec![
            LegalAction {
                label: "fold".into(),
                kind: ActionKind::Fold,
            },
            LegalAction {
                label: "check".into(),
                kind: ActionKind::Check,
            },
        ];
        let mut dense = RangeNode::new(0, &actions);
        let mut compact =
            RangeNode::new_with_layout(0, &actions, Some(Arc::new(ComboLayout::new(&legal))));
        for combo in 0..COMBO_COUNT {
            if !legal[combo] {
                continue;
            }
            let a = dense.storage_offset(combo);
            let b = compact.storage_offset(combo);
            let regrets = [combo as f64 - 42.5, 21.25 - combo as f64];
            let sums = if combo % 14 == 0 {
                [0.0, 0.0]
            } else {
                [0.125, 0.875]
            };
            dense.regrets[a..a + 2].copy_from_slice(&regrets);
            compact.regrets[b..b + 2].copy_from_slice(&regrets);
            dense.strategy_sum[a..a + 2].copy_from_slice(&sums);
            compact.strategy_sum[b..b + 2].copy_from_slice(&sums);
        }
        let parameters = BlueprintConfig::default().dcfr;
        for round in [0, 1, 2, 2, 17] {
            dense.discount_regrets(round, &parameters);
            compact.discount_regrets(round, &parameters);
            dense.finish_average_update(round, &parameters);
            compact.finish_average_update(round, &parameters);
            assert_eq!(dense.strategy(&legal), compact.strategy(&legal));
            assert_eq!(
                dense.average_strategy(&legal),
                compact.average_strategy(&legal)
            );
            for combo in 0..COMBO_COUNT {
                if !legal[combo] {
                    continue;
                }
                let a = dense.storage_offset(combo);
                let b = compact.storage_offset(combo);
                assert_eq!(dense.regrets[a..a + 2], compact.regrets[b..b + 2]);
                assert_eq!(dense.strategy_sum[a..a + 2], compact.strategy_sum[b..b + 2]);
            }
        }
    }
}
