//! Strict preflop-only inference. Stream and discard postflop nodes and all
//! training state; retain only trained frozen averages. Never invent a mix.
use super::*;

struct FrozenNode {
    descriptor: NodeDescriptor,
    action_labels: Arc<[Arc<str>]>,
    probabilities: Option<Box<[f64]>>,
    unavailable_reason: Option<String>,
}

#[derive(Deserialize)]
struct RawNode {
    descriptor: NodeDescriptor,
    action_labels: Arc<[Arc<str>]>,
    strategy_sum: Box<[f64]>,
    average_visits: u64,
    regret_updates: u64,
}

#[derive(Deserialize)]
struct Checkpoint {
    schema_version: u32,
    model: String,
    approximate: bool,
    config: BlueprintConfig,
    completed_iterations: u64,
    public_histories: BTreeMap<u64, Vec<String>>,
    #[serde(deserialize_with = "preflop_nodes")]
    nodes: BTreeMap<u64, FrozenNode>,
}

pub(super) struct FrozenPreflopPolicy {
    pub game: BlueprintConfig,
    pub rounds: u64,
    nodes: BTreeMap<u64, FrozenNode>,
}

impl FrozenPreflopPolicy {
    pub fn read(path: &Path) -> Result<Self, Box<dyn Error>> {
        let file = fs::File::open(path)?;
        let reader: Box<dyn Read> = if path.extension().is_some_and(|e| e == "gz") {
            Box::new(GzDecoder::new(BufReader::new(file)))
        } else {
            Box::new(BufReader::new(file))
        };
        let checkpoint: Checkpoint = if is_message_pack_checkpoint(path) {
            rmp_serde::from_read(reader)?
        } else {
            serde_json::from_reader(reader)?
        };
        if checkpoint.schema_version != checkpoint.config.checkpoint_schema_version()
            || checkpoint.model != MODEL
            || !checkpoint.approximate
            || checkpoint.completed_iterations > checkpoint.config.iterations
        {
            return Err("incompatible strict preflop checkpoint identity/schema".into());
        }
        checkpoint.config.validate()?;
        for node in checkpoint.nodes.values() {
            if !checkpoint
                .public_histories
                .contains_key(&node.descriptor.public_history_id)
            {
                return Err("strict preflop checkpoint references a missing history".into());
            }
        }
        Ok(Self {
            game: checkpoint.config,
            rounds: checkpoint.completed_iterations,
            nodes: checkpoint.nodes,
        })
    }

    pub fn strategy(&self, state: &GameState, combo: Combo) -> Result<Vec<f64>, String> {
        if state.street != Street::Preflop
            || state.terminal.is_some()
            || state.actor > 1
            || combo.cards()[0] == combo.cards()[1]
            || combo.cards().iter().any(|c| *c >= 52)
        {
            return Err(
                "strict preflop query requires a live preflop state and legal holding".into(),
            );
        }
        // Synthetic completion is used only by the existing observable-key
        // builder. No opponent cards or future board are accepted from callers.
        let deal = neural::deal_for_policy_combo_on_board(combo, state.actor, &[])?;
        let (key, descriptor, _) = information_set(state, &deal, &self.game);
        let node = self
            .nodes
            .get(&key)
            .ok_or("missing frozen preflop information set; no fallback")?;
        let actions = state.legal_actions(&self.game);
        if descriptor != node.descriptor
            || node
                .action_labels
                .iter()
                .map(|a| a.as_ref())
                .ne(actions.iter().map(|a| a.label.as_str()))
        {
            return Err("frozen preflop descriptor/action mismatch".into());
        }
        node.probabilities
            .as_ref()
            .map(|p| p.to_vec())
            .ok_or_else(|| node.unavailable_reason.clone().unwrap())
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

fn preflop_nodes<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<BTreeMap<u64, FrozenNode>, D::Error> {
    struct Visitor;
    impl<'de> serde::de::Visitor<'de> for Visitor {
        type Value = BTreeMap<u64, FrozenNode>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("frozen preflop average nodes")
        }
        fn visit_map<M: serde::de::MapAccess<'de>>(
            self,
            mut map: M,
        ) -> Result<Self::Value, M::Error> {
            let mut nodes = BTreeMap::new();
            let mut interner = NodeStorageInterner::default();
            while let Some((key, mut raw)) = map.next_entry::<u64, RawNode>()? {
                if raw.descriptor.street != Street::Preflop {
                    continue;
                }
                if raw.action_labels.is_empty()
                    || raw.action_labels.len() != raw.strategy_sum.len()
                    || raw.strategy_sum.iter().any(|v| !v.is_finite() || *v < 0.0)
                {
                    return Err(serde::de::Error::custom("invalid frozen preflop vector"));
                }
                raw.descriptor.canonicalize_money();
                raw.descriptor.hand_bucket_trajectory =
                    interner.intern_slice(&raw.descriptor.hand_bucket_trajectory);
                raw.descriptor.public_bucket_trajectory =
                    interner.intern_slice(&raw.descriptor.public_bucket_trajectory);
                raw.action_labels = interner.intern_slice(&raw.action_labels);
                let trained = raw.average_visits > 0
                    && raw.regret_updates > 0
                    && raw.strategy_sum.iter().any(|v| *v > 0.0);
                let unavailable_reason = (!trained).then(|| format!(
                    "untrained frozen preflop information set; average_visits={}, regret_updates={}, strategy_sum={:?}; no fallback",
                    raw.average_visits, raw.regret_updates, raw.strategy_sum));
                let average = AverageNode {
                    descriptor: raw.descriptor,
                    action_labels: raw.action_labels,
                    strategy_sum: raw.strategy_sum,
                    average_visits: raw.average_visits,
                };
                let probabilities = trained.then(|| average.average_strategy().into_boxed_slice());
                let node = FrozenNode {
                    descriptor: average.descriptor,
                    action_labels: average.action_labels,
                    probabilities,
                    unavailable_reason,
                };
                if nodes.insert(key, node).is_some() {
                    return Err(serde::de::Error::custom(
                        "duplicate frozen preflop information set",
                    ));
                }
            }
            Ok(nodes)
        }
    }
    deserializer.deserialize_map(Visitor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_preflop_streaming_preserves_averages_across_all_codecs_and_rejects_missing() {
        let (mut trainer, deal) = super::super::tests::fixture_trainer();
        trainer
            .nodes
            .values_mut()
            .for_each(|n| n.regret_updates = 1);
        let expected = trainer.nodes.values().next().unwrap().average_strategy();
        let mut postflop = trainer.nodes.values().next().unwrap().clone();
        postflop.descriptor.street = Street::Flop;
        trainer.nodes.insert(1, postflop);
        let state = GameState::initial(&trainer.config);
        let combo = Combo::new(deal.holes[0][0], deal.holes[0][1]);
        for codec in ["json", "json.gz", "msgpack", "msgpack.gz"] {
            let path =
                std::env::temp_dir().join(format!("strict-preflop-{}.{codec}", std::process::id()));
            trainer.write_checkpoint(&path).unwrap();
            let loaded = FrozenPreflopPolicy::read(&path).unwrap();
            assert_eq!(loaded.node_count(), 1);
            assert_eq!(loaded.rounds, trainer.completed_iterations);
            assert_eq!(loaded.strategy(&state, combo).unwrap(), expected);
            assert!(loaded.strategy(&state, Combo::new(0, 1)).is_err());
            let mut changed = state.clone();
            changed.invested[0] += 0.1;
            assert!(loaded.strategy(&changed, combo).is_err());
            changed.street = Street::Flop;
            assert!(loaded.strategy(&changed, combo).is_err());
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn strict_preflop_rejects_untrained_and_invalid_source_nodes() {
        let (mut trainer, deal) = super::super::tests::fixture_trainer();
        let path = std::env::temp_dir().join(format!(
            "strict-preflop-invalid-{}.json",
            std::process::id()
        ));
        trainer.write_checkpoint(&path).unwrap();
        let loaded = FrozenPreflopPolicy::read(&path).unwrap();
        assert!(loaded
            .strategy(
                &GameState::initial(&trainer.config),
                Combo::new(deal.holes[0][0], deal.holes[0][1])
            )
            .is_err());
        // Minimized retained-seed failure: regrets were updated, but this
        // holding has no realization-weighted average on a late all-in line.
        // Regret training alone must not turn absent averages into 50/50 play.
        let node = trainer.nodes.values_mut().next().unwrap();
        node.regret_updates = 459;
        node.average_visits = 0;
        node.strategy_sum.fill(0.0);
        trainer.write_checkpoint(&path).unwrap();
        let loaded = FrozenPreflopPolicy::read(&path).unwrap();
        let error = loaded
            .strategy(
                &GameState::initial(&trainer.config),
                Combo::new(deal.holes[0][0], deal.holes[0][1]),
            )
            .unwrap_err();
        assert!(error.contains("average_visits=0, regret_updates=459"));
        trainer.nodes.values_mut().next().unwrap().strategy_sum[0] = -1.0;
        trainer.write_checkpoint(&path).unwrap();
        assert!(FrozenPreflopPolicy::read(&path).is_err());
        fs::remove_file(path).unwrap();
    }
}
