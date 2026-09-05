//! Inference-only view of schema-5 checkpoints. Deserialize frozen average
//! accumulators directly; never allocate regrets, resumable state, or a second
//! expanded table. This reader cannot produce a training checkpoint.
use super::*;

#[derive(Deserialize)]
pub(super) struct AverageNode {
    pub descriptor: NodeDescriptor,
    pub action_labels: Arc<[Arc<str>]>,
    pub strategy_sum: Box<[f64]>,
    pub average_visits: u64,
}

impl AverageNode {
    pub fn average_strategy(&self) -> Vec<f64> {
        // Preserve Node::average_strategy's exact f64 operation order.
        let mut strategy = self.strategy_sum.to_vec();
        let maximum = strategy.iter().copied().fold(0.0f64, f64::max);
        if maximum > 0.0 {
            for probability in &mut strategy {
                *probability /= maximum;
            }
            let total = strategy.iter().sum::<f64>();
            for probability in &mut strategy {
                *probability /= total;
            }
            strategy
        } else {
            normalize_or_uniform(strategy)
        }
    }
}

#[cfg(test)]
impl From<Node> for AverageNode {
    fn from(node: Node) -> Self {
        Self {
            descriptor: node.descriptor,
            action_labels: node.action_labels,
            strategy_sum: node.strategy_sum,
            average_visits: node.average_visits,
        }
    }
}

#[derive(Deserialize)]
struct PolicyCheckpoint {
    schema_version: u32,
    model: String,
    approximate: bool,
    config: BlueprintConfig,
    completed_iterations: u64,
    public_histories: BTreeMap<u64, Vec<String>>,
    #[serde(deserialize_with = "average_nodes")]
    nodes: BTreeMap<u64, AverageNode>,
}

pub(super) struct InferenceTable {
    pub config: BlueprintConfig,
    pub rounds: u64,
    pub nodes: BTreeMap<u64, AverageNode>,
}

impl InferenceTable {
    pub fn read(path: &Path) -> Result<Self, Box<dyn Error>> {
        let file = fs::File::open(path)?;
        let reader: Box<dyn Read> = if path.extension().is_some_and(|e| e == "gz") {
            Box::new(GzDecoder::new(BufReader::new(file)))
        } else {
            Box::new(BufReader::new(file))
        };
        let checkpoint: PolicyCheckpoint = if is_message_pack_checkpoint(path) {
            rmp_serde::from_read(reader)?
        } else {
            serde_json::from_reader(reader)?
        };
        if checkpoint.schema_version != BLUEPRINT_CHECKPOINT_SCHEMA_VERSION
            || checkpoint.model != MODEL
            || !checkpoint.approximate
            || checkpoint.completed_iterations > checkpoint.config.iterations
        {
            return Err("incompatible frozen-policy checkpoint identity/schema".into());
        }
        checkpoint.config.validate()?;
        for node in checkpoint.nodes.values() {
            if !checkpoint
                .public_histories
                .contains_key(&node.descriptor.public_history_id)
            {
                return Err("frozen-policy checkpoint references a missing history".into());
            }
        }
        Ok(Self {
            config: checkpoint.config,
            rounds: checkpoint.completed_iterations,
            nodes: checkpoint.nodes,
        })
    }

    #[cfg(test)]
    pub fn from_trainer(trainer: Trainer) -> Self {
        Self {
            config: trainer.config,
            rounds: trainer.completed_iterations,
            nodes: trainer
                .nodes
                .into_iter()
                .map(|(key, node)| (key, node.into()))
                .collect(),
        }
    }
}

fn average_nodes<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<BTreeMap<u64, AverageNode>, D::Error> {
    struct Visitor;
    impl<'de> serde::de::Visitor<'de> for Visitor {
        type Value = BTreeMap<u64, AverageNode>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("unique frozen average policy nodes")
        }
        fn visit_map<M: serde::de::MapAccess<'de>>(
            self,
            mut map: M,
        ) -> Result<Self::Value, M::Error> {
            let mut nodes = BTreeMap::new();
            let mut interner = NodeStorageInterner::default();
            while let Some((key, mut node)) = map.next_entry::<u64, AverageNode>()? {
                if node.action_labels.is_empty()
                    || node.action_labels.len() != node.strategy_sum.len()
                    || node.strategy_sum.iter().any(|v| !v.is_finite() || *v < 0.0)
                {
                    return Err(serde::de::Error::custom(
                        "invalid frozen average strategy vector",
                    ));
                }
                node.descriptor.canonicalize_money();
                node.descriptor.hand_bucket_trajectory =
                    interner.intern_slice(&node.descriptor.hand_bucket_trajectory);
                node.descriptor.public_bucket_trajectory =
                    interner.intern_slice(&node.descriptor.public_bucket_trajectory);
                node.action_labels = interner.intern_slice(&node.action_labels);
                if nodes.insert(key, node).is_some() {
                    return Err(serde::de::Error::custom("duplicate frozen information set"));
                }
            }
            Ok(nodes)
        }
    }
    deserializer.deserialize_map(Visitor)
}
