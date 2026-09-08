//! Local training recovery only. These files must never enter serving exports.
use super::*;

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Receipt {
    schema: String,
    identity: serde_json::Value,
    checkpoint: String,
    checkpoint_sha256: String,
    completed_iterations: u64,
    progress: Vec<serde_json::Value>,
}

pub(super) fn save(trainer: &Trainer, directory: &Path, identity: &serde_json::Value,
    progress: &[serde_json::Value]) -> Result<PathBuf, String> {
    let round = trainer.completed_iterations;
    validate_progress(progress, round)?;
    fs::create_dir_all(directory).map_err(|e| e.to_string())?;
    let name = format!("round{round:04}.mpk.gz");
    let state = directory.join(&name);
    let receipt = directory.join(format!("round{round:04}.json"));
    if state.exists() || receipt.exists() { return Err("refuse to overwrite checkpoint generation".into()); }
    trainer.write_checkpoint(&state).map_err(|e| e.to_string())?;
    let record = Receipt { schema: "compact-preflop-checkpoint-v1".into(),
        identity: identity.clone(), checkpoint: name,
        checkpoint_sha256: format!("{:x}", Sha256::digest(fs::read(&state).map_err(|e| e.to_string())?)),
        completed_iterations: round, progress: progress.to_vec() };
    // Immutable generation: an interrupted state/receipt write cannot replace
    // the previous complete checkpoint. Only complete receipts are resumable.
    write_json_atomic(&receipt, &record).map_err(|e| e.to_string())?;
    Ok(receipt)
}

fn validate_progress(progress: &[serde_json::Value], round: u64) -> Result<(), String> {
    if round == 0 || progress.len() as u64 != round || progress.iter().enumerate()
        .any(|(i, row)| row["round"].as_u64() != Some(i as u64+1)) {
        return Err("checkpoint progress is not a complete contiguous prefix".into());
    }
    Ok(())
}

pub(super) fn restore(path: &Path, expected_sha: &str, identity: &serde_json::Value,
    config: &BlueprintConfig) -> Result<(Trainer, Vec<serde_json::Value>), String> {
    if fs::metadata(path).map_err(|e| e.to_string())?.len() > 32*1024*1024 {
        return Err("oversized compact checkpoint receipt".into());
    }
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    if format!("{:x}", Sha256::digest(&bytes)) != expected_sha {
        return Err("checkpoint receipt hash mismatch".into());
    }
    let receipt: Receipt = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    if receipt.schema != "compact-preflop-checkpoint-v1" || &receipt.identity != identity
        || receipt.checkpoint != format!("round{:04}.mpk.gz", receipt.completed_iterations) {
        return Err("checkpoint continuation identity or generation differs".into());
    }
    validate_progress(&receipt.progress, receipt.completed_iterations)?;
    let state = path.parent().ok_or("checkpoint receipt has no parent")?.join(&receipt.checkpoint);
    if fs::metadata(&state).map_err(|e| e.to_string())?.len() > 32*1024*1024
        || format!("{:x}", Sha256::digest(fs::read(&state).map_err(|e| e.to_string())?)) != receipt.checkpoint_sha256 {
        return Err("checkpoint state size/hash mismatch".into());
    }
    let saved = read_checkpoint(&state).map_err(|e| e.to_string())?;
    // The generic loader fills a missing legacy sampled-deal count from the
    // iteration count. Compact exact-private integration legitimately has zero
    // sampled private deals; retain that exact counter for replay/provenance.
    let sampled_deals = saved.sampled_deals;
    let mut trainer = Trainer::from_checkpoint(saved, config)?;
    trainer.sampled_deals = sampled_deals;
    if trainer.completed_iterations != receipt.completed_iterations || trainer.nodes.len() != 16900 {
        return Err("checkpoint state/progress mismatch".into());
    }
    Ok((trainer, receipt.progress))
}

#[test]
fn compact_checkpoint_replays_discounts_averages_rng_and_rejects_changed_oracle() {
    let config = BlueprintConfig { seed:27001, effective_stack_bb:20.0, iterations:8,
        averaging_delay:0, exact_preflop_averaging:true, max_information_sets:20000,
        traversal:BlueprintTraversal::PublicChanceSampling, ..BlueprintConfig::default() };
    let identity = serde_json::json!({"modelSha256":"fixed-model","proposalSha256":"fixed-proposal"});
    let mut trainer = Trainer::fresh(config.clone());
    let update = |trainer: &mut Trainer| {
        let snapshot = begin_update(trainer).unwrap();
        let draw = trainer.rng.next_f64();
        let values = snapshot.endpoints.keys().enumerate().map(|(i,h)| (h.clone(),
            [vec![draw*i as f64/100.0;1326],vec![-draw*i as f64/100.0;1326]])).collect();
        apply_snapshot_updates(trainer,&snapshot,&values,false).unwrap();
        trainer.completed_iterations+=1;
    };
    for _ in 0..4 { update(&mut trainer); }
    let dir = std::env::temp_dir().join(format!("compact-checkpoint-test-{}",std::process::id()));
    assert!(!dir.exists());
    let progress: Vec<_> = (1..=4).map(|n| serde_json::json!({"round":n})).collect();
    let receipt = save(&trainer,&dir,&identity,&progress).unwrap();
    let digest = format!("{:x}",Sha256::digest(fs::read(&receipt).unwrap()));
    assert!(save(&trainer,&dir,&identity,&progress).is_err());
    assert!(restore(&receipt,"wrong",&identity,&config).is_err());
    assert!(restore(&receipt,&digest,&serde_json::json!({"modelSha256":"changed"}),&config).is_err());
    let (mut resumed, restored_progress) = restore(&receipt,&digest,&identity,&config).unwrap();
    assert_eq!(progress,restored_progress);
    for _ in 4..8 { update(&mut trainer); update(&mut resumed); }
    let first = dir.join("continuous.mpk.gz"); let second = dir.join("resumed.mpk.gz");
    trainer.write_checkpoint(&first).unwrap(); resumed.write_checkpoint(&second).unwrap();
    assert!(fs::read(first).unwrap()==fs::read(second).unwrap(),"resumed checkpoint bytes differ");
    fs::remove_dir_all(dir).unwrap();
}

/// Read-only diagnostic for an existing local training state, not a policy export.
fn root_diagnostic(path: &Path, digest: &str) -> Result<serde_json::Value, String> {
    if fs::metadata(path).map_err(|e| e.to_string())?.len() > 32*1024*1024
        || format!("{:x}", Sha256::digest(fs::read(path).map_err(|e| e.to_string())?)) != digest {
        return Err("diagnostic checkpoint size/hash mismatch".into());
    }
    let saved = read_checkpoint(path).map_err(|e| e.to_string())?;
    let config = saved.config.clone();
    let trainer = Trainer::from_checkpoint(saved, &config)?;
    let mut rows = Vec::new();
    for node in trainer.nodes.values() {
        let history = trainer.public_histories.get(&node.descriptor.public_history_id)
            .ok_or("missing diagnostic history")?;
        if history.len() != 1 { continue; }
        rows.push(serde_json::json!({"hand":node.descriptor.hand_bucket_trajectory,
            "actions":node.action_labels,"regrets":node.regrets,
            "strategySum":node.strategy_sum,"average":node.average_strategy(),
            "current":node.current_strategy(),"averageVisits":node.average_visits}));
    }
    if rows.len() != 169 { return Err("diagnostic requires all169 root classes".into()); }
    rows.sort_by(|a,b| a["hand"].to_string().cmp(&b["hand"].to_string()));
    Ok(serde_json::json!({"schema":"compact-checkpoint-root-diagnostic-v1",
        "checkpointSha256":digest,"round":trainer.completed_iterations,
        "seed":config.seed,"rows":rows,"nativeQueries":0,"releaseAccepted":false}))
}

#[test]
#[ignore = "explicit hash-pinned local checkpoint diagnostic; no training"]
fn compact_checkpoint_root_diagnostic() {
    let path = PathBuf::from(std::env::var("POKER_COMPACT_DIAGNOSTIC_CHECKPOINT").unwrap());
    let digest = std::env::var("POKER_COMPACT_DIAGNOSTIC_CHECKPOINT_SHA").unwrap();
    let output = PathBuf::from(std::env::var("POKER_COMPACT_DIAGNOSTIC_OUTPUT").unwrap());
    assert!(!output.exists(), "refuse to overwrite diagnostic");
    assert!(root_diagnostic(&path, "incorrect-hash").is_err());
    let report = root_diagnostic(&path, &digest).unwrap();
    for row in report["rows"].as_array().unwrap() {
        for field in ["average", "current"] {
            let values = row[field].as_array().unwrap();
            assert!((values.iter().map(|v| v.as_f64().unwrap()).sum::<f64>()-1.0).abs()<1e-12);
        }
    }
    write_json_atomic(&output, &report).unwrap();
}
