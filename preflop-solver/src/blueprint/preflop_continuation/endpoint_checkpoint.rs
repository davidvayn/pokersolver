//! Immutable, checksummed endpoint values. A crash must not discard a whole
//! board or tempt recovery to fabricate CFVs from timing-only progress logs.
use super::*;

#[derive(Serialize, Deserialize)]
pub(super) struct Endpoint {
    pub identity: serde_json::Value,
    pub history: History,
    pub values: [Vec<f64>; 2],
    pub record: serde_json::Value,
}

fn validate(value: &Endpoint, identity: &serde_json::Value, history: &History) -> Result<(), String> {
    if &value.identity != identity || &value.history != history
        || value.record["history"] != serde_json::json!(history)
        || value.values.iter().any(|p| p.len()!=1326 || p.iter().any(|x| !x.is_finite())) {
        return Err("endpoint checkpoint identity, history or values differ".into());
    }
    Ok(())
}

pub(super) fn path(root: &Path, history: &History) -> PathBuf {
    root.join(format!("{:x}.json",Sha256::digest(serde_json::to_vec(history).unwrap())))
}

pub(super) fn save(root: &Path, value: &Endpoint) -> Result<(), String> {
    validate(value, &value.identity, &value.history)?;
    fs::create_dir_all(root).map_err(|e| e.to_string())?;
    let destination=path(root,&value.history);
    if destination.exists() { return Err("refuse to overwrite completed endpoint".into()); }
    // Hash the exact serialized body, avoiding parser/serialization ambiguity.
    // serde_json's float_roundtrip feature preserves every CFV bit on restore.
    let body=serde_json::to_string(value).map_err(|e| e.to_string())?;
    let checksum=format!("{:x}",Sha256::digest(body.as_bytes()));
    write_json_atomic(&destination,&serde_json::json!({
        "schema":"frozen-response-endpoint-v1","sha256":checksum,"body":body
    })).map_err(|e| e.to_string())?;
    fs::File::open(&destination).and_then(|f| f.sync_all()).map_err(|e| e.to_string())?;
    fs::File::open(root).and_then(|f| f.sync_all()).map_err(|e| e.to_string())
}

pub(super) fn load(root: &Path, identity: &serde_json::Value, history: &History) -> Result<Option<Endpoint>, String> {
    let source=path(root,history);
    if !source.exists() { return Ok(None); }
    if fs::metadata(&source).map_err(|e| e.to_string())?.len()>512*1024 {
        return Err("oversized endpoint checkpoint".into());
    }
    let envelope: serde_json::Value=serde_json::from_slice(&fs::read(source).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let body=envelope["body"].as_str().ok_or("missing endpoint body")?;
    if envelope["schema"]!="frozen-response-endpoint-v1"
        || envelope["sha256"]!=format!("{:x}",Sha256::digest(body.as_bytes())) {
        return Err("endpoint checkpoint checksum mismatch".into());
    }
    let value: Endpoint=serde_json::from_str(body).map_err(|e| e.to_string())?;
    validate(&value,identity,history)?;
    Ok(Some(value))
}

#[test]
fn endpoint_checkpoint_preserves_bits_and_rejects_corruption_and_changed_policy() {
    let dir=std::env::temp_dir().join(format!("poker-endpoint-{}",std::process::id()));
    assert!(!dir.exists());
    let history=vec!["call".into()];
    let identity=serde_json::json!({"policy":"pinned","board":[0,5,10]});
    let values=std::array::from_fn(|p| (0..1326).map(|c| (c as f64+0.1)/7.0*(if p==0 {1.0} else {-1.0})).collect());
    let value=Endpoint{identity:identity.clone(),history:history.clone(),values,
        record:serde_json::json!({"history":history,"seconds":1.0})};
    assert!(load(&dir,&identity,&history).unwrap().is_none());
    save(&dir,&value).unwrap();
    assert_eq!(load(&dir,&identity,&history).unwrap().unwrap().values,value.values);
    assert!(save(&dir,&value).is_err());
    assert!(load(&dir,&serde_json::json!({"policy":"changed"}),&history).is_err());
    let destination=path(&dir,&history);
    let mut envelope: serde_json::Value=serde_json::from_slice(&fs::read(&destination).unwrap()).unwrap();
    envelope["body"]=serde_json::json!("corrupted");
    write_json_atomic(&destination,&envelope).unwrap();
    assert!(load(&dir,&identity,&history).is_err());
    fs::remove_dir_all(dir).unwrap();
}
