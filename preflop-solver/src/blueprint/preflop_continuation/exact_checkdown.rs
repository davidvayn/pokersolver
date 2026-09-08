//! Exact preflop checkdown expectation for suit-invariant preflop classes.
//! Sum integer showdown units over every canonical flop orbit. This is a
//! chance-control-variate/payoff kernel, never a strategic policy or model.
use super::*;
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use std::io::Read;

fn classes() -> (Vec<String>, Vec<usize>, Vec<u64>, Vec<u64>) {
    let combos = all_combos();
    let labels: Vec<_> = combos
        .iter()
        .map(|c| c.label())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let index: Vec<_> = combos
        .iter()
        .map(|c| labels.binary_search(&c.label()).unwrap())
        .collect();
    let mut multiplicities = vec![0; 169];
    let mut pairs = vec![0; 169 * 169];
    for (i, first) in combos.iter().enumerate() {
        multiplicities[index[i]] += 1;
        for (j, second) in combos.iter().enumerate() {
            if !first.overlaps(*second) {
                pairs[index[i] * 169 + index[j]] += 1;
            }
        }
    }
    (labels, index, multiplicities, pairs)
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ExactCheckdown {
    schema: String,
    classes: Vec<String>,
    class_multiplicities: Vec<u64>,
    compatible_pairs: Vec<u64>,
    weighted_flop_pairs: Vec<u64>,
    weighted_win_units: Vec<u64>,
    canonical_flops: usize,
    raw_flops: usize,
    complete: bool,
}

impl ExactCheckdown {
    pub(super) fn read(path: &Path, expected_sha256: &str) -> Result<Self, String> {
        if fs::metadata(path).map_err(|e| e.to_string())?.len() > 4 * 1024 * 1024 {
            return Err("oversized exact preflop kernel".into());
        }
        let bytes = fs::read(path).map_err(|e| e.to_string())?;
        if bytes.len() > 4 * 1024 * 1024
            || format!("{:x}", Sha256::digest(&bytes)) != expected_sha256
        {
            return Err("exact preflop kernel size/hash mismatch".into());
        }
        let mut decoded = Vec::new();
        GzDecoder::new(bytes.as_slice())
            .take(4 * 1024 * 1024 + 1)
            .read_to_end(&mut decoded)
            .map_err(|e| e.to_string())?;
        if decoded.len() > 4 * 1024 * 1024 {
            return Err("oversized decoded preflop kernel".into());
        }
        let kernel: Self = serde_json::from_slice(&decoded).map_err(|e| e.to_string())?;
        kernel.validate()?;
        Ok(kernel)
    }

    fn validate(&self) -> Result<(), String> {
        let (classes, _, multiplicities, pairs) = classes();
        if self.schema != "exact-preflop-checkdown-class-kernel-v1"
            || !self.complete
            || self.canonical_flops != 1755
            || self.raw_flops != 22100
            || self.classes != classes
            || self.class_multiplicities != multiplicities
            || self.compatible_pairs != pairs
            || self.weighted_flop_pairs.len() != 169 * 169
            || self.weighted_win_units.len() != 169 * 169
        {
            return Err("incomplete or incompatible exact preflop checkdown kernel".into());
        }
        for a in 0..169 {
            for b in 0..169 {
                let k = a * 169 + b;
                let pair_flops = pairs[k] * 17296;
                if self.weighted_flop_pairs[k] != pair_flops
                    || self.weighted_win_units[k] > pair_flops * 1980
                    || self.weighted_win_units[k] + self.weighted_win_units[b * 169 + a]
                        != pair_flops * 1980
                {
                    return Err("exact kernel card-removal/orbit/zero-sum count mismatch".into());
                }
            }
        }
        Ok(())
    }

    /// Exact class-conditional CFVs averaged over the 1,225 legal opponent
    /// holdings. Both ranges must be constant within each preflop hand class.
    pub(super) fn values(
        &self,
        invested: [f64; 2],
        ranges: &[Vec<f64>; 2],
    ) -> Result<[Vec<f64>; 2], String> {
        if ranges
            .iter()
            .any(|r| r.len() != 1326 || r.iter().any(|v| !v.is_finite() || *v < 0.0))
            || invested.iter().any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err("invalid exact preflop payoff input".into());
        }
        let mut by_class = [vec![None; 169], vec![None; 169]];
        let index: Vec<_> = all_combos()
            .iter()
            .map(|c| self.classes.binary_search(&c.label()).unwrap())
            .collect();
        for p in 0..2 {
            for c in 0..1326 {
                let slot = &mut by_class[p][index[c]];
                if slot.is_some_and(|v| v != ranges[p][c]) {
                    return Err("preflop range is not class invariant".into());
                }
                *slot = Some(ranges[p][c]);
            }
        }
        let mut values = [vec![0.0; 169], vec![0.0; 169]];
        for p in 0..2 {
            for a in 0..169 {
                for b in 0..169 {
                    let k = a * 169 + b;
                    let wins = self.weighted_win_units[k] as f64 / (17296.0 * 1980.0);
                    values[p][a] += by_class[1 - p][b].unwrap()
                        * ((invested[0] + invested[1]) * wins
                            - invested[p] * self.compatible_pairs[k] as f64)
                        / (self.class_multiplicities[a] as f64 * OPPONENT_HANDS);
                }
            }
        }
        Ok(std::array::from_fn(|p| {
            index.iter().map(|c| values[p][*c]).collect()
        }))
    }
}

#[test]
#[ignore = "bounded exact-showdown orbit chunk; explicit indices, new output and resource guard required"]
fn exact_preflop_checkdown_chunk() {
    let indices: Vec<usize> =
        serde_json::from_str(&std::env::var("POKER_EXACT_ORBITS").unwrap()).unwrap();
    assert!(!indices.is_empty() && indices.len() <= 16 && indices.windows(2).all(|v| v[0] < v[1]));
    let orbits = preflop::canonical_flop_orbits();
    assert_eq!(orbits.len(), 1755);
    assert_eq!(orbits.iter().map(|o| o.orbit_size).sum::<usize>(), 22100);
    let (labels, index, multiplicities, pairs) = classes();
    let mut wins = vec![0u64; 169 * 169];
    let mut counts = vec![0u64; 169 * 169];
    let mut records = Vec::new();
    let started = Instant::now();
    for i in indices {
        let orbit = &orbits[i];
        let before = Instant::now();
        let kernel = exact_flop_kernel(orbit.board).unwrap();
        let (w, n) = kernel.class_totals(&index, 169).unwrap();
        for a in 0..169 {
            for b in 0..169 {
                let k = a * 169 + b;
                assert_eq!(n[k], n[b * 169 + a]);
                assert_eq!(w[k] + w[b * 169 + a], n[k] * 1980);
                wins[k] += w[k] * orbit.orbit_size as u64;
                counts[k] += n[k] * orbit.orbit_size as u64;
            }
        }
        let record = serde_json::json!({"index":i,"board":orbit.board,"orbitSize":orbit.orbit_size,
            "seconds":before.elapsed().as_secs_f64(),
            "integerCountsSha256":format!("{:x}",Sha256::digest(rmp_serde::to_vec(&(w,n)).unwrap()))});
        eprintln!("{record}");
        records.push(record);
    }
    let output = PathBuf::from(std::env::var("POKER_EXACT_OUTPUT").unwrap());
    let payload = serde_json::json!({"schema":"exact-preflop-checkdown-orbit-chunk-v1",
        "classes":labels,"classMultiplicities":multiplicities,"compatiblePairs":pairs,
        "weightedWinUnits":wins,"weightedFlopPairs":counts,"orbits":records,
        "seconds":started.elapsed().as_secs_f64(),"complete":false,"releaseAccepted":false,
        "interpretation":"Partial integer payoff counts only, never a usable kernel or a strategy."});
    let bytes = serde_json::to_vec(&payload).unwrap();
    assert!(bytes.len() < 4 * 1024 * 1024);
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap();
    let mut gzip = GzEncoder::new(file, Compression::default());
    gzip.write_all(&bytes).unwrap();
    gzip.finish().unwrap().sync_all().unwrap();
}

#[test]
fn exact_checkdown_counts_enforce_complete_chance_and_class_invariant_ranges() {
    let (classes, _, multiplicities, pairs) = classes();
    // Synthetic all-tie game tests accounting independently of poker equity.
    let mut kernel = ExactCheckdown {
        schema: "exact-preflop-checkdown-class-kernel-v1".into(),
        classes,
        class_multiplicities: multiplicities,
        compatible_pairs: pairs.clone(),
        weighted_flop_pairs: pairs.iter().map(|v| v * 17296).collect(),
        weighted_win_units: pairs.iter().map(|v| v * 17296 * 990).collect(),
        canonical_flops: 1755,
        raw_flops: 22100,
        complete: true,
    };
    kernel.validate().unwrap();
    let mut ranges = [vec![1.0; 1326], vec![1.0; 1326]];
    let values = kernel.values([1.0, 3.0], &ranges).unwrap();
    assert!(values[0].iter().all(|v| (v - 1.0).abs() < 1e-12));
    assert!(values[1].iter().all(|v| (v + 1.0).abs() < 1e-12));
    ranges[0][Combo::new(51, 50).key()] = 0.0;
    assert!(kernel.values([1.0, 3.0], &ranges).is_err());
    kernel.weighted_flop_pairs[0] += 1;
    assert!(kernel.validate().is_err());
    kernel.weighted_flop_pairs[0] -= 1;
    kernel.complete = false;
    assert!(kernel.validate().is_err());
}
