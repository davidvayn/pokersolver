//! Public-endpoint sampling only: no change to regrets, value targets or cards.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EndpointSampling {
    UniformOne,
    UniformBatch,
    RootStratified,
    OpponentReach,
    FixedImportance,
}

impl EndpointSampling {
    pub(super) fn parse(value: &str) -> Result<Self, String> {
        match value {
            "uniform_one" => Ok(Self::UniformOne),
            "uniform_batch" => Ok(Self::UniformBatch),
            "root_stratified" => Ok(Self::RootStratified),
            "opponent_reach" => Ok(Self::OpponentReach),
            "fixed_importance" => Ok(Self::FixedImportance),
            _ => Err("unknown compact endpoint sampling scheme".into()),
        }
    }

    pub(super) fn label(self) -> &'static str {
        match self {
            Self::UniformOne => "uniform_one",
            Self::UniformBatch => "uniform_batch",
            Self::RootStratified => "root_stratified",
            Self::OpponentReach => "opponent_reach",
            Self::FixedImportance => "fixed_importance",
        }
    }

    pub(super) fn select(
        self,
        live: &[History],
        root: &[String],
        rng: &mut SplitMix64,
    ) -> Result<BTreeMap<History, f64>, String> {
        if matches!(self, Self::OpponentReach | Self::FixedImportance) {
            return Err("weighted sampler requires its explicit proposal".into());
        }
        if live.is_empty()
            || live.iter().collect::<BTreeSet<_>>().len() != live.len()
            || live
                .iter()
                .any(|h| !h.starts_with(root) || h.len() <= root.len())
        {
            return Err("invalid live preflop endpoint population".into());
        }
        if self == Self::UniformOne {
            // Preserve the old draw and its exact probability/RNG stream.
            return Ok(BTreeMap::from([(
                live[rng.index(live.len())].clone(),
                1.0 / live.len() as f64,
            )]));
        }
        let mut groups: BTreeMap<&str, Vec<&History>> = BTreeMap::new();
        for history in live {
            groups
                .entry(&history[root.len()])
                .or_default()
                .push(history);
        }
        if self == Self::UniformBatch {
            // Compute-matched control: sample the same number of endpoints
            // without replacement, but without guaranteeing each root action.
            let count = groups.len();
            let mut available: Vec<_> = (0..live.len()).collect();
            let mut selected = BTreeMap::new();
            for _ in 0..count {
                let index = rng.index(available.len());
                selected.insert(
                    live[available.swap_remove(index)].clone(),
                    count as f64 / live.len() as f64,
                );
            }
            return Ok(selected);
        }
        // Independent fresh draws within this iteration. Never cycle through
        // a persistent shuffled list while the strategy adapts between draws.
        Ok(groups
            .values()
            .map(|group| {
                (
                    group[rng.index(group.len())].clone(),
                    1.0 / group.len() as f64,
                )
            })
            .collect())
    }
}

/// Frozen before training, independent of every new chance draw. Full support
/// and exact q correction preserve the conditional expectation even when this
/// fitted allocation is a poor match for a later training policy.
#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct FixedProposal {
    schema: String,
    live_histories: Vec<History>,
    probabilities_by_actor: [Vec<f64>; 2],
    uniform_mixture: f64,
    release_accepted: bool,
}

impl FixedProposal {
    pub(super) fn read(path: &Path, expected_sha: &str) -> Result<Self, String> {
        if fs::metadata(path).map_err(|e| e.to_string())?.len() > 2 * 1024 * 1024 {
            return Err("oversized endpoint proposal".into());
        }
        let bytes = fs::read(path).map_err(|e| e.to_string())?;
        if format!("{:x}", Sha256::digest(&bytes)) != expected_sha {
            return Err("endpoint proposal hash mismatch".into());
        }
        let proposal: Self = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        proposal.validate(&proposal.live_histories)?;
        Ok(proposal)
    }

    fn validate(&self, live: &[History]) -> Result<(), String> {
        if self.schema != "preflop-endpoint-importance-screen-v1"
            || self.release_accepted || self.uniform_mixture != 0.5
            || live.len() != 49 || live != self.live_histories
            || live.iter().collect::<BTreeSet<_>>().len() != 49
            || self.probabilities_by_actor.iter().any(|q| {
                q.len() != 49 || q.iter().any(|v| !v.is_finite() || *v < 0.5/49.0 || *v > 1.0)
                    || (q.iter().sum::<f64>()-1.0).abs() > 1e-12
            })
        {
            return Err("invalid or mismatched fixed endpoint proposal".into());
        }
        Ok(())
    }

    fn at_draw(&self, live: &[History], actor: usize, draw: f64) -> Result<BTreeMap<History, f64>, String> {
        self.validate(live)?;
        if actor > 1 || !(0.0..1.0).contains(&draw) {
            return Err("invalid proposal actor or draw".into());
        }
        let q = &self.probabilities_by_actor[actor];
        let mut cumulative = 0.0;
        let mut chosen = q.len()-1;
        for (index, probability) in q.iter().enumerate() {
            cumulative += probability;
            if draw < cumulative { chosen = index; break; }
        }
        Ok(BTreeMap::from([(live[chosen].clone(), q[chosen])]))
    }

    pub(super) fn select(&self, live: &[History], actor: usize, rng: &mut SplitMix64) -> Result<BTreeMap<History, f64>, String> {
        self.at_draw(live, actor, rng.next_f64())
    }
}

#[test]
fn fixed_importance_selection_returns_exact_q_for_both_actors_and_all_endpoints() {
    let live: Vec<History> = (0..49).map(|i| vec![i.to_string()]).collect();
    let q: Vec<_> = (0..49).map(|i| 0.5/49.0 + 0.5*(i+1) as f64/1225.0).collect();
    let mut proposal = FixedProposal { schema: "preflop-endpoint-importance-screen-v1".into(),
        live_histories: live.clone(), probabilities_by_actor: [q.clone(), q.iter().rev().copied().collect()],
        uniform_mixture: 0.5, release_accepted: false };
    for actor in 0..2 {
        let mut cumulative = 0.0;
        let mut expected = 0.0;
        for (i, probability) in proposal.probabilities_by_actor[actor].iter().enumerate() {
            let selected = proposal.at_draw(&live, actor, cumulative+probability/2.0).unwrap();
            assert_eq!(selected, BTreeMap::from([(live[i].clone(), *probability)]));
            let residual = i as f64-24.0;
            expected += probability*corrected_known_expectation(&[2.0], &[0.0], &[residual], *probability).unwrap()[0];
            cumulative += probability;
        }
        assert!((expected-2.0).abs() < 1e-10);
    }
    let mut reversed = live.clone(); reversed.reverse();
    assert!(proposal.at_draw(&reversed, 0, 0.5).is_err());
    assert!(proposal.at_draw(&live, 2, 0.5).is_err());
    assert!(proposal.at_draw(&live, 0, 1.0).is_err());
    proposal.probabilities_by_actor[0][0] = 0.0;
    assert!(proposal.at_draw(&live, 0, 0.5).is_err());
}

/// Conditional on the iteration's already-fixed policy, each endpoint keeps
/// positive support. The uniform 20% component bounds the largest correction;
/// the remainder prioritizes opponent reach, NOT joint or traverser own reach.
/// This is a variance heuristic, not an equilibrium or optimal-variance claim.
fn reach_probabilities(masses: &[f64]) -> Result<Vec<f64>, String> {
    if masses.is_empty() || masses.iter().any(|m| !m.is_finite() || *m < 0.0) {
        return Err("invalid counterfactual endpoint masses".into());
    }
    let total: f64 = masses.iter().sum();
    if !total.is_finite() {
        return Err("counterfactual endpoint mass overflow".into());
    }
    let uniform = 1.0 / masses.len() as f64;
    Ok(masses
        .iter()
        .map(|m| {
            if total == 0.0 {
                uniform
            } else {
                0.2 * uniform + 0.8 * m / total
            }
        })
        .collect())
}

pub(super) fn select_opponent_reach(
    snapshot: &Snapshot,
    live: &[History],
    traverser: usize,
    rng: &mut SplitMix64,
) -> Result<BTreeMap<History, f64>, String> {
    if traverser > 1 || live.iter().collect::<BTreeSet<_>>().len() != live.len() {
        return Err("invalid counterfactual endpoint population".into());
    }
    let masses = live
        .iter()
        .map(|h| {
            let (state, ranges) = snapshot.endpoints.get(h).ok_or("missing endpoint")?;
            if state.terminal.is_some() || state.street != Street::Flop {
                return Err("opponent-reach sample is not a live flop");
            }
            Ok(ranges[1 - traverser].iter().sum())
        })
        .collect::<Result<Vec<f64>, &str>>()?;
    let probabilities = reach_probabilities(&masses)?;
    let draw = rng.next_f64();
    let mut cumulative = 0.0;
    let mut chosen = live.len() - 1;
    for (index, q) in probabilities.iter().enumerate() {
        cumulative += q;
        if draw < cumulative {
            chosen = index;
            break;
        }
    }
    Ok(BTreeMap::from([(
        live[chosen].clone(),
        probabilities[chosen],
    )]))
}

#[test]
fn opponent_reach_proposal_preserves_support_and_exact_correction_expectation() {
    let masses = [0.0, 1.0, 9.0];
    let q = reach_probabilities(&masses).unwrap();
    assert!((q.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!(q.iter().all(|p| *p >= 0.2 / 3.0));
    assert!(q[2] > q[1] && q[1] > q[0]);
    let mean = [3.0, -1.0, 2.0];
    let residual = [0.0, -2.0, 8.0];
    for leaf in 0..3 {
        let expectation: f64 = (0..3)
            .map(|chosen| {
                q[chosen]
                    * if chosen == leaf {
                        corrected_known_expectation(
                            &[mean[leaf]],
                            &[0.0],
                            &[residual[leaf]],
                            q[leaf],
                        )
                        .unwrap()[0]
                    } else {
                        mean[leaf]
                    }
            })
            .sum();
        assert!((expectation - mean[leaf] - residual[leaf]).abs() < 1e-12);
    }
    assert_eq!(reach_probabilities(&[0.0, 0.0]).unwrap(), vec![0.5, 0.5]);
    assert!(reach_probabilities(&[f64::NAN]).is_err());
    assert!(reach_probabilities(&[-1.0]).is_err());
    assert!(reach_probabilities(&[]).is_err());
}

#[test]
fn stratified_endpoint_inclusion_probabilities_preserve_unbiased_corrections() {
    let root = vec!["blinds".to_owned()];
    let live: Vec<History> = [
        ("limp", 0),
        ("limp", 1),
        ("raise", 0),
        ("raise", 1),
        ("raise", 2),
    ]
    .into_iter()
    .map(|(a, n)| vec!["blinds".into(), a.into(), n.to_string()])
    .collect();
    let mut rng = SplitMix64::new(27001);
    let mut old = SplitMix64::new(27001);
    let expected = live[old.index(live.len())].clone();
    let one = EndpointSampling::UniformOne
        .select(&live, &root, &mut rng)
        .unwrap();
    assert_eq!(one, BTreeMap::from([(expected, 0.2)]));
    assert_eq!(rng.state(), old.state());
    let batch = EndpointSampling::UniformBatch
        .select(&live, &root, &mut rng)
        .unwrap();
    assert_eq!(batch.len(), 2);
    assert!(batch.values().all(|q| *q == 0.4));
    let selected = EndpointSampling::RootStratified
        .select(&live, &root, &mut rng)
        .unwrap();
    assert_eq!(selected.len(), 2);
    assert_eq!(
        selected
            .keys()
            .map(|h| h[1].clone())
            .collect::<BTreeSet<_>>()
            .len(),
        2
    );
    for (history, q) in selected {
        assert_eq!(q, if history[1] == "limp" { 0.5 } else { 1.0 / 3.0 });
    }
    // Enumerate all six joint samples instead of a flaky frequency assertion.
    let means = [1.0, 2.0, 3.0, 4.0, 5.0];
    let observed = [7.0, -2.0, 11.0, -3.0, 8.0];
    let mut expected = [0.0; 5];
    for a in 0..2 {
        for b in 2..5 {
            for i in 0..5 {
                let value = if i == a || i == b {
                    corrected_known_expectation(
                        &[means[i]],
                        &[means[i]],
                        &[observed[i]],
                        if i < 2 { 0.5 } else { 1.0 / 3.0 },
                    )
                    .unwrap()[0]
                } else {
                    means[i]
                };
                expected[i] += value / 6.0;
            }
        }
    }
    assert!(expected
        .iter()
        .zip(observed)
        .all(|(a, b)| (a - b).abs() < 1e-12));
    assert!(EndpointSampling::parse("unexpected").is_err());
    assert!(EndpointSampling::RootStratified
        .select(&[], &root, &mut rng)
        .is_err());
    assert!(EndpointSampling::RootStratified
        .select(&[live[0].clone(), live[0].clone()], &root, &mut rng)
        .is_err());
}

#[test]
fn full_preflop_batches_cover_six_opening_groups_and_49_live_endpoints() {
    let config = BlueprintConfig {
        effective_stack_bb: 20.0,
        iterations: 2,
        averaging_delay: 0,
        exact_preflop_averaging: true,
        traversal: BlueprintTraversal::PublicChanceSampling,
        ..BlueprintConfig::default()
    };
    let mut trainer = Trainer::fresh(config.clone());
    trainer.sweep_preflop_average().unwrap();
    let mut snapshot = Snapshot::capture(&trainer).unwrap();
    let live: Vec<_> = snapshot
        .endpoints
        .iter()
        .filter(|(_, (s, _))| s.terminal.is_none() && s.street == Street::Flop)
        .map(|(h, _)| h.clone())
        .collect();
    assert_eq!(live.len(), 49);
    let root = GameState::initial(&config).public_history;
    for scheme in [
        EndpointSampling::UniformBatch,
        EndpointSampling::RootStratified,
    ] {
        let selected = scheme
            .select(&live, &root, &mut SplitMix64::new(27001))
            .unwrap();
        assert_eq!(selected.len(), 6); // limp + 2/2.5/3/4/5bb; folds/all-ins are exact terminals
        for (history, q) in selected {
            let expected = if scheme == EndpointSampling::UniformBatch {
                6.0 / 49.0
            } else {
                1.0 / live
                    .iter()
                    .filter(|h| h[root.len()] == history[root.len()])
                    .count() as f64
            };
            assert_eq!(q, expected);
        }
    }
    let sample = select_opponent_reach(&snapshot, &live, 0, &mut SplitMix64::new(27001)).unwrap();
    for (_, ranges) in snapshot.endpoints.values_mut() {
        ranges[0].fill(0.0);
    }
    // A traverser's zero own reach must not remove counterfactual updates.
    assert_eq!(
        sample,
        select_opponent_reach(&snapshot, &live, 0, &mut SplitMix64::new(27001)).unwrap()
    );
    assert_eq!(sample.len(), 1);
    assert!(sample.values().all(|q| *q >= 0.2 / 49.0 && *q <= 1.0));
}
