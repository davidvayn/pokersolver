//! Matched full-grid PCS training diagnostic. External resource guard required.
use super::*;
use std::time::Instant;

#[test]
#[ignore = "explicit bounded paired preflop averaging pilot; requires external resource guard"]
fn matched_preflop_averaging_pilot() {
    let seed: u64 = std::env::var("POKER_AVERAGE_SEED")
        .unwrap()
        .parse()
        .unwrap();
    assert!([26001, 26002].contains(&seed));
    let rounds: u64 = std::env::var("POKER_AVERAGE_ROUNDS")
        .unwrap()
        .parse()
        .unwrap();
    assert!([2, 8, 16, 32, 64].contains(&rounds));
    let exact = match std::env::var("POKER_AVERAGE_MODE").as_deref() {
        Ok("sampled") => false,
        Ok("exact") => true,
        other => panic!("explicit averaging mode required: {other:?}"),
    };
    let path = PathBuf::from(std::env::var("POKER_AVERAGE_OUTPUT").unwrap());
    assert!(!path.exists());
    let config = BlueprintConfig {
        seed,
        iterations: rounds,
        effective_stack_bb: 20.0,
        averaging_delay: 0,
        max_information_sets: 4_000_000,
        traversal: BlueprintTraversal::PublicChanceSampling,
        integrate_terminal_actions: true,
        exact_preflop_averaging: exact,
        ..BlueprintConfig::default()
    };
    config.validate().unwrap();
    let mut trainer = Trainer::fresh(config.clone());
    let started = Instant::now();
    let mut progress = Vec::new();
    for round in 1..=rounds {
        trainer.config.iterations = round;
        trainer.train(&RunControl::default()).unwrap();
        assert_eq!(
            trainer.completed_iterations, round,
            "never silently compare truncated runs"
        );
        let tick = serde_json::json!({"round":round,"seconds":started.elapsed().as_secs_f64(),"nodes":trainer.nodes.len()});
        eprintln!("{tick}");
        progress.push(tick);
    }
    let training_seconds = started.elapsed().as_secs_f64();
    // Hash all touched regret rows, lazy discounts, and all postflop averages.
    // Untouched zero preflop rows exist only in the sweep and have no training
    // effect. Excluding only those makes the digest comparable, not weaker.
    let mut regret_hash = Sha256::new();
    let mut postflop_hash = Sha256::new();
    for (key, node) in &trainer.nodes {
        if node.last_discount_iteration > 0 || node.regret_updates > 0 {
            regret_hash.update(
                rmp_serde::to_vec_named(&(
                    key,
                    &node.descriptor,
                    &node.action_labels,
                    &node.regrets,
                    node.regret_updates,
                    node.last_discount_iteration,
                    node.last_regret_discount_cumulative_logs,
                ))
                .unwrap(),
            );
        } else {
            assert!(node.regrets.iter().all(|r| *r == 0.0));
        }
        if node.descriptor.street != Street::Preflop {
            postflop_hash.update(rmp_serde::to_vec_named(&(key, node)).unwrap());
        }
    }
    let mut classes = BTreeMap::<String, (Combo, usize)>::new();
    for combo in all_combos() {
        classes.entry(combo.label()).or_insert((combo, 0)).1 += 1;
    }
    let mut rows = Vec::new();
    let mut pending = vec![GameState::initial(&config)];
    let mut absent_average_combos = 0;
    let mut untrained_combos = 0;
    while let Some(state) = pending.pop() {
        if state.terminal.is_some() || state.street != Street::Preflop {
            continue;
        }
        let actions = state.legal_actions(&config);
        for (class, (combo, multiplicity)) in &classes {
            let deal = neural::deal_for_policy_combo_on_board(*combo, state.actor, &[]).unwrap();
            let (key, _, _) = information_set(&state, &deal, &config);
            let node = trainer.nodes.get(&key);
            let average =
                node.filter(|n| n.average_visits > 0 && n.strategy_sum.iter().any(|v| *v > 0.0));
            let trained = average.is_some_and(|n| n.regret_updates > 0);
            if average.is_none() {
                absent_average_combos += multiplicity;
            }
            if !trained {
                untrained_combos += multiplicity;
            }
            rows.push(
                serde_json::json!({"key":key.to_string(),"actor":state.actor,
                "history":state.public_history,"hand":class,"comboWeight":multiplicity,
                "actions":actions.iter().map(|a| a.label.as_str()).collect::<Vec<_>>(),
                "probabilities":average.map(Node::average_strategy),
                "averageMass":node.map(|n| n.strategy_sum.iter().sum::<f64>()),
                "averageVisits":node.map(|n| n.average_visits),
                "regretUpdates":node.map(|n| n.regret_updates),"trained":trained}),
            );
        }
        pending.extend(actions.iter().map(|a| state.apply(a, &config)));
    }
    assert_eq!(rows.len(), 16_900);
    let result = serde_json::json!({"schema":"matched-exact-preflop-average-pilot-v1",
        "config":config,"trainingSeconds":training_seconds,"progress":progress,
        "regretStateSha256":format!("{:x}",regret_hash.finalize()),
        "postflopStateSha256":format!("{:x}",postflop_hash.finalize()),
        "rngState":trainer.rng.state(),"sampledDeals":trainer.sampled_deals,
        "terminalEvaluations":trainer.terminal_evaluations,"totalNodes":trainer.nodes.len(),
        "exhaustiveQueries":132_600,"absentAverageCombos":absent_average_combos,
        "untrainedCombos":untrained_combos,"rows":rows,
        "interpretation":"Full-grid 20bb matched short PCS pilot. Coverage/stability diagnostics only; not full-game exploitability or a release-qualified policy."});
    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    serde_json::to_writer(&mut output, &result).unwrap();
    output.flush().unwrap();
    output.sync_all().unwrap();
}
