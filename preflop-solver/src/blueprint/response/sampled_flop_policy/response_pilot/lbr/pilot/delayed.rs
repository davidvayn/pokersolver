//! Separate research hypothesis: retain authentic baseline preflop play and
//! defer the greedy checkdown attack until the flop. No defender policy change.
use super::*;

#[test]
#[ignore = "delayed LBR pilot; finish the original pair and cache parity first, then use explicit source and guard"]
fn sampled_profile_delayed_lbr_pilot() {
    run_challenge_from(32, 64, 92004, Street::Flop);
}

#[test]
fn delayed_lbr_keeps_baseline_preflop_and_only_conditions_on_opponent_actions() {
    let (game, _) = super::super::tests::fixture(Street::Preflop);
    let policy = super::super::tests::Pattern::Call;
    let mut chance = SplitMix64::new(92301);
    let inactive = Lbr {
        seed: 90001,
        early_runouts_per_combo: 0,
    };
    let active = Lbr {
        seed: 90001,
        early_runouts_per_combo: 16,
    };
    let mut flop_decisions = 0;
    for index in 0..16 {
        let deal = Deal::sample(&mut chance);
        let action_seed = derived_seed(92301, index, 0);
        let base_p0 = baseline(&policy, &game, &deal, action_seed).unwrap();
        for seat in 0..2 {
            // The invalid equity budget would fail if this supposedly disabled
            // critic accidentally computed any action values.
            let disabled =
                play_from(&policy, &game, &deal, seat, action_seed, &inactive, None).unwrap();
            assert_eq!(disabled.utility, if seat == 0 { base_p0 } else { -base_p0 });
            assert!(disabled.actions.is_empty());
            assert_eq!(disabled.decisions, [0; 4]);
            let delayed = play_from(
                &policy,
                &game,
                &deal,
                seat,
                action_seed,
                &active,
                Some(Street::Flop),
            )
            .unwrap();
            let preflop = |hand: &HandAttack| {
                hand.history
                    .iter()
                    .take_while(|a| a.as_str() != "deal:Flop")
                    .cloned()
                    .collect::<Vec<_>>()
            };
            assert_eq!(preflop(&disabled), preflop(&delayed));
            assert_eq!(delayed.decisions[0], 0);
            assert!(delayed.actions.iter().all(|a| a["street"] != "preflop"));
            flop_decisions += delayed.decisions[1];
            let repeat = play_from(
                &policy,
                &game,
                &deal,
                seat,
                action_seed,
                &active,
                Some(Street::Flop),
            )
            .unwrap();
            assert_eq!(
                serde_json::to_vec(&delayed).unwrap(),
                serde_json::to_vec(&repeat).unwrap()
            );
            assert_eq!(
                serde_json::to_vec(
                    &play(&policy, &game, &deal, seat, action_seed, &active).unwrap()
                )
                .unwrap(),
                serde_json::to_vec(
                    &play_from(
                        &policy,
                        &game,
                        &deal,
                        seat,
                        action_seed,
                        &active,
                        Some(Street::Preflop)
                    )
                    .unwrap()
                )
                .unwrap()
            );
        }
    }
    assert!(flop_decisions >= 32);
}
