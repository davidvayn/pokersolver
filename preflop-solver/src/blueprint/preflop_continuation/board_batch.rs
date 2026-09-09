//! Average corrected CFVs under one frozen iteration policy, never policies or
//! regrets after separate updates. Endpoint q remains a single shared draw.
use super::*;

type EndpointValues = BTreeMap<History, [Vec<f64>; 2]>;

pub(super) fn chance_round(round: u64, index: usize, count: usize) -> Result<u64, String> {
    if round==0 || ![1,2].contains(&count) || index>=count {
        return Err("invalid independent board batch".into());
    }
    Ok(((round-1)/2)*(2*count as u64)+(round-1)%2+2*index as u64+1)
}

pub(super) fn mean_values(mut batches: Vec<EndpointValues>) -> Result<EndpointValues, String> {
    if ![1,2].contains(&batches.len()) { return Err("invalid board sample count".into()); }
    let count = batches.len() as f64;
    let mut mean = batches.remove(0);
    for next in batches {
        if !mean.keys().eq(next.keys()) { return Err("board endpoint support changed".into()); }
        for (history, values) in &mut mean {
            for p in 0..2 {
                if values[p].len()!=1326 || next[history][p].len()!=1326 {
                    return Err("board CFV shape changed".into());
                }
                for c in 0..1326 { values[p][c] += next[history][p][c]; }
            }
        }
    }
    for values in mean.values_mut() {
        for player in values {
            if player.len()!=1326 || player.iter().any(|v| !v.is_finite()) {
                return Err("invalid board CFVs".into());
            }
            if count!=1.0 { for v in player { *v /= count; } }
        }
    }
    Ok(mean)
}

#[test]
fn independent_board_stream_preserves_each_traverser_and_reference() {
    for r in 1..=32 { assert_eq!(chance_round(r,0,1).unwrap(),r); }
    for actor in 0..2 {
        let draws: Vec<_> = (1..=16).filter(|r| (r-1)%2==actor)
            .flat_map(|r| (0..2).map(move |b| chance_round(r,b,2).unwrap())).collect();
        assert_eq!(draws,(1..=32).filter(|r| (r-1)%2==actor).collect::<Vec<_>>());
    }
    assert!(chance_round(0,0,2).is_err());
    assert!(chance_round(1,2,2).is_err());
}

#[test]
fn independent_boards_average_corrected_values_not_twice_the_exact_baseline() {
    let base=vec![3.0;1326];
    let make = |native: f64| {
        let corrected=corrected_known_expectation(&base,&vec![2.0;1326],&vec![native;1326],0.25).unwrap();
        BTreeMap::from([(vec!["endpoint".into()],[corrected.clone(),corrected.iter().map(|x| -x).collect()])])
    };
    let a=make(4.0);
    assert_eq!(mean_values(vec![a.clone()]).unwrap(),a);
    let result=mean_values(vec![a,make(8.0)]).unwrap();
    assert_eq!(result[&vec!["endpoint".into()]][0],vec![19.0;1326]);
    assert!(mean_values(vec![]).is_err());
    assert!(mean_values(vec![make(1.0),BTreeMap::new()]).is_err());
}
