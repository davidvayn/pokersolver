// Read-only value/ranking diagnosis. Never an exploitability evaluator or policy.
import assert from 'node:assert/strict';

export function compareActionValues({history, actor, actionLabels, probabilities,
  nativeCfvs, predictedCfvs, ownReach, opponentMass, rootJoint, combos}) {
  const n = actionLabels.length, count = ownReach.length;
  assert.ok(n > 0 && rootJoint > 0);
  assert.equal(probabilities.length, count*n);
  assert.equal(opponentMass.length, count);
  for (const values of [nativeCfvs,predictedCfvs]) {
    assert.equal(values.length,n);
    assert.ok(values.every(row=>row.length===count && Array.from(row).every(Number.isFinite)));
  }
  let joint=0, agreement=0, loss=0, nativeLoss=0, predictedLoss=0, squared=0, contrast=0;
  const hands=[];
  for (let c=0;c<count;c++) {
    const mass=opponentMass[c], weight=ownReach[c]*mass;
    if (weight<=1e-15) continue;
    const actual=nativeCfvs.map(row=>row[c]/mass), predicted=predictedCfvs.map(row=>row[c]/mass);
    const best=Math.max(...actual), predictedBest=Math.max(...predicted);
    const a=actual.indexOf(best), b=predicted.indexOf(predictedBest);
    const gap=best-actual[b];
    let actualPolicy=0,predictedPolicy=0;
    for(let i=0;i<n;i++) {
      actualPolicy+=probabilities[c*n+i]*actual[i];
      predictedPolicy+=probabilities[c*n+i]*predicted[i];
      squared+=weight*(predicted[i]-actual[i])**2/n;
      contrast+=weight*((predicted[i]-predicted[0])-(actual[i]-actual[0]))**2/n;
    }
    joint+=weight; agreement+=weight*(a===b); loss+=weight*gap;
    nativeLoss+=weight*Math.max(0,best-actualPolicy);
    predictedLoss+=weight*Math.max(0,predictedBest-predictedPolicy);
    hands.push({combo:c,cards:combos[c],nativeActionEvBb:actual,predictedActionEvBb:predicted,
      probabilities:Array.from({length:n},(_,i)=>probabilities[c*n+i]),
      nativeBest:actionLabels[a],predictedBest:actionLabels[b],nativeLossFromPredictedBestBb:gap,
      rootWeightedRankingLossBb:weight*gap/rootJoint});
  }
  hands.sort((a,b)=>b.rootWeightedRankingLossBb-a.rootWeightedRankingLossBb || a.combo-b.combo);
  return {history,actor,actionLabels,jointReachRelativeToRoot:joint/rootJoint,
    nativeBestAgreement:joint>0?agreement/joint:null,
    nativeLossFromPredictedBestBb:joint>0?loss/joint:null,
    rootWeightedRankingLossBb:loss/rootJoint,
    nativePolicyDeviationLossBb:joint>0?nativeLoss/joint:null,
    predictedPolicyDeviationLossBb:joint>0?predictedLoss/joint:null,
    actionValueRmseBb:joint>0?Math.sqrt(squared/joint):null,
    actionContrastRmseBb:joint>0?Math.sqrt(contrast/joint):null,
    costlyHands:hands.slice(0,12),
    interpretation:'Same frozen future policy and all-turn integration; local ranking diagnostic, not exploitability or an EV sampling-error estimate.'};
}

export function sumPredictedLeaves(probe, candidateSha256, modelSha256, board, nativeLeaves, combos, stack) {
  assert.equal(probe.schema,'hu-native-flop-frozen-leaf-predictions-v1');
  assert.equal(probe.candidate_sha256,candidateSha256);
  assert.match(modelSha256??'',/^[a-f0-9]{64}$/);
  assert.equal(probe.model_sha256,modelSha256);
  assert.equal(probe.releaseAccepted,false);
  assert.equal(probe.packets.length,49);
  const seen=new Set(), result=new Map();
  for(const packet of probe.packets) {
    assert.ok(Number.isInteger(packet.turn)&&packet.turn>=0&&packet.turn<52&&!board.includes(packet.turn));
    assert.ok(!seen.has(packet.turn),'duplicate predicted turn'); seen.add(packet.turn);
    assert.equal(packet.leaves.length,nativeLeaves.size);
    const histories=new Set();
    for(const leaf of packet.leaves) {
      const id=JSON.stringify(leaf.history);
      assert.ok(nativeLeaves.has(id)&&!histories.has(id),'unknown or duplicate predicted leaf'); histories.add(id);
      if(!result.has(id)) result.set(id,[new Float64Array(1326),new Float64Array(1326)]);
      assert.equal(leaf.predicted_counterfactual_bb.length,2);
      for(let p=0;p<2;p++) {
        assert.equal(leaf.predicted_counterfactual_bb[p].length,1326);
        for(let c=0;c<1326;c++) {
          const value=leaf.predicted_counterfactual_bb[p][c];
          assert.ok(Number.isFinite(value)&&Math.abs(value)<=stack+1e-9);
          if(combos[c].some(card=>board.includes(card)||card===packet.turn)) assert.ok(value===0);
          result.get(id)[p][c]+=value/45;
        }
      }
    }
  }
  return result;
}
