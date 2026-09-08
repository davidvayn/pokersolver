// One-node deviations against unchanged subsequent play, not exploitability.
// CFVs contain opponent reach; joint weights contain both players' reach.
import assert from 'node:assert/strict';

export function summarizeActions({history, actor, actionLabels, probabilities,
  actionCfvs, ownReach, opponentMass, rootJoint, combos}) {
  const n = actionLabels.length, count = ownReach.length;
  assert.ok(n > 0 && rootJoint > 0);
  assert.equal(probabilities.length, count * n);
  assert.equal(actionCfvs.length, n);
  assert.ok(actionCfvs.every(v => v.length === count));
  assert.equal(opponentMass.length, count);
  const mix = Array(n).fill(0), actionEvs = Array(n).fill(0), hands = [];
  let joint = 0, weightedLoss = 0;
  for (let c = 0; c < count; c++) {
    const mass = opponentMass[c], weight = ownReach[c] * mass;
    if (weight <= 1e-15) continue;
    const ev = actionCfvs.map(v => v[c] / mass);
    const policy = Array.from({length: n}, (_, a) => probabilities[c * n + a]);
    assert.ok(ev.every(Number.isFinite));
    const expected = ev.reduce((total, v, a) => total + policy[a] * v, 0);
    const best = Math.max(...ev), bestIndex = ev.indexOf(best);
    const loss = Math.max(0, best - expected);
    joint += weight; weightedLoss += weight * loss;
    for (let a = 0; a < n; a++) {
      mix[a] += weight * policy[a]; actionEvs[a] += weight * ev[a];
    }
    hands.push({combo: c, cards: combos[c], probabilities: policy, actionEvBb: ev,
      bestAction: actionLabels[bestIndex], policyEvBb: expected, localDeviationLossBb: loss,
      rootWeightedContributionBb: weight * loss / rootJoint});
  }
  hands.sort((a, b) => b.rootWeightedContributionBb - a.rootWeightedContributionBb || a.combo - b.combo);
  return {history, actor, actionLabels, jointReachRelativeToRoot: joint / rootJoint,
    reachWeightedMix: joint > 0 ? mix.map(v => v / joint) : null,
    reachWeightedActionEvBb: joint > 0 ? actionEvs.map(v => v / joint) : null,
    localDeviationLossBb: joint > 0 ? weightedLoss / joint : null,
    rootWeightedContributionBb: weightedLoss / rootJoint, costlyHands: hands.slice(0, 12),
    interpretation: 'Single-node action deviations against this frozen subsequent policy; contributions across nodes are not additive full-game exploitability.'};
}
