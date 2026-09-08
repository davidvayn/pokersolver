import assert from 'node:assert/strict';
import test from 'node:test';
import { summarizeActions } from './native_action_diagnostics.mjs';
import { compareActionValues } from './native_action_value_probe.mjs';

test('action probe separates wrong rankings from action-independent value offsets', () => {
  const input = {history: ['test'], actor: 0, actionLabels: ['fold', 'call'],
    probabilities: [0.5, 0.5, 0.25, 0.75, 0, 0],
    nativeCfvs: [[-0.5, -0.25, 0], [1, 0, 0]],
    predictedCfvs: [[-0.5, -0.25, 0], [-1, -0.5, 0]],
    ownReach: [0.5, 0.25, 0], opponentMass: [0.5, 0.25, 1],
    rootJoint: 1, combos: [[1, 0], [2, 0], [2, 1]]};
  const r = compareActionValues(input);
  assert.equal(r.nativeBestAgreement, 0);
  assert.equal(r.nativeLossFromPredictedBestBb, 2.6);
  assert.equal(r.rootWeightedRankingLossBb, 0.8125);
  assert.equal(r.nativePolicyDeviationLossBb, 1.25);
  assert.equal(r.predictedPolicyDeviationLossBb, 0.55);
  const offset = compareActionValues({...input,
    predictedCfvs: input.nativeCfvs.map(row => row.map((v,c) => v + 10*input.opponentMass[c]))});
  assert.equal(offset.nativeBestAgreement, 1);
  assert.equal(offset.nativeLossFromPredictedBestBb, 0);
  assert.equal(offset.actionValueRmseBb, 10);
  assert.equal(offset.actionContrastRmseBb, 0);
  const absent = compareActionValues({...input, ownReach: [0,0,0]});
  assert.equal(absent.nativeBestAgreement, null);
  assert.equal(absent.nativeLossFromPredictedBestBb, null);
  const scaled = compareActionValues({...input, ownReach: input.ownReach.map(v=>v/4)});
  assert.equal(scaled.nativeLossFromPredictedBestBb, r.nativeLossFromPredictedBestBb);
  assert.equal(scaled.rootWeightedRankingLossBb, r.rootWeightedRankingLossBb/4);
  assert.throws(()=>compareActionValues({...input, predictedCfvs:[[NaN,0,0],[0,0,0]]}));
});

test('separates conditional EV from raw counterfactual and joint reach weights', () => {
  const input = {history: ['test'], actor: 0, actionLabels: ['fold', 'call'],
    probabilities: [0.5, 0.5, 0.25, 0.75, 0, 0],
    actionCfvs: [[-0.5, -0.25, 0], [1, 0, 0]], ownReach: [0.5, 0.25, 0],
    opponentMass: [0.5, 0.25, 1], rootJoint: 1, combos: [[1, 0], [2, 0], [2, 1]]};
  const row = summarizeActions(input);
  assert.equal(row.jointReachRelativeToRoot, 0.3125);
  assert.equal(row.rootWeightedContributionBb, 0.390625);
  assert.equal(row.localDeviationLossBb, 1.25);
  assert.equal(row.costlyHands.length, 2);
  assert.equal(row.costlyHands[0].localDeviationLossBb, 1.5);
  assert.deepEqual(row.costlyHands[0].actionEvBb, [-1, 2]);
  const scaled = summarizeActions({...input, ownReach: input.ownReach.map(v => v * 0.25)});
  assert.equal(scaled.localDeviationLossBb, row.localDeviationLossBb);
  assert.equal(scaled.rootWeightedContributionBb, row.rootWeightedContributionBb * 0.25);
  const absent = summarizeActions({...input, ownReach: [0, 0, 0]});
  assert.equal(absent.localDeviationLossBb, null);
  assert.equal(absent.reachWeightedMix, null);
  assert.deepEqual(absent.costlyHands, []);
});
