import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { audit } from './audit_native_flop_response.mjs';

test('royal-flush analytic backup; missing chance, changed values, and illegal actions fail', () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'native-flop-audit-test-'));
  const write = (name, value) => fs.writeFileSync(path.join(directory, name), JSON.stringify(value));
  const hash = value => crypto.createHash('sha256').update(value).digest('hex');
  const N = 1326, nuts = 48 * 47 / 2 + 44, villain = 30 * 29 / 2 + 29;
  const vectors = () => [Array(N).fill(0), Array(N).fill(0)];
  const ranges = vectors(); ranges[0][nuts] = 1; ranges[1][villain] = 1;
  const root = ['analytic:royal-flush-fixture'];
  const board = [32, 36, 40];
  const game = {effective_stack_bb: 2, big_blind_bb: 1,
    action_abstraction: {postflop_raise_cap: 1, flop_bet_pot_fractions: [1],
      postflop_raise_pot_fractions: [1], include_all_in: true}};
  const state = {street: 'flop', board, ranges, actor: 1, invested_bb: [1, 1],
    street_invested_bb: [0, 0], last_full_raise_bb: 1, aggressions: 0,
    checks: 0, raise_reopened: true, public_history: root};
  function row(history, actor, labels) {
    const probabilities = Array(N * labels.length).fill(0);
    for (let a = 0; a < labels.length; a++) probabilities[(actor === 0 ? nuts : villain) * labels.length + a] = 1 / labels.length;
    return {public_history: history, actor, action_labels: labels, probabilities};
  }
  const candidate = {schema: 'hu-native-counterfactual-turn-flop-pilot-v1', game, state,
    learned_leaf_model_sha256: 'b'.repeat(64), // Test-only prediction identity.
    turn_iterations: 4, strategies: [
      row(root, 1, ['check', 'bet_all_in_to_1.000bb']),
      row([...root, 'Flop:p1:check'], 0, ['check', 'bet_all_in_to_1.000bb']),
      row([...root, 'Flop:p1:bet_all_in_to_1.000bb'], 0, ['fold', 'call_all_in']),
      row([...root, 'Flop:p1:check', 'Flop:p0:bet_all_in_to_1.000bb'], 1, ['fold', 'call_all_in']),
    ]};
  const digest = hash(JSON.stringify(candidate));
  const equity = Buffer.alloc(N * N * 4);
  for (let i = 0; i < N * N; i++) equity.writeFloatLE(NaN, i * 4);
  equity.writeFloatLE(1, (nuts * N + villain) * 4);
  const response = {schema: 'hu-native-flop-frozen-response-v1', candidate_sha256: digest,
    public_turns: 49, turn_policy_count: 49, profile_bb: [0.875, -0.875],
    best_response_bb: [1.75, -0.5], gain_bb: [0.875, 0.375], half_summed_gain_bb: 0.625};
  const arguments_ = ['candidate.json', '.', 'equity.json', 'response.json'].map(name => path.join(directory, name));
  try {
    write('candidate.json', candidate); write('response.json', response);
    fs.writeFileSync(path.join(directory, 'equity.f32le'), equity);
    write('equity.json', {schema: 'native-flop-equity-audit-input-v1', board,
      legal: ranges.map(r => r.map(v => v > 0)), bytes: equity.length, sha256: hash(equity)});
    for (let turn = 0; turn < 52; turn++) {
      if (board.includes(turn)) continue;
      const profile = vectors();
      if (![48, 44, 30, 29].includes(turn)) { profile[0][nuts] = 0.5; profile[1][villain] = -0.5; }
      write(`turn-${turn}.json`, {schema: 'hu-native-flop-frozen-turn-packet-v1',
        candidate_sha256: digest, turn, turn_iterations: 4, leaves: [{
          history: [...root, 'Flop:p1:check', 'Flop:p0:check', 'deal:Turn'],
          policy_sha256: 'a'.repeat(64), profile_bb: profile, best_response_bb: profile}]});
    }
    const result = audit(...arguments_);
    assert.ok(result.maximumDifferenceBb < 1e-12);
    assert.ok(Math.abs(result.half_summed_gain_bb - 0.625) < 1e-12);
    // Diagnose individual actions against the same frozen subsequent policy.
    // At the root the villain gets -1.25bb checking and -0.5bb shoving;
    // the 50/50 policy therefore loses 0.375bb to this one-node deviation.
    const diagnosed = audit(...arguments_, {actionDiagnostics: true});
    assert.equal(diagnosed.half_summed_gain_bb, result.half_summed_gain_bb);
    assert.equal(diagnosed.actionDiagnostics.length, 4);
    const rootDiagnostic = diagnosed.actionDiagnostics.find(row =>
      JSON.stringify(row.history) === JSON.stringify(root));
    assert.deepEqual(rootDiagnostic.actionLabels, ['check', 'bet_all_in_to_1.000bb']);
    assert.deepEqual(rootDiagnostic.reachWeightedMix, [0.5, 0.5]);
    assert.ok(Math.abs(rootDiagnostic.localDeviationLossBb - 0.375) < 1e-12);
    assert.ok(Math.abs(rootDiagnostic.rootWeightedContributionBb - 0.375) < 1e-12);
    for (const [i, value] of [-1.25, -0.5].entries()) {
      assert.ok(Math.abs(rootDiagnostic.costlyHands[0].actionEvBb[i] - value) < 1e-12);
    }
    assert.equal(rootDiagnostic.costlyHands[0].bestAction, 'bet_all_in_to_1.000bb');
    const facedBet = diagnosed.actionDiagnostics.find(row => row.actor === 0
      && row.actionLabels.includes('fold'));
    assert.ok(Math.abs(facedBet.localDeviationLossBb - 1.5) < 1e-12);
    assert.ok(Math.abs(facedBet.rootWeightedContributionBb - 0.75) < 1e-12);
    const probe={schema:'hu-native-flop-frozen-leaf-predictions-v1',candidate_sha256:digest,
      model_sha256:'b'.repeat(64),releaseAccepted:false,packets:[]};
    for(let turn=0;turn<52;turn++) {
      if(board.includes(turn)) continue;
      const packet=JSON.parse(fs.readFileSync(path.join(directory,`turn-${turn}.json`)));
      probe.packets.push({turn,leaves:packet.leaves.map(leaf=>({history:leaf.history,
        predicted_counterfactual_bb:leaf.profile_bb}))});
    }
    const identical=audit(...arguments_,{leafPredictions:probe});
    assert.equal(identical.half_summed_gain_bb,result.half_summed_gain_bb);
    assert.ok(identical.actionValueDiagnostics.every(row=>row.actionValueRmseBb<1e-12));
    for(const packet of probe.packets) for(const leaf of packet.leaves)
      leaf.predicted_counterfactual_bb=leaf.predicted_counterfactual_bb.map(row=>row.map(v=>-v));
    const wrong=audit(...arguments_,{leafPredictions:probe});
    assert.equal(wrong.half_summed_gain_bb,result.half_summed_gain_bb,'prediction must never alter audited native response');
    const wrongRoot=wrong.actionValueDiagnostics.find(row=>JSON.stringify(row.history)===JSON.stringify(root));
    assert.ok(Math.abs(wrongRoot.nativeLossFromPredictedBestBb-.75)<1e-12);
    assert.equal(wrongRoot.costlyHands[0].predictedBest,'check');
    assert.throws(()=>audit(...arguments_,{leafPredictions:{...probe,packets:probe.packets.slice(1)}}));
    assert.throws(()=>audit(...arguments_,{leafPredictions:{...probe,model_sha256:'c'.repeat(64)}}));
    const alternative={...probe,model_sha256:'c'.repeat(64),
      prediction_role:'alternative_value_model',source_policy_model_sha256:'b'.repeat(64)};
    const alternateResult=audit(...arguments_,{leafPredictions:alternative,
      alternativeValueModelSha256:'c'.repeat(64)});
    assert.equal(alternateResult.half_summed_gain_bb,result.half_summed_gain_bb);
    assert.equal(alternateResult.predictionRole,'alternative_value_model');
    assert.equal(alternateResult.sourcePolicyModelSha256,'b'.repeat(64));
    assert.throws(()=>audit(...arguments_,{leafPredictions:alternative}));
    assert.throws(()=>audit(...arguments_,{leafPredictions:{...alternative,source_policy_model_sha256:'d'.repeat(64)},
      alternativeValueModelSha256:'c'.repeat(64)}));
    assert.throws(()=>audit(...arguments_,{leafPredictions:{...probe,packets:[probe.packets[0],...probe.packets.slice(0,48)]}}));
    write('response.json', {...response, half_summed_gain_bb: 0});
    assert.throws(() => audit(...arguments_), /half-summed gain/);
    write('response.json', response);
    const saved = path.join(directory, 'turn-0.json');
    fs.renameSync(saved, `${saved}.saved`);
    assert.throws(() => audit(...arguments_), /ENOENT/);
    fs.renameSync(`${saved}.saved`, saved);
    // A reconstruction budget is hash-pinned separately from the unchanged
    // training budget and flop rows. Packets must match the chosen route.
    candidate.response_turn_iterations = 8;
    write('candidate.json', candidate);
    const routedDigest = hash(JSON.stringify(candidate));
    write('response.json', {...response, candidate_sha256: routedDigest});
    for (let turn = 0; turn < 52; turn++) {
      if (board.includes(turn)) continue;
      const packet = JSON.parse(fs.readFileSync(path.join(directory, `turn-${turn}.json`)));
      write(`turn-${turn}.json`, {...packet, candidate_sha256: routedDigest, turn_iterations: 8});
    }
    assert.equal(candidate.turn_iterations, 4);
    assert.ok(Math.abs(audit(...arguments_).half_summed_gain_bb - 0.625) < 1e-12);
    const wrongBudget = JSON.parse(fs.readFileSync(saved));
    write('turn-0.json', {...wrongBudget, turn_iterations: 4});
    assert.throws(() => audit(...arguments_));
    write('turn-0.json', wrongBudget);
    candidate.game.action_abstraction.include_all_in = false;
    write('candidate.json', candidate);
    const changedDigest = hash(JSON.stringify(candidate));
    write('response.json', {...response, candidate_sha256: changedDigest});
    for (let turn = 0; turn < 52; turn++) {
      if (board.includes(turn)) continue;
      const packet = JSON.parse(fs.readFileSync(path.join(directory, `turn-${turn}.json`)));
      write(`turn-${turn}.json`, {...packet, candidate_sha256: changedDigest});
    }
    assert.throws(() => audit(...arguments_), /action grid does not match the game/);
  } finally {
    // Only the exclusively created test fixture directory is removed.
    fs.rmSync(directory, {recursive: true});
  }
});
