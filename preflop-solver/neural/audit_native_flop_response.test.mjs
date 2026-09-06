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
