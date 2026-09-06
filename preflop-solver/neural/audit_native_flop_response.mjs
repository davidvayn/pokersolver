// Independent flop backup/accounting check, NOT a second poker evaluator.
// node neural/audit_native_flop_response.mjs policy.json packets/ equity.json response.json
// Inputs are immutable. Missing turns or mismatched hashes fail, never score zero.
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const key = history => JSON.stringify(history);
const N = 1326;
const EPS = 1e-9;
const combos = [];
for (let high = 1; high < 52; high++) {
  for (let low = 0; low < high; low++) combos.push([high, low]);
}
const zero = () => new Float64Array(N);
const close = (a, b, message, tolerance = 1e-8) => {
  assert.ok(Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= tolerance,
    `${message}: ${a} versus ${b}`);
};

function compatibleMass(range) {
  let total = 0;
  const byCard = new Float64Array(52);
  for (let c = 0; c < N; c++) {
    total += range[c];
    for (const card of combos[c]) byCard[card] += range[c];
  }
  return Float64Array.from(combos, ([high, low], c) =>
    total - byCard[high] - byCard[low] + range[c]);
}

function actions(state, game) {
  const p = state.actor, o = 1 - p, committed = state.street_invested_bb;
  const remaining = game.effective_stack_bb - state.invested_bb[p];
  const call = Math.max(0, committed[o] - committed[p]);
  const high = Math.max(...committed), grid = game.action_abstraction;
  const result = call > EPS
    ? ['fold', call + EPS >= remaining ? 'call_all_in' : 'call'] : ['check'];
  if (remaining <= call + EPS || game.effective_stack_bb - state.invested_bb[o] <= EPS
      || !state.raise_reopened || state.aggressions >= grid.postflop_raise_cap + 1) return result;
  const pot = state.invested_bb[0] + state.invested_bb[1];
  const targets = call > EPS
    ? grid.postflop_raise_pot_fractions.map(f => committed[o] + (pot + call) * f)
    : grid.flop_bet_pot_fractions.map(f => committed[p] + pot * f);
  const maximum = committed[p] + remaining;
  const minimum = high + Math.max(state.last_full_raise_bb, game.big_blind_bb);
  const seen = new Set();
  for (const value of targets) {
    const target = Math.round(Math.min(value, maximum) * 1000) / 1000;
    if (target + EPS < minimum || target >= maximum - EPS || seen.has(target)) continue;
    seen.add(target);
    result.push(`${high <= EPS ? 'bet' : 'raise'}_to_${target.toFixed(3)}bb`);
  }
  if (grid.include_all_in) {
    result.push(`${high <= EPS ? 'bet' : 'raise'}_all_in_to_${maximum.toFixed(3)}bb`);
  }
  return result;
}

function apply(state, action, game) {
  const next = {...state, public_history: [...state.public_history, `Flop:p${state.actor}:${action}`],
    invested_bb: [...state.invested_bb], street_invested_bb: [...state.street_invested_bb]};
  const p = state.actor, o = 1 - p;
  const endRound = () => {
    close(next.invested_bb[0], next.invested_bb[1], 'unmatched closed round');
    if (next.invested_bb.some(v => v + EPS >= game.effective_stack_bb)) next.terminal = 'showdown';
    else { next.street = 'turn'; next.public_history.push('deal:Turn'); }
  };
  if (action === 'fold') { next.terminal = 'fold'; next.winner = o; }
  else if (action === 'check') {
    if (state.checks === 1) endRound();
    else { next.actor = o; next.checks = 1; }
  } else if (action === 'call' || action === 'call_all_in') {
    const paid = Math.min(state.street_invested_bb[o] - state.street_invested_bb[p],
      game.effective_stack_bb - state.invested_bb[p]);
    assert.ok(paid > 0);
    next.invested_bb[p] += paid; next.street_invested_bb[p] += paid; endRound();
  } else {
    const match = /^(?:bet|raise)_(?:all_in_)?to_([0-9.]+)bb$/.exec(action);
    assert.ok(match, `unknown action ${action}`);
    const target = Number(match[1]);
    const increment = target - Math.max(...state.street_invested_bb);
    next.invested_bb[p] += target - state.street_invested_bb[p];
    next.street_invested_bb[p] = target;
    next.raise_reopened = increment + EPS >= state.last_full_raise_bb;
    if (next.raise_reopened) next.last_full_raise_bb = increment;
    next.aggressions++; next.checks = 0; next.actor = o;
  }
  assert.ok(next.invested_bb.every(v => v >= 0 && v <= game.effective_stack_bb + EPS));
  return next;
}

export function audit(candidatePath, packetDirectory, equityPath, responsePath) {
  const candidateBytes = fs.readFileSync(candidatePath);
  const candidate = JSON.parse(candidateBytes), candidateSha256 = hash(candidateBytes);
  assert.equal(candidate.schema, 'hu-native-counterfactual-turn-flop-pilot-v1');
  assert.equal(candidate.state.street, 'flop');
  const {game, state} = candidate;
  const legal = state.ranges.map(range => range.map(v => v > 0));
  const rows = new Map();
  for (const row of candidate.strategies) {
    const id = key(row.public_history), n = row.action_labels.length;
    assert.ok(!rows.has(id));
    assert.equal(row.probabilities.length, N * n);
    const probabilities = Float64Array.from(row.probabilities, Math.fround);
    for (let c = 0; c < N; c++) {
      let sum = 0;
      for (let a = 0; a < n; a++) {
        const value = probabilities[c * n + a];
        assert.ok(Number.isFinite(value) && value >= 0 && value <= 1); sum += value;
      }
      close(sum, legal[row.actor][c] ? 1 : 0, 'frozen probability sum', 1e-6);
      if (sum > 0) for (let a = 0; a < n; a++) probabilities[c * n + a] /= sum;
    }
    rows.set(id, {...row, probabilities});
  }
  const equityMetadataBytes = fs.readFileSync(equityPath);
  const metadata = JSON.parse(equityMetadataBytes);
  const equityBytes = fs.readFileSync(equityPath.replace(/\.[^.]+$/, '.f32le'));
  assert.equal(metadata.schema, 'native-flop-equity-audit-input-v1');
  assert.deepEqual(metadata.board, state.board); assert.deepEqual(metadata.legal, legal);
  assert.equal(metadata.bytes, N * N * 4); assert.equal(equityBytes.length, metadata.bytes);
  assert.equal(metadata.sha256, hash(equityBytes));
  const equity = new Float64Array(N * N);
  for (let i = 0; i < equity.length; i++) equity[i] = equityBytes.readFloatLE(i * 4);

  const leaves = new Map(), packets = [];
  for (let turn = 0; turn < 52; turn++) {
    if (state.board.includes(turn)) continue;
    const bytes = fs.readFileSync(path.join(packetDirectory, `turn-${turn}.json`));
    const packet = JSON.parse(bytes);
    assert.equal(packet.schema, 'hu-native-flop-frozen-turn-packet-v1');
    assert.equal(packet.turn, turn); assert.equal(packet.candidate_sha256, candidateSha256);
    assert.equal(packet.turn_iterations, candidate.turn_iterations);
    const seen = new Set();
    for (const leaf of packet.leaves) {
      const id = key(leaf.history); assert.ok(!seen.has(id)); seen.add(id);
      assert.match(leaf.policy_sha256, /^[a-f0-9]{64}$/);
      if (!leaves.has(id)) leaves.set(id, {count: 0, profile_bb: [zero(), zero()], best_response_bb: [zero(), zero()]});
      const sum = leaves.get(id); sum.count++;
      for (const field of ['profile_bb', 'best_response_bb']) for (let p = 0; p < 2; p++) {
        assert.equal(leaf[field][p].length, N);
        for (let c = 0; c < N; c++) {
          const value = leaf[field][p][c];
          assert.ok(Number.isFinite(value) && Math.abs(value) <= game.effective_stack_bb + EPS);
          if (combos[c].some(v => state.board.includes(v) || v === turn)) assert.equal(value, 0);
          sum[field][p][c] += value / 45; // integrate chance BEFORE any flop maximization
        }
      }
    }
    packets.push({turn, sha256: hash(bytes), leaves: seen.size});
  }
  assert.equal(packets.length, 49);
  for (const leaf of leaves.values()) assert.equal(leaf.count, 49);
  assert.ok(packets.every(p => p.leaves === leaves.size));
  const visitedRows = new Set(), visitedLeaves = new Set();
  function backUp(cursor, ranges, seat, best, continueBest) {
    if (cursor.terminal === 'fold') {
      const value = cursor.winner === seat ? cursor.invested_bb[1 - seat] : -cursor.invested_bb[seat];
      return compatibleMass(ranges[1 - seat]).map(mass => mass * value);
    }
    if (cursor.terminal === 'showdown') {
      const result = zero();
      for (let p0 = 0; p0 < N; p0++) {
        if (!legal[0][p0]) continue;
        for (let p1 = 0; p1 < N; p1++) {
          const e = equity[p0 * N + p1];
          if (!Number.isFinite(e)) continue;
          assert.ok(e >= 0 && e <= 1);
          const value = e * cursor.invested_bb[1] - (1 - e) * cursor.invested_bb[0];
          if (seat === 0) result[p0] += ranges[1][p1] * value;
          else result[p1] -= ranges[0][p0] * value;
        }
      }
      return result;
    }
    const id = key(cursor.public_history);
    if (cursor.street === 'turn') {
      assert.ok(leaves.has(id), `missing live turn leaf ${id}`); visitedLeaves.add(id);
      return leaves.get(id)[best && continueBest ? 'best_response_bb' : 'profile_bb'][seat];
    }
    const row = rows.get(id); assert.ok(row, `missing flop row ${id}`); visitedRows.add(id);
    assert.equal(row.actor, cursor.actor);
    assert.deepEqual(row.action_labels, actions(cursor, game), 'action grid does not match the game');
    const actor = cursor.actor, n = row.action_labels.length;
    const children = row.action_labels.map((action, a) => {
      const child = ranges.map(r => Float64Array.from(r));
      if (!(best && actor === seat)) {
        for (let c = 0; c < N; c++) child[actor][c] *= row.probabilities[c * n + a];
      }
      return backUp(apply(cursor, action, game), child, seat, best, continueBest);
    });
    return Float64Array.from({length: N}, (_, c) => {
      if (actor === seat && best) return Math.max(...children.map(v => v[c]));
      let result = 0;
      for (let a = 0; a < n; a++) result += children[a][c] * (actor === seat ? row.probabilities[c * n + a] : 1);
      return result;
    });
  }
  const mass = compatibleMass(state.ranges[1]);
  const joint = mass.reduce((total, v, c) => total + state.ranges[0][c] * v, 0);
  assert.ok(joint > 0);
  const evaluate = (best, continueBest) => [0, 1].map(seat => {
    const values = backUp(state, state.ranges, seat, best, continueBest);
    return values.reduce((total, v, c) => total + state.ranges[seat][c] * v, 0) / joint;
  });
  const profile_bb = evaluate(false, false), best_response_bb = evaluate(true, true);
  const flopOnlyBest = evaluate(true, false);
  assert.equal(visitedRows.size, rows.size); assert.equal(visitedLeaves.size, leaves.size);
  close(profile_bb[0] + profile_bb[1], 0, 'zero sum');
  const gain_bb = best_response_bb.map((v, p) => v - profile_bb[p]);
  const half_summed_gain_bb = (gain_bb[0] + gain_bb[1]) / 2;
  const responseBytes = fs.readFileSync(responsePath), response = JSON.parse(responseBytes);
  assert.equal(response.schema, 'hu-native-flop-frozen-response-v1');
  assert.equal(response.candidate_sha256, candidateSha256);
  assert.equal(response.public_turns, 49); assert.equal(response.turn_policy_count, 49 * leaves.size);
  let maximumDifferenceBb = 0;
  for (const [name, values] of Object.entries({profile_bb, best_response_bb, gain_bb})) {
    for (let p = 0; p < 2; p++) {
      close(values[p], response[name][p], `${name} seat ${p}`);
      maximumDifferenceBb = Math.max(maximumDifferenceBb, Math.abs(values[p] - response[name][p]));
    }
  }
  close(half_summed_gain_bb, response.half_summed_gain_bb, 'half-summed gain');
  const flopOnlyGain = flopOnlyBest.map((v, p) => v - profile_bb[p]);
  assert.ok(gain_bb.every((v, p) => v >= -EPS && v + EPS >= flopOnlyGain[p] && flopOnlyGain[p] >= -EPS));
  return {schema: 'native-flop-independent-response-audit-v1', candidateSha256,
    responseSha256: hash(responseBytes), equityMetadataSha256: hash(equityMetadataBytes),
    packets, publicRows: rows.size, leaves: leaves.size, maximumDifferenceBb,
    profile_bb, best_response_bb, gain_bb, half_summed_gain_bb,
    flopOnlyRestrictedGainBb: flopOnlyGain,
    interpretation: 'Independent flop chance/back-up/betting audit with shared frozen turn CFVs and native f32 terminal equities. Conditional reused root, not full-game exploitability or independent equity validation.'};
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  assert.equal(process.argv.length, 6, 'usage: policy.json packets/ equity.json response.json');
  console.log(JSON.stringify(audit(...process.argv.slice(2)), null, 2));
}
