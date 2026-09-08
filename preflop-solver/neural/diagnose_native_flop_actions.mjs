// Reuse saved all-turn evaluations; never call a solver or overwrite a report.
// node diagnose_native_flop_actions.mjs candidate.json packets/ equity.json response.json output.json
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { audit } from './audit_native_flop_response.mjs';

assert.equal(process.argv.length, 7,
  'usage: candidate.json packets/ equity.json response.json output.json');
const paths = process.argv.slice(2), output = paths.pop();
assert.ok(!fs.existsSync(output), 'refusing to overwrite a diagnostic artifact');
const report = audit(...paths, {actionDiagnostics: true});
const sourceSha256 = Object.fromEntries([
  'audit_native_flop_response.mjs', 'native_action_diagnostics.mjs', 'diagnose_native_flop_actions.mjs',
].map(name => [name, crypto.createHash('sha256').update(
  fs.readFileSync(fileURLToPath(new URL(name, import.meta.url)))).digest('hex')]));
fs.writeFileSync(output, JSON.stringify({...report, sourceSha256, releaseAccepted: false}), {flag: 'wx'});
console.log(JSON.stringify({output, responseGainBb: report.half_summed_gain_bb,
  costliestDecisions: [...report.actionDiagnostics]
    .sort((a, b) => b.rootWeightedContributionBb - a.rootWeightedContributionBb)
    .slice(0, 5).map(({costlyHands, ...row}) => ({...row, costlyHands: costlyHands.slice(0, 2)}))}, null, 2));
