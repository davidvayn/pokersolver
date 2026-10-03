// Compare actual frozen-belief predictions with saved native packets. No solve.
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import {audit} from './audit_native_flop_response.mjs';

assert.ok([8,9,10].includes(process.argv.length),
  'usage: candidate packets equity response predictions output [alternative-model-sha256] [--action-diagnostics]');
const [candidate,packets,equity,response,predictions,output]=process.argv.slice(2);
const extras=process.argv.slice(8);
const actionDiagnostics=extras.includes('--action-diagnostics');
const alternativeValueModelSha256=extras.find(value=>value!=='--action-diagnostics');
assert.equal(extras.length,Number(actionDiagnostics)+Number(Boolean(alternativeValueModelSha256)));
assert.ok(fs.statSync(predictions).size<=64*1024*1024);
const bytes=fs.readFileSync(predictions), probe=JSON.parse(bytes);
const result=audit(candidate,packets,equity,response,{leafPredictions:probe,
  ...(actionDiagnostics ? {actionDiagnostics: true} : {}),
  ...(alternativeValueModelSha256 ? {alternativeValueModelSha256} : {})});
result.predictionsSha256=crypto.createHash('sha256').update(bytes).digest('hex');
result.predictionModelSha256=probe.model_sha256;
result.releaseAccepted=false;
fs.writeFileSync(output,JSON.stringify(result,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({output,measuredNativeResponseGainBb:result.half_summed_gain_bb,
  decisions:result.actionValueDiagnostics.map(({history,nativeBestAgreement,nativeLossFromPredictedBestBb,
    nativePolicyDeviationLossBb,predictedPolicyDeviationLossBb,actionContrastRmseBb})=>({history,
    nativeBestAgreement,nativeLossFromPredictedBestBb,nativePolicyDeviationLossBb,
    predictedPolicyDeviationLossBb,actionContrastRmseBb}))}));
