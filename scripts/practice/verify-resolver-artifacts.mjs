import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { gunzipSync } from 'node:zlib';
import { parseArgs } from '../policy/lib.mjs';

const args = parseArgs(process.argv.slice(2));
const repositoryRoot = path.resolve(import.meta.dirname, '..', '..');
const manifestPath = path.resolve(
  typeof args.get('--manifest') === 'string'
    ? args.get('--manifest')
    : path.join(repositoryRoot, 'data', 'practice', 'full-hand-manifests.json')
);
const modelRoot = path.resolve(
  typeof args.get('--model-dir') === 'string'
    ? args.get('--model-dir')
    : path.join(repositoryRoot, 'preflop-solver', 'models', 'practice')
);
const requestedVersion = args.get('--model-version');
const manifests = JSON.parse(await readFile(manifestPath, 'utf8'));

if (!Array.isArray(manifests)) throw new Error('Practice manifest registry must be an array');

const candidates = manifests.filter(
  (manifest) =>
    manifest?.runtime?.kind === 'rust-continual-resolver-v1' &&
    (typeof requestedVersion !== 'string' || manifest.version === requestedVersion)
);
if (candidates.length === 0) throw new Error('No matching continual resolver manifest exists');

const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
const safeFile = (file) =>
  typeof file === 'string' && /^[a-z0-9][a-z0-9.-]*\.json\.gz$/.test(file);
const finite = (value) => typeof value === 'number' && Number.isFinite(value);

const results = [];
for (const manifest of candidates) {
  const runtime = manifest.runtime;
  const expected = {
    networks: runtime.networkSha256,
    rangePolicy: runtime.rangePolicySha256,
    preflopActionValues: runtime.preflopActionValuesSha256,
    flopValueNetwork: runtime.valueNetworkSha256,
  };
  const decoded = {};
  const identities = {};
  for (const [kind, expectedSha256] of Object.entries(expected)) {
    const file = runtime.artifactFiles?.[kind];
    if (!safeFile(file) || !/^[a-f0-9]{64}$/.test(expectedSha256 ?? '')) {
      throw new Error(`${manifest.version} has invalid ${kind} artifact metadata`);
    }
    const compressed = await readFile(path.join(modelRoot, file));
    const bytes = gunzipSync(compressed);
    const actualSha256 = digest(bytes);
    if (actualSha256 !== expectedSha256) {
      throw new Error(
        `${manifest.version} ${kind} decoded SHA-256 ${actualSha256} differs from ${expectedSha256}`
      );
    }
    decoded[kind] = JSON.parse(bytes.toString('utf8'));
    identities[kind] = { file, decodedSha256: actualSha256, compressedBytes: compressed.length };
  }

  const actionValues = decoded.preflopActionValues;
  const evaluatedInformationSets = actionValues.evaluated_information_sets;
  if (
    !Array.isArray(evaluatedInformationSets) ||
    evaluatedInformationSets.length !== 2 ||
    evaluatedInformationSets.some((count) => !Number.isInteger(count) || count <= 0) ||
    evaluatedInformationSets.reduce((sum, count) => sum + count, 0) !== 16_900 ||
    !finite(actionValues.policy_lookup_coverage) ||
    actionValues.policy_lookup_coverage < 0.9999 ||
    !finite(actionValues.action_ev_standard_error_coverage) ||
    actionValues.action_ev_standard_error_coverage < 0 ||
    actionValues.action_ev_standard_error_coverage > 1
  ) {
    throw new Error(`${manifest.version} action-value artifact is incomplete`);
  }
  const declaredCoverage = manifest.validation?.actionEvStandardErrorCoverage;
  if (
    finite(declaredCoverage) &&
    Math.abs(declaredCoverage - actionValues.action_ev_standard_error_coverage) > 1e-12
  ) {
    throw new Error(`${manifest.version} action-EV coverage differs from its manifest`);
  }
  if (manifest.active === true || manifest.validation?.status === 'accepted') {
    if (
      actionValues.schema !== 'hu-preflop-canonical-range-action-values-v1' ||
      actionValues.action_ev_standard_error_coverage < 0.95 ||
      actionValues.source_policy_sha256 !== runtime.networkSha256 ||
      !/^[a-f0-9]{64}$/.test(actionValues.policy_artifact_sha256 ?? '')
    ) {
      throw new Error(`${manifest.version} cannot serve its preflop action values`);
    }
  }

  results.push({
    version: manifest.version,
    active: manifest.active === true,
    validationStatus: manifest.validation?.status,
    artifacts: identities,
    actionEvStandardErrorCoverage: actionValues.action_ev_standard_error_coverage,
    policyLookupCoverage: actionValues.policy_lookup_coverage,
    evaluatedInformationSets,
  });
}

process.stdout.write(`${JSON.stringify({ verified: true, models: results }, null, 2)}\n`);
