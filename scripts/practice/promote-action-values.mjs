import { constants } from 'node:fs';
import {
  copyFile,
  readFile,
  rename,
  stat,
  writeFile,
} from 'node:fs/promises';
import path from 'node:path';
import { gunzipSync } from 'node:zlib';
import { parseArgs, required, sha256 } from '../policy/lib.mjs';

const args = parseArgs(process.argv.slice(2));
const repositoryRoot = path.resolve(import.meta.dirname, '..', '..');
const artifactPath = path.resolve(required(args, '--artifact'));
const policyPath = path.resolve(required(args, '--policy'));
const version = required(args, '--model-version');
const targetFile = required(args, '--target-file');
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

if (!/^[a-z0-9][a-z0-9.-]*\.json\.gz$/.test(targetFile)) {
  throw new Error('Target action-value file must be a safe gzip JSON basename');
}

const compressed = await readFile(artifactPath);
const decoded = gunzipSync(compressed);
const artifact = JSON.parse(decoded.toString('utf8'));
const policySha256 = sha256(await readFile(policyPath));
if (
  artifact.schema !== 'hu-preflop-canonical-range-action-values-v1' ||
  artifact.policy_artifact_sha256 !== policySha256 ||
  !/^[a-f0-9]{64}$/.test(artifact.source_policy_sha256 ?? '') ||
  !Array.isArray(artifact.evaluated_information_sets) ||
  artifact.evaluated_information_sets.length !== 2 ||
  artifact.evaluated_information_sets.reduce((sum, count) => sum + count, 0) !== 16_900 ||
  !Number.isFinite(artifact.policy_lookup_coverage) ||
  artifact.policy_lookup_coverage < 0.9999 ||
  !Number.isFinite(artifact.action_ev_standard_error_coverage) ||
  artifact.action_ev_standard_error_coverage < 0.95
) {
  throw new Error('Canonical action-value artifact has not passed its serving gates');
}

const manifests = JSON.parse(await readFile(manifestPath, 'utf8'));
if (!Array.isArray(manifests)) throw new Error('Practice manifest registry must be an array');
const manifest = manifests.find((candidate) => candidate?.version === version);
if (manifest?.runtime?.kind !== 'rust-continual-resolver-v1') {
  throw new Error(`No continual resolver manifest matches ${version}`);
}
if (manifest.runtime.networkSha256 !== artifact.source_policy_sha256) {
  throw new Error('Canonical action values target a different served network');
}
if (manifest.active === true || manifest.validation?.status === 'accepted') {
  throw new Error('Refusing to mutate an already served resolver manifest');
}

const targetPath = path.join(modelRoot, targetFile);
try {
  const existing = await readFile(targetPath);
  if (!existing.equals(compressed)) {
    throw new Error(`Target artifact ${targetFile} already exists with different bytes`);
  }
} catch (error) {
  if (error?.code !== 'ENOENT') throw error;
  await copyFile(artifactPath, targetPath, constants.COPYFILE_EXCL);
}

manifest.runtime.artifactFiles.preflopActionValues = targetFile;
manifest.runtime.preflopActionValuesSha256 = sha256(decoded);
manifest.validation.actionEvStandardErrorCoverage =
  artifact.action_ev_standard_error_coverage;
manifest.generatedAt = new Date().toISOString();

const artifactFiles = Object.values(manifest.runtime.artifactFiles);
manifest.validation.projectedStorageBytes = (
  await Promise.all(
    artifactFiles.map(async (file) => {
      if (!/^[a-z0-9][a-z0-9.-]*\.json\.gz$/.test(file)) {
        throw new Error('Resolver manifest contains an unsafe artifact path');
      }
      return (await stat(path.join(modelRoot, file))).size;
    })
  )
).reduce((sum, size) => sum + size, 0);
const notes = Array.isArray(manifest.validation.notes)
  ? manifest.validation.notes.filter(
      (note) =>
        typeof note === 'string' &&
        !note.includes('preflop action-value corpus') &&
        !note.includes('feedback is low confidence') &&
        !note.includes('immutable losslessly compressed solver bundle')
    )
  : [];
notes.push(
  `The immutable losslessly compressed solver bundle is ${manifest.validation.projectedStorageBytes.toLocaleString('en-US')} bytes; no database is required for this embedded runtime.`
);
notes.push(
  `Canonical preflop action values pass ${(100 * artifact.action_ev_standard_error_coverage).toFixed(3)}% policy-action-weighted coverage at 0.02bb standard error over ${artifact.corpus_deals} raw flops. This is a conservative full-hand sampling-error lower bound: deterministic postflop resolver values require no Monte Carlo sampling, while their learned-continuation approximation remains separately labeled low confidence with uncalibrated uncertainty.`
);
manifest.validation.notes = notes;

const temporaryManifest = `${manifestPath}.tmp-${process.pid}`;
await writeFile(temporaryManifest, `${JSON.stringify(manifests, null, 2)}\n`, {
  flag: 'wx',
});
await rename(temporaryManifest, manifestPath);

process.stdout.write(
  `${JSON.stringify(
    {
      promoted: true,
      version,
      targetFile,
      decodedSha256: manifest.runtime.preflopActionValuesSha256,
      actionEvStandardErrorCoverage:
        manifest.validation.actionEvStandardErrorCoverage,
      projectedStorageBytes: manifest.validation.projectedStorageBytes,
      active: manifest.active,
      validationStatus: manifest.validation.status,
    },
    null,
    2
  )}\n`
);
