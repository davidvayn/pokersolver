import { readFile, rename, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { MAX_HOSTED_BYTES, parseArgs, required } from '../policy/lib.mjs';

const args = parseArgs(process.argv.slice(2));
const repositoryRoot = path.resolve(import.meta.dirname, '..', '..');
const version = required(args, '--model-version');
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
const verifier = path.join(import.meta.dirname, 'verify-resolver-artifacts.mjs');
const manifests = JSON.parse(await readFile(manifestPath, 'utf8'));
if (!Array.isArray(manifests)) throw new Error('Practice manifest registry must be an array');
const manifest = manifests.find((candidate) => candidate?.version === version);
if (!manifest) throw new Error(`No practice manifest matches ${version}`);
if (manifest.active === true || manifest.validation?.status === 'accepted') {
  throw new Error('Resolver manifest is already active or accepted');
}

const validation = manifest.validation ?? {};
const resolver = manifest.runtime?.resolver;
const dcfr = manifest.runtime?.dcfr;
const validResolvedActor = (actor) =>
  actor === null || actor === 0 || actor === 1;
const validExponent = (value) => Number.isFinite(value) && value >= 0;
const gates = {
  experimentalLabel: manifest.label === 'Experimental self-play',
  fullHandSubtype: manifest.subtype === 'full-hand',
  continualResolver: manifest.runtime?.kind === 'rust-continual-resolver-v1',
  resolverConfiguration:
    validExponent(dcfr?.positiveRegretExponent) &&
    validExponent(dcfr?.negativeRegretExponent) &&
    validExponent(dcfr?.strategyExponent) &&
    Number.isInteger(resolver?.flopIterations) &&
    resolver.flopIterations >= 2 &&
    validResolvedActor(resolver.flopResolvedActor) &&
    typeof resolver.flopDeploySolvedPolicy === 'boolean' &&
    Number.isInteger(resolver.turnIterations) &&
    resolver.turnIterations >= 2 &&
    validResolvedActor(resolver.turnResolvedActor) &&
    Number.isInteger(resolver.riverIterations) &&
    resolver.riverIterations >= 2 &&
    validResolvedActor(resolver.riverResolvedActor) &&
    resolver.deterministic === true,
  exploitabilityExplicitlyDeferred: validation.exploitabilityGateDeferred === true,
  crossSeedFrequencyMae:
    Number.isFinite(validation.crossSeedFrequencyMae) &&
    validation.crossSeedFrequencyMae <= 0.05,
  primaryActionAgreement:
    Number.isFinite(validation.primaryActionAgreement) &&
    validation.primaryActionAgreement >= 0.85,
  aggregateActionDelta:
    Number.isFinite(validation.maximumAggregateActionDelta) &&
    validation.maximumAggregateActionDelta <= 0.03,
  policyCoverage:
    Number.isFinite(validation.policyCoverage) &&
    validation.policyCoverage >= 0.9999,
  actionEvPrecision:
    Number.isFinite(validation.actionEvStandardErrorCoverage) &&
    validation.actionEvStandardErrorCoverage >= 0.95,
  storage:
    Number.isFinite(validation.projectedStorageBytes) &&
    validation.projectedStorageBytes <= MAX_HOSTED_BYTES,
  rawProbabilitySums: validation.rawProbabilitySumsValid === true,
  quantizedProbabilitySums: validation.quantizedProbabilitySumsValid === true,
  independentSeeds: validation.independentSeedCount === 2,
  trainingHours:
    Array.isArray(validation.trainingHoursPerSeed) &&
    validation.trainingHoursPerSeed.length === 2 &&
    validation.trainingHoursPerSeed.every(
      (hours) => Number.isFinite(hours) && hours >= 8 && hours <= 12
    ),
};
const failed = Object.entries(gates)
  .filter(([, passed]) => !passed)
  .map(([gate]) => gate);
if (failed.length > 0) {
  throw new Error(`Experimental resolver activation gates failed: ${failed.join(', ')}`);
}

manifest.active = true;
validation.status = 'accepted';
manifest.validation = validation;
const temporaryManifest = `${manifestPath}.activation-${process.pid}`;
await writeFile(temporaryManifest, `${JSON.stringify(manifests, null, 2)}\n`, {
  flag: 'wx',
});

const verification = spawnSync(
  process.execPath,
  [
    verifier,
    '--manifest',
    temporaryManifest,
    '--model-dir',
    modelRoot,
    '--model-version',
    version,
  ],
  { encoding: 'utf8' }
);
if (verification.status !== 0) {
  await rm(temporaryManifest, { force: true });
  throw new Error(
    `Experimental resolver artifact verification failed: ${verification.stderr.trim()}`
  );
}
await rename(temporaryManifest, manifestPath);

process.stdout.write(
  `${JSON.stringify(
    {
      activated: true,
      version,
      label: manifest.label,
      exploitabilityGateDeferred: true,
      normalGates: gates,
      artifactVerification: JSON.parse(verification.stdout),
    },
    null,
    2
  )}\n`
);
