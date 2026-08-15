import { access, readFile, stat } from 'node:fs/promises';
import { constants } from 'node:fs';
import path from 'node:path';
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
const executable = path.resolve(
  typeof args.get('--binary') === 'string'
    ? args.get('--binary')
    : path.join(
        repositoryRoot,
        'preflop-solver',
        'target',
        'release',
        'preflop-solver'
      )
);
const tracePath = path.resolve(
  typeof args.get('--trace') === 'string'
    ? args.get('--trace')
    : path.join(
        repositoryRoot,
        '.next',
        'server',
        'app',
        'api',
        'practice',
        'resolve',
        'route.js.nft.json'
      )
);
const requestedVersion = args.get('--model-version');
const manifests = JSON.parse(await readFile(manifestPath, 'utf8'));
if (!Array.isArray(manifests)) {
  throw new Error('Practice manifest registry must be an array');
}

const candidates = manifests.filter(
  (manifest) =>
    manifest?.runtime?.kind === 'rust-continual-resolver-v1' &&
    (typeof requestedVersion !== 'string' || manifest.version === requestedVersion)
);
if (candidates.length === 0) {
  throw new Error('No matching continual resolver manifest exists');
}

const safeFile = (file) =>
  typeof file === 'string' && /^[a-z0-9][a-z0-9.-]*\.json\.gz$/.test(file);
const expectedArtifacts = new Set();
for (const manifest of candidates) {
  const files = Object.values(manifest.runtime.artifactFiles ?? {});
  if (files.length !== 4 || files.some((file) => !safeFile(file))) {
    throw new Error(`${manifest.version} has invalid resolver artifact metadata`);
  }
  for (const file of files) expectedArtifacts.add(path.join(modelRoot, file));
}

const trace = JSON.parse(await readFile(tracePath, 'utf8'));
if (!Array.isArray(trace.files)) {
  throw new Error('Resolver route trace has no file list');
}
const tracedFiles = new Set(
  trace.files.map((file) => path.resolve(path.dirname(tracePath), file))
);
const requiredFiles = [executable, ...expectedArtifacts];
const missing = requiredFiles.filter((file) => !tracedFiles.has(file));
if (missing.length > 0) {
  throw new Error(
    `Resolver route trace omits pinned files: ${missing
      .map((file) => path.relative(repositoryRoot, file))
      .join(', ')}`
  );
}

const tracedArtifacts = [...tracedFiles].filter(
  (file) =>
    path.dirname(file) === modelRoot &&
    file.endsWith('.json.gz')
);
const unexpected = tracedArtifacts.filter((file) => !expectedArtifacts.has(file));
if (unexpected.length > 0) {
  throw new Error(
    `Resolver route trace contains unpinned model files: ${unexpected
      .map((file) => path.basename(file))
      .join(', ')}`
  );
}

await access(executable, constants.X_OK);
let pinnedBundleBytes = 0;
for (const file of requiredFiles) {
  const metadata = await stat(file);
  if (!metadata.isFile()) {
    throw new Error(`${file} is not a regular file`);
  }
  pinnedBundleBytes += metadata.size;
}

process.stdout.write(
  `${JSON.stringify(
    {
      verified: true,
      trace: path.relative(repositoryRoot, tracePath),
      executable: path.relative(repositoryRoot, executable),
      artifacts: [...expectedArtifacts]
        .map((file) => path.relative(repositoryRoot, file))
        .sort(),
      pinnedBundleBytes,
    },
    null,
    2
  )}\n`
);
