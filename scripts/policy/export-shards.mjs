import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import {
  MAX_HOSTED_BYTES,
  NORMAL_ITEM_BYTES,
  assertAcceptedManifest,
  encodeNodes,
  encodeSamples,
  parseArgs,
  required,
  sha256,
} from './lib.mjs';

const args = parseArgs(process.argv.slice(2));
const inputPath = path.resolve(required(args, '--input'));
const outputRoot = path.resolve(required(args, '--output'));
const source = JSON.parse(await readFile(inputPath, 'utf8'));
const { manifest, nodes, samples = [] } = source;
assertAcceptedManifest(manifest);
if (!Array.isArray(nodes) || nodes.length === 0) throw new Error('Policy export contains no nodes');
if (
  manifest.subtype === 'full-hand' &&
  nodes.some(
    (node) =>
      !Array.isArray(node.actions) ||
      node.actions.some(
        (action) =>
          !Number.isFinite(action.evBb) ||
          !Number.isFinite(action.standardErrorBb) ||
          !['high', 'low'].includes(action.confidence)
      )
  )
) {
  throw new Error(
    'Every served full-hand action requires an evaluated EV, standard error, and confidence grade'
  );
}

const output = path.join(outputRoot, manifest.version);
await mkdir(output, { recursive: true });
await writeFile(path.join(output, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`, { flag: 'wx' });
const shards = [];
let totalBytes = 0;
let estimatedItems = 0;

function createShardGroups(entries, encode) {
  const firstPass = new Map();
  for (const entry of entries) {
    const prefix = entry.stateHash.slice(0, 4);
    const group = firstPass.get(prefix) ?? [];
    group.push(entry);
    firstPass.set(prefix, group);
  }
  const finalGroups = new Map();
  for (const [prefix, group] of firstPass) {
    const encoded = encode(group);
    if (encoded.byteLength <= 256 * 1024) {
      finalGroups.set(prefix, encoded);
      continue;
    }
    const split = new Map();
    for (const entry of group) {
      const extended = entry.stateHash.slice(0, 6);
      const values = split.get(extended) ?? [];
      values.push(entry);
      split.set(extended, values);
    }
    for (const [extended, values] of split) {
      const payload = encode(values);
      if (payload.byteLength > 1024 * 1024) {
        throw new Error(`Extended shard ${extended} exceeds the 1MB serving guard`);
      }
      finalGroups.set(extended, payload);
    }
  }
  return finalGroups;
}

for (const depthBb of manifest.depthsBb) {
  const depthNodes = nodes.filter((node) => node.depthBb === depthBb);
  if (depthNodes.length === 0) throw new Error(`No nodes found for ${depthBb}bb`);
  const finalGroups = createShardGroups(depthNodes, encodeNodes);

  const depthDirectory = path.join(output, String(depthBb));
  await mkdir(depthDirectory, { recursive: true });
  for (const [prefix, payload] of [...finalGroups].sort(([first], [second]) => first.localeCompare(second))) {
    const filename = `${prefix}.bin`;
    await writeFile(path.join(depthDirectory, filename), payload, { flag: 'wx' });
    totalBytes += payload.byteLength;
    estimatedItems += Math.ceil(payload.byteLength / NORMAL_ITEM_BYTES);
    shards.push({
      kind: 'policy',
      version: manifest.version,
      depthBb,
      prefix,
      filename: path.join(manifest.version, String(depthBb), filename),
      bytes: payload.byteLength,
      sha256: sha256(payload),
    });
  }
}

if (!Array.isArray(samples)) throw new Error('Policy samples must be an array');
for (const depthBb of manifest.depthsBb) {
  for (const street of ['flop', 'turn', 'river']) {
    const streetSamples = samples.filter(
      (sample) => sample.depthBb === depthBb && sample.street === street
    );
    if (streetSamples.length === 0) continue;
    const finalGroups = createShardGroups(streetSamples, encodeSamples);
    const sampleDirectory = path.join(
      outputRoot,
      'samples',
      manifest.version,
      String(depthBb),
      street
    );
    await mkdir(sampleDirectory, { recursive: true });
    for (const [prefix, payload] of [...finalGroups].sort(([first], [second]) => first.localeCompare(second))) {
      const filename = `${prefix}.bin`;
      await writeFile(path.join(sampleDirectory, filename), payload, { flag: 'wx' });
      totalBytes += payload.byteLength;
      estimatedItems += Math.ceil(payload.byteLength / NORMAL_ITEM_BYTES);
      shards.push({
        kind: 'sample',
        version: manifest.version,
        depthBb,
        street,
        prefix,
        filename: path.join('samples', manifest.version, String(depthBb), street, filename),
        bytes: payload.byteLength,
        sha256: sha256(payload),
      });
    }
  }
}

const existingHostedBytes = Number(args.get('--existing-hosted-bytes') ?? 0);
if (!Number.isFinite(existingHostedBytes) || existingHostedBytes < 0) {
  throw new Error('--existing-hosted-bytes must be a non-negative number');
}
// DynamoDB table size includes key/attribute metadata as well as binary
// payloads. Reserve a conservative 1KB per 24KB item instead of auditing only
// the compact shard files.
const estimatedHostedBytes = totalBytes + estimatedItems * 1024;
if (existingHostedBytes + estimatedHostedBytes > MAX_HOSTED_BYTES) {
  throw new Error(`Projected hosted policy size ${existingHostedBytes + estimatedHostedBytes} exceeds 20GB`);
}

const index = {
  schemaVersion: 1,
  createdAt: new Date().toISOString(),
  manifest,
  sourceSha256: sha256(await readFile(inputPath)),
  totalBytes,
  estimatedHostedBytes,
  estimatedItems,
  projectedHostedBytes: existingHostedBytes + estimatedHostedBytes,
  shards,
};
await writeFile(path.join(output, 'export-index.json'), `${JSON.stringify(index, null, 2)}\n`, { flag: 'wx' });
process.stdout.write(`${JSON.stringify({ version: manifest.version, nodes: nodes.length, samples: samples.length, shards: shards.length, totalBytes, estimatedHostedBytes })}\n`);
