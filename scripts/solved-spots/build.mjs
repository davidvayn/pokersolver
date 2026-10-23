// Generate immutable, self-contained river artifacts with the MIT Rust engine.
// Run npm run build:solver first. No cloud, network, neural weights or database.
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtempSync, readFileSync, mkdirSync, writeFileSync, existsSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { isDeepStrictEqual } from 'node:util';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const binary = path.join(root, 'preflop-solver/target/release/preflop-solver');
const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
const verifyOnly = process.argv.includes('--verify');
function sameReport(actual, expected) {
  return Object.keys(actual).length === Object.keys(expected).length && Object.keys(actual).every((key) => {
    const a = actual[key], b = expected[key];
    if (typeof a === 'number') return Number.isFinite(a) && Number.isFinite(b) && Math.abs(a-b) <= 1e-9;
    if (Array.isArray(a)) return Array.isArray(b) && a.length === b.length && a.every((v,i) => Number.isFinite(v) && Math.abs(v-b[i]) <= 1e-9);
    return a === b;
  });
}
function run(args) {
  const result = spawnSync(binary, args, { cwd: root, encoding: 'utf8', timeout: 600_000, maxBuffer: 2_000_000 });
  if (result.error || result.status !== 0) throw new Error(result.error?.message ?? result.stderr);
  return JSON.parse(result.stdout);
}
const catalogPath = path.join(root, 'data/solved-spots/catalog.json');
if (verifyOnly) {
  const catalog = JSON.parse(readFileSync(catalogPath, 'utf8'));
  for (const entry of catalog) {
    if (!/^\/models\/solved-spots\/[a-z0-9-]+\/[a-f0-9]{64}\.json$/.test(entry.url)) throw new Error('Invalid artifact URL');
    const file = path.join(root, 'public', entry.url);
    const bytes = readFileSync(file);
    if (digest(bytes) !== entry.sha256 || bytes.length !== entry.bytes) throw new Error('Artifact identity mismatch');
    const spot = JSON.parse(bytes);
    const source = JSON.parse(readFileSync(path.join(root, 'data/solved-spots', `${entry.id}.json`), 'utf8'));
    if (!isDeepStrictEqual(source, spot.input)) throw new Error('Source input differs from artifact');
    const report = run(['solved-spot', '--verify', '--input', file]);
    if (!report.accepted || !sameReport(report,spot.verification)
      || !sameReport(report,entry.verification)) throw new Error('Verification mismatch');
    console.log(`${entry.id}: verified ${entry.bytes} bytes; total BR gain ${report.totalBestResponseGainBb.toFixed(6)}bb/spot`);
  }
} else {
  const temp = mkdtempSync(path.join(tmpdir(), 'poker-solved-spots-'));
  try {
    const entries = [];
    for (const id of ['river-50bb-paired']) {
      const input = path.join(root, 'data/solved-spots', `${id}.json`);
      const output = path.join(temp, `${id}.json`);
      run(['solved-spot', '--input', input, '--output', output]);
      const bytes = readFileSync(output);
      const spot = JSON.parse(bytes);
      const report = run(['solved-spot', '--verify', '--input', output]);
      if (!report.accepted) throw new Error('Independent exported-policy verification failed');
      const sha256 = digest(bytes);
      const url = `/models/solved-spots/${id}/${sha256}.json`;
      const target = path.join(root, 'public', url);
      mkdirSync(path.dirname(target), { recursive: true });
      if (existsSync(target)) {
        if (digest(readFileSync(target)) !== sha256) throw new Error('Immutable artifact collision');
      } else writeFileSync(target, bytes, { flag: 'wx' });
      entries.push({ id, title: spot.input.title, board: spot.input.board, depthBb: spot.input.depthBb,
        url, sha256, bytes: bytes.length, verification: report });
      console.log(`${id}: ${bytes.length} bytes; total BR gain ${report.totalBestResponseGainBb.toFixed(6)}bb/spot`);
    }
    writeFileSync(catalogPath, `${JSON.stringify(entries, null, 2)}\n`);
  } finally { rmSync(temp, { recursive: true }); }
}
