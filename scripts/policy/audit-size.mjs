import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { MAX_HOSTED_BYTES, parseArgs, required } from './lib.mjs';

const args = parseArgs(process.argv.slice(2));
const indexes = String(required(args, '--indexes')).split(',').map((value) => path.resolve(value.trim()));
let bytes = 0;
for (const file of indexes) {
  const index = JSON.parse(await readFile(file, 'utf8'));
  bytes += index.estimatedHostedBytes ?? index.totalBytes;
}
const result = { bytes, gibibytes: bytes / 1024 ** 3, limitBytes: MAX_HOSTED_BYTES, headroomBytes: MAX_HOSTED_BYTES - bytes, passed: bytes <= MAX_HOSTED_BYTES };
process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
if (!result.passed) process.exitCode = 1;
