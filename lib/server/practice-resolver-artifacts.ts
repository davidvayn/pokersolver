import 'server-only';

import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { open, stat } from 'node:fs/promises';
import path from 'node:path';
import { createGunzip } from 'node:zlib';
import type { ContinualResolverRuntime } from '@/lib/practice-types';

const MAX_COMPRESSED_BYTES = 64 * 1024 ** 2;
const MAX_DECODED_BYTES = 256 * 1024 ** 2;
const MAX_BINARY_BYTES = 128 * 1024 ** 2;
const HEX_DIGEST = /^[a-f0-9]{64}$/;
const SAFE_FILE = /^[a-z0-9][a-z0-9.-]*\.json\.gz$/;

/** Verify canonical identities before loading a model, without buffering JSON. */
export async function verifyPracticeResolverArtifacts(
  root: string,
  runtime: ContinualResolverRuntime
): Promise<void> {
  const expected = {
    networks: runtime.networkSha256,
    rangePolicy: runtime.rangePolicySha256,
    preflopActionValues: runtime.preflopActionValuesSha256,
    flopValueNetwork: runtime.valueNetworkSha256,
  };
  for (const kind of Object.keys(expected) as Array<keyof typeof expected>) {
    const file = runtime.artifactFiles[kind];
    const digest = expected[kind];
    if (!SAFE_FILE.test(file) || !HEX_DIGEST.test(digest)) {
      throw new Error(`Unsafe pinned resolver ${kind} metadata`);
    }
    const artifactPath = path.join(root, file);
    const info = await stat(artifactPath);
    if (!info.isFile() || info.size <= 0 || info.size > MAX_COMPRESSED_BYTES) {
      throw new Error(`Pinned resolver ${kind} exceeds its loading budget`);
    }
    const source = createReadStream(artifactPath);
    const decoded = createGunzip();
    source.on('error', (error) => decoded.destroy(error));
    source.pipe(decoded);
    const hash = createHash('sha256');
    let length = 0;
    try {
      for await (const bytes of decoded) {
        length += bytes.length;
        if (length > MAX_DECODED_BYTES) {
          throw new Error(`Pinned resolver ${kind} exceeds its decoded budget`);
        }
        hash.update(bytes);
      }
    } finally {
      source.destroy();
      decoded.destroy();
    }
    if (hash.digest('hex') !== digest) {
      throw new Error(`Pinned resolver ${kind} canonical hash mismatch`);
    }
    // Native loading prefers an optional compiled sidecar. Verify that too;
    // checking only gzip would not establish what the worker actually loads.
    const binaryPath = `${artifactPath}.bin`;
    let handle;
    try {
      handle = await open(binaryPath, 'r');
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') continue;
      throw error;
    }
    try {
      const binaryInfo = await handle.stat();
      const header = Buffer.alloc(80);
      const read = await handle.read(header, 0, 80, 0);
      if (!binaryInfo.isFile() || binaryInfo.size > MAX_BINARY_BYTES || read.bytesRead !== 80
          || header.subarray(0, 8).toString() !== 'PKRMODL2'
          || header.subarray(8, 40).toString('hex') !== digest
          || header.readBigUInt64LE(72) !== BigInt(binaryInfo.size - 80)) {
        throw new Error(`Pinned resolver ${kind} binary identity mismatch`);
      }
      const payloadHash = createHash('sha256');
      for await (const bytes of handle.createReadStream({ start: 80, autoClose: false })) {
        payloadHash.update(bytes);
      }
      if (payloadHash.digest('hex') !== header.subarray(40, 72).toString('hex')) {
        throw new Error(`Pinned resolver ${kind} binary payload mismatch`);
      }
    } finally {
      await handle.close();
    }
  }
}
