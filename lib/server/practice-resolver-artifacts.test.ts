import { afterEach, describe, expect, it, vi } from 'vitest';
import { createHash } from 'node:crypto';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { gzipSync } from 'node:zlib';
import fullHandManifests from '@/data/practice/full-hand-manifests.json';
import type { ContinualResolverRuntime } from '@/lib/practice-types';

vi.mock('server-only', () => ({}));
import { verifyPracticeResolverArtifacts } from '@/lib/server/practice-resolver-artifacts';

const directories: string[] = [];
const digest = (bytes: Buffer) => createHash('sha256').update(bytes).digest('hex');
async function fixture(binary = false) {
  const root = await mkdtemp(path.join(os.tmpdir(),'poker-resolver-integrity-'));
  directories.push(root);
  const runtime = structuredClone(fullHandManifests[0].runtime) as ContinualResolverRuntime;
  const canonical = Buffer.from('{"fixture":true}');
  const hash = digest(canonical);
  Object.assign(runtime,{networkSha256:hash,rangePolicySha256:hash,valueNetworkSha256:hash,preflopActionValuesSha256:hash});
  const payload = Buffer.from('fixture-payload');
  const header = Buffer.alloc(80);
  header.write('PKRMODL2');
  Buffer.from(hash,'hex').copy(header,8);
  Buffer.from(digest(payload),'hex').copy(header,40);
  header.writeBigUInt64LE(BigInt(payload.length),72);
  for (const file of Object.values(runtime.artifactFiles)) {
    await writeFile(path.join(root,file),gzipSync(canonical));
    if (binary) await writeFile(path.join(root,`${file}.bin`),Buffer.concat([header,payload]));
  }
  return {root,runtime,header,payload};
}
afterEach(async () => { await Promise.all(directories.splice(0).map(root=>rm(root,{recursive:true,force:true}))); });

describe('bounded pinned worker artifact loading',()=>{
  it('verifies decoded JSON hashes with and without native sidecars',async()=>{
    for (const binary of [false,true]) {
      const {root,runtime} = await fixture(binary);
      await expect(verifyPracticeResolverArtifacts(root,runtime)).resolves.toBeUndefined();
    }
  });
  it('rejects canonical identity drift before spawning a worker',async()=>{
    const {root,runtime} = await fixture();
    await writeFile(path.join(root,runtime.artifactFiles.networks),gzipSync(Buffer.from('{}')));
    await expect(verifyPracticeResolverArtifacts(root,runtime)).rejects.toThrow('canonical hash mismatch');
  });
  it('rejects wrong sidecar identity, truncated payloads, and payload corruption',async()=>{
    for (const mutation of ['identity','truncated','payload']) {
      const {root,runtime,header,payload} = await fixture(true);
      if (mutation === 'identity') header[8] ^= 1;
      if (mutation === 'payload') payload[0] ^= 1;
      await writeFile(path.join(root,`${runtime.artifactFiles.networks}.bin`),
        mutation === 'truncated' ? header : Buffer.concat([header,payload]));
      await expect(verifyPracticeResolverArtifacts(root,runtime)).rejects.toThrow('binary');
    }
  });
  it('rejects missing artifacts, malformed gzip, and unsafe metadata',async()=>{
    const {root,runtime} = await fixture();
    const file = path.join(root,runtime.artifactFiles.networks);
    await writeFile(file,Buffer.from('not-gzip'));
    await expect(verifyPracticeResolverArtifacts(root,runtime)).rejects.toThrow();
    await rm(file);
    await expect(verifyPracticeResolverArtifacts(root,runtime)).rejects.toThrow();
    runtime.artifactFiles.networks = '../outside.json.gz';
    await expect(verifyPracticeResolverArtifacts(root,runtime)).rejects.toThrow('Unsafe');
  });
});
