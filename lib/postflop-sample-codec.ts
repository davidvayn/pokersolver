import type { PostflopPracticeSample } from '@/lib/practice-types';

const MAGIC = new Uint8Array([0x50, 0x4c, 0x53, 0x31]); // PLS1
const SCHEMA_VERSION = 1;

function hexToBytes(hex: string): Uint8Array {
  if (!/^[a-f0-9]{64}$/.test(hex)) throw new Error('Invalid sample state hash');
  return Uint8Array.from(
    { length: 32 },
    (_, index) => Number.parseInt(hex.slice(index * 2, index * 2 + 2), 16)
  );
}

function bytesToHex(bytes: Uint8Array): string {
  return [...bytes]
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
}

function u32(value: number): Uint8Array {
  const bytes = new Uint8Array(4);
  new DataView(bytes.buffer).setUint32(0, value, true);
  return bytes;
}

function concatenate(parts: Uint8Array[]): Uint8Array {
  const result = new Uint8Array(
    parts.reduce((sum, part) => sum + part.byteLength, 0)
  );
  let offset = 0;
  for (const part of parts) {
    result.set(part, offset);
    offset += part.byteLength;
  }
  return result;
}

function validateSample(sample: PostflopPracticeSample): void {
  if (!/^[a-f0-9]{64}$/.test(sample.stateHash)) {
    throw new Error('Invalid sample state hash');
  }
  if (![20, 50, 100].includes(sample.depthBb)) {
    throw new Error('Invalid sample depth');
  }
  if (!['flop', 'turn', 'river'].includes(sample.street)) {
    throw new Error('Invalid sample street');
  }
  if (
    sample.state.depthBb !== sample.depthBb ||
    sample.state.street !== sample.street ||
    !Array.isArray(sample.replayActions)
  ) {
    throw new Error('Sample state metadata does not match its index');
  }
}

export function encodePostflopSampleShard(
  samples: PostflopPracticeSample[]
): Uint8Array {
  const encoder = new TextEncoder();
  const parts: Uint8Array[] = [MAGIC, Uint8Array.of(SCHEMA_VERSION), u32(samples.length)];
  for (const sample of [...samples].sort((first, second) =>
    first.stateHash.localeCompare(second.stateHash)
  )) {
    validateSample(sample);
    const payload = encoder.encode(JSON.stringify(sample));
    parts.push(hexToBytes(sample.stateHash), u32(payload.byteLength), payload);
  }
  return concatenate(parts);
}

export function decodePostflopSampleShard(
  bytes: Uint8Array
): PostflopPracticeSample[] {
  let offset = 0;
  const take = (length: number) => {
    if (length < 0 || offset + length > bytes.byteLength) {
      throw new Error('Truncated postflop sample shard');
    }
    const value = bytes.slice(offset, offset + length);
    offset += length;
    return value;
  };
  const readU32 = () => {
    const value = take(4);
    return new DataView(value.buffer, value.byteOffset, 4).getUint32(0, true);
  };
  if (!take(4).every((byte, index) => byte === MAGIC[index])) {
    throw new Error('Invalid postflop sample shard magic');
  }
  if (take(1)[0] !== SCHEMA_VERSION) {
    throw new Error('Unsupported postflop sample shard schema');
  }
  const count = readU32();
  const decoder = new TextDecoder();
  const samples: PostflopPracticeSample[] = [];
  for (let index = 0; index < count; index++) {
    const indexedHash = bytesToHex(take(32));
    const payload = JSON.parse(decoder.decode(take(readU32()))) as PostflopPracticeSample;
    validateSample(payload);
    if (payload.stateHash !== indexedHash) {
      throw new Error('Postflop sample hash index mismatch');
    }
    samples.push(payload);
  }
  if (offset !== bytes.byteLength) {
    throw new Error('Unexpected trailing postflop sample bytes');
  }
  return samples;
}
