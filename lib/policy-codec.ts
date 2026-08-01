import type {
  ActionKind,
  ConfidenceLevel,
  PolicyAction,
  PolicyNode,
} from '@/lib/practice-types';
import { validatePolicyNode } from '@/lib/practice-grading';

const MAGIC = new Uint8Array([0x50, 0x4c, 0x50, 0x31]); // PLP1
const SCHEMA_VERSION = 1;
const ACTION_KINDS: ActionKind[] = [
  'fold',
  'check',
  'call',
  'bet',
  'raise',
  'all-in',
];
const CONFIDENCE: ConfidenceLevel[] = ['high', 'low', 'unavailable'];

function hexToBytes(hex: string): Uint8Array {
  if (!/^[a-f0-9]{64}$/.test(hex)) throw new Error('Invalid state hash');
  const result = new Uint8Array(32);
  for (let index = 0; index < result.length; index++) {
    result[index] = Number.parseInt(hex.slice(index * 2, index * 2 + 2), 16);
  }
  return result;
}

function bytesToHex(bytes: Uint8Array): string {
  return [...bytes]
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
}

class Writer {
  private values: number[] = [];
  private encoder = new TextEncoder();

  bytes(value: Uint8Array): void {
    this.values.push(...value);
  }

  u8(value: number): void {
    this.values.push(value & 0xff);
  }

  u16(value: number): void {
    this.values.push(value & 0xff, (value >>> 8) & 0xff);
  }

  u32(value: number): void {
    this.values.push(
      value & 0xff,
      (value >>> 8) & 0xff,
      (value >>> 16) & 0xff,
      (value >>> 24) & 0xff
    );
  }

  f32(value: number | null | undefined): void {
    const bytes = new Uint8Array(4);
    new DataView(bytes.buffer).setFloat32(0, value ?? Number.NaN, true);
    this.bytes(bytes);
  }

  string(value: string): void {
    const bytes = this.encoder.encode(value);
    if (bytes.length > 255) throw new Error('Policy strings must fit in 255 bytes');
    this.u8(bytes.length);
    this.bytes(bytes);
  }

  finish(): Uint8Array {
    return Uint8Array.from(this.values);
  }
}

class Reader {
  private offset = 0;
  private decoder = new TextDecoder();

  constructor(private bytes: Uint8Array) {}

  take(length: number): Uint8Array {
    if (length < 0 || this.offset + length > this.bytes.length) {
      throw new Error('Truncated policy shard');
    }
    const result = this.bytes.slice(this.offset, this.offset + length);
    this.offset += length;
    return result;
  }

  u8(): number {
    return this.take(1)[0];
  }

  u16(): number {
    const bytes = this.take(2);
    return bytes[0] | (bytes[1] << 8);
  }

  u32(): number {
    const bytes = this.take(4);
    return new DataView(bytes.buffer, bytes.byteOffset, 4).getUint32(0, true);
  }

  f32(): number | null {
    const bytes = this.take(4);
    const value = new DataView(bytes.buffer, bytes.byteOffset, 4).getFloat32(0, true);
    return Number.isNaN(value) ? null : value;
  }

  string(): string {
    return this.decoder.decode(this.take(this.u8()));
  }

  done(): boolean {
    return this.offset === this.bytes.length;
  }
}

export function quantizeProbabilities(values: number[]): number[] {
  const quantized = values.map((value) =>
    Math.max(0, Math.min(65_535, Math.round(value * 65_535)))
  );
  const difference = 65_535 - quantized.reduce((sum, value) => sum + value, 0);
  if (difference !== 0 && quantized.length > 0) {
    const largest = values.reduce(
      (best, value, index) => (value > values[best] ? index : best),
      0
    );
    quantized[largest] += difference;
  }
  return quantized;
}

export function encodePolicyShard(nodes: PolicyNode[]): Uint8Array {
  const writer = new Writer();
  writer.bytes(MAGIC);
  writer.u8(SCHEMA_VERSION);
  writer.u32(nodes.length);
  for (const node of nodes) {
    const errors = validatePolicyNode(node);
    if (errors.length > 0) throw new Error(`Invalid policy node: ${errors.join('; ')}`);
    writer.bytes(hexToBytes(node.stateHash));
    writer.u8(node.actions.length);
    writer.u8(
      node.bestActionId === null
        ? 255
        : node.actions.findIndex((action) => action.id === node.bestActionId)
    );
    writer.f32(node.bestActionEvBb);
    writer.f32(node.reachProbability);
    const probabilities = quantizeProbabilities(
      node.actions.map((action) => action.probability)
    );
    for (const [actionIndex, action] of node.actions.entries()) {
      writer.u8(ACTION_KINDS.indexOf(action.kind));
      writer.string(action.id);
      writer.string(action.label);
      writer.f32(action.amountToBb);
      writer.u16(probabilities[actionIndex]);
      writer.f32(action.evBb);
      writer.f32(action.standardErrorBb);
      writer.u8(CONFIDENCE.indexOf(action.confidence));
    }
  }
  return writer.finish();
}

export function decodePolicyShard(bytes: Uint8Array): PolicyNode[] {
  const reader = new Reader(bytes);
  const magic = reader.take(4);
  if (!magic.every((byte, index) => byte === MAGIC[index])) {
    throw new Error('Invalid policy shard magic');
  }
  if (reader.u8() !== SCHEMA_VERSION) {
    throw new Error('Unsupported policy shard schema');
  }
  const count = reader.u32();
  const nodes: PolicyNode[] = [];
  for (let nodeIndex = 0; nodeIndex < count; nodeIndex++) {
    const stateHash = bytesToHex(reader.take(32));
    const actionCount = reader.u8();
    const bestIndex = reader.u8();
    const bestActionEvBb = reader.f32();
    const reachProbability = reader.f32();
    const actions: PolicyAction[] = [];
    for (let actionIndex = 0; actionIndex < actionCount; actionIndex++) {
      const kind = ACTION_KINDS[reader.u8()];
      if (!kind) throw new Error('Invalid action kind in policy shard');
      const id = reader.string();
      const label = reader.string();
      const amountToBb = reader.f32();
      const probability = reader.u16() / 65_535;
      const evBb = reader.f32();
      const standardErrorBb = reader.f32();
      const confidence = CONFIDENCE[reader.u8()];
      if (!confidence) throw new Error('Invalid confidence in policy shard');
      actions.push({
        id,
        kind,
        label,
        ...(amountToBb === null ? {} : { amountToBb }),
        probability,
        evBb,
        standardErrorBb,
        confidence,
      });
    }
    if (bestIndex !== 255 && bestIndex >= actions.length) {
      throw new Error('Invalid best-action index in policy shard');
    }
    nodes.push({
      stateHash,
      actions,
      bestActionId: bestIndex === 255 ? null : actions[bestIndex].id,
      bestActionEvBb,
      ...(reachProbability === null ? {} : { reachProbability }),
    });
  }
  if (!reader.done()) throw new Error('Unexpected trailing policy shard bytes');
  return nodes;
}

export function policyNodeFromShard(
  nodes: PolicyNode[],
  stateHash: string
): PolicyNode | null {
  return nodes.find((node) => node.stateHash === stateHash) ?? null;
}
