import 'server-only';

import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import {
  DynamoDBClient,
  GetItemCommand,
  QueryCommand,
} from '@aws-sdk/client-dynamodb';
import { awsCredentialsProvider } from '@vercel/oidc-aws-credentials-provider';
import type { PolicyManifest } from '@/lib/practice-types';
import { isValidatedFullHandManifest } from '@/lib/practice-models';

export const POLICY_REGION = 'us-west-2';
const TABLE_NAME = process.env.PRACTICE_POLICY_TABLE;
const LOCAL_DIRECTORY = process.env.PRACTICE_POLICY_LOCAL_DIR;
const RUNTIME_ROLE_ARN = process.env.AWS_ROLE_ARN;
const OIDC_AUDIENCE = process.env.VERCEL_OIDC_AUDIENCE;

let client: DynamoDBClient | null = null;

function dynamo(): DynamoDBClient {
  const credentials = RUNTIME_ROLE_ARN
    ? OIDC_AUDIENCE
      ? awsCredentialsProvider({
          roleArn: RUNTIME_ROLE_ARN,
          audience: OIDC_AUDIENCE,
        })
      : awsCredentialsProvider({ roleArn: RUNTIME_ROLE_ARN })
    : undefined;
  client ??= new DynamoDBClient({ region: POLICY_REGION, credentials });
  return client;
}

export function validPolicyRoutePart(value: string): boolean {
  return /^[a-z0-9][a-z0-9._-]{0,127}$/i.test(value);
}

export function validShardPrefix(value: string): boolean {
  return /^(?:[a-f0-9]{4}|[a-f0-9]{6})$/.test(value);
}

function localShardPath(version: string, depth: number, prefix: string): string {
  if (!LOCAL_DIRECTORY) throw new Error('Local policy directory is not configured');
  const root = path.resolve(LOCAL_DIRECTORY);
  const target = path.resolve(root, version, String(depth), `${prefix}.bin`);
  if (!target.startsWith(`${root}${path.sep}`)) throw new Error('Invalid local shard path');
  return target;
}

function localSamplePath(
  version: string,
  depth: number,
  street: string,
  prefix: string
): string {
  if (!LOCAL_DIRECTORY) throw new Error('Local policy directory is not configured');
  const root = path.resolve(LOCAL_DIRECTORY);
  const target = path.resolve(
    root,
    'samples',
    version,
    String(depth),
    street,
    `${prefix}.bin`
  );
  if (!target.startsWith(`${root}${path.sep}`)) throw new Error('Invalid local sample path');
  return target;
}

function concatenate(parts: Uint8Array[]): Uint8Array {
  const length = parts.reduce((sum, part) => sum + part.byteLength, 0);
  const result = new Uint8Array(length);
  let offset = 0;
  for (const part of parts) {
    result.set(part, offset);
    offset += part.byteLength;
  }
  return result;
}

function verifiedDynamoPayload(
  items: Array<Record<string, { B?: Uint8Array; N?: string; S?: string; BOOL?: boolean }>>
): Uint8Array | null {
  if (items.length === 0) return null;
  const ordered = [...items].sort(
    (first, second) => Number(first.part?.N ?? -1) - Number(second.part?.N ?? -1)
  );
  const expectedCount = Number(ordered[0].partCount?.N ?? -1);
  const expectedSha = ordered[0].shardSha256?.S;
  if (
    !Number.isInteger(expectedCount) ||
    expectedCount !== ordered.length ||
    !/^[a-f0-9]{64}$/.test(expectedSha ?? '') ||
    ordered.some(
      (item, index) =>
        Number(item.part?.N) !== index ||
        item.partCount?.N !== String(expectedCount) ||
        item.shardSha256?.S !== expectedSha ||
        item.immutable?.BOOL !== true ||
        !(item.payload?.B instanceof Uint8Array)
    )
  ) {
    return null;
  }
  const payload = concatenate(
    ordered.map((item) => item.payload!.B as Uint8Array)
  );
  return createHash('sha256').update(payload).digest('hex') === expectedSha
    ? payload
    : null;
}

export interface StoredShard {
  payload: Uint8Array;
  etag: string;
}

export async function readPolicyShard(input: {
  version: string;
  depth: number;
  prefix: string;
}): Promise<StoredShard | null> {
  if (
    !validPolicyRoutePart(input.version) ||
    !Number.isInteger(input.depth) ||
    input.depth <= 0 ||
    !validShardPrefix(input.prefix)
  ) {
    return null;
  }

  let payload: Uint8Array;
  if (LOCAL_DIRECTORY) {
    try {
      payload = await readFile(localShardPath(input.version, input.depth, input.prefix));
    } catch {
      return null;
    }
  } else if (TABLE_NAME) {
    const shardKey = `${input.version}#${input.depth}#${input.prefix}`;
    const response = await dynamo().send(
      new QueryCommand({
        TableName: TABLE_NAME,
        KeyConditionExpression: 'shardKey = :key',
        ExpressionAttributeValues: { ':key': { S: shardKey } },
        ProjectionExpression: 'part, payload, shardSha256, partCount, immutable',
        ConsistentRead: false,
        ScanIndexForward: true,
      })
    );
    const verified = verifiedDynamoPayload(response.Items ?? []);
    if (!verified) return null;
    payload = verified;
  } else {
    return null;
  }
  return {
    payload,
    etag: createHash('sha256').update(payload).digest('hex'),
  };
}

export async function readSampleShard(input: {
  version: string;
  depth: number;
  street: string;
  prefix: string;
}): Promise<StoredShard | null> {
  if (
    !validPolicyRoutePart(input.version) ||
    !Number.isInteger(input.depth) ||
    input.depth <= 0 ||
    !['flop', 'turn', 'river'].includes(input.street) ||
    !validShardPrefix(input.prefix)
  ) {
    return null;
  }
  let payload: Uint8Array;
  if (LOCAL_DIRECTORY) {
    try {
      payload = await readFile(
        localSamplePath(input.version, input.depth, input.street, input.prefix)
      );
    } catch {
      return null;
    }
  } else if (TABLE_NAME) {
    const shardKey = `sample#${input.version}#${input.depth}#${input.street}#${input.prefix}`;
    const response = await dynamo().send(
      new QueryCommand({
        TableName: TABLE_NAME,
        KeyConditionExpression: 'shardKey = :key',
        ExpressionAttributeValues: { ':key': { S: shardKey } },
        ProjectionExpression: 'part, payload, shardSha256, partCount, immutable',
        ConsistentRead: false,
        ScanIndexForward: true,
      })
    );
    const verified = verifiedDynamoPayload(response.Items ?? []);
    if (!verified) return null;
    payload = verified;
  } else {
    return null;
  }
  return { payload, etag: createHash('sha256').update(payload).digest('hex') };
}

function isAcceptedManifest(value: unknown): value is PolicyManifest {
  if (!value || typeof value !== 'object') return false;
  const manifest = value as Partial<PolicyManifest>;
  const validation = manifest.validation;
  const baseAccepted =
    manifest.schemaVersion === 1 &&
    typeof manifest.version === 'string' &&
    manifest.label === 'Approximate GTO' &&
    manifest.active === true &&
    Array.isArray(manifest.depthsBb) &&
    validation?.status === 'accepted';
  if (!baseAccepted) return false;
  if (manifest.subtype !== 'full-hand') return manifest.subtype === 'push-fold';
  return isValidatedFullHandManifest(value);
}

export async function readHostedManifests(): Promise<PolicyManifest[]> {
  if (!TABLE_NAME) return [];
  try {
    const response = await dynamo().send(
      new GetItemCommand({
        TableName: TABLE_NAME,
        Key: { shardKey: { S: 'manifest#active' }, part: { N: '0' } },
        ProjectionExpression: 'payload',
        ConsistentRead: false,
      })
    );
    const payload = response.Item?.payload?.B;
    if (!(payload instanceof Uint8Array)) return [];
    const parsed: unknown = JSON.parse(new TextDecoder().decode(payload));
    return Array.isArray(parsed) ? parsed.filter(isAcceptedManifest) : [];
  } catch {
    return [];
  }
}
