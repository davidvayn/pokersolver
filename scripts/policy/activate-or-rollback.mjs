import { readFile } from 'node:fs/promises';
import path from 'node:path';
import {
  DynamoDBClient,
  GetItemCommand,
  PutItemCommand,
} from '@aws-sdk/client-dynamodb';
import {
  POLICY_REGION,
  assertAcceptedManifest,
  parseArgs,
  required,
} from './lib.mjs';

const args = parseArgs(process.argv.slice(2));
const tableName = required(args, '--table');
const client = new DynamoDBClient({ region: POLICY_REGION });
const encoder = new TextEncoder();
const activeKey = { shardKey: { S: 'manifest#active' }, part: { N: '0' } };

async function get(key) {
  const response = await client.send(new GetItemCommand({ TableName: tableName, Key: key, ConsistentRead: true }));
  return response.Item?.payload?.B ?? null;
}

let nextPayload;
if (args.has('--manifest')) {
  const value = JSON.parse(await readFile(path.resolve(required(args, '--manifest')), 'utf8'));
  const manifests = Array.isArray(value) ? value : [value];
  manifests.forEach(assertAcceptedManifest);
  nextPayload = encoder.encode(JSON.stringify(manifests));
} else {
  const version = required(args, '--rollback-version');
  nextPayload = await get({ shardKey: { S: `manifest#history#${version}` }, part: { N: '0' } });
  if (!nextPayload) throw new Error(`No activation history for ${version}`);
}

const previous = await get(activeKey);
if (previous) {
  const parsed = JSON.parse(new TextDecoder().decode(previous));
  const version = parsed.map((manifest) => manifest.version).sort().join('+');
  await client.send(new PutItemCommand({
    TableName: tableName,
    Item: {
      shardKey: { S: `manifest#history#${version}` },
      part: { N: '0' },
      payload: { B: previous },
      savedAt: { N: String(Date.now()) },
    },
  }));
}
await client.send(new PutItemCommand({
  TableName: tableName,
  Item: { ...activeKey, payload: { B: nextPayload }, activatedAt: { N: String(Date.now()) } },
}));
process.stdout.write('active manifest updated; previous activation retained for rollback\n');
