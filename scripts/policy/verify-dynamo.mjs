import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { DynamoDBClient, QueryCommand } from '@aws-sdk/client-dynamodb';
import { POLICY_REGION, parseArgs, required, sha256, sleep } from './lib.mjs';

const args = parseArgs(process.argv.slice(2));
const index = JSON.parse(await readFile(path.resolve(required(args, '--index')), 'utf8'));
const tableName = required(args, '--table');
const client = new DynamoDBClient({ region: POLICY_REGION });
let verifiedBytes = 0;

for (const shard of index.shards) {
  const key = shard.kind === 'sample'
    ? `sample#${shard.version}#${shard.depthBb}#${shard.street}#${shard.prefix}`
    : `${shard.version}#${shard.depthBb}#${shard.prefix}`;
  const response = await client.send(new QueryCommand({
    TableName: tableName,
    KeyConditionExpression: 'shardKey = :key',
    ExpressionAttributeValues: { ':key': { S: key } },
    ProjectionExpression: 'part, payload',
    ConsistentRead: true,
    ScanIndexForward: true,
  }));
  const parts = (response.Items ?? [])
    .sort((first, second) => Number(first.part.N) - Number(second.part.N))
    .map((item) => item.payload.B);
  if (parts.length === 0) throw new Error(`Missing hosted shard ${key}`);
  const payload = Buffer.concat(parts);
  if (payload.byteLength !== shard.bytes || sha256(payload) !== shard.sha256) {
    throw new Error(`Hosted integrity mismatch for ${key}`);
  }
  verifiedBytes += payload.byteLength;
  await sleep(50);
}
process.stdout.write(`${JSON.stringify({ shards: index.shards.length, verifiedBytes, ok: true })}\n`);
