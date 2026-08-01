import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import {
  DynamoDBClient,
  GetItemCommand,
  PutItemCommand,
} from '@aws-sdk/client-dynamodb';
import {
  MAX_HOSTED_BYTES,
  POLICY_REGION,
  NORMAL_ITEM_BYTES,
  dynamoWriteDelayMs,
  parseArgs,
  required,
  sha256,
  sleep,
  splitBuffer,
} from './lib.mjs';

const args = parseArgs(process.argv.slice(2));
const indexPath = path.resolve(required(args, '--index'));
const tableName = required(args, '--table');
const statePath = path.resolve(args.get('--state') || `${indexPath}.import-state.json`);
const index = JSON.parse(await readFile(indexPath, 'utf8'));
if (index.projectedHostedBytes > MAX_HOSTED_BYTES) throw new Error('Import index exceeds the 20GB hosted limit');

let state = { completed: [], completedParts: {} };
try {
  state = JSON.parse(await readFile(statePath, 'utf8'));
} catch {}
const completed = new Set(state.completed);
const completedParts = new Map(
  Object.entries(state.completedParts ?? {}).map(([key, parts]) => [
    key,
    new Set(Array.isArray(parts) ? parts : []),
  ])
);
const client = new DynamoDBClient({ region: POLICY_REGION });
const root = path.dirname(indexPath);

async function saveState() {
  await writeFile(
    statePath,
    `${JSON.stringify({
      completed: [...completed].sort(),
      completedParts: Object.fromEntries(
        [...completedParts]
          .sort(([first], [second]) => first.localeCompare(second))
          .map(([key, parts]) => [key, [...parts].sort((first, second) => first - second)])
      ),
    }, null, 2)}\n`
  );
}

async function existingPartMatches(tableName, key, part, expected, shardSha256, partCount) {
  const response = await client.send(new GetItemCommand({
    TableName: tableName,
    Key: { shardKey: { S: key }, part: { N: String(part) } },
    ConsistentRead: true,
  }));
  const item = response.Item;
  return Boolean(
    item?.payload?.B &&
    Buffer.from(item.payload.B).equals(Buffer.from(expected)) &&
    item.shardSha256?.S === shardSha256 &&
    Number(item.partCount?.N) === partCount &&
    item.immutable?.BOOL === true
  );
}

for (const shard of index.shards) {
  const key = shard.kind === 'sample'
    ? `sample#${shard.version}#${shard.depthBb}#${shard.street}#${shard.prefix}`
    : `${shard.version}#${shard.depthBb}#${shard.prefix}`;
  if (completed.has(key)) continue;
  const payload = await readFile(path.resolve(root, '..', shard.filename));
  if (payload.byteLength !== shard.bytes || sha256(payload) !== shard.sha256) {
    throw new Error(`Integrity check failed before import: ${shard.filename}`);
  }
  const parts = splitBuffer(payload, NORMAL_ITEM_BYTES);
  const savedParts = completedParts.get(key) ?? new Set();
  completedParts.set(key, savedParts);
  for (let part = 0; part < parts.length; part++) {
    if (savedParts.has(part)) continue;
    const item = parts[part];
    try {
      await client.send(new PutItemCommand({
        TableName: tableName,
        Item: {
          shardKey: { S: key },
          part: { N: String(part) },
          payload: { B: item },
          shardSha256: { S: shard.sha256 },
          partCount: { N: String(parts.length) },
          immutable: { BOOL: true },
        },
        ConditionExpression: 'attribute_not_exists(shardKey) AND attribute_not_exists(part)',
      }));
    } catch (error) {
      if (
        error?.name !== 'ConditionalCheckFailedException' ||
        !(await existingPartMatches(
          tableName,
          key,
          part,
          item,
          shard.sha256,
          parts.length
        ))
      ) {
        throw error;
      }
    }
    savedParts.add(part);
    await saveState();
    await sleep(dynamoWriteDelayMs(item.byteLength, 25));
  }
  completed.add(key);
  completedParts.delete(key);
  await saveState();
  process.stdout.write(`imported ${key} (${parts.length} item${parts.length === 1 ? '' : 's'})\n`);
}
