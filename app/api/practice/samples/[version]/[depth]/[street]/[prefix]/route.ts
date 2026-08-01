import { readSampleShard } from '@/lib/server/practice-policy-store';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

interface RouteContext {
  params: Promise<{
    version: string;
    depth: string;
    street: string;
    prefix: string;
  }>;
}

export async function GET(_request: Request, context: RouteContext) {
  const { version, depth: rawDepth, street, prefix } = await context.params;
  const shard = await readSampleShard({
    version,
    depth: Number(rawDepth),
    street,
    prefix,
  });
  if (!shard) {
    return Response.json(
      { error: 'Sample shard not found' },
      { status: 404, headers: { 'Cache-Control': 'no-store' } }
    );
  }
  return new Response(shard.payload, {
    headers: {
      'Content-Type': 'application/vnd.poker-lab.sample-v1',
      ETag: `"${shard.etag}"`,
      'Cache-Control': 'public, max-age=31536000, immutable',
      'X-Content-Type-Options': 'nosniff',
    },
  });
}
