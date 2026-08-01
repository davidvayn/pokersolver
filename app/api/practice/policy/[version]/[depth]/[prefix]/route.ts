import { readPolicyShard } from '@/lib/server/practice-policy-store';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

interface RouteContext {
  params: Promise<{ version: string; depth: string; prefix: string }>;
}

export async function GET(_request: Request, context: RouteContext) {
  const { version, depth: rawDepth, prefix } = await context.params;
  const depth = Number(rawDepth);
  const shard = await readPolicyShard({ version, depth, prefix });
  if (!shard) {
    return Response.json(
      { error: 'Policy shard not found' },
      { status: 404, headers: { 'Cache-Control': 'no-store' } }
    );
  }
  return new Response(shard.payload, {
    status: 200,
    headers: {
      'Content-Type': 'application/vnd.poker-lab.policy-v1',
      'Content-Length': String(shard.payload.byteLength),
      ETag: `"${shard.etag}"`,
      'Cache-Control': 'public, max-age=31536000, immutable',
      'X-Content-Type-Options': 'nosniff',
    },
  });
}
