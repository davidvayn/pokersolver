import { NextResponse } from 'next/server';
import { activePracticeManifests } from '@/lib/practice-models';
import { readHostedManifests } from '@/lib/server/practice-policy-store';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET() {
  const embedded = activePracticeManifests();
  const hosted = (await readHostedManifests()).filter(
    (manifest) => manifest.subtype === 'full-hand'
  );
  const manifests = new Map(
    [...embedded, ...hosted]
      .filter(
        (manifest) =>
          manifest.active && manifest.validation.status === 'accepted'
      )
      .map((manifest) => [manifest.version, manifest])
  );
  return NextResponse.json(
    {
      schemaVersion: 1,
      manifests: [...manifests.values()],
    },
    {
      headers: {
        'Cache-Control': 'public, max-age=60, s-maxage=300, stale-while-revalidate=3600',
      },
    }
  );
}
