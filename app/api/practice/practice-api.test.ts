import { describe, expect, it, vi } from 'vitest';

vi.mock('server-only', () => ({}));
import { GET as getModels } from '@/app/api/practice/models/route';
import { GET as getPolicy } from '@/app/api/practice/policy/[version]/[depth]/[prefix]/route';
import { GET as getSamples } from '@/app/api/practice/samples/[version]/[depth]/[street]/[prefix]/route';

describe('practice public APIs', () => {
  it('returns only accepted active manifests with cache metadata', async () => {
    const response = await getModels();
    const body = await response.json();
    expect(response.status).toBe(200);
    expect(response.headers.get('cache-control')).toContain('s-maxage=300');
    expect(body.schemaVersion).toBe(1);
    expect(body.manifests.length).toBeGreaterThan(0);
    expect(
      body.manifests.every(
        (manifest: { active: boolean; validation: { status: string } }) =>
          manifest.active && manifest.validation.status === 'accepted'
      )
    ).toBe(true);
    expect(JSON.stringify(body)).not.toMatch(/AWS_(?:ACCESS|SECRET)|credential/i);
  });

  it('rejects invalid and missing shard prefixes without a fallback policy', async () => {
    const response = await getPolicy(new Request('http://localhost'), {
      params: Promise.resolve({
        version: 'missing-model',
        depth: '20',
        prefix: 'not-a-prefix',
      }),
    });
    expect(response.status).toBe(404);
    expect(response.headers.get('cache-control')).toBe('no-store');
    expect(await response.json()).toEqual({ error: 'Policy shard not found' });

    const sample = await getSamples(new Request('http://localhost'), {
      params: Promise.resolve({
        version: 'missing-model',
        depth: '20',
        street: 'river',
        prefix: 'not-a-prefix',
      }),
    });
    expect(sample.status).toBe(404);
    expect(sample.headers.get('cache-control')).toBe('no-store');
    expect(await sample.json()).toEqual({ error: 'Sample shard not found' });
  });
});
