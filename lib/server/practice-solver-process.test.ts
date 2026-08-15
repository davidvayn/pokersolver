import { describe, expect, it, vi } from 'vitest';
import fullHandManifests from '@/data/practice/full-hand-manifests.json';
import type { PolicyManifest } from '@/lib/practice-types';

vi.mock('server-only', () => ({}));

import {
  PRACTICE_RESOLVER_IDENTITY,
  practiceResolverCommand,
} from '@/lib/server/practice-solver-process';

function option(args: string[], flag: string): string | null {
  const index = args.indexOf(flag);
  return index < 0 ? null : (args[index + 1] ?? null);
}

describe('pinned practice resolver process', () => {
  it('passes the manifest iteration counts and per-street actor routing to Rust', () => {
    const manifest = (fullHandManifests as PolicyManifest[]).find(
      (candidate) =>
        candidate.version === PRACTICE_RESOLVER_IDENTITY.modelVersion
    );
    expect(manifest?.runtime?.kind).toBe('rust-continual-resolver-v1');
    if (manifest?.runtime?.kind !== 'rust-continual-resolver-v1') {
      throw new Error('the pinned resolver manifest is missing');
    }

    const { args } = practiceResolverCommand();
    const resolver = manifest.runtime.resolver;
    const streets = [
      ['flop', resolver.flopIterations, resolver.flopResolvedActor],
      ['turn', resolver.turnIterations, resolver.turnResolvedActor],
      ['river', resolver.riverIterations, resolver.riverResolvedActor],
    ] as const;

    for (const [street, iterations, resolvedActor] of streets) {
      expect(option(args, `--${street}-resolver-iterations`)).toBe(
        String(iterations)
      );
      expect(option(args, `--${street}-resolver-actor`)).toBe(
        resolvedActor === null ? null : String(resolvedActor)
      );
    }
  });
});
