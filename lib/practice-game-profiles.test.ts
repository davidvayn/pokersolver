import { describe, expect, it } from 'vitest';
import {
  DEFAULT_ONLINE_CASH_REFERENCE,
  REQUESTED_PRACTICE_STACK_DEPTHS_BB,
} from '@/lib/practice-game-profiles';
import { isFullHandDepth } from '@/lib/practice-types';
import { activePracticeManifests } from '@/lib/practice-models';

describe('planned cash-game profiles', () => {
  it('records all requested stack depths, including 1000bb and 2000bb', () => {
    expect(REQUESTED_PRACTICE_STACK_DEPTHS_BB).toEqual([
      20, 40, 50, 100, 200, 1000, 2000,
    ]);
    expect(REQUESTED_PRACTICE_STACK_DEPTHS_BB.every(isFullHandDepth)).toBe(true);
  });

  it('pins the published two-player USD NL25 reference without inventing NL20', () => {
    const reference = DEFAULT_ONLINE_CASH_REFERENCE;
    expect(reference.label).toBe('PokerStars NL25');
    expect(reference.playersDealt).toBe(2);
    expect(reference.smallBlindCents).toBe(10);
    expect(reference.bigBlindCents).toBe(25);
    expect(reference.rake.rateBasisPoints / 100).toBe(4.5);
    expect(reference.rake.capCents / reference.bigBlindCents).toBe(2);
    expect(reference.rake.noFlopNoDrop).toBe(true);
    expect(reference.rake.rounding).toBe('nearest-cent-half-to-even');
    expect(reference.sourceUrl).toBe('https://www.pokerstars.com/poker/room/rake/');
  });

  it('does not activate raked policies by adding a reference or training targets', () => {
    expect(DEFAULT_ONLINE_CASH_REFERENCE.modelStatus).toBe('requires-rake-aware-training');
    const manifests = activePracticeManifests();
    expect(manifests.length).toBeGreaterThan(0);
    expect(manifests.every((manifest) => manifest.abstraction.rake === 'none')).toBe(true);
    expect(manifests.some((manifest) => manifest.depthsBb.includes(2000))).toBe(false);
  });
});
