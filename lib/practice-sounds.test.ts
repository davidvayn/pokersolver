import { describe, expect, it } from 'vitest';
import {
  practiceSoundCues,
  type PracticeSoundSnapshot,
} from '@/lib/practice-sounds';

function snapshot(
  overrides: Partial<PracticeSoundSnapshot> = {}
): PracticeSoundSnapshot {
  return {
    handId: 'hand-1',
    boardCount: 0,
    actionKinds: [],
    ...overrides,
  };
}

describe('practice sound cues', () => {
  it('announces blinds and two dealt hole cards for a new hand', () => {
    expect(practiceSoundCues(null, snapshot())).toEqual([
      { kind: 'chips' },
      { kind: 'cards', count: 2 },
    ]);
    expect(practiceSoundCues(null, snapshot({ boardCount: 3 }))).toEqual([
      { kind: 'chips' },
      { kind: 'cards', count: 2 },
      { kind: 'cards', count: 3 },
    ]);
  });

  it('plays cards for a newly dealt street and chips for wager actions', () => {
    const previous = snapshot({ actionKinds: ['check'], boardCount: 0 });
    const flop = snapshot({ actionKinds: ['check', 'call'], boardCount: 3 });

    expect(practiceSoundCues(previous, flop)).toEqual([
      { kind: 'chips' },
      { kind: 'cards', count: 3 },
    ]);
  });

  it('stays quiet for checks, folds, and unchanged renders', () => {
    const previous = snapshot({ actionKinds: ['check'] });

    expect(
      practiceSoundCues(previous, snapshot({ actionKinds: ['check', 'fold'] }))
    ).toEqual([]);
    expect(practiceSoundCues(previous, previous)).toEqual([]);
    expect(practiceSoundCues(previous, null)).toEqual([]);
  });
});

describe('isTableSoundActive', () => {
  it('disables sounds during loading, solving, and error overlays', async () => {
    const { isTableSoundActive } = await import(
      '@/components/practice/usePracticeTableSounds'
    );
    expect(isTableSoundActive('loading')).toBe(false);
    expect(isTableSoundActive('solving')).toBe(false);
    expect(isTableSoundActive('unavailable')).toBe(false);
    expect(isTableSoundActive('error')).toBe(false);
  });

  it('enables sounds for visible decision, transitioning, feedback, and review states', async () => {
    const { isTableSoundActive } = await import(
      '@/components/practice/usePracticeTableSounds'
    );
    expect(isTableSoundActive('decision')).toBe(true);
    expect(isTableSoundActive('transitioning')).toBe(true);
    expect(isTableSoundActive('feedback')).toBe(true);
    expect(isTableSoundActive('review')).toBe(true);
    expect(isTableSoundActive()).toBe(true);
  });
});
