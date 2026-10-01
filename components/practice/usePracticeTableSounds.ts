'use client';

import { useEffect, useRef } from 'react';
import {
  clearPendingPracticeSounds,
  playPracticeSound,
  preloadPracticeSounds,
  practiceSoundCues,
  practiceSoundSnapshot,
  unlockPracticeAudio,
  type PracticeSoundSnapshot,
} from '@/lib/practice-sounds';
import type { HandState } from '@/lib/practice-types';
import type { TableStatus } from '@/components/practice/PracticeTable';

export function isTableSoundActive(status?: TableStatus): boolean {
  return (
    status !== 'loading' &&
    status !== 'solving' &&
    status !== 'unavailable' &&
    status !== 'error'
  );
}

export function usePracticeTableSounds(
  state: HandState | null,
  statusOrEnabled: TableStatus | boolean = 'decision',
  maybeEnabled = true
): void {
  const status: TableStatus =
    typeof statusOrEnabled === 'boolean' ? 'decision' : statusOrEnabled;
  const enabled =
    typeof statusOrEnabled === 'boolean' ? statusOrEnabled : maybeEnabled;

  const previous = useRef<PracticeSoundSnapshot | null>(null);

  useEffect(() => {
    if (!enabled) {
      clearPendingPracticeSounds();
      return;
    }
    preloadPracticeSounds();
    const unlock = () => void unlockPracticeAudio();
    window.addEventListener('pointerdown', unlock, {
      capture: true,
      passive: true,
    });
    window.addEventListener('keydown', unlock, { capture: true });
    return () => {
      window.removeEventListener('pointerdown', unlock, { capture: true });
      window.removeEventListener('keydown', unlock, { capture: true });
    };
  }, [enabled]);

  useEffect(() => {
    if (!enabled || !isTableSoundActive(status)) {
      if (status === 'loading') {
        clearPendingPracticeSounds();
      }
      return;
    }
    if (!state) return;

    const current = practiceSoundSnapshot(state);
    const cues = practiceSoundCues(previous.current, current);
    previous.current = current;
    cues.forEach((cue, index) => playPracticeSound(cue, index * 0.07));
  }, [enabled, state, status]);
}
