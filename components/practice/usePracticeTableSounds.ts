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

export function usePracticeTableSounds(
  state: HandState | null,
  enabled: boolean
): void {
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
    const current = practiceSoundSnapshot(state);
    const cues = practiceSoundCues(previous.current, current);
    previous.current = current;
    if (!enabled) return;
    cues.forEach((cue, index) => playPracticeSound(cue, index * 0.07));
  }, [enabled, state]);
}
