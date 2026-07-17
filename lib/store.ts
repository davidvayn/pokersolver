'use client';

import { create } from 'zustand';
import { Position, SIX_MAX, FULL_RING, TableFormat } from './positions';

interface SpotState {
  format: TableFormat;
  hero: Position;
  villain: Position;
  setFormat: (f: TableFormat) => void;
  setHero: (p: Position) => void;
  setVillain: (p: Position) => void;
  toggleFormat: () => void;
}

export const useSpot = create<SpotState>((set, get) => ({
  format: SIX_MAX,
  hero: 'BTN',
  villain: 'BB',
  setFormat: (format) => set({ format }),
  setHero: (hero) => set({ hero }),
  setVillain: (villain) => set({ villain }),
  toggleFormat: () => {
    const next = get().format.seats === 6 ? FULL_RING : SIX_MAX;
    const hero = next.positions.includes(get().hero) ? get().hero : 'BTN';
    const villain = next.positions.includes(get().villain) ? get().villain : 'BB';
    set({ format: next, hero, villain });
  },
}));
