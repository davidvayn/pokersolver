'use client';

import { create } from 'zustand';
import {
  Position,
  SIX_MAX,
  TABLE_FORMATS,
  TableFormat,
} from './positions';

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
  setFormat: (format) => {
    const current = get();
    const hero = format.positions.includes(current.hero)
      ? current.hero
      : format.positions[0];
    const villain =
      format.positions.includes(current.villain) && current.villain !== hero
        ? current.villain
        : format.positions.find((position) => position !== hero) ?? hero;
    set({ format, hero, villain });
  },
  setHero: (hero) => set({ hero }),
  setVillain: (villain) => set({ villain }),
  toggleFormat: () => {
    const currentIndex = TABLE_FORMATS.findIndex(
      (format) => format.seats === get().format.seats
    );
    const next = TABLE_FORMATS[(currentIndex + 1) % TABLE_FORMATS.length];
    get().setFormat(next);
  },
}));
