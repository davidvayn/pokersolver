'use client';

import { create } from 'zustand';

interface UiState {
  settingsOpen: boolean;
  showSolverStats: boolean;
  openSettings: () => void;
  closeSettings: () => void;
  toggleSettings: () => void;
  setShowSolverStats: (show: boolean) => void;
}

export const useUi = create<UiState>((set) => ({
  settingsOpen: false,
  showSolverStats: false,
  openSettings: () => set({ settingsOpen: true }),
  closeSettings: () => set({ settingsOpen: false }),
  toggleSettings: () => set((s) => ({ settingsOpen: !s.settingsOpen })),
  setShowSolverStats: (show) => set({ showSolverStats: show }),
}));
